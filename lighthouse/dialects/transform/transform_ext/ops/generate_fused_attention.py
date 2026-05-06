"""Transform extension to generate fused attention computation."""

from mlir import ir
from mlir.dialects import ext, transform, arith, scf, linalg, tensor
from mlir.dialects.transform import DiagnosedSilenceableFailure

from lighthouse.dialects.transform.transform_ext import TransformExtensionDialect


class GenerateFusedAttention(
    TransformExtensionDialect.Operation, name="generate_fused_attention"
):
    """Generate tiled fused attention computation (flash attention optimization).

    Takes Q, K, V slices and output tensor, and generates an inner tiled loop
    that computes fused attention with online softmax using running max and sum.

    This implements the flash attention algorithm where:
    1. The computation is tiled along the inner K dimension (sequence length)
    2. Online max and sum are maintained across tiles
    3. Output is incrementally updated with rescaled contributions

    Args:
        q_slice: Handle to Q slice operation (tensor.extract_slice)
        k_slice: Handle to K slice operation (tensor.extract_slice)
        scale_slice: Handle to scaling factor slice operation (tensor.extract_slice)
        v_slice: Handle to V slice operation (tensor.extract_slice)
        output: Handle to the output operation to replace
        tile_size: Size of inner dimension tiles (default: from attributes)
    """

    q_slice: ext.Operand[transform.AnyOpType]
    k_slice: ext.Operand[transform.AnyOpType]
    scale_slice: ext.Operand[transform.AnyOpType]
    v_slice: ext.Operand[transform.AnyOpType]
    output: ext.Operand[transform.AnyOpType]
    tile_size: ir.IntegerAttr
    new_output: ext.Result[transform.AnyOpType[()]] = ext.infer_result()

    @classmethod
    def attach_interface_impls(cls, ctx=None):
        cls.TransformOpInterfaceModel.attach(cls.OPERATION_NAME, context=ctx)
        cls.MemoryEffectsOpInterfaceModel.attach(cls.OPERATION_NAME, context=ctx)

    class TransformOpInterfaceModel(transform.TransformOpInterface):
        @staticmethod
        def apply(
            op: "GenerateFusedAttention",
            rewriter: transform.TransformRewriter,
            results: transform.TransformResults,
            state: transform.TransformState,
        ) -> DiagnosedSilenceableFailure:
            # Get payload operations
            q_slice_ops = state.get_payload_ops(op.q_slice)
            k_slice_ops = state.get_payload_ops(op.k_slice)
            scale_slice_ops = state.get_payload_ops(op.scale_slice)
            v_slice_ops = state.get_payload_ops(op.v_slice)
            output_ops = state.get_payload_ops(op.output)

            if (
                len(q_slice_ops) != 1
                or len(k_slice_ops) != 1
                or len(scale_slice_ops) != 1
                or len(v_slice_ops) != 1
                or len(output_ops) != 1
            ):
                return DiagnosedSilenceableFailure.emit_silenceable_error(
                    "Expected exactly one operation for each operand"
                )

            q_slice_op = q_slice_ops[0]
            k_slice_op = k_slice_ops[0]
            scale_slice_op = scale_slice_ops[0]
            v_slice_op = v_slice_ops[0]
            output_op = output_ops[0]

            # Get tile size
            tile_size_value = ir.IntegerAttr(op.tile_size).value

            # Get the result types and shapes
            q_result = q_slice_op.results[0]
            k_result = k_slice_op.results[0]
            scale_result = scale_slice_op.results[0]
            v_result = v_slice_op.results[0]
            output_result = output_op.results[0]

            # Extract shape information from the slice operations
            q_type = ir.RankedTensorType(q_result.type)
            k_type = ir.RankedTensorType(k_result.type)
            scale_type = ir.RankedTensorType(scale_result.type)
            v_type = ir.RankedTensorType(v_result.type)
            output_type = ir.RankedTensorType(output_result.type)
            print(f"Q type: {q_type}, K type: {k_type}, Scale type: {scale_type}, V type: {v_type}, Output type: {output_type}")

            element_type = q_type.element_type
            index_type = ir.IndexType.get()

            # Build the fused attention computation
            with ir.InsertionPoint(output_op):
                # Collapse the unit batch dimension to get 2D tensors
                # Q: [1, seq_q, head_dim] -> [seq_q, head_dim]
                # K: [1, seq_k, head_dim] -> [seq_k, head_dim]
                # V: [1, seq_k, head_dim] -> [seq_k, head_dim]
                # Scale: [1, seq_q, 1] -> [seq_q, 1]
                q_2d_ty = ir.RankedTensorType.get((q_type.shape[1], q_type.shape[2]), element_type)
                k_2d_ty = ir.RankedTensorType.get((k_type.shape[1], k_type.shape[2]), element_type)
                v_2d_ty = ir.RankedTensorType.get((v_type.shape[1], v_type.shape[2]), element_type)
                scale_2d_ty = ir.RankedTensorType.get((scale_type.shape[1], scale_type.shape[2]), element_type)
                q_2d = tensor.collapse_shape(q_2d_ty, src=q_result, reassociation=[[0, 1], [2]])
                k_2d = tensor.collapse_shape(k_2d_ty, src=k_result, reassociation=[[0, 1], [2]])
                v_2d = tensor.collapse_shape(v_2d_ty, src=v_result, reassociation=[[0, 1], [2]])
                scale_2d = tensor.collapse_shape(scale_2d_ty, src=scale_result, reassociation=[[0, 1], [2]])

                # Get dimensions from 2D tensors
                # Q: [seq_q, head_dim]
                # K: [seq_k, head_dim]
                # V: [seq_k, head_dim]
                seq_q_dim = q_type.shape[1]
                head_dim = q_type.shape[2]
                seq_k_size = arith.constant(index_type, k_type.shape[1])
                print(f"Seq Q dim: {seq_q_dim}, Head dim: {head_dim}, Sequence length K: {seq_k_size}")

                # Initialize max to -inf
                # Shape: [seq_q] (1D for 2D tensors)
                neg_inf = arith.constant(
                    element_type, float("-inf") if element_type == ir.F32Type.get() else -1e10
                )
                max_shape = [seq_q_dim]
                max_init = tensor.empty(max_shape, element_type)
                running_max = linalg.fill(neg_inf, outs=[max_init])

                # Initialize sum to 0
                # Shape: [seq_q] (1D for 2D tensors)
                zero = arith.constant(element_type, 0.0)
                sum_init = tensor.empty(max_shape, element_type)
                running_sum = linalg.fill(zero, outs=[sum_init])

                # Initialize output accumulator to 0
                # Shape: [seq_q, head_dim] (2D)
                output_2d_shape = [seq_q_dim, head_dim]
                output_2d_init = tensor.empty(output_2d_shape, element_type)
                output_acc = linalg.fill(zero, outs=[output_2d_init])

                # Create tiled loop over K dimension
                c0 = arith.constant(index_type, 0)
                tile_size_const = arith.constant(index_type, tile_size_value)

                # Build the scf.for loop
                loop_result = scf.ForOp(
                    c0,
                    seq_k_size,
                    tile_size_const,
                    [running_max, running_sum, output_acc],
                )

                with ir.InsertionPoint(loop_result.body):
                    # Get loop iteration variable and current state
                    k_idx = loop_result.induction_variable
                    old_max = loop_result.inner_iter_args[0]
                    old_sum = loop_result.inner_iter_args[1]
                    old_output = loop_result.inner_iter_args[2]

                    # Slice K and V for this tile
                    # TODO: Implement proper slicing logic here
                    # This is a placeholder - needs to be filled with actual slice operations

                    # Compute Q @ K^T for this tile
                    # Q: [seq_q, head_dim]
                    # K_tile: [tile_size, head_dim] (transposed from [head_dim, tile_size])
                    # Result: [seq_q, tile_size]

                    # Compute new max across this tile
                    # new_max: [seq_q] = max(old_max, row_max(Q @ K^T_tile))

                    # Compute exp(Q @ K^T_tile - new_max)
                    # exp_scores: [seq_q, tile_size]

                    # Update running sum with rescaling
                    # new_sum: [seq_q] = old_sum * exp(old_max - new_max) + row_sum(exp_scores)

                    # Update output with rescaling
                    # new_output: [seq_q, head_dim] = old_output * (old_sum * exp(old_max - new_max) / new_sum)
                    #                                + (exp_scores @ V_tile) / new_sum

                    # For now, yield the unchanged values (placeholder)
                    scf.yield_([old_max, old_sum, old_output])

                # Extract final results from loop
                final_max = loop_result.results[0]
                final_sum = loop_result.results[1]
                final_output_2d = loop_result.results[2]

                # Expand the 2D output back to 3D to match the original output shape
                # [seq_q, head_dim] -> [1, seq_q, head_dim]
                final_output_3d = tensor.expand_shape(output_type, src=final_output_2d, reassociation=[[0, 1], [2]], output_shape=[], static_output_shape=output_type.shape)

                # Create a dummy add operation to wrap the final output
                # This is needed because replace_op requires an operation, not a value
                zero_const = arith.constant(element_type, 0.0)
                # Create a linalg.add that adds 0 (identity operation)
                zero_tensor_shape = output_type.shape
                zero_tensor_init = tensor.empty(zero_tensor_shape, element_type)
                zero_tensor = linalg.fill(zero_const, outs=[zero_tensor_init])

                # Create the add operation: final_output_3d + 0
                output_init_for_add = tensor.empty(zero_tensor_shape, element_type)
                dummy_add = linalg.add(final_output_3d, zero_tensor, outs=[output_init_for_add])

                # Replace the original output operation with the dummy add
                rewriter.replace_op(output_op, dummy_add.owner)

            results.set_ops(op.new_output, [dummy_add.owner])
            return DiagnosedSilenceableFailure.Success

        @staticmethod
        def allow_repeated_handle_operands(_op: "GenerateFusedAttention") -> bool:
            return False

    class MemoryEffectsOpInterfaceModel(ir.MemoryEffectsOpInterface):
        @staticmethod
        def get_effects(op: ir.Operation, effects):
            # Read Q, K, scale, V slices
            transform.only_reads_handle(op.op_operands[:4], effects)
            # Consume and replace output
            transform.consumes_handle(op.op_operands[4:5], effects)
            # Produce new output handle
            transform.produces_handle(op.results, effects)
            # Modify the payload
            transform.modifies_payload(effects)


def generate_fused_attention(
    q_slice: ir.Value,
    k_slice: ir.Value,
    scale_slice: ir.Value,
    v_slice: ir.Value,
    output: ir.Value,
    tile_size: int | ir.IntegerAttr,
) -> ir.Value:
    """Generate fused attention computation with inner tiling.

    Args:
        q_slice: Handle to Q slice operation
        k_slice: Handle to K slice operation
        scale_slice: Handle to scaling factor slice operation
        v_slice: Handle to V slice operation
        output: Handle to output operation to replace
        tile_size: Size of tiles along the K dimension

    Returns:
        Handle to the new output operation
    """
    if not isinstance(tile_size, ir.IntegerAttr):
        tile_size = ir.IntegerAttr.get(ir.IntegerType.get_signless(64), tile_size)

    return GenerateFusedAttention(
        q_slice, k_slice, scale_slice, v_slice, output, tile_size=tile_size
    ).new_output
