"""Transform extension to generate fused attention computation."""

from mlir import ir
from mlir.dialects import ext, transform, arith, scf, linalg, tensor, math
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
            print(
                f"Q type: {q_type}, K type: {k_type}, Scale type: {scale_type}, V type: {v_type}, Output type: {output_type}"
            )

            element_type = q_type.element_type
            index_type = ir.IndexType.get()

            # Build the fused attention computation
            with ir.InsertionPoint(output_op):
                # Collapse the unit batch dimension to get 2D tensors
                # Q: [1, seq_q, head_dim] -> [seq_q, head_dim]
                # K: [1, seq_k, head_dim] -> [seq_k, head_dim]
                # V: [1, seq_k, head_dim] -> [seq_k, head_dim]
                # Scale: [1, seq_q, 1] -> [seq_q, 1]
                q_2d_ty = ir.RankedTensorType.get(
                    (q_type.shape[1], q_type.shape[2]), element_type
                )
                k_2d_ty = ir.RankedTensorType.get(
                    (k_type.shape[1], k_type.shape[2]), element_type
                )
                v_2d_ty = ir.RankedTensorType.get(
                    (v_type.shape[1], v_type.shape[2]), element_type
                )
                scale_2d_ty = ir.RankedTensorType.get(
                    (scale_type.shape[1], scale_type.shape[2]), element_type
                )
                q_2d = tensor.collapse_shape(
                    q_2d_ty, src=q_result, reassociation=[[0, 1], [2]]
                )
                k_2d = tensor.collapse_shape(
                    k_2d_ty, src=k_result, reassociation=[[0, 1], [2]]
                )
                v_2d = tensor.collapse_shape(
                    v_2d_ty, src=v_result, reassociation=[[0, 1], [2]]
                )
                scale_2d = tensor.collapse_shape(
                    scale_2d_ty, src=scale_result, reassociation=[[0, 1], [2]]
                )

                # Get dimensions from 2D tensors
                # Q: [seq_q, head_dim]
                # K: [seq_k, head_dim]
                # V: [seq_k, head_dim]
                seq_q_dim = q_type.shape[1]
                head_dim = q_type.shape[2]
                seq_k_size = arith.constant(index_type, k_type.shape[1])

                # Initialize max to -inf
                # Shape: [seq_q] (1D for 2D tensors)
                neg_inf = arith.constant(
                    element_type,
                    float("-inf") if element_type == ir.F32Type.get() else -1e10,
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
                    # K: [seq_k, head_dim] -> K_tile: [tile_size, head_dim]
                    # V: [seq_k, head_dim] -> V_tile: [tile_size, head_dim]
                    one = arith.constant(index_type, 1)
                    k_tile_type = ir.RankedTensorType.get(
                        [tile_size_value, head_dim], element_type
                    )
                    k_tile = tensor.extract_slice(
                        k_tile_type,
                        source=k_2d,
                        offsets=[k_idx],
                        sizes=[],
                        strides=[],
                        static_offsets=[ir.ShapedType.get_dynamic_size(), 0],
                        static_sizes=[tile_size_value, head_dim],
                        static_strides=[1, 1],
                    )

                    v_tile_type = ir.RankedTensorType.get(
                        [tile_size_value, head_dim], element_type
                    )
                    v_tile = tensor.extract_slice(
                        v_tile_type,
                        source=v_2d,
                        offsets=[k_idx],
                        sizes=[],
                        strides=[],
                        static_offsets=[ir.ShapedType.get_dynamic_size(), 0],
                        static_sizes=[tile_size_value, head_dim],
                        static_strides=[1, 1],
                    )
                    # Transpose K_tile: [tile_size, head_dim] -> [head_dim, tile_size]
                    k_tile_t_shape = [head_dim, tile_size_value]
                    k_tile_t_init = tensor.empty(k_tile_t_shape, element_type)
                    k_tile_t = linalg.transpose(
                        k_tile, outs=[k_tile_t_init], permutation=[1, 0]
                    )

                    # Compute Q @ K^T for this tile
                    # Q: [seq_q, head_dim]
                    # K_tile_T: [head_dim, tile_size]
                    # Result: [seq_q, tile_size]
                    qk_shape = [seq_q_dim, tile_size_value]
                    qk_init = tensor.empty(qk_shape, element_type)
                    qk_filled = linalg.fill(zero, outs=[qk_init])
                    qk = linalg.matmul(q_2d, k_tile_t, outs=[qk_filled])

                    # Compute row-wise max of qk
                    # row_max: [seq_q] = max(qk, axis=1)
                    row_max_init = tensor.empty([seq_q_dim], element_type)
                    row_max_filled = linalg.fill(neg_inf, outs=[row_max_init])
                    dims_attr = ir.DenseI64ArrayAttr.get([1])
                    f16 = ir.F16Type.get()

                    @linalg.reduce(
                        result=[ir.RankedTensorType.get([seq_q_dim], element_type)],
                        inputs=[qk],
                        inits=[row_max_filled],
                        dimensions=dims_attr,
                    )
                    def row_max(elem: f16, acc: f16):
                        return arith.maximumf(elem, acc)

                    # Compute new max across this tile
                    # new_max: [seq_q] = max(old_max, row_max)
                    new_max_init = tensor.empty([seq_q_dim], element_type)
                    new_max = linalg.max(old_max, row_max, outs=[old_max])

                    # Compute exp(qk - new_max)
                    # First broadcast new_max to [seq_q, 1] then to [seq_q, tile_size]
                    new_max_2d_type = ir.RankedTensorType.get(
                        [seq_q_dim, 1], element_type
                    )
                    new_max_2d_init = tensor.empty(
                        [seq_q_dim, tile_size_value], element_type
                    )
                    new_max_2d = linalg.broadcast(
                        new_max, outs=[new_max_2d_init], dimensions=[0]
                    )

                    # exp_scores: [seq_q, tile_size] = exp(qk - new_max_2d)
                    exp_scores_init = tensor.empty(qk_shape, element_type)

                    @linalg.map(
                        result=[ir.RankedTensorType.get(qk_shape, element_type)],
                        inputs=[qk, new_max_2d],
                        init=exp_scores_init,
                    )
                    def exp_scores(qk_val: f16, max_val: f16, _: f16):
                        diff = arith.subf(qk_val, max_val)
                        return math.exp(diff)

                    # Compute row-wise sum of exp_scores
                    # row_sum_exp: [seq_q] = sum(exp_scores, axis=1)
                    row_sum_exp_init = tensor.empty([seq_q_dim], element_type)
                    row_sum_exp_filled = linalg.fill(zero, outs=[row_sum_exp_init])

                    @linalg.reduce(
                        result=[ir.RankedTensorType.get([seq_q_dim], element_type)],
                        inputs=[exp_scores],
                        inits=[row_sum_exp_filled],
                        dimensions=dims_attr,
                    )
                    def row_sum_exp(elem: f16, acc: f16):
                        return arith.addf(elem, acc)

                    # Compute correction factor for old values: exp(old_max - new_max)
                    correction_init = tensor.empty([seq_q_dim], element_type)

                    @linalg.map(
                        result=[ir.RankedTensorType.get([seq_q_dim], element_type)],
                        inputs=[old_max, new_max],
                        init=correction_init,
                    )
                    def correction(old_val: f16, new_val: f16, _: f16):
                        diff = arith.subf(old_val, new_val)
                        return math.exp(diff)

                    # Update running sum with rescaling
                    # new_sum: [seq_q] = old_sum * correction + row_sum_exp
                    # new_sum_init = tensor.empty([seq_q_dim], element_type)

                    @linalg.map(
                        result=[ir.RankedTensorType.get([seq_q_dim], element_type)],
                        inputs=[old_sum, correction, row_sum_exp],
                        init=old_sum,
                    )
                    def new_sum(old_s: f16, corr: f16, new_s: f16, _: f16):
                        rescaled = arith.mulf(old_s, corr)
                        return arith.addf(rescaled, new_s)

                    # Compute exp_scores @ V_tile
                    # exp_scores: [seq_q, tile_size]
                    # V_tile: [tile_size, head_dim]
                    # Result: [seq_q, head_dim]
                    exp_v_init = tensor.empty([seq_q_dim, head_dim], element_type)
                    exp_v_filled = linalg.fill(zero, outs=[exp_v_init])
                    exp_v = linalg.matmul(exp_scores, v_tile, outs=[exp_v_filled])

                    # Update output with rescaling
                    # new_output: [seq_q, head_dim] = old_output * (correction * old_sum / new_sum) + (exp_v / new_sum)
                    # First compute rescale factor: correction / new_sum (broadcasted to [seq_q, 1])
                    rescale_factor_div_init = tensor.empty([seq_q_dim], element_type)
                    rescale_factor_div = linalg.div(
                        correction, new_sum, outs=[rescale_factor_div_init]
                    )
                    rescale_factor_mul_init = tensor.empty([seq_q_dim], element_type)
                    rescale_factor_mul = linalg.mul(
                        rescale_factor_div, old_sum, outs=[rescale_factor_mul_init]
                    )
                    rescale_factor_2d_init = tensor.empty(
                        [seq_q_dim, tile_size_value], element_type
                    )
                    rescale_factor_2d = linalg.broadcast(
                        rescale_factor_mul,
                        outs=[rescale_factor_2d_init],
                        dimensions=[0],
                    )

                    # Rescale old output
                    rescaled_old_init = tensor.empty(
                        [seq_q_dim, head_dim], element_type
                    )
                    rescaled_old = linalg.mul(
                        old_output, rescale_factor_2d, outs=[old_output]
                    )

                    # Compute: exp_v / new_sum (broadcast new_sum to [seq_q, tile_size])
                    norm_factor_2d_init = tensor.empty(
                        [seq_q_dim, tile_size_value], element_type
                    )
                    norm_factor_2d = linalg.broadcast(
                        new_sum, outs=[norm_factor_2d_init], dimensions=[0]
                    )

                    # Normalize new contribution
                    normalized_exp_v_init = tensor.empty(
                        [seq_q_dim, head_dim], element_type
                    )
                    normalized_exp_v = linalg.div(
                        exp_v, norm_factor_2d, outs=[normalized_exp_v_init]
                    )

                    # Add both contributions
                    new_output_init = tensor.empty([seq_q_dim, head_dim], element_type)
                    new_output = linalg.add(
                        rescaled_old, normalized_exp_v, outs=[old_output]
                    )

                    scf.yield_([new_max, new_sum, new_output])

                # Extract final result from loop
                final_output_2d = loop_result.results[2]

                # Expand the 2D output back to 3D to match the original output shape
                # [seq_q, head_dim] -> [1, seq_q, head_dim]
                final_output_3d = tensor.expand_shape(
                    output_type,
                    src=final_output_2d,
                    reassociation=[[0, 1], [2]],
                    output_shape=[],
                    static_output_shape=output_type.shape,
                )

                # Create a dummy add operation to wrap the final output
                # This is needed because replace_op requires an operation, not a value
                zero_const = arith.constant(element_type, 0.0)
                # Create a linalg.add that adds 0 (identity operation)
                zero_tensor_shape = output_type.shape
                zero_tensor_init = tensor.empty(zero_tensor_shape, element_type)
                zero_tensor = linalg.fill(zero_const, outs=[zero_tensor_init])

                # Create the add operation: final_output_3d + 0
                output_init_for_add = tensor.empty(zero_tensor_shape, element_type)
                dummy_add = linalg.add(
                    final_output_3d, zero_tensor, outs=[output_init_for_add]
                )

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
