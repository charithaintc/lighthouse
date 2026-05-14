"""Transform extension to generate fused attention computation."""

import numpy as np
from mlir import ir
from mlir.dialects import ext, transform, arith, scf, linalg, tensor, math, vector
from mlir.dialects.transform import DiagnosedSilenceableFailure

from lighthouse.dialects.transform.transform_ext import TransformExtensionDialect


class GenerateFusedAttention(
    TransformExtensionDialect.Operation, name="generate_fused_attention"
):
    """Generate tiled fused attention computation (flash attention optimization).

    Takes Q, K, V loads and scale constant from bufferized IR, and generates an inner
    tiled loop that computes fused attention with online softmax using running max and sum.

    This implements the flash attention algorithm where:
    1. The computation is tiled along the reduction dimension (K/V sequence length)
    2. Online max and sum are maintained across tiles
    3. Output is incrementally updated with rescaled contributions

    Args:
        q_load: Handle to Q load operation (vector.transfer_read)
        k_load: Handle to K load operation (vector.transfer_read)
        v_load: Handle to V load operation (vector.transfer_read)
        scale: Handle to scale constant operation (arith.constant)
        output: Handle to the output operation to replace (vector.contract)
        tile_size: Tile size for the reduction dimension tiling (K/V sequence length)
    """

    q_load: ext.Operand[transform.AnyOpType]
    k_load: ext.Operand[transform.AnyOpType]
    v_load: ext.Operand[transform.AnyOpType]
    scale: ext.Operand[transform.AnyOpType]
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
            q_load_ops = state.get_payload_ops(op.q_load)
            k_load_ops = state.get_payload_ops(op.k_load)
            v_load_ops = state.get_payload_ops(op.v_load)
            scale_ops = state.get_payload_ops(op.scale)
            output_ops = state.get_payload_ops(op.output)

            if (
                len(q_load_ops) != 1
                or len(k_load_ops) != 1
                or len(v_load_ops) != 1
                or len(scale_ops) != 1
                or len(output_ops) != 1
            ):
                return DiagnosedSilenceableFailure.emit_silenceable_error(
                    "Expected exactly one operation for each operand"
                )

            q_load_op = q_load_ops[0]
            k_load_op = k_load_ops[0]
            v_load_op = v_load_ops[0]
            scale_op = scale_ops[0]
            output_op = output_ops[0]

            # Extract the scale scalar value from scale_op (arith.constant)
            scale_attr = scale_op.attributes["value"]
            # Extract the scalar scale value from the scale_attr DenseElementsAttr
            scale_dense_attr = ir.DenseElementsAttr(scale_attr)
            # Get the first element as the scale value (all elements are the same in a splat)
            scale_np_array = np.array(scale_dense_attr)
            scale_value = float(scale_np_array.flat[0])

            # Extract wg_rows and d_head from q_load result type
            # q_load is vector.transfer_read that produces a vector
            q_load_result = q_load_op.results[0]
            q_vector_type = ir.VectorType(q_load_result.type)
            wg_rows = q_vector_type.shape[0]
            d_head = q_vector_type.shape[1]

            # Get tile size
            tile_size_value = ir.IntegerAttr(op.tile_size).value

            # Get element type from q_load result
            element_type = q_vector_type.element_type

            # Build the fused attention computation
            with ir.InsertionPoint(output_op):
                # 1. Define m_i_init: vector of shape [wg_rows] with neg_inf values
                m_i_vector_type = ir.VectorType.get([wg_rows], element_type)
                neg_inf_value = 0xFC00  if element_type == ir.F16Type.get() else float("-inf")
                m_i_values = np.full(wg_rows, neg_inf_value, dtype=np.float16 if element_type == ir.F16Type.get() else np.float32)
                m_i_init_attr = ir.DenseElementsAttr.get(m_i_values, type=m_i_vector_type)
                m_i_init = arith.constant(m_i_vector_type, m_i_init_attr)

                # 2. Define l_i_init: vector of shape [wg_rows] with zero values
                l_i_vector_type = ir.VectorType.get([wg_rows], element_type)
                l_i_values = np.zeros(wg_rows, dtype=np.float16 if element_type == ir.F16Type.get() else np.float32)
                l_i_init_attr = ir.DenseElementsAttr.get(l_i_values, type=l_i_vector_type)
                l_i_init = arith.constant(l_i_vector_type, l_i_init_attr)

                # 3. Define acc_init: vector of shape [wg_rows, d_head] with zero values
                acc_vector_type = ir.VectorType.get([wg_rows, d_head], element_type)
                acc_values = np.zeros((wg_rows, d_head), dtype=np.float16 if element_type == ir.F16Type.get() else np.float32)
                acc_init_attr = ir.DenseElementsAttr.get(acc_values, type=acc_vector_type)
                acc_init = arith.constant(acc_vector_type, acc_init_attr)

                # Get n_ctx from k_load result type (first dimension size)
                k_load_result = k_load_op.results[0]
                k_vector_type = ir.VectorType(k_load_result.type)
                n_ctx = k_vector_type.shape[0]



                scale_vector_type = ir.VectorType.get([wg_rows], element_type)
                scale_values = np.full((wg_rows), scale_value, dtype=np.float16 if element_type == ir.F16Type.get() else np.float32)
                scale_init_attr = ir.DenseElementsAttr.get(scale_values, type=scale_vector_type)
                scale_vector = arith.constant(scale_vector_type, scale_init_attr)

                # Create loop bounds
                index_type = ir.IndexType.get()
                c0 = arith.constant(index_type, 0)
                c_n_ctx = arith.constant(index_type, n_ctx)
                c_tile_size = arith.constant(index_type, tile_size_value)

                # Create scf.for loop that iterates from 0 to n_ctx in steps of tile_size
                loop = scf.ForOp(c0, c_n_ctx, c_tile_size, [m_i_init, l_i_init, acc_init])

                with ir.InsertionPoint(loop.body):
                    # Get the loop induction variable and iter_args
                    loop_idx = loop.induction_variable
                    m_i = loop.inner_iter_args[0]
                    l_i = loop.inner_iter_args[1]
                    acc = loop.inner_iter_args[2]

                    # Load the current K tile: shape [tile_size, d_head]
                    # Use the same memref and indices as k_load, but replace second-to-last index with loop_idx
                    k_memref = k_load_op.operands[0]
                    k_tile_type = ir.VectorType.get([tile_size_value, d_head], element_type)

                    # Get the indices from original k_load (all operands except the first one which is the memref)
                    # and the last one which is the padding value
                    k_load_indices = list(k_load_op.operands[1:-1])

                    # Replace the second-to-last index with loop_idx
                    k_tile_indices = k_load_indices
                    k_tile_indices[-2] = loop_idx  # Assuming the reduction dimension is the last index before padding

                    # Get the padding value (last operand of k_load)
                    padding = k_load_op.operands[-1]

                    # Get in_bounds attribute if it exists
                    in_bounds = k_load_op.attributes.get("in_bounds", None)

                    k_perm_map = k_load_op.attributes.get("permutation_map", None)

                    # Create vector.transfer_read for K tile
                    k_tile = vector.TransferReadOp(
                        k_tile_type,
                        k_memref,
                        k_load_indices,
                        k_perm_map,
                        padding,
                        in_bounds=in_bounds
                    ).result
                    # print(f"k_tile: {k_tile}")

                    # Step 1: Transpose K tile from [tile_size, d_head] to [d_head, tile_size]
                    k_transpose_type = ir.VectorType.get([d_head, tile_size_value], element_type)
                    # vector.transpose with permutation [1, 0] swaps the two dimensions
                    k_transpose = vector.transpose(k_transpose_type, k_tile, [1, 0])
                    # print(f"k_transpose: {k_transpose}")

                    # Step 2: Compute Q * K_transpose using vector.contract
                    # Q shape: [wg_rows, d_head]
                    # K_transpose shape: [d_head, tile_size]
                    # Output shape: [wg_rows, tile_size]
                    # Contraction: Q[i, k] * K_transpose[k, j] -> QKT[i, j]
                    # indexing_maps: affine_map<(i, j, k) -> (i, k)>, affine_map<(i, j, k) -> (k, j)>, affine_map<(i, j, k) -> (i, j)>
                    # iterator_types: ["parallel", "parallel", "reduction"]

                    q_value = q_load_op.results[0]
                    qkt_type = ir.VectorType.get([wg_rows, tile_size_value], element_type)

                    # Create zero-initialized accumulator for the contraction
                    qkt_acc_values = np.zeros((wg_rows, tile_size_value), dtype=np.float16 if element_type == ir.F16Type.get() else np.float32)
                    qkt_acc_attr = ir.DenseElementsAttr.get(qkt_acc_values, type=qkt_type)
                    qkt_acc = arith.constant(qkt_type, qkt_acc_attr)

                    # Create affine maps for the contraction
                    affine_d0 = ir.AffineExpr.get_dim(0)
                    affine_d1 = ir.AffineExpr.get_dim(1)
                    affine_d2 = ir.AffineExpr.get_dim(2)

                    # Map for Q: (i, j, k) -> (i, k)
                    q_map = ir.AffineMap.get(3, 0, [affine_d0, affine_d2])
                    # Map for K_transpose: (i, j, k) -> (k, j)
                    k_map = ir.AffineMap.get(3, 0, [affine_d2, affine_d1])
                    # Map for output QKT: (i, j, k) -> (i, j)
                    out_map = ir.AffineMap.get(3, 0, [affine_d0, affine_d1])

                    indexing_maps = ir.ArrayAttr.get([
                        ir.AffineMapAttr.get(q_map),
                        ir.AffineMapAttr.get(k_map),
                        ir.AffineMapAttr.get(out_map)
                    ])

                    iterator_types = ir.ArrayAttr.get([
                        ir.Attribute.parse("#vector.iterator_type<parallel>"),
                        ir.Attribute.parse("#vector.iterator_type<parallel>"),
                        ir.Attribute.parse("#vector.iterator_type<reduction>")
                    ])

                    qkt = vector.contract(
                        qkt_type,
                        q_value,
                        k_transpose,
                        qkt_acc,
                        indexing_maps=indexing_maps,
                        iterator_types=iterator_types
                    )
                    # print(f"qkt: {qkt}")

                    # Step 3: Max reduction over the inner dimension of QKT
                    # QKT shape: [wg_rows, tile_size]
                    # Result shape: [wg_rows]
                    # We need to compute max along dimension 1 (tile_size dimension)

                    qkt_max = vector.multi_reduction(
                        kind="maxnumf",
                        source=qkt,
                        acc=m_i_init,
                        reduction_dims=[1]
                    )

                    # Step 4: Scale the max: qkt_max_scaled = qkt_max * scale
                    # Both have shape [wg_rows]
                    qkt_max_scaled = arith.mulf(qkt_max, scale_vector)

                    # Step 5: Compute m_ij = max(m_i, qkt_max_scaled)
                    # Both have shape [wg_rows]
                    m_ij = arith.maximumf(m_i, qkt_max_scaled)

                    # Step 6: Scale QKT matrix: qkt_scaled = qkt * scale_2d
                    # Need to broadcast scale from [wg_rows] to [wg_rows, tile_size]
                    scale_2d_type = ir.VectorType.get([wg_rows, tile_size_value], element_type)
                    scale_2d_values = np.full((wg_rows, tile_size_value), scale_value, dtype=np.float16 if element_type == ir.F16Type.get() else np.float32)
                    scale_2d_attr = ir.DenseElementsAttr.get(scale_2d_values, type=scale_2d_type)
                    scale_2d = arith.constant(scale_2d_type, scale_2d_attr)
                    qkt_scaled = arith.mulf(qkt, scale_2d)

                    # Step 7: Broadcast m_ij from [wg_rows] to [wg_rows, tile_size]
                    m_ij_bcasted_type = ir.VectorType.get([wg_rows, tile_size_value], element_type)
                    m_ij_bcasted = vector.broadcast(m_ij_bcasted_type, m_ij)

                    # Step 8: Center the scores: qkt_centered = qkt_scaled - m_ij_bcasted
                    qkt_centered = arith.subf(qkt_scaled, m_ij_bcasted)

                    # Step 9: Compute exponential: qkt_exp = exp(qkt_centered)
                    qkt_exp = math.exp(qkt_centered)

                    # Step 10: Sum reduction along inner dimension: l_ij = sum(qkt_exp, dim=1)
                    # Shape [wg_rows, tile_size] -> [wg_rows]
                    l_ij = vector.multi_reduction(
                        kind="add",
                        source=qkt_exp,
                        acc=l_i_init,
                        reduction_dims=[1]
                    )

                    # Step 11: Compute alpha = exp(m_i - m_ij)
                    m_diff = arith.subf(m_i, m_ij)
                    alpha = math.exp(m_diff)

                    # Step 12: Update l_i: l_i_updated = l_i * alpha + l_ij
                    l_i_scaled = arith.mulf(l_i, alpha)
                    l_i_updated = arith.addf(l_i_scaled, l_ij)

                    # Step 13: Broadcast alpha from [wg_rows] to [wg_rows, d_head]
                    alpha_bcasted_type = ir.VectorType.get([wg_rows, d_head], element_type)
                    alpha_bcasted = vector.broadcast(alpha_bcasted_type, alpha)

                    # Step 14: Update accumulator: acc_updated = acc * alpha_bcasted
                    acc_updated = arith.mulf(acc, alpha_bcasted)

                    # Step 15: Load the current V tile: shape [tile_size, d_head]
                    # Use the same memref and indices as v_load, but replace second-to-last index with loop_idx
                    v_memref = v_load_op.operands[0]
                    v_tile_type = ir.VectorType.get([tile_size_value, d_head], element_type)

                    # Get the indices from original v_load (all operands except the first one which is the memref)
                    # and the last one which is the padding value
                    v_load_indices = list(v_load_op.operands[1:-1])

                    # Replace the second-to-last index with loop_idx
                    v_tile_indices = v_load_indices
                    v_tile_indices[-2] = loop_idx  # Assuming the reduction dimension is the second-to-last index

                    # Get the padding value (last operand of v_load)
                    v_padding = v_load_op.operands[-1]

                    # Get in_bounds attribute if it exists
                    v_in_bounds = v_load_op.attributes.get("in_bounds", None)

                    v_perm_map = v_load_op.attributes.get("permutation_map", None)

                    # Create vector.transfer_read for V tile
                    v_tile = vector.TransferReadOp(
                        v_tile_type,
                        v_memref,
                        v_load_indices,
                        v_perm_map,
                        v_padding,
                        in_bounds=v_in_bounds
                    ).result

                    # Step 16: Compute attention-weighted values: pv_out = qkt_exp @ v_tile
                    # qkt_exp shape: [wg_rows, tile_size]
                    # v_tile shape: [tile_size, d_head]
                    # Output shape: [wg_rows, d_head]
                    # Contraction: qkt_exp[i, k] * v_tile[k, j] -> pv_out[i, j]

                    # Create affine maps for the contraction
                    affine_d0 = ir.AffineExpr.get_dim(0)
                    affine_d1 = ir.AffineExpr.get_dim(1)
                    affine_d2 = ir.AffineExpr.get_dim(2)

                    # Map for qkt_exp: (i, j, k) -> (i, k)
                    qkt_exp_map = ir.AffineMap.get(3, 0, [affine_d0, affine_d2])
                    # Map for v_tile: (i, j, k) -> (k, j)
                    v_map = ir.AffineMap.get(3, 0, [affine_d2, affine_d1])
                    # Map for output pv_out: (i, j, k) -> (i, j)
                    pv_out_map = ir.AffineMap.get(3, 0, [affine_d0, affine_d1])

                    indexing_maps_pv = ir.ArrayAttr.get([
                        ir.AffineMapAttr.get(qkt_exp_map),
                        ir.AffineMapAttr.get(v_map),
                        ir.AffineMapAttr.get(pv_out_map)
                    ])

                    iterator_types_pv = ir.ArrayAttr.get([
                        ir.Attribute.parse("#vector.iterator_type<parallel>"),
                        ir.Attribute.parse("#vector.iterator_type<parallel>"),
                        ir.Attribute.parse("#vector.iterator_type<reduction>")
                    ])

                    pv_out = vector.contract(
                        acc_vector_type,
                        qkt_exp,
                        v_tile,
                        acc_updated,
                        indexing_maps=indexing_maps_pv,
                        iterator_types=iterator_types_pv
                    )

                    # Yield the updated iter args
                    scf.yield_([m_ij, l_i_updated, pv_out])

            # Extract the final accumulator result (3rd output) from the loop
            final_output = loop.results[2]


            # Replace all uses of the original output operation with the final loop result
            output_op.results[0].replace_all_uses_with(final_output)

            # Erase the original output operation
            rewriter.erase_op(output_op)

            # Return the final output handle
            results.set_ops(op.new_output, [final_output.owner])
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
    q_load: ir.Value,
    k_load: ir.Value,
    v_load: ir.Value,
    scale: ir.Value,
    output: ir.Value,
    tile_size: int | ir.IntegerAttr,
) -> ir.Value:
    """Generate fused attention computation with inner tiling on bufferized IR.

    Args:
        q_load: Handle to Q load operation (vector.transfer_read)
        k_load: Handle to K load operation (vector.transfer_read)
        v_load: Handle to V load operation (vector.transfer_read)
        scale: Handle to scale constant operation (arith.constant)
        output: Handle to output operation to replace (vector.contract)
        tile_size: Tile size for the reduction dimension tiling (K/V sequence length)

    Returns:
        Handle to the new output operation
    """
    if not isinstance(tile_size, ir.IntegerAttr):
        tile_size = ir.IntegerAttr.get(ir.IntegerType.get_signless(64), tile_size)

    return GenerateFusedAttention(
        q_load, k_load, v_load, scale, output, tile_size=tile_size
    ).new_output
