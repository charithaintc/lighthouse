"""Generate MLIR payload for GPU attention operation."""

import math

from mlir import ir
from mlir.dialects import arith, bufferization, linalg, memref, tensor
from mlir.dialects import math as math_dialect

from lighthouse.utils.mlir import func_cif
from lighthouse.ingress.mlir_gen.utils import emit_buf_to_tensor


def _generic(ins, out, indexing_maps, iterator_types, body):
    """Emit a single-result ``linalg.generic`` over statically shaped tensors.

    `body` receives one scalar block argument per input followed by the output's,
    and returns the value to yield.
    """
    maps = ir.ArrayAttr.get([ir.AffineMapAttr.get(m) for m in indexing_maps])
    iterators = ir.ArrayAttr.get(
        [ir.Attribute.parse(f"#linalg.iterator_type<{it}>") for it in iterator_types]
    )
    op = linalg.GenericOp(
        result_tensors=[out.type],
        inputs=ins,
        outputs=[out],
        indexing_maps=maps,
        iterator_types=iterators,
    )
    element_type = ir.ShapedType(out.type).element_type
    block = op.regions[0].blocks.append(*([element_type] * (len(ins) + 1)))
    with ir.InsertionPoint(block):
        linalg.yield_([body(*block.arguments)])
    return op.results[0]


def generate_gpu_attention_payload(
    func_name: str,
    batch_size: int,
    n_head: int,
    n_ctx: int,
    d_head: int,
    dtype: ir.Type,
) -> ir.Module:
    """
    Generate MLIR module for attention payload.

    Computes attention:
    output = softmax(Q @ K^T / sqrt(d_head)) @ V

    The softmax is emitted in the flash-attention form -- ``max``, ``exp``, row
    ``sum`` and the ``P@V`` contraction as separate ops, with the normalizing
    divide *after* the contraction -- rather than as a `linalg.softmax`. See
    step 4 for why.

    Args:
        func_name: Name of the payload function
        batch_size: Batch size
        n_head: Number of attention heads
        n_ctx: Context length (sequence length)
        d_head: Head dimension
        dtype: MLIR element type (e.g., F32Type)

    Returns:
        MLIR module containing the attention payload function
    """
    mod = ir.Module.create()
    shape = (batch_size, n_head, n_ctx, d_head)
    memref_t = ir.MemRefType.get(shape, dtype)

    with ir.InsertionPoint(mod.body):
        # Collapse first 2 dimensions (batch_size, n_head) into a batch dimension
        # From (batch_size, n_head, n_ctx, d_head) to (batch_size*n_head, n_ctx, d_head)
        batch_dim = batch_size * n_head
        collapsed_shape_3d = (batch_dim, n_ctx, d_head)
        memref_3d_t = ir.MemRefType.get(collapsed_shape_3d, dtype)

        # Function signature: payload(output, Q, K, V)
        @func_cif(memref_t, memref_t, memref_t, memref_t, name=func_name)
        def payload(output, Q_arg, K_arg, V_arg):
            # Collapse memrefs from 4D to 3D
            Q_3d_memref = memref.collapse_shape(
                memref_3d_t,
                Q_arg,
                reassociation=[[0, 1], [2], [3]],
            )
            K_3d_memref = memref.collapse_shape(
                memref_3d_t,
                K_arg,
                reassociation=[[0, 1], [2], [3]],
            )
            V_3d_memref = memref.collapse_shape(
                memref_3d_t,
                V_arg,
                reassociation=[[0, 1], [2], [3]],
            )
            output_3d_memref = memref.collapse_shape(
                memref_3d_t,
                output,
                reassociation=[[0, 1], [2], [3]],
            )

            # Convert 3D memrefs to tensors
            Q_3d = emit_buf_to_tensor(Q_3d_memref, restrict=True)
            K_3d = emit_buf_to_tensor(K_3d_memref, restrict=True)
            V_3d = emit_buf_to_tensor(V_3d_memref, restrict=True)

            # Step 1: Transpose K to get K^T
            # Permute from (batch_dim, n_ctx, d_head) to (batch_dim, d_head, n_ctx)
            kt_shape_3d = (batch_dim, d_head, n_ctx)
            kt_init = tensor.empty(kt_shape_3d, dtype)
            K_transposed = linalg.transpose(K_3d, outs=[kt_init], permutation=[0, 2, 1])

            # Step 2: Compute Q @ K^T using batch_matmul
            # Q: (batch_dim, n_ctx, d_head) @ K^T: (batch_dim, d_head, n_ctx)
            # Result: (batch_dim, n_ctx, n_ctx)
            qkt_shape_3d = (batch_dim, n_ctx, n_ctx)
            qkt_init = tensor.empty(qkt_shape_3d, dtype)
            # Initialize with zeros for matmul accumulation
            zero = arith.constant(dtype, 0.0)
            qkt_init_filled = linalg.fill(zero, outs=[qkt_init])

            # Batch matmul: Q @ K^T
            qkt = linalg.batch_matmul(Q_3d, K_transposed, outs=[qkt_init_filled])

            # Step 3: Scale by 1/sqrt(d_head)
            scale_factor = 1.0 / math.sqrt(d_head)
            scale_const = arith.constant(dtype, scale_factor)

            # Create a tensor filled with the scale factor
            scale_tensor_init = tensor.empty(qkt_shape_3d, dtype)
            scale_tensor = linalg.fill(scale_const, outs=[scale_tensor_init])

            # Elementwise multiply qkt with scale tensor
            scaled_qkt_init = tensor.empty(qkt_shape_3d, dtype)
            scaled_qkt = linalg.mul(qkt, scale_tensor, outs=[scaled_qkt_init])

            # Step 4: softmax over the last dimension, written in the
            # flash-attention form -- with the normalizing divide deferred past the
            # P@V contraction:
            #
            #   m   = max_k s          l   = sum_k P
            #   P   = exp(s - m)       O   = P @ V         out = O / l
            #
            # This is algebraically identical to `softmax(s) @ V`: dividing by the
            # per-row `l` commutes with a contraction that reduces the *other*
            # axis. Writing it this way (rather than as `linalg.softmax`, whose
            # decomposition normalizes before the contraction) leaves the
            # `max -> exp -> {sum, P@V}` dependency chain explicit, which is what
            # `transform_ext.fuse_dependant_reduction_ops` consumes to build the
            # online one-pass loop.
            d0, d1, d2 = (ir.AffineDimExpr.get(i) for i in range(3))
            # (batch, row, col) -> (batch, row, col) and -> (batch, row): the
            # per-row statistics are broadcast over the reduced axis.
            elementwise_map = ir.AffineMap.get(3, 0, [d0, d1, d2])
            row_map = ir.AffineMap.get(3, 0, [d0, d1])
            row_shape_3d = (batch_dim, n_ctx)

            # m = max_k s
            neg_inf = arith.constant(dtype, float("-inf"))
            m_init = linalg.fill(neg_inf, outs=[tensor.empty(row_shape_3d, dtype)])
            row_max = _generic(
                [scaled_qkt],
                m_init,
                [elementwise_map, row_map],
                ["parallel", "parallel", "reduction"],
                lambda s, acc: arith.maximumf(s, acc),
            )

            # P = exp(s - m), read by both the row sum and the P@V contraction.
            probs = _generic(
                [scaled_qkt, row_max],
                tensor.empty(qkt_shape_3d, dtype),
                [elementwise_map, row_map, elementwise_map],
                ["parallel", "parallel", "parallel"],
                lambda s, m, out: math_dialect.exp(arith.subf(s, m)),
            )

            # l = sum_k P
            l_init = linalg.fill(zero, outs=[tensor.empty(row_shape_3d, dtype)])
            row_sum = _generic(
                [probs],
                l_init,
                [elementwise_map, row_map],
                ["parallel", "parallel", "reduction"],
                lambda p, acc: arith.addf(p, acc),
            )

            # Step 5: O = P @ V, still unnormalized.
            # probs: (batch_dim, n_ctx, n_ctx) @ V: (batch_dim, n_ctx, d_head)
            # Result: (batch_dim, n_ctx, d_head)
            output_3d_init = tensor.empty(collapsed_shape_3d, dtype)
            output_3d_init_filled = linalg.fill(zero, outs=[output_3d_init])
            unnormalized = linalg.batch_matmul(
                probs, V_3d, outs=[output_3d_init_filled]
            )

            # Step 6: out = O / l, the deferred normalization. `l` is broadcast
            # over d_head, the contraction's free axis.
            result_3d = _generic(
                [unnormalized, row_sum],
                tensor.empty(collapsed_shape_3d, dtype),
                [elementwise_map, row_map, elementwise_map],
                ["parallel", "parallel", "parallel"],
                lambda o, denom, out: arith.divf(o, denom),
            )

            # Materialize 3D result back to 3D output memref
            bufferization.materialize_in_destination(
                None, result_3d, output_3d_memref, restrict=True, writable=True
            )

    return mod
