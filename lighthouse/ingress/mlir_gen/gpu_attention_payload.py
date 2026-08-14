"""Generate MLIR payload for GPU attention operation."""

import math

from mlir import ir
from mlir.dialects import arith, bufferization, linalg, memref, tensor
from mlir.dialects import math as math_dialect

from lighthouse.utils.mlir import func_cif
from lighthouse.ingress.mlir_gen.utils import (
    affine_map,
    emit_buf_to_tensor,
    parallel,
    reduction,
)


def generate_gpu_attention_payload(
    func_name: str,
    Z: int,
    H: int,
    n_ctx: int,
    n_head: int,
    dtype: ir.Type,
) -> ir.Module:
    """
    Generate MLIR module for attention payload.

    Computes attention:
    output = softmax(Q @ K^T / sqrt(n_head)) @ V

    The softmax is written out explicitly and, unlike `linalg.softmax`, with the
    normalizing divide deferred *past* the attention_weights @ V contraction:

        S[b, i, j] = (Q @ K^T)[b, i, j] * scale
        m[b, i]    = max_j S[b, i, j]
        p[b, i, j] = exp(S[b, i, j] - m[b, i])
        l[b, i]    = sum_j p[b, i, j]
        O[b, i, d] = sum_j p[b, i, j] * V[b, j, d]
        out        = O[b, i, d] / l[b, i]

    which is the same value as `softmax(S) @ V` because `l` does not depend on
    the contracted axis `j`. This is the form fused attention needs: `p` feeds
    the contraction directly, so both `l` and `O` are reductions of `p` over `j`
    and can be turned into running accumulators of a single loop over `j`.

    `p` is a single op shared by both `l` and `O`; the reduction fusion clones it
    so that each of those two reductions has its own elementwise term to move
    into the loop.

    Args:
        func_name: Name of the payload function
        Z: Batch size
        H: Number of attention heads
        n_ctx: Context length (sequence length)
        n_head: Head dimension
        dtype: MLIR element type (e.g., F32Type)

    Returns:
        MLIR module containing the attention payload function
    """
    mod = ir.Module.create()
    shape = (Z, H, n_ctx, n_head)
    memref_t = ir.MemRefType.get(shape, dtype)

    with ir.InsertionPoint(mod.body):
        # Collapse first 2 dimensions (Z, H) into a batch dimension
        # From (Z, H, n_ctx, n_head) to (Z*H, n_ctx, n_head)
        batch_dim = Z * H
        collapsed_shape_3d = (batch_dim, n_ctx, n_head)
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
            # Permute from (batch_dim, n_ctx, n_head) to (batch_dim, n_head, n_ctx)
            kt_shape_3d = (batch_dim, n_head, n_ctx)
            kt_init = tensor.empty(kt_shape_3d, dtype)
            K_transposed = linalg.transpose(K_3d, outs=[kt_init], permutation=[0, 2, 1])

            # Step 2: Compute Q @ K^T using batch_matmul
            # Q: (batch_dim, n_ctx, n_head) @ K^T: (batch_dim, n_head, n_ctx)
            # Result: (batch_dim, n_ctx, n_ctx)
            qkt_shape_3d = (batch_dim, n_ctx, n_ctx)
            qkt_init = tensor.empty(qkt_shape_3d, dtype)
            # Initialize with zeros for matmul accumulation
            zero = arith.constant(dtype, 0.0)
            qkt_init_filled = linalg.fill(zero, outs=[qkt_init])

            # Batch matmul: Q @ K^T
            qkt = linalg.batch_matmul(Q_3d, K_transposed, outs=[qkt_init_filled])

            # Step 3: Scale by 1/sqrt(n_head)
            scale_factor = 1.0 / math.sqrt(n_head)
            scale_const = arith.constant(dtype, scale_factor)

            # Create a tensor filled with the scale factor
            scale_tensor_init = tensor.empty(qkt_shape_3d, dtype)
            scale_tensor = linalg.fill(scale_const, outs=[scale_tensor_init])

            # Elementwise multiply qkt with scale tensor
            scaled_qkt_init = tensor.empty(qkt_shape_3d, dtype)
            scaled_qkt = linalg.mul(qkt, scale_tensor, outs=[scaled_qkt_init])

            # Step 4: Softmax numerator, written out over the iteration space
            # (b, i, j) with j -- the key/value axis -- reduced.
            all_parallel_map = affine_map(
                3,
                [
                    ir.AffineDimExpr.get(0),
                    ir.AffineDimExpr.get(1),
                    ir.AffineDimExpr.get(2),
                ],
            )
            row_map = affine_map(3, [ir.AffineDimExpr.get(0), ir.AffineDimExpr.get(1)])
            row_shape_2d = (batch_dim, n_ctx)

            # 4a) Running max: m[b, i] = max_j S[b, i, j]
            neg_inf = arith.constant(dtype, float("-inf"))
            max_init = linalg.fill(neg_inf, outs=[tensor.empty(row_shape_2d, dtype)])

            @linalg.generic(
                [scaled_qkt],
                [max_init],
                [all_parallel_map, row_map],
                [parallel, parallel, reduction],
            )
            def row_max(s, acc):
                return arith.MaximumFOp(s, acc)

            # 4b) Shifted exponential: p[b, i, j] = exp(S[b, i, j] - m[b, i])
            #
            # A single op, shared by both consumers below.
            # `transform.structured.fuse_dependant_reduction_ops` needs each
            # consumer reduction to have its own elementwise term to move into the
            # reduction loop, and clones this one as it fuses them.
            @linalg.generic(
                [scaled_qkt, row_max],
                [tensor.empty(qkt_shape_3d, dtype)],
                [all_parallel_map, row_map, all_parallel_map],
                [parallel, parallel, parallel],
            )
            def attention_weights(s, m, _out):
                return math_dialect.ExpOp(arith.SubFOp(s, m).result)

            # 4c) Normalizer: l[b, i] = sum_j p[b, i, j]
            sum_init = linalg.fill(zero, outs=[tensor.empty(row_shape_2d, dtype)])

            @linalg.generic(
                [attention_weights],
                [sum_init],
                [all_parallel_map, row_map],
                [parallel, parallel, reduction],
            )
            def row_sum(p, acc):
                return arith.AddFOp(p, acc)

            # Step 5: Contract the (unnormalized) weights with V using batch_matmul
            # p: (batch_dim, n_ctx, n_ctx) @ V: (batch_dim, n_ctx, n_head)
            # Result: (batch_dim, n_ctx, n_head)
            pv_init = tensor.empty(collapsed_shape_3d, dtype)
            pv_init_filled = linalg.fill(zero, outs=[pv_init])

            pv = linalg.batch_matmul(attention_weights, V_3d, outs=[pv_init_filled])

            # Step 6: Deferred normalization: out[b, i, d] = O[b, i, d] / l[b, i]
            output_3d_init = tensor.empty(collapsed_shape_3d, dtype)

            @linalg.generic(
                [pv, row_sum],
                [output_3d_init],
                [all_parallel_map, row_map, all_parallel_map],
                [parallel, parallel, parallel],
            )
            def result_3d(o, norm, _out):
                return arith.DivFOp(o, norm)

            # Materialize 3D result back to 3D output memref
            bufferization.materialize_in_destination(
                None, result_3d, output_3d_memref, restrict=True, writable=True
            )

    return mod
