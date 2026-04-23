"""Generate MLIR payload for GPU fused attention operation."""

import math

from mlir import ir
from mlir.dialects import arith, bufferization, linalg, tensor

from lighthouse.utils.mlir import func_cif
from lighthouse.ingress.mlir_gen.gpu_utils import emit_gpu_util_funcs
from lighthouse.ingress.mlir_gen.utils import emit_buf_to_tensor


def generate_gpu_fused_attention_payload(
    func_name: str,
    Z: int,
    H: int,
    n_ctx: int,
    n_head: int,
    dtype: ir.Type,
) -> ir.Module:
    """
    Generate MLIR module for fused attention payload.

    Computes fused attention:
    output = softmax(Q @ K^T / sqrt(n_head)) @ V

    Args:
        func_name: Name of the payload function
        Z: Batch size
        H: Number of attention heads
        n_ctx: Context length (sequence length)
        n_head: Head dimension
        dtype: MLIR element type (e.g., F32Type)

    Returns:
        MLIR module containing the fused attention payload function
    """
    mod = ir.Module.create()
    shape = (Z, H, n_ctx, n_head)
    memref_t = ir.MemRefType.get(shape, dtype)

    with ir.InsertionPoint(mod.body):
        # Function signature: payload(output, Q, K, V)
        @func_cif(memref_t, memref_t, memref_t, memref_t, name=func_name)
        def payload(output, Q_arg, K_arg, V_arg):
            # Convert memrefs to tensors
            emit_buf_to_tensor(output, restrict=True, writable=True)
            Q_tensor = emit_buf_to_tensor(Q_arg, restrict=True)
            K_tensor = emit_buf_to_tensor(K_arg, restrict=True)
            V_tensor = emit_buf_to_tensor(V_arg, restrict=True)

            # Collapse first 3 dimensions (Z, H, n_ctx) into a single batch dimension
            # From (Z, H, n_ctx, n_head) to (Z*H*n_ctx, n_head)
            batch_dim = Z * H * n_ctx
            collapsed_shape_2d = (batch_dim, n_head)

            Q_2d = tensor.collapse_shape(
                ir.RankedTensorType.get(collapsed_shape_2d, dtype),
                Q_tensor,
                reassociation=[[0, 1, 2], [3]],
            )
            K_2d = tensor.collapse_shape(
                ir.RankedTensorType.get(collapsed_shape_2d, dtype),
                K_tensor,
                reassociation=[[0, 1, 2], [3]],
            )
            V_2d = tensor.collapse_shape(
                ir.RankedTensorType.get(collapsed_shape_2d, dtype),
                V_tensor,
                reassociation=[[0, 1, 2], [3]],
            )

            # Step 1: Transpose K to get K^T
            # Transpose from (batch_dim, n_head) to (n_head, batch_dim)
            kt_shape_2d = (n_head, batch_dim)
            kt_init = tensor.empty(kt_shape_2d, dtype)
            K_transposed = linalg.transpose(K_2d, outs=[kt_init], permutation=[1, 0])

            # Step 2: Compute Q @ K^T using matmul
            # Q: (batch_dim, n_head) @ K^T: (n_head, batch_dim)
            # Result: (batch_dim, batch_dim)
            qkt_shape_2d = (batch_dim, batch_dim)
            qkt_init = tensor.empty(qkt_shape_2d, dtype)
            # Initialize with zeros for matmul accumulation
            zero = arith.constant(dtype, 0.0)
            qkt_init_filled = linalg.fill(zero, outs=[qkt_init])

            # Matmul: Q @ K^T
            qkt = linalg.matmul(Q_2d, K_transposed, outs=[qkt_init_filled])

            # # Step 3: Scale by 1/sqrt(n_head)
            # scale_factor = 1.0 / math.sqrt(n_head)
            # scale_const = arith.constant(dtype, scale_factor)

            # # Create a tensor filled with the scale factor
            # scale_tensor_init = tensor.empty(qkt_shape_2d, dtype)
            # scale_tensor = linalg.fill(scale_const, outs=[scale_tensor_init])

            # # Elementwise multiply qkt with scale tensor
            # scaled_qkt_init = tensor.empty(qkt_shape_2d, dtype)
            # scaled_qkt = linalg.mul(qkt, scale_tensor, outs=[scaled_qkt_init])

            # Step 4: Apply softmax along the last dimension (dim=1 in 2D)
            softmax_init = tensor.empty(qkt_shape_2d, dtype)
            attention_weights = linalg.softmax(
                result=[ir.RankedTensorType.get(qkt_shape_2d, dtype)],
                input=qkt,
                output=softmax_init,
                dimension=1,
            )

            # Step 5: Multiply attention weights by V using matmul
            # attention_weights: (batch_dim, batch_dim) @ V: (batch_dim, n_head)
            # Result: (batch_dim, n_head)
            output_2d_init = tensor.empty(collapsed_shape_2d, dtype)
            output_2d_init_filled = linalg.fill(zero, outs=[output_2d_init])

            result_2d = linalg.matmul(
                attention_weights, V_2d, outs=[output_2d_init_filled]
            )

            # Expand back to 4D: (Z*H*n_ctx, n_head) -> (Z, H, n_ctx, n_head)
            result = tensor.expand_shape(
                ir.RankedTensorType.get(shape, dtype),
                result_2d,
                reassociation=[[0, 1, 2], [3]],
                output_shape=[],
                static_output_shape=shape,
            )

            # Materialize result back to output memref
            bufferization.materialize_in_destination(
                None, result, output, restrict=True, writable=True
            )

        # Emit utility functions for GPU memory management
        emit_gpu_util_funcs(dtype, rank=4)

    return mod
