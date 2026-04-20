"""Generate MLIR payload for GPU fused attention operation."""

from mlir import ir
from mlir.dialects import bufferization, tensor

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

            # TODO: Implement fused attention computation
            # This will involve:
            # 1. Q @ K^T (batch matmul with transpose)
            # 2. Scale by 1/sqrt(n_head)
            # 3. Softmax along last dimension
            # 4. Result @ V (batch matmul)

            # Placeholder: create empty output tensor
            output_init = tensor.empty(shape, dtype)
            result = output_init

            # Materialize result back to output memref
            bufferization.materialize_in_destination(
                None, result, output, restrict=True, writable=True
            )

        # Emit utility functions for GPU memory management
        emit_gpu_util_funcs(dtype, rank=4)

    return mod
