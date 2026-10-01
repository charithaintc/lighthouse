"""Generate a matmul payload with static M/N and a runtime-sized K.

Arguments are ordered (C, A, B) to match the runner's buffer order.
"""

from mlir import ir
from mlir.dialects import arith, bufferization, linalg, tensor

from lighthouse.ingress.mlir_gen.generic import convert_float_type
from lighthouse.ingress.mlir_gen.utils import emit_buf_to_tensor
from lighthouse.utils.mlir import func_cif


def generate_dyn_shape_matmul_payload(
    func_name: str,
    M: int,
    N: int,
    ab_type: ir.Type,
    acc_type: ir.Type,
    result_type: ir.Type,
    accumulate_c: bool = True,
) -> ir.Module:
    """Build a dynamic-K matmul payload, optionally accumulating into C."""
    dyn = ir.ShapedType.get_dynamic_size()

    mod = ir.Module.create()
    memref_c_t = ir.MemRefType.get((M, N), result_type)
    memref_a_t = ir.MemRefType.get((M, dyn), ab_type)
    memref_b_t = ir.MemRefType.get((dyn, N), ab_type)

    with ir.InsertionPoint(mod.body):

        @func_cif(memref_c_t, memref_a_t, memref_b_t, name=func_name)
        def payload(c_buf, a_buf, b_buf):
            """Convert buffers to tensors, multiply, and write back to C."""
            c_tensor = emit_buf_to_tensor(c_buf, restrict=True, writable=True)
            a_tensor = emit_buf_to_tensor(a_buf, restrict=True)
            b_tensor = emit_buf_to_tensor(b_buf, restrict=True)

            if accumulate_c:
                acc_tensor = c_tensor
                if acc_type != result_type:
                    acc_tensor = convert_float_type(
                        acc_tensor, tensor.empty((M, N), acc_type)
                    )
            else:
                zero = arith.constant(acc_type, 0.0)
                acc_tensor = linalg.fill(zero, outs=[tensor.empty((M, N), acc_type)])

            # Accumulate over dynamic K into the static MxN result.
            out = linalg.matmul(a_tensor, b_tensor, outs=(acc_tensor,))

            if acc_type != result_type:
                out = convert_float_type(out, tensor.empty((M, N), result_type))

            bufferization.materialize_in_destination(
                None, out, c_buf, restrict=True, writable=True
            )

    return mod
