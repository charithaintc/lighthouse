"""
Payload generator for the dynamic-shape XeGPU matmul example.

Kept local to the example (rather than added to
``lighthouse.ingress.mlir_gen.gpu_matmul_payload``) so that experiments with a
dynamic reduction dimension cannot regress the existing static payloads.

The emitted payload is a plain matmul whose reduction dim K is `?`:

    func.func @payload(%C: memref<MxNxf32>,
                       %A: memref<Mx?xf16>,
                       %B: memref<?xNxf16>) {
      %tc = bufferization.to_tensor %C restrict writable
      %ta = bufferization.to_tensor %A restrict
      %tb = bufferization.to_tensor %B restrict
      %d  = linalg.matmul ins(%ta, %tb) outs(%tc)
      bufferization.materialize_in_destination %d in restrict writable %C
    }

M and N stay static, so the parallel iteration domain is fully static and the
MxN accumulator never needs a dynamically-sized `tensor.empty`. The only runtime
extent is K, which shows up as the trip count of the reduction loop.

Argument order is (C, A, B) to match ``XeGPUMatMul`` in ``matmul.py`` and the
buffer order that ``Runner`` expects.
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
    """
    Generate payload module for a matmul with a dynamic reduction dimension.

    Args:
        func_name: Name of the emitted payload function.
        M, N: Static parallel dimensions. K is always dynamic.
        ab_type: Element type of A and B.
        acc_type: Element type the matmul accumulates in.
        result_type: Element type of the C buffer.
        accumulate_c: Compute C += A*B instead of C = A*B.
    """
    dyn = ir.ShapedType.get_dynamic_size()

    mod = ir.Module.create()
    memref_c_t = ir.MemRefType.get((M, N), result_type)
    memref_a_t = ir.MemRefType.get((M, dyn), ab_type)
    memref_b_t = ir.MemRefType.get((dyn, N), ab_type)

    with ir.InsertionPoint(mod.body):
        # Argument order: output, then inputs.
        @func_cif(memref_c_t, memref_a_t, memref_b_t, name=func_name)
        def payload(c_buf, a_buf, b_buf):
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

            # ins are MxK and KxN with K dynamic; outs is the static MxN acc.
            out = linalg.matmul(a_tensor, b_tensor, outs=(acc_tensor,))

            if acc_type != result_type:
                out = convert_float_type(out, tensor.empty((M, N), result_type))

            bufferization.materialize_in_destination(
                None, out, c_buf, restrict=True, writable=True
            )

    return mod
