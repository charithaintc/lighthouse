"""Generate MLIR payload for GPU softmax operation."""

from mlir import ir
from mlir.dialects import linalg, bufferization, arith, tensor, math

from lighthouse.utils.mlir import func_cif
from lighthouse.ingress.mlir_gen.gpu_utils import (
    emit_gpu_util_funcs,
    emit_buf_to_tensor,
)


def generate_gpu_softmax_payload(
    func_name: str,
    M: int,
    N: int,
    dtype: ir.Type,
) -> ir.Module:
    """
    Generate MLIR module for softmax payload.

    Computes softmax along the last dimension (rows):
    output[i, j] = exp(input[i, j] - max_i) / sum_i(exp(input[i, j] - max_i))

    where max_i and sum_i are computed over row i.

    Args:
        func_name: Name of the payload function
        M: Number of rows
        N: Number of columns
        dtype: MLIR element type (e.g., F32Type)

    Returns:
        MLIR module containing the softmax payload function
    """
    mod = ir.Module.create()
    shape = (M, N)
    memref_t = ir.MemRefType.get(shape, dtype)

    with ir.InsertionPoint(mod.body):
        # Function signature: payload(output, input)
        @func_cif(memref_t, memref_t, name=func_name)
        def payload(output, input_arg):
            # Convert memrefs to tensors
            emit_buf_to_tensor(output, restrict=True, writable=True)
            input_tensor = emit_buf_to_tensor(input_arg, restrict=True)

            # Define affine maps for indexing
            # #map = affine_map<(d0, d1) -> (d0, d1)>  (identity 2D)
            # #map1 = affine_map<(d0, d1) -> (d0)>     (broadcast/reduce along d1)
            d0 = ir.AffineDimExpr.get(0)
            d1 = ir.AffineDimExpr.get(1)
            map_2d = ir.AffineMap.get(2, 0, [d0, d1])
            map_1d = ir.AffineMap.get(2, 0, [d0])

            # Step 1: Find max - linalg.generic reduction
            neg_inf = arith.constant(dtype, float("-inf"))
            max_init = tensor.empty((M,), dtype)
            max_filled = linalg.fill(neg_inf, outs=[max_init])

            @linalg.generic(
                [input_tensor],  # inputs
                [max_filled],  # outputs
                [map_2d, map_1d],  # indexing_maps
                [
                    linalg.IteratorType.parallel,
                    linalg.IteratorType.reduction,
                ],  # iterator_types
            )
            def row_max(in_val, acc):
                return arith.maximumf(in_val, acc)

            # Step 2: Subtract max (broadcast) - linalg.generic elementwise
            output_init = tensor.empty((M, N), dtype)

            @linalg.generic(
                [input_tensor, row_max],  # inputs
                [output_init],  # outputs
                [map_2d, map_1d, map_2d],  # indexing_maps
                [
                    linalg.IteratorType.parallel,
                    linalg.IteratorType.parallel,
                ],  # iterator_types
            )
            def shifted(in_val, max_val, out):
                return arith.subf(in_val, max_val)

            # Step 3: Compute exp - linalg.generic elementwise
            @linalg.generic(
                [shifted],  # inputs
                [output_init],  # outputs
                [map_2d, map_2d],  # indexing_maps
                [
                    linalg.IteratorType.parallel,
                    linalg.IteratorType.parallel,
                ],  # iterator_types
            )
            def exp_vals(in_val, out):
                return math.exp(in_val)

            # Step 4: Sum exp values - linalg.generic reduction
            sum_init = tensor.empty((M,), dtype)
            zero = arith.constant(dtype, 0.0)
            sum_filled = linalg.fill(zero, outs=[sum_init])

            @linalg.generic(
                [exp_vals],  # inputs
                [sum_filled],  # outputs
                [map_2d, map_1d],  # indexing_maps
                [
                    linalg.IteratorType.parallel,
                    linalg.IteratorType.reduction,
                ],  # iterator_types
            )
            def row_sum(in_val, acc):
                return arith.addf(in_val, acc)

            # Step 5: Divide by sum (broadcast) - linalg.generic elementwise
            @linalg.generic(
                [exp_vals, row_sum],  # inputs
                [output_init],  # outputs
                [map_2d, map_1d, map_2d],  # indexing_maps
                [
                    linalg.IteratorType.parallel,
                    linalg.IteratorType.parallel,
                ],  # iterator_types
            )
            def result(exp_val, sum_val, out):
                return arith.divf(exp_val, sum_val)

            # Materialize result back to output memref
            bufferization.materialize_in_destination(
                None, result, output, restrict=True, writable=True
            )

        # Emit utility functions for GPU memory management
        emit_gpu_util_funcs(dtype, rank=2)

    return mod
