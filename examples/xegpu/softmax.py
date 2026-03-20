# RUN: %PYTHON %s --dump-kernel=xegpu-wg | FileCheck %s
# CHECK: module attributes {gpu.container_module} {

"""
XeGPU softmax benchmark.
"""

import argparse
import ctypes
from typing import Optional
from functools import cached_property

import numpy as np
from mlir import ir
from mlir.execution_engine import ExecutionEngine
from mlir.dialects import linalg, gpu, bufferization, arith, tensor, func, math
from mlir.dialects import transform
from mlir.dialects.transform import structured, loop, xegpu

from lighthouse.workload import benchmark, get_bench_wrapper_schedule
from lighthouse.utils.memref import to_ctype as memref_to_ctype
from lighthouse.utils.numpy import numpy_to_ctype
from lighthouse.utils.mlir import func_cif
from lighthouse.ingress.mlir_gen import get_mlir_elem_type
from lighthouse.ingress.mlir_gen.gpu_utils import emit_gpu_util_funcs, emit_buf_to_tensor
from lighthouse.pipeline.helper import (
    apply_registered_pass,
    canonicalize,
    match,
)
from mlir.dialects.transform import bufferization as transform_bufferization
from mlir.dialects.bufferization import LayoutMapOption

from xegpu_workload import XeGPUWorkload

def match_and_split(*args, nhandles=1, **kwargs):
    """Helper function that splits matched handles."""
    matched = match(*args, **kwargs)
    anytype = transform.AnyOpType.get()
    matched_ops = transform.split_handle((anytype,) * nhandles, matched)
    if nhandles == 1:
        matched_ops = [matched_ops]
    return matched_ops


def softmax_complexity(M: int, N: int, nbytes: int):
    """
    Complexity of softmax operation.
    
    For each row:
    - O(N) to find max
    - O(N) to compute exp(x - max) and sum
    - O(N) to normalize
    Total: 3*N operations per row, but with transcendental (exp) operations
    """
    # Approximation: 5 FLOPs per element (max, sub, exp, sum, div)
    # exp is expensive but we count it as ~1 FLOP for simplicity
    flop_count = M * N * 5
    memory_reads = M * N * nbytes  # read input
    memory_writes = M * N * nbytes  # write output
    return flop_count, memory_reads, memory_writes


class XeGPUSoftmax(XeGPUWorkload):
    """
    Softmax workload on XeGPU.

    Computes softmax along the last dimension (rows):
    output[i, j] = exp(input[i, j] - max_i) / sum_i(exp(input[i, j] - max_i))

    where max_i and sum_i are computed over row i.
    """

    def __init__(
        self,
        M: int,
        N: int,
        dtype: str = "f32",
    ):
        super().__init__()
        self.M = M
        self.N = N
        self.shape = (M, N)
        assert dtype == "f32", "Only f32 type is supported for softmax"
        self.dtype_str = dtype
        type_str_to_numpy = {
            "f16": np.float16,
            "f32": np.float32,
        }
        self.dtype = type_str_to_numpy[dtype]

    @cached_property
    def _initial_host_arrays(self) -> tuple[np.ndarray]:
        """Generate initial values on host with numpy."""
        np.random.seed(42)
        # Use values in range [-0.5, 0.5] to avoid numerical issues
        input_arr = np.random.uniform(-0.5, 0.5, self.shape).astype(self.dtype)
        return (input_arr,)

    @cached_property
    def _reference_solution(self) -> np.ndarray:
        """Compute reference solution on host with numpy."""
        (input_arr,) = self._initial_host_arrays
        # Use float32 for computation
        x = input_arr.astype(np.float32)
        # Compute softmax along axis 1 (each row independently)
        # Numerically stable version: subtract max before exp
        max_vals = np.max(x, axis=1, keepdims=True)
        exp_vals = np.exp(x - max_vals)
        sum_vals = np.sum(exp_vals, axis=1, keepdims=True)
        output = exp_vals / sum_vals
        return output.astype(self.dtype)

    def _get_input_arrays(
        self, execution_engine: ExecutionEngine
    ) -> list[ctypes.Structure]:
        # Allocate device memory for input and output
        input_gpu = self._allocate_array(
            "input", self.shape, self.dtype_str, execution_engine
        )
        output_gpu = self._allocate_array(
            "output", self.shape, self.dtype_str, execution_engine
        )

        # Copy input to device
        (input_host,) = self._initial_host_arrays
        copy_fn = f"gpu_copy_2d_{self.dtype_str}"
        execution_engine.invoke(
            copy_fn, numpy_to_ctype(input_host), memref_to_ctype(input_gpu)
        )

        # Return memrefs: [output, input]
        return [output_gpu, input_gpu]

    def check_correctness(
        self, execution_engine: ExecutionEngine, verbose: int = 0
    ) -> bool:
        # Copy result from device to host
        output_gpu = self.gpu_memrefs[("output", self.dtype_str)]
        output_host = np.zeros(self.shape, dtype=self.dtype)
        execution_engine.invoke(
            f"gpu_copy_2d_{self.dtype_str}",
            memref_to_ctype(output_gpu),
            numpy_to_ctype(output_host),
        )

        output_ref = self._reference_solution
        output_computed = output_host.astype(np.float32)
        
        if verbose > 1:
            print("Reference solution (first 5 rows):")
            print(output_ref[:5])
            print("Computed solution (first 5 rows):")
            print(output_computed[:5])

        # Check row sums are close to 1.0
        row_sums = np.sum(output_computed, axis=1)
        sums_ok = np.allclose(row_sums, 1.0, rtol=1e-5, atol=1e-6)
        
        # Check values match reference
        values_ok = np.allclose(output_computed, output_ref, rtol=1e-4, atol=1e-6)
        
        success = sums_ok and values_ok

        if verbose:
            if success:
                print("PASSED")
            else:
                print("FAILED!")
                if not sums_ok:
                    print(f"  Row sums check failed. Min: {row_sums.min():.6f}, Max: {row_sums.max():.6f}")
                if not values_ok:
                    max_diff = np.abs(output_computed - output_ref).max()
                    print(f"  Values mismatch. Max abs diff: {max_diff:.6e}")
        return success

    def get_complexity(self) -> tuple[int, int, int]:
        nbytes = np.dtype(self.dtype).itemsize
        return softmax_complexity(self.M, self.N, nbytes)

    def payload_module(self) -> ir.Module:
        """Generate MLIR module for softmax payload."""
        mod = ir.Module.create()
        dtype = get_mlir_elem_type(self.dtype_str)
        memref_t = ir.MemRefType.get(self.shape, dtype)
        
        with ir.InsertionPoint(mod.body):
            # Function signature: payload(output, input)
            @func_cif(memref_t, memref_t, name=self.payload_function_name)
            def payload(output, input_arg):
                # Convert memrefs to tensors
                output_tensor = emit_buf_to_tensor(output, restrict=True, writable=True)
                input_tensor = emit_buf_to_tensor(input_arg, restrict=True)
                
                M, N = self.shape
                
                # Define affine maps for indexing
                # #map = affine_map<(d0, d1) -> (d0, d1)>  (identity 2D)
                # #map1 = affine_map<(d0, d1) -> (d0)>     (broadcast/reduce along d1)
                d0 = ir.AffineDimExpr.get(0)
                d1 = ir.AffineDimExpr.get(1)
                map_2d = ir.AffineMap.get(2, 0, [d0, d1])
                map_1d = ir.AffineMap.get(2, 0, [d0])
                
                # Step 1: Find max - linalg.generic reduction
                neg_inf = arith.constant(dtype, float('-inf'))
                max_init = tensor.empty((M,), dtype)
                max_filled = linalg.fill(neg_inf, outs=[max_init])
                
                @linalg.generic(
                    [input_tensor],  # inputs
                    [max_filled],  # outputs
                    [map_2d, map_1d],  # indexing_maps
                    [linalg.IteratorType.parallel, linalg.IteratorType.reduction],  # iterator_types
                )
                def row_max(in_val, acc):
                    return arith.maximumf(in_val, acc)
                
                # Step 2: Subtract max (broadcast) - linalg.generic elementwise
                output_init = tensor.empty((M, N), dtype)
                
                @linalg.generic(
                    [input_tensor, row_max],  # inputs
                    [output_init],  # outputs
                    [map_2d, map_1d, map_2d],  # indexing_maps
                    [linalg.IteratorType.parallel, linalg.IteratorType.parallel],  # iterator_types
                )
                def shifted(in_val, max_val, out):
                    return arith.subf(in_val, max_val)
                
                # Step 3: Compute exp - linalg.generic elementwise
                @linalg.generic(
                    [shifted],  # inputs
                    [output_init],  # outputs
                    [map_2d, map_2d],  # indexing_maps
                    [linalg.IteratorType.parallel, linalg.IteratorType.parallel],  # iterator_types
                )
                def exp_vals(in_val, out):
                    return math.exp(in_val)
                
                # Step 4: Sum exp values - linalg.generic reduction
                # Create collapsed tensor for sum init
                # sum_init_2d = tensor.empty((M, 1), dtype)
                sum_init = tensor.empty((M,), dtype)
                # tensor.CollapseShapeOp(sum_init, sum_init_2d, [[0, 1]])

                
                zero = arith.constant(dtype, 0.0)
                sum_filled = linalg.fill(zero, outs=[sum_init])
                
                @linalg.generic(
                    [exp_vals],  # inputs
                    [sum_filled],  # outputs
                    [map_2d, map_1d],  # indexing_maps
                    [linalg.IteratorType.parallel, linalg.IteratorType.reduction],  # iterator_types
                )
                def row_sum(in_val, acc):
                    return arith.addf(in_val, acc)
                
                # Step 5: Divide by sum (broadcast) - linalg.generic elementwise
                @linalg.generic(
                    [exp_vals, row_sum],  # inputs
                    [output_init],  # outputs
                    [map_2d, map_1d, map_2d],  # indexing_maps
                    [linalg.IteratorType.parallel, linalg.IteratorType.parallel],  # iterator_types
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

    def schedule_modules(
        self, stop_at_stage: Optional[str] = None, parameters: Optional[dict] = None
    ) -> list[ir.Module]:
        """
        Generate transform schedule for softmax.
        
        For now, returns an empty schedule. In the future, this would contain
        tiling, vectorization, and XeGPU-specific lowering transformations.
        """
        # TODO: Implement proper transform schedule
        # For now, create a minimal schedule that prints the last linalg operation
        mod = ir.Module.create()
        mod.operation.attributes["transform.with_named_sequence"] = ir.UnitAttr.get()
        
        with ir.InsertionPoint(mod.body):

            
            # Create a transform sequence with proper signature
            named_sequence = transform.named_sequence(
                "__transform_main",
                [transform.AnyOpType.get()],  # input: module
                [],  # no outputs
                arg_attrs=[{"transform.readonly": ir.UnitAttr.get()}]
            )
            
            with ir.InsertionPoint(named_sequence.body):
                # Get the input module (bodyTarget)
                payload_mod = named_sequence.bodyTarget
                
                # Match all linalg.generic operations
                # We have 5 generic ops in softmax: max, sub, exp, sum, div
                generic_ops = structured.structured_match(
                    transform.AnyOpType.get(),
                    payload_mod,
                    ops=["linalg.generic"]
                )
                
                # Split the handle into individual operation handles
                # For softmax, we have 5 operations
                anytype = transform.AnyOpType.get()
                split_ops = transform.split_handle(
                    (anytype, anytype, anytype, anytype, anytype),  # 5 result types
                    generic_ops
                )
                
                # Reverse split_ops to have operations in reverse order
                split_ops = list(reversed(split_ops))
                
                # The first operation (after reversal) is the division - this is the consumer
                last_op = split_ops[0]

                # Print the last operation before tiling
                # transform.print_(target=last_op, name="last_linalg_generic_before_tiling")

                # Tile the last operation using tile_using_forall
                # Tile sizes: [64, 64] for the two parallel dimensions (M, N)
                tiled_op, for_op = structured.structured_tile_using_forall(
                    anytype, anytype,
                    last_op,
                    num_threads=[],
                    tile_sizes=[],
                    static_tile_sizes=(parameters["wg_rows"],),
                )

                # Fuse the producer operations into the forall loop
                # Iterate through remaining operations (already in reverse order)
                current_forall = for_op
                for producer_op in split_ops[1:]:
                    fused_op, current_forall = structured.structured_fuse_into_containing_op(
                        anytype, anytype,
                        producer_op,
                        current_forall
                    )
                    
                func = transform.get_parent_op(
                    anytype,
                    current_forall,
                    op_name="func.func",
                    deduplicate=True,
                )
                transform.apply_cse(func)
                canonicalize(func)
                func = apply_registered_pass(func, "eliminate-empty-tensors")
                func = structured.VectorizeChildrenAndApplyPatternsOp(
                    func,
                    fold_type_extensions_into_contract=True,
                ).result
                identity_layout = LayoutMapOption.IdentityLayoutMap
                payload_mod = transform.get_parent_op(
                    anytype,
                    func,
                    op_name="builtin.module",
                    deduplicate=True,
                )
                payload_mod = transform_bufferization.OneShotBufferizeOp(
                    payload_mod,
                    allow_return_allocs_from_loops=True,
                    bufferize_function_boundaries=True,
                    function_boundary_type_conversion=identity_layout,
                ).result
                payload_mod = apply_registered_pass(payload_mod, "fold-memref-alias-ops")
                transform.apply_cse(payload_mod)
                canonicalize(payload_mod)
                
                # convert forall to parallel
                wg_loops = match_and_split(payload_mod, ops={"scf.forall"})
                for wg_loop in wg_loops:
                    wg_loop = loop.loop_forall_to_parallel([anytype], wg_loop)
                func = transform.get_parent_op(anytype, wg_loop)

                # convert to scf.parallel to gpu.launch
                func = apply_registered_pass(func, "gpu-map-parallel-loops")
                func = apply_registered_pass(func, "convert-parallel-loops-to-gpu")
                func = apply_registered_pass(func, "lower-affine")
                transform.apply_cse(func)
                canonicalize(func)
                # set the number of threads for the gpu.launch operation
                launch_op = match_and_split(func, ops={"gpu.launch"})
                num_threads = parameters["sg_rows"] * parameters["subgroup_size"]
                xegpu.set_gpu_launch_threads(launch_op[0], threads=[num_threads, 1, 1])
                
                # outline gpu func
                func = apply_registered_pass(func, "lower-affine")
                canonicalize(func)
                func = apply_registered_pass(func, "gpu-launch-sink-index-computations")
                payload_mod = transform.get_parent_op(
                    anytype,
                    func,
                    op_name="builtin.module",
                    deduplicate=True,
                )
                # payload = match(payload_mod, ops={"func.func"})
                # transform.PrintOp(target=payload_mod, name="before_gpu_outlining")
                payload_mod = apply_registered_pass(payload_mod, "gpu-kernel-outlining")
                transform.apply_cse(payload_mod)

                # set xevm target
                payload_mod = apply_registered_pass(
                    payload_mod,
                    "xevm-attach-target",
                    options={"O": "3", "chip": "bmg"},
                )

                # convert vector to xegpu
                gpu_mod = match_and_split(payload_mod, ops={"gpu.module"})
                # for gpu_mod in gpu_mod_ops:
                gpu_func = match(gpu_mod[0], ops={"gpu.func"})
                gpu_func = apply_registered_pass(gpu_func, "convert-vector-to-xegpu")
                transform.apply_cse(gpu_func)
                
                # Set layout attributes for xegpu.store_nd operations
                store_ops = match_and_split(gpu_func, ops={"xegpu.store_nd"}, nhandles=1)
                # for store_op in store_ops:
                xegpu.set_op_layout_attr(store_ops[0], sg_layout=[8, 1], sg_data=[8, 64])
                
                payload_mod = apply_registered_pass(
                    payload_mod, "gpu-lower-to-xevm-pipeline", options={"xegpu-op-level": "workgroup"}
                )
                
                
                

                # Required: yield to end the transform sequence
                transform.yield_()
        
        return [mod]

    def shared_libs(self) -> list[str]:
        return ["libmlir_levelzero_runtime.so"]


def parse_cli():
    parser = argparse.ArgumentParser(
        description="Softmax using MLIR XeGPU",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--sizes",
        type=int,
        nargs=2,
        default=[1024, 64],
        help="M,N matrix sizes (MxN)",
    )
    parser.add_argument(
        "--wg-rows",
        type=int,
        default=64,
        help="Number of rows per workgroup.",
    )
    parser.add_argument(
        "--sg-rows",
        type=int,
        default=8,
        help="Number of rows per subgroup.",
    )
    parser.add_argument(
        "--subgroup-size",
        type=int,
        default=16,
        help="Subgroup size.",
    )
    parser.add_argument(
        "--nruns",
        type=int,
        default=1000,
        help="Number of runs to average the execution time.",
    )
    parser.add_argument(
        "--nwarmup",
        type=int,
        default=20,
        help="Number of warm-up iterations before benchmarking.",
    )
    parser.add_argument(
        "--check-result",
        action="store_true",
        help="Check the result of the softmax computation.",
    )
    parser.add_argument(
        "--dump-kernel",
        type=str,
        choices=[
            "initial",
            "tiled",
            "vectorized",
            "bufferized",
            "xegpu-initial",
            "xegpu-wg",
            "final",
        ],
        help="Dump kernel IR at different stages of lowering and exit without "
        "executing the kernel.",
    )
    parser.add_argument(
        "--dump-schedule",
        action="store_true",
        help="Dump transform schedule.",
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_cli()

    params = {
        "wg_rows": args.wg_rows,
        "sg_rows": args.sg_rows,
        "subgroup_size": args.subgroup_size,
    }

    M, N = args.sizes
    dtype = "f32"

    with ir.Context(), ir.Location.unknown():
        wload = XeGPUSoftmax(M=M, N=N, dtype=dtype)

        if args.dump_kernel or args.dump_schedule:
            wload.lower_payload(
                dump_payload=args.dump_kernel,
                dump_schedule=args.dump_schedule,
                schedule_parameters=params,
            )
        else:
            times = benchmark(
                wload,
                nruns=args.nruns,
                nwarmup=args.nwarmup,
                schedule_parameters=params,
                check_correctness=args.check_result,
                verbose=1,
            )
            times *= 1e6  # convert to microseconds
            elapsed = np.mean(times)
            flop_count = wload.get_complexity()[0]
            gflops = flop_count / (elapsed * 1e-6) / 1e9

            def list2str(a):
                return ",".join(map(str, a))

            parts = [
                f"sizes={list2str(args.sizes)}",
                f"dt={dtype}",
                f"wg-rows={args.wg_rows}",
                f"sg-rows={args.sg_rows}",
                f"subgroup-size={args.subgroup_size}",
                f"time(us): {elapsed:.2f}",
                f"GFLOPS: {gflops:.2f}",
            ]
            print(" ".join(parts))
