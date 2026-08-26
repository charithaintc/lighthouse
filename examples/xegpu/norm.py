# RUN: %PYTHON %s --dump-kernel=xegpu-wg | FileCheck %s
# CHECK: module attributes {gpu.container_module} {

"""
XeGPU row-wise vector norm / normalization (L1 / L2) benchmark.
"""

import argparse
from functools import cached_property

import numpy as np
from mlir import ir

from lighthouse import dialects as lh_dialects
from lighthouse.execution.runner import Runner
from lighthouse.pipeline.driver import TransformDriver
from lighthouse.execution import GPUMemoryManager
from lighthouse.utils.numpy import mlir_to_numpy_dtype
from lighthouse.ingress.mlir_gen import get_mlir_elem_type
from lighthouse.ingress.mlir_gen.gpu_norm_payload import generate_gpu_norm_payload
from lighthouse.schedule.parameters import ScheduleParameters
from lighthouse.schedule.xegpu import reduction_schedule, xegpu_to_binary


def norm_complexity(M: int, N: int, nbytes: int, norm_kind: str, normalize: bool):
    """
    Complexity of a row-wise norm in the max-scaled stable form.

    Per element: abs, max-compare, divide by the row max, add -- plus a multiply
    for L2's square, and one more divide when normalizing. The per-row work (a
    multiply, and a sqrt for L2) is O(M) and ignored.
    """
    flops_per_elem = 4 if norm_kind == "l1" else 5
    if normalize:
        flops_per_elem += 1
    flop_count = M * N * flops_per_elem
    memory_reads = M * N * nbytes  # read input
    memory_writes = (M * N if normalize else M) * nbytes
    return flop_count, memory_reads, memory_writes


def check_correctness(
    input_arr: np.ndarray,
    output_arr: np.ndarray,
    norm_kind: str,
    normalize: bool,
    verbose: int = 0,
) -> bool:
    x = input_arr.astype(np.float32)
    if norm_kind == "l1":
        norms = np.sum(np.abs(x), axis=1)
    else:
        norms = np.sqrt(np.sum(x * x, axis=1))
    output_ref = x / norms[:, None] if normalize else norms

    output = output_arr.astype(np.float32)

    if verbose > 1:
        print("Reference solution (first 8 rows):")
        print(output_ref[:8])
        print("Computed solution (first 8 rows):")
        print(output[:8])

    # The kernel sums N scaled terms in a different order than numpy, so compare
    # on a relative tolerance rather than exactly.
    success = np.allclose(output, output_ref, rtol=1e-4, atol=1e-5)

    if verbose:
        if success:
            print("PASSED")
        else:
            print("FAILED!")
            rel = np.abs(output - output_ref) / np.maximum(np.abs(output_ref), 1e-30)
            print(f"  Max relative diff: {rel.max():.6e}")
    return success


class XeGPUNorm:
    """
    Row-wise Lp norm / normalization on XeGPU.

    Computes, per row of an (M, N) input, either the norm itself

        out[i] = sum_j |x[i, j]|           (L1)
        out[i] = sqrt(sum_j x[i, j]^2)     (L2)

    or, with `normalize`, the input divided through by it -- the
    `torch.nn.functional.normalize` / cosine-similarity step:

        out[i, j] = x[i, j] / norm[i]

    Both use the numerically stable max-scaled form, which makes the op a
    `max -> elementwise -> sum` dependent-reduction chain -- the same shape as
    softmax, and fusable into a single online loop the same way.
    """

    def __init__(
        self,
        M: int,
        N: int,
        dtype: str = "f32",
        norm_kind: str = "l2",
        normalize: bool = False,
    ):
        self.M = M
        self.N = N
        self.shape = (M, N)
        self.row_shape = (M,)
        self.norm_kind = norm_kind
        self.normalize = normalize
        self.out_shape = self.shape if normalize else self.row_shape
        assert dtype == "f32", "Only f32 type is supported for norm"
        self.elem_type = get_mlir_elem_type(dtype)
        self.dtype = mlir_to_numpy_dtype(self.elem_type)
        self.memory_manager_class = GPUMemoryManager
        self.payload_function_name = "payload"

    @cached_property
    def _initial_host_arrays(self) -> tuple[np.ndarray]:
        """Generate initial values on host with numpy."""
        np.random.seed(42)
        # Magnitudes in [0.5, 1.5] with random signs. Kept away from zero because
        # the stable form divides by the row maximum: an all-zero row would make
        # the scaled term 0/0.
        magnitude = np.random.uniform(0.5, 1.5, self.shape).astype(self.dtype)
        sign = np.random.choice([-1.0, 1.0], self.shape).astype(self.dtype)
        input_arr = magnitude * sign
        output_arr = np.zeros(self.out_shape, dtype=self.dtype)
        return (output_arr, input_arr)

    def get_complexity(self) -> tuple[int, int, int]:
        nbytes = np.dtype(self.dtype).itemsize
        return norm_complexity(self.M, self.N, nbytes, self.norm_kind, self.normalize)

    def payload_module(self) -> ir.Module:
        """Generate MLIR module for the norm payload."""
        mod = generate_gpu_norm_payload(
            func_name=self.payload_function_name,
            M=self.M,
            N=self.N,
            dtype=self.elem_type,
            norm_kind=self.norm_kind,
            normalize=self.normalize,
        )
        # The 2D input, and the 1D per-row result when not normalizing, each need
        # their own alloc/copy helpers.
        ranks_and_types = [(2, self.elem_type), (1, self.elem_type)]
        self.memory_manager_class.emit_memory_management_funcs(
            mod, ranks_and_types=ranks_and_types
        )
        return mod

    def schedule_modules(
        self,
        stop_at_stage: str | None = None,
        parameters: ScheduleParameters | None = None,
    ) -> list[ir.Module]:
        """Generate transform schedule for the norm."""
        schedules = []
        schedules.append(Runner.get_bench_wrapper_schedule(self.payload_function_name))

        schedules.append(
            reduction_schedule(
                payload_func_name=self.payload_function_name,
                stop_at_stage=stop_at_stage,
                params=parameters,
            )
        )

        if stop_at_stage and stop_at_stage != "final":
            return schedules

        schedules.append(xegpu_to_binary())

        return schedules

    def shared_libs(self) -> list[str]:
        return ["libmlir_levelzero_runtime.so"]


def parse_cli():
    parser = argparse.ArgumentParser(
        description="Row-wise L1/L2 norm using MLIR XeGPU",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--norm",
        type=str,
        choices=["l1", "l2"],
        default="l2",
        help="Which norm to compute.",
    )
    parser.add_argument(
        "--sizes",
        type=int,
        nargs=2,
        default=[1024, 512],
        help="M,N matrix sizes (MxN); the norm reduces N.",
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
        "--reduction-step-size",
        type=int,
        default=16,
        help="Step size for reduction loop tiling (optional).",
    )
    parser.add_argument(
        "--normalize",
        action="store_true",
        help="Divide the input through by the row norms (torch F.normalize) "
        "instead of returning the norms.",
    )
    parser.add_argument(
        "--online",
        action="store_true",
        help="Fuse the max and sum reductions into a single online (one-pass) "
        "reduction loop.",
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
        default=1000,
        help="Number of warm-up iterations before benchmarking.",
    )
    parser.add_argument(
        "--check-result",
        action="store_true",
        help="Check the result of the norm computation.",
    )
    parser.add_argument(
        "--dump-kernel",
        type=str,
        choices=[
            "initial",
            "tiled",
            "vectorized",
            "bufferized",
            "gpu-outlining",
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
    parser.add_argument(
        "--verbose",
        "-v",
        action="count",
        default=0,
        help="Increase output verbosity (e.g. print reference and computed solutions).",
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_cli()

    params = ScheduleParameters(
        [
            {
                "layer_kind": "reduction",
                "sizes": args.sizes,
                "wg_rows": args.wg_rows,
                "sg_rows": args.sg_rows,
                "subgroup_size": args.subgroup_size,
                "reduction_step_size": args.reduction_step_size,
                "fuse_dependant_reductions": args.online,
                # Normalizing writes a full (M, N) result, so its epilogue spans
                # the reduced axis and gets tiled along it. A bare norm's result
                # is one scalar per row, so there is no full-extent pass to tile.
                "epilogue_spans_reduction_dim": args.normalize,
            }
        ]
    )

    M, N = args.sizes
    dtype = "f32"

    with ir.Context(), ir.Location.unknown():
        lh_dialects.register_and_load()
        wload = XeGPUNorm(
            M=M, N=N, dtype=dtype, norm_kind=args.norm, normalize=args.normalize
        )

        if args.dump_kernel or args.dump_schedule:
            pipeline = TransformDriver(
                wload.schedule_modules(
                    stop_at_stage=args.dump_kernel, parameters=params
                )
            )
            payload = pipeline.apply(wload.payload_module())
            if args.dump_kernel:
                print(payload)
            if args.dump_schedule:
                for schedule_module in wload.schedule_modules(parameters=params):
                    print(schedule_module)
        else:
            pipeline = TransformDriver(wload.schedule_modules(parameters=params))
            payload = pipeline.apply(wload.payload_module())
            runner = Runner(
                payload,
                mem_manager_cls=wload.memory_manager_class,
                shared_libs=wload.shared_libs(),
            )
            if args.check_result:
                # Setup callback function to copy result from device to host.
                result_host_copy = np.zeros(wload.out_shape, dtype=wload.dtype)
                argument_access_callback = Runner.get_gpu_argument_access_callback(
                    result_host_copy, arg_index=0
                )

                # Execute kernel once.
                runner.execute(
                    host_input_buffers=wload._initial_host_arrays,
                    payload_function_name=wload.payload_function_name,
                    argument_access_callback=argument_access_callback,
                )

                # Compute reference solution on host.
                success = check_correctness(
                    wload._initial_host_arrays[1],
                    result_host_copy,
                    wload.norm_kind,
                    wload.normalize,
                    verbose=args.verbose,
                )
                if not success:
                    raise ValueError("Result mismatch!")
                else:
                    print("Result is correct. Proceeding to benchmark...")

            times = runner.benchmark(
                host_input_buffers=wload._initial_host_arrays,
                nruns=args.nruns,
                nwarmup=args.nwarmup,
            )
            times *= 1e6  # convert to microseconds
            elapsed = np.mean(times)
            flop_count = wload.get_complexity()[0]
            gflops = flop_count / (elapsed * 1e-6) / 1e9

            def list2str(a):
                return ",".join(map(str, a))

            print(
                f"norm={args.norm} "
                f"{'normalize ' if args.normalize else ''}"
                f"sizes={list2str(args.sizes)} "
                f"dt={dtype} "
                f"wg-rows={args.wg_rows} "
                f"sg-rows={args.sg_rows} "
                f"subgroup-size={args.subgroup_size} "
                f"time(us): {elapsed:.2f} "
                f"GFLOPS: {gflops:.2f} "
            )
