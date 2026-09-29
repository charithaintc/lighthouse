# RUN: %PYTHON %s --dump-kernel=xegpu-initial | FileCheck %s
# RUN: %PYTHON %s --sizes 512 1024 128 --dump-kernel=xegpu-initial | FileCheck %s
# RUN: %PYTHON %s --dump-kernel=xegpu-wg | FileCheck %s
# RUN: %PYTHON %s --sizes 512 1024 128 --dump-kernel=xegpu-wg | FileCheck %s
# CHECK: gpu.module @payload_kernel

"""XeGPU matmul example with static M/N and runtime-sized K.

Use `--dump-kernel` to inspect lowering stages through `final`.
"""

import argparse
import warnings
from dataclasses import dataclass, field
from typing import ClassVar

import numpy as np
from mlir import ir

from lighthouse import dialects as lh_dialects
from lighthouse.execution.runner import Runner
from lighthouse.pipeline.driver import TransformDriver
from lighthouse.execution import (
    MemoryManager,
    GPUMemoryManager,
)
from lighthouse.ingress.mlir_gen import get_mlir_elem_type
from lighthouse.schedule.parameters import ScheduleParameters
from lighthouse.schedule.xegpu import xegpu_to_binary
from lighthouse.utils.numpy import mlir_to_numpy_dtype

from dyn_shape_matmul_payload import generate_dyn_shape_matmul_payload
from dyn_shape_matmul_schedule import (
    IMPLEMENTED_STAGES,
    STAGES,
    dyn_shape_matmul_schedule,
)


def matmul_complexity(
    M: int,
    N: int,
    K: int,
    accumulate_c: bool,
    nbytes_ab: int,
    nbytes_c: int,
):
    """Estimate matmul FLOPs and memory traffic."""
    flop_count = 2 * M * N * K
    memory_reads = (M * K + K * N) * nbytes_ab  # read A and B
    memory_writes = M * N * nbytes_c  # write C
    if accumulate_c:
        memory_reads += M * N * nbytes_c  # read C for accumulation
    return flop_count, memory_reads, memory_writes


@dataclass
class XeGPUDynShapeMatMul:
    """Configure a dynamic-K matmul; nominal K sizes buffers and selects tiles."""

    payload_function_name: ClassVar[str] = "payload"
    memory_manager_class: ClassVar[type[MemoryManager]] = GPUMemoryManager

    M: int = 4096
    N: int = 4096
    K: int = 4096
    ab_type: ir.Type | str | None = None
    c_type: ir.Type | str | None = None
    acc_type: ir.Type | str | None = None
    accumulate_c: bool = True
    truncate_c: bool = False
    _input_arrays_cache: dict[bool, list[np.ndarray]] = field(
        default_factory=dict, init=False, repr=False
    )

    def __post_init__(self):
        """Resolve element types and derive host matrix shapes."""
        if isinstance(self.ab_type, str):
            self.ab_type = get_mlir_elem_type(self.ab_type)
        if isinstance(self.c_type, str):
            self.c_type = get_mlir_elem_type(self.c_type)
        if isinstance(self.acc_type, str):
            self.acc_type = get_mlir_elem_type(self.acc_type)
        if self.ab_type is None:
            self.ab_type = ir.F16Type.get()
        if self.acc_type is None:
            self.acc_type = ir.F32Type.get()
        if self.c_type is None:
            self.c_type = self.ab_type if self.truncate_c else self.acc_type
        assert isinstance(self.ab_type, (ir.F16Type, ir.BF16Type)), (
            "Only f16 and bf16 types are supported for A and B"
        )
        assert isinstance(self.acc_type, ir.F32Type), "Only f32 type is supported for C"
        if self.truncate_c:
            assert self.c_type == self.ab_type, (
                "C type must match A/B type when truncating"
            )
        self.ab_dtype = mlir_to_numpy_dtype(self.ab_type)
        self.c_dtype = mlir_to_numpy_dtype(self.c_type)
        row_pitch_bytes = self.K * np.dtype(self.ab_dtype).itemsize
        if row_pitch_bytes % 64:
            raise ValueError(
                f"XeGPU block loads require K to give A a 64-byte-aligned row "
                f"pitch; K={self.K} gives {row_pitch_bytes} bytes"
            )
        self.a_shape = (self.M, self.K)
        self.b_shape = (self.K, self.N)
        self.c_shape = (self.M, self.N)

    def get_input_arrays(self, init_int: bool = False) -> list[np.ndarray]:
        """Generate and cache host inputs for the selected initialization mode."""

        cached = self._input_arrays_cache.get(init_int)
        if cached is not None:
            return cached

        def gen_random(shape, dtype):
            """Generate integer or floating-point inputs of the requested shape."""
            if init_int:
                # Integers avoid rounding differences during correctness checks.
                a = np.random.randint(-3, 4, shape)
            else:
                a = np.random.rand(*shape)
            return a.astype(dtype)

        np.random.seed(2)
        A = gen_random(self.a_shape, self.ab_dtype)
        B = gen_random(self.b_shape, self.ab_dtype)
        C = gen_random(self.c_shape, self.c_dtype)
        arrays = [C, A, B]
        self._input_arrays_cache[init_int] = arrays
        return arrays

    def get_complexity(self) -> tuple[int, int, int]:
        """Return estimated FLOPs, bytes read, and bytes written."""
        nbytes_ab = np.dtype(self.ab_dtype).itemsize
        nbytes_c = np.dtype(self.c_dtype).itemsize
        return matmul_complexity(
            self.M,
            self.N,
            self.K,
            self.accumulate_c,
            nbytes_ab,
            nbytes_c,
        )

    def payload_module(self) -> ir.Module:
        """Build the payload and GPU buffer helpers."""
        mod = generate_dyn_shape_matmul_payload(
            func_name=self.payload_function_name,
            M=self.M,
            N=self.N,
            ab_type=self.ab_type,
            acc_type=self.acc_type,
            result_type=self.c_type,
            accumulate_c=self.accumulate_c,
        )
        ranks_and_types = list(set(((2, self.ab_type), (2, self.c_type))))
        self.memory_manager_class.emit_memory_management_funcs(
            mod, ranks_and_types=ranks_and_types
        )
        return mod

    def schedule_modules(
        self,
        stop_at_stage: str | None = None,
        parameters: ScheduleParameters | None = None,
        device: str | None = None,
    ) -> list[ir.Module]:
        """Build wrapper and lowering schedules through the requested stage."""
        assert parameters is not None, "Schedule parameters must be provided"
        schedules = []
        schedules.append(Runner.get_bench_wrapper_schedule(self.payload_function_name))

        schedules.append(
            dyn_shape_matmul_schedule(
                params=parameters,
                payload_func_name=self.payload_function_name,
                device=device,
                stop_at_stage=stop_at_stage,
                ab_type=self.ab_type,
                acc_type=self.acc_type,
            )
        )

        if stop_at_stage and stop_at_stage != "final":
            return schedules

        schedules.append(xegpu_to_binary())

        return schedules

    def shared_libs(self) -> list[str]:
        """List shared libraries needed for execution."""
        return ["libmlir_levelzero_runtime.so"]


def check_results(
    mmul: XeGPUDynShapeMatMul,
    host_inputs: list[np.ndarray],
    host_solution: np.ndarray,
    verbose: int = 0,
) -> bool:
    """Compare the computed output with a NumPy matmul reference."""
    C, A, B = host_inputs[:3]

    f32 = np.float32
    D_ref = A.astype(f32) @ B.astype(f32)
    if mmul.accumulate_c:
        D_ref += C.astype(f32)
    if mmul.truncate_c:
        D_ref = D_ref.astype(mmul.ab_dtype)

    D_host = host_solution.astype(np.float32)
    if verbose > 1:
        print("Reference solution:")
        print(D_ref)
        print("Computed solution:")
        print(D_host)
    success = np.allclose(D_host, D_ref)

    if verbose:
        if success:
            print("PASSED")
        else:
            print("FAILED Result mismatch!")
            diff = np.abs(D_host - D_ref)
            print(f"Max absolute difference: {np.max(diff)}")

    return success


def parse_cli_args(description):
    """Parse example and dump-stage options."""
    parser = argparse.ArgumentParser(
        description=description,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--sizes",
        type=int,
        nargs=3,
        default=[4096, 4096, 4096],
        help="M,N,K matrix sizes (A=MxK, B=KxN, C=MxN). K is the runtime extent "
        "of the dynamic reduction dimension. K times the A/B element size "
        "must be a multiple of 64 bytes for XeGPU block loads.",
    )
    parser.add_argument(
        "--ab-type",
        type=str,
        choices=["f16", "bf16"],
        default="f16",
        help="Data type for A and B matrices.",
    )
    parser.add_argument(
        "--no-accumulate-c",
        action="store_true",
        help="Compute plain matrix-multiply C=A*B instead of matrix-multiply-accumulate C+=A*B.",
    )
    parser.add_argument(
        "--truncate-c",
        action="store_true",
        help="Truncate C to A,B data type after accumulation (e.g., float32 to float16).",
    )
    parser.add_argument(
        "--wg-tile",
        type=int,
        nargs=2,
        help="Workgroup tile size M,N.",
    )
    parser.add_argument(
        "--k-tile",
        type=int,
        help="Inner reduction dimension tile size K.",
    )
    parser.add_argument(
        "--init-int",
        action="store_true",
        help="Initialize arrays with integers in [-3, 3] to make result checking more reliable.",
    )
    parser.add_argument(
        "--check-result",
        action="store_true",
        help="Check the result of the matrix multiplication.",
    )
    parser.add_argument(
        "--nruns",
        type=int,
        default=500,
        help="Number of runs to average the execution time.",
    )
    parser.add_argument(
        "--nwarmup",
        type=int,
        default=500,
        help="Number of warm-up iterations before benchmarking.",
    )
    parser.add_argument(
        "--dump-kernel",
        type=str,
        choices=list(STAGES),
        help="Dump kernel IR at different stages of lowering and exit without "
        "executing the kernel.",
    )
    parser.add_argument(
        "--dump-schedule",
        action="store_true",
        help="Dump transform schedule.",
    )
    parser.add_argument(
        "--target",
        choices=["B70", "B50"],
        help="Target GPU device, e.g., B70.",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="count",
        default=0,
        help="Increase output verbosity (e.g. print reference and computed solutions).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    description = """XeGPU matmul example with a dynamic reduction dimension.

The payload declares K as `?`, so the reduction extent is a runtime value. The
`--sizes` K entry only sizes the host buffers and the tile-size heuristics.

Use --dump-kernel to inspect implemented lowering stages.
"""
    args = parse_cli_args(description=description)

    # K sizes host buffers; its payload dimension remains dynamic.
    m, n, k = args.sizes

    params = ScheduleParameters(
        [
            {
                "layer_kind": "matmul",
                "m": m,
                "n": n,
                "k": k,
                "transpose_a": False,
                "transpose_b": False,
                "dynamic_k": True,
            }
        ]
    )

    # Override workgroup and reduction tiles from the CLI.
    cli_params = {}
    if args.wg_tile:
        cli_params["wg_m"], cli_params["wg_n"] = args.wg_tile
    if args.k_tile:
        cli_params["k_tile"] = args.k_tile
    params[0].update(cli_params)

    with ir.Context(), ir.Location.unknown():
        lh_dialects.register_and_load()

        wload = XeGPUDynShapeMatMul(
            M=m,
            N=n,
            K=k,
            ab_type=args.ab_type,
            accumulate_c=not args.no_accumulate_c,
            truncate_c=args.truncate_c,
        )

        if args.check_result and not args.init_int:
            warnings.warn(
                "Correctness checking with float initialization is not reliable: "
                "it may report a mismatch even when the kernel is correct, due to "
                "floating-point rounding differences. Consider using --init-int.",
                RuntimeWarning,
            )

        if args.dump_kernel or args.dump_schedule:
            if args.dump_kernel:
                pipeline = TransformDriver(
                    wload.schedule_modules(
                        stop_at_stage=args.dump_kernel,
                        parameters=params,
                        device=args.target,
                    )
                )
                payload = pipeline.apply(wload.payload_module())
                print(payload)
            if args.dump_schedule:
                # Dump the requested stage or the latest implemented one.
                for schedule_module in wload.schedule_modules(
                    stop_at_stage=args.dump_kernel or IMPLEMENTED_STAGES[-1],
                    parameters=params,
                    device=args.target,
                ):
                    print(schedule_module)
        else:
            pipeline = TransformDriver(
                wload.schedule_modules(parameters=params, device=args.target)
            )
            payload = pipeline.apply(wload.payload_module())
            runner = Runner(
                payload,
                mem_manager_cls=wload.memory_manager_class,
                shared_libs=wload.shared_libs(),
            )
            host_inputs = wload.get_input_arrays(args.init_int)
            if args.check_result:
                # Copy device output back for correctness checking.
                D_host_copy = np.zeros(wload.c_shape, dtype=wload.c_dtype)
                argument_access_callback = Runner.get_gpu_argument_access_callback(
                    D_host_copy, arg_index=0
                )

                runner.execute(
                    host_input_buffers=host_inputs,
                    payload_function_name=wload.payload_function_name,
                    argument_access_callback=argument_access_callback,
                )
                success = check_results(
                    wload,
                    host_inputs,
                    D_host_copy,
                    verbose=args.verbose,
                )
                if not success:
                    raise ValueError("Result mismatch!")

            times = runner.benchmark(
                host_input_buffers=host_inputs,
                nruns=args.nruns,
                nwarmup=args.nwarmup,
            )
            times *= 1e6  # convert to microseconds
            elapsed = np.mean(times)
            flop_count = wload.get_complexity()[0]
            gflops = flop_count / (elapsed * 1e-6) / 1e9

            p = params[0]
            print(
                f"sizes={p['m']},{p['n']},{p['k']} "
                f"dt={wload.ab_type},{wload.c_type} "
                f"dyn-k=1 "
                f"time(us): {elapsed:.2f} "
                f"GFLOPS: {gflops:.2f}"
            )
