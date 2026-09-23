# RUN: %PYTHON %s --dump-kernel=initial | FileCheck %s
# RUN: %PYTHON %s --sizes 512 1024 128 --dump-kernel=initial | FileCheck %s
# RUN: %PYTHON %s --dump-kernel=initial --ab-type bf16 | FileCheck %s
# RUN: %PYTHON %s --dump-kernel=initial --no-accumulate-c | FileCheck %s
# RUN: %PYTHON %s --dump-kernel=initial --truncate-c | FileCheck %s
# RUN: %PYTHON %s --dump-kernel=tiled | FileCheck %s --check-prefix=TILED
# RUN: %PYTHON %s --sizes 512 1024 128 --dump-kernel=tiled | FileCheck %s --check-prefix=TILED
# RUN: %PYTHON %s --dump-kernel=tiled --wg-tile 256 256 | FileCheck %s --check-prefix=TILED
# RUN: %PYTHON %s --dump-kernel=tiled --k-tile 64 | FileCheck %s --check-prefix=TILED
# RUN: %PYTHON %s --dump-kernel=tiled --ab-type bf16 | FileCheck %s --check-prefix=TILED
# RUN: %PYTHON %s --dump-kernel=tiled --no-accumulate-c | FileCheck %s --check-prefix=TILED
# RUN: %PYTHON %s --dump-kernel=tiled --truncate-c | FileCheck %s --check-prefix=TILED
# RUN: %PYTHON %s --dump-kernel=vectorized | FileCheck %s --check-prefix=VEC
# RUN: %PYTHON %s --sizes 512 1024 128 --dump-kernel=vectorized | FileCheck %s --check-prefix=VEC
# RUN: %PYTHON %s --dump-kernel=vectorized --k-tile 64 | FileCheck %s --check-prefix=VEC
# RUN: %PYTHON %s --dump-kernel=vectorized --truncate-c | FileCheck %s --check-prefix=VEC
# RUN: %PYTHON %s --dump-kernel=bufferized | FileCheck %s --check-prefix=BUF
# RUN: %PYTHON %s --sizes 512 1024 128 --dump-kernel=bufferized | FileCheck %s --check-prefix=BUF
# RUN: %PYTHON %s --dump-kernel=gpu-outlining | FileCheck %s --check-prefix=OUTLINE
# RUN: %PYTHON %s --sizes 512 1024 128 --dump-kernel=gpu-outlining | FileCheck %s --check-prefix=OUTLINE
# RUN: %PYTHON %s --dump-kernel=xegpu-initial | FileCheck %s --check-prefix=XEGPU
# RUN: %PYTHON %s --sizes 512 1024 128 --dump-kernel=xegpu-initial | FileCheck %s --check-prefix=XEGPU
# RUN: %PYTHON %s --dump-schedule | FileCheck %s --check-prefix=SCHEDULE

# K must stay dynamic all the way to the payload: A is Mx? and B is ?xN.
# CHECK-LABEL: func.func @payload(
# CHECK-SAME:    %{{.*}}: memref<{{[0-9]+}}x{{[0-9]+}}x{{[a-z0-9]+}}>,
# CHECK-SAME:    %{{.*}}: memref<{{[0-9]+}}x?x{{[a-z0-9]+}}>,
# CHECK-SAME:    %{{.*}}: memref<?x{{[0-9]+}}x{{[a-z0-9]+}}>)
# CHECK:         linalg.matmul ins(%{{.*}}, %{{.*}} : tensor<{{[0-9]+}}x?x{{[a-z0-9]+}}>, tensor<?x{{[0-9]+}}x{{[a-z0-9]+}}>)
# CHECK:         bufferization.materialize_in_destination

# M and N are tiled statically into an scf.forall, K into an scf.for whose trip
# count is only known at runtime. The affine.min clamps the last K tile, and the
# matmul operands keep `?` on the reduction dim.
# TILED-LABEL: func.func @payload(
# TILED:         scf.forall
# TILED:           scf.for
# TILED:             affine.min
# TILED:             linalg.matmul ins(%{{.*}}, %{{.*}} : tensor<{{[0-9]+}}x?x{{[a-z0-9]+}}>, tensor<?x{{[0-9]+}}x{{[a-z0-9]+}}>)
# TILED:           tensor.parallel_insert_slice
# TILED:         bufferization.materialize_in_destination

# The dynamic K tile is padded to k_tile with zeros before vectorizing, so masks
# land only on the A/B loads and the contraction is bare.
# The MxN accumulator is hoisted into the K loop's iter_args as a vector.
# VEC-LABEL: func.func @payload(
# VEC:         vector.transfer_read
# VEC:         scf.for {{.*}} iter_args({{.*}}) -> (vector<{{[0-9]+}}x{{[0-9]+}}xf32>)
# VEC:           affine.min
# VEC:           vector.create_mask
# VEC:           vector.mask {{.*}} { vector.transfer_read
# VEC:           vector.create_mask
# VEC:           vector.mask {{.*}} { vector.transfer_read
# VEC:           vector.contract
# VEC:         vector.transfer_write
# VEC-NOT:     vector.mask {{.*}} { vector.contract
# VEC-NOT:     tensor.pad

# convert-vector-to-xegpu: the contraction reaches xegpu.dpas, and the static MxN
# accumulator becomes create_nd_tdesc + load_nd/store_nd. The masked A/B loads do
# NOT convert -- transferPreconditions rejects a transfer with a mask operand --
# so they stay as vector.transfer_read. Flip those XEGPU-NOT/CHECK pairs when
# that gap is closed.
# XEGPU: gpu.module @payload_kernel [#xevm.target<O = 3>]
# XEGPU:   gpu.func @payload_kernel(%{{.*}}: memref<{{[0-9]+}}x?x{{[a-z0-9]+}}>,
# XEGPU:     xegpu.create_nd_tdesc
# XEGPU:     xegpu.load_nd
# XEGPU:     scf.for {{.*}} iter_args({{.*}}) -> (vector<{{[0-9]+}}x{{[0-9]+}}xf32>)
# XEGPU:       vector.create_mask
# XEGPU:       vector.transfer_read %{{.*}}, %{{.*}}, %{{.*}} {in_bounds
# XEGPU:       xegpu.dpas
# XEGPU:     xegpu.store_nd
# XEGPU-NOT: vector.mask

# After bufferization the masks survive and the subviews fold into direct
# indexed transfers off the function arguments.
# BUF-LABEL: func.func @payload(
# BUF:         memref.dim
# BUF:         scf.forall
# BUF:           vector.transfer_read %{{.*}}[
# BUF:           scf.for {{.*}} iter_args({{.*}}) -> (vector<{{[0-9]+}}x{{[0-9]+}}xf32>)
# BUF:             vector.create_mask
# BUF:             vector.mask {{.*}} { vector.transfer_read %{{.*}}[
# BUF:             vector.contract
# BUF:           vector.transfer_write
# BUF-NOT:     linalg.matmul
# BUF-NOT:     vector.mask {{.*}} { vector.contract

# The dynamic K extent travels into the kernel as a `?` in the argument type, so
# memref.dim still recovers it inside gpu.func. affine.min is lowered to
# arith.subi/minsi by lower-affine, and the masks are carried through untouched.
# OUTLINE: gpu.module @payload_kernel
# OUTLINE:   gpu.func @payload_kernel(%{{.*}}: memref<{{[0-9]+}}x?x{{[a-z0-9]+}}>,
# OUTLINE-SAME:  kernel
# OUTLINE:     %[[K:.*]] = memref.dim
# OUTLINE:     scf.for {{.*}} to %[[K]] step
# OUTLINE:       arith.minsi
# OUTLINE:       vector.mask {{.*}} { vector.transfer_read
# OUTLINE:       vector.contract
# OUTLINE:     gpu.return
# OUTLINE-NOT: linalg.matmul
# OUTLINE-NOT: vector.mask {{.*}} { vector.contract

# SCHEDULE: transform.named_sequence @__transform_main
# SCHEDULE: transform.structured.match ops{["func.func"]} attributes {sym_name = "payload"}
# SCHEDULE: transform.structured.fuse
# SCHEDULE: transform.yield

"""
XeGPU matrix multiplication example with a dynamic reduction dimension.

Same kernel as `matmul.py`, except the payload declares K as `?`, so the
reduction extent is only known at runtime. M and N stay static.

The transform schedule is built up stage by stage; currently only
`--dump-kernel=initial` is supported.
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
    """Complexity of the matmul operation."""
    flop_count = 2 * M * N * K
    memory_reads = (M * K + K * N) * nbytes_ab  # read A and B
    memory_writes = M * N * nbytes_c  # write C
    if accumulate_c:
        memory_reads += M * N * nbytes_c  # read C for accumulation
    return flop_count, memory_reads, memory_writes


@dataclass
class XeGPUDynShapeMatMul:
    """
    Matrix multiplication kernel on XeGPU with a dynamic reduction dimension.

    Computes C = A * B for input matrices A (M x K) and B (K x N), where the
    payload declares K as `?`. `K` here is only the runtime extent: it sizes the
    host buffers and feeds the tile-size heuristics, but never appears in the
    payload types.

    If `accumulate_c` is True, computes C = A * B + C instead.

    `ab_type` specifies the data type for A and B matrices (f16 or bf16).
    `c_type` specifies the data type of the result C (f32 by default, or ab_type
    if truncate_c is True).
    `acc_type` specifies the data type for accumulation (f32 by default).
    """

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
        self.a_shape = (self.M, self.K)
        self.b_shape = (self.K, self.N)
        self.c_shape = (self.M, self.N)

    def get_input_arrays(self, init_int: bool = False) -> list[np.ndarray]:
        """Generate initial values on host with numpy."""

        # Cache the generated arrays to avoid regenerating them for every run.
        cached = self._input_arrays_cache.get(init_int)
        if cached is not None:
            return cached

        def gen_random(shape, dtype):
            if init_int:
                # Use integer values to avoid f16/f32 floating point
                # discrepancies in the correctness check.
                a = np.random.randint(-3, 4, shape)
            else:
                # Use float values for benchmarking to get reliable performance
                # measurements.
                a = np.random.rand(*shape) - 0.5
            return a.astype(dtype)

        np.random.seed(2)
        A = gen_random(self.a_shape, self.ab_dtype)
        B = gen_random(self.b_shape, self.ab_dtype)
        C = gen_random(self.c_shape, self.c_dtype)
        arrays = [C, A, B]
        self._input_arrays_cache[init_int] = arrays
        return arrays

    def get_complexity(self) -> tuple[int, int, int]:
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
        return ["libmlir_levelzero_runtime.so"]


def check_results(
    mmul: XeGPUDynShapeMatMul,
    host_inputs: list[np.ndarray],
    host_solution: np.ndarray,
    verbose: int = 0,
) -> bool:
    """
    Check correctness of the result.
    """
    # Compute reference solution on host.
    C, A, B = host_inputs[:3]

    # use float32 data type for efficiency
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
        "of the dynamic reduction dimension.",
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
        # Derived from the schedule so the two cannot drift apart.
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

The schedule is still being built up; currently only --dump-kernel=initial works.
"""
    args = parse_cli_args(description=description)

    # Problem size. K is the runtime extent of the dynamic dimension.
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
                # The reduction dim is dynamic in the payload; record that so the
                # later schedule stages can pick masking over static tiling.
                "dynamic_k": True,
            }
        ]
    )

    # Collect parameters from CLI arguments
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

        # Remove this guard once dyn_shape_matmul_schedule reaches the "final"
        # stage; everything below it is already wired up for execution.
        if not (args.dump_kernel or args.dump_schedule):
            raise SystemExit(
                "Executing the kernel needs the 'final' schedule stage, which is not "
                f"implemented yet (implemented: {IMPLEMENTED_STAGES}). "
                "Use --dump-kernel=initial or --dump-schedule for now."
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
                # Only the stages listed in dyn_shape_matmul_schedule's
                # IMPLEMENTED_STAGES can be built, so dump the schedule as far
                # as it currently goes.
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
                # Setup callback function to copy result from device to host.
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
