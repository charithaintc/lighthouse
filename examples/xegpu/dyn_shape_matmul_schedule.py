"""Transform schedule for XeGPU matmul with a runtime-sized K dimension.

Implemented stages tile M/N and K, pad K before vectorization, bufferize,
outline a GPU kernel, and lower the contraction to XeGPU.
"""

from mlir import ir
from mlir.dialects.bufferization import LayoutMapOption
from mlir.dialects import transform
from mlir.dialects.transform import bufferization as transform_bufferization
from mlir.dialects.transform import memref as transform_memref
from mlir.dialects.transform import structured
from mlir.dialects.transform import vector as transform_vector

import lighthouse.transform as lh_transform
from lighthouse.dialects.transform import transform_ext
from lighthouse.pipeline.helper import (
    PipelineInterrupt,
    apply_registered_pass,
    canonicalize,
    match,
)
from lighthouse.schedule import schedule_boilerplate
from lighthouse.schedule.parameters import ScheduleParameters
from lighthouse.schedule.xegpu import XeGPUParameterSelector
from lighthouse.schedule.xegpu.lowering_common import (
    convert_to_gpu_launch,
    convert_vector_to_xegpu,
    get_payload_func,
    outline_gpu_function,
)
from lighthouse.schedule.xegpu.xegpu_specs import XeGPUSpecs

# Lowering stages in pipeline order.
STAGES = (
    "initial",
    "tiled",
    "vectorized",
    "bufferized",
    "gpu-outlining",
    "xegpu-initial",
    "xegpu-wg",
    "final",
)

IMPLEMENTED_STAGES = (
    "initial",
    "tiled",
    "vectorized",
    "bufferized",
    "gpu-outlining",
    "xegpu-initial",
)

REQUIRED_PARAMS = ("wg_m", "wg_n", "sg_m", "sg_n", "k_tile")


def dyn_shape_matmul_schedule(
    params: ScheduleParameters,
    payload_func_name: str = "payload",
    device: str | None = None,
    stop_at_stage: str = "",
    *,
    ab_type: ir.Type,
    acc_type: ir.Type,
) -> ir.Module:
    """Build the schedule, optionally stopping at an intermediate stage."""
    assert params is not None and len(params) > 0, "params must be provided."
    assert len(params) == 1, "the dynamic-shape matmul example has a single layer."

    layer_params = params[0]
    m, n, k = (layer_params.get(d) for d in ("m", "n", "k"))
    assert all(d is not None for d in (m, n, k)), "m, n, k must be provided in params"

    param_selector = XeGPUParameterSelector(device=device)
    if not all(p in layer_params for p in REQUIRED_PARAMS):
        # Nominal K selects tile sizes; the payload keeps K dynamic.
        generated = param_selector.get_parameters(
            (m, n, k),
            layer_params.get("transpose_a", False),
            layer_params.get("transpose_b", False),
        )
        for key, value in generated[0].items():
            layer_params.setdefault(key, value)

    with schedule_boilerplate() as (schedule, named_seq):
        anytype = transform.AnyOpType.get()
        func = get_payload_func(named_seq.bodyTarget, func_name=payload_func_name)
        payload_mod = transform.get_parent_op(
            anytype,
            func,
            op_name="builtin.module",
            deduplicate=True,
        )

        try:
            bundle_dyn_shape_matmul_schedule(
                payload_mod,
                payload_func_name=payload_func_name,
                params=params,
                gpu_specs=param_selector.gpu_specs,
                stop_at_stage=stop_at_stage,
                ab_type=ab_type,
                acc_type=acc_type,
            )
        except PipelineInterrupt:
            pass
        finally:
            transform.yield_()

    return schedule


def _pad_k_and_vectorize(
    func: ir.Value,
    matmul_op: ir.Value,
    *,
    k_tile: int,
    ab_type: ir.Type,
    acc_type: ir.Type,
) -> None:
    """Pad K to a static tile, then vectorize loads and contraction."""
    anytype = transform.AnyOpType.get()
    zero_ab = ir.FloatAttr.get(ab_type, 0.0)
    zero_acc = ir.FloatAttr.get(acc_type, 0.0)

    # Pad the clamped K tile to k_tile; skip copy-back for accumulator hoisting.
    structured.structured_pad(
        anytype,
        anytype,
        anytype,
        matmul_op,
        [],
        padding_values=[zero_ab, zero_ab, zero_acc],
        padding_dimensions=[2],
        static_pad_to_multiple_of=[k_tile],
        nofold_flags=[0, 0, 0],
        copy_back_op="none",
    )
    transform.apply_cse(func)
    canonicalize(func)

    # Vectorize padded loads and the contraction, then hoist the accumulator.
    func = structured.structured_vectorize_children_and_apply_patterns(
        anytype,
        func,
        fold_type_extensions_into_contract=True,
        vectorize_padding=True,
        vectorize_nd_extract=True,
    )
    k_loop = match(func, ops={"scf.for"})
    lh_transform.loop_hoisting(k_loop)
    lh_transform.cleanup(func)


def _bufferize(mod: ir.Value) -> ir.Value:
    """Bufferize tiled tensors and canonicalize memref aliases."""
    transform_bufferization.bufferization_eliminate_empty_tensors(mod)
    mod = transform_bufferization.bufferization_one_shot_bufferize(
        transform.any_op_t(),
        mod,
        function_boundary_type_conversion=LayoutMapOption.IdentityLayoutMap,
        bufferize_function_boundaries=True,
    )
    with ir.InsertionPoint(transform.apply_patterns(mod).patterns):
        transform_memref.apply_patterns_memref_fold_memref_alias_ops()
    transform.apply_cse(mod)
    canonicalize(mod)

    return mod


def bundle_dyn_shape_matmul_schedule(
    mod: ir.Value,
    payload_func_name: str,
    params: ScheduleParameters,
    gpu_specs: XeGPUSpecs,
    ab_type: ir.Type,
    acc_type: ir.Type,
    stop_at_stage: str = "",
) -> ir.Value:
    """Lower the dynamic-shape matmul payload towards XeGPU, stage by stage."""

    if stop_at_stage == "initial":
        raise PipelineInterrupt()

    layer_params = params[0]
    wg_tile = [layer_params["wg_m"], layer_params["wg_n"]]
    k_tile = layer_params["k_tile"]

    func = get_payload_func(mod, func_name=payload_func_name)
    # Fuse the epilogue so workgroup tiling includes its consumers.
    func = apply_registered_pass(func, "linalg-fuse-elementwise-ops")

    # Tile the final consumer to keep the epilogue inside the workgroup loop.
    matmul_op = match(func, ops={"linalg.matmul"})
    consumers = transform_ext.get_tileable_consumers(matmul_op)
    leaf_consumer_op = transform_ext.extract_handle(consumers, -1)

    # Tile static M/N in a workgroup forall.
    _, [wg_loop], _ = lh_transform.tile(
        leaf_consumer_op,
        tile_sizes=wg_tile,
        fuse_producers=True,
        use_forall=True,
        apply_cleanup=False,
    )
    # Tile dynamic K in an scf.for; affine.min clamps the final tile.
    wg_matmul = match(wg_loop, ops={"linalg.matmul"})
    _, [k_loop], _ = lh_transform.tile(wg_matmul, tile_sizes=[0, 0, k_tile])
    lh_transform.cleanup(wg_loop)

    lh_transform.cleanup(func)

    if stop_at_stage == "tiled":
        raise PipelineInterrupt()

    # Pad dynamic K before vectorization so the contraction has static sizes.
    func = get_payload_func(mod, func_name=payload_func_name)
    matmul_op = match(func, ops={"linalg.matmul"})
    _pad_k_and_vectorize(
        func,
        matmul_op,
        k_tile=k_tile,
        ab_type=ab_type,
        acc_type=acc_type,
    )
    if stop_at_stage == "vectorized":
        raise PipelineInterrupt()

    mod = _bufferize(mod)
    if stop_at_stage == "bufferized":
        raise PipelineInterrupt()

    # Map workgroups to gpu.launch and outline the kernel.
    convert_to_gpu_launch(mod, payload_func_name=payload_func_name)
    mod = outline_gpu_function(
        mod, payload_func_name=payload_func_name, gpu_specs=gpu_specs, params=params
    )
    if stop_at_stage == "gpu-outlining":
        raise PipelineInterrupt()

    # Normalize any remaining transfer masks before XeGPU conversion.
    gpu_func = match(mod, ops={"gpu.func"})
    with ir.InsertionPoint(transform.apply_patterns(gpu_func).patterns):
        transform_vector.apply_patterns_vector_lower_masked_transfers()
    lh_transform.cleanup(gpu_func)

    mod = convert_vector_to_xegpu(mod)
    if stop_at_stage == "xegpu-initial":
        raise PipelineInterrupt()

    raise NotImplementedError(
        f"stop_at_stage={stop_at_stage!r} is not implemented yet; "
        f"dyn_shape_matmul_schedule currently supports {IMPLEMENTED_STAGES}."
    )
