"""
Transform schedule for the dynamic-shape XeGPU matmul example.

Kept local to the example (rather than added to
``lighthouse.schedule.xegpu.mlp_schedule``) so that experiments with a dynamic
reduction dimension cannot regress the existing static schedules.

The schedule is being built up stage by stage. Right now only the ``initial``
stage is wired: the payload is matched and immediately handed back untouched, so
``--dump-kernel=initial`` shows the generated linalg payload. Requesting any
later stage raises, rather than silently dumping the initial IR.
"""

from mlir import ir
from mlir.dialects import transform

from lighthouse.pipeline.helper import PipelineInterrupt
from lighthouse.schedule import schedule_boilerplate
from lighthouse.schedule.parameters import ScheduleParameters
from lighthouse.schedule.xegpu.lowering_common import get_payload_func

# Lowering stages in pipeline order. Stages after `initial` are not implemented
# yet; extend IMPLEMENTED_STAGES as each one is added.
STAGES = (
    "initial",
    "tiled",
    "vectorized",
    "bufferized",
    "xegpu-initial",
    "xegpu-wg",
    "final",
)

IMPLEMENTED_STAGES = ("initial",)


def dyn_shape_matmul_schedule(
    params: ScheduleParameters,
    payload_func_name: str = "payload",
    device: str | None = None,
    stop_at_stage: str = "",
) -> ir.Module:
    """
    Generate the transform schedule module for the dynamic-shape matmul payload.

    Args:
        params: Per-layer schedule parameters (m, n, k and tile sizes).
        payload_func_name: Name of the payload function to transform.
        device: Target GPU device, used for parameter selection in later stages.
        stop_at_stage: Stop after this stage and leave the payload as-is.
    """
    assert params is not None and len(params) > 0, "params must be provided."
    assert len(params) == 1, "the dynamic-shape matmul example has a single layer."

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
                device=device,
                stop_at_stage=stop_at_stage,
            )
        except PipelineInterrupt:
            pass
        finally:
            transform.yield_()

    return schedule


def bundle_dyn_shape_matmul_schedule(
    mod: ir.Value,
    payload_func_name: str,
    params: ScheduleParameters,
    device: str | None = None,
    stop_at_stage: str = "",
) -> ir.Value:
    """Lower the dynamic-shape matmul payload towards XeGPU, stage by stage."""

    if stop_at_stage == "initial":
        raise PipelineInterrupt()

    # Next steps, in the order we will add them:
    #   - "tiled":         workgroup-tile the parallel dims (M, N) into an
    #                      scf.forall, then tile the dynamic K reduction loop.
    #   - "vectorized":    mask the vectorization so partial K tiles are safe.
    #   - "bufferized":    one-shot bufferize and outline the gpu.func.
    #   - "xegpu-initial": convert vector to xegpu.
    #   - "xegpu-wg":      attach workgroup layouts / prefetches.
    raise NotImplementedError(
        f"stop_at_stage={stop_at_stage!r} is not implemented yet; "
        f"dyn_shape_matmul_schedule currently supports {IMPLEMENTED_STAGES}."
    )
