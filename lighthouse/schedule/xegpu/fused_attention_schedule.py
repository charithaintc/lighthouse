"""Generate MLIR transform schedule for XeGPU fused attention operation."""

from typing import Optional

from mlir import ir
from mlir.dialects import transform

from lighthouse.pipeline.helper import (
    canonicalize,
    match,
    PipelineInterrupt,
)
from lighthouse.schedule.xegpu.helper import bundle_xegpu_to_binary


def get_fused_attention_schedule_module(
    stop_at_stage: Optional[str] = None,
    parameters: Optional[dict] = None,
) -> ir.Module:
    """
    Generate transform schedule for fused attention operation.

    The schedule performs the following transformations:
    1. Tile the fused attention operation
    2. Vectorize operations
    3. Bufferize tensors
    4. Convert to GPU dialect
    5. Lower to XeGPU operations

    Args:
        stop_at_stage: Optional stage name to stop early (for debugging)
        parameters: Dictionary with scheduling parameters:
            - batch_size: Batch size (Z)
            - num_heads: Number of attention heads (H)
            - n_ctx: Context length
            - n_head: Head dimension

    Returns:
        MLIR module containing the transform schedule
    """
    assert parameters is not None, "Schedule parameters must be provided"

    mod = ir.Module.create()
    mod.operation.attributes["transform.with_named_sequence"] = ir.UnitAttr.get()

    with ir.InsertionPoint(mod.body):
        # Create a transform sequence with proper signature
        named_sequence = transform.named_sequence(
            "__transform_main",
            [transform.AnyOpType.get()],  # input: module
            [],  # no outputs
            arg_attrs=[{"transform.readonly": ir.UnitAttr.get()}],
        )

        with ir.InsertionPoint(named_sequence.body):
            # match the payload module
            anytype = transform.AnyOpType.get()
            func = match(named_sequence.bodyTarget, ops={"func.func"})
            payload_mod = transform.get_parent_op(
                anytype,
                func,
                op_name="builtin.module",
                deduplicate=True,
            )

            xegpu_fused_attention_transform_schedule(
                payload_mod,
                parameters=parameters,
                stop_at_stage=stop_at_stage or "",
            )

    return mod


def xegpu_fused_attention_transform_schedule(
    mod: ir.Value[transform.AnyOpType],
    parameters: dict,
    stop_at_stage: str = "",
):
    """Transform schedule for fused attention payload."""
    try:
        mod = bundle_xegpu_fused_attention_schedule(
            mod,
            parameters=parameters,
            stop_at_stage=stop_at_stage,
        )

        mod = bundle_xegpu_to_binary(
            mod,
            stop_at_stage=stop_at_stage,
        )
    except PipelineInterrupt:
        pass
    finally:
        transform.yield_()


def bundle_xegpu_fused_attention_schedule(
    mod: ir.Value[transform.AnyOpType],
    parameters: dict,
    stop_at_stage: str = "",
) -> ir.Value[transform.AnyOpType]:
    """Schedule for lowering fused attention payload to xegpu wg level."""

    if stop_at_stage == "initial":
        raise PipelineInterrupt()

    anytype = transform.AnyOpType.get()

    # TODO: Implement tiling, fusion, and lowering for fused attention
    # This will involve:
    # 1. Matching and tiling matmul operations (Q @ K^T)
    # 2. Fusing softmax operation
    # 3. Tiling second matmul (attention @ V)
    # 4. Vectorization
    # 5. Bufferization
    # 6. GPU outlining
    # 7. XeGPU lowering

    func = match(mod, ops={"func.func"})

    if stop_at_stage == "tiled":
        raise PipelineInterrupt()

    # vectorize (placeholder)
    # func = structured.VectorizeChildrenAndApplyPatternsOp(
    #     func,
    #     fold_type_extensions_into_contract=True,
    # ).result
    transform.apply_cse(func)
    canonicalize(func)

    if stop_at_stage == "vectorized":
        raise PipelineInterrupt()

    # bufferize (placeholder)
    # mod = apply_registered_pass(mod, "eliminate-empty-tensors")
    # identity_layout = LayoutMapOption.IdentityLayoutMap
    # mod = transform_bufferization.OneShotBufferizeOp(
    #     mod,
    #     allow_return_allocs_from_loops=True,
    #     bufferize_function_boundaries=True,
    #     function_boundary_type_conversion=identity_layout,
    # ).result

    if stop_at_stage == "bufferized":
        raise PipelineInterrupt()

    if stop_at_stage == "gpu-outlining":
        raise PipelineInterrupt()

    if stop_at_stage == "xegpu-initial":
        raise PipelineInterrupt()

    if stop_at_stage == "xegpu-wg":
        raise PipelineInterrupt()

    return mod
