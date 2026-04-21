"""Generate MLIR transform schedule for XeGPU fused attention operation."""

from typing import Optional

from mlir import ir
from mlir.dialects import transform
from mlir.dialects.transform import structured

from lighthouse.pipeline.helper import (
    canonicalize,
    match,
    match_and_split,
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
            - wg_tile_size: Workgroup tile size for the collapsed batch dimension (Z*H*n_ctx)

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
    # Match all matmul operations - there should be 2:
    # 1. Q @ K^T
    # 2. attention_weights @ V
    matmul_ops = match_and_split(mod, ops={"linalg.matmul"}, nhandles=2)

    # Get the last matmul (attention_weights @ V)
    last_matmul = matmul_ops[1]
    func = transform.get_parent_op(
        anytype,
        last_matmul,
        op_name="func.func",
        deduplicate=True,
    )

    # Tile the last matmul in the batch dimension using tile_using_forall
    # Batch dimension is the first dimension (collapsed_dim = Z * H * n_ctx)
    # Extract workgroup tile size from parameters
    wg_tile_size = parameters["wg_tile_size"]

    tiled_matmul, forall_loop = structured.structured_tile_using_forall(
        anytype,
        anytype,
        last_matmul,
        num_threads=[],
        tile_sizes=[],
        static_tile_sizes=(wg_tile_size, 0),
    )

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
