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
from lighthouse.schedule import schedule_boilerplate


def fused_attention_schedule(
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

    with schedule_boilerplate() as (schedule, named_seq):
        # match the payload module
        anytype = transform.AnyOpType.get()
        func = match(named_seq.bodyTarget, ops={"func.func"})
        payload_mod = transform.get_parent_op(
            anytype,
            func,
            op_name="builtin.module",
            deduplicate=True,
        )

        try:
            bundle_xegpu_fused_attention_schedule(
                payload_mod,
                parameters=parameters,
                stop_at_stage=stop_at_stage or "",
            )
        except PipelineInterrupt:
            pass
        finally:
            transform.yield_()

    return schedule


def bundle_xegpu_fused_attention_schedule(
    mod: ir.Value[transform.AnyOpType],
    parameters: dict,
    stop_at_stage: str = "",
) -> ir.Value[transform.AnyOpType]:
    """Schedule for lowering fused attention payload to xegpu wg level."""

    if stop_at_stage == "initial":
        raise PipelineInterrupt()

    anytype = transform.AnyOpType.get()
    anyvalue = transform.AnyValueType.get()
    # Match all matmul operations - there should be 2:
    # 1. Q @ K^T
    # 2. attention_weights @ V
    matmul_ops = match_and_split(mod, ops={"linalg.batch_matmul"}, nhandles=2)

    # Get the last matmul (attention_weights @ V)
    last_matmul = matmul_ops[1]
    func = transform.get_parent_op(
        anytype,
        last_matmul,
        op_name="func.func",
        deduplicate=True,
    )

    # Tile the last matmul in both batch and M dimensions.
    wg_tile_size = parameters["wg_tile_size"]

    tiled_matmul, forall_loop = structured.structured_tile_using_forall(
        anytype,
        anytype,
        last_matmul,
        num_threads=[],
        tile_sizes=[],
        static_tile_sizes=(1, wg_tile_size, 0, 0),
    )
    # Fuse the zero initialization of the output of the last matmul (tensor.empty) into the forall loop.
    tiled_matmul_init = transform.get_producer_of_operand(
        anytype, forall_loop, operand_number=0
    )
    _, forall_loop = structured.structured_fuse_into_containing_op(
        anytype,
        anytype,
        producer_op=tiled_matmul_init,
        containing_op=forall_loop,
    )
    transform.apply_cse(func)
    canonicalize(func)

    # Decompose softmax into generic ops
    softmax_ops = match_and_split(func, ops={"linalg.softmax"}, nhandles=1)
    softmax_op = softmax_ops[0]
    structured.structured_decompose_interface(anytype, softmax_op)
    transform.apply_cse(func)
    canonicalize(func)

    # Fuse all linalg.generic ops from softmax decomposition (4 ops: max, sub+exp, sum, div)
    # Match and fuse in reverse order (from consumer to producer)
    generic_ops = match_and_split(func, ops={"linalg.generic"}, nhandles=4)
    for generic_op in reversed(generic_ops):
        _, forall_loop = structured.structured_fuse_into_containing_op(
            anytype,
            anytype,
            producer_op=generic_op,
            containing_op=forall_loop,
        )
    transform.apply_cse(func)
    canonicalize(func)

    # Max and add reductions use linalg.fill to intialize the reduction output. Fuse these fill ops as well.
    fill_ops = match_and_split(func, ops={"linalg.fill"}, nhandles=5)
    # Max fill is the third fill op and add fill is the fourth fill op (based on the pattern of decomposition)
    max_fill_op = fill_ops[2]
    add_fill_op = fill_ops[3]
    for fill_op in [max_fill_op, add_fill_op]:
        _, forall_loop = structured.structured_fuse_into_containing_op(
            anytype,
            anytype,
            producer_op=fill_op,
            containing_op=forall_loop,
        )
    transform.apply_cse(func)
    canonicalize(func)

    linalg_mul_op = match_and_split(func, ops={"linalg.mul"}, nhandles=1)[0]
    first_matmul = transform.get_producer_of_operand(
        anytype, linalg_mul_op, operand_number=0
    )
    scale_fill_op = transform.get_producer_of_operand(
        anytype, linalg_mul_op, operand_number=1
    )
    transpose_op = transform.get_producer_of_operand(
        anytype, first_matmul, operand_number=1
    )
    matmul_fill_op = transform.get_producer_of_operand(
        anytype, first_matmul, operand_number=2
    )
    for op in [
        linalg_mul_op,
        scale_fill_op,
        first_matmul,
        matmul_fill_op,
        transpose_op,
    ]:
        _, forall_loop = structured.structured_fuse_into_containing_op(
            anytype,
            anytype,
            producer_op=op,
            containing_op=forall_loop,
        )
    transform.apply_cse(func)
    canonicalize(func)

    if stop_at_stage == "outer-tiled":
        raise PipelineInterrupt()

    # # vectorize (placeholder)
    # # func = structured.VectorizeChildrenAndApplyPatternsOp(
    # #     func,
    # #     fold_type_extensions_into_contract=True,
    # # ).result
    # transform.apply_cse(func)
    # canonicalize(func)

    if stop_at_stage == "inner-tiled":
        raise PipelineInterrupt()

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
