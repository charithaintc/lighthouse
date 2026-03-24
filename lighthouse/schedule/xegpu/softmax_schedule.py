"""Generate MLIR transform schedule for XeGPU softmax operation."""

from typing import Optional

from mlir import ir
from mlir.dialects import transform
from mlir.dialects.transform import structured, loop, xegpu
from mlir.dialects.transform import bufferization as transform_bufferization
from mlir.dialects.bufferization import LayoutMapOption

from lighthouse.pipeline.helper import (
    apply_registered_pass,
    canonicalize,
    match,
)


def match_and_split(*args, nhandles=1, **kwargs):
    """Helper function that splits matched handles."""
    matched = match(*args, **kwargs)
    anytype = transform.AnyOpType.get()
    matched_ops = transform.split_handle((anytype,) * nhandles, matched)
    if nhandles == 1:
        matched_ops = [matched_ops]
    return matched_ops


def get_softmax_schedule_module(
    stop_at_stage: Optional[str] = None,
    parameters: Optional[dict] = None,
) -> ir.Module:
    """
    Generate transform schedule for softmax operation.
    
    The schedule performs the following transformations:
    1. Tile the consumer operation (division) using forall
    2. Fuse producer operations into the forall loop
    3. Vectorize operations
    4. Bufferize tensors
    5. Convert to GPU dialect
    6. Lower to XeGPU operations
    
    Args:
        stop_at_stage: Optional stage name to stop early (for debugging)
        parameters: Dictionary with scheduling parameters:
            - wg_rows: Number of rows per workgroup
            - sg_rows: Number of rows per subgroup  
            - subgroup_size: Size of subgroup
            
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
            arg_attrs=[{"transform.readonly": ir.UnitAttr.get()}]
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
            
            # Match all linalg.generic and linalg.fill operations
            # We have 7 operations in softmax: 
            # fill(max_init), max, sub, exp, fill(sum_init), sum, div
            generic_ops = structured.structured_match(
                transform.AnyOpType.get(),
                payload_mod,
                ops=["linalg.generic", "linalg.fill"]
            )
            
            # Split the handle into individual operation handles
            anytype = transform.AnyOpType.get()
            split_ops = transform.split_handle(
                (anytype, anytype, anytype, anytype, anytype, anytype, anytype),  # 7 result types
                generic_ops
            )
            
            # Reverse split_ops to have operations in reverse order
            split_ops = list(reversed(split_ops))
            
            # The first operation (after reversal) is the division - this is the consumer
            last_op = split_ops[0]

            # Tile the last operation using tile_using_forall
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
            
            func = structured.VectorizeChildrenAndApplyPatternsOp(
                func,
                fold_type_extensions_into_contract=True,
            ).result
            transform.apply_cse(func)
            canonicalize(func)
            payload_mod = apply_registered_pass(payload_mod, "eliminate-empty-tensors")
            identity_layout = LayoutMapOption.IdentityLayoutMap
            payload_mod = transform_bufferization.OneShotBufferizeOp(
                payload_mod,
                allow_return_allocs_from_loops=True,
                bufferize_function_boundaries=True,
                function_boundary_type_conversion=identity_layout,
            ).result
            # fold memref.subviews into vector.transfer_read/write ops
            payload_mod = apply_registered_pass(payload_mod, "fold-memref-alias-ops")
            transform.apply_cse(payload_mod)
            canonicalize(payload_mod)
            
            # convert forall to parallel
            wg_loops = match_and_split(payload_mod, ops={"scf.forall"})
            for wg_loop in wg_loops:
                wg_loop = loop.loop_forall_to_parallel([anytype], wg_loop)
            func = transform.get_parent_op(anytype, wg_loop)
            # convert scf.parallel to gpu.launch
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
            payload_mod = apply_registered_pass(payload_mod, "gpu-kernel-outlining")
            transform.apply_cse(payload_mod)
            
            # set xevm target
            payload_mod = apply_registered_pass(
                payload_mod,
                "xevm-attach-target",
                options={"O": "3", "chip": "bmg"},
            )

            # convert vector to xegpu
            gpu_mod_ops = match_and_split(payload_mod, ops={"gpu.module"})
            for gpu_mod in gpu_mod_ops:
                gpu_func = match(gpu_mod, ops={"gpu.func"})
                gpu_func = apply_registered_pass(gpu_func, "convert-vector-to-xegpu")
                transform.apply_cse(gpu_func)
                
            # Set layout attributes for xegpu.store_nd operations
            store_ops = match_and_split(gpu_func, ops={"xegpu.store_nd"}, nhandles=1)
            xegpu.set_op_layout_attr(store_ops[0], sg_layout=[8, 1], sg_data=[8, 64])
            
            payload_mod = apply_registered_pass(
                payload_mod, "gpu-lower-to-xevm-pipeline", options={"xegpu-op-level": "workgroup"}
            )
            # Required: yield to end the transform sequence
            transform.yield_()
    
    return mod
