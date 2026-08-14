"""Generate MLIR transform schedule for XeGPU fused attention operation."""

from mlir import ir
from mlir.dialects import transform
from mlir.dialects.transform import structured, loop, xegpu
from mlir.dialects.transform.structured import (
    apply_patterns_linalg_fold_unit_extent_dims_via_slices,
)
from mlir.dialects.transform.tensor import (
    apply_patterns_tensor_drop_redundant_insert_slice_rank_expansion,
    apply_patterns_tensor_fold_tensor_subset_ops,
    apply_patterns_tensor_merge_consecutive_insert_extract_slice,
)

from lighthouse.pipeline.helper import (
    canonicalize,
    match,
    match_and_split,
    PipelineInterrupt,
    apply_registered_pass,
)
from lighthouse.schedule import schedule_boilerplate
from lighthouse.schedule.xegpu.lowering_common import bufferize


def fused_attention_schedule(
    stop_at_stage: str | None = None,
    parameters: dict | None = None,
) -> ir.Module:
    """
    Generate transform schedule for attention kernel.

    The schedule performs the following transformations:
    1. Tile the fuse the strandard attention computation along parallel dims
    2. Vectorize operations
    3. Bufferize tensors
    4. Perform the fused attention optimization for the innermost computation
    5. Convert to GPU dialect
    6. Lower to XeGPU operations

    Args:
        stop_at_stage: Optional stage name to stop early (for debugging)
        parameters: Dictionary with scheduling parameters:
            - batch_size: Batch size (Z)
            - num_heads: Number of attention heads (H)
            - n_ctx: Context length
            - n_head: Head dimension
            - wg_rows: Number of Q*K^T*V rows computed by each work group
            - sg_rows: Number of Q*K^T*V rows computed by each subgroup
            - subgroup_size: Size of subgroup

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
    """Schedule for lowering attention payload to xegpu wg level."""

    if stop_at_stage == "initial":
        raise PipelineInterrupt()

    anytype = transform.AnyOpType.get()

    # The payload spells the softmax out as linalg.generic ops and defers the
    # normalizing divide past the P@V contraction (see
    # generate_gpu_attention_payload):
    #   S = (Q @ K^T) * scale ;  m = max_j S ;  p = exp(S - m) ;  l = sum_j p
    #   O = p @ V             ;  out = O / l
    # The divide -- not the contraction -- is therefore the last op of the chain
    # and the root of the work-group level tiling.
    max_op, p_op, sum_op, div_op = match_and_split(
        mod, ops={"linalg.generic"}, nhandles=4
    )
    func = transform.get_parent_op(
        anytype,
        div_op,
        op_name="func.func",
        deduplicate=True,
    )
    qk_matmul, pv_matmul = match_and_split(
        func, ops={"linalg.batch_matmul"}, nhandles=2
    )
    scale_mul_op = match_and_split(func, ops={"linalg.mul"}, nhandles=1)[0]

    # The reductions and both matmuls initialize their output with a linalg.fill,
    # and Q @ K^T reads a transposed K; reach all of them through their consumer.
    transpose_op = transform.get_producer_of_operand(
        anytype, qk_matmul, operand_number=1
    )
    qk_fill_op = transform.get_producer_of_operand(anytype, qk_matmul, operand_number=2)
    scale_fill_op = transform.get_producer_of_operand(
        anytype, scale_mul_op, operand_number=1
    )
    max_fill_op = transform.get_producer_of_operand(anytype, max_op, operand_number=1)
    sum_fill_op = transform.get_producer_of_operand(anytype, sum_op, operand_number=1)
    pv_fill_op = transform.get_producer_of_operand(anytype, pv_matmul, operand_number=2)

    # Tile the divide in both batch and M dimensions. Its destination is a bare
    # tensor.empty, which stays outside as the loop's shared_out; every other op
    # of the chain is fused in below.
    wg_rows = parameters["wg_rows"]
    tiled_div, forall_loop = structured.structured_tile_using_forall(
        anytype,
        anytype,
        div_op,
        num_threads=[],
        tile_sizes=[],
        static_tile_sizes=(1, wg_rows, 0),
    )

    # Sink the whole chain into the forall loop, walking from the consumer end to
    # the producer end so that each op is fused only once all of its consumers
    # are already inside the loop.
    for producer_op in [
        pv_matmul,
        pv_fill_op,
        sum_op,
        sum_fill_op,
        p_op,
        max_op,
        max_fill_op,
        scale_mul_op,
        scale_fill_op,
        qk_matmul,
        qk_fill_op,
        transpose_op,
    ]:
        _, forall_loop = structured.structured_fuse_into_containing_op(
            anytype,
            anytype,
            producer_op=producer_op,
            containing_op=forall_loop,
        )
    # CSE is what collapses the redundant tile slices the fusion leaves behind;
    # without it `fuse_into_containing_op` clones each producer once per slice use
    # and the whole score computation, row max included, gets duplicated.
    transform.apply_cse(func)
    canonicalize(func)

    if stop_at_stage == "outer-tiled":
        raise PipelineInterrupt()

    # Fold the whole softmax-and-contract chain into a single loop over the
    # key/value axis -- the fused (flash) attention inner loop -- while still at
    # linalg level. `fuse_dependant_reduction_ops` does the work: it moves the
    # elementwise term and one consumer reduction into an already-tiled producer
    # reduction loop and inserts the online correction that rescales that
    # reduction's running accumulator whenever the running max changes.
    tile_size = parameters.get("inner_loop_tile_size", 64)

    max_op, p_op, sum_op, _ = match_and_split(func, ops={"linalg.generic"}, nhandles=4)
    _, pv_matmul = match_and_split(func, ops={"linalg.batch_matmul"}, nhandles=2)
    # The fusion op wants both the elementwise term and the consumer reduction as
    # linalg.generic ops, so generalize the contraction.
    pv_op = structured.structured_generalize(anytype, pv_matmul)

    # Tile the row max along the key/value axis. This is the producer reduction
    # loop the rest of the chain gets folded into; the marker attribute is what
    # the fusion op recognizes it by.
    _, reduction_loop = structured.structured_tile_using_for(
        anytype,
        [anytype],
        max_op,
        dynamic_sizes=[],
        interchange=[],
        static_sizes=[0, 0, tile_size],
        scalable_sizes=[False, False, False],
    )
    transform.annotate(reduction_loop, "__reduction_loop__")

    # First chain: max -> p -> row sum. `p` also feeds the contraction, so the op
    # fuses a clone of it and leaves the original in place for the second chain.
    reduction_loop = structured.structured_fuse_dependant_reduction_ops(
        anytype, reduction_loop, p_op, sum_op
    )
    # Fusing replaces the loop, so re-mark the new one as the producer reduction.
    transform.annotate(reduction_loop, "__reduction_loop__")

    # Second chain: max -> p -> P@V, into that same loop. The first fusion
    # consumed the handle to `p`; the original is still the contraction's operand.
    p_op = transform.get_producer_of_operand(anytype, pv_op, operand_number=0)
    reduction_loop = structured.structured_fuse_dependant_reduction_ops(
        anytype, reduction_loop, p_op, pv_op
    )
    transform.apply_cse(func)

    # Sink the score computation into the reduction loop as well, so only one
    # [wg_rows, tile_size] score tile -- rather than the full [wg_rows, n_ctx]
    # matrix -- is ever live.
    for producer_name in ["linalg.mul", "linalg.batch_matmul", "linalg.transpose"]:
        producer_op = match_and_split(func, ops={producer_name}, nhandles=1)[0]
        _, reduction_loop = structured.structured_fuse_into_containing_op(
            anytype,
            anytype,
            producer_op=producer_op,
            containing_op=reduction_loop,
        )

    # The Q @ K^T zero accumulator and the scale tensor are still filled at full
    # [wg_rows, n_ctx] extent outside the loop, even though the ops now inside it
    # only ever read a [wg_rows, tile_size] slice. Sink those two fills as well so
    # neither tensor is materialized whole. Reach them through the in-loop consumer
    # they initialize -- one hop for the slice the fusion left behind, one more for
    # the fill itself. The three remaining fills initialize the loop's running
    # accumulators and must stay outside.
    scale_mul_op = match_and_split(reduction_loop, ops={"linalg.mul"}, nhandles=1)[0]
    qk_matmul = match_and_split(
        reduction_loop, ops={"linalg.batch_matmul"}, nhandles=1
    )[0]
    for consumer_op, operand_number in [(scale_mul_op, 1), (qk_matmul, 2)]:
        fill_slice = transform.get_producer_of_operand(
            anytype, consumer_op, operand_number=operand_number
        )
        fill_op = transform.get_producer_of_operand(
            anytype, fill_slice, operand_number=0
        )
        _, reduction_loop = structured.structured_fuse_into_containing_op(
            anytype,
            anytype,
            producer_op=fill_op,
            containing_op=reduction_loop,
        )

    transform.apply_cse(func)
    canonicalize(func)

    # Strip the batch dim, which the work-group tiling cut down to 1. Everything
    # downstream -- the XeGPU layouts below and the blocking/distribution passes in
    # `xegpu_to_binary` -- is built for rank-2 tiles, and the vector-level
    # `cast_away_leading_one_dim` patterns cannot finish the job: they have no
    # pattern for multi_reduction, broadcast or transpose, so the softmax row
    # reductions and the correction's broadcasts would keep a unit dim and drag
    # shape_casts (and rank-3 XeGPU layouts) along with them.
    #
    # Done here, after the reduction fusion rather than before it, for two reasons:
    # `fuse_dependant_reduction_ops` gets to run on the shape it already handles,
    # and every op above is still matched by name -- generalizing first would turn
    # the whole chain into linalg.generic.
    #
    # `fold_unit_extent_dims` only rewrites linalg.generic, hence the generalize;
    # and it leaves two-step slice chains behind (16x4096x64 -> 1x128x64 ->
    # 128x64), which plain canonicalization does not compose, hence the tensor
    # patterns. The scf.for's own iter_args keep their unit dim -- no pattern
    # retypes a loop signature -- but that no longer matters: with every op inside
    # rank 2, vectorization reads through them rank-reducing and the accumulators
    # come out as rank-1/2 vectors.
    named_ops = match(
        func,
        ops={
            "linalg.batch_matmul",
            "linalg.mul",
            "linalg.transpose",
            "linalg.fill",
            "linalg.elementwise",
        },
    )
    structured.structured_generalize(anytype, named_ops)
    with ir.InsertionPoint(transform.apply_patterns(func).patterns):
        apply_patterns_linalg_fold_unit_extent_dims_via_slices()
        transform.apply_patterns_canonicalization()
    transform.apply_cse(func)
    with ir.InsertionPoint(transform.apply_patterns(func).patterns):
        apply_patterns_tensor_merge_consecutive_insert_extract_slice()
        apply_patterns_tensor_drop_redundant_insert_slice_rank_expansion()
        apply_patterns_tensor_fold_tensor_subset_ops()
        transform.apply_patterns_canonicalization()
    transform.apply_cse(func)

    if stop_at_stage == "inner-tiled":
        raise PipelineInterrupt()

    # Vectorize. The shared helper also hoists the loop-invariant vector
    # transfers and subsets out of the reduction loop, which is what keeps the
    # running accumulators loop-carried rather than re-read every iteration.
    func = structured.structured_vectorize_children_and_apply_patterns(
        anytype,
        func,
        fold_type_extensions_into_contract=True,
    )

    # Hoist the loop-invariant vector transfers out of the reduction loop so the
    # running accumulators stay loop-carried instead of being re-read and
    # re-written every iteration. CSE first: vectorization emits one transfer_read
    # per linalg op that reads a given iter_arg -- the running max is read three
    # times, once by its own reduction and once by each chain's correction -- and
    # subset hoisting refuses to hoist a read/write pair while any other subset op
    # on the same iter_arg overlaps it. Merging the duplicate reads is what lets
    # that accumulator stay in registers.
    transform.apply_cse(func)
    kv_loop = match(func, ops={"scf.for"})
    transform.apply_licm(kv_loop)
    loop.loop_hoist_loop_invariant_subsets(kv_loop)
    transform.apply_cse(func)
    canonicalize(func)

    # `p` is fused into the reduction loop once per consumer reduction, and each
    # copy still writes its tile back into the full [wg_rows, n_ctx] iter_arg it
    # was carved out of -- two loop results that nothing outside the loop reads.
    # Neither DCE nor canonicalization can see that: the writes feed the loop's
    # yield, so the chain is only dead once the loop *results* are known dead, and
    # scf.for only folds away an iter_arg whose region argument is unused too.
    # `remove-dead-values` runs a liveness analysis across the loop and unwinds
    # the whole chain. It has to happen before bufferization -- the pass only
    # drops ops that do not affect memory, so once these are memref.alloc plus
    # vector.transfer_write nothing upstream will remove them.
    func = apply_registered_pass(func, "remove-dead-values")

    if stop_at_stage == "vectorized":
        raise PipelineInterrupt()

    mod = bufferize(mod)

    if stop_at_stage == "bufferized":
        raise PipelineInterrupt()

    # Convert forall to parallel
    wg_loops = match_and_split(mod, ops={"scf.forall"})
    for wg_loop in wg_loops:
        wg_loop = loop.loop_forall_to_parallel([anytype], wg_loop)
    func = transform.get_parent_op(anytype, wg_loop)

    # Convert scf.parallel to gpu.launch
    func = apply_registered_pass(func, "gpu-map-parallel-loops")
    func = apply_registered_pass(func, "convert-parallel-loops-to-gpu")
    func = apply_registered_pass(func, "lower-affine")
    transform.apply_cse(func)
    canonicalize(func)

    # Set the number of threads for the gpu.launch operation
    launch_op = match_and_split(func, ops={"gpu.launch"})
    wg_rows = parameters["wg_rows"]
    sg_rows = parameters["sg_rows"]
    subgroup_size = parameters["subgroup_size"]
    num_subgroups = wg_rows // sg_rows
    num_threads = num_subgroups * subgroup_size
    xegpu.set_gpu_launch_threads(launch_op[0], threads=[num_threads, 1, 1])

    # Outline gpu func
    func = apply_registered_pass(func, "lower-affine")
    canonicalize(func)
    func = apply_registered_pass(func, "gpu-launch-sink-index-computations")
    mod = apply_registered_pass(mod, "gpu-kernel-outlining")
    transform.apply_cse(mod)

    if stop_at_stage == "gpu-outlining":
        raise PipelineInterrupt()

    # Set xevm target
    mod = apply_registered_pass(
        mod,
        "xevm-attach-target",
        options={"O": "3", "chip": "bmg"},
    )

    # Convert vectot to xegpu
    gpu_mod_ops = match_and_split(mod, ops={"gpu.module"})
    for gpu_mod in gpu_mod_ops:
        gpu_func = match(gpu_mod, ops={"gpu.func"})
        gpu_func = apply_registered_pass(gpu_func, "convert-vector-to-xegpu")
        transform.apply_cse(gpu_func)
        gpu_func = apply_registered_pass(gpu_func, "loop-invariant-code-motion")

    # Insert prefetches for the K and V tiles of the reduction loop. Each inserts
    # nb_prefetch prefetches ahead of the loop plus one per iteration, at
    # induction_var + nb_prefetch * step. This must run before the wg-level layouts
    # are set below, since the prefetch descriptor is cloned from the load's
    # descriptor. The layouts of the emitted prefetch_nd ops are set together with
    # the other wg-level layouts.
    nb_prefetch = parameters.get("nb_prefetch", 1)
    if nb_prefetch > 0:
        reduction_loop = match(gpu_func, ops={"scf.for"})
        kv_load_ops = match_and_split(reduction_loop, ops={"xegpu.load_nd"}, nhandles=2)
        for load_op in kv_load_ops:
            xegpu.insert_prefetch(load_op, nb_prefetch=nb_prefetch)
        transform.apply_cse(gpu_func)
        canonicalize(gpu_func)

    if stop_at_stage == "xegpu-initial":
        raise PipelineInterrupt()

    # Define XeGPU layout parameters
    n_head = parameters["n_head"]
    sg_rows = parameters["sg_rows"]

    # Every tile is rank 2 here: the batch dim the work-group tiling cut down to 1
    # was dropped during vectorization, so these layouts describe [rows, cols].
    #
    # Q, the attention weights and the output are all [wg_rows, n_head] tiles that
    # are split by rows over the subgroups. Only the memory ops carry inst_data,
    # the DPAS operands are left to the default DPAS blocking.
    q_sg_layout = [num_subgroups, 1]
    q_sg_data = [sg_rows, n_head]
    q_load_inst_data = [16, 32]

    out_sg_layout = q_sg_layout
    out_sg_data = q_sg_data

    qk_sg_layout = q_sg_layout
    qk_sg_data = [sg_rows, tile_size]

    # The K and V tiles are consumed in full by every subgroup (each subgroup owns
    # its own rows of Q).
    kv_sg_layout = [1, 1]
    kv_load_sg_data = [tile_size, n_head]
    v_load_inst_data = [32, 32]
    # Load K column-major so that the transpose feeding the DPAS is a no-op. The
    # first entry of `order` is the fastest-changing dim, so [0, 1] makes the rows
    # innermost (the row-major default would be [1, 0]).
    k_load_order = [0, 1]

    # K^T operand of the Q@K^T DPAS: [n_head, tile_size]
    kt_sg_data = [n_head, tile_size]
    # V operand of the P@V DPAS: [tile_size, n_head]
    v_sg_data = [tile_size, n_head]

    # The K/V prefetches cover the same [tile_size, n_head] tile as the loads, but
    # are distributed over the subgroups to maximize prefetch bandwidth.
    prefetch_sg_layout = [2, 4]
    prefetch_sg_data = [tile_size // 2, n_head // 4]
    prefetch_inst_data = list(prefetch_sg_data)

    # Set layout attributes for xegpu.store_nd ops.
    store_nd_op = match_and_split(gpu_func, ops={"xegpu.store_nd"}, nhandles=1)[0]
    xegpu.set_anchor_layout(
        store_nd_op,
        sg_layout=out_sg_layout,
        sg_data=out_sg_data,
    )

    # Set layout for xegpu.load_nd ops (3 total: Q, K, V)
    load_nd_ops = match_and_split(gpu_func, ops={"xegpu.load_nd"}, nhandles=3)

    # First load_nd: Q layout
    xegpu.set_anchor_layout(
        load_nd_ops[0],
        sg_layout=q_sg_layout,
        sg_data=q_sg_data,
        inst_data=q_load_inst_data,
    )

    # Second load_nd: K layout
    xegpu.set_anchor_layout(
        load_nd_ops[1],
        sg_layout=kv_sg_layout,
        sg_data=kv_load_sg_data,
        order=k_load_order,
    )

    # Third load_nd: V layout
    xegpu.set_anchor_layout(
        load_nd_ops[2],
        sg_layout=kv_sg_layout,
        sg_data=kv_load_sg_data,
        inst_data=v_load_inst_data,
    )

    # Set layout for all K/V xegpu.prefetch_nd ops
    if nb_prefetch > 0:
        prefetch_ops = match(gpu_func, ops={"xegpu.prefetch_nd"})
        xegpu.set_anchor_layout(
            prefetch_ops,
            sg_layout=prefetch_sg_layout,
            sg_data=prefetch_sg_data,
            inst_data=prefetch_inst_data,
        )

    # Set layout for xegpu.dpas ops (2 total: Q@K^T and P@V)
    dpas_ops = match_and_split(gpu_func, ops={"xegpu.dpas"}, nhandles=2)

    # Layouts for the Q@K^T dpas:
    qk_dpas_op = dpas_ops[0]
    # Index 0: Q layout
    xegpu.set_anchor_layout(
        qk_dpas_op,
        sg_layout=q_sg_layout,
        sg_data=q_sg_data,
        index=0,
    )
    # Index 1: K^T layout
    xegpu.set_anchor_layout(
        qk_dpas_op,
        sg_layout=kv_sg_layout,
        sg_data=kt_sg_data,
        index=1,
    )
    # Index 2: QK output layout
    xegpu.set_anchor_layout(
        qk_dpas_op,
        sg_layout=qk_sg_layout,
        sg_data=qk_sg_data,
        index=2,
    )

    # Layouts for the P@V dpas:
    pv_dpas_op = dpas_ops[1]
    # Index 0: QK (attention weights) layout
    xegpu.set_anchor_layout(
        pv_dpas_op,
        sg_layout=qk_sg_layout,
        sg_data=qk_sg_data,
        index=0,
    )
    # Index 1: V layout
    xegpu.set_anchor_layout(
        pv_dpas_op,
        sg_layout=kv_sg_layout,
        sg_data=v_sg_data,
        index=1,
    )
    # Index 2: Output layout
    xegpu.set_anchor_layout(
        pv_dpas_op,
        sg_layout=out_sg_layout,
        sg_data=out_sg_data,
        index=2,
    )

    if stop_at_stage == "xegpu-wg":
        raise PipelineInterrupt()

    return mod
