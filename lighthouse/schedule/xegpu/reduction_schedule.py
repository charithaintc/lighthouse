"""Generate MLIR transform schedule for XeGPU softmax operation."""

from mlir import ir
from mlir.dialects import transform
from mlir.dialects.transform import structured, xegpu
import lighthouse.transform as lh_transform
from .lowering_common import (
    get_named_func,
    vectorize,
    bufferize,
    convert_to_gpu_launch,
    convert_vector_to_xegpu,
)
from lighthouse.pipeline.helper import (
    apply_registered_pass,
    canonicalize,
    match,
    match_and_split,
    PipelineInterrupt,
)
from lighthouse.schedule import schedule_boilerplate
from lighthouse.schedule.parameters import ScheduleParameters
from lighthouse.dialects.transform import transform_ext


def reduction_schedule(
    stop_at_stage: str | None = None,
    params: ScheduleParameters | None = None,
    payload_func_name: str = "payload",
) -> ir.Module:
    """
    Generate transform schedule for softmax operation.

    The schedule performs the following transformations:
    1. Tile the linalg.softmax operation using forall
    2. Vectorize operations
    3. Bufferize tensors
    4. Convert to GPU dialect
    5. Lower to XeGPU operations

    Args:
        stop_at_stage: Optional stage name to stop early (for debugging)
        params: ScheduleParameters object containing one reduction layer
            parameter dictionary with keys:
            - wg_rows: Number of rows per workgroup
            - sg_rows: Number of rows per subgroup
            - subgroup_size: Size of subgroup
            - sizes: Tuple with the sizes of the input tensors (e.g. (M, N))
            - reduction_step_size: Optional step size for tiling reduction loops
            - fuse_dependant_reductions: Optional; fold a two-reduction
              `R1 -> E -> R2` chain into a single online loop instead of giving
              each reduction its own. Off by default, and only sound when `E`'s
              data inputs cancel in the new/old ratio of the online correction --
              softmax's `exp(x - m) = exp(x)/exp(m)` does, layer norm's
              `(x - mean)^2` does not.
            - epilogue_spans_reduction_dim: Optional, default True. Whether the
              trailing elementwise op covers the reduction axis and so has to be
              tiled along it (softmax, layer norm). False for kernels whose
              result is one value per parallel slice, e.g. a row-wise norm: there
              the epilogue only consumes the reduced accumulators, so there is
              nothing to tile and the output is rank-reduced.

    Returns:
        MLIR module containing the transform schedule
    """
    assert params is not None and len(params) > 0, (
        "Schedule parameters must be provided"
    )

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
            bundle_xegpu_reduction_schedule(
                payload_mod,
                payload_func_name=payload_func_name,
                params=params,
                stop_at_stage=stop_at_stage,
            )
        except PipelineInterrupt:
            pass
        finally:
            transform.yield_()

    return schedule


def bundle_xegpu_reduction_schedule(
    mod: ir.Value[transform.AnyOpType],
    payload_func_name: str,
    params: ScheduleParameters,
    stop_at_stage: str = "",
) -> ir.Value[transform.AnyOpType]:
    """Schedule for lowering softmax payload to xegpu wg level."""

    layer_params = params[0]

    if stop_at_stage == "initial":
        raise PipelineInterrupt()

    reduction_step_size = layer_params["reduction_step_size"]

    anytype = transform.AnyOpType.get()

    # Match linalg.softmax operation if any and decompose it into generic ops
    softmax_ops = structured.structured_match(anytype, mod, ops=["linalg.softmax"])
    structured.structured_decompose_interface(anytype, softmax_ops)

    # Match payload function
    # TODO match with given function name instead?
    generic_ops = structured.structured_match(anytype, mod, ops=["linalg.generic"])
    func = transform.get_parent_op(
        anytype,
        generic_ops,
        op_name="func.func",
        deduplicate=True,
    )

    # Normalize possible singleton dimensions so tile+fuse logic works.
    with ir.InsertionPoint(transform.apply_patterns(func).patterns):
        structured.apply_patterns_linalg_fold_unit_extent_dims_via_slices()
    transform_ext.fold_singleton_extract_slice(func)
    lh_transform.cleanup(func)

    # Fuse elementwise ops, also removes unused linalg op results (if any).
    #
    # Skipped in the online form: `fuse_dependant_reduction_ops` needs the
    # elementwise term `E` of the `R1 -> E -> R2` chain to still be an op of its
    # own, and this pass folds it into every consumer. It runs after the fusion
    # instead.
    online_reduction = layer_params.get("fuse_dependant_reductions", False)
    epilogue_spans_reduction = layer_params.get("epilogue_spans_reduction_dim", True)
    if not online_reduction:
        func = apply_registered_pass(func, "linalg-fuse-elementwise-ops")
        lh_transform.cleanup(func)

    # WG row tiling
    generic_ops = structured.structured_match(anytype, mod, ops=["linalg.generic"])
    leaf_generic = transform_ext.extract_handle(generic_ops, -1)
    _, [wg_loop], _ = lh_transform.tile(
        leaf_generic,
        tile_sizes=(layer_params["wg_rows"],),
        fuse_producers=True,
        use_forall=True,
        apply_cleanup=False,
    )
    lh_transform.cleanup(func)
    wg_loop = match_and_split(func, ops={"scf.forall"}, nhandles=1)[0]

    # Reduction dimension tiling.
    # 1. Tile the leaf elemwise linalg.generic op and fuse its elemwise
    #    linalg.generic producers into the resulting loop.
    # 2. Tile each reduction linalg.generic op (from last to first) and fuse its
    #    elemwise producers into the resulting loop.

    def fuse_elemwise_producers_to_loop(target, parent_loop):
        """Fuses all elementwise producer ops of `target` into `parent_loop`."""
        producers = transform_ext.trace_producers(target)
        elemwise_producers = transform_ext.filter_elementwise(producers)
        elemwise_producers = transform_ext.filter_by_name(
            elemwise_producers,
            "linalg.generic",
        )
        _, fused_loop = structured.structured_fuse_into_containing_op(
            anytype,
            anytype,
            producer_op=elemwise_producers,
            containing_op=parent_loop,
        )
        return fused_loop

    generic_ops = match(wg_loop, ops={"linalg.generic"})
    elemwise_ops = transform_ext.filter_elementwise(generic_ops)
    leaf_elemwise = transform_ext.extract_handle(elemwise_ops, -1)
    reduction_ops = transform_ext.filter_reduction_ops(generic_ops)

    reduction_tile_size = [0, reduction_step_size]

    def tile_leaf_elemwise():
        """Tiles the trailing elemwise op and fuses its elemwise producers in.

        A no-op when the epilogue does not span the reduction axis: it then has
        no loop to tile (a row-wise norm's epilogue only reads the reduced
        accumulators) and its producers are already inside the reduction loops.
        """
        if not epilogue_spans_reduction:
            return
        tiled_elemwise, tile_loop = structured.TileUsingForOp(
            leaf_elemwise, sizes=reduction_tile_size
        ).results
        fuse_elemwise_producers_to_loop(tiled_elemwise, tile_loop)

    # Tile trailing elemwise op first, unless the reductions are fused: the
    # online form needs the chain still rooted at the untiled reductions.
    if not online_reduction:
        tile_leaf_elemwise()

    def tile_and_fuse_reduction(reduction_op, tile_sizes):
        # Tile the reduction op.
        _, tiled_op, _, tile_loop = structured.structured_tile_reduction_using_for(
            [anytype],
            anytype,
            anytype,
            anytype,
            target=reduction_op,
            tile_sizes=tile_sizes,
        )
        # Fuse all elemwise producers into the tiled leaf loop.
        fuse_elemwise_producers_to_loop(tiled_op, tile_loop)

    if online_reduction:
        # One-pass (online) form. Instead of one loop over the reduction axis
        # per reduction, fold the whole `R1 -> E -> R2` chain into R1's loop:
        # `fuse_dependant_reduction_ops` moves `E` and `R2` into it and inserts
        # the correction that rescales R2's running accumulator whenever R1's
        # changes. For softmax the chain is `max -> exp(x - max) -> sum`, so the
        # running max and the running normalizer are accumulated together and
        # the input is read once here instead of twice.
        #
        # `E` keeps a second, unfused copy outside the loop, which is what the
        # normalizing divide reads: the tiles the fused copy computes use the
        # *running* max, and only the accumulator the correction rescales is
        # valid at the end. That copy is tiled into the trailing elemwise loop
        # below, so it is recomputed per tile rather than materialized whole.
        r1_op = transform_ext.extract_handle(reduction_ops, 0)
        r2_op = transform_ext.extract_handle(reduction_ops, -1)
        # `E` is R2's only input; R2 reduces nothing else.
        e_op = transform.get_producer_of_operand(anytype, r2_op, operand_number=0)

        # Tile R1 along the reduction axis. Unlike `tile_reduction_using_for`
        # this carries the accumulator at its final shape rather than a partial
        # one, which is what the fusion needs: the correction reads the running
        # value per parallel slice. The marker attribute is how the fusion op
        # recognizes the producer loop.
        _, reduction_loop = structured.structured_tile_using_for(
            anytype,
            [anytype],
            r1_op,
            dynamic_sizes=[],
            interchange=[],
            static_sizes=reduction_tile_size,
            scalable_sizes=[False] * len(reduction_tile_size),
        )
        transform.annotate(reduction_loop, "__reduction_loop__")
        structured.structured_fuse_dependant_reduction_ops(
            anytype, reduction_loop, e_op, r2_op
        )

        # Now tile the divide, pulling the leftover full-extent `E` in with it.
        tile_leaf_elemwise()
    else:
        # Tile and fuse the reduction loops in reverse order. After each fusion
        # step, DCE removes the dead untiled elementwise epilogue so it cannot
        # create a cross-loop use that breaks the next tile-fuse iteration. Note
        # that DCE does not invalidate the reduction loop handles as the tracking
        # listener only invalidates modified handles and the reduction loops are
        # alive and thus not removed.
        reduction_ops = transform_ext.reverse_handles(reduction_ops)
        with lh_transform.foreach(reduction_ops) as reduction_op:
            tile_and_fuse_reduction(reduction_op, reduction_tile_size)
            transform.apply_dce(wg_loop)
            transform.yield_()

    # Fuse all sibling elementwise ops in scf.for loops.
    func = apply_registered_pass(func, "linalg-fuse-elementwise-ops")

    # Cleanup after tiling and fusion.
    transform.apply_cse(func)
    canonicalize(func)

    if online_reduction:
        # The fused `E` still writes each tile back into the full-extent
        # iter_arg it was carved out of, a loop result nothing reads now that
        # the divide recomputes the term. Neither DCE nor canonicalization can
        # see that -- the writes feed the loop's yield, so the chain is only
        # dead once the loop *results* are known dead, and scf.for only folds
        # away an iter_arg whose region argument is unused too.
        # `remove-dead-values` runs a liveness analysis across the loop and
        # unwinds the whole chain.
        func = apply_registered_pass(func, "remove-dead-values")
        transform.apply_cse(func)
        canonicalize(func)

    if stop_at_stage == "tiled":
        raise PipelineInterrupt()

    if online_reduction:
        # Same as `vectorize`, with a CSE inserted before the subset hoisting.
        # Vectorization emits one `transfer_read` per linalg op reading a given
        # iter_arg -- the running max is read by its own reduction *and* by the
        # correction's "old" term -- and subset hoisting refuses to hoist a
        # read/write pair while any other subset op on the same tensor overlaps
        # it. Merging the duplicate reads first is what keeps the accumulator
        # loop-carried in a register instead of re-read and re-written per tile.
        # `disable_multi_reduction_to_contract_patterns` keeps the reduction as a
        # `vector.multi_reduction`. Without it, a reduction whose per-element term
        # is a product -- an L2 norm's `sum_j (x/m)^2`, once the elementwise term
        # has been folded into the reduction body -- is rewritten to a
        # `vector.contract` reducing a 2-D tile to a 1-D result. XeGPU expects
        # contractions to be DPAS-shaped and asserts on that form, and lowering
        # the contract back afterwards is worse: the only strategy that applies
        # scalarizes it into one `vector.reduction` per row. Softmax and layer norm
        # are unaffected -- their reduction terms are not products, so no contract
        # was formed either way.
        func = get_named_func(mod, payload_func_name)
        func = structured.structured_vectorize_children_and_apply_patterns(
            anytype,
            func,
            fold_type_extensions_into_contract=True,
            disable_multi_reduction_to_contract_patterns=True,
        )
        transform.apply_cse(func)
        reduction_loops = match(func, ops={"scf.for"})
        lh_transform.loop_hoisting(reduction_loops)
        lh_transform.cleanup(func)
    else:
        func = vectorize(mod, payload_func_name=payload_func_name)

    # Relax the float semantics of the whole kernel. These reductions are
    # transcendental-heavy, and with strict IEEE `exp` the kernel is bound by the
    # precise polynomial expansion rather than by memory; `fast` lets the backend
    # use the hardware transcendental unit, which is worth more here than any
    # tiling choice.
    transform_ext.set_fastmath(func)

    # With that in place, rewrite `exp(a) / exp(b)` into `exp(a - b)`. The online
    # rescale factor comes out of the fusion as `exp(-m_new) / exp(-m_old)`, so
    # this trades two transcendentals and a divide per tile for one subtract and
    # one exp. A no-op for the unfused form, which has no such divide.
    #
    # Both run here rather than at linalg level: the two exponentials and the
    # divide live in separate linalg ops until vectorization merges them into a
    # single block, and the divide originates from a `linalg.elementwise`, which
    # carries no `fastmath` attribute for `fold_exp_div` to key off.
    transform_ext.fold_exp_div(func)

    transform.apply_cse(func)
    canonicalize(func)

    if stop_at_stage == "vectorized":
        raise PipelineInterrupt()

    # bufferize
    mod = bufferize(mod)

    if stop_at_stage == "bufferized":
        raise PipelineInterrupt()

    convert_to_gpu_launch(mod, payload_func_name)

    func = get_named_func(mod, payload_func_name)
    # set the number of threads for the gpu.launch operation
    launch_op = match_and_split(func, ops={"gpu.launch"})
    num_subgroups = layer_params["wg_rows"] // layer_params["sg_rows"]
    num_threads = num_subgroups * layer_params["subgroup_size"]
    xegpu.set_gpu_launch_threads(launch_op[0], threads=[num_threads, 1, 1])

    # outline gpu func
    func = apply_registered_pass(func, "lower-affine")
    canonicalize(func)
    func = apply_registered_pass(func, "gpu-launch-sink-index-computations")
    mod = apply_registered_pass(mod, "gpu-kernel-outlining")
    transform.apply_cse(mod)

    if stop_at_stage == "gpu-outlining":
        raise PipelineInterrupt()

    mod = convert_vector_to_xegpu(mod)
    lh_transform.cleanup(mod)

    if stop_at_stage == "xegpu-initial":
        raise PipelineInterrupt()

    # Set layout attributes on the stores. These are the anchors the wg-level
    # layout propagation works backwards from, so every store needs one.
    #
    # `xegpu.store` covers the scattered form. `convert-vector-to-xegpu` only
    # emits an `xegpu.store_nd` block store for a vector of rank >= 2 and routes
    # rank-1 writes to the scatter path, so a kernel whose result is one value per
    # row (a norm) reaches this point with no `store_nd` at all -- without an
    # anchor on the scatter, propagation has nothing to start from and fails with
    # "op has users but no layout assigned for its result".
    gpu_mod = match_and_split(mod, ops={"gpu.module"})[0]
    gpu_func = match(gpu_mod, ops={"gpu.func"})
    store_nd_ops = match(gpu_func, ops={"xegpu.store_nd"})
    store_matrix_ops = match(gpu_func, ops={"xegpu.store_matrix"})
    store_scatter_ops = match(gpu_func, ops={"xegpu.store"})
    if epilogue_spans_reduction:
        sg_layout = [layer_params["wg_rows"] // layer_params["sg_rows"], 1]
        sg_data = [layer_params["sg_rows"], layer_params["reduction_step_size"]]
    else:
        # The result is one value per row, so the store is rank 1: distribute the
        # rows over the subgroups, sg_rows each, with no column dim to split.
        sg_layout = [layer_params["wg_rows"] // layer_params["sg_rows"]]
        sg_data = [layer_params["sg_rows"]]
    with lh_transform.foreach(store_nd_ops) as store_op:
        xegpu.set_anchor_layout(store_op, sg_layout=sg_layout, sg_data=sg_data)
        transform.yield_()
    with lh_transform.foreach(store_matrix_ops) as store_op:
        xegpu.set_anchor_layout(store_op, sg_layout=sg_layout, sg_data=sg_data)
        transform.yield_()
    with lh_transform.foreach(store_scatter_ops) as store_op:
        xegpu.set_anchor_layout(store_op, sg_layout=sg_layout, sg_data=sg_data)
        transform.yield_()

    if stop_at_stage == "xegpu-wg":
        raise PipelineInterrupt()

    return mod
