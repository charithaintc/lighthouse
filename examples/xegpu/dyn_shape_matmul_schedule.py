"""
Transform schedule for the dynamic-shape XeGPU matmul example.

Kept local to the example (rather than added to
``lighthouse.schedule.xegpu.mlp_schedule``) so that experiments with a dynamic
reduction dimension cannot regress the existing static schedules.

The schedule is being built up stage by stage. Wired so far:

  * ``initial``: the payload is matched and handed back untouched.
  * ``tiled``:   workgroup tiling of the parallel dims (M, N) into an scf.forall,
                 then tiling of the dynamic reduction dim K into an scf.for with a
                 runtime trip count and an affine.min-clamped last tile.
  * ``vectorized``: pad the dynamic K tile to `k_tile` with zeros, then
                 vectorize; see `_pad_k_and_vectorize` below.
  * ``bufferized``: one-shot bufferization, see `_bufferize` below.
  * ``gpu-outlining``: map the workgroup scf.forall onto gpu.launch and outline
                 the body into a gpu.func. Masks survive unchanged.
  * ``xegpu-initial``: convert-vector-to-xegpu. The contraction reaches
                 `xegpu.dpas`, but the masked A/B loads do NOT reach
                 `xegpu.load_nd`. See the FIXME in
                 bundle_dyn_shape_matmul_schedule.

Requesting a later stage raises, rather than silently dumping earlier IR.

The vectorize and bufferize stages are re-implemented locally rather than reusing
`lowering_common.vectorize_bufferize_and_outline_gpu_func`, because the stock
`vectorize()` infers vector sizes from static shapes and silently no-ops on a
dynamic K tile. This is prototype code; if it works out, the masked path belongs
back in lowering_common behind a flag.
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
    match_and_split,
)
from lighthouse.schedule import schedule_boilerplate
from lighthouse.schedule.parameters import ScheduleParameters
from lighthouse.schedule.xegpu import XeGPUParameterSelector
from lighthouse.schedule.xegpu.lowering_common import (
    convert_to_gpu_launch,
    convert_vector_to_xegpu,
    get_payload_func,
    outline_gpu_function,
    vectorize,
)
from lighthouse.schedule.xegpu.xegpu_specs import XeGPUSpecs

# Lowering stages in pipeline order. Stages after `initial` are not implemented
# yet; extend IMPLEMENTED_STAGES as each one is added.
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
    # Runs to completion, but the result is only partially converted -- see the
    # FIXME in bundle_dyn_shape_matmul_schedule.
    "xegpu-initial",
)

# Parameters the schedule needs so far. Filled from the cost model / parameter DB
# when the caller does not supply them.
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
    """
    Generate the transform schedule module for the dynamic-shape matmul payload.

    Args:
        params: Per-layer schedule parameters (m, n, k and tile sizes).
        payload_func_name: Name of the payload function to transform.
        device: Target GPU device, used for parameter selection in later stages.
        stop_at_stage: Stop after this stage and leave the payload as-is.
        ab_type: Element type of A/B, needed for the K padding value.
        acc_type: Element type of the accumulator, needed for the padding value.
    """
    assert params is not None and len(params) > 0, "params must be provided."
    assert len(params) == 1, "the dynamic-shape matmul example has a single layer."

    layer_params = params[0]
    m, n, k = (layer_params.get(d) for d in ("m", "n", "k"))
    assert all(d is not None for d in (m, n, k)), "m, n, k must be provided in params"

    param_selector = XeGPUParameterSelector(device=device)
    if not all(p in layer_params for p in REQUIRED_PARAMS):
        # K is dynamic in the payload, so the nominal K only feeds the tile-size
        # heuristic here; it never reaches the IR.
        generated = param_selector.get_parameters(
            (m, n, k),
            layer_params.get("transpose_a", False),
            layer_params.get("transpose_b", False),
        )
        # Caller-provided values win over generated ones.
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
    wg_m: int,
    wg_n: int,
    k_tile: int,
    ab_type: ir.Type,
    acc_type: ir.Type,
) -> None:
    """
    Pad the dynamic K tile up to `k_tile` with zeros, then vectorize.

    Vectorizing the dynamic tile directly leaves a mask around the
    `vector.contract`, which cannot reach XeGPU: there is no maskable
    `xegpu.dpas`. This route avoids ever creating that mask.

    The trick is to make the matmul's operands *statically* shaped before
    vectorizing it. `tensor.pad` on the K dim does exactly that, and -- unlike
    `tensor.extract_slice` -- `tensor.pad` is vectorizable, so the pad itself
    becomes the masked load. The zeros it writes are what make the unmasked
    contraction correct: disabled K lanes contribute 0 * 0 = 0 to an `add`
    reduction.

    Before:
      %5 = affine.min #map(%arg6)[%dim]
      %a = tensor.extract_slice ... to tensor<128x?xf16>
      %b = tensor.extract_slice ... to tensor<?x128xf16>
      linalg.matmul ins(%a, %b) outs(%acc : tensor<128x128xf32>)

    After padding (note the static pad result types):
      %8 = affine.apply #map1(%5)                         // k_tile - %5
      %pa = tensor.pad %a low[0,0] high[0,%8] {0.0 : f16} : ... to tensor<128x16xf16>
      %pb = tensor.pad %b low[0,0] high[%8,0] {0.0 : f16} : ... to tensor<16x128xf16>
      linalg.matmul ins(%pa, %pb) outs(%acc)              // fully static

    After vectorizing (masks only on the loads, contract is bare):
      %9  = vector.create_mask %c128, %8 : vector<128x16xi1>
      %10 = vector.mask %9 { vector.transfer_read ... } -> vector<128x16xf16>
      %11 = vector.create_mask %8, %c128 : vector<16x128xi1>
      %12 = vector.mask %11 { vector.transfer_read ... } -> vector<16x128xf16>
      %15 = vector.contract ... %10, %12, %arg7 -> vector<128x128xf32>
    """
    anytype = transform.AnyOpType.get()
    zero_ab = ir.FloatAttr.get(ab_type, 0.0)
    zero_acc = ir.FloatAttr.get(acc_type, 0.0)

    # Pad iteration dim 2 (K) of the matmul up to a multiple of k_tile. Since the
    # tile size is already clamped to <= k_tile, "multiple of k_tile" means
    # exactly k_tile, which is why the pad result types come out static.
    #
    # copy_back_op="none" matters: the default inserts a
    # bufferization.materialize_in_destination that blocks hoisting the MxN
    # accumulator out of the K loop.
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

    # transform.print_(target=func, name="after padding matmul")

    # The two pads need different vector sizes: A's tile is wg_m x k_tile, B's is
    # k_tile x wg_n. Passing the same sizes to both silently fails the pad
    # vectorization precondition on one of them.
    pad_a, pad_b = match_and_split(func, ops={"tensor.pad"}, nhandles=2)
    structured.structured_vectorize(
        pad_a, [], static_vector_sizes=[wg_m, k_tile], scalable_sizes=[False, False]
    )
    structured.structured_vectorize(
        pad_b, [], static_vector_sizes=[k_tile, wg_n], scalable_sizes=[False, False]
    )
    # transform.print_(target=func, name="after vectorizing pads")

    # The matmul is fully static now, so the stock vectorizer works unchanged --
    # no explicit sizes, and no mask around the contraction.
    func = vectorize(mod=None, payload_func=func)

    # transform.print_(target=func, name="after vectorizing matmul")


def _bufferize(mod: ir.Value) -> ir.Value:
    """
    One-shot bufferization of the masked-vectorized payload.

    Same recipe as `lowering_common.bufferize`, inlined here to keep the
    prototype self-contained. Folding the memref aliases is what turns the
    per-tile `memref.subview` chains back into direct indexed transfers off the
    function arguments.
    """
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

    # NOTE The stock flow runs `buffer-deallocation-pipeline` here (see
    # lowering_common.convert_allocs_to_gpu). It is omitted on purpose:
    # `-ownership-based-buffer-deallocation` materializes an `arith.constant false`
    # ownership indicator *inside* the `vector.mask` regions we just created,
    # which violates vector.mask's single-op region constraint and fails with
    # "'vector.mask' op expects only one operation to mask".
    #
    # Safe for now because this payload accumulates in place and allocates
    # nothing. It has to be dealt with (mask lowering first, or a fix upstream)
    # before any variant that needs a temporary buffer.

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
    # Collapse the elementwise epilogue (the acc/result type conversions) so
    # there is a single tileable consumer to anchor the workgroup loop on.
    func = apply_registered_pass(func, "linalg-fuse-elementwise-ops")

    # Anchor on the last tileable consumer rather than the matmul itself, so any
    # elementwise epilogue lands inside the workgroup loop instead of after it.
    matmul_op = match(func, ops={"linalg.matmul"})
    consumers = transform_ext.get_tileable_consumers(matmul_op)
    leaf_consumer_op = transform_ext.extract_handle(consumers, -1)

    # Workgroup tiling of the parallel dims only. Passing two tile sizes for a
    # 3-loop matmul leaves the innermost (reduction) loop untiled, so K remains a
    # single dynamic-trip-count reduction inside each tile.
    _, [wg_loop], _ = lh_transform.tile(
        leaf_consumer_op,
        tile_sizes=wg_tile,
        fuse_producers=True,
        use_forall=True,
        apply_cleanup=False,
    )
    # Reduction tiling. The K extent is dynamic, so the loop gets a runtime upper
    # bound (memref.dim) and, since k_tile cannot be proven to divide it, an
    # affine.min clamps the last tile. That clamp is the seam the vectorizer will
    # later have to mask.
    wg_matmul = match(wg_loop, ops={"linalg.matmul"})
    _, [k_loop], _ = lh_transform.tile(wg_matmul, tile_sizes=[0, 0, k_tile])
    lh_transform.cleanup(wg_loop)

    lh_transform.cleanup(func)

    if stop_at_stage == "tiled":
        raise PipelineInterrupt()

    # The stock `lowering_common.vectorize` cannot be used directly on a dynamic K
    # tile: it goes through transform.structured.vectorize_children_and_apply_patterns,
    # whose VectorizationPattern calls linalg::vectorize with inputVectorSizes={}
    # and so infers sizes from the op's static shape. On
    # `linalg.matmul ins(tensor<256x?xf16>, tensor<?x256xf16>)` it returns a plain
    # pattern-match failure inside a greedy rewriter -- no diagnostic -- and the
    # matmul is left as-is, which then flows all the way down to a gpu.func
    # holding a raw linalg.matmul and no DPAS.
    func = get_payload_func(mod, func_name=payload_func_name)
    matmul_op = match(func, ops={"linalg.matmul"})
    _pad_k_and_vectorize(
        func,
        matmul_op,
        wg_m=layer_params["wg_m"],
        wg_n=layer_params["wg_n"],
        k_tile=k_tile,
        ab_type=ab_type,
        acc_type=acc_type,
    )
    if stop_at_stage == "vectorized":
        raise PipelineInterrupt()

    mod = _bufferize(mod)
    if stop_at_stage == "bufferized":
        raise PipelineInterrupt()

    # Map the workgroup scf.forall onto gpu.launch and outline the body into a
    # gpu.func. The masks are untouched here: `affine.min` is lowered to
    # arith.subi/arith.minsi by lower-affine, and the dynamic K extent travels
    # into the kernel as a `memref<M x ? x f16>` argument, so memref.dim still
    # recovers it inside gpu.func.
    convert_to_gpu_launch(mod, payload_func_name=payload_func_name)
    mod = outline_gpu_function(
        mod, payload_func_name=payload_func_name, gpu_specs=gpu_specs, params=params
    )
    if stop_at_stage == "gpu-outlining":
        raise PipelineInterrupt()

    # convert-vector-to-xegpu is not mask-aware. Two blockers were in the way;
    # padding K clears the second one, the first remains.
    #
    #  1. FIXME Masked A/B loads never reach XeGPU. `transferPreconditions` in
    #     VectorToXeGPU.cpp bails out on any transfer with a mask operand
    #     ("Masked transfer is not supported"), so the loads stay as
    #     `vector.transfer_read` -- no xegpu.load_nd, hence no block loads and
    #     nothing for the xegpu-wg stage to attach prefetches to. The C
    #     accumulator (static MxN, unmasked) does become create_nd_tdesc +
    #     load_nd/store_nd, so the kernel comes out half-converted.
    #     Note the descriptors are emitted with `boundary_check = false`;
    #     hardware bounds-checking against the dynamic extent may be the natural
    #     way to express this mask, rather than teaching the conversion about
    #     `vector.mask`.
    #
    #  2. RESOLVED by padding K. There is no maskable DPAS: vectorizing the
    #     dynamic tile directly turns `vector.contract` into `xegpu.dpas` inside a
    #     `vector.mask ... : vector<wg_m x wg_n x k_tile x i1>` region, and
    #     xegpu.dpas does not implement MaskableOpInterface, so the verifier
    #     rejects it ("expects a MaskableOpInterface within the mask region").
    #     `_pad_k_and_vectorize` never creates that mask, so the dpas is bare.
    #
    # Lower the remaining load masks to transfer mask operands. Without this the
    # conversion fires *inside* the `vector.mask` region, expands the read into
    # create_nd_tdesc + load_nd (two ops, so the region constraint breaks) and
    # drops the mask entirely.
    #
    # NOTE This cannot use the `lower-vector-mask` pass here: that pass is
    # declared `Pass<"lower-vector-mask", "func::FuncOp">`, and after outlining
    # the masks live in a `gpu.func`. Apply the same patterns directly instead --
    # the pass body is just populateVectorMaskLoweringPatternsForSideEffectingOps
    # plus MaskOp canonicalization, which is what `lower_masked_transfers` and
    # the canonicalizer give us.
    gpu_func = match(mod, ops={"gpu.func"})
    with ir.InsertionPoint(transform.apply_patterns(gpu_func).patterns):
        transform_vector.apply_patterns_vector_lower_masked_transfers()
    lh_transform.cleanup(gpu_func)

    mod = convert_vector_to_xegpu(mod)
    if stop_at_stage == "xegpu-initial":
        raise PipelineInterrupt()

    # Next steps, in the order we will add them:
    #   - "xegpu-wg":      attach workgroup layouts / prefetches.
    raise NotImplementedError(
        f"stop_at_stage={stop_at_stage!r} is not implemented yet; "
        f"dyn_shape_matmul_schedule currently supports {IMPLEMENTED_STAGES}."
    )
