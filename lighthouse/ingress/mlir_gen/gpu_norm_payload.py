"""Generate MLIR payload for GPU row-wise vector norms (L1 / L2)."""

from mlir import ir
from mlir.dialects import arith, bufferization, linalg, math, tensor

from lighthouse.utils.mlir import func_cif
from lighthouse.ingress.mlir_gen.utils import (
    affine_map,
    emit_buf_to_tensor,
    parallel,
    reduction,
)

NORM_KINDS = ("l1", "l2")


def emit_norm_generics(
    input_tensor, out_init, dtype, norm_kind: str, normalize: bool = False
):
    """Emit the linalg ops of a numerically stable row-wise Lp norm.

    Over the iteration space (i, j) reducing on j:

        m[i]    = max_j |x[i, j]|                    <- R1
        t[i, j] = |x[i, j] / m[i]|      (L1)         <- E
                  (x[i, j] / m[i])^2    (L2)
        s[i]    = sum_j t[i, j]                      <- R2
        n[i]    = m[i] * s[i]           (L1)
                  m[i] * sqrt(s[i])     (L2)

    and then, depending on `normalize`, either returns the norms themselves

        out[i]    = n[i]

    or divides the input through by them, i.e. `torch.nn.functional.normalize`

        out[i, j] = x[i, j] / n[i]

    The two differ only in the epilogue. `normalize` keeps `n` folded into a
    single full-extent op, so the result is rank 2 and the whole kernel has the
    same shape as softmax: a fused reduction loop plus one elementwise pass. The
    bare-norm form has no full-extent epilogue at all -- its result is one scalar
    per row -- so the fused version reads the input exactly once.

    Scaling by the row maximum is what makes this the *stable* form: it keeps
    every term of the sum in [0, 1], so `x[i, j]^2` cannot overflow for an L2
    norm whose true value is representable. `m` factors back out afterwards
    because it does not depend on the reduced axis.

    This is the same `R1 -> E -> R2` shape as softmax -- a max reduction feeding
    an elementwise term feeding a sum -- so it fuses into one online loop the
    same way. `E` is factorizable in `m` (`|x/m| = |x| * (1/m)`,
    `(x/m)^2 = x^2 * (1/m^2)`), which is what the online correction needs: the
    per-tile rescale works out to `m_old/m_new` for L1 and `(m_old/m_new)^2` for
    L2.

    Two details of `E` matter for the fused form:

    * The **divide comes first**, with `abs` (L1) or the square (L2) applied to
      its result. The correction is derived by re-evaluating `E` with its data
      inputs replaced by the neutral element of the op that consumes them, and
      `math.absf` has no such element -- writing `E` as `|x| / m` would put `x`
      directly under the `abs` and the correction could not be built. Ordered
      this way, `x` is consumed by the `divf` and neutralizes to `1.0`, giving
      `E' = |1/m|` and `E' = (1/m)^2` respectively. The two forms are equal
      because `m > 0`: `|x|/m = |x/m|`, and squaring discards the sign anyway.
      This also keeps `abs` out of the iteration space as a separate op, so
      nothing is materialized at full extent.
    * `m` is initialized to `0.0`, not `-inf`. `|x| >= 0` makes zero a valid
      lower bound, and the first tile's correction factor is `m_old/m_new`, which
      would be `-inf` (and `-inf * 0 = NaN` against the zero-initialized sum) had
      the accumulator started at `-inf`.

    Args:
        input_tensor: The (M, N) input.
        out_init: Destination for the result -- (M,) for a bare norm, (M, N) when
            `normalize` is set.
        dtype: Element type.
        norm_kind: "l1" or "l2".
        normalize: Divide the input through by the norms instead of returning them.

    Returns:
        The result tensor.
    """
    assert norm_kind in NORM_KINDS, f"norm_kind must be one of {NORM_KINDS}"

    par_map_2d = affine_map(2, [ir.AffineDimExpr.get(0), ir.AffineDimExpr.get(1)])
    row_map_2d = affine_map(2, [ir.AffineDimExpr.get(0)])
    id_map_1d = affine_map(1, [ir.AffineDimExpr.get(0)])

    zero = arith.constant(dtype, 0.0)
    shape = tuple(ir.RankedTensorType(input_tensor.type).shape)
    row_shape = (shape[0],)

    # R1: m[i] = max_j |x[i, j]|
    max_init = linalg.fill(zero, outs=[tensor.empty(row_shape, dtype)])

    @linalg.generic(
        [input_tensor],
        [max_init],
        [par_map_2d, row_map_2d],
        [parallel, reduction],
    )
    def row_max(x, acc):
        return arith.MaximumFOp(math.absf(x), acc)

    # E: the scaled term, all-parallel and factorizable in `m`. Note the divide
    # precedes the abs / square -- see the note above.
    @linalg.generic(
        [input_tensor, row_max],
        [tensor.empty(shape, dtype)],
        [par_map_2d, row_map_2d, par_map_2d],
        [parallel, parallel],
    )
    def scaled(x, m, _out):
        ratio = arith.DivFOp(x, m).result
        if norm_kind == "l1":
            return math.AbsFOp(ratio)
        return arith.MulFOp(ratio, ratio)

    # R2: s[i] = sum_j t[i, j]
    sum_init = linalg.fill(zero, outs=[tensor.empty(row_shape, dtype)])

    @linalg.generic(
        [scaled],
        [sum_init],
        [par_map_2d, row_map_2d],
        [parallel, reduction],
    )
    def row_sum(t, acc):
        return arith.AddFOp(t, acc)

    def undo_scaling(m, s):
        """n = m * s (L1) or m * sqrt(s) (L2), from the row max and scaled sum."""
        total = s if norm_kind == "l1" else math.SqrtOp(s).result
        return arith.MulFOp(m, total).result

    if not normalize:
        # out[i] = n[i]
        @linalg.generic(
            [row_max, row_sum],
            [out_init],
            [id_map_1d, id_map_1d, id_map_1d],
            [parallel],
        )
        def result(m, s, _out):
            return undo_scaling(m, s)

        return result

    # out[i, j] = x[i, j] / n[i]. Reads the input a second time, so this is the
    # pass the fused form cannot remove -- exactly like softmax's divide.
    @linalg.generic(
        [input_tensor, row_max, row_sum],
        [out_init],
        [par_map_2d, row_map_2d, row_map_2d, par_map_2d],
        [parallel, parallel],
    )
    def normalized(x, m, s, _out):
        return arith.DivFOp(x, undo_scaling(m, s))

    return normalized


def generate_gpu_norm_payload(
    func_name: str,
    M: int,
    N: int,
    dtype: ir.Type,
    norm_kind: str = "l2",
    normalize: bool = False,
) -> ir.Module:
    """
    Generate MLIR module for a row-wise Lp norm payload.

    Both forms use the numerically stable max-scaled reduction chain (see
    `emit_norm_generics`). With `normalize` unset, the result is one norm per row:

        out[i] = sum_j |x[i, j]|             (L1)
        out[i] = sqrt(sum_j x[i, j]^2)       (L2)

    With `normalize` set, the input is divided through by those norms -- the
    `torch.nn.functional.normalize` / cosine-similarity preprocessing step -- and
    the result has the input's shape:

        out[i, j] = x[i, j] / out_norm[i]

    Args:
        func_name: Name of the payload function
        M: Number of rows
        N: Number of columns (the reduced dimension)
        dtype: MLIR element type (e.g., F32Type)
        norm_kind: "l1" or "l2"
        normalize: Emit the normalized input rather than the norms

    Returns:
        MLIR module containing the norm payload function
    """
    mod = ir.Module.create()
    shape = (M, N)
    row_shape = (M,)
    out_shape = shape if normalize else row_shape
    memref_t = ir.MemRefType.get(shape, dtype)
    out_memref_t = ir.MemRefType.get(out_shape, dtype)

    with ir.InsertionPoint(mod.body):
        # Function signature: payload(output, input)
        @func_cif(out_memref_t, memref_t, name=func_name)
        def payload(output, input_arg):
            emit_buf_to_tensor(output, restrict=True, writable=True)
            input_tensor = emit_buf_to_tensor(input_arg, restrict=True)

            result = emit_norm_generics(
                input_tensor,
                tensor.empty(out_shape, dtype),
                dtype,
                norm_kind,
                normalize,
            )
            bufferization.materialize_in_destination(
                None, result, output, restrict=True, writable=True
            )

    return mod
