from mlir import ir
from mlir.dialects import arith, ext, math, transform
from mlir.dialects.transform import DiagnosedSilenceableFailure

from lighthouse.dialects.transform.transform_ext import TransformExtensionDialect


def _fastmath_allows_transform(op: ir.Operation) -> bool:
    """
    True if `op`'s `fastmath` flags permit `exp(a)/exp(b)` -> `exp(a-b)`.

    The rewrite is not IEEE-preserving: it changes the rounding of the two
    intermediate exponentials into a single one, and it turns a division into a
    subtraction. `reassoc` is what licenses that regrouping, and the flag set
    also has to cover the values the identity ignores -- `exp(a)/exp(b)` is
    finite when both exponentials overflow or underflow together, whereas
    `exp(a-b)` is not (and vice versa), so `nnan`/`ninf` are required too.
    `fastmath<fast>` is the only flag set that provides all three.
    """
    if "fastmath" not in op.attributes:
        return False
    return str(op.attributes["fastmath"]) == "#arith.fastmath<fast>"


def _exp_operand(value: ir.Value) -> ir.Value | None:
    """The argument of `value`'s defining `math.exp`, if it is a fast-math one."""
    producer = value.owner
    if not isinstance(producer, ir.Operation):
        producer = getattr(producer, "operation", None)
    if producer is None or producer.name != "math.exp":
        return None
    if not _fastmath_allows_transform(producer):
        return None
    return producer.operands[0]


def _collect_divf_ops(root: ir.Operation) -> list[ir.Operation]:
    ops: list[ir.Operation] = []

    def collect(op: ir.Operation) -> ir.WalkResult:
        if op.name == "arith.divf":
            ops.append(op)
        return ir.WalkResult.ADVANCE

    root.walk(collect, ir.WalkOrder.PRE_ORDER)
    return ops


def _rewrite_divf(div_op: ir.Operation, rewriter: transform.TransformRewriter) -> bool:
    if not _fastmath_allows_transform(div_op):
        return False

    numerator = _exp_operand(div_op.operands[0])
    denominator = _exp_operand(div_op.operands[1])
    if numerator is None or denominator is None:
        return False

    # exp(a)/exp(b) and exp(a-b) only agree when a and b -- hence a-b -- share the
    # result type; a mixed-precision chain is left alone.
    if numerator.type != denominator.type:
        return False
    if numerator.type != div_op.results[0].type:
        return False

    fast = arith.FastMathFlags.fast
    with ir.InsertionPoint(div_op), div_op.location:
        difference = arith.SubFOp(numerator, denominator, fastmath=fast)
        replacement = math.ExpOp(difference.result, fastmath=fast)
    # The two original exps are left in place; they are dead only if nothing else
    # reads them, which DCE/CSE afterwards decides.
    rewriter.replace_op(div_op, [replacement.result])
    return True


class FoldExpDivOp(TransformExtensionDialect.Operation, name="fold_exp_div"):
    """
    Rewrite `exp(a) / exp(b)` into `exp(a - b)`.

    Trades a transcendental and a divide for a subtract. Applies only when the
    `arith.divf` and both `math.exp` producers all carry `fastmath<fast>`; see
    `_fastmath_allows_transform` for why nothing weaker is accepted. Ops without
    the flag are skipped rather than rejected, so this is safe to run over a whole
    function.

    The motivating case is the online-softmax rescale factor emitted by
    `transform.structured.fuse_dependant_reduction_ops`, which is
    `exp(-m_new) / exp(-m_old)`; this turns it into the single `exp(m_old - m_new)`
    a hand-written flash-attention kernel uses. Run `set_fastmath` first: the
    divide originates from a `linalg.elementwise`, which has no `fastmath`
    attribute to inherit.

    Args:
        target: Handle to root ops to rewrite within (e.g. func.func).
    Returns:
        Handle to the (possibly) rewritten target roots.
    """

    target: ext.Operand[transform.AnyOpType]
    rewritten_ops: ext.Result[transform.AnyOpType[()]] = ext.infer_result()

    @classmethod
    def attach_interface_impls(cls, context=None):
        cls.TransformOpInterfaceModel.attach(cls.OPERATION_NAME, context=context)
        cls.MemoryEffectsOpInterfaceModel.attach(cls.OPERATION_NAME, context=context)

    class TransformOpInterfaceModel(transform.TransformOpInterface):
        @staticmethod
        def apply(
            op: "FoldExpDivOp",
            rewriter: transform.TransformRewriter,
            results: transform.TransformResults,
            state: transform.TransformState,
        ) -> DiagnosedSilenceableFailure:
            targets = state.get_payload_ops(op.target)

            for target in targets:
                for div_op in _collect_divf_ops(target):
                    _rewrite_divf(div_op, rewriter)

            results.set_ops(op.rewritten_ops, targets)
            return DiagnosedSilenceableFailure.Success

        @staticmethod
        def allow_repeated_handle_operands(_op: "FoldExpDivOp") -> bool:
            return False

    class MemoryEffectsOpInterfaceModel(ir.MemoryEffectsOpInterface):
        @staticmethod
        def get_effects(op: ir.Operation):
            return (
                transform.only_reads_handle(op.op_operands)
                + transform.produces_handle(op.results)
                + transform.modifies_payload()
            )


def fold_exp_div(
    target: ir.Value[transform.AnyOpType],
) -> ir.Value[transform.AnyOpType]:
    """snake_case wrapper to create FoldExpDivOp."""
    op = FoldExpDivOp(target=target)
    return op.rewritten_ops
