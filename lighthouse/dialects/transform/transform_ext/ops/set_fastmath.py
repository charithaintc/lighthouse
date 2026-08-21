from mlir import ir
from mlir.dialects import ext, transform
from mlir.dialects.transform import DiagnosedSilenceableFailure

from lighthouse.dialects.transform.transform_ext import TransformExtensionDialect

NO_FASTMATH = "#arith.fastmath<none>"

# Ops that carry an optional `fastmath` attribute. Structured ops that expand to
# these later (e.g. `linalg.elementwise`) have no such attribute, so anything they
# emit has to be annotated after it materializes.
FASTMATH_OP_NAMES = frozenset(
    {
        "arith.addf",
        "arith.divf",
        "arith.maximumf",
        "arith.maxnumf",
        "arith.minimumf",
        "arith.minnumf",
        "arith.mulf",
        "arith.negf",
        "arith.remf",
        "arith.subf",
        "math.absf",
        "math.exp",
        "math.exp2",
        "math.expm1",
        "math.fma",
        "math.log",
        "math.powf",
        "math.rsqrt",
        "math.sqrt",
        "math.tanh",
    }
)


def _collect_fastmath_ops(root: ir.Operation) -> list[ir.Operation]:
    ops: list[ir.Operation] = []

    def collect(op: ir.Operation) -> ir.WalkResult:
        if op.name in FASTMATH_OP_NAMES:
            ops.append(op)
        return ir.WalkResult.ADVANCE

    root.walk(collect, ir.WalkOrder.PRE_ORDER)
    return ops


class SetFastmathOp(TransformExtensionDialect.Operation, name="set_fastmath"):
    """
    Set `fastmath<fast>` on every floating-point arith/math op under `target`.

    Ops that already carry a non-default `fastmath` attribute are left alone, so an
    explicitly-chosen weaker flag set is never widened. `#arith.fastmath<none>`
    counts as absent: it is the attribute's default, so it is present in the
    attribute dictionary but elided when printed, and it carries no intent.

    The motivating case is the online-softmax rescale factor emitted by
    `transform.structured.fuse_dependant_reduction_ops`: it is built from
    `linalg.elementwise`, which has no `fastmath` attribute, so the `arith.divf`
    it expands to comes out unannotated even though the surrounding payload ops
    are `fastmath<fast>`. `fold_exp_div` needs the whole chain annotated.

    Args:
        target: Handle to root ops to annotate within (e.g. func.func).
    Returns:
        Handle to the annotated target roots.
    """

    target: ext.Operand[transform.AnyOpType]
    annotated_ops: ext.Result[transform.AnyOpType[()]] = ext.infer_result()

    @classmethod
    def attach_interface_impls(cls, context=None):
        cls.TransformOpInterfaceModel.attach(cls.OPERATION_NAME, context=context)
        cls.MemoryEffectsOpInterfaceModel.attach(cls.OPERATION_NAME, context=context)

    class TransformOpInterfaceModel(transform.TransformOpInterface):
        @staticmethod
        def apply(
            op: "SetFastmathOp",
            rewriter: transform.TransformRewriter,
            results: transform.TransformResults,
            state: transform.TransformState,
        ) -> DiagnosedSilenceableFailure:
            targets = state.get_payload_ops(op.target)

            for target in targets:
                with target.context:
                    fast = ir.Attribute.parse("#arith.fastmath<fast>")
                for payload_op in _collect_fastmath_ops(target):
                    existing = payload_op.attributes.get("fastmath")
                    if existing is not None and str(existing) != NO_FASTMATH:
                        continue
                    payload_op.attributes["fastmath"] = fast

            results.set_ops(op.annotated_ops, targets)
            return DiagnosedSilenceableFailure.Success

        @staticmethod
        def allow_repeated_handle_operands(_op: "SetFastmathOp") -> bool:
            return False

    class MemoryEffectsOpInterfaceModel(ir.MemoryEffectsOpInterface):
        @staticmethod
        def get_effects(op: ir.Operation):
            return (
                transform.only_reads_handle(op.op_operands)
                + transform.produces_handle(op.results)
                + transform.modifies_payload()
            )


def set_fastmath(
    target: ir.Value[transform.AnyOpType],
) -> ir.Value[transform.AnyOpType]:
    """snake_case wrapper to create SetFastmathOp."""
    op = SetFastmathOp(target=target)
    return op.annotated_ops
