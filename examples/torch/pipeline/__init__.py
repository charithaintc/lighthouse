from __future__ import annotations
import os
import re

from abc import ABC, abstractmethod

from numpy import mod
from mlir import ir, passmanager
from mlir.dialects._omp_ops_gen import target
from mlir.execution_engine import ExecutionEngine
from mlir.dialects.bufferization import LayoutMapOption
from mlir.dialects.transform import (AnnotateOp, AnyOpType, AnyValueType, ApplyCanonicalizationPatternsOp,
                                     ApplyDeadCodeEliminationOp, GetOperandOp, GetParentOp,
                                     ApplyCommonSubexpressionEliminationOp, ApplyPatternsOp, FailurePropagationMode,
                                     NamedSequenceOp, OptionValueTypes, ParamConstantOp, PrintOp, SequenceOp,
                                     SplitHandleOp, YieldOp, apply_licm, apply_registered_pass, get_producer_of_operand)
from mlir.dialects.transform.interpreter import apply_named_sequence
from mlir.dialects.transform.gpu import MapNestedForallToThreads
from mlir.dialects.transform.loop import HoistLoopInvariantSubsetsOp, ForallToParallelOp
from mlir.dialects.transform.memref import MemRefEraseDeadAllocAndStoresOp, ApplyAllocToAllocaOp
from mlir.dialects.transform.bufferization import OneShotBufferizeOp
from mlir.dialects.transform.structured import (ApplyFoldUnitExtentDimsViaReshapesPatternsOp, FuseIntoContainingOp,
                                                MatchOp, TileUsingForOp, TileUsingForallOp, TileReductionUsingForOp,
                                                TileReductionUsingForallOp, VectorizeChildrenAndApplyPatternsOp)
from mlir.dialects.transform.xegpu import GetDescOp, SetDescLayoutOp, SetGPULaunchThreadsOp, SetOpLayoutAttrOp
from mlir.ir import InsertionPoint, UnitAttr
from torch import Tensor, nn
from torch_mlir.fx import OutputType, export_and_import
from typing import Callable, ContextManager, Dict, Iterable, List, Literal, Optional, Set, Tuple, Union
from typing_extensions import override
from .utils import torch_to_packed_args


def match(target: Union[ir.Value, ir.OpResult], ops: Union[str, Iterable[str]]) -> MatchOp:
    matcher = MatchOp.match_op_names(target, (ops, ) if isinstance(ops, str) else ops)
    return matcher


def split_handle(handle: Union[ir.Value, ir.OpResult], count: int) -> ir.OpResultList:
    return SplitHandleOp([AnyOpType.get()] * count, handle).results


def cse(target: Union[ir.Value, ir.OpResult]):
    ApplyCommonSubexpressionEliminationOp(target)


def dce(target: Union[ir.Value, ir.OpResult]):
    ApplyDeadCodeEliminationOp(target)


def apply_patterns(target: Union[ir.Value, ir.OpResult]) -> ir.InsertionPoint:
    return ir.InsertionPoint(ApplyPatternsOp(target).patterns)


def canonicalize(target: Union[ir.Value, ir.OpResult]):
    with apply_patterns(target):
        ApplyCanonicalizationPatternsOp()


def apply_pass(
        target: Union[ir.Operation, ir.Value, ir.OpView], *name_and_options:
    Union[str, Tuple[str, Dict[Union[str, ir.StringAttr], OptionValueTypes]]]) -> ir.Value:
    any = AnyOpType.get()
    for name in name_and_options:
        if isinstance(name, str):
            if name == "cse":
                cse(target)
            elif name == "canonicalize":
                canonicalize(target)
            else:
                target = apply_registered_pass(any, target, name)
        else:
            target = apply_registered_pass(any, target, name[0], options=name[1])
    return target


class Phase(ABC):

    @abstractmethod
    def apply(self, ctx: Phase.Context):
        ...

    @property
    def pipeline(self) -> Pipeline:
        if (pl := getattr(self, "__pipeline__", None)) is None:
            raise Exception("Phase is not associated with any Pipeline!")
        return pl

    @property
    def name(self) -> str:
        return getattr(self, "__phase_name__", self.__class__.__name__)

    @property
    def depends(self) -> Optional[Phase]:
        return getattr(self, "__depends_on__", None)

    def __call__(
        self,
        target: Union[nn.Module, ir.Module, str],
        *sample_args: Tensor,
        ir_context: Optional[ir.Context] = None,
        dump: bool = False,
        **sample_kwargs,
    ) -> ir.Module:
        if isinstance(target, Phase.Context):
            if (dep := self.depends):
                dep(target, dump=dump)
            elif dump:
                PrintOp(target=target.mod, name="Initial")
            self.apply(target)
            if dump:
                PrintOp(target=target.mod, name=self.name)
            return

        pl = self.pipeline
        target = pl.to_mlir_module(target, *sample_args, ir_context=ir_context, **sample_kwargs)
        # print("module before phase:", target)
        with pl.transform_module(target, dump) as mod:
            self(Phase.Context(mod, target), dump=dump)
        return target

    def __repr__(self):
        d = self.depends
        return self.name if d is None else f"{self.name} -> {d}"

    class Context:

        def __init__(self, mod: Union[ir.Value, ir.OpResult], payload: ir.Module):
            self._mod = mod
            self._payload = payload
            self.tile_sizes: Dict[ir.Operation, Tuple[int, ...]] = {}
            self.gpu_loop_tile_sizes: Dict[ir.Operation, Tuple[int, ...]] = {}

        @property
        def payload(self) -> ir.Module:
            return self._payload

        @property
        def mod(self) -> Union[ir.Value, ir.OpResult]:
            return self._mod

        @mod.setter
        def mod(self, value: Union[ir.Value, ir.OpResult]):
            self._mod = value
            self.func = None
            self.gpu_mod = None

        @property
        def func(self) -> ir.OpResult:
            if (fn := getattr(self, "__func_matcher__", None)) is None:
                self.func = fn = match(self.mod, "func.func").result
            return fn

        @func.setter
        def func(self, value: ir.OpResult):
            self.__func_matcher__ = value

        @property
        def gpu_mod(self) -> ir.OpResult:
            if (mod := getattr(self, "__gpu_mod_matcher__", None)) is None:
                self.gpu_mod = mod = match(self.mod, "gpu.module").result
            return mod

        @gpu_mod.setter
        def gpu_mod(self, value: ir.OpResult):
            self.__gpu_mod_matcher__ = value
            self.gpu_func = None

        @property
        def gpu_func(self) -> ir.OpResult:
            if (fn := getattr(self, "__gpu_func_matcher__", None)) is None:
                self.gpu_func = fn = match(self.gpu_mod, "gpu.func").result
            return fn

        @gpu_func.setter
        def gpu_func(self, value: ir.OpResult):
            self.__gpu_func_matcher__ = value


class PipelineMeta(type):

    def __new__(cls, name, bases, attrs):
        target_class = super().__new__(cls, name, bases, attrs)
        if len(bases) == 0 and name == "Pipeline":
            return target_class

        prev = None
        for name, value in attrs.items():
            if isinstance(value, Phase):
                assert name not in ("apply", "compile", "exec", "to_mlir_module", "transform_module", "default_context",
                                    "get_shared_libs")
                value.__phase_name__ = name
                value.__pipeline__ = target_class
                if prev is not None:
                    value.__depends_on__ = prev
                prev = value
        assert prev is not None, f"Pipeline {target_class} does not have any Phase!"
        target_class.__last_phase__ = prev
        return target_class


class Pipeline(metaclass=PipelineMeta):

    @classmethod
    def apply(
        cls,
        target: Union[nn.Module, ir.Module, str],
        *sample_args: Tensor,
        ir_context: Optional[ir.Context] = None,
        dump: bool = False,
        **sample_kwargs,
    ) -> ir.Module:
        return cls.__last_phase__(  # type: ignore[attr-defined]
            target,
            *sample_args,
            ir_context=ir_context,
            dump=dump,
            **sample_kwargs,
        )

    @classmethod
    def compile(
        cls,
        target: Union[nn.Module, ir.Module, str],
        *sample_args: Tensor,
        ir_context: Optional[ir.Context] = None,
        dump: bool = False,
        opt_level=2,
        **sample_kwargs,
    ) -> CompiledModule:
        if not isinstance(target, ir.Module):
            target = cls.apply(target, *sample_args, ir_context=ir_context, dump=dump, **sample_kwargs)
        func_name = None
        num_args = 0
        for op in target.body.operations:
            if op.name == "llvm.func" and (name := op.attributes["sym_name"].value):
                if name.startswith("mgpu"):
                    continue
                if name.startswith("_mlir_ciface_"):
                    num_args = len(op.body.blocks[0].arguments)
                
                else:
                    func_name = name
        assert func_name is not None, "No llvm.func found in the module!"
        return CompiledModule(target, func_name, num_args, opt_level=opt_level)

    @classmethod
    def exec(
        cls,
        target: Union[nn.Module, ir.Module, str],
        *kernel_args: Tensor,
        ir_context: Optional[ir.Context] = None,
        dump: bool = False,
        opt_level=2,
        **kernel_kwargs,
    ):
        if isinstance(target, nn.Module):
            import inspect
            sig = inspect.signature(target.forward).parameters
            sample_args = kernel_args[0:len(sig)]
        else:
            sample_args = []
        mod = cls.compile(target, *sample_args, ir_context=ir_context, dump=dump, opt_level=opt_level, **kernel_kwargs)
        return mod(*kernel_args)

    @classmethod
    def to_mlir_module(
        cls,
        target: Union[nn.Module, str],
        *sample_args: Tensor,
        ir_context: Optional[ir.Context] = None,
        **sample_kwargs,
    ) -> ir.Module:
        if isinstance(target, nn.Module):
            target = export_and_import(
                target,
                *sample_args,
                output_type=OutputType.LINALG_ON_TENSORS,
                **sample_kwargs,
            )
            target = str(target)

        if isinstance(target, str):
            target = ir.Module.parse(target, context=ir_context or cls.default_context())

        assert isinstance(target, ir.Module)
        for fn in [op for op in target.body.operations if op.operation.name == "func.func"]:
            with ir.InsertionPoint(fn), fn.location:
                fn.attributes["llvm.emit_c_interface"] = UnitAttr.get()

        return target

    @staticmethod
    def transform_module(target: ir.Module, dump: bool = False) -> ContextManager:

        class TransormContextManager(ContextManager):

            def __init__(self, target: ir.Module, dump: bool):
                self.target = target
                self.dump = dump

            def __enter__(self) -> ir.Value:
                self.ctx_ip = self.target.context.__enter__()
                self.loc_ip = self.target.body.region.owner.location.__enter__()
                self.mod = ir.Module.create()
                self.mod.operation.attributes["transform.with_named_sequence"] = (UnitAttr.get())
                self.mod_ip = ir.InsertionPoint(self.mod.body)
                self.mod_ip.__enter__()
                self.seq = NamedSequenceOp(
                    "__transform_main",
                    [AnyOpType.get()],
                    [],
                    arg_attrs=[{
                        "transform.readonly": UnitAttr.get()
                    }],
                )
                self.seq_ip = ir.InsertionPoint(self.seq.body)
                self.seq_ip.__enter__()
                return self.seq.bodyTarget

            def __exit__(self, *args):
                YieldOp()
                self.seq_ip.__exit__(*args)

                # Cleanup the transform module
                # pm = passmanager.PassManager(context=self.mod.context)
                # pm.add("cse")
                # pm.add("canonicalize")
                # pm.run(self.mod.operation)

                if self.dump:
                    print(f"{self.mod}\n// -----\n")

                apply_named_sequence(
                    payload_root=self.target,
                    transform_root=self.seq,
                    transform_module=self.mod,
                )

                self.mod_ip.__exit__(*args)
                self.loc_ip.__exit__(*args)
                self.ctx_ip.__exit__(*args)
                return False

        return TransormContextManager(target, dump)

    @staticmethod
    def default_context() -> ir.Context:
        cls = Pipeline
        if (ctx := getattr(cls, "__default_context__", None)) is None:
            cls.__default_context__ = ctx = ir.Context()
        return ctx

    @staticmethod
    def get_shared_libs() -> List[str]:
        cls = Pipeline
        if (libs := getattr(cls, "__mlir_shared_libs__", None)) is None:
            mlir_path = ir.__file__.split("tools/mlir/python_packages")[0]
            lib_dir = os.path.join(mlir_path, "lib")
            names = ["libmlir_c_runner_utils.so", "libmlir_runner_utils.so", "libmlir_levelzero_runtime.so"]
            libs = [os.path.join(lib_dir, f) for f in names]
            cls.__mlir_shared_libs__ = libs
        return libs


class CompiledModule:

    def __init__(self, module: ir.Module, func_name: str, num_args: int, opt_level=2):
        self.eng = ExecutionEngine(module, opt_level=opt_level, shared_libs=Pipeline.get_shared_libs())
        self.eng.initialize()
        self.func = self.eng.lookup(func_name)
        self.num_args = num_args

    def __call__(self, *args: Tensor):
        assert len(args) == self.num_args, f"Expected {self.num_args} arguments, got {len(args)}!"
        return self.func(torch_to_packed_args(args))


class OpNameMatcher:
    op_name_pattern = re.compile(r'[^a-zA-Z0-9_.]')

    def __init__(self, *op_name_or_pattern: Union[str, re.Pattern]):
        self.op_names: Set[str] = set()
        self.patterns: Dict[str, re.Pattern] = {}
        self.add(*op_name_or_pattern)

    def add(self, *op_name_or_pattern: Union[str, re.Pattern, Iterable[Union[str, re.Pattern]]]) -> OpNameMatcher:
        op_name_or_pattern = [n for p in op_name_or_pattern for n in ((p, ) if isinstance(p, str | re.Pattern) else p)]
        for name in [n for p in op_name_or_pattern for n in (p.split(",") if isinstance(p, str) else (p, ))]:
            if isinstance(name, re.Pattern):
                self.patterns[name.pattern] = name
            elif (idx := name.find("#")) != -1:
                n = name[:idx]
                for num in name[idx + 1:].split("#"):
                    self.op_names.add(f"{n}#{num}")
            elif self.op_name_pattern.search(name):
                self.patterns[name] = re.compile(name)
            else:
                self.op_names.add(name)
        return self

    def copy(self) -> OpNameMatcher:
        matcher = OpNameMatcher()
        matcher.op_names = self.op_names.copy()
        matcher.patterns = self.patterns.copy()
        return matcher

    def update(self, other: OpNameMatcher):
        self.op_names.update(other.op_names)
        self.patterns.update(other.patterns)

    def matches(self, op_name: str, num: int = None) -> bool:
        if op_name in self.op_names:
            return True
        for pattern in self.patterns.values():
            if pattern.fullmatch(op_name):
                return True
        if num is not None:
            return self.matches(f"{op_name}#{num}")
        return False


class Suppress:

    def __init__(self, scope):
        self.scope = scope

    def __enter__(self):
        self.ip1 = InsertionPoint.current
        self.ip1.__enter__()
        op = SequenceOp(failure_propagation_mode=FailurePropagationMode.Suppress,
                        results=[AnyOpType.get()],
                        target=self.scope)
        self.ip2 = InsertionPoint(op.body)
        self.ip2.__enter__()
        return op.result

    def __exit__(self, *args):
        self.ip2.__exit__(*args)
        self.ip1.__exit__(*args)


class TilingAndFusion(Phase):
    TileSizeFnType = Callable[[ir.Operation], Tuple[int, ...]]
    TileSizeOrFnType = Union[Tuple[int, ...], TileSizeFnType]
    OpNameOrPatternType = Union[str, re.Pattern]
    OpNamesOrPatternsType = Union[OpNameOrPatternType, Iterable[OpNameOrPatternType]]
    OptionalOpNamesOrPatternsType = Optional[OpNamesOrPatternsType]
    LoopType = Literal["for", "forall"]
    TilingOpType = Callable[[Union[ir.Operation, ir.Value, ir.OpView], Tuple[int, ...]], ir.OpResultList]
    LoopOrTilingOpType = Union[LoopType, TilingOpType]

    def __init__(
        self,
        tile_sizes: Optional[TileSizeOrFnType] = None,
        tile_and_fuse: OptionalOpNamesOrPatternsType = None,
        loop: Optional[LoopOrTilingOpType] = "forall",
        reduction: bool = False,
        no_tile: OptionalOpNamesOrPatternsType = None,
        no_fuse: OptionalOpNamesOrPatternsType = None,
        no_tile_no_fuse: OptionalOpNamesOrPatternsType = None,
    ):
        self.supported_ops = OpNameMatcher()
        self.params: List[TilingAndFusion.Params] = []
        if tile_sizes is not None:
            assert tile_and_fuse
            self.tile_and_fuse(
                tile_sizes,
                tile_and_fuse,
                loop=loop,
                reduction=reduction,
                no_tile=no_tile,
                no_fuse=no_fuse,
                no_tile_no_fuse=no_tile_no_fuse,
            )

    def tile(
        self,
        tile_sizes: TileSizeOrFnType,
        op_names_or_patterns: OpNamesOrPatternsType,
        loop: LoopOrTilingOpType = "forall",
        reduction: bool = False,
        no_tile: OptionalOpNamesOrPatternsType = None,
    ) -> TilingAndFusion:
        self.params.append(
            TilingAndFusion.Params(
                self,
                tile_sizes,
                tile=op_names_or_patterns,
                no_tile=no_tile,
                loop=loop,
                reduction=reduction,
            ))
        return self

    def tile_and_fuse(
        self,
        tile_sizes: TileSizeOrFnType,
        op_names_or_patterns: OpNamesOrPatternsType,
        loop: LoopOrTilingOpType = "forall",
        reduction: bool = False,
        no_tile: OptionalOpNamesOrPatternsType = None,
        no_fuse: OptionalOpNamesOrPatternsType = None,
        no_tile_no_fuse: OptionalOpNamesOrPatternsType = None,
    ) -> TilingAndFusion:
        self.params.append(
            TilingAndFusion.Params(
                self,
                tile_sizes,
                tile_and_fuse=op_names_or_patterns,
                no_tile=no_tile,
                no_fuse=no_fuse,
                no_tile_no_fuse=no_tile_no_fuse,
                loop=loop,
                reduction=reduction,
            ))
        return self

    def fuse(self,
             op_names_or_patterns: OpNamesOrPatternsType,
             no_fuse: OptionalOpNamesOrPatternsType = None) -> TilingAndFusion:
        m = OpNameMatcher(op_names_or_patterns)
        self.params[-1].fuse.update(m)
        self.supported_ops.update(m)
        if no_fuse is not None:
            m = OpNameMatcher(no_fuse)
            self.params[-1].no_fuse.update(m)
            self.supported_ops.update(m)
        return self

    def then(
        self,
        tile_sizes: Optional[TileSizeOrFnType] = None,
        tile_and_fuse: OptionalOpNamesOrPatternsType = None,
        loop: Optional[TilingAndFusion.LoopType] = None,
        reduction: bool = False,
        tile: OptionalOpNamesOrPatternsType = None,
        fuse: OptionalOpNamesOrPatternsType = None,
        no_tile: OptionalOpNamesOrPatternsType = None,
        no_fuse: OptionalOpNamesOrPatternsType = None,
        no_tile_no_fuse: OptionalOpNamesOrPatternsType = None,
    ) -> TilingAndFusion:
        """Copy params from the current tile into a new one and update with the specified options."""
        copy = self.params[-1].copy(
            phase=self,
            tile_sizes=tile_sizes,
            tile_and_fuse=tile_and_fuse,
            no_tile_no_fuse=no_tile_no_fuse,
            tile=tile,
            fuse=fuse,
            no_tile=no_tile,
            no_fuse=no_fuse,
            loop=loop,
            reduction=reduction,
        )
        self.params.append(copy)
        return self

    @override
    def apply(self, ctx: Phase.Context):
        op_to_info: Dict[ir.Operation, TilingAndFusion.OpInfo] = {}
        names: Dict[str, int] = {}

        def visitor(op: ir.Operation):
            name = op.name
            # TODO: check if op implements TilingInterface
            if not name.endswith(".yield"):
                num = names.get(name, -1) + 1
                if num or self.supported_ops.matches(name, num):
                    names[name] = num
                    op_to_info[op] = TilingAndFusion.OpInfo(op, num)
            return ir.WalkResult.ADVANCE

        # Collect all supported ops
        ctx.payload.body.operations[0].walk(visitor)
        mod = ctx.mod
        matches = match(mod, names)
        num_handles = len(op_to_info)
        handles = split_handle(matches, num_handles) if num_handles > 1 else (matches.result, )
        # Assign handles to each op
        for i, h in zip(op_to_info.values(), handles):
            i.handle = h
        # Processing in reverse order in order to tile the last ops first and fuse the producers greedily
        op_to_info = dict(reversed(op_to_info.items()))

        for params in self.params:
            processed = set()
            for oi in op_to_info.values():
                if oi.op not in processed and params.tile_matches(oi):
                    fused = self.apply_to(ctx, params, oi, op_to_info)
                    processed.update(fused)

        cse(mod)
        canonicalize(mod)
        # with apply_patterns(mod):
        #     ApplyFoldUnitExtentDimsViaReshapesPatternsOp()
        #     ApplyCanonicalizationPatternsOp()

        ctx.gpu_loop_tile_sizes = dict(sorted(ctx.gpu_loop_tile_sizes.items(), key=lambda i: i[0].location.start_line))

    def apply_to(
        self,
        ctx: Phase.Context,
        params: TilingAndFusion.Params,
        oi: TilingAndFusion.OpInfo,
        op_to_info: Dict[ir.Operation, TilingAndFusion.OpInfo],
    ) -> Iterable[ir.Operation]:
        op = oi.op
        producers = self.collect_producers(op, op_to_info, params.fuse_matches)
        producers = self.filter_out_producers(op, producers)
        producers = self.sort_producers(producers)

        # Tile and fuse all the producers
        sizes = params.tile_sizes(op)
        oi.handle, loop = params.tiling_op(oi.handle, sizes, op.location)
        if params is self.params[0]:
            AnnotateOp(loop, "gpu_loop")
            ctx.gpu_loop_tile_sizes[op] = sizes

        for oi in producers.values():
            oi.handle, loop = FuseIntoContainingOp(oi.handle, loop).results

        return producers.keys()

    # Collect recursively all producers of the op operands, that are in op_to_info and match the given matcher
    def collect_producers(
        self,
        op: ir.Operation,
        op_to_info: Dict[ir.Operation, TilingAndFusion.OpInfo],
        match: Callable[[TilingAndFusion.OpInfo], bool],
        producers: Optional[Dict[ir.Operation, TilingAndFusion.OpInfo]] = None,
    ):
        if producers is None:
            producers = {}
        for operand in op.operands:
            if (oi := op_to_info.get(operand.owner, None)) is not None and match(oi):
                producers[operand.owner] = oi
                self.collect_producers(operand.owner, op_to_info, match, producers)
        return producers

    # Filter out all the producers, that have consumers not in the producers list
    @staticmethod
    def filter_out_producers(
        op: ir.Operation,
        producers: Dict[ir.Operation, TilingAndFusion.OpInfo],
    ) -> Dict[ir.Operation, TilingAndFusion.OpInfo]:
        return {
            p: oi
            for p, oi in producers.items()
            if all(u.owner.operation in producers or u.owner.operation == op for r in p.results for u in r.uses)
        }

    # Sort producers in the order, that allows to fuse them
    @staticmethod
    def sort_producers(
            producers: Dict[ir.Operation, TilingAndFusion.OpInfo]) -> Dict[ir.Operation, TilingAndFusion.OpInfo]:
        # FIXME: implement a topological sort instead of sorting by line number
        return dict(sorted(producers.items(), key=lambda i: i[0].location.start_line, reverse=True))

    class Params:
        _tiling_ops_map: Dict[Tuple[str, bool], TilingAndFusion.TilingOpType] = {
            ("forall", False):
            lambda t, s, loc=None: TileUsingForallOp(t, tile_sizes=s, loc=loc).results,
            ("forall", True):
            lambda t, s, loc=None: TileReductionUsingForallOp(
                [AnyOpType.get()],  # fill_op
                AnyOpType.get(),  # split_op
                AnyOpType.get(),  # combining_op
                AnyOpType.get(),  # forall_op
                target=t,
                tile_sizes=s,
                loc=loc,
            ).results[2:],
            ("for", False):
            lambda t, s, loc=None: TilingAndFusion.Params._split_for_loops(
                TileUsingForOp(t, sizes=s, loc=loc).results, s),
            ("for", True):
            lambda t, s, loc=None: TileReductionUsingForOp(
                [AnyOpType.get()],  # fill_op
                AnyOpType.get(),  # split_op
                AnyOpType.get(),  # combining_op
                AnyOpType.get(),  # for_op
                target=t,
                tile_sizes=s,
                loc=loc,
            ).results[2:],
        }

        def __init__(
            self,
            phase: TilingAndFusion,
            tile_sizes: TilingAndFusion.TileSizeOrFnType,
            tile_and_fuse: TilingAndFusion.OptionalOpNamesOrPatternsType = None,
            no_tile_no_fuse: TilingAndFusion.OptionalOpNamesOrPatternsType = None,
            tile: TilingAndFusion.OptionalOpNamesOrPatternsType = None,
            fuse: TilingAndFusion.OptionalOpNamesOrPatternsType = None,
            no_tile: TilingAndFusion.OptionalOpNamesOrPatternsType = None,
            no_fuse: TilingAndFusion.OptionalOpNamesOrPatternsType = None,
            loop: TilingAndFusion.LoopOrTilingOpType = "forall",
            reduction: bool = False,
        ) -> TilingAndFusion:
            self.tile_sizes = tile_sizes if isinstance(tile_sizes, Callable) else lambda _: tile_sizes
            self.tiling_op = loop if isinstance(loop, Callable) else self._tiling_ops_map[(loop, reduction)]
            self.tile = OpNameMatcher()
            self.fuse = OpNameMatcher()
            self.no_tile = OpNameMatcher()
            self.no_fuse = OpNameMatcher()
            self.reduction: bool = reduction

            for map in ((tile_and_fuse, self.tile, self.fuse), (no_tile_no_fuse, self.no_tile, self.no_fuse),
                        (tile, self.tile), (fuse, self.fuse), (no_tile, self.no_tile), (no_fuse, self.no_fuse)):
                if map[0] is None:
                    continue
                m = OpNameMatcher(map[0])
                for o in map[1:]:
                    o.update(m)
                phase.supported_ops.update(m)

        def tile_matches(self, oi: TilingAndFusion.OpInfo) -> bool:
            name = oi.op.name
            return self.tile.matches(name, oi.num) and not self.no_tile.matches(name, oi.num)

        def fuse_matches(self, oi: TilingAndFusion.OpInfo) -> bool:
            name = oi.op.name
            return self.fuse.matches(name, oi.num) and not self.no_fuse.matches(name, oi.num)

        def copy(
            self,
            phase: TilingAndFusion,
            tile_sizes: Optional[TilingAndFusion.TileSizeOrFnType] = None,
            tile_and_fuse: TilingAndFusion.OptionalOpNamesOrPatternsType = None,
            no_tile_no_fuse: TilingAndFusion.OptionalOpNamesOrPatternsType = None,
            tile: TilingAndFusion.OptionalOpNamesOrPatternsType = None,
            fuse: TilingAndFusion.OptionalOpNamesOrPatternsType = None,
            no_tile: TilingAndFusion.OptionalOpNamesOrPatternsType = None,
            no_fuse: TilingAndFusion.OptionalOpNamesOrPatternsType = None,
            loop: Optional[TilingAndFusion.LoopOrTilingOpType] = None,
            reduction: bool = False,
        ) -> TilingAndFusion.Params:
            copy = TilingAndFusion.Params(
                phase,
                tile_sizes or self.tile_sizes,
                tile_and_fuse,
                no_tile_no_fuse,
                tile,
                fuse,
                no_tile,
                no_fuse,
                loop or self.tiling_op,
                reduction,
            )
            copy.tile.update(self.tile)
            copy.fuse.update(self.fuse)
            copy.no_tile.update(self.no_tile)
            copy.no_fuse.update(self.no_fuse)
            return copy

        @staticmethod
        def _split_for_loops(tile_results, sizes):
            return tile_results
            loop = SplitHandleOp([AnyOpType.get(), AnyOpType.get()], tile_results[1], overflow_result=0).results[0]
            return tile_results[0], loop
            if len(sizes) == 1:
                return tile_results
            # loops = split_handle(tile_results[1], len(sizes))
            # loops = [l for l, s in zip(loops, sizes) if s > 0]
            # return tile_results[0], loops[0] if len(loops) == 1 else loops
            # PrintOp(target=GetParentOp(AnyOpType.get(), tile_results[0]).result, name="Parent of")
            return tile_results[0], GetParentOp(AnyOpType.get(), tile_results[0]).result

    class OpInfo:

        def __init__(self, op: ir.Operation, num: int):
            self.op = op
            self.num = num
            self.handle = None


class Vectorization(Phase):

    @override
    def apply(self, ctx: Phase.Context):
        func = VectorizeChildrenAndApplyPatternsOp(
            ctx.func,
            fold_type_extensions_into_contract=True,
        ).result

        # hoist loop invariant vector read/store ops
        HoistLoopInvariantSubsetsOp(match(func, ("scf.for", "scf.forall")))

        ctx.func = func
        cse(func)
        canonicalize(func)
        
class RewriteBroadcastTransposes(Phase):
    
    @override
    def apply(self, ctx: Phase.Context):
        from mlir.dialects import vector
        
        # match transpose ops.
        transpose_ops = match(ctx.func, "vector.transpose")
        PrintOp(target=transpose_ops, name="Matched transpose ops")
        
        # For each transpose, get the producer of operand 0 (the source)
        # broadcast_ops = []
        # for h in split_handle(transpose_ops, len(transpose_ops.results)):
        #     producer = get_producer_of_operand(AnyOpType.get(), h, 0)
        #     # Check if the producer is a broadcast op
        #     # broadcast = match(producer, "vector.broadcast")
        #     # broadcast_ops.append(broadcast)
        # PrintOp(target=transpose_ops, name="Matched transpose ops")
        # PrintOp(target=broadcast_ops, name="Collected broadcast ops")
        # # Print the collected broadcast ops
        # for i, bcast_op in enumerate(broadcast_ops):
        #     PrintOp(target=bcast_op, name=f"Broadcast op {i}")
        # use printop to print them
        



class Bufferization(Phase):

    @override
    def apply(self, ctx: Phase.Context):
        mod = OneShotBufferizeOp(
            ctx.mod,
            allow_return_allocs_from_loops=True,
            bufferize_function_boundaries=True,
            function_boundary_type_conversion=LayoutMapOption.IdentityLayoutMap,
        ).result
        mod = OneShotBufferizeOp(
            mod,
            allow_return_allocs_from_loops=False,
            bufferize_function_boundaries=True,
            function_boundary_type_conversion=LayoutMapOption.IdentityLayoutMap,
        ).result

        ctx.mod = apply_pass(
            mod,
            "fold-memref-alias-ops",  # fold memref.subviews into vector.transfer_read/write ops
            "drop-equivalent-buffer-results",
            ("buffer-results-to-out-params", {
                "add-result-attr": True,
                "hoist-static-allocs": True,
                "hoist-dynamic-allocs": True,
                "modify-public-functions": True,
            }),
            "cse",
            "canonicalize",
        )


class GpuKernelOutlining(Phase):

    def __init__(
        self,
        threads: Optional[Union[int, Tuple[int, int, int], Callable[[ir.Operation, Tuple[int, ...]],
                                                                    Tuple[int, int, int]]]] = None,
        sg_size: Optional[Union[int, Callable[[ir.Operation, Tuple[int, ...]], Optional[int]]]] = None,
        target={
            "O": "2",
            "chip": "pvc"
        },
    ):
        self.target = target
        if threads is None:
            self.threads = self.calculate_threads
        elif isinstance(threads, int):
            self.threads = lambda _, __: (threads, 1, 1)
        elif not isinstance(threads, Callable):
            self.threads = lambda _, __: threads
        if sg_size is None:
            self.sg_size = lambda _, __: None
        elif isinstance(sg_size, int):
            self.sg_size = lambda _, __: sg_size
        else:
            self.sg_size = sg_size
            

    @override
    def apply(self, ctx: Phase.Context):
        # convert forall to parallel
        tile_sizes = ctx.gpu_loop_tile_sizes
        num_kernels = len(tile_sizes)
        loops = MatchOp(AnyOpType.get(), ctx.func, op_attrs=ir.DictAttr.get({"gpu_loop": ir.UnitAttr.get()}))
        for h in split_handle(loops, num_kernels):
            ForallToParallelOp([AnyOpType.get()], h)

        # convert scf.parallel to gpu.launch
        func = apply_pass(ctx.func, "gpu-map-parallel-loops", "convert-parallel-loops-to-gpu", "lower-affine", "cse",
                          "canonicalize")

        # set correct number of gpu threads
        handles = split_handle(match(func, ops=("gpu.launch", )), num_kernels)
        for h, (o, t) in zip(handles, tile_sizes.items()):
            SetGPULaunchThreadsOp(h, threads=self.threads(o, t))

        # outline gpu func
        apply_pass(func, "lower-affine", "canonicalize", "gpu-launch-sink-index-computations")
        ctx.mod = apply_pass(ctx.mod, "gpu-kernel-outlining", ("xevm-attach-target", self.target), "cse")

        # set sg size attribute
        if any(self.sg_size(o, t) is not None for o, t in tile_sizes.items()):
            i32_type = ir.IntegerType.get_signless(32)
            param_type = ir.Type.parse(f"!transform.param<i32>")
            kernels = split_handle(match(ctx.mod, ops=("gpu.func", )), num_kernels)
            for h, (o, t) in zip(kernels, tile_sizes.items()):
                if (sg_size := self.sg_size(o, t)):
                    p = ParamConstantOp(param_type, ir.IntegerAttr.get(i32_type, sg_size)).result
                    AnnotateOp(h, "intel_reqd_sub_group_size", param=p)

    def calculate_threads(self, op: ir.Operation, tiles: Tuple[int, ...]) -> Tuple[int, int, int]:
        # TODO: implement
        sg_size = self.sg_size(op, tiles) or 32
        return sg_size, 1, 1


class VectorToXegpu(Phase):

    @override
    def apply(self, ctx: Phase.Context):
        # convert vector to xegpu
        ctx.gpu_func = apply_pass(ctx.gpu_func, "convert-vector-to-xegpu", "expand-strided-metadata",
                                  "cse", "canonicalize")


class XegpuLayout(Phase):

    @staticmethod
    def set_op_layout(
        op: Union[ir.Operation, ir.Value],
        idx: int,
        sg_layout: Tuple[int, int],
        sg_data: Tuple[int, int],
        *,
        inst_data: Optional[Tuple[int, int]] = None,
        order: Optional[Tuple[int, int]] = None,
        result_idx: Optional[int] = None,
    ):
        # print(order)
        # operand = GetOperandOp(AnyValueType.get(), op, (idx, ))
        # SetDescLayoutOp(GetDescOp(operand), sg_layout, sg_data, inst_data=inst_data)
        SetOpLayoutAttrOp(
            op,
            sg_layout,
            sg_data,
            inst_data=inst_data,
            order=order,
            index=idx if result_idx is None else result_idx,
            result=result_idx is not None,
        )

    # @staticmethod
    # def set_result_layout(
    #     op: Union[ir.Operation, ir.Value],
    #     sg_layout: Tuple[int, int],
    #     sg_data: Tuple[int, int],
    #     *,
    #     inst_data: Optional[Tuple[int, int]] = None,
    #     idx: int = 0,
    # ):
    #     # result = GetResultOp(AnyValueType.get(), op, (idx, ))
    #     # SetDescLayoutOp(GetDescOp(result), sg_layout, sg_data, inst_data=inst_data)
    #     SetOpLayoutAttrOp(op, sg_layout, sg_data, inst_data=inst_data, index=idx, result=True)


class DpasLayout(XegpuLayout):

    @override
    def apply(self, ctx: Phase.Context):
        dpas_tile = [8, 16, 16]

        # tunable parameters
        wg_tile = [256, 256]
        sg_tile = [32, 64]
        k_tile = 32
        load_tile_a = [8, 16]
        load_tile_b = [16, 16]

        # derived parameters
        sg_layout = [wg_tile[0] // sg_tile[0], wg_tile[1] // sg_tile[1]]

        # matmul matrix shapes
        sg_tile_a = [sg_tile[0], k_tile]
        sg_tile_b = [k_tile, sg_tile[1]]
        dpas_shape_a = [dpas_tile[0], dpas_tile[2]]
        dpas_shape_b = [dpas_tile[2], dpas_tile[1]]
        dpas_shape_c = [dpas_tile[0], dpas_tile[1]]

        # add layouts to DPAS op operands
        gpu_func = ctx.gpu_func
        k_loop = match(gpu_func, ops={"scf.for"})
        dpas_op = match(k_loop, ops={"xegpu.dpas"})

        # matmul matrix shapes
        sg_tile_a = [sg_tile[0], k_tile]
        sg_tile_b = [k_tile, sg_tile[1]]
        dpas_shape_a = [dpas_tile[0], dpas_tile[2]]
        dpas_shape_b = [dpas_tile[2], dpas_tile[1]]
        dpas_shape_c = [dpas_tile[0], dpas_tile[1]]

        # A tile load layout
        layout_load_a = {
            "sg_layout": sg_layout,
            "sg_data": sg_tile_a,
            "inst_data": load_tile_a,
        }
        self.set_op_layout(dpas_op, 0, sg_layout, sg_tile_a, inst_data=load_tile_a)

        # # B tile load layout
        layout_load_b = {
            "sg_layout": sg_layout,
            "sg_data": sg_tile_b,
            "inst_data": load_tile_b,
        }
        self.set_op_layout(dpas_op, 1, sg_layout, sg_tile_b, inst_data=load_tile_b)

        # C tile layout
        output_layout = {
            "sg_layout": sg_layout,
            "sg_data": sg_tile,
            "inst_data": dpas_shape_c,
        }
        self.set_op_layout(dpas_op, 2, sg_layout, sg_tile, inst_data=dpas_shape_c, result_idx=0)
        # self.set_result_layout(dpas_op, sg_layout, sg_tile, inst_data=dpas_shape_c)

        # for post ops we need to add C layout manually
        max_op = match(gpu_func, ops={"arith.maximumf"}).result
        SetOpLayoutAttrOp(target=max_op, result=True, index=0, **output_layout)
        # find zero constant buffer and annotate it
        const_buffer = get_producer_of_operand(AnyOpType.get(), max_op, 1)
        SetOpLayoutAttrOp(target=const_buffer, result=True, index=0, **output_layout)

        cse(gpu_func)
        canonicalize(gpu_func)

        # hoist desc ops out of reduction loop
        apply_licm(k_loop)

        canonicalize(gpu_func)
        cse(gpu_func)


class XeGpu(Phase):

    @override
    def apply(self, ctx: Phase.Context):
        propagate_sg_layout = ("xegpu-propagate-layout", {"layout-kind": "subgroup"})
        propagate_inst_data = ("xegpu-propagate-layout", {"layout-kind": "inst"})
        propagate_lane_layout = ("xegpu-propagate-layout", {"layout-kind": "lane"})
        # print before
        PrintOp(target=ctx.gpu_func, name="Before propagate subgroup layout")
        target = apply_pass(ctx.gpu_func, "xevm-attach-target", propagate_sg_layout)
        funcOp = match(target, "gpu.func")
        PrintOp(target=funcOp, name="After propagate subgroup layout")
        target = apply_pass(target, "xegpu-wg-to-sg-distribute")
        
        # xegpu distribution
        # propagate = ("xegpu-propagate-layout", {"layout-kind": "inst"})
        # apply_pass(ctx.gpu_func, "xegpu-wg-to-sg-distribute")
                   #propagate, "xegpu-blocking", "canonicalize", "cse")
        # propagate_lane = ("xegpu-propagate-layout", {"layout-kind": "lane"})
        # ctx.gpu_mod = apply_pass(ctx.gpu_mod, propagate_lane, "xegpu-subgroup-distribute", "canonicalize", "cse",
        #                          "loop-invariant-code-motion", "cse", "xegpu-vector-linearize", "convert-xegpu-to-xevm",
        #                          ("convert-gpu-to-llvm-spv", {
        #                              "use-64bit-index": "true"
        #                          }), "convert-xevm-to-llvm", "cse")

        # apply_pass(ctx.gpu_func, "gpu-async-region")

        # ctx.mod = apply_pass(ctx.mod, "reconcile-unrealized-casts", "convert-vector-to-scf", "convert-scf-to-cf",
        #                      "expand-strided-metadata", "finalize-memref-to-llvm", "convert-cf-to-llvm",
        #                      "convert-vector-to-llvm", "convert-arith-to-llvm", "convert-index-to-llvm",
        #                      "convert-func-to-llvm", "convert-math-to-llvm", "gpu-to-llvm", "lower-affine",
        #                      "reconcile-unrealized-casts", "cse", "gpu-module-to-binary")