import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from pipeline import match, split_handle, apply_patterns, cse, Bufferization, GpuKernelOutlining, Phase, Pipeline, TilingAndFusion, VectorToXegpu, Vectorization, XeGpu, XegpuLayout, apply_pass
from pipeline import RewriteBroadcastTransposes

from typing_extensions import override
import torch
import torch.nn as nn
from torch_mlir.fx import OutputType, export_and_import
from mlir import ir

from mlir.dialects import arith
from mlir.dialects import linalg
from mlir.dialects.transform.xegpu import SetDescLayoutOp, SetOpLayoutAttrOp
from mlir.dialects.transform import ApplyCanonicalizationPatternsOp
from mlir.dialects.transform.structured import ApplyFoldUnitExtentDimsViaReshapesPatternsOp

class Model(nn.Module):
    """
    Simple model that performs a Softmax activation.
    """

    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies Softmax activation to the input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, num_features).

        Returns:
            torch.Tensor: Output tensor with Softmax applied, same shape as input.
        """
        return torch.softmax(x, dim=1)


device = "xpu"
batch_size = 4096
dim = 64
# dim = 393216


def get_inputs():
    # x = torch.rand(batch_size, dim, device=device)
    x = torch.ones(batch_size, dim, device=device)
    return [x]


def get_init_inputs():
    return []  # No special initialization inputs needed


class LowerXeGPU(XegpuLayout):

    @override
    def apply(self, ctx: Phase.Context):
        gpu_func = ctx.gpu_func
        store_ops = match(gpu_func, ops={"xegpu.store_nd"})
        self.set_op_layout(store_ops, 0, [8, 1], [8, 64], order=[1, 0])
        
        # Apply xegpu-propagate-layout pass with layout-kind=subgroup
        ctx.gpu_mod = apply_pass(ctx.gpu_mod, ("xegpu-propagate-layout", {"layout-kind": "subgroup"}))
        ctx.gpu_mod = apply_pass(ctx.gpu_mod, "xegpu-wg-to-sg-distribute", "cse", "lower-affine", "cse")
        # Apply xegpu-propagate-layout again with layout-kind=inst
        ctx.gpu_mod = apply_pass(ctx.gpu_mod, ("xegpu-propagate-layout", {"layout-kind": "inst"}))
        ctx.gpu_mod = apply_pass(ctx.gpu_mod, "xegpu-blocking", "canonicalize", "cse")
        # Apply xegpu-propagate-layout again with layout-kind=lane
        ctx.gpu_mod = apply_pass(ctx.gpu_mod, ("xegpu-propagate-layout", {"layout-kind": "lane"}))
        ctx.gpu_mod = apply_pass(ctx.gpu_mod, "xegpu-subgroup-distribute", "canonicalize", "cse", "loop-invariant-code-motion", 
                             "cse", "xegpu-vector-linearize", "canonicalize", "cse")
        ctx.gpu_mod = apply_pass(ctx.gpu_mod, "convert-xegpu-to-xevm", "convert-math-to-xevm")
        ctx.gpu_mod = apply_pass(ctx.gpu_mod, ("convert-gpu-to-llvm-spv", {"use-64bit-index": True}))
        ctx.gpu_mod = apply_pass(ctx.gpu_mod, "convert-xevm-to-llvm", "cse")
                                  

class LowerToLLVM(Phase):
    
    @override
    def apply(self, ctx: Phase.Context):
        ctx.func = apply_pass(ctx.func, "gpu-async-region")
        ctx.mod = apply_pass(ctx.mod, "reconcile-unrealized-casts", "convert-vector-to-scf", "convert-scf-to-cf",
                            "expand-strided-metadata", "finalize-memref-to-llvm", "convert-cf-to-llvm", "convert-vector-to-llvm",
                            "convert-arith-to-llvm", "convert-index-to-llvm", "convert-func-to-llvm", "convert-math-to-llvm",
                            "gpu-to-llvm", "lower-affine", "reconcile-unrealized-casts", "cse", "gpu-module-to-binary")


class SoftmaxPipeline(Pipeline):
    tile = TilingAndFusion((64, ), "linalg.*")  #.tile((0, 256), "linalg.generic#0#3", loop="for", reduction=True)
    vectorize = Vectorization()
    # remove_transpose = RewriteBroadcastTransposes()
    bufferize = Bufferization()
    outline = GpuKernelOutlining(128, 16)
    vecToXegpu = VectorToXegpu()
    layout = LowerXeGPU()
    lowerToLLVM = LowerToLLVM()

    @classmethod
    def to_mlir_module(cls, target, *args, ir_context=None, **kwargs):
        mlir = super().to_mlir_module(target, *args, ir_context=ir_context, **kwargs)
        # print(mlir)

        # Rewrite the first generic and remove the second output (keep max computation only)
        op = next(op for op in mlir.body.operations[0].regions[0].blocks[0].operations if op.name == "linalg.generic")
        with ir.InsertionPoint(op), op.location:
            f32_type = ir.F32Type.get()
            max = linalg.GenericOp(
                (op.operands[1].type, ),
                inputs=op.operands[:1],
                outputs=op.operands[1:2],
                indexing_maps=ir.ArrayAttr.get(list(op.attributes["indexing_maps"])[:2]),
                iterator_types=op.attributes["iterator_types"],
            )
            body = max.regions[0].blocks.append(f32_type, f32_type)
            with ir.InsertionPoint(body):
                linalg.YieldOp((arith.MaximumFOp(body.arguments[0], body.arguments[1]).result, ))
        op.results[0].replace_all_uses_with(max.results[0])

        with cls.transform_module(mlir) as mod:
            func = match(mod, "func.func")
            with apply_patterns(func):
                # Remove tensor.expand_shape
                ApplyFoldUnitExtentDimsViaReshapesPatternsOp()
                ApplyCanonicalizationPatternsOp()
            cse(func)
        return mlir


if __name__ == "__main__":
    model = Model()
    inputs = get_inputs()
    output = torch.zeros_like(inputs[0])
    # mod = export_and_import(model, *inputs, output_type=OutputType.LINALG_ON_TENSORS)
    #convert mod to linalg generic
    # print(mod)
    if len(sys.argv) > 1 and sys.argv[1] == "dump":
        SoftmaxPipeline.apply(model, *inputs, dump=True)
    else:
    # print(sys.argv)
        SoftmaxPipeline.exec(model, *inputs, output)
        print(f"output: {output}")
        # print 64 th row of output
        print(f"output[64]: {output[64]}")
        expected = model(*inputs)
        print(f"expected: {expected}")
        # check results match
        if not torch.allclose(output, expected, atol=1e-5):
            print("Output does not match expected result!")
        else:
            print("Output matches expected result.")