import ctypes
import torch

from mlir.runtime.np_to_memref import (F16, make_nd_memref_descriptor)

from mlir import ir
from mlir.dialects import transform
from mlir.dialects.transform import structured
import ctypes


class F32(ctypes.Structure):
    """A ctype representation for MLIR's Float16."""

    _fields_ = [("f32", ctypes.c_int32)]


def get_ranked_memref_descriptor(torch_tensor):
    """Returns a ranked memref descriptor for the given numpy array."""
    if torch_tensor.dtype not in (torch.float16, torch.float32):
        raise NotImplementedError("Only float16 and float32 dtypes are supported in this example.")
    ctp = F16 if torch_tensor.dtype == torch.float16 else F32
    rank = torch_tensor.dim()
    base_addr = torch_tensor.untyped_storage().data_ptr()
    x = make_nd_memref_descriptor(2, ctp)()
    x.allocated = base_addr
    x.aligned = ctypes.cast(base_addr, ctypes.POINTER(ctp))
    x.offset = ctypes.c_longlong(0)

    # Numpy uses byte quantities to express strides, MLIR OTOH uses the
    # torch abstraction which specifies strides in terms of elements.
    shape_arr_t = ctypes.c_longlong * rank
    stride_arr_t = ctypes.c_longlong * rank
    x.shape = shape_arr_t(*[int(s) for s in torch_tensor.size()])
    x.strides = stride_arr_t(*[int(s) for s in torch_tensor.stride()])
    return x


def get_packed_arg(ctypes_args) -> list[ctypes.c_void_p]:
    """
    Return a list of packed ctype arguments compatible with
    jitted MLIR function's interface.

    Args:
        ctypes_args: A list of ctype pointer arguments.
    """
    packed_args = (ctypes.c_void_p * len(ctypes_args))()
    for argNum in range(len(ctypes_args)):
        packed_args[argNum] = ctypes.cast(ctypes_args[argNum], ctypes.c_void_p)
    return packed_args


def memref_to_ctype(memref_desc) -> ctypes._Pointer:
    """
    Convert a memref descriptor into a ctype argument.

    Args:
        memref_desc: An MLIR memref descriptor.
    """
    return ctypes.pointer(ctypes.pointer(memref_desc))


def memrefs_to_packed_args(memref_descs) -> list[ctypes.c_void_p]:
    """
    Convert a list of memref descriptors into packed ctype arguments.

    Args:
        memref_descs: A list of memref descriptors.
    """
    ctype_args = [memref_to_ctype(memref) for memref in memref_descs]
    return get_packed_arg(ctype_args)


def torch_to_memref(input: torch.Tensor) -> ctypes.Structure:
    """
    Convert a PyTorch tensor into a memref descriptor.

    Args:
        input: PyTorch tensor.
    """
    return get_ranked_memref_descriptor(input)


def torch_to_packed_args(inputs: list[torch.Tensor]) -> list[ctypes.c_void_p]:
    """
    Convert a list of PyTorch tensors into packed ctype arguments.

    Args:
        inputs: A list of PyTorch tensors.
    """
    memrefs = [torch_to_memref(input) for input in inputs]
    return memrefs_to_packed_args(memrefs)