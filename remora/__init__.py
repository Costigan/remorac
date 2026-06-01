"""Remora Dense Core compiler prototype."""

from remora.abi import (
    RemoraMemRef0,
    RemoraMemRef1,
    RemoraMemRef2,
    RemoraMemRef3,
    element_strides,
    make_memref_descriptor,
    make_numpy_memref_descriptor,
    memref_descriptor_type,
    numpy_from_memref_descriptor,
)
from remora.executor import RemoraExecutor
from remora.runtime import CUDAKernel, CUDAModule, CUDARuntime, CPUExecutor, CPUFunctionExecutor

__all__ = [
    "CUDAKernel",
    "CUDAModule",
    "CUDARuntime",
    "CPUExecutor",
    "CPUFunctionExecutor",
    "RemoraExecutor",
    "RemoraMemRef0",
    "RemoraMemRef1",
    "RemoraMemRef2",
    "RemoraMemRef3",
    "element_strides",
    "make_memref_descriptor",
    "make_numpy_memref_descriptor",
    "memref_descriptor_type",
    "numpy_from_memref_descriptor",
]
