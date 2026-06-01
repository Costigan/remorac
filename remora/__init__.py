"""Remora Dense Core compiler prototype."""

from remora.abi import (
    MAX_RANK,
    RemoraMemRef0,
    RemoraMemRef1,
    RemoraMemRef10,
    RemoraMemRef2,
    RemoraMemRef3,
    RemoraMemRef4,
    RemoraMemRef5,
    RemoraMemRef6,
    RemoraMemRef7,
    RemoraMemRef8,
    RemoraMemRef9,
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
    "MAX_RANK",
    "RemoraExecutor",
    "RemoraMemRef0",
    "RemoraMemRef1",
    "RemoraMemRef10",
    "RemoraMemRef2",
    "RemoraMemRef3",
    "RemoraMemRef4",
    "RemoraMemRef5",
    "RemoraMemRef6",
    "RemoraMemRef7",
    "RemoraMemRef8",
    "RemoraMemRef9",
    "element_strides",
    "make_memref_descriptor",
    "make_numpy_memref_descriptor",
    "memref_descriptor_type",
    "numpy_from_memref_descriptor",
]
