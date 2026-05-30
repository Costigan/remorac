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
)

__all__ = [
    "RemoraMemRef0",
    "RemoraMemRef1",
    "RemoraMemRef2",
    "RemoraMemRef3",
    "element_strides",
    "make_memref_descriptor",
    "make_numpy_memref_descriptor",
    "memref_descriptor_type",
]
