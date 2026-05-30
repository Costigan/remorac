"""ctypes definitions for the Remora Dense Core external ABI."""

from __future__ import annotations

import ctypes
from collections.abc import Sequence
from typing import TypeAlias

import numpy as np

from remora.errors import RemoraError


PointerValue: TypeAlias = int | ctypes.c_void_p


class RemoraMemRef0(ctypes.Structure):
    _fields_ = [
        ("allocated", ctypes.c_void_p),
        ("aligned", ctypes.c_void_p),
        ("offset", ctypes.c_int64),
    ]


class RemoraMemRef1(ctypes.Structure):
    _fields_ = [
        ("allocated", ctypes.c_void_p),
        ("aligned", ctypes.c_void_p),
        ("offset", ctypes.c_int64),
        ("size0", ctypes.c_int64),
        ("stride0", ctypes.c_int64),
    ]


class RemoraMemRef2(ctypes.Structure):
    _fields_ = [
        ("allocated", ctypes.c_void_p),
        ("aligned", ctypes.c_void_p),
        ("offset", ctypes.c_int64),
        ("size0", ctypes.c_int64),
        ("size1", ctypes.c_int64),
        ("stride0", ctypes.c_int64),
        ("stride1", ctypes.c_int64),
    ]


class RemoraMemRef3(ctypes.Structure):
    _fields_ = [
        ("allocated", ctypes.c_void_p),
        ("aligned", ctypes.c_void_p),
        ("offset", ctypes.c_int64),
        ("size0", ctypes.c_int64),
        ("size1", ctypes.c_int64),
        ("size2", ctypes.c_int64),
        ("stride0", ctypes.c_int64),
        ("stride1", ctypes.c_int64),
        ("stride2", ctypes.c_int64),
    ]


_DESCRIPTORS = {
    0: RemoraMemRef0,
    1: RemoraMemRef1,
    2: RemoraMemRef2,
    3: RemoraMemRef3,
}


def memref_descriptor_type(rank: int) -> type[ctypes.Structure]:
    """Return the rank-specialized descriptor type for ranks 0 through 3."""
    try:
        return _DESCRIPTORS[rank]
    except KeyError as exc:
        raise RemoraError("Remora Dense Core ABI supports only ranks 0 through 3") from exc


def element_strides(array: np.ndarray) -> tuple[int, ...]:
    """Convert numpy byte strides to ABI element strides."""
    np_array = np.asarray(array)
    itemsize = np_array.dtype.itemsize
    if itemsize <= 0:
        raise RemoraError(f"dtype {np_array.dtype} has invalid item size {itemsize}")

    strides: list[int] = []
    for byte_stride in np_array.strides:
        if byte_stride % itemsize != 0:
            raise RemoraError(
                f"byte stride {byte_stride} is not divisible by item size {itemsize}"
            )
        strides.append(byte_stride // itemsize)
    return tuple(strides)


def make_memref_descriptor(
    device_or_host_ptr: PointerValue,
    shape: Sequence[int],
    strides: Sequence[int],
    dtype: object,
    offset: int = 0,
) -> ctypes.Structure:
    """Create a rank-specialized descriptor from pointer, shape, and strides.

    ``dtype`` is accepted for call-site clarity and validation; the ABI stores
    element type in kernel metadata, not in the descriptor.
    """
    np.dtype(dtype)
    sizes = tuple(int(dim) for dim in shape)
    element_stride_values = tuple(int(stride) for stride in strides)
    rank = len(sizes)

    if rank != len(element_stride_values):
        raise RemoraError("shape and strides must have the same rank")
    if any(dim < 0 for dim in sizes):
        raise RemoraError("descriptor sizes must be non-negative")

    descriptor_type = memref_descriptor_type(rank)
    ptr = _pointer_value(device_or_host_ptr)
    descriptor = descriptor_type()
    descriptor.allocated = ptr
    descriptor.aligned = ptr
    descriptor.offset = int(offset)

    for axis, size in enumerate(sizes):
        setattr(descriptor, f"size{axis}", size)
    for axis, stride in enumerate(element_stride_values):
        setattr(descriptor, f"stride{axis}", stride)

    return descriptor


def make_numpy_memref_descriptor(array: np.ndarray) -> ctypes.Structure:
    """Create a host descriptor for a numpy array or numpy view."""
    np_array = np.asarray(array)
    rank = np_array.ndim
    memref_descriptor_type(rank)

    base = _base_array(np_array)
    itemsize = np_array.dtype.itemsize
    byte_offset = int(np_array.ctypes.data) - int(base.ctypes.data)
    if byte_offset % itemsize != 0:
        raise RemoraError(
            f"view byte offset {byte_offset} is not divisible by item size {itemsize}"
        )

    descriptor = make_memref_descriptor(
        int(base.ctypes.data),
        np_array.shape,
        element_strides(np_array),
        np_array.dtype,
        offset=byte_offset // itemsize,
    )
    descriptor.aligned = int(base.ctypes.data)
    return descriptor


def _pointer_value(pointer: PointerValue) -> int:
    if isinstance(pointer, ctypes.c_void_p):
        if pointer.value is None:
            raise RemoraError("null pointer is not a valid Remora memref allocation")
        return int(pointer.value)
    return int(pointer)


def _base_array(array: np.ndarray) -> np.ndarray:
    base = array
    while isinstance(base.base, np.ndarray):
        base = base.base
    return base
