"""ctypes definitions for the Remora Dense Core external ABI."""

from __future__ import annotations

import ctypes
from collections.abc import Sequence
from typing import TypeAlias

import numpy as np

from remora.errors import RemoraError


PointerValue: TypeAlias = int | ctypes.c_void_p

MAX_RANK = 10


def _memref_fields(rank: int) -> list[tuple[str, object]]:
    return (
        [
            ("allocated", ctypes.c_void_p),
            ("aligned", ctypes.c_void_p),
            ("offset", ctypes.c_int64),
        ]
        + [(f"size{axis}", ctypes.c_int64) for axis in range(rank)]
        + [(f"stride{axis}", ctypes.c_int64) for axis in range(rank)]
    )


def _make_memref_type(rank: int) -> type[ctypes.Structure]:
    return type(
        f"RemoraMemRef{rank}",
        (ctypes.Structure,),
        {
            "__module__": __name__,
            "_fields_": _memref_fields(rank),
        },
    )


_DESCRIPTORS = {rank: _make_memref_type(rank) for rank in range(MAX_RANK + 1)}
globals().update(
    {descriptor_type.__name__: descriptor_type for descriptor_type in _DESCRIPTORS.values()}
)

_DESCRIPTOR_RANKS = {descriptor_type: rank for rank, descriptor_type in _DESCRIPTORS.items()}


def memref_descriptor_type(rank: int) -> type[ctypes.Structure]:
    """Return the rank-specialized descriptor type for supported static ranks."""
    try:
        return _DESCRIPTORS[rank]
    except KeyError as exc:
        raise RemoraError(
            f"Remora Dense Core ABI supports only ranks 0 through {MAX_RANK}"
        ) from exc


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


def descriptor_rank(descriptor: ctypes.Structure) -> int:
    """Return the rank of a Remora memref descriptor instance or type."""
    descriptor_type = descriptor if isinstance(descriptor, type) else type(descriptor)
    try:
        return _DESCRIPTOR_RANKS[descriptor_type]
    except KeyError as exc:
        raise RemoraError(f"unknown Remora memref descriptor type {descriptor_type}") from exc


def descriptor_shape(descriptor: ctypes.Structure) -> tuple[int, ...]:
    """Return descriptor sizes as a shape tuple."""
    rank = descriptor_rank(descriptor)
    return tuple(int(getattr(descriptor, f"size{axis}")) for axis in range(rank))


def descriptor_strides(descriptor: ctypes.Structure) -> tuple[int, ...]:
    """Return descriptor strides in elements."""
    rank = descriptor_rank(descriptor)
    return tuple(int(getattr(descriptor, f"stride{axis}")) for axis in range(rank))


def numpy_from_memref_descriptor(
    descriptor: ctypes.Structure,
    dtype: object,
    *,
    copy: bool = True,
) -> np.ndarray:
    """Create a numpy value from a rank-specialized Remora memref descriptor."""
    np_dtype = np.dtype(dtype)
    shape = descriptor_shape(descriptor)
    itemsize = np_dtype.itemsize
    aligned = descriptor.aligned
    if aligned is None:
        raise RemoraError("descriptor aligned pointer is null")

    offset = int(descriptor.offset)
    data_address = int(aligned) + offset * itemsize

    if not shape:
        scalar_type = np.ctypeslib.as_ctypes_type(np_dtype)
        value = ctypes.cast(data_address, ctypes.POINTER(scalar_type)).contents.value
        return np.array(value, dtype=np_dtype)
    if any(dim == 0 for dim in shape):
        return np.empty(shape, dtype=np_dtype)

    byte_strides = tuple(stride * itemsize for stride in descriptor_strides(descriptor))
    lowest, highest = _span_byte_bounds(shape, byte_strides)
    buffer_address = data_address + lowest
    view = np.ndarray(
        shape=shape,
        dtype=np_dtype,
        buffer=_memory_at(buffer_address, highest - lowest + itemsize),
        offset=-lowest,
        strides=byte_strides,
    )
    return view.copy() if copy else view


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


def _span_byte_bounds(shape: tuple[int, ...], byte_strides: tuple[int, ...]) -> tuple[int, int]:
    lowest = 0
    highest = 0
    for dim, stride in zip(shape, byte_strides):
        extent = (dim - 1) * stride
        lowest += min(0, extent)
        highest += max(0, extent)
    return lowest, highest


def _memory_at(address: int, size: int) -> ctypes.Array:
    return (ctypes.c_char * size).from_address(address)
