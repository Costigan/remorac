import ctypes

import numpy as np

from remora.abi import (
    RemoraMemRef0,
    RemoraMemRef1,
    RemoraMemRef2,
    RemoraMemRef3,
    descriptor_shape,
    descriptor_strides,
    element_strides,
    make_memref_descriptor,
    make_numpy_memref_descriptor,
    numpy_from_memref_descriptor,
)


def field_names(struct_type):
    return [name for name, _field_type in struct_type._fields_]


def test_memref_field_names_and_order():
    assert field_names(RemoraMemRef0) == ["allocated", "aligned", "offset"]
    assert field_names(RemoraMemRef1) == [
        "allocated",
        "aligned",
        "offset",
        "size0",
        "stride0",
    ]
    assert field_names(RemoraMemRef2) == [
        "allocated",
        "aligned",
        "offset",
        "size0",
        "size1",
        "stride0",
        "stride1",
    ]
    assert field_names(RemoraMemRef3) == [
        "allocated",
        "aligned",
        "offset",
        "size0",
        "size1",
        "size2",
        "stride0",
        "stride1",
        "stride2",
    ]


def test_memref_struct_sizes_are_stable_on_64_bit():
    pointer_size = ctypes.sizeof(ctypes.c_void_p)
    assert pointer_size == 8
    assert ctypes.sizeof(RemoraMemRef0) == 2 * pointer_size + 1 * 8
    assert ctypes.sizeof(RemoraMemRef1) == 2 * pointer_size + 3 * 8
    assert ctypes.sizeof(RemoraMemRef2) == 2 * pointer_size + 5 * 8
    assert ctypes.sizeof(RemoraMemRef3) == 2 * pointer_size + 7 * 8


def test_contiguous_rank_1_descriptor_from_numpy():
    array = np.arange(5, dtype=np.float32)
    descriptor = make_numpy_memref_descriptor(array)

    assert isinstance(descriptor, RemoraMemRef1)
    assert descriptor.allocated == array.ctypes.data
    assert descriptor.aligned == array.ctypes.data
    assert descriptor.offset == 0
    assert descriptor.size0 == 5
    assert descriptor.stride0 == 1


def test_contiguous_rank_2_descriptor_from_numpy():
    array = np.arange(12, dtype=np.float32).reshape(3, 4)
    descriptor = make_numpy_memref_descriptor(array)

    assert isinstance(descriptor, RemoraMemRef2)
    assert (descriptor.size0, descriptor.size1) == (3, 4)
    assert (descriptor.stride0, descriptor.stride1) == (4, 1)


def test_contiguous_rank_3_descriptor_from_numpy():
    array = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
    descriptor = make_numpy_memref_descriptor(array)

    assert isinstance(descriptor, RemoraMemRef3)
    assert (descriptor.size0, descriptor.size1, descriptor.size2) == (2, 3, 4)
    assert (descriptor.stride0, descriptor.stride1, descriptor.stride2) == (12, 4, 1)


def test_rank_0_scalar_descriptor_creation():
    array = np.asarray(42, dtype=np.int32)
    descriptor = make_numpy_memref_descriptor(array)

    assert isinstance(descriptor, RemoraMemRef0)
    assert descriptor.allocated == array.ctypes.data
    assert descriptor.aligned == array.ctypes.data
    assert descriptor.offset == 0


def test_manual_descriptor_creation():
    descriptor = make_memref_descriptor(
        device_or_host_ptr=0x1000,
        shape=(2, 3),
        strides=(3, 1),
        dtype=np.float32,
    )

    assert isinstance(descriptor, RemoraMemRef2)
    assert descriptor.allocated == 0x1000
    assert descriptor.aligned == 0x1000
    assert descriptor.offset == 0
    assert (descriptor.size0, descriptor.size1) == (2, 3)
    assert (descriptor.stride0, descriptor.stride1) == (3, 1)


def test_element_strides_for_transposed_numpy_view():
    array = np.arange(12, dtype=np.float32).reshape(3, 4)
    view = array.T

    assert element_strides(view) == (1, 4)

    descriptor = make_numpy_memref_descriptor(view)
    assert isinstance(descriptor, RemoraMemRef2)
    assert descriptor.allocated == array.base.ctypes.data
    assert descriptor.aligned == array.base.ctypes.data
    assert descriptor.offset == 0
    assert (descriptor.size0, descriptor.size1) == (4, 3)
    assert (descriptor.stride0, descriptor.stride1) == (1, 4)


def test_element_strides_and_offset_for_sliced_numpy_view():
    array = np.arange(20, dtype=np.float32).reshape(4, 5)
    view = array[1:, 2:]

    descriptor = make_numpy_memref_descriptor(view)

    assert descriptor.allocated == array.base.ctypes.data
    assert descriptor.aligned == array.base.ctypes.data
    assert descriptor.offset == 7
    assert (descriptor.size0, descriptor.size1) == (3, 3)
    assert (descriptor.stride0, descriptor.stride1) == (5, 1)


def test_element_strides_and_offset_for_negative_stride_numpy_view():
    array = np.arange(6, dtype=np.float32)
    view = array[::-1]

    descriptor = make_numpy_memref_descriptor(view)

    assert descriptor.allocated == array.ctypes.data
    assert descriptor.aligned == array.ctypes.data
    assert descriptor.offset == 5
    assert descriptor.size0 == 6
    assert descriptor.stride0 == -1


def test_descriptor_shape_and_strides_helpers():
    descriptor = make_memref_descriptor(
        device_or_host_ptr=0x1000,
        shape=(2, 3, 4),
        strides=(12, 4, 1),
        dtype=np.float32,
    )

    assert descriptor_shape(descriptor) == (2, 3, 4)
    assert descriptor_strides(descriptor) == (12, 4, 1)


def test_numpy_from_rank_0_descriptor_round_trip():
    array = np.asarray(42, dtype=np.int32)
    descriptor = make_numpy_memref_descriptor(array)

    result = numpy_from_memref_descriptor(descriptor, np.int32)

    assert result.shape == ()
    assert result.dtype == np.int32
    assert result.item() == 42


def test_numpy_from_rank_1_descriptor_round_trip():
    array = np.arange(5, dtype=np.float32)
    descriptor = make_numpy_memref_descriptor(array)

    result = numpy_from_memref_descriptor(descriptor, np.float32)

    np.testing.assert_array_equal(result, array)


def test_numpy_from_rank_2_descriptor_round_trip_with_view():
    array = np.arange(20, dtype=np.float32).reshape(4, 5)
    view = array[1:, 2:]
    descriptor = make_numpy_memref_descriptor(view)

    result = numpy_from_memref_descriptor(descriptor, np.float32)

    np.testing.assert_array_equal(result, view)


def test_numpy_from_rank_3_descriptor_round_trip():
    array = np.arange(24, dtype=np.int32).reshape(2, 3, 4)
    descriptor = make_numpy_memref_descriptor(array)

    result = numpy_from_memref_descriptor(descriptor, np.int32)

    np.testing.assert_array_equal(result, array)


def test_numpy_from_negative_stride_descriptor_round_trip():
    array = np.arange(6, dtype=np.float32)
    view = array[::-1]
    descriptor = make_numpy_memref_descriptor(view)

    result = numpy_from_memref_descriptor(descriptor, np.float32)

    np.testing.assert_array_equal(result, view)
