import numpy as np

from remora.display import format_result
from remora.types import BOOL, FLOAT, INT, ArrayType, StaticDim


def test_formats_scalars_in_remora_style():
    assert format_result(True, BOOL) == "true"
    assert format_result(False, BOOL) == "false"
    assert format_result(3, INT) == "3"
    assert format_result(3.0, FLOAT) == "3.0"
    assert format_result(3.25, FLOAT) == "3.25"


def test_formats_vectors_in_remora_style():
    assert (
        format_result(np.array([0, 2, 4], dtype=np.int32), ArrayType(INT, (StaticDim(3),)))
        == "[0, 2, 4]"
    )
    assert (
        format_result(
            np.array([0.0, 2.0, 4.5], dtype=np.float32),
            ArrayType(FLOAT, (StaticDim(3),)),
        )
        == "[0.0, 2.0, 4.5]"
    )
    assert (
        format_result(
            np.array([True, False], dtype=np.bool_),
            ArrayType(BOOL, (StaticDim(2),)),
        )
        == "[true, false]"
    )


def test_formats_matrices_and_rank3_arrays():
    matrix = np.array([[1, 2], [3, 4]], dtype=np.int32)
    tensor = np.array([[[1.0], [2.0]], [[3.0], [4.0]]], dtype=np.float32)

    assert (
        format_result(matrix, ArrayType(INT, (StaticDim(2), StaticDim(2))))
        == "[[1, 2],\n [3, 4]]"
    )
    assert (
        format_result(tensor, ArrayType(FLOAT, (StaticDim(2), StaticDim(2), StaticDim(1))))
        == "[[[1.0],\n  [2.0]],\n\n [[3.0],\n  [4.0]]]"
    )


def test_formats_rank4_and_rank10_arrays_with_numpy_style():
    rank4 = np.full((1, 1, 1, 1), 2, dtype=np.int32)
    rank10 = np.full((1, 1, 1, 1, 1, 1, 1, 1, 1, 1), 2, dtype=np.int32)

    assert (
        format_result(rank4, ArrayType(INT, tuple(StaticDim(1) for _axis in range(4))))
        == "[[[[2]]]]"
    )
    assert (
        format_result(rank10, ArrayType(INT, tuple(StaticDim(1) for _axis in range(10))))
        == "[[[[[[[[[[2]]]]]]]]]]"
    )
