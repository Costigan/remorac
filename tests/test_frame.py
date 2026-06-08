"""Tests for remora/frame.py -- shared frame/cell decomposition module."""

from remora.frame import (
    FrameCell,
    apply_frame,
    cell_matches_array_suffix,
    cell_type_candidates,
    decompose_argument,
    infer_lifting,
    principal_frame,
    scalar_cell_and_frame,
    validate_cell_rank,
)
from remora.types import (
    FLOAT,
    INT,
    ArrayType,
    FuncType,
    RemoraTypeError,
    ScalarType,
    StaticDim,
)


def test_scalar_cell_and_frame_scalar():
    s, f = scalar_cell_and_frame(INT)
    assert s == INT
    assert f == ()


def test_scalar_cell_and_frame_array():
    s, f = scalar_cell_and_frame(ArrayType(FLOAT, (StaticDim(3), StaticDim(4))))
    assert s == FLOAT
    assert f == (StaticDim(3), StaticDim(4))


def test_validate_cell_rank_ok():
    validate_cell_rank(5, 3)


def test_validate_cell_rank_equal():
    validate_cell_rank(3, 3)


def test_validate_cell_rank_too_high():
    try:
        validate_cell_rank(2, 3)
        assert False, "expected error"
    except RemoraTypeError as e:
        assert "rank 2 is too low for cell rank 3" in str(e).lower() or "too low" in str(e).lower()


def test_cell_matches_scalar_scalar():
    assert cell_matches_array_suffix(INT, INT)
    assert not cell_matches_array_suffix(INT, FLOAT)


def test_cell_matches_scalar_array():
    assert cell_matches_array_suffix(INT, ArrayType(INT, (StaticDim(3), StaticDim(4))))
    assert not cell_matches_array_suffix(FLOAT, ArrayType(INT, (StaticDim(3),)))


def test_cell_matches_array_suffix_exact():
    cell = ArrayType(FLOAT, (StaticDim(4),))
    array = ArrayType(FLOAT, (StaticDim(3), StaticDim(4)))
    assert cell_matches_array_suffix(cell, array)


def test_cell_matches_array_suffix_mismatched():
    cell = ArrayType(INT, (StaticDim(3),))
    array = ArrayType(INT, (StaticDim(4),))
    assert not cell_matches_array_suffix(cell, array)


def test_cell_matches_array_suffix_cell_too_long():
    cell = ArrayType(FLOAT, (StaticDim(4), StaticDim(5)))
    array = ArrayType(FLOAT, (StaticDim(3),))
    assert not cell_matches_array_suffix(cell, array)


def test_cell_type_candidates_scalar():
    assert cell_type_candidates(INT) == [INT]


def test_cell_type_candidates_rank3():
    s3 = StaticDim(3)
    s4 = StaticDim(4)
    s5 = StaticDim(5)
    arr = ArrayType(FLOAT, (s3, s4, s5))
    candidates = cell_type_candidates(arr)
    assert len(candidates) == 4  # element + 3 trailing ranks
    assert candidates[0] == FLOAT
    assert candidates[1] == ArrayType(FLOAT, (s5,))
    assert candidates[2] == ArrayType(FLOAT, (s4, s5))
    assert candidates[3] == ArrayType(FLOAT, (s3, s4, s5))


def test_decompose_argument_scalar_cell():
    s3 = StaticDim(3)
    s4 = StaticDim(4)
    func_type = FuncType((INT,), INT)
    array = ArrayType(INT, (s3, s4))
    fc = decompose_argument(func_type, array)
    assert fc.cell_type == INT
    assert fc.frame_shape == (s3, s4)
    assert fc.cell_rank == 0
    assert fc.frame_rank == 2


def test_decompose_argument_array_cell():
    s3 = StaticDim(3)
    s4 = StaticDim(4)
    cell = ArrayType(FLOAT, (s4,))
    func_type = FuncType((cell,), FLOAT)
    array = ArrayType(FLOAT, (s3, s4))
    fc = decompose_argument(func_type, array)
    assert fc.cell_type == cell
    assert fc.frame_shape == (s3,)
    assert fc.cell_rank == 1
    assert fc.frame_rank == 1


def test_decompose_argument_full_array():
    s3 = StaticDim(3)
    s4 = StaticDim(4)
    cell = ArrayType(FLOAT, (s3, s4))
    func_type = FuncType((cell,), FLOAT)
    array = ArrayType(FLOAT, (s3, s4))
    fc = decompose_argument(func_type, array)
    assert fc.cell_type == cell
    assert fc.frame_shape == ()


def test_decompose_argument_cell_too_deep():
    s3 = StaticDim(3)
    cell = ArrayType(FLOAT, (s3, s3))  # rank 2 cell
    func_type = FuncType((cell,), FLOAT)
    array = ArrayType(FLOAT, (s3,))  # rank 1
    try:
        decompose_argument(func_type, array)
        assert False, "expected error"
    except RemoraTypeError as e:
        assert "too low" in str(e).lower() or "rank" in str(e).lower()


def test_principal_frame_equal():
    pf = principal_frame([(StaticDim(3),), (StaticDim(3),)])
    assert pf == (StaticDim(3),)


def test_principal_frame_one_empty():
    pf = principal_frame([(), (StaticDim(3), StaticDim(4))])
    assert pf == (StaticDim(3), StaticDim(4))


def test_principal_frame_prefix():
    pf = principal_frame([(StaticDim(3),), (StaticDim(3), StaticDim(4))])
    assert pf == (StaticDim(3), StaticDim(4))


def test_principal_frame_incompatible():
    pf = principal_frame([(StaticDim(3),), (StaticDim(4),)])
    assert pf is None


def test_principal_frame_empty_list():
    assert principal_frame([]) == ()


def test_apply_frame_no_frame():
    assert apply_frame(INT, ()) == INT


def test_apply_frame_scalar_result():
    result = apply_frame(INT, (StaticDim(3), StaticDim(4)))
    assert result == ArrayType(INT, (StaticDim(3), StaticDim(4)))


def test_apply_frame_array_result():
    result = apply_frame(ArrayType(FLOAT, (StaticDim(5),)), (StaticDim(3), StaticDim(4)))
    assert result == ArrayType(FLOAT, (StaticDim(3), StaticDim(4), StaticDim(5)))


def test_infer_lifting_legacy():
    s3 = StaticDim(3)
    s4 = StaticDim(4)
    func_type = FuncType((INT,), INT)
    array = ArrayType(INT, (s3, s4))
    frame_shape, result_type = infer_lifting(func_type, array)
    assert frame_shape == (s3, s4)
    assert result_type == ArrayType(INT, (s3, s4))


def test_infer_lifting_legacy_array_cell():
    s3 = StaticDim(3)
    s4 = StaticDim(4)
    cell = ArrayType(FLOAT, (s4,))
    func_type = FuncType((cell,), FLOAT)
    array = ArrayType(FLOAT, (s3, s4))
    frame_shape, result_type = infer_lifting(func_type, array)
    assert frame_shape == (s3,)
    assert result_type == ArrayType(FLOAT, (s3,))


def test_infer_lifting_binary():
    """infer_lifting rejects multi-param function types."""
    s3 = StaticDim(3)
    func_type = FuncType((INT, INT), INT)
    array = ArrayType(INT, (s3,))
    try:
        infer_lifting(func_type, array)
        assert False, "expected error"
    except RemoraTypeError:
        pass


def test_frame_cell_str():
    fc = FrameCell(INT, (StaticDim(3), StaticDim(4)))
    s = str(fc)
    assert "FrameCell" in s
    assert "3" in s
    assert "4" in s
