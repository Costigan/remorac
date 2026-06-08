import pytest

from remora.index import (
    DimAdd,
    DimLit,
    DimSub,
    DimVar,
    IndexError,
    ShapeConcat,
    ShapeLit,
    ShapeVar,
    free_index_vars,
    normalize_index,
    substitute_index,
)


def test_dim_literal_rejects_negative_values():
    with pytest.raises(ValueError, match="non-negative"):
        DimLit(-1)


def test_substitutes_dimension_variables():
    expr = DimAdd(DimVar("n"), DimLit(1))

    assert substitute_index(expr, {"n": DimLit(4)}) == DimLit(5)


def test_substitution_rejects_sort_mismatch():
    with pytest.raises(IndexError, match="expected Dim"):
        substitute_index(DimVar("n"), {"n": ShapeLit(())})


def test_normalizes_dimension_addition_identities():
    assert normalize_index(DimAdd(DimLit(0), DimVar("n"))) == DimVar("n")
    assert normalize_index(DimAdd(DimLit(2), DimLit(3))) == DimLit(5)


def test_normalizer_does_not_create_negative_dimension_literals():
    expr = DimSub(DimLit(2), DimLit(3))

    assert normalize_index(expr) == expr


def test_normalizes_shape_concat_literals_and_empty_shapes():
    expr = ShapeConcat(
        ShapeConcat(ShapeLit((DimLit(2),)), ShapeLit(())),
        ShapeLit((DimLit(3),)),
    )

    assert normalize_index(expr) == ShapeLit((DimLit(2), DimLit(3)))


def test_shape_concat_preserves_shape_variables():
    expr = ShapeConcat(ShapeLit((DimLit(2),)), ShapeVar("rest"))

    assert normalize_index(expr) == ShapeConcat(ShapeLit((DimLit(2),)), ShapeVar("rest"))


def test_free_index_vars_tracks_dim_and_shape_vars():
    expr = ShapeConcat(
        ShapeLit((DimVar("n"), DimAdd(DimVar("m"), DimLit(1)))),
        ShapeVar("rest"),
    )

    assert free_index_vars(expr) == frozenset({"n", "m", "rest"})
