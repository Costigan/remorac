import pytest

from remora.dependent_types import (
    free_type_index_vars,
    instantiate_pi_type,
    substitute_type,
)
from remora.index import DimLit, DimVar, IndexBinder, IndexSort, ShapeLit
from remora.types import FLOAT, ArrayType, FuncType, PiType


def test_pi_type_string_includes_index_binders():
    pi = PiType(
        (IndexBinder("n", IndexSort.DIM),),
        FuncType((ArrayType(FLOAT, (DimVar("n"),)),), FLOAT),
    )

    assert "n Dim" in str(pi)


def test_substitute_type_rewrites_array_shape_dimensions():
    value_type = FuncType((ArrayType(FLOAT, (DimVar("n"),)),), FLOAT)

    assert substitute_type(value_type, {"n": DimLit(8)}) == FuncType(
        (ArrayType(FLOAT, (DimLit(8),)),),
        FLOAT,
    )


def test_substitute_type_respects_pi_binder_shadowing():
    pi = PiType(
        (IndexBinder("n", IndexSort.DIM),),
        ArrayType(FLOAT, (DimVar("n"),)),
    )

    assert substitute_type(pi, {"n": DimLit(8)}) == pi


def test_instantiate_pi_type_substitutes_index_binders():
    pi = PiType(
        (IndexBinder("n", IndexSort.DIM),),
        FuncType((ArrayType(FLOAT, (DimVar("n"),)),), FLOAT),
    )

    assert instantiate_pi_type(pi, (DimLit(8),)) == FuncType(
        (ArrayType(FLOAT, (DimLit(8),)),),
        FLOAT,
    )


def test_instantiate_pi_type_rejects_wrong_index_sort():
    pi = PiType(
        (IndexBinder("n", IndexSort.DIM),),
        ArrayType(FLOAT, (DimVar("n"),)),
    )

    with pytest.raises(ValueError, match="sort Dim"):
        instantiate_pi_type(pi, (ShapeLit(()),))


def test_free_type_index_vars_respects_pi_binders():
    value_type = PiType(
        (IndexBinder("n", IndexSort.DIM),),
        FuncType(
            (ArrayType(FLOAT, (DimVar("n"), DimVar("m"))),),
            ArrayType(FLOAT, (DimVar("n"),)),
        ),
    )

    assert free_type_index_vars(value_type) == frozenset({"m"})
