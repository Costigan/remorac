import pytest

from remora.ast_nodes import SourceLoc
from remora.constraints import ConstraintError, DimEq, ShapeEq, match_shape_template, solve_exact
from remora.index import DimAdd, DimLit, DimVar, ShapeLit, ShapeVar
from remora.types import StaticDim


def test_exact_solver_binds_dimension_variable_to_static_dimension():
    bindings = solve_exact([DimEq(DimVar("n"), StaticDim(5))])

    assert bindings == {"n": StaticDim(5)}


def test_exact_solver_rejects_conflicting_repeated_dimension_variable():
    with pytest.raises(ConstraintError, match="dimension mismatch"):
        solve_exact([
            DimEq(DimVar("n"), StaticDim(3)),
            DimEq(DimVar("n"), StaticDim(4)),
        ])


def test_exact_solver_matches_fixed_rank_shape_templates():
    bindings = match_shape_template(
        (DimVar("m"), DimVar("n")),
        (StaticDim(3), StaticDim(4)),
    )

    assert bindings == {"m": StaticDim(3), "n": StaticDim(4)}


def test_exact_solver_rejects_shape_rank_mismatch():
    with pytest.raises(ConstraintError, match="rank mismatch"):
        match_shape_template((DimVar("n"),), (StaticDim(3), StaticDim(4)))


def test_exact_solver_rejects_mismatched_concrete_dimensions():
    with pytest.raises(ConstraintError, match="dimension mismatch"):
        solve_exact([DimEq(StaticDim(3), StaticDim(4))])


def test_exact_solver_rejects_symbolic_arithmetic_binding():
    with pytest.raises(ConstraintError, match="non-concrete"):
        solve_exact([DimEq(DimVar("n"), DimAdd(DimVar("m"), DimLit(1)))])


def test_exact_solver_rejects_shape_variables_in_phase_7a():
    with pytest.raises(ConstraintError, match="fixed-rank shape literals"):
        solve_exact([ShapeEq(ShapeVar("s"), ShapeLit((StaticDim(3),)))])


def test_constraint_errors_include_source_location():
    loc = SourceLoc("bad.lisp", 2, 5)

    with pytest.raises(ConstraintError, match=r"bad\.lisp:2:5"):
        solve_exact([DimEq(StaticDim(3), StaticDim(4), loc)])
