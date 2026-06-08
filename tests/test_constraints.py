import pytest

from remora.ast_nodes import SourceLoc
from remora.constraints import (
    ConstraintError,
    DimEq,
    ShapeEq,
    match_shape_expr_pattern,
    match_shape_template,
    solve_exact,
    solve_linear,
    solve_with_shapes,
)
from remora.index import DimAdd, DimLit, DimSub, DimVar, ShapeLit, ShapeVar
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


# ── Phase 7.4: Arithmetic constraint solver ───────────────────────────────

def test_arithmetic_solver_add_var_left():
    bindings = solve_with_shapes(
        [DimEq(DimAdd(DimVar("a"), DimLit(3)), DimLit(8))]
    )
    assert "a" in bindings
    assert _dim_value(bindings["a"]) == 5


def test_arithmetic_solver_add_var_right():
    bindings = solve_with_shapes(
        [DimEq(DimAdd(DimLit(3), DimVar("b")), DimLit(10))]
    )
    assert "b" in bindings
    assert _dim_value(bindings["b"]) == 7


def test_arithmetic_solver_add_both_known():
    bindings = solve_with_shapes(
        [DimEq(DimAdd(DimLit(2), DimLit(3)), DimLit(5))]
    )
    assert bindings == {}


def test_arithmetic_solver_add_mismatch():
    # DimLit(2) + DimLit(3) = 5 normalizes to 5=6 → dimension mismatch
    with pytest.raises(ConstraintError, match="dimension mismatch"):
        solve_with_shapes([DimEq(DimAdd(DimLit(2), DimLit(3)), DimLit(6))])


def test_arithmetic_solver_add_negative_rejected():
    with pytest.raises(ConstraintError, match="negative"):
        solve_with_shapes([DimEq(DimAdd(DimVar("a"), DimLit(8)), DimLit(5))])


def test_arithmetic_solver_sub_var_left():
    bindings = solve_with_shapes(
        [DimEq(DimSub(DimVar("a"), DimLit(3)), DimLit(5))]
    )
    assert "a" in bindings
    assert _dim_value(bindings["a"]) == 8


def test_arithmetic_solver_sub_var_right():
    bindings = solve_with_shapes(
        [DimEq(DimSub(DimLit(10), DimVar("b")), DimLit(3))]
    )
    assert "b" in bindings
    assert _dim_value(bindings["b"]) == 7


def test_arithmetic_solver_sub_negative_rejected():
    # 3 - b = 5  →  b = -2  (rejected as negative)
    with pytest.raises(ConstraintError, match="negative"):
        solve_with_shapes([DimEq(DimSub(DimLit(3), DimVar("b")), DimLit(5))])


def test_arithmetic_solver_add_in_shape_template():
    # a + b = 7 with both unknown → needs one known operand
    with pytest.raises(ConstraintError, match="need one known operand"):
        match_shape_expr_pattern(
            ShapeLit((DimAdd(DimVar("a"), DimVar("b")),)),
            (DimLit(7),),
        )


def test_arithmetic_solver_add_with_known_in_shape():
    bindings = match_shape_expr_pattern(
        ShapeLit((DimAdd(DimVar("a"), DimLit(2)),)),
        (DimLit(7),),
    )
    assert "a" in bindings
    assert _dim_value(bindings["a"]) == 5


def _dim_value(binding) -> int:
    if hasattr(binding, "value"):
        return binding.value
    return binding.value


# ── Full linear solver (multi-equation, multi-unknown) ─────────────────────

def test_linear_solver_multi_equation():
    bindings = solve_linear([
        DimEq(DimAdd(DimVar("a"), DimVar("b")), DimLit(10)),
        DimEq(DimVar("b"), DimLit(3)),
    ])
    assert _dim_value(bindings["a"]) == 7
    assert _dim_value(bindings["b"]) == 3


def test_linear_solver_interdependent_adds():
    bindings = solve_linear([
        DimEq(DimAdd(DimVar("a"), DimVar("b")), DimLit(10)),
        DimEq(DimAdd(DimVar("b"), DimLit(2)), DimLit(5)),
    ])
    assert _dim_value(bindings["a"]) == 7
    assert _dim_value(bindings["b"]) == 3


def test_linear_solver_sub_and_add():
    """a + b = 10, b = 3 → a = 7 (one equation anchors the system)"""
    bindings = solve_linear([
        DimEq(DimSub(DimVar("a"), DimLit(3)), DimLit(4)),  # a - 3 = 4 → a = 7
        DimEq(DimAdd(DimVar("a"), DimVar("b")), DimLit(10)),  # 7 + b = 10 → b = 3
    ])
    assert _dim_value(bindings["a"]) == 7
    assert _dim_value(bindings["b"]) == 3


def test_linear_solver_three_equations():
    bindings = solve_linear([
        DimEq(DimAdd(DimVar("a"), DimVar("b")), DimLit(12)),
        DimEq(DimSub(DimVar("a"), DimVar("c")), DimLit(5)),
        DimEq(DimVar("c"), DimLit(3)),
    ])
    assert _dim_value(bindings["a"]) == 8
    assert _dim_value(bindings["b"]) == 4
    assert _dim_value(bindings["c"]) == 3


def test_linear_solver_detects_inconsistency():
    with pytest.raises(ConstraintError):
        solve_linear([
            DimEq(DimAdd(DimVar("a"), DimVar("b")), DimLit(10)),
            DimEq(DimVar("a"), DimLit(3)),
            DimEq(DimVar("b"), DimLit(8)),  # 3 + 8 != 10
        ])


def test_linear_solver_rejects_underconstrained():
    with pytest.raises(ConstraintError):
        solve_linear([
            DimEq(DimAdd(DimVar("a"), DimVar("b")), DimLit(10)),
            # missing: neither a nor b is known
        ])
