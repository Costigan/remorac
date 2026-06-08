"""Constraint representation and restricted solvers for dependent indices."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TypeAlias

from remora.ast_nodes import SourceLoc
from remora.errors import RemoraError
from remora.index import (
    AnyIndexExpr,
    DimExpr,
    DimLit,
    DimVar,
    IndexSubstitution,
    ShapeExpr,
    ShapeLit,
    normalize_index,
)


class ConstraintError(RemoraError):
    """Raised when index constraints cannot be solved."""

    def __init__(self, message: str, loc: SourceLoc | None = None):
        if loc is not None:
            message = f"{loc.file}:{loc.line}:{loc.col}: {message}"
        super().__init__(message)
        self.loc = loc


@dataclass(frozen=True)
class DimEq:
    left: DimExpr
    right: DimExpr
    loc: SourceLoc | None = None


@dataclass(frozen=True)
class ShapeEq:
    left: ShapeExpr
    right: ShapeExpr
    loc: SourceLoc | None = None


Constraint: TypeAlias = DimEq | ShapeEq


def solve_exact(constraints: list[Constraint]) -> dict[str, DimExpr]:
    """Solve Phase 7a exact dimension constraints.

    This solver only handles fixed-rank shape literals and dimension variables
    bound to concrete dimensions.  It deliberately rejects symbolic arithmetic
    and shape variables; later Phase 7 solvers can extend this contract.
    """
    bindings: dict[str, DimExpr] = {}
    for constraint in constraints:
        if isinstance(constraint, DimEq):
            _solve_dim_eq(constraint.left, constraint.right, bindings, constraint.loc)
        elif isinstance(constraint, ShapeEq):
            _solve_shape_eq(constraint.left, constraint.right, bindings, constraint.loc)
    return bindings


def match_shape_template(
    expected: tuple[DimExpr, ...],
    actual: tuple[DimExpr, ...],
    *,
    loc: SourceLoc | None = None,
) -> dict[str, DimExpr]:
    """Bind dimension variables in an expected shape to an actual shape."""
    return solve_exact([ShapeEq(ShapeLit(expected), ShapeLit(actual), loc)])


def apply_dim_bindings(expr: AnyIndexExpr, bindings: IndexSubstitution) -> AnyIndexExpr:
    """Normalize an index expression after exact-solver substitution."""
    from remora.index import substitute_index

    return normalize_index(substitute_index(expr, bindings))


def _solve_shape_eq(
    left: ShapeExpr,
    right: ShapeExpr,
    bindings: dict[str, DimExpr],
    loc: SourceLoc | None,
) -> None:
    left = _require_shape_lit(normalize_index(left), loc)
    right = _require_shape_lit(normalize_index(right), loc)
    if len(left.dims) != len(right.dims):
        raise ConstraintError(
            f"shape rank mismatch: expected rank {len(left.dims)}, got rank {len(right.dims)}",
            loc,
        )
    for left_dim, right_dim in zip(left.dims, right.dims):
        _solve_dim_eq(left_dim, right_dim, bindings, loc)


def _solve_dim_eq(
    left: DimExpr,
    right: DimExpr,
    bindings: dict[str, DimExpr],
    loc: SourceLoc | None,
) -> None:
    left = _normalize_bound_dim(left, bindings)
    right = _normalize_bound_dim(right, bindings)
    if left == right:
        return
    if isinstance(left, DimVar):
        _bind_dim(left.name, right, bindings, loc)
        return
    if isinstance(right, DimVar):
        _bind_dim(right.name, left, bindings, loc)
        return
    left_value = _static_dim_value(left)
    right_value = _static_dim_value(right)
    if left_value is not None and right_value is not None:
        if left_value != right_value:
            raise ConstraintError(
                f"dimension mismatch: expected {left_value}, got {right_value}",
                loc,
            )
        return
    raise ConstraintError(f"cannot solve dimension equality {left} = {right}", loc)


def _bind_dim(
    name: str,
    value: DimExpr,
    bindings: dict[str, DimExpr],
    loc: SourceLoc | None,
) -> None:
    if isinstance(value, DimVar) and value.name == name:
        return
    existing = bindings.get(name)
    if existing is not None:
        _solve_dim_eq(existing, value, bindings, loc)
        return
    if _static_dim_value(value) is None:
        raise ConstraintError(f"cannot bind dimension {name} to non-concrete {value}", loc)
    bindings[name] = value


def _normalize_bound_dim(expr: DimExpr, bindings: dict[str, DimExpr]) -> DimExpr:
    if isinstance(expr, DimVar):
        return bindings.get(expr.name, expr)
    normalized = normalize_index(expr)
    if not isinstance(normalized, DimExpr):
        raise AssertionError("dimension normalization returned non-dimension")
    return normalized


def _require_shape_lit(expr: AnyIndexExpr, loc: SourceLoc | None) -> ShapeLit:
    if not isinstance(expr, ShapeLit):
        raise ConstraintError(f"exact solver requires fixed-rank shape literals, got {expr}", loc)
    return expr


def _static_dim_value(expr: DimExpr) -> int | None:
    if isinstance(expr, DimLit):
        return expr.value
    value = getattr(expr, "value", None)
    if isinstance(value, int):
        return value
    return None
