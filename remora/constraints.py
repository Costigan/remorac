"""Constraint representation and restricted solvers for dependent indices."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TypeAlias

from remora.ast_nodes import SourceLoc
from remora.errors import RemoraError
from remora.index import (
    AnyIndexExpr,
    DimAdd,
    DimExpr,
    DimLit,
    DimSub,
    DimVar,
    IndexSubstitution,
    ShapeConcat,
    ShapeExpr,
    ShapeLit,
    ShapeVar,
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


def solve_with_shapes(constraints: list[Constraint]) -> dict[str, AnyIndexExpr]:
    """Solve constraints, possibly producing ShapeExpr bindings for ShapeVars.

    This extends ``solve_exact`` by allowing:
    - ShapeVar matching against concrete shapes
    - ShapeConcat equations with finite split search over concrete shapes
    """
    bindings: dict[str, AnyIndexExpr] = {}
    for constraint in constraints:
        if isinstance(constraint, DimEq):
            _solve_dim_eq_any(
                constraint.left, constraint.right, bindings, constraint.loc
            )
        elif isinstance(constraint, ShapeEq):
            _solve_shape_eq_any(
                constraint.left, constraint.right, bindings, constraint.loc
            )
    return bindings


def match_shape_template_with_shapes(
    expected: tuple[DimExpr, ...],
    actual: tuple[DimExpr, ...],
    shape_binders: set[str],
    *,
    loc: SourceLoc | None = None,
) -> dict[str, AnyIndexExpr]:
    """Bind dim and shape variables in an expected shape to an actual shape.

    For Shape binder names in *shape_binders*, the corresponding DimVar in
    the expected shape tuple is treated as a pointer to a shape variable
    that should bind to the entire actual shape tuple.
    """
    # Check if any of the expected dims is a DimVar that maps to a shape binder
    if expected and len(expected) == 1 and isinstance(expected[0], DimVar):
        name = expected[0].name
        if name in shape_binders:
            # This is a shape-variable reference: bind the entire shape
            return {name: ShapeLit(actual)}

    # Standard dim-level matching
    return {
        name: expr
        for name, expr in solve_exact(
            [ShapeEq(ShapeLit(expected), ShapeLit(actual), loc)]
        ).items()
    }


def match_shape_expr_pattern(
    pattern: ShapeExpr,
    actual: tuple[DimExpr, ...],
    *,
    loc: SourceLoc | None = None,
) -> dict[str, AnyIndexExpr]:
    """Match a shape expression pattern against a concrete shape tuple.

    Supports:
      - ``ShapeLit(dims)`` → exact rank/dim matching
      - ``ShapeVar(name)`` → bind name to the entire actual shape
      - ``ShapeConcat(left, right)`` → finite split search with
        concrete suffixes/prefixes
    """
    if isinstance(pattern, ShapeVar):
        return {pattern.name: ShapeLit(actual)}

    if isinstance(pattern, ShapeLit):
        return {
            name: expr
            for name, expr in solve_with_shapes(
                [ShapeEq(pattern, ShapeLit(actual), loc)]
            ).items()
        }

    if isinstance(pattern, ShapeConcat):
        return _solve_shape_concat(pattern, actual, loc)

    raise ConstraintError(
        f"cannot match shape pattern {pattern} against concrete shape", loc
    )


def _solve_shape_concat(
    pattern: ShapeConcat,
    actual: tuple[DimExpr, ...],
    loc: SourceLoc | None,
) -> dict[str, AnyIndexExpr]:
    """Enumerate splits of *actual* to solve a ShapeConcat pattern.

    Both sides of the concat must individually match prefixes / suffixes.
    """
    right_norm = normalize_index(pattern.right)

    # If the right side is a shape variable (rest var), it absorbs everything
    # after the left side matches
    if isinstance(right_norm, ShapeVar):
        left_norm = normalize_index(pattern.left)
        if isinstance(left_norm, ShapeLit):
            left_len = len(left_norm.dims)
            left_bindings = match_shape_expr_pattern(
                left_norm, actual[:left_len], loc=loc
            )
            # Right variable binds to the remaining shape
            right_shape = actual[left_len:]
            left_bindings[right_norm.name] = ShapeLit(right_shape)
            return left_bindings

    left_norm = normalize_index(pattern.left)

    # If the left side is a shape variable (prefix var), it absorbs the prefix
    if isinstance(left_norm, ShapeVar):
        if isinstance(right_norm, ShapeLit):
            right_len = len(right_norm.dims)
            if tuple(actual[-right_len:]) != right_norm.dims:
                raise ConstraintError(
                    f"shape suffix mismatch: expected {right_norm.dims}, "
                    f"got {tuple(actual[-right_len:])}",
                    loc,
                )
            # Left variable binds to the prefix
            left_shape = actual[:-right_len]
            return {left_norm.name: ShapeLit(left_shape)}

    # General split search: try all split points
    total = len(actual)
    left_norm = normalize_index(pattern.left)
    right_norm = normalize_index(pattern.right)

    errors: list[str] = []
    for split_at in range(total + 1):
        prefix = actual[:split_at]
        suffix = actual[split_at:]
        try:
            left_bindings = match_shape_expr_pattern(
                left_norm, prefix, loc=loc
            )
        except ConstraintError as e:
            errors.append(str(e))
            continue
        try:
            right_bindings = match_shape_expr_pattern(
                right_norm, suffix, loc=loc
            )
        except ConstraintError as e:
            errors.append(str(e))
            continue
        # Merge bindings, checking for conflicts
        merged: dict[str, AnyIndexExpr] = dict(left_bindings)
        for name, expr in right_bindings.items():
            if name in merged:
                if merged[name] != expr:
                    raise ConstraintError(
                        f"conflicting binding for {name}: "
                        f"{merged[name]} vs {expr}",
                        loc,
                    )
            else:
                merged[name] = expr
        return merged

    raise ConstraintError(
        f"cannot split shape {actual} to match concat pattern {pattern}: "
        f"{'; '.join(errors)}",
        loc,
    )


def _solve_shape_eq_any(
    left: ShapeExpr,
    right: ShapeExpr,
    bindings: dict[str, AnyIndexExpr],
    loc: SourceLoc | None,
) -> None:
    """Solve a ShapeEq allowing ShapeVar bindings."""
    left = normalize_index(left)
    right = normalize_index(right)

    # Try to reduce to simple cases first
    if isinstance(left, ShapeVar):
        _bind_shape(left.name, right, bindings, loc)
        return
    if isinstance(right, ShapeVar):
        _bind_shape(right.name, left, bindings, loc)
        return

    if isinstance(left, ShapeLit) and isinstance(right, ShapeLit):
        if len(left.dims) != len(right.dims):
            raise ConstraintError(
                f"shape rank mismatch: expected rank {len(left.dims)}, "
                f"got rank {len(right.dims)}",
                loc,
            )
        for left_dim, right_dim in zip(left.dims, right.dims):
            _solve_dim_eq_any(left_dim, right_dim, bindings, loc)
        return

    if isinstance(left, ShapeConcat) or isinstance(right, ShapeConcat):
        # Convert concrete sides to ShapeLit
        if isinstance(left, ShapeConcat) and isinstance(right, ShapeLit):
            result = _solve_shape_concat(
                left, right.dims, loc
            )
            _merge_any_bindings(bindings, result, loc)
            return
        if isinstance(right, ShapeConcat) and isinstance(left, ShapeLit):
            result = _solve_shape_concat(
                right, left.dims, loc
            )
            _merge_any_bindings(bindings, result, loc)
            return

    raise ConstraintError(
        f"cannot solve shape equality {left} = {right}", loc
    )


def _solve_dim_eq_any(
    left: DimExpr,
    right: DimExpr,
    bindings: dict[str, AnyIndexExpr],
    loc: SourceLoc | None,
) -> None:
    """Solve a DimEq allowing both Dim and Shape variable bindings."""
    left = _normalize_bound_dim_any(left, bindings)
    right = _normalize_bound_dim_any(right, bindings)
    if left == right:
        return
    if isinstance(left, DimVar):
        _bind_dim_any(left.name, right, bindings, loc)
        return
    if isinstance(right, DimVar):
        _bind_dim_any(right.name, left, bindings, loc)
        return
    left_value = _static_dim_value(left)
    right_value = _static_dim_value(right)
    if left_value is not None and right_value is not None:
        if left_value != right_value:
            raise ConstraintError(
                f"dimension mismatch: expected {left_value}, got {right_value}", loc
            )
        return
    if isinstance(left, DimAdd):
        _solve_dim_add_eq(left, right, bindings, loc)
        return
    if isinstance(right, DimAdd):
        _solve_dim_add_eq(right, left, bindings, loc)
        return
    if isinstance(left, DimSub):
        _solve_dim_sub_eq(left, right, bindings, loc)
        return
    if isinstance(right, DimSub):
        _solve_dim_sub_eq(right, left, bindings, loc)
        return
    raise ConstraintError(
        f"cannot solve dimension equality {left} = {right}", loc
    )


def _bind_dim_any(
    name: str,
    value: DimExpr,
    bindings: dict[str, AnyIndexExpr],
    loc: SourceLoc | None,
) -> None:
    if isinstance(value, DimVar) and value.name == name:
        return
    existing = bindings.get(name)
    if existing is not None:
        if isinstance(existing, DimExpr):
            _solve_dim_eq_any(existing, _require_dim(value), bindings, loc)
        else:
            raise ConstraintError(
                f"cannot rebind shape variable {name} to dimension {value}", loc
            )
        return
    if _static_dim_value(value) is None:
        raise ConstraintError(
            f"cannot bind dimension {name} to non-concrete {value}", loc
        )
    bindings[name] = value


def _bind_shape(
    name: str,
    value: ShapeExpr,
    bindings: dict[str, AnyIndexExpr],
    loc: SourceLoc | None,
) -> None:
    if isinstance(value, ShapeVar) and value.name == name:
        return
    existing = bindings.get(name)
    if existing is not None:
        _solve_shape_eq_any(existing, value, bindings, loc)  # type: ignore[arg-type]
        return
    bindings[name] = value


def _normalize_bound_dim_any(
    expr: DimExpr, bindings: dict[str, AnyIndexExpr]
) -> DimExpr:
    if isinstance(expr, DimVar):
        bound = bindings.get(expr.name)
        if bound is not None and isinstance(bound, DimExpr):
            return bound
    normalized = normalize_index(expr)
    if not isinstance(normalized, DimExpr):
        raise AssertionError("dimension normalization returned non-dimension")
    return normalized


def _merge_any_bindings(
    bindings: dict[str, AnyIndexExpr],
    new_bindings: dict[str, AnyIndexExpr],
    loc: SourceLoc | None,
) -> None:
    for name, expr in new_bindings.items():
        if name in bindings:
            if bindings[name] != expr:
                raise ConstraintError(
                    f"conflicting binding for {name}: "
                    f"{bindings[name]} vs {expr}",
                    loc,
                )
        else:
            bindings[name] = expr


def _require_dim(expr: AnyIndexExpr) -> DimExpr:
    if not isinstance(expr, DimExpr):
        raise ConstraintError(f"expected dimension expression, got {expr.sort}")
    return expr


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


# ── Dimension arithmetic solvers ──────────────────────────────────────────


def _solve_dim_add_eq(
    add_expr: DimAdd,
    other: DimExpr,
    bindings: dict[str, AnyIndexExpr],
    loc: SourceLoc | None,
) -> None:
    """Solve ``DimAdd(left, right) = other`` for a free variable."""
    left = add_expr.left
    right = add_expr.right
    target = _static_dim_value(other)
    if target is None:
        raise ConstraintError(
            f"cannot solve {add_expr} = {other}: right-hand side must be concrete",
            loc,
        )

    left_value = _static_dim_value(_normalize_bound_dim_any(left, bindings))
    right_value = _static_dim_value(_normalize_bound_dim_any(right, bindings))

    if left_value is not None and right_value is not None:
        if left_value + right_value != target:
            raise ConstraintError(
                f"arithmetic mismatch: {left_value} + {right_value} != {target}",
                loc,
            )
        return

    # Try to solve for a single free variable
    if left_value is not None and isinstance(_normalize_bound_dim_any(right, bindings), DimVar):
        solved = target - left_value
        if solved < 0:
            raise ConstraintError(
                f"dimension subtraction {target} - {left_value} would be negative",
                loc,
            )
        rv = _normalize_bound_dim_any(right, bindings)
        _bind_dim_any(rv.name, DimLit(solved), bindings, loc)
        return

    if right_value is not None and isinstance(_normalize_bound_dim_any(left, bindings), DimVar):
        solved = target - right_value
        if solved < 0:
            raise ConstraintError(
                f"dimension subtraction {target} - {right_value} would be negative",
                loc,
            )
        lv = _normalize_bound_dim_any(left, bindings)
        _bind_dim_any(lv.name, DimLit(solved), bindings, loc)
        return

    raise ConstraintError(
        f"cannot solve {add_expr} = {target}: need one known operand", loc
    )


def _solve_dim_sub_eq(
    sub_expr: DimSub,
    other: DimExpr,
    bindings: dict[str, AnyIndexExpr],
    loc: SourceLoc | None,
) -> None:
    """Solve ``DimSub(left, right) = other`` for a free variable."""
    left = sub_expr.left
    right = sub_expr.right
    target = _static_dim_value(other)
    if target is None:
        raise ConstraintError(
            f"cannot solve {sub_expr} = {other}: right-hand side must be concrete",
            loc,
        )

    left_value = _static_dim_value(_normalize_bound_dim_any(left, bindings))
    right_value = _static_dim_value(_normalize_bound_dim_any(right, bindings))

    if left_value is not None and right_value is not None:
        if left_value - right_value != target:
            raise ConstraintError(
                f"arithmetic mismatch: {left_value} - {right_value} != {target}",
                loc,
            )
        if left_value - right_value < 0:
            raise ConstraintError(
                f"dimension subtraction {left_value} - {right_value} is negative",
                loc,
            )
        return

    # Solve left - concrete = target  →  left = target + concrete
    if right_value is not None and isinstance(_normalize_bound_dim_any(left, bindings), DimVar):
        solved = target + right_value
        lv = _normalize_bound_dim_any(left, bindings)
        _bind_dim_any(lv.name, DimLit(solved), bindings, loc)
        return

    # Solve concrete - right = target  →  right = concrete - target
    if left_value is not None and isinstance(_normalize_bound_dim_any(right, bindings), DimVar):
        solved = left_value - target
        if solved < 0:
            raise ConstraintError(
                f"dimension subtraction {left_value} - {target} would be negative for {right}",
                loc,
            )
        rv = _normalize_bound_dim_any(right, bindings)
        _bind_dim_any(rv.name, DimLit(solved), bindings, loc)
        return

    raise ConstraintError(
        f"cannot solve {sub_expr} = {target}: need one known operand", loc
    )
