"""Compile-time index expressions for dependent Remora shapes.

This module is intentionally independent from the typechecker.  Phase 7 uses
it as the shared vocabulary for dimension and shape variables before programs
are specialized back to backend-compatible static shapes.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Mapping, TypeAlias

from remora.errors import RemoraError


class IndexError(RemoraError):
    """Raised when an index expression is ill-sorted or invalid."""


class IndexSort(Enum):
    DIM = "Dim"
    SHAPE = "Shape"

    def __str__(self) -> str:
        return self.value


@dataclass(frozen=True)
class IndexBinder:
    name: str
    sort: IndexSort


class IndexExpr:
    """Base class for compile-time index expressions."""

    @property
    def sort(self) -> IndexSort:
        raise NotImplementedError


class DimExpr(IndexExpr):
    @property
    def sort(self) -> IndexSort:
        return IndexSort.DIM


class ShapeExpr(IndexExpr):
    @property
    def sort(self) -> IndexSort:
        return IndexSort.SHAPE


@dataclass(frozen=True)
class DimLit(DimExpr):
    value: int

    def __post_init__(self) -> None:
        if self.value < 0:
            raise ValueError("dimension literals must be non-negative")

    def __str__(self) -> str:
        return str(self.value)


@dataclass(frozen=True)
class DimVar(DimExpr):
    name: str

    def __str__(self) -> str:
        return self.name


@dataclass(frozen=True)
class DimAdd(DimExpr):
    left: DimExpr
    right: DimExpr

    def __str__(self) -> str:
        return f"(+ {self.left} {self.right})"


@dataclass(frozen=True)
class DimSub(DimExpr):
    left: DimExpr
    right: DimExpr

    def __str__(self) -> str:
        return f"(- {self.left} {self.right})"


@dataclass(frozen=True)
class ShapeLit(ShapeExpr):
    dims: tuple[DimExpr, ...]

    def __str__(self) -> str:
        dims = " ".join(str(dim) for dim in self.dims)
        return f"(shape {dims})"


@dataclass(frozen=True)
class ShapeVar(ShapeExpr):
    name: str

    def __str__(self) -> str:
        return f"@{self.name}"


@dataclass(frozen=True)
class ShapeConcat(ShapeExpr):
    left: ShapeExpr
    right: ShapeExpr

    def __str__(self) -> str:
        return f"(++ {self.left} {self.right})"


AnyIndexExpr: TypeAlias = DimExpr | ShapeExpr
IndexSubstitution: TypeAlias = Mapping[str, AnyIndexExpr]


def substitute_index(expr: AnyIndexExpr, bindings: IndexSubstitution) -> AnyIndexExpr:
    """Substitute free index variables by name, preserving index sorts."""
    if isinstance(expr, DimVar):
        return _lookup_substitution(expr.name, IndexSort.DIM, bindings) or expr
    if isinstance(expr, ShapeVar):
        return _lookup_substitution(expr.name, IndexSort.SHAPE, bindings) or expr
    if isinstance(expr, DimAdd):
        return normalize_index(
            DimAdd(
                _require_dim(substitute_index(expr.left, bindings)),
                _require_dim(substitute_index(expr.right, bindings)),
            )
        )
    if isinstance(expr, DimSub):
        return normalize_index(
            DimSub(
                _require_dim(substitute_index(expr.left, bindings)),
                _require_dim(substitute_index(expr.right, bindings)),
            )
        )
    if isinstance(expr, ShapeLit):
        return ShapeLit(
            tuple(_require_dim(substitute_index(dim, bindings)) for dim in expr.dims)
        )
    if isinstance(expr, ShapeConcat):
        return normalize_index(
            ShapeConcat(
                _require_shape(substitute_index(expr.left, bindings)),
                _require_shape(substitute_index(expr.right, bindings)),
            )
        )
    return expr


def normalize_index(expr: AnyIndexExpr) -> AnyIndexExpr:
    """Apply local simplifications without solving constraints."""
    if isinstance(expr, DimAdd):
        left = _require_dim(normalize_index(expr.left))
        right = _require_dim(normalize_index(expr.right))
        if isinstance(left, DimLit) and left.value == 0:
            return right
        if isinstance(right, DimLit) and right.value == 0:
            return left
        vl = _dim_value(left)
        vr = _dim_value(right)
        if vl is not None and vr is not None:
            return _static_dim(vl + vr)
        return DimAdd(left, right)
    if isinstance(expr, DimSub):
        left = _require_dim(normalize_index(expr.left))
        right = _require_dim(normalize_index(expr.right))
        if isinstance(right, DimLit) and right.value == 0:
            return left
        vl = _dim_value(left)
        vr = _dim_value(right)
        if vl is not None and vr is not None and vl >= vr:
            return _static_dim(vl - vr)
        return DimSub(left, right)
    if isinstance(expr, ShapeLit):
        return ShapeLit(tuple(_require_dim(normalize_index(dim)) for dim in expr.dims))
    if isinstance(expr, ShapeConcat):
        pieces = _flatten_concat(expr)
        literal_dims: list[DimExpr] = []
        normalized: list[ShapeExpr] = []
        for piece in pieces:
            piece = _require_shape(normalize_index(piece))
            if isinstance(piece, ShapeLit):
                literal_dims.extend(piece.dims)
                continue
            if literal_dims:
                normalized.append(ShapeLit(tuple(literal_dims)))
                literal_dims = []
            normalized.append(piece)
        if literal_dims:
            normalized.append(ShapeLit(tuple(literal_dims)))
        if not normalized:
            return ShapeLit(())
        result = normalized[0]
        for piece in normalized[1:]:
            result = ShapeConcat(result, piece)
        return result
    return expr


def free_index_vars(expr: AnyIndexExpr) -> frozenset[str]:
    if isinstance(expr, (DimVar, ShapeVar)):
        return frozenset({expr.name})
    if isinstance(expr, (DimAdd, DimSub)):
        return free_index_vars(expr.left) | free_index_vars(expr.right)
    if isinstance(expr, ShapeLit):
        result: set[str] = set()
        for dim in expr.dims:
            result.update(free_index_vars(dim))
        return frozenset(result)
    if isinstance(expr, ShapeConcat):
        return free_index_vars(expr.left) | free_index_vars(expr.right)
    return frozenset()


def _lookup_substitution(
    name: str,
    expected_sort: IndexSort,
    bindings: IndexSubstitution,
) -> AnyIndexExpr | None:
    replacement = bindings.get(name)
    if replacement is None:
        return None
    if replacement.sort is not expected_sort:
        raise IndexError(
            f"index substitution for {name!r} has sort {replacement.sort}, "
            f"expected {expected_sort}"
        )
    return replacement


def _flatten_concat(expr: ShapeExpr) -> list[ShapeExpr]:
    if isinstance(expr, ShapeConcat):
        return _flatten_concat(expr.left) + _flatten_concat(expr.right)
    return [expr]


def _require_dim(expr: AnyIndexExpr) -> DimExpr:
    if not isinstance(expr, DimExpr):
        raise IndexError(f"expected dimension expression, got {expr.sort}")
    return expr


def _require_shape(expr: AnyIndexExpr) -> ShapeExpr:
    if not isinstance(expr, ShapeExpr):
        raise IndexError(f"expected shape expression, got {expr.sort}")
    return expr


def _dim_value(expr: DimExpr) -> int | None:
    """Extract a concrete integer from a DimExpr, or None."""
    if isinstance(expr, DimLit):
        return expr.value
    # Also check StaticDim (from types.py) which has .value
    value = getattr(expr, "value", None)
    if isinstance(value, int):
        return value
    return None


def _static_dim(value: int) -> DimExpr:
    """Create a StaticDim for a concrete value, avoiding circular import."""
    from remora.types import StaticDim
    return StaticDim(value)


# ── Alpha-equivalence ──────────────────────────────────────────────────────


def index_alpha_equivalent(
    a: AnyIndexExpr,
    b: AnyIndexExpr,
    env: dict[str, str] | None = None,
) -> bool:
    """Check if two index expressions are equivalent up to bound-variable renaming.

    *env* maps bound variable names to canonical names.  Variables not in
    *env* are treated as free and must match exactly.
    """
    if env is None:
        env = {}

    # Concrete values must match
    if isinstance(a, DimLit) and isinstance(b, DimLit):
        return a.value == b.value
    if isinstance(a, (DimLit,)) != isinstance(b, (DimLit,)):
        return False

    # Variables: check env mapping
    if isinstance(a, DimVar) and isinstance(b, DimVar):
        ca = env.get(a.name, a.name)
        cb = env.get(b.name, b.name)
        return ca == cb

    if isinstance(a, ShapeVar) and isinstance(b, ShapeVar):
        ca = env.get(a.name, a.name)
        cb = env.get(b.name, b.name)
        return ca == cb

    # Structural recursion
    if isinstance(a, DimAdd) and isinstance(b, DimAdd):
        return index_alpha_equivalent(a.left, b.left, env) and index_alpha_equivalent(
            a.right, b.right, env
        )
    if isinstance(a, DimSub) and isinstance(b, DimSub):
        return index_alpha_equivalent(a.left, b.left, env) and index_alpha_equivalent(
            a.right, b.right, env
        )
    if isinstance(a, ShapeLit) and isinstance(b, ShapeLit):
        return len(a.dims) == len(b.dims) and all(
            index_alpha_equivalent(ad, bd, env) for ad, bd in zip(a.dims, b.dims)
        )
    if isinstance(a, ShapeConcat) and isinstance(b, ShapeConcat):
        return index_alpha_equivalent(a.left, b.left, env) and index_alpha_equivalent(
            a.right, b.right, env
        )

    # Different kinds
    return type(a) is type(b) and a == b
