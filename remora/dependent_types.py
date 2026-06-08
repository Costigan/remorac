"""Helpers for dependent Remora types."""

from __future__ import annotations

from remora.index import (
    AnyIndexExpr,
    IndexSort,
    IndexSubstitution,
    free_index_vars,
    substitute_index,
)
from remora.types import (
    ArrayType,
    ForallType,
    FuncType,
    PiType,
    RemoraType,
    ScalarType,
    SigmaType,
    TypeBinder,
    TypeVar,
)


def substitute_type(value_type: RemoraType, bindings: IndexSubstitution) -> RemoraType:
    """Substitute index variables inside a Remora type."""
    if isinstance(value_type, ArrayType):
        if value_type.shape_expr is not None:
            # Substitute into the shape expression and extract concrete dims
            from remora.index import ShapeLit, normalize_index
            new_expr = normalize_index(substitute_index(value_type.shape_expr, bindings))
            if isinstance(new_expr, ShapeLit):
                new_dims = new_expr.dims
                # If the new dims are all concrete (no free variables), drop shape_expr
                if not any(_has_index_var(d) for d in new_dims):
                    return ArrayType(value_type.element, new_dims)
                return ArrayType(value_type.element, new_dims, new_expr)
            # Shape var could not be fully resolved; keep existing dims
            new_dims = tuple(
                substitute_index(dim, bindings) for dim in value_type.shape
            )
            return ArrayType(value_type.element, new_dims, new_expr)
        return ArrayType(
            value_type.element,
            tuple(substitute_index(dim, bindings) for dim in value_type.shape),
        )
    if isinstance(value_type, FuncType):
        return FuncType(
            tuple(substitute_type(param, bindings) for param in value_type.params),
            substitute_type(value_type.result, bindings),
        )
    if isinstance(value_type, SigmaType):
        shadowed = _drop_shadowed(bindings, value_type.hidden_names)
        return SigmaType(value_type.hidden_names, substitute_type(value_type.body, shadowed))
    if isinstance(value_type, PiType):
        shadowed = _drop_shadowed(bindings, tuple(binder.name for binder in value_type.binders))
        return PiType(value_type.binders, substitute_type(value_type.body, shadowed))
    if isinstance(value_type, ForallType):
        # Forall binders shadow element-type variables (TypeVars)
        shadowed = _drop_shadowed(bindings, tuple(binder.name for binder in value_type.binders))
        return ForallType(value_type.binders, substitute_type(value_type.body, shadowed))
    if isinstance(value_type, TypeVar):
        # TypeVars are not index variables; they are left alone by index substitution
        return value_type
    return value_type


def instantiate_pi_type(value_type: PiType, args: tuple[AnyIndexExpr, ...]) -> RemoraType:
    """Instantiate a Pi type with explicit index arguments."""
    if len(args) != len(value_type.binders):
        raise ValueError(
            f"Pi type expects {len(value_type.binders)} index argument(s), got {len(args)}"
        )
    bindings: dict[str, AnyIndexExpr] = {}
    for binder, arg in zip(value_type.binders, args):
        if binder.sort is IndexSort.DIM and arg.sort is not IndexSort.DIM:
            raise ValueError(f"index argument for {binder.name} must have sort Dim")
        if binder.sort is IndexSort.SHAPE and arg.sort is not IndexSort.SHAPE:
            raise ValueError(f"index argument for {binder.name} must have sort Shape")
        bindings[binder.name] = arg
    return substitute_type(value_type.body, bindings)


def free_type_index_vars(value_type: RemoraType) -> frozenset[str]:
    """Return free dimension and shape variables occurring in a type."""
    if isinstance(value_type, ArrayType):
        result: set[str] = set()
        for dim in value_type.shape:
            result.update(free_index_vars(dim))
        return frozenset(result)
    if isinstance(value_type, FuncType):
        result = set(free_type_index_vars(value_type.result))
        for param in value_type.params:
            result.update(free_type_index_vars(param))
        return frozenset(result)
    if isinstance(value_type, SigmaType):
        return free_type_index_vars(value_type.body) - frozenset(value_type.hidden_names)
    if isinstance(value_type, PiType):
        bound = frozenset(binder.name for binder in value_type.binders)
        return free_type_index_vars(value_type.body) - bound
    if isinstance(value_type, ForallType):
        return free_type_index_vars(value_type.body)
    if isinstance(value_type, TypeVar):
        return frozenset()
    return frozenset()


def _drop_shadowed(
    bindings: IndexSubstitution,
    names: tuple[str, ...],
) -> dict[str, AnyIndexExpr]:
    return {
        name: expr
        for name, expr in bindings.items()
        if name not in names
    }


def _has_index_var(dim_expr: AnyIndexExpr) -> bool:
    from remora.index import free_index_vars as _fiv
    return bool(_fiv(dim_expr))


# ── Forall / element-type variable helpers ────────────────────────────────


def substitute_element_types(
    value_type: RemoraType,
    bindings: dict[str, ScalarType],
) -> RemoraType:
    """Substitute ``TypeVar`` names with concrete ``ScalarType`` values."""
    if isinstance(value_type, ArrayType):
        return ArrayType(
            substitute_element_types(value_type.element, bindings),  # type: ignore[return-value]
            value_type.shape,
            value_type.shape_expr,
        )
    if isinstance(value_type, FuncType):
        return FuncType(
            tuple(substitute_element_types(p, bindings) for p in value_type.params),
            substitute_element_types(value_type.result, bindings),
        )
    if isinstance(value_type, SigmaType):
        return SigmaType(
            value_type.hidden_names,
            substitute_element_types(value_type.body, bindings),
        )
    if isinstance(value_type, PiType):
        return PiType(
            value_type.binders,
            substitute_element_types(value_type.body, bindings),
        )
    if isinstance(value_type, ForallType):
        shadowed = {
            name: ty
            for name, ty in bindings.items()
            if name not in {b.name for b in value_type.binders}
        }
        return ForallType(
            value_type.binders,
            substitute_element_types(value_type.body, shadowed),
        )
    if isinstance(value_type, TypeVar):
        if value_type.name in bindings:
            return bindings[value_type.name]
        return value_type
    return value_type


def instantiate_forall_type(
    value_type: ForallType,
    args: tuple[ScalarType, ...],
) -> RemoraType:
    """Instantiate a Forall type with concrete element-type arguments."""
    if len(args) != len(value_type.binders):
        raise ValueError(
            f"Forall expects {len(value_type.binders)} type argument(s), got {len(args)}"
        )
    bindings: dict[str, ScalarType] = {}
    for binder, arg in zip(value_type.binders, args):
        bindings[binder.name] = arg
    return substitute_element_types(value_type.body, bindings)


def free_type_vars(value_type: RemoraType) -> frozenset[str]:
    """Return free element-type variable names (TypeVar names) in a type."""
    if isinstance(value_type, ArrayType):
        result = set(free_type_vars(value_type.element))
        for dim in value_type.shape:
            pass  # shapes contain DimExpr, not TypeVars
        return frozenset(result)
    if isinstance(value_type, FuncType):
        result = set(free_type_vars(value_type.result))
        for param in value_type.params:
            result.update(free_type_vars(param))
        return frozenset(result)
    if isinstance(value_type, SigmaType):
        return free_type_vars(value_type.body)
    if isinstance(value_type, PiType):
        return free_type_vars(value_type.body)
    if isinstance(value_type, ForallType):
        bound = frozenset(binder.name for binder in value_type.binders)
        return free_type_vars(value_type.body) - bound
    if isinstance(value_type, TypeVar):
        return frozenset({value_type.name})
    return frozenset()
