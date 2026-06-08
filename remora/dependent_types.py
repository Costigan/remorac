"""Helpers for dependent Remora types."""

from __future__ import annotations

from remora.index import (
    AnyIndexExpr,
    IndexSort,
    IndexSubstitution,
    free_index_vars,
    substitute_index,
)
from remora.types import ArrayType, FuncType, PiType, RemoraType, SigmaType


def substitute_type(value_type: RemoraType, bindings: IndexSubstitution) -> RemoraType:
    """Substitute index variables inside a Remora type."""
    if isinstance(value_type, ArrayType):
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
