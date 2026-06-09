"""Elaborate typed source programs into the structural typed core IR."""

from __future__ import annotations

from dataclasses import fields, is_dataclass

from remora.ast_nodes import AppExpr, FuncDef, MapExpr, VarExpr
from remora.elaborated import (
    CoreDefinition,
    CoreExpr,
    CoreIndexApplication,
    CoreProgram,
    CoreSpecialization,
    FrameCellDecision,
)
from remora.typechecker import (
    TypedDefinition,
    TypedExpr,
    TypedExprNode,
    TypedIndexApp,
    TypedLambda,
    TypedMap,
    TypedProgram,
)
from remora.types import FuncType


def elaborate_program(program: TypedProgram) -> CoreProgram:
    """Build a structural core and collect concrete dependent instances."""
    applications: list[CoreIndexApplication] = []
    specializations: dict[str, CoreSpecialization] = {}
    definitions = tuple(
        _elaborate_definition(definition, applications, specializations)
        for definition in program.definitions
    )
    body = (
        _elaborate_expr(program.body, applications, specializations)
        if program.body is not None
        else None
    )
    return CoreProgram(
        definitions,
        body,
        program.type,
        tuple(applications),
        tuple(specializations.values()),
    )


def _elaborate_definition(
    definition: TypedDefinition,
    applications: list[CoreIndexApplication],
    specializations: dict[str, CoreSpecialization],
) -> CoreDefinition:
    value = (
        _elaborate_expr(definition.value, applications, specializations)
        if definition.value is not None
        else None
    )
    return CoreDefinition(
        definition.definition,
        definition.type,
        value,
        definition,
    )


def _elaborate_expr(
    value: TypedExpr,
    applications: list[CoreIndexApplication],
    specializations: dict[str, CoreSpecialization],
) -> CoreExpr:
    children = tuple(
        _elaborate_expr(child, applications, specializations)
        for child in _typed_expr_children(value)
    )
    frame: FrameCellDecision | None = None
    if isinstance(value, TypedMap):
        # Derive the cell type from the function's first parameter
        cell_type = None
        if isinstance(value.func, TypedExprNode) and isinstance(value.func.type, FuncType):
            cell_type = value.func.type.params[0]
        elif isinstance(value.func, TypedLambda) and isinstance(value.func.type, FuncType):
            cell_type = value.func.type.params[0]
        frame = FrameCellDecision(
            value.frame_shape,
            value.cell_shape,
            cell_type=cell_type,
            is_implicit=isinstance(value.expr, AppExpr),
            is_binary=len(value.arrays) == 2,
            principal_frame=value.frame_shape if len(value.arrays) == 2 else None,
        )
    core = CoreExpr(type(value).__name__, value.type, children, value, frame)

    if isinstance(value, TypedIndexApp):
        function = value.expr.func
        if not isinstance(function, VarExpr):
            raise AssertionError("typed index application must target a named function")
        specialization_name = value.function.specialization_name
        if specialization_name is None:
            raise AssertionError("typed index application is missing specialization identity")
        applications.append(
            CoreIndexApplication(
                value.expr,
                function.name,
                specialization_name,
                value.index_args,
                value.type,
            )
        )

    if (
        isinstance(value, TypedLambda)
        and isinstance(value.expr, FuncDef)
        and value.specialization_name is not None
    ):
        if len(children) != 1:
            raise AssertionError("typed specialization must have one body child")
        body = children[0]
        specialization = CoreSpecialization(
            value.specialization_name,
            value.expr.name,
            value.index_args,
            tuple(value.params),
            body,
            value.type,
        )
        existing = specializations.get(specialization.name)
        if existing is not None and existing != specialization:
            raise AssertionError(f"conflicting specialization {specialization.name}")
        specializations[specialization.name] = specialization

    return core


def _typed_expr_children(value: TypedExpr) -> list[TypedExpr]:
    result: list[TypedExpr] = []
    if not is_dataclass(value):
        return result
    for field in fields(value):
        if field.name in {
            "expr",
            "type",
            "from_type",
            "to_type",
            "params",
            "index_args",
            "specialization_name",
            "hidden_names",
            "name",
            "value_name",
            "frame_shape",
            "cell_shape",
            "reduction_dim",
            "offsets",
            "sizes",
            "shift",
        }:
            continue
        _append_typed_children(getattr(value, field.name), result)
    return result


def _append_typed_children(value: object, result: list[TypedExpr]) -> None:
    if _is_typed_expr(value):
        result.append(value)
        return
    if isinstance(value, (list, tuple)):
        for item in value:
            if _is_typed_expr(item):
                result.append(item)


def _is_typed_expr(value: object) -> bool:
    cls = type(value)
    return (
        cls.__module__ == "remora.typechecker"
        and cls.__name__.startswith("Typed")
        and cls.__name__ not in {"TypedProgram", "TypedDefinition"}
        and hasattr(value, "type")
    )
