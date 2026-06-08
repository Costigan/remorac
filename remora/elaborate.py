"""Elaborate typed source programs into the typed core IR."""

from __future__ import annotations

from dataclasses import fields, is_dataclass

from remora.ast_nodes import VarExpr
from remora.elaborated import CoreIndexApplication, CoreProgram
from remora.typechecker import TypedIndexApp, TypedProgram


def elaborate_program(program: TypedProgram) -> CoreProgram:
    """Create the initial typed core representation for a checked program."""
    applications: list[CoreIndexApplication] = []
    _collect_index_applications(program, applications)
    return CoreProgram(program, tuple(applications))


def _collect_index_applications(
    value: object,
    applications: list[CoreIndexApplication],
) -> None:
    if isinstance(value, TypedIndexApp):
        function = value.expr.func
        if not isinstance(function, VarExpr):
            raise AssertionError("typed index application must target a named function")
        applications.append(
            CoreIndexApplication(
                value.expr,
                function.name,
                value.index_args,
                value.type,
            )
        )
    if isinstance(value, (list, tuple)):
        for item in value:
            _collect_index_applications(item, applications)
        return
    if is_dataclass(value):
        for field in fields(value):
            _collect_index_applications(getattr(value, field.name), applications)
