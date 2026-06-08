"""Verifier for the typed elaborated core."""

from __future__ import annotations

from remora.ast_nodes import FuncDef
from remora.elaborated import CoreProgram
from remora.errors import RemoraError
from remora.dependent_types import free_type_index_vars
from remora.types import StaticDim
from remora.types import FuncType, PiType


class CoreVerificationError(RemoraError):
    """Raised when typed core invariants are violated."""


def verify_core_program(program: CoreProgram) -> None:
    typed = program.typed
    if typed.body is None:
        if typed.type is not None:
            raise CoreVerificationError("definition-only core program has a result type")
    elif typed.body.type != typed.type:
        raise CoreVerificationError(
            f"core body type {typed.body.type} does not match program type {typed.type}"
        )

    for definition in typed.definitions:
        if definition.value is None:
            if isinstance(definition.definition, FuncDef) and (
                definition.type is None or isinstance(definition.type, (FuncType, PiType))
            ):
                continue
            if definition.type is not None:
                raise CoreVerificationError(
                    f"function definition {definition.definition.name} has an unexpected value type"
                )
            continue
        if definition.type != definition.value.type:
            raise CoreVerificationError(
                f"definition {definition.definition.name} type {definition.type} "
                f"does not match value type {definition.value.type}"
            )

    for application in program.index_applications:
        if any(not isinstance(arg, StaticDim) for arg in application.index_args):
            raise CoreVerificationError(
                f"index application of {application.function_name} is not concrete"
            )
        free = free_type_index_vars(application.type)
        if free:
            names = ", ".join(sorted(free))
            raise CoreVerificationError(
                f"index application of {application.function_name} has "
                f"unspecialized index variables: {names}"
            )
