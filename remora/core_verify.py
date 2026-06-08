"""Verifier for the structural typed elaborated core."""

from __future__ import annotations

from remora.ast_nodes import FuncDef
from remora.dependent_types import free_type_index_vars
from remora.elaborated import CoreExpr, CoreProgram
from remora.errors import RemoraError
from remora.index import DimExpr, ShapeLit
from remora.types import ArrayType, ForallType, FuncType, PiType, StaticDim


class CoreVerificationError(RemoraError):
    """Raised when typed core invariants are violated."""


def _is_concrete_index(arg) -> bool:
    if isinstance(arg, StaticDim):
        return True
    if isinstance(arg, DimExpr):
        value = getattr(arg, "value", None)
        return isinstance(value, int) and value >= 0
    if isinstance(arg, ShapeLit):
        return all(_is_concrete_index(d) for d in arg.dims)
    return False


def verify_core_program(program: CoreProgram) -> None:
    if program.body is None:
        if program.type is not None:
            raise CoreVerificationError("definition-only core program has a result type")
    else:
        if program.body.type != program.type:
            raise CoreVerificationError(
                f"core body type {program.body.type} does not match program type {program.type}"
            )
        _verify_expr(program.body, require_concrete=True)

    for definition in program.definitions:
        if definition.value is None:
            if isinstance(definition.source, FuncDef) and (
                definition.type is None
                or isinstance(definition.type, (FuncType, PiType, ForallType))
            ):
                continue
            if definition.type is not None:
                raise CoreVerificationError(
                    f"function definition {definition.source.name} "
                    "has an unexpected value type"
                )
            continue
        if definition.type != definition.value.type:
            raise CoreVerificationError(
                f"definition {definition.source.name} type {definition.type} "
                f"does not match value type {definition.value.type}"
            )
        _verify_expr(definition.value, require_concrete=True)

    names: set[str] = set()
    specialization_keys: set[tuple[str, tuple[StaticDim, ...], FuncType]] = set()
    for specialization in program.specializations:
        if specialization.name in names:
            raise CoreVerificationError(
                f"duplicate core specialization {specialization.name}"
            )
        names.add(specialization.name)
        if any(not _is_concrete_index(arg) for arg in specialization.index_args):
            raise CoreVerificationError(
                f"specialization {specialization.name} is not concrete"
            )
        _require_concrete_type(
            specialization.type,
            f"specialization {specialization.name}",
        )
        if specialization.body.type != specialization.type.result:
            raise CoreVerificationError(
                f"specialization {specialization.name} body type "
                f"{specialization.body.type} does not match {specialization.type.result}"
            )
        _verify_expr(specialization.body, require_concrete=True)
        specialization_keys.add(
            (
                specialization.function_name,
                specialization.index_args,
                specialization.type,
            )
        )

    for application in program.index_applications:
        if any(not _is_concrete_index(arg) for arg in application.index_args):
            raise CoreVerificationError(
                f"index application of {application.function_name} is not concrete"
            )
        _require_concrete_type(
            application.type,
            f"index application of {application.function_name}",
        )
        application_key = (
            application.function_name,
            application.index_args,
            application.type,
        )
        if application_key not in specialization_keys:
            raise CoreVerificationError(
                f"index application of {application.function_name} has no "
                "matching core specialization"
            )
        if application.specialization_name not in names:
            raise CoreVerificationError(
                f"index application names unknown specialization "
                f"{application.specialization_name}"
            )


def _verify_expr(expr: CoreExpr, *, require_concrete: bool) -> None:
    if expr.type != expr.typed.type:
        raise CoreVerificationError(
            f"{expr.kind} core type {expr.type} does not match typed node {expr.typed.type}"
        )
    if require_concrete:
        _require_concrete_type(expr.type, expr.kind)
    if expr.frame is not None:
        _verify_frame_decision(expr.frame, expr)
    for child in expr.children:
        _verify_expr(child, require_concrete=require_concrete)


def _verify_frame_decision(frame, expr: CoreExpr) -> None:
    from remora.index import free_index_vars as _free_iv

    if frame.frame_rank + frame.cell_rank > 0 and not isinstance(expr.type, ArrayType):
        pass  # result type may be a scalar or function, not always an array
    if frame.frame_shape:
        for dim in frame.frame_shape:
            _check_dim_concrete(dim, f"frame dimension in {expr.kind}", _free_iv)
    if frame.cell_shape:
        for dim in frame.cell_shape:
            _check_dim_concrete(dim, f"cell dimension in {expr.kind}", _free_iv)


def _check_dim_concrete(value, context: str, free_iv) -> None:
    free = free_iv(value)
    if free:
        variables = ", ".join(sorted(free))
        raise CoreVerificationError(
            f"{context} has unspecialized index variables: {variables}"
        )


def _require_concrete_dim(value, context: str) -> None:
    from remora.index import free_index_vars as _free_iv

    _check_dim_concrete(value, context, _free_iv)


def _require_concrete_type(value_type, context: str) -> None:
    free = free_type_index_vars(value_type)
    if free:
        variables = ", ".join(sorted(free))
        raise CoreVerificationError(
            f"{context} has unspecialized index variables: {variables}"
        )
