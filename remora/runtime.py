"""CPU runtime helpers for Remora Dense Core."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from remora.ast_nodes import BoolLit, FloatLit, FuncDef, IfExpr, IntLit, IotaExpr, VarExpr
from remora.display import format_result
from remora.errors import RemoraError
from remora.parser import parse_program
from remora.prelude import with_prelude
from remora.typechecker import (
    TypeChecker,
    TypedApp,
    TypedArray,
    TypedCast,
    TypedDefinition,
    TypedExpr,
    TypedExprNode,
    TypedFold,
    TypedIf,
    TypedIndex,
    TypedLambda,
    TypedLeftSection,
    TypedLet,
    TypedMap,
    TypedOperatorFunc,
    TypedProgram,
    TypedRank,
    TypedRightSection,
    TypedShape,
)
from remora.types import ArrayType, BOOL, FLOAT, INT, RemoraType, ScalarType, StaticDim


class RuntimeUnavailable(RemoraError):
    """Raised when a requested runtime target is unavailable."""


class EvaluationError(RemoraError):
    """Raised when CPU evaluation hits unsupported typed syntax."""


@dataclass(frozen=True)
class EvaluationResult:
    value: object
    type: RemoraType


Value = object
Env = dict[str, Value]
CallableValue = Callable[..., Value]


def evaluate_source(source: str, *, include_prelude: bool = True) -> EvaluationResult:
    program_source = with_prelude(source) if include_prelude else source
    typed = TypeChecker().check_program(parse_program(program_source))
    return evaluate_typed_program(typed)


def evaluate_typed_program(program: TypedProgram) -> EvaluationResult:
    if program.body is None or program.type is None:
        raise EvaluationError("definition-only programs cannot be evaluated")

    env: Env = {}
    for definition in program.definitions:
        _bind_definition(definition, env)
    return EvaluationResult(_eval_expr(program.body, env), program.type)


def format_value(value: object) -> str:
    if isinstance(value, np.ndarray):
        if value.dtype == np.bool_:
            return format_result(value, ArrayType(BOOL, tuple(StaticDim(size) for size in value.shape)))
        if np.issubdtype(value.dtype, np.integer):
            return format_result(value, ArrayType(INT, tuple(StaticDim(size) for size in value.shape)))
        if np.issubdtype(value.dtype, np.floating):
            return format_result(value, ArrayType(FLOAT, tuple(StaticDim(size) for size in value.shape)))
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, bool):
        return format_result(value, BOOL)
    if isinstance(value, int):
        return format_result(value, INT)
    if isinstance(value, float):
        return format_result(value, FLOAT)
    return repr(value)


def _bind_definition(definition: TypedDefinition, env: Env) -> None:
    if isinstance(definition.definition, FuncDef):
        return
    if definition.value is None:
        raise EvaluationError("function definitions are deferred")
    env[definition.definition.name] = _eval_expr(definition.value, env)


def _eval_expr(expr: TypedExpr, env: Env) -> Value:
    if isinstance(expr, TypedCast):
        return _cast_scalar(_eval_expr(expr.value, env), expr.to_type)

    if isinstance(expr, TypedArray):
        values = [_eval_expr(element, env) for element in expr.elements]
        return np.array(values, dtype=_numpy_dtype(expr.type.element))

    if isinstance(expr, TypedMap):
        arrays = [_eval_expr(array, env) for array in expr.arrays]
        callable_value = _eval_callable(expr.func, env)
        return _map_value(callable_value, arrays, expr.cell_shape, expr.type)

    if isinstance(expr, TypedFold):
        array = _eval_expr(expr.array, env)
        acc = _eval_expr(expr.init, env)
        callable_value = _eval_callable(expr.func, env)
        if not isinstance(array, np.ndarray):
            raise EvaluationError("fold expects an array value")
        for item in array:
            acc = callable_value(acc, item)
        return _coerce_runtime_value(acc, expr.type)

    if isinstance(expr, TypedShape):
        return np.array(
            [dim.value for dim in _shape_dims(expr.array.type)],
            dtype=np.int32,
        )

    if isinstance(expr, TypedRank):
        return int(expr.array.type.rank)

    if isinstance(expr, TypedIndex):
        array = _eval_expr(expr.array, env)
        indices = tuple(int(_eval_expr(index, env)) for index in expr.indices)
        if not isinstance(array, np.ndarray):
            raise EvaluationError("indexing expects an array value")
        value = array[indices]
        return _coerce_runtime_value(value, expr.type)

    if isinstance(expr, TypedLambda):
        return _lambda_callable(expr, env)

    if isinstance(expr, TypedApp):
        if isinstance(expr.func, TypedExprNode) and isinstance(expr.func.expr, VarExpr):
            if expr.func.expr.name in _OPS:
                args = [_eval_expr(arg, env) for arg in expr.args]
                return _coerce_runtime_value(_apply_op(expr.func.expr.name, *args), expr.type)
        func = _eval_expr(expr.func, env)
        if not callable(func):
            raise EvaluationError("application target is not callable")
        args = [_eval_expr(arg, env) for arg in expr.args]
        return _coerce_runtime_value(func(*args), expr.type)

    if isinstance(expr, TypedLet):
        inner_env = dict(env)
        inner_env[expr.name] = _eval_expr(expr.value, env)
        return _eval_expr(expr.body, inner_env)

    if isinstance(expr, TypedIf):
        condition = _eval_expr(expr.condition, env)
        return _eval_expr(expr.then_branch if bool(condition) else expr.else_branch, env)

    if isinstance(expr, (TypedOperatorFunc, TypedLeftSection, TypedRightSection)):
        return _eval_callable(expr, env)

    if isinstance(expr, TypedExprNode):
        return _eval_expr_node(expr, env)

    raise EvaluationError(f"CPU evaluation for {type(expr).__name__} is deferred")


def _eval_expr_node(expr: TypedExprNode, env: Env) -> Value:
    ast = expr.expr
    if isinstance(ast, IntLit):
        return int(ast.value)
    if isinstance(ast, FloatLit):
        return float(ast.value)
    if isinstance(ast, BoolLit):
        return bool(ast.value)
    if isinstance(ast, VarExpr):
        try:
            return env[ast.name]
        except KeyError as exc:
            raise EvaluationError(f"unbound runtime variable '{ast.name}'") from exc
    if isinstance(ast, IotaExpr):
        if not isinstance(expr.type, ArrayType):
            raise EvaluationError("iota must have an array type")
        size = expr.type.shape[0]
        if not isinstance(size, StaticDim):
            raise EvaluationError("iota requires a static dimension")
        return np.arange(size.value, dtype=np.int32)
    raise EvaluationError(f"CPU evaluation for {type(ast).__name__} is deferred")


def _eval_callable(expr: TypedExpr, env: Env) -> CallableValue:
    if isinstance(expr, TypedLambda):
        return _lambda_callable(expr, env)
    if isinstance(expr, TypedOperatorFunc):
        return lambda left, right: _coerce_runtime_value(
            _apply_op(expr.expr.op, left, right), expr.type.result
        )
    if isinstance(expr, TypedLeftSection):
        left = _eval_expr(expr.arg, env)
        return lambda right: _coerce_runtime_value(
            _apply_op(expr.expr.op, left, right), expr.type.result
        )
    if isinstance(expr, TypedRightSection):
        right = _eval_expr(expr.arg, env)
        return lambda left: _coerce_runtime_value(
            _apply_op(expr.expr.op, left, right), expr.type.result
        )
    value = _eval_expr(expr, env)
    if not callable(value):
        raise EvaluationError("expected a callable value")
    return value


def _lambda_callable(expr: TypedLambda, env: Env) -> CallableValue:
    closed_env = dict(env)

    def call(*args: Value) -> Value:
        if len(args) != len(expr.params):
            raise EvaluationError("lambda arity mismatch")
        inner_env = dict(closed_env)
        for (name, _param_type), arg in zip(expr.params, args):
            inner_env[name] = arg
        return _eval_expr(expr.body, inner_env)

    return call


def _map_value(
    callable_value: CallableValue,
    arrays: list[Value],
    cell_shape: tuple[StaticDim, ...],
    result_type: RemoraType,
) -> Value:
    if len(arrays) == 2:
        return _binary_map_value(callable_value, arrays[0], arrays[1], result_type)
    if len(arrays) != 1:
        raise EvaluationError("map currently supports one or two arrays")
    array = arrays[0]
    if not isinstance(array, np.ndarray):
        return _coerce_runtime_value(callable_value(array), result_type)

    cell_rank = len(cell_shape)
    frame_rank = array.ndim - cell_rank
    if frame_rank < 0:
        raise EvaluationError("map cell rank exceeds array rank")

    if frame_rank == 0:
        return _coerce_runtime_value(callable_value(array), result_type)

    frame_shape = array.shape[:frame_rank]
    values = [
        callable_value(array[index] if cell_rank else array[index].item())
        for index in np.ndindex(frame_shape)
    ]
    if isinstance(result_type, ArrayType):
        return np.array(values, dtype=_numpy_dtype(result_type.element)).reshape(
            tuple(dim.value for dim in result_type.shape)
        )
    return _coerce_runtime_value(values[0], result_type)


def _binary_map_value(
    callable_value: CallableValue,
    left: Value,
    right: Value,
    result_type: RemoraType,
) -> Value:
    if not isinstance(left, np.ndarray) and not isinstance(right, np.ndarray):
        return _coerce_runtime_value(callable_value(left, right), result_type)
    if not isinstance(left, np.ndarray) or not isinstance(right, np.ndarray):
        raise EvaluationError("binary map currently expects both operands to be arrays or scalars")
    if left.shape != right.shape:
        raise EvaluationError("binary map expects matching array shapes")
    values = [
        callable_value(left[index].item(), right[index].item())
        for index in np.ndindex(left.shape)
    ]
    if isinstance(result_type, ArrayType):
        return np.array(values, dtype=_numpy_dtype(result_type.element)).reshape(
            tuple(dim.value for dim in result_type.shape)
        )
    return _coerce_runtime_value(values[0], result_type)


def _apply_op(op: str, left: Value, right: Value) -> Value:
    if op == "+":
        return left + right
    if op == "-":
        return left - right
    if op == "*":
        return left * right
    if op == "/":
        return float(left) / float(right)
    if op == "<":
        return left < right
    if op == "<=":
        return left <= right
    if op == "==":
        return left == right
    if op == "!=":
        return left != right
    if op == "&&":
        return bool(left) and bool(right)
    if op == "||":
        return bool(left) or bool(right)
    raise EvaluationError(f"operator {op} is not supported by CPU evaluation")


def _coerce_runtime_value(value: Value, value_type: RemoraType) -> Value:
    if isinstance(value_type, ScalarType):
        return _cast_scalar(value, value_type)
    if isinstance(value_type, ArrayType):
        return np.asarray(value, dtype=_numpy_dtype(value_type.element)).reshape(
            tuple(dim.value for dim in value_type.shape)
        )
    return value


def _shape_dims(value_type: RemoraType) -> tuple[StaticDim, ...]:
    if isinstance(value_type, ArrayType):
        return value_type.shape
    if isinstance(value_type, ScalarType):
        return ()
    raise EvaluationError("shape/rank of function values is deferred")


def _cast_scalar(value: Value, value_type: ScalarType) -> Value:
    if isinstance(value, np.generic):
        value = value.item()
    if value_type == INT:
        return int(value)
    if value_type == FLOAT:
        return float(value)
    if value_type == BOOL:
        return bool(value)
    raise EvaluationError(f"cannot cast runtime value to {value_type}")


def _numpy_dtype(element_type: ScalarType) -> np.dtype:
    if element_type == INT:
        return np.dtype(np.int32)
    if element_type == FLOAT:
        return np.dtype(np.float32)
    if element_type == BOOL:
        return np.dtype(np.bool_)
    raise EvaluationError(f"unsupported array element type {element_type}")


_OPS = {"+", "-", "*", "/", "<", "<=", "==", "!=", "&&", "||"}
