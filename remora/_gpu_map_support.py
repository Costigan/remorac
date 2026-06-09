"""Shared support for the current narrow GPU map slice."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from remora.hir import (
    HIRExpr,
    HIRFunction,
    HIRIf,
    HIRLambda,
    HIRLit,
    HIRMap,
    HIRPrimCallable,
    HIRPrimOp,
    HIRVar,
)
from remora.types import BOOL, FLOAT, INT, ArrayType, ScalarType


@dataclass(frozen=True)
class F32MapOperation:
    op: str
    constant: float | None = None
    constant_side: str | None = None


@dataclass(frozen=True)
class F32InputExpr:
    index: int


@dataclass(frozen=True)
class F32ConstantExpr:
    value: float


@dataclass(frozen=True)
class F32BinaryExpr:
    op: str
    left: "F32Expr"
    right: "F32Expr"


@dataclass(frozen=True)
class F32SelectExpr:
    condition: "F32Expr"
    then_expr: "F32Expr"
    else_expr: "F32Expr"


@dataclass(frozen=True)
class F32CmpExpr:
    op: str
    left: "F32Expr"
    right: "F32Expr"


F32Expr = F32InputExpr | F32ConstantExpr | F32BinaryExpr | F32SelectExpr | F32CmpExpr


@dataclass(frozen=True)
class F32MapKernel:
    shape: tuple[int, ...]
    operation: F32MapOperation
    num_inputs: int
    expression: F32Expr | None = None


@dataclass(frozen=True)
class I32MapOperation:
    op: str
    constant: int | None = None
    constant_side: str | None = None


@dataclass(frozen=True)
class I32MapKernel:
    shape: tuple[int, ...]
    operation: I32MapOperation
    num_inputs: int


def analyze_supported_map_function(
    function: HIRFunction,
    *,
    on_unsupported: Callable[[str], Exception],
    context: str,
    element_type: ScalarType,
) -> F32MapKernel | I32MapKernel:
    type_name = "float" if element_type == FLOAT else "int" if element_type == INT else "bool"

    if len(function.params) not in (1, 2):
        raise on_unsupported(f"{context} currently supports one or two input parameters")

    input_types: list[ArrayType] = []
    for param in function.params:
        if not (
            isinstance(param.type, ArrayType)
            and param.type.element == element_type
            and 1 <= param.type.rank <= 10
        ):
            raise on_unsupported(f"{context} currently supports rank-1 through rank-10 {type_name} inputs only")
        input_types.append(param.type)

    if not (
        isinstance(function.return_type, ArrayType)
        and function.return_type.element == element_type
        and 1 <= function.return_type.rank <= 10
    ):
        raise on_unsupported(f"{context} currently supports rank-1 through rank-10 {type_name} outputs only")

    if any(input_type.shape != function.return_type.shape for input_type in input_types):
        raise on_unsupported(f"{context} input and output shapes must match")

    if not (
        isinstance(function.body, HIRMap)
        and len(function.body.arrays) == len(function.params)
        and all(isinstance(array, HIRVar) for array in function.body.arrays)
        and [array.name for array in function.body.arrays] == [param.name for param in function.params]
        and isinstance(function.body.func, HIRPrimCallable)
    ):
        raise on_unsupported(f"{context} currently supports primitive maps over function parameters only")

    callable_ = function.body.func
    if element_type == FLOAT:
        allowed_ops = {"+", "-", "*", "/"}
    elif element_type == INT:
        allowed_ops = {"+", "-", "*", "/"}
    else: # BOOL
        allowed_ops = {"&&", "||", "==", "!="}
        
    if callable_.op not in allowed_ops:
         raise on_unsupported(f"{context} does not support operator {callable_.op} for {type_name}")

    if len(function.params) == 1:
        if isinstance(callable_.left_arg, HIRLit) and callable_.left_arg.type == element_type:
            operation: F32MapOperation | I32MapOperation = F32MapOperation(callable_.op, callable_.left_arg.value, "left") if element_type == FLOAT else I32MapOperation(callable_.op, int(callable_.left_arg.value), "left")
        elif isinstance(callable_.right_arg, HIRLit) and callable_.right_arg.type == element_type:
            operation = F32MapOperation(callable_.op, callable_.right_arg.value, "right") if element_type == FLOAT else I32MapOperation(callable_.op, int(callable_.right_arg.value), "right")
        else:
            raise on_unsupported(f"{context} unary map requires a literal {type_name} section")
    elif callable_.left_arg is None and callable_.right_arg is None:
        operation = F32MapOperation(callable_.op) if element_type == FLOAT else I32MapOperation(callable_.op)
    else:
        raise on_unsupported(f"{context} binary map does not support operator sections")

    KernelClass = F32MapKernel if element_type == FLOAT else I32MapKernel
    return KernelClass(
        tuple(dim.value for dim in function.return_type.shape),
        operation,
        len(function.params),
    )


def analyze_supported_f32_map_function(
    function: HIRFunction,
    *,
    on_unsupported: Callable[[str], Exception],
    context: str,
) -> F32MapKernel:
    try:
        simple = analyze_supported_map_function(
            function,
            on_unsupported=on_unsupported,
            context=context,
            element_type=FLOAT,
        )
        assert isinstance(simple, F32MapKernel)
        return simple
    except Exception as simple_error:
        try:
            return _analyze_fused_f32_map(function, on_unsupported, context)
        except Exception:
            raise simple_error


def _simple_f32_expression(kernel: F32MapKernel) -> F32Expr:
    operation = kernel.operation
    if kernel.num_inputs == 2:
        return F32BinaryExpr(operation.op, F32InputExpr(0), F32InputExpr(1))
    assert operation.constant is not None
    constant = F32ConstantExpr(float(operation.constant))
    value = F32InputExpr(0)
    if operation.constant_side == "left":
        return F32BinaryExpr(operation.op, constant, value)
    return F32BinaryExpr(operation.op, value, constant)


def _analyze_fused_f32_map(
    function: HIRFunction,
    on_unsupported: Callable[[str], Exception],
    context: str,
) -> F32MapKernel:
    input_types = _require_scalar_array_params(function, FLOAT, context, on_unsupported)
    if len(input_types) not in (1, 2):
        raise on_unsupported(f"{context} currently supports one or two input parameters")
    if not isinstance(function.return_type, ArrayType) or function.return_type.element != FLOAT:
        raise on_unsupported(f"{context} currently supports float array outputs only")
    if any(input_type.shape != function.return_type.shape for input_type in input_types):
        raise on_unsupported(f"{context} input and output shapes must match")
    param_indices = {param.name: index for index, param in enumerate(function.params)}
    expression = _f32_expr_from_array(function.body, param_indices, {}, on_unsupported, context)
    root = expression if isinstance(expression, (F32BinaryExpr, F32SelectExpr, F32CmpExpr)) else None
    if root is None:
        raise on_unsupported(f"{context} fused map result must be arithmetic")
    root_op = root.op if isinstance(root, F32BinaryExpr) else "select"
    return F32MapKernel(
        tuple(dim.value for dim in function.return_type.shape),
        F32MapOperation(root_op),
        len(function.params),
        expression,
    )


def _f32_expr_from_array(
    expr: HIRExpr,
    param_indices: dict[str, int],
    scalar_env: dict[str, F32Expr],
    on_unsupported: Callable[[str], Exception],
    context: str,
) -> F32Expr:
    if isinstance(expr, HIRVar) and expr.name in param_indices:
        return F32InputExpr(param_indices[expr.name])
    if isinstance(expr, HIRIf):
        return F32SelectExpr(
            _f32_expr_from_array(expr.condition, param_indices, scalar_env, on_unsupported, context),
            _f32_expr_from_array(expr.then_branch, param_indices, scalar_env, on_unsupported, context),
            _f32_expr_from_array(expr.else_branch, param_indices, scalar_env, on_unsupported, context),
        )
    if not isinstance(expr, HIRMap):
        raise on_unsupported(f"{context} fused maps require parameter or map operands")
    array_exprs = [
        _f32_expr_from_array(array, param_indices, scalar_env, on_unsupported, context)
        for array in expr.arrays
    ]
    if isinstance(expr.func, HIRPrimCallable):
        op = expr.func.op
        if op not in {"+", "-", "*", "/"}:
            raise on_unsupported(f"{context} does not support fused operator {op}")
        if len(array_exprs) == 2:
            if expr.func.left_arg is not None or expr.func.right_arg is not None:
                raise on_unsupported(
                    f"{context} binary map does not support operator sections"
                )
            return F32BinaryExpr(op, array_exprs[0], array_exprs[1])
        if len(array_exprs) == 1:
            if isinstance(expr.func.left_arg, HIRLit):
                return F32BinaryExpr(op, F32ConstantExpr(float(expr.func.left_arg.value)), array_exprs[0])
            if isinstance(expr.func.right_arg, HIRLit):
                return F32BinaryExpr(op, array_exprs[0], F32ConstantExpr(float(expr.func.right_arg.value)))
    if isinstance(expr.func, HIRLambda) and len(expr.func.params) == len(array_exprs):
        env = dict(scalar_env)
        env.update({param.name: value for param, value in zip(expr.func.params, array_exprs)})
        return _f32_expr_from_scalar(expr.func.body, env, on_unsupported, context)
    raise on_unsupported(f"{context} fused map callable is not supported")


def _f32_expr_from_scalar(
    expr: HIRExpr,
    env: dict[str, F32Expr],
    on_unsupported: Callable[[str], Exception],
    context: str,
) -> F32Expr:
    if isinstance(expr, HIRVar) and expr.name in env:
        return env[expr.name]
    if isinstance(expr, HIRLit) and expr.type == FLOAT:
        return F32ConstantExpr(float(expr.value))
    if isinstance(expr, HIRPrimOp) and len(expr.args) == 2:
        op = expr.op
        for suffix in ("f", "b", "i"):
            if op.endswith(suffix):
                op = op[:-1]
                break
        if op in {"+", "-", "*", "/"}:
            return F32BinaryExpr(
                op,
                _f32_expr_from_scalar(expr.args[0], env, on_unsupported, context),
                _f32_expr_from_scalar(expr.args[1], env, on_unsupported, context),
            )
        if op in {">", "<", ">=", "<=", "==", "!="}:
            return F32CmpExpr(
                op,
                _f32_expr_from_scalar(expr.args[0], env, on_unsupported, context),
                _f32_expr_from_scalar(expr.args[1], env, on_unsupported, context),
            )
        raise on_unsupported(f"{context} does not support fused scalar operator {expr.op}")
    if isinstance(expr, HIRIf):
        return F32SelectExpr(
            _f32_expr_from_scalar(expr.condition, env, on_unsupported, context),
            _f32_expr_from_scalar(expr.then_branch, env, on_unsupported, context),
            _f32_expr_from_scalar(expr.else_branch, env, on_unsupported, context),
        )
    raise on_unsupported(f"{context} fused scalar expression is not supported")


def analyze_supported_i32_map_function(
    function: HIRFunction,
    *,
    on_unsupported: Callable[[str], Exception],
    context: str,
) -> I32MapKernel:
    return analyze_supported_map_function(
        function,
        on_unsupported=on_unsupported,
        context=context,
        element_type=INT,
    ) # type: ignore


def analyze_supported_bool_map_function(
    function: HIRFunction,
    *,
    on_unsupported: Callable[[str], Exception],
    context: str,
) -> I32MapKernel:
    return analyze_supported_map_function(
        function,
        on_unsupported=on_unsupported,
        context=context,
        element_type=BOOL,
    ) # type: ignore


def _require_scalar_array_params(
    function: HIRFunction,
    element_type: ScalarType,
    context: str,
    on_unsupported: Callable[[str], Exception],
) -> list[ArrayType]:
    input_types: list[ArrayType] = []
    type_name = "float" if element_type == FLOAT else "int" if element_type == INT else "bool"
    for param in function.params:
        if not (
            isinstance(param.type, ArrayType)
            and param.type.element == element_type
            and 1 <= param.type.rank <= 10
        ):
            raise on_unsupported(
                f"{context} currently supports rank-1 through rank-10 {type_name} inputs only"
            )
        input_types.append(param.type)
    return input_types
