"""High-level IR for typed Remora Dense Core programs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TypeAlias

from remora.ast_nodes import (
    BoolLit,
    FloatLit,
    IntLit,
    IotaExpr,
    VarExpr,
)
from remora.errors import RemoraError
from remora.typechecker import (
    TypedApp,
    TypedArray,
    TypedCast,
    TypedDefinition,
    TypedExpr,
    TypedExprNode,
    TypedFold,
    TypedLambda,
    TypedLeftSection,
    TypedLet,
    TypedMap,
    TypedOperatorFunc,
    TypedProgram,
    TypedRightSection,
)
from remora.types import (
    BOOL,
    FLOAT,
    INT,
    ArrayType,
    DimExpr,
    FuncType,
    RemoraType,
    ScalarType,
    StaticDim,
    eval_static_dim,
)


@dataclass(frozen=True)
class HIRProgram:
    functions: list[HIRFunction]
    main: HIRExpr | None
    return_type: RemoraType | None


@dataclass(frozen=True)
class HIRFunction:
    name: str
    params: list[HIRParam]
    body: HIRExpr
    return_type: RemoraType


@dataclass(frozen=True)
class HIRParam:
    name: str
    type: RemoraType


@dataclass(frozen=True)
class HIRMap:
    frame_shape: tuple[DimExpr, ...]
    cell_shape: tuple[DimExpr, ...]
    func: HIRCallable
    array: HIRExpr
    result_type: RemoraType


@dataclass(frozen=True)
class HIRFold:
    reduction_dim: DimExpr
    func: HIRCallable
    init: HIRExpr
    array: HIRExpr
    result_type: RemoraType


@dataclass(frozen=True)
class HIRLet:
    name: str
    value_type: RemoraType
    value: HIRExpr
    body: HIRExpr
    result_type: RemoraType


@dataclass(frozen=True)
class HIRCall:
    func_name: str
    args: list[HIRExpr]
    result_type: RemoraType


@dataclass(frozen=True)
class HIRLambda:
    params: list[HIRParam]
    body: HIRExpr
    result_type: FuncType


@dataclass(frozen=True)
class HIRPrimCallable:
    op: str
    params: tuple[RemoraType, ...]
    result_type: RemoraType
    left_arg: HIRExpr | None = None
    right_arg: HIRExpr | None = None


@dataclass(frozen=True)
class HIRPrimOp:
    op: str
    args: list[HIRExpr]
    result_type: RemoraType


@dataclass(frozen=True)
class HIRIota:
    size: StaticDim
    result_type: ArrayType


@dataclass(frozen=True)
class HIRCast:
    value: HIRExpr
    from_type: ScalarType
    to_type: ScalarType
    result_type: ScalarType


@dataclass(frozen=True)
class HIRArrayLit:
    elements: list[HIRExpr]
    result_type: ArrayType


@dataclass(frozen=True)
class HIRVar:
    name: str
    type: RemoraType


@dataclass(frozen=True)
class HIRLit:
    value: int | float | bool
    type: ScalarType


HIRExpr: TypeAlias = (
    HIRMap
    | HIRFold
    | HIRLet
    | HIRCall
    | HIRLambda
    | HIRPrimOp
    | HIRIota
    | HIRCast
    | HIRArrayLit
    | HIRVar
    | HIRLit
)

HIRCallable: TypeAlias = HIRLambda | HIRPrimCallable | HIRVar


class HIRLoweringError(RemoraError):
    """Raised when typed AST to HIR lowering hits deferred syntax."""


def lower_to_hir(program: TypedProgram) -> HIRProgram:
    main = program.body
    if main is None:
        if program.definitions:
            raise HIRLoweringError(
                "definition-only programs cannot be lowered to HIR without a body"
            )
        return HIRProgram([], None, None)

    lowered_main = lower_expr(main)
    for definition in reversed(program.definitions):
        lowered_main = _wrap_value_definition(definition, lowered_main)

    return HIRProgram([], lowered_main, program.type)


def lower_expr(expr: TypedExpr) -> HIRExpr:
    if isinstance(expr, TypedCast):
        return HIRCast(lower_expr(expr.value), expr.from_type, expr.to_type, expr.type)

    if isinstance(expr, TypedArray):
        return HIRArrayLit([lower_expr(element) for element in expr.elements], expr.type)

    if isinstance(expr, TypedMap):
        return HIRMap(
            expr.frame_shape,
            expr.cell_shape,
            lower_callable(expr.func),
            lower_expr(expr.array),
            expr.type,
        )

    if isinstance(expr, TypedFold):
        return HIRFold(
            expr.reduction_dim,
            lower_callable(expr.func),
            lower_expr(expr.init),
            lower_expr(expr.array),
            expr.type,
        )

    if isinstance(expr, TypedLambda):
        return HIRLambda(
            [HIRParam(name, param_type) for name, param_type in expr.params],
            lower_expr(expr.body),
            expr.type,
        )

    if isinstance(expr, TypedApp):
        if _typed_node_var_name(expr.func) in _PRIMITIVE_OPS:
            return HIRPrimOp(
                _prim_op_name(_typed_node_var_name(expr.func), expr.type),
                [lower_expr(arg) for arg in expr.args],
                expr.type,
            )
        func_name = _typed_node_var_name(expr.func)
        if func_name is None:
            raise HIRLoweringError("only direct calls are supported in HIR lowering")
        return HIRCall(func_name, [lower_expr(arg) for arg in expr.args], expr.type)

    if isinstance(expr, TypedLet):
        return HIRLet(
            expr.name,
            expr.value.type,
            lower_expr(expr.value),
            lower_expr(expr.body),
            expr.type,
        )

    if isinstance(expr, TypedExprNode):
        return _lower_typed_node(expr)

    raise HIRLoweringError(f"cannot lower typed node {type(expr).__name__}")


def lower_callable(expr: TypedExpr) -> HIRCallable:
    if isinstance(expr, TypedOperatorFunc):
        return HIRPrimCallable(expr.expr.op, expr.type.params, expr.type.result)
    if isinstance(expr, TypedLeftSection):
        return HIRPrimCallable(
            expr.expr.op,
            expr.type.params,
            expr.type.result,
            left_arg=lower_expr(expr.arg),
        )
    if isinstance(expr, TypedRightSection):
        return HIRPrimCallable(
            expr.expr.op,
            expr.type.params,
            expr.type.result,
            right_arg=lower_expr(expr.arg),
        )

    lowered = lower_expr(expr)
    if isinstance(lowered, (HIRLambda, HIRVar)):
        return lowered
    if isinstance(lowered, HIRPrimOp):
        raise HIRLoweringError("primitive operation expression is not a callable")
    if isinstance(expr, TypedExprNode) and isinstance(expr.type, FuncType):
        op = _operator_like_expr(expr)
        if op is not None:
            return HIRPrimCallable(op, expr.type.params, expr.type.result)
    raise HIRLoweringError(f"cannot lower callable {type(expr).__name__}")


def _lower_typed_node(expr: TypedExprNode) -> HIRExpr:
    ast = expr.expr
    if isinstance(ast, IntLit):
        return HIRLit(ast.value, INT)
    if isinstance(ast, FloatLit):
        return HIRLit(ast.value, FLOAT)
    if isinstance(ast, BoolLit):
        return HIRLit(ast.value, BOOL)
    if isinstance(ast, VarExpr):
        return HIRVar(ast.name, expr.type)
    if isinstance(ast, IotaExpr):
        if not isinstance(expr.type, ArrayType):
            raise HIRLoweringError("iota must have an array type")
        return HIRIota(eval_static_dim(ast.size, ast.loc), expr.type)
    raise HIRLoweringError(f"lowering for {type(ast).__name__} is deferred")


def _wrap_value_definition(definition: TypedDefinition, body: HIRExpr) -> HIRExpr:
    if definition.value is None:
        raise HIRLoweringError("function definitions are deferred in HIR lowering")
    return HIRLet(
        definition.definition.name,
        definition.type,
        lower_expr(definition.value),
        body,
        body_result_type(body),
    )


def body_result_type(expr: HIRExpr) -> RemoraType:
    if isinstance(expr, HIRMap):
        return expr.result_type
    if isinstance(expr, HIRFold):
        return expr.result_type
    if isinstance(expr, HIRLet):
        return expr.result_type
    if isinstance(expr, HIRCall):
        return expr.result_type
    if isinstance(expr, HIRLambda):
        return expr.result_type
    if isinstance(expr, HIRPrimOp):
        return expr.result_type
    if isinstance(expr, HIRIota):
        return expr.result_type
    if isinstance(expr, HIRCast):
        return expr.result_type
    if isinstance(expr, HIRArrayLit):
        return expr.result_type
    if isinstance(expr, HIRVar):
        return expr.type
    if isinstance(expr, HIRLit):
        return expr.type
    raise AssertionError(f"unknown HIR expression {type(expr).__name__}")


def _typed_node_var_name(expr: TypedExpr) -> str | None:
    if isinstance(expr, TypedExprNode) and isinstance(expr.expr, VarExpr):
        return expr.expr.name
    return None


def _operator_like_expr(expr: TypedExprNode) -> str | None:
    ast = expr.expr
    op = getattr(ast, "op", None)
    if isinstance(op, str):
        return op
    return None


def _prim_op_name(op: str | None, result_type: RemoraType) -> str:
    if op is None:
        raise HIRLoweringError("missing primitive operator name")
    if result_type == FLOAT:
        suffix = "f"
    elif result_type == INT:
        suffix = "i"
    elif result_type == BOOL:
        suffix = "b"
    else:
        raise HIRLoweringError(f"unsupported primitive result type {result_type}")
    return f"{op}{suffix}"


_PRIMITIVE_OPS = {"+", "-", "*", "/", "<", "<=", "==", "!=", "&&", "||"}
