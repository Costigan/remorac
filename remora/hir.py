"""High-level IR for typed Remora Dense Core programs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TypeAlias

from remora.ast_nodes import (
    AppExpr,
    BoolLit,
    FloatLit,
    IntLit,
    IotaExpr,
    LambdaExpr,
    LengthExpr,
    VarExpr,
    WithShapeExpr,
)
from remora.errors import RemoraError
from remora.operators import ALL_PRIMITIVE_OPS, is_primitive_op
from remora.typechecker import (
    TypedApp,
    TypedAppend,
    TypedArray,
    TypedBox,
    TypedCast,
    TypedDefinition,
    TypedExpr,
    TypedExprNode,
    TypedFold,
    TypedFoldRight,
    TypedGrade,
    TypedIf,
    TypedIndex,
    TypedIndexApp,
    TypedIndicesOf,
    TypedLambda,
    TypedLeftSection,
    TypedLength,
    TypedLet,
    TypedMap,
    TypedOperatorFunc,
    TypedProgram,
    TypedRank,
    TypedRavel,
    TypedReshape,
    TypedRightSection,
    TypedRotate,
    TypedScan,
    TypedShape,
    TypedSlice,
    TypedSort,
    TypedSubarray,
    TypedReverse,
    TypedTake,
    TypedTranspose,
    TypedUnbox,
    TypedWithShape,
    TypedScatterAdd,
    TypedDrop,
    TypedFilter,
    TypedReplicate,
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
    SigmaType,
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
    arrays: list[HIRExpr]
    result_type: RemoraType

    @property
    def array(self) -> HIRExpr:
        return self.arrays[0]


@dataclass(frozen=True)
class HIRApply:
    """General rank-polymorphic application (supersedes HIRMap for implicit maps)."""
    frame_shape: tuple[DimExpr, ...]
    cell_shape: tuple[DimExpr, ...]
    func: HIRCallable
    arrays: list[HIRExpr]
    result_type: RemoraType

    @property
    def array(self) -> HIRExpr:
        return self.arrays[0]


@dataclass(frozen=True)
class HIRFold:
    reduction_dim: DimExpr
    func: HIRCallable
    init: HIRExpr
    array: HIRExpr
    result_type: RemoraType


@dataclass(frozen=True)
class HIRReduce:
    """Parallel reduction over a leading dimension (supersedes HIRFold)."""
    reduction_dim: DimExpr
    func: HIRCallable
    init: HIRExpr
    array: HIRExpr
    result_type: RemoraType


@dataclass(frozen=True)
class HIRFoldRight:
    """Right-to-left fold reduction."""
    reduction_dim: DimExpr
    func: HIRCallable
    init: HIRExpr
    array: HIRExpr
    result_type: RemoraType


@dataclass(frozen=True)
class HIRScan:
    """Prefix-sum (scan) operation."""
    reduction_dim: DimExpr
    func: HIRCallable
    init: HIRExpr
    array: HIRExpr
    exclusive: bool
    right: bool
    result_type: RemoraType


@dataclass(frozen=True)
class HIRRotate:
    """Circular rotation of an array along the leading dimension."""
    array: HIRExpr
    shift: StaticDim
    result_type: ArrayType


@dataclass(frozen=True)
class HIRSubarray:
    """Extract a rectangular sub-region."""
    array: HIRExpr
    offsets: tuple[StaticDim, ...]
    sizes: tuple[StaticDim, ...]
    result_type: ArrayType


@dataclass(frozen=True)
class HIRIndicesOf:
    """Coordinate tensor from array shape."""
    array: HIRExpr
    result_type: ArrayType


@dataclass(frozen=True)
class HIRWithShape:
    """Replicate scalar/array to match a target shape."""
    source: HIRExpr
    result_type: ArrayType


@dataclass(frozen=True)
class HIRBox:
    """Box an array (type erasure: no runtime effect)."""
    value: HIRExpr
    result_type: SigmaType


@dataclass(frozen=True)
class HIRUnbox:
    """Unbox a boxed array (type erasure: no runtime effect, like let)."""
    box_value: HIRExpr
    hidden_names: list[str]
    value_name: str
    body: HIRExpr
    result_type: RemoraType


@dataclass(frozen=True)
class HIRFilter:
    """Filter array elements by predicate (dynamic output size)."""
    predicate: HIRCallable
    array: HIRExpr
    result_type: SigmaType


@dataclass(frozen=True)
class HIRReplicate:
    """Replicate array elements by counts (dynamic output size)."""
    counts: HIRExpr
    array: HIRExpr
    result_type: SigmaType


@dataclass(frozen=True)
class HIRSort:
    """Sort array elements in ascending order."""
    array: HIRExpr
    result_type: ArrayType


@dataclass(frozen=True)
class HIRGrade:
    """Return permutation indices that would sort the array."""
    array: HIRExpr
    result_type: ArrayType


@dataclass(frozen=True)
class HIRAppend:
    """Concatenate two arrays along the leading dimension."""
    left: HIRExpr
    right: HIRExpr
    result_type: ArrayType


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
class HIRScatterAdd:
    target: HIRExpr
    index: HIRExpr
    update: HIRExpr
    result_type: ArrayType


@dataclass(frozen=True)
class HIRPrimOp:
    op: str
    args: list[HIRExpr]
    result_type: RemoraType


@dataclass(frozen=True)
class HIRIf:
    condition: HIRExpr
    then_branch: HIRExpr
    else_branch: HIRExpr
    result_type: RemoraType


@dataclass(frozen=True)
class HIRIndex:
    array: HIRExpr
    indices: list[HIRExpr | HIRSlice]
    result_type: RemoraType


@dataclass(frozen=True)
class HIRSlice:
    start: int
    end: int
    result_type: ArrayType


@dataclass(frozen=True)
class HIRTranspose:
    array: HIRExpr
    result_type: ArrayType


@dataclass(frozen=True)
class HIRReshape:
    array: HIRExpr
    result_type: ArrayType


@dataclass(frozen=True)
class HIRRavel:
    array: HIRExpr
    result_type: ArrayType


@dataclass(frozen=True)
class HIRReverse:
    array: HIRExpr
    result_type: ArrayType


@dataclass(frozen=True)
class HIRTake:
    count: int
    array: HIRExpr
    result_type: ArrayType


@dataclass(frozen=True)
class HIRDrop:
    count: int
    array: HIRExpr
    result_type: ArrayType


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
    | HIRApply
    | HIRFold
    | HIRReduce
    | HIRFoldRight
    | HIRScan
    | HIRRotate
    | HIRSubarray
    | HIRIndicesOf
    | HIRWithShape
    | HIRScatterAdd
    | HIRBox
    | HIRUnbox
    | HIRFilter
    | HIRReplicate
    | HIRSort
    | HIRGrade
    | HIRAppend
    | HIRLet
    | HIRCall
    | HIRLambda
    | HIRPrimOp
    | HIRIf
    | HIRIndex
    | HIRSlice
    | HIRTranspose
    | HIRReshape
    | HIRRavel
    | HIRReverse
    | HIRTake
    | HIRDrop
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
        if definition.value is None:
            continue
        lowered_main = _wrap_value_definition(definition, lowered_main)

    return HIRProgram([], lowered_main, program.type)


def lower_expr(expr: TypedExpr) -> HIRExpr:
    if isinstance(expr, TypedCast):
        return HIRCast(lower_expr(expr.value), expr.from_type, expr.to_type, expr.type)

    if isinstance(expr, TypedArray):
        return HIRArrayLit([lower_expr(element) for element in expr.elements], expr.type)

    if isinstance(expr, TypedMap):
        # Use HIRApply for implicit (AppExpr-based) maps, HIRMap for explicit ones
        node_cls = HIRApply if isinstance(expr.expr, AppExpr) else HIRMap
        return node_cls(
            expr.frame_shape,
            expr.cell_shape,
            lower_callable(expr.func),
            [lower_expr(array) for array in expr.arrays],
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

    if isinstance(expr, TypedFoldRight):
        return HIRFoldRight(
            expr.reduction_dim,
            lower_callable(expr.func),
            lower_expr(expr.init),
            lower_expr(expr.array),
            expr.type,
        )

    if isinstance(expr, TypedScan):
        return HIRScan(
            expr.reduction_dim,
            lower_callable(expr.func),
            lower_expr(expr.init),
            lower_expr(expr.array),
            expr.exclusive,
            expr.right,
            expr.type,
        )

    if isinstance(expr, TypedShape):
        return HIRArrayLit(
            [HIRLit(dim.value, INT) for dim in _shape_dims(expr.array.type)],
            expr.type,
        )

    if isinstance(expr, TypedRank):
        return HIRLit(expr.array.type.rank, INT)

    if isinstance(expr, TypedLength):
        return HIRLit(expr.dim.value, INT)

    if isinstance(expr, TypedIndex):
        return HIRIndex(
            lower_expr(expr.array),
            [_lower_index_element(index) for index in expr.indices],
            expr.type,
        )

    if isinstance(expr, TypedSlice):
        return _lower_slice(expr)

    if isinstance(expr, TypedTranspose):
        return HIRTranspose(lower_expr(expr.array), expr.type)

    if isinstance(expr, TypedReshape):
        return HIRReshape(lower_expr(expr.array), expr.type)

    if isinstance(expr, TypedRavel):
        return HIRRavel(lower_expr(expr.array), expr.type)

    if isinstance(expr, TypedReverse):
        return HIRReverse(lower_expr(expr.array), expr.type)

    if isinstance(expr, TypedRotate):
        return HIRRotate(lower_expr(expr.array), expr.shift, expr.type)

    if isinstance(expr, TypedSubarray):
        return HIRSubarray(lower_expr(expr.array), expr.offsets, expr.sizes, expr.type)

    if isinstance(expr, TypedIndicesOf):
        return HIRIndicesOf(lower_expr(expr.array), expr.type)

    if isinstance(expr, TypedWithShape):
        return HIRWithShape(lower_expr(expr.source), expr.type)

    if isinstance(expr, TypedScatterAdd):
        return HIRScatterAdd(
            lower_expr(expr.array),
            lower_expr(expr.index),
            lower_expr(expr.update),
            expr.type,
        )

    if isinstance(expr, TypedBox):
        return HIRBox(lower_expr(expr.value), expr.type)

    if isinstance(expr, TypedUnbox):
        return HIRUnbox(
            lower_expr(expr.box_value),
            expr.hidden_names,
            expr.value_name,
            lower_expr(expr.body),
            expr.type,
        )

    if isinstance(expr, TypedAppend):
        return HIRAppend(
            lower_expr(expr.left),
            lower_expr(expr.right),
            expr.type,
        )

    if isinstance(expr, TypedTake):
        return HIRTake(expr.count.expr.value, lower_expr(expr.array), expr.type) # type: ignore

    if isinstance(expr, TypedDrop):
        return HIRDrop(expr.count.expr.value, lower_expr(expr.array), expr.type) # type: ignore

    if isinstance(expr, TypedLambda):
        return HIRLambda(
            [HIRParam(name, param_type) for name, param_type in expr.params],
            lower_expr(expr.body),
            expr.type,
        )

    if isinstance(expr, TypedIndexApp):
        return lower_expr(expr.function)

    if isinstance(expr, TypedApp):
        if _typed_node_var_name(expr.func) in ALL_PRIMITIVE_OPS:
            return HIRPrimOp(
                _prim_op_name(_typed_node_var_name(expr.func), expr.type),
                [lower_expr(arg) for arg in expr.args],
                expr.type,
            )
        if isinstance(expr.func, TypedLambda):
            return _inline_lambda_call(expr.func, expr.args, expr.type)
        if isinstance(expr.func, TypedIndexApp):
            return _inline_lambda_call(expr.func.function, expr.args, expr.type)
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

    if isinstance(expr, TypedIf):
        return HIRIf(
            lower_expr(expr.condition),
            lower_expr(expr.then_branch),
            lower_expr(expr.else_branch),
            expr.type,
        )

    if isinstance(expr, TypedFilter):
        return HIRFilter(
            lower_callable(expr.predicate),
            lower_expr(expr.array),
            expr.type,
        )

    if isinstance(expr, TypedReplicate):
        return HIRReplicate(
            lower_expr(expr.counts),
            lower_expr(expr.array),
            expr.type,
        )

    if isinstance(expr, TypedSort):
        return HIRSort(lower_expr(expr.array), expr.type)

    if isinstance(expr, TypedGrade):
        return HIRGrade(lower_expr(expr.array), expr.type)

    if isinstance(expr, TypedExprNode):
        return _lower_typed_node(expr)

    raise HIRLoweringError(f"cannot lower typed node {type(expr).__name__}")


def _lower_index_element(index: TypedExpr | TypedSlice) -> HIRExpr | HIRSlice:
    if isinstance(index, TypedSlice):
        return _lower_slice(index)
    return lower_expr(index)


def _lower_slice(expr: TypedSlice) -> HIRSlice:
    start_val = 0
    if expr.expr.start is not None:
        # confirmed in typechecker to be IntLit for now
        start_val = expr.expr.start.value

    # typedslice type is its own result type
    # but we need end = start + length
    length = expr.type.shape[0].value
    end_val = start_val + length

    return HIRSlice(start_val, end_val, expr.type)


def lower_callable(expr: TypedExpr) -> HIRCallable:
    if isinstance(expr, TypedOperatorFunc):
        return HIRPrimCallable(expr.expr.op, expr.type.params, expr.type.result)
    if isinstance(expr, TypedLeftSection):
        return HIRPrimCallable(
            expr.expr.op,
            expr.type.params,
            expr.type.result,
            right_arg=lower_expr(expr.arg),
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
    if isinstance(ast, LambdaExpr):
        if not isinstance(expr.type, FuncType):
            raise HIRLoweringError("lambda must have a function type")
        return HIRLambda(
            [HIRParam(name, param_type) for name, param_type in zip(ast.params, expr.type.params)],
            HIRLit(0, INT),
            expr.type,
        )
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


def _inline_lambda_call(
    function: TypedLambda,
    args: list[TypedExpr],
    result_type: RemoraType,
) -> HIRExpr:
    if len(function.params) != len(args):
        raise HIRLoweringError("function arity mismatch in HIR lowering")
    body = lower_expr(function.body)
    for (name, param_type), arg in reversed(list(zip(function.params, args))):
        body = HIRLet(name, param_type, lower_expr(arg), body, result_type)
    return body


def body_result_type(expr: HIRExpr) -> RemoraType:
    if isinstance(expr, (HIRMap, HIRApply)):
        return expr.result_type
    if isinstance(expr, (HIRFold, HIRReduce)):
        return expr.result_type
    if isinstance(expr, HIRLet):
        return expr.result_type
    if isinstance(expr, HIRCall):
        return expr.result_type
    if isinstance(expr, HIRLambda):
        return expr.result_type
    if isinstance(expr, HIRPrimOp):
        return expr.result_type
    if isinstance(expr, HIRIndex):
        return expr.result_type
    if isinstance(expr, HIRSlice):
        return expr.result_type
    if isinstance(expr, HIRTranspose):
        return expr.result_type
    if isinstance(expr, HIRReshape):
        return expr.result_type
    if isinstance(expr, HIRRavel):
        return expr.result_type
    if isinstance(expr, HIRReverse):
        return expr.result_type
    if isinstance(expr, HIRTake):
        return expr.result_type
    if isinstance(expr, HIRDrop):
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
    if isinstance(expr, HIRWithShape):
        return expr.result_type
    if isinstance(expr, (HIRFilter, HIRReplicate)):
        return expr.result_type
    if isinstance(expr, (HIRSort, HIRGrade)):
        return expr.result_type
    raise AssertionError(f"unknown HIR expression {type(expr).__name__}")


def _shape_dims(value_type: RemoraType) -> tuple[DimExpr, ...]:
    if isinstance(value_type, ArrayType):
        return value_type.shape
    if isinstance(value_type, ScalarType):
        return ()
    raise HIRLoweringError("shape/rank of function values is deferred")


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
    if isinstance(result_type, ArrayType):
        result_type = result_type.element
    if result_type == FLOAT:
        suffix = "f"
    elif result_type == INT:
        suffix = "i"
    elif result_type == BOOL:
        suffix = "b"
    else:
        raise HIRLoweringError(f"unsupported primitive result type {result_type}")
    return f"{op}{suffix}"
