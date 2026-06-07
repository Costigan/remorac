"""Small Dense Core type checker for the parser AST."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TypeAlias

from remora.ast_nodes import (
    AppExpr,
    ArrayLit,
    BoolLit,
    Definition,
    Expr,
    FloatLit,
    FoldExpr,
    FuncDef,
    IfExpr,
    IndexExpr,
    IntLit,
    IotaExpr,
    LambdaExpr,
    LeftSectionExpr,
    LetExpr,
    MapExpr,
    OperatorFuncExpr,
    Program,
    RankExpr,
    RavelExpr,
    ReshapeExpr,
    RightSectionExpr,
    ShapeExpr,
    SliceRange,
    ReverseExpr,
    TakeExpr,
    TransposeExpr,
    DropExpr,
    ValDef,
    VarExpr,
)
from remora.operators import ALL_PRIMITIVE_OPS
from remora.types import (
    BOOL,
    FLOAT,
    INT,
    ArrayType,
    DimExpr,
    FuncType,
    RemoraType,
    RemoraTypeError,
    ScalarType,
    StaticDim,
    common_numeric_type,
    enforce_rank_limit,
    eval_static_dim,
    infer_lifting,
    is_numeric,
)


@dataclass(frozen=True)
class TypedProgram:
    definitions: list[TypedDefinition]
    body: TypedExpr | None
    type: RemoraType | None


@dataclass(frozen=True)
class TypedDefinition:
    definition: Definition
    value: TypedExpr | None
    type: RemoraType | None


@dataclass(frozen=True)
class TypedExprNode:
    expr: Expr
    type: RemoraType


@dataclass(frozen=True)
class TypedCast:
    value: TypedExpr
    from_type: ScalarType
    to_type: ScalarType
    type: ScalarType


@dataclass(frozen=True)
class TypedArray:
    expr: ArrayLit
    elements: list[TypedExpr]
    type: ArrayType


@dataclass(frozen=True)
class TypedMap:
    expr: MapExpr
    func: TypedExpr
    arrays: list[TypedExpr]
    frame_shape: tuple[DimExpr, ...]
    cell_shape: tuple[DimExpr, ...]
    type: RemoraType

    @property
    def array(self) -> TypedExpr:
        return self.arrays[0]


@dataclass(frozen=True)
class TypedFold:
    expr: FoldExpr
    func: TypedExpr
    init: TypedExpr
    array: TypedExpr
    reduction_dim: DimExpr
    type: RemoraType


@dataclass(frozen=True)
class TypedShape:
    expr: ShapeExpr
    array: TypedExpr
    type: ArrayType


@dataclass(frozen=True)
class TypedRank:
    expr: RankExpr
    array: TypedExpr
    type: ScalarType


@dataclass(frozen=True)
class TypedTranspose:
    expr: TransposeExpr
    array: TypedExpr
    type: ArrayType


@dataclass(frozen=True)
class TypedReshape:
    expr: ReshapeExpr
    shape_expr: TypedExpr
    array: TypedExpr
    type: ArrayType


@dataclass(frozen=True)
class TypedRavel:
    expr: RavelExpr
    array: TypedExpr
    type: ArrayType


@dataclass(frozen=True)
class TypedReverse:
    expr: ReverseExpr
    array: TypedExpr
    type: ArrayType


@dataclass(frozen=True)
class TypedTake:
    expr: TakeExpr
    count: TypedExpr
    array: TypedExpr
    type: ArrayType


@dataclass(frozen=True)
class TypedDrop:
    expr: DropExpr
    count: TypedExpr
    array: TypedExpr
    type: ArrayType


@dataclass(frozen=True)
class TypedSlice:
    expr: SliceRange
    start: TypedExpr | None
    end: TypedExpr | None
    type: ArrayType


@dataclass(frozen=True)
class TypedIndex:
    expr: IndexExpr
    array: TypedExpr
    indices: list[TypedExpr | TypedSlice]
    type: RemoraType


@dataclass(frozen=True)
class TypedLambda:
    expr: LambdaExpr | FuncDef
    params: list[tuple[str, RemoraType]]
    body: TypedExpr
    type: FuncType


@dataclass(frozen=True)
class TypedOperatorFunc:
    expr: OperatorFuncExpr
    type: FuncType


@dataclass(frozen=True)
class TypedLeftSection:
    expr: LeftSectionExpr
    arg: TypedExpr
    type: FuncType


@dataclass(frozen=True)
class TypedRightSection:
    expr: RightSectionExpr
    arg: TypedExpr
    type: FuncType


@dataclass(frozen=True)
class TypedApp:
    expr: AppExpr
    func: TypedExpr
    args: list[TypedExpr]
    type: RemoraType


@dataclass(frozen=True)
class TypedLet:
    expr: LetExpr
    name: str
    value: TypedExpr
    body: TypedExpr
    type: RemoraType


@dataclass(frozen=True)
class TypedIf:
    expr: IfExpr
    condition: TypedExpr
    then_branch: TypedExpr
    else_branch: TypedExpr
    type: RemoraType


TypedExpr: TypeAlias = (
    TypedExprNode
    | TypedCast
    | TypedArray
    | TypedMap
    | TypedFold
    | TypedShape
    | TypedRank
    | TypedTranspose
    | TypedReshape
    | TypedRavel
    | TypedReverse
    | TypedTake
    | TypedDrop
    | TypedSlice
    | TypedIndex
    | TypedLambda
    | TypedOperatorFunc
    | TypedLeftSection
    | TypedRightSection
    | TypedApp
    | TypedLet
    | TypedIf
)


class TypeEnv:
    def __init__(self, bindings: dict[str, RemoraType] | None = None):
        self._bindings = dict(bindings or {})

    def extend(self, name: str, value_type: RemoraType) -> TypeEnv:
        return TypeEnv({**self._bindings, name: value_type})

    def lookup(self, name: str) -> RemoraType:
        try:
            return self._bindings[name]
        except KeyError as exc:
            raise RemoraTypeError(f"unbound variable '{name}'") from exc


class TypeChecker:
    def __init__(self) -> None:
        self._functions: dict[str, FuncDef] = {}
        self._active_functions: set[str] = set()

    def check_program(self, program: Program) -> TypedProgram:
        env = self._build_prelude_env()
        typed_definitions: list[TypedDefinition] = []
        self._functions = {
            definition.name: definition
            for definition in program.definitions
            if isinstance(definition, FuncDef)
        }
        self._active_functions = set()

        for definition in program.definitions:
            typed_definition, env = self._check_definition(definition, env)
            typed_definitions.append(typed_definition)

        if program.body is None:
            return TypedProgram(typed_definitions, None, None)

        typed_body = self.infer(program.body, env)
        return TypedProgram(typed_definitions, typed_body, typed_body.type)

    def infer(self, expr: Expr, env: TypeEnv | None = None) -> TypedExpr:
        env = env or self._build_prelude_env()

        if isinstance(expr, IntLit):
            return TypedExprNode(expr, INT)
        if isinstance(expr, FloatLit):
            return TypedExprNode(expr, FLOAT)
        if isinstance(expr, BoolLit):
            return TypedExprNode(expr, BOOL)
        if isinstance(expr, VarExpr):
            return TypedExprNode(expr, env.lookup(expr.name))
        if isinstance(expr, ArrayLit):
            return self._infer_array(expr, env)
        if isinstance(expr, IotaExpr):
            size = eval_static_dim(expr.size, expr.loc)
            return TypedExprNode(expr, ArrayType(INT, (size,)))
        if isinstance(expr, ShapeExpr):
            typed_array = self.infer(expr.array, env)
            self._require_shape_operand(typed_array.type, "shape", expr.loc)
            return TypedShape(
                expr,
                typed_array,
                ArrayType(INT, (StaticDim(typed_array.type.rank),)),
            )
        if isinstance(expr, RankExpr):
            typed_array = self.infer(expr.array, env)
            self._require_shape_operand(typed_array.type, "rank", expr.loc)
            return TypedRank(expr, typed_array, INT)
        if isinstance(expr, TransposeExpr):
            return self._infer_transpose(expr, env)
        if isinstance(expr, ReshapeExpr):
            return self._infer_reshape(expr, env)
        if isinstance(expr, RavelExpr):
            return self._infer_ravel(expr, env)
        if isinstance(expr, TakeExpr):
            return self._infer_take(expr, env)
        if isinstance(expr, DropExpr):
            return self._infer_drop(expr, env)
        if isinstance(expr, SliceRange):
            raise RemoraTypeError("slice range must be used within indexing", expr.loc)
        if isinstance(expr, IndexExpr):
            return self._infer_indexing(expr, env)
        if isinstance(expr, LetExpr):
            if isinstance(expr.value, LambdaExpr):
                return self._infer_let_lambda(expr, env)
            typed_value = self.infer(expr.value, env)
            inner_env = env.extend(expr.name, typed_value.type)
            typed_body = self.infer(expr.body, inner_env)
            return TypedLet(expr, expr.name, typed_value, typed_body, typed_body.type)
        if isinstance(expr, IfExpr):
            condition = self.infer(expr.condition, env)
            if isinstance(condition.type, ArrayType):
                if condition.type.element != BOOL:
                    raise RemoraTypeError("if condition array must have boolean elements", expr.loc)
            elif condition.type != BOOL:
                self._require(condition.type, BOOL, expr.loc)
            then_branch = self.infer(expr.then_branch, env)
            else_branch = self.infer(expr.else_branch, env)
            self._require(then_branch.type, else_branch.type, expr.loc)
            return TypedIf(expr, condition, then_branch, else_branch, then_branch.type)
        if isinstance(expr, AppExpr):
            return self._infer_app(expr, env)
        if isinstance(expr, LambdaExpr):
            raise RemoraTypeError(
                "lambda expressions require an expected function type", expr.loc
            )
        if isinstance(expr, (OperatorFuncExpr, LeftSectionExpr, RightSectionExpr)):
            raise RemoraTypeError(
                "operator sections require an expected function type", expr.loc
            )
        if isinstance(expr, MapExpr):
            return self._infer_map(expr, env)
        if isinstance(expr, FoldExpr):
            return self._infer_fold(expr, env)
        if isinstance(expr, ReverseExpr):
            return self._infer_reverse(expr, env)

        raise RemoraTypeError(f"type checking for {type(expr).__name__} is deferred")

    def check_callable(
        self, expr: Expr, expected_type: FuncType, env: TypeEnv
    ) -> TypedExpr:
        if isinstance(expr, LambdaExpr):
            if len(expr.params) != len(expected_type.params):
                raise RemoraTypeError("lambda arity does not match expected type", expr.loc)
            inner_env = env
            for name, param_type in zip(expr.params, expected_type.params):
                inner_env = inner_env.extend(name, param_type)
            typed_body = self.infer(expr.body, inner_env)
            typed_body = self._coerce(typed_body, expected_type.result, expr.loc)
            return TypedLambda(
                expr,
                list(zip(expr.params, expected_type.params)),
                typed_body,
                expected_type,
            )

        if isinstance(expr, OperatorFuncExpr):
            if len(expected_type.params) != 2:
                raise RemoraTypeError("operator function must be checked as binary")
            return self._check_operator_func(expr, expected_type)

        if isinstance(expr, LeftSectionExpr):
            if len(expected_type.params) != 1:
                raise RemoraTypeError("left operator section must be unary")
            return self._check_left_section(expr, expected_type, env)

        if isinstance(expr, RightSectionExpr):
            if len(expected_type.params) != 1:
                raise RemoraTypeError("right operator section must be unary")
            return self._check_right_section(expr, expected_type, env)

        if isinstance(expr, VarExpr) and expr.name in self._functions:
            return self._typed_top_level_function(
                self._functions[expr.name],
                expected_type,
                env,
            )

        typed = self.infer(expr, env)
        self._require(typed.type, expected_type, expr.loc)
        return typed

    def _infer_array(self, expr: ArrayLit, env: TypeEnv) -> TypedExpr:
        if not expr.elements:
            raise RemoraTypeError("empty array literals require annotations", expr.loc)

        typed_elements = [self.infer(element, env) for element in expr.elements]
        element_type = typed_elements[0].type
        for typed_element in typed_elements[1:]:
            self._require(typed_element.type, element_type, expr.loc)

        if isinstance(element_type, FuncType):
            raise RemoraTypeError("arrays of functions are deferred", expr.loc)

        if isinstance(element_type, ArrayType):
            array_type = ArrayType(element_type.element, (StaticDim(len(expr.elements)),) + element_type.shape)
        else:
            array_type = ArrayType(element_type, (StaticDim(len(expr.elements)),))

        enforce_rank_limit(array_type, expr.loc)
        return TypedArray(expr, typed_elements, array_type)

    def _infer_app(self, expr: AppExpr, env: TypeEnv) -> TypedExpr:
        if isinstance(expr.func, VarExpr) and expr.func.name in ALL_PRIMITIVE_OPS:
            return self._infer_primitive_app(expr, env)
        if isinstance(expr.func, VarExpr) and expr.func.name in self._functions:
            return self._infer_top_level_function_app(expr, env)

        typed_func = self.infer(expr.func, env)
        if not isinstance(typed_func.type, FuncType):
            raise RemoraTypeError(f"not a function: {typed_func.type}", expr.loc)
        if len(expr.args) != len(typed_func.type.params):
            raise RemoraTypeError("function arity mismatch", expr.loc)

        typed_args = [
            self._coerce(self.infer(arg, env), param_type, expr.loc)
            for arg, param_type in zip(expr.args, typed_func.type.params)
        ]
        return TypedApp(expr, typed_func, typed_args, typed_func.type.result)

    def _infer_primitive_app(self, expr: AppExpr, env: TypeEnv) -> TypedExpr:
        if len(expr.args) != 2:
            raise RemoraTypeError("primitive operators are binary", expr.loc)
        left = self.infer(expr.args[0], env)
        right = self.infer(expr.args[1], env)
        op = expr.func.name

        if op in {"+", "-", "*"}:
            result_type = self._common_fold_operator_type(left.type, right.type, expr.loc)
            return TypedApp(
                expr,
                TypedExprNode(expr.func, FuncType((result_type, result_type), result_type)),
                [
                    self._coerce(left, result_type, expr.loc) if not isinstance(left.type, ArrayType) else left,
                    self._coerce(right, result_type, expr.loc) if not isinstance(right.type, ArrayType) else right,
                ],
                result_type,
            )
        if op == "/":
            if not is_numeric(left.type) or not is_numeric(right.type):
                raise RemoraTypeError("division expects numeric operands", expr.loc)
            return TypedApp(
                expr,
                TypedExprNode(expr.func, FuncType((FLOAT, FLOAT), FLOAT)),
                [
                    self._coerce(left, FLOAT, expr.loc),
                    self._coerce(right, FLOAT, expr.loc),
                ],
                FLOAT,
            )
        if op in {"<", "<=", "==", "!="}:
            result_type = common_numeric_type(left.type, right.type)
            return TypedApp(
                expr,
                TypedExprNode(expr.func, FuncType((result_type, result_type), BOOL)),
                [
                    self._coerce(left, result_type, expr.loc),
                    self._coerce(right, result_type, expr.loc),
                ],
                BOOL,
            )
        if op in {"&&", "||"}:
            self._require(left.type, BOOL, expr.loc)
            self._require(right.type, BOOL, expr.loc)
            return TypedApp(
                expr,
                TypedExprNode(expr.func, FuncType((BOOL, BOOL), BOOL)),
                [left, right],
                BOOL,
            )
        raise RemoraTypeError(f"unknown primitive operator '{op}'", expr.loc)

    def _infer_map(self, expr: MapExpr, env: TypeEnv) -> TypedMap:
        if len(expr.arrays) == 2:
            return self._infer_binary_map(expr, env)
        if len(expr.arrays) != 1:
            raise RemoraTypeError("map currently supports one or two arrays", expr.loc)
        typed_array = self.infer(expr.array, env)
        candidates = self._cell_type_candidates(typed_array.type)
        errors: list[Exception] = []

        for cell_type in candidates:
            try:
                expected_func_type = self._infer_callable_type_for_map(
                    expr.func, cell_type, env
                )
                typed_func = self.check_callable(expr.func, expected_func_type, env)
                frame_shape, result_type = infer_lifting(expected_func_type, typed_array.type)
                cell_shape = cell_type.shape if isinstance(cell_type, ArrayType) else ()
                return TypedMap(
                    expr,
                    typed_func,
                    [typed_array],
                    frame_shape,
                    cell_shape,
                    result_type,
                )
            except RemoraTypeError as exc:
                errors.append(exc)

        raise RemoraTypeError(f"could not type-check map callable: {errors[-1]}", expr.loc)

    def _infer_binary_map(self, expr: MapExpr, env: TypeEnv) -> TypedMap:
        left = self.infer(expr.arrays[0], env)
        right = self.infer(expr.arrays[1], env)
        left_cell, frame_shape = self._scalar_cell_and_frame(left.type, expr.loc)
        right_cell, right_frame = self._scalar_cell_and_frame(right.type, expr.loc)
        if frame_shape != right_frame:
            raise RemoraTypeError(
                f"binary map expects matching shapes, got {left.type} and {right.type}",
                expr.loc,
            )

        func_type = self._infer_binary_map_callable_type(
            expr.func,
            left_cell,
            right_cell,
            env,
        )
        typed_func = self.check_callable(expr.func, func_type, env)
        result_type = func_type.result
        if frame_shape:
            if isinstance(result_type, FuncType):
                raise RemoraTypeError("function-valued map results are deferred", expr.loc)
            if isinstance(result_type, ArrayType):
                raise RemoraTypeError("binary map over array-valued cells is deferred", expr.loc)
            result_type = ArrayType(result_type, frame_shape)
        return TypedMap(expr, typed_func, [left, right], frame_shape, (), result_type)

    def _scalar_cell_and_frame(
        self, value_type: RemoraType, loc
    ) -> tuple[ScalarType, tuple[DimExpr, ...]]:
        if isinstance(value_type, ScalarType):
            return value_type, ()
        if isinstance(value_type, ArrayType):
            return value_type.element, value_type.shape
        raise RemoraTypeError("map over function values is deferred", loc)

    def _infer_binary_map_callable_type(
        self,
        expr: Expr,
        left_cell: ScalarType,
        right_cell: ScalarType,
        env: TypeEnv,
    ) -> FuncType:
        if isinstance(expr, OperatorFuncExpr):
            if expr.op in {"+", "-", "*"}:
                result = common_numeric_type(left_cell, right_cell)
                return FuncType((result, result), result)
            if expr.op == "/":
                self._require_numeric(left_cell, expr.loc)
                self._require_numeric(right_cell, expr.loc)
                return FuncType((FLOAT, FLOAT), FLOAT)
            if expr.op in {"<", "<=", "==", "!="}:
                result = common_numeric_type(left_cell, right_cell)
                return FuncType((result, result), BOOL)
            if expr.op in {"&&", "||"}:
                self._require(left_cell, BOOL, expr.loc)
                self._require(right_cell, BOOL, expr.loc)
                return FuncType((BOOL, BOOL), BOOL)
            raise RemoraTypeError(f"operator {expr.op} is deferred", expr.loc)

        if isinstance(expr, LambdaExpr):
            if len(expr.params) != 2:
                raise RemoraTypeError("binary map expects a binary callable", expr.loc)
            inner_env = env.extend(expr.params[0], left_cell).extend(
                expr.params[1],
                right_cell,
            )
            body = self.infer(expr.body, inner_env)
            return FuncType((left_cell, right_cell), body.type)

        if isinstance(expr, VarExpr) and expr.name in self._functions:
            function = self._functions[expr.name]
            if len(function.params) != 2:
                raise RemoraTypeError("binary map expects a binary callable", expr.loc)
            return self._infer_top_level_function_type(
                function,
                (left_cell, right_cell),
                env,
            )

        typed = self.infer(expr, env)
        if not isinstance(typed.type, FuncType) or len(typed.type.params) != 2:
            raise RemoraTypeError("binary map expects a binary callable", expr.loc)
        self._require(typed.type.params[0], left_cell, expr.loc)
        self._require(typed.type.params[1], right_cell, expr.loc)
        return typed.type

    def _infer_transpose(self, expr: TransposeExpr, env: TypeEnv) -> TypedTranspose:
        typed_array = self._require_array(expr.array, "transpose", env)
        if typed_array.type.rank < 2:
            raise RemoraTypeError("transpose expects an array of rank at least 2", expr.loc)

        shape = typed_array.type.shape
        transposed_shape = (shape[1], shape[0]) + shape[2:]
        return TypedTranspose(
            expr,
            typed_array,
            ArrayType(typed_array.type.element, transposed_shape),
        )

    def _infer_reshape(self, expr: ReshapeExpr, env: TypeEnv) -> TypedReshape:
        typed_shape = self.infer(expr.shape, env)
        typed_array = self._require_array(expr.array, "reshape", env)
        
        # In Dense Core, target shape must be literal to keep ArrayType static
        if not isinstance(typed_shape, TypedArray) or not all(isinstance(e, TypedExprNode) and isinstance(e.expr, IntLit) for e in typed_shape.elements):
             raise RemoraTypeError("reshape target shape must be a literal integer array", expr.loc)
        
        new_shape_values = [int(e.expr.value) for e in typed_shape.elements] # type: ignore
        new_total = 1
        for v in new_shape_values: new_total *= v
        
        old_total = 1
        for dim in typed_array.type.shape:
             old_total *= dim.value
             
        if new_total != old_total:
             raise RemoraTypeError(f"reshape mismatch: target shape {new_shape_values} (size {new_total}) does not match input size {old_total}", expr.loc)
             
        return TypedReshape(
            expr,
            typed_shape,
            typed_array,
            ArrayType(typed_array.type.element, tuple(StaticDim(v) for v in new_shape_values))
        )

    def _infer_ravel(self, expr: RavelExpr, env: TypeEnv) -> TypedRavel:
        typed_array = self._require_array(expr.array, "ravel", env)
        total = 1
        for dim in typed_array.type.shape: total *= dim.value
        return TypedRavel(
            expr,
            typed_array,
            ArrayType(typed_array.type.element, (StaticDim(total),))
        )

    def _infer_reverse(self, expr: ReverseExpr, env: TypeEnv) -> TypedReverse:
        typed_array = self._require_array(expr.array, "reverse", env)
        if typed_array.type.rank < 1:
            raise RemoraTypeError("reverse expects an array operand", expr.loc)
        return TypedReverse(expr, typed_array, typed_array.type)

    def _infer_take(self, expr: TakeExpr, env: TypeEnv) -> TypedTake:
        typed_count = self.infer(expr.count, env)
        typed_array = self._require_array(expr.array, "take", env)
        self._require(typed_count.type, INT, expr.loc)
        
        if typed_array.type.rank < 1:
             raise RemoraTypeError("take expects a non-scalar array", expr.loc)
             
        if not (isinstance(typed_count, TypedExprNode) and isinstance(typed_count.expr, IntLit)):
             raise RemoraTypeError("take count must be a literal integer", expr.loc)
             
        count = typed_count.expr.value
        extent = typed_array.type.shape[0].value
        if count < 0 or count > extent:
             raise RemoraTypeError(f"take count {count} is out of bounds for axis 0 with extent {extent}", expr.loc)
             
        new_shape = (StaticDim(count),) + typed_array.type.shape[1:]
        return TypedTake(expr, typed_count, typed_array, ArrayType(typed_array.type.element, new_shape))

    def _infer_drop(self, expr: DropExpr, env: TypeEnv) -> TypedDrop:
        typed_count = self.infer(expr.count, env)
        typed_array = self._require_array(expr.array, "drop", env)
        self._require(typed_count.type, INT, expr.loc)
        
        if typed_array.type.rank < 1:
             raise RemoraTypeError("drop expects a non-scalar array", expr.loc)

        if not (isinstance(typed_count, TypedExprNode) and isinstance(typed_count.expr, IntLit)):
             raise RemoraTypeError("drop count must be a literal integer", expr.loc)
             
        count = typed_count.expr.value
        extent = typed_array.type.shape[0].value
        if count < 0 or count > extent:
             raise RemoraTypeError(f"drop count {count} is out of bounds for axis 0 with extent {extent}", expr.loc)
             
        new_shape = (StaticDim(extent - count),) + typed_array.type.shape[1:]
        return TypedDrop(expr, typed_count, typed_array, ArrayType(typed_array.type.element, new_shape))

    def _require_array(self, expr: Expr, context: str, env: TypeEnv) -> TypedExpr:
        typed = self.infer(expr, env)
        if not isinstance(typed.type, ArrayType):
            raise RemoraTypeError(f"{context} expects an array operand", expr.loc)
        return typed

    def _infer_slice_range(
        self,
        expr: SliceRange,
        axis_extent: StaticDim,
        env: TypeEnv,
    ) -> TypedSlice:
        typed_start = self.infer(expr.start, env) if expr.start is not None else None
        typed_end = self.infer(expr.end, env) if expr.end is not None else None
        if typed_start is not None:
            self._require(typed_start.type, INT, expr.loc)
        if typed_end is not None:
            self._require(typed_end.type, INT, expr.loc)

        # For now, we only support static slices to keep ArrayType static
        start_val = 0
        if expr.start is not None:
            if isinstance(expr.start, IntLit):
                start_val = expr.start.value
            else:
                raise RemoraTypeError("only literal integer slice bounds are supported so far", expr.loc)

        end_val = axis_extent.value
        if expr.end is not None:
            if isinstance(expr.end, IntLit):
                end_val = expr.end.value
            else:
                raise RemoraTypeError("only literal integer slice bounds are supported so far", expr.loc)

        if start_val < 0 or end_val < 0 or start_val > axis_extent.value or end_val > axis_extent.value or start_val > end_val:
             raise RemoraTypeError(
                 f"invalid slice {start_val}:{end_val} for axis with extent {axis_extent.value}",
                 expr.loc,
             )

        slice_extent = end_val - start_val
        return TypedSlice(
            expr,
            typed_start,
            typed_end,
            ArrayType(INT, (StaticDim(slice_extent),)),
        )

    def _infer_indexing(self, expr: IndexExpr, env: TypeEnv) -> TypedIndex:
        typed_array = self.infer(expr.array, env)
        if not isinstance(typed_array.type, ArrayType):
            raise RemoraTypeError("indexing expects an array operand", expr.loc)

        if len(expr.indices) > typed_array.type.rank:
            raise RemoraTypeError(
                f"too many indices for rank-{typed_array.type.rank} array",
                expr.loc,
            )

        typed_indices: list[TypedExpr | TypedSlice] = []
        result_shape_parts: list[DimExpr] = []

        for position, index in enumerate(expr.indices):
            axis_extent = typed_array.type.shape[position]
            if isinstance(index, SliceRange):
                typed_slice = self._infer_slice_range(index, axis_extent, env)
                typed_indices.append(typed_slice)
                result_shape_parts.append(typed_slice.type.shape[0])
            else:
                typed_index = self.infer(index, env)
                self._require(typed_index.type, INT, expr.loc)
                if isinstance(index, IntLit):
                    value = index.value
                    if value < 0 or value >= axis_extent.value:
                        raise RemoraTypeError(
                            f"index {value} is out of bounds for axis {position} with extent {axis_extent.value}",
                            index.loc,
                        )
                typed_indices.append(typed_index)
                # scalar index drops the dimension, so no part added to result_shape_parts

        # Add remaining dimensions from the original array
        result_shape_parts.extend(typed_array.type.shape[len(expr.indices) :])

        result_type: RemoraType
        if not result_shape_parts:
            result_type = typed_array.type.element
        else:
            result_type = ArrayType(typed_array.type.element, tuple(result_shape_parts))

        return TypedIndex(expr, typed_array, typed_indices, result_type)

    def _infer_fold(self, expr: FoldExpr, env: TypeEnv) -> TypedFold:
        typed_init = self.infer(expr.init, env)
        typed_array = self.infer(expr.array, env)
        if not isinstance(typed_array.type, ArrayType) or typed_array.type.rank < 1:
            raise RemoraTypeError("fold expects a non-scalar array", expr.loc)

        element_type = typed_array.type.drop_outer(1)
        if isinstance(element_type, ArrayType):
            self._require(typed_init.type, element_type, expr.loc)

        expected_func_type = FuncType((typed_init.type, element_type), typed_init.type)
        typed_func = self.check_callable(expr.func, expected_func_type, env)
        return TypedFold(
            expr,
            typed_func,
            typed_init,
            typed_array,
            typed_array.type.shape[0],
            typed_init.type,
        )

    def _infer_callable_type_for_map(
        self, expr: Expr, cell_type: RemoraType, env: TypeEnv
    ) -> FuncType:
        if isinstance(expr, (LeftSectionExpr, RightSectionExpr)):
            section_arg = expr.arg
            typed_arg = self.infer(section_arg, env)
            if expr.op == "/":
                self._require_numeric(cell_type, expr.loc)
                self._require_numeric(typed_arg.type, expr.loc)
                return FuncType((cell_type,), FLOAT)
            if expr.op in {"+", "-", "*"}:
                result_type = common_numeric_type(cell_type, typed_arg.type)
                return FuncType((cell_type,), result_type)
            if expr.op in {"==", "!=", "<", "<="}:
                # For now, allow any scalar elements that match
                # Wait, common_numeric_type would handle numeric promotion.
                # If one is bool and other is not, it might fail.
                if is_numeric(cell_type) and is_numeric(typed_arg.type):
                    return FuncType((cell_type,), BOOL)
                if cell_type == BOOL and typed_arg.type == BOOL:
                    return FuncType((cell_type,), BOOL)
                raise RemoraTypeError(f"operator {expr.op} expects matching types", expr.loc)
            if expr.op in {"&&", "||"}:
                self._require(cell_type, BOOL, expr.loc)
                self._require(typed_arg.type, BOOL, expr.loc)
                return FuncType((cell_type,), BOOL)
            raise RemoraTypeError("operator section expects supported operands", expr.loc)

        if isinstance(expr, OperatorFuncExpr):
            raise RemoraTypeError("binary operator function is not a unary map callable", expr.loc)

        if isinstance(expr, LambdaExpr):
            # Try the lambda against its expected input and use its inferred body type.
            inner_env = env
            if len(expr.params) != 1:
                raise RemoraTypeError("map expects a unary callable", expr.loc)
            inner_env = inner_env.extend(expr.params[0], cell_type)
            body = self.infer(expr.body, inner_env)
            return FuncType((cell_type,), body.type)

        if isinstance(expr, VarExpr) and expr.name in self._functions:
            function = self._functions[expr.name]
            if len(function.params) != 1:
                raise RemoraTypeError("map expects a unary callable", expr.loc)
            return self._infer_top_level_function_type(
                function,
                (cell_type,),
                env,
            )

        typed = self.infer(expr, env)
        if not isinstance(typed.type, FuncType) or len(typed.type.params) != 1:
            raise RemoraTypeError("map expects a unary callable", expr.loc)
        self._require(typed.type.params[0], cell_type, expr.loc)
        return typed.type

    def _check_operator_func(
        self, expr: OperatorFuncExpr, expected_type: FuncType
    ) -> TypedExpr:
        params = expected_type.params
        if expr.op in {"+", "-", "*"}:
            result = self._common_fold_operator_type(params[0], params[1], expr.loc)
            self._require(result, expected_type.result, expr.loc)
        elif expr.op == "/":
            self._require_numeric(params[0], expr.loc)
            self._require_numeric(params[1], expr.loc)
            self._require(expected_type.result, FLOAT, expr.loc)
        else:
            raise RemoraTypeError(f"operator {expr.op} is deferred", expr.loc)
        return TypedOperatorFunc(expr, expected_type)

    def _check_left_section(
        self, expr: LeftSectionExpr, expected_type: FuncType, env: TypeEnv
    ) -> TypedExpr:
        typed_arg = self.infer(expr.arg, env)
        expected_arg_type = expected_type.params[0]
        if expr.op in {"+", "-", "*"}:
            result = self._common_fold_operator_type(expected_arg_type, typed_arg.type, expr.loc)
            self._require(result, expected_type.result, expr.loc)
        elif expr.op == "/":
            self._require_numeric(expected_arg_type, expr.loc)
            self._require_numeric(typed_arg.type, expr.loc)
            self._require(expected_type.result, FLOAT, expr.loc)
        elif expr.op in {"==", "!=", "<", "<="}:
            if is_numeric(expected_arg_type) and is_numeric(typed_arg.type):
                self._require(expected_type.result, BOOL, expr.loc)
            elif expected_arg_type == BOOL and typed_arg.type == BOOL:
                self._require(expected_type.result, BOOL, expr.loc)
            else:
                 raise RemoraTypeError(f"operator {expr.op} expects matching types", expr.loc)
        elif expr.op in {"&&", "||"}:
            self._require(expected_arg_type, BOOL, expr.loc)
            self._require(typed_arg.type, BOOL, expr.loc)
            self._require(expected_type.result, BOOL, expr.loc)
        else:
            raise RemoraTypeError(f"operator {expr.op} section is deferred", expr.loc)
        return TypedLeftSection(expr, typed_arg, expected_type)

    def _check_right_section(
        self, expr: RightSectionExpr, expected_type: FuncType, env: TypeEnv
    ) -> TypedExpr:
        typed_arg = self.infer(expr.arg, env)
        expected_arg_type = expected_type.params[0]
        if expr.op in {"+", "-", "*"}:
            result = self._common_fold_operator_type(typed_arg.type, expected_arg_type, expr.loc)
            self._require(result, expected_type.result, expr.loc)
        elif expr.op == "/":
            self._require_numeric(typed_arg.type, expr.loc)
            self._require_numeric(expected_arg_type, expr.loc)
            self._require(expected_type.result, FLOAT, expr.loc)
        elif expr.op in {"==", "!=", "<", "<="}:
            if is_numeric(expected_arg_type) and is_numeric(typed_arg.type):
                self._require(expected_type.result, BOOL, expr.loc)
            elif expected_arg_type == BOOL and typed_arg.type == BOOL:
                self._require(expected_type.result, BOOL, expr.loc)
            else:
                 raise RemoraTypeError(f"operator {expr.op} expects matching types", expr.loc)
        elif expr.op in {"&&", "||"}:
            self._require(expected_arg_type, BOOL, expr.loc)
            self._require(typed_arg.type, BOOL, expr.loc)
            self._require(expected_type.result, BOOL, expr.loc)
        else:
            raise RemoraTypeError(f"operator {expr.op} section is deferred", expr.loc)
        return TypedRightSection(expr, typed_arg, expected_type)

    def _cell_type_candidates(self, value_type: RemoraType) -> list[RemoraType]:
        if isinstance(value_type, ScalarType):
            return [value_type]
        candidates: list[RemoraType] = [value_type.element]
        for rank in range(1, value_type.rank + 1):
            candidates.append(ArrayType(value_type.element, value_type.shape[-rank:]))
        return candidates

    def check_definition(
        self, definition: Definition, env: TypeEnv
    ) -> tuple[TypedDefinition, TypeEnv]:
        """Type-check a single definition and return it with the extended environment."""
        return self._check_definition(definition, env)

    def _check_definition(
        self, definition: Definition, env: TypeEnv
    ) -> tuple[TypedDefinition, TypeEnv]:
        if isinstance(definition, ValDef):
            typed_value = self.infer(definition.value, env)
            return TypedDefinition(definition, typed_value, typed_value.type), env.extend(
                definition.name, typed_value.type
            )
        if isinstance(definition, FuncDef):
            self._functions[definition.name] = definition
            return TypedDefinition(definition, None, None), env
        raise AssertionError(f"unknown definition type {type(definition).__name__}")

    def _infer_top_level_function_app(self, expr: AppExpr, env: TypeEnv) -> TypedExpr:
        function = self._functions[expr.func.name]
        if len(expr.args) != len(function.params):
            raise RemoraTypeError("function arity mismatch", expr.loc)
        typed_args = [self.infer(arg, env) for arg in expr.args]
        func_type = self._infer_top_level_function_type(
            function,
            tuple(arg.type for arg in typed_args),
            env,
        )
        typed_func = self._typed_top_level_function(function, func_type, env)
        typed_args = [
            self._coerce(arg, param_type, expr.loc)
            for arg, param_type in zip(typed_args, func_type.params)
        ]
        return TypedApp(expr, typed_func, typed_args, func_type.result)

    def infer_top_level_function_type(
        self,
        function: FuncDef,
        param_types: tuple[RemoraType, ...],
        env: TypeEnv,
    ) -> FuncType:
        """Infer the result type of a top-level function given concrete parameter types."""
        return self._infer_top_level_function_type(function, param_types, env)

    def _infer_top_level_function_type(
        self,
        function: FuncDef,
        param_types: tuple[RemoraType, ...],
        env: TypeEnv,
    ) -> FuncType:
        if len(function.params) != len(param_types):
            raise RemoraTypeError("function arity mismatch", function.loc)
        typed_func = self._typed_top_level_function(
            function,
            FuncType(param_types, INT),
            env,
            infer_result=True,
        )
        return typed_func.type

    def typed_top_level_function(
        self,
        function: FuncDef,
        func_type: FuncType,
        env: TypeEnv,
        *,
        infer_result: bool = False,
    ) -> TypedLambda:
        """Build a typed lambda for a top-level function given concrete parameter types."""
        return self._typed_top_level_function(function, func_type, env, infer_result=infer_result)

    def _typed_top_level_function(
        self,
        function: FuncDef,
        func_type: FuncType,
        env: TypeEnv,
        *,
        infer_result: bool = False,
    ) -> TypedLambda:
        if function.name in self._active_functions:
            raise RemoraTypeError("recursive function definitions are deferred", function.loc)
        self._active_functions.add(function.name)
        try:
            inner_env = env
            for name, param_type in zip(function.params, func_type.params):
                inner_env = inner_env.extend(name, param_type)
            typed_body = self.infer(function.body, inner_env)
            if infer_result:
                result_type = typed_body.type
                typed_result = typed_body
            else:
                typed_result = self._coerce(typed_body, func_type.result, function.loc)
                result_type = func_type.result
            inferred_type = FuncType(func_type.params, result_type)
            return TypedLambda(
                function,
                list(zip(function.params, func_type.params)),
                typed_result,
                inferred_type,
            )
        finally:
            self._active_functions.remove(function.name)

    def _infer_let_lambda(self, expr: LetExpr, env: TypeEnv) -> TypedExpr:
        if not isinstance(expr.value, LambdaExpr):
            raise AssertionError("_infer_let_lambda expects a lambda value")
        if not (
            isinstance(expr.body, AppExpr)
            and isinstance(expr.body.func, VarExpr)
            and expr.body.func.name == expr.name
        ):
            raise RemoraTypeError(
                "standalone lambda bindings are only supported for direct application",
                expr.loc,
            )
        if len(expr.body.args) != len(expr.value.params):
            raise RemoraTypeError("function arity mismatch", expr.body.loc)

        typed_args = [self.infer(arg, env) for arg in expr.body.args]
        param_types = tuple(arg.type for arg in typed_args)
        inner_env = env
        for name, param_type in zip(expr.value.params, param_types):
            inner_env = inner_env.extend(name, param_type)
        typed_lambda_body = self.infer(expr.value.body, inner_env)
        func_type = FuncType(param_types, typed_lambda_body.type)
        typed_lambda = TypedLambda(
            expr.value,
            list(zip(expr.value.params, param_types)),
            typed_lambda_body,
            func_type,
        )
        typed_body = TypedApp(
            expr.body,
            TypedExprNode(expr.body.func, func_type),
            typed_args,
            typed_lambda_body.type,
        )
        return TypedLet(expr, expr.name, typed_lambda, typed_body, typed_body.type)

    def _coerce(
        self, typed: TypedExpr, expected_type: RemoraType, loc
    ) -> TypedExpr:
        if typed.type == expected_type:
            return typed
        if typed.type == INT and expected_type == FLOAT:
            return TypedCast(typed, INT, FLOAT, FLOAT)
        raise RemoraTypeError(f"expected {expected_type}, got {typed.type}", loc)

    def _require(self, actual: RemoraType, expected: RemoraType, loc) -> None:
        if actual != expected:
            raise RemoraTypeError(f"expected {expected}, got {actual}", loc)

    def _require_shape_operand(
        self, value_type: RemoraType, operator: str, loc
    ) -> None:
        if isinstance(value_type, FuncType):
            raise RemoraTypeError(f"{operator} of function values is deferred", loc)

    def _require_numeric(self, value_type: RemoraType, loc) -> None:
        if not is_numeric(value_type):
            raise RemoraTypeError(f"expected numeric type, got {value_type}", loc)

    def _common_fold_operator_type(
        self,
        left: RemoraType,
        right: RemoraType,
        loc,
    ) -> RemoraType:
        if isinstance(left, ArrayType) or isinstance(right, ArrayType):
            if not isinstance(left, ArrayType) or not isinstance(right, ArrayType):
                raise RemoraTypeError(f"expected matching array operands, got {left} and {right}", loc)
            if left != right:
                raise RemoraTypeError(f"expected matching array operands, got {left} and {right}", loc)
            if not is_numeric(left.element):
                raise RemoraTypeError(f"expected numeric array elements, got {left.element}", loc)
            return left
        return common_numeric_type(left, right)

    def _build_prelude_env(self) -> TypeEnv:
        return TypeEnv()
