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
    IntLit,
    IotaExpr,
    LambdaExpr,
    LeftSectionExpr,
    LetExpr,
    MapExpr,
    OperatorFuncExpr,
    Program,
    RightSectionExpr,
    ValDef,
    VarExpr,
)
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
    type: RemoraType


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
    array: TypedExpr
    frame_shape: tuple[DimExpr, ...]
    cell_shape: tuple[DimExpr, ...]
    type: RemoraType


@dataclass(frozen=True)
class TypedFold:
    expr: FoldExpr
    func: TypedExpr
    init: TypedExpr
    array: TypedExpr
    reduction_dim: DimExpr
    type: RemoraType


@dataclass(frozen=True)
class TypedLambda:
    expr: LambdaExpr
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


TypedExpr: TypeAlias = (
    TypedExprNode
    | TypedCast
    | TypedArray
    | TypedMap
    | TypedFold
    | TypedLambda
    | TypedOperatorFunc
    | TypedLeftSection
    | TypedRightSection
    | TypedApp
    | TypedLet
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
    def check_program(self, program: Program) -> TypedProgram:
        env = self._build_prelude_env()
        typed_definitions: list[TypedDefinition] = []

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
        if isinstance(expr, LetExpr):
            typed_value = self.infer(expr.value, env)
            inner_env = env.extend(expr.name, typed_value.type)
            typed_body = self.infer(expr.body, inner_env)
            return TypedLet(expr, expr.name, typed_value, typed_body, typed_body.type)
        if isinstance(expr, IfExpr):
            condition = self.infer(expr.condition, env)
            self._require(condition.type, BOOL, expr.loc)
            then_branch = self.infer(expr.then_branch, env)
            else_branch = self.infer(expr.else_branch, env)
            self._require(then_branch.type, else_branch.type, expr.loc)
            return TypedExprNode(expr, then_branch.type)
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
        if isinstance(expr.func, VarExpr) and expr.func.name in _INFIX_OPERATORS:
            return self._infer_primitive_app(expr, env)

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
            result_type = common_numeric_type(left.type, right.type)
            return TypedApp(
                expr,
                TypedExprNode(expr.func, FuncType((result_type, result_type), result_type)),
                [
                    self._coerce(left, result_type, expr.loc),
                    self._coerce(right, result_type, expr.loc),
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
                    typed_array,
                    frame_shape,
                    cell_shape,
                    result_type,
                )
            except RemoraTypeError as exc:
                errors.append(exc)

        raise RemoraTypeError(f"could not type-check map callable: {errors[-1]}", expr.loc)

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
            raise RemoraTypeError("operator section expects numeric operands", expr.loc)

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
            result = common_numeric_type(expected_arg_type, typed_arg.type)
            self._require(result, expected_type.result, expr.loc)
        elif expr.op == "/":
            self._require_numeric(expected_arg_type, expr.loc)
            self._require_numeric(typed_arg.type, expr.loc)
            self._require(expected_type.result, FLOAT, expr.loc)
        else:
            raise RemoraTypeError(f"operator {expr.op} section is deferred", expr.loc)
        return TypedLeftSection(expr, typed_arg, expected_type)

    def _check_right_section(
        self, expr: RightSectionExpr, expected_type: FuncType, env: TypeEnv
    ) -> TypedExpr:
        typed_arg = self.infer(expr.arg, env)
        expected_arg_type = expected_type.params[0]
        if expr.op in {"+", "-", "*"}:
            result = common_numeric_type(typed_arg.type, expected_arg_type)
            self._require(result, expected_type.result, expr.loc)
        elif expr.op == "/":
            self._require_numeric(typed_arg.type, expr.loc)
            self._require_numeric(expected_arg_type, expr.loc)
            self._require(expected_type.result, FLOAT, expr.loc)
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

    def _check_definition(
        self, definition: Definition, env: TypeEnv
    ) -> tuple[TypedDefinition, TypeEnv]:
        if isinstance(definition, ValDef):
            typed_value = self.infer(definition.value, env)
            return TypedDefinition(definition, typed_value, typed_value.type), env.extend(
                definition.name, typed_value.type
            )
        if isinstance(definition, FuncDef):
            raise RemoraTypeError(
                "function definition type inference requires annotations and is deferred",
                definition.loc,
            )
        raise AssertionError(f"unknown definition type {type(definition).__name__}")

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


_INFIX_OPERATORS = {"+", "-", "*", "/", "<", "<=", "==", "!=", "&&", "||"}
