"""Small Dense Core type checker for the parser AST."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TypeAlias

from remora.ast_nodes import (
    AppExpr,
    AppendExpr,
    ArrayLit,
    BoolLit,
    BoxesExpr,
    BoxExpr,
    Definition,
    Expr,
    FloatLit,
    FoldExpr,
    FoldRightExpr,
    FuncDef,
    GradeExpr,
    FilterExpr,
    IfExpr,
    IndexAppExpr,
    IndexExpr,
    IndicesOfExpr,
    IntLit,
    Iota1Expr,
    IotaExpr,
    IotaNExpr,
    LambdaExpr,
    LeftSectionExpr,
    LengthExpr,
    LetExpr,
    MapExpr,
    OperatorFuncExpr,
    Program,
    RankExpr,
    RavelExpr,
    ReduceExpr,
    RerankExpr,
    ReplicateExpr,
    ReshapeExpr,
    RightSectionExpr,
    RotateExpr,
    ScanExpr,
    SelectExpr,
    ShapeExpr,
    SliceRange,
    SortExpr,
    SubarrayExpr,
    ReverseExpr,
    TakeExpr,
    TraceExpr,
    TransposeExpr,
    DropExpr,
    UnboxExpr,
    ValDef,
    VarExpr,
    WithShapeExpr,
)
from remora.constraints import ConstraintError, match_shape_template
from remora.dependent_types import (
    free_type_index_vars,
    instantiate_pi_type,
    substitute_type,
)
from remora.frame import (
    apply_frame,
    cell_matches_array_suffix,
    cell_type_candidates,
    principal_frame,
    scalar_cell_and_frame,
    infer_lifting as frame_infer_lifting,
)
from remora.index import IndexBinder, IndexSort
from remora.operators import ALL_PRIMITIVE_OPS
from remora.types import (
    BOOL,
    FLOAT,
    INT,
    ArrayType,
    DimExpr,
    FuncType,
    PiType,
    RemoraType,
    RemoraTypeError,
    ScalarType,
    SigmaType,
    StaticDim,
    common_numeric_type,
    enforce_rank_limit,
    eval_static_dim,
    infer_lifting,
    is_numeric,
)


@dataclass(frozen=True)
class TypedProgram:
    """A fully typed program with top-level definitions and an optional body."""
    definitions: list[TypedDefinition]
    body: TypedExpr | None
    type: RemoraType | None


@dataclass(frozen=True)
class TypedDefinition:
    """A typed top-level definition with an optional value and type."""
    definition: Definition
    value: TypedExpr | None
    type: RemoraType | None


@dataclass(frozen=True)
class TypedExprNode:
    """Leaf typed expression wrapping an AST node and its inferred type."""
    expr: Expr
    type: RemoraType


@dataclass(frozen=True)
class TypedCast:
    """A typed integer-to-float numeric coercion."""
    value: TypedExpr
    from_type: ScalarType
    to_type: ScalarType
    type: ScalarType


@dataclass(frozen=True)
class TypedArray:
    """A typed array literal with recursively typed elements."""
    expr: ArrayLit
    elements: list[TypedExpr]
    type: ArrayType


@dataclass(frozen=True)
class TypedMap:
    """A typed map (lifted) application over one or two arrays."""
    expr: Expr
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
    """A typed fold (prefix reduction) over an array."""
    expr: FoldExpr
    func: TypedExpr
    init: TypedExpr
    array: TypedExpr
    reduction_dim: DimExpr
    type: RemoraType


@dataclass(frozen=True)
class TypedFoldRight:
    """A typed right-to-left fold reduction."""
    expr: FoldRightExpr
    func: TypedExpr
    init: TypedExpr
    array: TypedExpr
    reduction_dim: DimExpr
    type: RemoraType


@dataclass(frozen=True)
class TypedScan:
    """A typed scan (prefix-sum) operation."""
    expr: ScanExpr | TraceExpr
    func: TypedExpr
    init: TypedExpr
    array: TypedExpr
    reduction_dim: DimExpr
    exclusive: bool
    right: bool
    type: RemoraType


@dataclass(frozen=True)
class TypedShape:
    """A typed shape operator applied to an array."""
    expr: ShapeExpr
    array: TypedExpr
    type: ArrayType


@dataclass(frozen=True)
class TypedRank:
    """A typed rank operator returning the dimensionality of an array."""
    expr: RankExpr
    array: TypedExpr
    type: ScalarType


@dataclass(frozen=True)
class TypedLength:
    """A typed length operator returning the size of the leading dimension."""
    expr: LengthExpr
    array: TypedExpr
    dim: DimExpr
    type: ScalarType


@dataclass(frozen=True)
class TypedTranspose:
    """A typed transpose swapping the first two axes of an array."""
    expr: TransposeExpr
    array: TypedExpr
    type: ArrayType


@dataclass(frozen=True)
class TypedReshape:
    """A typed reshape giving an array a new shape with the same total size."""
    expr: ReshapeExpr
    shape_expr: TypedExpr
    array: TypedExpr
    type: ArrayType


@dataclass(frozen=True)
class TypedRavel:
    """A typed ravel (flatten) reducing an array to a single dimension."""
    expr: RavelExpr
    array: TypedExpr
    type: ArrayType


@dataclass(frozen=True)
class TypedReverse:
    """A typed reverse along the first axis of an array."""
    expr: ReverseExpr
    array: TypedExpr
    type: ArrayType


@dataclass(frozen=True)
class TypedTake:
    """A typed take (prefix truncation) of an array along its first axis."""
    expr: TakeExpr
    count: TypedExpr
    array: TypedExpr
    type: ArrayType


@dataclass(frozen=True)
class TypedDrop:
    """A typed drop (prefix removal) from an array along its first axis."""
    expr: DropExpr
    count: TypedExpr
    array: TypedExpr
    type: ArrayType


@dataclass(frozen=True)
class TypedSlice:
    """A typed slice range within an indexing expression."""
    expr: SliceRange
    start: TypedExpr | None
    end: TypedExpr | None
    type: ArrayType


@dataclass(frozen=True)
class TypedIndex:
    """A typed indexing expression on an array with scalar and/or slice indices."""
    expr: IndexExpr
    array: TypedExpr
    indices: list[TypedExpr | TypedSlice]
    type: RemoraType


@dataclass(frozen=True)
class TypedLambda:
    """A typed lambda expression or top-level function definition."""
    expr: LambdaExpr | FuncDef
    params: list[tuple[str, RemoraType]]
    body: TypedExpr
    type: FuncType
    specialization_name: str | None = None
    index_args: tuple[DimExpr, ...] = ()


@dataclass(frozen=True)
class TypedOperatorFunc:
    """A typed operator function value with a binary function type."""
    expr: OperatorFuncExpr
    type: FuncType


@dataclass(frozen=True)
class TypedLeftSection:
    """A typed left operator section with a partially applied argument."""
    expr: LeftSectionExpr
    arg: TypedExpr
    type: FuncType


@dataclass(frozen=True)
class TypedRightSection:
    """A typed right operator section with a partially applied argument."""
    expr: RightSectionExpr
    arg: TypedExpr
    type: FuncType


@dataclass(frozen=True)
class TypedApp:
    """A typed function application with typed arguments."""
    expr: AppExpr
    func: TypedExpr
    args: list[TypedExpr]
    type: RemoraType


@dataclass(frozen=True)
class TypedIndexApp:
    """A Pi-typed function specialized at explicit compile-time indices."""
    expr: IndexAppExpr
    function: TypedLambda
    index_args: tuple[DimExpr, ...]
    type: FuncType


@dataclass(frozen=True)
class TypedLet:
    """A typed let-expression binding a name to a value within a body."""
    expr: LetExpr
    name: str
    value: TypedExpr
    body: TypedExpr
    type: RemoraType


@dataclass(frozen=True)
class TypedIf:
    """A typed conditional expression with then and else branches."""
    expr: IfExpr | SelectExpr
    condition: TypedExpr
    then_branch: TypedExpr
    else_branch: TypedExpr
    type: RemoraType


@dataclass(frozen=True)
class TypedAppend:
    """A typed append (concatenation) of two arrays along the leading dimension."""
    expr: AppendExpr
    left: TypedExpr
    right: TypedExpr
    type: ArrayType


@dataclass(frozen=True)
class TypedRotate:
    """A typed circular rotation of an array along the leading dimension."""
    expr: RotateExpr
    array: TypedExpr
    shift: DimExpr
    type: ArrayType


@dataclass(frozen=True)
class TypedSubarray:
    """A typed subarray extraction."""
    expr: SubarrayExpr
    array: TypedExpr
    offsets: tuple[DimExpr, ...]
    sizes: tuple[DimExpr, ...]
    type: ArrayType


@dataclass(frozen=True)
class TypedIndicesOf:
    """A typed indices-of expression."""
    expr: IndicesOfExpr
    array: TypedExpr
    type: ArrayType


@dataclass(frozen=True)
class TypedWithShape:
    """A typed with-shape (broadcast replication) expression."""
    expr: WithShapeExpr
    source: TypedExpr
    type: ArrayType


@dataclass(frozen=True)
class TypedSort:
    """A typed sort expression."""
    expr: SortExpr
    array: TypedExpr
    type: ArrayType


@dataclass(frozen=True)
class TypedGrade:
    """A typed grade (argsort) expression."""
    expr: GradeExpr
    array: TypedExpr
    type: ArrayType


@dataclass(frozen=True)
class TypedFilter:
    """A typed filter expression with boxed result."""
    expr: FilterExpr
    predicate: TypedExpr
    array: TypedExpr
    type: SigmaType


@dataclass(frozen=True)
class TypedReplicate:
    """A typed replicate expression with boxed result."""
    expr: ReplicateExpr
    counts: TypedExpr
    array: TypedExpr
    type: SigmaType


@dataclass(frozen=True)
class TypedBox:
    """A typed box expression wrapping an array with a hidden dimension."""
    expr: BoxExpr
    value: TypedExpr
    type: SigmaType


@dataclass(frozen=True)
class TypedUnbox:
    """A typed unbox expression opening an existential type."""
    expr: UnboxExpr
    box_value: TypedExpr
    hidden_names: list[str]
    value_name: str
    body: TypedExpr
    type: RemoraType


TypedExpr: TypeAlias = (
    TypedExprNode
    | TypedCast
    | TypedArray
    | TypedMap
    | TypedFold
    | TypedFoldRight
    | TypedScan
    | TypedShape
    | TypedRank
    | TypedLength
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
    | TypedIndexApp
    | TypedAppend
    | TypedRotate
    | TypedSubarray
    | TypedIndicesOf
    | TypedWithShape
    | TypedBox
    | TypedUnbox
    | TypedSort
    | TypedGrade
    | TypedFilter
    | TypedReplicate
    | TypedLet
    | TypedIf
)


class TypeEnv:
    """Immutable environment with separate value and compile-time index namespaces."""

    def __init__(
        self,
        bindings: dict[str, RemoraType] | None = None,
        index_bindings: dict[str, IndexSort] | None = None,
    ):
        """Create a type environment with optional initial variable bindings."""
        self._bindings = dict(bindings or {})
        self._index_bindings = dict(index_bindings or {})

    def extend(self, name: str, value_type: RemoraType) -> TypeEnv:
        """Return a new environment with the given name-type binding added."""
        return TypeEnv(
            {**self._bindings, name: value_type},
            self._index_bindings,
        )

    def extend_index(self, binder: IndexBinder) -> TypeEnv:
        """Return an environment extended with one compile-time index binder."""
        if binder.name in self._index_bindings:
            raise RemoraTypeError(f"duplicate index binder {binder.name!r}")
        return TypeEnv(
            self._bindings,
            {**self._index_bindings, binder.name: binder.sort},
        )

    def lookup_index(self, name: str) -> IndexSort:
        """Return the sort of a compile-time index variable."""
        try:
            return self._index_bindings[name]
        except KeyError as exc:
            raise RemoraTypeError(f"unbound index variable '{name}'") from exc

    def lookup(self, name: str) -> RemoraType:
        """Return the type bound to a variable name, or raise an error."""
        try:
            return self._bindings[name]
        except KeyError as exc:
            raise RemoraTypeError(f"unbound variable '{name}'") from exc


class TypeChecker:
    """Dense Core type checker that infers types for expressions and programs."""

    def __init__(self) -> None:
        """Create a new type checker with empty function registries."""
        self._functions: dict[str, FuncDef] = {}
        self._active_functions: set[str] = set()
        self._specializations: dict[
            tuple[str, tuple[DimExpr, ...]], TypedLambda
        ] = {}

    def check_program(self, program: Program) -> TypedProgram:
        """Type-check an entire program and return a typed program."""
        env = self._build_prelude_env()
        typed_definitions: list[TypedDefinition] = []
        self._functions = {
            definition.name: definition
            for definition in program.definitions
            if isinstance(definition, FuncDef)
        }
        self._active_functions = set()
        self._specializations = {}

        for definition in program.definitions:
            typed_definition, env = self._check_definition(definition, env)
            typed_definitions.append(typed_definition)

        if program.body is None:
            return TypedProgram(typed_definitions, None, None)

        typed_body = self.infer(program.body, env)
        return TypedProgram(typed_definitions, typed_body, typed_body.type)

    def infer(self, expr: Expr, env: TypeEnv | None = None) -> TypedExpr:
        """Infer the type of an expression and return a typed expression."""
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
        if isinstance(expr, LengthExpr):
            typed_array = self.infer(expr.array, env)
            self._require_shape_operand(typed_array.type, "length", expr.loc)
            if isinstance(typed_array.type, ArrayType):
                dim = typed_array.type.shape[0]
            else:
                raise RemoraTypeError("length expects an array operand", expr.loc)
            return TypedLength(expr, typed_array, dim, INT)
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
        if isinstance(expr, SelectExpr):
            condition = self.infer(expr.condition, env)
            then_branch = self.infer(expr.then_branch, env)
            else_branch = self.infer(expr.else_branch, env)
            if isinstance(condition.type, ArrayType):
                if condition.type.element != BOOL:
                    raise RemoraTypeError("select condition array must have boolean elements", expr.loc)
            elif condition.type != BOOL:
                self._require(condition.type, BOOL, expr.loc)
            self._require(then_branch.type, else_branch.type, expr.loc)
            return TypedIf(expr, condition, then_branch, else_branch, then_branch.type)  # type: ignore[arg-type]
        if isinstance(expr, AppendExpr):
            left = self.infer(expr.left, env)
            right = self.infer(expr.right, env)
            if not isinstance(left.type, ArrayType) or not isinstance(right.type, ArrayType):
                raise RemoraTypeError("append expects two arrays", expr.loc)
            if left.type.element != right.type.element:
                raise RemoraTypeError(f"append expects matching element types, got {left.type.element} and {right.type.element}", expr.loc)
            if left.type.shape[1:] != right.type.shape[1:]:
                raise RemoraTypeError(f"append expects matching non-leading dimensions", expr.loc)
            new_leading = StaticDim(left.type.shape[0].value + right.type.shape[0].value)
            result_shape = (new_leading,) + left.type.shape[1:]
            return TypedAppend(expr, left, right, ArrayType(left.type.element, result_shape))
        if isinstance(expr, RotateExpr):
            typed_array = self.infer(expr.array, env)
            if not isinstance(typed_array.type, ArrayType):
                raise RemoraTypeError("rotate expects an array", expr.loc)
            shift_dim = eval_static_dim(expr.shift, expr.loc)
            return TypedRotate(expr, typed_array, shift_dim, typed_array.type)
        if isinstance(expr, RerankExpr):
            return self._infer_rerank(expr, env)
        if isinstance(expr, SubarrayExpr):
            return self._infer_subarray(expr, env)
        if isinstance(expr, IndicesOfExpr):
            return self._infer_indices_of(expr, env)
        if isinstance(expr, WithShapeExpr):
            return self._infer_with_shape(expr, env)
        if isinstance(expr, BoxExpr):
            return self._infer_box(expr, env)
        if isinstance(expr, UnboxExpr):
            return self._infer_unbox(expr, env)
        if isinstance(expr, BoxesExpr):
            return self._infer_boxes(expr, env)
        if isinstance(expr, Iota1Expr):
            return self._infer_iota1(expr, env)
        if isinstance(expr, IotaNExpr):
            return self._infer_iotan(expr, env)
        if isinstance(expr, FilterExpr):
            return self._infer_filter(expr, env)
        if isinstance(expr, ReplicateExpr):
            return self._infer_replicate(expr, env)
        if isinstance(expr, SortExpr):
            return self._infer_sort(expr, env)
        if isinstance(expr, GradeExpr):
            return self._infer_grade(expr, env)
        if isinstance(expr, AppExpr):
            return self._infer_app(expr, env)
        if isinstance(expr, IndexAppExpr):
            return self._infer_index_app(expr, env)
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
        if isinstance(expr, ReduceExpr):
            return self._infer_reduce(expr, env)
        if isinstance(expr, FoldRightExpr):
            return self._infer_fold_right(expr, env)
        if isinstance(expr, ScanExpr):
            return self._infer_scan(expr, env)
        if isinstance(expr, TraceExpr):
            return self._infer_trace(expr, env)
        if isinstance(expr, ReverseExpr):
            return self._infer_reverse(expr, env)

        raise RemoraTypeError(f"type checking for {type(expr).__name__} is deferred")

    def check_callable(
        self, expr: Expr, expected_type: FuncType, env: TypeEnv
    ) -> TypedExpr:
        """Type-check a callable expression against an expected function type."""
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

        if isinstance(expr, RerankExpr):
            return self._check_rerank(expr, expected_type, env)

        if isinstance(expr, VarExpr) and expr.name in self._functions:
            return self._typed_top_level_function(
                self._functions[expr.name],
                expected_type,
                env,
            )

        if isinstance(expr, VarExpr) and expr.name in ALL_PRIMITIVE_OPS:
            if len(expected_type.params) != 2:
                raise RemoraTypeError(
                    f"primitive operator {expr.name} must be binary", expr.loc
                )
            op_expr = OperatorFuncExpr(expr.name, expr.loc)
            return self._check_operator_func(op_expr, expected_type)

        typed = self.infer(expr, env)
        self._require(typed.type, expected_type, expr.loc)
        return typed

    def _check_rerank(
        self, expr: RerankExpr, expected_type: FuncType, env: TypeEnv
    ) -> TypedExpr:
        """Check ~(r1 r2) f → desugared lambda against expected type."""
        n = len(expr.ranks)
        if n != len(expected_type.params):
            raise RemoraTypeError(
                f"reranking expects {n} params but callable expects {len(expected_type.params)}",
                expr.loc,
            )
        param_names = [f"__r{i}" for i in range(n)]

        inner_env = env
        for name, param_type in zip(param_names, expected_type.params):
            inner_env = inner_env.extend(name, param_type)

        app_expr = AppExpr(
            expr.func,
            [VarExpr(name, expr.loc) for name in param_names],
            expr.loc,
        )
        typed_body = self.infer(app_expr, inner_env)
        typed_body = self._coerce(typed_body, expected_type.result, expr.loc)

        return TypedLambda(
            LambdaExpr(param_names, app_expr, expr.loc, param_ranks=expr.ranks),
            list(zip(param_names, expected_type.params)),
            typed_body,
            expected_type,
        )

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
        # Implicit rank-polymorphic map: for primitive ops, lambdas, and
        # named functions with cell-rank annotations, attempt to lift
        # array applications to a map.
        is_primitive = isinstance(expr.func, VarExpr) and expr.func.name in ALL_PRIMITIVE_OPS
        is_lambda = isinstance(expr.func, LambdaExpr)
        is_ranked_func = (
            isinstance(expr.func, VarExpr)
            and expr.func.name in self._functions
            and getattr(self._functions[expr.func.name], "param_ranks", None) is not None
        )
        if is_primitive or is_lambda or is_ranked_func:
            typed_args = [self.infer(arg, env) for arg in expr.args]
            if any(isinstance(a.type, ArrayType) for a in typed_args):
                result = self._try_implicit_map(expr, typed_args, env)
                if result is not None:
                    return result

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

    def _infer_index_app(self, expr: IndexAppExpr, env: TypeEnv) -> TypedIndexApp:
        if not isinstance(expr.func, VarExpr) or expr.func.name not in self._functions:
            raise RemoraTypeError(
                "explicit index application requires a named top-level function",
                expr.loc,
            )
        function = self._functions[expr.func.name]
        declared_type = self._declared_function_type(function)
        if not isinstance(declared_type, PiType):
            raise RemoraTypeError(
                f"function {function.name!r} does not have a Pi type",
                expr.loc,
            )
        for arg in expr.args:
            if not isinstance(arg, StaticDim):
                raise RemoraTypeError(
                    "Phase 7a explicit index arguments must be dimension literals",
                    expr.loc,
                )
        try:
            instantiated = instantiate_pi_type(declared_type, expr.args)
        except ValueError as exc:
            raise RemoraTypeError(str(exc), expr.loc) from exc
        if not isinstance(instantiated, FuncType):
            raise RemoraTypeError(
                "explicit index application did not produce a function type",
                expr.loc,
            )
        typed_function = self._typed_top_level_function(
            function,
            instantiated,
            env,
            index_args=expr.args,
        )
        return TypedIndexApp(expr, typed_function, expr.args, instantiated)

    def _try_implicit_map(
        self, expr: AppExpr, typed_args: list[TypedExpr], env: TypeEnv
    ) -> TypedExpr | None:
        """Attempt implicit rank-polymorphic map lifting for array arguments."""

        if len(typed_args) == 2:
            return self._try_implicit_binary_map(expr, typed_args, env)
        if len(typed_args) == 1:
            return self._try_implicit_unary_map(expr, typed_args, env)
        return None

    def _try_implicit_unary_map(
        self, expr: AppExpr, typed_args: list[TypedExpr], env: TypeEnv
    ) -> TypedExpr | None:
        typed_array = typed_args[0]
        if not isinstance(typed_array.type, ArrayType):
            return None

        candidates = self._cell_type_candidates(typed_array.type)
        for cell_type in candidates:
            try:
                func_type = self._infer_callable_type_for_map(
                    expr.func, cell_type, env
                )
                typed_func = self.check_callable(expr.func, func_type, env)
                frame_shape, result_type = frame_infer_lifting(func_type, typed_array.type)
                cell_shape = cell_type.shape if isinstance(cell_type, ArrayType) else ()
                # If frame_shape is empty and cell_shape matches the full array,
                # this is a direct application – no implicit map needed.
                if not frame_shape and cell_shape == typed_array.type.shape:
                    return None
                return TypedMap(
                    expr,
                    typed_func,
                    typed_args,
                    frame_shape,
                    cell_shape,
                    result_type,
                )
            except RemoraTypeError:
                continue
        return None

    def _try_implicit_binary_map(
        self, expr: AppExpr, typed_args: list[TypedExpr], env: TypeEnv
    ) -> TypedExpr | None:
        left = typed_args[0]
        right = typed_args[1]
        if not isinstance(left.type, ArrayType) and not isinstance(right.type, ArrayType):
            return None

        left_cell, left_frame = self._scalar_cell_and_frame(left.type, expr.loc)
        right_cell, right_frame = self._scalar_cell_and_frame(right.type, expr.loc)

        # Principal-frame broadcasting: determine the principal (longest) frame.
        # Shorter frames get their cells replicated to match the principal shape.
        princ_frame = self._principal_frame(left_frame, right_frame, expr.loc)
        if princ_frame is None:
            return None

        try:
            func_type = self._infer_binary_map_callable_type(
                expr.func, left_cell, right_cell, env
            )
            typed_func = self.check_callable(expr.func, func_type, env)
            result_type = func_type.result
            if princ_frame:
                if isinstance(result_type, FuncType):
                    return None
                if isinstance(result_type, ArrayType):
                    return None
                result_type = apply_frame(result_type, princ_frame)
            return TypedMap(
                expr, typed_func, typed_args, princ_frame, (), result_type,
            )
        except RemoraTypeError:
            return None

    def _principal_frame(
        self,
        left_frame: tuple[DimExpr, ...],
        right_frame: tuple[DimExpr, ...],
        loc,
    ) -> tuple[DimExpr, ...] | None:
        return principal_frame([left_frame, right_frame], loc)

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
        if op in {"<", "<=", ">", ">=", "==", "!="}:
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
                frame_shape, result_type = frame_infer_lifting(expected_func_type, typed_array.type)
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
            result_type = apply_frame(result_type, frame_shape)
        return TypedMap(expr, typed_func, [left, right], frame_shape, (), result_type)

    def _scalar_cell_and_frame(
        self, value_type: RemoraType, loc
    ) -> tuple[ScalarType, tuple[DimExpr, ...]]:
        return scalar_cell_and_frame(value_type, loc)

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

        if isinstance(expr, VarExpr) and expr.name in ALL_PRIMITIVE_OPS:
            return self._infer_binary_callable_type_for_primitive(
                expr.name, left_cell, right_cell, expr.loc
            )

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

    def _infer_binary_callable_type_for_primitive(
        self, op: str, left_cell: ScalarType, right_cell: ScalarType, loc
    ) -> FuncType:
        if op in {"+", "-", "*"}:
            result = common_numeric_type(left_cell, right_cell)
            return FuncType((result, result), result)
        if op == "/":
            self._require_numeric(left_cell, loc)
            self._require_numeric(right_cell, loc)
            return FuncType((FLOAT, FLOAT), FLOAT)
        if op in {"<", "<=", ">", ">=", "==", "!="}:
            result = common_numeric_type(left_cell, right_cell)
            return FuncType((result, result), BOOL)
        if op in {"&&", "||"}:
            self._require(left_cell, BOOL, loc)
            self._require(right_cell, BOOL, loc)
            return FuncType((BOOL, BOOL), BOOL)
        raise RemoraTypeError(f"operator {op} cannot be used in binary map", loc)

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

    def _infer_reduce(self, expr: ReduceExpr, env: TypeEnv) -> TypedFold:
        typed_init = self.infer(expr.init, env)
        typed_array = self.infer(expr.array, env)
        if not isinstance(typed_array.type, ArrayType) or typed_array.type.rank < 1:
            raise RemoraTypeError("reduce expects a non-scalar array", expr.loc)
        if expr.require_nonempty and isinstance(typed_array.type.shape[0], StaticDim) and typed_array.type.shape[0].value == 0:
            raise RemoraTypeError("reduce/1 expects a non-empty leading dimension", expr.loc)

        element_type = typed_array.type.drop_outer(1)
        if isinstance(element_type, ArrayType):
            self._require(typed_init.type, element_type, expr.loc)

        expected_func_type = FuncType((typed_init.type, element_type), typed_init.type)
        typed_func = self.check_callable(expr.func, expected_func_type, env)
        return TypedFold(
            expr,  # type: ignore[arg-type]
            typed_func,
            typed_init,
            typed_array,
            typed_array.type.shape[0],
            typed_init.type,
        )

    def _infer_fold_right(self, expr: FoldRightExpr, env: TypeEnv) -> TypedFoldRight:
        typed_init = self.infer(expr.init, env)
        typed_array = self.infer(expr.array, env)
        if not isinstance(typed_array.type, ArrayType) or typed_array.type.rank < 1:
            raise RemoraTypeError("fold-right expects a non-scalar array", expr.loc)

        element_type = typed_array.type.drop_outer(1)
        if isinstance(element_type, ArrayType):
            self._require(typed_init.type, element_type, expr.loc)

        expected_func_type = FuncType((element_type, typed_init.type), typed_init.type)
        typed_func = self.check_callable(expr.func, expected_func_type, env)
        return TypedFoldRight(
            expr,
            typed_func,
            typed_init,
            typed_array,
            typed_array.type.shape[0],
            typed_init.type,
        )

    def _infer_scan(self, expr: ScanExpr, env: TypeEnv) -> TypedScan:
        typed_init = self.infer(expr.init, env)
        typed_array = self.infer(expr.array, env)
        if not isinstance(typed_array.type, ArrayType) or typed_array.type.rank < 1:
            raise RemoraTypeError("scan expects a non-scalar array", expr.loc)

        element_type = typed_array.type.drop_outer(1)
        if isinstance(element_type, ArrayType):
            self._require(typed_init.type, element_type, expr.loc)

        expected_func_type = FuncType((typed_init.type, element_type), typed_init.type)
        typed_func = self.check_callable(expr.func, expected_func_type, env)
        return TypedScan(
            expr,
            typed_func,
            typed_init,
            typed_array,
            typed_array.type.shape[0],
            expr.exclusive,
            False,  # right
            typed_array.type,
        )

    def _infer_trace(self, expr: TraceExpr, env: TypeEnv) -> TypedScan:
        typed_init = self.infer(expr.init, env)
        typed_array = self.infer(expr.array, env)
        if not isinstance(typed_array.type, ArrayType) or typed_array.type.rank < 1:
            raise RemoraTypeError("trace expects a non-scalar array", expr.loc)

        element_type = typed_array.type.drop_outer(1)
        if isinstance(element_type, ArrayType):
            self._require(typed_init.type, element_type, expr.loc)

        expected_func_type = FuncType((typed_init.type, element_type), typed_init.type)
        typed_func = self.check_callable(expr.func, expected_func_type, env)
        return TypedScan(
            expr,  # type: ignore[arg-type]
            typed_func,
            typed_init,
            typed_array,
            typed_array.type.shape[0],
            False,  # exclusive
            expr.right,  # right
            typed_array.type,
        )

    def _infer_rerank(self, expr: RerankExpr, env: TypeEnv) -> TypedExpr:
        """Desugar ~(r1 r2) f → lambda with rank-annotated params."""
        n = len(expr.ranks)
        param_names = [f"__r{i}" for i in range(n)]
        param_ranks = expr.ranks

        lambda_expr = LambdaExpr(
            param_names,
            AppExpr(
                expr.func,
                [VarExpr(name, expr.loc) for name in param_names],
                expr.loc,
            ),
            expr.loc,
            param_ranks=param_ranks if any(r != 0 for r in param_ranks) else None,
        )

        return TypedExprNode(
            lambda_expr,
            FuncType(tuple(INT for _ in range(n)), INT),
        )

    def _infer_subarray(self, expr: SubarrayExpr, env: TypeEnv) -> TypedExpr:
        typed_array = self.infer(expr.array, env)
        if not isinstance(typed_array.type, ArrayType):
            raise RemoraTypeError("subarray expects an array", expr.loc)
        if len(expr.offsets) != typed_array.type.rank:
            raise RemoraTypeError(
                f"subarray offset count {len(expr.offsets)} != rank {typed_array.type.rank}",
                expr.loc,
            )
        if len(expr.shape) != typed_array.type.rank:
            raise RemoraTypeError(
                f"subarray shape count {len(expr.shape)} != rank {typed_array.type.rank}",
                expr.loc,
            )
        offsets = tuple(eval_static_dim(o, expr.loc) for o in expr.offsets)
        sizes = tuple(eval_static_dim(s, expr.loc) for s in expr.shape)
        result_type = ArrayType(typed_array.type.element, sizes)
        return TypedSubarray(expr, typed_array, offsets, sizes, result_type)

    def _infer_indices_of(self, expr: IndicesOfExpr, env: TypeEnv) -> TypedExpr:
        typed_array = self.infer(expr.array, env)
        if not isinstance(typed_array.type, ArrayType):
            raise RemoraTypeError("indices-of expects an array", expr.loc)
        rank = StaticDim(typed_array.type.rank)
        result_shape = (rank,) + typed_array.type.shape
        return TypedIndicesOf(expr, typed_array, ArrayType(INT, result_shape))

    def _infer_with_shape(self, expr: WithShapeExpr, env: TypeEnv) -> TypedExpr:
        typed_target = self.infer(expr.target, env)
        typed_shape = self.infer(expr.shape, env)
        if not isinstance(typed_shape.type, ArrayType) or typed_shape.type.element != INT:
            raise RemoraTypeError("with-shape expects an integer shape vector", expr.loc)
        shape_dims = tuple(
            StaticDim(int(e.value)) if isinstance(e, IntLit) else StaticDim(0)
            for e in (expr.shape.elements if isinstance(expr.shape, ArrayLit) else [expr.shape])
        )
        if isinstance(typed_target.type, ScalarType):
            result_type = ArrayType(typed_target.type, shape_dims)
        elif isinstance(typed_target.type, ArrayType):
            result_type = ArrayType(typed_target.type.element, shape_dims + typed_target.type.shape)
        else:
            raise RemoraTypeError("with-shape expects a scalar or array target", expr.loc)
        return TypedWithShape(expr, typed_target, result_type)

    def _infer_box(self, expr: BoxExpr, env: TypeEnv) -> TypedExpr:
        typed_value = self.infer(expr.value, env)
        if not isinstance(typed_value.type, ArrayType):
            raise RemoraTypeError("box expects an array", expr.loc)
        # Create hidden name from the leading dimension
        hidden_name = "len"
        sigma = SigmaType((hidden_name,), typed_value.type)
        return TypedBox(expr, typed_value, sigma)

    def _infer_unbox(self, expr: UnboxExpr, env: TypeEnv) -> TypedExpr:
        typed_box = self.infer(expr.box_expr, env)
        if not isinstance(typed_box.type, SigmaType):
            raise RemoraTypeError("unbox expects a boxed value (Sigma type)", expr.loc)
        sigma = typed_box.type
        if len(expr.hidden_names) != len(sigma.hidden_names):
            raise RemoraTypeError(
                f"unbox expects {len(sigma.hidden_names)} hidden name(s), got {len(expr.hidden_names)}",
                expr.loc,
            )
        # For the value binding, use the sigma body type
        # In a full implementation, we'd substitute the hidden dimensions
        inner_env = env
        for name in expr.hidden_names:
            inner_env = inner_env.extend(name, INT)
        inner_env = inner_env.extend(expr.value_name, sigma.body)
        typed_body = self.infer(expr.body, inner_env)
        # The hidden dimension must not leak into the body type
        if isinstance(typed_body.type, SigmaType):
            raise RemoraTypeError("hidden dimension escapes in unbox body", expr.loc)
        return TypedUnbox(expr, typed_box, expr.hidden_names, expr.value_name, typed_body, typed_body.type)

    def _infer_boxes(self, expr: BoxesExpr, env: TypeEnv) -> TypedExpr:
        """boxes e1 e2 ... : [(Σ (len) elem_type)] — array of individually-boxed elements."""
        if not expr.elements:
            raise RemoraTypeError("boxes requires at least one element", expr.loc)
        typed_elements = [self.infer(e, env) for e in expr.elements]
        first_type = typed_elements[0].type
        if not isinstance(first_type, ArrayType):
            raise RemoraTypeError("boxes elements must be arrays", expr.loc)
        # Each element is boxed: (Σ (len) element_type)
        element_sigma = SigmaType(("len",), first_type)
        for te in typed_elements[1:]:
            if not isinstance(te.type, ArrayType):
                raise RemoraTypeError("boxes elements must be arrays", expr.loc)
            if te.type.element != first_type.element:
                raise RemoraTypeError("boxes elements must have the same element type", expr.loc)
        result_type = ArrayType(element_sigma, (StaticDim(len(typed_elements)),))
        return TypedExprNode(expr, result_type)

    def _infer_iota1(self, expr: Iota1Expr, env: TypeEnv) -> TypedExpr:
        """iota1 n : (Σ (len) [int len]) — boxed iota of runtime size."""
        size = eval_static_dim(expr.size, expr.loc)
        iota_type = ArrayType(INT, (size,))
        sigma = SigmaType(("len",), iota_type)
        # Infer the iota itself, then wrap in a box
        typed_iota = TypedExprNode(IotaExpr(expr.size, expr.loc), iota_type)
        synthetic_box = BoxExpr(IotaExpr(expr.size, expr.loc), expr.loc)
        return TypedBox(synthetic_box, typed_iota, sigma)

    def _infer_iotan(self, expr: IotaNExpr, env: TypeEnv) -> TypedExpr:
        """iotaN d1 ... dN : (Σ (dims...) [int d1 ... dN]) — boxed multi-dim iota."""
        dims = tuple(eval_static_dim(s, expr.loc) for s in expr.sizes)
        if len(dims) != expr.rank:
            raise RemoraTypeError(f"iota{expr.rank} expects {expr.rank} dimensions", expr.loc)
        iota_type = ArrayType(INT, dims)
        hidden_names = tuple(f"d{i}" for i in range(expr.rank))
        sigma = SigmaType(hidden_names, iota_type)
        typed_iota = TypedExprNode(IotaExpr(IntLit(dims[0].value, expr.loc), expr.loc), iota_type)
        synthetic_box = BoxExpr(IotaExpr(expr.sizes[0], expr.loc), expr.loc)
        return TypedBox(synthetic_box, typed_iota, sigma)

    def _infer_sort(self, expr: SortExpr, env: TypeEnv) -> TypedExpr:
        typed_array = self.infer(expr.array, env)
        if not isinstance(typed_array.type, ArrayType):
            raise RemoraTypeError("sort expects an array", expr.loc)
        return TypedSort(expr, typed_array, typed_array.type)

    def _infer_grade(self, expr: GradeExpr, env: TypeEnv) -> TypedExpr:
        typed_array = self.infer(expr.array, env)
        if not isinstance(typed_array.type, ArrayType):
            raise RemoraTypeError("grade expects an array", expr.loc)
        result_shape = (typed_array.type.shape[0],)
        return TypedGrade(expr, typed_array, ArrayType(INT, result_shape))

    def _infer_filter(self, expr: FilterExpr, env: TypeEnv) -> TypedExpr:
        """filter pred xs : (Σ (len) [elem len]) — boxed filtered array."""
        # Handle operator sections like (filter (> 0) xs)
        pred_expr = expr.predicate
        if isinstance(pred_expr, AppExpr) and isinstance(pred_expr.func, VarExpr):
            if pred_expr.func.name in ALL_PRIMITIVE_OPS and len(pred_expr.args) == 1:
                pred_expr = LeftSectionExpr(pred_expr.func.name, pred_expr.args[0], pred_expr.loc)
        typed_array = self.infer(expr.array, env)
        if not isinstance(typed_array.type, ArrayType):
            raise RemoraTypeError("filter expects an array", expr.loc)
        element_type = typed_array.type.element
        expected_func = FuncType((element_type,), BOOL)
        typed_pred = self.check_callable(pred_expr, expected_func, env)
        sigma = SigmaType(("len",), typed_array.type)
        return TypedFilter(expr, typed_pred, typed_array, sigma)

    def _infer_replicate(self, expr: ReplicateExpr, env: TypeEnv) -> TypedExpr:
        """replicate counts xs : (Σ (len) [elem len]) — boxed repeated array."""
        typed_counts = self.infer(expr.counts, env)
        typed_array = self.infer(expr.array, env)
        if not isinstance(typed_counts.type, ArrayType) or typed_counts.type.element != INT:
            raise RemoraTypeError("replicate expects an integer count vector", expr.loc)
        if not isinstance(typed_array.type, ArrayType):
            raise RemoraTypeError("replicate expects an array", expr.loc)
        sigma = SigmaType(("len",), typed_array.type)
        return TypedReplicate(expr, typed_counts, typed_array, sigma)

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
        elif expr.op in {"<", "<=", "==", "!="}:
            _common = common_numeric_type(params[0], params[1])
            self._require(expected_type.result, BOOL, expr.loc)
        elif expr.op in {"&&", "||"}:
            self._require(params[0], BOOL, expr.loc)
            self._require(params[1], BOOL, expr.loc)
            self._require(expected_type.result, BOOL, expr.loc)
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
        elif expr.op in {"==", "!=", "<", "<=", ">", ">="}:
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
        elif expr.op in {"==", "!=", "<", "<=", ">", ">="}:
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
        return cell_type_candidates(value_type)

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
            declared_type = self._declared_function_type(definition)
            if isinstance(declared_type, PiType):
                if not isinstance(declared_type.body, FuncType):
                    raise RemoraTypeError(
                        "dependent function annotation must contain a function type",
                        definition.loc,
                    )
                symbolic_env = env
                for binder in declared_type.binders:
                    symbolic_env = symbolic_env.extend_index(binder)
                self._typed_top_level_function(
                    definition,
                    declared_type.body,
                    symbolic_env,
                )
            return TypedDefinition(
                definition,
                None,
                declared_type,
            ), env
        raise AssertionError(f"unknown definition type {type(definition).__name__}")

    def _infer_top_level_function_app(self, expr: AppExpr, env: TypeEnv) -> TypedExpr:
        function = self._functions[expr.func.name]
        if len(expr.args) != len(function.params):
            raise RemoraTypeError("function arity mismatch", expr.loc)
        typed_args = [self.infer(arg, env) for arg in expr.args]
        actual_param_types = tuple(arg.type for arg in typed_args)
        func_type = self._infer_top_level_function_type(
            function,
            actual_param_types,
            env,
        )
        index_args = self._inferred_index_args(function, actual_param_types)
        typed_func = self._typed_top_level_function(
            function,
            func_type,
            env,
            index_args=index_args,
        )
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

    def specialize_top_level_function(
        self,
        function: FuncDef,
        param_types: tuple[RemoraType, ...],
        env: TypeEnv,
    ) -> TypedLambda:
        """Build or reuse a concrete top-level function specialization."""
        func_type = self._infer_top_level_function_type(function, param_types, env)
        index_args = self._inferred_index_args(function, param_types)
        return self._typed_top_level_function(
            function,
            func_type,
            env,
            index_args=index_args,
        )

    def _infer_top_level_function_type(
        self,
        function: FuncDef,
        param_types: tuple[RemoraType, ...],
        env: TypeEnv,
    ) -> FuncType:
        if len(function.params) != len(param_types):
            raise RemoraTypeError("function arity mismatch", function.loc)
        declared_param_types = self._declared_param_types(function)
        if declared_param_types is not None:
            bindings = self._infer_index_bindings(
                function,
                declared_param_types,
                param_types,
            )
            specialized_params = tuple(
                substitute_type(param_type, bindings)
                for param_type in declared_param_types
            )
            declared_result = self._declared_result_type(function)
            if declared_result is None:
                raise RemoraTypeError(
                    "dependent function definitions require a result type",
                    function.loc,
                )
            specialized_result = substitute_type(declared_result, bindings)
            typed_func = self._typed_top_level_function(
                function,
                FuncType(specialized_params, specialized_result),
                env,
                index_args=tuple(
                    bindings[binder.name]
                    for binder in self._index_binders(function)
                ),
            )
            return typed_func.type
        typed_func = self._typed_top_level_function(
            function,
            FuncType(param_types, INT),
            env,
            infer_result=True,
        )
        return typed_func.type

    def _declared_function_type(self, function: FuncDef) -> RemoraType | None:
        declared_param_types = self._declared_param_types(function)
        if declared_param_types is None:
            return None
        declared_result = self._declared_result_type(function)
        if declared_result is None:
            raise RemoraTypeError(
                "dependent function definitions require a result type",
                function.loc,
            )
        body: RemoraType = FuncType(declared_param_types, declared_result)
        binders = self._index_binders(function)
        unbound = free_type_index_vars(body) - frozenset(
            binder.name for binder in binders
        )
        if unbound:
            names = ", ".join(sorted(unbound))
            raise RemoraTypeError(
                f"unbound index variable(s) in function annotation: {names}",
                function.loc,
            )
        if binders:
            return PiType(binders, body)
        return body

    def _declared_param_types(self, function: FuncDef) -> tuple[RemoraType, ...] | None:
        raw = getattr(function, "param_types", None)
        if raw is None:
            return None
        if len(raw) != len(function.params):
            raise RemoraTypeError("function annotation arity mismatch", function.loc)
        return tuple(self._require_remora_type(value, function.loc) for value in raw)

    def _index_binders(self, function: FuncDef) -> tuple[IndexBinder, ...]:
        raw = getattr(function, "index_binders", ())
        binders: list[IndexBinder] = []
        names: set[str] = set()
        for binder in raw:
            if not isinstance(binder, IndexBinder):
                raise RemoraTypeError("invalid index binder in function definition", function.loc)
            if binder.name in names:
                raise RemoraTypeError(
                    f"duplicate index binder {binder.name!r}",
                    function.loc,
                )
            names.add(binder.name)
            binders.append(binder)
        return tuple(binders)

    def _declared_result_type(self, function: FuncDef) -> RemoraType | None:
        raw = getattr(function, "result_type", None)
        if raw is None:
            return None
        return self._require_remora_type(raw, function.loc)

    def _require_remora_type(self, value: object, loc) -> RemoraType:
        if isinstance(value, (ScalarType, ArrayType, FuncType, SigmaType, PiType)):
            return value
        raise RemoraTypeError(f"invalid type annotation {value!r}", loc)

    def _infer_index_bindings(
        self,
        function: FuncDef,
        declared_param_types: tuple[RemoraType, ...],
        actual_param_types: tuple[RemoraType, ...],
    ) -> dict[str, DimExpr]:
        binders = self._index_binders(function)
        binder_names = {binder.name for binder in binders}
        for binder in binders:
            if binder.sort is not IndexSort.DIM:
                raise RemoraTypeError(
                    "Phase 7a only supports Dim binders in function annotations",
                    function.loc,
                )
        bindings: dict[str, DimExpr] = {}
        for declared, actual in zip(declared_param_types, actual_param_types):
            try:
                inferred = self._match_declared_type(declared, actual, function.loc)
            except ConstraintError as exc:
                raise RemoraTypeError(str(exc), function.loc) from exc
            for name, value in inferred.items():
                if name not in binder_names:
                    raise RemoraTypeError(
                        f"unbound dimension variable {name!r} in function annotation",
                        function.loc,
                    )
                existing = bindings.get(name)
                if existing is not None and existing != value:
                    raise RemoraTypeError(
                        f"dimension mismatch for {name!r}: expected {existing}, got {value}",
                        function.loc,
                    )
                bindings[name] = value
        missing = [binder.name for binder in binders if binder.name not in bindings]
        if missing:
            names = ", ".join(missing)
            raise RemoraTypeError(
                f"could not infer index argument(s): {names}",
                function.loc,
            )
        return bindings

    def _inferred_index_args(
        self,
        function: FuncDef,
        actual_param_types: tuple[RemoraType, ...],
    ) -> tuple[DimExpr, ...] | None:
        declared_param_types = self._declared_param_types(function)
        binders = self._index_binders(function)
        if declared_param_types is None or not binders:
            return None
        bindings = self._infer_index_bindings(
            function,
            declared_param_types,
            actual_param_types,
        )
        return tuple(bindings[binder.name] for binder in binders)

    def _match_declared_type(
        self,
        declared: RemoraType,
        actual: RemoraType,
        loc,
    ) -> dict[str, DimExpr]:
        if isinstance(declared, ScalarType):
            self._require(actual, declared, loc)
            return {}
        if isinstance(declared, ArrayType):
            if not isinstance(actual, ArrayType):
                raise RemoraTypeError(f"expected array type {declared}, got {actual}", loc)
            self._require(actual.element, declared.element, loc)
            return match_shape_template(declared.shape, actual.shape, loc=loc)
        raise RemoraTypeError(
            f"Phase 7a function annotations only support scalar and array parameter types, got {declared}",
            loc,
        )

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
        index_args: tuple[DimExpr, ...] | None = None,
    ) -> TypedLambda:
        cache_key = (
            (function.name, index_args)
            if index_args is not None
            else None
        )
        if cache_key is not None and cache_key in self._specializations:
            cached = self._specializations[cache_key]
            if cached.type != func_type:
                raise RemoraTypeError(
                    f"inconsistent specialization type for {cached.specialization_name}",
                    function.loc,
                )
            return cached
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
            typed_lambda = TypedLambda(
                function,
                list(zip(function.params, func_type.params)),
                typed_result,
                inferred_type,
                self._specialization_name(function, index_args)
                if index_args is not None
                else None,
                index_args or (),
            )
            if cache_key is not None:
                self._specializations[cache_key] = typed_lambda
            return typed_lambda
        finally:
            self._active_functions.remove(function.name)

    def _specialization_name(
        self,
        function: FuncDef,
        index_args: tuple[DimExpr, ...],
    ) -> str:
        binders = self._index_binders(function)
        suffix = "__".join(
            f"{binder.name}_{arg}"
            for binder, arg in zip(binders, index_args)
        )
        return f"{function.name}__{suffix}"

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
