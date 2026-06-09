"""Static types for the Remora Dense Core prototype."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TypeAlias

from remora.ast_nodes import Expr, IntLit, SourceLoc
from remora.errors import RemoraError
from remora.index import DimExpr, IndexBinder, ShapeExpr
from remora.limits import MAX_DENSE_RANK


@dataclass(frozen=True)
class StaticDim(DimExpr):
    value: int

    def __post_init__(self) -> None:
        if self.value < 0:
            raise ValueError("static dimensions must be non-negative")

    def __str__(self) -> str:
        return str(self.value)

@dataclass(frozen=True)
class ScalarType:
    name: str

    @property
    def rank(self) -> int:
        return 0

    def __str__(self) -> str:
        return self.name


@dataclass(frozen=True)
class TypeVar(ScalarType):
    """A type variable standing for an unknown scalar element type.

    TypeVars are resolved during Forall instantiation before backend lowering.
    They must not appear in backend HIR types.
    """

    def __str__(self) -> str:
        return f"?{self.name}"


@dataclass(frozen=True)
class TypeBinder:
    """Binds a type variable name in a ForallType."""
    name: str


@dataclass(frozen=True)
class ArrayType:
    element: ScalarType
    shape: tuple[DimExpr, ...]
    shape_expr: ShapeExpr | None = None

    @property
    def rank(self) -> int:
        return len(self.shape)

    def with_shape_expr(self, expr: ShapeExpr) -> ArrayType:
        return ArrayType(self.element, self.shape, expr)

    def with_frame(self, frame: tuple[DimExpr, ...]) -> ArrayType:
        new_shape = frame + self.shape
        new_expr = None
        if self.shape_expr is not None:
            from remora.index import ShapeConcat, ShapeLit
            new_expr = ShapeConcat(ShapeLit(frame), self.shape_expr)
        return ArrayType(self.element, new_shape, new_expr)

    def drop_outer(self, n: int) -> RemoraType:
        if n < 0 or n > self.rank:
            raise ValueError("cannot drop more dimensions than an array has")
        remaining_shape = self.shape[n:]
        if not remaining_shape:
            return self.element
        return ArrayType(self.element, remaining_shape)

    def concrete_shape(self) -> tuple[DimExpr, ...]:
        return self.shape

    def __str__(self) -> str:
        if self.shape_expr is not None:
            dims = str(self.shape_expr)
        else:
            dims = ",".join(str(dim) for dim in self.shape)
        return f"{self.element}[{dims}]"


@dataclass(frozen=True)
class FuncType:
    params: tuple[RemoraType, ...]
    result: RemoraType

    @property
    def rank(self) -> int:
        return 0

    def __str__(self) -> str:
        params = ", ".join(str(param) for param in self.params)
        return f"({params}) -> {self.result}"


@dataclass(frozen=True)
class SigmaType:
    """Existential type: (Σ (name) body_type) — there exists some dimension."""
    hidden_names: tuple[str, ...]
    body: RemoraType

    @property
    def rank(self) -> int:
        return 0

    def __str__(self) -> str:
        names = " ".join(self.hidden_names)
        return f"(Σ ({names}) {self.body})"


@dataclass(frozen=True)
class PiType:
    """Dependent product over compile-time dimension/shape indices."""
    binders: tuple[IndexBinder, ...]
    body: RemoraType

    @property
    def rank(self) -> int:
        return 0

    def __str__(self) -> str:
        binders = " ".join(f"({binder.name} {binder.sort})" for binder in self.binders)
        return f"(Π ({binders}) {self.body})"


@dataclass(frozen=True)
class ForallType:
    """Universally quantified type over element-type variables.

    ``ForallType([TypeBinder("t")], ArrayType(TypeVar("t"), ...))``
    means "for all element types t, this function works".
    """
    binders: tuple[TypeBinder, ...]
    body: RemoraType

    @property
    def rank(self) -> int:
        return 0

    def __str__(self) -> str:
        binders = " ".join(binder.name for binder in self.binders)
        return f"(∀ ({binders}) {self.body})"


@dataclass(frozen=True)
class PairType:
    """Product type for pairs: (PairType A B)."""
    left: RemoraType
    right: RemoraType

    @property
    def element(self) -> ScalarType | None:
        if isinstance(self.left, ScalarType) and isinstance(self.right, ScalarType):
            return self.left
        return None

    @property
    def rank(self) -> int:
        return 0

    def __str__(self) -> str:
        return f"(Pair {self.left} {self.right})"


RemoraType: TypeAlias = ScalarType | ArrayType | FuncType | SigmaType | PiType | ForallType | PairType

FLOAT = ScalarType("float")
INT = ScalarType("int")
BOOL = ScalarType("bool")

class RemoraTypeError(RemoraError):
    """Raised when Dense Core type checking fails."""

    def __init__(self, message: str, loc: SourceLoc | None = None):
        if loc is not None:
            message = f"{loc.file}:{loc.line}:{loc.col}: {message}"
        super().__init__(message)
        self.loc = loc


def eval_static_dim(expr: Expr, loc: SourceLoc | None = None) -> StaticDim:
    """Evaluate a Dense Core dimension expression.

    Phase 2 intentionally accepts only integer literals as compile-time
    dimensions. Constant folding can broaden this later.
    """
    if isinstance(expr, IntLit) and expr.value >= 0:
        return StaticDim(expr.value)
    raise RemoraTypeError("expected a non-negative integer constant dimension", loc)


def enforce_rank_limit(value_type: RemoraType, loc: SourceLoc | None = None) -> None:
    if value_type.rank > MAX_DENSE_RANK:
        raise RemoraTypeError("rank limit exceeded in Dense Core", loc)


def common_numeric_type(left: RemoraType, right: RemoraType) -> ScalarType:
    if left == FLOAT or right == FLOAT:
        if is_numeric(left) and is_numeric(right):
            return FLOAT
    if left == INT and right == INT:
        return INT
    raise RemoraTypeError(f"expected numeric operands, got {left} and {right}")


def is_numeric(value_type: RemoraType) -> bool:
    return value_type in (INT, FLOAT)
