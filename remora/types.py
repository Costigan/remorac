"""Static types for the Remora Dense Core prototype."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TypeAlias

from remora.ast_nodes import Expr, IntLit, SourceLoc
from remora.errors import RemoraError
from remora.index import DimExpr, IndexBinder
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
class ArrayType:
    element: ScalarType
    shape: tuple[DimExpr, ...]

    @property
    def rank(self) -> int:
        return len(self.shape)

    def with_frame(self, frame: tuple[DimExpr, ...]) -> ArrayType:
        return ArrayType(self.element, frame + self.shape)

    def drop_outer(self, n: int) -> RemoraType:
        if n < 0 or n > self.rank:
            raise ValueError("cannot drop more dimensions than an array has")
        remaining_shape = self.shape[n:]
        if not remaining_shape:
            return self.element
        return ArrayType(self.element, remaining_shape)

    def __str__(self) -> str:
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


RemoraType: TypeAlias = ScalarType | ArrayType | FuncType | SigmaType | PiType

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


def infer_lifting(
    func_type: FuncType,
    array_type: RemoraType,
) -> tuple[tuple[DimExpr, ...], RemoraType]:
    if len(func_type.params) != 1:
        raise RemoraTypeError("map expects a unary function")

    cell_type = func_type.params[0]
    cell_rank = cell_type.rank
    array_rank = array_type.rank

    if array_rank < cell_rank:
        raise RemoraTypeError(
            f"array rank {array_rank} is too low for cell rank {cell_rank}"
        )

    if not _cell_matches_array_suffix(cell_type, array_type):
        raise RemoraTypeError(f"function cell type {cell_type} does not match {array_type}")

    frame_shape: tuple[DimExpr, ...]
    if isinstance(array_type, ArrayType):
        frame_shape = array_type.shape[: array_type.rank - cell_rank]
    else:
        frame_shape = ()

    result_type = with_frame(func_type.result, frame_shape)
    enforce_rank_limit(result_type)
    return frame_shape, result_type


def with_frame(value_type: RemoraType, frame: tuple[DimExpr, ...]) -> RemoraType:
    if not frame:
        return value_type
    if isinstance(value_type, FuncType):
        raise RemoraTypeError("function-valued map results are deferred")
    if isinstance(value_type, ArrayType):
        return value_type.with_frame(frame)
    return ArrayType(value_type, frame)


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


def _cell_matches_array_suffix(cell_type: RemoraType, array_type: RemoraType) -> bool:
    if isinstance(cell_type, FuncType) or isinstance(array_type, FuncType):
        return False
    if isinstance(cell_type, ScalarType):
        if isinstance(array_type, ScalarType):
            return cell_type == array_type
        return cell_type == array_type.element
    if isinstance(array_type, ScalarType):
        return False
    return (
        cell_type.element == array_type.element
        and cell_type.shape == array_type.shape[-cell_type.rank :]
    )
