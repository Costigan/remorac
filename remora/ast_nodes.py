"""AST nodes for the Remora Dense Core parser slice."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, TypeAlias

from remora.index import DimExpr, IndexBinder
from remora.index import ShapeExpr as IndexShapeExpr  # avoid name clash with field name

if TYPE_CHECKING:
    from remora.types import RemoraType


@dataclass(frozen=True)
class SourceLoc:
    file: str
    line: int
    col: int


@dataclass(frozen=True)
class Program:
    definitions: list[Definition]
    body: Expr | None
    loc: SourceLoc


@dataclass(frozen=True)
class FuncDef:
    name: str
    params: list[str]
    body: Expr
    loc: SourceLoc
    param_ranks: list[int | None] | None = None
    index_binders: tuple[IndexBinder, ...] = ()
    type_binders: tuple[str, ...] = ()
    param_types: list[RemoraType] | None = None
    result_type: RemoraType | None = None


@dataclass(frozen=True)
class ValDef:
    name: str
    value: Expr
    loc: SourceLoc


@dataclass(frozen=True)
class LetExpr:
    name: str
    value: Expr
    body: Expr
    loc: SourceLoc


@dataclass(frozen=True)
class IfExpr:
    condition: Expr
    then_branch: Expr
    else_branch: Expr
    loc: SourceLoc


@dataclass(frozen=True)
class SelectExpr:
    condition: Expr
    then_branch: Expr
    else_branch: Expr
    loc: SourceLoc


@dataclass(frozen=True)
class AppendExpr:
    left: Expr
    right: Expr
    loc: SourceLoc


@dataclass(frozen=True)
class RotateExpr:
    array: Expr
    shift: Expr
    loc: SourceLoc


@dataclass(frozen=True)
class RerankExpr:
    ranks: list[int]
    func: Expr
    loc: SourceLoc


@dataclass(frozen=True)
class SubarrayExpr:
    array: Expr
    offsets: list[Expr]
    shape: list[Expr]
    loc: SourceLoc


@dataclass(frozen=True)
class IndicesOfExpr:
    array: Expr
    loc: SourceLoc


@dataclass(frozen=True)
class WithShapeExpr:
    target: Expr
    shape: Expr
    loc: SourceLoc


@dataclass(frozen=True)
class ScatterAddExpr:
    array: Expr
    index: Expr
    update: Expr
    loc: SourceLoc


@dataclass(frozen=True)
class Im2colExpr:
    image: Expr
    kernel_shape: Expr
    stride: Expr
    loc: SourceLoc


@dataclass(frozen=True)
class Col2imExpr:
    columns: Expr
    image_shape: Expr
    kernel_shape: Expr
    stride: Expr
    loc: SourceLoc


@dataclass(frozen=True)
class PairExpr:
    left: Expr
    right: Expr
    loc: SourceLoc


@dataclass(frozen=True)
class FirstExpr:
    pair: Expr
    loc: SourceLoc


@dataclass(frozen=True)
class SecondExpr:
    pair: Expr
    loc: SourceLoc


@dataclass(frozen=True)
class BoxExpr:
    value: Expr
    loc: SourceLoc


@dataclass(frozen=True)
class Iota1Expr:
    size: Expr
    loc: SourceLoc


@dataclass(frozen=True)
class IotaNExpr:
    rank: int
    sizes: list[Expr]
    loc: SourceLoc


@dataclass(frozen=True)
class BoxesExpr:
    elements: list[Expr]
    loc: SourceLoc


@dataclass(frozen=True)
class FilterExpr:
    predicate: Expr
    array: Expr
    loc: SourceLoc


@dataclass(frozen=True)
class ReplicateExpr:
    counts: Expr
    array: Expr
    loc: SourceLoc


@dataclass(frozen=True)
class SortExpr:
    func: Expr
    array: Expr
    loc: SourceLoc


@dataclass(frozen=True)
class GradeExpr:
    func: Expr
    array: Expr
    loc: SourceLoc


@dataclass(frozen=True)
class UnboxExpr:
    box_expr: Expr
    hidden_names: list[str]
    value_name: str
    body: Expr
    loc: SourceLoc


@dataclass(frozen=True)
class LambdaExpr:
    params: list[str]
    body: Expr
    loc: SourceLoc
    param_ranks: list[int | None] | None = None


@dataclass(frozen=True)
class AppExpr:
    func: Expr
    args: list[Expr]
    loc: SourceLoc


@dataclass(frozen=True)
class IndexAppExpr:
    func: Expr
    args: tuple[DimExpr | IndexShapeExpr, ...]
    loc: SourceLoc


@dataclass(frozen=True)
class GradExpr:
    """grad f: differentiates a unary Float→Float function."""
    func: Expr
    loc: SourceLoc


@dataclass(frozen=True)
class MapExpr:
    func: Expr
    arrays: list[Expr]
    loc: SourceLoc

    @property
    def array(self) -> Expr:
        return self.arrays[0]


@dataclass(frozen=True)
class FoldExpr:
    func: Expr
    init: Expr
    array: Expr
    loc: SourceLoc


@dataclass(frozen=True)
class ReduceExpr:
    func: Expr
    init: Expr
    array: Expr
    loc: SourceLoc
    require_nonempty: bool = False


@dataclass(frozen=True)
class FoldRightExpr:
    func: Expr
    init: Expr
    array: Expr
    loc: SourceLoc


@dataclass(frozen=True)
class ScanExpr:
    func: Expr
    init: Expr
    array: Expr
    loc: SourceLoc
    exclusive: bool = False
    require_nonempty: bool = False


@dataclass(frozen=True)
class TraceExpr:
    func: Expr
    init: Expr
    array: Expr
    loc: SourceLoc
    right: bool = False


@dataclass(frozen=True)
class IotaExpr:
    size: Expr
    loc: SourceLoc


@dataclass(frozen=True)
class ShapeExpr:
    array: Expr
    loc: SourceLoc


@dataclass(frozen=True)
class LengthExpr:
    array: Expr
    loc: SourceLoc


@dataclass(frozen=True)
class RankExpr:
    array: Expr
    loc: SourceLoc


@dataclass(frozen=True)
class TransposeExpr:
    array: Expr
    loc: SourceLoc


@dataclass(frozen=True)
class ReshapeExpr:
    shape: Expr
    array: Expr
    loc: SourceLoc


@dataclass(frozen=True)
class RavelExpr:
    array: Expr
    loc: SourceLoc


@dataclass(frozen=True)
class ReverseExpr:
    array: Expr
    loc: SourceLoc


@dataclass(frozen=True)
class TakeExpr:
    count: Expr
    array: Expr
    loc: SourceLoc


@dataclass(frozen=True)
class DropExpr:
    count: Expr
    array: Expr
    loc: SourceLoc


@dataclass(frozen=True)
class ComposeExpr:
    outer: Expr
    inner: Expr
    loc: SourceLoc


@dataclass(frozen=True)
class OperatorFuncExpr:
    op: str
    loc: SourceLoc


@dataclass(frozen=True)
class LeftSectionExpr:
    op: str
    arg: Expr
    loc: SourceLoc


@dataclass(frozen=True)
class RightSectionExpr:
    arg: Expr
    op: str
    loc: SourceLoc


@dataclass(frozen=True)
class VarExpr:
    name: str
    loc: SourceLoc


@dataclass(frozen=True)
class IntLit:
    value: int
    loc: SourceLoc


@dataclass(frozen=True)
class FloatLit:
    value: float
    loc: SourceLoc


@dataclass(frozen=True)
class BoolLit:
    value: bool
    loc: SourceLoc


@dataclass(frozen=True)
class ArrayLit:
    elements: list[Expr]
    loc: SourceLoc


@dataclass(frozen=True)
class IndexExpr:
    array: Expr
    indices: list[Expr | SliceRange]
    loc: SourceLoc


@dataclass(frozen=True)
class SliceRange:
    start: Expr | None
    end: Expr | None
    loc: SourceLoc


Expr: TypeAlias = (
    LetExpr
    | IfExpr
    | SelectExpr
    | AppendExpr
    | RotateExpr
    | RerankExpr
    | SubarrayExpr
    | IndicesOfExpr
    | WithShapeExpr
    | ScatterAddExpr
    | Im2colExpr
    | Col2imExpr
    | PairExpr
    | FirstExpr
    | SecondExpr
    | BoxExpr
    | UnboxExpr
    | Iota1Expr
    | IotaNExpr
    | BoxesExpr
    | FilterExpr
    | ReplicateExpr
    | SortExpr
    | GradeExpr
    | LambdaExpr
    | AppExpr
    | IndexAppExpr
    | GradExpr
    | MapExpr
    | FoldExpr
    | ReduceExpr
    | FoldRightExpr
    | ScanExpr
    | TraceExpr
    | IotaExpr
    | ShapeExpr
    | LengthExpr
    | RankExpr
    | TransposeExpr
    | ReshapeExpr
    | RavelExpr
    | ReverseExpr
    | TakeExpr
    | DropExpr
    | ComposeExpr
    | OperatorFuncExpr
    | LeftSectionExpr
    | RightSectionExpr
    | VarExpr
    | IntLit
    | FloatLit
    | BoolLit
    | ArrayLit
    | IndexExpr
    | SliceRange
)

Definition: TypeAlias = FuncDef | ValDef
