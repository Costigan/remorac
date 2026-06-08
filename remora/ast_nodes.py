"""AST nodes for the Remora Dense Core parser slice."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TypeAlias


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


@dataclass(frozen=True)
class IotaExpr:
    size: Expr
    loc: SourceLoc


@dataclass(frozen=True)
class ShapeExpr:
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
    | LambdaExpr
    | AppExpr
    | MapExpr
    | FoldExpr
    | ReduceExpr
    | FoldRightExpr
    | ScanExpr
    | IotaExpr
    | ShapeExpr
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
