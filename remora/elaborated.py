"""Typed elaborated core IR between source typing and backend HIR."""

from __future__ import annotations

from dataclasses import dataclass

from remora.ast_nodes import Definition, IndexAppExpr
from remora.index import DimExpr
from remora.typechecker import TypedDefinition, TypedExpr
from remora.types import FuncType, RemoraType


@dataclass(frozen=True)
class FrameCellDecision:
    """A frame/cell decomposition decision recorded in the typed core.

    Downstream passes (HIR lowering, AD, ...) reuse this instead of
    rediscovering the decomposition.
    """

    frame_shape: tuple[DimExpr, ...]
    cell_shape: tuple[DimExpr, ...]
    cell_type: RemoraType | None = None
    is_implicit: bool = False  # True for auto-lifting, False for explicit map
    is_binary: bool = False    # True for binary (broadcasting) operations
    principal_frame: tuple[DimExpr, ...] | None = None  # for broadcasting

    @property
    def cell_rank(self) -> int:
        return len(self.cell_shape)

    @property
    def frame_rank(self) -> int:
        return len(self.frame_shape)

    @property
    def result_rank(self) -> int:
        return self.frame_rank + self.cell_rank


@dataclass(frozen=True)
class CoreExpr:
    """A structural typed-core expression with its elaborated children."""

    kind: str
    type: RemoraType
    children: tuple[CoreExpr, ...]
    typed: TypedExpr
    frame: FrameCellDecision | None = None


@dataclass(frozen=True)
class CoreDefinition:
    """A source definition and its elaborated value, when value-bound."""

    source: Definition
    type: RemoraType | None
    value: CoreExpr | None
    typed: TypedDefinition


@dataclass(frozen=True)
class CoreSpecialization:
    """A concrete dependent-function instance available for backend erasure."""

    name: str
    function_name: str
    index_args: tuple[DimExpr, ...]
    params: tuple[tuple[str, RemoraType], ...]
    body: CoreExpr
    type: FuncType


@dataclass(frozen=True)
class CoreIndexApplication:
    """An explicit Pi specialization retained until backend erasure."""

    source: IndexAppExpr
    function_name: str
    specialization_name: str
    index_args: tuple[DimExpr, ...]
    type: FuncType


@dataclass(frozen=True)
class CoreProgram:
    definitions: tuple[CoreDefinition, ...]
    body: CoreExpr | None
    type: RemoraType | None
    index_applications: tuple[CoreIndexApplication, ...] = ()
    specializations: tuple[CoreSpecialization, ...] = ()
