"""Typed elaborated core IR.

Phase 7 will move dependent type elaboration here.  The initial core is a
compatibility shell around the existing typed program so the compiler pipeline
has a stable insertion point before backend HIR lowering.
"""

from __future__ import annotations

from dataclasses import dataclass

from remora.ast_nodes import IndexAppExpr
from remora.index import DimExpr
from remora.typechecker import TypedProgram
from remora.types import FuncType


@dataclass(frozen=True)
class CoreIndexApplication:
    """An explicit Pi specialization retained until backend erasure."""
    source: IndexAppExpr
    function_name: str
    index_args: tuple[DimExpr, ...]
    type: FuncType


@dataclass(frozen=True)
class CoreProgram:
    typed: TypedProgram
    index_applications: tuple[CoreIndexApplication, ...] = ()
