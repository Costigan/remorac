"""Shared HIR expression dispatch utilities.

Provides a single dispatch table so that adding a new HIR expression
node type requires changes in fewer places.  Each module that walks HIR
expressions calls ``hir_dispatch(expr, handlers)`` with a dict of
handler functions keyed by ``type``.

Example::

    from remora.hir_dispatch import hir_dispatch

    def my_walk(expr):
        return hir_dispatch(expr, {
            HIRVar: lambda e: ...,
            HIRLit: lambda e: ...,
        }, default=lambda e: e)
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, TypeVar

from remora.hir import (
    HIRApply,
    HIRArrayLit,
    HIRCall,
    HIRCast,
    HIRDrop,
    HIRExpr,
    HIRFold,
    HIRFoldRight,
    HIRIf,
    HIRIndex,
    HIRIndicesOf,
    HIRIota,
    HIRLambda,
    HIRLet,
    HIRLit,
    HIRMap,
    HIRParam,
    HIRPrimCallable,
    HIRPrimOp,
    HIRRavel,
    HIRReduce,
    HIRReshape,
    HIRReverse,
    HIRRotate,
    HIRScan,
    HIRSlice,
    HIRSubarray,
    HIRTake,
    HIRTranspose,
    HIRVar,
)

# Ordered list of all HIR expression types for dispatch.
_ALL_HIR_TYPES: tuple[type, ...] = (
    HIRApply,
    HIRArrayLit,
    HIRCall,
    HIRCast,
    HIRDrop,
    HIRFold,
    HIRFoldRight,
    HIRIf,
    HIRIndex,
    HIRIndicesOf,
    HIRIota,
    HIRLambda,
    HIRLet,
    HIRLit,
    HIRMap,
    HIRPrimOp,
    HIRPrimCallable,
    HIRRavel,
    HIRReduce,
    HIRReshape,
    HIRReverse,
    HIRRotate,
    HIRScan,
    HIRSlice,
    HIRSubarray,
    HIRTake,
    HIRTranspose,
    HIRVar,
)


def hir_dispatch(
    expr: HIRExpr,
    handlers: dict[type, Handler],
    *,
    default: Handler | None = None,
) -> Any:
    """Dispatch *expr* to the handler registered for its concrete type.

    Raises ``AssertionError`` if no handler matches and no *default* is given.
    """
    for cls in _ALL_HIR_TYPES:
        if isinstance(expr, cls):
            handler = handlers.get(cls)
            if handler is not None:
                return handler(expr)
            if default is not None:
                return default(expr)
            raise AssertionError(
                f"no handler for HIR expression {type(expr).__name__}"
            )
    raise AssertionError(f"unknown HIR expression {type(expr).__name__}")
