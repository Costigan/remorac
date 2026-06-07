"""MLIR type mapping and shared lowering utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from remora.errors import RemoraError
from remora.hir import (
    HIRArrayLit,
    HIRCast,
    HIRCall,
    HIRDrop,
    HIRExpr,
    HIRFold,
    HIRIota,
    HIRIndex,
    HIRLet,
    HIRLit,
    HIRMap,
    HIRPrimOp,
    HIRRavel,
    HIRReshape,
    HIRReverse,
    HIRTake,
    HIRTranspose,
    HIRVar,
)
from remora.types import (
    BOOL,
    FLOAT,
    INT,
    ArrayType,
    FuncType,
    RemoraType,
    ScalarType,
)


class RemoraLoweringError(RemoraError):
    """Raised when the current MLIR lowering slice cannot handle a program."""


def type_to_mlir(value_type: RemoraType) -> str:
    if value_type == FLOAT:
        return "f32"
    if value_type == INT:
        return "i32"
    if value_type == BOOL:
        return "i1"
    if isinstance(value_type, ArrayType):
        dims = "x".join(str(dim.value) for dim in value_type.shape)
        element = type_to_mlir(value_type.element)
        if dims:
            return f"tensor<{dims}x{element}>"
        return f"tensor<{element}>"
    if isinstance(value_type, FuncType):
        params = ", ".join(type_to_mlir(param) for param in value_type.params)
        return f"({params}) -> {type_to_mlir(value_type.result)}"
    if isinstance(value_type, ScalarType):
        raise RemoraLoweringError(f"unknown scalar type {value_type.name}")
    raise RemoraLoweringError(f"cannot lower type {value_type}")


def _is_scalar_type(value_type: RemoraType) -> bool:
    return isinstance(value_type, ScalarType)


def _join_prefix(prefix: str, suffix: str) -> str:
    return f"{prefix}_{suffix}" if prefix else suffix


def _expr_result_type(expr: HIRExpr) -> RemoraType:
    if isinstance(expr, HIRLit):
        return expr.type
    if isinstance(expr, HIRVar):
        return expr.type
    if isinstance(expr, HIRIota):
        return expr.result_type
    if isinstance(expr, HIRArrayLit):
        return expr.result_type
    if isinstance(expr, HIRMap):
        return expr.result_type
    if isinstance(expr, HIRFold):
        return expr.result_type
    if isinstance(expr, HIRLet):
        return expr.result_type
    if isinstance(expr, HIRCall):
        return expr.result_type
    if isinstance(expr, HIRCast):
        return expr.result_type
    if isinstance(expr, HIRPrimOp):
        return expr.result_type
    if isinstance(expr, HIRIndex):
        return expr.result_type
    if isinstance(expr, HIRTranspose):
        return expr.result_type
    if isinstance(expr, HIRReshape):
        return expr.result_type
    if isinstance(expr, HIRRavel):
        return expr.result_type
    if isinstance(expr, HIRReverse):
        return expr.result_type
    if isinstance(expr, HIRTake):
        return expr.result_type
    if isinstance(expr, HIRDrop):
        return expr.result_type
    raise AssertionError(f"unknown HIR expression {type(expr).__name__}")


@dataclass(frozen=True)
class _TensorValue:
    name: str
    type: str
    element_type: str


TensorEnv = dict[str, _TensorValue]
