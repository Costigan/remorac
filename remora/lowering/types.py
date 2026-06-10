"""MLIR type mapping and shared lowering utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from remora.errors import RemoraError
from remora.hir import (
    HIRAppend,
    HIRApply,
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
    HIRReduce,
    HIRReshape,
    HIRReverse,
    HIRRotate,
    HIRScatterAdd,
    HIRIm2col,
    HIRCol2im,
    HIRSubarray,
    HIRTake,
    HIRTranspose,
    HIRVar,
    HIRWithShape,
)
from remora.types import (
    BOOL,
    FLOAT,
    INT,
    ArrayType,
    FuncType,
    RemoraType,
    ScalarType,
    SigmaType,
    StaticDim,
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
        dim_parts: list[str] = []
        for dim in value_type.shape:
            value = getattr(dim, "value", None)
            if value is None:
                raise RemoraLoweringError(
                    f"cannot lower type {value_type}: shape contains "
                    f"non-concrete dimension {dim}"
                )
            dim_parts.append(str(value))
        dims = "x".join(dim_parts)
        element = type_to_mlir(value_type.element)
        if dims:
            return f"tensor<{dims}x{element}>"
        return f"tensor<{element}>"
    if isinstance(value_type, FuncType):
        params = ", ".join(type_to_mlir(param) for param in value_type.params)
        return f"({params}) -> {type_to_mlir(value_type.result)}"
    if isinstance(value_type, ScalarType):
        raise RemoraLoweringError(f"unknown scalar type {value_type.name}")
    if isinstance(value_type, SigmaType):
        elem = type_to_mlir(value_type.body.element if isinstance(value_type.body, ArrayType) else value_type.body)
        return f"tensor<?x{elem}>"
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
    if isinstance(expr, (HIRMap, HIRApply)):
        return expr.result_type
    if isinstance(expr, (HIRFold, HIRReduce)):
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
    if isinstance(expr, HIRAppend):
        return expr.result_type
    if isinstance(expr, HIRSubarray):
        return expr.result_type
    if isinstance(expr, HIRRotate):
        return expr.result_type
    if isinstance(expr, HIRScatterAdd):
        return expr.result_type
    if isinstance(expr, (HIRIm2col, HIRCol2im)):
        return expr.result_type
    if isinstance(expr, HIRWithShape):
        return expr.result_type
    raise AssertionError(f"unknown HIR expression {type(expr).__name__}")


@dataclass(frozen=True)
class _TensorValue:
    name: str
    type: str
    element_type: str


TensorEnv = dict[str, _TensorValue]
