"""Shared operator metadata for the Remora Dense Core language.

Centralizes operator definitions so that adding or changing an operator
requires an update in exactly one place.  Used by the type checker, HIR
lowering, MLIR lowering (CPU + GPU), and the typed-AST interpreter.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

# ---------------------------------------------------------------------------
# Numeric / comparison / boolean core operators
# ---------------------------------------------------------------------------

ARITHMETIC_OPS = frozenset({"+", "-", "*", "/", "modulo"})
UNARY_FLOAT_OPS = frozenset({"exp", "log", "sqrt", "cos", "sin", "ceil", "floor"})
BINARY_FLOAT_OPS = frozenset({"expt"})
COMPARISON_OPS = frozenset({"<", "<=", ">", ">=", "==", "!="})
BOOLEAN_OPS = frozenset({"&&", "||"})
ALL_PRIMITIVE_OPS = (
    ARITHMETIC_OPS | UNARY_FLOAT_OPS | BINARY_FLOAT_OPS | COMPARISON_OPS | BOOLEAN_OPS
)


def is_primitive_op(name: str) -> bool:
    """Return True if *name* is a primitive Remora operator."""
    return name in ALL_PRIMITIVE_OPS


# ---------------------------------------------------------------------------
# Per-result-type dispatch tables for MLIR ``arith`` dialect
# ---------------------------------------------------------------------------

_ARITH_OPS_F32: dict[str, str] = {
    "+": "arith.addf",
    "-": "arith.subf",
    "*": "arith.mulf",
    "/": "arith.divf",
    "exp": "math.exp",
    "log": "math.log",
    "sqrt": "math.sqrt",
    "cos": "math.cos",
    "sin": "math.sin",
    "ceil": "math.ceil",
    "floor": "math.floor",
    "expt": "math.powf",
    "modulo": "arith.remf",
    "==": "arith.cmpf oeq",
    "!=": "arith.cmpf une",
    "<": "arith.cmpf olt",
    "<=": "arith.cmpf ole",
}

_ARITH_OPS_F64: dict[str, str] = _ARITH_OPS_F32  # same ops, MLIR types determine precision

_ARITH_OPS_I32: dict[str, str] = {
    "+": "arith.addi",
    "-": "arith.subi",
    "*": "arith.muli",
    "/": "arith.divsi",
    "modulo": "arith.remsi",
    "==": "arith.cmpi eq",
    "!=": "arith.cmpi ne",
    "<": "arith.cmpi slt",
    "<=": "arith.cmpi sle",
}

_ARITH_OPS_I64: dict[str, str] = {
    "+": "arith.addi",
    "-": "arith.subi",
    "*": "arith.muli",
    "/": "arith.divsi",
    "modulo": "arith.remsi",
    "==": "arith.cmpi eq",
    "!=": "arith.cmpi ne",
    "<": "arith.cmpi slt",
    "<=": "arith.cmpi sle",
}

_ARITH_OPS_I1: dict[str, str] = {
    "&&": "arith.andi",
    "||": "arith.ori",
    "==": "arith.cmpi eq",
    "!=": "arith.cmpi ne",
}

_ARITH_OPS_BY_TYPE: dict[str, dict[str, str]] = {
    "f32": _ARITH_OPS_F32,
    "f64": _ARITH_OPS_F64,
    "i32": _ARITH_OPS_I32,
    "i64": _ARITH_OPS_I64,
    "i1": _ARITH_OPS_I1,
}


def arith_op(op: str, result_type: str) -> str:
    """Return the ``arith.*`` MLIR operation for *op* / *result_type*."""
    table = _ARITH_OPS_BY_TYPE.get(result_type)
    if table is None:
        raise KeyError(f"no arith ops defined for MLIR type {result_type!r}")
    try:
        return table[op]
    except KeyError:
        raise KeyError(f"operator {op!r} not defined for MLIR type {result_type!r}")


# ---------------------------------------------------------------------------
# Per-result-type dispatch tables for GPU ``llvm`` dialect
# ---------------------------------------------------------------------------

_LLVM_OPS_F32: dict[str, str] = {
    "+": "llvm.fadd",
    "-": "llvm.fsub",
    "*": "llvm.fmul",
    "/": "llvm.fdiv",
    "modulo": "llvm.frem",
}

_LLVM_OPS_F64: dict[str, str] = _LLVM_OPS_F32  # same LLVM ops for f64

_LLVM_OPS_I32: dict[str, str] = {
    "+": "llvm.add",
    "-": "llvm.sub",
    "*": "llvm.mul",
    "/": "llvm.sdiv",
    "modulo": "llvm.srem",
}

_LLVM_OPS_I1: dict[str, str] = {
    "&&": "llvm.and",
    "||": "llvm.or",
    "==": "llvm.icmp \"eq\"",
    "!=": "llvm.icmp \"ne\"",
}

_LLVM_OPS_BY_TYPE: dict[str, dict[str, str]] = {
    "f32": _LLVM_OPS_F32,
    "f64": _LLVM_OPS_F64,
    "i32": _LLVM_OPS_I32,
    "i1": _LLVM_OPS_I1,
}


def llvm_op(op: str, element_type: str) -> str:
    """Return the ``llvm.*`` MLIR operation for GPU kernels."""
    table = _LLVM_OPS_BY_TYPE.get(element_type)
    if table is None:
        raise KeyError(f"no llvm ops defined for type {element_type!r}")
    try:
        return table[op]
    except KeyError:
        raise KeyError(f"operator {op!r} not defined for LLVM type {element_type!r}")


# ---------------------------------------------------------------------------
# Comparison predicate helpers
# ---------------------------------------------------------------------------

_CMP_PREDICATES: dict[str, tuple[str, str | None]] = {
    # (i32_predicate, f32_predicate)
    "<": ("slt", "olt"),
    "<=": ("sle", "ole"),
    ">": ("sgt", "ogt"),
    ">=": ("sge", "oge"),
    "==": ("eq", "oeq"),
    "!=": ("ne", "one"),
}


def comparison_predicate(op: str, operand_type: str) -> str:
    """Return the ``arith.cmpi`` / ``arith.cmpf`` predicate attribute."""
    preds = _CMP_PREDICATES.get(op)
    if preds is None:
        raise KeyError(f"no comparison predicate for {op!r}")
    i_pred, f_pred = preds
    if operand_type == "i32":
        return i_pred
    if operand_type in ("f32", "f64"):
        return f_pred
    raise KeyError(f"no comparison predicate for operand type {operand_type!r}")


def comparison_mlir_op(op: str, operand_type: str) -> str:
    """Return ``arith.cmpi`` or ``arith.cmpf`` for the given comparison."""
    if op not in COMPARISON_OPS:
        raise KeyError(f"{op!r} is not a comparison operator")
    return "arith.cmpi" if operand_type == "i32" else "arith.cmpf"


# ---------------------------------------------------------------------------
# PTX assembly operators (legacy codegen path)
# ---------------------------------------------------------------------------

def ptx_op(op: str) -> str:
    """Return the PTX assembly operation for an f32 arithmetic operator."""
    ptx = {
        "*": "mul.rn.f32",
        "+": "add.rn.f32",
        "-": "sub.rn.f32",
        "/": "div.rn.f32",
    }
    try:
        return ptx[op]
    except KeyError:
        raise KeyError(f"no PTX op for {op!r}")


# ---------------------------------------------------------------------------
# Operator metadata record
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class OperatorMeta:
    """Metadata for one primitive Remora operator."""

    name: str
    arity: int = 2
    is_arithmetic: bool = False
    is_comparison: bool = False
    is_boolean: bool = False

    @property
    def is_numeric(self) -> bool:
        return self.is_arithmetic or self.is_comparison


_OPERATOR_TABLE: ClassVar[dict[str, OperatorMeta]] = {
    "+": OperatorMeta("+", is_arithmetic=True),
    "-": OperatorMeta("-", is_arithmetic=True),
    "*": OperatorMeta("*", is_arithmetic=True),
    "/": OperatorMeta("/", is_arithmetic=True),
    "modulo": OperatorMeta("modulo", is_arithmetic=True),
    "expt": OperatorMeta("expt", is_arithmetic=True),
    "exp": OperatorMeta("exp", arity=1, is_arithmetic=True),
    "log": OperatorMeta("log", arity=1, is_arithmetic=True),
    "sqrt": OperatorMeta("sqrt", arity=1, is_arithmetic=True),
    "cos": OperatorMeta("cos", arity=1, is_arithmetic=True),
    "sin": OperatorMeta("sin", arity=1, is_arithmetic=True),
    "ceil": OperatorMeta("ceil", arity=1, is_arithmetic=True),
    "floor": OperatorMeta("floor", arity=1, is_arithmetic=True),
    "<": OperatorMeta("<", is_comparison=True),
    "<=": OperatorMeta("<=", is_comparison=True),
    ">": OperatorMeta(">", is_comparison=True),
    ">=": OperatorMeta(">=", is_comparison=True),
    "==": OperatorMeta("==", is_comparison=True),
    "!=": OperatorMeta("!=", is_comparison=True),
    "&&": OperatorMeta("&&", is_boolean=True),
    "||": OperatorMeta("||", is_boolean=True),
}


def operator_meta(op: str) -> OperatorMeta:
    """Return the metadata record for *op*."""
    try:
        return _OPERATOR_TABLE[op]
    except KeyError:
        raise KeyError(f"unknown operator {op!r}")
