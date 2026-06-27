"""Shared lowering emission helpers.

The canonical descriptor-ABI helpers live in ``remora/gpu_lowering.py``:

* ``_descriptor_type(rank)`` — descriptor struct type name
* ``_descriptor_load_lines(prefix, descriptor_name, rank)`` — load runtime
  sizes and strides from a descriptor pointer (reused ~80×)
* ``_multi_index_lines(rank)`` — linear → multi-index via precomputed planes
  (llvm dialect, descriptor ABI)
* ``_linear_index_lines(prefix, rank)`` — multi-index → linear via strides
  (llvm dialect, descriptor ABI)

This module provides additional shared helpers and documentation so
that new lowering code does not re-invent these idioms.
"""

from __future__ import annotations

from typing import Literal


IRType = Literal["i8", "i1", "i32", "f32", "f64", "bool"]
LLVM_DIALECT_OPS: dict[tuple[str, str], str] = {
    ("+", "f32"): "llvm.fadd",
    ("-", "f32"): "llvm.fsub",
    ("*", "f32"): "llvm.fmul",
    ("/", "f32"): "llvm.fdiv",
    ("+", "i32"): "llvm.add",
    ("-", "i32"): "llvm.sub",
    ("*", "i32"): "llvm.mul",
    ("/", "i32"): "llvm.sdiv",
    ("&&", "i1"): "llvm.and",
    ("||", "i1"): "llvm.or",
    ("==", "i32"): "llvm.icmp \"eq\"",
    ("!=", "i32"): "llvm.icmp \"ne\"",
    ("<", "i32"): "llvm.icmp \"slt\"",
    ("<=", "i32"): "llvm.icmp \"sle\"",
    (">", "i32"): "llvm.icmp \"sgt\"",
    (">=", "i32"): "llvm.icmp \"sge\"",
    ("==", "f32"): "llvm.fcmp \"oeq\"",
    ("!=", "f32"): "llvm.fcmp \"one\"",
    ("<", "f32"): "llvm.fcmp \"olt\"",
    ("<=", "f32"): "llvm.fcmp \"ole\"",
    (">", "f32"): "llvm.fcmp \"ogt\"",
    (">=", "f32"): "llvm.fcmp \"oge\"",
}


def llvm_op(op: str, element_type: IRType) -> str:
    if op in ("&&", "||"):
        element_type = "i1"
    if op in ("==", "!=", "<", "<=", ">", ">="):
        if element_type in ("bool", "i8"):
            element_type = "i32"
    return LLVM_DIALECT_OPS[(op, element_type)]


def emit_delinearize(
    linear_var: str,
    sizes: list[str],
    *,
    indent: str = "      ",
    dialect: Literal["llvm", "arith"] = "llvm",
) -> list[str]:
    rank = len(sizes)
    if rank <= 0:
        return []

    lines: list[str] = []

    if dialect == "llvm":
        div_op = f"{indent}%{{idx_result}} = llvm.udiv {{remainder}}, {{plane_var}} : i64"
        rem_op = f"{indent}%{{rem_result}} = llvm.urem {{remainder}}, {{plane_var}} : i64"
        last_op = f"{indent}%{{idx_result}} = {{remainder}}"
    else:
        div_op = f"{indent}%{{idx_result}} = arith.divui {{remainder}}, {{plane_var}} : index"
        rem_op = f"{indent}%{{rem_result}} = arith.remui {{remainder}}, {{plane_var}} : index"
        last_op = f"{indent}%{{idx_result}} = {{remainder}}"

    if rank == 1:
        lines.append(last_op.format(idx_result=f"i0", remainder=linear_var))
        return lines

    planes: list[str] = []
    for axis in range(rank - 1):
        plane_vars = " * ".join(sizes[axis + 1 :])
        planes.append(f"{indent}%plane{axis} = arith.constant ({plane_vars}) : i64")

    for axis in range(rank - 1):
        lines.append(planes[axis])
    current = linear_var
    for axis in range(rank - 1):
        lines.append(div_op.format(idx_result=f"i{axis}", remainder=current, plane_var=f"%plane{axis}"))
        lines.append(rem_op.format(rem_result=f"rem{axis}", remainder=current, plane_var=f"%plane{axis}"))
        current = f"%rem{axis}"
    lines.append(f"{indent}%i{rank - 1} = llvm.add {current}, %index_zero : i64")

    return lines


def emit_2d_decompose(
    flat_var: str,
    row_var: str,
    col_var: str,
    n_cols: str,
    *,
    indent: str = "      ",
    dialect: Literal["llvm", "arith"] = "llvm",
) -> list[str]:
    if dialect == "llvm":
        div_v = f"{indent}%{{row}} = llvm.udiv {{flat}}, {{ncol}} : i64"
        rem_v = f"{indent}%{{col}} = llvm.urem {{flat}}, {{ncol}} : i64"
    else:
        div_v = f"{indent}%{{row}} = arith.divui {{flat}}, {{ncol}} : index"
        rem_v = f"{indent}%{{col}} = arith.remui {{flat}}, {{ncol}} : index"
    return [
        div_v.format(row=row_var, flat=flat_var, ncol=n_cols),
        rem_v.format(col=col_var, flat=flat_var, ncol=n_cols),
    ]
