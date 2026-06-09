"""Reverse-mode AD via evaluation tape for Remora.

AD4: primitive derivative registry, conditionals, negation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from remora.ast_nodes import FloatLit, IntLit, OperatorFuncExpr, VarExpr
from remora.typechecker import (
    TypedApp,
    TypedCast,
    TypedExpr,
    TypedExprNode,
    TypedFold,
    TypedIf,
    TypedLambda,
    TypedLet,
    TypedMap,
    TypedOperatorFunc,
)
from remora.types import FLOAT


@dataclass
class TapeEntry:
    kind: str
    inputs: tuple[int, ...]
    saved: tuple[Any, ...]


@dataclass
class EvalTape:
    entries: list[TapeEntry] = field(default_factory=list)
    values: list[np.ndarray] = field(default_factory=list)
    input_indices: list[int] = field(default_factory=list)
    has_data_dependent_control_flow: bool = False

    def push(self, entry: TapeEntry, value: np.ndarray) -> int:
        idx = len(self.values)
        self.entries.append(entry)
        self.values.append(value)
        return idx

    def push_const(self, value: float) -> int:
        return self.push(TapeEntry("const", (), (value,)), np.asarray(value, dtype=np.float64))

    def push_input(self, value: np.ndarray) -> int:
        idx = self.push(TapeEntry("var", (), ()), value)
        self.input_indices.append(idx)
        return idx

    def reverse(self) -> dict[int, np.ndarray]:
        adjs: list[np.ndarray | None] = [None] * len(self.values)
        adjs[-1] = np.ones_like(self.values[-1], dtype=np.float64)
        for i in reversed(range(len(self.entries))):
            adj = adjs[i]
            if adj is None:
                continue
            e = self.entries[i]
            if e.kind == "add":
                _bcast_acc(adjs, e.inputs[0], adj, self.values[e.inputs[0]])
                _bcast_acc(adjs, e.inputs[1], adj, self.values[e.inputs[1]])
            elif e.kind == "sub":
                _bcast_acc(adjs, e.inputs[0], adj, self.values[e.inputs[0]])
                _bcast_acc(adjs, e.inputs[1], -adj, self.values[e.inputs[1]])
            elif e.kind == "mul":
                rv = np.asarray(e.saved[0], dtype=np.float64)
                lv = np.asarray(e.saved[1], dtype=np.float64)
                _bcast_acc(adjs, e.inputs[0], adj * rv, self.values[e.inputs[0]])
                _bcast_acc(adjs, e.inputs[1], adj * lv, self.values[e.inputs[1]])
            elif e.kind == "div":
                rv = np.asarray(e.saved[0], dtype=np.float64)
                lv = np.asarray(e.saved[1], dtype=np.float64)
                _bcast_acc(adjs, e.inputs[0], adj / rv, self.values[e.inputs[0]])
                _bcast_acc(adjs, e.inputs[1], -adj * lv / (rv * rv), self.values[e.inputs[1]])
            elif e.kind == "fold":
                iv = np.asarray(e.saved[0], dtype=np.float64)
                _accum(adjs, e.inputs[0], np.full_like(iv, adj.item()))
            elif e.kind == "neg":
                _bcast_acc(adjs, e.inputs[0], -adj, self.values[e.inputs[0]])
        return {idx: adjs[idx] for idx in range(len(adjs)) if adjs[idx] is not None}


def _bcast_acc(adjs, idx, contrib, target_val):
    contrib = np.asarray(contrib, dtype=np.float64)
    target = np.asarray(target_val, dtype=np.float64)
    if contrib.shape != target.shape and target.ndim < contrib.ndim:
        reduce_axes = tuple(range(contrib.ndim - target.ndim))
        contrib = contrib.sum(axis=reduce_axes, keepdims=False)
        if contrib.shape != target.shape:
            squeeze_axes = tuple(
                i for i, (cs, ts) in enumerate(zip(contrib.shape, target.shape))
                if ts == 1 and cs > 1
            )
            if squeeze_axes:
                contrib = contrib.sum(axis=squeeze_axes, keepdims=True)
    _accum(adjs, idx, contrib)


def _accum(adjs, idx, c):
    if adjs[idx] is None:
        adjs[idx] = np.asarray(c, dtype=np.float64)
    else:
        adjs[idx] = adjs[idx] + c


# ── Primitive derivative registry ─────────────────────────────────────────

_VJP_REGISTRY: dict[str, tuple[str, int]] = {
    "+": ("add", 0),
    "-": ("sub", 0),
    "*": ("mul", 2),
    "/": ("div", 2),
}

_INACTIVE_BINARY_OPS = {"<", "<=", ">", ">=", "==", "!=", "&&", "||"}


def _record_primitive(tape, op, left_idx, right_idx, left_val, right_val) -> int:
    info = _VJP_REGISTRY.get(op)
    if info is None:
        raise NotImplementedError(f"no VJP registered for {op!r}")
    kind, n_saved = info
    saved = (right_val, left_val) if n_saved == 2 else ()
    result = _apply_bin_op(op, left_val, right_val)
    return tape.push(TapeEntry(kind, (left_idx, right_idx), saved), result)


def _apply_bin_op(op, lv, rv):
    if op == "+": return lv + rv
    if op == "-": return lv - rv
    if op == "*": return lv * rv
    if op == "/": return lv / rv
    raise NotImplementedError(f"apply bin op: {op}")


# ── Traced evaluation ─────────────────────────────────────────────────────


def trace_expr(expr: TypedExpr, env: dict[str, int], tape: EvalTape) -> int:
    if isinstance(expr, TypedExprNode):
        return _trace_node(expr, env, tape)
    if isinstance(expr, TypedCast):
        return trace_expr(expr.value, env, tape)
    if isinstance(expr, TypedLet):
        vidx = trace_expr(expr.value, env, tape)
        return trace_expr(expr.body, {**env, expr.expr.name: vidx}, tape)
    if isinstance(expr, TypedApp):
        return _trace_app(expr, env, tape)
    if isinstance(expr, TypedFold):
        return _trace_fold(expr, env, tape)
    if isinstance(expr, TypedMap):
        return _trace_map(expr, env, tape)
    if isinstance(expr, TypedIf):
        return _trace_if(expr, env, tape)
    raise NotImplementedError(f"trace: {type(expr).__name__}")


def _value(tape: EvalTape, idx: int) -> np.ndarray:
    return tape.values[idx]


def _trace_node(expr: TypedExprNode, env: dict[str, int], tape: EvalTape) -> int:
    ast = expr.expr
    if isinstance(ast, FloatLit):
        return tape.push_const(ast.value)
    if isinstance(ast, IntLit):
        return tape.push_const(float(ast.value))
    if isinstance(ast, VarExpr):
        if ast.name in env:
            return env[ast.name]
        raise RuntimeError(f"free variable {ast.name!r}")
    raise NotImplementedError(f"trace node: {type(ast).__name__}")


def _trace_app(expr: TypedApp, env: dict[str, int], tape: EvalTape) -> int:
    args = [trace_expr(a, env, tape) for a in expr.args]
    if len(args) != 2:
        raise NotImplementedError("trace app: non-binary")
    left_idx, right_idx = args
    op = _get_op(expr.func)
    if op in _INACTIVE_BINARY_OPS:
        result = _apply_inactive_bin_op(
            op, _value(tape, left_idx), _value(tape, right_idx)
        )
        return tape.push(TapeEntry("inactive", (left_idx, right_idx), ()), result)
    return _record_primitive(tape, op, left_idx, right_idx, _value(tape, left_idx), _value(tape, right_idx))


def _apply_inactive_bin_op(op: str, left, right):
    if op == "<":
        return left < right
    if op == "<=":
        return left <= right
    if op == ">":
        return left > right
    if op == ">=":
        return left >= right
    if op == "==":
        return left == right
    if op == "!=":
        return left != right
    if op == "&&":
        return np.logical_and(left, right)
    if op == "||":
        return np.logical_or(left, right)
    raise NotImplementedError(f"inactive binary op: {op}")


def _trace_fold(expr: TypedFold, env: dict[str, int], tape: EvalTape) -> int:
    arr_idx = trace_expr(expr.array, env, tape)
    arr_val = _value(tape, arr_idx)
    return tape.push(TapeEntry("fold", (arr_idx,), (arr_val,)), arr_val.sum())


def _trace_map(expr, env: dict[str, int], tape: EvalTape) -> int:
    if not expr.cell_shape:
        if len(expr.arrays) == 2:
            left_idx = trace_expr(expr.arrays[0], env, tape)
            right_idx = trace_expr(expr.arrays[1], env, tape)
            op = _get_op(expr.func)
            return _record_primitive(tape, op, left_idx, right_idx, _value(tape, left_idx), _value(tape, right_idx))
    if expr.arrays:
        return trace_expr(expr.arrays[0], env, tape)
    raise NotImplementedError("trace map: no arrays")


def _trace_if(expr, env: dict[str, int], tape: EvalTape) -> int:
    tape.has_data_dependent_control_flow = True
    cond_idx = trace_expr(expr.condition, env, tape)
    if np.any(_value(tape, cond_idx)):
        return trace_expr(expr.then_branch, env, tape)
    return trace_expr(expr.else_branch, env, tape)


def _get_op(func: object) -> str:
    from remora.ast_nodes import OperatorFuncExpr, VarExpr
    if isinstance(func, TypedExprNode):
        f = func.expr
        if isinstance(f, OperatorFuncExpr): return f.op
        if isinstance(f, VarExpr): return f.name
    if isinstance(func, TypedOperatorFunc):
        return func.expr.op
    return ""


# ── Gradient computation ──────────────────────────────────────────────────


def grad_via_tape(body: TypedExpr, param_name: str, x: np.ndarray) -> np.ndarray:
    tape, x_idx = trace_via_tape(body, param_name, x)
    adjs = tape.reverse()
    grad = adjs.get(x_idx)
    if grad is None:
        raise RuntimeError("AD: input not found on tape")
    return np.asarray(grad).reshape(np.asarray(x).shape)


def trace_via_tape(
    body: TypedExpr,
    param_name: str,
    x: np.ndarray,
) -> tuple[EvalTape, int]:
    """Trace one specialized unary function body and return its tape and input."""
    tape = EvalTape()
    x_idx = tape.push_input(np.asarray(x, dtype=np.float64))
    trace_expr(body, {param_name: x_idx}, tape)
    return tape, x_idx
