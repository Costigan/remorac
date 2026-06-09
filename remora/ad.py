"""Reverse-mode AD via evaluation tape for the Remora interpreter.

AD3: preserves Pi-typed gradient signatures.
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
    TypedLambda,
    TypedLet,
    TypedMap,
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


# ── Traced evaluation (returns tape index) ─────────────────────────────────


def trace_expr(expr: TypedExpr, env: dict[str, int], tape: EvalTape) -> int:
    if isinstance(expr, TypedExprNode):
        return _trace_node(expr, env, tape)
    if isinstance(expr, TypedCast):
        return trace_expr(expr.value, env, tape)
    if isinstance(expr, TypedLet):
        vidx = trace_expr(expr.value, env, tape)
        local = dict(env)
        local[expr.expr.name] = vidx
        return trace_expr(expr.body, local, tape)
    if isinstance(expr, TypedApp):
        return _trace_app(expr, env, tape)
    if isinstance(expr, TypedFold):
        return _trace_fold(expr, env, tape)
    if isinstance(expr, TypedMap):
        return _trace_map(expr, env, tape)
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
        raise RuntimeError(f"unexpected free variable {ast.name!r} in trace")
    raise NotImplementedError(f"trace node: {type(ast).__name__}")


def _trace_app(expr: TypedApp, env: dict[str, int], tape: EvalTape) -> int:
    func = expr.func
    args = [trace_expr(a, env, tape) for a in expr.args]
    if len(args) != 2:
        raise NotImplementedError("trace app: non-binary")
    left_idx, right_idx = args
    left_val = _value(tape, left_idx)
    right_val = _value(tape, right_idx)
    op = ""
    if isinstance(func, TypedExprNode):
        f = func.expr
        if isinstance(f, OperatorFuncExpr):
            op = f.op
        elif isinstance(f, VarExpr):
            op = f.name
    return _record_primitive(tape, op, left_idx, right_idx, left_val, right_val)


def _record_primitive(tape, op, left_idx, right_idx, left_val, right_val) -> int:
    if op == "+":
        return tape.push(TapeEntry("add", (left_idx, right_idx), ()), left_val + right_val)
    elif op == "-":
        return tape.push(TapeEntry("sub", (left_idx, right_idx), ()), left_val - right_val)
    elif op == "*":
        return tape.push(TapeEntry("mul", (left_idx, right_idx), (right_val, left_val)), left_val * right_val)
    elif op == "/":
        return tape.push(TapeEntry("div", (left_idx, right_idx), (right_val, left_val)), left_val / right_val)
    raise NotImplementedError(f"record primitive: {op}")


def _trace_fold(expr: TypedFold, env: dict[str, int], tape: EvalTape) -> int:
    arr_idx = trace_expr(expr.array, env, tape)
    arr_val = _value(tape, arr_idx)
    result = arr_val.sum()
    return tape.push(TapeEntry("fold", (arr_idx,), (arr_val,)), result)


def _trace_map(expr, env: dict[str, int], tape: EvalTape) -> int:
    if not expr.cell_shape:
        if len(expr.arrays) == 1:
            return trace_expr(expr.arrays[0], env, tape)
        if len(expr.arrays) == 2:
            left_idx = trace_expr(expr.arrays[0], env, tape)
            right_idx = trace_expr(expr.arrays[1], env, tape)
            func = expr.func
            op = _get_op(func)
            return _record_primitive(tape, op, left_idx, right_idx, _value(tape, left_idx), _value(tape, right_idx))
    if expr.arrays:
        return trace_expr(expr.arrays[0], env, tape)
    raise NotImplementedError("trace map: no arrays")


def _get_op(func: object) -> str:
    """Extract the operator name from a function expression."""
    from remora.typechecker import TypedOperatorFunc, TypedExprNode
    from remora.ast_nodes import OperatorFuncExpr, VarExpr
    if isinstance(func, TypedExprNode):
        f = func.expr
        if isinstance(f, OperatorFuncExpr):
            return f.op
        if isinstance(f, VarExpr):
            return f.name
    if isinstance(func, TypedOperatorFunc):
        return func.expr.op
    return ""


# ── Top-level gradient computation ─────────────────────────────────────────


def grad_via_tape(body: TypedExpr, param_name: str, x: np.ndarray) -> np.ndarray:
    tape = EvalTape()
    input_val = np.asarray(x, dtype=np.float64)
    x_idx = tape.push_input(input_val)
    env: dict[str, int] = {param_name: x_idx}
    trace_expr(body, env, tape)
    adjs = tape.reverse()
    grad = adjs.get(x_idx)
    if grad is None:
        raise RuntimeError("AD: input not found on tape")
    return np.asarray(grad).reshape(np.asarray(x).shape)
