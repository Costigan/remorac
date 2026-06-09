"""Generate reusable Remora gradient source from an evaluation tape."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TypeAlias

import numpy as np

from remora.ad import EvalTape


Shape = tuple[int, ...]


@dataclass(frozen=True)
class _Atom:
    text: str
    shape: Shape


@dataclass(frozen=True)
class _Op:
    op: str
    left: "_Expr"
    right: "_Expr | None"
    shape: Shape


@dataclass(frozen=True)
class _Fill:
    value: "_Expr"
    like: "_Expr"
    shape: Shape


_Expr: TypeAlias = _Atom | _Op | _Fill


def generate_gradient_source(
    tape: EvalTape,
    param_name: str,
    param_shape: tuple[int, ...],
    *,
    function_name: str = "grad-f",
) -> str:
    """Return a unary Float gradient function for the traced tape graph.

    The generated function recomputes symbolic primal intermediates from its
    parameter. Concrete tape values are used only to recover source constants.
    """
    if not tape.entries:
        raise ValueError("cannot generate a gradient from an empty tape")
    if len(tape.input_indices) != 1:
        raise ValueError("gradient source currently requires exactly one tape input")

    input_idx = tape.input_indices[0]
    shape = tuple(int(dim) for dim in param_shape)
    if tuple(np.asarray(tape.values[input_idx]).shape) != shape:
        raise ValueError("parameter shape does not match the traced tape input")

    primals = _reconstruct_primals(tape, input_idx, param_name)
    adjs: list[_Expr | None] = [None] * len(tape.entries)
    output_shape = _shape_of(tape.values[-1])
    adjs[-1] = _constant(1.0, output_shape)

    for index in reversed(range(len(tape.entries))):
        adj = adjs[index]
        if adj is None:
            continue
        entry = tape.entries[index]
        if entry.kind == "add":
            _accumulate(
                adjs, entry.inputs[0], _unbroadcast(adj, primals[entry.inputs[0]])
            )
            _accumulate(
                adjs, entry.inputs[1], _unbroadcast(adj, primals[entry.inputs[1]])
            )
        elif entry.kind == "sub":
            _accumulate(
                adjs, entry.inputs[0], _unbroadcast(adj, primals[entry.inputs[0]])
            )
            _accumulate(
                adjs,
                entry.inputs[1],
                _unbroadcast(_neg(adj), primals[entry.inputs[1]]),
            )
        elif entry.kind == "mul":
            left, right = entry.inputs
            _accumulate(
                adjs,
                left,
                _unbroadcast(_binary("*", adj, primals[right]), primals[left]),
            )
            _accumulate(
                adjs,
                right,
                _unbroadcast(_binary("*", adj, primals[left]), primals[right]),
            )
        elif entry.kind == "div":
            left, right = entry.inputs
            _accumulate(
                adjs,
                left,
                _unbroadcast(_binary("/", adj, primals[right]), primals[left]),
            )
            numerator = _binary("*", adj, primals[left])
            denominator = _binary("*", primals[right], primals[right])
            _accumulate(
                adjs,
                right,
                _unbroadcast(_neg(_binary("/", numerator, denominator)), primals[right]),
            )
        elif entry.kind == "fold":
            operand = entry.inputs[0]
            _accumulate(adjs, operand, _fill(adj, primals[operand]))
        elif entry.kind == "neg":
            _accumulate(adjs, entry.inputs[0], _neg(adj))
        elif entry.kind in {"const", "var"}:
            continue
        else:
            raise NotImplementedError(f"gradient source VJP: {entry.kind}")

    gradient = adjs[input_idx]
    if gradient is None:
        raise RuntimeError("AD source: input not found on tape")
    param_type = _source_type(shape)
    body = _emit(gradient)
    return (
        f"(define/pi () ({function_name} [{param_name} {param_type}] {param_type}) "
        f"{body})"
    )


def _reconstruct_primals(tape: EvalTape, input_idx: int, param_name: str) -> list[_Expr]:
    primals: list[_Expr] = []
    for index, entry in enumerate(tape.entries):
        shape = _shape_of(tape.values[index])
        if entry.kind == "var":
            if index != input_idx:
                raise ValueError("gradient source encountered an unsupported extra input")
            expr = _Atom(param_name, shape)
        elif entry.kind == "const":
            expr = _constant_value(entry.saved[0], shape)
        elif entry.kind in {"add", "sub", "mul", "div"}:
            op = {"add": "+", "sub": "-", "mul": "*", "div": "/"}[entry.kind]
            expr = _binary(op, primals[entry.inputs[0]], primals[entry.inputs[1]])
        elif entry.kind == "fold":
            operand = primals[entry.inputs[0]]
            expr = _Op("fold", operand, None, shape)
        elif entry.kind == "neg":
            expr = _neg(primals[entry.inputs[0]])
        else:
            raise NotImplementedError(f"gradient source primal: {entry.kind}")
        primals.append(expr)
    return primals


def _shape_of(value: object) -> Shape:
    return tuple(int(dim) for dim in np.asarray(value).shape)


def _constant(value: float, shape: Shape = ()) -> _Atom:
    return _Atom(_format_float(value), shape)


def _constant_value(value: object, shape: Shape) -> _Atom:
    array = np.asarray(value)
    if array.ndim != 0:
        raise NotImplementedError("array constants are not supported in gradient source")
    return _constant(float(array), shape)


def _format_float(value: float) -> str:
    if not np.isfinite(value):
        raise ValueError("gradient source cannot serialize non-finite constants")
    text = repr(float(value))
    return text if "." in text or "e" in text.lower() else f"{text}.0"


def _source_type(shape: Shape) -> str:
    if not shape:
        return "Float"
    return f"(Array Float {' '.join(str(dim) for dim in shape)})"


def _binary(op: str, left: _Expr, right: _Expr) -> _Expr:
    result_shape = left.shape if len(left.shape) >= len(right.shape) else right.shape
    if op == "+":
        if _is_zero(left):
            return right
        if _is_zero(right):
            return left
        if left == right:
            return _binary("*", _constant(2.0), left)
    if op == "-":
        if _is_zero(right):
            return left
        if left == right and not result_shape:
            return _constant(0.0, result_shape)
    if op == "*":
        if not result_shape and (_is_zero(left) or _is_zero(right)):
            return _constant(0.0, result_shape)
        if _is_one(left):
            return right
        if _is_one(right):
            return left
        if isinstance(left, _Fill) and _is_one(left.value) and left.shape == right.shape:
            return right
        if isinstance(right, _Fill) and _is_one(right.value) and right.shape == left.shape:
            return left
    if op == "/":
        if not result_shape and _is_zero(left):
            return _constant(0.0, result_shape)
        if _is_one(right):
            return left
    return _Op(op, left, right, result_shape)


def _neg(expr: _Expr) -> _Expr:
    if _is_zero(expr):
        return expr
    if isinstance(expr, _Op) and expr.op == "neg":
        assert expr.right is None
        return expr.left
    return _binary("-", _constant(0.0), expr)


def _fill(value: _Expr, like: _Expr) -> _Expr:
    if value.shape == like.shape:
        return value
    return _Fill(value, like, like.shape)


def _unbroadcast(expr: _Expr, target: _Expr) -> _Expr:
    if expr.shape == target.shape:
        return expr
    if target.shape == ():
        return _Op("fold", expr, None, ())
    raise NotImplementedError(
        f"gradient source cannot unbroadcast {expr.shape} to {target.shape}"
    )


def _accumulate(adjs: list[_Expr | None], index: int, contribution: _Expr) -> None:
    current = adjs[index]
    adjs[index] = contribution if current is None else _binary("+", current, contribution)


def _is_constant(expr: _Expr, value: float) -> bool:
    if not isinstance(expr, _Atom) or expr.shape:
        return False
    try:
        return float(expr.text) == value
    except ValueError:
        return False


def _is_zero(expr: _Expr) -> bool:
    return _is_constant(expr, 0.0)


def _is_one(expr: _Expr) -> bool:
    return _is_constant(expr, 1.0)


def _emit(expr: _Expr) -> str:
    if isinstance(expr, _Atom):
        return expr.text
    if isinstance(expr, _Fill):
        return _emit(_binary("+", _binary("*", _constant(0.0), expr.like), expr.value))
    if expr.op == "fold":
        return f"(fold + 0.0 {_emit(expr.left)})"
    if expr.op == "neg":
        return f"(- {_emit(expr.left)})"
    assert expr.right is not None
    left = _emit(expr.left)
    right = _emit(expr.right)
    if expr.left.shape == () and expr.right.shape:
        if expr.op in {"+", "*"}:
            return f"(map ({expr.op} {left}) {right})"
        return f"(map (lambda (v) ({expr.op} {left} v)) {right})"
    if expr.left.shape and expr.right.shape == ():
        return f"(map (lambda (v) ({expr.op} v {right})) {left})"
    if expr.left.shape or expr.right.shape:
        return f"(map {expr.op} {left} {right})"
    return f"({expr.op} {left} {right})"
