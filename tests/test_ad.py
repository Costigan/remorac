"""Tests for AD1: scalar reverse-mode with evaluation tape."""

import numpy as np
import pytest

from remora.ad import EvalTape, TapeEntry, grad_via_tape
from remora.ad_testing import finite_difference_grad, grad_check
from remora.typechecker import TypedApp, TypedExprNode
from remora.ast_nodes import FloatLit, VarExpr, OperatorFuncExpr, IntLit, SourceLoc
from remora.types import FLOAT

_LOC = SourceLoc("test", 0, 0)


def _var(name: str) -> TypedExprNode:
    return TypedExprNode(VarExpr(name, _LOC), FLOAT)

def _lit(v: float) -> TypedExprNode:
    return TypedExprNode(FloatLit(v, _LOC), FLOAT)

def _op(op: str) -> TypedExprNode:
    return TypedExprNode(OperatorFuncExpr(op, _LOC), FLOAT)

def _app(func, left, right):
    return TypedApp(None, func, [left, right], FLOAT)  # type: ignore


# ── Tape unit tests ─────────────────────────────────────────────────────────


def test_tape_add():
    t = EvalTape()
    x = t.push_input(np.asarray(3.0))
    y = t.push_input(np.asarray(4.0))
    r = t.push(TapeEntry("add", (x, y), ()), np.asarray(7.0))
    adjs = t.reverse()
    assert adjs[x] == pytest.approx(1.0)
    assert adjs[y] == pytest.approx(1.0)


def test_tape_mul():
    t = EvalTape()
    x = t.push_input(np.asarray(3.0))
    y = t.push_input(np.asarray(4.0))
    r = t.push(TapeEntry("mul", (x, y), (np.asarray(4.0), np.asarray(3.0))), np.asarray(12.0))
    adjs = t.reverse()
    assert adjs[x] == pytest.approx(4.0)
    assert adjs[y] == pytest.approx(3.0)


def test_tape_div():
    t = EvalTape()
    x = t.push_input(np.asarray(6.0))
    y = t.push_input(np.asarray(2.0))
    r = t.push(TapeEntry("div", (x, y), (np.asarray(2.0), np.asarray(6.0))), np.asarray(3.0))
    adjs = t.reverse()
    assert adjs[x] == pytest.approx(0.5)
    assert adjs[y] == pytest.approx(-1.5)


def test_tape_composition():
    t = EvalTape()
    x = t.push_input(np.asarray(3.0))
    x2 = t.push(TapeEntry("mul", (x, x), (np.asarray(3.0), np.asarray(3.0))), np.asarray(9.0))
    r = t.push(TapeEntry("add", (x2, x), ()), np.asarray(12.0))
    adjs = t.reverse()
    assert adjs[x] == pytest.approx(7.0)  # 2*3 + 1


def test_tape_fold():
    t = EvalTape()
    arr = t.push_input(np.asarray([1.0, 2.0, 3.0]))
    r = t.push(TapeEntry("fold", (arr,), (np.asarray([1.0, 2.0, 3.0]),)), np.asarray(6.0))
    adjs = t.reverse()
    np.testing.assert_array_equal(adjs[arr], [1.0, 1.0, 1.0])


# ── grad_via_tape tests ────────────────────────────────────────────────────


def test_grad_square():
    """grad of x*x at x=3 → 6"""
    body = _app(_op("*"), _var("x"), _var("x"))
    g = grad_via_tape(body, "x", np.array([3.0]))
    assert g[0] == pytest.approx(6.0, rel=1e-4)


def test_grad_quadratic():
    """grad of x*x + x at x=3 → 7"""
    inner = _app(_op("*"), _var("x"), _var("x"))
    body = _app(_op("+"), inner, _var("x"))
    g = grad_via_tape(body, "x", np.array([3.0]))
    assert g[0] == pytest.approx(7.0, rel=1e-4)


def test_grad_polynomial():
    """grad of (x+1)*(x-2) at x=3 → 2x-1 = 5"""
    t1 = _app(_op("+"), _var("x"), _lit(1.0))
    t2 = _app(_op("-"), _var("x"), _lit(2.0))
    body = _app(_op("*"), t1, t2)
    g = grad_via_tape(body, "x", np.array([3.0]))
    assert g[0] == pytest.approx(5.0, rel=1e-4)


def test_grad_vs_finite_diff():
    """Tape gradient matches finite differences for random input."""
    rng = np.random.RandomState(42)
    x = rng.randn(1) * 2.0
    body = _app(
        _op("*"),
        _app(_op("+"), _var("x"), _lit(1.0)),
        _app(_op("-"), _var("x"), _lit(2.0)),
    )
    tape_g = grad_via_tape(body, "x", x)

    def f(v):
        return float((v[0] + 1.0) * (v[0] - 2.0))
    grad_check(f, x, tape_g, label="polynomial")
