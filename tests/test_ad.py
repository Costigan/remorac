"""Tests for AD1: scalar reverse-mode with evaluation tape."""

import numpy as np
import pytest

from remora.ad import EvalTape, TapeEntry, grad_via_tape
from remora.ad_testing import finite_difference_grad, grad_check
from remora.typechecker import TypedApp, TypedExprNode
from remora.ast_nodes import FloatLit, VarExpr, OperatorFuncExpr, IntLit, SourceLoc, FoldExpr
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


# ── AD2: array broadcasting ────────────────────────────────────────────────


def test_tape_broadcast_add_scalar():
    """(+ arr 1.0) where arr is [3], 1.0 is scalar — cotangent of 1.0 is sum."""
    t = EvalTape()
    arr = t.push_input(np.asarray([1.0, 2.0, 3.0]))
    one = t.push_const(1.0)
    r = t.push(TapeEntry("add", (arr, one), ()), np.asarray([2.0, 3.0, 4.0]))
    adjs = t.reverse()

    np.testing.assert_array_equal(adjs[arr], [1.0, 1.0, 1.0])
    assert adjs[one] == pytest.approx(3.0)  # sum([1,1,1])


def test_tape_broadcast_mul_scalar():
    """(* arr 2.0) — d/darr = 2, d/d2 = sum(arr) = 6"""
    t = EvalTape()
    arr = t.push_input(np.asarray([1.0, 2.0, 3.0]))
    two = t.push_const(2.0)
    r = t.push(
        TapeEntry("mul", (arr, two), (np.asarray(2.0), np.asarray([1.0, 2.0, 3.0]))),
        np.asarray([2.0, 4.0, 6.0]),
    )
    adjs = t.reverse()
    np.testing.assert_array_equal(adjs[arr], [2.0, 2.0, 2.0])
    assert adjs[two] == pytest.approx(6.0)  # sum([1,2,3])


def test_grad_array_square():
    """grad of fold + 0 (* x x) at x=[1,2,3] → 2*x = [2,4,6]"""
    body = _make_fold(_app(_op("*"), _var("x"), _var("x")), _var("x"))
    g = grad_via_tape(body, "x", np.array([1.0, 2.0, 3.0]))
    np.testing.assert_array_almost_equal(g, [2.0, 4.0, 6.0])


def test_grad_array_sum_of_squares():
    """grad of fold + 0 (* x x) matches finite differences."""
    body = _make_fold(_app(_op("*"), _var("x"), _var("x")), _var("x"))
    rng = np.random.RandomState(42)
    x = rng.randn(4)
    tape_g = grad_via_tape(body, "x", x)

    def f(v):
        return float(np.sum(v * v))
    grad_check(f, x, tape_g, label="sum_sq")


def _make_fold(body, array_expr):
    """Build a TypedFold: fold + 0 where body is the element-wise expression
    applied to the array expression.  The 'array' field is the element-wise
    result, which the fold then sums."""
    from remora.typechecker import TypedFold
    from remora.ast_nodes import FoldExpr
    from remora.types import StaticDim

    init = _lit(0.0)
    func = _op("+")
    return TypedFold(
        FoldExpr(func.expr, init.expr, array_expr.expr, _LOC),
        func, init, body,
        StaticDim(0), FLOAT,
    )


# ── AD4: additional VJPs ──────────────────────────────────────────────────

def test_tape_neg():
    """neg(x): VJP = -adj"""
    t = EvalTape()
    x = t.push_input(np.asarray([1.0, 2.0, 3.0]))
    r = t.push(TapeEntry("neg", (x,), ()), np.asarray([-1.0, -2.0, -3.0]))
    adjs = t.reverse()
    np.testing.assert_array_equal(adjs[x], [-1.0, -1.0, -1.0])


# ── AD5: performance / correctness benchmarks ──────────────────────────────

def test_tape_vs_fd_accuracy():
    """Tape gradient must match finite differences within tight tolerance."""
    rng = np.random.RandomState(123)
    for _ in range(20):
        x = rng.randn(rng.randint(1, 10)) * 2.0
        body = _make_fold(_app(_op("*"), _var("x"), _var("x")), _var("x"))
        tape_g = grad_via_tape(body, "x", x)

        def f(v):
            return float(np.sum(v * v))
        grad_check(f, x, tape_g, rtol=1e-6, atol=1e-8, label="bench_fd")


def test_tape_vs_fd_speed():
    """Tape should be faster than finite differences for moderate arrays."""
    import time
    rng = np.random.RandomState(42)
    body = _make_fold(_app(_op("*"), _var("x"), _var("x")), _var("x"))

    sizes = [5, 10, 50, 100]
    for n in sizes:
        x = rng.randn(n)
        t0 = time.perf_counter()
        for _ in range(10):
            grad_via_tape(body, "x", x)
        tape_time = (time.perf_counter() - t0) / 10

        def f(v):
            return float(np.sum(v * v))
        t0 = time.perf_counter()
        for _ in range(10):
            finite_difference_grad(f, x)
        fd_time = (time.perf_counter() - t0) / 10

        speedup = fd_time / tape_time if tape_time > 0 else float('inf')
        # Tape should be faster (n evaluations vs 2n+1 evaluations)
        assert speedup > 1.0, f"Tape not faster than FD for n={n}"


@pytest.mark.parametrize("shape,func_name", [
    ((3,), "sq"),
    ((5,), "sq"),
    ((10,), "sq"),
    ((4,), "sq"),
])
def test_compiled_cross_validation(shape, func_name):
    """Tape gradient on compiled specialized body must match finite differences."""
    from remora.compiler import compile_function_source
    from remora.typechecker import TypeChecker
    from remora.lisp_reader import parse_lisp
    from remora.types import ArrayType, FLOAT, StaticDim, FuncType

    src = '''(define/pi ([n Dim]) (sq [x (Array Float n)] Float) (fold + 0.0 (* x x)))'''
    static_dim = StaticDim(shape[0])

    # Compile it
    art = compile_function_source(
        src, func_name,
        (ArrayType(FLOAT, (static_dim,)),),
        verify=False, include_prelude=False, syntax='lisp',
    )
    assert art.specialization_name is not None

    # Get specialized body for tape
    tc = TypeChecker()
    tc.check_program(parse_lisp(src))
    spec = tc._typed_top_level_function(
        tc._functions[func_name],
        FuncType((ArrayType(FLOAT, (static_dim,)),), FLOAT),
        tc._build_prelude_env(),
        index_args=(static_dim,),
    )

    rng = np.random.RandomState(42 + shape[0])
    x = rng.randn(*shape).astype(np.float64) * 2.0
    tape_g = grad_via_tape(spec.body, "x", x)

    def f(v):
        return float(np.sum(v * v))
    grad_check(f, x, tape_g, label=f"compiled_{func_name}_{shape}")
