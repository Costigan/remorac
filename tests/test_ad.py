"""Tests for AD: reverse-mode with evaluation tape."""

import numpy as np
import pytest

from remora.ad import EvalTape, TapeEntry, grad_via_tape, trace_via_tape_multi
from remora.ad_testing import finite_difference_grad, grad_check
from remora.typechecker import TypedApp, TypedExprNode, TypedAppend, TypedSubarray
from remora.typechecker import TypedIf
from remora.ast_nodes import FloatLit, VarExpr, OperatorFuncExpr, IntLit, SourceLoc, FoldExpr, IfExpr, AppendExpr
from remora.types import FLOAT, ArrayType, StaticDim, FuncType

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


def test_tape_exp_and_log():
    exp_tape = EvalTape()
    exp_x = exp_tape.push_input(np.asarray(2.0))
    exp_tape.push(TapeEntry("exp", (exp_x,), ()), np.exp(2.0))
    assert exp_tape.reverse()[exp_x] == pytest.approx(np.exp(2.0))

    log_tape = EvalTape()
    log_x = log_tape.push_input(np.asarray(2.0))
    log_tape.push(TapeEntry("log", (log_x,), ()), np.log(2.0))
    assert log_tape.reverse()[log_x] == pytest.approx(0.5)


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


def test_tape_fold_reduces_only_the_recorded_axis():
    values = np.arange(1.0, 7.0).reshape(2, 3)
    tape = EvalTape()
    array = tape.push_input(values)
    tape.push(
        TapeEntry("fold", (array,), (values, 1)),
        values.sum(axis=1),
    )

    adjs = tape.reverse()

    np.testing.assert_array_equal(adjs[array], np.ones_like(values))


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


def test_conditional_predicate_is_inactive():
    condition = _app(_op(">"), _var("x"), _lit(0.0))
    square = _app(_op("*"), _var("x"), _var("x"))
    negative = _app(_op("-"), _lit(0.0), _var("x"))
    body = TypedIf(
        IfExpr(condition.expr, square.expr, negative.expr, _LOC),
        condition,
        square,
        negative,
        FLOAT,
    )

    assert grad_via_tape(body, "x", np.asarray(3.0)) == pytest.approx(6.0)
    assert grad_via_tape(body, "x", np.asarray(-3.0)) == pytest.approx(-1.0)


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


# ── AD append VJP ───────────────────────────────────────────────────────────


def test_tape_append_distinct():
    t = EvalTape()
    left = t.push_input(np.asarray([1.0, 2.0, 3.0]))
    right = t.push_input(np.asarray([4.0, 5.0]))
    t.push(
        TapeEntry("append", (left, right), ((3,), 3)),
        np.asarray([1.0, 2.0, 3.0, 4.0, 5.0]),
    )
    adjs = t.reverse()
    np.testing.assert_array_equal(adjs[left], [1.0, 1.0, 1.0])
    np.testing.assert_array_equal(adjs[right], [1.0, 1.0])


def test_tape_append_repeated():
    t = EvalTape()
    x = t.push_input(np.asarray([1.0, 2.0, 3.0]))
    t.push(
        TapeEntry("append", (x, x), ((3,), 3)),
        np.asarray([1.0, 2.0, 3.0, 1.0, 2.0, 3.0]),
    )
    adjs = t.reverse()
    np.testing.assert_array_equal(adjs[x], [2.0, 2.0, 2.0])


def test_tape_append_rank2():
    t = EvalTape()
    left = t.push_input(np.asarray([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]))
    right = t.push_input(np.asarray([[7.0, 8.0], [9.0, 10.0]]))
    t.push(
        TapeEntry("append", (left, right), ((3, 2), 3)),
        np.asarray(
            [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0], [9.0, 10.0]]
        ),
    )
    adjs = t.reverse()
    np.testing.assert_array_equal(adjs[left], np.ones((3, 2)))
    np.testing.assert_array_equal(adjs[right], np.ones((2, 2)))


def _specialize_body(source: str, func_name: str, param_type):
    from remora.typechecker import TypeChecker
    from remora.lisp_reader import parse_lisp
    from remora.types import FuncType, ScalarType

    tc = TypeChecker()
    tc.check_program(parse_lisp(source))
    if isinstance(param_type, ScalarType):
        index_args = ()
    else:
        index_args = tuple(d for d in param_type.shape if hasattr(d, "value"))
    spec = tc._typed_top_level_function(
        tc._functions[func_name],
        FuncType((param_type,), FLOAT),
        tc._build_prelude_env(),
        index_args=index_args,
    )
    return spec.body, spec.params[0][0]


def test_grad_append_same_input_square():
    source = (
        "(define/pi () "
        "  (loss [x (Array Float 3)] Float) "
        "  (fold + 0.0 (* (append x x) (append x x))))"
    )
    from remora.types import ArrayType, StaticDim
    body, pname = _specialize_body(
        source, "loss", ArrayType(FLOAT, (StaticDim(3),))
    )
    x = np.array([1.0, 2.0, 3.0])
    g = grad_via_tape(body, pname, x)
    np.testing.assert_array_almost_equal(g, 4.0 * x)


def test_grad_append_transformed_operands():
    source = (
        "(define/pi () "
        "  (loss [x (Array Float 3)] Float) "
        "  (fold + 0.0 (* (append x (* 2.0 x)) "
        "                 (append x (* 2.0 x)))))"
    )
    from remora.types import ArrayType, StaticDim
    body, pname = _specialize_body(
        source, "loss", ArrayType(FLOAT, (StaticDim(3),))
    )
    x = np.array([1.0, 2.0, 3.0])
    g = grad_via_tape(body, pname, x)
    np.testing.assert_array_almost_equal(g, 10.0 * x)


def test_append_vs_finite_diff():
    source = (
        "(define/pi () "
        "  (loss [x (Array Float 4)] Float) "
        "  (fold + 0.0 (* (append x x) (append x x))))"
    )
    from remora.types import ArrayType, StaticDim
    body, pname = _specialize_body(
        source, "loss", ArrayType(FLOAT, (StaticDim(4),))
    )
    rng = np.random.RandomState(123)
    x = rng.randn(4) * 2.0
    tape_g = grad_via_tape(body, pname, x)

    def f(v):
        v = np.asarray(v, dtype=np.float64)
        cat = np.concatenate([v, v])
        return float(np.sum(cat * cat))
    grad_check(f, x, tape_g, label="append_sq")


def test_append_rank2_vs_finite_diff():
    source = (
        "(define/pi () "
        "  (loss [x (Array Float 4 2)] Float) "
        "  (fold + 0.0 (ravel (* (append x x) (append x x)))))"
    )
    from remora.types import ArrayType, StaticDim
    body, pname = _specialize_body(
        source, "loss", ArrayType(FLOAT, (StaticDim(4), StaticDim(2)))
    )
    rng = np.random.RandomState(456)
    x = rng.randn(4, 2) * 2.0
    tape_g = grad_via_tape(body, pname, x)

    def f(v):
        v = np.asarray(v, dtype=np.float64)
        cat = np.concatenate([v, v])
        return float(np.sum(cat * cat))
    grad_check(f, x, tape_g, label="append_rank2_sq")


# ── AD subarray VJP ──────────────────────────────────────────────────────────


def test_tape_subarray():
    t = EvalTape()
    arr = t.push_input(np.asarray([10.0, 20.0, 30.0, 40.0, 50.0]))
    t.push(
        TapeEntry("subarray", (arr,), ((5,), (2,), (3,))),
        np.asarray([30.0, 40.0, 50.0]),
    )
    adjs = t.reverse()
    np.testing.assert_array_equal(adjs[arr], [0.0, 0.0, 1.0, 1.0, 1.0])


def test_tape_subarray_from_start():
    t = EvalTape()
    arr = t.push_input(np.asarray([1.0, 2.0, 3.0, 4.0]))
    t.push(
        TapeEntry("subarray", (arr,), ((4,), (0,), (2,))),
        np.asarray([1.0, 2.0]),
    )
    adjs = t.reverse()
    np.testing.assert_array_equal(adjs[arr], [1.0, 1.0, 0.0, 0.0])


def test_grad_subarray_square():
    source = (
        "(define/pi () "
        "  (loss [x (Array Float 5)] Float) "
        "  (fold + 0.0 (* (subarray x [2] [3]) (subarray x [2] [3]))))"
    )
    from remora.types import ArrayType, StaticDim
    body, pname = _specialize_body(
        source, "loss", ArrayType(FLOAT, (StaticDim(5),))
    )
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    g = grad_via_tape(body, pname, x)
    expected = np.zeros(5, dtype=np.float64)
    expected[2:5] = 2.0 * np.array([3.0, 4.0, 5.0])
    np.testing.assert_array_almost_equal(g, expected)


def test_subarray_vs_finite_diff():
    source = (
        "(define/pi () "
        "  (loss [x (Array Float 6)] Float) "
        "  (fold + 0.0 (* (subarray x [1] [4]) (subarray x [1] [4]))))"
    )
    from remora.types import ArrayType, StaticDim
    body, pname = _specialize_body(
        source, "loss", ArrayType(FLOAT, (StaticDim(6),))
    )
    rng = np.random.RandomState(789)
    x = rng.randn(6) * 2.0
    tape_g = grad_via_tape(body, pname, x)

    def f(v):
        v = np.asarray(v, dtype=np.float64)
        sub = v[1:5]
        return float(np.sum(sub * sub))
    grad_check(f, x, tape_g, label="subarray_sq")


# ── AD rotate VJP ──────────────────────────────────────────────────────────


def test_tape_rotate():
    t = EvalTape()
    arr = t.push_input(np.asarray([1.0, 2.0, 3.0, 4.0]))
    t.push(
        TapeEntry("rotate", (arr,), (2, 4)),
        np.asarray([3.0, 4.0, 1.0, 2.0]),
    )
    adjs = t.reverse()
    np.testing.assert_array_equal(adjs[arr], [1.0, 1.0, 1.0, 1.0])


def test_grad_rotate_square():
    source = (
        "(define/pi () "
        "  (loss [x (Array Float 4)] Float) "
        "  (fold + 0.0 (* (rotate x 1) (rotate x 1))))"
    )
    from remora.types import ArrayType, StaticDim
    body, pname = _specialize_body(
        source, "loss", ArrayType(FLOAT, (StaticDim(4),))
    )
    x = np.array([1.0, 2.0, 3.0, 4.0])
    g = grad_via_tape(body, pname, x)
    np.testing.assert_array_almost_equal(g, 2.0 * x)


def test_rotate_vs_finite_diff():
    source = (
        "(define/pi () "
        "  (loss [x (Array Float 5)] Float) "
        "  (fold + 0.0 (* (rotate x 2) (rotate x 2))))"
    )
    from remora.types import ArrayType, StaticDim
    body, pname = _specialize_body(
        source, "loss", ArrayType(FLOAT, (StaticDim(5),))
    )
    rng = np.random.RandomState(42)
    x = rng.randn(5) * 2.0
    tape_g = grad_via_tape(body, pname, x)

    def f(v):
        v = np.asarray(v, dtype=np.float64)
        rotated = np.roll(v, -2)
        return float(np.sum(rotated * rotated))
    grad_check(f, x, tape_g, label="rotate_sq")


# ── AD index VJP ──────────────────────────────────────────────────────────


def test_tape_index():
    t = EvalTape()
    arr = t.push_input(np.asarray([10.0, 20.0, 30.0, 40.0]))
    t.push(
        TapeEntry("index", (arr,), ((4,), (2,))),
        np.asarray(30.0),
    )
    adjs = t.reverse()
    np.testing.assert_array_equal(adjs[arr], [0.0, 0.0, 1.0, 0.0])


def test_grad_index_square():
    source = (
        "(define/pi () "
        "  (loss [x (Array Float 4)] Float) "
        "  (* (index x 2) (index x 2)))"
    )
    from remora.types import ArrayType, StaticDim
    body, pname = _specialize_body(
        source, "loss", ArrayType(FLOAT, (StaticDim(4),))
    )
    x = np.array([1.0, 2.0, 3.0, 4.0])
    g = grad_via_tape(body, pname, x)
    expected = np.zeros(4, dtype=np.float64)
    expected[2] = 6.0
    np.testing.assert_array_almost_equal(g, expected)


def test_index_vs_finite_diff():
    source = (
        "(define/pi () "
        "  (loss [x (Array Float 5)] Float) "
        "  (* (index x 3) (index x 3)))"
    )
    from remora.types import ArrayType, StaticDim
    body, pname = _specialize_body(
        source, "loss", ArrayType(FLOAT, (StaticDim(5),))
    )
    rng = np.random.RandomState(99)
    x = rng.randn(5) * 2.0
    tape_g = grad_via_tape(body, pname, x)

    def f(v):
        v = np.asarray(v, dtype=np.float64)
        return float(v[3] * v[3])
    grad_check(f, x, tape_g, label="index_sq")


# ── AD select VJP ─────────────────────────────────────────────────────────


def test_tape_select_scalar():
    t = EvalTape()
    x = t.push_input(np.asarray(3.0))
    cond = t.push(TapeEntry("inactive", (x, x), (">",)), np.asarray(True))
    then_val = t.push(TapeEntry("mul", (x, x), (np.asarray(3.0), np.asarray(3.0))), np.asarray(9.0))
    else_val = t.push(TapeEntry("neg", (x,), ()), np.asarray(-3.0))
    t.push(TapeEntry("select", (cond, then_val, else_val), ()), np.asarray(9.0))
    adjs = t.reverse()
    # adj = 1.0, cond=True → then branch gets adj → mul(x,x) d/dx = 2*x = 6.0
    assert adjs[x] == pytest.approx(6.0)


def test_grad_select_scalar():
    source = (
        "(define/pi () "
        "  (loss [x Float] Float) "
        "  (if (> x 0.0) (* x x) (- 0.0 x)))"
    )
    from remora.types import FuncType
    body, pname = _specialize_body(
        source, "loss", FLOAT
    )
    # at x=3: loss = 9.0, grad = 6.0
    g = grad_via_tape(body, pname, np.asarray(3.0))
    assert g == pytest.approx(6.0, rel=1e-4)
    # at x=-3: loss = 3.0, grad = -1.0
    g = grad_via_tape(body, pname, np.asarray(-3.0))
    assert g == pytest.approx(-1.0, rel=1e-4)


# ── Liveness / buffer reuse test ────────────────────────────────────────────


def test_tape_liveness_frees_values():
    """Chained operations should not accumulate all primals in memory."""
    t = EvalTape()
    x = t.push_input(np.array([1.0, 2.0, 3.0]))
    cur = x
    for _ in range(10):
        cur = t.push(
            TapeEntry("add", (cur, cur), ()),
            t.values[cur] + t.values[cur],
        )
    t.push(TapeEntry("fold", (cur,), (t.values[cur],)), t.values[cur].sum())
    adjs = t.reverse()
    expected = np.full(3, 2.0**10, dtype=np.float64)
    np.testing.assert_array_equal(adjs[x], expected)
    freed_count = sum(1 for v in t.values if v is None)
    assert freed_count >= 8, f"expected >= 8 freed, got {freed_count}"


# ── Section 1 helpers: multi-param source specialization ────────────────────


def _substitute_bindings(type_, bindings):
    """Substitute DimExpr bindings in a type."""
    from remora.typechecker import substitute_type
    return substitute_type(type_, bindings)


def _specialize_source_multi(source: str, func_name: str, *param_types):
    """Typecheck Lisp source, specialize at concrete types, return (body, param_names)."""
    from remora.typechecker import TypeChecker
    from remora.lisp_reader import parse_lisp

    tc = TypeChecker()
    tc.check_program(parse_lisp(source))
    function = tc._functions[func_name]

    declared_param_types = tc._declared_param_types(function)
    if declared_param_types is not None:
        bindings = tc._infer_index_bindings(
            function, declared_param_types, param_types,
        )
        specialized_params = tuple(
            _substitute_bindings(pt, bindings) for pt in declared_param_types
        )
        declared_result = tc._declared_result_type(function)
        specialized_result = (
            _substitute_bindings(declared_result, bindings)
            if declared_result is not None
            else FLOAT
        )
    else:
        specialized_params = param_types
        specialized_result = FLOAT

    index_args = tc._inferred_index_args(function, param_types)
    spec = tc._typed_top_level_function(
        function,
        FuncType(specialized_params, specialized_result),
        tc._build_prelude_env(),
        index_args=index_args,
    )
    return spec.body, [name for name, _ in spec.params]


def _multi_tape_gradients(body, param_names, values):
    """Trace multi-param body and return gradient dict {name: np.array}."""
    tape, indices = trace_via_tape_multi(
        body, [np.asarray(v, dtype=np.float64) for v in values], param_names,
    )
    adjs = tape.reverse()
    return {name: np.asarray(adjs[idx]) for name, idx in zip(param_names, indices)}


# ── Section 1: vector-cell map + fold AD ────────────────────────────────────


_VEC_SUM_SRC = """\
(define/pi ([h Dim] [w Dim])
  (vec-sum [x (Array Float h w)] (Array Float h))
  (map (lambda (row) (fold + 0.0 row)) x))
"""


def test_vec_sum_forward():
    """Forward: (map (lambda (row) (fold + 0.0 row)) x) sums each row."""
    body, [pname] = _specialize_source_multi(
        _VEC_SUM_SRC, "vec-sum",
        ArrayType(FLOAT, (StaticDim(2), StaticDim(3))),
    )
    from remora.ad import trace_expr
    tape = EvalTape()
    x = tape.push_input(np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float64))
    trace_expr(body, {pname: x}, tape)
    np.testing.assert_array_equal(tape.values[-1], [6.0, 15.0])


def test_vec_sum_gradient():
    """Gradient of row sum = all-ones of same shape."""
    body, [pname] = _specialize_source_multi(
        _VEC_SUM_SRC, "vec-sum",
        ArrayType(FLOAT, (StaticDim(2), StaticDim(3))),
    )
    x = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    g = grad_via_tape(body, pname, x)
    np.testing.assert_array_equal(g, np.ones_like(x))


def test_vec_sum_finite_diff():
    """Row-sum gradient matches finite differences."""
    body, [pname] = _specialize_source_multi(
        _VEC_SUM_SRC, "vec-sum",
        ArrayType(FLOAT, (StaticDim(4), StaticDim(3))),
    )
    rng = np.random.RandomState(99)
    x = rng.randn(4, 3)
    tape_g = grad_via_tape(body, pname, x)

    def f(v):
        return float(np.sum(v.sum(axis=1)))
    grad_check(f, x, tape_g, label="vec_sum", rtol=1e-6)


def test_tape_vector_cell_map_fold():
    """Tape correctly traces vector-cell map with fold (constructed directly)."""
    from remora.typechecker import TypedMap, TypedFold, TypedLambda
    from remora.ast_nodes import MapExpr, LambdaExpr, FoldExpr

    frame = (StaticDim(2),)
    cell = (StaticDim(3),)
    array_type = ArrayType(FLOAT, frame + cell)
    cell_type = ArrayType(FLOAT, cell)

    _loc = SourceLoc("test", 0, 0)
    row_var = TypedExprNode(VarExpr("row", _loc), cell_type)
    init = TypedExprNode(FloatLit(0.0, _loc), FLOAT)
    plus_op = TypedExprNode(OperatorFuncExpr("+", _loc), FLOAT)

    body = TypedFold(
        FoldExpr(plus_op.expr, init.expr, row_var.expr, _loc),
        plus_op, init, row_var,
        StaticDim(0), FLOAT,
    )
    lam = TypedLambda(
        LambdaExpr(["row"], body.expr, _loc),
        [("row", cell_type)], body,
        FuncType((cell_type,), FLOAT),
    )
    x = TypedExprNode(VarExpr("x", _loc), array_type)
    map_expr = MapExpr(lam.expr, [x.expr], _loc)
    typed_map = TypedMap(map_expr, lam, [x], frame, cell, ArrayType(FLOAT, frame))

    tape = EvalTape()
    arr = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float64)
    arr_idx = tape.push_input(arr)
    from remora.ad import trace_expr
    trace_expr(typed_map, {"x": arr_idx}, tape)
    np.testing.assert_array_equal(tape.values[-1], [6.0, 15.0])

    adjs = tape.reverse()
    np.testing.assert_array_equal(adjs[arr_idx], np.ones_like(arr))


def test_tape_vector_cell_map_fold_captured():
    """Tape correctly handles captured variable in vector-cell map + fold."""
    from remora.typechecker import TypedMap, TypedFold, TypedLambda, TypedApp
    from remora.ast_nodes import MapExpr, LambdaExpr, FoldExpr, AppExpr

    frame = (StaticDim(2),)
    cell = (StaticDim(3),)
    array_type = ArrayType(FLOAT, frame + cell)
    cell_type = ArrayType(FLOAT, cell)

    _loc = SourceLoc("test", 0, 0)
    row_var = TypedExprNode(VarExpr("row", _loc), cell_type)
    x_var = TypedExprNode(VarExpr("x", _loc), cell_type)
    init = TypedExprNode(FloatLit(0.0, _loc), FLOAT)
    plus_op = TypedExprNode(OperatorFuncExpr("+", _loc), FLOAT)
    mul_op = TypedExprNode(OperatorFuncExpr("*", _loc), FLOAT)

    mul_body = TypedApp(
        AppExpr(mul_op.expr, [row_var.expr, x_var.expr], _loc),
        mul_op, [row_var, x_var], cell_type,
    )
    fold_body = TypedFold(
        FoldExpr(plus_op.expr, init.expr, mul_body.expr, _loc),
        plus_op, init, mul_body,
        StaticDim(0), FLOAT,
    )
    lam = TypedLambda(
        LambdaExpr(["row"], fold_body.expr, _loc),
        [("row", cell_type)], fold_body,
        FuncType((cell_type,), FLOAT),
    )
    w = TypedExprNode(VarExpr("w", _loc), array_type)
    map_expr = MapExpr(lam.expr, [w.expr], _loc)
    typed_map = TypedMap(map_expr, lam, [w], frame, cell, ArrayType(FLOAT, frame))

    weights = np.array([[0.5, -1.0, 2.0], [1.5, 0.25, -0.75]], dtype=np.float64)
    features = np.array([2.0, -0.5, 1.25], dtype=np.float64)

    tape = EvalTape()
    w_idx = tape.push_input(weights)
    x_idx = tape.push_input(features)
    from remora.ad import trace_expr
    trace_expr(typed_map, {"w": w_idx, "x": x_idx}, tape)
    np.testing.assert_array_almost_equal(tape.values[-1], [4.0, 1.9375])

    adjs = tape.reverse()
    expected_w_grad = np.tile(features, (2, 1))
    expected_x_grad = weights.sum(axis=0)
    np.testing.assert_array_almost_equal(adjs[w_idx], expected_w_grad)
    np.testing.assert_array_almost_equal(adjs[x_idx], expected_x_grad)


def test_tape_vector_cell_map_fold_captured_finite_diff():
    """Captured-variable vector-cell map gradient matches finite differences."""
    from remora.typechecker import TypedMap, TypedFold, TypedLambda, TypedApp
    from remora.ast_nodes import MapExpr, LambdaExpr, FoldExpr, AppExpr

    for h, w in [(2, 3), (4, 3), (3, 5)]:
        frame = (StaticDim(h),)
        cell = (StaticDim(w),)
        array_type = ArrayType(FLOAT, frame + cell)
        cell_type = ArrayType(FLOAT, cell)

        _loc = SourceLoc("test", 0, 0)
        row_var = TypedExprNode(VarExpr("row", _loc), cell_type)
        x_var = TypedExprNode(VarExpr("x", _loc), cell_type)
        init = TypedExprNode(FloatLit(0.0, _loc), FLOAT)
        plus_op = TypedExprNode(OperatorFuncExpr("+", _loc), FLOAT)
        mul_op = TypedExprNode(OperatorFuncExpr("*", _loc), FLOAT)

        mul_body = TypedApp(
            AppExpr(mul_op.expr, [row_var.expr, x_var.expr], _loc),
            mul_op, [row_var, x_var], cell_type,
        )
        fold_body = TypedFold(
            FoldExpr(plus_op.expr, init.expr, mul_body.expr, _loc),
            plus_op, init, mul_body,
            StaticDim(0), FLOAT,
        )
        lam = TypedLambda(
            LambdaExpr(["row"], fold_body.expr, _loc),
            [("row", cell_type)], fold_body,
            FuncType((cell_type,), FLOAT),
        )
        w_var = TypedExprNode(VarExpr("w", _loc), array_type)
        map_expr = MapExpr(lam.expr, [w_var.expr], _loc)
        typed_map = TypedMap(map_expr, lam, [w_var], frame, cell, ArrayType(FLOAT, frame))

        rng = np.random.RandomState(77 + h)
        weights = rng.randn(h, w)
        features = rng.randn(w)

        tape = EvalTape()
        w_idx = tape.push_input(weights)
        x_idx = tape.push_input(features)
        from remora.ad import trace_expr
        trace_expr(typed_map, {"w": w_idx, "x": x_idx}, tape)
        adjs = tape.reverse()

        def loss_w(candidate):
            return float(np.sum(candidate @ features))

        def loss_x(candidate):
            return float(np.sum(weights @ candidate))

        grad_check(loss_w, weights, adjs[w_idx], label=f"vec_map_fd_w_{h}x{w}", rtol=1e-6)
        grad_check(loss_x, features, adjs[x_idx], label=f"vec_map_fd_x_{h}x{w}", rtol=1e-6)


# ── Section 2: model composition (named helpers) ────────────────────────────


_NESTED_HELPER_SRC = """\
(define/pi ()
  (dot [a (Array Float 3) b (Array Float 3)] Float)
  (fold + 0.0 (map * a b)))

(define/pi ()
  (linear [w (Array Float 2 3) x (Array Float 3)] (Array Float 2))
  (map (lambda (row) (dot row x)) w))

(define/pi ()
  (loss [w (Array Float 2 3) x (Array Float 3)] Float)
  (fold + 0.0 (* (linear w x) (linear w x))))
"""


def test_nested_helper_forward():
    """Named helpers dot -> linear -> loss. Forward execution through tape."""
    body, pnames = _specialize_source_multi(
        _NESTED_HELPER_SRC, "loss",
        ArrayType(FLOAT, (StaticDim(2), StaticDim(3))),
        ArrayType(FLOAT, (StaticDim(3),)),
    )
    weights = np.array([[0.5, -1.0, 2.0], [1.5, 0.25, -0.75]], dtype=np.float64)
    features = np.array([2.0, -0.5, 1.25], dtype=np.float64)

    from remora.ad import trace_expr
    tape = EvalTape()
    w_idx = tape.push_input(weights)
    x_idx = tape.push_input(features)
    trace_expr(body, {pnames[0]: w_idx, pnames[1]: x_idx}, tape)

    projected = weights @ features
    expected_loss = float(np.sum(projected * projected))
    np.testing.assert_almost_equal(tape.values[-1], expected_loss)


def test_nested_helper_gradient():
    """Named helper composition produces correct gradients."""
    body, pnames = _specialize_source_multi(
        _NESTED_HELPER_SRC, "loss",
        ArrayType(FLOAT, (StaticDim(2), StaticDim(3))),
        ArrayType(FLOAT, (StaticDim(3),)),
    )
    weights = np.array([[0.5, -1.0, 2.0], [1.5, 0.25, -0.75]], dtype=np.float64)
    features = np.array([2.0, -0.5, 1.25], dtype=np.float64)

    grads = _multi_tape_gradients(body, pnames, [weights, features])

    projected = weights @ features
    expected_w = 2.0 * projected[:, np.newaxis] * features[np.newaxis, :]
    expected_x = 2.0 * (weights.T @ projected)

    np.testing.assert_array_almost_equal(grads[pnames[0]], expected_w)
    np.testing.assert_array_almost_equal(grads[pnames[1]], expected_x)


def test_nested_helper_finite_diff():
    """Named helpers: tape gradient matches finite differences for both params."""
    body, pnames = _specialize_source_multi(
        _NESTED_HELPER_SRC, "loss",
        ArrayType(FLOAT, (StaticDim(2), StaticDim(3))),
        ArrayType(FLOAT, (StaticDim(3),)),
    )
    rng = np.random.RandomState(44)
    weights = rng.randn(2, 3)
    features = rng.randn(3)
    grads = _multi_tape_gradients(body, pnames, [weights, features])

    def loss_w(candidate):
        projected = candidate @ features
        return float(np.sum(projected * projected))

    def loss_x(candidate):
        projected = weights @ candidate
        return float(np.sum(projected * projected))

    grad_check(loss_w, weights, grads[pnames[0]], label="nested_w", rtol=1e-5)
    grad_check(loss_x, features, grads[pnames[1]], label="nested_x", rtol=1e-5)


def test_scalar_named_helper():
    """A simple named helper (square) in a scalar loss."""
    _SRC = """\
(define/pi ()
  (sq [x Float] Float)
  (* x x))
(define/pi ()
  (loss [x Float] Float)
  (sq x))
"""
    body, [pname] = _specialize_source_multi(
        _SRC, "loss", FLOAT,
    )
    g = grad_via_tape(body, pname, np.asarray(3.0))
    assert g == pytest.approx(6.0)
    g = grad_via_tape(body, pname, np.asarray(-2.0))
    assert g == pytest.approx(-4.0)
