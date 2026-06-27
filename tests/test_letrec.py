"""Tests for ``letrec`` (local recursive function bindings) in Lisp syntax.

``letrec`` is desugared (lambda-lifted) to top-level recursive functions before
type checking, so it rides the existing interpreter / CPU / GPU recursion
support.  See ``remora/_letrec_lift.py``.
"""

import typing
from dataclasses import fields

import pytest

from remora._letrec_lift import desugar_letrec
from remora.ast_nodes import AppExpr, Expr, FuncDef, LetRecExpr, VarExpr
from remora.errors import RemoraError
from remora.lisp_reader import parse_lisp
from remora.runtime import evaluate_source, evaluate_source_compiled

_EXPR_CLASSES = tuple(t for t in typing.get_args(Expr) if isinstance(t, type))


def _contains_letrec(expr) -> bool:
    if isinstance(expr, LetRecExpr):
        return True
    for f in fields(expr):
        value = getattr(expr, f.name)
        if isinstance(value, _EXPR_CLASSES) and _contains_letrec(value):
            return True
        if isinstance(value, (list, tuple)):
            for item in value:
                if isinstance(item, _EXPR_CLASSES) and _contains_letrec(item):
                    return True
    return False


def _lifted(program):
    return [
        d
        for d in program.definitions
        if isinstance(d, FuncDef) and d.name.startswith("__letrec")
    ]


class TestLetrecDesugar:
    def test_self_recursive_lifts_to_toplevel(self):
        prog = parse_lisp(
            "(letrec ((loop (lambda (i acc) "
            "  (if (== i 0) acc (loop (- i 1) (+ acc i)))))) (loop 5 0))"
        )
        lifted = _lifted(prog)
        assert len(lifted) == 1
        assert lifted[0].params == ["i", "acc"]  # no capture
        # No LetRecExpr survives anywhere.
        assert not _contains_letrec(prog.body)
        for d in prog.definitions:
            if isinstance(d, FuncDef):
                assert not _contains_letrec(d.body)
        # The body is now a call to the lifted function.
        assert isinstance(prog.body, AppExpr)
        assert isinstance(prog.body.func, VarExpr)
        assert prog.body.func.name == lifted[0].name

    def test_capture_threads_free_var_as_leading_param(self):
        prog = parse_lisp(
            "(define/pi () (f [base Float n Float] Float) "
            "  (letrec ((go (lambda (i acc) "
            "    (if (< i 0.5) acc (go (- i 1.0) (+ acc base)))))) (go n 0.0)))"
        )
        lifted = _lifted(prog)
        assert len(lifted) == 1
        # captured 'base' becomes the leading parameter
        assert lifted[0].params == ["base", "i", "acc"]

    def test_mutual_recursion_lifts_two_functions(self):
        prog = parse_lisp(
            "(define/pi () (g [n Float] Float) "
            "  (letrec ((even (lambda (k) (if (== k 0.0) 1.0 (odd (- k 1.0))))) "
            "           (odd (lambda (k) (if (== k 0.0) 0.0 (even (- k 1.0)))))) (even n)))"
        )
        assert len(_lifted(prog)) == 2

    def test_no_letrec_is_identity(self):
        prog = parse_lisp("(define (f [x]) (+ x 1)) (f 2)")
        assert desugar_letrec(prog) is prog


class TestLetrecInterpreter:
    def _ev(self, src):
        return evaluate_source(src, syntax="lisp", include_prelude=False).value

    def test_toplevel_int_loop(self):
        assert (
            self._ev(
                "(letrec ((loop (lambda (i acc) "
                "  (if (== i 0) acc (loop (- i 1) (+ acc i)))))) (loop 5 0))"
            )
            == 15
        )

    def test_capture(self):
        assert self._ev(
            "(define/pi () (f [base Float n Float] Float) "
            "  (letrec ((go (lambda (i acc) "
            "    (if (< i 0.5) acc (go (- i 1.0) (+ acc base)))))) (go n 0.0)))"
            "(f 3.0 4.0)"
        ) == pytest.approx(12.0)

    def test_mutual(self):
        assert self._ev(
            "(define/pi () (g [n Float] Float) "
            "  (letrec ((even (lambda (k) (if (== k 0.0) 1.0 (odd (- k 1.0))))) "
            "           (odd (lambda (k) (if (== k 0.0) 0.0 (even (- k 1.0)))))) (even n)))"
            "(g 7.0)"
        ) == pytest.approx(0.0)

    def test_nested(self):
        assert (
            self._ev(
                "(letrec ((outer (lambda (n) "
                "  (letrec ((inner (lambda (i acc) "
                "    (if (== i 0) acc (inner (- i 1) (+ acc 1)))))) (inner n 0))))) "
                "(outer 4))"
            )
            == 4
        )

    def test_newton(self):
        assert self._ev(
            "(define/pi () (sqrt2 [x0 Float] Float) "
            "  (letrec ((go (lambda (x) "
            "    (if (< (* (- (* x x) 2.0) (- (* x x) 2.0)) 0.00000001) "
            "        x (go (- x (/ (- (* x x) 2.0) (* 2.0 x)))))))) (go x0)))"
            "(sqrt2 1.0)"
        ) == pytest.approx(2.0 ** 0.5, abs=1e-3)


class TestLetrecCompiledCPU:
    def test_capture_cpu(self):
        r = evaluate_source_compiled(
            "(define/pi () (f [base Float n Float] Float) "
            "  (letrec ((go (lambda (i acc) "
            "    (if (< i 0.5) acc (go (- i 1.0) (+ acc base)))))) (go n 0.0)))"
            "(f 3.0 4.0)",
            syntax="lisp",
            include_prelude=False,
        )
        assert r.value == pytest.approx(12.0)

    def test_toplevel_loop_cpu(self):
        r = evaluate_source_compiled(
            "(letrec ((loop (lambda (i acc) "
            "  (if (== i 0) acc (loop (- i 1) (+ acc i)))))) (loop 5 0))",
            syntax="lisp",
            include_prelude=False,
        )
        assert r.value == 15

    def test_let_in_recursive_body_cpu(self):
        # An untyped (lifted) recursive helper whose body wraps the terminal
        # `if` in a `let`, with an inferred result type.  This exercises the
        # recursion result-type back-substitution over `TypedLet`
        # (regression lock for the `_substitute_type_var` fix).
        src = (
            "(define/pi () (f [n Float] Float) "
            "  (letrec ((go (lambda (k acc) "
            "    (let ((d (- k 1.0))) "
            "      (if (< k 0.5) acc (go d (+ acc k))))))) (go n 0.0)))"
            "(f 4.0)"
        )
        assert evaluate_source(
            src, syntax="lisp", include_prelude=False
        ).value == pytest.approx(10.0)
        assert evaluate_source_compiled(
            src, syntax="lisp", include_prelude=False
        ).value == pytest.approx(10.0)


class TestLetrecRejected:
    def test_non_lambda_binding(self):
        with pytest.raises(RemoraError, match="must be a"):
            evaluate_source(
                "(letrec ((x 5)) x)", syntax="lisp", include_prelude=False
            )

    def test_used_as_value(self):
        with pytest.raises(RemoraError, match="used as a value"):
            evaluate_source(
                "(define/pi () (f [xs (Array Float 2)] (Array Float 2)) "
                "  (letrec ((g (lambda (x) (g x)))) (map g xs)))"
                "(f [1.0 2.0])",
                syntax="lisp",
                include_prelude=False,
            )

    def test_param_capture_collision(self):
        with pytest.raises(RemoraError, match="shadow captured"):
            evaluate_source(
                "(define/pi () (f [c Float n Float] Float) "
                "  (letrec ((a (lambda (k) (if (< k 0.5) c (b (- k 1.0))))) "
                "           (b (lambda (c) (if (< c 0.5) 0.0 (a (- c 1.0)))))) (a n)))"
                "(f 1.0 3.0)",
                syntax="lisp",
                include_prelude=False,
            )


class TestWhileDotimes:
    """``while`` and ``dotimes`` desugar to tail-recursive ``letrec`` loops."""

    def _ev(self, src):
        return evaluate_source(src, syntax="lisp", include_prelude=False).value

    def _cpu(self, src):
        return evaluate_source_compiled(
            src, syntax="lisp", include_prelude=False
        ).value

    def test_while_countdown_sum(self):
        src = "(while (< 0.0 n) ((n 5.0 (- n 1.0)) (acc 0.0 (+ acc n))) acc)"
        assert self._ev(src) == pytest.approx(15.0)
        assert self._cpu(src) == pytest.approx(15.0)

    def test_while_parallel_update_fib(self):
        # Bindings update simultaneously: 10 steps of (a, b) <- (a+b, a)
        # starting from (0, 1) yields fib(10) = 55.
        src = "(while (< i 10) ((i 0 (+ i 1)) (a 0 (+ a b)) (b 1 a)) a)"
        assert self._ev(src) == 55

    def test_while_desugars_to_lifted_letrec(self):
        prog = parse_lisp(
            "(while (< 0.0 n) ((n 5.0 (- n 1.0)) (acc 0.0 (+ acc n))) acc)"
        )
        assert _lifted(prog)
        assert not _contains_letrec(prog.body)

    def test_dotimes_int_sum(self):
        src = "(dotimes (i 5) (acc 0) (+ acc i))"  # 0+1+2+3+4
        assert self._ev(src) == 10
        assert self._cpu(src) == 10

    def test_dotimes_float_acc(self):
        src = "(dotimes (i 4) (acc 0.0) (+ acc 2.0))"  # add 2.0 four times
        assert self._ev(src) == pytest.approx(8.0)

    def test_dotimes_desugars_to_lifted_letrec(self):
        prog = parse_lisp("(dotimes (i 5) (acc 0) (+ acc i))")
        assert _lifted(prog)
        assert not _contains_letrec(prog.body)
