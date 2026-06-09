"""End-to-end differentiable programming demo.

Exercises the full Phase 7 + AD pipeline:
  Pi-typed loss functions → grad → tape → finite-diff validation → compiled execution
"""

import numpy as np
import pytest

from remora.ad import grad_via_tape
from remora.ad_testing import grad_check
from remora.compiler import compile_function_source
from remora.lisp_reader import parse_lisp
from remora.typechecker import TypeChecker
from remora.types import ArrayType, FLOAT, FuncType, StaticDim


def _get_specialized_body(src, func_name, shape):
    """Compile source, return specialized typed body for the tape."""
    tc = TypeChecker()
    tc.check_program(parse_lisp(src))
    dim = StaticDim(shape[0])
    spec = tc._typed_top_level_function(
        tc._functions[func_name],
        FuncType((ArrayType(FLOAT, (dim,)),), FLOAT),
        tc._build_prelude_env(),
        index_args=(dim,),
    )
    return spec.body


def _compile_and_check(src, func_name, shape):
    """Verify the function compiles successfully."""
    dim = StaticDim(shape[0])
    art = compile_function_source(
        src, func_name,
        (ArrayType(FLOAT, (dim,)),),
        verify=False, include_prelude=False, syntax='lisp',
    )
    assert art.specialization_name is not None
    return art


def test_demo_square_loss():
    """f(x) = sum(x²).  Gradient at x is 2*x."""
    src = (
        "(define/pi ([n Dim]) "
        "  (sq-loss [x (Array Float n)] Float) "
        "  (fold + 0.0 (* x x)))"
    )

    # Compile
    art = _compile_and_check(src, "sq-loss", (5,))
    assert "sq-loss__n_5" in art.specialization_name

    # Gradient via tape
    body = _get_specialized_body(src, "sq-loss", (5,))
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    g = grad_via_tape(body, "x", x)
    np.testing.assert_array_almost_equal(g, [2.0, 4.0, 6.0, 8.0, 10.0])

    # Finite-diff check at random point
    rng = np.random.RandomState(42)
    x2 = rng.randn(5).astype(np.float64) * 2.0
    grad_check(lambda v: float(np.sum(v * v)), x2, grad_via_tape(body, "x", x2),
               label="sq-loss@5")


def test_demo_square_loss_multi_shape():
    """Same Pi-typed loss differentiates correctly at multiple shapes."""
    src = (
        "(define/pi ([n Dim]) "
        "  (sq-loss [x (Array Float n)] Float) "
        "  (fold + 0.0 (* x x)))"
    )

    for n in [1, 3, 7, 12]:
        body = _get_specialized_body(src, "sq-loss", (n,))
        x = np.arange(1.0, float(n + 1))
        g = grad_via_tape(body, "x", x)
        expected = 2.0 * x
        np.testing.assert_array_almost_equal(g, expected, decimal=4,
            err_msg=f"sq-loss gradient mismatch at n={n}")


def test_demo_linear_model_loss():
    """f(w) = sum((y - w*x)²).  Requires closure support in tape (deferred).

    Conceptual test: documents the expected behavior for multi-variable AD.
    """
    # When tape supports closures, this will work:
    # f(w) = (2-w)² + (4-2w)² = 5w² - 20w + 20
    # Gradient: 10w - 20. At w=0: -20. At w=2: 0.
    pass


def test_demo_grad_type_preserves_pi():
    """grad of Pi-typed loss is also Pi-typed."""
    src = (
        "(define/pi ([n Dim]) "
        "  (sq-loss [x (Array Float n)] Float) "
        "  (fold + 0.0 (* x x))) "
        "(grad sq-loss)"
    )
    tc = TypeChecker()
    typed = tc.check_program(parse_lisp(src))
    from remora.types import PiType
    assert isinstance(typed.type, PiType)
    inner = typed.type.body
    assert isinstance(inner, FuncType)
    assert inner.params[0] == inner.result  # input type = gradient type


def test_demo_grad_vs_fd_random_sizes():
    """Stress test: tape vs FD at many random sizes."""
    rng = np.random.RandomState(99)
    src = (
        "(define/pi ([n Dim]) "
        "  (sq-loss [x (Array Float n)] Float) "
        "  (fold + 0.0 (* x x)))"
    )

    for n in rng.randint(1, 25, size=8):
        body = _get_specialized_body(src, "sq-loss", (n,))
        x = rng.randn(n).astype(np.float64) * 2.0
        tape_g = grad_via_tape(body, "x", x)
        grad_check(lambda v: float(np.sum(v * v)), x, tape_g,
                   rtol=1e-4, label=f"stress@{n}")
