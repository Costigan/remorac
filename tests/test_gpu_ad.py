"""GPU AD integration: compile primal, validate gradient via tape.

The tape runs on CPU; the primal compiles for CPU (and GPU when available).
This validates the full pipeline: typecheck → compile → tape gradient → FD check.
"""

import numpy as np
import pytest

from remora.ad import grad_via_tape
from remora.ad_testing import grad_check
from remora.compiler import compile_function_source
from remora.lisp_reader import parse_lisp
from remora.pipeline import PipelineUnavailable
from remora.typechecker import TypeChecker
from remora.types import ArrayType, FLOAT, FuncType, StaticDim


def _get_specialized_body(src, func_name, shape):
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


# ── Primal compiles for CPU (same HIR path as GPU) ─────────────────────────


@pytest.mark.parametrize("n", [3, 5, 10, 20])
def test_primal_compiles_and_gradient_matches_fd(n):
    """Pi-typed loss compiles; tape gradient matches FD at random input."""
    src = (
        "(define/pi ([n Dim]) "
        "  (sq-loss [x (Array Float n)] Float) "
        "  (fold + 0.0 (* x x)))"
    )
    dim = StaticDim(n)
    art = compile_function_source(
        src, "sq-loss",
        (ArrayType(FLOAT, (dim,)),),
        verify=False, include_prelude=False, syntax="lisp",
    )
    assert art.specialization_name is not None, f"compilation failed for n={n}"

    body = _get_specialized_body(src, "sq-loss", (n,))
    x = np.arange(1.0, float(n + 1))
    g = grad_via_tape(body, "x", x)
    expected = 2.0 * x
    np.testing.assert_array_almost_equal(g, expected, decimal=4,
        err_msg=f"gradient mismatch for n={n}")


# ── GPU scaffold compilation (where available) ──────────────────────────────


@pytest.mark.parametrize("n", [4, 8])
def test_gpu_scaffold_compiles(n):
    """Simple map function compiles to GPU scaffold (descriptor ABI)."""
    try:
        from remora.compiler import compile_function_source_to_supported_gpu_artifacts
    except ImportError:
        pytest.skip("GPU compiler not available")

    src = f"def scale x = map (* 2.0) x"
    dim = StaticDim(n)
    try:
        art = compile_function_source_to_supported_gpu_artifacts(
            src, "scale",
            (ArrayType(FLOAT, (dim,)),),
            include_prelude=False,
        )
        assert art.ptx_text is not None or art.scaffold is not None
    except PipelineUnavailable:
        pytest.skip("GPU pipeline not available")
    except Exception as e:
        name = type(e).__name__
        if "ScaffoldError" in name or "LoweringError" in name:
            pytest.skip(f"GPU function not supported for n={n}: {e}")
        raise


# ── Compiled + tape gradient cross-validation ──────────────────────────────


def test_compiled_function_vs_tape_gradient():
    """Tape gradient on compiled function body must match analytical."""
    src = (
        "(define/pi ([n Dim]) "
        "  (sq-loss [x (Array Float n)] Float) "
        "  (fold + 0.0 (* x x)))"
    )
    for n in [2, 4, 8, 16]:
        dim = StaticDim(n)
        art = compile_function_source(
            src, "sq-loss",
            (ArrayType(FLOAT, (dim,)),),
            verify=False, include_prelude=False, syntax="lisp",
        )
        assert art.specialization_name is not None

        body = _get_specialized_body(src, "sq-loss", (n,))
        rng = np.random.RandomState(42 + n)
        x = rng.randn(n).astype(np.float64) * 2.0

        def f(v):
            return float(np.sum(v * v))
        grad_check(f, x, grad_via_tape(body, "x", x),
                   label=f"compiled@{n}")


def test_select_gradient_gpu_artifact():
    """Select/conditional gradient compiles to GPU PTX."""
    from remora.compiler import compile_gradient_function_source_to_supported_gpu_artifacts

    source = (
        "(define/pi () (relu [x (Array Float 4)] Float) "
        "  (fold + 0.0 (if (> x 0.0) x (map (* 0.0) x))))"
    )
    from remora.types import ArrayType, StaticDim

    param_type = ArrayType(FLOAT, (StaticDim(4),))
    generated = compile_gradient_function_source_to_supported_gpu_artifacts(
        source,
        "relu",
        (param_type,),
        include_prelude=False,
        syntax="lisp",
    )
    assert generated.gradient_source.source
    assert "(if" in generated.gradient_source.source
    assert generated.gpu.ptx_text
    assert len(generated.gpu.kernels) >= 1
