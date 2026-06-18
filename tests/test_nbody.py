"""Tests for N-body force computation (interpreter and CPU compiled)."""

import numpy as np
import pytest

from remora.runtime import evaluate_source


def _nbody_source(N: int, eps: float = 0.01) -> str:
    """Interpreter-friendly source (no let bindings)."""
    D = "(- (index pos j) (index pos i))"
    sd = f"(exp (* 1.5 (log (+ (fold + 0.0 (* {D} {D})) {eps}))))"
    force = f"(map (lambda (v) (/ v {sd})) {D})"
    return (
        f"(define/pi () (forces [pos (Array Float {N} 3)] (Array Float {N} 3))"
        f" (map (lambda (i)"
        f" (fold + [0.0 0.0 0.0]"
        f" (map (lambda (j) {force}) (iota {N}))))"
        f" (iota {N})))"
    )


def _nbody_source_compiled(N: int, eps: float = 0.01) -> str:
    """Compilable source using :: let bindings to hoist complex sub-expressions
    out of map callable bodies.  The :: syntax creates HIRLet nodes that the
    new scf.for-based lowering path processes step-by-step."""
    return (
        f"(define/pi () (forces [pos (Array Float {N} 3)] (Array Float {N} 3))"
        f" (map (lambda (i)"
        f" (fold + [0.0 0.0 0.0]"
        f" (map (lambda (j)"
        f" (:: D (- (index pos j) (index pos i))"
        f" (:: dsq (fold + 0.0 (* D D))"
        f" (:: sd (exp (* 1.5 (log (+ dsq {eps}))))"
        f" (map (lambda (v) (/ v sd)) D)))))"
        f" (iota {N}))))"
        f" (iota {N})))"
    )


def _ref_forces(pos: np.ndarray, eps: float = 0.01) -> np.ndarray:
    N = len(pos)
    ref = np.zeros((N, 3), dtype=np.float32)
    for i in range(N):
        for j in range(N):
            if i == j:
                continue
            d = pos[j] - pos[i]
            dsq = float((d * d).sum()) + eps
            ref[i] += d / (dsq ** 1.5)
    return ref


def test_nbody_forces_match_reference():
    N = 4
    rng = np.random.default_rng(0)
    pos = rng.standard_normal((N, 3)).astype(np.float32)
    src = _nbody_source(N)
    pt = "[" + " ".join("[" + " ".join(f"{v:.6f}" for v in r) + "]" for r in pos) + "]"
    result = evaluate_source(
        f"{src} (forces {pt})", include_prelude=False, syntax="lisp"
    )
    out = np.asarray(result.value, dtype=np.float32)
    ref = _ref_forces(pos)
    np.testing.assert_allclose(out, ref, rtol=1e-4, atol=1e-5)


def test_nbody_self_force_is_zero():
    """A single particle must have zero net force (no self-interaction)."""
    N = 1
    src = _nbody_source(N)
    rng = np.random.default_rng(1)
    pos = rng.standard_normal((N, 3)).astype(np.float32)
    pt = "[" + " ".join("[" + " ".join(f"{v:.6f}" for v in r) + "]" for r in pos) + "]"
    result = evaluate_source(
        f"{src} (forces {pt})", include_prelude=False, syntax="lisp"
    )
    out = np.asarray(result.value, dtype=np.float32)
    np.testing.assert_allclose(out, np.zeros((1, 3), dtype=np.float32), atol=1e-6)


# ---------------------------------------------------------------------------
# CPU compiled tests
# ---------------------------------------------------------------------------


def test_nbody_compiles():
    """Verify the compiled N-body source generates valid MLIR."""
    from remora.compiler import compile_function_source
    from remora.types import ArrayType, FLOAT, StaticDim

    N = 4
    src = _nbody_source_compiled(N)
    pt = (ArrayType(FLOAT, (StaticDim(N), StaticDim(3))),)
    artifact = compile_function_source(src, "forces", pt, syntax="lisp")
    assert artifact.mlir_module is not None, "N-body compilation should succeed"
    assert len(artifact.mlir_text) > 0


def test_simple_map_fold_compiles():
    """Verify the simpler map+fold pattern also compiles (no :: let needed)."""
    from remora.compiler import compile_function_source
    from remora.types import ArrayType, FLOAT, StaticDim

    N = 4
    src = '''(define/pi () (f [pos (Array Float 4 3)] (Array Float 4 3))
  (map (lambda (i)
    (fold + [0.0 0.0 0.0]
      (map (lambda (j)
        (- (index pos j) (index pos i)))
      (iota 4))))
  (iota 4)))'''
    pt = (ArrayType(FLOAT, (StaticDim(N), StaticDim(3))),)
    artifact = compile_function_source(src, "f", pt, syntax="lisp")
    assert artifact.mlir_module is not None, "Simple map-fold compilation should succeed"
