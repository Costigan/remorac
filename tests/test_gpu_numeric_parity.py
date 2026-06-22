"""GPU vs CPU numeric-parity tests.

The CPU-compiled path is the validated oracle; each program is compiled to
both GPU (descriptor-ABI PTX) and CPU and their outputs compared.  This
guards the class of *silent* GPU miscompiles that compile-only tests miss
(e.g. the vector-valued cell-fold that collapsed each force to a broadcast
scalar — fixed in the let / binary-op / register-vector-reduce emitters).
"""
import numpy as np
import pytest

from remora.compiler import compile_function_source_to_mlir_gpu_ptx
from remora.executor import RemoraExecutor
from remora.runtime import CPUFunctionExecutor, CUDARuntime, RuntimeUnavailable
from remora.types import FLOAT, ArrayType, StaticDim
from conftest import gpu_required_or_skip


def _arr(*dims):
    return ArrayType(FLOAT, tuple(StaticDim(d) for d in dims))


def _nbody(N):
    return (
        f"(define/pi () (forces [pos (Array Float {N} 3)] (Array Float {N} 3))"
        f" (map (lambda (i) (fold + [0.0 0.0 0.0]"
        f" (map (lambda (j) (let* ((D (- (index pos j) (index pos i)))"
        f" (dsq (fold + 0.0 (* D D)))"
        f" (sd (exp (* 1.5 (log (+ dsq 0.01)))))) (map (lambda (v) (/ v sd)) D)))"
        f" (iota {N}))))"
        f" (iota {N})))"
    )


# (id, source, func, param_types, inputs, syntax)
_CASES = [
    ("map_unary", "def f xs = map (* 2.0) xs", "f", (_arr(1024),),
     lambda: [np.random.default_rng(0).standard_normal(1024).astype(np.float32)], "ml"),
    ("map_binary", "def f xs ys = map (+) xs ys", "f", (_arr(1024), _arr(1024)),
     lambda: [np.random.default_rng(1).standard_normal(1024).astype(np.float32),
              np.random.default_rng(2).standard_normal(1024).astype(np.float32)], "ml"),
    ("map_compound", "def f xs = map (\\x -> x * x + 1.0) xs", "f", (_arr(1024),),
     lambda: [np.random.default_rng(3).standard_normal(1024).astype(np.float32)], "ml"),
    ("vector_cell_fold",
     "(define/pi () (f [pos (Array Float 4 3)] (Array Float 4 3))"
     " (map (lambda (i) (fold + [0.0 0.0 0.0]"
     " (map (lambda (j) (let* ((D (- (index pos j) (index pos i)))) D)) (iota 4))))"
     " (iota 4)))", "f", (_arr(4, 3),),
     lambda: [np.random.default_rng(4).standard_normal((4, 3)).astype(np.float32)], "lisp"),
    ("nbody", _nbody(64), "forces", (_arr(64, 3),),
     lambda: [np.random.default_rng(5).standard_normal((64, 3)).astype(np.float32)], "lisp"),
]


@pytest.mark.parametrize("case", _CASES, ids=[c[0] for c in _CASES])
def test_gpu_cpu_numeric_parity_when_available(case):
    name, src, fn, pt, make_inputs, syntax = case
    runtime = None
    try:
        try:
            runtime = CUDARuntime()
        except RuntimeUnavailable as exc:
            gpu_required_or_skip(str(exc))
        inputs = make_inputs()
        # CPU oracle
        art = CPUFunctionExecutor.compile_source(src, fn, pt, include_prelude=False, syntax=syntax)
        try:
            cpu = np.asarray(CPUFunctionExecutor(art).execute(*inputs).value, dtype=np.float32)
        finally:
            art.close()
        # GPU
        ptx, kernels, _ = compile_function_source_to_mlir_gpu_ptx(
            src, fn, pt, include_prelude=False, kernel_name="k", syntax=syntax)
        executor = RemoraExecutor(ptx, kernels, runtime=runtime)
        try:
            gpu = np.asarray(executor.execute("k", list(inputs)), dtype=np.float32)
        finally:
            executor.close()
        np.testing.assert_allclose(
            gpu.reshape(cpu.shape), cpu, rtol=1e-3, atol=1e-4,
            err_msg=f"GPU/CPU mismatch for {name}")
    except RuntimeUnavailable as exc:
        gpu_required_or_skip(str(exc))
    finally:
        if runtime is not None:
            runtime.close()
