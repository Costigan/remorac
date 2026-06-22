"""Benchmark suite: Remora vs NumPy vs JAX runtime execution performance.

Measures execution time (not compilation time) for common array operations
across backends.  Compile-once-execute-many pattern for compiled backends.

Run: uv run remora-perf --ops map,fold --backends numpy,remora-cpu --sizes 1000,10000
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from dataclasses import asdict, dataclass
from typing import Any, Callable

import numpy as np


@dataclass
class BenchResult:
    operation: str
    backend: str
    size: int
    median_s: float
    min_s: float
    std_s: float
    throughput_elem_per_s: float


def _measure(fn: Callable[[], Any], warmup: int, trials: int) -> list[float]:
    for _ in range(warmup):
        fn()
    times: list[float] = []
    for _ in range(trials):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    return times


def _result(op: str, backend: str, size: int, times: list[float]) -> BenchResult:
    med = statistics.median(times)
    return BenchResult(
        operation=op,
        backend=backend,
        size=size,
        median_s=med,
        min_s=min(times),
        std_s=statistics.stdev(times) if len(times) > 1 else 0.0,
        throughput_elem_per_s=size / med if med > 0 else float("inf"),
    )


# ---------------------------------------------------------------------------
# NumPy benchmarks
# ---------------------------------------------------------------------------

def bench_map_numpy(n: int, warmup: int, trials: int) -> BenchResult:
    xs = np.random.default_rng(0).standard_normal(n).astype(np.float32)
    return _result("map", "numpy", n, _measure(lambda: xs * 2.0, warmup, trials))


def bench_fold_numpy(n: int, warmup: int, trials: int) -> BenchResult:
    xs = np.random.default_rng(0).standard_normal(n).astype(np.float32)
    return _result("fold", "numpy", n, _measure(lambda: np.sum(xs), warmup, trials))


def bench_scan_numpy(n: int, warmup: int, trials: int) -> BenchResult:
    xs = np.random.default_rng(0).standard_normal(n).astype(np.float32)
    return _result("scan", "numpy", n, _measure(lambda: np.cumsum(xs), warmup, trials))


def bench_matmul_numpy(n: int, warmup: int, trials: int) -> BenchResult:
    rng = np.random.default_rng(0)
    a = rng.standard_normal((n, n)).astype(np.float32)
    b = rng.standard_normal((n, n)).astype(np.float32)
    return _result("matmul", "numpy", n * n, _measure(lambda: a @ b, warmup, trials))


def bench_sort_numpy(n: int, warmup: int, trials: int) -> BenchResult:
    xs = np.random.default_rng(0).standard_normal(n).astype(np.float32)
    return _result("sort", "numpy", n, _measure(lambda: np.sort(xs), warmup, trials))


def bench_stencil_numpy(n: int, warmup: int, trials: int) -> BenchResult:
    rng = np.random.default_rng(0)
    img = rng.standard_normal((n, n)).astype(np.float32)
    kernel = np.full((3, 3), 1.0 / 9.0, dtype=np.float32)
    from numpy.lib.stride_tricks import sliding_window_view
    def run():
        patches = sliding_window_view(img, (3, 3))
        return (patches * kernel).sum(axis=(-2, -1))
    out_elems = (n - 2) * (n - 2)
    return _result("stencil", "numpy", out_elems, _measure(run, warmup, trials))


# ---------------------------------------------------------------------------
# JAX benchmarks
# ---------------------------------------------------------------------------

def _try_import_jax():
    try:
        import jax
        import jax.numpy as jnp
        return jax, jnp
    except ImportError:
        return None, None


def bench_map_jax(n: int, warmup: int, trials: int) -> BenchResult | None:
    jax, jnp = _try_import_jax()
    if jax is None:
        return None
    xs = jnp.array(np.random.default_rng(0).standard_normal(n).astype(np.float32))
    f = jax.jit(lambda x: x * 2.0)
    def run():
        f(xs).block_until_ready()
    backend = f"jax-{jax.devices()[0].platform}"
    return _result("map", backend, n, _measure(run, warmup, trials))


def bench_fold_jax(n: int, warmup: int, trials: int) -> BenchResult | None:
    jax, jnp = _try_import_jax()
    if jax is None:
        return None
    xs = jnp.array(np.random.default_rng(0).standard_normal(n).astype(np.float32))
    f = jax.jit(jnp.sum)
    def run():
        f(xs).block_until_ready()
    backend = f"jax-{jax.devices()[0].platform}"
    return _result("fold", backend, n, _measure(run, warmup, trials))


def bench_scan_jax(n: int, warmup: int, trials: int) -> BenchResult | None:
    jax, jnp = _try_import_jax()
    if jax is None:
        return None
    xs = jnp.array(np.random.default_rng(0).standard_normal(n).astype(np.float32))
    f = jax.jit(jnp.cumsum)
    def run():
        f(xs).block_until_ready()
    backend = f"jax-{jax.devices()[0].platform}"
    return _result("scan", backend, n, _measure(run, warmup, trials))


def bench_matmul_jax(n: int, warmup: int, trials: int) -> BenchResult | None:
    jax, jnp = _try_import_jax()
    if jax is None:
        return None
    rng = np.random.default_rng(0)
    a = jnp.array(rng.standard_normal((n, n)).astype(np.float32))
    b = jnp.array(rng.standard_normal((n, n)).astype(np.float32))
    f = jax.jit(jnp.matmul)
    def run():
        f(a, b).block_until_ready()
    backend = f"jax-{jax.devices()[0].platform}"
    return _result("matmul", backend, n * n, _measure(run, warmup, trials))


def bench_sort_jax(n: int, warmup: int, trials: int) -> BenchResult | None:
    jax, jnp = _try_import_jax()
    if jax is None:
        return None
    xs = jnp.array(np.random.default_rng(0).standard_normal(n).astype(np.float32))
    f = jax.jit(jnp.sort)
    def run():
        f(xs).block_until_ready()
    backend = f"jax-{jax.devices()[0].platform}"
    return _result("sort", backend, n, _measure(run, warmup, trials))


def bench_stencil_jax(n: int, warmup: int, trials: int) -> BenchResult | None:
    jax, jnp = _try_import_jax()
    if jax is None:
        return None
    img = jnp.array(np.random.default_rng(0).standard_normal((n, n)).astype(np.float32))
    kernel = jnp.full((3, 3), 1.0 / 9.0, dtype=jnp.float32)
    kernel_4d = kernel.reshape(1, 1, 3, 3)
    img_4d = img.reshape(1, 1, n, n)
    @jax.jit
    def run(x):
        return jax.lax.conv(x, kernel_4d, (1, 1), "VALID")
    def go():
        run(img_4d).block_until_ready()
    out_elems = (n - 2) * (n - 2)
    backend = f"jax-{jax.devices()[0].platform}"
    return _result("stencil", backend, out_elems, _measure(go, warmup, trials))


# ---------------------------------------------------------------------------
# Remora CPU benchmarks
# ---------------------------------------------------------------------------

def _bench_remora_cpu_func(
    op: str, source: str, func_name: str, param_types: tuple, n: int,
    inputs_fn: Callable[[int], list[np.ndarray]],
    warmup: int, trials: int,
    *, syntax: str = "ml", include_prelude: bool = True,
    output_elems: int | None = None,
) -> BenchResult | None:
    try:
        from remora.runtime import CPUFunctionExecutor
        artifact = CPUFunctionExecutor.compile_source(
            source, func_name, param_types,
            include_prelude=include_prelude, syntax=syntax,
        )
    except Exception:
        return None
    try:
        exe = CPUFunctionExecutor(artifact)
        inputs = inputs_fn(n)
        def run():
            exe.execute(*inputs)
        run()
        elems = output_elems if output_elems is not None else n
        return _result(op, "remora-cpu", elems, _measure(run, warmup, trials))
    except Exception:
        return None
    finally:
        artifact.close()


def bench_map_remora_cpu(n: int, warmup: int, trials: int) -> BenchResult | None:
    from remora.types import FLOAT, ArrayType, StaticDim
    return _bench_remora_cpu_func(
        "map",
        "def scale xs = map (* 2.0) xs", "scale",
        (ArrayType(FLOAT, (StaticDim(n),)),), n,
        lambda n: [np.random.default_rng(0).standard_normal(n).astype(np.float32)],
        warmup, trials,
    )


def bench_fold_remora_cpu(n: int, warmup: int, trials: int) -> BenchResult | None:
    from remora.types import FLOAT, ArrayType, StaticDim
    return _bench_remora_cpu_func(
        "fold",
        "def sumit xs = fold (+) 0.0 xs", "sumit",
        (ArrayType(FLOAT, (StaticDim(n),)),), n,
        lambda n: [np.random.default_rng(0).standard_normal(n).astype(np.float32)],
        warmup, trials, include_prelude=False,
    )


def bench_scan_remora_cpu(n: int, warmup: int, trials: int) -> BenchResult | None:
    from remora.types import FLOAT, ArrayType, StaticDim
    return _bench_remora_cpu_func(
        "scan",
        "def scanit xs = iscan (+) 0.0 xs", "scanit",
        (ArrayType(FLOAT, (StaticDim(n),)),), n,
        lambda n: [np.random.default_rng(0).standard_normal(n).astype(np.float32)],
        warmup, trials, include_prelude=False,
    )


def bench_sort_remora_cpu(n: int, warmup: int, trials: int) -> BenchResult | None:
    from remora.types import FLOAT, ArrayType, StaticDim
    source = (
        f"(define/pi () (sortit [xs (Array Float {n})] (Array Float {n}))"
        f" (sort < xs))"
    )
    return _bench_remora_cpu_func(
        "sort", source, "sortit",
        (ArrayType(FLOAT, (StaticDim(n),)),), n,
        lambda n: [np.random.default_rng(0).standard_normal(n).astype(np.float32)],
        warmup, trials, syntax="lisp", include_prelude=False,
    )


def bench_stencil_remora_cpu(n: int, warmup: int, trials: int) -> BenchResult | None:
    from remora.types import FLOAT, ArrayType, StaticDim
    interior = n - 2
    patches = interior * interior
    source = (
        f"(define/pi () "
        f"(blur [image (Array Float {n} {n}) k (Array Float 3 3)] "
        f"(Array Float {patches})) "
        f"(map (lambda (p) (fold + 0.0 (map * p (ravel k)))) "
        f"(im2col image [3 3] 1)))"
    )
    ptype = (
        ArrayType(FLOAT, (StaticDim(n), StaticDim(n))),
        ArrayType(FLOAT, (StaticDim(3), StaticDim(3))),
    )
    kernel = np.full((3, 3), 1.0 / 9.0, dtype=np.float32)
    return _bench_remora_cpu_func(
        "stencil", source, "blur", ptype, n,
        lambda n: [
            np.random.default_rng(0).standard_normal((n, n)).astype(np.float32),
            kernel,
        ],
        warmup, trials, syntax="lisp", include_prelude=False,
        output_elems=patches,
    )


def bench_matmul_remora_cpu(n: int, warmup: int, trials: int) -> BenchResult | None:
    from remora.types import FLOAT, ArrayType, StaticDim
    source = (
        f"(define/pi () "
        f"(mm [a (Array Float {n} {n}) b (Array Float {n} {n})] "
        f"(Array Float {n} {n})) "
        f"(matmul a b))"
    )
    ptype = (
        ArrayType(FLOAT, (StaticDim(n), StaticDim(n))),
        ArrayType(FLOAT, (StaticDim(n), StaticDim(n))),
    )
    rng = np.random.default_rng(42)
    return _bench_remora_cpu_func(
        "matmul", source, "mm", ptype, n,
        lambda n: [
            rng.standard_normal((n, n)).astype(np.float32),
            rng.standard_normal((n, n)).astype(np.float32),
        ],
        warmup, trials, syntax="lisp", include_prelude=False,
        output_elems=n * n,
    )


# ---------------------------------------------------------------------------
# Remora GPU benchmarks
# ---------------------------------------------------------------------------

def _get_gpu_runtime():
    from remora.runtime import CUDARuntime, RuntimeUnavailable
    try:
        return CUDARuntime()
    except RuntimeUnavailable:
        return None


_POOL_ENABLED = True


def _set_pool_enabled(enabled: bool) -> None:
    global _POOL_ENABLED
    _POOL_ENABLED = enabled


def _apply_pool(exe) -> None:
    if not _POOL_ENABLED:
        exe.set_pool_enabled(False)


_DEVICE_RESIDENT = False


def _set_device_resident(enabled: bool) -> None:
    global _DEVICE_RESIDENT
    _DEVICE_RESIDENT = enabled


def _device_resident_run(exe, kernel_name: str, xs: np.ndarray):
    """Upload xs once, allocate output once, return a run() that launches
    the kernel on the device-resident buffers (no host-device transfer)."""
    in_ptr = exe.alloc_and_upload(xs)
    out_ptr = exe.execute_device(kernel_name, [in_ptr], [xs])

    def run():
        exe.execute_device(kernel_name, [in_ptr], [xs], output_ptr=out_ptr)

    return run


def bench_map_remora_gpu(n: int, warmup: int, trials: int) -> BenchResult | None:
    try:
        from remora.compiler import compile_function_source_to_mlir_gpu_ptx
        from remora.executor import RemoraExecutor
        from remora.types import FLOAT, ArrayType, StaticDim
    except ImportError:
        return None
    rt = _get_gpu_runtime()
    if rt is None:
        return None
    try:
        ptx, kernels, _ = compile_function_source_to_mlir_gpu_ptx(
            "def scale xs = map (* 2.0) xs", "scale",
            (ArrayType(FLOAT, (StaticDim(n),)),),
            kernel_name="bench_map",
        )
        xs = np.random.default_rng(0).standard_normal(n).astype(np.float32)
        with RemoraExecutor(ptx, kernels, runtime=rt) as exe:
            _apply_pool(exe)
            if _DEVICE_RESIDENT:
                run = _device_resident_run(exe, "bench_map", xs)
            else:
                def run():
                    exe.execute("bench_map", [xs])
            return _result("map", "remora-gpu", n, _measure(run, warmup, trials))
    except Exception:
        return None
    finally:
        rt.close()


def bench_fold_remora_gpu(n: int, warmup: int, trials: int) -> BenchResult | None:
    try:
        from remora.compiler import compile_function_source_to_mlir_gpu_ptx
        from remora.executor import RemoraExecutor
        from remora.types import FLOAT, ArrayType, StaticDim
    except ImportError:
        return None
    rt = _get_gpu_runtime()
    if rt is None:
        return None
    try:
        ptx, kernels, _ = compile_function_source_to_mlir_gpu_ptx(
            "def sumit xs = fold (+) 0.0 xs", "sumit",
            (ArrayType(FLOAT, (StaticDim(n),)),),
            include_prelude=False,
            kernel_name="bench_fold",
        )
        xs = np.random.default_rng(0).standard_normal(n).astype(np.float32)
        with RemoraExecutor(ptx, kernels, runtime=rt) as exe:
            _apply_pool(exe)
            if _DEVICE_RESIDENT:
                run = _device_resident_run(exe, "bench_fold", xs)
            else:
                def run():
                    exe.execute("bench_fold", [xs])
            return _result("fold", "remora-gpu", n, _measure(run, warmup, trials))
    except Exception:
        return None
    finally:
        rt.close()


def bench_sort_remora_gpu(n: int, warmup: int, trials: int) -> BenchResult | None:
    try:
        from remora.codegen import generate_mlir_descriptor_abi_ptx
        from remora.executor import RemoraExecutor
        from remora.hir import HIRFunction, HIRParam, HIRSort, HIRVar
        from remora.types import FLOAT, ArrayType, StaticDim
    except ImportError:
        return None
    rt = _get_gpu_runtime()
    if rt is None:
        return None
    try:
        arr_type = ArrayType(FLOAT, (StaticDim(n),))
        hf = HIRFunction(
            "s", [HIRParam("xs", arr_type)],
            HIRSort(HIRVar("xs", arr_type), result_type=arr_type),
            return_type=arr_type,
        )
        ptx, kernels, plan = generate_mlir_descriptor_abi_ptx(hf, kernel_name="bench_sort")
        xs = np.random.default_rng(42).standard_normal(n).astype(np.float32)
        with RemoraExecutor(ptx, kernels, runtime=rt) as exe:
            _apply_pool(exe)
            if plan is not None:
                def run():
                    exe.execute_plan(plan, [xs])
            else:
                def run():
                    exe.execute("bench_sort", [xs])
            return _result("sort", "remora-gpu", n, _measure(run, warmup, trials))
    except Exception:
        return None
    finally:
        rt.close()


def bench_matmul_remora_gpu(n: int, warmup: int, trials: int) -> BenchResult | None:
    try:
        from remora.codegen import generate_mlir_descriptor_abi_ptx
        from remora.executor import RemoraExecutor
        from remora.hir import HIRFunction, HIRMatmul, HIRParam, HIRVar
        from remora.types import FLOAT, ArrayType, StaticDim
    except ImportError:
        return None
    rt = _get_gpu_runtime()
    if rt is None:
        return None
    try:
        tNN = ArrayType(FLOAT, (StaticDim(n), StaticDim(n)))
        hf = HIRFunction(
            "mm", [HIRParam("a", tNN), HIRParam("b", tNN)],
            HIRMatmul(HIRVar("a", tNN), HIRVar("b", tNN), result_type=tNN),
            return_type=tNN,
        )
        ptx, kernels, plan = generate_mlir_descriptor_abi_ptx(hf, kernel_name="bench_mm")
        rng = np.random.default_rng(42)
        a = rng.standard_normal((n, n)).astype(np.float32)
        b = rng.standard_normal((n, n)).astype(np.float32)
        with RemoraExecutor(ptx, kernels, runtime=rt) as exe:
            _apply_pool(exe)
            if plan is not None:
                def run():
                    exe.execute_plan(plan, [a, b])
            else:
                def run():
                    exe.execute("bench_mm", [a, b])
            return _result("matmul", "remora-gpu", n * n, _measure(run, warmup, trials))
    except Exception:
        return None
    finally:
        rt.close()


def bench_scan_remora_gpu(n: int, warmup: int, trials: int) -> BenchResult | None:
    try:
        from remora.compiler import compile_function_source
        from remora.codegen import generate_mlir_descriptor_abi_ptx
        from remora.executor import RemoraExecutor
        from remora.types import FLOAT, ArrayType, StaticDim
    except ImportError:
        return None
    rt = _get_gpu_runtime()
    if rt is None:
        return None
    try:
        artifact = compile_function_source(
            "def scanit xs = iscan (+) 0.0 xs", "scanit",
            (ArrayType(FLOAT, (StaticDim(n),)),),
            verify=False, include_prelude=False,
        )
        ptx, kernels, plan = generate_mlir_descriptor_abi_ptx(
            artifact.hir_function, kernel_name="bench_scan",
        )
        xs = np.random.default_rng(0).standard_normal(n).astype(np.float32)
        with RemoraExecutor(ptx, kernels, runtime=rt) as exe:
            _apply_pool(exe)
            if plan is not None:
                def run():
                    exe.execute_plan(plan, [xs])
            else:
                def run():
                    exe.execute("bench_scan", [xs])
            return _result("scan", "remora-gpu", n, _measure(run, warmup, trials))
    except Exception:
        return None
    finally:
        rt.close()


def bench_stencil_remora_gpu(n: int, warmup: int, trials: int) -> BenchResult | None:
    try:
        from remora.compiler import compile_function_source_to_mlir_gpu_ptx
        from remora.executor import RemoraExecutor
        from remora.types import FLOAT, ArrayType, StaticDim
    except ImportError:
        return None
    rt = _get_gpu_runtime()
    if rt is None:
        return None
    interior = n - 2
    patches = interior * interior
    source = (
        f"(define/pi () "
        f"(blur [image (Array Float {n} {n}) k (Array Float 3 3)] "
        f"(Array Float {patches})) "
        f"(map (lambda (p) (fold + 0.0 (map * p (ravel k)))) "
        f"(im2col image [3 3] 1)))"
    )
    try:
        ptx, kernels, _ = compile_function_source_to_mlir_gpu_ptx(
            source, "blur",
            (
                ArrayType(FLOAT, (StaticDim(n), StaticDim(n))),
                ArrayType(FLOAT, (StaticDim(3), StaticDim(3))),
            ),
            include_prelude=False,
            kernel_name="bench_stencil",
            syntax="lisp",
        )
        img = np.random.default_rng(0).standard_normal((n, n)).astype(np.float32)
        kernel = np.full((3, 3), 1.0 / 9.0, dtype=np.float32)
        with RemoraExecutor(ptx, kernels, runtime=rt) as exe:
            _apply_pool(exe)
            def run():
                exe.execute("bench_stencil", [img, kernel])
            return _result("stencil", "remora-gpu", patches, _measure(run, warmup, trials))
    except Exception:
        return None
    finally:
        rt.close()


# ---------------------------------------------------------------------------
# Application benchmark: gradient descent (polynomial curve fitting)
# ---------------------------------------------------------------------------
#
# Fits f(x) = c0 + c1*x + c2*x^2 to the points (0,1),(1,2),(2,5),(3,10)
# by `n` steps of gradient descent (lr=0.001) from c=[0,0,0].  At n=200
# all backends converge to [0.512337, 0.433115, 0.911621].

_GRAD_DESCENT_X = (0.0, 1.0, 2.0, 3.0)
_GRAD_DESCENT_Y = (1.0, 2.0, 5.0, 10.0)
_GRAD_DESCENT_LR = 0.001


def _grad_descent_source(steps: int) -> str:
    return f"""(define/pi ()
  (poly-eval [coeffs (Array Float 3) x Float] Float)
  (let* ((c0 (index-item coeffs 0))
         (c1 (index-item coeffs 1))
         (c2 (index-item coeffs 2)))
    (+ c0 (+ (* c1 x) (* c2 (* x x))))))
(define/pi ()
  (poly-loss [coeffs (Array Float 3)] Float)
  (let* ((r0 (- (poly-eval coeffs 0.0) 1.0))
         (r1 (- (poly-eval coeffs 1.0) 2.0))
         (r2 (- (poly-eval coeffs 2.0) 5.0))
         (r3 (- (poly-eval coeffs 3.0) 10.0)))
    (+ (* r0 r0) (+ (* r1 r1) (+ (* r2 r2) (* r3 r3))))))
(fold (lambda (params step)
        (- params (* {_GRAD_DESCENT_LR} ((grad poly-loss) params))))
      [0.0 0.0 0.0]
      (iota {steps}))"""


def grad_descent_numpy(steps: int) -> np.ndarray:
    x = np.array(_GRAD_DESCENT_X, dtype=np.float64)
    y = np.array(_GRAD_DESCENT_Y, dtype=np.float64)
    c = np.zeros(3, dtype=np.float64)
    for _ in range(steps):
        pred = c[0] + c[1] * x + c[2] * x * x
        r = pred - y
        grad = np.array([
            2.0 * np.sum(r),
            2.0 * np.sum(r * x),
            2.0 * np.sum(r * x * x),
        ])
        c = c - _GRAD_DESCENT_LR * grad
    return c


def bench_grad_descent_numpy(n: int, warmup: int, trials: int) -> BenchResult:
    return _result(
        "grad_descent", "numpy", n,
        _measure(lambda: grad_descent_numpy(n), warmup, trials),
    )


def bench_grad_descent_jax(n: int, warmup: int, trials: int) -> BenchResult | None:
    jax, jnp = _try_import_jax()
    if jax is None:
        return None
    x = jnp.array(_GRAD_DESCENT_X)
    y = jnp.array(_GRAD_DESCENT_Y)

    def loss(c):
        pred = c[0] + c[1] * x + c[2] * x * x
        return jnp.sum((pred - y) ** 2)

    grad = jax.grad(loss)

    @jax.jit
    def optimize(c0):
        return jax.lax.fori_loop(
            0, n, lambda i, c: c - _GRAD_DESCENT_LR * grad(c), c0
        )

    def run():
        optimize(jnp.zeros(3)).block_until_ready()

    backend = f"jax-{jax.devices()[0].platform}"
    return _result("grad_descent", backend, n, _measure(run, warmup, trials))


def bench_grad_descent_remora_cpu(n: int, warmup: int, trials: int) -> BenchResult | None:
    try:
        from remora.runtime import CPUExecutor
    except ImportError:
        return None
    try:
        artifact = CPUExecutor.compile_source(
            _grad_descent_source(n), include_prelude=True, syntax="lisp",
        )
    except Exception:
        return None
    try:
        exe = CPUExecutor(artifact)
        exe.execute_main([])
        return _result(
            "grad_descent", "remora-cpu", n,
            _measure(lambda: exe.execute_main([]), warmup, trials),
        )
    except Exception:
        return None
    finally:
        artifact.close()


def bench_grad_descent_remora_gpu(n: int, warmup: int, trials: int) -> BenchResult | None:
    try:
        from remora.compiler import compile_source
        from remora.codegen import try_compile_state_fold_gpu
        from remora.executor import RemoraExecutor
    except ImportError:
        return None
    rt = _get_gpu_runtime()
    if rt is None:
        return None
    try:
        art = compile_source(
            _grad_descent_source(n), include_prelude=True, syntax="lisp", verify=False,
        )
        fold_result = try_compile_state_fold_gpu(art.hir)
        if fold_result is None:
            return None
        ptx, kernels, plan = fold_result
        with RemoraExecutor(ptx, kernels, runtime=rt) as exe:
            _apply_pool(exe)
            exe.execute_plan(plan, [])
            return _result(
                "grad_descent", "remora-gpu", n,
                _measure(lambda: exe.execute_plan(plan, []), warmup, trials),
            )
    except Exception:
        return None
    finally:
        rt.close()


# ---------------------------------------------------------------------------
# Application benchmark: convolution pipeline (conv -> relu -> pool)
# ---------------------------------------------------------------------------
#
# A 3x3 box-filter convolution (valid, stride 1) followed by ReLU and a
# 4-window sum pool over the flattened activation.  All backends compute
# the identical math so outputs match within float32 tolerance.  (Average
# pooling would require a per-element division inside the map, which hits a
# CPU-lowering gap; sum pooling keeps the pipeline faithful and matchable.)

CONV_PIPELINE_SIZES = (32, 64, 128)


def _conv_pipeline_source(n: int) -> str:
    p4 = ((n - 2) * (n - 2)) // 4
    conv = "(map (lambda (p) (fold + 0.0 (map * p (ravel k)))) (im2col image [3 3] 1))"
    relu = f"(map (lambda (v) (if (> v 0.0) v 0.0)) {conv})"
    return (
        f"(define/pi () (conv-pipe [image (Array Float {n} {n}) k (Array Float 3 3)] "
        f"(Array Float {p4})) "
        f"(map (lambda (row) (fold + 0.0 row)) (reshape {relu} [{p4} 4])))"
    )


def _conv_pipeline_numpy(img: np.ndarray, k: np.ndarray) -> np.ndarray:
    from numpy.lib.stride_tricks import sliding_window_view
    p4 = ((img.shape[0] - 2) * (img.shape[1] - 2)) // 4
    win = sliding_window_view(img, (3, 3))
    conv = np.tensordot(win, k, axes=([2, 3], [0, 1])).reshape(-1).astype(np.float32)
    relu = np.maximum(conv, 0.0)
    return relu.reshape(p4, 4).sum(axis=1).astype(np.float32)


def bench_conv_pipeline_numpy(n: int, warmup: int, trials: int) -> BenchResult:
    img = np.random.default_rng(0).standard_normal((n, n)).astype(np.float32)
    k = np.full((3, 3), 1.0 / 9.0, dtype=np.float32)
    return _result(
        "conv_pipeline", "numpy", n * n,
        _measure(lambda: _conv_pipeline_numpy(img, k), warmup, trials),
    )


def bench_conv_pipeline_jax(n: int, warmup: int, trials: int) -> BenchResult | None:
    jax, jnp = _try_import_jax()
    if jax is None:
        return None
    p4 = ((n - 2) * (n - 2)) // 4

    def crp(img, k):
        h, w = img.shape
        out = jnp.zeros((h - 2, w - 2))
        for a in range(3):
            for b in range(3):
                out = out + img[a:a + h - 2, b:b + w - 2] * k[a, b]
        relu = jnp.maximum(out.reshape(-1), 0.0)
        return relu.reshape(p4, 4).sum(axis=1)

    f = jax.jit(crp)
    img = jnp.array(np.random.default_rng(0).standard_normal((n, n)).astype(np.float32))
    k = jnp.array(np.full((3, 3), 1.0 / 9.0, dtype=np.float32))

    def run():
        f(img, k).block_until_ready()

    backend = f"jax-{jax.devices()[0].platform}"
    return _result("conv_pipeline", backend, n * n, _measure(run, warmup, trials))


def bench_conv_pipeline_remora_cpu(n: int, warmup: int, trials: int) -> BenchResult | None:
    from remora.types import FLOAT, ArrayType, StaticDim
    ptype = (
        ArrayType(FLOAT, (StaticDim(n), StaticDim(n))),
        ArrayType(FLOAT, (StaticDim(3), StaticDim(3))),
    )
    img = np.random.default_rng(0).standard_normal((n, n)).astype(np.float32)
    k = np.full((3, 3), 1.0 / 9.0, dtype=np.float32)
    return _bench_remora_cpu_func(
        "conv_pipeline", _conv_pipeline_source(n), "conv-pipe", ptype, n * n,
        lambda _n: [img, k],
        warmup, trials, syntax="lisp", include_prelude=False,
        output_elems=n * n,
    )


# ---------------------------------------------------------------------------
# Application benchmark: N-body step (all-pairs gravitational forces)
# ---------------------------------------------------------------------------
#
# One gravitational timestep: force[i] = sum_j d/(|d|^2 + eps)^1.5 where
# d = pos[j] - pos[i].  Self-interaction (i==j) contributes zero (d=0).
# Backends: numpy, jax, remora-cpu (all verified vs the O(N^2) reference).
# remora-gpu is intentionally omitted: the general-map GPU lowering
# miscompiles the vector-valued (3-component) cell fold, collapsing each
# force into a single broadcast scalar (a known GPU-lowering bug).

NBODY_SIZES = (64, 256, 1024)
_NBODY_EPS = 0.01


def _nbody_source(n: int, eps: float = _NBODY_EPS) -> str:
    return (
        f"(define/pi () (forces [pos (Array Float {n} 3)] (Array Float {n} 3))"
        f" (map (lambda (i) (fold + [0.0 0.0 0.0]"
        f" (map (lambda (j) (let* ((D (- (index pos j) (index pos i)))"
        f" (dsq (fold + 0.0 (* D D))) (sd (exp (* 1.5 (log (+ dsq {eps}))))))"
        f" (map (lambda (v) (/ v sd)) D))) (iota {n}))))"
        f" (iota {n})))"
    )


def _nbody_numpy(pos: np.ndarray, eps: float = _NBODY_EPS) -> np.ndarray:
    d = pos[None, :, :] - pos[:, None, :]
    dsq = (d * d).sum(-1) + eps
    inv = dsq ** -1.5
    return (d * inv[..., None]).sum(axis=1).astype(np.float32)


def bench_nbody_numpy(n: int, warmup: int, trials: int) -> BenchResult:
    pos = np.random.default_rng(0).standard_normal((n, 3)).astype(np.float32)
    return _result(
        "nbody", "numpy", n * n,
        _measure(lambda: _nbody_numpy(pos), warmup, trials),
    )


def bench_nbody_jax(n: int, warmup: int, trials: int) -> BenchResult | None:
    jax, jnp = _try_import_jax()
    if jax is None:
        return None
    eps = _NBODY_EPS

    def nbody(pos):
        d = pos[None, :, :] - pos[:, None, :]
        dsq = (d * d).sum(-1) + eps
        inv = dsq ** -1.5
        return (d * inv[..., None]).sum(axis=1)

    f = jax.jit(nbody)
    pos = jnp.array(np.random.default_rng(0).standard_normal((n, 3)).astype(np.float32))

    def run():
        f(pos).block_until_ready()

    backend = f"jax-{jax.devices()[0].platform}"
    return _result("nbody", backend, n * n, _measure(run, warmup, trials))


def bench_nbody_remora_cpu(n: int, warmup: int, trials: int) -> BenchResult | None:
    from remora.types import FLOAT, ArrayType, StaticDim
    ptype = (ArrayType(FLOAT, (StaticDim(n), StaticDim(3))),)
    pos = np.random.default_rng(0).standard_normal((n, 3)).astype(np.float32)
    return _bench_remora_cpu_func(
        "nbody", _nbody_source(n), "forces", ptype, n * n,
        lambda _n: [pos],
        warmup, trials, syntax="lisp", include_prelude=False,
        output_elems=n * n,
    )


# ---------------------------------------------------------------------------
# Fusion benchmarks (composed op-chains vs manually fused single passes)
# ---------------------------------------------------------------------------
#
# Each case is run as a composed chain and as a hand-fused single map; the
# composed/manual median ratio is the fusion-efficiency metric (1.0 means
# the compiler fuses the chain into one pass).  All variants are remora-cpu.

FUSION_SIZES = (100_000, 1_000_000)

_FUSION_SOURCES = {
    "mapchain-composed": "def f xs = map (* 2.0) (map (+ 1.0) xs)",
    "mapchain-manual": "def f xs = map (\\x -> (x + 1.0) * 2.0) xs",
    "triple-composed": (
        "def f xs = map (\\x -> if x < 0.0 then 0.0 - x else x) "
        "(map (\\x -> 0.0 - x) (map (* 2.0) xs))"
    ),
    "triple-manual": (
        "def f xs = map (\\x -> if (0.0 - (x * 2.0)) < 0.0 "
        "then 0.0 - (0.0 - (x * 2.0)) else (0.0 - (x * 2.0))) xs"
    ),
}


def _bench_fusion_unary(backend_label: str, source: str, n: int,
                        warmup: int, trials: int) -> BenchResult | None:
    from remora.runtime import CPUFunctionExecutor
    from remora.types import FLOAT, ArrayType, StaticDim
    try:
        art = CPUFunctionExecutor.compile_source(
            source, "f", (ArrayType(FLOAT, (StaticDim(n),)),), include_prelude=True,
        )
    except Exception:
        return None
    try:
        exe = CPUFunctionExecutor(art)
        xs = np.random.default_rng(0).standard_normal(n).astype(np.float32)
        exe.execute(xs)
        return _result("fusion", backend_label, n,
                       _measure(lambda: exe.execute(xs), warmup, trials))
    except Exception:
        return None
    finally:
        art.close()


def _make_fusion_bench(backend_label: str, source: str):
    def bench(n: int, warmup: int, trials: int) -> BenchResult | None:
        return _bench_fusion_unary(backend_label, source, n, warmup, trials)
    return bench


def bench_fusion_dot(n: int, warmup: int, trials: int) -> BenchResult | None:
    from remora.runtime import CPUFunctionExecutor
    from remora.types import FLOAT, ArrayType, StaticDim
    source = "def f xs ys = fold (+) 0.0 (map (*) xs ys)"
    try:
        art = CPUFunctionExecutor.compile_source(
            source, "f",
            (ArrayType(FLOAT, (StaticDim(n),)), ArrayType(FLOAT, (StaticDim(n),))),
            include_prelude=True,
        )
    except Exception:
        return None
    try:
        exe = CPUFunctionExecutor(art)
        xs = np.random.default_rng(0).standard_normal(n).astype(np.float32)
        ys = np.random.default_rng(1).standard_normal(n).astype(np.float32)
        exe.execute(xs, ys)
        return _result("fusion", "dot-composed", n,
                       _measure(lambda: exe.execute(xs, ys), warmup, trials))
    except Exception:
        return None
    finally:
        art.close()


# ---------------------------------------------------------------------------
# Pipeline benchmark: cumsum(sort(xs)) — a representative multi-op chain
# ---------------------------------------------------------------------------
#
# sort -> prefix-sum is a real idiom (CDF/quantiles).  Measures end-to-end
# pipeline parity, and isolates the device-residency win: `remora-gpu`
# chains the two GPU plans on-device (execute_plan_to_device); `remora-gpu-htod`
# runs the naive host-round-trip between the two ops.

PIPELINE_SIZES = (100_000, 1_000_000)


def _build_pipeline_exe(rt, N):
    from remora.compiler import compile_function_source
    from remora.codegen import generate_mlir_descriptor_abi_ptx
    from remora.executor import RemoraExecutor
    from remora.hir import HIRFunction, HIRParam, HIRSort, HIRVar
    from remora.types import FLOAT, ArrayType, StaticDim
    at = ArrayType(FLOAT, (StaticDim(N),))
    hf = HIRFunction("s", [HIRParam("xs", at)], HIRSort(HIRVar("xs", at), result_type=at), return_type=at)
    sptx, skern, splan = generate_mlir_descriptor_abi_ptx(hf, kernel_name="psort")
    art = compile_function_source("def sc xs = iscan (+) 0.0 xs", "sc", (at,), verify=False, include_prelude=False)
    cptx, ckern, cplan = generate_mlir_descriptor_abi_ptx(art.hir_function, kernel_name="pscan")
    exe = RemoraExecutor(sptx, skern, runtime=rt)
    exe.add_module(cptx, ckern)
    return exe, splan, cplan


def bench_pipeline_numpy(n: int, warmup: int, trials: int) -> BenchResult:
    xs = np.random.default_rng(0).standard_normal(n).astype(np.float32)
    return _result("pipeline", "numpy", n, _measure(lambda: np.cumsum(np.sort(xs)), warmup, trials))


def bench_pipeline_jax(n: int, warmup: int, trials: int) -> BenchResult | None:
    jax, jnp = _try_import_jax()
    if jax is None:
        return None
    f = jax.jit(lambda x: jnp.cumsum(jnp.sort(x)))
    xj = jnp.array(np.random.default_rng(0).standard_normal(n).astype(np.float32))

    def run():
        f(xj).block_until_ready()

    backend = f"jax-{jax.devices()[0].platform}"
    return _result("pipeline", backend, n, _measure(run, warmup, trials))


def bench_pipeline_remora_gpu(n: int, warmup: int, trials: int) -> BenchResult | None:
    from remora.executor import DeviceArray
    rt = _get_gpu_runtime()
    if rt is None:
        return None
    try:
        exe, splan, cplan = _build_pipeline_exe(rt, n)
        _apply_pool(exe)
        xs = np.random.default_rng(0).standard_normal(n).astype(np.float32)

        def run():
            xd = DeviceArray.from_numpy(exe, xs)
            sd = exe.execute_plan_to_device(splan, [xd]); xd.free()
            rd = exe.execute_plan_to_device(cplan, [sd]); sd.free()
            rd.to_numpy(); rd.free()

        return _result("pipeline", "remora-gpu", n, _measure(run, warmup, trials))
    except Exception:
        return None
    finally:
        rt.close()


def bench_pipeline_remora_gpu_htod(n: int, warmup: int, trials: int) -> BenchResult | None:
    rt = _get_gpu_runtime()
    if rt is None:
        return None
    try:
        exe, splan, cplan = _build_pipeline_exe(rt, n)
        _apply_pool(exe)
        xs = np.random.default_rng(0).standard_normal(n).astype(np.float32)

        def run():
            s = np.asarray(exe.execute_plan(splan, [xs]))
            np.asarray(exe.execute_plan(cplan, [s]))

        return _result("pipeline", "remora-gpu-htod", n, _measure(run, warmup, trials))
    except Exception:
        return None
    finally:
        rt.close()


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

ALL_OPS = ("map", "fold", "scan", "matmul", "sort", "stencil", "grad_descent", "conv_pipeline", "nbody", "fusion", "pipeline")

DEFAULT_SIZES = (1_000, 10_000, 100_000, 1_000_000, 10_000_000)
MATMUL_SIZES = (32, 64, 128, 256, 512, 1024)
STENCIL_SIZES = (32, 64, 128, 256, 512, 1024)
GRAD_DESCENT_SIZES = (200,)

BENCHMARKS: dict[tuple[str, str], Callable[..., BenchResult | None]] = {
    ("map", "numpy"): bench_map_numpy,
    ("map", "jax"): bench_map_jax,
    ("map", "remora-cpu"): bench_map_remora_cpu,
    ("map", "remora-gpu"): bench_map_remora_gpu,
    ("fold", "numpy"): bench_fold_numpy,
    ("fold", "jax"): bench_fold_jax,
    ("fold", "remora-cpu"): bench_fold_remora_cpu,
    ("fold", "remora-gpu"): bench_fold_remora_gpu,
    ("scan", "numpy"): bench_scan_numpy,
    ("scan", "jax"): bench_scan_jax,
    ("scan", "remora-cpu"): bench_scan_remora_cpu,
    ("scan", "remora-gpu"): bench_scan_remora_gpu,
    ("matmul", "numpy"): bench_matmul_numpy,
    ("matmul", "jax"): bench_matmul_jax,
    ("matmul", "remora-cpu"): bench_matmul_remora_cpu,
    ("matmul", "remora-gpu"): bench_matmul_remora_gpu,
    ("sort", "numpy"): bench_sort_numpy,
    ("sort", "jax"): bench_sort_jax,
    ("sort", "remora-cpu"): bench_sort_remora_cpu,
    ("sort", "remora-gpu"): bench_sort_remora_gpu,
    ("stencil", "numpy"): bench_stencil_numpy,
    ("stencil", "jax"): bench_stencil_jax,
    ("stencil", "remora-cpu"): bench_stencil_remora_cpu,
    ("stencil", "remora-gpu"): bench_stencil_remora_gpu,
    ("grad_descent", "numpy"): bench_grad_descent_numpy,
    ("grad_descent", "jax"): bench_grad_descent_jax,
    ("grad_descent", "remora-cpu"): bench_grad_descent_remora_cpu,
    ("grad_descent", "remora-gpu"): bench_grad_descent_remora_gpu,
    ("conv_pipeline", "numpy"): bench_conv_pipeline_numpy,
    ("conv_pipeline", "jax"): bench_conv_pipeline_jax,
    ("conv_pipeline", "remora-cpu"): bench_conv_pipeline_remora_cpu,
    ("nbody", "numpy"): bench_nbody_numpy,
    ("nbody", "jax"): bench_nbody_jax,
    ("nbody", "remora-cpu"): bench_nbody_remora_cpu,
    ("fusion", "mapchain-composed"): _make_fusion_bench("mapchain-composed", _FUSION_SOURCES["mapchain-composed"]),
    ("fusion", "mapchain-manual"): _make_fusion_bench("mapchain-manual", _FUSION_SOURCES["mapchain-manual"]),
    ("fusion", "triple-composed"): _make_fusion_bench("triple-composed", _FUSION_SOURCES["triple-composed"]),
    ("fusion", "triple-manual"): _make_fusion_bench("triple-manual", _FUSION_SOURCES["triple-manual"]),
    ("fusion", "dot-composed"): bench_fusion_dot,
    ("pipeline", "numpy"): bench_pipeline_numpy,
    ("pipeline", "jax"): bench_pipeline_jax,
    ("pipeline", "remora-gpu"): bench_pipeline_remora_gpu,
    ("pipeline", "remora-gpu-htod"): bench_pipeline_remora_gpu_htod,
}


def _sizes_for_op(op: str) -> tuple[int, ...]:
    if op == "matmul":
        return MATMUL_SIZES
    if op == "stencil":
        return STENCIL_SIZES
    if op == "grad_descent":
        return GRAD_DESCENT_SIZES
    if op == "conv_pipeline":
        return CONV_PIPELINE_SIZES
    if op == "nbody":
        return NBODY_SIZES
    if op == "fusion":
        return FUSION_SIZES
    if op == "pipeline":
        return PIPELINE_SIZES
    return DEFAULT_SIZES


def _all_backends_for_op(op: str) -> list[str]:
    return sorted({b for (o, b) in BENCHMARKS if o == op})


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------

def _fmt_time(s: float) -> str:
    if s < 1e-6:
        return f"{s * 1e9:.0f}ns"
    if s < 1e-3:
        return f"{s * 1e6:.1f}us"
    if s < 1.0:
        return f"{s * 1e3:.2f}ms"
    return f"{s:.3f}s"


def _fmt_throughput(t: float) -> str:
    if t >= 1e9:
        return f"{t / 1e9:.2f}G"
    if t >= 1e6:
        return f"{t / 1e6:.2f}M"
    if t >= 1e3:
        return f"{t / 1e3:.1f}K"
    return f"{t:.0f}"


def _print_results(results: list[BenchResult]) -> None:
    if not results:
        return
    hdr = f"{'Operation':<12s} {'Backend':<14s} {'Size':>10s}  {'Median':>10s}  {'Min':>10s}  {'Stddev':>10s}  {'Elem/s':>10s}"
    print(hdr)
    print("-" * len(hdr))
    for r in results:
        print(
            f"{r.operation:<12s} {r.backend:<14s} {r.size:>10d}  "
            f"{_fmt_time(r.median_s):>10s}  {_fmt_time(r.min_s):>10s}  "
            f"{_fmt_time(r.std_s):>10s}  {_fmt_throughput(r.throughput_elem_per_s):>10s}"
        )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="remora-perf",
        description="Benchmark Remora vs NumPy vs JAX runtime execution performance.",
    )
    parser.add_argument(
        "--ops", default=",".join(ALL_OPS),
        help=f"Comma-separated operations (default: {','.join(ALL_OPS)})",
    )
    parser.add_argument(
        "--backends", default=None,
        help="Comma-separated backends (default: all available for each op)",
    )
    parser.add_argument(
        "--sizes", default=None,
        help="Comma-separated sizes (overrides per-op defaults)",
    )
    parser.add_argument("--warmup", type=int, default=5, help="Warmup iterations (default: 5)")
    parser.add_argument("--trials", type=int, default=20, help="Timed iterations (default: 20)")
    parser.add_argument("--json", metavar="FILE", default=None, help="Write JSON results to FILE")
    parser.add_argument(
        "--no-pool", action="store_true",
        help="Disable the GPU device memory pool (allocate/free per call)",
    )
    parser.add_argument(
        "--device-resident", action="store_true",
        help="For GPU map/fold, keep data on the device (no host-device "
             "transfer in the timed loop) to isolate pure kernel time",
    )
    args = parser.parse_args(argv)

    _set_pool_enabled(not args.no_pool)
    _set_device_resident(args.device_resident)

    ops = [o.strip() for o in args.ops.split(",")]
    backends = [b.strip() for b in args.backends.split(",")] if args.backends else None
    size_override = (
        tuple(int(s.strip()) for s in args.sizes.split(",")) if args.sizes else None
    )

    results: list[BenchResult] = []

    for op in ops:
        sizes = size_override if size_override else _sizes_for_op(op)
        op_backends = backends if backends else _all_backends_for_op(op)

        for backend in op_backends:
            bench_fn = BENCHMARKS.get((op, backend))
            if bench_fn is None:
                continue

            for size in sizes:
                sys.stdout.write(f"  {op}/{backend} n={size} ... ")
                sys.stdout.flush()
                try:
                    r = bench_fn(size, args.warmup, args.trials)
                except Exception as exc:
                    print(f"ERROR: {exc}")
                    continue
                if r is None:
                    print("skipped (unavailable)")
                    continue
                print(f"{_fmt_time(r.median_s)} ({_fmt_throughput(r.throughput_elem_per_s)} elem/s)")
                results.append(r)

    print()
    _print_results(results)

    if args.json:
        with open(args.json, "w") as f:
            json.dump([asdict(r) for r in results], f, indent=2)
        print(f"\nResults written to {args.json}")


if __name__ == "__main__":
    main()
