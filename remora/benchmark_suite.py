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


# ---------------------------------------------------------------------------
# Remora GPU benchmarks
# ---------------------------------------------------------------------------

def _get_gpu_runtime():
    from remora.runtime import CUDARuntime, RuntimeUnavailable
    try:
        return CUDARuntime()
    except RuntimeUnavailable:
        return None


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
        from remora.codegen import generate_mlir_descriptor_abi_ptx
        from remora.executor import RemoraExecutor
        from remora.hir import HIRFunction, HIRParam, HIRScan, HIRVar
        from remora.types import FLOAT, ArrayType, StaticDim, StaticDim as SD
        from remora.hir import HIRLit, HIRLambda
    except ImportError:
        return None
    rt = _get_gpu_runtime()
    if rt is None:
        return None
    try:
        from remora.compiler import compile_function_source_to_mlir_gpu_ptx
        ptx, kernels, _ = compile_function_source_to_mlir_gpu_ptx(
            "def scanit xs = iscan (+) 0.0 xs", "scanit",
            (ArrayType(FLOAT, (StaticDim(n),)),),
            include_prelude=False,
            kernel_name="bench_scan",
        )
        xs = np.random.default_rng(0).standard_normal(n).astype(np.float32)
        with RemoraExecutor(ptx, kernels, runtime=rt) as exe:
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
            def run():
                exe.execute("bench_stencil", [img, kernel])
            return _result("stencil", "remora-gpu", patches, _measure(run, warmup, trials))
    except Exception:
        return None
    finally:
        rt.close()


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

ALL_OPS = ("map", "fold", "scan", "matmul", "sort", "stencil")

DEFAULT_SIZES = (1_000, 10_000, 100_000, 1_000_000)
MATMUL_SIZES = (32, 64, 128, 256, 512)
STENCIL_SIZES = (32, 64, 128, 256, 512)

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
    ("matmul", "remora-gpu"): bench_matmul_remora_gpu,
    ("sort", "numpy"): bench_sort_numpy,
    ("sort", "jax"): bench_sort_jax,
    ("sort", "remora-cpu"): bench_sort_remora_cpu,
    ("sort", "remora-gpu"): bench_sort_remora_gpu,
    ("stencil", "numpy"): bench_stencil_numpy,
    ("stencil", "jax"): bench_stencil_jax,
    ("stencil", "remora-cpu"): bench_stencil_remora_cpu,
    ("stencil", "remora-gpu"): bench_stencil_remora_gpu,
}


def _sizes_for_op(op: str) -> tuple[int, ...]:
    if op == "matmul":
        return MATMUL_SIZES
    if op == "stencil":
        return STENCIL_SIZES
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
    args = parser.parse_args(argv)

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
