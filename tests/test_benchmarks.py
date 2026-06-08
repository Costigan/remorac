"""Benchmark: compile vs execute time for CPU and GPU.

Reports three numbers per size:
  compile — one-time cost (parse → typecheck → HIR → codegen)
  exec   — median execution time after compile
  result — correctness check

Run: uv run python tests/test_benchmarks.py
"""

import time
import numpy as np

from remora.runtime import evaluate_source, CPUExecutor


def bench_cpu_detail(source: str, *, syntax: str = "ml", repeats: int = 200):
    """Returns (compile_time_sec, median_exec_time_sec)."""
    t0 = time.perf_counter()
    artifact = CPUExecutor.compile_source(source, include_prelude=False, syntax=syntax)
    compile_time = time.perf_counter() - t0

    executor = CPUExecutor(artifact)
    for _ in range(10):
        executor.execute_main([])

    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        executor.execute_main([])
        times.append(time.perf_counter() - t0)
    artifact.close()
    return compile_time, float(np.median(times))


def bench_gpu_detail(source: str, *, repeats: int = 50):
    """Returns (compile_time_sec, median_exec_time_sec).

    Compiles to PTX once, pre-loads CUDA module once, then launches
    the program repeatedly with shared runtime/module.
    """
    from remora.compiler import compile_source_to_ptx
    from remora.executor import execute_program_from_ptx, GPUPtxContext

    t0 = time.perf_counter()
    artifact = compile_source_to_ptx(source, include_prelude=True)
    compile_time = time.perf_counter() - t0

    ctx = GPUPtxContext(artifact.ptx_text)
    try:
        for _ in range(3):
            execute_program_from_ptx(artifact, context=ctx)

        times = []
        for _ in range(repeats):
            t0 = time.perf_counter()
            execute_program_from_ptx(artifact, context=ctx)
            times.append(time.perf_counter() - t0)
    finally:
        ctx.close()

    return compile_time, float(np.median(times))


def bench_interp(source: str, *, syntax: str = "ml", repeats: int = 10):
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        evaluate_source(source, include_prelude=False, syntax=syntax)
        times.append(time.perf_counter() - t0)
    return float(np.median(times))


def fmt_sec(s):
    if s < 1e-6:
        return f"{(s * 1e9):4.0f}ns"
    if s < 1e-3:
        return f"{(s * 1e6):5.1f}us"
    return f"{(s * 1e3):6.2f}ms"


if __name__ == "__main__":
    print("Remora Benchmark Suite")
    print("=" * 80)
    print(f"{'Op':20s} {'N':>8s}  {'CPU compile':>11s} {'CPU exec':>10s}  {'GPU compile':>12s} {'GPU exec':>10s}  Interp")
    print("-" * 80)

    for n in [100, 1000, 10000, 100000]:
        src = f"fold (+) 0 (iota {n})"
        cc, ce = bench_cpu_detail(src)
        gc, ge = bench_gpu_detail(src)
        interp = bench_interp(src)
        print(f"{'fold':20s} {n:8d}  {fmt_sec(cc):>11s} {fmt_sec(ce):>10s}  {fmt_sec(gc):>12s} {fmt_sec(ge):>10s}  {fmt_sec(interp)}")

    for n in [100, 1000, 10000, 100000]:
        src = f"map (* 2) (iota {n})"
        cc, ce = bench_cpu_detail(src)
        gc, ge = bench_gpu_detail(src)
        interp = bench_interp(src)
        print(f"{'map':20s} {n:8d}  {fmt_sec(cc):>11s} {fmt_sec(ce):>10s}  {fmt_sec(gc):>12s} {fmt_sec(ge):>10s}  {fmt_sec(interp)}")

    print()
    print("Legend: compile = one-time, exec = median post-compile, Interp = interpreter")
