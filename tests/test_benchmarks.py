"""Benchmark: compiled CPU execution vs interpreter.

Compiles once, executes many times for fair comparison.
Run: uv run python tests/test_benchmarks.py
"""

import time
import numpy as np

from remora.runtime import evaluate_source, CPUExecutor


def bench_cpu(source: str, *, syntax: str = "ml", repeats: int = 200) -> float:
    """Compile once, execute *repeats* times. Returns median seconds."""
    artifact = CPUExecutor.compile_source(source, include_prelude=False, syntax=syntax)
    executor = CPUExecutor(artifact)
    for _ in range(10):
        executor.execute_main([])
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        executor.execute_main([])
        times.append(time.perf_counter() - t0)
    artifact.close()
    return float(np.median(times))


def bench_interp(source: str, *, syntax: str = "ml", repeats: int = 10) -> float:
    """Time interpreter execution. Returns median seconds."""
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        evaluate_source(source, include_prelude=False, syntax=syntax)
        times.append(time.perf_counter() - t0)
    return float(np.median(times))


def report(name: str, cpu: float, interp: float):
    speedup = interp / cpu if cpu > 0 else 0
    icon = "✅" if speedup > 10 else "⚡" if speedup > 1 else "🐢"
    print(f"  {icon} {name:28s}  cpu={cpu*1e6:8.1f}us  interp={interp*1e6:8.1f}us  {speedup:7.1f}x")


if __name__ == "__main__":
    print("Remora Benchmark Suite")
    print("=" * 65)
    print()

    # --- Fold (ML syntax) ---
    print("Fold sum (ML):")
    for n in [100, 1000, 10000, 100000]:
        c = bench_cpu(f"fold (+) 0 (iota {n})")
        i = bench_interp(f"fold (+) 0 (iota {n})")
        report(f"fold N={n}", c, i)

    # --- Map (ML syntax) ---
    print("\nMap double (ML):")
    for n in [100, 1000, 10000, 100000]:
        c = bench_cpu(f"map (* 2) (iota {n})")
        i = bench_interp(f"map (* 2) (iota {n})")
        report(f"map N={n}", c, i)

    # --- Binary map (ML) ---
    print("\nBinary map (ML):")
    for n in [100, 1000, 10000]:
        c = bench_cpu(f"map (+) (iota {n}) (iota {n})")
        i = bench_interp(f"map (+) (iota {n}) (iota {n})")
        report(f"bimap N={n}", c, i)

    # --- Fold (Lisp syntax) ---
    print("\nFold sum (Lisp):")
    for n in [100, 1000, 10000]:
        arr = " ".join(str(x) for x in range(n))
        c = bench_cpu(f"(fold + 0 [{arr}])", syntax="lisp")
        i = bench_interp(f"(fold + 0 [{arr}])", syntax="lisp")
        report(f"lisp fold N={n}", c, i)

    # --- Implicit vs Explicit (ML) ---
    print("\nImplicit vs Explicit add (ML):")
    for n in [100, 1000, 10000]:
        c_imp = bench_cpu(f"(iota {n}) + (iota {n})")
        c_exp = bench_cpu(f"map (+) (iota {n}) (iota {n})")
        ovh = c_imp / c_exp if c_exp > 0 else 1.0
        print(f"  N={n:5d}  implicit={c_imp*1e6:8.1f}us  explicit={c_exp*1e6:8.1f}us  ratio={ovh:.2f}x")

    # --- Scan (ML) ---
    print("\nScan (ML):")
    for n in [100, 1000, 5000]:
        c = bench_cpu(f"iscan (+) 0 (iota {n})", repeats=50)
        i = bench_interp(f"iscan (+) 0 (iota {n})")
        report(f"iscan N={n}", c, i)

    # --- Compilation time ---
    print("\nCompilation overhead:")
    t0 = time.perf_counter()
    bench_cpu("iota 1000", repeats=1)
    comp_time = time.perf_counter() - t0
    print(f"  First compile+execute: {comp_time*1000:.1f}ms")

    print("\nDone.")
