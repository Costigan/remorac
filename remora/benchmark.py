"""Small benchmark harness for Remora Dense Core."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import sys
import time

from remora.compiler import compile_source_to_mlir
from remora.errors import RemoraError
from remora.pipeline import (
    PipelineToolchain,
    detect_toolchain,
    run_cpu_pipeline_text,
    run_fusion_pipeline_text,
)
from remora.runtime import evaluate_source_compiled, resolve_cpu_threads


BASELINE_SOURCES = {
    "vector_scale": "map (* 2.0) (iota 1000)",
    "map_chain": "map (* 3.0) (map (* 2.0) (iota 1000))",
    "vector_sum": "fold (+) 0.0 (iota 1000)",
    "dot": (
        "let xs = [1.0, 2.0, 3.0] in "
        "let ys = [4.0, 5.0, 6.0] in "
        "dot xs ys"
    ),
    "row_reduce": (
        "let xs = [[1.0, 2.0], [3.0, 4.0]] in "
        "map (\\row -> fold (+) 0.0 row) xs"
    ),
}


@dataclass(frozen=True)
class BenchmarkResult:
    name: str
    cpu_threads: int | None
    cpu_vectorize: bool
    mlir_compile_s: float
    fusion_pipeline_s: float
    cpu_pipeline_s: float
    compiled_execution_s: float
    linalg_generic_before: int
    linalg_generic_after_fusion: int
    llvm_func_count: int
    allocation_count: int

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def benchmark_source(
    source: str,
    *,
    name: str = "program",
    cpu_threads: int | None = None,
    cpu_vectorize: bool = False,
    toolchain: PipelineToolchain | None = None,
) -> BenchmarkResult:
    """Compile and execute one source string, returning coarse timing metrics."""
    resolved_cpu_threads = resolve_cpu_threads(cpu_threads)
    toolchain = detect_toolchain() if toolchain is None else toolchain

    start = time.perf_counter()
    mlir = compile_source_to_mlir(source, verify=False)
    mlir_compile_s = time.perf_counter() - start

    start = time.perf_counter()
    fused = run_fusion_pipeline_text(mlir, toolchain=toolchain)
    fusion_pipeline_s = time.perf_counter() - start

    start = time.perf_counter()
    lowered = run_cpu_pipeline_text(mlir, toolchain=toolchain, vectorize=cpu_vectorize)
    cpu_pipeline_s = time.perf_counter() - start

    start = time.perf_counter()
    evaluate_source_compiled(
        source,
        cpu_threads=resolved_cpu_threads,
        cpu_vectorize=cpu_vectorize,
    )
    compiled_execution_s = time.perf_counter() - start

    return BenchmarkResult(
        name=name,
        cpu_threads=resolved_cpu_threads,
        cpu_vectorize=cpu_vectorize,
        mlir_compile_s=mlir_compile_s,
        fusion_pipeline_s=fusion_pipeline_s,
        cpu_pipeline_s=cpu_pipeline_s,
        compiled_execution_s=compiled_execution_s,
        linalg_generic_before=mlir.count("linalg.generic"),
        linalg_generic_after_fusion=fused.count("linalg.generic"),
        llvm_func_count=lowered.count("llvm.func"),
        allocation_count=_allocation_count(lowered),
    )


def _allocation_count(lowered_mlir: str) -> int:
    return lowered_mlir.count("llvm.call @malloc") + lowered_mlir.count("memref.alloc")


def check_result_against_baseline(
    result: BenchmarkResult,
    baselines: dict[str, object],
) -> list[str]:
    cases = baselines.get("cases")
    if not isinstance(cases, list):
        return ["benchmark baseline file must contain a cases list"]
    baseline = next(
        (case for case in cases if isinstance(case, dict) and case.get("name") == result.name),
        None,
    )
    if baseline is None:
        return [f"benchmark baseline for {result.name!r} was not found"]

    failures: list[str] = []
    max_fused = baseline.get("max_linalg_generic_after_fusion")
    if isinstance(max_fused, int) and result.linalg_generic_after_fusion > max_fused:
        failures.append(
            f"{result.name}: linalg_generic_after_fusion {result.linalg_generic_after_fusion} > {max_fused}"
        )
    max_allocs = baseline.get("max_allocation_count")
    if isinstance(max_allocs, int) and result.allocation_count > max_allocs:
        failures.append(f"{result.name}: allocation_count {result.allocation_count} > {max_allocs}")
    return failures


def run_benchmark_suite(
    baseline_path: Path,
    *,
    cpu_threads: int | None = None,
    cpu_vectorize: bool = False,
) -> tuple[list[BenchmarkResult], list[str]]:
    """Run all cases defined in the baseline file."""
    baselines = json.loads(baseline_path.read_text(encoding="utf-8"))
    cases = baselines.get("cases", [])
    results: list[BenchmarkResult] = []
    all_failures: list[str] = []

    for case in cases:
        name = case.get("name")
        if not name:
            continue
        source = BASELINE_SOURCES.get(name)
        if not source:
            all_failures.append(f"No source string defined for baseline case {name!r}")
            continue

        result = benchmark_source(
            source,
            name=name,
            cpu_threads=cpu_threads,
            cpu_vectorize=cpu_vectorize,
        )
        results.append(result)
        all_failures.extend(check_result_against_baseline(result, baselines))

    return results, all_failures


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Benchmark a Remora Dense Core source file")
    parser.add_argument("file", type=Path, nargs="?", help="Remora source file")
    parser.add_argument("--name", default=None, help="benchmark case name")
    parser.add_argument(
        "--cpu-threads",
        type=int,
        default=None,
        help="requested CPU worker thread count; defaults to REMORA_NUM_THREADS when set",
    )
    vectorize_group = parser.add_mutually_exclusive_group()
    vectorize_group.add_argument(
        "--cpu-vectorize",
        dest="cpu_vectorize",
        action="store_true",
        help="use the experimental affine/vector CPU lowering pipeline",
    )
    vectorize_group.add_argument(
        "--no-cpu-vectorize",
        dest="cpu_vectorize",
        action="store_false",
        help="use the scalar CPU lowering pipeline",
    )
    parser.set_defaults(cpu_vectorize=False)
    parser.add_argument(
        "--baseline",
        type=Path,
        default=None,
        help="optional benchmark baseline JSON file to check this result against",
    )
    parser.add_argument(
        "--suite",
        action="store_true",
        help="run all cases from the baseline file; requires --baseline",
    )
    args = parser.parse_args(argv)

    if args.suite and not args.baseline:
        print("remora-bench: --suite requires --baseline", file=sys.stderr)
        return 1
    if not args.suite and not args.file:
        print("remora-bench: file is required unless using --suite", file=sys.stderr)
        return 1

    try:
        if args.suite:
            results, failures = run_benchmark_suite(
                args.baseline,
                cpu_threads=args.cpu_threads,
                cpu_vectorize=args.cpu_vectorize,
            )
            print(json.dumps([r.to_dict() for r in results], sort_keys=True))
        else:
            source = args.file.read_text(encoding="utf-8")
            result = benchmark_source(
                source,
                name=args.name or args.file.stem,
                cpu_threads=args.cpu_threads,
                cpu_vectorize=args.cpu_vectorize,
            )
            failures = []
            if args.baseline is not None:
                baseline = json.loads(args.baseline.read_text(encoding="utf-8"))
                failures = check_result_against_baseline(result, baseline)
            print(json.dumps(result.to_dict(), sort_keys=True))
    except (OSError, RemoraError, json.JSONDecodeError) as exc:
        print(f"remora-bench: {exc}", file=sys.stderr)
        return 1

    if failures:
        for failure in failures:
            print(f"remora-bench: {failure}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
