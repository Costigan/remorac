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


@dataclass(frozen=True)
class BenchmarkResult:
    name: str
    cpu_threads: int | None
    mlir_compile_s: float
    fusion_pipeline_s: float
    cpu_pipeline_s: float
    compiled_execution_s: float
    linalg_generic_before: int
    linalg_generic_after_fusion: int
    llvm_func_count: int

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def benchmark_source(
    source: str,
    *,
    name: str = "program",
    cpu_threads: int | None = None,
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
    lowered = run_cpu_pipeline_text(mlir, toolchain=toolchain)
    cpu_pipeline_s = time.perf_counter() - start

    start = time.perf_counter()
    evaluate_source_compiled(source, cpu_threads=resolved_cpu_threads)
    compiled_execution_s = time.perf_counter() - start

    return BenchmarkResult(
        name=name,
        cpu_threads=resolved_cpu_threads,
        mlir_compile_s=mlir_compile_s,
        fusion_pipeline_s=fusion_pipeline_s,
        cpu_pipeline_s=cpu_pipeline_s,
        compiled_execution_s=compiled_execution_s,
        linalg_generic_before=mlir.count("linalg.generic"),
        linalg_generic_after_fusion=fused.count("linalg.generic"),
        llvm_func_count=lowered.count("llvm.func"),
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Benchmark a Remora Dense Core source file")
    parser.add_argument("file", type=Path, help="Remora source file")
    parser.add_argument("--name", default=None, help="benchmark case name")
    parser.add_argument(
        "--cpu-threads",
        type=int,
        default=None,
        help="requested CPU worker thread count; defaults to REMORA_NUM_THREADS when set",
    )
    args = parser.parse_args(argv)

    try:
        source = args.file.read_text(encoding="utf-8")
        result = benchmark_source(
            source,
            name=args.name or args.file.stem,
            cpu_threads=args.cpu_threads,
        )
    except (OSError, RemoraError) as exc:
        print(f"remora-bench: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(result.to_dict(), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
