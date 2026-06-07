"""Command-line entry points for Remora Dense Core."""

from __future__ import annotations

import argparse
from pathlib import Path
from pprint import pformat
import sys

from remora.codegen import CodegenUnavailable
from remora.compiler import compile_source, compile_source_to_ptx
from remora.defunc import defunctionalize
from remora.display import format_result
from remora.errors import RemoraError
from remora.hir import lower_to_hir
from remora.parser import parse_program
from remora.pipeline import PipelineUnavailable
from remora.prelude import with_prelude
from remora.runtime import evaluate_source, evaluate_source_compiled
from remora.typechecker import TypeChecker


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Remora Dense Core compiler")
    parser.add_argument("file", type=Path, help="Remora source file")
    parser.add_argument(
        "--target",
        choices=("cpu", "interp", "mlir", "ptx", "gpu-nvidia"),
        default="cpu",
        help="output target; cpu runs compiled CPU code, interp uses the reference evaluator, gpu-nvidia validates GPU compilation",
    )
    parser.add_argument("--emit-ast", action="store_true", help="print parsed AST and exit")
    parser.add_argument(
        "--emit-typed-ast",
        action="store_true",
        help="print typed AST and exit",
    )
    parser.add_argument("--emit-hir", action="store_true", help="print defunctionalized HIR and exit")
    parser.add_argument("--emit-mlir", action="store_true", help="print validated MLIR and exit")
    parser.add_argument("--emit-ptx", action="store_true", help="print generated PTX and exit")
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
    args = parser.parse_args(argv)

    try:
        source = args.file.read_text(encoding="utf-8")
        if args.emit_ast:
            print(pformat(parse_program(source, str(args.file))))
            return 0
        if args.emit_typed_ast:
            print(pformat(TypeChecker().check_program(parse_program(with_prelude(source), str(args.file)))))
            return 0
        if args.emit_hir:
            typed = TypeChecker().check_program(parse_program(with_prelude(source), str(args.file)))
            print(pformat(defunctionalize(lower_to_hir(typed))))
            return 0
        if args.emit_mlir:
            print(compile_source(source).mlir_text)
            return 0
        if args.emit_ptx:
            artifact = compile_source_to_ptx(source)
            print(artifact.ptx_text)
            return 0

        if args.target == "cpu":
            result = evaluate_source_compiled(
                source,
                cpu_threads=args.cpu_threads,
                cpu_vectorize=args.cpu_vectorize,
            )
            print(format_result(result.value, result.type))
            return 0
        if args.target == "interp":
            result = evaluate_source(source)
            print(format_result(result.value, result.type))
            return 0
        if args.target == "mlir":
            print(compile_source(source).mlir_text)
            return 0
        if args.target == "ptx":
            artifact = compile_source_to_ptx(source)
            print(artifact.ptx_text)
            return 0
        if args.target == "gpu-nvidia":
            _handle_gpu_target(source)
            return 0
        raise AssertionError(f"unknown target {args.target}")
    except (OSError, RemoraError, CodegenUnavailable, PipelineUnavailable) as exc:
        print(f"remorac: {exc}", file=sys.stderr)
        return 1


def _handle_gpu_target(source: str) -> None:
    """Compile a Remora body program to GPU and print diagnostics."""
    try:
        artifact = compile_source_to_ptx(source)
    except CodegenUnavailable as exc:
        raise CodegenUnavailable(
            f"GPU compilation failed: {exc}\n\n"
            "This program may use operations not supported on GPU. "
            "Currently supported GPU operations: element-wise maps (f32/i32/bool), "
            "and scalar reductions. Views, indexing, and scalar-only programs are "
            "not supported on GPU."
        ) from exc
    kernels = artifact.kernels
    if not kernels:
        raise CodegenUnavailable(
            "No GPU kernels were generated. This program may contain only "
            "scalar operations with no tensor-level parallelism. "
            "GPU target requires tensor operations (map, fold, etc.)."
        )
    print(f"GPU compilation succeeded — {len(kernels)} kernel(s) generated")


if __name__ == "__main__":
    raise SystemExit(main())
