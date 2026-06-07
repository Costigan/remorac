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
from remora.lisp_reader import parse_lisp as parse_lisp_program
from remora.pipeline import PipelineUnavailable
from remora.prelude import with_prelude
from remora.runtime import evaluate_source, evaluate_source_compiled
from remora.typechecker import TypeChecker


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Remora Dense Core compiler")
    parser.add_argument("file", type=Path, help="Remora source file")
    parser.add_argument(
        "--syntax",
        choices=("ml", "lisp"),
        default="ml",
        help="syntax for reading source files; ml (default) or lisp",
    )
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
        help="use the affine/vector CPU lowering pipeline",
    )
    vectorize_group.add_argument(
        "--no-cpu-vectorize",
        dest="cpu_vectorize",
        action="store_false",
        help="use the scalar CPU lowering pipeline (default)",
    )
    parser.set_defaults(cpu_vectorize=False)
    parser.add_argument(
        "--call",
        type=str,
        default=None,
        help="call a named function with descriptor ABI (requires --input for each param)",
    )
    parser.add_argument(
        "--input",
        type=Path,
        action="append",
        default=None,
        help="load a .npy file as input to a --call function",
    )
    args = parser.parse_args(argv)

    try:
        source = args.file.read_text(encoding="utf-8")
        syntax: str = args.syntax
        if args.call is not None:
            return _handle_function_call(args, source)
        if args.emit_ast:
            print(pformat(_parse(source, str(args.file), syntax)))
            return 0
        if args.emit_typed_ast:
            print(pformat(TypeChecker().check_program(_parse(with_prelude(source), str(args.file), syntax))))
            return 0
        if args.emit_hir:
            typed = TypeChecker().check_program(_parse(with_prelude(source), str(args.file), syntax))
            print(pformat(defunctionalize(lower_to_hir(typed))))
            return 0
        if args.emit_mlir:
            print(compile_source(source, syntax=syntax).mlir_text)
            return 0
        if args.emit_ptx:
            artifact = compile_source_to_ptx(source, syntax=syntax)
            print(artifact.ptx_text)
            return 0

        if args.target == "cpu":
            result = evaluate_source_compiled(
                source,
                cpu_threads=args.cpu_threads,
                cpu_vectorize=args.cpu_vectorize,
                syntax=syntax,
            )
            print(format_result(result.value, result.type))
            return 0
        if args.target == "interp":
            result = evaluate_source(source, syntax=syntax)
            print(format_result(result.value, result.type))
            return 0
        if args.target == "mlir":
            print(compile_source(source, syntax=syntax).mlir_text)
            return 0
        if args.target == "ptx":
            artifact = compile_source_to_ptx(source, syntax=syntax)
            print(artifact.ptx_text)
            return 0
        if args.target == "gpu-nvidia":
            _handle_gpu_target(source, syntax=syntax)
            return 0
        raise AssertionError(f"unknown target {args.target}")
    except (OSError, RemoraError, CodegenUnavailable, PipelineUnavailable) as exc:
        print(f"remorac: {exc}", file=sys.stderr)
        return 1


def _parse(source: str, filename: str = "<input>", syntax: str = "ml"):
    if syntax == "lisp":
        return parse_lisp_program(source, filename)
    return parse_program(source, filename)


def _handle_function_call(args, source: str) -> int:
    """Compile and execute a named function with .npy input arrays."""
    import numpy as np
    from remora.compiler import compile_function_source_to_supported_gpu_artifacts
    from remora.executor import RemoraExecutor
    from remora.types import ArrayType, FLOAT, INT, StaticDim
    from remora.display import format_result

    if args.input is None:
        print("remorac: --call requires at least one --input FILE.npy", file=sys.stderr)
        return 1

    arrays = []
    for path in args.input:
        arr = np.load(str(path))
        arrays.append(arr)

    param_types = tuple(
        ArrayType(
            FLOAT if arr.dtype == np.float32 else INT,
            tuple(StaticDim(d) for d in arr.shape),
        )
        for arr in arrays
    )

    try:
        artifact = compile_function_source_to_supported_gpu_artifacts(
            source,
            args.call,
            param_types,
        )
    except Exception as exc:
        print(f"remorac: GPU function compilation failed: {exc}", file=sys.stderr)
        return 1

    try:
        executor = RemoraExecutor(artifact.ptx_text, artifact.kernels)
        kernel_name = args.call
        if kernel_name not in executor._kernels:
            kernel_name = next(iter(executor._kernels))
        result = executor.execute(kernel_name, arrays)
        print(format_result(result, artifact.compiler.return_type))
        executor.close()
        return 0
    except Exception as exc:
        print(f"remorac: GPU execution failed: {exc}", file=sys.stderr)
        return 1


def _handle_gpu_target(source: str, syntax: str = "ml") -> None:
    """Compile a Remora body program to GPU, execute it, and print the result."""
    from remora.executor import execute_program_on_gpu
    from remora.display import format_result
    from remora.compiler import compile_source

    try:
        result = execute_program_on_gpu(source)
        artifact = compile_source(source)
        from remora.types import ArrayType, ScalarType
        from remora.runtime import _result_dtype
        rtype = artifact.return_type
        if rtype is None:
            raise CodegenUnavailable("Cannot determine result type")
        print(format_result(result, rtype))
    except CodegenUnavailable as exc:
        raise CodegenUnavailable(
            f"GPU execution failed: {exc}\n\n"
            "This program may use operations not supported on GPU. "
            "Currently supported GPU operations: element-wise maps (f32/i32/bool), "
            "and scalar reductions. Views, indexing, and scalar-only programs are "
            "not supported on GPU."
        ) from exc


if __name__ == "__main__":
    raise SystemExit(main())
