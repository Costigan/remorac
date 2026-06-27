"""Command-line entry point for Remora Dense Core compiler."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from pprint import pformat
from typing import Any

from remora.codegen import CodegenUnavailable
from remora.compiler import compile_source, compile_source_to_ptx, explain_lowering
from remora.defunc import defunctionalize
from remora.display import format_result
from remora.errors import RemoraError
from remora.hir import lower_to_hir
from remora.lisp_reader import parse_lisp as parse_lisp_program
from remora.parser import parse_program
from remora.pipeline import (
    PipelineUnavailable,
    detect_toolchain,
    run_cpu_pipeline_text,
    translate_mlir_to_llvmir,
)
from remora.prelude import prelude_definition_sources, with_prelude
from remora.runtime import (
    CPUExecutor,
    CPUFunctionExecutor,
    CompiledCPUArtifact,
    EvaluationResult,
    check_metadata,
    compile_llvm_ir_to_executable,
    compile_llvm_ir_to_path,
    resolve_cpu_threads,
    write_metadata,
)
from remora.types import ArrayType, BOOL, FLOAT, FLOAT64, INT, RemoraType, ScalarType
from remora.typechecker import TypeChecker


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Remora Dense Core compiler")
    parser.add_argument("files", nargs="*", type=Path, help="Remora source files (.remora or .lisp)")
    parser.add_argument(
        "--syntax",
        choices=("ml", "lisp"),
        default=None,
        help="syntax override; inferred from file extension by default",
    )
    parser.add_argument(
        "--target",
        choices=("cpu", "interp", "cuda"),
        default="cpu",
        help="output target: cpu (compile and run), interp (reference evaluator), cuda (compile and run on GPU)",
    )
    parser.add_argument(
        "--repl", action="store_true",
        help="load source files and start interactive REPL",
    )
    parser.add_argument(
        "--compile-only", action="store_true",
        help="compile to executable and exit without running",
    )
    parser.add_argument(
        "-shared", "--shared", action="store_true",
        help="produce a shared library (.so) instead of an executable",
    )
    parser.add_argument(
        "-o", "--output", type=Path, default=None,
        help="output .so path (default: a.so in current directory)",
    )
    parser.add_argument(
        "--cleanup", "--rm", action="store_true", dest="cleanup",
        help="remove .so and .json after execution",
    )
    parser.add_argument("--emit-ast", action="store_true", help="print parsed AST and exit")
    parser.add_argument("--emit-typed-ast", action="store_true", help="print typed AST and exit")
    parser.add_argument("--emit-hir", action="store_true", help="print defunctionalized HIR and exit")
    parser.add_argument("--emit-mlir", action="store_true", help="print validated MLIR and exit")
    parser.add_argument("--emit-ptx", action="store_true", help="print generated PTX and exit")
    parser.add_argument(
        "--explain-lowering", nargs="?", const="text", default=None,
        choices=("text", "json"),
        help="explain which lowering route was selected for GPU (text or json)",
    )
    parser.add_argument(
        "--cpu-threads", type=int, default=None,
        help="requested CPU worker thread count; defaults to REMORA_NUM_THREADS when set",
    )
    vectorize_group = parser.add_mutually_exclusive_group()
    vectorize_group.add_argument("--cpu-vectorize", dest="cpu_vectorize", action="store_true",
                                 help="use the affine/vector CPU lowering pipeline (default)")
    vectorize_group.add_argument("--no-cpu-vectorize", dest="cpu_vectorize", action="store_false",
                                 help="use the scalar CPU lowering pipeline")
    parser.set_defaults(cpu_vectorize=True)
    parser.add_argument(
        "--call", type=str, default=None,
        help="call a named function with descriptor ABI (requires --input for each param)",
    )
    parser.add_argument(
        "--input", type=Path, action="append", default=None,
        help="load a .npy file as input to a --call function",
    )
    parser.add_argument(
        "--args",
        nargs=argparse.REMAINDER,
        default=None,
        help="Remora literal arguments passed to main or the sole defined function",
    )
    args = parser.parse_args(argv)

    try:
        return _run(args)
    except (OSError, RemoraError, CodegenUnavailable, PipelineUnavailable) as exc:
        print(f"remorac: {exc}", file=sys.stderr)
        return 1


def _run(args: argparse.Namespace) -> int:
    # --- validate contradictory flags ---
    if args.compile_only and args.repl:
        print("remorac: --compile-only and --repl cannot be used together", file=sys.stderr)
        return 1
    if args.cleanup and args.compile_only:
        print("remorac: --cleanup cannot be used with --compile-only", file=sys.stderr)
        return 1

    emit_flags = [args.emit_ast, args.emit_typed_ast, args.emit_hir, args.emit_mlir, args.emit_ptx]
    if any(emit_flags) and args.compile_only:
        print("remorac: --emit-* flags cannot be used with --compile-only", file=sys.stderr)
        return 1

    # --- handle --call separately ---
    if args.call is not None:
        if not args.files:
            print("remorac: --call requires at least one source file", file=sys.stderr)
            return 1
        return _handle_function_call(args)

    # --- handle no files ---
    if not args.files:
        if args.repl:
            return _run_repl([], args.syntax, args.target)
        print("remorac: at least one source file is required", file=sys.stderr)
        return 1

    # --- resolve syntax ---
    syntax = _resolve_syntax(args.files, args.syntax)
    if syntax is None:
        print("remorac: cannot mix .remora and .lisp source files", file=sys.stderr)
        return 1

    # --- load and concat sources ---
    sources: dict[str, str] = {}
    for f in args.files:
        if not f.is_file():
            print(f"remorac: file not found: {f}", file=sys.stderr)
            return 1
        sources[str(f)] = f.read_text(encoding="utf-8")

    combined_source = _concat_sources(list(sources.items()), syntax)
    entry_name: str | None = None
    if args.args is not None:
        entry_name = _resolve_args_entry_name(combined_source, str(args.files[-1]), syntax)
        if entry_name is None:
            print(
                "remorac: --args requires a function named 'main' or a single function definition",
                file=sys.stderr,
            )
            return 1

    # --- --repl ---
    if args.repl:
        source_items = list(sources.items())
        return _run_repl(source_items, syntax, args.target)

    # --- --emit-* flags ---
    if args.emit_ast:
        print(pformat(_parse_source(combined_source, str(args.files[0]), syntax)))
        return 0
    if args.emit_typed_ast:
        print(pformat(TypeChecker().check_program(
            _parse_source(with_prelude(combined_source) if syntax == "ml" else combined_source,
                          str(args.files[0]), syntax))))
        return 0
    if args.emit_hir:
        typed = TypeChecker().check_program(
            _parse_source(with_prelude(combined_source) if syntax == "ml" else combined_source,
                          str(args.files[0]), syntax))
        print(pformat(defunctionalize(lower_to_hir(typed))))
        return 0
    if args.emit_mlir:
        print(compile_source(combined_source, syntax=syntax).mlir_text)
        return 0
    if args.emit_ptx:
        artifact = compile_source_to_ptx(combined_source, syntax=syntax)
        print(artifact.ptx_text)
        return 0
    if args.explain_lowering is not None:
        explanation = explain_lowering(combined_source, syntax=syntax)
        if args.explain_lowering == "json":
            import json
            print(json.dumps({
                "target": explanation.target,
                "route_selected": explanation.route_selected,
                "capability_keys": explanation.capability_keys,
                "decisions": explanation.decisions,
            }, indent=2))
        else:
            print(f"target: {explanation.target}")
            if explanation.route_selected:
                print(f"route selected: {explanation.route_selected}")
                print(f"capability keys: {', '.join(explanation.capability_keys) if explanation.capability_keys else 'none'}")
            print("decisions:")
            for d in explanation.decisions:
                status = "ACCEPTED" if d["accepted"] else "REJECTED"
                print(f"  [{status}] {d['route_name']}: {d['reason']}")
        return 0

    # --- resolve output path ---
    is_shared = args.shared
    default_name = "a.so" if is_shared else "a.out"
    if args.output is not None:
        out_path = args.output.resolve()
    else:
        out_path = Path(default_name).resolve()

    # --- target dispatch ---
    if args.target == "interp":
        if entry_name is not None:
            combined_source = _source_with_entry_call(combined_source, syntax, entry_name, args.args)
        result = _evaluate_interp(combined_source, syntax)
        print(format_result(result.value, result.type))
        return 0
    if args.target == "cuda":
        if entry_name is not None:
            print("remorac: --args is not supported with --target cuda yet", file=sys.stderr)
            return 1
        _handle_gpu_target(combined_source, syntax)
        return 0

    # --- cpu target: compile and optionally run ---
    if entry_name is not None:
        return _run_cpu_function_with_args(
            combined_source,
            syntax,
            entry_name,
            args.args,
            args.compile_only,
            resolved_threads=resolve_cpu_threads(args.cpu_threads),
            cpu_vectorize=args.cpu_vectorize,
        )

    resolved_threads = resolve_cpu_threads(args.cpu_threads)
    threaded = resolved_threads is not None and resolved_threads > 1

    # Check metadata for incremental rebuild
    needs_compile = not check_metadata(
        out_path, sources,
        cpu_threads=resolved_threads or 1,
        cpu_vectorize=args.cpu_vectorize,
    )

    if needs_compile:
        compiler_artifact = compile_source(
            combined_source,
            verify=False,
            include_prelude=(syntax == "ml"),
            export_output_descriptor=True,
            syntax=syntax,
        )
        if compiler_artifact.return_type is None:
            print("remorac: definition-only programs cannot be compiled for CPU execution", file=sys.stderr)
            return 1

        toolchain = detect_toolchain()
        if threaded:
            from remora.runtime import has_openmp_runtime
            if not has_openmp_runtime():
                raise PipelineUnavailable(
                    "cpu_threads > 1 requires an OpenMP runtime; install libomp or use --cpu-threads 1"
                )
        try:
            lowered = run_cpu_pipeline_text(
                compiler_artifact.mlir_text,
                toolchain=toolchain,
                threaded=threaded,
                vectorize=args.cpu_vectorize,
            )
        except PipelineUnavailable as exc:
            if threaded:
                raise PipelineUnavailable(
                    f"threaded CPU lowering is not available for this program: {exc}"
                ) from exc
            raise
        llvm_ir = translate_mlir_to_llvmir(lowered, toolchain=toolchain)

        if is_shared:
            compile_llvm_ir_to_path(
                llvm_ir, out_path,
                toolchain=toolchain,
                threaded=threaded,
            )
        else:
            compile_llvm_ir_to_executable(
                llvm_ir, out_path, compiler_artifact.return_type,
                toolchain=toolchain,
                threaded=threaded,
            )

        write_metadata(
            out_path, sources,
            cpu_threads=resolved_threads or 1,
            cpu_vectorize=args.cpu_vectorize,
        )
        return_type: Any = compiler_artifact.return_type
    else:
        compiler_artifact = compile_source(
            combined_source,
            verify=False,
            include_prelude=(syntax == "ml"),
            export_output_descriptor=True,
            syntax=syntax,
        )
        if compiler_artifact.return_type is None:
            print("remorac: definition-only programs cannot be compiled for CPU execution", file=sys.stderr)
            return 1
        return_type = compiler_artifact.return_type

    if args.compile_only:
        print(f"Compiled: {out_path}")
        return 0

    return _execute_and_cleanup(out_path, return_type, resolved_threads, args.cpu_vectorize, args.cleanup, is_shared)


def _execute_and_cleanup(
    out_path: Path,
    return_type: Any,
    cpu_threads: int | None,
    cpu_vectorize: bool,
    cleanup: bool,
    is_shared: bool = False,
) -> int:
    if is_shared:
        artifact = CompiledCPUArtifact(
            out_path, None,
            return_type,
            cpu_threads, cpu_vectorize,
        )
        try:
            executor = CPUExecutor(artifact)
            value = executor.execute_main([])
            result = EvaluationResult(value, return_type)
            print(format_result(result.value, result.type))
        finally:
            if cleanup:
                out_path.unlink(missing_ok=True)
                json_path = out_path.with_suffix(".json")
                json_path.unlink(missing_ok=True)
    else:
        import subprocess
        result = subprocess.run([str(out_path)], capture_output=True, text=True)
        sys.stdout.write(result.stdout)
        if result.returncode != 0:
            sys.stderr.write(result.stderr)
            return result.returncode
        if cleanup:
            out_path.unlink(missing_ok=True)
            json_path = out_path.with_suffix(".json")
            json_path.unlink(missing_ok=True)
    return 0


def _evaluate_interp(source: str, syntax: str) -> EvaluationResult:
    from remora.runtime import evaluate_source
    return evaluate_source(source, syntax=syntax)


def _run_repl(source_items: list[tuple[str, str]], syntax_override: str | None, target: str) -> int:
    from remora.repl import ReplSession, ReplError

    try:
        session = ReplSession(target=target)
    except ReplError as exc:
        print(f"remorac: {exc}", file=sys.stderr)
        return 1

    for filename, content in source_items:
        ext = Path(filename).suffix
        file_syntax = "lisp" if ext == ".lisp" else "ml"
        effective_syntax = syntax_override or file_syntax
        session.state.syntax = effective_syntax

        try:
            session.load_source(
                content,
                filename,
                syntax=effective_syntax,
                evaluate_body=False,
            )
        except RemoraError as exc:
            print(f"remorac: error loading {filename}: {exc}", file=sys.stderr)
            return 1

    try:
        session.run()
    except SystemExit:
        pass
    return 0


def _concat_sources(source_items: list[tuple[str, str]], syntax: str) -> str:
    """Combine multiple source files: definitions from all, body from the last."""
    if len(source_items) == 1:
        return source_items[0][1]

    from remora.ast_nodes import FuncDef, ValDef

    def_parts: list[str] = []
    for filename, content in source_items:
        file_syntax = "lisp" if Path(filename).suffix == ".lisp" else "ml"
        try:
            program = (parse_lisp_program(content, filename) if file_syntax == "lisp"
                     else parse_program(content, filename))
        except Exception:
            def_parts.append(content)
            continue

        for definition in program.definitions:
            if isinstance(definition, (FuncDef, ValDef)):
                src = _extract_definition_source(content, definition)
                if src:
                    def_parts.append(src)

    last_content = source_items[-1][1]
    last_file_syntax = "lisp" if Path(source_items[-1][0]).suffix == ".lisp" else "ml"
    try:
        last_program = (parse_lisp_program(last_content, source_items[-1][0]) if last_file_syntax == "lisp"
                      else parse_program(last_content, source_items[-1][0]))
        if last_program.body is not None:
            body_src = _extract_body_source(last_content)
            if body_src:
                return "\n".join(def_parts) + "\n" + body_src
    except Exception:
        pass

    return "\n".join(def_parts) + "\n" + last_content


def _extract_definition_source(source: str, definition: Any) -> str | None:
    """Extract the source text of a definition from the original source."""
    lines = source.splitlines()
    loc = getattr(definition, "loc", None)
    if loc is not None and hasattr(loc, "line") and loc.line > 0:
        start_line = loc.line - 1
        end_line = start_line
        if hasattr(loc, "end_line") and loc.end_line > 0:
            end_line = loc.end_line - 1
        elif hasattr(loc, "ecl") and loc.ecl > 0:
            end_line = loc.ecl - 1
        end_line = min(end_line, len(lines) - 1)
        return "\n".join(lines[start_line:end_line + 1])
    name = getattr(definition, "name", None)
    if name and hasattr(name, "__str__"):
        if hasattr(definition, "params"):
            params = " ".join(str(p) for p in definition.params) if hasattr(definition, "params") else ""
            return f"def {name} {params} = ..."
        return f"def {name} = ..."
    return None


def _extract_body_source(source: str) -> str | None:
    """Extract the body expression source from a Remora program."""
    lines = source.splitlines()
    # Skip comment lines at the start
    start = 0
    for line in lines:
        stripped = line.strip()
        if stripped and not stripped.startswith("--"):
            break
        start += 1
    if start >= len(lines):
        return None
    return "\n".join(lines[start:])


def _resolve_args_entry_name(source: str, filename: str, syntax: str) -> str | None:
    from remora.ast_nodes import FuncDef

    program = _parse_source(source, filename, syntax)
    functions = [definition for definition in program.definitions if isinstance(definition, FuncDef)]
    main_defs = [definition for definition in functions if str(definition.name) == "main"]

    if main_defs:
        return "main"
    if len(functions) == 1 and program.body is None:
        return str(functions[0].name)
    return None


def _source_with_entry_call(source: str, syntax: str, entry: str, arg_exprs: list[str]) -> str:
    if syntax == "lisp":
        call = f"({entry}{(' ' + ' '.join(arg_exprs)) if arg_exprs else ''})"
    else:
        call = " ".join([entry, *arg_exprs])
    return f"{source.rstrip()}\n{call}\n"


def _run_cpu_function_with_args(
    source: str,
    syntax: str,
    entry_name: str,
    arg_exprs: list[str],
    compile_only: bool,
    *,
    resolved_threads: int | None,
    cpu_vectorize: bool,
) -> int:
    if compile_only:
        print("remorac: --compile-only with --args is not supported yet", file=sys.stderr)
        return 1

    values, param_types = _evaluate_arg_exprs(arg_exprs, syntax)
    artifact = CPUFunctionExecutor.compile_source(
        source,
        entry_name,
        param_types,
        include_prelude=(syntax == "ml"),
        syntax=syntax,
        cpu_threads=resolved_threads,
        cpu_vectorize=cpu_vectorize,
    )
    result = CPUFunctionExecutor(artifact).execute(*values)
    print(format_result(result.value, result.type))
    return 0


def _evaluate_arg_exprs(arg_exprs: list[str], syntax: str) -> tuple[list[object], tuple[RemoraType, ...]]:
    values: list[object] = []
    types: list[RemoraType] = []
    for expr in arg_exprs:
        result = _evaluate_interp(expr, syntax)
        values.append(_value_for_compiled_arg(result.value, result.type))
        types.append(result.type)
    return values, tuple(types)


def _value_for_compiled_arg(value: object, value_type: RemoraType) -> object:
    import numpy as np

    if isinstance(value_type, ScalarType):
        dtype = {
            INT.name: np.int32,
            FLOAT.name: np.float32,
            FLOAT64.name: np.float64,
            BOOL.name: np.bool_,
        }.get(value_type.name)
        if dtype is None:
            return np.asarray(value)
        return np.asarray(value, dtype=dtype)
    if isinstance(value_type, ArrayType):
        dtype = {
            INT.name: np.int32,
            FLOAT.name: np.float32,
            FLOAT64.name: np.float64,
            BOOL.name: np.bool_,
        }.get(value_type.element.name)
        if dtype is None:
            return np.asarray(value)
        return np.asarray(value, dtype=dtype)
    return value


def _resolve_syntax(files: list[Path], explicit_syntax: str | None) -> str | None:
    if explicit_syntax:
        return explicit_syntax
    extensions = set()
    for f in files:
        ext = f.suffix
        if ext == ".remora":
            extensions.add("ml")
        elif ext == ".lisp":
            extensions.add("lisp")
        else:
            extensions.add("ml")
    if len(extensions) > 1:
        return None
    return extensions.pop() if extensions else "ml"


def _parse_source(source: str, filename: str, syntax: str):
    if syntax == "lisp":
        return parse_lisp_program(source, filename)
    return parse_program(source, filename)


def _handle_function_call(args: argparse.Namespace) -> int:
    import numpy as np
    from remora.compiler import compile_function_source_to_supported_gpu_artifacts
    from remora.executor import RemoraExecutor
    from remora.types import ArrayType, FLOAT, INT, StaticDim
    from remora.display import format_result

    if args.input is None:
        print("remorac: --call requires at least one --input FILE.npy", file=sys.stderr)
        return 1

    sources: dict[str, str] = {}
    for f in args.files:
        sources[str(f)] = f.read_text(encoding="utf-8")
    source = "\n".join(sources.values())

    syntax = _resolve_syntax(args.files, args.syntax)
    if syntax is None:
        print("remorac: cannot mix .remora and .lisp source files", file=sys.stderr)
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
            source, args.call, param_types, syntax=syntax,
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


def _handle_gpu_target(source: str, syntax: str) -> None:
    from remora.executor import execute_program_on_gpu
    from remora.display import format_result
    from remora.compiler import compile_source

    result = execute_program_on_gpu(source, syntax=syntax)
    artifact = compile_source(source, syntax=syntax)
    rtype = artifact.return_type
    if rtype is None:
        raise CodegenUnavailable("Cannot determine result type")
    print(format_result(result, rtype))


if __name__ == "__main__":
    raise SystemExit(main())
