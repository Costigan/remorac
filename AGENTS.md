# AGENTS.md

## Project

RemoraC: a compiler for the rank-polymorphic Remora array language.
Compiles ML-syntax (default) and Lisp-syntax programs to CPU and GPU via MLIR.
Python 3.11+, managed with `uv` and `hatchling`.

Currently, only the dense core of Remora is implemented, although the
end goal is a full implementation.

## Commands

```bash
# Install / sync deps
uv sync

# Run all tests
uv run pytest

# Run a single test file
uv run pytest tests/test_parser.py

# Run a single test by name
uv run pytest tests/test_execution.py -k test_name

# Compile-check the package (no linter or type checker is configured)
uv run python -m compileall -q remora

# Run a .remora example on compiled CPU
uv run remorac examples/prelude_sum.remora

# Run on the reference interpreter (supports more syntax than CPU backend)
uv run remorac --target interp examples/conditional.remora

# Inspect compiler stages
uv run remorac --emit-ast examples/prelude_sum.remora
uv run remorac --emit-typed-ast examples/prelude_sum.remora
uv run remorac --emit-hir examples/prelude_sum.remora
uv run remorac --emit-mlir examples/prelude_sum.remora

# Start the REPL
uv run remora
```

There is no configured linter, formatter, or Python type checker. Use `uv run python -m compileall -q remora` as the fast verification step after edits.

## Architecture (compilation pipeline)

```
Source (.remora / .lisp)
  → parser.py / lisp_reader.py   → AST (ast_nodes.py)
  → typechecker.py                → Typed AST
  → elaborate.py                  → Core IR
  → hir.py                        → HIR
  → hir_opt.py, defunc.py         → Optimized HIR
  → lowering/ (tensor_ops, module, scalar, view_ops) → MLIR
  → CPU: mlir-opt pipeline → .so → ctypes  (pipeline.py, runtime.py)
  → GPU: LLVM dialect → LLVM IR → PTX      (gpu_lowering.py, codegen.py)
```

Entry points: `remorac` (CLI, `remora/cli.py`), `remora` (REPL, `remora/repl.py`), `remora-bench` (benchmark, `remora/benchmark.py`).

## Key conventions

- **No example-specific code.** Lowering paths must handle arbitrary HIR, not pattern-match specific programs.
- **Static shapes only.** All dimensions come from HIR type annotations.
- **Descriptor ABI** for GPU kernels (aligned pointer + offset + sizes + strides).
- **Prelude auto-prepended.** `stdlib/prelude.rem` is injected by `remora/prelude.py` before compilation.
- **Two syntaxes.** ML syntax is the default; Lisp syntax is selected with `--syntax lisp`. The ML grammar is in `remora/grammar.lark` (Lark). The Lisp grammar is inline in `remora/lisp_reader.py`.
- **`_` prefix** on filenames means internal helper module.

## Testing

- **Framework:** pytest only, no plugins. Config is in `pyproject.toml` (`testpaths = ["tests"]`).
- **GPU tests are opt-in.** Set `REMORA_TEST_GPU=1` to require GPU tests to pass; otherwise they skip. GPU tests need `iree-compiler` and CUDA.
- **Acceptance tests** use `tests/acceptance/manifest.json`. Each case declares a `.remora` file, target, expected exit code, and optional stdout/stderr checks. Categories: `supported`, `rejected`, `deferred`.
- **Golden MLIR files** in `tests/golden_mlir/` are used by lowering tests for output comparison.
- CPU compiled tests typically use `CPUFunctionExecutor.compile_source()` or `evaluate_source_compiled()`.

## External toolchain

The MLIR pipeline (`remora/pipeline.py`) discovers tools at runtime: `mlir-opt`, `mlir-translate`, `iree-opt`, `iree-compile`, `llc`, `ptxas`. These come from `iree-compiler` (Python package) or system installs. If missing, pipeline operations raise `PipelineUnavailable`. You can validate the toolchain with:

```bash
uv run python tools/validate_mlir_toolchain.py
```

## Directory map

| Path | What |
|------|------|
| `remora/` | Main compiler package |
| `remora/lowering/` | HIR → MLIR lowering (tensor ops, scalar, views, module builder) |
| `remora/grammar.lark` | Lark grammar for ML syntax |
| `remora/remora_rt.c` | C runtime linked into compiled CPU programs |
| `tests/` | ~48 test files |
| `tests/acceptance/` | Acceptance test cases + manifest.json |
| `tests/golden_mlir/` | Golden MLIR output files |
| `stdlib/prelude.rem` | Standard library prelude |
| `examples/` | `.remora` examples and Python drivers |
| `docs/` | Design docs; `PROJECT_OVERVIEW.md` is the best starting point |
| `docs/remora-reference/` | Academic papers on the Remora language |
| `tools/` | MLIR toolchain validation scripts |

## CI

The `python-tests` job in `.github/workflows/ci.yml` runs `uv sync` then `uv run pytest -q`. The rest of the CI workflow is for an unrelated .NET (ILGPU) project and does not apply to this Python codebase.
