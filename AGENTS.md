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

# Run the whole suite — CPU tests AND GPU tests (GPU runs when a CUDA runtime is present)
uv run pytest

# Same suite, but tolerate a missing GPU: GPU-unavailable degrades to skip, not failure
# (on a machine that HAS a GPU this still runs the GPU tests)
REMORA_TEST_GPU=0 uv run pytest

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

# Run AD optimization example (interpreter)
uv run remorac --syntax lisp --target interp examples/ad_optimize.lisp

# Run AD optimization example (compiled CPU)
uv run remorac --syntax lisp --target cpu examples/ad_optimize.lisp

# Inspect compiler stages
uv run remorac --emit-ast examples/prelude_sum.remora
uv run remorac --emit-typed-ast examples/prelude_sum.remora
uv run remorac --emit-hir examples/prelude_sum.remora
uv run remorac --emit-mlir examples/prelude_sum.remora

# Start the REPL
uv run remorac --repl
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

Entry points: `remorac` (CLI + REPL, `remora/cli.py`), `remora-bench` (benchmark, `remora/benchmark.py`).

## Key conventions

- **No example-specific code.** Lowering paths must handle arbitrary HIR, not pattern-match specific programs.
- **Static shapes only.** All dimensions come from HIR type annotations.
- **Descriptor ABI** for GPU kernels (aligned pointer + offset + sizes + strides).
- **Prelude auto-prepended.** `stdlib/prelude.rem` is injected by `remora/prelude.py` before compilation.
- **Two syntaxes.** ML syntax is the default; Lisp syntax is selected with `--syntax lisp` (or inferred from `.remora`/`.lisp` extension). The ML grammar is in `remora/grammar.lark` (Lark). The Lisp grammar is inline in `remora/lisp_reader.py`.
- **`_` prefix** on filenames means internal helper module.
- **Lisp syntax local bindings.** Use `let`/`let*` for local bindings (standard Scheme-style). The `::` let-form was removed.

## Testing

- **Framework:** pytest only, no plugins. Config is in `pyproject.toml` (`testpaths = ["tests"]`).
- **GPU is a first-class target. `uv run pytest` exercises the GPU path by default.**
  `tests/conftest.py` defaults `REMORA_TEST_GPU=1` when unset, so locally the GPU path
  gets the *same emphasis as the CPU path*: GPU tests run, and a missing/broken GPU or
  toolchain is a **hard failure** (one clear error at startup), never a silent skip.
  - Run everything (CPU + GPU): `uv run pytest` exercises both paths in one run.
  - Tolerate a missing GPU: `REMORA_TEST_GPU=0 uv run pytest` — GPU-unavailable then
    degrades to a skip instead of a failure. This does **not** disable GPU tests: on a
    machine that has a GPU they still run (there is no CPU-only switch).
  - GPU tests need `iree-compiler` and CUDA (installed by `uv sync` + a working driver).
  - On a GPU machine, GPU tests run regardless of the flag (the runtime simply
    succeeds); the flag only controls what happens when the GPU is *unavailable*
    (fail vs skip). The point of the default is that lost GPU coverage is loud.
- **Acceptance tests** use `tests/acceptance/manifest.json`. Each case declares a `.remora` file, target, expected exit code, and optional stdout/stderr checks. Categories: `supported`, `rejected`, `deferred`.
- **Golden MLIR files** in `tests/golden_mlir/` are used by lowering tests for output comparison.
- CPU compiled tests typically use `CPUFunctionExecutor.compile_source()` or `evaluate_source_compiled()`.

### Coverage rules (learned the hard way)

Silent miscompiles — code that *compiles cleanly but computes the wrong values* — are
the worst failure mode here and have shipped before (e.g. a vector cell-fold and
several view ops inside `map` bodies on GPU). Follow these rules to avoid them:

- **"Compiles" is not "correct".** A test like `assert ".visible .entry" in ptx`
  proves nothing about results. Every GPU op / lowering path needs a **numeric-parity**
  test that runs the kernel and compares against an oracle. Compile-only tests are
  acceptable only as a secondary smoke check, never as the sole coverage for an op.
- **Oracle = the interpreter.** `evaluate_source(...)` (the reference interpreter) is
  the most capable oracle and supports more constructs than the CPU-compiled backend
  (which `deferred`s some ops, e.g. `drop`). Prefer it for parity. Existing GPU-vs-oracle
  harnesses: `TestGPUNumericParity._run_parity` in `tests/test_gpu_general_lowering.py`
  and `tests/test_gpu_numeric_parity.py`. Add new cases there.
- **Sweep element types and shapes.** Cover `f32` *and* `i32` (and `bool` where
  relevant) at non-trivial sizes — not just one f32 example. Real bugs hid in i32-only
  paths (hardcoded `f32` stores / `output_dtype`). Use parametrized tests.
- **Test ops in compound contexts, not just standalone.** An op at the top level and
  the same op *inside a `map`/`fold` body* take different lowering paths (top-level
  linalg vs the general-expr emitter in `remora/_gpu_expr_lowering.py`). Cover both.
- **Prefer end-to-end source compilation over hand-built HIR.** Tests that construct
  HIR directly and call one specific builder bypass the `codegen.py` routing cascade,
  so whole code paths (and crashes within them) go untested. Use `compile_function_source*`
  / `evaluate_source` for coverage; reserve direct-HIR tests for unit-testing one builder.
- **Unsupported constructs must fail loudly, never silently.** When a path can't
  guarantee a correct result, raise `GPUScaffoldError`/`CodegenUnavailable` rather than
  emitting a kernel. Lock this in with a "rejected-not-silent" test (see
  `test_unsupported_view_in_map_rejected_not_silent`).
- **When you add or change a GPU op or lowering path, add a parity test in the same
  change.** Locally `uv run pytest` runs it on the GPU by default; it only proves
  something where a GPU is present (it does on the dev machine). Don't rely on CI for
  GPU coverage — CI sets `REMORA_TEST_GPU=0` and skips it (see CI).

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
| `examples/ad_*.lisp` | AD examples: `ad_polynomial`, `ad_circle`, `ad_spring`, `ad_softmax`, `ad_optimize` |
| `docs/` | Design docs; `PROJECT_OVERVIEW.md` is the best starting point |
| `docs/remora-reference/` | Academic papers on the Remora language |
| `tools/` | MLIR toolchain validation scripts |

## CI

The `python-tests` job in `.github/workflows/ci.yml` runs `uv sync` then `uv run pytest -q`. The rest of the CI workflow is for an unrelated .NET (ILGPU) project and does not apply to this Python codebase.

> **Coverage gap:** CI sets `REMORA_TEST_GPU=0` (`.github/workflows/ci.yml`), so every
> GPU test **skips** in CI — whereas locally it defaults to `1` and GPU runs. GPU
> silent-miscompiles (the worst failure mode — see "Coverage rules") therefore cannot be
> caught by CI today. Until a CUDA-capable CI runner exists, GPU parity tests must be run
> **locally** (the default `uv run pytest` on the GPU dev machine does this) before
> merging any change to a GPU lowering path (`gpu_lowering.py`, `codegen.py`,
> `_gpu_expr_lowering.py`, `_gpu_*`). State in the PR that you did so.
