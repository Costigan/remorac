# Remora Project Overview

*For LLMs and humans starting a new session. Read this first.*

## What is Remora?

Remora is an array programming language compiler. It compiles a dense,
statically-shaped subset of the Remora language (both ML and Lisp syntax)
to CPU and GPU executables via MLIR.

**Design principle:** the implementation must handle valid dense Remora
programs, not specific examples. The compiled-CPU path now covers every
construct the dense-core typechecker accepts — closure conversion,
ForallType HOF monomorphization, recursion, and all text-path lowering
are complete.

**24 kLOC** across 50+ source modules, **1,055 tests** in 50 test files.

## Language Subset Implemented (Dense Core)

The dense core covers:

| Feature                                                              | CPU | Interpreter |     GPU     |
| -------------------------------------------------------------------- | :-: | :---------: | :---------: |
| Scalar arithmetic (+, -, \*, /)                                      |  ✓  |      ✓      |      ✓      |
| Arrays (rank 1–10, Int, Float, Float64, Bool)                        |  ✓  |      ✓      |      ✓      |
| `let` bindings (scalar and array)                                    |  ✓  |      ✓      |      ✓      |
| `if` / `select` conditionals                                         |  ✓  |      ✓      |      ✓      |
| `map` (unary, binary, operator sections)                             |  ✓  |      ✓      |      ✓      |
| `fold` / `reduce` / `fold-right`                                     |  ✓  |      ✓      |      ✓      |
| `scan` (inclusive, exclusive, left, right)                           |  ✓  |      ✓      |   limited   |
| `trace` / `trace-right`                                              |  ✓  |      ✓      |   limited   |
| `lambda` expressions                                                 |  ✓  |      ✓      |      ✓      |
| `define` (plain, `define/pi`, `define/forall`)                       |  ✓  |      ✓      |   limited   |
| Function values as arguments                                         |  ✓  |      ✓      |   limited   |
| Closure capture (lambdas with free vars)                             |  ✓  |      ✓      |      ✗      |
| Operator sections (`(* 2)`, `(+ 1)`)                                 |  ✓  |      ✓      |      ✓      |
| Recursion (self, mutual, tail, non-tail)                             |  ✓  |      ✓      | tail helper |
| `rerank` (`~(r1 r2)`)                                                |  ✓  |      ✓      |      ✓      |
| Views (index, slice, transpose, reshape, ravel, reverse, take, drop) |  ✓  |      ✓      |      ✓      |
| `im2col` / `col2im`                                                  |  ✓  |      ✓      |      ✗      |
| `pair` / `first` / `second`                                          |  ✓  |      ✓      |      ✗      |
| `box` / `unbox` (type erasure only)                                  |  ✓  |      ✓      |      ✗      |
| `sort` / `grade` (f32/i32)                                           |  ✓  |      ✓      |  f32 only   |
| `append`, `rotate`, `subarray`, `indices-of`, `with-shape`           |  ✓  |      ✓      |   limited   |
| `matmul`                                                             |  ✓  |      ✓      |  f32 only   |
| `compose`                                                            |  ✓  |      ✓      |      ✗      |
| Automatic differentiation (`grad`)                                   |  ✓  |      ✓      |      ✗      |
| Rank limit                                                           | 10  |     10      |     10      |
| Dynamic shapes                                                       |  ✗  |      ✗      |      ✗      |

**Legend:** ✓ = complete, limited = partial/subset supported, ✗ = not yet supported.

## Architecture

```
Source (ML or Lisp syntax)
  → Parser (parser.py / lisp_reader.py)              → AST (ast_nodes.py: 61 node classes)
  → Type Checker (typechecker.py: 3,459 lines)      → Typed AST + constraints
  → Elaboration (elaborate.py, frame.py)             → Elaborated Core IR (elaborated.py)
  → Erasure (erase.py, constraints.py, index.py)     → Core IR
  → HIR Lowering (hir.py: 1,066 lines, 47 node types) → HIR
  → HIR Optimizations (hir_opt.py, defunc.py, ad_hir.py, ad_opt.py)
  → MLIR Lowering (lowering/: 11 modules)            → MLIR
  → Codegen dispatch (pipeline.py, codegen.py)
      CPU:  mlir-opt → LLVM IR → llc → gcc → a.out (executable) or a.so (shared)
      GPU:  LLVM dialect descriptor-ABI → LLVM IR → PTX (gpu_lowering.py: 6,505 lines, codegen.py)
  → Execution: a.out subprocess / RemoraFunction API / CPUExecutor / RemoraExecutor
```

### Pipeline stages in detail

| Stage        | Input           | Output           | Key modules                                                          |
| ------------ | --------------- | ---------------- | -------------------------------------------------------------------- |
| Parse        | Source text     | AST              | `parser.py`, `lisp_reader.py`                                        |
| Type check   | AST             | Typed AST        | `typechecker.py`, `types.py`, `constraints.py`, `dependent_types.py` |
| Elaborate    | Typed AST       | Elaborated Core  | `elaborate.py`, `frame.py`, `index.py`                               |
| Core verify  | Elaborated Core | (validated)      | `core_verify.py`                                                     |
| Erase        | Elaborated Core | Backend Core     | `erase.py`                                                           |
| HIR          | Backend Core    | HIR              | `hir.py`                                                             |
| HIR optimize | HIR             | Optimized HIR    | `hir_opt.py`, `defunc.py`, `ad_hir.py`, `ad_opt.py`                  |
| MLIR lower   | HIR             | MLIR module      | `lowering/` package                                                  |
| Codegen      | MLIR            | Native executable / PTX | `pipeline.py`, `gpu_lowering.py`, `codegen.py`                       |
| Execute      | Native artifact | Numeric result   | `runtime.py`, `executor.py`, `compiler.py`, `api.py`                 |

## Key Directories

| Directory            | Purpose                                                                                |
| -------------------- | -------------------------------------------------------------------------------------- |
| `remora/`            | Main source package (50+ modules)                                                      |
| `remora/lowering/`   | HIR → MLIR lowering: tensor ops, scalar, views, indexing, module building (11 modules) |
| `remora/jupyter/`    | IPython/Jupyter magics for interactive use                                             |
| `tests/`             | 1,055 tests across 50 files, plus `tests/golden_mlir/`                                 |
| `tests/acceptance/`  | Acceptance test cases with manifest.json                                               |
| `tests/golden_mlir/` | Golden MLIR output files for lowering tests                                            |
| `examples/`          | 30 `.remora` + 12 `.lisp` example programs and Python drivers                          |
| `docs/`              | Design documents and plans (18 markdown files)                                         |
| `stdlib/`            | Standard library (`prelude.rem`: 17 functions)                                         |
| `tools/`             | Toolchain validation scripts                                                           |

## Key Files by Layer

### Frontend — Parsing

| File                    | Lines | Purpose                                                                    |
| ----------------------- | ----- | -------------------------------------------------------------------------- |
| `remora/parser.py`      | 288   | ML-syntax parser (Lark, grammar in `remora/grammar.lark`)                  |
| `remora/lisp_reader.py` | 767   | Lisp-syntax parser (Lark grammar inline), maps s-expressions to same AST   |
| `remora/operators.py`   | —     | Centralized operator metadata: `+ - * / exp log sqrt && \|\| < <= > == !=` |

### Frontend — AST

| File                  | Lines | Purpose                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                |
| --------------------- | ----- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `remora/ast_nodes.py` | —     | 61 AST node classes: `LetExpr`, `IfExpr`, `LambdaExpr`, `AppExpr`, `MapExpr`, `FoldExpr`, `FoldRightExpr`, `ReduceExpr`, `ScanExpr`, `TraceExpr`, `IotaExpr`, `IotaNExpr`, `Iota1Expr`, `PairExpr`, `FirstExpr`, `SecondExpr`, `BoxExpr`, `UnboxExpr`, `BoxesExpr`, `SortExpr`, `GradeExpr`, `GradExpr`, `Im2colExpr`, `Col2imExpr`, `MatmulExpr`, `FilterExpr`, `ReplicateExpr`, `ScatterAddExpr`, `ComposeExpr`, `ReverseExpr`, `TransposeExpr`, `ReshapeExpr`, `RavelExpr`, `TakeExpr`, `DropExpr`, `ShapeExpr`, `RankExpr`, `LengthExpr`, `IndexExpr`, `IndexAppExpr`, `SelectExpr`, `AppendExpr`, `RotateExpr`, `RerankExpr`, `SliceRange`, `LeftSectionExpr`, `RightSectionExpr`, `OperatorFuncExpr`, `VarExpr`, lit nodes, program, definitions |

### Frontend — Type System

| File                        | Lines | Purpose                                                                                                                         |
| --------------------------- | ----- | ------------------------------------------------------------------------------------------------------------------------------- |
| `remora/typechecker.py`     | 3,459 | Full type checker (largest module), bidirectional, handles ForallType, dependent types, rank polymorphism, HOF monomorphization |
| `remora/types.py`           | —     | Static type definitions: scalars, arrays, function types, ForallType                                                            |
| `remora/constraints.py`     | —     | Constraint representation and restricted solvers for dependent indices                                                          |
| `remora/dependent_types.py` | —     | Helpers for dependent Remora types: type/index substitution, normalization                                                      |
| `remora/index.py`           | —     | Compile-time index/dimension/shape expressions for dependent shapes                                                             |

### Core IR — Elaboration & Erasure

| File                    | Lines | Purpose                                                                  |
| ----------------------- | ----- | ------------------------------------------------------------------------ |
| `remora/elaborate.py`   | 181   | Converts typed AST to elaborated core IR (rank-polymorphic lowering)     |
| `remora/elaborated.py`  | —     | Typed elaborated core IR datatypes between source typing and backend HIR |
| `remora/frame.py`       | —     | Shared frame/cell decomposition for rank-polymorphic elaboration         |
| `remora/erase.py`       | —     | Erases dependent typed-core programs down to backend HIR                 |
| `remora/core_verify.py` | —     | Structural verifier for the elaborated core IR                           |

### HIR — Intermediate Representation

| File                       | Lines | Purpose                                                                                                                                                                                                                        |
| -------------------------- | ----- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `remora/hir.py`            | 1,066 | 47 HIR node types: `HIRMap`, `HIRFold`, `HIRReduce`, `HIRIndex`, `HIRLet`, `HIRApply`, `HIRPrimOp`, `HIRIf`, `HIRVar`, `HIRLit`, `HIRCast`, `HIRScan`, `HIRTrace`, `HIRSort`, `HIRMatmul`, `HIRPair`, `HIRBox`, view ops, etc. |
| `remora/hir_opt.py`        | 900   | CSE (common subexpression elimination), DCE (dead code elimination), duplicate analysis                                                                                                                                        |
| `remora/defunc.py`         | 458   | Defunctionalization pass: lowers higher-order functions to first-order                                                                                                                                                         |
| `remora/hir_dispatch.py`   | —     | Shared HIR expression dispatch table utilities                                                                                                                                                                                 |
| `remora/execution_plan.py` | —     | Multi-kernel GPU execution plans: buffer specs, kernel steps, host loops                                                                                                                                                       |

### Automatic Differentiation

| File                   | Lines | Purpose                                                                        |
| ---------------------- | ----- | ------------------------------------------------------------------------------ |
| `remora/ad.py`         | 674   | Reverse-mode AD: builds evaluation tape from `_Expr` IR, propagates cotangents |
| `remora/ad_source.py`  | 1,208 | Generates Remora gradient source from AD tape                                  |
| `remora/ad_hir.py`     | —     | Validates AD tape gradient computations are expressible/compilable as HIR      |
| `remora/ad_opt.py`     | —     | Bottom-up simplification of AD `_Expr` trees before source emission            |
| `remora/ad_testing.py` | —     | Central finite-difference utilities for testing AD gradient correctness        |

### CPU Lowering

| File                                  | Lines | Purpose                                                                                                                |
| ------------------------------------- | ----- | ---------------------------------------------------------------------------------------------------------------------- |
| `remora/lowering/tensor_ops.py`       | —     | Map/fold/iota lowering. `_body_needs_tensor_lowering()`, `_lower_map_body_with_loops()`, `_lower_expression_in_loop()` |
| `remora/lowering/module.py`           | —     | MLIR module building, descriptor ABI, entry point generation                                                           |
| `remora/lowering/scalar.py`           | —     | Scalar region emitter (`_RegionEmitter`): arithmetic, comparisons, conditionals                                        |
| `remora/lowering/view_ops.py`         | —     | View operation lowering: transpose, reshape, slice, reverse, take/drop                                                 |
| `remora/lowering/indexing.py`         | —     | Index expression lowering                                                                                              |
| `remora/lowering/types.py`            | —     | MLIR type conversion: Remora types → MLIR types                                                                        |
| `remora/lowering/_builder_emitter.py` | —     | Builder-based emitter (auxiliary path, largely superseded by text path)                                                |
| `remora/lowering/_builder_ops.py`     | —     | Builder-based operation factories                                                                                      |
| `remora/lowering/scalar_builder.py`   | —     | Builder-based scalar region emitter                                                                                    |
| `remora/lowering/_gpu_builder.py`     | —     | GPU-specific builder utilities                                                                                         |
| `remora/pipeline.py`                  | 701   | MLIR pass pipelines for CPU (`mlir-opt`) and GPU; toolchain discovery                                                  |

### GPU Lowering

| File                           | Lines | Purpose                                                                                                                                                                                                                                                                                                                                                                            |
| ------------------------------ | ----- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `remora/gpu_lowering.py`       | 6,505 | LLVM dialect GPU kernel builders + general lowering via recursive expression compiler. Handles folds, index expressions, nested maps, conditionals, casts, let bindings, math intrinsics (exp/log/sqrt via NVVM), view ops, descriptor reinterpretation, type-aware i32 ops, array-typed conditionals, grad-lifting, state fold lowering, monomorphization, tail-recursive helpers |
| `remora/codegen.py`            | 1,343 | `generate_mlir_descriptor_abi_ptx()` dispatch cascade: routes HIR programs to appropriate GPU lowering path                                                                                                                                                                                                                                                                        |
| `remora/_gpu_expr_lowering.py` | 1,410 | GPU expression IR (`GpuExpr`) and HIR→GPUExpr compiler for map-body expressions                                                                                                                                                                                                                                                                                                    |
| `remora/_gpu_map_support.py`   | 459   | GPU map analysis and `F32MapKernel` fast path for homogeneous element-wise maps                                                                                                                                                                                                                                                                                                    |
| `remora/_gpu_radix_sort.py`    | —     | GPU 256-bin LSD radix sort for f32 arrays, emitted as an `ExecutionPlan`                                                                                                                                                                                                                                                                                                           |
| `remora/abi.py`                | —     | ctypes definitions for the Dense Core external ABI (memref descriptor layout)                                                                                                                                                                                                                                                                                                      |

### Runtime & Public API

| File                 | Lines | Purpose                                                                                                  |
| -------------------- | ----- | -------------------------------------------------------------------------------------------------------- |
| `remora/runtime.py`  | 2,310 | `CPUExecutor`: compiled .so/.out loader; `CPUFunctionExecutor`; standalone executable compilation with auto-generated C stub; CUDA stubs |
| `remora/executor.py` | 801   | `RemoraExecutor`: GPU kernel launch, `DeviceArray`, `_DeviceBuffer`, memory pool, kernel chaining        |
| `remora/compiler.py` | 1,535 | Public compiler API: `compile_function_source()`, `compile_gradient_function_source()`, monomorphization |
| `remora/api.py`      | —     | `RemoraFunction` and `compile_function()`: call compiled Remora from NumPy                               |
| output metadata JSON | —     | Sidecar rebuild metadata (`a.json` / `<output>.json`) written alongside CLI artifacts |
| `remora/limits.py`   | —     | Shared Dense Core implementation limits (`MAX_DENSE_RANK = 10`)                                          |
| `remora/lowering.py` | —     | Compatibility shim re-exporting the `remora.lowering` package                                            |

### Entry Points & Tooling

| File                        | Lines | Purpose                                                       |
| --------------------------- | ----- | ------------------------------------------------------------- |
| `remora/cli.py` | — | CLI entry point (`remorac` command: compile to `a.out`, `--repl`, `-o`, `--shared`, `--compile-only`, `--cleanup`, multi-file, syntax inference) |
| `remora/repl.py`            | —     | CPU-first interactive REPL                                    |
| `remora/benchmark.py`       | —     | Small benchmark harness for the Dense Core compiler/pipeline  |
| `remora/benchmark_suite.py` | —     | Benchmark suite: Remora vs NumPy vs JAX execution performance |
| `remora/display.py`         | —     | Result formatting/display helpers for scalars and arrays      |
| `remora/errors.py`          | —     | Base `RemoraError` exception class                            |
| `remora/prelude.py`         | —     | Prelude source helpers; auto-prepends `stdlib/prelude.rem`    |

### Jupyter

| File                       | Purpose                                                  |
| -------------------------- | -------------------------------------------------------- |
| `remora/jupyter/magics.py` | IPython/Jupyter cell magics for inline Remora evaluation |

## Current State

### CPU

Complete lowering for every dense-core construct. Closure conversion, ForallType
HOF monomorphization, recursion (self/mutual, tail/non-tail), operator sections,
exclusive/right scans, binary/cell maps, and fold sections all compile. Single
text-based lowering path (builder API disabled as slower and less capable).

### GPU

General lowering via recursive expression compiler (`remora/_gpu_expr_lowering.py`,
1,410 lines). Handles folds, index expressions, nested maps, conditionals, casts,
let bindings, scalar and array-valued reductions, element-wise operations, math
intrinsics (exp/log/sqrt via NVVM), descriptor-level view ops (take/drop/subarray/
reverse/rotate/transpose), descriptor reinterpretation (reshape/ravel/append/
with-shape), type-aware i32 arithmetic and comparisons, and array-typed conditionals.
Includes grad-lifting pass, state fold lowering, monomorphization, helper inlining,
and scalar self-tail-recursive helper loops inside map bodies (Float, Int, Bool).
The general path serves as a universal dispatch fallback for any map program.
GPU radix sort (`_gpu_radix_sort.py`) provides f32 sorting via 256-bin LSD algorithm.

### Interpreter

Handles dense-core programs and serves as the semantic oracle. Tail recursion is
trampolined; tested deep non-tail recursion runs beyond Python's default recursion
depth. Supports constructs not yet on GPU (boxes, pairs, closures, im2col/col2im,
dynamic shapes).

### AD

Reverse-mode AD for scalar-cost functions. Tape → source compilation. AD works
through compiled CPU path for optimization loops (grad-lifting + state fold); see
`examples/ad_optimize.lisp`. AD HIR verification (`ad_hir.py`) and optimization
(`ad_opt.py`) ensure generated gradients are valid and efficient.

### Code size and tests

| Metric             | Count                                          |
| ------------------ | ---------------------------------------------- |
| Source modules     | 50+                                            |
| Total lines        | ~24,000                                        |
| Largest module     | `gpu_lowering.py` (6,505 lines)                |
| Test files         | 50                                             |
| Test functions     | 1,055                                          |
| GPU test files     | 6                                              |
| GPU lowering tests | 120+ (in `tests/test_gpu_general_lowering.py`) |
| N-body tests       | 5 (GPU compilation + numeric parity)           |
| Example programs   | 30 `.remora` + 12 `.lisp`                      |

## Testing Approach

**Framework:** pytest only, no plugins. Config in `pyproject.toml` (`testpaths = ["tests"]`).

**GPU is a first-class target.** `uv run pytest` exercises the GPU path by default.
`tests/conftest.py` defaults `REMORA_TEST_GPU=1`. On a GPU machine, GPU tests run
regardless of the flag. The flag only controls behavior when GPU is *unavailable*:
fail (default) vs skip (`REMORA_TEST_GPU=0`).

```bash
uv run pytest                           # CPU + GPU (default)
REMORA_TEST_GPU=0 uv run pytest         # tolerate missing GPU
uv run pytest tests/test_parser.py      # single file
uv run pytest tests/test_execution.py -k test_name  # single test
```

**Acceptance tests** use `tests/acceptance/manifest.json`. Each case declares a
`.remora` file, target, expected exit code, and optional stdout/stderr checks.
Categories: `supported`, `rejected`, `deferred`.

**Golden MLIR files** in `tests/golden_mlir/` are used by lowering tests for
output comparison.

**CPU compiled tests** typically use `CPUFunctionExecutor.compile_source()` or
`evaluate_source_compiled()`.

### Coverage rules

Silent miscompiles — code that *compiles cleanly but computes the wrong values* —
are the worst failure mode. Follow these rules:

- **"Compiles" is not "correct".** Every GPU op / lowering path needs a
  **numeric-parity** test that runs the kernel and compares against an oracle.
  Compile-only tests are acceptable only as a secondary smoke check.
- **Oracle = the interpreter.** `evaluate_source(...)` (reference interpreter) is
  the most capable oracle. Prefer it for parity. Existing GPU-vs-oracle harnesses:
  `TestGPUNumericParity._run_parity` in `tests/test_gpu_general_lowering.py` and
  `tests/test_gpu_numeric_parity.py`.
- **Sweep element types and shapes.** Cover `f32` *and* `i32` (and `bool` where
  relevant) at non-trivial sizes. Use parametrized tests.
- **Test ops in compound contexts, not just standalone.** An op at top level and
  the same op *inside a `map`/`fold` body* take different lowering paths. Cover both.
- **Prefer end-to-end source compilation over hand-built HIR.** Tests that construct
  HIR directly bypass the `codegen.py` routing cascade. Use
  `compile_function_source*` / `evaluate_source` for coverage; reserve direct-HIR
  tests for unit-testing one builder.
- **Unsupported constructs must fail loudly.** When a path can't guarantee a correct
  result, raise `GPUScaffoldError`/`CodegenUnavailable`. Lock with a
  "rejected-not-silent" test.
- **When adding/changing a GPU op, add a parity test in the same change.** Locally
  `uv run pytest` runs on GPU by default. Don't rely on CI for GPU coverage — CI
  sets `REMORA_TEST_GPU=0` and skips it.

## Conventions

1. **AOT compilation.** `remorac` compiles Remora source to a standalone
   ELF executable (`a.out`) that can be run independently. `--shared` produces
   a shared library for embedding. The default executable is built by
   auto-generating a C `main()` stub (from the return type), linking it with
   the Remora object code and `remora_rt.o`, and producing a PIE binary.
1. **No example-specific code.** Lowering paths must handle arbitrary HIR, not
   pattern-match against specific programs.
1. **Static shapes only.** All dimensions come from HIR type annotations.
1. **Descriptor ABI.** GPU and CPU kernels use Remora descriptor structs
   (aligned pointer + offset + sizes + strides) for input/output.  Normative
   specification in [`docs/ABI.md`](ABI.md); ctypes implementation in
   `remora/abi.py`.  LLMs working on lowering code should read ABI.md first.
1. **File naming.** `_` prefix for internal helper modules.
1. **Naming in HIR.** `HIRMap`, `HIRFold`, `HIRReduce`, `HIRScan`, `HIRTrace`,
   `HIRIndex`, `HIRLet`, `HIRApply`, `HIRPrimOp`, `HIRIf`, `HIRVar`, `HIRLit`,
   `HIRCast`, `HIRPair`, `HIRBox`, `HIRSort`, `HIRMatmul`, view ops.
1. **Prelude auto-prepended.** `stdlib/prelude.rem` is injected by
   `remora/prelude.py` before compilation. Contains 17 function definitions
   (`add`, `sub`, `mul`, `div`, `neg`, `id`, `const`, `sum`, `product`, `scale`,
   `dot`, `max`, `min`, `abs`, `any`, `all`).
1. **Two syntaxes, one AST.** ML syntax (default, `--syntax ml`) and Lisp syntax
   (`--syntax lisp`) both produce the same AST; all backends work with either.
1. **Verification step.** `uv run python -m compileall -q remora` for fast
   compile-check after edits (no linter, formatter, or type checker configured).

## Key Design Documents

| Document                                    | Purpose                                                        |
| ------------------------------------------- | -------------------------------------------------------------- |
| `docs/USER_GUIDE.md`                        | User-facing language reference with syntax tables and examples |
| `docs/ABI.md`                               | Descriptor ABI specification for GPU kernels                   |
| `docs/DENSE_CORE.md`                        | Dense Core language specification                              |
| `docs/IMPLEMENT_RECURSION.md`               | Recursion implementation plan and status                       |
| `docs/IMPLEMENTATION_NOTES.md`              | Implementation notes and decisions                             |
| `docs/MLIR_IMPLEMENTATION_PLAN.md`          | MLIR lowering plan                                             |
| `docs/DYNAMIC_SHAPES_AND_BOXES_PLAN.md`     | Plan for dynamic shapes and boxes                              |
| `docs/BENCHMARK_PLAN.md`                    | Benchmark design and goals                                     |
| `docs/BENCHMARK_IMPROVEMENT_PLAN.md`        | Benchmark optimization plan                                    |
| `docs/BENCHMARK_SESSION_PROMPT.md`          | Benchmark session reference                                    |
| `docs/FUTURE_WORK.md`                       | Future work and planned features                               |
| `docs/HEAT1D_PLAN.md`                       | Heat equation 1D plan                                          |
| `docs/ANCHOR_FREE_CRATER_DETECTION_PLAN.md` | Crater detection plan                                          |
| `docs/PYTHON_INTEGRATION_PLAN.md`           | Python/NumPy integration plan                                  |
| `docs/HOW_TO_RUN.md`                        | Quick-start execution guide                                    |
| `docs/UPDATE_CI.md`                         | CI update reference                                            |
| `docs/remorac-vs-futhark.md`                | Comparison with Futhark                                        |
