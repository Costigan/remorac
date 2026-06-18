# Remora Project Overview

*For LLMs and humans starting a new session.  Read this first.*

## What is Remora?

Remora is an array programming language compiler.  It compiles a dense,
statically-shaped subset of the Remora language (both ML and Lisp syntax)
to CPU and GPU executables via MLIR.

**Design principle**: the implementation must handle valid dense Remora
programs, not specific examples.  The GPU backend now handles general
compound-body maps through a recursive expression compiler targeting
MLIR LLVM dialect — no example-specific pattern matching remains.

## Architecture

```
Source (ML or Lisp syntax)
  → Parser (parser.py / lisp_reader.py) → AST (ast_nodes.py)
  → Type Checker (typechecker.py) → Typed AST
  → Elaboration (elaborate.py) → Typed Core IR
  → HIR Lowering (hir.py) → HIR (hir.py)
  → HIR Optimizations (hir_opt.py, defunc.py)
  → MLIR Lowering (lowering/tensor_ops.py, lowering/module.py, ...) → MLIR
  → CPU: mlir-opt → .so → ctypes (pipeline.py, runtime.py)
  → GPU: LLVM dialect descriptor-ABI → LLVM IR → PTX (gpu_lowering.py, codegen.py)
  → Execution: CPUExecutor / CPUFunctionExecutor / RemoraExecutor
```

## Key Directories

| Directory | Purpose |
|-----------|---------|
| `remora/` | Main source package |
| `remora/lowering/` | HIR → MLIR lowering (tensor ops, scalar ops, view ops, module building) |
| `tests/` | ≈886 tests across 43 files |
| `examples/` | Example programs and Python drivers |
| `docs/` | Design documents and plans |
| `stdlib/` | Standard library (prelude.rem) |

## Key Files by Layer

### Frontend
- `remora/parser.py` — ML-syntax parser (Lark grammar: `remora/grammar.lark`)
- `remora/lisp_reader.py` — Lisp-syntax parser (Lark grammar inline)
- `remora/typechecker.py` — Full type checker (3081 lines, largest module)

### HIR
- `remora/hir.py` — HIR datatypes (HIRMap, HIRFold, HIRIndex, HIRLet, etc.)
- `remora/hir_opt.py` — CSE, DCE, duplicate analysis
- `remora/defunc.py` — Defunctionalization pass

### CPU Lowering
- `remora/lowering/tensor_ops.py` — Map/fold/iota lowering.  Contains:
  - `_body_needs_tensor_lowering()` — detects map bodies needing tensor ops
  - `_lower_map_body_with_loops()` — `scf.for`-based general compound-body path
  - `_lower_expression_in_loop()` — recursive expression lowerer inside loops
- `remora/lowering/module.py` — MLIR module building, descriptor ABI
- `remora/lowering/scalar.py` — Scalar region emitter (`_RegionEmitter`)
- `remora/pipeline.py` — MLIR pass pipelines for CPU and GPU

### GPU Lowering
- `remora/codegen.py` — `generate_mlir_descriptor_abi_ptx()` dispatch cascade
- `remora/gpu_lowering.py` — LLVM dialect GPU kernel builders + general lowering
- `remora/_gpu_expr_lowering.py` — GPU expression IR (GpuExpr) and HIR→GPUExpr compiler
- `remora/_gpu_map_support.py` — GPU map analysis and `F32MapKernel`

### Runtime
- `remora/runtime.py` — CPUExecutor, CPUFunctionExecutor, CUDA stubs
- `remora/executor.py` — RemoraExecutor for GPU kernel launch
- `remora/compiler.py` — Public compiler API (`compile_function_source`, etc.)

### Automatic Differentiation
- `remora/ad.py` — Reverse-mode AD via evaluation tape
- `remora/ad_source.py` — Generates Remora gradient source from tape

## Current State

### Working
- **CPU**: General lowering for any valid dense program.  Compound bodies
  (maps containing fold/index/nested-map) lowered via `scf.for` loops.
- **GPU**: General lowering via recursive expression compiler
  (`remora/_gpu_expr_lowering.py`).  Handles folds, index expressions,
  nested maps, conditionals, casts, let bindings, scalar and array-valued
  reductions, element-wise operations, math intrinsics (exp/log/sqrt via
  NVVM), descriptor-level view ops (take/drop/subarray/reverse/rotate/
  transpose), descriptor reinterpretation (reshape/ravel/append/with-shape),
  type-aware i32 arithmetic and comparisons, and array-typed conditionals.
  The general path serves as a universal dispatch fallback for any map
  program.
- **Interpreter**: Handles most programs (used for test validation).
- **AD**: Reverse-mode works for scalar-cost functions.  Tape → source
  compilation.

### Test Counts
- CPU tests: ≈340 (all passing)
- GPU tests: ≈97 (all passing)
- General GPU lowering tests: 31 (all passing, in `tests/test_gpu_general_lowering.py`)
- N-body tests: 5 (all passing, including GPU compilation + numeric parity)

## Conventions

1. **No example-specific code** — any lowering path must handle arbitrary
   HIR, not pattern-match against specific programs.
2. **Static shapes only** — dimensions from HIR type annotations.
3. **Descriptor ABI** — GPU kernels use Remora descriptor structs
   (aligned pointer + offset + sizes + strides) for input/output.
4. **File-naming**: `_` prefix for internal helpers, `lowering/` subdirectory
   for MLIR lowering modules.
5. **Testing**: pytest, `uv run pytest tests/`.  GPU tests need IREE.
   CPU compiled tests use `CPUFunctionExecutor.compile_source()`.
6. **Naming in HIR**: `HIRMap`, `HIRFold`, `HIRReduce`, `HIRIndex`,
   `HIRLet`, `HIRApply`, `HIRPrimOp`, `HIRIf`, `HIRVar`, `HIRLit`,
   `HIRCast`.

## Key Design Documents

| Document | Purpose |
|----------|---------|
| `docs/GPU_GENERAL_LOWERING_PLAN.md` | Plan for general GPU lowering **(COMPLETE)** |
| `docs/DEEPSEEK_CONTINUATION_PLAN.md` | Overall development roadmap |
| `docs/COMPILER_MATURITY_EXAMPLES.md` | What examples compile on which backend |
| `docs/ABI.md` | Descriptor ABI specification |
