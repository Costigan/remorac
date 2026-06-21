# Changelog

All notable changes to RemoraC are documented here, organized by
feature area.  See also the per-phase changelog in the git history.

## GPU Expression Compiler (AD on GPU)

### `ad_optimize.lisp` runs on GPU
- 200-step gradient descent executes entirely on GPU via `LoopPlan`,
  producing the same result as CPU: `[0.512337, 0.433115, 0.911621]`.
- The AD source transform's 32,769-node gradient expression is
  collapsed by CSE before GPU compilation.

### HIRScatterAdd support (`_gpu_expr_lowering.py`)
- Static path: when the target is a `GpuArrayExpr` and the index is a
  compile-time literal, the scatter-add modifies the component directly.
- Dynamic path: when the target is a per-element expression, emits a
  runtime `GpuSelect(coord == index, target + update, target)` using
  the thread coordinate.

### HIRLet input alias propagation
- When `let coeffs = params` and `params` is in `input_map`, the let
  name `coeffs` is added to `input_map` as an alias.  This allows
  `HIRIndex(HIRVar('coeffs'), 0)` to resolve through the alias.
- Save/restore ensures the alias doesn't leak outside the let scope.

### `hir_optimize` bug fix (`hir_opt.py`)
- `hir_optimize` was discarding the CSE bindings returned by `hir_cse`
  — it passed only the rewritten expression (with dangling variable
  references) to DCE.  Fixed to wrap bindings as nested `HIRLet` before
  running DCE.

### CSE on computed map operands (`gpu_lowering.py`)
- When a map/apply operand is a non-trivial expression (not a variable,
  literal, or view op), it is CSE-optimized via `hir_optimize` before
  being wrapped as an `HIRLet` in the lambda body.
- Coordinate env seeded in `_gpu_emit_expr` so `GpuIndexCoordinate`
  resolves during MLIR emission.

### SSA name uniqueness fix
- `%gen_zidx_N` constants replaced with `_fresh_ssa()` counter to
  avoid duplicate SSA definitions when multiple scatter-adds emit
  the same zero-index constant.

### `GpuCompareOp` i64 support
- Integer comparisons (`i32`, `i64`) now emit `llvm.icmp` with
  signed predicates instead of `llvm.fcmp`.

## GPU Performance Kernels

### Tiled Shared-Memory Matmul
- TILE=16 cooperative loading with shared memory tiles.
- Each thread block computes one 16×16 output tile; edge tiles are
  bounds-checked and zero-padded for non-aligned dimensions.
- Falls back to the naive per-thread dot-product if the tiled version
  fails to compile.

### Bitonic Sort and Grade
- **Sort** (N ≤ 1024): O(N log²N) parallel bitonic sort in shared
  memory.  Non-power-of-2 sizes padded with +∞ sentinels.
- **Grade / argsort** (N ≤ 1024): same algorithm with a paired index
  array; outputs i32 permutation indices.
- **Multi-block sort** (N > 1024): per-block local sort followed by
  global compare-swap merge steps orchestrated via `ExecutionPlan`
  with double-buffered global memory.  Odd blocks reverse output to
  form bitonic sequences at boundaries.  Supports up to ~1M elements.
- **Multi-block grade** (N > 1024): per-block local grade followed by
  global merge steps that compare `values[idx]` and swap i32 indices.
  Same odd-block reversal for correct bitonic merge.

### Parallel Scatter-Add
- Single-block kernel (N ≤ 1024): all threads copy target→output in
  parallel, then thread 0 performs the scalar add after a barrier.
- Falls back to serial for N > 1024.

### Multi-Block Parallel Scan
- Four-kernel plan for N > 1024: per-block Hillis-Steele scan →
  extract block sums → scan block sums → propagate prefixes.
- Supports up to 1,048,576 elements (1024 blocks × 1024 threads).
- Single-block Hillis-Steele scan unchanged for N ≤ 1024.

### Parallel Filter
- Three-kernel `ExecutionPlan`: predicate evaluation → i32 prefix
  sum (Hillis-Steele) → scatter-write.
- Replaces the serial single-thread kernel for N ≤ 1024.

### Parallel Replicate
- Two-kernel plan: prefix sum on counts → scatter-replicate.
- Each thread writes `counts[i]` copies of `values[i]` to computed
  output positions.
- Replaces the serial single-thread kernel for N ≤ 1024.

## Multi-Kernel Orchestration

### ExecutionPlan Infrastructure (`remora/execution_plan.py`)
- `BufferSpec`: named device buffers with shape, dtype, optional
  `init` data for pre-populating from host.
- `KernelStep`: single GPU kernel launch with named buffer refs.
- `LoopPlan`: host-side iteration with buffer swapping (double-buffer
  pattern for optimization loops).
- `ExecutionPlan`: ordered sequence of steps with validation.

### RemoraExecutor Extensions (`remora/executor.py`)
- `execute_plan(plan, inputs)`: runs multi-step plans with named
  device buffers, kernel launches, host-side loops, and buffer
  swapping.
- `add_module(ptx, kernels)`: load kernels from additional PTX
  modules.
- `KernelMeta.grid_size`: explicit grid size override for 2D tiling.

### State-Fold GPU Detection (`remora/codegen.py`)
- `try_compile_state_fold_gpu(program)`: detects
  `fold body init (iota N)` where the body ignores the step variable,
  compiles the body as a GPU map kernel, and emits a `LoopPlan` with
  N iterations and buffer swapping.
- Wired into `execute_program_on_gpu()` as a first-pass attempt
  before the IREE pipeline.

## GPU Codegen Improvements

### HIRApply Support
- `build_descriptor_abi_general_map_gpu_module` now accepts `HIRApply`
  (the generalized rank-polymorphic application node) alongside
  `HIRMap`.
- Codegen fallback also accepts `HIRApply` with any callable type
  (not just `HIRLambda`).

### ArrayType Import Fix
- `ArrayType` imported at module level in `codegen.py` to avoid
  `UnboundLocalError` from Python's scoping rules when local imports
  inside try blocks shadowed the name.

### Tiled Matmul i1/i64 Fix
- `llvm.and` of `llvm.icmp` results uses `: i1` (not `: i64`) in
  the tiled matmul bounds-checking logic.

## Python Integration (Phases 1-3)

### Phase 1 — Core Wrapper
- `RemoraFunction` callable wrapper with JIT rank/shape checking.
- `compile_function()`, `compile_all()`, and `remora.define()` APIs.
- `RemoraRankMismatchError` for clear boundary violations.

### Phase 2 — Jupyter Cell Magic
- `%%remora` cell magic with `--target cpu|interp|gpu`, `--syntax`,
  `--out`, `--types` flags.
- Multi-definition cells via `compile_all()`.
- Error reporting: compiler errors propagate to notebook output.

### Phase 3 — Developer Experience
- `remora.define(source)`: returns `RemoraFunction` (single def) or
  `dict` (multiple defs).
- `%remora_eval` line magic: persistent REPL session in IPython with
  `--target`, `--syntax`, `--reset` flags.
- `%%remora --types`: inline type display for compiled definitions.

## AD and Optimization

- Five AD example files: `ad_polynomial`, `ad_circle`, `ad_spring`,
  `ad_softmax`, `ad_optimize`.
- `ad_optimize.lisp`: 200-step gradient descent on polynomial
  curve-fitting loss; works on interpreter, compiled CPU, and GPU
  (result: `[0.512337, 0.433115, 0.911621]`).
- Grad-lifting pass (`_rewrite_applied_source_gradient`):
  `(grad f)` resolved at any depth via `_collect_typed_grads`.
- State fold lowering (`_lower_state_fold_result`): `scf.for` with
  scalar decomposition for array-valued fold accumulators.

## Lisp Syntax

- `::` let-form replaced entirely with standard `let`/`let*`
  (Scheme-style).  Grammar, transformer, all tests and docs updated.

## Test Infrastructure

- Pre-existing test failures fixed: restored `docs/DENSE_CORE.md`,
  `docs/ABI.md`, `docs/IMPLEMENTATION_NOTES.md`, and
  `docs/BENCHMARK_BASELINES.json` from `docs/old/`.
- GPU integration tests (`tests/test_gpu_integration.py`): 13 tests
  verified on NVIDIA RTX 5090 (compute capability 12.0) covering
  map, sort, grade, matmul, reduction, AD gradient descent, and
  multi-block grade.
- Execution plan unit tests (`tests/test_execution_plan.py`):
  19 tests for plan construction, validation, and state-fold
  detection.
- Python integration tests (`tests/test_api.py`, `tests/test_magics.py`):
  27 tests for `RemoraFunction`, codec, `define()`, cell magic, and
  REPL integration.
- Total: 943 non-GPU + 13 GPU tests passing, 0 xfails, 0 regressions.
