# Remora Dense Core: Complete Implementation Plan

_Generated 2026-06-06 from evaluation and implementation notes._

This document integrates findings from `deepseek-evaluation.md` and
`IMPLEMENTATION_NOTES.md` into a concrete, sequenced plan to reach a complete
Dense Core implementation that runs on CPU alone, or CPU and GPU. It builds on
the existing milestone roadmap in `IMPLEMENTATION_PLAN_UPDATE.md`.

---

## 1. End-State Definition: "Complete Dense Core"

A complete Dense Core implementation must satisfy the contract in
`docs/DENSE_CORE.md` across two backends:

### 1.1 CPU Backend (Complete)

| Capability | Current | Target |
|---|---|---|
| Scalar arithmetic, bool ops, casts | Done | Done |
| Array literals rank 0–10 | Done | Done |
| `iota`, `shape`, `rank` | Done | Done |
| Unary/binary scalar-cell `map` rank 0–10 | Done | Done |
| Scalar `fold` rank 1–10 | Done | Done |
| Array-cell `fold` rank 1–10 | Partial (primitive callables only) | General static callables |
| Cell `map` (body = fold over 1 cell dim) | Done | General cell maps |
| `let` (scalar + tensor) | Scalar SSA; tensor inlined | Full tensor SSA |
| `if` over scalar booleans | Done | Done |
| Full-rank indexing, partial indexing | Done (literal indices) | Dynamic index expressions |
| View ops: `transpose`, `slice`, `reshape`, `ravel`, `take`, `drop`, `reverse` | Done | Done |
| Top-level function definitions | Call-site specialization only | General top-level function types |
| Prelude (`add`, `sub`, `mul`, `div`, `sum`, `product`, `scale`, `dot`) | Done | Done |
| Multicore threading (`--cpu-threads`) | Experimental (maps, reductions, row-reductions) | Broad nested tensor coverage |
| Vectorization (`--cpu-vectorize`) | Experimental | Stable, non-experimental |
| Buffer reuse / arena allocation | Foundation exists | Integrated into all pipelines |
| Benchmark harness | Done | Done |

### 1.2 GPU Backend (Complete)

| Capability | Current | Target |
|---|---|---|
| Unary/binary `map` f32/i32/bool rank 1–10 | Done (descriptor ABI) | Done |
| Scalar reductions f32 rank 1 | Done (parallel) | i32, bool, rank 2+ |
| Dot-shaped reductions f32 | Done | i32 |
| Strided/non-contiguous views | Minimal | Full descriptor stride support |
| Whole-program GPU lowering | None | Full program → `gpu.module` |
| GPU REPL (`:target gpu-nvidia`) | Rejected | Functional with input binding |
| GPU CLI (`remorac --target gpu-nvidia`) | None | Functional with `.npy` inputs |
| GPU multi-kernel programs | None | Supported |
| GPU bool ABI (byte-backed) | Done | Done |
| GPU int32 ops beyond maps | None | Reductions, fold |
| Direct hand-written PTX | Legacy fallback | Removed, replaced by MLIR path |
| `ptxas` validation | Available when installed | Integrated into test path |

### 1.3 Cross-Cutting

| Capability | Current | Target |
|---|---|---|
| Text-based MLIR generation | Primary path | Replaced with builder API |
| `lowering.py` god module (2,286 lines) | Monolithic | Split into focused modules |
| Operator dispatch duplication | 11 copies | Single shared dispatch table |
| Docstrings | Sparse (19 functions) | Comprehensive |
| Python CI | None | GitHub Actions running full suite |
| GPU CI gates | None | GPU tests gated on `REMORA_TEST_GPU=1` |

---

## 2. Gap Analysis: What Stands Between Current State and "Complete"

### 2.1 CPU Gaps

1. **General array-cell `fold` callables.** Currently `fold` over array cells
   only accepts primitive callables (`+`, `*`, etc.) and inlined lifted lambdas.
   General static callables (named functions, operator sections) over array
   cells must be supported.

2. **General cell `map` beyond rank-1-cell fold bodies.** Cell maps currently
   only support bodies that are folds over a single cell dimension. General
   cell maps whose body is an arbitrary scalar-producing expression must be
   lowered.

3. **Full tensor SSA environment for `let`.** Tensor `let` values are currently
   inlined before MLIR emission. A proper tensor SSA environment (which already
   exists for scalars) must be extended to tensors to avoid code duplication
   and enable correct bufferization for large programs.

4. **General top-level function types.** Function definitions are specialized
   only at direct call sites. General top-level function type inference and
   MLIR function emission (with proper parameter/return types beyond scalar
   callables) is needed for reusable library code.

5. **Dynamic index expression lowering.** Indices that are not literal
   integers currently cannot be lowered. Runtime index expressions in
   `tensor.extract` / `tensor.extract_slice` must be supported.

6. **Broad nested-tensor threaded/vectorized coverage.** The threaded CPU
   pipeline (OpenMP) works for maps, scalar reductions, dot-shaped reductions,
   and row reductions. Broader coverage is needed for general nested tensor
   programs.

7. **Stable vectorization.** The `--cpu-vectorize` path is experimental and
   non-default. It must be hardened and tested across the acceptance suite.

8. **Buffer reuse integration.** The arena allocator foundation exists in
   `remora.runtime` but `buffer-hoisting` and `buffer-loop-hoisting` passes
   must be validated across all CPU pipeline variants.

### 2.2 GPU Gaps

1. **Whole-program GPU lowering.** Currently only function-level
   descriptor-ABI kernels are generated. Full programs with body expressions
   must compile to GPU, not just named functions with explicit parameter types.

2. **General HIR-to-`gpu.module` lowering.** The current GPU path emits
   descriptor-ABI kernels through a specialized bridge (not the general
   `tensor`/`linalg`-to-`gpu` pipeline). A general lowering that consumes the
   same normalized tensor graph as CPU lowering is needed.

3. **GPU reductions beyond f32 rank-1.** Scalar reductions for `int32` and
   `bool` types, plus reductions over rank-2+ arrays (cell reductions), must
   be supported.

4. **GPU dot-shaped reductions for int32.** Currently f32 only.

5. **Non-contiguous GPU view support.** Strided descriptors must be handled
   correctly on GPU, with proper device copies or strided kernel access.

6. **GPU REPL and CLI.** No user-facing GPU target exists. Requires:
   - Input-binding model (`.npy` files for CLI, `:load-npy` for REPL)
   - Whole-program GPU compilation
   - GPU target selection and fallback diagnostics

7. **Multi-kernel GPU programs.** Programs that produce more than one kernel
   (e.g., map-then-fold where fusion does not eliminate the intermediate)
   must be supported.

8. **Hand-written PTX removal.** The legacy `codegen.py` PTX path for
   rank-1 through rank-3 f32 maps must be replaced by the MLIR-derived path
   and removed from the production code path.

9. **Broad GPU int32 support.** Beyond maps, `fold` and other operations
   must work with `int32` inputs.

### 2.3 Technical Debt Blocking Progress

These items, identified in `deepseek-evaluation.md`, must be addressed
_before or alongside_ feature work to avoid compounding the problem:

1. **Text-based MLIR generation → Builder API** (HIGHEST PRIORITY).
   The entire `lowering.py` (2,286 lines) and `gpu_lowering.py` (944 lines)
   construct MLIR through raw f-string concatenation. This is the single
   biggest risk to correctness, maintainability, and future extension.
   Resolution: install PyYAML, enable `iree.compiler.dialects.linalg` (or
   standalone MLIR Python bindings), port emission to the builder API.

2. **God module `lowering.py`** — split into focused modules:
   - `remora/lowering/types.py` — MLIR type mapping
   - `remora/lowering/scalar.py` — scalar region emission (`_RegionEmitter`)
   - `remora/lowering/tensor_ops.py` — map, fold, iota lowering
   - `remora/lowering/view_ops.py` — transpose, slice, reshape, etc.
   - `remora/lowering/module.py` — main module builder, SSA environment

3. **Operator dispatch duplication** — centralize into one shared dispatch
   table used by `lowering.py`, `gpu_lowering.py`, `typechecker.py`, `hir.py`,
   and `runtime.py`. Define operator metadata (name, arity, operand types,
   result type, MLIR op, PTX op) in one place.

4. **Sparse docstrings** — add function-level docstrings to every public
   function in `typechecker.py`, `gpu_lowering.py`, `codegen.py`, and
   the split `lowering/` modules.

5. **`isinstance` chains** — replace with a proper visitor or method dispatch
   where practical (at minimum, extract the node-type traversal into shared
   utility functions so adding a node type requires changes in one place).

6. **Text-processing MLIR hacks** — `_strip_trivial_memref_alloca_scopes`
   in `pipeline.py` performs line-by-line text manipulation. Replace with
   a pass-manager approach or a proper MLIR transformation.

7. **Type alias duplication** — consolidate the `Expr` / `TypedExpr` /
   `HIRExpr` type unions so adding a new node type does not require changes
   in 5 separate files.

8. **Encapsulation violations** — `compiler.py` calls private `TypeChecker`
   methods (`_check_definition`, `_infer_top_level_function_type`,
   `_typed_top_level_function`). Add proper public methods to `TypeChecker`.

---

## 3. Implementation Plan

The plan is organized into 8 work streams. Streams A–D are feature work.
Streams E–F are technical debt remediation. Stream G is infrastructure.
Stream H is cleanup and hardening.

Streams can be partially parallelized but have internal dependencies
as noted.

### Stream A: CPU Dense Core Completion

**Goal:** Every Dense Core form accepted by the typechecker lowers and
executes correctly on CPU.

#### A1. Full tensor SSA environment for `let`
- Extend the existing scalar SSA environment in `lowering.py` to handle
  tensor-typed `let` bindings.
- Remove the HIR-let-inlining fallback for tensors.
- Dependencies: none (standalone change in `lowering.py`).
- Estimate: 2–3 days.

#### A2. General array-cell `fold` callables
- Extend `_lower_fold` to accept named functions, operator sections, and
  lifted lambdas as the fold callable for array-cell reductions.
- Currently only primitive callables work for array-cell folds.
- Dependencies: A1 (for let-bound fold init operands).
- Estimate: 2–3 days.

#### A3. General cell `map` lowering
- Extend cell-map lowering beyond rank-1-cell fold bodies.
- Support cell maps whose body is any scalar-producing HIR expression
  (not just a fold).
- Dependencies: A2 (fold callable generalization feeds into this).
- Estimate: 3–4 days.

#### A4. Dynamic index expression lowering
- Extend `_lower_index` to accept non-literal index expressions.
- Lower runtime indices through `tensor.extract` with dynamic offsets.
- Dependencies: A1 (tensor SSA for index expressions).
- Estimate: 2 days.

#### A5. General top-level function type inference and MLIR lowering
- Add proper function type inference beyond call-site specialization.
- Emit `func.func` with array-parameter types for general top-level
  functions (not just scalar callables).
- Dependencies: A1 (let environment feeds function body lowering).
- Estimate: 4–5 days.

#### A6. Broad threaded pipeline coverage
- Extend the OpenMP threaded lowering to general nested tensor programs.
- Validate `buffer-hoisting` and `buffer-loop-hoisting` in threaded pipeline.
- Dependencies: A1–A5 (programs must lower correctly before threading).
- Estimate: 3–4 days.

#### A7. Stable vectorization
- Remove experimental flag from vectorization path.
- Run vectorized pipeline through full acceptance suite.
- Fix any correctness issues with row-reduction + vectorization combination.
- Dependencies: A6 (threading and vectorization share pipeline infrastructure).
- Estimate: 2–3 days.

#### A8. Buffer reuse integration
- Integrate `buffer-hoisting` and `buffer-loop-hoisting` into ALL CPU
  pipeline variants (default, threaded, vectorized).
- Validate allocation count reduction in `remora-bench`.
- Dependencies: A6, A7.
- Estimate: 2 days.

### Stream B: GPU Dense Core Completion

**Goal:** Every Dense Core form executes on GPU with the same semantics
as CPU, through a general `tensor`/`linalg`-to-`gpu.module` pipeline.

#### B1. General HIR-to-`gpu.module` lowering
- Build a GPU lowering layer that consumes the same normalized tensor graph
  used by CPU lowering.
- Emit `gpu.module` / `gpu.func` kernels with descriptor-ABI entry points.
- Support rank 0–10 descriptor loads/stores through generated loops/index
  decomposition (no hand-written rank branches).
- Support all element types: `f32`, `i32`, `bool` (byte-backed `i8`).
- Dependencies: Stream A completion (CPU lowering must be correct first).
- Estimate: 8–10 days.

#### B2. GPU reductions: i32, bool, rank-2+
- Extend the block-parallel shared-memory reduction to `int32` and `bool`
  element types.
- Add cell-reduction support (reduce over outer dimensions of rank-2+
  arrays).
- Dependencies: B1.
- Estimate: 4–5 days.

#### B3. GPU dot-shaped reductions for int32
- Extend dot-shaped reductions from f32 only to i32.
- Dependencies: B2.
- Estimate: 1–2 days.

#### B4. Non-contiguous GPU view support
- Verify strided descriptors work correctly in GPU kernels.
- Add tests for transposed, sliced, and offset views on GPU.
- Dependencies: B1.
- Estimate: 2–3 days.

#### B5. Whole-program GPU lowering
- Enable `remorac --target gpu-nvidia file.remora` for body programs.
- When the body lowers to a supported GPU kernel, compile and launch.
- When unsupported, emit a clear target diagnostic (not silent CPU fallback).
- Dependencies: B1–B4.
- Estimate: 3–4 days.

#### B6. Multi-kernel GPU programs
- Support programs that produce multiple GPU kernels (e.g., map-then-fold
  where fusion does not fire).
- Sequence kernel launches with correct data dependencies.
- Dependencies: B5.
- Estimate: 3–4 days.

#### B7. GPU CLI with `.npy` input binding
- Add `remorac --target gpu-nvidia --call NAME --input xs.npy` interface.
- Support `.npy` as the first stable array interchange format.
- Dependencies: B5.
- Estimate: 2–3 days.

#### B8. GPU REPL
- Add `:target gpu-nvidia` to REPL with `:load-npy name path` command.
- Support expression evaluation with loaded arrays as descriptor inputs.
- Add clear fallback diagnostics for unsupported GPU programs.
- Dependencies: B5, B7.
- Estimate: 3–4 days.

#### B9. Remove hand-written PTX fallback
- Delete or isolate the legacy `codegen.py` hand-written PTX functions
  (`_unary_f32_map_ptx`, `_binary_f32_map_ptx`, etc.).
- Ensure the MLIR-derived path covers all cases the legacy path handled.
- Dependencies: B1 (MLIR-derived path must be complete).
- Estimate: 1 day.

### Stream C: `if` Conditional Broadening

**Goal:** Extend `if` support beyond scalar booleans to tensor booleans
(where both branches are tensor-typed with matching shapes).

#### C1. Typechecker support for tensor `if`
- Extend `TypeChecker.infer` to accept `if` over boolean tensors where
  both branches have identical array types.
- Dependencies: none (typechecker-only change).
- Estimate: 1–2 days.

#### C2. HIR and MLIR lowering for tensor `if`
- Lower tensor `if` to `scf.if` returning tensors.
- Handle the case where both branches produce tensors of identical shape.
- Dependencies: C1, A1 (tensor SSA for branch results).
- Estimate: 2–3 days.

### Stream D: Prelude and Standard Library Expansion

**Goal:** Expand the starter prelude to cover the full Dense Core surface.

#### D1. Additional prelude functions
- Add: `neg`, `abs`, `maximum`, `minimum`, `any`, `all`, `count`,
  `compose`, `flip`, `const`, `id`.
- These are pure Remora source; no compiler changes needed for the
  interpreter path, but MLIR lowering must be validated for each.
- Dependencies: A5 (general function types for `compose`, `flip`, etc.).
- Estimate: 2 days.

#### D2. `zip` / `zipwith` prelude support
- Implement `zip f a b` which applies a binary function elementwise
  using indexing.
- Requires dynamic index lowering (A4) plus binary map generalization.
- Dependencies: A4, B1.
- Estimate: 1–2 days.

### Stream E: Technical Debt — MLIR Builder API

**Goal:** Replace all text-based MLIR generation with the MLIR builder API.

This is the highest-priority refactor. It affects `lowering.py` and
`gpu_lowering.py`. It must be done early to avoid compounding the problem.

#### E1. Resolve builder API dependency
- Install PyYAML so `iree.compiler.dialects.linalg` (or standalone MLIR
  Python bindings) work.
- Verify the builder API can construct every operation currently emitted
  via text.
- Dependencies: none.
- Estimate: 1–2 days.

#### E2. Port scalar region emission to builder API
- Convert `_RegionEmitter` and `_lower_prim_op` from text to builder API.
- This is the innermost emission layer; porting it first establishes the
  pattern for the rest of `lowering.py`.
- Dependencies: E1.
- Estimate: 2–3 days.

#### E3. Port `iota`, scalar `map`, scalar `fold` to builder API
- Convert the core `linalg.generic` emitters to the builder API.
- Dependencies: E2.
- Estimate: 3–4 days.

#### E4. Port binary `map`, cell `map`, array-cell `fold` to builder API
- Convert the remaining tensor operations.
- Dependencies: E3.
- Estimate: 2–3 days.

#### E5. Port view operations to builder API
- Convert `transpose`, `slice`, `reshape`, `ravel`, `take`, `drop`,
  `reverse` lowering.
- Dependencies: E4.
- Estimate: 2–3 days.

#### E6. Port `gpu_lowering.py` to builder API
- Convert all GPU scaffold and descriptor-ABI kernel emission from text
  to builder API.
- Dependencies: E5.
- Estimate: 3–4 days.

#### E7. Remove text-based emission infrastructure
- Delete string-manipulation helpers (SSA name parsing from MLIR text,
  `_strip_trivial_memref_alloca_scopes` text processing).
- Dependencies: E6.
- Estimate: 1 day.

#### E8. Update golden MLIR fixtures
- Regenerate all checked-in golden fixtures against builder API output.
- Dependencies: E7.
- Estimate: 1 day.

### Stream F: Technical Debt — Module Structure

**Goal:** Split `lowering.py` god module, eliminate duplication,
add docstrings.

#### F1. Split `lowering.py` into focused modules
- Create `remora/lowering/` package with:
  - `types.py` — MLIR type mapping
  - `scalar.py` — scalar region emission
  - `tensor_ops.py` — map, fold, iota, let, if
  - `view_ops.py` — transpose, slice, reshape, ravel, take, drop, reverse
  - `indexing.py` — full-rank and partial indexing
  - `module.py` — main module builder, SSA environment, function lowering
  - `__init__.py` — public `MLIRLowering` facade
- Dependencies: Stream E (do builder API port before or during the split
  to avoid splitting text-based code that will be rewritten).
- Estimate: 3–4 days.

#### F2. Centralize operator dispatch
- Create `remora/operators.py` with a single operator metadata table.
- Each operator entry defines: Remora name, arity, operand types, result
  type rule, MLIR op constructor, typechecker behavior.
- Update `typechecker.py`, `hir.py`, `lowering/scalar.py`,
  `gpu_lowering.py`, and `runtime.py` to use the shared table.
- Remove the 11 duplicated dispatch sites.
- Dependencies: E2 (scalar emission must be ported first).
- Estimate: 2–3 days.

#### F3. Add comprehensive docstrings
- Add function-level docstrings to every public function in:
  - `typechecker.py` (1,134 lines, currently 0 docstrings)
  - `gpu_lowering.py` (944 lines, currently 0 docstrings)
  - `codegen.py` (592 lines, currently 0 docstrings)
  - All new `remora/lowering/` modules
- Dependencies: F1, F2.
- Estimate: 2–3 days.

#### F4. Replace `isinstance` chains with visitor pattern
- Extract the repeated node-type traversal logic into a shared
  `NodeVisitor` base class or utility function.
- At minimum, add a `map_expr(func, expr)` utility that dispatches by
  node type, so adding a node type requires adding one handler in one
  place plus explicit opt-in in each module that supports the type.
- Dependencies: F1 (do after module split).
- Estimate: 2–3 days.

#### F5. Fix encapsulation violations
- Add public methods to `TypeChecker`:
  - `TypeChecker.check_definition(defn, env)`
  - `TypeChecker.infer_top_level_function_type(func_def, env)`
  - `TypeChecker.typed_top_level_function(func_def, typed_body, func_type)`
- Update `compiler.py` to call public methods instead of private ones.
- Dependencies: none (standalone change).
- Estimate: 1 day.

#### F6. Address `assert isinstance` guards
- Replace the 3 `assert isinstance(...)` type-narrowing guards in
  `lowering.py` (lines 721, 745, 863) with explicit `raise TypeError`
  or proper type-narrowing blocks.
- Dependencies: F1 (do during module split).
- Estimate: 0.5 day.

### Stream G: Infrastructure

**Goal:** Add Python CI, improve test coverage, add GPU CI gates.

#### G1. Python CI (GitHub Actions)
- Add a workflow that runs on push/PR:
  - `uv run python tools/validate_mlir_toolchain.py`
  - `uv run python tools/validate_mlir_pipeline.py`
  - `uv run pytest -q`
- Dependencies: none (can be done at any time).
- Estimate: 1 day.

#### G2. GPU CI gates
- Add `REMORA_TEST_GPU=1` gating for live GPU tests.
- Add a GPU CI workflow that runs on self-hosted runners with CUDA.
- Dependencies: B1 (GPU lowering must be stable enough for CI).
- Estimate: 1–2 days.

#### G3. Broaden test coverage
- Add typechecker tests for all deferred features (to lock in expected
  diagnostics).
- Add lowering tests for rank-0 and rank-10 edge cases.
- Add acceptance tests for every Dense Core form in `docs/DENSE_CORE.md`.
- Dependencies: Stream A (features must exist before testing).
- Estimate: ongoing, 1 day per feature.

#### G4. Property-based / fuzz tests
- Add property tests comparing compiled CPU output against the
  typed-AST interpreter for randomly generated valid Dense Core programs.
- Dependencies: A1–A4 (broad lowering coverage needed).
- Estimate: 2–3 days.

### Stream H: Cleanup, Hardening, and Documentation

#### H1. Consolidate type alias updates
- Design a pattern so adding a new expression node type requires changes
  in at most 2 files instead of 5+.
- Dependencies: F4 (visitor pattern).
- Estimate: 1–2 days.

#### H2. Remove hardcoded rank-1 special cases
- Generalize any remaining rank-1 special-case code paths in
  `gpu_lowering.py` and `codegen.py`.
- All lowering should be rank-parametric against `MAX_RANK = 10`.
- Dependencies: B1.
- Estimate: 1–2 days.

#### H3. User documentation
- Add examples for CPU execution, GPU function calls, REPL array
  loading, and performance tuning.
- Add a `docs/USER_GUIDE.md`.
- Dependencies: B8 (GPU REPL must work for docs to be accurate).
- Estimate: 2–3 days.

#### H4. Packaging checks
- Add toolchain discovery diagnostics for missing `ptxas`, LLVM version
  mismatches, CUDA unavailability.
- Include in `tools/validate_mlir_toolchain.py`.
- Dependencies: B7 (need GPU CLI to test packaging).
- Estimate: 1 day.

---

## 4. Milestone Sequence

The milestones align with the existing M1–M7 framework from
`IMPLEMENTATION_PLAN_UPDATE.md`, refined to incorporate technical debt
remediation and the full gap analysis above.

### Phase 1: Foundation Hardening (6–8 weeks)

| Milestone | Description | Streams |
|---|---|---|
| **M-refactor-1** | Builder API port for scalar emission + operator centralization | E1, E2, F2 |
| **M-refactor-2** | `lowering.py` split into modules + docstrings | F1, F3 |
| **M-refactor-3** | Builder API port for all lowering + golden fixtures | E3–E8 |
| **M-infra-1** | Python CI running on GitHub Actions | G1 |

**Gate:** All existing 421 tests pass with builder API output. No text-based
MLIR generation remains. `lowering.py` is split. Operator dispatch is
centralized.

### Phase 2: CPU Completion (4–6 weeks)

| Milestone | Description | Streams |
|---|---|---|
| **M-cpu-1** | Full tensor SSA, general fold callables, general cell maps | A1–A3 |
| **M-cpu-2** | Dynamic index lowering, general function types | A4, A5 |
| **M-cpu-3** | Tensor `if` support | C1, C2 |
| **M-cpu-4** | Threaded/vectorized hardening + buffer reuse | A6–A8 |
| **M-cpu-5** | Prelude expansion + zip support | D1, D2 |
| **M-cpu-6** | Visitor pattern + encapsulation fix + isinstance cleanup | F4–F6, H1 |

**Gate:** Every form in `docs/DENSE_CORE.md` lowers and executes correctly on
CPU. `remorac --target cpu` passes all acceptance tests. Threaded and
vectorized pipelines are stable. Buffer reuse is active.

### Phase 3: GPU Completion (6–8 weeks)

| Milestone | Description | Streams |
|---|---|---|
| **M-gpu-1** | General HIR-to-`gpu.module` lowering | B1 |
| **M-gpu-2** | GPU reductions (i32, bool, rank-2+, dot i32) | B2, B3 |
| **M-gpu-3** | Non-contiguous GPU views + whole-program GPU | B4, B5 |
| **M-gpu-4** | Multi-kernel GPU programs | B6 |
| **M-gpu-5** | GPU CLI + REPL with `.npy` input binding | B7, B8 |
| **M-gpu-6** | Remove hand-written PTX + rank-1 special cases | B9, H2 |

**Gate:** Every form in `docs/DENSE_CORE.md` executes on GPU. `remorac
--target gpu-nvidia` works for body programs and named function calls. GPU
REPL supports `:target gpu-nvidia` with array binding. Hand-written PTX
is removed.

### Phase 4: Polish (2–3 weeks)

| Milestone | Description | Streams |
|---|---|---|
| **M-infra-2** | GPU CI gates + property tests | G2, G4 |
| **M-docs-1** | User guide + packaging checks | H3, H4 |
| **M-final** | Full acceptance suite passes on CPU and GPU | G3 |

**Gate:** `uv run pytest` passes all tests on CPU. `REMORA_TEST_GPU=1 uv run
pytest` passes all GPU-gated tests. User documentation exists. CI is green.

---

## 5. Dependency Graph

```
E1 ──► E2 ──► E3 ──► E4 ──► E5 ──► E6 ──► E7 ──► E8
                │                         │
                ▼                         ▼
               F2                        F1 ──► F3 ──► F4 ──► F6
                │                         │
                └─────────┬───────────────┘
                          ▼
                    Stream A (A1─►A2─►A3─►A4─►A5─►A6─►A7─►A8)
                          │                         │
                          ▼                         ▼
                    Stream C (C1─►C2)          Stream D (D1─►D2)
                          │                         │
                          └─────────┬───────────────┘
                                    ▼
                              Stream B (B1─►B2─►B3─►B4─►B5─►B6─►B7─►B8─►B9)
                                    │
                                    ▼
                              Stream H (H2, H1, H3, H4)

Stream G (G1, G2, G3, G4) — can run in parallel with any phase
Stream F5 (encapsulation fix) — can run at any time
```

## 6. Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| Builder API port introduces subtle MLIR correctness bugs | Medium | High | Run full test suite after each builder API sub-milestone; keep text-based path as reference until port is complete |
| Module split breaks imports | Medium | Medium | Use deprecation re-exports during transition; run import tests first |
| GPU `tensor`/`linalg`-to-`gpu.module` pipeline doesn't preserve descriptor ABI | High | High | Add adapter kernel layer; validate ABI at every GPU milestone |
| Vectorization + threading combination remains unstable | Medium | Medium | Keep combined mode rejected until proven; don't block other milestones |
| LLVM/MLIR 18 toolchain becomes unavailable | Low | High | Document exact versions and sources in `MLIR_TOOLCHAIN.md`; maintain offline cache |
| CUDA toolkit version drift breaks `ptxas` validation | Low | Medium | Pin CUDA 12+ in docs; test on multiple CUDA versions in CI |

## 7. Effort Estimate

| Phase | Work Streams | Estimated Calendar Time |
|---|---|---|
| Phase 1: Foundation Hardening | E, F (partial), G1 | 6–8 weeks |
| Phase 2: CPU Completion | A, C, D, F (remaining), H1 | 4–6 weeks |
| Phase 3: GPU Completion | B, H2 | 6–8 weeks |
| Phase 4: Polish | G2–G4, H3, H4 | 2–3 weeks |
| **Total** | | **18–25 weeks** |

These estimates assume one full-time engineer. With multiple engineers,
Phases 2 and 3 can be partially overlapped (GPU lowering can start once
CPU lowering is ported to builder API, even before CPU feature completion).

## 8. Success Criteria

A complete Dense Core implementation is achieved when:

1. **`uv run pytest` passes all 421+ tests with zero skipped (on CPU).**
2. **Every form listed in `docs/DENSE_CORE.md` "Implemented forms" has:**
   - A passing acceptance test on CPU.
   - A passing acceptance test on GPU (gated on `REMORA_TEST_GPU=1`).
3. **`remorac --target cpu file.remora` runs all acceptance programs.**
4. **`remorac --target gpu-nvidia file.remora` runs all acceptance programs
   with a GPU available.**
5. **`remora --target cpu` and `remora --target gpu-nvidia` REPL sessions
   support definitions, expressions, array binding, and all documented
   commands.**
6. **No text-based MLIR generation remains. The MLIR builder API is used
   throughout.**
7. **`lowering.py` is split into focused modules. Operator dispatch is
   centralized. All public functions have docstrings.**
8. **CI (GitHub Actions) runs the full test suite on every push/PR. GPU
   tests run on self-hosted runners with CUDA.**
9. **No hand-written PTX remains in the production code path.**
10. **`remora-bench --suite` reports allocation counts, compile times,
    and kernel counts for all baseline programs on both CPU and GPU.**
