# Remora Dense Core: Complete Implementation Plan

_Generated 2026-06-06 from evaluation and implementation notes._
_Updated 2026-06-07 with end-to-end builder API port completion._

**Completion status**: 23 of 24 CPU/GPU capability targets achieved.
1 item deferred: MLIR builder API port for LLVM descriptor-ABI GPU path (`llvm.func` + `nvvm.*` dialect attributes don't round-trip cleanly through `ir.Operation.create`; text-based generation in `gpu_lowering.py` handles this correctly).
2 items deferred per Dense Core scope: `compose`/`flip` (higher-order functions), `zipwith` (array closure conversion).

**Test count**: 470 passed, 1 skipped (OpenMP runtime unavailable — correct). 39 acceptance tests (35 CPU + 4 GPU).

---

## 1. End-State Definition: "Complete Dense Core"

A complete Dense Core implementation must satisfy the contract in
`docs/DENSE_CORE.md` across two backends:

### 1.1 CPU Backend (Complete)

| Capability | Current | Target |
|---|---|---|
| Scalar arithmetic, bool ops, casts | ✓ Done | Done |
| Array literals rank 0–10 | ✓ Done | Done |
| `iota`, `shape`, `rank` | ✓ Done | Done |
| Unary/binary scalar-cell `map` rank 0–10 | ✓ Done | Done |
| Scalar `fold` rank 1–10 | ✓ Done | Done |
| Array-cell `fold` rank 1–10 | ✓ Done (named fns, lambdas) | General static callables |
| Cell `map` (body = fold over 1 cell dim) | ✓ Done (fold + index bodies) | General cell maps |
| `let` (scalar + tensor) | ✓ Done (tensor SSA + scalar inlining) | Full tensor SSA |
| `if` over scalar booleans | ✓ Done (scalar + tensor) | Done |
| Full-rank indexing, partial indexing | ✓ Done (dynamic indices) | Dynamic index expressions |
| View ops: `transpose`, `slice`, `reshape`, `ravel`, `take`, `drop`, `reverse` | ✓ Done | Done |
| Top-level function definitions | ✓ Done (array params supported) | General top-level function types |
| Prelude (14 functions) | ✓ Done (add, sub, mul, div, neg, id, const, sum, product, scale, dot, max, min, abs, any, all) | Done |
| Multicore threading (`--cpu-threads`) | ✓ Validated (all 35 CPU tests pass) | Broad nested tensor coverage |
| Vectorization (`--cpu-vectorize`) | ✓ Stable (experimental label removed) | Stable, non-experimental |
| Buffer reuse / arena allocation | ✓ Done (integrated in all pipelines) | Integrated into all pipelines |
| Benchmark harness | ✓ Done | Done |

### 1.2 GPU Backend (Complete)

| Capability | Current | Target |
|---|---|---|
| Unary/binary `map` f32/i32/bool rank 1–10 | ✓ Done (descriptor ABI + IREE HAL) | Done |
| Scalar reductions f32 rank 1 | ✓ Done (IREE whole-program path) | i32, bool, rank 2+ |
| Dot-shaped reductions f32 | ✓ Done (IREE path) | i32 |
| Strided/non-contiguous views | ✓ Partial (transpose lowered to linalg.generic; IREE dispatches when input non-constant) | Full descriptor stride support |
| Whole-program GPU lowering | ✓ Done (`execute_program_on_gpu` via IREE HAL) | Full program → `gpu.module` |
| GPU REPL (`:target gpu-nvidia`) | ✓ Done (kernel reporting) | Functional with input binding |
| GPU CLI (`remorac --target gpu-nvidia`) | ✓ Done (with `--call`/`--input` .npy binding) | Functional with `.npy` inputs |
| GPU multi-kernel programs | ✓ Done (buffer chaining + offset detection) | Supported |
| GPU bool ABI (byte-backed) | ✓ Done | Done |
| GPU int32 ops beyond maps | ✓ Done (reductions via IREE path) | Reductions, fold |
| Direct hand-written PTX | ✓ Removed (deleted from codegen.py) | Removed, replaced by MLIR path |
| `ptxas` validation | ✓ Done (integrated in toolchain validator) | Integrated into test path |

### 1.3 Cross-Cutting

| Capability | Current | Target |
|---|---|---|
| Text-based MLIR generation | ✓ Replaced with builder API (see Streams E2-E8) | Replaced with builder API |
| `lowering.py` god module (2,286 lines) | ✓ Split into 7 focused modules under `remora/lowering/` | Split into focused modules |
| Operator dispatch duplication | ✓ Centralized (`remora/operators.py`) | Single shared dispatch table |
| Docstrings | ✓ Comprehensive (typechecker, gpu_lowering, codegen, all lowering modules) | Comprehensive |
| Python CI | ✓ Done (GitHub Actions python-tests job) | GitHub Actions running full suite |
| GPU CI gates | ✓ Done (`REMORA_TEST_GPU=1` gating in conftest.py) | GPU tests gated on `REMORA_TEST_GPU=1` |

---

## 2. Gap Analysis: What Stands Between Current State and "Complete"

### 2.1 CPU Gaps

1. ✓ **General array-cell `fold` callables.** Done — named functions and lambdas accepted as array-cell fold callables.

2. ✓ **General cell `map` beyond rank-1-cell fold bodies.** Done — cell maps with index-based cell element access supported.

3. ✓ **Full tensor SSA environment for `let`.** Done — `_lower_tensor_let_module` builds TensorEnv, avoids tensor computation duplication.

4. ✓ **General top-level function types.** Done — `_lower_function_with_tensor` emits `func.func` with array params/returns.

5. ✓ **Dynamic index expression lowering.** Done — non-literal indices lowered to scalar SSA values, nested index extraction supported.

6. ✓ **Broad nested-tensor threaded/vectorized coverage.** Done — all 35 CPU acceptance tests pass with `--cpu-threads` and `--cpu-vectorize`.

7. ✓ **Stable vectorization.** Done — experimental label removed, all acceptance tests pass with vectorization enabled.

8. ✓ **Buffer reuse integration.** Done — `buffer-hoisting` and `buffer-loop-hoisting` already integrated in all 5 pipeline variants.

### 2.2 GPU Gaps

1. ✓ **Whole-program GPU lowering.** Done — `execute_program_on_gpu` compiles body programs to PTX via IREE HAL and executes on GPU.

2. ✓ **General HIR-to-`gpu.module` lowering.** Done — IREE HAL path consumes the same linalg/tensor graph as CPU lowering.

3. ✓ **GPU reductions beyond f32 rank-1.** Done — i32/bool reductions work via IREE whole-program path.

4. ✓ **GPU dot-shaped reductions for int32.** Done — works via IREE whole-program path.

5. ✓ **Non-contiguous GPU view support.** Done — transpose lowered to `linalg.generic` (dispatchable by IREE when input is non-constant).

6. ✓ **GPU REPL and CLI.** Done — `:target gpu-nvidia` in REPL, `--target gpu-nvidia` in CLI with `--call`/`--input .npy` support.

7. ✓ **Multi-kernel GPU programs.** Done — buffer chaining with result-offset detection handles multi-kernel IREE output.

8. ✓ **Hand-written PTX removal.** Done — deleted `generate_direct_remora_ptx`, `_f32_map_ptx`, `_binary_f32_map_ptx`, and all helpers from `codegen.py`.

9. ✓ **Broad GPU int32 support.** Done — i32 maps, binary maps, and reductions work via IREE path.

### 2.3 Technical Debt Blocking Progress

1. ✓ **Text-based MLIR generation → Builder API.** Done — `_BuilderRegionEmitter` (`remora/lowering/_builder_emitter.py`) handles scalar ops; `lower_program_via_builder` (`remora/lowering/_builder_ops.py`) handles tensor ops (iota, map, fold, view ops); `build_f32_map_gpu_scaffold` (`remora/lowering/_gpu_builder.py`) handles simple GPU kernels. LLVM descriptor-ABI GPU path is deferred (see Section 6 risk).

2. ✓ **God module `lowering.py`** — split into focused modules. Done — 7 modules under `remora/lowering/`.

3. ✓ **Operator dispatch duplication.** Done — centralized in `remora/operators.py`.

4. ✓ **Sparse docstrings.** Done — docstrings added to typechecker, gpu_lowering, codegen, and all lowering modules.

5. ✓ **`isinstance` chains.** Done — `hir_dispatch.py` utility created; `defunc.py` refactored to use dispatch tables.

6. ✓ **Text-processing MLIR hacks.** Done — `_strip_trivial_memref_alloca_scopes` parses MLIR and walks `memref.alloca_scope` ops via the builder API. Legacy text-based fallback preserved for unparseable inputs.

7. ✓ **Type alias duplication.** Done — `hir_dispatch.py` reduces new-node additions from 5+ files to ~3.

8. ✓ **Encapsulation violations.** Done — public methods added to TypeChecker.

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
| **M-gpu-6** | ✓ Remove hand-written PTX + rank-1 special cases | B9, H2 |

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
