# Remora Implementation Plan Update

This document updates the older MLIR phase plan with a direct roadmap from the
current prototype to the intended system: a full-language Remora compiler and
REPL that execute efficiently on multicore CPUs and NVIDIA GPUs.

## 1. Current Baseline

The repository is a working Dense Core prototype, not yet a full Remora system.

Implemented today:

- Parser, source locations, typechecker, HIR, defunctionalization, MLIR lowering,
  compiled CPU execution, CLI, REPL, and test infrastructure for a useful static
  Dense Core subset.
- Rank-0 through rank-10 ABI descriptor structs and descriptor-input CPU
  callable functions.
- Compiled CPU execution for scalar values, static arrays, `iota`, maps, folds,
  binary maps, dot-shaped programs, static `shape`/`rank`, indexing, booleans,
  and the starter prelude.
- Function-level NVIDIA execution for current descriptor-ABI slices:
  rank-1 through rank-3 `float32`/`int32` unary and binary maps, plus rank-1
  `float32` scalar reductions and dot-shaped reductions.
- Fusion and performance smoke tests for the current CPU/linalg path.

Major gaps:

- Full Remora semantics are not implemented: dynamic shapes/rank, boxes,
  generalized rank-polymorphic annotations, first-class dynamic functions,
  broad views/slices/transposes, richer reductions/scans, and full standard
  library behavior remain future work.
- CPU performance is not yet a mature multicore backend. The current path uses
  MLIR/LLVM, but it lacks a deliberate bufferization/vectorization/threading
  contract and benchmark gates.
- GPU performance is still early. Generated kernels cover narrow slices,
  reductions are serial, bool ABI layout is unresolved, non-contiguous GPU views
  are not proven, and whole-program GPU lowering is not general.
- CLI/REPL GPU execution is not user-facing yet because there is no input-binding
  model for descriptor-input GPU functions.

## 2. Target Architecture

The durable architecture is:

1. Parse source into AST with precise source ranges.
2. Typecheck into static Dense Core types plus explicit shape/function metadata.
3. Lower into HIR that preserves frame/cell semantics, array views, and static
   callable specialization.
4. Normalize HIR into a fused, backend-independent tensor graph.
5. Lower the tensor graph to standard MLIR (`tensor`, `linalg`, `arith`, `scf`,
   `func`, `memref`, `gpu`, LLVM/NVVM), not a custom Remora dialect for v1.
6. Execute through one Remora descriptor ABI for CPU and GPU.
7. Drive CLI and REPL through the same compiler/runtime APIs.

The implementation should keep two execution tiers:

- **Dense Core v1**: static rank and static dimensions, rank 0 through 10,
  dense rectangular arrays, static callable specialization, views via descriptors,
  CPU and NVIDIA execution.
- **Full Remora v2**: dynamic shape/rank, boxes/hidden shapes, richer
  rank-polymorphic typing, and dynamic higher-order values. These should not be
  mixed into the v1 performance path until Dense Core is stable.

## 3. Milestone Roadmap

### M1. Lock the Dense Core Language Contract

Goal: make the supported language surface explicit and testable before expanding
backend work.

Implementation:

- Add a concise `docs/DENSE_CORE.md` that defines accepted syntax, type rules,
  rank/shape restrictions, callable restrictions, and deferred full-language
  features.
- Add acceptance manifests for every supported feature and every intentionally
  deferred feature. Deferred cases should assert stable diagnostics, not silently
  skip.
- Replace stale roadmap examples that still imply `gpu-nvidia` is the default
  CLI target.
- Decide and document bool memory layout for the external ABI:
  `bool` arrays use one byte per element at descriptor boundaries. GPU code may
  compute predicates internally as `i1`, but global memory load/store for public
  bool arrays must use `i8` values normalized to `0` or `1`.

Acceptance gates:

- `docs/DENSE_CORE.md` exists and is referenced from `IMPLEMENTATION_NOTES.md`.
- Acceptance tests distinguish supported, rejected, and deferred programs.
- Bool ABI policy is documented in `docs/ABI.md` and used by tests.

### M2. Replace Text-Splicing Lowering with Structured Tensor Lowering

Goal: stop growing ad hoc textual lowering and create a reliable tensor IR
builder layer.

Implementation:

- Introduce a small internal lowering builder that owns SSA names, emits MLIR
  text deterministically, tracks tensor values, and supports insertion without
  string surgery.
- Replace tensor-let inlining with a real tensor SSA environment for `let`,
  top-level values, map inputs, fold inputs, indexing, and shape/rank metadata.
- Generalize lowering for rank 0 through 10:
  scalar-cell unary/binary maps, scalar folds, array-cell folds, nested maps,
  partial/full indexing, static shape/rank, casts, conditionals over scalars,
  and top-level function specialization.
- Lower views as descriptor-compatible tensor/memref metadata where possible.
  Surface syntax for slices/transposes can come later, but internal view
  semantics must not require ABI replacement.

Acceptance gates:

- Existing CPU acceptance tests still pass.
- New tests cover rank-4 and rank-10 maps/folds/indexing where type rules allow.
- No lowering path relies on rewriting emitted MLIR by searching for `@main`.
- MLIR verifier catches every generated module in CI when tools are available.

### M3. Build a Real Multicore CPU Backend

Goal: make CPU execution a performant backend, not just correctness execution.

Implementation:

- Define the CPU pipeline in documented stages:
  canonicalize/fuse, one-shot bufferize, convert linalg to loops or affine,
  vectorize where legal, lower to LLVM, and enable OpenMP or another explicit
  threading strategy.
- Add CPU target options:
  `--cpu-threads`, `--cpu-vectorize/--no-cpu-vectorize`, and
  `REMORA_NUM_THREADS` as the environment default.
- Replace temporary shared-library compilation only if it blocks performance or
  stability. Otherwise keep it until a stable in-process LLVM execution binding
  is available.
- Add buffer reuse/arena planning for intermediate arrays so fused and
  non-fused programs do not allocate unnecessary temporaries.
- Add a benchmark harness that records compile time, run time, allocation count,
  and linalg/kernel counts for vector scale, map chains, reductions, dot,
  row reductions, and rank-3/rank-10 maps.

Acceptance gates:

- CPU backend uses multiple cores for large maps/reductions when requested.
- Map chains and dot-shaped programs do not materialize avoidable intermediates.
- Benchmarks have checked-in thresholds or trend baselines.
- Compiled CPU results match the interpreter/NumPy oracle for all acceptance
  programs.

### M4. Generalize NVIDIA GPU Lowering

Goal: replace narrow generated kernels with a general HIR/tensor-to-GPU path for
Dense Core.

Implementation:

- Add a GPU lowering layer that consumes the same normalized tensor graph used
  by CPU lowering and emits descriptor-ABI `gpu.module` kernels.
- Support rank 0 through 10 descriptor loads/stores by generated loops/index
  decomposition rather than hand-written rank branches.
- Support element types:
  `float32`, `int32`, and public `bool` as byte-backed `i8` arrays.
- Support operations:
  unary/binary maps, scalar comparisons, boolean maps, casts, scalar folds,
  dot-shaped reductions, row/outer-axis reductions, and map-then-fold fusion.
- Replace serial reductions with parallel reductions:
  per-block shared-memory reduction, multi-block reduction for large arrays, and
  a second-stage reduction kernel when needed.
- Add non-contiguous input and output descriptor support. Runtime device copies
  must either preserve the host view span correctly or pack into a contiguous
  device buffer and pass matching contiguous descriptors deliberately.
- Make `ptxas` validation part of the normal GPU test path when installed.

Acceptance gates:

- Live CUDA tests pass for maps, bool maps, reductions, dot, rank-3 tensors,
  rank-10 descriptor metadata, and strided inputs/outputs when CUDA is present.
- PTX parameter ABI matches `docs/ABI.md`; no IREE HAL PTX is treated as
  launchable Remora code.
- Direct hand-written PTX fallback is removed or isolated as a legacy test-only
  fixture.
- GPU reductions are parallel and benchmarked against the serial prototype.

### M5. Add User-Facing GPU CLI and REPL Execution

Goal: make `gpu-nvidia` a real user target, not only a Python API.

Implementation:

- Add an explicit input-binding model for descriptor-input functions:
  `remorac --target gpu-nvidia --call NAME --input xs.npy --input ys.npy file.remora`.
- Use `.npy` as the first stable array interchange format. It preserves dtype,
  shape, and contiguous layout; view/stride inputs can be added later through
  an explicit metadata format.
- For programs with a body and no external inputs, allow direct
  `remorac --target gpu-nvidia file.remora` when the body lowers to a supported
  GPU kernel.
- Add REPL commands:
  `:target gpu-nvidia`, `:load-npy name path`, `:vars`, and `:clear name`.
  The REPL should compile expressions using loaded arrays as descriptor inputs.
- Add clear fallback behavior:
  unsupported GPU programs fail with a diagnostic that names the unsupported
  construct and suggests `--target cpu`; they do not silently run on CPU.

Acceptance gates:

- CLI can run a GPU `scale`, `sum`, `dot`, bool map, and row-reduction example
  using `.npy` inputs.
- REPL can load arrays, switch targets, run the same examples, and recover after
  unsupported GPU diagnostics.
- CPU and GPU formatting paths produce identical Remora output for matching
  results.

### M6. Expand the Language Toward Full Remora

Goal: move beyond Dense Core once CPU/GPU execution is stable.

Implementation order:

1. Add surface view operations: slice, take/drop, transpose, reshape/ravel where
   semantics are shape-preserving or statically known.
2. Add generalized rank-polymorphic annotations and function type inference so
   reusable functions are not specialized only at direct call sites.
3. Add shape variables and symbolic static dimensions where constraints can be
   solved at compile time.
4. Add boxed/hidden-shape arrays as a separate runtime representation, not by
   weakening Dense Core tensors.
5. Add dynamic shape/rank dispatch by specializing and caching rank-specific
   kernels; do not replace optimized static kernels with generic interpreters.
6. Add dynamic higher-order values only after static specialization and boxed
   shape representations are settled.

Acceptance gates:

- Each new feature has parser/typechecker/lowering/runtime acceptance tests.
- Dynamic features do not regress Dense Core benchmark baselines.
- Diagnostics clearly distinguish unsupported full-language features from type
  errors.

### M7. Productionize Performance, Packaging, and Developer Workflow

Goal: make the compiler usable for real programs and maintainable by multiple
engineers.

Implementation:

- Add a benchmark suite with machine-readable JSON output and comparison tooling.
- Track compile time, run time, memory traffic proxies, allocation counts,
  kernel counts, and CPU thread counts.
- Add golden MLIR/PTX tests only for stable contracts; prefer semantic and
  metadata tests for incidental compiler output.
- Add debug flags:
  `--emit-mlir-after`, `--emit-llvm`, `--emit-ptx`, `--save-temps`,
  `--explain-fusion`, and `--explain-target`.
- Add packaging checks for toolchain discovery, CUDA availability, missing
  `ptxas`, and LLVM version mismatches.
- Add user docs with examples for CPU execution, GPU function calls, REPL array
  loading, and performance tuning.

Acceptance gates:

- A fresh environment can run validators and examples with documented setup.
- Benchmark regressions are visible in CI or a documented local command.
- Debug output is sufficient to diagnose why a program used CPU, failed GPU
  lowering, or materialized an intermediate.

## 4. API and Interface Changes

Compiler/runtime APIs:

- Keep `compile_source` for whole-program CPU-oriented compilation.
- Add `compile_function_source_to_target(source, function_name, param_types,
  target=...)` as the stable function-level entry point for CPU/GPU descriptor
  execution.
- Add a target capability result object:
  supported, selected target, diagnostics, kernel metadata, input requirements,
  and fallback suggestion.
- Keep `RemoraExecutor` for direct descriptor-ABI GPU kernels only.
- Add a symmetric high-level `FunctionExecutor` facade for CPU and GPU so CLI
  and REPL do not call backend-specific executor constructors directly.

CLI:

- Keep current `cpu`, `interp`, `mlir`, and `ptx` behavior.
- Add `gpu-nvidia` only with explicit input binding and no silent CPU fallback.
- Add `--call`, repeated `--input NAME=PATH` or ordered `--input PATH`, and
  `--output PATH` for `.npy` result output.

REPL:

- Keep CPU as default.
- Add GPU only after array binding exists.
- Add `:load-npy`, `:vars`, and target diagnostics before adding incremental
  compilation.

## 5. Testing Strategy

Required test layers:

- **Acceptance tests**: supported, rejected, and deferred programs with stable
  expected behavior.
- **Typechecker tests**: shape/rank/function diagnostics, static bounds, bool
  ABI decisions, and dynamic-feature rejection.
- **Lowering tests**: parse/verify MLIR for representative rank 0, 1, 2, 3, 4,
  and 10 programs.
- **CPU execution tests**: compare compiled CPU, interpreter, and NumPy oracles.
- **GPU execution tests**: live CUDA tests skip when CUDA is absent but validate
  maps, bool maps, reductions, dot, strided descriptors, and rank coverage when
  present.
- **Performance tests**: non-flaky smoke thresholds plus optional extended
  benchmark runs.
- **CLI/REPL tests**: target selection, `.npy` input binding, diagnostics,
  formatting, session recovery, and loaded definitions.

The minimum green command set after each milestone is:

```bash
env UV_CACHE_DIR=/tmp/uv-cache UV_LINK_MODE=copy uv run python tools/validate_mlir_toolchain.py
env UV_CACHE_DIR=/tmp/uv-cache UV_LINK_MODE=copy uv run python tools/validate_mlir_pipeline.py
env UV_CACHE_DIR=/tmp/uv-cache UV_LINK_MODE=copy uv run pytest -q
```

## 6. Sequencing Rules

- Do not broaden full-language features before Dense Core CPU/GPU performance
  gates exist.
- Do not add user-facing `gpu-nvidia` CLI/REPL execution until input binding is
  explicit and unsupported GPU programs produce target diagnostics.
- Do not treat IREE HAL PTX as a Remora runtime artifact.
- Do not add new rank-specialized branches for ranks below 10 unless they are
  explicitly marked as temporary and have a removal milestone.
- Do not silently fall back from GPU to CPU in user-facing commands.
- Keep the typed-AST interpreter as a correctness oracle, not the default user
  execution path.

## 7. Completed Sprint and Next Work

Completed in this sprint:

1. Added `docs/DENSE_CORE.md`.
2. Updated `docs/ABI.md` with the byte-backed public bool array policy.
3. Added explicit acceptance manifest categories for supported, rejected, and
   deferred programs.
4. Introduced a small internal MLIR main-module builder and typed tensor SSA
   environment values.
5. Ported tensor-let/map/fold/index module assembly away from `@main`
   string-splicing.
6. Added tests for Dense Core docs, acceptance categories, and the no-splicing
   tensor-let lowering contract.

Completed in the follow-on sprint:

1. Moved the remaining straightforward whole-program scalar, iota, array
   literal, and scalar map emitters onto the shared main-module builder.
2. Added chained tensor-let coverage for let-bound maps feeding later tensor
   expressions.
3. Added rank-4 and rank-10 fold acceptance cases.
4. Added requested CPU thread-count plumbing through CLI/runtime artifacts and
   `REMORA_NUM_THREADS`.
5. Added `remora-bench`, a JSON benchmark harness for compile/pipeline/execution
   timing and coarse operation counts.

Completed in the M3 CPU sprint:

1. Added an experimental threaded CPU pipeline that lowers linalg maps through
   `scf.parallel`, the OpenMP dialect, OpenMP-to-LLVM, and LLVM lowering.
2. Made `--cpu-threads` and `REMORA_NUM_THREADS` select that threaded pipeline
   for requests above one thread, with explicit libomp runtime detection and a
   stable diagnostic when the runtime is unavailable.
3. Added execution-time `OMP_NUM_THREADS` scoping for compiled whole-program and
   descriptor-input CPU calls.
4. Added coarse allocation counts to `remora-bench`.
5. Added `docs/BENCHMARK_BASELINES.json` and smoke coverage for the benchmark
   baseline structure.
6. Kept the focused CPU/CLI/performance/lowering/acceptance suite and MLIR
   validators green.

The next sprint should harden M3 rather than broaden the language:

1. Make threaded lowering work for reductions and nested tensor programs, not
   just parallel map-shaped programs.
2. Add CI or a documented local validation profile with libomp installed so
   `--cpu-threads > 1` is exercised through link and execution, not only MLIR
   pipeline text.
3. Add `--cpu-vectorize/--no-cpu-vectorize` and settle the vectorization stage
   before GPU generalization resumes.
4. Turn benchmark baselines into executable gates for allocation counts, fusion
   counts, and optional machine-local trend comparisons.
5. Start buffer reuse/arena planning for intermediate tensors that survive
   fusion.
