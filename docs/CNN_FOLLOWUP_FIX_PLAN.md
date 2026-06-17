# CNN Follow-up Fix Plan

## Status

**Status: COMPLETE (2026-06-16)**

This plan addressed the issues found in
`docs/review_of_cnn_changes_made_by_deepseek.md` after considering Deepseek's
response.  The goal is not to reopen the whole CNN scalability effort.  The
core IR-size work is useful and should remain.  This plan focuses on concrete
correctness fixes, stricter native-mode validation, and status wording that
matches what is actually validated.

Implemented follow-up changes:

- `TypeChecker.infer()` now honors `_caching_enabled`.
- `examples/crater_train.py --compiled` now requires compiled native execution.
- Auto training mode reports the compiled fallback reason when verbose.
- The compiled crater parity test skips only for the known `memrefCopy`
  runtime blocker on environments that still lack the runtime helper.
- Native cache keys include tool version/stat fingerprint data and the cache
  pipeline version was bumped.
- Unused `HIRRelu` HIR surface was removed.
- Status documents now distinguish descriptor-compiler completion from native
  crater execution validation.

## Linker Follow-up (2026-06-16) [COMPLETE]

DeepSeek and Codex agreed that the remaining `memrefCopy` failure is a native
link/runtime bug, not just a documentation caveat.  The repo had two divergent
compiled CPU link paths:

- `CPUExecutor.compile_source()` linked `remora_rt.o` inline.
- `CPUFunctionExecutor.compile_source()` used `_compile_llvm_ir_to_shared_library()`,
  whose linker command omitted Remora runtime support.

The fix plan is:

1. Make Remora's C runtime provide the MLIR bufferization helper `memrefCopy`
   used by the lowered LLVM IR.
2. Make `_compile_llvm_ir_to_shared_library()` link `remora_rt.o` consistently.
3. Route `CPUExecutor.compile_source()` through the same shared helper.
4. Bump the native artifact cache pipeline version so older `.so` files with
   unresolved runtime symbols are not reused.
5. Add tests that check the shared linker includes the runtime object and that
   a compiled function needing `memrefCopy` can load and execute.

Implemented:

- `remora/remora_rt.c` now defines a generic strided `memrefCopy` helper.
- `_compile_llvm_ir_to_shared_library()` links `remora_rt.o`.
- `CPUExecutor.compile_source()` now uses the shared link helper instead of a
  separate inline linker command.
- Native artifact cache `pipeline_version` was bumped to avoid reusing old
  `.so` files with unresolved runtime symbols.
- `tests/test_runtime_linking.py` verifies the runtime object is linked.
- `tests/test_execution.py::test_cpu_function_executor_links_memref_copy_runtime_support`
  verifies the compiled function path loads and executes with runtime support.

## Scope

In scope:

- Fix the typechecker memoization guard.
- Wire the `examples/crater_train.py --compiled` CLI flag.
- Add strict compiled-mode behavior so fallback is explicit.
- Narrow the compiled parity test skip condition to the known `memrefCopy`
  runtime blocker.
- Improve native artifact cache invalidation for toolchain changes.
- Remove or complete dead `HIRRelu` API surface.
- Update plan/status docs to distinguish "implemented with fallback" from
  "validated native execution."

Out of scope:

- Adding batch dimensions to the descriptor ABI.
- Making GPU crater training work.
- Adding typed-HIR disk caching.  Phase C remains rejected.
- Treating `im2col`/`col2im` memref allocation as a leak without post-pipeline
  evidence.  At most, add a verification note or test later.

## Phase 1: Fix Typechecker Memoization Guard

**Goal:** make the implemented memoization behavior match the plan's stated
design: speculative/type-directed retry paths must be able to disable infer
caching.

### Problem

`TypeChecker` has `_caching_enabled` and retry paths set it to `False`, but
`infer()` ignores the flag and always reads/writes `_infer_cache` for
non-`VarExpr` nodes.

### Implementation

1. Update `TypeChecker.infer()` so cache lookup and cache writes happen only
   when `_caching_enabled` is true.
2. Preserve timing/profiling behavior regardless of caching state.
3. Add or update tests covering:
   - Cache is used for repeated non-speculative inference.
   - Cache is bypassed while `_caching_enabled` is false.
   - A rank-polymorphic retry scenario does not reuse a typed result inferred
     under a different expected callable/cell type.

### Acceptance

- Focused typechecker tests pass.
- Existing crater/CNN smoke tests still pass.
- No meaningful regression in the crater gradient preparation benchmark.

## Phase 2: Make Compiled Training Mode Explicit

**Goal:** keep auto fallback for ergonomics, but provide a strict compiled mode
that raises when native execution is unavailable.

### Problem

`train_tiny_dataset(use_compiled=None)` tries compiled mode and falls back to
the interpreter.  This is acceptable for default usage, but the current CLI
`--compiled` flag is ignored and there is no strict mode for validation.

### Implementation

1. Wire the CLI flag through to `train_tiny_dataset()`.
2. Define mode semantics:
   - `use_compiled=None`: auto mode; try compiled first, fall back with a clear
     message when verbose.
   - `use_compiled=False`: force interpreter.
   - `use_compiled=True`: require compiled native execution; raise on failure.
3. Change `_try_compiled()` or replace it with a helper that returns both the
   compiled object and the failure reason, so fallback is visible in verbose
   output.
4. Ensure `TrainingResult.compiled` accurately reports the mode actually used.

### Acceptance

- `train_tiny_dataset(use_compiled=False)` uses interpreter mode.
- `train_tiny_dataset(use_compiled=None)` falls back cleanly when compiled
  execution is unavailable.
- `train_tiny_dataset(use_compiled=True)` raises when compiled execution is
  unavailable.
- `python examples/crater_train.py --compiled` requires compiled mode instead
  of silently falling back.

## Phase 3: Narrow Compiled Parity Test Skips

**Goal:** make tests skip only for the known missing-runtime blocker, while
still failing on unrelated compiler/lowering regressions.

### Problem

`test_compiled_gradients_match_interpreter()` catches every exception and calls
`pytest.skip()`.  This can hide unrelated bugs.

### Implementation

1. Add a small helper, local to the test or shared in test utilities, that
   identifies the known native-runtime blocker:
   - missing `mlir-opt` or `llc`
   - missing native runtime support
   - exception text containing `undefined symbol: memrefCopy`
2. Skip only for those known conditions.
3. Let all other exceptions fail the test.
4. If strict compiled mode from Phase 2 exists, use it in the test so fallback
   cannot accidentally satisfy the test.

### Acceptance

- In environments without the runtime helper, the compiled parity test skips
  specifically because of `memrefCopy`.
- If a lowering/typechecking/codegen error occurs before the known runtime
  blocker, the test fails.
- The test message clearly states the skipped dependency.

## Phase 4: Improve Native Cache Toolchain Fingerprint

**Goal:** prevent stale shared libraries when LLVM/MLIR tools are upgraded in
place at the same path.

### Problem

The cache key currently fingerprints toolchain executable paths, not versions
or binary contents.  Upgrading `mlir-opt`, `mlir-translate`, or `llc` at the
same path can reuse stale `.so` files.

### Implementation

1. Update `_toolchain_fingerprint()` in `remora/cache.py` to include:
   - executable path
   - `--version` output when available
   - optionally file mtime/size as a fallback
2. Keep failures non-fatal; if version probing fails, include an explicit
   fallback marker in the fingerprint.
3. Bump the cache `pipeline_version` to invalidate existing entries.
4. Add tests for:
   - fingerprint changes when mocked version output changes
   - cache key changes when fingerprint changes

### Acceptance

- Cache key is stable for identical source/options/tool versions.
- Cache key changes under mocked toolchain version changes.
- Existing cache behavior still works when version probing is unavailable.

## Phase 5: Remove or Complete `HIRRelu`

**Goal:** avoid dead HIR API surface.

### Problem

`HIRRelu` is defined and added to dispatch/result-type handling, but no pass
emits it and no lowerer handles it.

### Implementation Options

Prefer the smallest safe option:

1. Remove `HIRRelu` from:
   - `remora/hir.py`
   - `remora/hir_dispatch.py`
   - `remora/hir_opt.py`
2. If removal reveals intended tests or call sites, either restore with full
   lowering or document why it is needed.

Alternative:

1. Add recognition and lowering for ReLU.
2. Add tests proving it is emitted and lowered correctly.

### Acceptance

- No unused `HIRRelu` references remain, unless full support is implemented.
- HIR dispatch tests and lowering tests pass.

## Phase 6: Update Status Documentation

**Goal:** make the docs match the validated behavior without discounting the
real scalability work.

### Problem

The original follow-up used caveat language while native execution was blocked
by `memrefCopy`.  The linker follow-up fixed that runtime support in this
environment, so the docs should now distinguish retained fallback behavior from
validated strict native execution.

### Implementation

Update:

- `docs/CNN_GRADIENT_NATIVE_COMPILE_SCALABILITY.md`
- `docs/CRATER_DETECTION_PERFORMANCE_PLAN.md`
- optionally `docs/crater_train_status.txt` if it still claims stronger status

Recommended wording:

- Descriptor MLIR scalability phases are complete.
- Crater training auto mode is implemented with interpreter fallback.
- Strict compiled mode exists and raises when native execution is unavailable.
- `memrefCopy` runtime linkage is provided by `remora_rt.c`.
- Strict native crater training has been validated on the tiny one-epoch run.

### Acceptance

- Status tables distinguish:
  - implemented
  - validated
  - deferred
  - blocked by external/runtime dependency
- Phase E is not simply labeled complete without the runtime caveat.
- Rejected Phase C remains rejected, with cache-after-preparation described as
  intentional.

## Verification Checklist

Run focused tests first:

```bash
env UV_CACHE_DIR=/tmp/uv-cache uv run pytest \
  tests/test_crater_train.py \
  tests/test_hir.py \
  tests/test_im2col.py \
  tests/test_ad_source.py::test_array_gradient_with_captured_scalar_map_and_fold_compiles_and_executes \
  -q
```

Run any new typechecker/cache tests added by this plan.

If time permits, run the broader suite:

```bash
env UV_CACHE_DIR=/tmp/uv-cache uv run pytest -q
```

Manual checks:

```bash
env UV_CACHE_DIR=/tmp/uv-cache uv run python examples/crater_train.py --epochs 1 --examples 2
env UV_CACHE_DIR=/tmp/uv-cache uv run python examples/crater_train.py --compiled --epochs 1 --examples 2
```

Expected behavior after the linker follow-up:

- Default command should prefer compiled execution when native support is
  available.
- `--compiled` should run compiled execution and raise only on real
  toolchain/runtime failures.

2026-06-16 verification:

```text
49 passed, 1 skipped in 683.58s
```

Manual checks:

- `uv run python examples/crater_train.py --epochs 1 --examples 2` fell back to
  interpreter mode, reported the `memrefCopy` load failure, and completed one
  epoch.
- `uv run python examples/crater_train.py --compiled --epochs 1 --examples 2`
  previously raised with `undefined symbol: memrefCopy`.

2026-06-16 linker follow-up verification:

```text
tests/test_runtime_linking.py
tests/test_execution.py::test_cpu_function_executor_links_memref_copy_runtime_support
2 passed in 0.40s
```

Manual strict compiled crater check:

```text
Using compiled native execution (single value-and-grad function)
epoch   1 loss 0.694780
compile_seconds=364.285
mean_step_seconds=0.001202
loss: 0.696294 -> 0.694780
```

## Completion Criteria

This follow-up plan is complete when:

- Typechecker caching is correct and covered by tests.
- CLI compiled mode is wired and strict.
- Compiled parity test skips only for known missing runtime support.
- Cache keys include toolchain version/fingerprint information beyond paths.
- Dead `HIRRelu` surface is removed or fully supported.
- Status docs accurately describe implemented, validated, blocked, and deferred
  work.
