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
  runtime blocker.
- Native cache keys include tool version/stat fingerprint data and the cache
  pipeline version was bumped.
- Unused `HIRRelu` HIR surface was removed.
- Status documents now distinguish descriptor-compiler completion from native
  crater execution validation.

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

- Implementing the missing MLIR runtime `memrefCopy` linkage.
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
that fails when native execution is unavailable.

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
- `train_tiny_dataset(use_compiled=None)` falls back cleanly when `memrefCopy`
  is missing.
- `train_tiny_dataset(use_compiled=True)` raises when `memrefCopy` is missing.
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

- In the current environment, the compiled parity test skips specifically
  because of `memrefCopy`.
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

The current plans use "COMPLETE" language that can be read as native crater
training being validated end-to-end.  The code currently validates
implementation pieces and an interpreter fallback, while native execution is
blocked by `memrefCopy` in this environment.

### Implementation

Update:

- `docs/CNN_GRADIENT_NATIVE_COMPILE_SCALABILITY.md`
- `docs/CRATER_DETECTION_PERFORMANCE_PLAN.md`
- optionally `docs/crater_train_status.txt` if it still claims stronger status

Recommended wording:

- Descriptor MLIR scalability phases are complete.
- Crater training auto mode is implemented with interpreter fallback.
- Strict compiled mode exists and raises when native runtime support is
  missing.
- Compiled parity is skipped only for the known missing `memrefCopy` runtime
  dependency.
- Native crater training is not considered end-to-end validated until the
  runtime linkage is fixed and parity passes.

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

Expected current behavior until `memrefCopy` is fixed:

- Default command runs via interpreter fallback.
- `--compiled` raises with the native runtime/linkage error instead of falling
  back.

2026-06-16 verification:

```text
49 passed, 1 skipped in 683.58s
```

Manual checks:

- `uv run python examples/crater_train.py --epochs 1 --examples 2` fell back to
  interpreter mode, reported the `memrefCopy` load failure, and completed one
  epoch.
- `uv run python examples/crater_train.py --compiled --epochs 1 --examples 2`
  failed fast with `undefined symbol: memrefCopy`.

## Completion Criteria

This follow-up plan is complete when:

- Typechecker caching is correct and covered by tests.
- CLI compiled mode is wired and strict.
- Compiled parity test skips only for known missing runtime support.
- Cache keys include toolchain version/fingerprint information beyond paths.
- Dead `HIRRelu` surface is removed or fully supported.
- Status docs accurately describe implemented, validated, blocked, and deferred
  work.
