# Remora Crater Detection Performance Plan

## Goal

Make Remora usable for training and inference of convolutional neural networks
on large (≥ 256×256) lunar surface images for crater detection.  Compilation
must finish in seconds, not minutes, and inference must run at native speed.

## Diagnosis

The current pipeline compiles a 32×32 6-parameter CNN gradient in ~112 s and
executes interpreted gradients at ~7 s per training step.  The root causes are:

### 1. Typechecking dominates compilation (111 of 112 seconds)

The generated gradient source (~29 KB for a single parameter, ~172 KB for
all six) describes a deeply-nested computation DAG.  When `prepare_function_source`
parses and typechecks this source, it expands into ~3,790 HIR nodes that the
typechecker must elaborate, specialize, and lower.  Every subexpression is
fully elaborated by the dependent-type checker — there is no memoisation of
identical subexpressions during type inference.

Measurements from Phase 1-10 benchmarks:

| Stage | Time | Notes |
|---|---|---|
| Gradient source generation | 0.01 s | AD tape + emission |
| Function preparation (typecheck + HIR) | 111.0 s | Parsing, type inference, elaboration, defunctionalization |
| Descriptor MLIR generation | 0.06 s | HIR → MLIR text |
| `mlir-opt-18` CPU pipeline | 0.03 s | Bufferization, lowering, LLVM conversion |
| MLIR-to-LLVM + `llc` + link | 0.07 s | Native code generation |
| **Total** | **~112 s** | |

Everything after HIR construction takes under 0.2 seconds.  The typechecker
is the sole bottleneck.

### 2. Compilation repeats for every gradient parameter

The original `crater_train.py` generated and typechecked 6 separate gradient
functions (one per trainable parameter).  Phase 6 consolidated this into a
single multi-output function, but the total typechecking work is similar
since all six backward paths share the forward structure.

### 3. Interpreted execution is slow

The current interpreter evaluates Remora expressions recursively on
NumPy arrays.  A single forward pass on 32×32 images takes ~0.02 s, but
a gradient step (forward + 6 backward evaluations) takes ~7 s in
interpreted mode.  Native compiled execution (via `CPUFunctionExecutor`)
would be much faster, but the first-time compilation cost discourages
its use.

### 4. Image size scaling is unknown

Phases 0-10 were benchmarked exclusively on the 32×32 crater CNN.  The
compact `im2col` implementation (Phase 3) should scale the IR size
proportionally to loop-nest depth, not pixel count, but this has not
been verified for 128×128 or 256×256 images.

### 5. Missing optimisations for large models

- **Matmul recognition (Phase 8)** does not trigger for the CNN because
  dot products sit inside defunctionalised cell-maps.
- **Saved-value tape (Phase 7)** is opt-in and defaults to off.
- **No batch dimension** in the descriptor ABI — each example requires
  a separate `execute()` call.
- **No GPU path** for convolutions — only elementwise maps.

---

## Plan

### Phase A: Measure the typechecker bottleneck

**Goal:** identify which type-inference pass(es) consume the 111 seconds.

**Approach:**

1. Add wall-clock timing to `TypeChecker.infer()` or to key dispatch
   functions (`_infer_fold`, `_infer_map`, `_infer_app`, `_infer_lambda`).
   Use `perf_counter()` around each call and accumulate totals per kind.
2. Instrument `prepare_function_source` to report:
   - Parse time
   - Typecheck time (per definition + main body)
   - Specialize time
   - `lower_expr` (typed AST → HIR) time
3. Run on the CNN gradient-0 source (single parameter, 29 KB) and the
   multi-output source (all 6 parameters, 172 KB).
4. Produce a table: "typecheck pass X: Y seconds (Z% of total)".

**Exit criteria:** a ranked list of the most expensive type-inference
paths with measured times.

### Phase B: Add type-level memoisation

**Goal:** avoid re-elaborating structurally identical subexpressions
during typechecking.

**Approach:**

1. The typechecker infers types bottom-up on the AST.  When it encounters
   a subexpression it has already inferred (same AST identity or same
   typed result), return the cached result instead of recursing.
2. Use `id(expr)` as a memo key for the current specialization context
   (since the AST is shared within a single `infer` call).
3. Alternatively, memoize `_typed_top_level_function` results by a
   hash of (function source, param types, index args).

**Exit criteria:** typechecking time for the CNN gradient is reduced by
at least 50% (from ~111 s to < 55 s).

### Phase C: Cache typed HIR (pre-MLIR compilation cache)

**Goal:** make repeated compilations of the same function instant.

**Approach:**

1. Extend `remora/cache.py` to cache `PreparedFunctionArtifact` (typed HIR
   + function metadata) keyed by (source hash, function name, param types).
2. Serialize the HIR function to a portable format (e.g., pickle the frozen
   dataclass tree, or regenerate from source on cache miss and store the
   prepared artifact).
3. On cache hit, skip `prepare_function_source` entirely and proceed
   directly to descriptor lowering.
4. Invalidate when the Remora compiler version changes (already in the
   cache key).

**Exit criteria:** second compilation of the CNN gradient takes < 1 second
(all work skipped except cache lookup and descriptor lowering).

### Phase D: Scale to larger images

**Goal:** verify and fix compilation for 64×64, 128×128, and 256×256 images.

**Approach:**

1. Create a parametric test suite that generates gradient functions for
   different image sizes (64×64, 128×128, 256×256) with the same CNN
   architecture.
2. Measure:
   - Gradient source generation time
   - Function preparation (typecheck + HIR) time
   - Descriptor MLIR size (bytes, `linalg.generic` count)
   - CPU pipeline time
   - Compilation success/failure
3. If the typechecker scales superlinearly, apply Phase B memoisation.
4. If the MLIR pipeline scales superlinearly (unlikely after Phase 3),
   investigate loop-fusion or bufferization bottlenecks.
5. Fix any static-shape validation failures at larger sizes.

**Exit criteria:** 256×256 CNN gradient compiles successfully in
proportional time to the 32×32 baseline (after memoisation).

### Phase E: Enable production training loop

**Goal:** `crater_train.py` trains with compiled native execution by default.

**Approach:**

1. Wire `CompiledTrainingFunctions` as the default path in
   `train_tiny_dataset`.  Keep interpreter as fallback.
2. Add a numerical parity assertion: compare compiled gradient output with
   interpreted output for a random input, assert `allclose` at 1e-4.
3. Add a batch dimension to the CNN: accept `(batch, 32, 32)` input and
   return `(batch,)` loss.  This requires changes to the descriptor ABI
   (batch dimension is a leading `?` dynamic dimension in the memref)
   and the runtime's `execute_into` to handle batched outputs.
4. Enable the saved-value tape by default (`use_saved_values=True`) after
   verifying numerical parity.
5. Extend `HIRMatmul` recognition to trigger for cell-map dot products
   (the `linear` layer pattern: `map(lambda row: fold(+, 0, map(*, row, x)), w)`).

**Exit criteria:** `uv run examples/crater_train.py` completes in native
compiled mode with loss decreasing across epochs, and total wall time
dominated by training steps, not compilation.

### Phase F: GPU acceleration (optional, deferred)

**Goal:** run convolution and linear layers on GPU.

**Approach:**

1. Extend the GPU scaffold (`remora/gpu_lowering.py`) to handle `HIRMatmul`
   and `HIRIm2col` operations via structured linalg GPU kernels.
2. Add a CUDA descriptor ABI for multi-output (Pair-returning) functions.
3. Profile against CPU baseline.

**Exit criteria:** a GPU-accelerated training step is faster than CPU
native execution for image sizes ≥ 128×128.

---

## Priority and Dependencies

```text
Phase A (profile typechecker)
  → Phase B (memoisation) ──┐
                              → Phase D (scale images)
Phase C (HIR cache) ─────────┘
                              → Phase E (production training)
                                → Phase F (GPU, optional)
```

Phases A, B, and C can run in parallel.  Phase D depends on B.  Phase E
depends on B, C, and D.  Phase F is independent and optional.

---

## Acceptance Targets

- [ ] Typechecker profile identifies dominant pass(es).
- [ ] Typechecking time for 32×32 CNN gradient < 55 s (50% reduction).
- [ ] Second compilation of 32×32 CNN gradient < 1 s (HIR cached).
- [ ] 256×256 CNN gradient compiles successfully.
- [ ] 256×256 CNN gradient compilation time proportional to 32×32 baseline.
- [ ] `crater_train.py` trains with compiled native execution by default.
- [ ] Compiled and interpreted gradients agree within 1e-4 relative tolerance.
- [ ] Full non-training test suite passes after final implementation.
