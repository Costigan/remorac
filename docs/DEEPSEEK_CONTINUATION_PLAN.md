# DeepSeek Continuation Plan

## Purpose

This is the working plan for continuing `remorac` development after the current
crater CNN classifier work.  It expands the recommended direction from
`docs/REMORAC_CAPABILITY_SUMMARY.md` into an execution roadmap and folds in the
anchor-free crater detector plan from
`docs/ANCHOR_FREE_CRATER_DETECTION_PLAN.md`.

The core strategy is incremental compiler maturation through examples:

1. Dense static AD workloads.
2. Dense non-AD numeric kernels.
3. Memory-pressure workloads.
4. Static padded irregular workloads.
5. Sparse, dynamic, or library-backed workloads.

The anchor-free crater detector is the central near-term example.  Supporting
examples such as logistic regression, image filters, PDE stencils, and N-body
are used to isolate compiler features before those features are required by the
detector or later production workloads.

## Ground Rules for DeepSeek

- [ ] Preserve existing crater classifier behavior.  Do not regress
  `examples/crater_train.py` or its tests.
- [ ] Prefer small, executable milestones over broad architectural rewrites.
- [ ] Keep Python responsible for orchestration, data wrangling, irregular
  post-processing, and checkpointing.
- [ ] Keep Remora responsible for dense differentiable array computation and
  compiler validation.
- [ ] Start with static shapes and static batch sizes.  Dynamic shapes are deferred.
- [ ] Add tests before scaling model size or moving work to GPU.
- [ ] Record status, timings, and known limitations in docs after each completed
  phase.
- [ ] Do not implement full YOLO, U-Net, sparse graph analytics, FFT libraries, or
  production geospatial pipelines until the smaller prerequisite phases pass.

## Current Baseline

`remorac` currently has:

- [x] Dense static arrays.
- [x] Rank-polymorphic arithmetic.
- [x] `map`, `fold`, reductions, scans, and common views.
- [x] Reverse-mode AD for scalar losses over many elementwise, view, reduction, and
  indexing operations.
- [x] `exp`, `log`, `select`, and stable BCE-style scalar math.
- [x] Static-shape `im2col` and `col2im`.
- [x] Multi-output value-and-grad source generation.
- [x] Native CPU compilation through the descriptor path.
- [x] Python/NumPy orchestration proven by `examples/crater_train.py`.

Important gaps this plan addresses:

- [ ] Static batch support for training workloads.
- [ ] CPU/GPU parity on dense kernels.
- [ ] GPU ABI and lowering for larger, multi-input, multi-output examples.
- [ ] Buffer reuse and memory planning for large intermediate arrays.
- [ ] Matmul/dot pattern recognition.
- [ ] Gather/scatter and padded irregular data support.

## Roadmap Overview

| Done | Phase | Main Example | Main Compiler Capability |
|---|---|---|---|
| [x] | 0 | Current crater classifier | Baseline preservation |
| [x] | 1 | Logistic regression / softmax | Static batched AD, stable loss |
| [x] | 2 | Synthetic crater grid detector | `examples/crater_detect_train.py` trains on synthetic data; 6/6 tests pass |
| [x] | 3 | Batched crater detector | Mini-batch SGD with Python gradient accumulation; parity tests |
| [ ] | 4 | Image filters / PDE stencil | Dense non-AD kernels and CPU/GPU parity |
| [ ] | 5 | Real crater tile pipeline | Python orchestration and detector validation |
| [ ] | 6 | Pyramid/overlap crater inference | Scaled inference with Python post-processing |
| [ ] | 7 | N-body / differentiable renderer | Memory pressure and buffer planning |
| [ ] | 8 | Molecular dynamics / graph step | Padded irregular gather/scatter |
| [ ] | 9 | Later sparse/library workloads | PageRank, tomography, FFT, production scale |

## Phase 0: Preserve the Current Classifier Baseline

### Goal

Keep the existing crater classifier as the regression suite and reference
training path while adding new examples.

### Tasks

- [x] Run the existing focused crater/classifier tests before major changes.
- [x] Run and record the current tests in `tests/test_crater_train.py`:
  - [x] `test_tiny_crater_training_decreases_loss`,
  - [x] `test_benchmark_produces_reasonable_numbers`,
  - [x] `test_strict_compiled_mode_raises_on_compile_failure`,
  - [x] `test_compiled_gradients_match_interpreter`.
- [x] Identify any additional focused tests that validate:
  - [x] `im2col`/`col2im`,
  - [x] CNN gradients,
  - [x] compiled value-and-grad.
- [x] Add a short status note if any test is currently skipped due to local
  toolchain limitations.
- [x] Confirm that any skip in `test_compiled_gradients_match_interpreter` is
  still limited to the documented native-runtime/toolchain blocker, historically
  the `memrefCopy` undefined-symbol failure.
- [x] Do not move classifier source or change its public script interface unless a
  later phase explicitly requires it.

### Acceptance

- [x] Existing crater classifier tests pass or have documented, pre-existing skips.
- [x] Strict compiled mode remains available where the native toolchain supports it.
- [x] New files do not alter classifier output or checkpoint format.

### Phase 0 Status (2026-06-17)

**Command lines run:**
```
uv run pytest tests/test_crater_train.py -v
uv run pytest tests/test_im2col.py -v
uv run pytest tests/test_ad_source.py -v
uv run pytest tests/test_ad.py -v
```

**Tests passed/skipped/failed:**
- `tests/test_crater_train.py`: 4/4 passed (no skips, no failures)
  - `test_tiny_crater_training_decreases_loss` — PASSED
  - `test_benchmark_produces_reasonable_numbers` — PASSED
  - `test_strict_compiled_mode_raises_on_compile_failure` — PASSED
  - `test_compiled_gradients_match_interpreter` — PASSED
- `tests/test_im2col.py`: 17/17 passed (im2col/col2im, CNN forward/gradients, BCE stability)
- `tests/test_ad_source.py`: 51/51 passed (value-and-grad generation, compiled gradients, select/index/append/rotate VJPs)
- `tests/test_ad.py`: 53/53 passed (tape AD, finite difference checks, compiled cross-validation)

**Known skip reason:**
No tests were skipped. `test_compiled_gradients_match_interpreter` did not skip because the native toolchain (mlir-opt-18, llc-18) is available on this machine. The `memrefCopy` undefined-symbol blocker was not triggered.

**Current strict compiled-mode status:**
Compiled native execution is fully available. `CompiledTrainingFunctions` constructs and executes successfully, with forward loss and multi-output gradients matching the interpreter within tolerance (rtol=1e-3, atol=1e-5). The interpreted gradient source compilation takes ~163s for 6 separate per-input gradient functions. The compiled value-and-grad source generation is much faster (single multi-output call).

## Phase 1: Dense Static AD Baseline

### Goal

Use logistic regression and optional softmax classification to harden the
simplest batched AD path before the detector depends on it.

### Why This Comes First

The crater detector needs batched scalar losses, stable objectness BCE, and
gradient accumulation across batch and grid axes.  Logistic regression isolates
those requirements without convolution, image tiling, or object detection target
assignment.

### Tasks

- [x] Add `examples/logistic_train.py` with synthetic data generation.
- [x] Add a Remora source for binary logistic loss:

```text
loss(w [D], b Float, x [B,D], y [B]) -> Float
```

- [x] Treat the shape line above as planning notation only.  The implementation
  must use real Remora syntax, following the `define/pi` style used by
  `examples/crater_train.py` and the AD examples in `docs/USER_GUIDE.md`.
- [x] Use stable BCE with logits.
- [x] Generate value-and-grad for `w` and `b`.
- [x] Train with Python-owned SGD.
- [x] Compare one run against a NumPy reference for loss reduction.
- [x] If static batch ABI is insufficient, start with per-example loss and document
  the exact missing shape support before implementing batch changes.

### Compiler Work

- [x] Validate descriptor execution for rank-2 inputs and rank-1 labels.
- [x] Validate reduction to scalar mean loss.
- [ ] Validate AD accumulation over the leading batch axis.
- [ ] Add or improve dot/matvec recognition only if the explicit Remora expression
  compiles but performs pathologically.

### Tests

- [x] Forward loss matches a NumPy reference.
- [x] Gradients pass finite-difference spot checks.
- [x] One optimizer step lowers loss on a fixed batch.
- [x] Short synthetic training run decreases loss.

### Acceptance

- [x] A static batched AD example trains through Python.
- [x] Failures in the detector's objectness loss can now be compared against this
  simpler baseline.

### Phase 1 Status (2026-06-17)

**Deliverables:**
- `examples/logistic_train.py` — logistic regression training script
- `tests/test_logistic_train.py` — 7 tests covering determinism, forward loss,
  one-step loss reduction, training convergence, strict compiled-mode error
  handling, compiled-vs-interpreter gradient parity, and compiled-vs-interpreter
  forward parity

**Exact shapes used:**
- Per-example Remora function `logistic-loss(w [D], b Float, x [D], y Float) -> Float`
- Default D=4, batch size B=8
- Python loops over the batch accumulating per-example gradients
- Batched `[B,D]` loss was deferred; per-example loss compiles and trains
  correctly. The inline `map *` + `fold +` dot-product compiles through the
  descriptor path for rank-1 arrays.

**Remora source/function names:**
- `bce(logit, y)` — stable binary cross-entropy with logits using `select` pattern
- `logistic-loss(w, b, x, y)` — per-example logistic loss = BCE(logit, y)

**Tests added:**
```
uv run pytest tests/test_logistic_train.py -v
```
7/7 passed:
- `test_synthetic_data_is_deterministic`
- `test_forward_loss_is_finite`
- `test_one_step_lowers_loss`
- `test_training_decreases_loss`
- `test_strict_compiled_mode_raises_on_compile_failure`
- `test_compiled_gradients_match_interpreter`
- `test_forward_losses_match_compiled_vs_interpreter`

**Loss before/after (default settings, 5 epochs, B=4, D=4):**
- Initial: 0.7172 → Final: 0.4808 (compiled mode, learning_rate=0.1)
- Loss decreases consistently across interpreted and compiled modes.

**Compiled/interpreter mode status:**
- Compiled native execution works on this machine (mlir-opt-18, llc-18 available).
  `CompiledLogisticFunctions` constructs in ~1s and executes value-and-grad
  correctly.
- Interpreted gradient source generation takes ~163s for the first call
  (2 per-input gradient functions) but is cached via `@lru_cache`.
- Compiled and interpreted gradients match within rtol=1e-3, atol=1e-5.
- Compiled and interpreted forward losses match within rel=1e-6.

**Compiler limitations found:**

### Gap: `define/pi` index parameter inference does not propagate through nested `define/pi` calls — **FIXED**

**Status:** Resolved (2026-06-17).  Symbolic `(iapp dot D)` now works inside
`define/pi ([D Dim])` after the specialization-binding fix below.

**Changes made:**
- `remora/typechecker.py:_infer_index_app` — validates symbolic DimVar args
  against both the TypeEnv index bindings AND the current specialization's
  `_current_index_bindings` dict.  Resolves DimVars to concrete values during
  specialization (e.g., `D → StaticDim(4)`).
- `remora/typechecker.py:TypeChecker.__init__` — added `_current_index_bindings`
  instance attribute.
- `remora/typechecker.py:_infer_top_level_function_type` — saves/restores
  `_current_index_bindings` around body typechecking so nested iapp calls
  resolve symbolic indices to concrete values.
- `remora/typechecker.py:_is_dim_bound_in_env` — new static helper checking
  whether an index name is bound as Dim in a TypeEnv.

**What works now:**
- `(iapp dot 3)` — concrete, works as before.
- `((iapp dot D) w x)` inside `define/pi ([D Dim])` — **now works**.
- Wrong-sort (e.g., `D` bound as Shape) correctly rejected.
- Unbound symbolic args correctly rejected.
- Generated HIR/source/AD path works for a small nested helper using symbolic iapp.

**Original gap description (for reference):**

A `define/pi` helper with index bindings cannot be called from within another
`define/pi` function when the index arguments are symbolic (DimVar).  The
typechecker infers indices correctly from concrete arrays at the top level, and
from concrete arrays inside a `define/pi ()` (no index bindings), but not from
symbolic indices in a `define/pi ([D Dim] ...)` context.

**Minimal failing source (now fixed):**
```lisp
(define/pi ([D Dim])
  (dot [a (Array Float D) b (Array Float D)] Float)
  (fold + 0.0 (map * a b)))

(define/pi ([D Dim])
  (loss [w (Array Float D) x (Array Float D)] Float)
  (dot w x))    ;; FAILS
```

**Exact error:**
```
remora.types.RemoraTypeError: <input>:3:4: could not infer index argument(s): D
```

**Command to reproduce:**
```python
from remora.lisp_reader import parse_lisp
from remora.typechecker import TypeChecker
TypeChecker().check_program(parse_lisp(source))
```

**What works:**
- Top-level call with concrete arrays: `(dot [1.0 2.0] [3.0 4.0])` — PASSES
- Call from `define/pi ()` with concrete param types: `(define/pi () (call-dot [xs (Array Float 3) ys (Array Float 3)] Float) (dot xs ys))` — PASSES
- Explicit `iapp` with concrete index: `((iapp dot 3) w x)` — PASSES

**What does not work:**
- Call from `define/pi ([D Dim])` with symbolic D: FAILS (above)
- Explicit `iapp` with symbolic index: `((iapp dot D) w x)` — FAILS with "explicit index argument D must be concrete"

**Proposed compiler fix (smallest):**
Allow `iapp` to accept symbolic index arguments that reference the enclosing
`define/pi`'s index bindings.  When the typechecker encounters
`(iapp dot D)` inside a `define/pi` with `[D Dim]` in its binder list, it
should unify `D` with the callee's index binders rather than requiring a
concrete `StaticDim`.  This is a localized change in the iapp typechecking path
(`_infer_index_app` or equivalent).

**Alternative fix (broader):**
Propagate index inference through nested `define/pi` calls.  When `(dot w x)`
is encountered and `w` has type `(Array Float D)` where `D` is a symbolic index
from the enclosing function's binder, the typechecker should attempt to unify
that with `dot`'s index binders.  This requires changes in
`_infer_top_level_function_app`.

**Workaround used:**
Inlined the dot-product expression `(fold + 0.0 (map * x w))` directly in the
loss function rather than using a parameterized helper.  This is acceptable for
the current small model but would become unwieldy for larger Remora sources.

**Impact on this phase:**
Prevented a clean factored `dot` helper.  Not blocking — the per-example
logistic loss compiles and trains correctly.  Would block a batched version
that maps a helper over a `[B,D]` input if the helper needed index parameters.

**Command lines run:**
```
uv run pytest tests/test_logistic_train.py -v
uv run pytest tests/test_crater_train.py -v
uv run python examples/logistic_train.py --epochs 5 --examples 4
```

## Phase 2: Synthetic Anchor-Free Crater Grid Detector

### Goal

Implement the first detector entirely on synthetic data with fixed shapes.

### Model Contract

Start with:

```text
images:  [B, 1, 64, 64]
targets: [B, 8, 8, 4]
preds:   [B, 8, 8, 4]
loss:    Float
```

Each target cell is:

```text
[objectness, dx, dy, log_radius]
```

For the model output, objectness is an unbounded logit.

### Python Tasks

- [x] Add `examples/crater_detect_data.py` (synthetic data module).
- [x] Add synthetic crater image generation:
  - [x] grayscale `float32`,
  - [x] one to a few craters per tile,
  - [x] configurable noise,
  - [x] deterministic seeds.
- [x] Add dense target assignment:

```text
gx = floor(local_x / cell_width)
gy = floor(local_y / cell_height)
dx = (local_x - gx * cell_width) / cell_width
dy = (local_y - gy * cell_height) / cell_height
log_radius = log(radius / radius_scale)
```

- [x] Add target decoding back to circles.
- [x] Record conflicts when multiple crater centers land in the same grid cell.
- [ ] Add `examples/crater_detect_train.py` (blocked by compiler gap below).
- [ ] Add `--synthetic`, `--dry-run-data`, `--examples`, `--seed`, and
  `--checkpoint` options.

### Remora Tasks

- [x] Add a detector Remora source string (typechecks, interpreter forward works).
- [x] Start with the current classifier's proven architecture shape rather than
  inventing a larger detector backbone:
  - [x] one `3x3` convolution/im2col stage (stride 8),
  - [x] ReLU via `select`,
  - [x] the smallest downsampling/pooling needed to reach the `8x8` grid
    (stride-8 im2col gives 8×8 directly),
  - [x] head producing `[8, 8, 4]` (scalar-per-cell head; objectness-only
    variant typechecks and runs via interpreter).
- [ ] Add a second convolution stage only after the single-convolution detector
  loss, value-and-grad, and synthetic training run are stable.
- [x] Implement per-cell loss (objectness BCE works elementwise over [64]).
- [x] Reduce cell losses to a scalar mean loss (fold+).
- [ ] Generate multi-output value-and-grad for trainable parameters
  (generates, but native .so has missing entry point — see compiler gap).
- [ ] Keep optimizer state in Python.

### Compiler Work

- [ ] Validate AD through all detector operations.
- [ ] Validate cotangent accumulation through frame/grid axes.
- [x] Validate `im2col` on `64x64` inputs (pure im2col compiles; combined with
  map+fold now compiles too — lowering gap fixed; see Phase 2 Status).
- [x] Explicitly verify that the existing multi-output value-and-grad path works
  for the detector's multi-parameter setup: a scalar loss built from
  `map`+`fold` over `im2col` now generates, compiles (`verify=True`), executes,
  and returns correct gradients (bias grad = patch count).  See
  `tests/test_im2col.py::test_map_fold_over_im2col_value_and_grad_compiles`.

### Tests

- [x] Synthetic target assignment and decoding are inverse within one pixel.
- [x] Empty cells have objectness `0.0`; assigned cells have `1.0`.
- [ ] Perfect prediction loss is lower than shifted prediction loss.
- [ ] Detector loss is finite for extreme logits and radii.
- [ ] Gradients are finite.
- [ ] One optimizer step lowers fixed-batch loss.
- [ ] Short synthetic training run decreases total loss.

### Acceptance

- [x] `examples/crater_detect_train.py --synthetic` trains the detector on fixed
  synthetic data.
- [ ] The run reports total, objectness, center, and radius loss components.
- [ ] Checkpoints can be saved and restored for inference on synthetic tiles.

**Note (2026-06-17):** the training script is objectness-only (BCE).  Center and
radius heads are deferred to a later phase (requires a 4-channel output head and
corresponding target decoding).  Checkpointing is deferred to Phase 5 (real
pipeline).

### Phase 2 Status (2026-06-17) — PARTIAL, lowering gap fixed; detector train script still TODO

**Completed:**
- `examples/crater_detect_data.py` — synthetic image generation, target
  assignment, decoding, conflict recording
- `tests/test_crater_detect_data.py` — 10 tests (all pass, 0.07s)
- Detector Remora source typechecks; interpreter forward loss = 44.36
  (finite, reasonable)
- AD source generation succeeds; per-input gradient sources generated
- Value-and-grad source generation succeeds
- **Compiler lowering gap FIXED** (see below): `map` of an inline lambda
  containing a `fold` over `im2col` output now lowers to valid MLIR and
  executes correctly.  Value-and-grad for a scalar loss built from this
  pattern now generates and compiles.

### Gap (FIXED): `map` lambda containing `fold` over `im2col` output produced empty MLIR

**Status:** Resolved (2026-06-17).

**Root cause:** The cell-map lowerer (`remora/lowering/tensor_ops.py`,
`_lower_map_cell_result`) only accepted `HIRVar` callables (lifted/named
functions) and raised `RemoraLoweringError("only lifted lambda cell maps lower
to MLIR so far")` for inline `HIRLambda` callables.  `compile_prepared_function`
(`remora/compiler.py`) swallows `RemoraLoweringError` into a 0-byte MLIR
artifact, so the failure surfaced only as an empty `.so` with no
`_mlir_ciface_remora_call` entry point.  The HIR and typechecker were always
correct; only the lowerer rejected the inline lambda.

This affected *every* cell-map whose callable was an inline lambda (with or
without `im2col`); `im2col` was not the trigger.  The handoff's "what works"
list was inaccurate: `map (+ p 0.0)` over im2col actually raises a typecheck
error (`(+ p 0.0)` keeps the cell shape, so the result is `[900,9]` not
`[900]`), and `map (fold ...)` alone (no im2col) hit the *same* lowering error.

**Minimal failing source (now fixed):**
```lisp
(define/pi ()
  (f [image (Array Float 32 32)] (Array Float 900))
  (map (lambda (p) (fold + 0.0 p)) (im2col image [3 3] 1)))
```

**Changes made:**
- `remora/lowering/tensor_ops.py:_lower_map_cell_result` — when `node.func` is
  an inline `HIRLambda`, synthesize a `HIRFunction` (`__cell_lambda`) from the
  lambda's `params`/`body`/`result_type.result` and proceed through the existing
  cell-fold / cell-index lowering.  The downstream
  `_lower_map_cell_fold_result` and `_lower_map_cell_index_result` only read
  `function.body` and `function.params`, both present on `HIRLambda`, so no
  further changes were required.

**Verification:**
- The minimal source compiles with `verify=True` (full MLIR validation) and
  executes; output matches a NumPy patch-sum reference (max abs diff ~1e-6).
- Parametrized over `(8,3,1)`, `(32,3,1)`, `(8,3,2)`, `(7,2,2)` — all match.
- Value-and-grad for a scalar loss `fold + 0.0 (map (lambda (p) (+ (fold + 0.0 p) bias)) (im2col image [3 3] 1))`
  generates, compiles (`verify=True`), executes, and returns correct gradients
  (bias gradient = 36 = patch count, analytically confirmed).
- Tests added in `tests/test_im2col.py`:
  `test_map_fold_over_im2col_lowers_and_matches_reference` (parametrized, 4
  cases) and `test_map_fold_over_im2col_value_and_grad_compiles`.
- `tests/test_im2col.py`: 22/22 pass (17 existing + 5 new).  Fast regression
  subsets (`test_im2col.py`, `test_ad.py`, `test_crater_detect_data.py`,
  `test_phase7_dependent_functions.py`): 132/132 pass.

**Classifier execution model (confirmed):** `examples/crater_train.py` always
runs the forward loss through the *interpreter* (`_prepare_interpreted_function`,
crater_train.py:199); only the value-and-grad is compiled.  `conv2d`'s forward
therefore never compiled before (and still does not — see remaining gaps), so
this was never a regression.  The detector should follow the same pattern:
interpreter forward, compiled value-and-grad.

### Remaining / pre-existing lowering gaps (not blockers; documented for next steps)

These are *not* caused by the fix above and do not block Phase 2 because the
detector can follow the classifier's source pattern and rely on AD
restructuring for the value-and-grad.  They are recorded so the next compiler
maturation step has exact failing shapes.

**Update (2026-06-17, second pass):** gaps 1–4 and 6–7 below are now FIXED.
The detector **forward** compiles natively and matches a NumPy reference
(see `tests/test_im2col.py::test_detector_forward_compiles_and_matches_reference`).
The fully-inlined scalar detector **loss** (`fold + 0.0 (map (lambda (l) (* l l)) <detect>)`)
also compiles natively and matches NumPy exactly.  The remaining blocker for a
fully-compiled training loop is gap 8 (variable operator sections in the
AD-generated value-and-grad source).

1. **Free scalars inside a `map` lambda lower to wrong MLIR** — FIXED.
   `_lower_binary_map_fold_input` / `_lower_binary_map_result` now splat scalar
   operands (literal *or* variable) to the result rank.  `(map (lambda (d) (+ (* w2 d) b2)) dots)` is still best written as `(+ (* w2 (map ...)) b2)`
   (scalars outside the lambda), but the splat makes rank-polymorphic
   `(+ array scalar_var)` / `(* scalar_var array)` compile and execute
   correctly.

2. **Dot-product cell-fold (`fold + 0.0 (map * p k)`)** — FIXED.
   `_reduces_param` now recognizes a `HIRMap`/`HIRApply` referencing the param,
   and `_lower_cell_fold_producer_result` lowers `fold <f> init (map OP p free...)`
   by feeding the cell element plus each cell-shaped free array (broadcast
   across the frame) into a single `linalg.generic` reduction.  This is the
   `dot-patch` pattern; the classifier's `conv2d` now compiles natively
   (`test_classifier_conv2d_pattern_compiles_and_matches`).

3. **Reduce-then-transform cell-map body** — still deferred, but no longer
   needed: the detector follows the classifier's source pattern (scalars
   outside the lambda), so this shape does not arise.

4. **`HIRLet`-wrapped cell-map bodies** — FIXED.  `_lower_map_cell_result`
   now inlines the lambda body's `HIRLet` chain (via `_inline_lets`) before
   fold/index routing, so a named `dot-patch` called inside a map lambda
   reaches the cell-fold lowerer.

6. **Function-call inlining** — still deferred.  A function that *calls*
   another user function (e.g. `detect-loss` calling `detect`) and uses the
   result as a tensor fails with "only tensor literals and iota values lower
   as tensor inputs so far".  Workaround: write the loss with the detect body
   inlined (no call), which compiles.  The classifier avoids this by
   interpreting the forward; the AD path also restructures calls away in some
   cases.

7. **SSA name collision in nested cell-maps** — FIXED.  The cell-fold lowerers
   (`_lower_map_cell_fold_result`, `_lower_cell_fold_producer_result`,
   `_lower_map_cell_index_result`) used hardcoded `%map_empty`/`%mapped`/`%init`/`%filled`;
   when a cell-fold result fed a sibling map, the names collided
   ("redefinition of SSA value '%map_empty'").  A `prefix` is now threaded
   from `_lower_tensor_input`/`_lower_fold_input` through the cell-map
   lowerers and incorporated into the SSA names.

8. **Variable operator sections in maps** — FIXED.
   `_lower_callable_operand` (`remora/lowering/scalar.py:97`) now resolves a
   `HIRVar` section operand via `scalar_env` (threaded through
   `_lower_primitive_callable_result`/`_lower_map_callable_result`), so
   `(map (* w) arr)`, `(map (+ b) arr)`, `(map (- w) arr)` compile and match
   NumPy (`test_variable_operator_section_in_map_compiles`).

**Two more bugs found and fixed while closing gap 8 (the compiled multi-output
value-and-grad was silently wrong):**

9. **Runtime multi-output (Pair) scalar-input aliasing** — FIXED.  The Pair
   path of `CPUFunctionExecutor.execute_into` built input memref descriptors
   from `np.asarray(input)` *temporaries* but only retained a *different* set
   of `np.asarray` copies as keepalive, so scalar-input temporaries were GC'd
   and their addresses reused — every scalar input aliased to the last one.
   Symptom: the compiled value-and-grad returned the wrong gradient for every
   output after the first (e.g. g_b2≈14.8 vs correct 188.1).  The non-Pair
   path was already fixed earlier; the Pair path now materializes input arrays
   once and reuses them for both the descriptors and the keepalive.
   (`test_detector_value_and_grad_matches_finite_differences`.)

10. **Non-unique SSA names in Pair scalar-component lowering** — FIXED.
    `_lower_descriptor_scalar_result_body` and `_RegionEmitter` used fixed
    names (`%init`, `%folded`, `%extracted`, `%v0`, `%result_cast`) that
    collided across pair components (parse error for simple losses, scoped but
    fragile for hoisted folds).  A `prefix` is now threaded from
    `_lower_pair_result` (`pair_{i}`) through
    `_lower_descriptor_scalar_result_body` and `_RegionEmitter.temp`.

**Phase 2 compiler unblock — COMPLETE.**  The detector forward, the inlined
detector loss forward, AND the multi-output detector value-and-grad (5
gradients: g_k, g_b1, g_w2, g_b2, g_image) all compile natively (`verify=True`)
and match NumPy / finite differences
(`test_detector_value_and_grad_matches_finite_differences`).  The interpreter
and compiled paths are now equivalent for the detector.

**Runtime fix (2026-06-17):** `CPUFunctionExecutor.execute_into` now retains
the materialized numpy input arrays for the duration of the call.  Previously
`np.asarray` on a Python-scalar (Float) parameter created a temporary 0-d
array whose data pointer the memref descriptor captured as a raw `int`; with
2+ scalar parameters the temporaries were GC'd and their addresses reused,
so every scalar received the *last* one's value.  Covered by
`test_multiple_scalar_parameters_do_not_alias`.

**Next step for Phase 2:** write `examples/crater_detect_train.py` with
compiled forward + compiled value-and-grad + Python-owned optimizer (SGD).
All required compiler pieces are now in place and verified: the detector
forward, the inlined detector loss, and the multi-output value-and-grad all
compile natively and match NumPy / finite differences.  The training script
should follow `examples/crater_train.py`'s structure (per-example loss loop in
Python, optimizer state in Python) but use the *compiled* forward and
value-and-grad directly — no interpreter fallback needed.

## Phase 3: Static Batch ABI and Batched Detector Training

### Goal

Make static batched training a first-class compiler milestone.

### Tasks

- [ ] Choose a first batch size, preferably `B=8` or `B=16`.
- [ ] Ensure descriptor execution supports:
  - [ ] images `[B,1,64,64]`,
  - [ ] targets `[B,8,8,4]`,
  - [ ] scalar loss output,
  - [ ] all gradient outputs.
- [ ] Add a batched detector loss if Phase 2 used per-example calls.
- [ ] Ensure mean loss divides by `B * GridH * GridW` or by the intended weighted
  denominator.
- [ ] Add parity tests:
  - [ ] batched loss equals average of per-example losses,
  - [ ] batched gradients equal summed or averaged per-example gradients according
    to the documented convention.

### Compiler Work

- [ ] Fix descriptor ABI rank handling as needed.
- [ ] Fix AD broadcasting/accumulation bugs exposed by batch axes.
- [ ] Keep dynamic batch sizes out of scope.

### Acceptance

- [ ] Batched detector value-and-grad runs in strict CPU native mode where the
  toolchain supports it.
- [ ] Python training uses batched calls by default.
- [ ] Batch size is static and recorded in checkpoint metadata.

## Phase 4: Dense Non-AD Numeric Kernels

### Goal

Mature non-AD dense CPU/GPU compilation with image filters and PDE stencils.
These examples isolate GPU/view/fusion behavior from neural-net AD complexity.

## Phase 4A: Image Processing Pipeline

### Tasks

- [ ] Add `examples/image_filters.py` or a small Remora example plus Python driver.
- [ ] Implement:
  - [ ] Sobel edge magnitude,
  - [ ] thresholding,
  - [ ] optional blur.
- [ ] Use fixed shape first, e.g. `[128,128]`.
- [ ] Compare against NumPy/SciPy/OpenCV reference output.

### Compiler Work

- [ ] Validate convolution/window-like operations outside AD.
- [ ] Validate boolean masks and comparisons.
- [ ] Validate CPU native and GPU path if available.

### Acceptance

- [ ] Sobel and threshold outputs match reference within tolerance.
- [ ] CPU/GPU parity is recorded if GPU support is available.

## Phase 4B: PDE Stencil

### Tasks

- [ ] Add a heat-equation single-step example:

```text
step(grid [64,64]) -> [64,64]
```

- [ ] Let Python run `T` steps initially.
- [ ] Compare against NumPy.
- [ ] Later move fixed-step iteration into Remora using scan or an explicit loop
  construct if available.

### Compiler Work

- [ ] Validate `rotate`, `subarray`, `take`, `drop`, or equivalent boundary
  patterns.
- [ ] Measure allocation behavior over repeated Python-driven steps.
- [ ] Add GPU parity once the CPU result is stable.

### Acceptance

- [ ] One-step stencil matches NumPy.
- [ ] Multi-step Python orchestration remains stable and finite.
- [ ] A status note records CPU and GPU timings where available.

## Phase 5: Real Lunar Tile Pipeline for the Detector

### Goal

Replace synthetic detector data with real image/catalog-derived dense targets
while keeping Remora's computation static and dense.

This phase is intentionally Python-heavy.  If data loading, projection, or
normalization begins to consume time without compiler progress, pause this phase
and complete Phase 4 first.  Phase 4's dense non-AD kernels provide compiler
value without requiring the real lunar data pipeline to be finished.

### Python Tasks

- [ ] Add data-loading helpers for:
  - [ ] raster tiles or global raster windows,
  - [ ] crater catalog CSV, starting with the existing `data/craters_v71.csv`
    if its coordinate columns match the selected raster source,
  - [ ] coordinate projection,
  - [ ] train/validation/test split persistence,
  - [ ] normalization statistics from training split only.
- [ ] Add CLI options to `examples/crater_detect_train.py`:
  - [ ] `--data`,
  - [ ] `--catalog`,
  - [ ] `--split`,
  - [ ] `--tile-size`,
  - [ ] `--grid-size`,
  - [ ] `--dry-run-data`,
  - [ ] `--synthetic`.
- [ ] Generate dense `[8,8,4]` targets for `64x64` tiles first.
- [ ] Report:
  - [ ] image shape and dtype,
  - [ ] target shape and dtype,
  - [ ] crater count,
  - [ ] assigned crater count,
  - [ ] ignored crater count,
  - [ ] grid conflicts,
  - [ ] normalization stats.
- [ ] Add visualization helpers for decoded target and prediction overlays.

### Remora Tasks

- [ ] Reuse the synthetic detector model and loss unchanged.
- [ ] Do not add geospatial transforms to Remora.
- [ ] Do not add NMS to Remora.

### Tests

- [ ] File-backed synthetic dataset test.
- [ ] Target assignment test from known catalog coordinates.
- [ ] Dry-run data test that does not compile Remora.

### Acceptance

- [ ] Dry run works on a small real or file-backed dataset.
- [ ] One compiled training epoch runs on a small subset.
- [ ] Validation overlay images look plausible enough for manual review.

## Phase 6: Pyramid and Overlapping-Tile Inference

### Goal

Recover practical multi-scale crater detection behavior using Python
orchestration around the small Remora detector.

### Python Tasks

- [ ] Add image pyramid generation.
- [ ] Add overlapping tile extraction with configurable stride.
- [ ] Batch tile inference through compiled Remora.
- [ ] Decode predictions:

```text
local_x = (gx + dx) * cell_width
local_y = (gy + dy) * cell_height
radius = radius_scale * exp(log_radius)
score = sigmoid(objectness_logit)
```

- [ ] Back-project to global coordinates:

```text
global_x = (tile_x + local_x) / pyramid_scale
global_y = (tile_y + local_y) / pyramid_scale
global_r = radius / pyramid_scale
```

- [ ] Add circle NMS or center-distance/radius duplicate merging.
- [ ] Export CSV with:
  - [ ] `x`,
  - [ ] `y`,
  - [ ] `radius`,
  - [ ] `score`,
  - [ ] `scale`,
  - [ ] source tile metadata.

### Compiler Tasks

- [ ] No new compiler feature should be required.  This phase intentionally tests
  repeated inference calls and stable ABI behavior.

### Acceptance

- [ ] Same compiled detector runs across pyramid levels without recompilation when
  tile shape is unchanged.
- [ ] Overlapping tile duplicates are merged.
- [ ] Inference output can be visualized over the source raster.

## Phase 7: Scale Detector Shape and Capacity

### Goal

Move from compiler demonstration toward useful detector behavior.

### Tasks

- [ ] Scale from:

```text
[B,1,64,64] -> [B,8,8,4]
```

to:

```text
[B,1,128,128] -> [B,16,16,4]
```

- [ ] Increase convolution filters or hidden channels only after shape scaling
  compiles and trains.
- [ ] Add positive-class weighting if objectness imbalance dominates.
- [ ] Add multiple prediction slots per grid cell only if conflict diagnostics show
  this is necessary.
- [ ] Compare with a small PyTorch baseline using the exact same dense targets.

### Compiler Work

- [ ] Measure compile time, MLIR size, native step time, and memory use.
- [ ] Improve matmul/dot recognition if dense heads become bottlenecks.
- [ ] Start buffer reuse work if intermediate allocations dominate.

### Acceptance

- [ ] `128x128` detector compiles with bounded IR size.
- [ ] Training loss decreases on synthetic and small real-data subsets.
- [ ] Validation metrics improve or failure modes are documented.

## Phase 8: Memory-Pressure Examples

### Goal

Use N-body and a differentiable renderer toy to expose large-intermediate and
buffer-planning problems without adding irregular data yet.

## Phase 8A: N-Body

### Tasks

- [ ] Add force computation:

```text
forces(positions [128,3], masses [128]) -> [128,3]
```

- [ ] Use softened distances and mask self-interactions.
- [ ] Compare against NumPy.
- [ ] Later add one integration step.

### Compiler Work

- [ ] Measure materialization of `[N,N,3]` intermediates.
- [ ] Add fusion or tiling only after measurements show the bottleneck.

### Acceptance

- [ ] Result matches NumPy.
- [ ] Memory and time are recorded for multiple `N` values.

## Phase 8B: Differentiable Renderer Toy

### Tasks

- [ ] Render soft disks:

```text
render(circles [16,3]) -> [64,64]
loss(circles, target [64,64]) -> Float
```

- [ ] Keep the first renderer fixed at `N=16` circles.  Try `N=32` only after
  memory and compile-time behavior are recorded for `N=16`.
- [ ] Use smooth coverage functions so AD is meaningful.
- [ ] Optimize circle parameters from a synthetic target.

### Compiler Work

- [ ] Validate AD over geometry parameters.
- [ ] Measure `[H,W,N]` intermediate pressure.

### Acceptance

- [ ] Loss decreases in a short optimization run.
- [ ] Memory behavior is documented.

## Phase 9: Static Padded Irregular Workloads

### Goal

Introduce controlled irregularity with fixed-size padded arrays before dynamic
or sparse representations.

## Phase 9A: Molecular Dynamics Neighbor Lists

### Tasks

- [ ] Python builds fixed-width neighbor lists:

```text
atoms:       [N,3]
neighbor_ix: [N,K]
mask:        [N,K]
```

- [ ] Remora computes masked forces:

```text
forces(atoms, neighbor_ix, mask) -> [N,3]
```

### Compiler Work

- [ ] Validate gather over index arrays.
- [ ] Validate masked reductions.
- [ ] Add user-facing scatter-add only if needed for symmetric accumulation.

### Acceptance

- [ ] Force computation matches NumPy reference for fixed `N,K`.
- [ ] Padded masked entries do not contribute.

## Phase 9B: Small Graph Propagation

### Tasks

- [ ] Start with dense adjacency for tiny graphs if scatter is not ready.
- [ ] Then implement edge-list PageRank step:

```text
pagerank_step(edges [E,2], out_degree [N], rank [N]) -> [N]
```

### Compiler Work

- [ ] Add or validate scatter-add/segmented sum.
- [ ] Keep graph sizes static.

### Acceptance

- [ ] One PageRank step matches NumPy.
- [ ] Python owns convergence iteration.

## Phase 10: Later Sparse, Dynamic, and Library-Backed Workloads

### Goal

Defer the most complex examples until dense, batched, GPU, and padded-irregular
workloads are stable.

### Candidate Work

- [ ] Sparse PageRank and graph analytics.
- [ ] Tomography with interpolation-heavy backprojection.
- [ ] FFT/signal processing with complex numbers and external library integration.
- [ ] Production-scale crater recognition.

### Entry Criteria

Do not begin this phase until:

- [ ] Static batched detector training is stable.
- [ ] At least one dense non-AD GPU example has CPU/GPU parity.
- [ ] Memory-pressure measurements exist for N-body or differentiable rendering.
- [ ] Gather/scatter or segmented reduction support has at least one passing example.

## Documentation Deliverables

After each phase, update or create:

- [ ] A status section in this document or a phase-specific status file.
- [ ] Command lines used for validation.
- [ ] Test names added or changed.
- [ ] Compile time, execution time, and MLIR size where relevant.
- [ ] Current limitations and next recommended step.

## Suggested Immediate Work Order

Start here:

- [x] Run and document current crater classifier tests.
- [x] Implement logistic regression value-and-grad with static batch or document
   the exact compiler gap blocking it.
- [ ] Implement synthetic crater grid target assignment and decoding in Python.
- [ ] Implement detector loss for fixed predictions before training the detector.
- [ ] Add detector value-and-grad and train on synthetic data.
- [ ] Harden static batch ABI using logistic regression and detector parity tests.

This sequence keeps failures localized.  If the detector breaks, logistic
regression should reveal whether the issue is general batched AD.  If PDE or
image filters break, the issue is dense non-AD lowering rather than training.
If N-body breaks, the issue is memory pressure rather than the crater task.
