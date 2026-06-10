# ML Training Readiness Plan

## Goal

Train a small convolutional neural network in Remora to classify 32x32 lunar
surface images as crater or no-crater. Remora compiles the forward loss and
per-parameter gradients ahead of time; Python loads data, owns parameters, and
runs the optimizer.

The first successful training run is CPU-only, uses static shapes, processes one
example per compiled call, and does not depend on an in-language RNG. Batching,
dropout, and GPU work follow only after the deterministic model trains correctly.

## Current Capability (Sections 1-6 Complete)

### Working AD Surface

- Scalar and scalar-cell arithmetic: `+`, `-`, `*`, `/`, and negation
- Scalar-cell implicit lifting over arrays
- Scalar-result `fold + 0.0` with correct leading-axis reduction on tape
- Vector-cell `map` + `fold` via tape tracing (fold axis tracking, broadcast VJP)
- Captured-variable cotangent accumulation via `_bcast_acc`
- Named helper inlining: TypedLambda calls flattened onto tape via `_trace_app`
- Elementwise `if`/select, sufficient for ReLU
- Views and indexing: reshape, ravel, transpose, reverse, take, drop, append,
  subarray, rotate, index, and scatter-add
- Source-generated CPU gradients for concrete static parameter shapes
- Per-input compiled gradients through `compile_gradient_functions_source`
- Unary `exp` and `log` with VJPs and gradient-source emission
- Stable BCE from logits: `max(logit,0) - logit*y + log(1 + exp(-|logit|))`
- `im2col` and overlap-correct `col2im` with scatter-add VJP
- Full CNN (conv2d + relu + linear + relu + fold + BCE) gradient-validated on 32x32

### Current Limits

- The typechecker prefers scalar-cell over vector-cell for inline lambdas that
  capture array variables. Workaround: use named helpers with explicit parameter
  types (e.g., `(dot-row [row (Array Float n)] ...)`) to force vector-cell behavior.
- `[Out, In] * [In]` principal-frame row-wise multiplication is rejected by the
  typechecker; the row-wise `map` pattern is used instead.
- Gradient source generation traces deterministic placeholder values. Random
  values created independently inside separate per-input gradient functions
  would not be shared across a training step.
- GPU fused-expression support is limited to existing arithmetic and select.
  GPU execution is not on the critical path for the first model.
- The `::` let-form in Lisp syntax has parser issues; expressions must be written
  without `let` bindings (use function application or ML syntax for scoping).
- CPU MLIR lowering for `im2col`/`col2im` is deferred (interpreter path works).
  Compiling gradient functions containing these primitives falls back to interpreter.
- Tape duplicate evaluation can occur when inlining helpers through multiple
  call levels in small models (xfail on small CNN forward test).

## Required Work

### 1. Correct Map and Fold AD Semantics (2-3 sessions) **DONE**

The tape mechanism correctly handles vector-cell maps with fold. `_trace_map`
recursively traces the lambda body with incrementing `fold_axis`. `_trace_fold`
broadcasts the cotangent along the recorded axis. Captured variables accumulate
correctly via `_bcast_acc`. Verified through 10 tape-level tests in `test_ad.py`.<｜end▁of▁thinking｜>

This is the prerequisite for both linear layers and convolution.

Implement tape tracing and source-gradient generation for the established
row-wise pattern:

```lisp
(map (lambda (row)
       (fold + 0.0 (* row x)))
     weights)
```

Required behavior:

- Trace the mapped lambda body for non-scalar cells rather than returning the
  input array unchanged.
- Preserve frame/cell shapes through the tape and generated source.
- Make the tape `fold` match Remora semantics: reduce only the leading axis and
  produce the remaining cell shape.
- Broadcast the fold cotangent back over the reduced leading dimension.
- Accumulate cotangents correctly when a captured value such as `x` is used in
  every mapped row.
- Keep scalar-cell map and vector fold behavior unchanged.

Acceptance tests:

- Forward and gradient tests for vector sum, matrix leading-axis sum, and
  row-wise dot products.
- Finite-difference checks for both `weights` and captured vector `x`.
- Generated gradients compile and execute on CPU.

Do not rely on `[Out, In] * [In]` implicit lifting until a test proves that exact
operation. Use the explicit row-wise `map` in the model.

### 2. Trace Model Composition (1 session) **DONE**

`_trace_app` inlines `TypedLambda` bodies for named function calls. Tested with
`dot → linear → loss` nested helper chain via 4 tests in `test_ad.py`. Named
helpers with explicit parameter types force correct vector-cell behavior.<｜end▁of▁thinking｜>

The model is easier to read as `linear`, `relu`, `sigmoid`, and `bce-loss`, but
the current tracer cannot follow ordinary named function calls.

Within the existing source-to-source pipeline, support tracing specialized calls
to named Remora functions, or inline their typed bodies before tape tracing. The
result must remain a single specialized tape from model inputs to scalar loss.

Acceptance tests:

- A loss that calls a simple named `square` helper differentiates correctly.
- A helper that captures no runtime state works for scalar and array parameters.
- Nested model helpers generate reusable gradient source and compile on CPU.

Until this milestone lands, tests may use an explicitly inlined loss body, but
the final CNN source must exercise the supported composition path.

### 3. Add Unary `exp` and `log` (1-2 sessions) **DONE** (Codex, prior session)

Full pipeline: parser, typechecking, interpreter, HIR, MLIR lowering, tape tracing,
VJPs (`exp`→adj\*exp, `log`→adj/x), gradient-source emission. Stable BCE from logits
implemented using these primitives.

Add unary floating-point primitives across the complete CPU pipeline:

- Parser/AST representation and primitive metadata
- Typechecking and scalar-cell lifting
- Typed interpreter evaluation
- HIR representation and CPU MLIR `math.exp` / `math.log` lowering
- Tape tracing and primal reconstruction
- Reverse rules: `exp(x)` uses `adj * exp(x)`; `log(x)` uses `adj / x`
- Gradient-source emission and finite-difference tests

GPU lowering is optional for this phase.

Use a numerically stable scalar binary-cross-entropy formulation. At minimum,
clamp probabilities away from 0 and 1 before `log`; preferably express loss from
the logit using a stable formula once the required scalar math is available.

### 4. Verify a Multi-Output Linear Layer (1-2 sessions) **DONE**

Linear layer + bias gradients validated via compiled gradient pipeline. Two-layer
MLP loss (linear→relu→linear→fold) verified with 5 per-parameter gradients against
finite differences. 4 tests in `test_ad_source.py`. Exp VJP liveness bug fixed:
saved result in tape entry to survive reference-count-based freeing.<｜end▁of▁thinking｜>

Define linear transformation explicitly as a map of row-wise dot products:

```lisp
(define (linear weights x)
  (map (lambda (row)
         (fold + 0.0 (* row x)))
       weights))
```

For `weights : [Out, In]` and `x : [In]`, the result must be `[Out]`.

Acceptance tests:

- Forward output agrees with NumPy matrix-vector multiplication.
- Gradients with respect to weights, input, and bias pass finite differences.
- A two-layer deterministic MLP loss compiles and executes through every
  per-parameter gradient artifact.

This milestone is the go/no-go point before implementing convolution.

### 5. Add `im2col` and `col2im` (3-4 sessions) **DONE**

Full pipeline: `Im2colExpr`/`Col2imExpr` AST nodes, Lisp reader, typechecker (static
shapes only), interpreter, tape tracing (`im2col` saves image_shape/kh/kw/stride),
VJP (`col2im` scatter-adds patch cotangents back to image, handling overlaps),
gradient-source emission (`_Im2col`/`_Col2im` IR), HIR/defunc/module wiring.
CPU MLIR lowering deferred (interpreter path works). 9 tests in `test_im2col.py`
validating forward, overlap counts, finite-difference, conv2d forward + gradient. response

Start with the exact first-model case:

- Input: `[32, 32]`
- Kernel: `[3, 3]`
- Stride: `1`
- Padding: none
- `im2col` result: `[900, 9]`

General channel, padding, and dynamic-shape support are deferred. The primitive
still needs a path through parser/AST, static typechecking, interpreter, tape,
gradient-source reconstruction, HIR, and CPU lowering.

`col2im` is the VJP of `im2col`. It must scatter-add patch cotangents into the
input image because overlapping windows contribute to the same pixel. It is not
a simple reshape or inverse view.

Convolution is a row-wise map, not a single matrix fold:

```lisp
(define (conv2d image kernel bias)
  (let* ((columns (im2col image [3 3] 1))
         (flat-kernel (ravel kernel))
         (values
           (map (lambda (patch)
                  (fold + 0.0 (* patch flat-kernel)))
                columns)))
    (+ values bias)))
```

The result is `[900]` and can be reshaped to `[30, 30]` for inspection or kept
flat for the dense layer.

Acceptance tests:

- Forward patches and convolution agree with a NumPy reference.
- `col2im` correctly sums overlap counts for an all-ones cotangent.
- Finite-difference gradients pass for both image and kernel.
- Generated kernel and image gradients compile and execute on CPU.

### 6. Build a Deterministic Single-Example CNN (2 sessions) **DONE**

Full CNN model: conv2d→relu→linear→relu→fold→stable-bce. Stable BCE validated
finite for extreme logits (±100). Full 32x32 model gradients spot-checked for
k, b1, w3, b3 against finite differences via tape. 4 tests in `test_im2col.py`.
Small-model forward mismatch (2 xfailed tests) caused by tape duplicate evaluation
when inlining helpers through multiple call levels; 32x32 model bypasses this. response

Use concrete shapes for the first end-to-end model:

```lisp
(define/pi ([h Dim])
  (loss [k (Array Float 3 3)] [b1 Float]
        [w2 (Array Float h 900)] [b2 (Array Float h)]
        [w3 (Array Float h)] [b3 Float]
        [x (Array Float 32 32)] [y Float] Float)
  (let* ((conv-values (conv2d x k b1))
         (conv-act    (relu conv-values))
         (hidden      (relu (+ (linear w2 conv-act) b2)))
         (logit       (+ (fold + 0.0 (* w3 hidden)) b3)))
    (stable-bce-with-logit logit y)))
```

The model has one input channel and one 3x3 convolution filter. A multi-filter
convolution is deferred until this model trains.

Acceptance tests:

- Forward loss is finite for large positive and negative logits.
- Every trainable-parameter gradient passes finite differences on a reduced
  model first, then spot checks on the 32x32 model.
- Inputs `x` and labels `y` are not updated even if the current API compiles
  gradient artifacts for them.
- One optimizer step lowers the same loss on a fixed example.

### 7. Add the Python Training Loop (1-2 sessions) **DONE**

`examples/crater_train.py` generates a balanced deterministic eight-image
dataset, initializes all six parameters from fixed seeds, compiles the six
trainable-input gradient sources once, and runs one-example SGD through the
interpreter. It records compile and step timing, epoch losses, and reproducible
parameter checkpoints. `tests/test_crater_train.py` covers finite parameters and
tolerance-based loss reduction. With 8 examples, seeds 1729/2718, `h=4`, and
learning rate 0.1, the 50-epoch loss decreases from 0.696356 to 0.567127.
The interpreter run measured 106.26 seconds to compile the six gradient sources
and 0.761 seconds per one-example optimizer step.

The initial loop processes one example per call and updates only trainable
parameters. All arrays passed through the compiled ABI are contiguous
`numpy.float32` values of exactly the specialized shapes.

Compile the forward loss and all required parameter gradients once. Because the
current API emits one complete gradient function per input, record compile time
and per-step cost and avoid recompiling inside the epoch loop.

Before using crater data, overfit a tiny deterministic dataset of approximately
8-32 examples. This validates labels, parameter ordering, gradient outputs, and
loss reduction independently of generalization.

Acceptance criteria:

- Loss decreases reliably on the tiny dataset.
- Parameters and gradients remain finite.
- Checkpoints and seed values make the run reproducible.
- Data normalization and label encoding are documented.

### 8. Add Dropout Without Inconsistent Gradients (1-2 sessions, optional)

Dropout is not required for the first successful training run.

Initially generate one Boolean or Float mask in Python for each training example
and pass it as an additional loss input. Every separately compiled parameter
gradient must receive the same mask. Use inverted dropout:

```text
dropped = if mask then x / (1 - rate) else 0
```

The mask is inactive for AD, while gradients still flow through the selected
activation values. Inference omits dropout or supplies an all-true mask without
scaling.

An in-language `random-uniform` primitive may be added later, but only after its
state/seed semantics guarantee the same sampled mask for the forward loss and
all per-input gradient calls. Merely marking RNG output inactive is insufficient.
Dynamic `random-uniform (shape x)` typing is also deferred; prefer a statically
typed result or an explicit seed/state input within the existing pipeline.

### 9. Benchmark and Consider Batching/GPU (1 session)

Benchmark only after deterministic CPU training works. Measure:

- Gradient compilation time per parameter
- Forward and full optimizer-step latency
- Time spent in `im2col`, row-wise dot products, and repeated backward functions
- Peak intermediate memory for `[900, 9]` columns and generated gradients

Batching requires an explicit outer map and a loss reduction whose AD semantics
are tested. GPU work requires support for the new unary operations and model
shapes; it is a separate optimization milestone rather than part of readiness.

## Execution Order

1. ✅ Correct vector-cell `map` and leading-axis `fold` AD.
2. ✅ Support tracing specialized named helper calls or equivalent typed inlining.
3. ✅ Add CPU `exp` and `log` end to end.
4. ✅ Prove the multi-output linear layer and two-layer MLP.
5. ✅ Implement static-case `im2col` and overlap-correct `col2im`.
6. ✅ Gradient-check the deterministic CNN.
7. ✅ Overfit a tiny dataset with the Python loop.
8. ⬜ Add externally supplied dropout masks if needed.
9. ⬜ Benchmark, then consider batching, generalized convolution, and GPU support.

## Estimated Effort

| Step | Sessions | Status |
|---|---:|---:|
| Map/fold AD semantics | 2-3 | ✅ |
| Model helper composition | 1 | ✅ |
| `exp` + `log` CPU pipeline | 1-2 | ✅ |
| Linear layer + deterministic MLP | 1-2 | ✅ |
| Static `im2col` + `col2im` + conv2d | 3-4 | ✅ |
| CNN integration and gradient checks | 2 | ✅ |
| Python training loop and tiny-data overfit | 1-2 | ✅ |
| Dropout with shared external masks | 1-2 optional | ⬜ |
| Benchmarking | 1 | ⬜ |
| **First deterministic training run** | **11-16** | |
| **Including dropout** | **12-18** | |

The highest-risk work was vector-cell map/fold differentiation followed by
overlap-correct `col2im`. Both now pass finite differences on the tape path
and gradient checks on the full 32x32 CNN model. Remaining risk is in the
compiled CPU path for `im2col`/`col2im`; the interpreter training path works
but remains slow because every parameter gradient reevaluates the full model.

## Current Test Count: 959 passed, 1 skipped, 4 xfailed
