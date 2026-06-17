# Production Crater Recognition Plan

## Status

**Status: READY FOR IMPLEMENTATION REVIEW**

This plan replaces the first draft after DeepSeek's review.  The main change is
route: do **not** build a parallel crater training system from scratch.  Evolve
the existing `examples/crater_train.py`, which already proves the key pieces:

- a Remora CNN source for crater classification,
- compiled multi-output value-and-grad through `CompiledTrainingFunctions`,
- strict compiled mode and interpreter fallback,
- a Python training loop,
- checkpoint snapshots in `TrainingResult`,
- native execution from Python with NumPy arrays,
- compiled/interpreted gradient parity.

The production example should harden and scale that existing path.

## Responsibility Split

Remora is responsible for:

- [ ] Defining the neural-net architecture.
- [ ] Defining forward inference.
- [ ] Defining binary classification loss.
- [ ] Defining value-and-grad for all trainable parameters.
- [ ] Compiling functions that Python can call repeatedly with NumPy arrays.

Python is responsible for:

- [ ] Loading image tiles and labels.
- [ ] Splitting train/validation/test sets.
- [ ] Normalization and augmentation.
- [ ] Calling compiled Remora functions repeatedly.
- [ ] Optimizer updates and optimizer state.
- [ ] Metrics, checkpoints, logs, and prediction outputs.

NumPy arrays remain the interop format.  The current compiled descriptor ABI
already accepts NumPy arrays for static shapes.

## Data Assumptions

These assumptions are intentionally concrete so implementation can begin.  They
can be adjusted before coding if the real dataset differs.

- [ ] Input data is pre-tiled lunar orbital imagery.
- [ ] Each tile is grayscale `float32`.
- [ ] First real-data target tile shape is `(128, 128)`.
- [ ] First batched target shape is `(16, 128, 128)`.
- [ ] Labels are binary crater-present values, shape `(16,)`, `float32`.
- [ ] Python normalizes pixels to either `[-1, 1]` or train-split
  standardized values.
- [ ] Python owns augmentation: flips, rotations, noise, contrast jitter.
- [ ] Python owns train/validation/test split persistence.
- [ ] Dataset storage is initially `.npz` shards or image files plus a CSV
  manifest.  Choose one during Phase 1.

Later work may add segmentation masks or crater-center heatmaps, but the MVP is
binary tile classification.

## MVP Success Criteria

The MVP is complete when:

- [ ] A PyTorch baseline using the same dataset split demonstrates that the
  data/task framing is learnable.
- [ ] `examples/crater_train.py` can train from a real dataset path, not only
  the built-in toy dataset.
- [ ] The Remora model runs on `(128, 128)` tiles.
- [ ] Static batch size 16 works end to end for forward, loss, and
  value-and-grad.
- [ ] Strict compiled mode trains at least one epoch on a small real-data
  subset.
- [ ] Training and validation loss decrease on a real subset.
- [ ] Python reports accuracy, precision, recall, F1, PR AUC or ROC AUC, and
  confusion matrix.
- [ ] Checkpoints contain parameters, optimizer state, shape metadata, Remora
  source hash, and normalization metadata.
- [ ] A checkpoint can be restored and used for validation or prediction.

## Staged Scope

The plan is staged.  CPU static-batch work is the correctness and ABI staging
ground; GPU work is part of this same roadmap because production-scale CNN
training is unlikely to be viable on CPU alone.

MVP scope:

- [ ] Real dataset integration.
- [ ] `128x128` single-example compiled training.
- [ ] Static batch size 16 on CPU.
- [ ] Production-shaped metrics, checkpoints, and Python orchestration.

Later phases in this plan:

- [ ] GPU ABI and runtime support for the same batched crater model.
- [ ] GPU lowering for the operations used by the crater CNN.
- [ ] CPU/GPU parity and production-throughput validation.

Still deferred out of this plan:

- [ ] Dynamic batch sizes.  Use static batch first, including on GPU.
- [ ] Segmentation/detection heads.
- [ ] Large model architecture search.
- [ ] Full packaging or cloud training infrastructure.

## Phase 1: Real Dataset Integration in Existing Training Script

**Goal:** teach `examples/crater_train.py` to train from Python-loaded data
while preserving the existing tiny deterministic dataset for tests.

### Tasks

- [ ] Keep `make_tiny_dataset()` for fast CI and compiler smoke tests.
- [ ] Add a dataset manifest format:
  - option A: `.npz` shards containing `images` and `labels`,
  - option B: CSV with image paths and labels.
- [ ] Add a `CraterDataset` or equivalent loader in `examples/crater_train.py`
  or a small adjacent module.
- [ ] Add `--data`, `--split`, `--validation-split`, and `--seed` CLI options.
- [ ] Add `--dry-run-data` to print batch shapes, dtype, label balance, and
  normalization stats without compiling Remora.
- [ ] Add train/validation/test split persistence.
- [ ] Add Python-side normalization computed from the training split only.
- [ ] Add optional Python-side augmentation for training batches only.
- [ ] Add tests using a synthetic file-backed dataset.
- [ ] Put shared data-loading and metric helpers somewhere both Remora and
  PyTorch baselines can use without duplicating split logic.

### Acceptance

- [ ] `uv run python examples/crater_train.py --dry-run-data --data ...`
  prints deterministic dataset stats.
- [ ] Tiny in-memory dataset tests still pass.
- [ ] File-backed synthetic dataset tests validate shape `(N, 128, 128)` and
  label shape `(N,)`.
- [ ] No Remora compiler changes are required for this phase.

### Risk Mitigations

- Risk: real labels are imbalanced.
  - [ ] Dry-run output reports class balance.
  - [ ] Loader supports class weighting or balanced sampling flag.
- Risk: train/validation leakage through augmented variants.
  - [ ] Split is done before augmentation.

## Phase 1.5: PyTorch Data and Model Baseline

**Goal:** understand the crater dataset and establish a reference model before
asking Remora to carry the full training workload.

This is a baseline and diagnostic tool, not the final production path.  It
should share the same dataset loader, splits, normalization, augmentation
policy, and metric code that Remora will use.

### Tasks

- [ ] Add `examples/crater_pytorch_baseline.py`.
- [ ] Use the Phase 1 dataset loader and deterministic splits.
- [ ] Use the same `(128, 128)` grayscale tile input contract.
- [ ] Train a modest PyTorch CNN comparable to the planned Remora Model A.
- [ ] Support the same train/validation/test split names.
- [ ] Log training and validation loss per epoch.
- [ ] Compute accuracy, precision, recall, F1, PR AUC or ROC AUC, and confusion
  matrix.
- [ ] Save false-positive and false-negative examples for inspection.
- [ ] Save baseline checkpoint and metrics JSON.
- [ ] Record throughput in examples/sec for CPU and GPU if PyTorch GPU is
  available.

### Acceptance

- [ ] Baseline training loss decreases on synthetic crater data.
- [ ] Baseline training loss decreases on a small real-data subset.
- [ ] Validation metrics are stable enough to guide Remora work.
- [ ] False-positive/false-negative outputs reveal whether tile classification
  is a reasonable framing.
- [ ] The Remora plan is updated if the baseline shows that a different label
  target, normalization strategy, class weighting, or model shape is needed.

### Risk Mitigations

- Risk: PyTorch baseline becomes a competing implementation.
  - [ ] Keep PyTorch math isolated to the baseline script.
  - [ ] Shared code is limited to data loading, splitting, augmentation,
    metrics, and output formatting.
- Risk: Remora chases an unlearnable dataset.
  - [ ] Do not start Model A Remora scaling until the baseline shows the data
    has signal or identifies what must change.
- Risk: metrics look good only because of leakage.
  - [ ] Baseline uses the same persisted split manifest that Remora will use.

## Phase 2: Scale Existing Single-Example CNN to 128x128

**Goal:** prove the current compiled single-example path scales from `32x32`
to `128x128` before adding batch complexity, using lessons from the PyTorch
baseline about normalization, class weighting, and model size.

This phase should evolve the existing `_CNN_FULL_LISP_SRC` pattern rather than
creating an unrelated model file.

### Tasks

- [ ] Parameterize image size in the crater CNN source generation or add a
  clearly named `128x128` source variant.
- [ ] Choose the initial Remora architecture based on the smallest PyTorch
  baseline model that learns the task.
- [ ] Update parameter shape helpers for `128x128`.
- [ ] Keep the initial architecture close to the current working model:
  one compact convolution, ReLU, optional pooling/downsampling, dense head,
  BCE loss.
- [ ] Compile strict native forward/loss/value-and-grad for one `128x128`
  example.
- [ ] Run compiled/interpreted parity on a small deterministic `128x128`
  example.
- [ ] Measure descriptor MLIR size, `linalg.generic` count, `scf.for` count,
  compile time, and first-step time.
- [ ] Add a benchmark case for `crater-cnn-gradient-128`.

### Acceptance

- [ ] PyTorch baseline has already established that the dataset/task framing is
  learnable, or this phase explicitly uses synthetic data only.
- [ ] Strict compiled single-example `128x128` training step runs.
- [ ] Compiled/interpreted gradients agree within tolerance.
- [ ] Descriptor MLIR size remains bounded by loop structure, not pixel count.
  Record the exact byte count in this doc or a status file.
- [ ] Compile time and step time are recorded.
- [ ] Existing `32x32` crater tests continue to pass.

### Risk Mitigations

- Risk: dense layer explodes after `128x128` convolution.
  - [ ] Add pooling or global average pooling before dense if flattened
    features are too large.
- Risk: typechecking or lowering time regresses.
  - [ ] Record phase timing before declaring acceptance.

## Phase 3: Static Batch ABI and Batched Training

**Goal:** implement the central compiler feature needed for production
training: static batch size 16 for images and labels.

This is the hard technical phase.  Treat it as a compiler milestone, not a
minor training-loop task.

### Design Constraints

- [ ] First implementation uses static batch size: `(16, 128, 128)`.
- [ ] Dynamic batch sizes are deferred.
- [ ] Python may drop or pad the final partial batch.
- [ ] Remora computes mean batch loss and gradients over the batch.
- [ ] Python still applies optimizer updates.

### Compiler Tasks

- [ ] Audit descriptor lowering for rank-3 inputs and rank-1 outputs.
- [ ] Audit `CPUFunctionExecutor.execute_into()` for batched input/output
  descriptor validation.
- [ ] Audit Pair/multi-output wrapper lowering for many batched-gradient
  outputs.
- [ ] Add tests for compiled functions with input shape `(16, 128, 128)`.
- [ ] Add tests for outputs shape `(16,)`.
- [ ] Add tests for scalar mean reduction over batch.
- [ ] Add tests for multi-output gradients where some gradients are high-rank
  arrays.
- [ ] Ensure cache keys include static batch shapes through parameter types.

### Remora Model Tasks

- [ ] Convert forward pass to accept `images: (16, 128, 128)`.
- [ ] Convert labels to `labels: (16,)`.
- [ ] Compute per-example logits `(16,)`.
- [ ] Compute per-example BCE `(16,)`.
- [ ] Reduce to scalar mean loss.
- [ ] Generate one value-and-grad function for all trainable parameters.

### Measurements Required

Before Phase 3 can be marked done, collect:

- [ ] Descriptor MLIR bytes for batch 1 and batch 16.
- [ ] `tensor.extract` count.
- [ ] `tensor.insert` count.
- [ ] `scf.for` count.
- [ ] `linalg.generic` count.
- [ ] Function preparation time.
- [ ] CPU pipeline time.
- [ ] Native link/load time.
- [ ] Mean compiled step time over at least 20 batches.

### Acceptance

- [ ] `batch=16`, `128x128` compiled forward runs.
- [ ] `batch=16`, `128x128` compiled value-and-grad runs.
- [ ] Compiled/interpreted parity passes on a tiny batched example.
- [ ] IR size is acceptable and recorded.  If IR size scales with
  `batch * pixels` due to unrolled operations, stop and fix lowering before
  continuing.
- [ ] Python training loop can run at least 10 compiled batched steps.

### Risk Mitigations

- Risk: batched convolution IR grows too quickly.
  - [ ] Acceptance requires recorded IR counts for batch 16.
- Risk: Pair output ABI becomes unwieldy for many gradients.
  - [ ] Add a focused multi-output ABI test before integrating the full model.
- Risk: interpreted parity is too slow for batch 16.
  - [ ] Allow parity on batch 2 or smaller image size, plus finite-difference
    checks on tiny shapes.

## Phase 4: Upgrade Model and Training Loop for Production MVP

**Goal:** turn the batched `128x128` path into a credible binary crater
classifier while keeping the architecture modest.

### Model Architecture

Start from the existing crater CNN and evolve only as needed:

- [ ] Input: `(16, 128, 128)`.
- [ ] Convolution block 1: small number of filters, ReLU.
- [ ] Pooling or stride to reduce spatial size.
- [ ] Optional convolution block 2 if Phase 3 performance is acceptable.
- [ ] Global average pooling or compact dense head.
- [ ] Binary logit per example.
- [ ] Mean BCE-with-logits loss.

Avoid a large flattened dense layer unless measurements show it is safe.
Use the PyTorch baseline to justify any architecture expansion.

### Python Training Tasks

- [ ] Add SGD with momentum.
- [ ] Add Adam or AdamW.
- [ ] Add gradient clipping.
- [ ] Add checkpoint save/restore:
  parameters, optimizer state, epoch/step, RNG seeds, normalization stats,
  Remora source hash, parameter shapes.
- [ ] Add validation loop using compiled inference.
- [ ] Add prediction output for held-out examples.

### Metrics

- [ ] Accuracy.
- [ ] Precision.
- [ ] Recall.
- [ ] F1.
- [ ] ROC AUC or PR AUC.
- [ ] Confusion matrix.
- [ ] Loss curves.
- [ ] Optional false-positive/false-negative image grids.

### Acceptance

- [ ] Training loss decreases on synthetic crater data.
- [ ] Training loss decreases on a small real-data subset.
- [ ] Remora metrics are compared against the PyTorch baseline on the same
  split, with any large gaps investigated.
- [ ] Validation metrics are computed and written to disk.
- [ ] Checkpoint restore reproduces the next step with fixed seeds.
- [ ] `--evaluate-checkpoint` runs compiled inference on validation/test data.
- [ ] `--predict` writes tile IDs and crater probabilities.

### Risk Mitigations

- Risk: real dataset is noisy or imbalanced.
  - [ ] Report PR AUC in addition to accuracy.
  - [ ] Support class weights or balanced sampling.
- Risk: Python optimizer copies dominate runtime.
  - [ ] Measure time spent in Remora call, optimizer update, and data loading.
- Risk: first compile time is too slow for iteration.
  - [ ] Require cache-hit startup measurement in logs.

## Phase 5: Documentation and Reproducibility

**Goal:** make the MVP runnable by another developer.

### Tasks

- [ ] Document dataset layout.
- [ ] Document preprocessing and normalization.
- [ ] Document all training commands.
- [ ] Document checkpoint format.
- [ ] Add a synthetic smoke command that works without real data.
- [ ] Add a real-data command template.
- [ ] Record toolchain versions and native cache keys in logs.

### Acceptance

- [ ] Fresh checkout can run synthetic smoke training.
- [ ] Real-data training command is documented with expected files.
- [ ] Checkpoint metadata is sufficient to reproduce model source and shapes.

## Phase 6: GPU ABI and Runtime for Batched CNN Training

**Goal:** extend Remora's GPU path enough to launch the same static-batched
crater model used by the CPU MVP.

This is not optional for production-scale training.  It is placed after the CPU
MVP because the CPU path defines the exact model signatures, array layouts,
multi-output gradient structure, and parity tests the GPU path must match.

### Entry Criteria

- [ ] Phase 3 static batch CPU value-and-grad works.
- [ ] Phase 4 training loop can train on synthetic and small real CPU subsets.
- [ ] The model operation inventory is frozen for Model A:
  convolution/im2col, ReLU, pooling/global pooling, dense, BCE, reductions.
- [ ] CPU reference outputs are available for forward, loss, and gradients on
  tiny deterministic batches.

### Current GPU Gaps to Close

- [ ] GPU function ABI supports only 1-2 input parameters; crater training needs
  many model parameters plus images and labels.
- [ ] GPU ABI does not support multi-output value-and-grad returns.
- [ ] GPU lowering does not handle the `scf.for` loops used by compact
  `im2col`/`col2im`.
- [ ] GPU lowering does not yet cover the required matmul/convolution path.
- [ ] GPU runtime wrappers need reliable NumPy/device memory transfer for many
  inputs and outputs.

### Tasks

- [ ] Define the GPU descriptor ABI for static batch crater functions.
- [ ] Support at least the full Model A input list:
  parameters, batch images, batch labels.
- [ ] Support multiple output buffers for all gradients and scalar loss.
- [ ] Add GPU runtime support for allocating/copying all inputs and outputs.
- [ ] Add compile/load/call API parallel to `CPUFunctionExecutor` where
  possible.
- [ ] Add shape and dtype validation for GPU calls.
- [ ] Add cache keys that include target `gpu-nvidia` and relevant toolchain
  versions.
- [ ] Add clear errors for unsupported GPU toolchains or devices.

### Acceptance

- [ ] A compiled GPU function accepts at least 4 input arrays and 2 output
  arrays in a test.
- [ ] A compiled GPU value-and-grad-like test returns multiple outputs.
- [ ] GPU launch works repeatedly from Python without recompiling.
- [ ] CPU/GPU transfer overhead is measured separately from kernel time.

### Risk Mitigations

- Risk: trying to support arbitrary dynamic signatures will balloon scope.
  - [ ] Support only static Model A signatures first.
- Risk: GPU runtime bugs hide compiler bugs.
  - [ ] Add tiny elementwise/multi-output ABI tests before CNN kernels.

## Phase 7: GPU Lowering for Model A Operations

**Goal:** lower the operations used by the batched crater CNN to GPU-compatible
kernels.

### Tasks

- [ ] Decide convolution strategy for GPU:
  compact loop lowering, direct convolution kernel, or im2col plus matmul.
- [ ] Lower or transform compact `im2col` loops for GPU execution.
- [ ] Add GPU lowering for ReLU and simple elementwise maps over batched tensors.
- [ ] Add GPU lowering for reductions used by BCE and mean batch loss.
- [ ] Add GPU lowering for pooling/global average pooling.
- [ ] Add GPU lowering for dense layer/matmul pattern used by Model A.
- [ ] Add GPU lowering for `col2im` or equivalent convolution VJP path.
- [ ] Add GPU support for value-and-grad output gradients.

### Measurements Required

- [ ] GPU descriptor/module size for forward.
- [ ] GPU descriptor/module size for value-and-grad.
- [ ] Compile time.
- [ ] Host-to-device transfer time.
- [ ] Kernel execution time.
- [ ] Device-to-host transfer time.
- [ ] End-to-end step time.

### Acceptance

- [ ] GPU forward logits match CPU within tolerance on a tiny batch.
- [ ] GPU scalar loss matches CPU within tolerance.
- [ ] GPU gradients match CPU within tolerance on a tiny batch.
- [ ] GPU value-and-grad compiles for the static batch-16 Model A signature.
- [ ] GPU step time is measured against CPU step time.

### Risk Mitigations

- Risk: direct convolution is too large to implement immediately.
  - [ ] Start with the smallest operation-preserving lowering that passes
    parity, then optimize.
- Risk: GPU compile path cannot handle generated loop structure.
  - [ ] Add an early stop: compile isolated batched `im2col` before full model.

## Phase 8: GPU Training Integration and Throughput Target

**Goal:** make Python training choose CPU or GPU compiled Remora kernels and
demonstrate that GPU is the viable production-throughput path.

### Tasks

- [ ] Add `--target cpu` / `--target gpu-nvidia` to `examples/crater_train.py`.
- [ ] Compile CPU and GPU inference/value-and-grad through comparable wrappers.
- [ ] Keep Python optimizer and checkpoint format target-independent.
- [ ] Add CPU/GPU parity mode for one batch.
- [ ] Add GPU training loop over synthetic data.
- [ ] Add GPU training loop over a small real subset.
- [ ] Record throughput in examples/sec and images/sec.
- [ ] Record GPU memory usage if available.

### Acceptance

- [ ] `--target gpu-nvidia --synthetic --batch-size 16 --epochs 1` trains and
  decreases loss.
- [ ] GPU and CPU predictions match within tolerance before training updates.
- [ ] GPU gradients match CPU on a tiny deterministic batch.
- [ ] GPU training is faster than CPU for the agreed production-shaped batch
  size.  Initial target: at least 2x faster for batch 16 on `128x128` tiles,
  then revisit after measurements.
- [ ] If GPU is not faster, the bottleneck is identified and a follow-up
  optimization task is written.

### Risk Mitigations

- Risk: transfer overhead dominates for batch 16.
  - [ ] Measure kernel-only and end-to-end times separately.
  - [ ] Consider larger static batch sizes only after batch 16 is correct.
- Risk: Python optimizer copies dominate.
  - [ ] Measure optimizer update time separately.
  - [ ] Consider fused Remora update functions only if measurements justify it.

## Proposed CLI Evolution

Keep `examples/crater_train.py` as the main entry point.

Dry run:

```bash
uv run python examples/crater_train.py \
  --data data/crater_tiles \
  --dry-run-data
```

Synthetic smoke:

```bash
uv run python examples/crater_train.py \
  --synthetic \
  --compiled \
  --image-size 128 \
  --batch-size 16 \
  --epochs 1
```

Real training:

```bash
uv run python examples/crater_train.py \
  --data data/crater_tiles \
  --compiled \
  --image-size 128 \
  --batch-size 16 \
  --epochs 20 \
  --optimizer adamw \
  --learning-rate 1e-3 \
  --checkpoint-dir runs/crater_model_a
```

Evaluate:

```bash
uv run python examples/crater_train.py \
  --data data/crater_tiles \
  --evaluate-checkpoint runs/crater_model_a/best.npz \
  --split test
```

Predict:

```bash
uv run python examples/crater_train.py \
  --data data/new_lunar_tiles \
  --predict \
  --checkpoint runs/crater_model_a/best.npz \
  --output predictions/crater_scores.csv
```

## Immediate Execution Order

1. [ ] Add real/synthetic file-backed dataset loading to `crater_train.py`.
2. [ ] Train the PyTorch baseline on the same loader/splits and inspect
   metrics/errors.
3. [ ] Scale existing single-example Remora CNN to `128x128`; measure IR and
   timing.
4. [ ] Implement static batch ABI tests and fix compiler/runtime gaps.
5. [ ] Convert the model/loss/value-and-grad to static batch 16.
6. [ ] Add optimizer/checkpoint/metrics around the compiled batched kernel.
7. [ ] Train on synthetic batched data until loss decreases.
8. [ ] Train on a small real-data subset and compare to the PyTorch baseline.
9. [ ] Implement GPU ABI support for the frozen Model A static signature.
10. [ ] Add GPU lowering for Model A operations and value-and-grad.
11. [ ] Train the same Python workflow with `--target gpu-nvidia` and compare
   throughput to CPU.

## Explicitly Deferred Follow-up Plans

- [ ] Segmentation/detection model plan.
- [ ] Dynamic batch ABI plan.
- [ ] Larger architecture/model-quality plan after the CPU/GPU Model A path is
  measured.
