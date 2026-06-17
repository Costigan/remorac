# Anchor-Free Crater Detection Plan

## Goal

Evolve Remora's current crater CNN example from binary tile classification into
a compact crater detector that predicts crater center and radius.  The detector
should remain small enough to be a clear compiler-driving example while still
matching the structure of real lunar crater recognition:

- Python owns geospatial data preparation, tiling, augmentation, batching,
  validation, checkpointing, and post-processing.
- Remora owns the dense differentiable computation: forward pass, detection
  loss, reverse-mode AD, and optionally a single-batch training step.
- The first detector is an anchor-free grid model over fixed-size image tiles.
- Multi-scale image pyramids and overlapping tiles recover much of the practical
  behavior of larger YOLO-style systems without requiring the first milestone to
  implement the full object-detection stack.

This plan builds on the current `examples/crater_train.py` path rather than
replacing it.  The existing classifier remains the smoke test for CNN training;
this detector becomes the next end-to-end benchmark.

## Model Contract

The initial detector uses a fixed tile size and fixed grid.  Concrete shapes are
chosen to keep the first Remora implementation tractable.

Recommended first milestone:

```text
images:  [B, 1, 64, 64]
targets: [B, 8, 8, 4]
preds:   [B, 8, 8, 4]
loss:    Float
```

Larger follow-up milestone:

```text
images:  [B, 1, 128, 128]
targets: [B, 16, 16, 4]
preds:   [B, 16, 16, 4]
loss:    Float
```

Each target/prediction cell has four values:

```text
[objectness_logit_or_target, dx, dy, log_radius]
```

- `objectness`: target is `1.0` when a crater center is assigned to that grid
  cell, otherwise `0.0`.  The model should emit an unbounded logit.
- `dx`, `dy`: center offset within the grid cell, normalized to `[0, 1]`.
- `log_radius`: log-scaled radius normalized to tile units or pixels.

The first model supports at most one crater center per grid cell.  Python should
drop, reassign, or record conflicts during target generation.  Multiple slots
per cell are deferred until this simpler shape trains and compiles reliably.

## Python Responsibilities

Python is deliberately responsible for the irregular and geospatial parts of the
workflow.

### Data Preparation

- Load global lunar mosaics or pre-cut tiles from GeoTIFF/COG/NumPy shards.
- Load crater catalogs containing longitude, latitude, and diameter/radius.
- Project catalog coordinates into raster pixel coordinates using the image CRS.
- Split train/validation/test sets before augmentation.
- Normalize imagery using training-split statistics only.
- Optionally combine optical imagery and DEM channels later; the first detector
  is one grayscale channel.

### Tile and Target Generation

- Extract fixed-size tiles, initially `64x64`, later `128x128`.
- Use overlapping tiles for better boundary coverage.
- For each tile, select catalog craters whose centers fall inside the tile and
  whose radii fall inside the current scale range.
- Assign each crater center to a grid cell:

```text
gx = floor(local_x / cell_width)
gy = floor(local_y / cell_height)
dx = (local_x - gx * cell_width) / cell_width
dy = (local_y - gy * cell_height) / cell_height
log_radius = log(radius / radius_scale)
```

- Emit a dense target tensor `[GridH, GridW, 4]`.
- Record assignment conflicts and ignored craters for diagnostics.
- Support synthetic target generation for compiler and training tests.

### Training Orchestration

- Construct static batches compatible with Remora's compiled ABI.
- Call compiled Remora functions for `loss`, `value_and_grad`, or `train_step`.
- Own optimizer state initially.  SGD is the first target; Adam can follow.
- Save checkpoints containing:
  - Remora source hash,
  - parameter arrays,
  - optimizer state,
  - tile/grid shape metadata,
  - normalization metadata,
  - training split identifier.
- Report training and validation losses split into objectness and localization
  components.

### Inference and Post-Processing

- Build image pyramids for multi-scale detection.
- Extract overlapping tiles at each pyramid level.
- Batch tile inference through Remora.
- Decode grid predictions into local circles:

```text
local_x = (gx + dx) * cell_width
local_y = (gy + dy) * cell_height
radius = radius_scale * exp(log_radius)
score = sigmoid(objectness_logit)
```

- Map local tile detections back to global image coordinates:

```text
global_x = (tile_x + local_x) / pyramid_scale
global_y = (tile_y + local_y) / pyramid_scale
global_r = radius / pyramid_scale
```

- Merge duplicate detections with circle NMS or center-distance/radius matching.
- Export detections as CSV/GeoJSON and optional overlay images.

## Remora Responsibilities

Remora should stay focused on dense, regular array computation.

### Initial Remora Functions

The initial source should expose these logical functions, even if implementation
details are inlined to fit current compiler constraints:

```lisp
forward(params, image) -> [8, 8, 4]
cell-loss(pred_cell, target_cell) -> Float
detection-loss(params, images, targets) -> Float
value-and-grad(detection-loss, params...) -> (Float, grads...)
```

The first production-facing Python API can call either:

- a compiled `value_and_grad` function and update parameters in Python, or
- a compiled `train_step` function that returns updated parameters.

Prefer `value_and_grad` first because it matches the current crater training
script and keeps optimizer state out of Remora.

### Loss Definition

Use a stable scalar loss per grid cell:

```text
loss_cell =
  objectness_bce_with_logits(pred_obj_logit, target_obj)
  + target_obj * lambda_xy * mse(pred_dxdy, target_dxdy)
  + target_obj * lambda_r  * mse(pred_log_r, target_log_r)
```

Then reduce:

```text
loss = mean(loss_cell over B, GridH, GridW)
```

The localization terms are masked by target objectness so empty cells do not
train offsets or radius.  Objectness may need positive-class weighting because
most grid cells are empty.

### Architecture

Start with the smallest architecture that exercises the compiler clearly:

- one or two `3x3` convolution/im2col stages,
- ReLU via `select`,
- simple downsampling by stride or pooling,
- a dense or convolutional head producing `[GridH, GridW, 4]`.

Avoid full YOLO feature pyramids in the first detector.  Multi-scale behavior is
handled by Python image pyramids until the simple detector is validated.

## Compiler Capabilities Required

The current compiler already supports many ingredients used by
`examples/crater_train.py`.  The detector requires tightening and extending them
in a focused order.

### Required for Milestone 1

- Static-shape Remora functions for image, target, and parameter tensors.
- `im2col` or equivalent convolution lowering for `64x64` images.
- Elementwise arithmetic, comparison, `select`, `exp`, and `log`.
- `map`, `fold`, `reshape`, `ravel`, and array indexing patterns used by the
  model.
- Reverse-mode AD through all model and loss operations.
- Multi-output value-and-grad for all trainable parameters.
- Correct cotangent accumulation over batch/grid frame axes.
- Native CPU compiled execution from NumPy arrays.

### Required for Scaled Training

- Static batch ABI for shapes such as `[B, 1, 64, 64]` and `[B, 8, 8, 4]`.
- Efficient batched reduction to a scalar mean loss.
- CPU/GPU parity tests for the detector loss and gradients.
- GPU lowering for the operations used by the detector:
  - elementwise maps,
  - reductions,
  - compact loop-based `im2col` or convolution,
  - multi-input/multi-output ABI.
- Buffer reuse or equivalent memory planning for larger tiles and batches.

### Deferred Compiler Work

- Dynamic batch sizes.
- Variable-length target lists.
- In-language non-max suppression.
- In-language geospatial coordinate transforms.
- Full YOLO matching logic, feature pyramid heads, or anchor boxes.

## Staged Implementation

### Phase 0: Preserve Current Classifier Baseline

Goal: keep the existing crater classifier as a compiler smoke test while adding
the detector.

Tasks:

- Keep `examples/crater_train.py` and its tests passing.
- Record the current compiled strict-mode status in any new detector status
  notes.
- Reuse its value-and-grad wrapper pattern for the detector script.

Acceptance:

- Existing crater training tests pass unchanged.
- The new detector code does not weaken classifier behavior.

### Phase 1: Synthetic Grid-Target Dataset in Python

Goal: prove the target contract without involving real lunar map projection.

Tasks:

- Add a synthetic crater image generator for `64x64` tiles.
- Generate one to a few circular/rim-like craters per tile.
- Emit dense `[8, 8, 4]` targets.
- Include conflict diagnostics when two centers land in the same cell.
- Add decode utilities that convert target tensors back to circles.
- Add tests for assignment, normalization, decoding, and conflict counting.

Acceptance:

- Synthetic dataset produces contiguous `float32` arrays:
  - images `[N, 1, 64, 64]`,
  - targets `[N, 8, 8, 4]`.
- Decoded target circles match the source circles within one pixel.
- Empty cells have objectness `0.0`; assigned cells have objectness `1.0`.

### Phase 2: Remora Detector Loss Without Training

Goal: compile and run the scalar loss for fixed parameters and synthetic data.

Tasks:

- Add a detector Remora source string or `.lisp` example.
- Implement `forward` for one `64x64` image.
- Implement per-cell objectness/localization loss.
- Implement single-example and static-batch loss variants if batch ABI is ready;
  otherwise start with one example.
- Add finite output tests: no NaNs for extreme objectness logits or radius
  values.

Acceptance:

- Remora loss returns a finite scalar for synthetic inputs.
- A perfect prediction has lower loss than a deliberately shifted prediction.
- The compiled CPU path runs in strict mode where toolchain support is present.

### Phase 3: Value-and-Grad for Detector Parameters

Goal: train the detector on synthetic data with Python-owned optimizer updates.

Tasks:

- Generate a multi-output value-and-grad function for detector parameters.
- Add finite-difference gradient spot checks on a reduced model.
- Add a Python training script, likely `examples/crater_detect_train.py`.
- Train on a small synthetic dataset with fixed seeds.
- Log total, objectness, and localization losses.

Acceptance:

- One optimizer step lowers the loss on a fixed mini-batch.
- Training loss decreases over a short synthetic run.
- All parameter gradients are finite.
- Checkpoints can be saved and restored for inference.

### Phase 4: Static Batch Support

Goal: make the detector use a real batch shape instead of one compiled call per
example.

Tasks:

- Choose initial static batch size, e.g. `B=8` or `B=16`.
- Extend or validate descriptor execution for:
  - images `[B, 1, 64, 64]`,
  - targets `[B, 8, 8, 4]`,
  - scalar loss output,
  - multi-output gradients.
- Ensure mean loss divides by batch/grid cell count.
- Add parity tests between batched loss and average of per-example losses.

Acceptance:

- Batched compiled value-and-grad runs in strict CPU native mode.
- Batched and per-example losses agree within tolerance.
- Python training uses batched calls by default.

### Phase 5: Real Lunar Tile Pipeline

Goal: replace synthetic data with real image/catalog-derived dense targets.

Tasks:

- Add a data module for:
  - raster loading,
  - crater catalog loading,
  - coordinate projection,
  - tile extraction,
  - dense grid target assignment.
- Add `--data`, `--catalog`, `--split`, `--tile-size`, `--grid-size`,
  `--dry-run-data`, and `--synthetic` CLI options.
- Persist split manifests and normalization statistics.
- Add train/validation metrics:
  - objectness PR curve or AP proxy,
  - center error for matched craters,
  - radius error for matched craters,
  - decoded detection visualizations.

Acceptance:

- Dry run prints image/target shapes, dtype, crater counts, assignment conflicts,
  ignored crater counts, and normalization stats.
- A small real-data subset can run one compiled training epoch.
- Validation decode produces plausible circle overlays.

### Phase 6: Pyramid and Overlapping-Tile Inference

Goal: recover practical multi-scale behavior while keeping the Remora model
small.

Tasks:

- Add Python image pyramid inference.
- Add overlapping tile extraction with configurable stride.
- Decode and back-project detections to global image coordinates.
- Add circle NMS or radius-aware duplicate merging.
- Evaluate recall/precision over held-out catalog regions.

Acceptance:

- The same Remora detector runs over multiple pyramid scales without recompiling
  when tile shape is unchanged.
- Duplicate detections from overlapping tiles are merged.
- Output CSV contains at least `x`, `y`, `radius`, `score`, `scale`, and source
  tile metadata.

### Phase 7: Scale Tile Size and Model Capacity

Goal: move from compiler demonstration toward useful crater recognition.

Tasks:

- Scale to `128x128` tiles and `16x16` grid.
- Add additional convolution filters or hidden channels.
- Add positive-class weighting and tune loss weights.
- Add multiple prediction slots per grid cell if conflict diagnostics show many
  missed nested/overlapping craters.
- Compare against a PyTorch baseline using the same generated targets.

Acceptance:

- `128x128` detector compiles with bounded IR size.
- Training loss decreases on real data.
- Validation metrics improve over the `64x64` synthetic-first detector.

### Phase 8: GPU Execution

Goal: run the detector training kernel on GPU after CPU correctness is stable.

Tasks:

- Extend GPU ABI as needed for detector parameter/input count.
- Lower detector convolution/im2col path to GPU.
- Support multi-output gradients on GPU or return gradients through a stable
  descriptor ABI.
- Add CPU/GPU forward and gradient parity tests.
- Benchmark throughput against CPU compiled mode and a small PyTorch baseline.

Acceptance:

- Strict GPU mode runs one detector training step.
- CPU/GPU losses and gradients agree within tolerance.
- GPU throughput is reported in examples/sec and tile-pixels/sec.

## Risks and Mitigations

- One crater per cell may miss nested craters.
  - Mitigation: track target conflicts; add multiple slots per cell only if
    conflicts are common enough to matter.
- Empty cells dominate objectness loss.
  - Mitigation: positive weighting or focal-style objectness loss later.
- Boundary craters are missed.
  - Mitigation: overlapping tiles and shifted inference in Python.
- Scale variation exceeds one model's range.
  - Mitigation: image pyramids and radius-range filtering per scale.
- Full GPU training depends on compiler features not yet validated.
  - Mitigation: complete CPU native detector first; keep GPU as an explicit
    later phase.
- Real crater catalogs and imagery may have projection or labeling noise.
  - Mitigation: add dry-run overlays, split persistence, and visual inspection
    outputs before long training runs.

## Near-Term Deliverables

1. `examples/crater_detect_train.py` with synthetic data mode.
2. A detector Remora source with fixed `64x64 -> 8x8` shapes.
3. Tests for Python target assignment and decoding.
4. Tests for Remora detector loss and value-and-grad on synthetic tensors.
5. A short status update recording compile time, step time, and loss reduction.

