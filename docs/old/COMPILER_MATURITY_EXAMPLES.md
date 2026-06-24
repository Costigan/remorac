# ARCHIVED — out of date, kept for reference only. See AGENTS.md, PROJECT_OVERVIEW.md, and FUTURE_WORK.md for current information.

# Compiler Maturity Example Roadmap

## Current Status (2026-06-18)

| #   | Example             | Interp | CPU | GPU | Notes                                                                    |
| --- | ------------------- | ------ | --- | --- | ------------------------------------------------------------------------ |
| 1   | PDE solver (heat)   | ✓      | ✓   | ✓   | Subarray+fused-map path; 13/13 GPU tests                                 |
| 2   | N-body              | ✓      | ✓   | ~   | General path handles compound map bodies (fold+index+nested map)         |
| 3   | Image processing    | ✓      | ✓   | ✓   | Sobel, threshold, blur (Phases 4A/4G)                                    |
| 4   | Logistic regression | ✓      | ✓   | ✓   | Matmul kernel, type-aware arithmetic, general map fallback               |
| 5   | K-Means             | ✓      | ~   | ~   | Sort/grade GPU kernels exist; broadcasting via with-shape                |
| 6   | Molecular dynamics  | ✓      | ✗   | ✗   | CPU-level gaps (neighbor lists, periodic boundaries) block both backends |
| 7   | Tomography          | ✓      | ✗   | ✗   | Interpolation primitives not yet a compiler surface                      |
| 8   | Monte Carlo         | ✓      | ✓   | ✓   | Parallel scan (all ops, exclusive, right), reductions                    |
| 9   | Kalman filter       | ✓      | ~   | ~   | Matmul + full scan now available on GPU                                  |
| 10  | PageRank            | ✓      | ~   | ~   | Scatter-add GPU kernel available; edge-list patterns partially supported |
| 11  | FFT                 | ✓      | ✗   | ✗   | Complex numbers not supported                                            |
| 12  | Diff. renderer      | ✓      | ~   | ~   | General map fallback handles AD-generated compound bodies                |

**Interpreter:** `evaluate_source` / `_lambda_callable` path — works for all
examples (interpreter handles the full Remora surface). **CPU:** descriptor-ABI
native compilation. **GPU:** `compile_function_source_to_mlir_gpu_ptx` →
PTX → CUDA launch. ✓ = verified, ~ = likely works / not fully tested, ✗ =
blocked by missing compiler capability.

## Purpose

This document lists candidate end-to-end examples that could mature `remorac` in
the same way the crater neural-net work does. A good driving example should be
large enough to expose real compiler gaps, small enough to validate in stages,
and shaped around Remora's strengths: dense array programming, rank
polymorphism, shape checking, AD, reductions, scans, and CPU/GPU lowering.

The crater detector remains the primary near-term example. These examples are
options for broadening the compiler roadmap after, or alongside, that work.

## Current Baseline

As of the current state, `remorac` has useful coverage for:

- Lisp and ML syntaxes for dense array programs.
- Scalar and array arithmetic with rank-polymorphic lifting.
- `map`, `fold`, `fold-right`, reductions, scans (parallel Hillis-Steele),
  and a full set of shape/view primitives.
- Static-shape `im2col`/`col2im` for CNN-style image kernels.
- Reverse-mode AD for scalar losses over many elementwise, view, reduction, and
  indexing operations.
- Multi-output value-and-grad source generation.
- Native CPU compilation through the descriptor path.
- GPU compilation for the full dense statically-shaped subset: all view ops
  (take/drop/subarray/reverse/rotate/transpose), descriptor reinterpretation
  (reshape/ravel/append/with-shape), scatter-add, matmul, sort/grade,
  filter/replicate, indices-of, scan (all variants), and a general map path
  that serves as a universal fallback for any compound-body map.
- Type-aware i32/f32/bool loads, stores, and arithmetic on GPU.
- Automatic monomorphization of higher-order function parameters for GPU.

Important known gaps for larger examples:

- Interpolation and coordinate-transform primitives are not yet a polished
  surface (blocks tomography, some advanced image processing).
- Complex number support is absent (blocks FFT).
- Sparse, segmented, and irregular data structures are limited (affects
  molecular dynamics neighbor lists, graph algorithms).
- Dynamic shapes and variable-length outputs use serial GPU kernels
  (filter/replicate) pending multi-kernel executor orchestration.
- GPU sort, grade, scatter-add, and filter use serial single-thread kernels;
  parallel versions are documented in `docs/FUTURE_WORK.md`.
- Buffer reuse and memory planning are not yet mature enough for large
  intermediates (e.g. N-body pairwise `[N,N,3]` tensors).

## Evaluation Criteria

Each candidate is assessed by:

- **Value:** what compiler capability it would mature.
- **Fit:** how naturally it maps to Remora.
- **Current gaps:** what `remorac` likely lacks today.
- **First milestone:** a tractable version that can be implemented before the
  full problem.

## 1. Finite Difference PDE Solver

Examples: heat equation, wave equation, diffusion, shallow water, or a small
Navier-Stokes-like stencil.

Typical shapes:

```text
grid:       [H, W]
velocity:   [H, W, 2]
next_grid:  [H, W]
trajectory: [T, H, W]
```

### Value

This is one of the best non-ML examples for maturing GPU array compilation.
Stencil kernels are common in scientific computing, visually inspectable, and
mostly dense and regular.

It would exercise:

- `rotate`, `subarray`, `take`, `drop`, and boundary handling.
- Repeated time steps.
- Buffer reuse between current and next state.
- Fusion of elementwise operations around stencil reads.
- CPU/GPU parity for dense numeric kernels.

### Current Gaps

- Repeated time stepping needs either host-side Python orchestration or a
  reliable in-language loop/scan strategy for fixed iteration counts.
- Boundary-condition utilities are not yet a polished surface.
- GPU lowering for the exact combination of views and stencil operations needs
  validation.
- Without buffer reuse, long simulations may allocate excessively.

### First Milestone

Implement one heat-equation step:

```text
step(grid [64,64]) -> [64,64]
```

Then let Python run `T` steps and compare against a NumPy reference. Later,
move fixed-step iteration into Remora and scale to `512x512`.

## 2. N-Body Simulation

Compute gravitational or electrostatic interactions among particles.

Typical shapes:

```text
positions:  [N, 3]
velocities: [N, 3]
masses:     [N]
forces:     [N, 3]
```

### Value

N-body is a compact dense benchmark for pairwise maps, broadcasting, reductions,
and GPU memory pressure. It is also a good AD target if optimizing initial
conditions or fitting parameters.

It would exercise:

- Pairwise expansion from `[N,3]` to `[N,N,3]`.
- Reductions over interaction axes.
- Masking self-interactions.
- Numeric stability with softened distances.
- Later: tiling to avoid materializing full `[N,N,3]` intermediates.

### Current Gaps

- A naive all-pairs tensor can become too large quickly; the compiler needs
  fusion or tiling to avoid memory blowups.
- In-language multi-step integration requires loop or scan support with good
  buffer behavior.
- Masking diagonal/self-interactions needs clean array patterns.
- ~~Scalar-map-with-fold lowering gap~~ — **resolved** by the general GPU map
  path (compound bodies with folds, index expressions, nested maps).

### First Milestone

Implement single-step force computation for `N=128`:

```text
forces(positions [128,3], masses [128]) -> [128,3]
```

Validate against NumPy, then add one velocity/position update. Python can run
multiple steps until Remora loop support is ready.

## 3. Image Processing Pipeline

Examples: Gaussian blur, Sobel edges, thresholding, morphology, and simple
candidate extraction.

Typical shapes:

```text
image: [H, W]
edges: [H, W]
mask:  [H, W]
```

### Value

This is a natural companion to crater detection. It exercises real image
operators without neural-network AD complexity and gives visually inspectable
outputs.

It would exercise:

- Convolution or `im2col` outside the CNN path.
- Boolean arrays and threshold masks.
- Morphological operations via local windows.
- Reductions over windows.
- Potentially connected components as a later irregular-output challenge.

### Current Gaps

- `window` is used aspirationally in older crater examples, but the documented
  compiler surface centers on `im2col`, `subarray`, and views.
- Morphology wants local-window primitives over arbitrary shapes.
- Connected components needs irregular labels, scans, union-find, or repeated
  relaxation; that is beyond the current dense core.
- GPU lowering of image windows should be validated independently of CNN AD.

### First Milestone

Implement Sobel edge magnitude and threshold:

```text
sobel(image [128,128]) -> [128,128]
threshold(edges [128,128], t Float) -> [128,128 Bool]
```

Python can display overlays and compare against SciPy/OpenCV references.

## 4. Logistic Regression and Softmax Classifier

Train linear models with Remora AD.

Typical shapes:

```text
x:      [N, D]
w:      [D]
y:      [N]
logits: [N]
```

Multiclass:

```text
x:      [N, D]
w:      [C, D]
y:      [N, C]
logits: [N, C]
```

### Value

This is a smaller AD benchmark than neural nets and is excellent for testing
batch semantics, stable `log`/`exp`, reductions, and value-and-grad generation.

It would exercise:

- Batched matrix-vector and matrix-matrix patterns.
- Stable BCE or softmax cross-entropy.
- Gradient accumulation over batch axes.
- Optimizer orchestration in Python.
- CPU/GPU parity for simple differentiable workloads.

### Current Gaps

- Efficient matmul recognition is not fully mature for cell-map dot patterns.
- Static batch support still needs to be hardened.
- Softmax needs careful stable reductions: subtract max, exponentiate, normalize.
- `max` reductions and argmax-style metrics may need clearer primitive support.

### First Milestone

Binary logistic regression on synthetic data:

```text
loss(w [D], b Float, x [N,D], y [N]) -> Float
```

Compile value-and-grad for `w` and `b`, train with Python SGD, and compare to a
NumPy or scikit-learn baseline.

## 5. K-Means and Gaussian Mixture Models

Cluster points by nearest center, then update centers.

Typical shapes:

```text
points:  [N, D]
centers: [K, D]
dists:   [N, K]
labels:  [N]
```

### Value

K-means is a good bridge from dense pairwise arrays to irregular grouped
reductions. It is not AD-heavy, but it is shape-heavy and reduction-heavy.

It would exercise:

- Pairwise distance computation `[N,K,D] -> [N,K]`.
- Argmin or grade-like operations.
- Masked/segmented reductions by cluster.
- Iterative convergence.

### Current Gaps

- ~~`sort` and `grade` are documented as typecheck-only~~ — **resolved**:
  both have dedicated GPU kernels (serial insertion sort).
- Argmin/argmax support may need explicit patterns using `grade`.
- Updating centers requires segmented sums/counts or one-hot masks. Dense
  one-hot is possible but inefficient for large `N` and `K`.
- Convergence checks need scalar reductions and host orchestration.
- Empty clusters need conditionals and safe division.

### First Milestone

Implement pairwise squared distances:

```text
dists(points [N,D], centers [K,D]) -> [N,K]
```

Then add Python-owned argmin/update as an intermediate step. Later move dense
one-hot center updates into Remora for small `K`.

## 6. Molecular Dynamics With Neighbor Lists

Compute particle forces with cutoff radii and fixed-size neighbor lists.

Typical shapes:

```text
atoms:       [N, 3]
neighbor_ix: [N, K]
forces:      [N, 3]
```

### Value

This matures gather/scatter, masked computation, and more realistic scientific
simulation than all-pairs N-body. It introduces controlled irregularity through
a padded fixed-width neighbor list.

It would exercise:

- Gather from index arrays.
- Masked neighbors for rows with fewer than `K` neighbors.
- Reductions over neighbor axis.
- Potential scatter-add if computing pair contributions symmetrically.
- Later: Python-built neighbor lists versus Remora-built neighbor lists.

### Current Gaps

- Advanced gather patterns over `[N,K]` index arrays need validation.
- ~~Scatter-add~~ — **resolved**: `HIRScatterAdd` has a dedicated GPU kernel.
  However, the kernel is serial (single-thread); parallel scatter is in
  `docs/FUTURE_WORK.md`.
- Dynamic neighbor counts are not first-class; padding and masks are required.
- Periodic boundary conditions need careful vector arithmetic and wrapping.
- CPU-level gaps (neighbor-list construction, masked boundaries) block both
  backends equally.

### First Milestone

Use Python to build fixed-width neighbor lists and Remora to compute Lennard-
Jones-like forces:

```text
forces(atoms [N,3], neighbor_ix [N,K], mask [N,K]) -> [N,3]
```

## 7. Computed Tomography Reconstruction

Implement filtered backprojection or a simple iterative reconstruction.

Typical shapes:

```text
sinogram: [Angles, Detectors]
image:    [H, W]
```

### Value

Tomography is a real scientific imaging workload with structured transforms,
large reductions, and interpolation. It is a good stress test for layout,
transposes, and memory bandwidth.

It would exercise:

- Coordinate transforms over pixel grids.
- Gather/interpolation from sinogram coordinates.
- Reductions over projection angles.
- Large regular arrays and GPU execution.

### Current Gaps

- Interpolation primitives are not currently a polished compiler surface.
- Efficient coordinate-grid generation and trigonometric functions may need work.
- GPU lowering for gather-heavy interpolation is more difficult than pure
  elementwise maps.
- Backprojection can be written densely but may be slow without fusion.

### First Milestone

Implement a tiny nearest-neighbor backprojection:

```text
backproject(sinogram [32,64]) -> [64,64]
```

Use Python to generate a phantom and reference reconstruction.

## 8. Monte Carlo Option Pricing

Simulate many stochastic price paths and reduce payoffs.

Typical shapes:

```text
paths:   [Sims, Steps]
payoffs: [Sims]
price:   Float
```

### Value

This is an embarrassingly parallel workload with scans over time and reductions
over simulations. It can later use AD to compute sensitivities.

It would exercise:

- Prefix scans for cumulative returns.
- Reductions over large simulation axes.
- Host-provided random numbers or future Remora RNG support.
- Differentiation of payoff approximations for Greeks.

### Current Gaps

- Remora does not appear to have a mature RNG story; Python should generate
  random normal tensors first.
- Discontinuous payoffs make AD tricky unless smoothed or handled analytically.
- Large `[Sims, Steps]` arrays need memory planning on GPU.
- In-language simulation loops may need scan-based formulation.

### First Milestone

Python generates Gaussian shocks:

```text
price(shocks [Sims,Steps], s0 Float, drift Float, vol Float) -> Float
```

Remora computes cumulative paths with scan and reduces discounted payoff.

## 9. Kalman Filter or Particle Filter

Estimate hidden state from a stream of observations.

Typical shapes:

```text
observations: [T, ObsDim]
states:       [Particles, StateDim]
weights:      [Particles]
```

### Value

Filtering exercises scans over time, normalization, reductions, and controlled
state updates. Particle filters add irregular resampling pressure later.

It would exercise:

- `scan` for time-recursive computation.
- Matrix-vector operations.
- Probability normalization.
- Reductions and stable log-sum-exp variants.

### Current Gaps

- ~~General matrix multiplication~~ — **resolved**: `HIRMatmul` has a dedicated
  GPU kernel (per-thread dot-product; tiled version in `docs/FUTURE_WORK.md`).
- ~~Scan~~ — **resolved**: parallel Hillis-Steele scan handles all operators,
  exclusive/inclusive, left/right, any size.
- Resampling is irregular and likely should remain Python-owned initially.
- Dynamic observation lengths are deferred; use static `T` first.
- Stable probability computations need `max` reductions and log-domain helpers.

### First Milestone

Implement a fixed-shape linear Kalman predict/update for one time step, then use
Python to loop over `T`. Move to Remora `scan` after one-step correctness.

## 10. PageRank or Graph Propagation

Compute rank propagation over a graph.

Typical shapes:

```text
edges:     [E, 2]
rank:      [N]
next_rank: [N]
```

### Value

Graph workloads push Remora toward sparse and irregular data. This should be a
later example, after dense examples are solid.

It would exercise:

- Gather rank values by edge source.
- Scatter-add contributions to edge destination.
- Iterative convergence.
- Sparse representations at the Python/Remora boundary.

### Current Gaps

- ~~User-facing scatter-add~~ — **resolved**: `HIRScatterAdd` has a dedicated
  GPU kernel.
- Segmented reduction support is limited.
- Sparse storage formats are not yet central to the compiler.
- GPU lowering for scatter-heavy kernels is nontrivial.
- Dynamic graph sizes need either recompilation or shape padding.

### First Milestone

Use fixed-size edge arrays and Python-owned iteration:

```text
pagerank_step(edges [E,2], out_degree [N], rank [N]) -> [N]
```

If scatter-add is not available, start with dense adjacency `[N,N]` for very
small graphs, then graduate to edge-list scatter.

## 11. FFT-Based Signal Processing

Examples: spectrograms, convolution via FFT, or frequency-domain filtering.

Typical shapes:

```text
signal:      [T]
windows:     [Frames, Window]
spectrogram: [Frames, Freq]
```

### Value

This is a useful benchmark if Remora eventually needs primitive-library
integration. FFTs are foundational, but they are less ideal as an early
compiler example because a good implementation often depends on specialized
runtime primitives.

It would exercise:

- Window extraction.
- Complex numbers or real/imag pair representation.
- Reductions over frequency/time axes.
- Primitive calls into optimized FFT libraries.

### Current Gaps

- Complex number support is not clearly present.
- Efficient FFT is better implemented as a primitive or external library call,
  not as naive Remora source.
- GPU FFT requires library integration or substantial codegen work.
- Windowing can be done with `im2col`-like primitives but needs a 1D surface.

### First Milestone

Implement direct DFT for tiny fixed sizes using real/imag pair arrays:

```text
dft(signal [32]) -> [32,2]
```

Use this as a semantics test, not a performance target. Defer real FFT
performance to library integration.

## 12. Differentiable Renderer Toy

Render simple circles, disks, spheres, or heightfields and optimize parameters
from pixels.

Typical shapes:

```text
objects: [N, Params]
pixels:  [H, W]
loss:    Float
```

### Value

This is a visually inspectable AD workload. It is related to crater geometry
but avoids CNN training. It can test differentiable control flow and reductions
over object and pixel axes.

It would exercise:

- Broadcasting over pixels and objects.
- Smooth approximations to visibility and coverage.
- Reductions from object contributions to pixel image.
- AD over geometry parameters.

### Current Gaps

- Hard visibility and min/max operations are not smooth; the first version needs
  soft masks or differentiable approximations.
- Pairing pixel grids with object arrays can create large `[H,W,N]`
  intermediates.
- Efficient reductions over object axes need fusion and memory planning.
- More math functions may be needed for richer shading.

### First Milestone

Render a small set of soft disks:

```text
render(circles [N,3]) -> [64,64]
loss(circles, target [64,64]) -> Float
```

Use AD to optimize circle positions and radii from a synthetic target image.

## Recommended Priority

The strongest complements to crater neural-net training are:

1. **Finite difference PDE solver**: best dense GPU/stencil maturity path.
1. **N-body simulation**: best pairwise map/reduction and memory-pressure path.
1. **Logistic regression / softmax classifier**: best small AD and static-batch
   validation path.
1. **Image processing pipeline**: best non-neural companion to crater detection.
1. **K-means**: best transition toward argmin and segmented reductions.
1. **PageRank or molecular dynamics**: later sparse/irregular milestones once
   dense examples are solid.

Together with the anchor-free crater detector, these examples cover a broad
compiler maturity surface: dense maps, reductions, scans, stencils, AD,
convolution, static batch ABI, buffer reuse, scatter, segmented reductions,
irregular data, and GPU lowering.
