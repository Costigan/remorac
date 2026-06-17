# Remorac Capability Summary

## Summary

`remorac` is currently strongest as a static-shape dense array compiler with
reverse-mode AD, especially for small-to-moderate CNN-style numeric kernels on
CPU.  It is not yet a general production GPU compiler for irregular, sparse, or
dynamically shaped workloads.

The best near-term driving examples are dense, static, differentiable array
programs.  The next major compiler frontier is batched execution plus CPU/GPU
parity.  After that, the maturity frontier becomes memory planning,
sparse/irregular primitives, and dynamic or padded-shape workflows.

## Current Strengths

`remorac` already covers a useful core:

- Dense rectangular arrays with static shapes.
- Rank-polymorphic scalar and array arithmetic.
- `map`, `fold`, reductions, scans, and common shape/view operations.
- Reverse-mode AD for scalar losses over elementwise operations, reductions,
  views, selected indexing patterns, `exp`, `log`, and `select`.
- CNN-style image kernels through static-shape `im2col` and `col2im`.
- Multi-output value-and-grad generation for trainable parameters.
- Native CPU compilation through the descriptor path.
- Python orchestration with NumPy interop, proven by the current crater CNN
  training path.

This makes the following examples plausible near-term targets:

- Logistic regression and softmax classifiers.
- Small CNNs.
- Simple image filters and thresholding pipelines.
- Heat-equation-style stencil kernels.
- The synthetic first version of the anchor-free crater grid detector.
- Small all-pairs N-body force computation.
- Simple Monte Carlo reductions with Python-provided randomness.

## Main Gaps

### Static Batch Support

Many examples need inputs such as `[B, ...]` and a scalar mean loss over the
batch.  Static batch support is still a key missing production feature for
training workloads.

### GPU Coverage

GPU support is partial.  Existing support covers some elementwise maps,
reductions, views, scan, and append cases, but full CNN gradient training,
multi-input/multi-output ABI, loop-heavy `im2col` kernels, and larger batched
workloads still need implementation and parity testing.

### Buffer Reuse and Memory Planning

Purely functional array allocation can create large intermediate tensors.  This
matters for neural nets, N-body, differentiable rendering, PDE time loops, and
any workload that would otherwise materialize large pairwise or frame-expanded
arrays.

### Matmul and Dot Recognition

Matrix-vector and matrix-matrix patterns are not fully optimized, especially
when dot products are expressed inside higher-order maps or defunctionalized
helper functions.

### Irregular Data

Sparse and irregular workloads need more compiler surface.  PageRank, molecular
dynamics neighbor lists, connected components, graph propagation, and segmented
cluster updates require gather, scatter-add, segmented reductions, or padded
fixed-width representations.

### Dynamic Shapes

Dynamic shapes and variable-length outputs are not first-class.  Near-term
examples should prefer fixed tile sizes, fixed batch sizes, padded neighbor
lists, fixed grids, and Python-owned orchestration.

### Iteration and Time Loops

Repeated simulation steps are best handled by Python initially unless they map
cleanly to existing scan forms.  Moving fixed-step loops into Remora becomes more
attractive after buffer behavior and GPU lowering are stronger.

## Example Fit

### Best-Fit Examples Now

- **Logistic regression / softmax:** excellent for AD, static batch, stable
  `log`/`exp`, and value-and-grad validation.
- **Image processing:** good dense-kernel benchmark, especially Sobel, blur, and
  threshold pipelines.
- **PDE stencils:** strong dense GPU/compiler benchmark once stencil view
  lowering is validated.
- **Anchor-free crater detector:** good next step from CNN classification if
  kept static, dense, and grid-shaped.
- **Small N-body:** useful stress test for broadcasting and reductions, though
  memory pressure appears quickly.

### Later Examples

- **K-means:** needs argmin/grade and segmented or grouped reductions.
- **Molecular dynamics:** needs gather/scatter and masked neighbor-list support.
- **PageRank:** needs sparse/scatter support.
- **Tomography:** needs interpolation and gather-heavy lowering.
- **FFT/signal processing:** likely needs complex numbers and library
  integration.
- **Differentiable renderer:** feasible in toy form, but large `[H,W,N]`
  intermediates will pressure memory planning.

## Recommended Direction

Use the next wave of examples to mature the compiler in this order:

1. Dense static AD workloads: logistic regression, classifier CNNs, and the
   synthetic crater grid detector.
2. Dense non-AD numeric kernels: image processing and PDE stencils.
3. Memory-pressure workloads: N-body and differentiable rendering.
4. Static padded irregular workloads: molecular dynamics neighbor lists and
   small graph propagation.
5. Sparse, dynamic, or library-backed workloads: PageRank, tomography, FFT, and
   production-scale graph analytics.

