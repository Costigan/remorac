# Architectural Design Breakdown

This document details the structural, mathematical, and parallel computing mechanics behind the Remora multi-scale crater detection application.

---

## 1. System Topology & Data Flow

The processing pipeline is organized into three decoupled phases: spatial tensor materialization, rank-polymorphic execution (forward/backward), and geometric scale projection.

+-------------------------------------------------------+
|                Input: Lunar Raster Map                |
+-------------------------------------------------------+
|
v
+-------------------------------------------------------+
|           Image Pyramid Gen (Recursive Avg)           |
|  Scale 1.0 -> Scale 2.0 -> Scale 4.0 -> Scale 8.0     |
+-------------------------------------------------------+
|
v
+-------------------------------------------------------+
|              Spatial Tensor Windowing                 |
|   Extracts overlapping [32, 32] pixel sub-matrices    |
+-------------------------------------------------------+
|
v
+-------------------------------------------------------+
|          Rank-Polymorphic Processing (CNN)            |
|  - Implicit Cell Replication                          |
|  - Concurrent Feature Extraction                      |
+-------------------------------------------------------+
|
v
+-------------------------------------------------------+
|            Coordinate Space Back-Projection           |
|      Coordinates * Scale-Factor = Global Space        |
+-------------------------------------------------------+
|
v
+-------------------------------------------------------+
|       Output: Multi-Scale Bounding Box Array          |
+-------------------------------------------------------+

---

## 2. Mathematical Formalism of the Layers

### Convolution and Feature Extraction
The core feature extractor uses a localized spatial operator. For a given 2D patch $P$ and a kernel $K$ of shape $3 \times 3$, the operation is defined as a sum of element-wise products plus a scalar bias $b$:

$$c = \max\left(0, \left( \sum_{i=1}^{3} \sum_{j=1}^{3} P_{i,j} \cdot K_{i,j} \right) + b\right)$$

In Remora, the array syntax `(* patch kernel)` evaluates the element-wise multiplication natively across matching inner dimensions.

### Optimization Objective & Automatic Differentiation
Training parameters are updated using reverse-mode automatic differentiation targeting the Binary Cross-Entropy (BCE) loss function:

$$\mathcal{L} = -\frac{1}{N} \sum_{n=1}^{N} \left[ y_n \log(\hat{y}_n) + (1 - y_n) \log(1 - \hat{y}_n) \right]$$

Where:
* $y_n$ is the ground-truth binary label (1.0 for a crater, 0.0 for smooth regolith).
* $\hat{y}_n$ is the network scalar prediction bounded by the Sigmoid activation.

The `(grad loss-objective params ...)` primitive evaluates the vector Jacobian product across the structural parameters, yielding identical geometric shapes for parameter updates:

$$\theta_{t+1} = \theta_t - \eta \cdot \nabla_\theta \mathcal{L}$$

---

## 3. Remora Parallelization Mechanics

Remora entirely bypasses traditional loops, block allocations, or thread management primitives. It achieves extreme parallel efficiency via two core language paradigms:

### Windowing vs. Nested Iteration
Instead of writing an nested imperative loop over the spatial indices $(x, y)$ of a large lunar image, the `window` primitive materializes all potential candidate blocks into an independent high-rank axis. This transforms a single 2D canvas of shape `[H, W]` into an array of patches with a unified frame structure:

$$\text{Shape}(\text{patches}) = [N, 32, 32]$$

Where $N$ represents the total number of valid spatial strides across the image.

### Principal-Frame Cell Replication
When evaluating `(forward params patches)`, the network functions are written to expect an array of rank 2 (a singular `[32, 32]` patch). 
1. The runtime identifies that the input array has an extra leading dimension ($N$).
2. The runtime designates this outer axis as the **Frame**.
3. The parameter tuple `params` is implicitly replicated across this frame dimension.
4. The forward pass is dispatched concurrently across all $N$ cells, mapping perfectly to SIMD or multi-threaded hardware architectures.

---

## 4. Multi-Scale Coordinate Back-Projection

To catch craters larger than the base window size, the image pyramid downsamples the canvas recursively. Detections found on smaller scales must be transformed back to the coordinate space of the original high-resolution master image.

Given a coordinate vector $\mathbf{x}_{\text{local}} = [x, y]^T$ evaluated at a pyramid tier downsampled by a cumulative factor $S$, the true top-left pixel index on the high-resolution master map $\mathbf{x}_{\text{global}}$ is calculated by:

$$\mathbf{x}_{\text{global}} = \mathbf{x}_{\text{local}} \cdot S$$

The resulting bounding box configuration for any crater identified at scale $S$ is expressed as:

$$\text{Bounding Box} = \left[ x \cdot S, \; y \cdot S, \; (x \cdot S) + (\text{patch\_size} \cdot S), \; (y \cdot S) + (\text{patch\_size} \cdot S) \right]$$

This uniform scaling behavior is handled in the source code via the rank-polymorphic term `(* current-detections scale-factor)`.
