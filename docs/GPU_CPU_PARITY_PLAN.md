# GPU / CPU Backend Parity Plan

## Goal

Achieve parity between the CPU and GPU backends for the dense statically-shaped
Remora subset.  Any program that compiles and runs on the CPU backend should
also compile and run on the GPU backend (via the general lowering path),
producing numerically equivalent results.

**Design principle**: no example-specific code.  Every lowering path must
handle arbitrary HIR, not pattern-match against specific programs.

## Current State

The CPU backend handles **all HIR node types** in the dense subset via
`linalg.generic`-based tensor lowering plus `scf.for` fallback for compound
bodies.

The GPU general lowering path (`remora/_gpu_expr_lowering.py`) handles
**12 HIR node types**: `HIRMap`, `HIRApply`, `HIRFold`, `HIRReduce`,
`HIRIndex`, `HIRIf`, `HIRLet`, `HIRVar`, `HIRLit`, `HIRCast`, `HIRPrimOp`,
`HIRIota`.  This covers the core compound-body flow — maps containing folds,
indexing, arithmetic, conditionals, let bindings, and nested maps — plus
scalar and rank-1 array-valued reductions, element-wise operations on
arrays, and math intrinsics.

**16 tensor-input HIR node types** are handled by CPU but not by the GPU
general path.  These are the gaps we must close.

## Gap Analysis

| # | HIR Node | Category | Approach |
|---|----------|----------|----------|
| 1 | `HIRTake` | A: descriptor adjust | Adjust descriptor `size[0]` |
| 2 | `HIRDrop` | A: descriptor adjust | Add count to `offset[0]`, reduce `size[0]` |
| 3 | `HIRSubarray` | A: descriptor adjust | Adjust all `offset[k]` and `size[k]` |
| 4 | `HIRSlice` | A: descriptor adjust | Sub-range on first axis (offset + size) |
| 5 | `HIRReverse` | A: index transform | `load(N - 1 - i, j, …)` — affine rewrite |
| 6 | `HIRRotate` | A: index transform | `load((i + shift) % N, j, …)` — modulo |
| 7 | `HIRTranspose` | A: descriptor adjust | Swap `stride[0]` / `stride[1]`, reorder coords |
| 8 | `HIRArrayLit` | A: literal constant | Lower to `GpuConstant` list, return `GpuArrayExpr` |
| 9 | `HIRReshape` | B: descriptor reinterpret | New sizes/strides, same buffer |
| 10 | `HIRRavel` | B: descriptor reinterpret | Reshape to rank-1 (stride=1, size=total) |
| 11 | `HIRAppend` | B: concatenation | Two descriptor loads with conditional select |
| 12 | `HIRWithShape` | B: broadcast | Load from source at trailing coordinates only |
| 13 | `HIRScatterAdd` | B: atomic write | `llvm.atomicrmw fadd` on output |
| 14 | `HIRIm2col` | C: complex algorithm | Retain specialised kernel |
| 15 | `HIRCol2im` | C: complex algorithm | Defer — no existing scaffolding |
| 16 | `HIRMatmul` | C: complex algorithm | Defer — needs tiled shared-memory design |

**6 module-level nodes** handled by CPU but not in scope for the general
map-body path (`HIRSort`, `HIRGrade`, `HIRFilter`, `HIRReplicate`,
`HIRScan`, `HIRIndicesOf`) — deferred to a future phase.

## Implementation Plan

### Phase A — Descriptor-level view ops (easy, 7 types)

These nodes adjust which subset of an existing descriptor's data is
accessed.  On the GPU, they can be implemented by transforming the
descriptor offset/size/stride fields before the kernel sees them, or
by adjusting the thread-to-coordinate mapping inside the kernel.

The approach is to handle them in the HIR→GPUExpr compiler:
when a view op wraps a descriptor input, adjust the `GpuInputLoad`
coordinates rather than emitting a new kernel.

- [x] **A.1 — `HIRTake`**: When lowering `HIRTake(count, array)`,
      reduce the descriptor size and bound the thread index accordingly.
      The coordinated load uses the same index expressions; just the
      size changes.

- [x] **A.2 — `HIRDrop`**: When lowering `HIRDrop(count, array)`,
      add `count` to the logical offset for dimension 0.  Thread
      coordinates add `count` before the stride multiplication.

- [x] **A.3 — `HIRSubarray`**: When lowering `HIRSubarray(array, offsets, sizes)`,
      add each `offsets[k]` to the per-dimension coordinate before
      stride multiplication, and clamp the thread range to `sizes[k]`.
      The GPU descriptor load path already has subarray offset support
      (lines 964–984 in `gpu_lowering.py`); generalize it to all ranks.

- [x] **A.4 — `HIRSlice`**: Same mechanism as Subarray for a single
      axis with a start offset and a size.  Essentially `Subarray`
      with `offsets=(start,)` and `sizes=(end-start,)`.

- [x] **A.5 — `HIRReverse`**: When a load references dimension 0,
      replace coordinate `i` with `size[0] - 1 - i`.  No descriptor
      changes needed — just an affine transform on the coordinate
      expression before the stride multiplication.

- [x] **A.6 — `HIRRotate`**: Replace coordinate `i` with
      `(i + shift) % size[0]`.  Requires a modulo operation in the
      coordinate computation.  Alternatively, emit two conditional
      loads (left and right portions) but the modulo approach is
      simpler to generate.

- [x] **A.7 — `HIRTranspose`**: For rank-2, swap `stride[0]` and
      `stride[1]` in the descriptor, and swap the coordinate
      computation order (`%i0` ↔ `%i1`).  For higher ranks, reorder
      the stride array and coordinate mapping.

- [x] **A.8 — `HIRArrayLit`**: Already partially supported as fold-init
      values.  Extend `_lower_hir` to produce `GpuArrayExpr` containing
      `GpuConstant` nodes.  When it appears as a free operand to a map,
      each component resolves to a constant.

- [x] **A tests** — Add compilation + numeric parity tests for each
      view op type.  Verify that a program like
      `map (fn x -> x) (take 3 arr)` produces correct GPU output
      matching CPU.

### Phase B — Descriptor reinterpretation ops (moderate, 5 types)

These nodes change how a buffer's data is interpreted or accessed
across multiple inputs.

- [x] **B.1 — `HIRReshape`**: When reshaping an input descriptor,
      reinterpret it with new sizes and strides while keeping the same
      base pointer and offset.  The thread coordinate decomposition is
      recomputed for the new shape.  Requires that the total element
      count stays the same (statically guaranteed by the type checker).

- [x] **B.2 — `HIRRavel`**: Special case of reshape to rank-1 with
      stride 1.  `GpuInputLoad` with a single flat coordinate.

- [x] **B.3 — `HIRAppend`**: Concatenates `left` and `right` along
      dimension 0.  On GPU, emit a conditional: if the thread's
      index `i` is less than `left_size[0]`, load from the left
      descriptor at `i`; otherwise load from the right descriptor at
      `i - left_size[0]`.  This is a per-thread `llvm.select` on the
      coordinate and descriptor base pointer.

- [x] **B.4 — `HIRWithShape`**: Broadcast a smaller source array
      to a larger target shape.  On GPU, this means the source has
      fewer dimensions than the target.  `GpuInputLoad` with the
      trailing coordinates only (e.g. for `with-shape [M, N, K] src`
      where `src` has shape `[K]`, load from `src` at coordinate `k`).
      Requires mapping target thread coordinates to source coordinates
      by dropping leading dimensions.

- [ ] **B.5 — `HIRScatterAdd`**: `target[i] += update`.  On GPU,
      this is an atomic add: `llvm.atomicrmw fadd %target_ptr, %update`.
      Requires a separate kernel (scatter-add is a write, not a read)
      or handling as a special node in the expression tree that lowers
      to a store-with-atomic.

- [x] **B tests** — Compilation + numeric parity tests for reshape,
      ravel, append, with-shape, and scatter-add.

### Phase C — Remaining gaps and hardening

- [ ] **C.1 — i32 arithmetic support in GpuBinaryOp**: Currently the
      emitter defaults to `f32` LLVM ops.  Add type-aware emission
      so `i32` operands use `llvm.add`/`llvm.sub`/`llvm.mul`/`llvm.sdiv`.

- [ ] **C.2 — i32/int comparison support**: `GpuCompareOp` currently
      uses `llvm.fcmp`.  Add `llvm.icmp` path for integer operands.

- [ ] **C.3 — Mixed-type casts**: Support implicit casts when an
      expression mixes `i32` and `f32` operands (insert `GpuCast`).

- [ ] **C.4 — Array-typed `HIRIf`**: When both branches produce
      arrays of the same shape, emit element-wise `GpuSelect` for
      each component.

- [ ] **C.5 — Multi-rank array-valued folds**: Currently only rank-1
      array results are supported.  Extend to rank > 1 by recursively
      decomposing into nested component iterations.

- [ ] **C.6 — Non-contiguous stride support**: Test and fix descriptor
      stride handling for non-contiguous layouts (e.g. subarray views
      with stride != 1).  The existing `_linear_index_lines` already
      multiplies by per-dimension strides, so this should mostly work.

- [ ] **C.7 — Zero-size shape handling**: Bounds-check before any
      load/store when a dimension is zero.  Currently untested.

- [ ] **C tests** — Targeted tests for each hardening item.

### Phase D — Integration and regression

- [ ] **D.1 — Remove redundant specialised kernels**: The
      `build_descriptor_abi_f32_append_gpu_module` and
      `build_descriptor_abi_f32_scan_gpu_module` specialised kernels
      overlap with general-path capabilities after Phase B.  Remove them
      once the general path covers Append and the scan is deferred.

- [ ] **D.2 — Update dispatch chain priority**: After Phase A and B,
      the general path handles most programs.  Move the general-path
      dispatch earlier in the chain (or make it the default fallback)
      and keep only truly specialised kernels (im2col, matmul) as
      early-dispatch overrides.

- [ ] **D.3 — Full regression suite**: Run all CPU tests, all GPU
      tests, and all new parity tests.  Zero regressions.

- [ ] **D.4 — Update PROJECT_OVERVIEW.md** to reflect achieved parity.

## Out of Scope (this plan)

These nodes either require complex multi-pass algorithms, dynamic memory
allocation, or are better served by dedicated kernel designs.  They remain
handled by specialised kernels or deferred:

- `HIRIm2col` / `HIRCol2im` — retain specialised GPU kernels
- `HIRMatmul` — dedicated kernel required (tiling, shared memory)
- `HIRSort` / `HIRGrade` — GPU sorting algorithms (bitonic, radix)
- `HIRFilter` / `HIRReplicate` — dynamic output sizes
- `HIRScan` — parallel prefix sum (Blelloch/Kogge-Stone)
- `HIRIndicesOf` — coordinate generation kernel
- Named-function / section callables (non-lambda)
- `HIRFoldRight`

## Success Criteria

1. All HIR nodes in categories A and B (12 types) lowered by the GPU
   general path, producing correct numeric results matching CPU.
2. Categories A and B have compilation + numeric parity tests (GPU
   hardware required for parity).
3. i32 arithmetic and comparison fully supported in the general path.
4. All existing tests pass (≈340 CPU, ≈97 GPU, 16 GPU general lowering).
5. Redundant specialised kernels removed (Append, Scan).
6. `PROJECT_OVERVIEW.md` states "GPU backend achieved parity with CPU
   backend for the dense statically-shaped subset."
