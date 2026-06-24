# ARCHIVED — out of date, kept for reference only. See AGENTS.md, PROJECT_OVERVIEW.md, and FUTURE_WORK.md for current information.

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

- [x] **B.5 — `HIRScatterAdd`**: `target[i] += update`.  On GPU,
      this is an atomic add: `llvm.atomicrmw fadd %target_ptr, %update`.
      Requires a separate kernel (scatter-add is a write, not a read)
      or handling as a special node in the expression tree that lowers
      to a store-with-atomic.

- [x] **B tests** — Compilation + numeric parity tests for reshape,
      ravel, append, with-shape, and scatter-add.

### Phase C — Remaining gaps and hardening

- [x] **C.1 — i32 arithmetic support in GpuBinaryOp**: Currently the
      emitter defaults to `f32` LLVM ops.  Add type-aware emission
      so `i32` operands use `llvm.add`/`llvm.sub`/`llvm.mul`/`llvm.sdiv`.

- [x] **C.2 — i32/int comparison support**: `GpuCompareOp` currently
      uses `llvm.fcmp`.  Add `llvm.icmp` path for integer operands.

- [x] **C.3 — Mixed-type casts**: Support implicit casts when an
      expression mixes `i32` and `f32` operands (insert `GpuCast`).

- [x] **C.4 — Array-typed `HIRIf`**: When both branches produce
      arrays of the same shape, emit element-wise `GpuSelect` for
      each component.

- [x] **C.5 — Multi-rank array-valued folds**: Currently only rank-1
      array results are supported.  Extend to rank > 1 by recursively
      decomposing into nested component iterations.

- [x] **C.6 — Non-contiguous stride support**: Test and fix descriptor
      stride handling for non-contiguous layouts (e.g. subarray views
      with stride != 1).  The existing `_linear_index_lines` already
      multiplies by per-dimension strides, so this should mostly work.

- [x] **C.7 — Zero-size shape handling**: Bounds-check before any
      load/store when a dimension is zero.  Currently untested.

- [x] **C tests** — Targeted tests for each hardening item.

### Phase D — Integration and regression

- [x] **D.1 — Remove redundant specialised kernels**: The
      `build_descriptor_abi_f32_append_gpu_module` and
      `build_descriptor_abi_f32_scan_gpu_module` specialised kernels
      overlap with general-path capabilities after Phase B.  Remove them
      once the general path covers Append and the scan is deferred.

- [x] **D.2 — Update dispatch chain priority**: After Phase A and B,
      the general path handles most programs.  Move the general-path
      dispatch earlier in the chain (or make it the default fallback)
      and keep only truly specialised kernels (im2col, matmul) as
      early-dispatch overrides.

- [x] **D.3 — Full regression suite**: Run all CPU tests, all GPU
      tests, and all new parity tests.  Zero regressions.

- [x] **D.4 — Update PROJECT_OVERVIEW.md** to reflect achieved parity.

## Out of Scope (this plan)

These nodes either require complex multi-pass algorithms, dynamic memory
allocation, or are better served by dedicated kernel designs.  They remain
handled by specialised kernels or deferred:

- `HIRIm2col` / `HIRCol2im` — retain specialised GPU kernels
- Named-function / section callables (non-lambda)

**Resolved since original plan**: `HIRMatmul` (per-thread dot-product
kernel), `HIRSort` / `HIRGrade` (serial insertion sort), `HIRScan`
(parallel Hillis-Steele), `HIRFilter` / `HIRReplicate` (serial
single-thread — see Future Work for parallel upgrade), `HIRIndicesOf`
(parallel coordinate generation), `HIRFoldRight` (reverse fold in
general path).

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

## Future Work

### Phase E — Remaining correctness gaps

Programs that the CPU backend handles correctly but the GPU backend
gets wrong (silent wrong results) or rejects (compile error).

#### E.1 — Wrong-result gaps

These compile and run on GPU but produce incorrect output.

- [x] **E.1.1 — f32-only loads/stores**: The general map path, and all
      specialised kernels (matmul, sort, scan, scatter-add, filter,
      replicate), hardcode `f32` for `llvm.load` / `llvm.store` /
      `llvm.getelementptr`.  Integer (`i32`) and boolean (`i1`/`i8`)
      arrays that reach these paths get reinterpreted as `f32`, producing
      garbage.  Fix: thread `element_type` through `GpuInputLoad`,
      `GpuFlatLoad`, `GpuAppendLoad`, the output store in the general
      map builder, and all specialised kernel builders.

- [x] **E.1.2 — Scan operator**: `build_descriptor_abi_f32_scan_gpu_module`
      always uses `llvm.fadd` regardless of the `HIRScan.func` field.
      A product scan (`scan (*) 1 arr`) silently produces a sum scan.
      Fix: read the operator from the `HIRPrimCallable` and emit the
      correct LLVM op (`fadd` / `fmul`).

- [x] **E.1.3 — Scan exclusive flag**: The parallel scan kernel always
      produces an inclusive scan.  `escan` (exclusive) programs get
      inclusive results.  Fix: for exclusive scan, shift the output
      right by one position and insert the identity element at index 0.

- [x] **E.1.4 — Scan direction**: The kernel always scans left-to-right.
      Right-scan programs get left-scan results.  Fix: for right scan,
      reverse the input before scanning and reverse the output, or
      iterate the Hillis-Steele steps in reverse.

- [x] **E.1.5 — Rank > 1 append**: `GpuAppendLoad` only branches on
      the first coordinate.  Rank-2+ arrays appended along dimension 0
      need multi-dimensional coordinate decomposition for both the left
      and right descriptors.  **Resolved**: rank > 1 now raises
      `GPUScaffoldError` (compile-error) instead of producing wrong results.
      Full rank > 1 support deferred.

- [x] **E.1 tests** — Tests that verify correct output for i32 maps,
      product scan, exclusive scan, right scan, and rank-2 append.

#### E.2 — Compile-error gaps

These raise `GPUScaffoldError` or `CodegenUnavailable` on GPU but
compile and run correctly on CPU.

- [x] **E.2.1 — Named-function / section callables**: The general map
      builder requires `HIRLambda` as the map callable.  Named functions
      (`HIRVar` referencing a top-level def) and operator sections used
      directly as callables are rejected.  Fix: inline named functions
      by looking them up in the HIR function table, or lower them to
      equivalent lambda expressions before GPU compilation.

- [x] **E.2.2 — Filter with complex predicates**: The filter kernel
      only handles `HIRPrimCallable` comparisons with a literal constant.
      Lambda predicates (e.g. `\x -> x > 0 && x < 5`) are rejected.
      Fix: use the `GpuExpr` compiler to lower the predicate body to
      LLVM IR inline in the filter loop.

- [x] **E.2.3 — Scan for arrays > 1024**: The parallel Hillis-Steele
      scan uses one block with N threads, limiting N to 1024.  Larger
      arrays are rejected.  Fix: multi-block scan with inter-block
      prefix propagation, or fall back to a serial scan for N > 1024.

- [x] **E.2 tests** — Tests that verify compilation succeeds for named
      callables, complex filter predicates, and large scan arrays.

### Parallel GPU Filter and Replicate

`HIRFilter` and `HIRReplicate` currently use serial single-thread GPU
kernels (correct, O(N) on one core).  These should be upgraded to a
two-kernel parallel execution plan:

1. **Prefix-sum pass** — parallel Hillis-Steele scan on predicate results
   (filter) or counts (replicate) to compute per-element output positions.
2. **Scatter-write pass** — one thread per input element writes to its
   computed output position.

**Blocked on** `RemoraExecutor` multi-kernel orchestration.  Today the
executor assumes one kernel per function call.

See also `docs/FUTURE_WORK.md`.

### Tiled Shared-Memory Matmul

`HIRMatmul` currently uses a per-thread dot-product kernel.  A standard
CUDA tiled matmul with shared-memory tiles would improve throughput for
matrices larger than ~64×64.
