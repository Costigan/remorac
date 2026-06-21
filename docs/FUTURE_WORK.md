# Future Work

Items that are correct today but have a clear upgrade path for performance
or completeness.

## Parallel GPU Filter and Replicate

**Current state**: `HIRFilter` uses a parallel three-kernel plan for
N ≤ 1024: predicate evaluation, Hillis-Steele i32 prefix sum in shared
memory, and scatter-write.  `HIRReplicate` uses a parallel two-kernel
plan for N ≤ 1024: prefix sum on counts followed by scatter-replicate.
All kernels live in single PTX modules and are orchestrated by
``ExecutionPlan`` objects.  For N > 1024 or unsupported predicates,
serial single-thread fallbacks are used.

**Remaining work**: Multi-block parallel versions for N > 1024
(requires inter-block prefix propagation for scan).

**Where this is tracked**:
- `docs/GPU_CPU_PARITY_PLAN.md` — Future Work section
- Kernel docstrings in `remora/gpu_lowering.py`
- This file

## Host-Orchestrated Optimization Loops (GPU+CPU)

**Current state**: `examples/ad_optimize.lisp` compiles to a GPU
``LoopPlan`` via ``try_compile_state_fold_gpu`` in ``codegen.py``.
The pattern detector recognises ``fold body init (iota N)`` where the
body function does not use the step variable, compiles the body as a
single GPU map kernel via ``generate_mlir_descriptor_abi_ptx``, and
emits a ``LoopPlan`` with N iterations and double-buffer swapping.
``execute_program_on_gpu`` tries this path first; if the pattern does
not match or the GPU toolchain is unavailable, it falls back to the
standard IREE pipeline.

The fold body's init values are pre-loaded to the device via
``BufferSpec.init``.  On a CUDA-capable system, running
``remorac --syntax lisp --target gpu examples/ad_optimize.lisp``
executes the 200-step gradient descent on GPU.

## Tiled Shared-Memory Matmul

**Current state**: `HIRMatmul` uses a tiled shared-memory kernel with
TILE=16.  Each thread block cooperatively loads 16×16 tiles of A and B
into shared memory, then computes a partial dot product from the tiles.
This reduces global memory traffic by a factor of TILE compared to the
naive per-thread dot-product.  Edge tiles are bounds-checked and
zero-padded so non-TILE-aligned dimensions work correctly.  Falls back
to the naive per-thread kernel if the tiled version fails to compile.

## Multi-block Parallel Scan

**Current state**: The f32 scan uses a parallel Hillis-Steele kernel
for N ≤ 1024 (single block).  For 1024 < N ≤ 1,048,576, a four-kernel
multi-block scan is used: per-block local scan, extract block sums,
scan block sums, propagate prefixes.  All four kernels are in one PTX
module, orchestrated by an ``ExecutionPlan``.  For N > 1,048,576, the
serial single-thread fallback is used.

**Remaining work**: extend to arbitrary N via recursive multi-level scan.

## Parallel Sort and Grade

**Current state**: `HIRSort` and `HIRGrade` use parallel bitonic sort/grade
in shared memory for N ≤ 1024 (single block).  For N > 1024, multi-block
bitonic sort and grade use per-block local sorting followed by global
compare-swap merge steps via ``ExecutionPlan`` with double-buffered
global memory.  Odd blocks reverse their output to form bitonic sequences
at block boundaries.  Multi-block grade uses i32 index buffers with
value-lookup global merge steps.  Supports up to ~1M elements.

## Parallel Scatter-Add

**Current state**: `HIRScatterAdd` uses a parallel single-block kernel
for N ≤ 1024: all threads copy target to output in parallel, then
thread 0 performs the scalar add after a barrier.  For N > 1024, the
serial single-thread fallback is used.

**Upgrade path**: for N > 1024, use a two-kernel ``ExecutionPlan``
(parallel copy + single-thread add) or ``llvm.atomicrmw fadd`` for
the add step.
