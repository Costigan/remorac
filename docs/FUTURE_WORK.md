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

**Current state**: `HIRMatmul` uses a per-thread dot-product kernel.  Each
thread computes one output element by iterating over K.  Correct but
memory-bandwidth-limited for large matrices.

**Upgrade path**: standard CUDA tiled matmul with TILE×TILE shared-memory
tiles, cooperative loading, and register blocking.  Would improve
throughput significantly for matrices larger than ~64×64.

## Multi-block Parallel Scan

**Current state**: The parallel Hillis-Steele scan handles arrays up to
1024 elements (one block).  Arrays larger than 1024 fall back to a serial
single-thread kernel — correct but O(N) on one core.

**Upgrade path**: multi-block scan with inter-block prefix propagation
(Blelloch or decoupled look-back).  Each block scans its tile in shared
memory, then a second pass propagates block-level prefixes.  Would bring
large-array scan to O(N/P).

## Parallel Sort and Grade

**Current state**: `HIRSort` and `HIRGrade` use serial insertion sort
(single-thread, O(N²)).  Correct for any size but impractical for
arrays larger than a few hundred elements.

**Upgrade path**: bitonic sort in shared memory for small arrays
(N ≤ 1024), or a radix sort for larger arrays.  Both are well-known
GPU algorithms with O(N log²N) or O(N·W) work.

## Parallel Scatter-Add

**Current state**: `HIRScatterAdd` uses a serial single-thread kernel
that copies the target to the output and then performs a single add.
Correct but sequential.

**Upgrade path**: parallel copy (one thread per element) followed by
a single atomic add, or a fully parallel scatter with `llvm.atomicrmw
fadd` for each update position.  Would scale with array size.
