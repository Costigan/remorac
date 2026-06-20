# Future Work

Items that are correct today but have a clear upgrade path for performance
or completeness.

## Parallel GPU Filter and Replicate

**Current state**: `HIRFilter` uses a parallel three-kernel plan for
N ≤ 1024: predicate evaluation, Hillis-Steele i32 prefix sum in shared
memory, and scatter-write.  All three kernels live in one PTX module
and are orchestrated by an ``ExecutionPlan``.  For N > 1024 or
non-comparison predicates, the serial single-thread fallback is used.
`HIRReplicate` still uses the serial single-thread kernel.

**Upgrade path for replicate**: same two-kernel pattern as filter —
prefix-sum the counts array, then scatter values to computed positions.
The ``ExecutionPlan`` infrastructure is in place; the remaining work is
writing the replicate-specific kernels.

**Where this is tracked**:
- `docs/GPU_CPU_PARITY_PLAN.md` — Future Work section
- Kernel docstrings in `remora/gpu_lowering.py`
- This file

## Host-Orchestrated Optimization Loops (GPU+CPU)

**Current state**: `examples/ad_optimize.lisp` runs a 200-step gradient
descent loop in pure Remora using `fold` with `(grad loss)`.  This works
on the interpreter and compiled CPU (via the state-fold `scf.for` lowering
with scalar decomposition).  On GPU, the fold loop has no lowering path —
the GPU backend only handles individual map/reduce/scan kernels, not
sequential state-carrying loops.

**Right architecture**: host-orchestrated heterogeneous execution, matching
what PyTorch does.  The optimization loop runs on the **CPU host**.  Each
iteration launches **GPU kernels** for the gradient computation and
parameter update:

```
for step in range(N):                  # CPU loop
    grad = gpu_kernel(grad_fn, params) # GPU: parallel gradient
    params = params - lr * grad        # GPU or CPU: element-wise update
```

The parallelism is *within* each step (the gradient computation), not
*across* steps (which are inherently sequential).

**Implementation plan**:

1. Compile the gradient function (`(grad loss)`) as a standalone GPU
   kernel via the existing dispatch chain.
2. Compile the parameter update (element-wise scale + subtract) as a
   second GPU kernel.
3. Add a **loop execution plan** to `RemoraExecutor`: a sequence of
   kernel launches with CPU-side loop control and small tensor transfers
   between steps.
4. The fold lowering detects state folds whose bodies contain
   GPU-compilable operations and emits a loop plan instead of a single
   kernel.

**Blocked on**: the same `RemoraExecutor` multi-kernel orchestration
needed for parallel filter/replicate.  The executor needs to support
launching multiple kernels per function call, with CPU-side control flow
between them.

**Status**: The ``ExecutionPlan`` infrastructure is now implemented with
``LoopPlan`` support for host-side iteration and buffer swapping.  The
next step is to compile the gradient function and parameter update as
separate GPU kernels, then emit a ``LoopPlan``-based
``ExecutionPlan`` from the state-fold lowering path.

**Example**: `examples/ad_optimize.lisp` — gradient descent on a
polynomial curve-fitting loss.  Currently produces
`[0.512337, 0.433115, 0.911621]` on interpreter and compiled CPU.
GPU execution would launch the gradient kernel 200 times from a CPU loop.

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
