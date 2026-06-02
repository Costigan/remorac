# Remora Buffer Reuse and Arena Allocation Plan

This document outlines the strategy for reducing memory allocation overhead and
enabling buffer reuse for intermediate tensors in Remora Dense Core.

## Current State

- Every intermediate tensor that survives fusion results in a new `memref.alloc`
  (which lowers to `malloc`).
- Many allocations could be reused if they have compatible shapes and disjoint
  lifetimes.
- The current MLIR pipeline uses `one-shot-bufferize`, which does some in-place
  optimization but still leaves many allocations.

## Goal

- Reduce the number of dynamic allocations per execution.
- Enable reuse of buffers for intermediate results.
- Prepare for a "static arena" or "pre-allocated scratchpad" model for better
  performance and predictability.

## Proposed Strategy

### 1. Buffer Reuse Pass

We will leverage MLIR's buffer optimization passes. Specifically, we should
investigate:
- `buffer-deallocation`: to ensure buffers are freed as early as possible.
- `buffer-hoisting`: to move allocations out of loops.
- `buffer-loop-hoisting`: to move allocations out of loops.

### 2. Manual Buffer Reuse in Remora Pipeline

If MLIR's built-in passes are not sufficient, we can implement a custom
Remora-level pass that:
- Analyzes the lifetimes of intermediate tensors.
- Identifies tensors that can share the same buffer.
- Replaces multiple `memref.alloc` calls with a single allocation and
  appropriate `memref.view` or `memref.subview` operations.

### 3. Arena Allocator

Implement an arena allocator where a large chunk of memory is allocated once
per program execution (or even once per session) and used to satisfy all
internal allocation requests.

- **CPU Arena**: Use a single large `malloc` or a pool of pre-allocated buffers.
- **GPU Arena**: Use `cudaMalloc` once and manage it with a simple sub-allocator
  to avoid the high cost of frequent device allocations/frees.

### 4. Integration with `RemoraExecutor`

- The `RemoraExecutor` should be aware of the total scratchpad memory required
  by a kernel.
- The compiler should emit metadata indicating the maximum temporary memory
  needed.
- `RemoraExecutor` can then provide a pre-allocated "workspace" buffer to the
  kernel.

## Implementation Steps

1. **Research**: Evaluate MLIR's `buffer-deallocation-pipeline` more deeply on
   current Remora-generated MLIR.
2. **Analysis**: Use `remora-bench` to track `allocation_count` as we experiment
   with these passes.
3. **Prototype**: Add `buffer-deallocation-pipeline` to the `CPU_PIPELINE` and
   measure the impact.
4. **Arena implementation**: Modify `lowering.py` to optionally use a provided
   workspace buffer instead of generating `memref.alloc`.

## Success Metrics

- Decrease in `allocation_count` reported by `remora-bench`.
- Reduced execution time for programs with many intermediate tensors.
- Zero regressions in correctness tests.
