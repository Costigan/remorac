# Future Work

Items that are correct today but have a clear upgrade path for performance
or completeness.

## Parallel GPU Filter and Replicate

**Current state**: `HIRFilter` and `HIRReplicate` use serial single-thread
GPU kernels.  Thread 0 iterates the input, evaluates the predicate (filter)
or reads the counts (replicate), and writes matching/replicated elements
contiguously to the output.  Correct for any input size but O(N) on one
SM core.

**Upgrade path**: two-kernel parallel execution plan.

1. **Prefix-sum pass** — run the existing parallel Hillis-Steele scan
   kernel on the predicate results (filter) or counts array (replicate)
   to compute per-element output positions.
2. **Scatter-write pass** — one thread per input element reads its
   output position from the scan result and writes to that position.

This requires `RemoraExecutor` to support **multi-step execution plans**
(launch scan, read back output length, allocate output, launch scatter).
The kernel builders for both passes are straightforward; the orchestration
is the new work.

**Blocked on**: `RemoraExecutor` multi-kernel orchestration.  Today the
executor assumes one kernel per function.

**Where this is tracked**:
- `docs/GPU_CPU_PARITY_PLAN.md` — Future Work section
- Kernel docstrings in `remora/gpu_lowering.py`
- This file

## Tiled Shared-Memory Matmul

**Current state**: `HIRMatmul` uses a per-thread dot-product kernel.  Each
thread computes one output element by iterating over K.  Correct but
memory-bandwidth-limited for large matrices.

**Upgrade path**: standard CUDA tiled matmul with TILE×TILE shared-memory
tiles, cooperative loading, and register blocking.  Would improve
throughput significantly for matrices larger than ~64×64.
