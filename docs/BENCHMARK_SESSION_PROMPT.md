Read @AGENTS.md for project conventions and commands.

Read @docs/PROJECT_OVERVIEW.md for architecture context.

Read @docs/BENCHMARK_IMPROVEMENT_PLAN.md for the detailed plan you
will be executing.  It has six phases with checkboxes.  Start with
Phase 1 (Quick Wins) and work through the items in order.

Read @benchmarks/results/REPORT.md for the current benchmark
baseline numbers you are trying to improve.

## What was done in the previous session

1. **GPU device memory pool**: added `_pool_alloc`/`_pool_free` to
   `RemoraExecutor` in `remora/executor.py`.  Buffers are recycled
   by size across `execute()` and `execute_plan()` calls.  `close()`
   drains the pool.

2. **Benchmark suite**: created `remora/benchmark_suite.py` with
   `remora-perf` CLI entry point.  Benchmarks six operations (map,
   fold, scan, matmul, sort, stencil) across four backends (numpy,
   jax-gpu, remora-cpu, remora-gpu).  Compile-once-execute-many
   pattern.  JSON output via `--json`.

3. **Compiler fixes to support all operations on CPU**:
   - Added `_lower_scan_tensor_input()` and `_lower_sort_tensor_input()`
     to `remora/lowering/tensor_ops.py`, with handlers in
     `_lower_tensor_input()`.
   - Added sort extern auto-detection in
     `_lower_function_descriptor_module()` in `remora/lowering/module.py`.
   - Added `(matmul a b)` as a source-level Lisp operation across
     `ast_nodes.py`, `lisp_reader.py`, `typechecker.py`, `hir.py`,
     `runtime.py`, and `hir_opt.py`.
   - Fixed `_lower_matmul_tensor_input()` to zero-initialize the
     output tensor before `linalg.matmul`.

4. **Documentation**: `docs/BENCHMARK_PLAN.md`,
   `docs/BENCHMARK_IMPROVEMENT_PLAN.md`,
   `benchmarks/results/REPORT.md`.

## Key performance gaps to address (Phase 1)

- **CPU matmul**: 1.1M elem/s at 512x512 (248ms).  Fix: call BLAS
  `cblas_sgemm` from `remora_rt.c` instead of `linalg.matmul` naive
  loops.  Target: >50M elem/s.

- **CPU sort**: 11M elem/s at 1M (30x slower than NumPy).  Fix:
  eliminate tensor→memref copy loops.  Target: >30M elem/s.

- **GPU scan**: limited to n<=1024.  Fix: wire multi-block scan plan
  into the function-compilation path.  Target: works at 1M.

## How to verify

```bash
uv run python -m compileall -q remora       # compile check
uv run pytest tests/test_executor.py -q -x  # executor tests
uv run remora-perf --ops matmul --backends remora-cpu,numpy --sizes 64,128,256,512  # benchmark
```

## Rules

- Do not add comments to code unless asked.
- All compiler changes must be general (no benchmark-specific code
  in the compiler).
- Run `uv run python -m compileall -q remora` after every edit.
- Test correctness before benchmarking performance.
- Update `benchmarks/results/REPORT.md` after each performance
  improvement.
- Check off completed items in `docs/BENCHMARK_IMPROVEMENT_PLAN.md`.
