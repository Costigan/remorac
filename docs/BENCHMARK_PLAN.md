# Benchmark Plan: Remora vs NumPy / JAX / Futhark

Systematic performance comparison of Remora-compiled code against
hand-written NumPy, JAX `jit`, and Futhark for common array
operations. Validates whether rank polymorphism compiles to
competitive code.

## Current State

The benchmark suite (`remora/benchmark_suite.py`) and CLI
(`remora-perf` / `remora-bench`) are fully implemented.

- `remora/benchmark_suite.py` measures **execution time** (not compilation
  time) for 6 core operations across 5 backends: numpy, jax-gpu,
  remora-cpu, remora-gpu, and remora-gpu-htod (device-resident GPU).
- `remora/benchmark.py` (`remora-bench`) measures **compiler metrics**
  (compile time, MLIR sizes, fusion counts, allocation counts, peak RSS)
  for a single source program.
- Both compiled CPU (`compile_function_source` → `.so` → ctypes) and
  compiled GPU (`compile_function_source_to_mlir_gpu_ptx` →
  `RemoraExecutor`) paths are exercised.
- A representative **pipeline benchmark** (`cumsum(sort(xs))`) exercises
  multi-kernel device-resident chaining.
- Results are saved as JSON with timestamps and hardware metadata.
- A plotting tool (`tools/plot_benchmarks.py`) generates charts from
  result files.

Two entry points are registered in `pyproject.toml`:

```
[project.scripts]
remora-bench = "remora.benchmark:main"
remora-perf = "remora.benchmark_suite:main"
```

## CLI

### `remora-perf` — execution time benchmarks

```
uv run remora-perf --ops map,fold,scan,matmul,sort,stencil \
    --backends numpy,jax,remora-cpu,remora-gpu \
    --sizes 100000,1000000 \
    --warmup 5 --trials 20 \
    --json results.json
```

Flags:

- `--ops` — one or more of: `map`, `fold`, `scan`, `matmul`, `sort`, `stencil`,
  `fusion`, `grad_descent`, `conv_pipeline`, `nbody`, `pipeline`
- `--backends` — one or more of: `numpy`, `jax`, `remora-cpu`, `remora-gpu`,
  `remora-gpu-htod`
- `--sizes` — comma-separated problem sizes
- `--warmup N --trials N` — warmup and timed iterations
- `--json FILE` — save results
- `--device-resident` — measure GPU kernels without H↔D transfer
- `--no-pool` — bypass device memory pool (measure pool overhead)
- `--profile` — per-kernel timing breakdown (plan-based ops)

### `remora-bench` — compiler metrics

```
uv run remora-bench examples/prelude_sum.remora
uv run remora-bench --cpu-threads 1 --json result.json examples/prelude_sum.remora
```

Measures compile time, MLIR/LLVM sizes, linalg generic counts (before
and after fusion), allocation counts, and peak RSS. Does not measure
execution throughput.

## Operation Coverage

| Operation          | Remora CPU           | Remora GPU            | NumPy          | JAX            |
| ------------------ | -------------------- | --------------------- | -------------- | -------------- |
| map (scale)        | compiled             | compiled              | `xs * 2.0`    | `jnp.multiply` |
| fold (sum)         | compiled             | compiled              | `np.sum`       | `jnp.sum`      |
| scan (prefix sum)  | compiled             | compiled (≤1M)        | `np.cumsum`    | `jnp.cumsum`   |
| matmul             | compiled (C kernel)  | tiled shared-memory   | `a @ b`        | `jnp.matmul`   |
| sort               | compiled (LSD radix) | GPU radix (N>1024)    | `np.sort`      | `jnp.sort`     |
| stencil (3x3 blur) | compiled (im2col)    | compiled (im2col)     | sliding window | `jax.lax.conv` |

Additional benchmarks:

| Benchmark         | Description                                                     |
| ----------------- | --------------------------------------------------------------- |
| fusion            | Composed vs manually-fused op chains (map_chain, triple, dot)   |
| grad_descent      | 200-step AD gradient descent (polynomial curve fitting)         |
| conv_pipeline     | conv → relu → sum-pool forward pass                             |
| nbody             | All-pairs gravitational N-body step (CPU only; GPU miscompile)  |
| pipeline          | `cumsum ∘ sort` device-resident vs host-chained vs JAX          |

## Remora Benchmark Programs

CPU benchmarks are compiled via `compile_function_source` with
parameterized element types. GPU benchmarks compile to HIR
and call `generate_mlir_descriptor_abi_ptx` (for sort, scan, matmul)
or `RemoraExecutor` directly (for map, fold, stencil). The GPU
scan benchmark uses multi-block Hillis-Steele kernels for
N > 1024.

All benchmarks use a compile-once-execute-many pattern: the artifact
is compiled during warmup, then reused across timed trials.

## NumPy Baselines

Uses direct `numpy` calls (already a dependency). No special setup.

## JAX Baselines

Uses `jax.numpy` + `@jax.jit` with `.block_until_ready()` for timing.
JAX is an optional dependency. When available, JAX automatically uses
GPU when `cuda12` is installed.

## Futhark Baselines

Deferred. Requires the `futhark` compiler binary (via `cabal` or
prebuilt). Would live in `benchmarks/futhark/` with one `.fut` file
per operation. Python driver via ctypes or subprocess with
`futhark-data` format.

## Results

See `benchmarks/results/REPORT.md` for the latest performance data
along with analysis. JSON result files are timestamped and stored in
`benchmarks/results/`. Plotting via `tools/plot_benchmarks.py`.

## Performance Targets

(Aspirational — from `docs/BENCHMARK_IMPROVEMENT_PLAN.md`.)

| Operation   | remora-cpu target      | remora-gpu target    |
| ----------- | ---------------------- | -------------------- |
| map 1M      | > 2G (vectorized)      | > 2G                 |
| fold 1M     | > 2G (threaded)        | > 5G                 |
| scan 1M     | > 1G (vectorized)      | > 1G (multi-block)   |
| matmul 512  | > 100M (BLAS)          | > 1G                 |
| sort 1M     | > 50M (radix)          | > 500M (radix)       |
| stencil 512 | > 100M (vectorized)    | > 2G                 |

## Future Work

Items from `docs/BENCHMARK_IMPROVEMENT_PLAN.md` not yet completed:

- [ ] CPU vectorization pass calibration (now enabled by default, needs
  throughput measurement and tuning)
- [ ] Multi-threaded CPU fold via OpenMP / `scf.parallel`
- [ ] Kernel fusion: merge element-wise chains into single kernels
- [ ] Futhark benchmarks (requires `futhark` compiler installation)
- [ ] GPU plan profiling: per-kernel CUDA event timing and resource
  (register/shared-mem/occupancy) reporting via `--profile`
- [ ] Nsight ground-truth cross-validation of host-side timing
- [ ] Compiler plan composition: sub-expression GPU ops lowering into
  combined device-resident plans
