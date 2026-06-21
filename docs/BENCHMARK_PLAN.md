# Benchmark Plan: Remora vs NumPy / JAX / Futhark

Systematic performance comparison of Remora-compiled code against
hand-written NumPy, JAX `jit`, and Futhark for common array
operations. Validates whether rank polymorphism compiles to
competitive code.

## Current State

- `remora/benchmark.py` measures **compiler metrics** (compile time,
  MLIR sizes, fusion counts, peak RSS) — not runtime execution.
- `tests/test_benchmarks.py` has warmup+median **execution timing**
  for `map` and `fold` on CPU, GPU, and interpreter.
- `docs/BENCHMARK_BASELINES.json` contains structural ceilings only
  (fused-generic counts, allocation counts) — no wall-clock baselines.
- JAX and Futhark are discussed in docs but neither is installed or
  benchmarked.

## Operation Coverage by Backend

| Operation          | Remora CPU          | Remora GPU           | NumPy          | JAX            | Futhark      |
| ------------------ | ------------------- | -------------------- | -------------- | -------------- | ------------ |
| map (scale)        | compiled            | compiled             | `xs * 2.0`     | `jnp.multiply` | `map (*2.0)` |
| fold (sum)         | compiled            | compiled             | `np.sum`       | `jnp.sum`      | `reduce (+)` |
| scan (prefix sum)  | interp only         | Hillis-Steele kernel | `np.cumsum`    | `jnp.cumsum`   | `scan (+)`   |
| matmul             | compiled (fold+map) | tiled shared-memory  | `a @ b`        | `jnp.matmul`   | built-in     |
| sort               | interp only         | bitonic kernel       | `np.sort`      | `jnp.sort`     | `radix_sort` |
| stencil (3x3 blur) | compiled (im2col)   | compiled (im2col)    | sliding window | `jax.lax.conv` | stencil map  |

Scan and sort have no CPU compiled backend. CPU benchmarks for
those operations report interpreter time with a note.

## Phase 1: Benchmark Harness

Create `remora/benchmark_suite.py` (separate from the compiler-metrics
`benchmark.py`), focused on runtime execution performance.

### Timer utility

Warmup N runs, then measure M runs. Report median, min, and stddev
in seconds. Use `time.perf_counter` for CPU. For GPU, synchronize
before each timing boundary (wall-clock with sync, or CUDA event
elapsed if available).

### Size sweep

Each benchmark runs at sizes `[100, 1_000, 10_000, 100_000, 1_000_000]`. Matmul uses `[32, 64, 128, 256, 512, 1024]`.

### Result schema

JSON output per run:

```json
{
  "operation": "map_scale",
  "backend": "remora-cpu",
  "size": 100000,
  "median_s": 0.00042,
  "min_s": 0.00039,
  "std_s": 0.00003,
  "throughput_elem_per_s": 238095238
}
```

### CLI entry point

Add `remora-perf` script in `pyproject.toml`:

```
remora-perf = "remora.benchmark_suite:main"
```

Flags:

- `--ops map,fold,scan,matmul,sort,stencil` (select operations)
- `--backends remora-cpu,remora-gpu,numpy,jax-cpu,jax-gpu,futhark`
- `--sizes 1000,10000,100000`
- `--warmup 5 --trials 20`
- `--json results.json`

## Phase 2: Remora Benchmark Programs

Write each as a Remora source string compiled via the existing API.

| Operation          | Remora Source                                                 | CPU path                                       | GPU path                                                      |
| ------------------ | ------------------------------------------------------------- | ---------------------------------------------- | ------------------------------------------------------------- |
| map (scale)        | `def f xs = map (* 2.0) xs`                                   | `compile_function_source` -> `.so` -> `ctypes` | `compile_function_source_to_mlir_gpu_ptx` -> `RemoraExecutor` |
| fold (sum)         | `def f xs = fold (+) 0.0 xs`                                  | same                                           | same                                                          |
| scan (prefix sum)  | `(define/pi () (f [xs (Array Float N)] ...) (scan + 0.0 xs))` | interpreter                                    | GPU scan kernel                                               |
| matmul             | `(define/pi () (f [a ... b ...] ...) (matmul a b))`           | CPU compiled (fold+map)                        | GPU tiled matmul                                              |
| sort               | `(define/pi () (f [xs (Array Float N)] ...) (sort xs))`       | interpreter                                    | GPU bitonic sort                                              |
| stencil (3x3 blur) | `im2col image [3 3] 1` + fold-dot                             | CPU compiled                                   | GPU compiled                                                  |

## Phase 3: NumPy Baselines

NumPy is already a dependency. Each operation has a direct
equivalent:

| Operation          | NumPy implementation                                    |
| ------------------ | ------------------------------------------------------- |
| map (scale)        | `xs * 2.0`                                              |
| fold (sum)         | `np.sum(xs)`                                            |
| scan (prefix sum)  | `np.cumsum(xs)`                                         |
| matmul             | `a @ b`                                                 |
| sort               | `np.sort(xs)`                                           |
| stencil (3x3 blur) | `scipy.ndimage.uniform_filter` or manual sliding-window |

## Phase 4: JAX Baselines

Add `jax` as an optional dependency (extras group `[bench]` in
`pyproject.toml`).

Each benchmark follows the pattern:

```python
import jax, jax.numpy as jnp

@jax.jit
def scale(xs):
    return xs * 2.0

# warmup
scale(jnp.ones(n)).block_until_ready()

# measure
for _ in range(trials):
    t0 = time.perf_counter()
    scale(data).block_until_ready()
    t1 = time.perf_counter()
```

JAX operations:

| Operation         | JAX implementation |
| ----------------- | ------------------ |
| map (scale)       | `xs * 2.0`         |
| fold (sum)        | `jnp.sum(xs)`      |
| scan (prefix sum) | `jnp.cumsum(xs)`   |
| matmul            | `jnp.matmul(a, b)` |
| sort              | `jnp.sort(xs)`     |
| stencil           | `jax.lax.conv`     |

For GPU: JAX automatically uses GPU when available. Use
`jax.devices('gpu')` to confirm placement.

## Phase 5: Futhark Baselines (lower priority)

Futhark requires the `futhark` compiler binary (Haskell, installed
via `cabal` or prebuilt binaries).

Each benchmark is a `.fut` file, e.g.:

```futhark
-- map_scale.fut
entry main (xs: []f32) : []f32 = map (*2.0) xs
```

Compilation: `futhark c --library map_scale.fut` (CPU) or
`futhark cuda --library map_scale.fut` (GPU). Call from Python
via `ctypes` or the `futhark-data` format with `subprocess`.

This phase is the most labor-intensive. Defer until the
Remora/NumPy/JAX comparison is complete. Futhark results can be
added later without changing the harness.

Store `.fut` files in `benchmarks/futhark/`.

## Phase 6: Results and Analysis

### Persistence

Store results in `benchmarks/results/` with a timestamp and
hardware description (CPU model, GPU model, driver version).

### Plotting

`tools/plot_benchmarks.py` — matplotlib bar charts grouped by
operation, series by backend. Log-scale throughput (elements/sec)
vs problem size.

### Regression baseline

Save `benchmarks/baselines.json` with acceptable performance ranges.
CI can optionally check for regressions (GPU CI may not be
available).

### Summary table

Auto-generate a markdown table showing relative speedup vs NumPy
for inclusion in docs.

## Integration

Add to `pyproject.toml`:

```toml
[project.optional-dependencies]
bench = ["jax", "matplotlib"]

[project.scripts]
remora-perf = "remora.benchmark_suite:main"
```

## Recommended Order

1. Harness + Remora CPU + NumPy for map/fold/matmul/stencil
   (quickest value).
1. Add JAX CPU comparison.
1. Add Remora GPU for all 6 ops + JAX GPU.
1. Add Futhark (if the compiler is available on the target machine).

## Estimated Effort

| Phase                 | Lines    | Notes                                       |
| --------------------- | -------- | ------------------------------------------- |
| 1 (harness)           | ~200     | `benchmark_suite.py` + CLI                  |
| 2 (Remora programs)   | ~150     | source strings + compile/execute wrappers   |
| 3 (NumPy)             | ~80      | direct NumPy calls                          |
| 4 (JAX)               | ~120     | `@jax.jit` wrappers, `.block_until_ready()` |
| 5 (Futhark)           | ~200     | 6 `.fut` files + subprocess driver          |
| 6 (plotting/analysis) | ~150     | matplotlib + JSON persistence               |
| **Total**             | **~900** | excluding Futhark: ~700                     |

The core Remora-vs-NumPy-vs-JAX comparison (phases 1-4, 6) is a
1-2 day task given the existing infrastructure. Futhark adds
another day.
