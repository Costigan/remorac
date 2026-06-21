# Benchmark Improvement and Performance Plan

Observations from the initial benchmark run (2026-06-21, RTX 5090
Laptop GPU) and plan for improving both the benchmarks and the
compiler.  Organized into phases by effort and dependency order.

## Baseline Numbers (2026-06-21)

Reference throughput at the largest tested size per operation:

| Operation | numpy | jax-gpu | remora-cpu | remora-gpu |
|-----------|-------|---------|------------|------------|
| map 1M    | 6.24G | 27.2G   | 536M       | 1.57G      |
| fold 1M   | 8.89G | 34.9G   | 454M       | 4.03G      |
| scan 1M   | 522M  | 19.1G   | 502M       | n<=1024    |
| matmul 512| 87M   | 3.10G   | 1.1M       | 787M       |
| sort 1M   | 329M  | 4.19G   | 11.2M      | 68.4M      |
| stencil 512| 23.6M| 5.52G   | 58.3M      | 1.07G      |

---

## Phase 1: Quick Wins

Low effort, high impact.  No architectural changes required.
Estimated time: 1-2 days.

### 1.1 CPU matmul — BLAS dispatch

Replace naive `linalg.matmul` → nested loops with a C runtime
call to BLAS `cblas_sgemm`.  Same pattern used for sort (memref
→ C runtime → memref).

**Target**: 100-1000x improvement (1.1M → 100M-1G elem/s).

Files to modify:
- `remora/remora_rt.c`
- `remora/lowering/tensor_ops.py` (`_lower_matmul_tensor_input`)
- `remora/lowering/module.py` (extern detection regex)
- `remora/runtime.py` (linker flags for `-lblas`)

Tasks:
- [x] Add `remora_matmul_f32(a, b, c, M, K, N)` to `remora_rt.c`
      calling `cblas_sgemm` with `CblasRowMajor`, `CblasNoTrans`
      (only when an optimized BLAS — OpenBLAS/BLIS — is present;
      otherwise a register-tiled, cache-blocked `-O3 -march=native`
      C kernel is used, since the reference Netlib BLAS gives no
      speedup over naive loops)
- [x] Add `#include <cblas.h>` and link with `-lblas` in the
      `_get_remora_rt_o()` compile command (`runtime.py`).
      Implemented as: manual `cblas_sgemm` prototype gated behind
      `-DREMORA_HAVE_BLAS`; runtime links `-lopenblas`/`-lblis` only
      when `_find_optimized_blas()` finds one
- [x] Modify `_lower_matmul_tensor_input()` to emit:
      copy left tensor → memref, copy right tensor → memref,
      `func.call @remora_matmul_f32(...)`, copy result memref →
      tensor
- [x] Add `remora_matmul_f32` to the extern auto-detection regex
      in `_lower_function_descriptor_module()`
- [x] Verify: `uv run python -m compileall -q remora`

Tests:
- [x] Existing matmul test still passes:
      `uv run pytest tests/test_executor.py -q -x`
- [x] Manual correctness check: 512×512 matmul matches `np.matmul`
      (allclose rtol=1e-4, atol=1e-3; exact 1e-5 is too tight for
      a different accumulation order)
- [x] Benchmark: `uv run remora-perf --ops matmul --backends
      remora-cpu,numpy --sizes 64,128,256,512`
- [~] Verify remora-cpu matmul throughput > 50M elem/s at 512×512
      (exceeds 50M at sizes ≤256: 80-107M; at 512 the median is
      ~37M / min ~54M — single-thread cache-bound.  Up from 1.1M
      baseline, a ~34x improvement.  Reaching >50M median at 512
      needs threading or a full GEMM microkernel; deferred.)

### 1.2 CPU sort — eliminate copy overhead

The current sort path copies tensor → memref → qsort → tensor.
The two copy loops were assumed to dominate (30x overhead over
NumPy).  Use MLIR bufferization or in-place memref operations to
eliminate copies.

**Target**: 2-5x improvement (11M → 30-50M elem/s at 1M).

**NOTE (revised after profiling):** the tensor→memref copy is
*not* the bottleneck.  Profiling `remora_rt.c` directly showed that
for 100K f32, the `memcpy`-equivalent copy costs ~0.01ms while
`qsort` costs ~7.9ms — i.e. `qsort` (with its per-comparison
function-pointer indirect call) is ~99.9% of the time.  Removing
the copy would save nothing measurable.  The actual fix was to
replace `qsort` with an LSD radix sort for f32 in the C runtime.

Files to modify:
- `remora/remora_rt.c` (`remora_sort_f32` → radix sort)

Tasks:
- [x] Replace `qsort` for f32 with a 4-pass 8-bit LSD radix sort
      (`_remora_radix_sort_f32`).  f32 values are mapped to
      monotonic uint32 keys (flip sign bit for positives, flip all
      bits for negatives), radix-sorted, then mapped back.  Falls
      back to `qsort` for n < 64.
- [x] (Copy elimination was found unnecessary — see note above.)
- [x] Verify: `uv run python -m compileall -q remora`

Tests:
- [x] Sort correctness: random arrays (n=1000, 50000) match
      `np.sort` exactly; negative floats handled correctly
- [x] Benchmark: `uv run remora-perf --ops sort --backends
      remora-cpu,numpy --sizes 1000,10000,100000`
- [x] Verify remora-cpu sort throughput > 30M elem/s at 100K
      (achieved 167.75M elem/s at 100K, 143M at 1M — a ~12.6x
      improvement over the 13.3M qsort baseline)

### 1.3 GPU scan — multi-block

Wire the existing four-kernel multi-block scan infrastructure
into the function-compilation path.  The kernels exist in the
main-module path; they need to be accessible via
`generate_mlir_descriptor_abi_ptx()` or a similar entry point.

**Target**: enable GPU scan for n > 1024 (up to 1M).

Files to modify:
- `remora/codegen.py` or `remora/gpu_lowering.py`
- `remora/benchmark_suite.py` (remove n>1024 skip)

Tasks:
- [x] Identify how the main-module scan path dispatches to the
      multi-block plan (`_lower_scan_module` in `tensor_ops.py`).
      Found that the multi-block 4-kernel scan plan already existed
      in `codegen.py` (lines 843-887) and the kernels in
      `gpu_lowering.py` (`build_descriptor_abi_multiblock_f32_scan_gpu_module`),
      but were dead code: the single-block builder never raised
      `GPUScaffoldError` for N>1024 (it had a serial fallback), so
      the multi-block branch was never reached.
- [x] Expose multi-block scan through
      `generate_mlir_descriptor_abi_ptx()` when the scan dimension
      exceeds 1024.  Implemented by gating the single-block builder
      (`build_descriptor_abi_f32_scan_gpu_module`) to raise
      `GPUScaffoldError` for inclusive left-to-right add scans with
      1024 < N <= 1024*1024, which triggers the existing multi-block
      dispatch.  Exclusive/right/mul scans and N>1M keep the serial
      fallback.
- [x] Return an `ExecutionPlan` with the four scan kernels
      (already built in `codegen.py`; now reachable)
- [x] Update `bench_scan_remora_gpu` to use `execute_plan` when
      a plan is returned (now compiles to HIR then calls
      `generate_mlir_descriptor_abi_ptx` directly, like sort)
- [x] Verify: `uv run python -m compileall -q remora`

Tests:
- [x] GPU scan at n=10000 produces correct prefix sums (matches
      `np.cumsum`)
- [x] GPU scan at n=100000 and n=1000000 produce correct results
- [x] Benchmark: `uv run remora-perf --ops scan --backends
      remora-gpu --sizes 1000,10000,100000,1000000`
      (10K: 68M, 100K: 578M, 1M: 1.50G elem/s — exceeds the 1G
      multi-block goal).  Existing GPU scan tests still pass
      (`REMORA_TEST_GPU=1 pytest -k scan`: 23 passed).

### 1.4 Larger benchmark sizes

Extend the default size arrays to test GPU at scale.

**Target**: show GPU performance at 10M+ elements.

Files to modify:
- `remora/benchmark_suite.py` (size constants)

Tasks:
- [x] Add 10_000_000 to `DEFAULT_SIZES`
- [x] Add 1024 to `MATMUL_SIZES`
- [x] Add 1024 to `STENCIL_SIZES`
- [x] Run full benchmark and verify no crashes at larger sizes

Tests:
- [x] `uv run remora-perf --ops map,fold --sizes 10000000
      --backends numpy,remora-cpu,remora-gpu` (no crashes; map-gpu
      548M, fold-gpu 3.08G at 10M)
- [x] `uv run remora-perf --ops matmul,stencil --sizes 1024
      --backends remora-cpu,remora-gpu` (no crashes; matmul-gpu
      719M, stencil-gpu 1.63G at 1024)

### 1.5 Memory pool impact benchmark

Add a `--no-pool` flag that bypasses the device memory pool to
quantify allocation savings.

**Target**: measure pool overhead reduction in microseconds.

Files to modify:
- `remora/benchmark_suite.py`
- `remora/executor.py` (optional pool bypass)

Tasks:
- [x] Add `RemoraExecutor.set_pool_enabled(bool)` method that
      routes `_pool_alloc`/`_pool_free` to direct `_rt.alloc`/
      `_rt.free` when disabled (and drains the pool when disabled)
- [x] Add `--no-pool` flag to `remora-perf` CLI
- [x] When `--no-pool` is set, disable pool before GPU benchmarks
      (module-level `_POOL_ENABLED` applied via `_apply_pool(exe)`
      in every GPU benchmark)
- [x] Run: map and fold at 100K with pool on and off, report
      median difference

Tests:
- [x] Pool-disabled mode still produces correct results
      (executor tests pass with GPU enabled)
- [x] Report pool savings in microseconds per call:
      map 100K — pool ON 100.4us vs OFF 153.4us (~53us saved);
      fold 100K — pool ON 57.7us vs OFF 111.0us (~53us saved).
      The pool avoids a per-call cudaMalloc/cudaFree pair,
      saving ~53us/call.

---

## Phase 2: Benchmark Coverage

Expand what is measured.  No compiler changes required (except
device-resident API).  Estimated time: 2-3 days.

### 2.1 End-to-end application benchmarks

Add three real-world benchmarks to `benchmark_suite.py`.

#### 2.1a Gradient descent

Compile `ad_optimize.lisp` (200-step polynomial curve-fitting)
and measure total wall-clock time.  Compare against JAX
equivalent.

Tasks:
- [x] Write JAX gradient descent: define polynomial loss, use
      `jax.grad` + `jax.jit` (with `jax.lax.fori_loop`), run 200
      steps, time execution
- [x] Write Remora GPU benchmark: compile the parameterized
      `ad_optimize` source, build the state-fold GPU plan via
      `try_compile_state_fold_gpu`, time `execute_plan` (compile once)
- [x] Add CPU Remora version via `CPUExecutor.compile_source` +
      `execute_main` (compile once, execute many)
- [x] Add `bench_grad_descent_{numpy,jax,remora_cpu,remora_gpu}`
- [x] Add `"grad_descent"` to `ALL_OPS` (+ `GRAD_DESCENT_SIZES`)

Tests:
- [x] All backends produce the same result within `rtol=1e-3`:
      `[0.512337, 0.433115, 0.911621]` (verified numpy, jax,
      remora-cpu, remora-gpu)
- [x] Benchmark runs without error.  Timings (200 steps):
      remora-cpu 16us, numpy 1.15ms, jax-gpu 1.41ms, remora-gpu
      5.18ms.  remora-cpu wins big — the whole optimization runs as
      tight native code, while jax/GPU pay per-step launch overhead
      on this tiny 3-parameter problem.

#### 2.1b Convolution pipeline

Three-layer conv → relu → pool forward pass.

Tasks:
- [x] Write Remora source: im2col → fold-dot → map relu → pool.
      Pooling is a 4-window **sum** pool over the flattened
      activation (reshape → fold).  Average pooling needs a
      per-element `/` inside the map, which hits a CPU-lowering gap
      (missing `_mlir_ciface_remora_call` symbol); sum pooling keeps
      the pipeline faithful and exactly matchable.
- [x] Write JAX equivalent: 9 shifted-slice multiply-adds for the
      conv (exactly matches im2col fold-dot, unlike `lax.conv`'s
      correlation/padding conventions) → `jnp.maximum` relu →
      reshape-sum pool
- [x] Write NumPy equivalent using `sliding_window_view` +
      `np.maximum` relu + reshape-sum pool
- [x] Add `bench_conv_pipeline_{numpy,jax,remora_cpu}` functions
- [x] Add `"conv_pipeline"` to `ALL_OPS` (+ `CONV_PIPELINE_SIZES`)

Tests:
- [x] All backends produce numerically equivalent outputs
      (remora vs numpy maxerr 2.4e-7; jax vs numpy maxerr 4.8e-7)
- [x] Benchmark runs at input sizes 32×32, 64×64, 128×128

#### 2.1c N-body step

One gravitational N-body timestep (all-pairs forces + update).

Tasks:
- [x] Extract N-body step source from `tests/test_nbody.py`
      (`_nbody_source_compiled` form with `let*` bindings)
- [x] Write JAX equivalent: vectorized all-pairs force computation
- [x] Write NumPy equivalent: broadcasting-based pairwise distances
- [x] Add `bench_nbody_{numpy,jax,remora_cpu}` functions.
      `remora_gpu` is **omitted**: the general-map GPU lowering
      miscompiles the vector-valued (3-component) cell fold,
      collapsing each force into one broadcast scalar (verified:
      GPU output rows are `[s s s]`; maxerr 92.7 vs reference).
      The CPU path is correct.  This is a pre-existing GPU-lowering
      bug — `test_nbody_gpu_compiles` only checks compilation, never
      numeric parity.
- [x] Add `"nbody"` to `ALL_OPS` with sizes `(64, 256, 1024)`

Tests:
- [x] numpy/jax/remora-cpu produce pairwise-matching outputs within
      `rtol=1e-3` (remora-cpu maxerr 3.8e-6, jax maxerr 7.6e-6)
- [x] Benchmark runs without error at N=64, 256, 1024.
      remora-cpu beats numpy at every size (e.g. N=1024: 12.8ms vs
      39.2ms) — the compiled native all-pairs loop avoids numpy's
      large broadcast intermediates.

### 2.2 Device-resident GPU benchmarks

Measure pure kernel execution without host-device transfer.

Tasks:
- [x] Add `RemoraExecutor.execute_device(kernel_name, device_inputs,
      input_templates)` that accepts pre-allocated device pointers
      and returns a device pointer (no H→D or D→H copy)
- [x] Add `RemoraExecutor.alloc_and_upload(np_array) -> int` that
      allocates and copies once, returning the device pointer
- [x] Add `RemoraExecutor.download(device_ptr, shape, dtype) ->
      np.ndarray` for explicit D→H copy (+ `free_device`)
- [x] Add `--device-resident` flag to `remora-perf`
- [x] When set, GPU map/fold benchmarks pre-upload data and call
      `execute_device` in the timed loop
- [x] For JAX, data is already on device after warmup; no change
      needed

Tests:
- [x] Device-resident execution produces correct results
      (matches normal `execute` output exactly)
- [x] Map at 1M: device-resident remora-gpu 85.7us median vs 690us
      with transfer (target was <20us; the residual is descriptor
      build + launch + sync per call, not H↔D copy)
- [x] Report transfer overhead as the difference between
      device-resident and normal modes: map ~605us, fold ~210us
      at 1M

### 2.3 Fusion benchmarks

Measure whether the compiler fuses operation chains.

Tasks:
- [x] Add `bench_fusion_map_chain`: `map (*2) (map (+1) xs)`
      vs `map (\x -> (x+1)*2) xs` (manually fused)
- [x] Add `bench_fusion_dot`: `fold (+) 0 (map (*) xs ys)`
      (the 1D dot; matmul recognition is rank-2 only, so this is a
      single composed entry without a manual pair)
- [x] Add `bench_fusion_triple`: `map abs (map neg (map (* 2) xs))`
      (inline float-safe abs/neg to avoid the int-literal prelude
      definitions)
- [x] For each, benchmark both the composed and manually-fused
      versions (registered as `mapchain/triple-{composed,manual}`
      backends under op `fusion`)
- [x] Add `"fusion"` to `ALL_OPS` (+ `FUSION_SIZES`)

Tests:
- [x] Composed and manually-fused versions produce identical
      results (`np.array_equal` True for both map_chain and triple)
- [x] Report throughput ratio (composed / manual) as a fusion
      efficiency metric: **map_chain fuses perfectly** (ratio ~1.00
      at 100K and 1M).  **triple shows a 1.46x penalty at 100K**
      (composed 104us vs manual 72us) — the 3-deep map chain is not
      fully fused into one pass.  At 1M both converge (~1.0) as the
      workload becomes memory-bandwidth bound.

---

## Phase 3: GPU Performance

Improve GPU kernel implementations.  Estimated time: 3-5 days.

### 3.1 GPU sort — radix sort

Replace bitonic sort with LSB radix sort for f32 arrays.

**Target**: 10-50x improvement (68M → 700M-3G elem/s at 1M).

**STATUS: DONE.**  Implemented the **fast 256-bin (8-bit-digit, 4-pass)
radix sort** in `remora/_gpu_radix_sort.py`, wired into the GPU sort
dispatch (`codegen.py`), exposed as a 12-kernel `ExecutionPlan`.
Radix runs for 1024 < N ≤ 1024², bitonic remains the fallback below.

The original "deferred, highest-risk" assessment was over-cautious: an
empirical probe showed the warp intrinsics (`match.any.sync`,
`vote.ballot`, `shfl.sync`, `ctpop`) lower cleanly through this repo's
`mlir-translate-18 → llc-18 → ptxas` path (validated to a cubin), which
makes the *fast* design's hard step — the stable per-digit local-rank
scatter — tractable: `rank = popc(match.any.sync(digit) & lanemask.lt)`
per warp, aggregated across the 32 warps in shared memory.  Built
incrementally, each kernel validated against a NumPy oracle.

Pipeline: f32→uint32 key map (`bitcast` + sign flip) → per pass:
per-block digit-major histogram (shared atomics) → exclusive scan
(digit-major decomposition into single-block scans) → stable
warp-rank scatter → key→f32.

Tasks:
- [x] f32 key map + 4 hist + rowscan/digitscan/combine + 4 scatter
      MLIR kernels (one module, shared `.shared` globals)
- [x] f32 sign bit via the monotonic uint32 key mapping
- [x] 12-kernel `ExecutionPlan` (ping-pong key buffers) wired into
      `generate_mlir_descriptor_abi_ptx` sort dispatch
- [x] Fall back to bitonic sort for N ≤ 1024

Tests:
- [x] Correctness at N=2K..1M vs `np.sort` **exactly**, incl. heavy
      duplicates (stable), negatives, zeros, ±inf
      (`test_gpu_radix_sort_matches_numpy_when_available`; all 13
      existing GPU sort tests still pass)
- [x] Benchmark: **607M elem/s at 1M** through the official
      `remora-perf` path (with H↔D transfer + per-step syncs);
      ~1.35G device-resident.  ~9x the old bitonic (68M); above the
      500M target.


### 3.2 Reduce GPU launch overhead

Add device-resident array support to `RemoraExecutor`.

**Target**: 2-10x for iterative workflows.

Files to modify:
- `remora/executor.py`
- `remora/runtime.py`

Tasks:
- [x] Add `DeviceArray` class: holds device pointer, shape, dtype,
      nbytes; allocated from pool
- [x] Add `RemoraExecutor.execute_to_device(kernel_name,
      device_inputs) -> DeviceArray`
- [x] Add `DeviceArray.to_numpy() -> np.ndarray` (D→H copy)
- [x] Add `DeviceArray.from_numpy(executor, array) -> DeviceArray`
      (H→D copy) (+ `DeviceArray.free`)
- [x] Modify Mandelbrot example to use device-resident arrays —
      **deviation**: Mandelbrot's per-step escape masking
      (`np.where`, `counts[~escaped]`) is host-side data-dependent
      control flow, so it forces a round-trip every step and is a
      poor fit for pure device-residency.  Added a dedicated
      `examples/device_resident_iter.py` (a control-flow-free
      `z=z*z*a+c` recurrence) that demonstrates the pattern cleanly.
- [x] Verify: `uv run python -m compileall -q remora`

Tests:
- [x] Device-resident map at 1M: round-trip produces correct
      results (`tests/test_executor.py::
      test_device_array_round_trip_and_iteration_when_available`)
- [x] Device-resident iteration (100 steps of map): total time
      9.87ms vs 72.14ms with per-call transfer (7.31x) — meets the
      <10ms target
- [x] Existing `tests/test_executor.py` still passes

---

## Phase 4: CPU Performance

Improve CPU code generation quality.  Estimated time: 2-3 days.

### 4.1 CPU vectorization

Enable SIMD vectorization in the MLIR CPU pipeline.

**Target**: 2-8x for element-wise ops (536M → 1-4G elem/s for
map at 1M).

Files to modify:
- `remora/pipeline.py` (CPU pass pipeline)

Tasks:
- [ ] Research which MLIR passes enable vectorization for
      `linalg.generic` → LLVM:
      - `--linalg-generalize-named-ops`
      - `--linalg-fuse-elementwise-ops`
      - `--transform-interpreter` with vectorize schedule
      - `--convert-vector-to-llvm`
- [ ] Add vectorization passes to `CPU_PIPELINE_PASSES` in
      `pipeline.py`
- [ ] Test that existing programs still compile and produce
      correct results
- [ ] If full vectorization breaks some programs, gate it behind
      a `--cpu-vectorize` flag (already exists in benchmark CLI)
- [ ] Verify: `uv run python -m compileall -q remora`

Tests:
- [ ] All existing CPU tests pass:
      `uv run pytest tests/test_execution.py -q`
- [ ] Map at 1M: remora-cpu throughput > 1G elem/s
- [ ] Fold at 1M: remora-cpu throughput > 1G elem/s
- [ ] Scan at 100K: remora-cpu throughput > 1G elem/s
- [ ] Benchmark: `uv run remora-perf --ops map,fold,scan
      --backends remora-cpu,numpy --sizes 100000,1000000`

### 4.2 CPU fold — multi-threaded reduction

Parallelize fold across CPU cores using OpenMP.

**Target**: 4-16x (454M → 2-7G elem/s at 1M, depending on core
count).

Files to modify:
- `remora/pipeline.py`
- Possibly `remora/lowering/tensor_ops.py` (emit `scf.parallel`
  instead of `scf.for` for reductions)

Tasks:
- [ ] Research MLIR `--convert-scf-to-openmp` pass: does it handle
      `scf.for` with reduction semantics?
- [ ] If yes: add the pass to `CPU_PIPELINE_PASSES` and link with
      `-fopenmp` in the shared library build
- [ ] If no: modify fold lowering to emit `scf.parallel` with
      `scf.reduce` instead of `scf.for` with carry
- [ ] Gate behind `--cpu-threads N` flag (already exists in
      benchmark CLI)
- [ ] Verify: `uv run python -m compileall -q remora`

Tests:
- [ ] Fold at 1M produces correct result with 1, 2, 4, 8 threads
- [ ] Benchmark: `uv run remora-perf --ops fold --backends
      remora-cpu,numpy --sizes 1000000`
- [ ] Verify speedup scales with thread count

---

## Phase 5: Kernel Fusion

Fuse chains of element-wise operations into single kernels.
High effort, high architectural impact.  Estimated time: 5-10
days.

### 5.1 Source-level fusion (simpler)

Allow `remora.define()` to compile multi-output bodies.

Tasks:
- [ ] Design tuple return type for `remora.define()`:
      `f = remora.define("def step x = (x*2, x+1)")`
      returns `(array, array)` tuple
- [ ] Add `PairType` support to `CPUFunctionExecutor` return
      handling
- [ ] Add `PairType` support to GPU descriptor ABI (multiple
      output descriptors)
- [ ] Test: Mandelbrot `(step_real, step_imag, mag_sq)` as a
      single fused definition

Tests:
- [ ] Fused Mandelbrot produces correct results
- [ ] Fused version throughput > 2x unfused (3 separate kernels)
- [ ] Benchmark: add `bench_mandelbrot_fused` vs
      `bench_mandelbrot_unfused`

### 5.2 Lazy fusion (harder, more general)

Record computation graphs and fuse at execution time.

Tasks:
- [ ] Add `LazyArray` class that records operations instead of
      executing them
- [ ] Add `.numpy()` method that triggers compilation and
      execution of the recorded graph
- [ ] Implement graph → HIR conversion for chains of element-wise
      maps
- [ ] Fuse HIR before lowering: merge consecutive `HIRMap` nodes
      into a single map with a composed body
- [ ] Test with Mandelbrot iteration loop

Tests:
- [ ] Lazy evaluation produces identical results to eager
- [ ] 3-map chain is fused into 1 kernel (verify via
      `--emit-mlir` showing single `linalg.generic`)
- [ ] Throughput of fused chain > 2x unfused

---

## Phase 6: External Comparisons

Add Futhark as a comparison baseline.  Estimated time: 1-2 days
(assuming `futhark` binary is installed).

### 6.1 Futhark benchmarks

Tasks:
- [ ] Create `benchmarks/futhark/` directory
- [ ] Write `map_scale.fut`:
      `entry main (xs: []f32) : []f32 = map (*2.0) xs`
- [ ] Write `fold_sum.fut`:
      `entry main (xs: []f32) : f32 = reduce (+) 0.0 xs`
- [ ] Write `scan_prefix.fut`:
      `entry main (xs: []f32) : []f32 = scan (+) 0.0 xs`
- [ ] Write `matmul.fut` (using Futhark's built-in matmul or
      manual implementation)
- [ ] Write `sort.fut`:
      `entry main (xs: []f32) : []f32 = radix_sort_float xs`
- [ ] Write `stencil_blur.fut` (3×3 box blur via stencil map)
- [ ] Add `futhark` backend to `benchmark_suite.py`:
      - Compile with `futhark c --library` (CPU) or
        `futhark cuda --library` (GPU)
      - Call compiled library via `ctypes` or `subprocess` with
        `futhark-data` format
- [ ] Add `--backends futhark-cpu,futhark-gpu` options

Tests:
- [ ] All 6 Futhark programs compile and produce correct results
- [ ] Benchmark: `uv run remora-perf --ops map,fold,scan,sort
      --backends futhark-cpu,futhark-gpu,remora-cpu,remora-gpu`
- [ ] Results included in updated `REPORT.md`

---

## Verification Checklist

After each phase, run the full verification:

- [ ] `uv run python -m compileall -q remora`
- [ ] `uv run pytest tests/test_executor.py -q -x`
- [ ] `uv run pytest tests/test_parser.py tests/test_typechecker.py -q -x`
- [ ] `uv run remora-perf --warmup 5 --trials 20
      --json benchmarks/results/benchmark_report.json`
- [ ] Update `benchmarks/results/REPORT.md` with new numbers
- [ ] Update `CHANGELOG.md` with completed items

## Target Performance Goals

After all phases, target throughput at largest tested size:

| Operation | remora-cpu target | remora-gpu target |
|-----------|-------------------|-------------------|
| map 1M    | > 2G (vectorized) | > 2G              |
| fold 1M   | > 2G (threaded)   | > 5G              |
| scan 1M   | > 1G (vectorized) | > 1G (multi-block)|
| matmul 512| > 100M (BLAS)     | > 1G              |
| sort 1M   | > 50M (no copy)   | > 500M (radix)    |
| stencil 512| > 100M (vectorized)| > 2G             |
