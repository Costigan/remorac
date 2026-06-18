# GPU Descriptor ABI Parity — Codex Prompt

## Goal

Make the GPU descriptor-ABI path (`compile_function_source_to_mlir_gpu_ptx`
→ `CUDARuntime` → `RemoraExecutor`) compile and execute the same image filter
examples that already work on CPU (`examples/image_filters.py`).  Currently
the GPU path only handles simple homogeneous elementwise maps / folds / scans
/ appends.  Four gaps block the image filters.


## Ground Rules

- Work from `docs/DEEPSEEK_CONTINUATION_PLAN.md` — the Phase 4G section
  and the Phase 4 Status are the roadmap.
- Preserve existing GPU tests: `tests/test_executor.py` GPU round-trip tests
  (9 tests, currently all pass).
- `uv run pytest tests/test_executor.py -k "gpu_ptx_round_trip" -v`
- Keep the CPU path unchanged — no regressions in the 165 CPU tests.
- Fix gaps in order: Gap 1 first (smallest, immediately verifiable), then
  2 → 3 → 4.
- For each gap, add a GPU round-trip test in `tests/test_executor.py` that
  compiles the relevant filter to PTX and compares against NumPy.
- If a design choice blocks progress, document the exact HIR shape, error,
  and proposed path before proceeding.


## Current State

**What works on GPU** (10 tests pass):
- Rank-1/2/3 elementwise unary maps: `(map (+ 1.0) xs)`, `(map (* 2.0) xs)`.
- Rank-1/2/3 elementwise binary maps: `(map (+) xs ys)`.
- Fused f32 maps with scalar `Float` parameters, e.g. threshold:
  `(map (lambda (v) (select (> v t) 1.0 0.0)) image)`.
- Rank-1 fold/reduction: `(fold + 0.0 xs)`.
- Rank-1/2 scans and appends.

**What does NOT work** (the gaps below):

**Key files:**
- `remora/compiler.py` — `compile_function_source_to_mlir_gpu_ptx` (line 213,
  added `syntax` param).
- `remora/codegen.py` — `generate_mlir_descriptor_abi_ptx` (line 130+),
  tries f32→i32→bool→reduction→scan→append builders in sequence.
- `remora/_gpu_map_support.py` — `analyze_supported_f32_map_function` (line
  157), `_analyze_fused_f32_map` (line 191), `F32MapKernel`, `F32MapOperation`.
- `remora/gpu_lowering.py` — GPU module builders:
  `build_descriptor_abi_f32_map_gpu_module`,
  `build_descriptor_abi_f32_reduction_gpu_module`,
  `build_descriptor_abi_f32_scan_gpu_module`,
  `build_descriptor_abi_f32_append_gpu_module`.
- `remora/runtime.py` — `CUDARuntime`, `CUDAModule`, `CUDAKernel.launch`.
- `remora/executor.py` — `RemoraExecutor` (loads PTX, manages buffers, launches).
- `examples/image_filters.py` — CPU image filters (Sobel, threshold, blur);
  already has `--target gpu` CLI flag wired to `_compile_and_run_gpu`.
- `tests/test_executor.py` — GPU round-trip tests (lines 457–545).

**Verification commands:**
```bash
# Fast regression (no GPU needed)
uv run pytest tests/test_im2col.py tests/test_ad.py tests/test_crater_detect_data.py tests/test_phase7_dependent_functions.py tests/test_crater_detect_train.py tests/test_image_filters.py tests/test_pde_stencil.py -q

# GPU tests
uv run pytest tests/test_executor.py -k "gpu_ptx_round_trip" -v

# Manual GPU filter test
uv run python -c "
from examples.image_filters import run_sobel, run_threshold, run_blur
run_threshold(32, target='gpu')   # should work after Gap 1
run_blur(32, target='gpu')        # should work after Gaps 2-4
"
```


## Four Gaps (in implementation order)

### Gap 1 — Scalar parameters in GPU fused-map pattern — DONE

**What was blocked:** `(map (lambda (v) (select (> v t) 1.0 0.0)) image)`
— elementwise threshold with scalar `t`.  Works on CPU and now works on GPU.

**Original error:** `CodegenUnavailable` (swallowed; the f32 map matcher raises first,
then i32, then bool whose error message surfaces).

**Root cause:** `_analyze_fused_f32_map` (`_gpu_map_support.py:196-214`)
checks that ALL input params match the return type shape (line 201-202):
```python
if any(input_type.shape != function.return_type.shape
       for input_type in input_types):
    raise on_unsupported(...)
```
A scalar `t` has shape `()` ≠ `(32,32)` → rejected.

The same check is in `analyze_supported_map_function` (the non-fused path).

**Implemented fix:**
1. In `_analyze_fused_f32_map`, separate array params from scalar `Float`
   params.  Keep the shape check only for array params.
2. Preserve parameter order explicitly.  Today `F32MapKernel.num_inputs` means
   "number of memref descriptor inputs" throughout lowering and execution.
   Add explicit metadata for descriptor inputs vs scalar kernel arguments,
   rather than treating the scalar as another descriptor input.
3. Extend the fused expression IR so scalar params lower to the scalar kernel
   argument, while array params still lower to `%xN` loads from descriptors at
   the current output index.
4. In the descriptor-ABI GPU builder, generate a kernel signature like
   `input_desc..., scalar..., output_desc` and emit scalar params as `f32`
   arguments (not memref descriptors).
5. In `RemoraExecutor.execute_main` / `execute`, pack array inputs as device
   buffers + descriptors and pass scalar inputs as kernel args in generated
   ABI order.  `_pack_cuda_kernel_args` (`runtime.py:419`) already handles
   Python/NumPy floating scalar values.
6. Add `KernelMeta` fields such as `input_kinds` or `scalar_count` so runtime
   validation, dtype handling, and launch packing do not infer this from
   `num_inputs`.

**Verification:** after the fix, this should compile and run correctly:
```python
from remora.compiler import compile_function_source_to_mlir_gpu_ptx
from remora.executor import RemoraExecutor
from remora.runtime import CUDARuntime
from remora.types import ArrayType, FLOAT, StaticDim
import numpy as np

src = '(define/pi () (f [x (Array Float 4) t Float] (Array Float 4)) (map (lambda (v) (select (> v t) 1.0 0.0)) x))'
pt = (ArrayType(FLOAT,(StaticDim(4),)), FLOAT)
runtime = CUDARuntime()
ptx, kernels, _ = compile_function_source_to_mlir_gpu_ptx(src,'f',pt,include_prelude=False,syntax='lisp')
exe = RemoraExecutor(ptx, kernels, runtime=runtime)
r = exe.execute_main([np.array([1,2,3,4],dtype=np.float32), np.float32(2.5)])
print(r)  # should be [0., 0., 1., 1.]
runtime.close()
```

**Verification:** `tests/test_executor.py -k "gpu_ptx_round_trip" -v` now has
10 passing round-trip tests, including scalar threshold.


### Gap 2 — GPU im2col kernel builder

**What's blocked:** Any filter that uses `(im2col image [k k] stride)` —
Sobel, blur.  The CPU path lowers im2col to `memref.alloc` + `scf.for` loops
extracting patches; the GPU path has no im2col builder.

**Root cause:** `generate_mlir_descriptor_abi_ptx` tries builders in sequence
(f32 map → i32 map → bool map → reduction → scan → append).  None handles
im2col.

**Fix:**
1. Add a pattern recognizer: in `generate_mlir_descriptor_abi_ptx` (or a new
   helper in `codegen.py`), detect when the top-level HIR is an im2col
   (HIRIm2col).  The im2col takes an image [H,W] + kernel_shape + stride and
   produces [patches, patch_size].
2. Add `build_descriptor_abi_im2col_gpu_module(function, kernel_name)` in
   `gpu_lowering.py`.  Generate a descriptor-ABI `llvm.func` kernel in a
   `gpu.module`, matching the existing direct CUDA launch path:
   - Takes input descriptor pointer (image), output descriptor pointer
     (patches buffer).
   - One GPU thread per output element: `thread_id` maps to
     `(patch_idx, elem_idx)`.
   - Each thread computes `(patch_row, patch_col) = divmod(patch_idx,
     patches_per_axis)`, `(kernel_row, kernel_col) = divmod(elem_idx,
     kernel_size)`, then `image[patch_row*stride + kernel_row,
     patch_col*stride + kernel_col]`.
   - Writes to `output[patch_idx, elem_idx]`.
3. Add a `KernelMeta` entry for im2col (num_inputs=1, num_outputs=1,
   output_shape=(patches, patch_size)).
4. Wire it into the builder chain in `generate_mlir_descriptor_abi_ptx`
   BEFORE the f32 map attempt (since im2col is more specific).

**HIR shape to recognize:**
```
HIRIm2col(
  image=HIRVar("image"),
  kernel_shape=(3, 3),
  stride=1,
  result_type=ArrayType(float, (36, 9))
)
```

**Verification:** `im2col` alone should compile and produce the same result
as the CPU `im2col` for a given image:
```python
src = '(define/pi () (f [image (Array Float 8 8)] (Array Float 36 9)) (im2col image [3 3] 1))'
pt = (ArrayType(FLOAT,(StaticDim(8),StaticDim(8))),)
# compile GPU, execute, compare to numpy im2col
```

**Expected scope:** ~80–150 lines (new GPU kernel builder + pattern match +
kernel meta + test).


### Gap 3 — GPU cell-fold (convolution) kernel builder

**What's blocked:** The blur filter `(map (lambda (p) (fold + 0.0 (map * p
(ravel kb)))) (im2col image [3 3] 1))` — a per-patch dot-product reduction.
The GPU fold builder handles scalar-output `(fold + 0.0 xs)` but not a
map-of-folds or row-wise array reduction.  Rewriting uniform blur as
`fold + 0.0 (transpose (im2col …))` still requires an array-output reduction
over one axis, so it is not unlocked by the existing scalar fold builder.

**Approach (recommended):**
1. **Minimum useful implementation:** add an array-output row-reduction /
   cell-fold GPU builder.  For uniform blur it can compute one output element
   per patch by summing the im2col row and applying the scalar scale.
2. **General cell-fold (Sobel with custom kernels):** extend that builder to:
   - Recognizes `HIRMap(cell_shape != (), func=HIRLambda(HIRFold), …)`.
   - Generates a GPU kernel: each thread computes one output element by
     iterating over the cell dimension of the im2col output, accumulating
     the dot product with the kernel (passed as a memref descriptor).

   If this is too large, document the exact HIR shape and defer after Gap 2,
   but do not mark blur unblocked until an array-output row reduction or
   specialized combined blur kernel exists.

**Verification:** blur on GPU should match NumPy:
```python
from examples.image_filters import run_blur
run_blur(32, target='gpu')
```

**Expected scope:** ~100–250 lines depending on whether a row-sum-only kernel
or the general dot-product cell-fold builder is implemented.


### Gap 4 — Multi-operation GPU pipeline

**What's blocked:** Real functions compose multiple ops.  The GPU matcher
expects the ENTIRE function to be one pattern (map/fold/scan/append).  For
example, the full blur is `map (* 0.111) (fold + 0.0 (transpose (im2col
…)))` — a scale *after* a fold *after* an im2col.  This is three GPU
operations.

**Fix:**
1. In `generate_mlir_descriptor_abi_ptx`, after the single-pattern matchers
   fail, add a *decomposition* pass:
   - Walk the HIR expression tree.
   - Identify a sequence of supported GPU operations (im2col → transpose →
     fold → unary map).
   - Generate one GPU kernel per operation.
   - Connect them via intermediate `memref.alloc` buffers.
   - Return a list of `(gpu_module_text, KernelMeta)` pairs.
2. In `RemoraExecutor`, extend `execute_main` to handle multi-kernel
   execution: allocate intermediate buffers, launch kernels in sequence,
   copy only the final result back.
3. **Transpose as a view:** `transpose` is a zero-cost metadata operation.
   For GPU, two approaches:
   a. Pass a transposed memref descriptor (strides swapped) — no kernel needed.
   b. Implement a transpose kernel if descriptors aren't flexible enough.
   Approach (a) is preferred.
4. **Elementwise scale:** `(map (* 0.111) xs)` is already handled by the
   existing unary f32 map builder (Gap 1 extension for scalar constants).

**If decomposition is too large for one pass:** document the exact HIR for
blur and implement a specialized combined kernel: im2col + fold + scale in
one GPU kernel.  This is more code but avoids the multi-kernel orchestration.

**Verification:** blur end-to-end on GPU:
```python
from examples.image_filters import run_blur
run_blur(32, target='gpu')
```

**Expected scope:** 200–500 lines (decomposition pass + multi-kernel executor
extension + tests).


## Suggested Work Order

1. **Gap 1** — scalar params.  DONE; threshold now compiles to PTX and runs on
   GPU.
2. **Gap 2** — GPU im2col.  Unlocks the im2col primitive needed by all
   convolution-like filters.  ~80–150 lines.
3. **Gap 3** — cell-fold / row reduction.  Existing fold support is scalar
   output only, so blur still needs an array-output row-reduction or combined
   blur kernel.  ~100–250 lines.
4. **Gap 4** — multi-op pipeline.  ~200–500 lines.

After each gap, verify:
- No regression: `uv run pytest tests/test_executor.py -k gpu -q`
- Add a GPU round-trip test for the newly unblocked filter.
- Update `docs/DEEPSEEK_CONTINUATION_PLAN.md` Phase 4 Status.
