# MLIR Toolchain

This project uses a CPU-first standalone LLVM/MLIR toolchain for Phase 6
pipeline validation. IREE remains an inspection backend only.

## Current Local Toolchain

- Standalone LLVM/MLIR: Ubuntu LLVM 18.1.3
- `mlir-opt`: `/usr/bin/mlir-opt-18`
- `mlir-translate`: `/usr/bin/mlir-translate-18`
- `llc`: `/usr/bin/llc-18`
- OpenMP runtime: `libomp-18-dev` is installed in the current environment;
  threaded CPU execution with `--cpu-threads > 1` can link against the
  LLVM/MLIR 18-compatible `__kmpc` runtime
- `ptxas`: not installed in the current environment; PTX assembly validation
  is skipped when unavailable
- IREE inspection tools: `.venv/bin/iree-opt`, `.venv/bin/iree-compile`
- IREE package: `iree-compiler==20241104.1068`

The IREE tools report LLVM 20.0.0git and therefore must not be treated as the
same production toolchain as the standalone LLVM/MLIR 18 tools.

## Validation Commands

```bash
env UV_CACHE_DIR=/tmp/uv-cache UV_LINK_MODE=copy uv run python tools/validate_mlir_toolchain.py
env UV_CACHE_DIR=/tmp/uv-cache UV_LINK_MODE=copy uv run python tools/validate_mlir_pipeline.py
```

The pipeline validator checks:

- `docs/mlir-pipeline-cpu.txt` matches `remora.pipeline.CPU_PIPELINE`.
- `docs/mlir-pipeline-fusion.txt` matches `remora.pipeline.FUSION_PIPELINE`.
- Dense Core cases lower through standalone `mlir-opt-18` to LLVM dialect.
- The lowered LLVM dialect translates to LLVM IR with `mlir-translate-18`.
- Fusion reduces nested map programs at the `linalg.generic` level.

The experimental threaded CPU pipeline is exposed as
`remora.pipeline.CPU_THREADED_PIPELINE`. It lowers map-shaped `linalg` programs
through `scf.parallel` and OpenMP dialect operations before LLVM lowering. Link
and execution require libomp; libgomp does not satisfy the current MLIR
OpenMP-to-LLVM symbols.

## GPU Backend Status

The final NVIDIA backend requires Remora to lower to explicit `gpu.module` /
`gpu.func` kernels using the descriptor ABI in `docs/ABI.md`, then translate
device code through NVVM and assemble/check PTX.

The current executable MLIR-derived slice covers rank-1 through rank-3
`float32` and `int32` unary/binary scalar-cell maps, plus rank-1 `float32`
scalar reductions and dot-shaped `fold (+) init (map (*) xs ys)` kernels. It
emits a descriptor-pointer ABI `gpu.module` kernel directly, translates the
device body through NVVM/LLVM IR, emits PTX with `llc`, and launches through
`RemoraExecutor` when CUDA is available.
`ptxas` assembly checks are available through tests when `ptxas` is installed.

`remora.codegen.generate_ptx` remains only an IREE HAL PTX inspection path. Its
launch ABI is not the Remora descriptor ABI.
