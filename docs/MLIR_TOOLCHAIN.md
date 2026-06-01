# MLIR Toolchain

This project uses a CPU-first standalone LLVM/MLIR toolchain for Phase 6
pipeline validation. IREE remains an inspection backend only.

## Current Local Toolchain

- Standalone LLVM/MLIR: Ubuntu LLVM 18.1.3
- `mlir-opt`: `/usr/bin/mlir-opt-18`
- `mlir-translate`: `/usr/bin/mlir-translate-18`
- `llc`: `/usr/bin/llc-18`
- `ptxas`: not installed in the current environment
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

## Deferred GPU Backend

The final NVIDIA backend requires Remora to lower to explicit `gpu.module` /
`gpu.func` kernels using the descriptor ABI in `docs/ABI.md`, then translate
device code through NVVM and assemble/check PTX. That path is not complete yet.

Until then, `remora.codegen.generate_ptx` is only an IREE HAL PTX inspection
path. Its launch ABI is not the Remora descriptor ABI.
