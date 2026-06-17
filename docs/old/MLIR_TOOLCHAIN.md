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
- `ptxas`: `/usr/local/cuda/bin/ptxas`, CUDA compilation tools 13.2; PTX
  assembly validation is available
- IREE inspection tools: `.venv/bin/iree-opt`, `.venv/bin/iree-compile`
- IREE package: `iree-compiler==20241104.1068`

The IREE tools report LLVM 20.0.0git and therefore must not be treated as the
same production toolchain as the standalone LLVM/MLIR 18 tools.

## Toolchain Policy

The local toolchain has two intentionally separate roles:

- **Standalone LLVM/MLIR** (`mlir-opt`, `mlir-translate`, `llc`) is the source
  of truth for CPU pipeline validation and standalone PTX text generation.
- **IREE** is an inspection/smoke backend only. IREE HAL PTX is not a Remora
  runtime artifact and must not be counted as descriptor-ABI GPU completion.

LLVM major mismatch policy:

- Development validation allows IREE inspection tools to report a different
  LLVM major than standalone MLIR, but the validator prints
  `warning(dev-ok)`.
- Release or pinned-toolchain validation should use:

```bash
env UV_CACHE_DIR=/tmp/uv-cache UV_LINK_MODE=copy uv run python tools/validate_mlir_toolchain.py --require-unified-llvm
```

GPU validation policy:

- Missing `ptxas` is a skip for CPU/Dense Core development validation.
- Missing `ptxas` is a blocker for GPU release validation because generated PTX
  has not been assembled.
- GPU validation should use:

```bash
env UV_CACHE_DIR=/tmp/uv-cache UV_LINK_MODE=copy uv run python tools/validate_mlir_toolchain.py --require-gpu-validation
```

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
and row-reduction-shaped programs through `scf.parallel` and OpenMP dialect
operations before LLVM lowering. Runtime execution uses a split pipeline around
the OpenMP conversion so trivial no-allocation `memref.alloca_scope` wrappers do
not block nested loop lowering. Link and execution require libomp; libgomp does
not satisfy the current MLIR OpenMP-to-LLVM symbols.

The experimental vectorized CPU pipeline is exposed as
`remora.pipeline.CPU_VECTORIZED_PIPELINE` and selected by `--cpu-vectorize`. It
uses affine loop lowering plus `affine-super-vectorize` before lowering vector,
affine, SCF, and standard operations to LLVM. This mode is currently
single-threaded; combining `--cpu-vectorize` with `--cpu-threads > 1` is rejected
until the threaded/vectorized pipeline contract is deliberately designed.

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

## Installing `ptxas`

`ptxas` is part of the CUDA Toolkit developer tools. Runtime-only CUDA packages
are not sufficient.

Native Ubuntu/Debian with NVIDIA's CUDA apt repository:

```bash
sudo apt update
sudo apt install cuda-toolkit
```

Pop!_OS/Ubuntu without NVIDIA's CUDA apt repository may expose Ubuntu's
`nvidia-cuda-toolkit` package instead:

```bash
sudo apt update
sudo apt install nvidia-cuda-toolkit
```

To pin a toolkit release instead of tracking the latest toolkit metapackage,
install a versioned package, for example:

```bash
sudo apt install cuda-toolkit-13-2
```

WSL 2:

- Install the NVIDIA Windows driver on Windows.
- In the WSL Ubuntu instance, use NVIDIA's WSL-Ubuntu CUDA repository.
- Prefer `cuda-toolkit-X-Y`; avoid packages such as `cuda`, `cuda-13`, or
  `cuda-drivers` inside WSL because they may try to install a Linux display
  driver.

For Ubuntu 24.04 WSL, the current NVIDIA/Ubuntu example is:

```bash
wget https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update
sudo apt-get -y install cuda-toolkit-13-2
```

Conda environment:

```bash
conda install cuda -c nvidia
```

or a smaller developer-tools subset:

```bash
conda install cuda-nvcc -c nvidia
```

After installation, ensure the CUDA `bin` directory is on `PATH`, for example:

```bash
export PATH=/usr/local/cuda/bin:$PATH
which ptxas
ptxas --version
```

Then rerun:

```bash
env UV_CACHE_DIR=/tmp/uv-cache UV_LINK_MODE=copy uv run python tools/validate_mlir_toolchain.py --require-gpu-validation
env UV_CACHE_DIR=/tmp/uv-cache UV_LINK_MODE=copy uv run pytest tests/test_gpu_lowering.py tests/test_executor.py tests/test_cuda_runtime.py -q
```
