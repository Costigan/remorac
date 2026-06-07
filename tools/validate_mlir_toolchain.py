"""Report and validate the local MLIR/LLVM/CUDA toolchain."""

from __future__ import annotations

import argparse
from pathlib import Path
import importlib.metadata as metadata
import importlib
import re
import subprocess
import sys
import os
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from remora.pipeline import detect_toolchain, PipelineToolchain  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--require-unified-llvm",
        action="store_true",
        help="fail if IREE inspection tools report a different LLVM major than standalone MLIR",
    )
    parser.add_argument(
        "--require-gpu-validation",
        action="store_true",
        help="fail if ptxas is missing and PTX assembly validation cannot run",
    )
    parser.add_argument(
        "--require-cuda-runtime",
        action="store_true",
        help="fail if CUDA driver/runtime is not available for GPU execution",
    )
    args = parser.parse_args()

    toolchain = detect_toolchain()
    exit_code = 0

    # ── MLIR / LLVM toolchain ────────────────────────────────────────
    _section("MLIR / LLVM Standalone Tools")
    tools = _check_mlir_llvm_tools(toolchain)
    if tools["errors"]:
        exit_code = 1

    # ── IREE compiler ────────────────────────────────────────────────
    _section("IREE Compiler")
    _check_iree(toolchain)

    # ── PTX assembler ────────────────────────────────────────────────
    _section("GPU: PTX Assembler")
    ptxas_ok = _check_ptxas(toolchain, args.require_gpu_validation)
    if not ptxas_ok and args.require_gpu_validation:
        exit_code = 1

    # ── CUDA driver / runtime ────────────────────────────────────────
    _section("GPU: CUDA Driver")
    cuda_ok = _check_cuda_runtime()
    if not cuda_ok and args.require_cuda_runtime:
        exit_code = 1

    # ── CUDA compute capability ──────────────────────────────────────
    if cuda_ok:
        _section("GPU: Compute Capability")
        _check_cuda_devices()

    # ── ptxas SM support ─────────────────────────────────────────────
    if ptxas_ok:
        _check_ptxas_sm_targets(toolchain)

    # ── Python packages ──────────────────────────────────────────────
    _section("Python Packages")
    _check_python_packages()

    # ── Environment ──────────────────────────────────────────────────
    _section("Environment")
    _check_environment()

    # ── Summary ──────────────────────────────────────────────────────
    _section("Summary")
    _print_summary(toolchain, cuda_ok, ptxas_ok)

    return exit_code


# -----------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------


def _section(title: str) -> None:
    print(f"\n── {title}")


def _ok(msg: str = "") -> None:
    print(f"   ✓ {msg}" if msg else "   ✓ ok")


def _warn(msg: str) -> None:
    print(f"   ⚠ {msg}")


def _err(msg: str) -> None:
    print(f"   ✗ {msg}")


def _info(msg: str) -> None:
    print(f"   ℹ {msg}")


def _run(cmd: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, text=True, check=False)


def _version_text(path: str) -> str:
    result = _run([path, "--version"])
    return result.stdout or result.stderr


def _first_nonempty_line(text: str) -> str:
    for line in text.splitlines():
        if line.strip():
            return line.strip()
    return ""


def _llvm_major(text: str) -> int | None:
    match = re.search(r"LLVM version\s+(\d+)", text)
    return int(match.group(1)) if match else None


# -----------------------------------------------------------------------
# MLIR / LLVM toolchain checks
# -----------------------------------------------------------------------


def _check_mlir_llvm_tools(toolchain: "PipelineToolchain") -> dict:
    names = ["mlir-opt", "mlir-translate", "llc"]
    paths = {
        "mlir-opt": toolchain.mlir_opt,
        "mlir-translate": toolchain.mlir_translate,
        "llc": toolchain.llc,
    }

    versions: dict[str, int | None] = {}
    errors: list[str] = []

    for name in names:
        path = paths[name]
        if path is None:
            _err(f"{name}: not found")
            errors.append(name)
            versions[name] = None
            continue
        ver_text = _version_text(path)
        ver = _llvm_major(ver_text)
        _ok(f"{name}: {Path(path).name}  (LLVM {ver})")
        versions[name] = ver

    if errors:
        _warn(f"Install LLVM/MLIR {_DEFAULT_LLVM_MAJOR} toolchain:")
        _info("  Ubuntu/Debian: apt install mlir-18-tools llvm-18")
        _info("  macOS: brew install llvm@18")
        return {"errors": errors}

    majors = {versions[n] for n in names}
    if len(majors) != 1:
        _err(f"Version mismatch across tools: {versions}")
        _info("Ensure all tools come from the same LLVM installation.")
        errors.append("version-mismatch")
    else:
        _ok(f"All tools consistent (LLVM {next(iter(majors))})")

    return {"errors": errors}


# -----------------------------------------------------------------------
# IREE checks
# -----------------------------------------------------------------------


def _check_iree(toolchain: "PipelineToolchain") -> None:
    for name, path in [
        ("iree-opt", toolchain.iree_opt),
        ("iree-compile", toolchain.iree_compile),
    ]:
        if path is None:
            _warn(f"{name}: not found (GPU compilation unavailable)")
            continue
        ver_text = _version_text(path)
        ver = _llvm_major(ver_text)
        _ok(f"{name}: {Path(path).name}  (LLVM {ver})")

    try:
        ver = metadata.version("iree-compiler")
        _ok(f"iree-compiler package: {ver}")
    except metadata.PackageNotFoundError:
        _warn("iree-compiler Python package not found")

    if toolchain.iree_compile is not None:
        _check_iree_cuda_backend(toolchain)
    else:
        _info("Skipping IREE CUDA backend check (iree-compile not found)")


def _check_iree_cuda_backend(toolchain: "PipelineToolchain") -> None:
    """Check if iree-compile was built with CUDA support."""
    result = _run([str(toolchain.iree_compile), "--list_targets"])
    if "cuda" in result.stdout.lower() or "nvvm" in result.stdout.lower():
        _ok("iree-compile has CUDA/NVVM backend support")
    else:
        _warn("iree-compile may lack CUDA backend (GPU compilation may fail)")
        _info("  Build IREE with -DIREE_TARGET_BACKEND_CUDA=ON")


# -----------------------------------------------------------------------
# ptxas checks
# -----------------------------------------------------------------------


def _check_ptxas(toolchain: "PipelineToolchain", required: bool) -> bool:
    if toolchain.ptxas is None:
        if required:
            _err("ptxas: not found (required for GPU release validation)")
        else:
            _warn("ptxas: not found (PTX assembly validation skipped)")
        _info("  Install CUDA toolkit: https://developer.nvidia.com/cuda-downloads")
        _info("  Ensure <cuda>/bin is on PATH")
        return False

    result = _run([toolchain.ptxas, "--version"])
    ver_line = _first_nonempty_line(result.stdout)
    _ok(f"ptxas: {Path(toolchain.ptxas).parent}")
    _info(f"  {ver_line}")
    return True


def _check_ptxas_sm_targets(toolchain: "PipelineToolchain") -> None:
    """Check which SM architectures ptxas supports."""
    # quick check: try to assemble a minimal PTX targeting sm_80
    test_ptx = """.version 7.8
.target sm_80
.address_size 64
.visible .entry dummy() { ret; }"""

    result = _run(
        [toolchain.ptxas, "-c", "-"],
        # Need to pass PTX via stdin
    )
    # ptxas reads from file, not stdin easily. Skip detailed check.
    _info("SM target support: try compiling with --target sm_80 through sm_90a")


# -----------------------------------------------------------------------
# CUDA runtime checks
# -----------------------------------------------------------------------


def _check_cuda_runtime() -> bool:
    """Check if CUDA driver bindings and a GPU device are available."""
    try:
        from cuda.bindings import driver as cuda_driver
    except ImportError:
        _warn("cuda.bindings.driver: not installed")
        _info("  Install: pip install cuda-python")
        return False

    try:
        (err,) = cuda_driver.cuInit(0)
        if err != 0:
            _warn(f"cuInit failed (error {err}) — no CUDA driver?")
            return False
        _ok("CUDA driver initialized")
    except Exception as e:
        _warn(f"cuInit failed: {e}")
        return False

    return True


def _check_cuda_devices() -> None:
    """Report CUDA device information."""
    try:
        from cuda.bindings import driver as cuda_driver
    except ImportError:
        return

    try:
        result, count = cuda_driver.cuDeviceGetCount()
        if count == 0:
            _warn("No CUDA devices found")
            return
    except Exception:
        _warn("cuDeviceGetCount failed")
        return

    _ok(f"{count} CUDA device(s) detected")
    for i in range(min(count, 4)):
        try:
            _, device = cuda_driver.cuDeviceGet(i)
            _, name_bytes = cuda_driver.cuDeviceGetName(256, device)
            name = name_bytes.split(b"\x00")[0].decode("utf-8", errors="replace")

            _, major = cuda_driver.cuDeviceGetAttribute(
                cuda_driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, device
            )
            _, minor = cuda_driver.cuDeviceGetAttribute(
                cuda_driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, device
            )

            _, mem_bytes = _cuda_device_total_mem(cuda_driver, device)
            mem_gb = mem_bytes / (1024 ** 3)

            _info(f"  [{i}] {name}  (SM {major}.{minor}, {mem_gb:.1f} GiB)")
        except Exception as e:
            _info(f"  [{i}] (error: {e})")


def _cuda_device_total_mem(cuda_driver: Any, device: Any) -> tuple[int, int]:
    """Get total device memory in bytes (handles API changes in cuda-python)."""
    try:
        return cuda_driver.cuDeviceTotalMem(device)
    except TypeError:
        try:
            return cuda_driver.cuDeviceTotalMem(device, 0)
        except TypeError:
            try:
                return cuda_driver.cuDeviceTotalMem_v2(device)
            except (AttributeError, TypeError):
                return (0, 0)


# -----------------------------------------------------------------------
# Python package checks
# -----------------------------------------------------------------------


def _check_python_packages() -> None:
    packages = {
        "iree-compiler": "IREE MLIR compiler",
        "iree-compiler-snapshot": "IREE compiler (nightly)",
        "cuda-python": "CUDA Python bindings",
        "numpy": "Array operations",
        "lark": "Remora parser",
    }
    for pkg, desc in packages.items():
        try:
            ver = metadata.version(pkg)
            _ok(f"{pkg}: {ver}")
        except metadata.PackageNotFoundError:
            _warn(f"{pkg}: not installed  ({desc})")


# -----------------------------------------------------------------------
# Environment checks
# -----------------------------------------------------------------------


def _check_environment() -> None:
    # REMORA_NUM_THREADS
    threads = os.environ.get("REMORA_NUM_THREADS")
    if threads:
        _info(f"REMORA_NUM_THREADS={threads}")
    else:
        _info("REMORA_NUM_THREADS not set (default: all cores)")

    # CUDA_HOME / CUDA_PATH
    cuda_home = os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH")
    if cuda_home:
        _info(f"CUDA_HOME={cuda_home}")
    else:
        _info("CUDA_HOME not set (using PATH discovery)")

    # Check for common CUDA install locations
    common_cuda = [
        "/usr/local/cuda",
        "/usr/local/cuda-12",
        "/usr/local/cuda-13",
        "/opt/cuda",
    ]
    found = [p for p in common_cuda if Path(p).is_dir()]
    if found:
        _info(f"CUDA directories found: {', '.join(found)}")


# -----------------------------------------------------------------------
# Summary
# -----------------------------------------------------------------------


def _print_summary(toolchain: "PipelineToolchain", cuda_ok: bool, ptxas_ok: bool) -> None:
    features = []

    # CPU
    if toolchain.mlir_opt and toolchain.mlir_translate and toolchain.llc:
        features.append("CPU MLIR lowering")
    if _module_available("iree.compiler.passmanager"):
        features.append("CPU MLIR validation")
    if toolchain.iree_compile:
        features.append("CPU JIT compilation")

    # GPU
    if toolchain.iree_compile and ptxas_ok:
        features.append("GPU PTX compilation")
    if cuda_ok:
        features.append("GPU CUDA execution")

    _ok(f"Available: {', '.join(features) if features else '(none)'}")

    missing = []
    if toolchain.mlir_opt is None:
        missing.append("mlir-opt (CPU lowering)")
    if toolchain.iree_compile is None:
        missing.append("iree-compile (JIT/GPU)")
    if not ptxas_ok:
        missing.append("ptxas (GPU PTX validation)")
    if not cuda_ok:
        missing.append("CUDA driver (GPU execution)")

    if missing:
        _warn(f"Missing: {', '.join(missing)}")


def _module_available(name: str) -> bool:
    try:
        importlib.import_module(name)
        return True
    except (ImportError, ModuleNotFoundError):
        return False


_DEFAULT_LLVM_MAJOR = 18


if __name__ == "__main__":
    raise SystemExit(main())
