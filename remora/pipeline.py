"""MLIR pass pipeline helpers for Remora Dense Core.

Phase 6 starts with toolchain detection and parse-validated pass-manager
plumbing. Full CPU/GPU lowering pipelines are intentionally gated on the exact
MLIR toolchain being available and pinned.
"""

from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
from pathlib import Path
from shutil import which
import subprocess
import sys
from typing import Any

from remora.errors import RemoraError


VALIDATION_PIPELINE = "builtin.module(canonicalize,cse)"
FUSION_PIPELINE = "builtin.module(linalg-fuse-elementwise-ops,canonicalize,cse)"

CPU_PIPELINE = "builtin.module(" + ",".join(
    [
        "linalg-fuse-elementwise-ops",
        "one-shot-bufferize{bufferize-function-boundaries allow-return-allocs-from-loops}",
        "convert-linalg-to-loops",
        "convert-scf-to-cf",
        "convert-to-llvm",
        "reconcile-unrealized-casts",
    ]
) + ")"

GPU_NVIDIA_PIPELINE = "builtin.module(" + ",".join(
    [
        "linalg-fuse-elementwise-ops",
        "linalg-generalize-named-ops",
        "one-shot-bufferize{bufferize-function-boundaries allow-return-allocs-from-loops}",
        "buffer-deallocation-pipeline",
        "affine-loop-fusion",
        "affine-parallelize",
        "lower-affine",
        "convert-linalg-to-parallel-loops",
        "gpu-map-parallel-loops",
        "convert-parallel-loops-to-gpu",
        "gpu-kernel-outlining",
        "lower-affine",
        "convert-scf-to-cf",
        "convert-gpu-to-nvvm{index-bitwidth=64}",
        "convert-arith-to-llvm",
        "convert-math-to-llvm",
        "convert-func-to-llvm",
        "gpu-to-llvm",
        "reconcile-unrealized-casts",
    ]
) + ")"

GPU_NVIDIA_SCAFFOLD_NVVM_PIPELINE = (
    "builtin.module(gpu.module(convert-gpu-to-nvvm{index-bitwidth=64}))"
)

GPU_NVIDIA_SCAFFOLD_LLVM_DIALECT_PIPELINE = (
    "builtin.module("
    "gpu.module(convert-gpu-to-nvvm{index-bitwidth=64},convert-scf-to-cf),"
    "convert-cf-to-llvm,"
    "reconcile-unrealized-casts"
    ")"
)


class PipelineUnavailable(RemoraError):
    """Raised when the installed MLIR toolchain cannot build a requested pipeline."""


@dataclass(frozen=True)
class PipelineToolchain:
    mlir_opt: str | None
    mlir_translate: str | None
    iree_opt: str | None
    iree_compile: str | None
    llc: str | None
    ptxas: str | None
    iree_passmanager: bool

    @property
    def has_external_verifier(self) -> bool:
        return self.mlir_opt is not None or self.iree_opt is not None

    @property
    def has_standalone_mlir(self) -> bool:
        return self.mlir_opt is not None and self.mlir_translate is not None

    @property
    def external_verifier(self) -> str | None:
        return self.mlir_opt or self.iree_opt

    @property
    def has_ptx_toolchain(self) -> bool:
        return self.llc is not None and self.ptxas is not None


def detect_toolchain() -> PipelineToolchain:
    return PipelineToolchain(
        mlir_opt=_find_executable("mlir-opt", "mlir-opt-18"),
        mlir_translate=_find_executable("mlir-translate", "mlir-translate-18"),
        iree_opt=_find_executable("iree-opt"),
        iree_compile=_find_executable("iree-compile"),
        llc=_find_executable("llc", "llc-18"),
        ptxas=_find_executable("ptxas"),
        iree_passmanager=_module_available("iree.compiler.passmanager"),
    )


def build_validation_pipeline() -> Any:
    return build_pipeline(VALIDATION_PIPELINE)


def build_fusion_pipeline() -> Any:
    return build_pipeline(FUSION_PIPELINE)


def build_cpu_pipeline() -> Any:
    return build_pipeline(CPU_PIPELINE)


def build_gpu_nvidia_pipeline() -> Any:
    return build_pipeline(GPU_NVIDIA_PIPELINE)


def run_gpu_nvidia_scaffold_nvvm_pipeline_text(
    mlir_text: str,
    *,
    toolchain: PipelineToolchain | None = None,
) -> str:
    return run_external_pipeline_text(
        mlir_text,
        GPU_NVIDIA_SCAFFOLD_NVVM_PIPELINE,
        toolchain=toolchain,
    )


def run_gpu_nvidia_scaffold_llvm_dialect_pipeline_text(
    mlir_text: str,
    *,
    toolchain: PipelineToolchain | None = None,
) -> str:
    return run_external_pipeline_text(
        mlir_text,
        GPU_NVIDIA_SCAFFOLD_LLVM_DIALECT_PIPELINE,
        toolchain=toolchain,
    )


def build_pipeline(pipeline_text: str) -> Any:
    passmanager = _load_iree_passmanager()
    try:
        return passmanager.PassManager.parse(pipeline_text)
    except Exception as exc:
        raise PipelineUnavailable(
            f"MLIR pass pipeline is not available in this toolchain: {exc}"
        ) from exc


def run_pipeline(module: Any, pass_manager: Any, *, debug: bool = False) -> None:
    if debug:
        pass_manager.enable_ir_printing()
    pass_manager.enable_verifier(True)
    operation = getattr(module, "operation", module)
    pass_manager.run(operation)


def run_validation_pipeline(module: Any, *, debug: bool = False) -> None:
    with module.context:
        pass_manager = build_validation_pipeline()
        run_pipeline(module, pass_manager, debug=debug)


def run_external_pipeline_text(
    mlir_text: str,
    pipeline_text: str,
    *,
    toolchain: PipelineToolchain | None = None,
) -> str:
    toolchain = detect_toolchain() if toolchain is None else toolchain
    if toolchain.mlir_opt is None:
        raise PipelineUnavailable("mlir-opt is required for standalone MLIR pipeline validation")

    result = subprocess.run(
        [toolchain.mlir_opt, f"--pass-pipeline={pipeline_text}", "-"],
        input=mlir_text,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        stderr = result.stderr.strip()
        raise PipelineUnavailable(
            f"standalone MLIR pipeline failed with {Path(toolchain.mlir_opt).name}: {stderr}"
        )
    return result.stdout


def run_cpu_pipeline_text(
    mlir_text: str,
    *,
    toolchain: PipelineToolchain | None = None,
) -> str:
    return run_external_pipeline_text(mlir_text, CPU_PIPELINE, toolchain=toolchain)


def run_fusion_pipeline_text(
    mlir_text: str,
    *,
    toolchain: PipelineToolchain | None = None,
) -> str:
    return run_external_pipeline_text(mlir_text, FUSION_PIPELINE, toolchain=toolchain)


def translate_mlir_to_llvmir(
    mlir_text: str,
    *,
    toolchain: PipelineToolchain | None = None,
) -> str:
    toolchain = detect_toolchain() if toolchain is None else toolchain
    if toolchain.mlir_translate is None:
        raise PipelineUnavailable("mlir-translate is required for LLVM IR translation")

    result = subprocess.run(
        [toolchain.mlir_translate, "--mlir-to-llvmir"],
        input=mlir_text,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        stderr = result.stderr.strip()
        raise PipelineUnavailable(
            f"LLVM IR translation failed with {Path(toolchain.mlir_translate).name}: {stderr}"
        )
    return result.stdout


def verify_module_text(mlir_text: str, toolchain: PipelineToolchain | None = None) -> None:
    toolchain = detect_toolchain() if toolchain is None else toolchain
    verifier = toolchain.external_verifier
    if verifier is None:
        raise PipelineUnavailable("no external MLIR verifier is available")

    args = [verifier, "--verify-diagnostics", "-"]
    result = subprocess.run(
        args,
        input=mlir_text,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        stderr = result.stderr.strip()
        raise PipelineUnavailable(
            f"external MLIR verifier failed with {Path(verifier).name}: {stderr}"
        )


def _module_available(module_name: str) -> bool:
    try:
        import_module(module_name)
    except ModuleNotFoundError:
        return False
    return True


def _load_iree_passmanager() -> Any:
    try:
        return import_module("iree.compiler.passmanager")
    except ModuleNotFoundError as exc:
        raise PipelineUnavailable(
            "IREE compiler pass manager bindings are required for MLIR pipelines"
        ) from exc


def _find_executable(*names: str) -> str | None:
    for name in names:
        path = which(name)
        if path is not None:
            return path

        sibling = Path(sys.executable).parent / name
        if sibling.is_file():
            return str(sibling)
    return None
