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
import tempfile
from typing import Any

from remora.errors import RemoraError


VALIDATION_PIPELINE = "builtin.module(canonicalize,cse)"
FUSION_PIPELINE = "builtin.module(linalg-fuse-elementwise-ops,canonicalize,cse)"

CPU_PIPELINE = "builtin.module(" + ",".join(
    [
        "linalg-fuse-elementwise-ops",
        "one-shot-bufferize{bufferize-function-boundaries allow-return-allocs-from-loops}",
        "func.func(buffer-hoisting,buffer-loop-hoisting,buffer-deallocation)",
        "convert-linalg-to-loops",
        "convert-scf-to-cf",
        "expand-strided-metadata",
        "lower-affine",
        "finalize-memref-to-llvm",
        "convert-arith-to-llvm",
        "convert-index-to-llvm",
        "convert-to-llvm",
        "reconcile-unrealized-casts",
    ]
) + ")"

CPU_VECTORIZED_PIPELINE = "builtin.module(" + ",".join(
    [
        "linalg-fuse-elementwise-ops",
        "one-shot-bufferize{bufferize-function-boundaries allow-return-allocs-from-loops}",
        "func.func(buffer-hoisting,buffer-loop-hoisting,buffer-deallocation)",
        "convert-linalg-to-affine-loops",
        "func.func(affine-super-vectorize{virtual-vector-size=4 vectorize-reductions})",
        "lower-affine",
        "convert-scf-to-cf",
        "expand-strided-metadata",
        "finalize-memref-to-llvm",
        "convert-arith-to-llvm",
        "convert-index-to-llvm",
        "convert-vector-to-llvm",
        "convert-to-llvm",
        "reconcile-unrealized-casts",
    ]
) + ")"

CPU_THREADED_VECTORIZED_PIPELINE = "builtin.module(" + ",".join(
    [
        "linalg-fuse-elementwise-ops",
        "one-shot-bufferize{bufferize-function-boundaries allow-return-allocs-from-loops}",
        "func.func(buffer-hoisting,buffer-loop-hoisting,buffer-deallocation)",
        "convert-linalg-to-affine-loops",
        "func.func(affine-parallelize,affine-super-vectorize{virtual-vector-size=4 vectorize-reductions})",
        "lower-affine",
        "convert-scf-to-openmp",
        "convert-scf-to-cf",
        "expand-strided-metadata",
        "finalize-memref-to-llvm",
        "convert-arith-to-llvm",
        "convert-index-to-llvm",
        "convert-openmp-to-llvm",
        "convert-vector-to-llvm",
        "convert-to-llvm",
        "reconcile-unrealized-casts",
    ]
) + ")"

CPU_THREADED_PIPELINE = "builtin.module(" + ",".join(
    [
        "linalg-fuse-elementwise-ops",
        "one-shot-bufferize{bufferize-function-boundaries allow-return-allocs-from-loops}",
        "func.func(buffer-hoisting,buffer-loop-hoisting,buffer-deallocation)",
        "convert-linalg-to-parallel-loops",
        "convert-scf-to-openmp",
        "convert-scf-to-cf",
        "expand-strided-metadata",
        "lower-affine",
        "finalize-memref-to-llvm",
        "convert-arith-to-llvm",
        "convert-index-to-llvm",
        "convert-openmp-to-llvm",
        "convert-to-llvm",
        "reconcile-unrealized-casts",
    ]
) + ")"

CPU_THREADED_PRE_PIPELINE = "builtin.module(" + ",".join(
    [
        "linalg-fuse-elementwise-ops",
        "one-shot-bufferize{bufferize-function-boundaries allow-return-allocs-from-loops}",
        "func.func(buffer-hoisting,buffer-loop-hoisting,buffer-deallocation)",
        "convert-linalg-to-parallel-loops",
        "convert-scf-to-openmp",
        "expand-strided-metadata",
        "lower-affine",
    ]
) + ")"


CPU_THREADED_POST_PIPELINE = "builtin.module(" + ",".join(
    [
        "convert-scf-to-cf",
        "finalize-memref-to-llvm",
        "convert-arith-to-llvm",
        "convert-index-to-llvm",
        "convert-openmp-to-llvm",
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

    @property
    def has_nvptx_codegen(self) -> bool:
        return self.mlir_translate is not None and self.llc is not None


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


def build_cpu_threaded_pipeline() -> Any:
    return build_pipeline(CPU_THREADED_PIPELINE)


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
    threaded: bool = False,
    vectorize: bool = False,
) -> str:
    if threaded and vectorize:
        raise PipelineUnavailable("threaded CPU vectorization is not supported yet")
    if vectorize:
        return run_external_pipeline_text(mlir_text, CPU_VECTORIZED_PIPELINE, toolchain=toolchain)
    if not threaded:
        return run_external_pipeline_text(mlir_text, CPU_PIPELINE, toolchain=toolchain)
    lowered = run_external_pipeline_text(
        mlir_text,
        CPU_THREADED_PRE_PIPELINE,
        toolchain=toolchain,
    )
    lowered = _strip_trivial_memref_alloca_scopes(lowered)
    return run_external_pipeline_text(
        lowered,
        CPU_THREADED_POST_PIPELINE,
        toolchain=toolchain,
    )


def _strip_trivial_memref_alloca_scopes(mlir_text: str) -> str:
    """Remove no-allocation `memref.alloca_scope` wrappers from MLIR text.

    MLIR 18's linalg-to-parallel-loops path emits alloca scopes around loop
    bodies even when they contain no stack allocations. Lowering nested
    ``scf.for`` loops to CFG inside those wrappers can make the scope region
    invalid. This function strips only wrappers whose body has no
    ``memref.alloca`` operations.

    This is a text-based workaround for an MLIR 18 pipeline artifact. A
    future MLIR version may remove the need for this by emitting tighter
    alloca scopes.
    """
    lines = mlir_text.splitlines()
    output: list[str] = []
    index = 0
    while index < len(lines):
        line = lines[index]
        if "memref.alloca_scope" not in line or "{" not in line:
            output.append(line)
            index += 1
            continue

        depth = _brace_delta(line)
        end = index + 1
        while end < len(lines) and depth > 0:
            depth += _brace_delta(lines[end])
            end += 1
        if depth != 0:
            output.append(line)
            index += 1
            continue

        body = lines[index + 1 : end - 1]
        has_alloca = any("memref.alloca" in body_line for body_line in body)
        if has_alloca:
            output.extend(lines[index:end])
        else:
            output.extend(body)
        index = end
    return "\n".join(output) + ("\n" if mlir_text.endswith("\n") else "")


def _brace_delta(line: str) -> int:
    return line.count("{") - line.count("}")


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


def translate_llvmir_to_nvptx_text(
    llvm_ir: str,
    *,
    sm_version: str = "sm_80",
    toolchain: PipelineToolchain | None = None,
) -> str:
    toolchain = detect_toolchain() if toolchain is None else toolchain
    if toolchain.llc is None:
        raise PipelineUnavailable("llc is required for NVPTX text generation")

    with tempfile.TemporaryDirectory(prefix="remora-gpu-") as temp_dir:
        llvm_ir_path = Path(temp_dir) / "module.ll"
        llvm_ir_path.write_text(llvm_ir, encoding="utf-8")
        result = subprocess.run(
            [
                toolchain.llc,
                "-march=nvptx64",
                f"-mcpu={sm_version}",
                "-filetype=asm",
                str(llvm_ir_path),
                "-o",
                "-",
            ],
            capture_output=True,
            text=True,
            check=False,
        )
    if result.returncode != 0:
        stderr = result.stderr.strip()
        raise PipelineUnavailable(
            f"NVPTX text generation failed with {Path(toolchain.llc).name}: {stderr}"
        )
    return result.stdout


def assemble_ptx_text(
    ptx_text: str,
    *,
    sm_version: str = "sm_80",
    toolchain: PipelineToolchain | None = None,
) -> bytes:
    """Assemble PTX to a cubin with ptxas and return the generated bytes."""
    toolchain = detect_toolchain() if toolchain is None else toolchain
    if toolchain.ptxas is None:
        raise PipelineUnavailable("ptxas is required for PTX assembly validation")

    with tempfile.TemporaryDirectory(prefix="remora-ptxas-") as temp_dir:
        root = Path(temp_dir)
        ptx_path = root / "kernel.ptx"
        cubin_path = root / "kernel.cubin"
        ptx_path.write_text(ptx_text, encoding="utf-8")
        result = subprocess.run(
            [
                toolchain.ptxas,
                f"--gpu-name={sm_version}",
                str(ptx_path),
                "-o",
                str(cubin_path),
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode != 0:
            stderr = result.stderr.strip()
            raise PipelineUnavailable(
                f"PTX assembly failed with {Path(toolchain.ptxas).name}: {stderr}"
            )
        return cubin_path.read_bytes()


def lower_gpu_scaffold_to_nvptx_text(
    mlir_text: str,
    *,
    sm_version: str = "sm_80",
    toolchain: PipelineToolchain | None = None,
) -> str:
    """Lower a scaffold-only gpu.module artifact to inspection PTX text.

    This path is intentionally inspection-only: the resulting PTX uses MLIR's
    exploded memref ABI at the kernel boundary, not the external descriptor ABI
    required by `docs/ABI.md` and `RemoraExecutor`.
    """
    from remora.gpu_lowering import extract_gpu_module_body_as_module

    toolchain = detect_toolchain() if toolchain is None else toolchain
    lowered = run_gpu_nvidia_scaffold_llvm_dialect_pipeline_text(
        mlir_text,
        toolchain=toolchain,
    )
    device_module = extract_gpu_module_body_as_module(lowered)
    llvm_ir = translate_mlir_to_llvmir(device_module, toolchain=toolchain)
    return translate_llvmir_to_nvptx_text(
        llvm_ir,
        sm_version=sm_version,
        toolchain=toolchain,
    )


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

        if name == "ptxas":
            for base in ("/usr/local/cuda/bin", "/usr/local/cuda-13/bin", "/usr/local/cuda-13.2/bin"):
                cuda_path = Path(base) / name
                if cuda_path.is_file():
                    return str(cuda_path)
    return None
