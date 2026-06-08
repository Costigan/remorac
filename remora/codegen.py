"""PTX code generation helpers for Remora Dense Core.

The current Phase 6 path uses the installed IREE compiler as the practical
starter backend. It emits PTX for CUDA HAL dispatch kernels, not yet final
Remora ABI kernels intended for direct manual CUDA launches.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
import subprocess
import tempfile
from typing import Any

from remora._gpu_map_support import (
    analyze_supported_bool_map_function,
    analyze_supported_f32_map_function,
    analyze_supported_i32_map_function,
    F32MapKernel,
    F32MapOperation,
    I32MapKernel,
    I32MapOperation,
)
from remora.errors import RemoraError
from remora.hir import HIRFunction
from remora.pipeline import (
    PipelineToolchain,
    detect_toolchain,
    translate_llvmir_to_nvptx_text,
    translate_mlir_to_llvmir,
)
from remora.gpu_lowering import (
    GPUScaffoldError,
    build_descriptor_abi_bool_map_gpu_module,
    build_descriptor_abi_f32_append_gpu_module,
    build_descriptor_abi_f32_map_gpu_module,
    build_descriptor_abi_f32_reduction_gpu_module,
    build_descriptor_abi_f32_scan_gpu_module,
    build_descriptor_abi_i32_map_gpu_module,
    extract_gpu_module_body_as_module,
)


class CodegenUnavailable(RemoraError):
    """Raised when PTX generation cannot run with the installed toolchain."""


@dataclass(frozen=True)
class KernelMeta:
    name: str
    grid_dims: int
    block_size: int
    num_inputs: int
    num_outputs: int
    input_elem_types: list[str]
    output_elem_types: list[str]
    output_shape: tuple[int, ...] | None = None
    output_dtype: str | None = None
    is_reduction: bool = False


def generate_ptx(
    module: Any,
    *,
    sm_version: str = "sm_80",
    ptx_features: str = "+ptx75",
    toolchain: PipelineToolchain | None = None,
) -> tuple[str, list[KernelMeta]]:
    """Compile a lowered MLIR module to PTX text.

    This uses `iree-compile` and asks it to dump executable files. The returned
    PTX is suitable for syntax checks and Phase 6 pipeline validation. It is
    not yet the stable external Remora kernel ABI described in `docs/ABI.md`.
    """

    toolchain = detect_toolchain() if toolchain is None else toolchain
    if toolchain.iree_compile is None:
        raise CodegenUnavailable("iree-compile is required for PTX generation")

    module_text = str(module)
    with tempfile.TemporaryDirectory() as temp_dir:
        command = [
            toolchain.iree_compile,
            "--iree-hal-target-backends=cuda",
            f"--iree-cuda-target={sm_version}",
            "--iree-hal-dump-executable-files-to",
            temp_dir,
            "--output-format=vm-asm",
            "-",
        ]
        result = subprocess.run(
            command,
            input=module_text,
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode != 0:
            stderr = result.stderr.strip()
            raise CodegenUnavailable(f"iree-compile failed: {stderr}")

        ptx_files = sorted(Path(temp_dir).glob("*.ptx"))
        if not ptx_files:
            raise CodegenUnavailable("iree-compile did not emit any PTX files")

        ptx_parts = [path.read_text(encoding="utf-8") for path in ptx_files]

    ptx_text = "\n".join(ptx_parts)
    return ptx_text, _extract_kernel_metadata(ptx_text)




def generate_mlir_descriptor_abi_ptx(
    function: HIRFunction,
    *,
    kernel_name: str | None = None,
    toolchain: PipelineToolchain | None = None,
) -> tuple[str, list[KernelMeta]]:
    """Generate the first MLIR-derived descriptor-ABI PTX execution slice.

    This is intentionally narrow: rank-1 through rank-3 unary/binary
    `float32` maps. The generated GPU kernel accepts Remora descriptor pointers
    directly, so the exported entry can be launched by `RemoraExecutor`.
    """
    toolchain = detect_toolchain() if toolchain is None else toolchain
    name = kernel_name or f"remora_{function.name}"
    try:
        map_kernel = _direct_f32_map_kernel(function)
        rank = len(map_kernel.shape)
        if rank < 1 or rank > 10:
            raise CodegenUnavailable(
                "MLIR-derived descriptor-ABI PTX currently supports rank-1 through rank-10 f32 maps only"
            )
        if map_kernel.num_inputs == 1 and map_kernel.operation.constant is None:
            raise CodegenUnavailable(
                "MLIR-derived descriptor-ABI PTX currently supports unary literal-section or binary f32 maps only"
            )
        if map_kernel.num_inputs not in (1, 2):
            raise CodegenUnavailable(
                "MLIR-derived descriptor-ABI PTX currently supports one or two f32 input descriptors only"
            )
        gpu_module = build_descriptor_abi_f32_map_gpu_module(function, kernel_name=name)
        meta = KernelMeta(
            name=name,
            grid_dims=1,
            block_size=0,
            num_inputs=map_kernel.num_inputs,
            num_outputs=1,
            input_elem_types=["f32"],
            output_elem_types=["f32"],
            output_shape=map_kernel.shape,
            output_dtype="float32",
        )
    except CodegenUnavailable as f32_map_error:
        try:
            map_kernel = _direct_i32_map_kernel(function)
            rank = len(map_kernel.shape)
            if rank < 1 or rank > 10:
                raise CodegenUnavailable(
                    "MLIR-derived descriptor-ABI PTX currently supports rank-1 through rank-10 i32 maps only"
                )
            if map_kernel.num_inputs == 1 and map_kernel.operation.constant is None:
                raise CodegenUnavailable(
                    "MLIR-derived descriptor-ABI PTX currently supports unary literal-section or binary i32 maps only"
                )
            if map_kernel.num_inputs not in (1, 2):
                raise CodegenUnavailable(
                    "MLIR-derived descriptor-ABI PTX currently supports one or two i32 input descriptors only"
                )
            gpu_module = build_descriptor_abi_i32_map_gpu_module(function, kernel_name=name)
            meta = KernelMeta(
                name=name,
                grid_dims=1,
                block_size=0,
                num_inputs=map_kernel.num_inputs,
                num_outputs=1,
                input_elem_types=["i32"] * map_kernel.num_inputs,
                output_elem_types=["i32"],
                output_shape=map_kernel.shape,
                output_dtype="int32",
            )
        except CodegenUnavailable as i32_map_error:
            try:
                map_kernel = analyze_supported_bool_map_function(
                    function,
                    on_unsupported=CodegenUnavailable,
                    context="MLIR-derived descriptor-ABI PTX",
                )
                gpu_module = build_descriptor_abi_bool_map_gpu_module(function, kernel_name=name)
                meta = KernelMeta(
                    name=name,
                    grid_dims=1,
                    block_size=0,
                    num_inputs=map_kernel.num_inputs,
                    num_outputs=1,
                    input_elem_types=["i8"] * map_kernel.num_inputs,
                    output_elem_types=["i8"],
                    output_shape=map_kernel.shape,
                    output_dtype="bool",
                )
            except CodegenUnavailable as bool_map_error:
                try:
                    gpu_module = build_descriptor_abi_f32_reduction_gpu_module(function, kernel_name=name)
                    num_inputs = len(function.params)
                    meta = KernelMeta(
                        name=name,
                        grid_dims=1,
                        block_size=0,
                        num_inputs=num_inputs,
                        num_outputs=1,
                        input_elem_types=["f32"] * num_inputs,
                        output_elem_types=["f32"],
                        output_shape=(),
                        output_dtype="float32",
                        is_reduction=True,
                    )
                except GPUScaffoldError as reduction_error:
                    try:
                        gpu_module = build_descriptor_abi_f32_scan_gpu_module(function, kernel_name=name)
                        meta = KernelMeta(
                            name=name,
                            grid_dims=1,
                            block_size=1,  # single-thread scan
                            num_inputs=1,
                            num_outputs=1,
                            input_elem_types=["f32"],
                            output_elem_types=["f32"],
                            output_shape=(),
                            output_dtype="float32",
                        )
                    except GPUScaffoldError as scan_error:
                        try:
                            gpu_module = build_descriptor_abi_f32_append_gpu_module(function, kernel_name=name)
                            meta = KernelMeta(
                                name=name,
                                grid_dims=1,
                                block_size=0,
                                num_inputs=2,
                                num_outputs=1,
                                input_elem_types=["f32", "f32"],
                                output_elem_types=["f32"],
                                output_dtype="float32",
                            )
                        except GPUScaffoldError as append_error:
                            raise CodegenUnavailable(str(bool_map_error)) from append_error

    device_module = extract_gpu_module_body_as_module(gpu_module.text)
    llvm_ir = translate_mlir_to_llvmir(device_module, toolchain=toolchain)
    ptx = translate_llvmir_to_nvptx_text(llvm_ir, toolchain=toolchain)
    return ptx, [meta]


def generate_rank1_f32_unary_mlir_descriptor_abi_ptx(
    function: HIRFunction,
    *,
    kernel_name: str | None = None,
    toolchain: PipelineToolchain | None = None,
) -> tuple[str, list[KernelMeta]]:
    """Backward-compatible wrapper for the first MLIR-derived executable slice."""
    return generate_mlir_descriptor_abi_ptx(
        function,
        kernel_name=kernel_name,
        toolchain=toolchain,
    )


def _extract_kernel_metadata(ptx_text: str) -> list[KernelMeta]:
    """Parse PTX text to extract kernel entry names, block sizes, and param counts."""
    metas: list[KernelMeta] = []
    entry_matches = list(re.finditer(r"\.visible\s+\.entry\s+([A-Za-z_.$][\w.$]*)\s*\(", ptx_text))
    for index, match in enumerate(entry_matches):
        start = match.start()
        end = entry_matches[index + 1].start() if index + 1 < len(entry_matches) else len(ptx_text)
        body = ptx_text[start:end]
        metas.append(
            KernelMeta(
                name=match.group(1),
                grid_dims=1,
                block_size=_extract_block_size(body),
                num_inputs=_count_ptx_params(body),
                num_outputs=0,
                input_elem_types=[],
                output_elem_types=[],
            )
        )
    return metas


def _extract_block_size(ptx_entry_text: str) -> int:
    """Extract the .maxntid block size from a PTX entry body, returning 0 if absent."""
    match = re.search(r"\.maxntid\s+(\d+)", ptx_entry_text)
    if match is None:
        return 0
    return int(match.group(1))


def _count_ptx_params(ptx_entry_text: str) -> int:
    """Count .param declarations in a PTX entry body."""
    return len(re.findall(r"\.param\s+\.\w+\s+[A-Za-z_.$][\w.$]*", ptx_entry_text))


def _direct_f32_map_kernel(function: HIRFunction) -> F32MapKernel:
    """Analyze an HIRFunction into a supported F32MapKernel or raise CodegenUnavailable."""
    try:
        return analyze_supported_f32_map_function(
            function,
            on_unsupported=CodegenUnavailable,
            context="direct PTX",
        )
    except CodegenUnavailable as exc:
        message = str(exc).replace("float", "f32").replace(
            "one or two input parameters",
            "one or two input descriptors",
        ).replace(
            "literal float section",
            "literal f32 section constant",
        )
        raise CodegenUnavailable(message) from exc


def _direct_i32_map_kernel(function: HIRFunction) -> I32MapKernel:
    """Analyze an HIRFunction into a supported I32MapKernel or raise CodegenUnavailable."""
    try:
        return analyze_supported_i32_map_function(
            function,
            on_unsupported=CodegenUnavailable,
            context="direct MLIR descriptor PTX",
        )
    except CodegenUnavailable as exc:
        message = str(exc).replace("int", "i32").replace(
            "one or two input parameters",
            "one or two input descriptors",
        ).replace(
            "literal i32 section",
            "literal i32 section constant",
        )
        raise CodegenUnavailable(message) from exc

