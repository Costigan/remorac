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
from remora.operators import ptx_op
from remora.pipeline import (
    PipelineToolchain,
    detect_toolchain,
    translate_llvmir_to_nvptx_text,
    translate_mlir_to_llvmir,
)
from remora.gpu_lowering import (
    GPUScaffoldError,
    build_descriptor_abi_bool_map_gpu_module,
    build_descriptor_abi_f32_map_gpu_module,
    build_descriptor_abi_f32_reduction_gpu_module,
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


def generate_direct_remora_ptx(
    function: HIRFunction,
    *,
    kernel_name: str | None = None,
    block_size: int = 128,
) -> tuple[str, list[KernelMeta]]:
    """Generate direct Remora ABI PTX for the current narrow GPU slice.

    Supported today: rank-1 through rank-3 `float32` unary maps with a literal
    float section constant, and binary maps over two matching `float32` inputs.
    Unlike the scaffold-only NVPTX inspection path, this emitter produces the
    descriptor-pointer kernel ABI required by `docs/ABI.md` and
    `RemoraExecutor`.
    """
    name = kernel_name or f"remora_{function.name}"
    map_kernel = _direct_f32_map_kernel(function)
    ptx = _f32_map_ptx(name, map_kernel, block_size)
    return ptx, [
        KernelMeta(
            name=name,
            grid_dims=1,
            block_size=block_size,
            num_inputs=map_kernel.num_inputs,
            num_outputs=1,
            input_elem_types=["f32"],
            output_elem_types=["f32"],
            output_shape=map_kernel.shape,
            output_dtype="float32",
        )
    ]


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
                    raise CodegenUnavailable(str(bool_map_error)) from reduction_error

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
    match = re.search(r"\.maxntid\s+(\d+)", ptx_entry_text)
    if match is None:
        return 0
    return int(match.group(1))


def _count_ptx_params(ptx_entry_text: str) -> int:
    return len(re.findall(r"\.param\s+\.\w+\s+[A-Za-z_.$][\w.$]*", ptx_entry_text))


def _direct_f32_map_kernel(function: HIRFunction) -> F32MapKernel:
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


def _f32_map_ptx(
    kernel_name: str,
    kernel: F32MapKernel,
    block_size: int,
) -> str:
    if kernel.num_inputs == 2:
        return _binary_f32_map_ptx(kernel_name, kernel, block_size)
    op_line = _unary_f32_ptx_op(kernel.operation)
    index_lines = _unary_f32_map_index_lines(kernel.shape)
    constant = 0.0 if kernel.operation.constant is None else kernel.operation.constant
    return f""".version 6.0
.target sm_50
.address_size 64

.visible .entry {kernel_name}(
    .param .u64 input_desc_param,
    .param .u64 output_desc_param
)
.maxntid {block_size}, 1, 1
{{
    .reg .pred %p;
    .reg .b32 %r<5>;
    .reg .b64 %rd<32>;
    .reg .f32 %f<4>;

    ld.param.u64 %rd1, [input_desc_param];
    ld.param.u64 %rd2, [output_desc_param];
    mov.u32 %r1, %tid.x;
    mov.u32 %r2, %ctaid.x;
    mov.u32 %r3, %ntid.x;
    mad.lo.s32 %r4, %r2, %r3, %r1;
    cvt.s64.s32 %rd3, %r4;

{index_lines}
    setp.ge.s64 %p, %rd3, %rd4;
    @%p bra DONE;

    ld.u64 %rd5, [%rd1+8];
    ld.u64 %rd6, [%rd1+16];
    add.s64 %rd8, %rd6, %rd20;
    mul.lo.s64 %rd9, %rd8, 4;
    add.s64 %rd10, %rd5, %rd9;
    ld.global.f32 %f1, [%rd10];
    mov.f32 %f2, {constant:.8e};
    {op_line}

    ld.u64 %rd11, [%rd2+8];
    ld.u64 %rd12, [%rd2+16];
    add.s64 %rd14, %rd12, %rd21;
    mul.lo.s64 %rd15, %rd14, 4;
    add.s64 %rd16, %rd11, %rd15;
    st.global.f32 [%rd16], %f3;

DONE:
    ret;
}}
"""


def _unary_f32_map_index_lines(shape: tuple[int, ...]) -> str:
    if len(shape) == 1:
        return """    ld.u64 %rd4, [%rd1+24];
    ld.u64 %rd7, [%rd1+32];
    ld.u64 %rd13, [%rd2+32];
    mul.lo.s64 %rd20, %rd3, %rd7;
    mul.lo.s64 %rd21, %rd3, %rd13;"""
    if len(shape) == 2:
        return """    ld.u64 %rd22, [%rd1+24];
    ld.u64 %rd23, [%rd1+32];
    mul.lo.s64 %rd4, %rd22, %rd23;
    div.u64 %rd24, %rd3, %rd23;
    rem.u64 %rd25, %rd3, %rd23;
    ld.u64 %rd7, [%rd1+40];
    ld.u64 %rd26, [%rd1+48];
    mul.lo.s64 %rd20, %rd24, %rd7;
    mad.lo.s64 %rd20, %rd25, %rd26, %rd20;
    ld.u64 %rd13, [%rd2+40];
    ld.u64 %rd27, [%rd2+48];
    mul.lo.s64 %rd21, %rd24, %rd13;
    mad.lo.s64 %rd21, %rd25, %rd27, %rd21;"""
    if len(shape) == 3:
        return """    ld.u64 %rd22, [%rd1+24];
    ld.u64 %rd23, [%rd1+32];
    ld.u64 %rd24, [%rd1+40];
    mul.lo.s64 %rd25, %rd23, %rd24;
    mul.lo.s64 %rd4, %rd22, %rd25;
    div.u64 %rd26, %rd3, %rd25;
    rem.u64 %rd27, %rd3, %rd25;
    div.u64 %rd28, %rd27, %rd24;
    rem.u64 %rd29, %rd27, %rd24;
    ld.u64 %rd7, [%rd1+48];
    ld.u64 %rd30, [%rd1+56];
    ld.u64 %rd31, [%rd1+64];
    mul.lo.s64 %rd20, %rd26, %rd7;
    mad.lo.s64 %rd20, %rd28, %rd30, %rd20;
    mad.lo.s64 %rd20, %rd29, %rd31, %rd20;
    ld.u64 %rd13, [%rd2+48];
    ld.u64 %rd30, [%rd2+56];
    ld.u64 %rd31, [%rd2+64];
    mul.lo.s64 %rd21, %rd26, %rd13;
    mad.lo.s64 %rd21, %rd28, %rd30, %rd21;
    mad.lo.s64 %rd21, %rd29, %rd31, %rd21;"""
    raise CodegenUnavailable("direct PTX currently supports rank-1 through rank-3 maps only")


def _binary_f32_map_ptx(
    kernel_name: str,
    kernel: F32MapKernel,
    block_size: int,
) -> str:

    index_lines = _binary_f32_map_index_lines(kernel.shape)
    op_line = _binary_f32_ptx_op(kernel.operation)
    return f""".version 6.0
.target sm_50
.address_size 64

.visible .entry {kernel_name}(
    .param .u64 input0_desc_param,
    .param .u64 input1_desc_param,
    .param .u64 output_desc_param
)
.maxntid {block_size}, 1, 1
{{
    .reg .pred %p;
    .reg .b32 %r<5>;
    .reg .b64 %rd<50>;
    .reg .f32 %f<4>;

    ld.param.u64 %rd1, [input0_desc_param];
    ld.param.u64 %rd2, [input1_desc_param];
    ld.param.u64 %rd17, [output_desc_param];
    mov.u32 %r1, %tid.x;
    mov.u32 %r2, %ctaid.x;
    mov.u32 %r3, %ntid.x;
    mad.lo.s32 %r4, %r2, %r3, %r1;
    cvt.s64.s32 %rd3, %r4;

{index_lines}
    setp.ge.s64 %p, %rd3, %rd4;
    @%p bra DONE;

    ld.u64 %rd5, [%rd1+8];
    ld.u64 %rd6, [%rd1+16];
    add.s64 %rd8, %rd6, %rd20;
    mul.lo.s64 %rd9, %rd8, 4;
    add.s64 %rd10, %rd5, %rd9;
    ld.global.f32 %f1, [%rd10];

    ld.u64 %rd11, [%rd2+8];
    ld.u64 %rd12, [%rd2+16];
    add.s64 %rd14, %rd12, %rd21;
    mul.lo.s64 %rd15, %rd14, 4;
    add.s64 %rd16, %rd11, %rd15;
    ld.global.f32 %f2, [%rd16];
    {op_line}

    ld.u64 %rd30, [%rd17+8];
    ld.u64 %rd31, [%rd17+16];
    add.s64 %rd32, %rd31, %rd22;
    mul.lo.s64 %rd33, %rd32, 4;
    add.s64 %rd34, %rd30, %rd33;
    st.global.f32 [%rd34], %f3;

DONE:
    ret;
}}
"""


def _binary_f32_map_index_lines(shape: tuple[int, ...]) -> str:
    if len(shape) == 1:
        return """    ld.u64 %rd4, [%rd1+24];
    ld.u64 %rd7, [%rd1+32];
    ld.u64 %rd13, [%rd2+32];
    ld.u64 %rd35, [%rd17+32];
    mul.lo.s64 %rd20, %rd3, %rd7;
    mul.lo.s64 %rd21, %rd3, %rd13;
    mul.lo.s64 %rd22, %rd3, %rd35;"""
    if len(shape) == 2:
        return """    ld.u64 %rd23, [%rd1+24];
    ld.u64 %rd24, [%rd1+32];
    mul.lo.s64 %rd4, %rd23, %rd24;
    div.u64 %rd25, %rd3, %rd24;
    rem.u64 %rd26, %rd3, %rd24;
    ld.u64 %rd7, [%rd1+40];
    ld.u64 %rd27, [%rd1+48];
    mul.lo.s64 %rd20, %rd25, %rd7;
    mad.lo.s64 %rd20, %rd26, %rd27, %rd20;
    ld.u64 %rd13, [%rd2+40];
    ld.u64 %rd28, [%rd2+48];
    mul.lo.s64 %rd21, %rd25, %rd13;
    mad.lo.s64 %rd21, %rd26, %rd28, %rd21;
    ld.u64 %rd35, [%rd17+40];
    ld.u64 %rd36, [%rd17+48];
    mul.lo.s64 %rd22, %rd25, %rd35;
    mad.lo.s64 %rd22, %rd26, %rd36, %rd22;"""
    if len(shape) == 3:
        return """    ld.u64 %rd23, [%rd1+24];
    ld.u64 %rd24, [%rd1+32];
    ld.u64 %rd25, [%rd1+40];
    mul.lo.s64 %rd26, %rd24, %rd25;
    mul.lo.s64 %rd4, %rd23, %rd26;
    div.u64 %rd27, %rd3, %rd26;
    rem.u64 %rd28, %rd3, %rd26;
    div.u64 %rd29, %rd28, %rd25;
    rem.u64 %rd40, %rd28, %rd25;
    ld.u64 %rd7, [%rd1+48];
    ld.u64 %rd41, [%rd1+56];
    ld.u64 %rd42, [%rd1+64];
    mul.lo.s64 %rd20, %rd27, %rd7;
    mad.lo.s64 %rd20, %rd29, %rd41, %rd20;
    mad.lo.s64 %rd20, %rd40, %rd42, %rd20;
    ld.u64 %rd13, [%rd2+48];
    ld.u64 %rd43, [%rd2+56];
    ld.u64 %rd44, [%rd2+64];
    mul.lo.s64 %rd21, %rd27, %rd13;
    mad.lo.s64 %rd21, %rd29, %rd43, %rd21;
    mad.lo.s64 %rd21, %rd40, %rd44, %rd21;
    ld.u64 %rd35, [%rd17+48];
    ld.u64 %rd36, [%rd17+56];
    ld.u64 %rd37, [%rd17+64];
    mul.lo.s64 %rd22, %rd27, %rd35;
    mad.lo.s64 %rd22, %rd29, %rd36, %rd22;
    mad.lo.s64 %rd22, %rd40, %rd37, %rd22;"""
    raise CodegenUnavailable("direct PTX currently supports rank-1 through rank-3 maps only")


def _unary_f32_ptx_op(operation: F32MapOperation) -> str:
    if operation.op not in {"*", "+", "-", "/"}:
        raise CodegenUnavailable(f"direct PTX does not support operator {operation.op}")
    ptx = ptx_op(operation.op)
    if operation.op in {"-", "/"} and operation.constant_side == "left":
        return f"{ptx} %f3, %f2, %f1;"
    return f"{ptx} %f3, %f1, %f2;"


def _binary_f32_ptx_op(operation: F32MapOperation) -> str:
    if operation.op not in {"*", "+", "-", "/"}:
        raise CodegenUnavailable(f"direct PTX does not support operator {operation.op}")
    return f"{ptx_op(operation.op)} %f3, %f1, %f2;"
