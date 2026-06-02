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

from remora._gpu_map_support import F32MapKernel, F32MapOperation, analyze_supported_f32_map_function
from remora.errors import RemoraError
from remora.hir import HIRFunction
from remora.pipeline import (
    PipelineToolchain,
    detect_toolchain,
    run_gpu_nvidia_scaffold_llvm_dialect_pipeline_text,
    translate_llvmir_to_nvptx_text,
    translate_mlir_to_llvmir,
)
from remora.gpu_lowering import build_gpu_scaffold_for_function, extract_gpu_module_body_as_module


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
    output_shape: tuple[int, ...] = ()
    output_dtype: str | None = None


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
            f"--iree-cuda-target-features={ptx_features}",
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

    This is intentionally narrow: rank-1 unary/binary and rank-2 unary
    `float32` maps. The inner kernel still comes from the scaffold GPU path; a
    descriptor-pointer ABI wrapper is injected before NVPTX emission so the
    exported entry can be launched by `RemoraExecutor`.
    """
    toolchain = detect_toolchain() if toolchain is None else toolchain
    name = kernel_name or f"remora_{function.name}"
    map_kernel = _direct_f32_map_kernel(function)
    rank = len(map_kernel.shape)
    if rank not in (1, 2):
        raise CodegenUnavailable(
            "MLIR-derived descriptor-ABI PTX currently supports rank-1 and rank-2 f32 maps only"
        )
    if map_kernel.num_inputs == 1 and map_kernel.operation.constant is None:
        raise CodegenUnavailable(
            "MLIR-derived descriptor-ABI PTX currently supports unary literal-section or binary rank-1, and unary literal-section rank-2, f32 maps only"
        )
    if map_kernel.num_inputs not in (1, 2):
        raise CodegenUnavailable(
            "MLIR-derived descriptor-ABI PTX currently supports one or two rank-1/rank-2 f32 input descriptors only"
        )
    if rank == 2 and map_kernel.num_inputs != 1:
        raise CodegenUnavailable(
            "MLIR-derived descriptor-ABI PTX currently supports rank-2 unary f32 maps only"
        )

    scaffold = build_gpu_scaffold_for_function(function, kernel_name=name)
    lowered = run_gpu_nvidia_scaffold_llvm_dialect_pipeline_text(
        scaffold.text,
        toolchain=toolchain,
    )
    device_module = extract_gpu_module_body_as_module(lowered)
    llvm_ir = translate_mlir_to_llvmir(device_module, toolchain=toolchain)
    wrapped_llvm_ir = _wrap_descriptor_abi_kernel_llvm_ir(
        llvm_ir,
        kernel_name=name,
        rank=rank,
        num_inputs=map_kernel.num_inputs,
    )
    ptx = translate_llvmir_to_nvptx_text(wrapped_llvm_ir, toolchain=toolchain)
    return ptx, [
        KernelMeta(
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
    ]


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


def _wrap_descriptor_abi_kernel_llvm_ir(
    llvm_ir: str,
    *,
    kernel_name: str,
    rank: int,
    num_inputs: int,
) -> str:
    inner_name = f"{kernel_name}_inner"
    wrapped = llvm_ir.replace(f"@{kernel_name}", f"@{inner_name}")
    wrapped = re.sub(r"!nvvm\.annotations = !\{[^\n]*\}\n", "", wrapped)
    wrapped = re.sub(
        rf"!\d+ = !\{{ptr @{re.escape(inner_name)}, !\"kernel\", i32 1\}}\n",
        "",
        wrapped,
    )
    wrapped = wrapped.replace(
        'source_filename = "LLVMDialectModule"\n',
        'source_filename = "LLVMDialectModule"\n'
        f'%remora.memref{rank} = type {{ ptr, ptr, i64, '
        + ", ".join(["i64"] * rank)
        + (", " if rank else "")
        + ", ".join(["i64"] * rank)
        + " }\n",
        1,
    )
    params = [f"ptr %input{index}_desc" for index in range(num_inputs)] + ["ptr %output_desc"]
    descriptor_lines: list[str] = []
    call_args: list[str] = []
    for index in range(num_inputs):
        prefix = f"in{index}"
        descriptor_name = f"%input{index}_desc"
        descriptor_lines.extend(_descriptor_wrapper_lines(prefix, descriptor_name, rank=rank))
        call_args.extend(_descriptor_call_args(prefix, rank=rank))
    descriptor_lines.extend(_descriptor_wrapper_lines("out", "%output_desc", rank=rank))
    call_args.extend(_descriptor_call_args("out", rank=rank))
    wrapper = (
        f"\ndefine void @{kernel_name}(" + ", ".join(params) + ") {\n"
        + "\n".join(descriptor_lines)
        + f"\n  call void @{inner_name}(" + ", ".join(call_args) + ")\n"
        + "  ret void\n}\n"
    )
    wrapped = wrapped.replace("\nattributes #0 =", f"{wrapper}\nattributes #0 =", 1)
    metadata_ids = [int(match.group(1)) for match in re.finditer(r"!(\d+) =", wrapped)]
    next_id = max(metadata_ids, default=-1) + 1
    wrapped = wrapped.rstrip() + f"\n\n!nvvm.annotations = !{{!{next_id}}}\n!{next_id} = !{{ptr @{kernel_name}, !\"kernel\", i32 1}}\n"
    return wrapped


def _descriptor_wrapper_lines(prefix: str, descriptor_name: str, *, rank: int) -> list[str]:
    struct_name = f"%remora.memref{rank}"
    lines = [
        f"  %{prefix}_alloc_ptr = getelementptr {struct_name}, ptr {descriptor_name}, i32 0, i32 0",
        f"  %{prefix}_aligned_ptr = getelementptr {struct_name}, ptr {descriptor_name}, i32 0, i32 1",
        f"  %{prefix}_offset_ptr = getelementptr {struct_name}, ptr {descriptor_name}, i32 0, i32 2",
    ]
    for index in range(rank):
        lines.append(
            f"  %{prefix}_size{index}_ptr = getelementptr {struct_name}, ptr {descriptor_name}, i32 0, i32 {3 + index}"
        )
    for index in range(rank):
        lines.append(
            f"  %{prefix}_stride{index}_ptr = getelementptr {struct_name}, ptr {descriptor_name}, i32 0, i32 {3 + rank + index}"
        )
    lines.extend(
        [
            f"  %{prefix}_alloc = load ptr, ptr %{prefix}_alloc_ptr, align 8",
            f"  %{prefix}_aligned = load ptr, ptr %{prefix}_aligned_ptr, align 8",
            f"  %{prefix}_offset = load i64, ptr %{prefix}_offset_ptr, align 8",
        ]
    )
    for index in range(rank):
        lines.append(f"  %{prefix}_size{index} = load i64, ptr %{prefix}_size{index}_ptr, align 8")
    for index in range(rank):
        lines.append(f"  %{prefix}_stride{index} = load i64, ptr %{prefix}_stride{index}_ptr, align 8")
    return lines


def _descriptor_call_args(prefix: str, *, rank: int) -> list[str]:
    args = [
        f"ptr %{prefix}_alloc",
        f"ptr %{prefix}_aligned",
        f"i64 %{prefix}_offset",
    ]
    args.extend(f"i64 %{prefix}_size{index}" for index in range(rank))
    args.extend(f"i64 %{prefix}_stride{index}" for index in range(rank))
    return args


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
    kernel: _F32MapKernel,
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
    if operation.op == "*":
        return "mul.rn.f32 %f3, %f1, %f2;"
    if operation.op == "+":
        return "add.rn.f32 %f3, %f1, %f2;"
    if operation.op == "-":
        if operation.constant_side == "left":
            return "sub.rn.f32 %f3, %f2, %f1;"
        return "sub.rn.f32 %f3, %f1, %f2;"
    if operation.op == "/":
        if operation.constant_side == "left":
            return "div.rn.f32 %f3, %f2, %f1;"
        return "div.rn.f32 %f3, %f1, %f2;"
    raise CodegenUnavailable(f"direct PTX does not support operator {operation.op}")


def _binary_f32_ptx_op(operation: F32MapOperation) -> str:
    if operation.op == "*":
        return "mul.rn.f32 %f3, %f1, %f2;"
    if operation.op == "+":
        return "add.rn.f32 %f3, %f1, %f2;"
    if operation.op == "-":
        return "sub.rn.f32 %f3, %f1, %f2;"
    if operation.op == "/":
        return "div.rn.f32 %f3, %f1, %f2;"
    raise CodegenUnavailable(f"direct PTX does not support operator {operation.op}")
