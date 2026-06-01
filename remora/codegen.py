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

from remora.errors import RemoraError
from remora.hir import HIRFunction, HIRLit, HIRMap, HIRPrimCallable, HIRVar
from remora.pipeline import PipelineToolchain, detect_toolchain
from remora.types import FLOAT, ArrayType


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

    Supported today: one rank-1 `float32` input, one rank-1 `float32` output,
    and a primitive unary map section with a literal float constant.
    """
    name = kernel_name or f"remora_{function.name}"
    operation = _direct_rank1_f32_map_operation(function)
    extent = function.return_type.shape[0].value
    ptx = _rank1_f32_map_ptx(name, operation, block_size)
    return ptx, [
        KernelMeta(
            name=name,
            grid_dims=1,
            block_size=block_size,
            num_inputs=1,
            num_outputs=1,
            input_elem_types=["f32"],
            output_elem_types=["f32"],
            output_shape=(extent,),
            output_dtype="float32",
        )
    ]


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


@dataclass(frozen=True)
class _Rank1F32MapOperation:
    op: str
    constant: float
    constant_side: str


def _direct_rank1_f32_map_operation(function: HIRFunction) -> _Rank1F32MapOperation:
    if len(function.params) != 1:
        raise CodegenUnavailable("direct PTX currently supports one input descriptor")
    param = function.params[0]
    if not (
        isinstance(param.type, ArrayType)
        and param.type.element == FLOAT
        and param.type.rank == 1
    ):
        raise CodegenUnavailable("direct PTX currently supports rank-1 f32 inputs only")
    if not (
        isinstance(function.return_type, ArrayType)
        and function.return_type.element == FLOAT
        and function.return_type.rank == 1
    ):
        raise CodegenUnavailable("direct PTX currently supports rank-1 f32 outputs only")
    if param.type.shape != function.return_type.shape:
        raise CodegenUnavailable("direct PTX input and output shapes must match")
    if not (
        isinstance(function.body, HIRMap)
        and len(function.body.arrays) == 1
        and isinstance(function.body.arrays[0], HIRVar)
        and function.body.arrays[0].name == param.name
        and isinstance(function.body.func, HIRPrimCallable)
    ):
        raise CodegenUnavailable("direct PTX currently supports unary primitive maps only")

    callable_ = function.body.func
    if isinstance(callable_.left_arg, HIRLit) and callable_.left_arg.type == FLOAT:
        return _Rank1F32MapOperation(callable_.op, float(callable_.left_arg.value), "left")
    if isinstance(callable_.right_arg, HIRLit) and callable_.right_arg.type == FLOAT:
        return _Rank1F32MapOperation(callable_.op, float(callable_.right_arg.value), "right")
    raise CodegenUnavailable("direct PTX map requires a literal f32 section constant")


def _rank1_f32_map_ptx(
    kernel_name: str,
    operation: _Rank1F32MapOperation,
    block_size: int,
) -> str:
    op_line = _rank1_f32_ptx_op(operation)
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
    .reg .b64 %rd<20>;
    .reg .f32 %f<4>;

    ld.param.u64 %rd1, [input_desc_param];
    ld.param.u64 %rd2, [output_desc_param];
    mov.u32 %r1, %tid.x;
    mov.u32 %r2, %ctaid.x;
    mov.u32 %r3, %ntid.x;
    mad.lo.s32 %r4, %r2, %r3, %r1;
    cvt.s64.s32 %rd3, %r4;

    ld.u64 %rd4, [%rd1+24];
    setp.ge.s64 %p, %rd3, %rd4;
    @%p bra DONE;

    ld.u64 %rd5, [%rd1+8];
    ld.u64 %rd6, [%rd1+16];
    ld.u64 %rd7, [%rd1+32];
    mad.lo.s64 %rd8, %rd3, %rd7, %rd6;
    mul.lo.s64 %rd9, %rd8, 4;
    add.s64 %rd10, %rd5, %rd9;
    ld.global.f32 %f1, [%rd10];
    mov.f32 %f2, {operation.constant:.8e};
    {op_line}

    ld.u64 %rd11, [%rd2+8];
    ld.u64 %rd12, [%rd2+16];
    ld.u64 %rd13, [%rd2+32];
    mad.lo.s64 %rd14, %rd3, %rd13, %rd12;
    mul.lo.s64 %rd15, %rd14, 4;
    add.s64 %rd16, %rd11, %rd15;
    st.global.f32 [%rd16], %f3;

DONE:
    ret;
}}
"""


def _rank1_f32_ptx_op(operation: _Rank1F32MapOperation) -> str:
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
