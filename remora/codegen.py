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
from remora.execution_plan import BufferSpec, ExecutionPlan, KernelStep, LoopPlan
from remora.hir import HIRFunction, HIRParam, HIRProgram
from remora.hir import HIRArrayLit, HIRFold, HIRIota, HIRLit, HIRVar
from remora.hir import HIRFilter, HIRIndicesOf, HIRMatmul, HIRReplicate, HIRScatterAdd, HIRSort, HIRGrade
from remora.types import ArrayType, BOOL, FLOAT64 as _FLOAT64, INT, ScalarType
from remora.pipeline import (
    PipelineToolchain,
    detect_toolchain,
    translate_llvmir_to_nvptx_text,
    translate_mlir_to_llvmir,
)
from remora.gpu_lowering import (
    GPUScaffoldError,
    build_descriptor_abi_bitonic_grade_gpu_module,
    build_descriptor_abi_bitonic_sort_gpu_module,
    build_descriptor_abi_bool_map_gpu_module,
    build_descriptor_abi_cell_fold_dot_gpu_module,
    build_descriptor_abi_f32_map_gpu_module,
    build_descriptor_abi_f32_compound_fold_gpu_module,
    build_descriptor_abi_f32_reduction_gpu_module,
    build_descriptor_abi_f32_scan_gpu_module,
    build_descriptor_abi_filter_gpu_module,
    build_descriptor_abi_general_map_gpu_module,
    build_descriptor_abi_grade_gpu_module,
    build_descriptor_abi_i32_map_gpu_module,
    build_descriptor_abi_im2col_gpu_module,
    build_descriptor_abi_indices_of_gpu_module,
    build_descriptor_abi_matmul_gpu_module,
    build_descriptor_abi_multiblock_bitonic_sort_gpu_module,
    build_descriptor_abi_multiblock_bitonic_grade_gpu_module,
    build_descriptor_abi_multiblock_f32_scan_gpu_module,
    build_descriptor_abi_parallel_filter_gpu_module,
    build_descriptor_abi_parallel_replicate_gpu_module,
    build_descriptor_abi_parallel_scatter_add_gpu_module,
    build_descriptor_abi_replicate_gpu_module,
    build_descriptor_abi_scatter_add_gpu_module,
    build_descriptor_abi_sobel_gpu_module,
    build_descriptor_abi_sort_gpu_module,
    build_descriptor_abi_tiled_matmul_gpu_module,
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
    input_kinds: list[str] | None = None
    output_shape: tuple[int, ...] | None = None
    output_dtype: str | None = None
    is_reduction: bool = False
    grid_size: int | None = None


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
    functions: dict | None = None,
) -> tuple[str, list[KernelMeta], ExecutionPlan | None]:
    """Generate the first MLIR-derived descriptor-ABI PTX execution slice.

    This is intentionally narrow: rank-1 through rank-3 unary/binary
    `float32` maps. The generated GPU kernel accepts Remora descriptor pointers
    directly, so the exported entry can be launched by `RemoraExecutor`.
    """
    toolchain = detect_toolchain() if toolchain is None else toolchain
    name = kernel_name or f"remora_{function.name}"

    # ── try GPU im2col first (most specific) ──
    from remora.hir import HIRIm2col

    if isinstance(function.body, HIRIm2col):
        from remora.types import ArrayType as _AT_unused

        im2col = function.body
        kh, kw = im2col.kernel_shape
        param_type = function.params[0].type
        if isinstance(param_type, ArrayType) and param_type.rank == 2:
            h, w = int(param_type.shape[0].value), int(param_type.shape[1].value)
            ppa = (h - kh) // im2col.stride + 1
            pc = ppa * ppa
            ps = kh * kw
            gpu_module = build_descriptor_abi_im2col_gpu_module(function, kernel_name=name)
            meta = KernelMeta(
                name=name,
                grid_dims=1,
                block_size=0,
                num_inputs=1,
                num_outputs=1,
                input_elem_types=["f32"],
                output_elem_types=["f32"],
                output_shape=(pc, ps),
                output_dtype="float32",
            )
            device_module = extract_gpu_module_body_as_module(gpu_module.text)
            llvm_ir = translate_mlir_to_llvmir(device_module, toolchain=toolchain)
            ptx = translate_llvmir_to_nvptx_text(llvm_ir, toolchain=toolchain)
            return ptx, [meta], None

    # ── try GPU scatter-add ──
    if isinstance(function.body, HIRScatterAdd):
        try:
            gpu_module = build_descriptor_abi_parallel_scatter_add_gpu_module(function, kernel_name=name)
            sa_result = function.body.result_type
            sa_shape = tuple(int(d.value) for d in sa_result.shape) if isinstance(sa_result, ArrayType) else ()
            sa_N = sa_shape[0] if sa_shape else 1
            num_array = sum(1 for p in function.params if isinstance(p.type, ArrayType))
            num_scalar = sum(1 for p in function.params if not isinstance(p.type, ArrayType))
            meta = KernelMeta(
                name=name,
                grid_dims=1,
                block_size=sa_N,
                num_inputs=num_array + num_scalar,
                num_outputs=1,
                input_elem_types=["f32"] * (num_array + num_scalar),
                output_elem_types=["f32"],
                output_shape=sa_shape,
                output_dtype="float32",
            )
            device_module = extract_gpu_module_body_as_module(gpu_module.text)
            llvm_ir = translate_mlir_to_llvmir(device_module, toolchain=toolchain)
            ptx = translate_llvmir_to_nvptx_text(llvm_ir, toolchain=toolchain)
            return ptx, [meta], None
        except GPUScaffoldError:
            pass

        try:
            gpu_module = build_descriptor_abi_scatter_add_gpu_module(function, kernel_name=name)
            sa_result = function.body.result_type
            sa_shape = tuple(int(d.value) for d in sa_result.shape) if isinstance(sa_result, ArrayType) else ()
            num_array = sum(1 for p in function.params if isinstance(p.type, ArrayType))
            num_scalar = sum(1 for p in function.params if not isinstance(p.type, ArrayType))
            meta = KernelMeta(
                name=name,
                grid_dims=1,
                block_size=1,
                num_inputs=num_array + num_scalar,
                num_outputs=1,
                input_elem_types=["f32"] * (num_array + num_scalar),
                output_elem_types=["f32"],
                output_shape=sa_shape,
                output_dtype="float32",
            )
            device_module = extract_gpu_module_body_as_module(gpu_module.text)
            llvm_ir = translate_mlir_to_llvmir(device_module, toolchain=toolchain)
            ptx = translate_llvmir_to_nvptx_text(llvm_ir, toolchain=toolchain)
            return ptx, [meta], None
        except GPUScaffoldError:
            pass

    # ── try GPU matmul ──
    if isinstance(function.body, HIRMatmul):
        try:
            TILE = 16
            gpu_module = build_descriptor_abi_tiled_matmul_gpu_module(function, kernel_name=name, tile_size=TILE)
            mm_result = function.body.result_type
            mm_shape = tuple(int(d.value) for d in mm_result.shape) if isinstance(mm_result, ArrayType) else ()
            mm_M = mm_shape[0] if len(mm_shape) >= 1 else 1
            mm_N = mm_shape[1] if len(mm_shape) >= 2 else 1
            gridRows = (mm_M + TILE - 1) // TILE
            gridCols = (mm_N + TILE - 1) // TILE
            meta = KernelMeta(
                name=name,
                grid_dims=1,
                block_size=TILE * TILE,
                num_inputs=2,
                num_outputs=1,
                input_elem_types=["f32", "f32"],
                output_elem_types=["f32"],
                output_shape=mm_shape,
                output_dtype="float32",
                grid_size=gridRows * gridCols,
            )
            device_module = extract_gpu_module_body_as_module(gpu_module.text)
            llvm_ir = translate_mlir_to_llvmir(device_module, toolchain=toolchain)
            ptx = translate_llvmir_to_nvptx_text(llvm_ir, toolchain=toolchain)
            return ptx, [meta], None
        except GPUScaffoldError:
            pass

        try:
            gpu_module = build_descriptor_abi_matmul_gpu_module(function, kernel_name=name)
            mm_result = function.body.result_type
            mm_shape = tuple(int(d.value) for d in mm_result.shape) if isinstance(mm_result, ArrayType) else ()
            meta = KernelMeta(
                name=name,
                grid_dims=1,
                block_size=0,
                num_inputs=2,
                num_outputs=1,
                input_elem_types=["f32", "f32"],
                output_elem_types=["f32"],
                output_shape=mm_shape,
                output_dtype="float32",
            )
            device_module = extract_gpu_module_body_as_module(gpu_module.text)
            llvm_ir = translate_mlir_to_llvmir(device_module, toolchain=toolchain)
            ptx = translate_llvmir_to_nvptx_text(llvm_ir, toolchain=toolchain)
            return ptx, [meta], None
        except GPUScaffoldError:
            pass

    # ── try GPU sort / grade ──
    if isinstance(function.body, (HIRSort, HIRGrade)):
        if isinstance(function.body, HIRSort):
            try:
                from remora._gpu_radix_sort import build_radix_sort_gpu_module
                rx_text, rx_metas, rx_plan = build_radix_sort_gpu_module(function, kernel_name=name)
                rx_dev = extract_gpu_module_body_as_module(rx_text)
                rx_ir = translate_mlir_to_llvmir(rx_dev, toolchain=toolchain)
                rx_ptx = translate_llvmir_to_nvptx_text(rx_ir, toolchain=toolchain)
                return rx_ptx, rx_metas, rx_plan
            except GPUScaffoldError:
                pass
        try:
            # Detect element type for sort/grade
            _sg_elem = getattr(function.params[0].type, 'element', None)
            _sg_ename = getattr(_sg_elem, 'name', 'float') if _sg_elem else 'float'
            out_et = "i32" if _sg_ename == "int" else "f32"
            out_dtype = "int32" if _sg_ename == "int" else "float32"
            if isinstance(function.body, HIRSort):
                gpu_module = build_descriptor_abi_bitonic_sort_gpu_module(function, kernel_name=name)
            else:
                gpu_module = build_descriptor_abi_bitonic_grade_gpu_module(function, kernel_name=name)
                out_dtype = "int32"
            sg_result = function.body.result_type
            sg_shape = tuple(int(d.value) for d in sg_result.shape) if isinstance(sg_result, ArrayType) else ()
            sg_N = sg_shape[0] if sg_shape else 1
            sg_NP = 1
            while sg_NP < sg_N:
                sg_NP *= 2
            meta = KernelMeta(
                name=name, grid_dims=1, block_size=sg_NP, num_inputs=1, num_outputs=1,
                input_elem_types=[out_et],
                output_elem_types=[out_et],
                output_shape=sg_shape, output_dtype=out_dtype, grid_size=1,
            )
            device_module = extract_gpu_module_body_as_module(gpu_module.text)
            llvm_ir = translate_mlir_to_llvmir(device_module, toolchain=toolchain)
            ptx = translate_llvmir_to_nvptx_text(llvm_ir, toolchain=toolchain)
            return ptx, [meta], None
        except GPUScaffoldError:
            pass

        if isinstance(function.body, HIRSort):
            try:
                gpu_module = build_descriptor_abi_multiblock_bitonic_sort_gpu_module(function, kernel_name=name)
                sg_result = function.body.result_type
                sg_shape = tuple(int(d.value) for d in sg_result.shape) if isinstance(sg_result, ArrayType) else ()
                mb_N = sg_shape[0] if sg_shape else 1
                mb_NP = 1
                while mb_NP < mb_N:
                    mb_NP *= 2
                mb_BS = 1024
                mb_nblocks = mb_NP // mb_BS
                mb_local_stages = 10
                mb_num_stages = mb_NP.bit_length() - 1
                mb_local_name = f"{name}_local"
                mb_all_kernels = [KernelMeta(
                    name=mb_local_name, grid_dims=1, block_size=mb_BS, num_inputs=1, num_outputs=1,
                    input_elem_types=["f32"], output_elem_types=["f32"],
                    output_shape=(mb_NP,), output_dtype="float32", grid_size=mb_nblocks,
                )]
                mb_steps_list: list[KernelStep] = [KernelStep(mb_local_name, ["input_0"], "sorted_a")]
                mb_step_idx = 0
                for mb_k in range(mb_local_stages, mb_num_stages):
                    for mb_j in range(mb_k, -1, -1):
                        sn = f"{name}_gstep_{mb_step_idx}"
                        mb_all_kernels.append(KernelMeta(
                            name=sn, grid_dims=1, block_size=0, num_inputs=1, num_outputs=1,
                            input_elem_types=["f32"], output_elem_types=["f32"],
                            output_shape=(mb_NP,), output_dtype="float32",
                        ))
                        if mb_step_idx % 2 == 0:
                            mb_steps_list.append(KernelStep(sn, ["sorted_a"], "sorted_b"))
                        else:
                            mb_steps_list.append(KernelStep(sn, ["sorted_b"], "sorted_a"))
                        mb_step_idx += 1
                mb_final = "sorted_b" if mb_step_idx % 2 == 1 else "sorted_a"
                mb_plan = ExecutionPlan(
                    buffers=[BufferSpec("sorted_a", (mb_NP,), "f32"), BufferSpec("sorted_b", (mb_NP,), "f32")],
                    steps=mb_steps_list,
                    final_output=mb_final,
                    output_shape=sg_shape,
                    output_dtype="f32",
                )
                device_module = extract_gpu_module_body_as_module(gpu_module.text)
                llvm_ir = translate_mlir_to_llvmir(device_module, toolchain=toolchain)
                ptx = translate_llvmir_to_nvptx_text(llvm_ir, toolchain=toolchain)
                return ptx, mb_all_kernels, mb_plan
            except GPUScaffoldError:
                pass

        from remora.hir import HIRGrade as _HIRGrade_mb
        if isinstance(function.body, _HIRGrade_mb):
            try:
                gpu_module = build_descriptor_abi_multiblock_bitonic_grade_gpu_module(function, kernel_name=name)
                sg_result = function.body.result_type
                sg_shape = tuple(int(d.value) for d in sg_result.shape) if isinstance(sg_result, ArrayType) else ()
                mg_N = sg_shape[0] if sg_shape else 1
                mg_NP = 1
                while mg_NP < mg_N:
                    mg_NP *= 2
                mg_BS = 1024
                mg_nblocks = mg_NP // mg_BS
                mg_local_stages = 10
                mg_num_stages = mg_NP.bit_length() - 1
                mg_pad = f"{name}_pad"
                mg_local = f"{name}_local"
                mg_kernels = [
                    KernelMeta(name=mg_pad, grid_dims=1, block_size=0, num_inputs=1, num_outputs=1,
                               input_elem_types=["f32"], output_elem_types=["f32"],
                               output_shape=(mg_NP,), output_dtype="float32"),
                    KernelMeta(name=mg_local, grid_dims=1, block_size=mg_BS, num_inputs=1, num_outputs=1,
                               input_elem_types=["f32"], output_elem_types=["i32"],
                               output_shape=(mg_NP,), output_dtype="int32", grid_size=mg_nblocks),
                ]
                mg_steps: list[KernelStep] = [
                    KernelStep(mg_pad, ["input_0"], "values_padded"),
                    KernelStep(mg_local, ["values_padded"], "indices_a"),
                ]
                mg_step_idx = 0
                for mg_k in range(mg_local_stages, mg_num_stages):
                    for mg_j in range(mg_k, -1, -1):
                        sn = f"{name}_gstep_{mg_step_idx}"
                        mg_kernels.append(KernelMeta(
                            name=sn, grid_dims=1, block_size=0, num_inputs=2, num_outputs=1,
                            input_elem_types=["f32", "i32"], output_elem_types=["i32"],
                            output_shape=(mg_NP,), output_dtype="int32",
                        ))
                        if mg_step_idx % 2 == 0:
                            mg_steps.append(KernelStep(sn, ["values_padded", "indices_a"], "indices_b"))
                        else:
                            mg_steps.append(KernelStep(sn, ["values_padded", "indices_b"], "indices_a"))
                        mg_step_idx += 1
                mg_final = "indices_b" if mg_step_idx % 2 == 1 else "indices_a"
                mg_plan = ExecutionPlan(
                    buffers=[
                        BufferSpec("values_padded", (mg_NP,), "f32"),
                        BufferSpec("indices_a", (mg_NP,), "i32"),
                        BufferSpec("indices_b", (mg_NP,), "i32"),
                    ],
                    steps=mg_steps,
                    final_output=mg_final,
                    output_shape=sg_shape,
                    output_dtype="i32",
                )
                device_module = extract_gpu_module_body_as_module(gpu_module.text)
                llvm_ir = translate_mlir_to_llvmir(device_module, toolchain=toolchain)
                ptx = translate_llvmir_to_nvptx_text(llvm_ir, toolchain=toolchain)
                return ptx, mg_kernels, mg_plan
            except GPUScaffoldError:
                pass

        try:
            if isinstance(function.body, HIRSort):
                gpu_module = build_descriptor_abi_sort_gpu_module(function, kernel_name=name)
                out_dtype = "float32"
            else:
                gpu_module = build_descriptor_abi_grade_gpu_module(function, kernel_name=name)
                out_dtype = "int32"
            sg_result = function.body.result_type
            sg_shape = tuple(int(d.value) for d in sg_result.shape) if isinstance(sg_result, ArrayType) else ()
            meta = KernelMeta(
                name=name, grid_dims=1, block_size=1, num_inputs=1, num_outputs=1,
                input_elem_types=["f32"],
                output_elem_types=["f32" if out_dtype == "float32" else "i32"],
                output_shape=sg_shape, output_dtype=out_dtype,
            )
            device_module = extract_gpu_module_body_as_module(gpu_module.text)
            llvm_ir = translate_mlir_to_llvmir(device_module, toolchain=toolchain)
            ptx = translate_llvmir_to_nvptx_text(llvm_ir, toolchain=toolchain)
            return ptx, [meta], None
        except GPUScaffoldError:
            pass

    # ── try GPU indices-of ──
    if isinstance(function.body, HIRIndicesOf):
        try:
            gpu_module = build_descriptor_abi_indices_of_gpu_module(function, kernel_name=name)
            io_result = function.body.result_type
            io_shape = tuple(int(d.value) for d in io_result.shape) if isinstance(io_result, ArrayType) else ()
            meta = KernelMeta(
                name=name,
                grid_dims=1,
                block_size=0,
                num_inputs=1,
                num_outputs=1,
                input_elem_types=["f32"],
                output_elem_types=["i32"],
                output_shape=io_shape,
                output_dtype="int32",
            )
            device_module = extract_gpu_module_body_as_module(gpu_module.text)
            llvm_ir = translate_mlir_to_llvmir(device_module, toolchain=toolchain)
            ptx = translate_llvmir_to_nvptx_text(llvm_ir, toolchain=toolchain)
            return ptx, [meta], None
        except GPUScaffoldError:
            pass

    # ── try GPU filter ──
    if isinstance(function.body, HIRFilter):
        try:
            gpu_module = build_descriptor_abi_parallel_filter_gpu_module(function, kernel_name=name)
            f_shape = tuple(int(d.value) for d in function.params[0].type.shape) if isinstance(function.params[0].type, ArrayType) else ()
            N = f_shape[0] if f_shape else 1
            pred_name = f"{name}_pred"
            scan_name = f"{name}_scan"
            scatter_name = f"{name}_scatter"
            kernels = [
                KernelMeta(
                    name=pred_name, grid_dims=1, block_size=0,
                    num_inputs=1, num_outputs=1,
                    input_elem_types=["f32"], output_elem_types=["i32"],
                    output_shape=f_shape, output_dtype="int32",
                ),
                KernelMeta(
                    name=scan_name, grid_dims=1, block_size=N,
                    num_inputs=1, num_outputs=1,
                    input_elem_types=["i32"], output_elem_types=["i32"],
                    output_shape=f_shape, output_dtype="int32",
                ),
                KernelMeta(
                    name=scatter_name, grid_dims=1, block_size=0,
                    num_inputs=3, num_outputs=1,
                    input_elem_types=["f32", "i32", "i32"], output_elem_types=["f32"],
                    output_shape=f_shape, output_dtype="float32",
                ),
            ]
            plan = ExecutionPlan(
                buffers=[
                    BufferSpec("pred", f_shape, "i32"),
                    BufferSpec("scan", f_shape, "i32"),
                    BufferSpec("output", f_shape, "f32"),
                ],
                steps=[
                    KernelStep(pred_name, ["input_0"], "pred"),
                    KernelStep(scan_name, ["pred"], "scan"),
                    KernelStep(scatter_name, ["input_0", "pred", "scan"], "output"),
                ],
                final_output="output",
                output_shape=f_shape,
                output_dtype="f32",
            )
            device_module = extract_gpu_module_body_as_module(gpu_module.text)
            llvm_ir = translate_mlir_to_llvmir(device_module, toolchain=toolchain)
            ptx = translate_llvmir_to_nvptx_text(llvm_ir, toolchain=toolchain)
            return ptx, kernels, plan
        except GPUScaffoldError:
            pass

        try:
            gpu_module = build_descriptor_abi_filter_gpu_module(function, kernel_name=name)
            f_shape = tuple(int(d.value) for d in function.params[0].type.shape) if isinstance(function.params[0].type, ArrayType) else ()
            meta = KernelMeta(
                name=name, grid_dims=1, block_size=1, num_inputs=1, num_outputs=1,
                input_elem_types=["f32"], output_elem_types=["f32"],
                output_shape=f_shape, output_dtype="float32",
            )
            device_module = extract_gpu_module_body_as_module(gpu_module.text)
            llvm_ir = translate_mlir_to_llvmir(device_module, toolchain=toolchain)
            ptx = translate_llvmir_to_nvptx_text(llvm_ir, toolchain=toolchain)
            return ptx, [meta], None
        except GPUScaffoldError:
            pass

    # ── try GPU replicate ──
    if isinstance(function.body, HIRReplicate):
        try:
            gpu_module = build_descriptor_abi_parallel_replicate_gpu_module(function, kernel_name=name)
            r_N = int(function.params[1].type.shape[0].value) if isinstance(function.params[1].type, ArrayType) else 0
            out_N = r_N * r_N
            scan_name_r = f"{name}_scan"
            scatter_name_r = f"{name}_scatter"
            kernels = [
                KernelMeta(
                    name=scan_name_r, grid_dims=1, block_size=r_N,
                    num_inputs=1, num_outputs=1,
                    input_elem_types=["i32"], output_elem_types=["i32"],
                    output_shape=(r_N,), output_dtype="int32",
                ),
                KernelMeta(
                    name=scatter_name_r, grid_dims=1, block_size=0,
                    num_inputs=3, num_outputs=1,
                    input_elem_types=["i32", "f32", "i32"], output_elem_types=["f32"],
                    output_shape=(out_N,), output_dtype="float32",
                ),
            ]
            plan = ExecutionPlan(
                buffers=[
                    BufferSpec("scan", (r_N,), "i32"),
                    BufferSpec("output", (out_N,), "f32"),
                ],
                steps=[
                    KernelStep(scan_name_r, ["input_0"], "scan"),
                    KernelStep(scatter_name_r, ["input_0", "input_1", "scan"], "output"),
                ],
                final_output="output",
                output_shape=(out_N,),
                output_dtype="f32",
            )
            device_module = extract_gpu_module_body_as_module(gpu_module.text)
            llvm_ir = translate_mlir_to_llvmir(device_module, toolchain=toolchain)
            ptx = translate_llvmir_to_nvptx_text(llvm_ir, toolchain=toolchain)
            return ptx, kernels, plan
        except GPUScaffoldError:
            pass

        try:
            gpu_module = build_descriptor_abi_replicate_gpu_module(function, kernel_name=name)
            r_N = int(function.params[1].type.shape[0].value) if isinstance(function.params[1].type, ArrayType) else 0
            meta = KernelMeta(
                name=name, grid_dims=1, block_size=1, num_inputs=2, num_outputs=1,
                input_elem_types=["i32", "f32"], output_elem_types=["f32"],
                output_shape=(r_N * r_N,), output_dtype="float32",
            )
            device_module = extract_gpu_module_body_as_module(gpu_module.text)
            llvm_ir = translate_mlir_to_llvmir(device_module, toolchain=toolchain)
            ptx = translate_llvmir_to_nvptx_text(llvm_ir, toolchain=toolchain)
            return ptx, [meta], None
        except GPUScaffoldError:
            pass

    # ── try GPU cell-fold dot-product (convolution) ──
    try:
        from remora.gpu_lowering import _cell_fold_dot_kernel
        from remora.types import ArrayType as _AT_unused

        _, (kh, kw), stride = _cell_fold_dot_kernel(function)
        param_type2 = function.params[0].type
        if isinstance(param_type2, ArrayType) and param_type2.rank == 2:
            h2, w2 = int(param_type2.shape[0].value), int(param_type2.shape[1].value)
            ppa2 = (h2 - kh) // stride + 1
            pc2 = ppa2 * ppa2
            gpu_module = build_descriptor_abi_cell_fold_dot_gpu_module(function, kernel_name=name)
            meta = KernelMeta(
                name=name,
                grid_dims=1,
                block_size=0,
                num_inputs=2,
                num_outputs=1,
                input_elem_types=["f32", "f32"],
                output_elem_types=["f32"],
                output_shape=(pc2,),
                output_dtype="float32",
            )
            device_module = extract_gpu_module_body_as_module(gpu_module.text)
            llvm_ir = translate_mlir_to_llvmir(device_module, toolchain=toolchain)
            ptx = translate_llvmir_to_nvptx_text(llvm_ir, toolchain=toolchain)
            return ptx, [meta], None
    except GPUScaffoldError:
        pass

    # ── try GPU view ops (Reverse, Rotate, Take, Drop, Slice, Subarray, Reshape, Ravel, Append) ──
    try:
        from remora.hir import HIRReverse, HIRRotate, HIRTake, HIRDrop, HIRSlice, HIRSubarray, HIRReshape, HIRRavel, HIRAppend, HIRTranspose
        from remora.gpu_lowering import (
            build_descriptor_abi_reverse_gpu_module,
            build_descriptor_abi_rotate_gpu_module,
            build_descriptor_abi_take_gpu_module,
            build_descriptor_abi_drop_gpu_module,
            build_descriptor_abi_slice_gpu_module,
            build_descriptor_abi_subarray_gpu_module,
            build_descriptor_abi_reshape_gpu_module,
            build_descriptor_abi_ravel_gpu_module,
            build_descriptor_abi_append_gpu_module,
            build_descriptor_abi_transpose_gpu_module,
        )
        body = function.body
        if isinstance(body, HIRReverse):
            gpu_module = build_descriptor_abi_reverse_gpu_module(function, kernel_name=name)
        elif isinstance(body, HIRRotate):
            gpu_module = build_descriptor_abi_rotate_gpu_module(function, kernel_name=name)
        elif isinstance(body, HIRTake):
            gpu_module = build_descriptor_abi_take_gpu_module(function, kernel_name=name)
        elif isinstance(body, HIRDrop):
            gpu_module = build_descriptor_abi_drop_gpu_module(function, kernel_name=name)
        elif isinstance(body, HIRSlice):
            gpu_module = build_descriptor_abi_slice_gpu_module(function, kernel_name=name)
        elif isinstance(body, HIRSubarray):
            gpu_module = build_descriptor_abi_subarray_gpu_module(function, kernel_name=name)
        elif isinstance(body, HIRReshape):
            gpu_module = build_descriptor_abi_reshape_gpu_module(function, kernel_name=name)
        elif isinstance(body, HIRRavel):
            gpu_module = build_descriptor_abi_ravel_gpu_module(function, kernel_name=name)
        elif isinstance(body, HIRAppend):
            gpu_module = build_descriptor_abi_append_gpu_module(function, kernel_name=name)
        elif isinstance(body, HIRTranspose):
            gpu_module = build_descriptor_abi_transpose_gpu_module(function, kernel_name=name)
        else:
            raise GPUScaffoldError("not a supported view op")
        rt = body.result_type
        v_shape = tuple(int(d.value) for d in rt.shape) if isinstance(rt, ArrayType) else ()
        v_total = 1
        for d in v_shape:
            v_total *= d
        v_num_inputs = 2 if isinstance(body, HIRAppend) else 1
        v_ie = ["f32"] * v_num_inputs
        meta = KernelMeta(
            name=name, grid_dims=1, block_size=max(1, min(v_total, 1024)),
            num_inputs=v_num_inputs, num_outputs=1,
            input_elem_types=v_ie, output_elem_types=["f32"],
            output_shape=v_shape, output_dtype="float32",
        )
        device_module = extract_gpu_module_body_as_module(gpu_module.text)
        llvm_ir = translate_mlir_to_llvmir(device_module, toolchain=toolchain)
        ptx = translate_llvmir_to_nvptx_text(llvm_ir, toolchain=toolchain)
        return ptx, [meta], None
    except (GPUScaffoldError, CodegenUnavailable):
        pass

    # ── try GPU Sobel (combined 2-kernel cell-fold) ──
    try:
        from remora.gpu_lowering import _sobel_kernel

        _, (kh, kw), stride = _sobel_kernel(function)
        param_type3 = function.params[0].type
        if isinstance(param_type3, ArrayType) and param_type3.rank == 2:
            h3, w3 = int(param_type3.shape[0].value), int(param_type3.shape[1].value)
            ppa3 = (h3 - kh) // stride + 1
            pc3 = ppa3 * ppa3
            gpu_module = build_descriptor_abi_sobel_gpu_module(function, kernel_name=name)
            meta = KernelMeta(
                name=name,
                grid_dims=1,
                block_size=0,
                num_inputs=3,
                num_outputs=1,
                input_elem_types=["f32", "f32", "f32"],
                output_elem_types=["f32"],
                output_shape=(pc3,),
                output_dtype="float32",
            )
            device_module = extract_gpu_module_body_as_module(gpu_module.text)
            llvm_ir = translate_mlir_to_llvmir(device_module, toolchain=toolchain)
            ptx = translate_llvmir_to_nvptx_text(llvm_ir, toolchain=toolchain)
            return ptx, [meta], None
    except GPUScaffoldError:
        pass

    # ── try GPU general map kernel for compound-body maps ──
    try:
        from remora.hir import HIRLambda, HIRMap
        from remora.lowering.tensor_ops import _body_needs_tensor_lowering

        if (isinstance(function.body, HIRMap)
                and isinstance(function.body.func, HIRLambda)
                and _body_needs_tensor_lowering(function.body.func.body)):
            gpu_module = build_descriptor_abi_general_map_gpu_module(
                function, kernel_name=name, functions=functions,
            )

            body_map = function.body
            result_type = body_map.result_type
            if not isinstance(result_type, ArrayType):
                raise CodegenUnavailable(
                    "general GPU map requires an array result type"
                )

            output_shape = tuple(
                int(d.value) for d in result_type.shape
            )

            # Determine input types from function params
            num_array_inputs = sum(
                1 for p in function.params
                if isinstance(p.type, ArrayType)
            )
            num_scalar_inputs = sum(
                1 for p in function.params
                if not isinstance(p.type, ArrayType)
            )
            total_inputs = num_array_inputs + num_scalar_inputs

            input_elem_types: list[str] = []
            for param in function.params:
                if isinstance(param.type, ArrayType):
                    elem = param.type.element.name
                    if elem == "float":
                        input_elem_types.append("f32")
                    elif elem == "float64":
                        input_elem_types.append("f64")
                    elif elem == "int":
                        input_elem_types.append("i32")
                    elif elem == "bool":
                        input_elem_types.append("i1")
                    else:
                        input_elem_types.append("f32")
                else:
                    input_elem_types.append("f32")

            _out_elem_name = getattr(result_type.element, "name", "float")
            if _out_elem_name == "int":
                _out_et, _out_dtype = "i32", "int32"
            elif _out_elem_name == "bool":
                _out_et, _out_dtype = "i8", "bool"
            elif _out_elem_name == "float64":
                _out_et, _out_dtype = "f64", "float64"
            else:
                _out_et, _out_dtype = "f32", "float32"

            _total = 1
            for d in output_shape:
                _total *= d
            _kind: list[str] = []
            for param in function.params:
                _kind.append("array" if isinstance(param.type, ArrayType) else "scalar")
            meta = KernelMeta(
                name=name,
                grid_dims=1,
                block_size=max(1, min(_total, 1024)),
                num_inputs=total_inputs,
                num_outputs=1,
                input_elem_types=input_elem_types,
                output_elem_types=[_out_et],
                output_shape=output_shape,
                output_dtype=_out_dtype,
                input_kinds=_kind,
            )
            device_module = extract_gpu_module_body_as_module(gpu_module.text)
            llvm_ir = translate_mlir_to_llvmir(device_module, toolchain=toolchain)
            ptx = translate_llvmir_to_nvptx_text(llvm_ir, toolchain=toolchain)
            return ptx, [meta], None
    except (GPUScaffoldError, CodegenUnavailable):
        pass

    try:
        map_kernel = _direct_f32_map_kernel(function)
        rank = len(map_kernel.shape)
        if rank < 1 or rank > 10:
            raise CodegenUnavailable(
                "MLIR-derived descriptor-ABI PTX currently supports rank-1 through rank-10 f32 maps only"
            )
        if (
            map_kernel.expression is None
            and map_kernel.num_inputs == 1
            and map_kernel.operation.constant is None
            and map_kernel.scalar_count == 0
        ):
            raise CodegenUnavailable(
                "MLIR-derived descriptor-ABI PTX currently supports unary literal-section or binary f32 maps only"
            )
        if map_kernel.num_inputs not in (1, 2) and map_kernel.expression is None:
            raise CodegenUnavailable(
                "MLIR-derived descriptor-ABI PTX currently supports one or two f32 input descriptors only"
            )
        gpu_module = build_descriptor_abi_f32_map_gpu_module(function, kernel_name=name)
        input_kinds = list(map_kernel.input_kinds) if map_kernel.input_kinds else None
        total_inputs = map_kernel.num_inputs + map_kernel.scalar_count
        input_elem_types = ["f32"] * total_inputs if total_inputs > 1 else ["f32"]
        meta = KernelMeta(
            name=name,
            grid_dims=1,
            block_size=0,
            num_inputs=map_kernel.num_inputs + map_kernel.scalar_count,
            num_outputs=1,
            input_elem_types=input_elem_types,
            output_elem_types=["f32"],
            input_kinds=input_kinds,
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
                    _red_ename = function.params[0].type.element.name if isinstance(function.params[0].type, ArrayType) else "float"
                    _red_et = ("f32" if _red_ename == "float" else "i32" if _red_ename == "int"
                               else "f64" if _red_ename == "float64" else "f32")
                    _red_dt = ("float32" if _red_ename == "float" else "int32" if _red_ename == "int"
                               else "float64" if _red_ename == "float64" else "float32")
                    meta = KernelMeta(
                        name=name,
                        grid_dims=1,
                        block_size=0,
                        num_inputs=num_inputs,
                        num_outputs=1,
                        input_elem_types=[_red_et] * num_inputs,
                        output_elem_types=[_red_et],
                        output_shape=(),
                        output_dtype=_red_dt,
                        is_reduction=True,
                    )
                except GPUScaffoldError as reduction_error:
                    try:
                        gpu_module = build_descriptor_abi_f32_compound_fold_gpu_module(
                            function, kernel_name=name, functions=functions,
                        )
                        meta = KernelMeta(
                            name=name,
                            grid_dims=1,
                            block_size=1,
                            num_inputs=1,
                            num_outputs=1,
                            input_elem_types=["f32"],
                            output_elem_types=["f32"],
                            output_shape=(),
                            output_dtype="float32",
                            is_reduction=True,
                        )
                    except GPUScaffoldError as compound_fold_error:
                        try:
                            gpu_module = build_descriptor_abi_f32_scan_gpu_module(
                                function, kernel_name=name, functions=functions,
                            )
                            scan_shape = tuple(int(d.value) for d in function.params[0].type.shape)
                            _num_inputs = len(function.params)
                            _scan_elem_types: list[str] = []
                            _scan_kinds: list[str] = []
                            for p in function.params:
                                _scan_kinds.append("array")
                                if isinstance(p.type, ArrayType):
                                    en = p.type.element.name
                                    _scan_elem_types.append("f32" if en == "float" else "i1" if en == "bool" else "f32")
                                else:
                                    _scan_elem_types.append("f32")
                            _out_ename = function.params[0].type.element.name if isinstance(function.params[0].type, ArrayType) else "float"
                            _out_et = ("f32" if _out_ename == "float" else "i1" if _out_ename == "bool"
                                       else "i32" if _out_ename == "int" else "f64" if _out_ename == "float64"
                                       else "f32")
                            _out_dt = ("float32" if _out_ename == "float" else "bool" if _out_ename == "bool"
                                       else "int32" if _out_ename == "int" else "float64" if _out_ename == "float64"
                                       else "float32")
                            meta = KernelMeta(
                                name=name,
                                grid_dims=1,
                                block_size=scan_shape[0],
                                num_inputs=_num_inputs,
                                num_outputs=1,
                                input_elem_types=_scan_elem_types,
                                output_elem_types=[_out_et],
                                output_shape=scan_shape,
                                output_dtype=_out_dt,
                                input_kinds=_scan_kinds,
                            )
                        except GPUScaffoldError as scan_error:
                            try:
                                mb_module, mb_kernels, mb_buffers, mb_steps, sc_shape = build_descriptor_abi_multiblock_f32_scan_gpu_module(function, kernel_name=name)
                                mb_plan = ExecutionPlan(
                                    buffers=mb_buffers,
                                    steps=mb_steps,
                                    final_output="scanned",
                                    output_shape=sc_shape,
                                    output_dtype="f32",
                                )
                                mb_dev = extract_gpu_module_body_as_module(mb_module.text)
                                mb_ir = translate_mlir_to_llvmir(mb_dev, toolchain=toolchain)
                                mb_ptx = translate_llvmir_to_nvptx_text(mb_ir, toolchain=toolchain)
                                return mb_ptx, mb_kernels, mb_plan
                            except GPUScaffoldError:
                                pass
                            try:
                                from remora.hir import HIRLambda as _HIRLambda2, HIRMap as _HIRMap2, HIRApply as _HIRApply2
                                if not isinstance(function.body, (_HIRMap2, _HIRApply2)):
                                    raise CodegenUnavailable(
                                        "general GPU fallback requires a HIRMap with HIRLambda"
                                    )
                                gpu_module = build_descriptor_abi_general_map_gpu_module(
                                    function, kernel_name=name, functions=functions,
                                )
                                body_map2 = function.body
                                result_type2 = body_map2.result_type
                                if not isinstance(result_type2, ArrayType):
                                    raise CodegenUnavailable(
                                        "general GPU map fallback requires an array result type"
                                    )
                                output_shape2 = tuple(
                                    int(d.value) for d in result_type2.shape
                                )
                                num_array_inputs2 = sum(
                                    1 for p in function.params
                                    if isinstance(p.type, ArrayType)
                                )
                                num_scalar_inputs2 = sum(
                                    1 for p in function.params
                                    if isinstance(p.type, ScalarType)
                                )
                                _input_types2: list[str] = []
                                _input_kinds2: list[str] = []
                                for p in function.params:
                                    if isinstance(p.type, ArrayType):
                                        _input_kinds2.append("array")
                                        if p.type.element == INT:
                                            _input_types2.append("i32")
                                        elif p.type.element == BOOL:
                                            _input_types2.append("i1")
                                        elif p.type.element == _FLOAT64:
                                            _input_types2.append("f64")
                                        else:
                                            _input_types2.append("f32")
                                    elif isinstance(p.type, ScalarType):
                                        _input_kinds2.append("scalar")
                                        if p.type == INT:
                                            _input_types2.append("i32")
                                        elif p.type == BOOL:
                                            _input_types2.append("i1")
                                        elif p.type == _FLOAT64:
                                            _input_types2.append("f64")
                                        else:
                                            _input_types2.append("f32")
                                _out_elem2 = result_type2.element
                                _out_et2 = ("i32" if _out_elem2 == INT else "i1" if _out_elem2 == BOOL
                                            else "f64" if _out_elem2 == _FLOAT64 else "f32")
                                _out_dt2 = ("int32" if _out_elem2 == INT else "bool" if _out_elem2 == BOOL
                                            else "float64" if _out_elem2 == _FLOAT64 else "float32")
                                meta = KernelMeta(
                                    name=name,
                                    grid_dims=1,
                                    block_size=0,
                                    num_inputs=num_array_inputs2 + num_scalar_inputs2,
                                    num_outputs=1,
                                    input_elem_types=_input_types2,
                                    output_elem_types=[_out_et2],
                                    input_kinds=_input_kinds2,
                                    output_shape=output_shape2,
                                    output_dtype=_out_dt2,
                                )
                            except Exception as general_map_error:
                                for _rec_err in (
                                    general_map_error,
                                    scan_error,
                                    compound_fold_error,
                                    reduction_error,
                                ):
                                    _rec_msg = str(_rec_err)
                                    if "GPU recursion supports" in _rec_msg:
                                        raise CodegenUnavailable(_rec_msg) from _rec_err
                                raise CodegenUnavailable(
                                    "MLIR-derived descriptor-ABI PTX could not lower function to any GPU kernel: "
                                    f"f32_map={f32_map_error}; i32_map={i32_map_error}; bool_map={bool_map_error}; "
                                    f"reduction={reduction_error}; compound_fold={compound_fold_error}; "
                                    f"scan={scan_error}; general_map={general_map_error}"
                                ) from general_map_error
    device_module = extract_gpu_module_body_as_module(gpu_module.text)
    llvm_ir = translate_mlir_to_llvmir(device_module, toolchain=toolchain)
    ptx = translate_llvmir_to_nvptx_text(llvm_ir, toolchain=toolchain)
    return ptx, [meta], None


def generate_rank1_f32_unary_mlir_descriptor_abi_ptx(
    function: HIRFunction,
    *,
    kernel_name: str | None = None,
    toolchain: PipelineToolchain | None = None,
) -> tuple[str, list[KernelMeta], ExecutionPlan | None]:
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


def _hir_uses_var(node: Any, var_name: str) -> bool:
    """Check if a variable name is referenced anywhere in an HIR expression."""
    if isinstance(node, HIRVar):
        return node.name == var_name
    if isinstance(node, HIRLit):
        return False
    for attr in ("body", "func", "array", "init", "value", "callable"):
        child = getattr(node, attr, None)
        if child is not None and _hir_uses_var(child, var_name):
            return True
    for attr in ("args", "arrays", "elements"):
        children = getattr(node, attr, None)
        if children is not None:
            for child in children:
                if hasattr(child, "__dataclass_fields__") or isinstance(child, (HIRVar, HIRLit)):
                    if _hir_uses_var(child, var_name):
                        return True
    return False


def try_compile_state_fold_gpu(
    program: HIRProgram,
    *,
    toolchain: PipelineToolchain | None = None,
) -> tuple[str, list[KernelMeta], ExecutionPlan] | None:
    """Try to compile a state-fold-over-iota as a GPU loop plan.

    Detects the pattern::

        fold body_fn init (iota N)

    where ``body_fn(accumulator, step)`` does not use ``step`` and
    produces a result with the same type as the accumulator.  The body
    is compiled as a single GPU kernel, and a ``LoopPlan`` iterates
    it N times with buffer swapping.

    Returns ``(ptx, kernels, plan)`` on success, or ``None`` if the
    pattern does not match.
    """
    from remora.types import ArrayType

    if program.main is None or not isinstance(program.main, HIRFold):
        return None

    fold = program.main
    if not isinstance(fold.array, HIRIota):
        return None

    if not isinstance(fold.func, HIRVar):
        return None

    func_name = fold.func.name
    body_func = None
    for f in program.functions:
        if f.name == func_name:
            body_func = f
            break
    if body_func is None:
        return None

    if len(body_func.params) != 2:
        return None
    acc_param = body_func.params[0]
    step_param = body_func.params[1]

    if not isinstance(acc_param.type, ArrayType):
        return None
    if acc_param.type.element.name != "float":
        return None

    if _hir_uses_var(body_func.body, step_param.name):
        return None

    result_type = fold.result_type
    if not isinstance(result_type, ArrayType):
        return None
    shape = tuple(int(d.value) for d in result_type.shape)
    if result_type.rank != 1:
        return None

    init_values: tuple[float, ...] | None = None
    if isinstance(fold.init, HIRArrayLit):
        try:
            init_values = tuple(float(e.value) for e in fold.init.elements if isinstance(e, HIRLit))
            if len(init_values) != len(fold.init.elements):
                return None
        except (ValueError, AttributeError):
            return None
    else:
        return None

    N = int(fold.array.size.value)

    step_func = HIRFunction(
        name="__gpu_fold_step",
        params=[acc_param],
        body=body_func.body,
        return_type=result_type,
    )

    try:
        kernel_name = "fold_step"
        from remora.gpu_lowering import (
            build_descriptor_abi_general_map_gpu_module as _build_general,
            extract_gpu_module_body_as_module as _extract,
        )
        from remora.pipeline import translate_mlir_to_llvmir, translate_llvmir_to_nvptx_text
        gpu_module = _build_general(step_func, kernel_name=kernel_name)
        device_module = _extract(gpu_module.text)
        llvm_ir = translate_mlir_to_llvmir(device_module, toolchain=toolchain)
        ptx = translate_llvmir_to_nvptx_text(llvm_ir, toolchain=toolchain)
        kernels = [KernelMeta(
            name=kernel_name, grid_dims=1, block_size=0,
            num_inputs=1, num_outputs=1,
            input_elem_types=["f32"], output_elem_types=["f32"],
            output_shape=shape, output_dtype="float32",
        )]
    except (GPUScaffoldError, CodegenUnavailable):
        return None

    plan = ExecutionPlan(
        buffers=[
            BufferSpec("params", shape, "f32", init=init_values),
            BufferSpec("params_new", shape, "f32"),
        ],
        steps=[
            LoopPlan(
                count=N,
                body=[KernelStep(kernel_name, ["params"], "params_new")],
                swap_pairs=[("params", "params_new")],
            ),
        ],
        final_output="params",
        output_shape=shape,
        output_dtype="f32",
    )

    return ptx, kernels, plan
