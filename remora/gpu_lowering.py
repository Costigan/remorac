"""Experimental MLIR GPU module scaffolds for the production NVIDIA path."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from remora._gpu_map_support import (
    F32BinaryExpr,
    F32CmpExpr,
    F32ConstantExpr,
    F32Expr,
    F32InputExpr,
    F32ScalarParamExpr,
    F32MapKernel,
    F32MapOperation,
    F32SelectExpr,
    I32MapKernel,
    I32MapOperation,
    analyze_supported_f32_map_function,
    analyze_supported_i32_map_function,
)
from remora.errors import RemoraError
from remora.hir import HIRFold, HIRFunction, HIRLambda, HIRLit, HIRMap, HIRPrimCallable, HIRVar
from remora.hir import HIRApply
from remora.hir import HIRAppend, HIRDrop, HIRFilter, HIRIndicesOf, HIRMatmul, HIRRavel, HIRReplicate, HIRReshape, HIRReverse, HIRRotate, HIRScatterAdd, HIRSort, HIRGrade, HIRSubarray, HIRTake, HIRTranspose, HIRWithShape
from remora.operators import arith_op, llvm_op
from remora.types import FLOAT, ArrayType


class GPUScaffoldError(RemoraError):
    """Raised when an experimental GPU scaffold cannot be built."""


@dataclass(frozen=True)
class GPUModuleScaffold:
    text: str
    module_name: str
    kernel_name: str


def extract_gpu_module_body_as_module(
    mlir_text: str,
    *,
    module_name: str = "remora_gpu",
) -> str:
    """Extract one `gpu.module` body and wrap it as a top-level MLIR module.

    `mlir-translate --mlir-to-llvmir` does not translate a nested `gpu.module`
    body from the full host module. This helper is a narrow scaffold utility for
    device-module translation experiments after GPU-to-NVVM conversion.
    """
    marker = f"gpu.module @{module_name}"
    start = mlir_text.find(marker)
    if start < 0:
        raise GPUScaffoldError(f"gpu.module @{module_name} was not found")
    body_start = mlir_text.find("{", start)
    if body_start < 0:
        raise GPUScaffoldError(f"gpu.module @{module_name} has no body")
    body_end = _matching_brace_index(mlir_text, body_start)
    body = mlir_text[body_start + 1 : body_end]
    return "module {\n" + _dedent_gpu_module_body(body) + "\n}\n"


def build_rank1_f32_unary_map_gpu_scaffold(
    *,
    size: int,
    multiplier: float = 2.0,
    module_name: str = "remora_gpu",
    kernel_name: str = "remora_map_rank1_f32",
) -> GPUModuleScaffold:
    """Build a parseable `gpu.module` scaffold for a rank-1 f32 scale map.

    This intentionally stops before NVVM lowering and runtime launch support.
    It gives the production GPU path a concrete MLIR target shape while the CPU
    path remains the correctness oracle.
    """
    return build_f32_unary_map_gpu_scaffold(
        shape=(size,),
        operation="*",
        constant=multiplier,
        constant_side="right",
        module_name=module_name,
        kernel_name=kernel_name,
    )


def build_f32_unary_map_gpu_scaffold(
    *,
    shape: tuple[int, ...],
    operation: str,
    constant: float,
    constant_side: str = "right",
    module_name: str = "remora_gpu",
    kernel_name: str = "remora_map_f32",
) -> GPUModuleScaffold:
    """Build a gpu.module scaffold for a rank-N f32 unary map kernel."""
    _validate_scaffold_names(module_name, kernel_name)
    return _build_f32_map_gpu_scaffold(
        F32MapKernel(
            shape=shape,
            operation=F32MapOperation(
                operation,
                float(constant),
                constant_side,
            ),
            num_inputs=1,
        ),
        module_name=module_name,
        kernel_name=kernel_name,
    )


def build_f32_binary_map_gpu_scaffold(
    *,
    shape: tuple[int, ...],
    operation: str,
    module_name: str = "remora_gpu",
    kernel_name: str = "remora_map_f32_binary",
) -> GPUModuleScaffold:
    """Build a gpu.module scaffold for a rank-N f32 binary map kernel."""
    _validate_scaffold_names(module_name, kernel_name)
    return _build_f32_map_gpu_scaffold(
        F32MapKernel(
            shape=shape,
            operation=F32MapOperation(operation),
            num_inputs=2,
        ),
        module_name=module_name,
        kernel_name=kernel_name,
    )


def _matching_brace_index(text: str, open_index: int) -> int:
    """Return the index of the closing brace matching the brace at `open_index`."""
    depth = 0
    for index in range(open_index, len(text)):
        char = text[index]
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return index
    raise GPUScaffoldError("unterminated gpu.module body")


def _dedent_gpu_module_body(body: str) -> str:
    """Strip leading/trailing blank lines and remove 4-space indentation."""
    lines = body.splitlines()
    while lines and not lines[0].strip():
        lines.pop(0)
    while lines and not lines[-1].strip():
        lines.pop()
    dedented: list[str] = []
    for line in lines:
        dedented.append(line[4:] if line.startswith("    ") else line)
    return "\n".join(dedented)


def build_gpu_scaffold_for_function(
    function: HIRFunction,
    *,
    module_name: str = "remora_gpu",
    kernel_name: str | None = None,
) -> GPUModuleScaffold:
    """Build the experimental GPU scaffold from a supported HIR function."""
    kernel = _f32_map_kernel(function)
    return _build_f32_map_gpu_scaffold(
        kernel,
        module_name=module_name,
        kernel_name=kernel_name or f"remora_{function.name}_f32",
    )


def build_descriptor_abi_f32_map_gpu_module(
    function: HIRFunction,
    *,
    module_name: str = "remora_gpu",
    kernel_name: str | None = None,
) -> GPUModuleScaffold:
    """Build an executable descriptor-ABI GPU module for a supported f32 map."""
    kernel = _f32_map_kernel(function)
    name = kernel_name or f"remora_{function.name}_f32"
    _validate_scaffold_names(module_name, name)
    return _build_descriptor_abi_f32_map_gpu_module(
        kernel,
        module_name=module_name,
        kernel_name=name,
    )


def build_descriptor_abi_i32_map_gpu_module(
    function: HIRFunction,
    *,
    module_name: str = "remora_gpu",
    kernel_name: str | None = None,
) -> GPUModuleScaffold:
    """Build an executable descriptor-ABI GPU module for a supported i32 map."""
    kernel = _i32_map_kernel(function)
    name = kernel_name or f"remora_{function.name}_i32"
    _validate_scaffold_names(module_name, name)
    return _build_descriptor_abi_i32_map_gpu_module(
        kernel,
        module_name=module_name,
        kernel_name=name,
    )


def build_descriptor_abi_f32_reduction_gpu_module(
    function: HIRFunction,
    *,
    module_name: str = "remora_gpu",
    kernel_name: str | None = None,
) -> GPUModuleScaffold:
    """Build an executable descriptor-ABI GPU module for a supported f32 reduction."""
    kernel = _f32_reduction_kernel(function)
    name = kernel_name or f"remora_{function.name}_f32"
    _validate_scaffold_names(module_name, name)
    return _build_descriptor_abi_f32_reduction_gpu_module(
        kernel,
        module_name=module_name,
        kernel_name=name,
    )


def _cell_fold_dot_kernel(function: HIRFunction) -> tuple[HIRFunction, tuple[int, int], int]:
    """Return ``(function, kernel_shape, stride)`` if *function* is a valid cell-fold
    dot-product over im2col (convolution pattern)."""
    from remora.hir import HIRApply, HIRFold, HIRIm2col, HIRLambda, HIRMap, HIRPrimCallable, HIRRavel, HIRReduce, HIRVar

    if not isinstance(function.body, (HIRMap, HIRApply)):
        raise GPUScaffoldError("cell-fold dot requires a map as the top-level expression")
    outer_map = function.body
    if not outer_map.cell_shape or len(outer_map.cell_shape) != 1:
        raise GPUScaffoldError("cell-fold dot requires a cell-map with rank-1 cells")
    if not isinstance(outer_map.func, HIRLambda):
        raise GPUScaffoldError("cell-fold dot requires an inline lambda callable")
    body_expr = outer_map.func.body
    if not isinstance(body_expr, (HIRFold, HIRReduce)):
        raise GPUScaffoldError("cell-fold dot requires a fold as the lambda body")
    if not isinstance(body_expr.func, HIRPrimCallable) or body_expr.func.op != "+":
        raise GPUScaffoldError("cell-fold dot currently supports fold + only")
    if not isinstance(body_expr.array, (HIRMap, HIRApply)):
        raise GPUScaffoldError("cell-fold dot requires map * as the fold's array")
    inner_map = body_expr.array
    if not isinstance(inner_map.func, HIRPrimCallable) or inner_map.func.op != "*":
        raise GPUScaffoldError("cell-fold dot requires map * as the inner binary operation")
    inner_arrays = inner_map.arrays
    if len(inner_arrays) != 2:
        raise GPUScaffoldError("cell-fold dot requires exactly two inner operands")

    # One operand must be the cell param (HIRVar named after the lambda param)
    param_name = outer_map.func.params[0].name
    free_operand = None
    for a in inner_arrays:
        if isinstance(a, HIRVar) and a.name == param_name:
            continue
        free_operand = a
    if free_operand is None:
        raise GPUScaffoldError("cell-fold dot could not identify the free kernel operand")

    # The free operand should be a ravel of a kernel parameter
    if not isinstance(free_operand, HIRRavel):
        raise GPUScaffoldError("cell-fold dot requires the kernel operand to be raveled")
    kernel_var = free_operand.array
    if not isinstance(kernel_var, HIRVar):
        raise GPUScaffoldError("cell-fold dot requires the kernel to be a function parameter")

    # The cell-map's array must be an im2col
    if not isinstance(outer_map.array, HIRIm2col):
        raise GPUScaffoldError("cell-fold dot requires im2col as the cell-map array")
    im2col = outer_map.array
    kh, kw = im2col.kernel_shape
    return function, (kh, kw), im2col.stride


def build_descriptor_abi_cell_fold_dot_gpu_module(
    function: HIRFunction,
    *,
    module_name: str = "remora_gpu",
    kernel_name: str | None = None,
) -> GPUModuleScaffold:
    """Build a GPU module for a cell-fold dot-product over im2col (convolution).

    One thread per output patch.  Each thread loads the patch from the image,
    loads the kernel elements, computes the dot product, and stores the result.
    """
    _, (kh, kw), stride = _cell_fold_dot_kernel(function)
    param_type = function.params[0].type
    if not isinstance(param_type, ArrayType):
        raise GPUScaffoldError("cell-fold dot requires an array image parameter")
    h, w = int(param_type.shape[0].value), int(param_type.shape[1].value)
    patches_per_axis = (h - kh) // stride + 1
    patch_count = patches_per_axis * patches_per_axis

    name = kernel_name or f"remora_{function.name}_celldot"
    _validate_scaffold_names(module_name, name)

    # ── descriptor loads (image rank 2, kernel rank 1, output rank 1) ──
    desc_lines = _descriptor_load_lines("img", "%input_desc", 2)
    desc_lines.extend(_descriptor_load_lines("kern", "%kernel_desc", 1))
    desc_lines.extend(_descriptor_load_lines("out", "%output_desc", 1))

    # ── thread ID → patch index ──
    thread_lines = [
        "      %cf_tid32 = nvvm.read.ptx.sreg.tid.x : i32",
        "      %cf_tid = llvm.sext %cf_tid32 : i32 to i64",
        "      %cf_bid32 = nvvm.read.ptx.sreg.ctaid.x : i32",
        "      %cf_bid = llvm.sext %cf_bid32 : i32 to i64",
        "      %cf_bdim32 = nvvm.read.ptx.sreg.ntid.x : i32",
        "      %cf_bdim = llvm.sext %cf_bdim32 : i32 to i64",
        "      %cf_blk = llvm.mul %cf_bid, %cf_bdim  : i64",
        "      %cf_idx = llvm.add %cf_blk, %cf_tid  : i64",
        f"      %cf_n = llvm.mlir.constant({patch_count} : index) : i64",
        "      %cf_ok = llvm.icmp \"ult\" %cf_idx, %cf_n : i64",
        "      llvm.cond_br %cf_ok, ^cf_process, ^cf_done",
    ]

    # ── compute patch coordinates ──
    process_lines = [
        "    ^cf_process:",
        f"      %cf_ppa = llvm.mlir.constant({patches_per_axis} : index) : i64",
        "      %cf_prow = llvm.udiv %cf_idx, %cf_ppa : i64",
        "      %cf_pcol = llvm.urem %cf_idx, %cf_ppa : i64",
    ]

    # ── loop over kh × kw kernel elements ──
    cell_size = kh * kw
    loop_lines = [
        f"      %cf_init = llvm.mlir.constant(0.000000e+00 : f32) : f32",
        f"      %cf_zero = llvm.mlir.constant(0 : index) : i64",
        f"      %cf_one = llvm.mlir.constant(1 : index) : i64",
        f"      %cf_end = llvm.mlir.constant({cell_size} : index) : i64",
        f"      %cf_st = llvm.mlir.constant({stride} : index) : i64",
        f"      %cf_kw = llvm.mlir.constant({kw} : index) : i64",
        "      llvm.br ^cf_loop(%cf_zero, %cf_init : i64, f32)",
        "",
        "    ^cf_loop(%cf_elem: i64, %cf_acc: f32):",
        "      %cf_done = llvm.icmp \"uge\" %cf_elem, %cf_end : i64",
        "      llvm.cond_br %cf_done, ^cf_write(%cf_acc : f32), ^cf_body",
        "",
        "    ^cf_body:",
        "      %cf_krow = llvm.udiv %cf_elem, %cf_kw : i64",
        "      %cf_kcol = llvm.urem %cf_elem, %cf_kw : i64",
        # image row/col
        "      %cf_ir1 = llvm.mul %cf_prow, %cf_st : i64",
        "      %cf_ir2 = llvm.add %cf_ir1, %cf_krow : i64",
        "      %cf_ic1 = llvm.mul %cf_pcol, %cf_st : i64",
        "      %cf_ic2 = llvm.add %cf_ic1, %cf_kcol : i64",
        # linearize image index
        "      %cf_ir3 = llvm.add %cf_ir2, %cf_zero : i64",
        "      %cf_ic3 = llvm.add %cf_ic2, %cf_zero : i64",
        "      %cf_t0 = llvm.mul %cf_ir3, %img_stride0 : i64",
        "      %cf_roff = llvm.add %img_offset, %cf_t0 : i64",
        "      %cf_lin = llvm.add %cf_roff, %cf_ic3 : i64",
        "      %cf_pix_p = llvm.getelementptr %img_aligned[%cf_lin] : (!llvm.ptr, i64) -> !llvm.ptr, f32",
        "      %cf_pix = llvm.load %cf_pix_p : !llvm.ptr -> f32",
        # load kernel element
        "      %cf_ek = llvm.add %cf_elem, %cf_zero : i64",
        "      %cf_koff = llvm.add %kern_offset, %cf_ek : i64",
        "      %cf_kp = llvm.getelementptr %kern_aligned[%cf_koff] : (!llvm.ptr, i64) -> !llvm.ptr, f32",
        "      %cf_kv = llvm.load %cf_kp : !llvm.ptr -> f32",
        # accumulate
        "      %cf_mul = llvm.fmul %cf_pix, %cf_kv : f32",
        "      %cf_new = llvm.fadd %cf_acc, %cf_mul : f32",
        "      %cf_next = llvm.add %cf_elem, %cf_one : i64",
        "      llvm.br ^cf_loop(%cf_next, %cf_new : i64, f32)",
        "",
        "    ^cf_write(%cf_result: f32):",
        # store: output[patch_idx]
        "      %cf_out_off = llvm.add %out_offset, %cf_idx : i64",
        "      %cf_out_p = llvm.getelementptr %out_aligned[%cf_out_off] : (!llvm.ptr, i64) -> !llvm.ptr, f32",
        "      llvm.store %cf_result, %cf_out_p : f32, !llvm.ptr",
        "      llvm.br ^cf_done",
    ]

    done_lines = [
        "    ^cf_done:",
        "      llvm.return",
    ]

    body = desc_lines + thread_lines + process_lines + loop_lines + done_lines
    text = f"""module {{
  gpu.module @{module_name} {{
    llvm.func @{name}(%input_desc: !llvm.ptr, %kernel_desc: !llvm.ptr, %output_desc: !llvm.ptr) attributes {{gpu.kernel, nvvm.kernel}} {{
{chr(10).join(body)}
    }}
  }}
}}"""
    return GPUModuleScaffold(text, module_name, name)
    """Return ``(function, kernel_shape, stride)`` if *function* is a valid im2col."""
    from remora.hir import HIRIm2col

    if not isinstance(function.body, HIRIm2col):
        raise GPUScaffoldError("GPU im2col requires the body to be a single im2col")
    im2col = function.body
    kh, kw = im2col.kernel_shape
    stride = im2col.stride
    if kh <= 0 or kw <= 0:
        raise GPUScaffoldError(f"invalid im2col kernel shape ({kh}, {kw})")
    if stride <= 0:
        raise GPUScaffoldError(f"invalid im2col stride {stride}")
    return function, (kh, kw), stride


def _im2col_kernel(function: HIRFunction) -> tuple[HIRFunction, tuple[int, int], int]:
    """Return ``(function, kernel_shape, stride)`` if *function* is a valid im2col."""
    from remora.hir import HIRIm2col

    if not isinstance(function.body, HIRIm2col):
        raise GPUScaffoldError("GPU im2col requires the body to be a single im2col")
    im2col = function.body
    kh, kw = im2col.kernel_shape
    stride = im2col.stride
    if kh <= 0 or kw <= 0:
        raise GPUScaffoldError(f"invalid im2col kernel shape ({kh}, {kw})")
    if stride <= 0:
        raise GPUScaffoldError(f"invalid im2col stride {stride}")
    return function, (kh, kw), stride


def build_descriptor_abi_im2col_gpu_module(
    function: HIRFunction,
    *,
    module_name: str = "remora_gpu",
    kernel_name: str | None = None,
) -> GPUModuleScaffold:
    """Build a descriptor-ABI GPU module for a 2-D im2col operation.

    One thread per output element: each thread loads the corresponding image
    pixel and stores it into the patches buffer.
    """
    _, (kh, kw), stride = _im2col_kernel(function)
    param_type = function.params[0].type
    if not isinstance(param_type, ArrayType):
        raise GPUScaffoldError("im2col requires an array image parameter")
    h, w = int(param_type.shape[0].value), int(param_type.shape[1].value)
    patches_per_axis = (h - kh) // stride + 1
    patch_count = patches_per_axis * patches_per_axis
    patch_size = kh * kw

    name = kernel_name or f"remora_{function.name}_im2col"
    _validate_scaffold_names(module_name, name)

    # ── descriptor loads (input rank 2, output rank 2) ──
    desc_lines = _descriptor_load_lines("in", "%input_desc", 2)
    desc_lines.extend(_descriptor_load_lines("out", "%output_desc", 2))

    # ── thread ID → flat output index ──
    thread_lines = [
        "      %im2c_tid32 = nvvm.read.ptx.sreg.tid.x : i32",
        "      %im2c_tid = llvm.sext %im2c_tid32 : i32 to i64",
        "      %im2c_bid32 = nvvm.read.ptx.sreg.ctaid.x : i32",
        "      %im2c_bid = llvm.sext %im2c_bid32 : i32 to i64",
        "      %im2c_bdim32 = nvvm.read.ptx.sreg.ntid.x : i32",
        "      %im2c_bdim = llvm.sext %im2c_bdim32 : i32 to i64",
        "      %im2c_block_base = llvm.mul %im2c_bid, %im2c_bdim  : i64",
        "      %im2c_idx = llvm.add %im2c_block_base, %im2c_tid  : i64",
        f"      %im2c_outdim0 = llvm.mlir.constant({patch_count} : index) : i64",
        f"      %im2c_outdim1 = llvm.mlir.constant({patch_size} : index) : i64",
        f"      %im2c_total = llvm.mul %im2c_outdim0, %im2c_outdim1 : i64",
        "      %im2c_in_bounds = llvm.icmp \"ult\" %im2c_idx, %im2c_total : i64",
        "      llvm.cond_br %im2c_in_bounds, ^im2c_process, ^im2c_done",
    ]

    # ── compute patch coordinates and image coordinates ──
    process_lines = [
        f"    ^im2c_process:",
        f"      %im2c_ostride = llvm.mlir.constant({patch_size} : index) : i64",
        "      %im2c_patch_idx = llvm.udiv %im2c_idx, %im2c_ostride  : i64",
        "      %im2c_elem_idx = llvm.urem %im2c_idx, %im2c_ostride  : i64",
        f"      %im2c_ppl = llvm.mlir.constant({patches_per_axis} : index) : i64",
        "      %im2c_prow = llvm.udiv %im2c_patch_idx, %im2c_ppl  : i64",
        "      %im2c_pcol = llvm.urem %im2c_patch_idx, %im2c_ppl  : i64",
        f"      %im2c_ksz = llvm.mlir.constant({kw} : index) : i64",
        "      %im2c_krow = llvm.udiv %im2c_elem_idx, %im2c_ksz  : i64",
        "      %im2c_kcol = llvm.urem %im2c_elem_idx, %im2c_ksz  : i64",
        f"      %im2c_st = llvm.mlir.constant({stride} : index) : i64",
        "      %im2c_ir = llvm.mul %im2c_prow, %im2c_st : i64",
        "      %im2c_ir2 = llvm.add %im2c_ir, %im2c_krow : i64",
        "      %im2c_ic = llvm.mul %im2c_pcol, %im2c_st : i64",
        "      %im2c_ic2 = llvm.add %im2c_ic, %im2c_kcol : i64",
    ]

    # ── linearize image index, load pixel ──
    load_lines = [
        "      %im2c_z = llvm.mlir.constant(0 : index) : i64",
        "      %im2c_ir3 = llvm.add %im2c_ir2, %im2c_z : i64",
        "      %im2c_ic3 = llvm.add %im2c_ic2, %im2c_z : i64",
        "      %im2c_t0 = llvm.mul %im2c_ir3, %in_stride0 : i64",
        "      %im2c_roff = llvm.add %in_offset, %im2c_t0 : i64",
        "      %im2c_lin = llvm.add %im2c_roff, %im2c_ic3 : i64",
        "      %im2c_pix_ptr = llvm.getelementptr %in_aligned[%im2c_lin] : (!llvm.ptr, i64) -> !llvm.ptr, f32",
        "      %im2c_pixel = llvm.load %im2c_pix_ptr : !llvm.ptr -> f32",
    ]

    # ── store to output: out[patch_idx, elem_idx] ──
    store_lines = [
        "      %im2c_z0 = llvm.mlir.constant(0 : index) : i64",
        "      %im2c_z1 = llvm.mlir.constant(0 : index) : i64",
        "      %im2c_po = llvm.add %im2c_patch_idx, %im2c_z0 : i64",
        "      %im2c_pe = llvm.add %im2c_elem_idx, %im2c_z1 : i64",
        "      %im2c_ot0 = llvm.mul %im2c_po, %im2c_ostride : i64",
        "      %im2c_oo0 = llvm.add %out_offset, %im2c_ot0 : i64",
        "      %im2c_olin = llvm.add %im2c_oo0, %im2c_pe : i64",
        "      %im2c_out_ptr = llvm.getelementptr %out_aligned[%im2c_olin] : (!llvm.ptr, i64) -> !llvm.ptr, f32",
        "      llvm.store %im2c_pixel, %im2c_out_ptr : f32, !llvm.ptr",
        "      llvm.br ^im2c_done",
    ]

    done_lines = [
        "    ^im2c_done:",
        "      llvm.return",
    ]

    body = desc_lines + thread_lines + process_lines + load_lines + store_lines + done_lines
    text = f"""module {{
  gpu.module @{module_name} {{
    llvm.func @{name}(%input_desc: !llvm.ptr, %output_desc: !llvm.ptr) attributes {{gpu.kernel, nvvm.kernel}} {{
{chr(10).join(body)}
    }}
  }}
}}"""
    return GPUModuleScaffold(text, module_name, name)
    """Analyze the HIR function and return it if it's a valid scan kernel."""
    if not function.params:
        raise GPUScaffoldError("GPU scan requires a single-parameter function")
    param_type = function.params[0].type
    if not isinstance(param_type, ArrayType):
        raise GPUScaffoldError("GPU scan requires an array input parameter")
    if param_type.element != FLOAT:
        raise GPUScaffoldError("GPU scan currently supports f32 input only")
    if param_type.rank != 1:
        raise GPUScaffoldError("GPU scan currently supports rank-1 input only")
    return function


def build_descriptor_abi_f32_scan_gpu_module(
    function: HIRFunction,
    *,
    module_name: str = "remora_gpu",
    kernel_name: str | None = None,
) -> GPUModuleScaffold:
    """Build a descriptor-ABI GPU module for f32 scan (prefix-sum).

    Uses a parallel Hillis-Steele scan in shared memory.
    One block with N threads for arrays up to 1024 elements.
    Supports inclusive/exclusive, left/right, and + or * operators.

    A production implementation would use a parallel prefix-sum + scatter
    with multi-kernel orchestration for arrays > 1024 (see docs/FUTURE_WORK.md).
    """
    if len(function.params) != 1:
        raise GPUScaffoldError("GPU scan supports single-parameter functions only")
    param_type = function.params[0].type
    if not isinstance(param_type, ArrayType) or param_type.element != FLOAT:
        raise GPUScaffoldError("GPU scan supports rank-1 f32 input only")
    if param_type.rank != 1:
        raise GPUScaffoldError("GPU scan supports rank-1 input only")

    shape = _validate_shape(tuple(int(d.value) for d in param_type.shape))
    N = shape[0]
    name = kernel_name or f"remora_{function.name}_f32_scan"
    _validate_scaffold_names(module_name, name)

    from remora.hir import HIRScan as _HIRScan
    scan_op = "+"
    is_exclusive = False
    is_right = False
    if isinstance(function.body, _HIRScan):
        if isinstance(function.body.func, HIRPrimCallable):
            scan_op = function.body.func.op
        is_exclusive = function.body.exclusive
        is_right = function.body.right

    if scan_op == "+":
        llvm_scan_op = "llvm.fadd"
        identity = "0.000000e+00"
    elif scan_op == "*":
        llvm_scan_op = "llvm.fmul"
        identity = "1.000000e+00"
    else:
        raise GPUScaffoldError(f"GPU scan op '{scan_op}' not supported (only + and *)")

    import math
    max_d = math.ceil(math.log2(N)) if N > 1 else 0

    rank = 1
    desc_lines = _descriptor_load_lines("in", "%input_desc", rank)
    desc_lines.extend(_descriptor_load_lines("out", "%output_desc", rank))

    if N > 1024:
        idx_expr = "llvm.sub %ss_Nm1, %ss_i" if is_right else "llvm.add %ss_i, %ss_c0"
        if is_exclusive:
            body_block = f"""    ^ss_body:
      %ss_idx = {idx_expr}  : i64
      %ss_si = llvm.add %in_offset, %ss_idx  : i64
      %ss_sp = llvm.getelementptr %in_aligned[%ss_si] : (!llvm.ptr, i64) -> !llvm.ptr, f32
      %ss_elem = llvm.load %ss_sp : !llvm.ptr -> f32
      %ss_di = llvm.add %out_offset, %ss_idx  : i64
      %ss_dp = llvm.getelementptr %out_aligned[%ss_di] : (!llvm.ptr, i64) -> !llvm.ptr, f32
      llvm.store %ss_acc, %ss_dp : f32, !llvm.ptr
      %ss_nacc = {llvm_scan_op} %ss_acc, %ss_elem  : f32
      %ss_ni = llvm.add %ss_i, %ss_c1 : i64
      llvm.br ^ss_loop(%ss_ni, %ss_nacc : i64, f32)"""
        else:
            body_block = f"""    ^ss_body:
      %ss_idx = {idx_expr}  : i64
      %ss_si = llvm.add %in_offset, %ss_idx  : i64
      %ss_sp = llvm.getelementptr %in_aligned[%ss_si] : (!llvm.ptr, i64) -> !llvm.ptr, f32
      %ss_elem = llvm.load %ss_sp : !llvm.ptr -> f32
      %ss_nacc = {llvm_scan_op} %ss_acc, %ss_elem  : f32
      %ss_di = llvm.add %out_offset, %ss_idx  : i64
      %ss_dp = llvm.getelementptr %out_aligned[%ss_di] : (!llvm.ptr, i64) -> !llvm.ptr, f32
      llvm.store %ss_nacc, %ss_dp : f32, !llvm.ptr
      %ss_ni = llvm.add %ss_i, %ss_c1 : i64
      llvm.br ^ss_loop(%ss_ni, %ss_nacc : i64, f32)"""

        text = f"""module {{
  gpu.module @{module_name} {{
    llvm.func @{name}(%input_desc: !llvm.ptr, %output_desc: !llvm.ptr) attributes {{gpu.kernel, nvvm.kernel}} {{
{chr(10).join(desc_lines)}
      %ss_tid32 = nvvm.read.ptx.sreg.tid.x : i32
      %ss_tid = llvm.sext %ss_tid32 : i32 to i64
      %ss_c0 = llvm.mlir.constant(0 : index) : i64
      %ss_is_t0 = llvm.icmp "eq" %ss_tid, %ss_c0 : i64
      llvm.cond_br %ss_is_t0, ^ss_work, ^ss_done

    ^ss_work:
      %ss_N = llvm.mlir.constant({N} : index) : i64
      %ss_Nm1 = llvm.mlir.constant({N - 1} : index) : i64
      %ss_c1 = llvm.mlir.constant(1 : index) : i64
      %ss_init = llvm.mlir.constant({identity} : f32) : f32
      llvm.br ^ss_loop(%ss_c0, %ss_init : i64, f32)

    ^ss_loop(%ss_i: i64, %ss_acc: f32):
      %ss_done_cond = llvm.icmp "uge" %ss_i, %ss_N : i64
      llvm.cond_br %ss_done_cond, ^ss_done, ^ss_body

{body_block}

    ^ss_done:
      llvm.return
    }}
  }}
}}"""
        return GPUModuleScaffold(text, module_name, name)

    import math
    max_d = math.ceil(math.log2(N)) if N > 1 else 0

    if is_right:
        load_idx = f"      %sc_load_idx = llvm.sub %sc_Nm1, %sc_tid  : i64"
        write_idx = f"      %sc_write_idx = llvm.sub %sc_Nm1, %sc_tid  : i64"
    else:
        load_idx = "      %sc_load_idx = llvm.add %sc_tid, %sc_c0  : i64"
        write_idx = "      %sc_write_idx = llvm.add %sc_tid, %sc_c0  : i64"

    if is_exclusive:
        write_block = f"""    ^sc_write:
      %sc_is_zero = llvm.icmp "eq" %sc_tid, %sc_c0 : i64
      %sc_id_val = llvm.mlir.constant({identity} : f32) : f32
      %sc_prev_idx_raw = llvm.sub %sc_tid, %sc_c1  : i64
      %sc_prev_idx = llvm.select %sc_is_zero, %sc_c0, %sc_prev_idx_raw : i1, i64
      %sc_prev_ptr = llvm.getelementptr %sc_shmem_base[0, %sc_prev_idx] : (!llvm.ptr<3>, i64) -> !llvm.ptr<3>, !llvm.array<1024 x f32>
      %sc_prev_val = llvm.load %sc_prev_ptr : !llvm.ptr<3> -> f32
      %sc_final = llvm.select %sc_is_zero, %sc_id_val, %sc_prev_val : i1, f32
{write_idx}
      %sc_out_off = llvm.add %out_offset, %sc_write_idx  : i64
      %sc_out_ptr = llvm.getelementptr %out_aligned[%sc_out_off] : (!llvm.ptr, i64) -> !llvm.ptr, f32
      llvm.store %sc_final, %sc_out_ptr : f32, !llvm.ptr
      llvm.br ^sc_done"""
    else:
        write_block = f"""    ^sc_write:
      %sc_final = llvm.load %sc_shmem_me : !llvm.ptr<3> -> f32
{write_idx}
      %sc_out_off = llvm.add %out_offset, %sc_write_idx  : i64
      %sc_out_ptr = llvm.getelementptr %out_aligned[%sc_out_off] : (!llvm.ptr, i64) -> !llvm.ptr, f32
      llvm.store %sc_final, %sc_out_ptr : f32, !llvm.ptr
      llvm.br ^sc_done"""

    text = f"""module {{
  gpu.module @{module_name} {{
    llvm.mlir.global internal @scan_shmem() {{addr_space = 3 : i32}} : !llvm.array<1024 x f32>
    llvm.func @{name}(%input_desc: !llvm.ptr, %output_desc: !llvm.ptr) attributes {{gpu.kernel, nvvm.kernel}} {{
{chr(10).join(desc_lines)}
      %sc_tid32 = nvvm.read.ptx.sreg.tid.x : i32
      %sc_tid = llvm.sext %sc_tid32 : i32 to i64
      %sc_cN = llvm.mlir.constant({N} : index) : i64
      %sc_in_bounds = llvm.icmp "ult" %sc_tid, %sc_cN : i64
      llvm.cond_br %sc_in_bounds, ^sc_load, ^sc_done

    ^sc_load:
      %sc_c0 = llvm.mlir.constant(0 : index) : i64
      %sc_c1 = llvm.mlir.constant(1 : index) : i64
      %sc_c2 = llvm.mlir.constant(2 : index) : i64
      %sc_Nm1 = llvm.mlir.constant({N - 1} : index) : i64
{load_idx}
      %sc_in_off = llvm.add %in_offset, %sc_load_idx  : i64
      %sc_in_ptr = llvm.getelementptr %in_aligned[%sc_in_off] : (!llvm.ptr, i64) -> !llvm.ptr, f32
      %sc_elem = llvm.load %sc_in_ptr : !llvm.ptr -> f32
      %sc_shmem_base = llvm.mlir.addressof @scan_shmem : !llvm.ptr<3>
      %sc_shmem_me = llvm.getelementptr %sc_shmem_base[0, %sc_tid] : (!llvm.ptr<3>, i64) -> !llvm.ptr<3>, !llvm.array<1024 x f32>
      llvm.store %sc_elem, %sc_shmem_me : f32, !llvm.ptr<3>
      nvvm.barrier0
      %sc_max_d = llvm.mlir.constant({max_d} : index) : i64
      llvm.br ^sc_loop(%sc_c0, %sc_c1 : i64, i64)

    ^sc_loop(%sc_d: i64, %sc_stride: i64):
      %sc_loop_done = llvm.icmp "uge" %sc_d, %sc_max_d : i64
      llvm.cond_br %sc_loop_done, ^sc_write, ^sc_step

    ^sc_step:
      %sc_active = llvm.icmp "uge" %sc_tid, %sc_stride : i64
      %sc_partner_raw = llvm.sub %sc_tid, %sc_stride  : i64
      %sc_safe_partner = llvm.select %sc_active, %sc_partner_raw, %sc_c0 : i1, i64
      %sc_pptr = llvm.getelementptr %sc_shmem_base[0, %sc_safe_partner] : (!llvm.ptr<3>, i64) -> !llvm.ptr<3>, !llvm.array<1024 x f32>
      %sc_pval = llvm.load %sc_pptr : !llvm.ptr<3> -> f32
      %sc_id_f = llvm.mlir.constant({identity} : f32) : f32
      %sc_temp = llvm.select %sc_active, %sc_pval, %sc_id_f : i1, f32
      nvvm.barrier0
      %sc_cur = llvm.load %sc_shmem_me : !llvm.ptr<3> -> f32
      %sc_new = {llvm_scan_op} %sc_cur, %sc_temp  : f32
      %sc_result = llvm.select %sc_active, %sc_new, %sc_cur : i1, f32
      llvm.store %sc_result, %sc_shmem_me : f32, !llvm.ptr<3>
      nvvm.barrier0
      %sc_next_d = llvm.add %sc_d, %sc_c1 : i64
      %sc_next_stride = llvm.mul %sc_stride, %sc_c2 : i64
      llvm.br ^sc_loop(%sc_next_d, %sc_next_stride : i64, i64)

{write_block}

    ^sc_done:
      llvm.return
    }}
  }}
}}"""
    return GPUModuleScaffold(text, module_name, name)


def build_descriptor_abi_multiblock_f32_scan_gpu_module(
    function: HIRFunction,
    *,
    module_name: str = "remora_gpu",
    kernel_name: str | None = None,
    block_size: int = 1024,
) -> GPUModuleScaffold:
    """Build a multi-block f32 inclusive-add scan using 4 kernels.

    Phase 1 (scan_local): per-block Hillis-Steele scan.
    Phase 2 (extract_sums): extract last element of each block.
    Phase 3 (scan_sums): single-block scan of block sums.
    Phase 4 (propagate): add block prefix to each element.

    Supports N up to block_size * 1024 (default: 1,048,576).
    """
    if len(function.params) != 1:
        raise GPUScaffoldError("GPU multi-block scan requires one parameter")
    param_type = function.params[0].type
    if not isinstance(param_type, ArrayType) or param_type.element != FLOAT:
        raise GPUScaffoldError("GPU multi-block scan supports f32 only")
    if param_type.rank != 1:
        raise GPUScaffoldError("GPU multi-block scan supports rank-1 only")

    N = int(param_type.shape[0].value)
    BS = block_size
    if N <= BS:
        raise GPUScaffoldError("Use single-block scan for N <= block_size")
    num_blocks = (N + BS - 1) // BS
    if num_blocks > 1024:
        raise GPUScaffoldError(f"N={N} requires {num_blocks} blocks (max 1024)")

    import math
    max_d_local = math.ceil(math.log2(BS))
    max_d_sums = math.ceil(math.log2(num_blocks)) if num_blocks > 1 else 1

    base = kernel_name or f"remora_{function.name}_scan"
    local_name = f"{base}_local"
    extract_name = f"{base}_extract"
    sums_name = f"{base}_sums"
    propagate_name = f"{base}_propagate"
    _validate_scaffold_names(module_name, local_name)

    k1_desc = _descriptor_load_lines("sl_in", "%input_desc", 1)
    k1_desc.extend(_descriptor_load_lines("sl_out", "%output_desc", 1))

    k2_desc = _descriptor_load_lines("se_sc", "%scanned_desc", 1)
    k2_desc.extend(_descriptor_load_lines("se_bs", "%block_sums_desc", 1))

    k3_desc = _descriptor_load_lines("ss_in", "%sums_desc", 1)
    k3_desc.extend(_descriptor_load_lines("ss_out", "%prefix_desc", 1))

    k4_desc = _descriptor_load_lines("sp_pr", "%prefix_desc", 1)
    k4_desc.extend(_descriptor_load_lines("sp_sc", "%scanned_desc", 1))

    text = f"""module {{
  gpu.module @{module_name} {{
    llvm.mlir.global internal @mb_shmem() {{addr_space = 3 : i32}} : !llvm.array<{BS} x f32>
    llvm.mlir.global internal @mb_shmem2() {{addr_space = 3 : i32}} : !llvm.array<1024 x f32>

    llvm.func @{local_name}(%input_desc: !llvm.ptr, %output_desc: !llvm.ptr) attributes {{gpu.kernel, nvvm.kernel}} {{
{chr(10).join(k1_desc)}
      %sl_tid32 = nvvm.read.ptx.sreg.tid.x : i32
      %sl_tid = llvm.sext %sl_tid32 : i32 to i64
      %sl_bid32 = nvvm.read.ptx.sreg.ctaid.x : i32
      %sl_bid = llvm.sext %sl_bid32 : i32 to i64
      %sl_BS = llvm.mlir.constant({BS} : index) : i64
      %sl_N = llvm.mlir.constant({N} : index) : i64
      %sl_base = llvm.mul %sl_bid, %sl_BS  : i64
      %sl_gidx = llvm.add %sl_base, %sl_tid  : i64
      %sl_ok = llvm.icmp "ult" %sl_gidx, %sl_N : i64
      %sl_shmem = llvm.mlir.addressof @mb_shmem : !llvm.ptr<3>
      %sl_zf = llvm.mlir.constant(0.000000e+00 : f32) : f32
      llvm.cond_br %sl_ok, ^sl_load, ^sl_load_oob

    ^sl_load:
      %sl_si = llvm.add %sl_in_offset, %sl_gidx  : i64
      %sl_sp = llvm.getelementptr %sl_in_aligned[%sl_si] : (!llvm.ptr, i64) -> !llvm.ptr, f32
      %sl_val = llvm.load %sl_sp : !llvm.ptr -> f32
      %sl_mp = llvm.getelementptr %sl_shmem[0, %sl_tid] : (!llvm.ptr<3>, i64) -> !llvm.ptr<3>, !llvm.array<{BS} x f32>
      llvm.store %sl_val, %sl_mp : f32, !llvm.ptr<3>
      llvm.br ^sl_sync1

    ^sl_load_oob:
      %sl_mp2 = llvm.getelementptr %sl_shmem[0, %sl_tid] : (!llvm.ptr<3>, i64) -> !llvm.ptr<3>, !llvm.array<{BS} x f32>
      llvm.store %sl_zf, %sl_mp2 : f32, !llvm.ptr<3>
      llvm.br ^sl_sync1

    ^sl_sync1:
      nvvm.barrier0
      %sl_c0 = llvm.mlir.constant(0 : index) : i64
      %sl_c1 = llvm.mlir.constant(1 : index) : i64
      %sl_c2 = llvm.mlir.constant(2 : index) : i64
      %sl_maxd = llvm.mlir.constant({max_d_local} : index) : i64
      llvm.br ^sl_loop(%sl_c0, %sl_c1 : i64, i64)

    ^sl_loop(%sl_d: i64, %sl_stride: i64):
      %sl_ld = llvm.icmp "uge" %sl_d, %sl_maxd : i64
      llvm.cond_br %sl_ld, ^sl_write, ^sl_step

    ^sl_step:
      %sl_act = llvm.icmp "uge" %sl_tid, %sl_stride : i64
      %sl_pr = llvm.sub %sl_tid, %sl_stride  : i64
      %sl_sp2 = llvm.select %sl_act, %sl_pr, %sl_c0 : i1, i64
      %sl_pp = llvm.getelementptr %sl_shmem[0, %sl_sp2] : (!llvm.ptr<3>, i64) -> !llvm.ptr<3>, !llvm.array<{BS} x f32>
      %sl_pv = llvm.load %sl_pp : !llvm.ptr<3> -> f32
      %sl_tmp = llvm.select %sl_act, %sl_pv, %sl_zf : i1, f32
      nvvm.barrier0
      %sl_mp3 = llvm.getelementptr %sl_shmem[0, %sl_tid] : (!llvm.ptr<3>, i64) -> !llvm.ptr<3>, !llvm.array<{BS} x f32>
      %sl_cur = llvm.load %sl_mp3 : !llvm.ptr<3> -> f32
      %sl_new = llvm.fadd %sl_cur, %sl_tmp  : f32
      %sl_res = llvm.select %sl_act, %sl_new, %sl_cur : i1, f32
      llvm.store %sl_res, %sl_mp3 : f32, !llvm.ptr<3>
      nvvm.barrier0
      %sl_nd = llvm.add %sl_d, %sl_c1  : i64
      %sl_ns = llvm.mul %sl_stride, %sl_c2  : i64
      llvm.br ^sl_loop(%sl_nd, %sl_ns : i64, i64)

    ^sl_write:
      llvm.cond_br %sl_ok, ^sl_wb, ^sl_done

    ^sl_wb:
      %sl_mp4 = llvm.getelementptr %sl_shmem[0, %sl_tid] : (!llvm.ptr<3>, i64) -> !llvm.ptr<3>, !llvm.array<{BS} x f32>
      %sl_fv = llvm.load %sl_mp4 : !llvm.ptr<3> -> f32
      %sl_oi = llvm.add %sl_out_offset, %sl_gidx  : i64
      %sl_op = llvm.getelementptr %sl_out_aligned[%sl_oi] : (!llvm.ptr, i64) -> !llvm.ptr, f32
      llvm.store %sl_fv, %sl_op : f32, !llvm.ptr
      llvm.br ^sl_done

    ^sl_done:
      llvm.return
    }}

    llvm.func @{extract_name}(%scanned_desc: !llvm.ptr, %block_sums_desc: !llvm.ptr) attributes {{gpu.kernel, nvvm.kernel}} {{
{chr(10).join(k2_desc)}
      %se_tid32 = nvvm.read.ptx.sreg.tid.x : i32
      %se_tid = llvm.sext %se_tid32 : i32 to i64
      %se_NB = llvm.mlir.constant({num_blocks} : index) : i64
      %se_BS = llvm.mlir.constant({BS} : index) : i64
      %se_c1 = llvm.mlir.constant(1 : index) : i64
      %se_ok = llvm.icmp "ult" %se_tid, %se_NB : i64
      llvm.cond_br %se_ok, ^se_work, ^se_done

    ^se_work:
      %se_next = llvm.add %se_tid, %se_c1  : i64
      %se_end = llvm.mul %se_next, %se_BS  : i64
      %se_last = llvm.sub %se_end, %se_c1  : i64
      %se_N = llvm.mlir.constant({N} : index) : i64
      %se_clamp = llvm.icmp "ult" %se_last, %se_N : i64
      %se_safe = llvm.sub %se_N, %se_c1  : i64
      %se_idx = llvm.select %se_clamp, %se_last, %se_safe : i1, i64
      %se_si = llvm.add %se_sc_offset, %se_idx  : i64
      %se_sp = llvm.getelementptr %se_sc_aligned[%se_si] : (!llvm.ptr, i64) -> !llvm.ptr, f32
      %se_val = llvm.load %se_sp : !llvm.ptr -> f32
      %se_oi = llvm.add %se_bs_offset, %se_tid  : i64
      %se_op = llvm.getelementptr %se_bs_aligned[%se_oi] : (!llvm.ptr, i64) -> !llvm.ptr, f32
      llvm.store %se_val, %se_op : f32, !llvm.ptr
      llvm.br ^se_done

    ^se_done:
      llvm.return
    }}

    llvm.func @{sums_name}(%sums_desc: !llvm.ptr, %prefix_desc: !llvm.ptr) attributes {{gpu.kernel, nvvm.kernel}} {{
{chr(10).join(k3_desc)}
      %ss_tid32 = nvvm.read.ptx.sreg.tid.x : i32
      %ss_tid = llvm.sext %ss_tid32 : i32 to i64
      %ss_NB = llvm.mlir.constant({num_blocks} : index) : i64
      %ss_ok = llvm.icmp "ult" %ss_tid, %ss_NB : i64
      %ss_shmem = llvm.mlir.addressof @mb_shmem2 : !llvm.ptr<3>
      %ss_zf = llvm.mlir.constant(0.000000e+00 : f32) : f32
      llvm.cond_br %ss_ok, ^ss_load, ^ss_load_oob

    ^ss_load:
      %ss_si = llvm.add %ss_in_offset, %ss_tid  : i64
      %ss_sp = llvm.getelementptr %ss_in_aligned[%ss_si] : (!llvm.ptr, i64) -> !llvm.ptr, f32
      %ss_val = llvm.load %ss_sp : !llvm.ptr -> f32
      %ss_mp = llvm.getelementptr %ss_shmem[0, %ss_tid] : (!llvm.ptr<3>, i64) -> !llvm.ptr<3>, !llvm.array<1024 x f32>
      llvm.store %ss_val, %ss_mp : f32, !llvm.ptr<3>
      llvm.br ^ss_sync

    ^ss_load_oob:
      %ss_mp2 = llvm.getelementptr %ss_shmem[0, %ss_tid] : (!llvm.ptr<3>, i64) -> !llvm.ptr<3>, !llvm.array<1024 x f32>
      llvm.store %ss_zf, %ss_mp2 : f32, !llvm.ptr<3>
      llvm.br ^ss_sync

    ^ss_sync:
      nvvm.barrier0
      %ss_c0 = llvm.mlir.constant(0 : index) : i64
      %ss_c1 = llvm.mlir.constant(1 : index) : i64
      %ss_c2 = llvm.mlir.constant(2 : index) : i64
      %ss_maxd = llvm.mlir.constant({max_d_sums} : index) : i64
      llvm.br ^ss_loop(%ss_c0, %ss_c1 : i64, i64)

    ^ss_loop(%ss_d: i64, %ss_stride: i64):
      %ss_ld = llvm.icmp "uge" %ss_d, %ss_maxd : i64
      llvm.cond_br %ss_ld, ^ss_write, ^ss_step

    ^ss_step:
      %ss_act = llvm.icmp "uge" %ss_tid, %ss_stride : i64
      %ss_pr = llvm.sub %ss_tid, %ss_stride  : i64
      %ss_sp2 = llvm.select %ss_act, %ss_pr, %ss_c0 : i1, i64
      %ss_pp = llvm.getelementptr %ss_shmem[0, %ss_sp2] : (!llvm.ptr<3>, i64) -> !llvm.ptr<3>, !llvm.array<1024 x f32>
      %ss_pv = llvm.load %ss_pp : !llvm.ptr<3> -> f32
      %ss_tmp = llvm.select %ss_act, %ss_pv, %ss_zf : i1, f32
      nvvm.barrier0
      %ss_mp3 = llvm.getelementptr %ss_shmem[0, %ss_tid] : (!llvm.ptr<3>, i64) -> !llvm.ptr<3>, !llvm.array<1024 x f32>
      %ss_cur = llvm.load %ss_mp3 : !llvm.ptr<3> -> f32
      %ss_new = llvm.fadd %ss_cur, %ss_tmp  : f32
      %ss_res = llvm.select %ss_act, %ss_new, %ss_cur : i1, f32
      llvm.store %ss_res, %ss_mp3 : f32, !llvm.ptr<3>
      nvvm.barrier0
      %ss_nd = llvm.add %ss_d, %ss_c1  : i64
      %ss_ns = llvm.mul %ss_stride, %ss_c2  : i64
      llvm.br ^ss_loop(%ss_nd, %ss_ns : i64, i64)

    ^ss_write:
      llvm.cond_br %ss_ok, ^ss_wb, ^ss_done

    ^ss_wb:
      %ss_mp4 = llvm.getelementptr %ss_shmem[0, %ss_tid] : (!llvm.ptr<3>, i64) -> !llvm.ptr<3>, !llvm.array<1024 x f32>
      %ss_fv = llvm.load %ss_mp4 : !llvm.ptr<3> -> f32
      %ss_oi = llvm.add %ss_out_offset, %ss_tid  : i64
      %ss_op = llvm.getelementptr %ss_out_aligned[%ss_oi] : (!llvm.ptr, i64) -> !llvm.ptr, f32
      llvm.store %ss_fv, %ss_op : f32, !llvm.ptr
      llvm.br ^ss_done

    ^ss_done:
      llvm.return
    }}

    llvm.func @{propagate_name}(%prefix_desc: !llvm.ptr, %scanned_desc: !llvm.ptr) attributes {{gpu.kernel, nvvm.kernel}} {{
{chr(10).join(k4_desc)}
      %sp_tid32 = nvvm.read.ptx.sreg.tid.x : i32
      %sp_tid = llvm.sext %sp_tid32 : i32 to i64
      %sp_bid32 = nvvm.read.ptx.sreg.ctaid.x : i32
      %sp_bid = llvm.sext %sp_bid32 : i32 to i64
      %sp_c0 = llvm.mlir.constant(0 : index) : i64
      %sp_is_b0 = llvm.icmp "eq" %sp_bid, %sp_c0 : i64
      llvm.cond_br %sp_is_b0, ^sp_done, ^sp_work

    ^sp_work:
      %sp_BS = llvm.mlir.constant({BS} : index) : i64
      %sp_N = llvm.mlir.constant({N} : index) : i64
      %sp_c1 = llvm.mlir.constant(1 : index) : i64
      %sp_prev = llvm.sub %sp_bid, %sp_c1  : i64
      %sp_pi = llvm.add %sp_pr_offset, %sp_prev  : i64
      %sp_pp = llvm.getelementptr %sp_pr_aligned[%sp_pi] : (!llvm.ptr, i64) -> !llvm.ptr, f32
      %sp_prefix = llvm.load %sp_pp : !llvm.ptr -> f32
      %sp_base = llvm.mul %sp_bid, %sp_BS  : i64
      %sp_gidx = llvm.add %sp_base, %sp_tid  : i64
      %sp_ok = llvm.icmp "ult" %sp_gidx, %sp_N : i64
      llvm.cond_br %sp_ok, ^sp_add, ^sp_done

    ^sp_add:
      %sp_si = llvm.add %sp_sc_offset, %sp_gidx  : i64
      %sp_sp = llvm.getelementptr %sp_sc_aligned[%sp_si] : (!llvm.ptr, i64) -> !llvm.ptr, f32
      %sp_val = llvm.load %sp_sp : !llvm.ptr -> f32
      %sp_new = llvm.fadd %sp_val, %sp_prefix  : f32
      llvm.store %sp_new, %sp_sp : f32, !llvm.ptr
      llvm.br ^sp_done

    ^sp_done:
      llvm.return
    }}
  }}
}}"""
    return GPUModuleScaffold(text, module_name, propagate_name)


def _validate_scaffold_names(module_name: str, kernel_name: str) -> None:
    """Raise GPUScaffoldError if module or kernel name is not a valid identifier."""
    if not module_name.isidentifier() or not kernel_name.isidentifier():
        raise GPUScaffoldError("GPU scaffold names must be valid identifiers")


def _build_f32_map_gpu_scaffold(
    kernel: F32MapKernel,
    *,
    module_name: str,
    kernel_name: str,
) -> GPUModuleScaffold:
    """Assemble an MLIR gpu.module scaffold string for an f32 map kernel."""
    _validate_scaffold_names(module_name, kernel_name)
    shape = _validate_shape(kernel.shape)
    memref_type = _memref_type(shape)
    total_size = _product(shape)
    indexing_lines, indices = _indexing_lines(shape)
    input_params = ", ".join(
        [f"%input{index}: {memref_type}" for index in range(kernel.num_inputs)]
    )
    operation_lines = _operation_lines(kernel, memref_type, indices)
    text = f"""module {{
  gpu.module @{module_name} {{
    gpu.func @{kernel_name}({input_params}, %output: {memref_type}) kernel {{
      %tid = gpu.thread_id x
      %bid = gpu.block_id x
      %bdim = gpu.block_dim x
      %block_base = arith.muli %bid, %bdim : index
      %idx = arith.addi %block_base, %tid : index
      %size = arith.constant {total_size} : index
      %inside = arith.cmpi ult, %idx, %size : index
      scf.if %inside {{
{indexing_lines}
{operation_lines}
      }}
      gpu.return
    }}
  }}
}}"""
    return GPUModuleScaffold(text, module_name, kernel_name)


def _validate_shape(shape: tuple[int, ...]) -> tuple[int, ...]:
    """Validate rank 1-10 with non-negative dimensions; return normalized int tuple."""
    if not 1 <= len(shape) <= 10:
        raise GPUScaffoldError("GPU scaffold currently supports rank-1 through rank-10 shapes only")
    if any(dim < 0 for dim in shape):
        raise GPUScaffoldError("GPU scaffold shape dimensions must be non-negative")
    return tuple(int(dim) for dim in shape)


def _memref_type(shape: tuple[int, ...]) -> str:
    """Return a memref<...xf32> type string for the given shape."""
    return f"memref<{'x'.join(str(dim) for dim in shape)}xf32>"


def _product(shape: tuple[int, ...]) -> int:
    """Return the product of all dimensions in the shape."""
    total = 1
    for dim in shape:
        total *= dim
    return total


def _indexing_lines(shape: tuple[int, ...]) -> tuple[str, list[str]]:
    """Return MLIR lines and index variable names for multi-dimensional indexing.

    Handles rank 1–10 by decomposing a flat index into row-major coordinates
    via ``arith.divui`` / ``arith.remui``.
    """
    rank = len(shape)
    if rank == 1:
        return "", ["%idx"]
    if rank == 2:
        return (
            "\n".join([
                f"        %dim1 = arith.constant {shape[1]} : index",
                "        %i0 = arith.divui %idx, %dim1 : index",
                "        %i1 = arith.remui %idx, %dim1 : index",
            ]),
            ["%i0", "%i1"],
        )
    # General case: rank >= 3
    # plane[k] = product of dimensions k+1 .. rank-1
    lines: list[str] = []
    for axis in range(1, rank):
        plane = 1
        for d in shape[axis:]:
            plane *= d
        lines.append(
            f"        %plane{axis - 1} = arith.constant {plane} : index"
        )
    current = "%idx"
    for axis in range(rank - 1):
        lines.append(
            f"        %i{axis} = arith.divui {current}, %plane{axis} : index"
        )
        lines.append(
            f"        %rem{axis} = arith.remui {current}, %plane{axis} : index"
        )
        current = f"%rem{axis}"
    # Last axis: use the final remainder directly
    indices = [f"%i{axis}" for axis in range(rank - 1)] + [current]
    return "\n".join(lines), indices


def _operation_lines(kernel: F32MapKernel, memref_type: str, indices: list[str]) -> str:
    """Return MLIR lines loading inputs, applying the map op, and storing the result."""
    index_text = ", ".join(indices)
    lines = [f"        %x0 = memref.load %input0[{index_text}] : {memref_type}"]
    if kernel.expression is not None:
        for input_index in range(1, kernel.num_inputs):
            lines.append(
                f"        %x{input_index} = memref.load %input{input_index}"
                f"[{index_text}] : {memref_type}"
            )
        expression_lines, result = _f32_expression_lines(kernel.expression, "arith", "        ")
        lines.extend(expression_lines)
        lines.append(f"        %y = arith.addf {result}, %zero_expr : f32")
        lines.insert(len(lines) - 1, "        %zero_expr = arith.constant 0.000000e+00 : f32")
    elif kernel.num_inputs == 2:
        lines.append(f"        %x1 = memref.load %input1[{index_text}] : {memref_type}")
        lines.append(f"        %y = {_binary_op_expr(kernel.operation)}")
    else:
        assert kernel.operation.constant is not None
        lines.append(f"        %c = arith.constant {kernel.operation.constant:.6e} : f32")
        lines.append(f"        %y = {_unary_op_expr(kernel.operation)}")
    lines.append(f"        memref.store %y, %output[{index_text}] : {memref_type}")
    return "\n".join(lines)


def _unary_op_expr(operation: F32MapOperation) -> str:
    """Return the MLIR arith expression for a unary f32 map operation."""
    left = "%x0"
    right = "%c"
    if operation.constant_side == "left":
        left, right = right, left
    if operation.op not in {"*", "+", "-", "/"}:
        raise GPUScaffoldError(f"GPU scaffold does not support operator {operation.op}")
    mlir_op = arith_op(operation.op, "f32")
    return f"{mlir_op} {left}, {right} : f32"


def _binary_op_expr(operation: F32MapOperation) -> str:
    """Return the MLIR arith expression for a binary f32 map operation."""
    if operation.op not in {"*", "+", "-", "/"}:
        raise GPUScaffoldError(f"GPU scaffold does not support operator {operation.op}")
    mlir_op = arith_op(operation.op, "f32")
    return f"{mlir_op} %x0, %x1 : f32"


def build_descriptor_abi_bool_map_gpu_module(
    function: HIRFunction,
    *,
    module_name: str = "remora_gpu",
    kernel_name: str | None = None,
) -> GPUModuleScaffold:
    """Build an executable descriptor-ABI GPU module for a supported bool map."""
    from remora._gpu_map_support import analyze_supported_bool_map_function
    kernel = analyze_supported_bool_map_function(
        function,
        on_unsupported=GPUScaffoldError,
        context="descriptor ABI GPU module",
    )
    name = kernel_name or f"remora_{function.name}_bool"
    _validate_scaffold_names(module_name, name)
    
    shape = _validate_shape(kernel.shape)
    params = [
        *(f"%input{index}_desc: !llvm.ptr" for index in range(kernel.num_inputs)),
        "%output_desc: !llvm.ptr",
    ]
    body_lines = _descriptor_kernel_body_lines(
        kernel,
        element_type="i8",
        operation_lines=_descriptor_bool_operation_lines,
    )
    text = f"""module {{
  gpu.module @{module_name} {{
    llvm.func @{name}({", ".join(params)}) attributes {{gpu.kernel, nvvm.kernel}} {{
{chr(10).join(body_lines)}
      llvm.return
    }}
  }}
}}"""
    return GPUModuleScaffold(text, module_name, name)


def _descriptor_bool_operation_lines(kernel: I32MapKernel) -> list[str]:
    """Return LLVM IR lines for bool-valued map operations via the descriptor ABI."""
    lines: list[str] = []
    # inputs are i8, cast to i1
    for i in range(kernel.num_inputs):
        lines.append(f"      %x{i}_i1 = llvm.trunc %x{i} : i8 to i1")
    
    if kernel.num_inputs == 2:
        res_i1 = _descriptor_bool_binary_op_expr(kernel.operation)
        lines.append(f"      %y_i1 = {res_i1}")
    else:
        assert kernel.operation.constant is not None
        c_val = "1" if kernel.operation.constant else "0"
        lines.append(f"      %c_i1 = llvm.mlir.constant({c_val} : i1) : i1")
        res_i1 = _descriptor_bool_unary_op_expr(kernel.operation)
        lines.append(f"      %y_i1 = {res_i1}")
    
    # cast result back to i8
    lines.append("      %y = llvm.zext %y_i1 : i1 to i8")
    return lines


def _descriptor_bool_unary_op_expr(operation: I32MapOperation) -> str:
    """Return the LLVM expression for a unary bool operation."""
    left = "%x0_i1"
    right = "%c_i1"
    if operation.constant_side == "left":
        left, right = right, left
    if operation.op not in {"&&", "||", "==", "!="}:
        raise GPUScaffoldError(f"descriptor ABI GPU module does not support operator {operation.op} for bool")
    mlir_op = llvm_op(operation.op, "i1")
    return f"{mlir_op} {left}, {right} : i1"


def _descriptor_bool_binary_op_expr(operation: I32MapOperation) -> str:
    """Return the LLVM expression for a binary bool operation."""
    if operation.op not in {"&&", "||", "==", "!="}:
        raise GPUScaffoldError(f"descriptor ABI GPU module does not support operator {operation.op} for bool")
    mlir_op = llvm_op(operation.op, "i1")
    return f"{mlir_op} %x0_i1, %x1_i1 : i1"


def _f32_map_kernel(function: HIRFunction) -> F32MapKernel:
    """Analyze the HIR function and return an F32MapKernel or raise GPUScaffoldError."""
    return analyze_supported_f32_map_function(
        function,
        on_unsupported=GPUScaffoldError,
        context="GPU scaffold",
    )


def _i32_map_kernel(function: HIRFunction) -> I32MapKernel:
    """Analyze the HIR function and return an I32MapKernel or raise GPUScaffoldError."""
    return analyze_supported_i32_map_function(
        function,
        on_unsupported=GPUScaffoldError,
        context="descriptor ABI GPU module",
    )


def _build_descriptor_abi_f32_map_gpu_module(
    kernel: F32MapKernel,
    *,
    module_name: str,
    kernel_name: str,
) -> GPUModuleScaffold:
    """Assemble a descriptor-ABI GPU module scaffold for an f32 map kernel."""
    shape = _validate_shape(kernel.shape)
    rank = len(shape)
    params = [
        *(f"%input{index}_desc: !llvm.ptr" for index in range(kernel.num_inputs)),
        *(f"%scalar{index}: f32" for index in range(kernel.scalar_count)),
        "%output_desc: !llvm.ptr",
    ]
    body_lines = _descriptor_kernel_body_lines(kernel)
    text = f"""module {{
  gpu.module @{module_name} {{
    llvm.func @{kernel_name}({", ".join(params)}) attributes {{gpu.kernel, nvvm.kernel}} {{
{chr(10).join(body_lines)}
      llvm.return
    }}
  }}
}}"""
    return GPUModuleScaffold(text, module_name, kernel_name)


def _build_descriptor_abi_i32_map_gpu_module(
    kernel: I32MapKernel,
    *,
    module_name: str,
    kernel_name: str,
) -> GPUModuleScaffold:
    """Assemble a descriptor-ABI GPU module scaffold for an i32 map kernel."""
    shape = _validate_shape(kernel.shape)
    params = [
        *(f"%input{index}_desc: !llvm.ptr" for index in range(kernel.num_inputs)),
        "%output_desc: !llvm.ptr",
    ]
    body_lines = _descriptor_kernel_body_lines(
        kernel,
        element_type="i32",
        operation_lines=_descriptor_i32_operation_lines,
    )
    text = f"""module {{
  gpu.module @{module_name} {{
    llvm.func @{kernel_name}({", ".join(params)}) attributes {{gpu.kernel, nvvm.kernel}} {{
{chr(10).join(body_lines)}
      llvm.return
    }}
  }}
}}"""
    return GPUModuleScaffold(text, module_name, kernel_name)


def _descriptor_kernel_body_lines(
    kernel: F32MapKernel | I32MapKernel,
    *,
    element_type: str = "f32",
    operation_lines: Callable[[F32MapKernel | I32MapKernel], list[str]] | None = None,
) -> list[str]:
    """Return MLIR body lines for a descriptor-ABI kernel (descriptor loads, thread indexing, map op)."""
    rank = len(kernel.shape)
    operation_builder = _descriptor_operation_lines if operation_lines is None else operation_lines
    descriptor_names = [f"%input{index}_desc" for index in range(kernel.num_inputs)] + [
        "%output_desc"
    ]
    prefixes = [f"in{index}" for index in range(kernel.num_inputs)] + ["out"]
    lines: list[str] = []
    for prefix, descriptor_name in zip(prefixes, descriptor_names):
        lines.extend(_descriptor_load_lines(prefix, descriptor_name, rank))

    lines.extend(
        [
            "      %tid32 = nvvm.read.ptx.sreg.tid.x : i32",
            "      %tid = llvm.sext %tid32 : i32 to i64",
            "      %bid32 = nvvm.read.ptx.sreg.ctaid.x : i32",
            "      %bid = llvm.sext %bid32 : i32 to i64",
            "      %bdim32 = nvvm.read.ptx.sreg.ntid.x : i32",
            "      %bdim = llvm.sext %bdim32 : i32 to i64",
            "      %block_base = llvm.mul %bid, %bdim  : i64",
            "      %idx = llvm.add %block_base, %tid  : i64",
        ]
    )

    # Compute planes in reverse: plane[rank-1] = 1, plane[rank-2] = size[rank-1], etc.
    # plane[k] is the size of the trailing sub-array starting at axis k+1.
    lines.append("      %plane_last = llvm.mlir.constant(1 : index) : i64")
    prev_plane = "%plane_last"
    for axis in range(rank - 1, 0, -1):
        plane_name = f"%plane{axis - 1}"
        lines.append(f"      {plane_name} = llvm.mul {prev_plane}, %out_size{axis}  : i64")
        prev_plane = plane_name
    
    total_name = f"%total_size"
    lines.append(f"      {total_name} = llvm.mul {prev_plane}, %out_size0 : i64")

    lines.extend(
        [
            f"      %inside = llvm.icmp \"ult\" %idx, {total_name} : i64",
            "      llvm.cond_br %inside, ^bb1, ^bb2",
            "    ^bb1:",
        ]
    )
    lines.extend(_multi_index_lines(rank))
    for prefix in prefixes:
        lines.extend(_linear_index_lines(prefix, rank))
    for index in range(kernel.num_inputs):
        pf = prefixes[index]
        sub = (
            getattr(kernel, 'subarray_offsets', None)[index]
            if getattr(kernel, 'subarray_offsets', None) and index < len(getattr(kernel, 'subarray_offsets', None) or ())
            else None
        )
        if sub is not None and rank == 2:
            orow, ocol = sub
            radj = f"%{pf}_radj_{index}"
            cadj = f"%{pf}_cadj_{index}"
            cor = f"%{pf}_cor_{index}"
            coc = f"%{pf}_coc_{index}"
            linadj = f"%{pf}_linadj_{index}"
            lines.extend([
                f"      {cor} = llvm.mlir.constant({orow} : index) : i64",
                f"      {coc} = llvm.mlir.constant({ocol} : index) : i64",
                f"      {radj} = llvm.add %i0, {cor} : i64",
                f"      {cadj} = llvm.add %i1, {coc} : i64",
                f"      %_{pf}_tadj_{index} = llvm.mul {radj}, %{pf}_stride0 : i64",
                f"      %_{pf}_roffadj_{index} = llvm.add %{pf}_offset, %_{pf}_tadj_{index} : i64",
                f"      {linadj} = llvm.add %_{pf}_roffadj_{index}, {cadj} : i64",
                f"      %{pf}_elem_ptr = llvm.getelementptr %{pf}_aligned[{linadj}] : (!llvm.ptr, i64) -> !llvm.ptr, {element_type}",
            ])
        else:
            lines.extend(
                [
                    f"      %{pf}_elem_ptr = llvm.getelementptr %{pf}_aligned[%{pf}_linear] : (!llvm.ptr, i64) -> !llvm.ptr, {element_type}",
                ]
            )
        lines.append(
            f"      %x{index} = llvm.load %{pf}_elem_ptr : !llvm.ptr -> {element_type}"
        )
    lines.extend(operation_builder(kernel))
    lines.extend(
        [
            f"      %out_elem_ptr = llvm.getelementptr %out_aligned[%out_linear] : (!llvm.ptr, i64) -> !llvm.ptr, {element_type}",
            f"      llvm.store %y, %out_elem_ptr : {element_type}, !llvm.ptr",
            "      llvm.br ^bb2",
            "    ^bb2:",
        ]
    )
    return lines


def _descriptor_type(rank: int) -> str:
    """Return the LLVM struct type string for a cuda data descriptor of given rank."""
    fields = ["ptr", "ptr", "i64", *(["i64"] * rank), *(["i64"] * rank)]
    return f"!llvm.struct<({', '.join(fields)})>"


def _descriptor_load_lines(prefix: str, descriptor_name: str, rank: int) -> list[str]:
    """Return MLIR lines that load aligned pointer, offset, sizes, and strides from a descriptor."""
    descriptor_type = _descriptor_type(rank)
    lines = [
        f"      %{prefix}_aligned_ptr = llvm.getelementptr {descriptor_name}[0, 1] : (!llvm.ptr) -> !llvm.ptr, {descriptor_type}",
        f"      %{prefix}_offset_ptr = llvm.getelementptr {descriptor_name}[0, 2] : (!llvm.ptr) -> !llvm.ptr, {descriptor_type}",
        f"      %{prefix}_aligned = llvm.load %{prefix}_aligned_ptr : !llvm.ptr -> !llvm.ptr",
        f"      %{prefix}_offset = llvm.load %{prefix}_offset_ptr : !llvm.ptr -> i64",
    ]
    for axis in range(rank):
        field_index = 3 + axis
        lines.extend(
            [
                f"      %{prefix}_size{axis}_ptr = llvm.getelementptr {descriptor_name}[0, {field_index}] : (!llvm.ptr) -> !llvm.ptr, {descriptor_type}",
                f"      %{prefix}_size{axis} = llvm.load %{prefix}_size{axis}_ptr : !llvm.ptr -> i64",
            ]
        )
    for axis in range(rank):
        field_index = 3 + rank + axis
        lines.extend(
            [
                f"      %{prefix}_stride{axis}_ptr = llvm.getelementptr {descriptor_name}[0, {field_index}] : (!llvm.ptr) -> !llvm.ptr, {descriptor_type}",
                f"      %{prefix}_stride{axis} = llvm.load %{prefix}_stride{axis}_ptr : !llvm.ptr -> i64",
            ]
        )
    return lines


def _multi_index_lines(rank: int) -> list[str]:
    """Return MLIR lines computing multi-dimensional indices from a flat thread index."""
    lines: list[str] = []
    current_rem = "%idx"
    for axis in range(rank - 1):
        plane_name = f"%plane{axis}"
        lines.extend([
            f"      %i{axis} = llvm.udiv {current_rem}, {plane_name}  : i64",
            f"      %rem{axis} = llvm.urem {current_rem}, {plane_name}  : i64",
        ])
        current_rem = f"%rem{axis}"
    
    lines.append("      %index_zero = llvm.mlir.constant(0 : index) : i64")
    lines.append(f"      %i{rank - 1} = llvm.add {current_rem}, %index_zero  : i64")

    return lines


def _linear_index_lines(prefix: str, rank: int) -> list[str]:
    """Return MLIR lines computing a linear index from multi-dimensional indices and strides."""
    lines = [
        f"      %{prefix}_term0 = llvm.mul %i0, %{prefix}_stride0  : i64",
        f"      %{prefix}_linear0 = llvm.add %{prefix}_offset, %{prefix}_term0  : i64",
    ]
    previous = f"%{prefix}_linear0"
    for axis in range(1, rank):
        lines.extend(
            [
                f"      %{prefix}_term{axis} = llvm.mul %i{axis}, %{prefix}_stride{axis}  : i64",
                f"      %{prefix}_linear{axis} = llvm.add {previous}, %{prefix}_term{axis}  : i64",
            ]
        )
        previous = f"%{prefix}_linear{axis}"
    lines.append(f"      %{prefix}_linear_zero = llvm.mlir.constant(0 : index) : i64")
    lines.append(f"      %{prefix}_linear = llvm.add {previous}, %{prefix}_linear_zero  : i64")
    return lines


def _descriptor_operation_lines(kernel: F32MapKernel) -> list[str]:
    """Return LLVM IR lines for f32 map operations using the descriptor ABI."""
    if kernel.expression is not None:
        lines, result = _f32_expression_lines(kernel.expression, "llvm", "      ")
        lines.append("      %zero_expr = llvm.mlir.constant(0.000000e+00 : f32) : f32")
        lines.append(f"      %y = llvm.fadd {result}, %zero_expr : f32")
        return lines
    if kernel.num_inputs == 2:
        return [f"      %y = {_descriptor_binary_op_expr(kernel.operation)}"]
    assert kernel.operation.constant is not None
    lines = [f"      %c = llvm.mlir.constant({kernel.operation.constant:.6e} : f32) : f32"]
    lines.append(f"      %y = {_descriptor_unary_op_expr(kernel.operation)}")
    return lines


def _descriptor_unary_op_expr(operation: F32MapOperation) -> str:
    """Return the LLVM expression for a unary f32 operation."""
    left = "%x0"
    right = "%c"
    if operation.constant_side == "left":
        left, right = right, left
    if operation.op not in {"*", "+", "-", "/"}:
        raise GPUScaffoldError(f"descriptor ABI GPU module does not support operator {operation.op}")
    mlir_op = llvm_op(operation.op, "f32")
    return f"{mlir_op} {left}, {right}  : f32"


def _descriptor_binary_op_expr(operation: F32MapOperation) -> str:
    """Return the LLVM expression for a binary f32 operation."""
    if operation.op not in {"*", "+", "-", "/"}:
        raise GPUScaffoldError(f"descriptor ABI GPU module does not support operator {operation.op}")
    mlir_op = llvm_op(operation.op, "f32")
    return f"{mlir_op} %x0, %x1  : f32"


def _f32_expression_lines(
    expression: F32Expr,
    dialect: str,
    indent: str,
) -> tuple[list[str], str]:
    lines: list[str] = []
    counter = 0

    def emit(expr: F32Expr) -> str:
        nonlocal counter
        if isinstance(expr, F32InputExpr):
            return f"%x{expr.index}"
        if isinstance(expr, F32ScalarParamExpr):
            return f"%scalar{expr.index}"
        name = f"%expr{counter}"
        counter += 1
        if isinstance(expr, F32ConstantExpr):
            if dialect == "llvm":
                lines.append(
                    f"{indent}{name} = llvm.mlir.constant({expr.value:.6e} : f32) : f32"
                )
            else:
                lines.append(f"{indent}{name} = arith.constant {expr.value:.6e} : f32")
            return name
        assert isinstance(expr, (F32BinaryExpr, F32SelectExpr, F32CmpExpr))
        if isinstance(expr, F32CmpExpr):
            # Emit comparison, then convert i1 → f32 for branchless select
            left = emit(expr.left)
            right = emit(expr.right)
            pred = {"<": "olt", "<=": "ole", ">": "ogt", ">=": "oge",
                    "==": "oeq", "!=": "one"}.get(expr.op, "ogt")
            i1_name = f"%cmp_i1_{counter}"
            counter += 1
            if dialect == "llvm":
                lines.append(f'{indent}{i1_name} = llvm.fcmp "{pred}" {left}, {right} : f32')
                # Convert i1 → i32 → f32
                i32_name = f"%cmp_i32_{counter}"
                counter += 1
                lines.append(f"{indent}{i32_name} = llvm.zext {i1_name} : i1 to i32")
                lines.append(f"{indent}{name} = llvm.sitofp {i32_name} : i32 to f32")
            else:
                lines.append(f"{indent}{i1_name} = arith.cmpf {pred}, {left}, {right} : f32")
                i32_name = f"%cmp_i32_{counter}"
                counter += 1
                lines.append(f"{indent}{i32_name} = arith.extui {i1_name} : i1 to i32")
                lines.append(f"{indent}{name} = arith.uitofp {i32_name} : i32 to f32")
            return name
        if isinstance(expr, F32SelectExpr):
            # Compute as: cond_f32 * then + (1 - cond_f32) * else
            cond_f = emit(expr.condition)  # already f32 from F32CmpExpr
            then_v = emit(expr.then_expr)
            else_v = emit(expr.else_expr)
            one_name = f"%one_{counter}"
            counter += 1
            neg_name = f"%neg_cond_{counter}"
            counter += 1
            w_then = f"%w_then_{counter}"
            counter += 1
            w_else = f"%w_else_{counter}"
            counter += 1
            if dialect == "llvm":
                op = llvm_op("*", "f32")
                add = llvm_op("+", "f32")
                sub = llvm_op("-", "f32")
                spacing = "  "
                lines.append(f'{indent}{one_name} = llvm.mlir.constant(1.000000e+00 : f32) : f32')
                lines.append(f'{indent}{neg_name} = {sub} {one_name}, {cond_f}  : f32')
                lines.append(f'{indent}{w_then} = {op} {cond_f}, {then_v}  : f32')
                lines.append(f'{indent}{w_else} = {op} {neg_name}, {else_v}  : f32')
                lines.append(f'{indent}{name} = {add} {w_then}, {w_else}  : f32')
            else:
                op = arith_op("*", "f32")
                add = arith_op("+", "f32")
                sub = arith_op("-", "f32")
                spacing = " "
                lines.append(f"{indent}{one_name} = arith.constant 1.000000e+00 : f32")
                lines.append(f"{indent}{neg_name} = {sub} {one_name}, {cond_f}{spacing}: f32")
                lines.append(f"{indent}{w_then} = {op} {cond_f}, {then_v}{spacing}: f32")
                lines.append(f"{indent}{w_else} = {op} {neg_name}, {else_v}{spacing}: f32")
                lines.append(f"{indent}{name} = {add} {w_then}, {w_else}{spacing}: f32")
            return name
        left = emit(expr.left)
        right = emit(expr.right)
        op = llvm_op(expr.op, "f32") if dialect == "llvm" else arith_op(expr.op, "f32")
        spacing = "  " if dialect == "llvm" else " "
        lines.append(f"{indent}{name} = {op} {left}, {right}{spacing}: f32")
        return name

    return lines, emit(expression)


def _descriptor_i32_operation_lines(kernel: I32MapKernel) -> list[str]:
    """Return LLVM IR lines for i32 map operations using the descriptor ABI."""
    if kernel.num_inputs == 2:
        return [f"      %y = {_descriptor_i32_binary_op_expr(kernel.operation)}"]
    assert kernel.operation.constant is not None
    lines = [f"      %c = llvm.mlir.constant({kernel.operation.constant} : i32) : i32"]
    lines.append(f"      %y = {_descriptor_i32_unary_op_expr(kernel.operation)}")
    return lines


def _descriptor_i32_unary_op_expr(operation: I32MapOperation) -> str:
    """Return the LLVM expression for a unary i32 operation."""
    left = "%x0"
    right = "%c"
    if operation.constant_side == "left":
        left, right = right, left
    if operation.op not in {"*", "+", "-", "/"}:
        raise GPUScaffoldError(f"descriptor ABI GPU module does not support operator {operation.op}")
    mlir_op = llvm_op(operation.op, "i32")
    return f"{mlir_op} {left}, {right}  : i32"


def _descriptor_i32_binary_op_expr(operation: I32MapOperation) -> str:
    """Return the LLVM expression for a binary i32 operation."""
    if operation.op not in {"*", "+", "-", "/"}:
        raise GPUScaffoldError(f"descriptor ABI GPU module does not support operator {operation.op}")
    mlir_op = llvm_op(operation.op, "i32")
    return f"{mlir_op} %x0, %x1  : i32"


@dataclass(frozen=True)
class F32ReductionKernel:
    shape: tuple[int, ...]
    fold_op: str
    init: float
    input_op: str | None = None
    num_inputs: int = 1

    @property
    def size(self) -> int:
        """Total number of elements (product of shape)."""
        total = 1
        for d in self.shape:
            total *= d
        return total

    @property
    def rank(self) -> int:
        return len(self.shape)


def _f32_reduction_kernel(function: HIRFunction) -> F32ReductionKernel:
    """Analyze a HIR fold/map+fold function and return an F32ReductionKernel or raise."""
    if not isinstance(function.body, HIRFold):
        raise GPUScaffoldError("descriptor ABI GPU reduction currently supports fold bodies only")
    if function.return_type != FLOAT:
        raise GPUScaffoldError("descriptor ABI GPU reduction currently supports float scalar outputs only")
    if not (
        isinstance(function.body.func, HIRPrimCallable)
        and function.body.func.left_arg is None
        and function.body.func.right_arg is None
        and function.body.func.result_type == FLOAT
    ):
        raise GPUScaffoldError("descriptor ABI GPU reduction currently supports primitive float fold callables only")
    if function.body.func.op not in {"+", "*"}:
        raise GPUScaffoldError("descriptor ABI GPU reduction currently supports + and * folds only")
    if not isinstance(function.body.init, HIRLit) or function.body.init.type != FLOAT:
        raise GPUScaffoldError("descriptor ABI GPU reduction currently requires a literal float initializer")

    if isinstance(function.body.array, HIRVar):
        if len(function.params) != 1 or function.body.array.name != function.params[0].name:
            raise GPUScaffoldError("descriptor ABI GPU reduction input must be the function parameter")
        param_type = _require_rank1_f32_param(function.params[0].type)
        shape = tuple(int(d.value) for d in param_type.shape)
        return F32ReductionKernel(shape, function.body.func.op, float(function.body.init.value))

    if isinstance(function.body.array, HIRMap):
        mapped = function.body.array
        if len(function.params) != 2:
            raise GPUScaffoldError("descriptor ABI GPU dot reduction requires two input parameters")
        if not (
            len(mapped.arrays) == 2
            and all(isinstance(array, HIRVar) for array in mapped.arrays)
            and [array.name for array in mapped.arrays] == [param.name for param in function.params]
            and isinstance(mapped.func, HIRPrimCallable)
            and mapped.func.left_arg is None
            and mapped.func.right_arg is None
            and mapped.func.result_type == FLOAT
        ):
            raise GPUScaffoldError("descriptor ABI GPU dot reduction requires a primitive binary map over parameters")
        if mapped.func.op not in {"*", "+", "-", "/"}:
            raise GPUScaffoldError("descriptor ABI GPU dot reduction map operator is not supported")
        first_type = _require_rank1_f32_param(function.params[0].type)
        second_type = _require_rank1_f32_param(function.params[1].type)
        if first_type.shape != second_type.shape:
            raise GPUScaffoldError("descriptor ABI GPU dot reduction input shapes must match")
        shape = tuple(int(d.value) for d in first_type.shape)
        return F32ReductionKernel(
            shape,
            function.body.func.op,
            float(function.body.init.value),
            input_op=mapped.func.op,
            num_inputs=2,
        )

    raise GPUScaffoldError("descriptor ABI GPU reduction input must be a parameter or binary map over parameters")


def _require_rank1_f32_param(param_type: object) -> ArrayType:
    """Deprecated: historically required rank-1; now accepts any-rank float arrays."""
    if not (
        isinstance(param_type, ArrayType)
        and param_type.element == FLOAT
    ):
        raise GPUScaffoldError(
            "descriptor ABI GPU reduction currently supports float inputs only"
        )
    return param_type


def _build_descriptor_abi_f32_reduction_gpu_module(
    kernel: F32ReductionKernel,
    *,
    module_name: str,
    kernel_name: str,
) -> GPUModuleScaffold:
    """Assemble a descriptor-ABI GPU module scaffold for an f32 reduction kernel."""
    params = [
        *(f"%input{index}_desc: !llvm.ptr" for index in range(kernel.num_inputs)),
        "%output_desc: !llvm.ptr",
    ]
    body_lines = _reduction_kernel_body_lines(kernel)
    text = f"""module {{
  gpu.module @{module_name} {{
    llvm.mlir.global internal @shmem() {{addr_space = 3 : i32}} : !llvm.array<256 x f32>
    llvm.func @{kernel_name}({", ".join(params)}) attributes {{gpu.kernel, nvvm.kernel}} {{
{chr(10).join(body_lines)}
      llvm.return
    }}
  }}
}}"""
    return GPUModuleScaffold(text, module_name, kernel_name)


def _reduction_kernel_body_lines(kernel: F32ReductionKernel) -> list[str]:
    """Return MLIR body lines for a grid-strided reduction kernel with shared-memory tree reduce."""
    rank = kernel.rank
    prefixes = [f"in{index}" for index in range(kernel.num_inputs)]
    lines: list[str] = []
    for index, prefix in enumerate(prefixes):
        lines.extend(_descriptor_load_lines(prefix, f"%input{index}_desc", rank))
    lines.extend(_descriptor_load_lines("out", "%output_desc", 0))

    # Compute total size for bounds check (product of all dims)
    total_size = kernel.size
    lines.extend(
        [
            "      %tid32 = nvvm.read.ptx.sreg.tid.x : i32",
            "      %tid = llvm.sext %tid32 : i32 to i64",
            "      %bid32 = nvvm.read.ptx.sreg.ctaid.x : i32",
            "      %bid = llvm.sext %bid32 : i32 to i64",
            "      %bdim32 = nvvm.read.ptx.sreg.ntid.x : i32",
            "      %bdim = llvm.sext %bdim32 : i32 to i64",
            "      %gdim32 = nvvm.read.ptx.sreg.nctaid.x : i32",
            "      %gdim = llvm.sext %gdim32 : i32 to i64",
            "      %grid_stride = llvm.mul %bdim, %gdim : i64",
            "      %block_offset = llvm.mul %bid, %bdim : i64",
            "      %start_idx = llvm.add %tid, %block_offset : i64",
            f"      %init = llvm.mlir.constant({kernel.init:.6e} : f32) : f32",
            f"      %total = llvm.mlir.constant({total_size} : index) : i64",
            "      llvm.br ^bb_loop(%start_idx, %init : i64, f32)",
            "    ^bb_loop(%i: i64, %current_acc: f32):",
            "      %is_inside_loop = llvm.icmp \"ult\" %i, %total : i64",
            "      llvm.cond_br %is_inside_loop, ^bb_body, ^bb_reduce",
            "    ^bb_body:",
        ]
    )

    if rank > 1:
        lines.append("      %zero = llvm.mlir.constant(0 : index) : i64")
        lines.append("      %idx = llvm.add %i, %zero  : i64")
        lines.extend(_multi_index_lines(rank))

    for prefix in prefixes:
        if rank == 1:
            lines.extend([
                f"      %{prefix}_term = llvm.mul %i, %{prefix}_stride0  : i64",
                f"      %{prefix}_linear = llvm.add %{prefix}_offset, %{prefix}_term  : i64",
            ])
        else:
            lines.extend(_linear_index_lines(prefix, rank))
        lines.extend([
            f"      %{prefix}_elem_ptr = llvm.getelementptr %{prefix}_aligned[%{prefix}_linear] : (!llvm.ptr, i64) -> !llvm.ptr, f32",
            f"      %{prefix}_x = llvm.load %{prefix}_elem_ptr : !llvm.ptr -> f32",
        ])
    if kernel.num_inputs == 2:
        lines.append(f"      %item = {_reduction_binary_input_expr(kernel.input_op)}")
    else:
        lines.append("      %item = llvm.fadd %in0_x, %zero_f  : f32")
        lines.insert(-1, "      %zero_f = llvm.mlir.constant(0.000000e+00 : f32) : f32")
    
    # We use a trick for fold_op to handle %current_acc instead of %acc
    fold_expr = _reduction_fold_expr(kernel.fold_op).replace("%acc", "%current_acc")
    lines.extend(
        [
            f"      %next_acc = {fold_expr}",
            "      %next_i = llvm.add %i, %grid_stride : i64",
            "      llvm.br ^bb_loop(%next_i, %next_acc : i64, f32)",
            "    ^bb_reduce:",
            "      %shmem_ptr_uncasted = llvm.mlir.addressof @shmem : !llvm.ptr<3>",
            "      %shmem_ptr_mine = llvm.getelementptr %shmem_ptr_uncasted[0, %tid] : (!llvm.ptr<3>, i64) -> !llvm.ptr<3>, !llvm.array<256 x f32>",
            "      llvm.store %current_acc, %shmem_ptr_mine : f32, !llvm.ptr<3>",
            "      nvvm.barrier0",
        ]
    )
    
    # Tree reduction in shmem
    # For simplicity, we only support power-of-2 block sizes for the tree reduction logic here,
    # or we can just loop. 256 is power of 2.
    current_stride = 128
    while current_stride > 0:
        lines.extend([
            f"      %stride_{current_stride} = llvm.mlir.constant({current_stride} : i64) : i64",
            f"      %can_reduce_{current_stride} = llvm.icmp \"ult\" %tid, %stride_{current_stride} : i64",
            f"      llvm.cond_br %can_reduce_{current_stride}, ^bb_red_{current_stride}, ^bb_sync_{current_stride}",
            f"    ^bb_red_{current_stride}:",
            f"      %idx_other_{current_stride} = llvm.add %tid, %stride_{current_stride} : i64",
            f"      %ptr_other_{current_stride} = llvm.getelementptr %shmem_ptr_uncasted[0, %idx_other_{current_stride}] : (!llvm.ptr<3>, i64) -> !llvm.ptr<3>, !llvm.array<256 x f32>",
            f"      %val_other_{current_stride} = llvm.load %ptr_other_{current_stride} : !llvm.ptr<3> -> f32",
            f"      %val_mine_{current_stride} = llvm.load %shmem_ptr_mine : !llvm.ptr<3> -> f32",
        ])
        # Use fold_op logic
        red_expr = _reduction_fold_expr(kernel.fold_op).replace("%acc", f"%val_mine_{current_stride}").replace("%item", f"%val_other_{current_stride}")
        lines.extend([
            f"      %res_{current_stride} = {red_expr}",
            f"      llvm.store %res_{current_stride}, %shmem_ptr_mine : f32, !llvm.ptr<3>",
            f"      llvm.br ^bb_sync_{current_stride}",
            f"    ^bb_sync_{current_stride}:",
            "      nvvm.barrier0",
        ])
        current_stride //= 2

    # Final atomicAdd to global output
    if kernel.fold_op == "+":
        lines.extend([
            "      %zero_i64_atomic = llvm.mlir.constant(0 : index) : i64",
            "      %is_first = llvm.icmp \"eq\" %tid, %zero_i64_atomic : i64",
            "      llvm.cond_br %is_first, ^bb_atomic, ^bb_done",
            "    ^bb_atomic:",
            "      %final_val = llvm.load %shmem_ptr_mine : !llvm.ptr<3> -> f32",
            "      %out_ptr = llvm.getelementptr %out_aligned[%out_offset] : (!llvm.ptr, i64) -> !llvm.ptr, f32",
            "      %unused_atomic = llvm.atomicrmw fadd %out_ptr, %final_val monotonic : !llvm.ptr, f32",
            "      llvm.br ^bb_done",
            "    ^bb_done:",
        ])
    else:
        # For non-sum, we only support 1 block for now to avoid atomics
        # RemoraExecutor will launch 1 block if it's not a sum?
        # Actually, let's just use atomicAdd for everything if possible, or skip.
        # Implementation Plan says "serial GPU reductions with parallel block reductions".
        # If it's 1 block, tree reduction is enough.
        lines.extend([
            "      %zero_i64_store = llvm.mlir.constant(0 : index) : i64",
            "      %is_first = llvm.icmp \"eq\" %tid, %zero_i64_store : i64",
            "      llvm.cond_br %is_first, ^bb_store, ^bb_done",
            "    ^bb_store:",
            "      %final_val = llvm.load %shmem_ptr_mine : !llvm.ptr<3> -> f32",
            "      %out_ptr = llvm.getelementptr %out_aligned[%out_offset] : (!llvm.ptr, i64) -> !llvm.ptr, f32",
            "      llvm.store %final_val, %out_ptr : f32, !llvm.ptr",
            "      llvm.br ^bb_done",
            "    ^bb_done:",
        ])

    return lines


def _reduction_binary_input_expr(operation: str | None) -> str:
    """Return the LLVM expression for a binary input operation in a reduction."""
    if operation not in {"*", "+", "-", "/"}:
        raise GPUScaffoldError(f"descriptor ABI GPU reduction does not support map operator {operation}")
    mlir_op = llvm_op(operation, "f32")
    return f"{mlir_op} %in0_x, %in1_x  : f32"


def _reduction_fold_expr(operation: str) -> str:
    """Return the LLVM expression for a fold (accumulate) operation in a reduction."""
    if operation not in {"+", "*"}:
        raise GPUScaffoldError(f"descriptor ABI GPU reduction does not support fold operator {operation}")
    mlir_op = llvm_op(operation, "f32")
    return f"{mlir_op} %acc, %item  : f32"


def _sobel_kernel(function: HIRFunction) -> tuple[HIRFunction, tuple[int, int], int]:
    """Return ``(function, kernel_shape, stride)`` if *function* is a valid Sobel."""
    from remora.hir import HIRIm2col
    if len(function.params) != 3:
        raise GPUScaffoldError("Sobel requires exactly 3 parameters (image, kx, ky)")
    def _find(expr):
        if isinstance(expr, HIRIm2col):
            return (expr.kernel_shape[0], expr.kernel_shape[1], expr.stride)
        for fld in ("array", "arrays", "args", "body", "value"):
            ch = getattr(expr, fld, None)
            if ch is None: continue
            if isinstance(ch, (list, tuple)):
                for c in ch:
                    r = _find(c)
                    if r: return r
            else:
                r = _find(ch)
                if r: return r
        return None
    s = _find(function.body)
    if s is None: raise GPUScaffoldError("Sobel requires an im2col in the body")
    return function, s[:2], s[2]


def build_descriptor_abi_sobel_gpu_module(
    function: HIRFunction, *, module_name: str = "remora_gpu", kernel_name: str | None = None,
) -> GPUModuleScaffold:
    """Build a combined GPU kernel for Sobel (Gx dot-product + Gy dot-product → Gx²+Gy²)."""
    _, (kh, kw), stride = _sobel_kernel(function)
    param_type = function.params[0].type
    if not isinstance(param_type, ArrayType):
        raise GPUScaffoldError("Sobel requires an array image parameter")
    h, w = int(param_type.shape[0].value), int(param_type.shape[1].value)
    ppa = (h - kh) // stride + 1; pc = ppa * ppa; cs = kh * kw
    name = kernel_name or f"remora_{function.name}_sobel"
    _validate_scaffold_names(module_name, name)
    ds = _descriptor_load_lines("img", "%input_desc", 2)
    ds.extend(_descriptor_load_lines("kx", "%kernx_desc", 1))
    ds.extend(_descriptor_load_lines("ky", "%kerny_desc", 1))
    ds.extend(_descriptor_load_lines("out", "%output_desc", 1))
    def _cloop(pf, ka, ko):
        return [
            f"      %{pf}_init = llvm.mlir.constant(0.000000e+00 : f32) : f32",
            f"      llvm.br ^sb_{pf}_loop(%sb_zero, %{pf}_init : i64, f32)",
            f"    ^sb_{pf}_loop(%{pf}_e: i64, %{pf}_acc: f32):",
            f"      %{pf}_done = llvm.icmp \"uge\" %{pf}_e, %sb_end : i64",
            f"      llvm.cond_br %{pf}_done, ^sb_{pf}_end(%{pf}_acc : f32), ^sb_{pf}_body",
            f"    ^sb_{pf}_body:",
            f"      %{pf}_krow = llvm.udiv %{pf}_e, %sb_kw : i64",
            f"      %{pf}_kcol = llvm.urem %{pf}_e, %sb_kw : i64",
            f"      %{pf}_ir1 = llvm.mul %sb_prow, %sb_st : i64",
            f"      %{pf}_ir2 = llvm.add %{pf}_ir1, %{pf}_krow : i64",
            f"      %{pf}_ic1 = llvm.mul %sb_pcol, %sb_st : i64",
            f"      %{pf}_ic2 = llvm.add %{pf}_ic1, %{pf}_kcol : i64",
            f"      %{pf}_ir3 = llvm.add %{pf}_ir2, %sb_zero : i64",
            f"      %{pf}_ic3 = llvm.add %{pf}_ic2, %sb_zero : i64",
            f"      %{pf}_t0 = llvm.mul %{pf}_ir3, %img_stride0 : i64",
            f"      %{pf}_roff = llvm.add %img_offset, %{pf}_t0 : i64",
            f"      %{pf}_lin = llvm.add %{pf}_roff, %{pf}_ic3 : i64",
            f"      %{pf}_pp = llvm.getelementptr %img_aligned[%{pf}_lin] : (!llvm.ptr, i64) -> !llvm.ptr, f32",
            f"      %{pf}_pix = llvm.load %{pf}_pp : !llvm.ptr -> f32",
            f"      %{pf}_ek = llvm.add %{pf}_e, %sb_zero : i64",
            f"      %{pf}_koff = llvm.add {ko}, %{pf}_ek : i64",
            f"      %{pf}_kp = llvm.getelementptr {ka}[%{pf}_koff] : (!llvm.ptr, i64) -> !llvm.ptr, f32",
            f"      %{pf}_kv = llvm.load %{pf}_kp : !llvm.ptr -> f32",
            f"      %{pf}_mul = llvm.fmul %{pf}_pix, %{pf}_kv : f32",
            f"      %{pf}_new = llvm.fadd %{pf}_acc, %{pf}_mul : f32",
            f"      %{pf}_next = llvm.add %{pf}_e, %sb_one : i64",
            f"      llvm.br ^sb_{pf}_loop(%{pf}_next, %{pf}_new : i64, f32)",
            f"    ^sb_{pf}_end(%{pf}_result: f32):",
        ]
    body = ds + [
        "      %sb_tid32 = nvvm.read.ptx.sreg.tid.x : i32",
        "      %sb_tid = llvm.sext %sb_tid32 : i32 to i64",
        "      %sb_bid32 = nvvm.read.ptx.sreg.ctaid.x : i32",
        "      %sb_bid = llvm.sext %sb_bid32 : i32 to i64",
        "      %sb_bdim32 = nvvm.read.ptx.sreg.ntid.x : i32",
        "      %sb_bdim = llvm.sext %sb_bdim32 : i32 to i64",
        "      %sb_blk = llvm.mul %sb_bid, %sb_bdim  : i64",
        "      %sb_idx = llvm.add %sb_blk, %sb_tid  : i64",
        f"      %sb_n = llvm.mlir.constant({pc} : index) : i64",
        "      %sb_ok = llvm.icmp \"ult\" %sb_idx, %sb_n : i64",
        "      llvm.cond_br %sb_ok, ^sb_body, ^sb_done",
        f"    ^sb_body:",
        f"      %sb_ppa = llvm.mlir.constant({ppa} : index) : i64",
        "      %sb_prow = llvm.udiv %sb_idx, %sb_ppa : i64",
        "      %sb_pcol = llvm.urem %sb_idx, %sb_ppa : i64",
        f"      %sb_zero = llvm.mlir.constant(0 : index) : i64",
        f"      %sb_one = llvm.mlir.constant(1 : index) : i64",
        f"      %sb_end = llvm.mlir.constant({cs} : index) : i64",
        f"      %sb_st = llvm.mlir.constant({stride} : index) : i64",
        f"      %sb_kw = llvm.mlir.constant({kw} : index) : i64",
    ] + _cloop("gx", "%kx_aligned", "%kx_offset") + [
        "      %sb_gx2 = llvm.fmul %gx_result, %gx_result : f32",
    ] + _cloop("gy", "%ky_aligned", "%ky_offset") + [
        "      %sb_gy2 = llvm.fmul %gy_result, %gy_result : f32",
        "      %sb_sum = llvm.fadd %sb_gx2, %sb_gy2 : f32",
        "      %sb_out_off = llvm.add %out_offset, %sb_idx : i64",
        "      %sb_out_p = llvm.getelementptr %out_aligned[%sb_out_off] : (!llvm.ptr, i64) -> !llvm.ptr, f32",
        "      llvm.store %sb_sum, %sb_out_p : f32, !llvm.ptr",
        "      llvm.br ^sb_done",
        "    ^sb_done:",
        "      llvm.return",
    ]
    text = f"""module {{
  gpu.module @{module_name} {{
    llvm.func @{name}(%input_desc: !llvm.ptr, %kernx_desc: !llvm.ptr, %kerny_desc: !llvm.ptr, %output_desc: !llvm.ptr) attributes {{gpu.kernel, nvvm.kernel}} {{
{chr(10).join(body)}
    }}
  }}
"""
    return GPUModuleScaffold(name=module_name, text=text)


# ---------------------------------------------------------------------------
# Matmul kernel
# ---------------------------------------------------------------------------


def build_descriptor_abi_matmul_gpu_module(
    function: HIRFunction,
    *,
    module_name: str = "remora_gpu",
    kernel_name: str | None = None,
) -> GPUModuleScaffold:
    """Build a descriptor-ABI GPU module for matrix multiplication.

    One thread per output element.  Each thread iterates over K to compute
    its dot product.  Correct for all sizes; a tiled shared-memory version
    would improve performance for large matrices.
    """
    if not isinstance(function.body, HIRMatmul):
        raise GPUScaffoldError("GPU matmul requires HIRMatmul body")
    if len(function.params) != 2:
        raise GPUScaffoldError("GPU matmul requires exactly two parameters")

    left_type = function.params[0].type
    right_type = function.params[1].type
    if not isinstance(left_type, ArrayType) or left_type.rank != 2:
        raise GPUScaffoldError("GPU matmul left operand must be rank-2 float array")
    if not isinstance(right_type, ArrayType) or right_type.rank != 2:
        raise GPUScaffoldError("GPU matmul right operand must be rank-2 float array")
    if left_type.element != FLOAT or right_type.element != FLOAT:
        raise GPUScaffoldError("GPU matmul currently supports f32 only")

    M = int(left_type.shape[0].value)
    K = int(left_type.shape[1].value)
    K2 = int(right_type.shape[0].value)
    N = int(right_type.shape[1].value)
    if K != K2:
        raise GPUScaffoldError(f"GPU matmul inner dimension mismatch: {K} vs {K2}")

    name = kernel_name or f"remora_{function.name}_matmul"
    _validate_scaffold_names(module_name, name)
    total = M * N

    desc_lines = _descriptor_load_lines("left", "%input0_desc", 2)
    desc_lines.extend(_descriptor_load_lines("right", "%input1_desc", 2))
    desc_lines.extend(_descriptor_load_lines("out", "%output_desc", 2))

    text = f"""module {{
  gpu.module @{module_name} {{
    llvm.func @{name}(%input0_desc: !llvm.ptr, %input1_desc: !llvm.ptr, %output_desc: !llvm.ptr) attributes {{gpu.kernel, nvvm.kernel}} {{
{chr(10).join(desc_lines)}
      %mm_tid32 = nvvm.read.ptx.sreg.tid.x : i32
      %mm_tid = llvm.sext %mm_tid32 : i32 to i64
      %mm_bid32 = nvvm.read.ptx.sreg.ctaid.x : i32
      %mm_bid = llvm.sext %mm_bid32 : i32 to i64
      %mm_bdim32 = nvvm.read.ptx.sreg.ntid.x : i32
      %mm_bdim = llvm.sext %mm_bdim32 : i32 to i64
      %mm_blk = llvm.mul %mm_bid, %mm_bdim  : i64
      %mm_flat = llvm.add %mm_blk, %mm_tid  : i64
      %mm_total = llvm.mlir.constant({total} : index) : i64
      %mm_ok = llvm.icmp "ult" %mm_flat, %mm_total : i64
      llvm.cond_br %mm_ok, ^mm_body, ^mm_done

    ^mm_body:
      %mm_cN = llvm.mlir.constant({N} : index) : i64
      %mm_row = llvm.udiv %mm_flat, %mm_cN  : i64
      %mm_col = llvm.urem %mm_flat, %mm_cN  : i64
      %mm_cK = llvm.mlir.constant({K} : index) : i64
      %mm_c0 = llvm.mlir.constant(0 : index) : i64
      %mm_c1 = llvm.mlir.constant(1 : index) : i64
      %mm_init = llvm.mlir.constant(0.000000e+00 : f32) : f32
      llvm.br ^mm_kloop(%mm_c0, %mm_init : i64, f32)

    ^mm_kloop(%mm_k: i64, %mm_acc: f32):
      %mm_kdone = llvm.icmp "uge" %mm_k, %mm_cK : i64
      llvm.cond_br %mm_kdone, ^mm_store(%mm_acc : f32), ^mm_kbody

    ^mm_kbody:
      %mm_a_t0 = llvm.mul %mm_row, %left_stride0  : i64
      %mm_a_t1 = llvm.mul %mm_k, %left_stride1  : i64
      %mm_a_off = llvm.add %left_offset, %mm_a_t0  : i64
      %mm_a_lin = llvm.add %mm_a_off, %mm_a_t1  : i64
      %mm_a_ptr = llvm.getelementptr %left_aligned[%mm_a_lin] : (!llvm.ptr, i64) -> !llvm.ptr, f32
      %mm_a_val = llvm.load %mm_a_ptr : !llvm.ptr -> f32
      %mm_b_t0 = llvm.mul %mm_k, %right_stride0  : i64
      %mm_b_t1 = llvm.mul %mm_col, %right_stride1  : i64
      %mm_b_off = llvm.add %right_offset, %mm_b_t0  : i64
      %mm_b_lin = llvm.add %mm_b_off, %mm_b_t1  : i64
      %mm_b_ptr = llvm.getelementptr %right_aligned[%mm_b_lin] : (!llvm.ptr, i64) -> !llvm.ptr, f32
      %mm_b_val = llvm.load %mm_b_ptr : !llvm.ptr -> f32
      %mm_prod = llvm.fmul %mm_a_val, %mm_b_val  : f32
      %mm_nacc = llvm.fadd %mm_acc, %mm_prod  : f32
      %mm_nk = llvm.add %mm_k, %mm_c1 : i64
      llvm.br ^mm_kloop(%mm_nk, %mm_nacc : i64, f32)

    ^mm_store(%mm_result: f32):
      %mm_o_t0 = llvm.mul %mm_row, %out_stride0  : i64
      %mm_o_t1 = llvm.mul %mm_col, %out_stride1  : i64
      %mm_o_off = llvm.add %out_offset, %mm_o_t0  : i64
      %mm_o_lin = llvm.add %mm_o_off, %mm_o_t1  : i64
      %mm_o_ptr = llvm.getelementptr %out_aligned[%mm_o_lin] : (!llvm.ptr, i64) -> !llvm.ptr, f32
      llvm.store %mm_result, %mm_o_ptr : f32, !llvm.ptr
      llvm.br ^mm_done

    ^mm_done:
      llvm.return
    }}
  }}
}}"""
    return GPUModuleScaffold(text, module_name, name)


def build_descriptor_abi_tiled_matmul_gpu_module(
    function: HIRFunction,
    *,
    module_name: str = "remora_gpu",
    kernel_name: str | None = None,
    tile_size: int = 16,
) -> GPUModuleScaffold:
    """Build a tiled shared-memory matmul GPU kernel.

    Uses cooperative tile loading with TILE×TILE shared memory blocks.
    Each thread block computes one TILE×TILE tile of the output matrix.
    Improves memory bandwidth utilization vs the per-thread dot product.
    """
    from remora.hir import HIRMatmul

    if not isinstance(function.body, HIRMatmul):
        raise GPUScaffoldError("GPU tiled matmul requires HIRMatmul body")
    if len(function.params) != 2:
        raise GPUScaffoldError("GPU tiled matmul requires exactly two parameters")

    left_type = function.params[0].type
    right_type = function.params[1].type
    if not isinstance(left_type, ArrayType) or not isinstance(right_type, ArrayType):
        raise GPUScaffoldError("GPU tiled matmul requires array parameters")
    if left_type.element != FLOAT or right_type.element != FLOAT:
        raise GPUScaffoldError("GPU tiled matmul supports f32 only")
    if left_type.rank != 2 or right_type.rank != 2:
        raise GPUScaffoldError("GPU tiled matmul requires rank-2 arrays")

    M = int(left_type.shape[0].value)
    K = int(left_type.shape[1].value)
    K2 = int(right_type.shape[0].value)
    N = int(right_type.shape[1].value)
    if K != K2:
        raise GPUScaffoldError(f"matmul inner dimension mismatch: {K} vs {K2}")

    T = tile_size
    gridCols = (N + T - 1) // T
    num_tiles = (K + T - 1) // T

    name = kernel_name or f"remora_{function.name}_mm"
    _validate_scaffold_names(module_name, name)

    desc_lines = _descriptor_load_lines("la", "%input0_desc", 2)
    desc_lines.extend(_descriptor_load_lines("rb", "%input1_desc", 2))
    desc_lines.extend(_descriptor_load_lines("oc", "%output_desc", 2))

    text = f"""module {{
  gpu.module @{module_name} {{
    llvm.mlir.global internal @mm_sha() {{addr_space = 3 : i32}} : !llvm.array<{T*T} x f32>
    llvm.mlir.global internal @mm_shb() {{addr_space = 3 : i32}} : !llvm.array<{T*T} x f32>

    llvm.func @{name}(%input0_desc: !llvm.ptr, %input1_desc: !llvm.ptr, %output_desc: !llvm.ptr) attributes {{gpu.kernel, nvvm.kernel}} {{
{chr(10).join(desc_lines)}
      %tm_tid32 = nvvm.read.ptx.sreg.tid.x : i32
      %tm_tid = llvm.sext %tm_tid32 : i32 to i64
      %tm_bid32 = nvvm.read.ptx.sreg.ctaid.x : i32
      %tm_bid = llvm.sext %tm_bid32 : i32 to i64
      %tm_T = llvm.mlir.constant({T} : index) : i64
      %tm_lr = llvm.udiv %tm_tid, %tm_T  : i64
      %tm_lc = llvm.urem %tm_tid, %tm_T  : i64
      %tm_gcols = llvm.mlir.constant({gridCols} : index) : i64
      %tm_brow = llvm.udiv %tm_bid, %tm_gcols  : i64
      %tm_bcol = llvm.urem %tm_bid, %tm_gcols  : i64
      %tm_br_off = llvm.mul %tm_brow, %tm_T  : i64
      %tm_bc_off = llvm.mul %tm_bcol, %tm_T  : i64
      %tm_gr = llvm.add %tm_br_off, %tm_lr  : i64
      %tm_gc = llvm.add %tm_bc_off, %tm_lc  : i64
      %tm_sha = llvm.mlir.addressof @mm_sha : !llvm.ptr<3>
      %tm_shb = llvm.mlir.addressof @mm_shb : !llvm.ptr<3>
      %tm_c0 = llvm.mlir.constant(0 : index) : i64
      %tm_c1 = llvm.mlir.constant(1 : index) : i64
      %tm_cM = llvm.mlir.constant({M} : index) : i64
      %tm_cN = llvm.mlir.constant({N} : index) : i64
      %tm_cK = llvm.mlir.constant({K} : index) : i64
      %tm_nt = llvm.mlir.constant({num_tiles} : index) : i64
      %tm_zf = llvm.mlir.constant(0.000000e+00 : f32) : f32
      llvm.br ^tm_tloop(%tm_c0, %tm_zf : i64, f32)

    ^tm_tloop(%tm_t: i64, %tm_acc: f32):
      %tm_td = llvm.icmp "uge" %tm_t, %tm_nt : i64
      llvm.cond_br %tm_td, ^tm_store, ^tm_tbody

    ^tm_tbody:
      %tm_tk = llvm.mul %tm_t, %tm_T  : i64
      %tm_ac = llvm.add %tm_tk, %tm_lc  : i64
      %tm_a_r_ok = llvm.icmp "ult" %tm_gr, %tm_cM : i64
      %tm_a_c_ok = llvm.icmp "ult" %tm_ac, %tm_cK : i64
      %tm_a_ok = llvm.and %tm_a_r_ok, %tm_a_c_ok  : i1
      %tm_sa_r = llvm.select %tm_a_ok, %tm_gr, %tm_c0 : i1, i64
      %tm_sa_c = llvm.select %tm_a_ok, %tm_ac, %tm_c0 : i1, i64
      %tm_a_off = llvm.mul %tm_sa_r, %la_stride0  : i64
      %tm_a_off2 = llvm.add %tm_a_off, %tm_sa_c  : i64
      %tm_a_off3 = llvm.add %la_offset, %tm_a_off2  : i64
      %tm_a_ptr = llvm.getelementptr %la_aligned[%tm_a_off3] : (!llvm.ptr, i64) -> !llvm.ptr, f32
      %tm_a_raw = llvm.load %tm_a_ptr : !llvm.ptr -> f32
      %tm_a_val = llvm.select %tm_a_ok, %tm_a_raw, %tm_zf : i1, f32
      %tm_sha_idx = llvm.mul %tm_lr, %tm_T  : i64
      %tm_sha_idx2 = llvm.add %tm_sha_idx, %tm_lc  : i64
      %tm_sha_p = llvm.getelementptr %tm_sha[0, %tm_sha_idx2] : (!llvm.ptr<3>, i64) -> !llvm.ptr<3>, !llvm.array<{T*T} x f32>
      llvm.store %tm_a_val, %tm_sha_p : f32, !llvm.ptr<3>
      %tm_br = llvm.add %tm_tk, %tm_lr  : i64
      %tm_b_r_ok = llvm.icmp "ult" %tm_br, %tm_cK : i64
      %tm_b_c_ok = llvm.icmp "ult" %tm_gc, %tm_cN : i64
      %tm_b_ok = llvm.and %tm_b_r_ok, %tm_b_c_ok  : i1
      %tm_sb_r = llvm.select %tm_b_ok, %tm_br, %tm_c0 : i1, i64
      %tm_sb_c = llvm.select %tm_b_ok, %tm_gc, %tm_c0 : i1, i64
      %tm_b_off = llvm.mul %tm_sb_r, %rb_stride0  : i64
      %tm_b_off2 = llvm.add %tm_b_off, %tm_sb_c  : i64
      %tm_b_off3 = llvm.add %rb_offset, %tm_b_off2  : i64
      %tm_b_ptr = llvm.getelementptr %rb_aligned[%tm_b_off3] : (!llvm.ptr, i64) -> !llvm.ptr, f32
      %tm_b_raw = llvm.load %tm_b_ptr : !llvm.ptr -> f32
      %tm_b_val = llvm.select %tm_b_ok, %tm_b_raw, %tm_zf : i1, f32
      %tm_shb_idx = llvm.mul %tm_lr, %tm_T  : i64
      %tm_shb_idx2 = llvm.add %tm_shb_idx, %tm_lc  : i64
      %tm_shb_p = llvm.getelementptr %tm_shb[0, %tm_shb_idx2] : (!llvm.ptr<3>, i64) -> !llvm.ptr<3>, !llvm.array<{T*T} x f32>
      llvm.store %tm_b_val, %tm_shb_p : f32, !llvm.ptr<3>
      nvvm.barrier0
      llvm.br ^tm_kloop(%tm_c0, %tm_acc : i64, f32)

    ^tm_kloop(%tm_k: i64, %tm_kacc: f32):
      %tm_kd = llvm.icmp "uge" %tm_k, %tm_T : i64
      llvm.cond_br %tm_kd, ^tm_kend(%tm_kacc : f32), ^tm_kbody

    ^tm_kbody:
      %tm_ai = llvm.mul %tm_lr, %tm_T  : i64
      %tm_ai2 = llvm.add %tm_ai, %tm_k  : i64
      %tm_ap = llvm.getelementptr %tm_sha[0, %tm_ai2] : (!llvm.ptr<3>, i64) -> !llvm.ptr<3>, !llvm.array<{T*T} x f32>
      %tm_av = llvm.load %tm_ap : !llvm.ptr<3> -> f32
      %tm_bi = llvm.mul %tm_k, %tm_T  : i64
      %tm_bi2 = llvm.add %tm_bi, %tm_lc  : i64
      %tm_bp = llvm.getelementptr %tm_shb[0, %tm_bi2] : (!llvm.ptr<3>, i64) -> !llvm.ptr<3>, !llvm.array<{T*T} x f32>
      %tm_bv = llvm.load %tm_bp : !llvm.ptr<3> -> f32
      %tm_prod = llvm.fmul %tm_av, %tm_bv  : f32
      %tm_nacc = llvm.fadd %tm_kacc, %tm_prod  : f32
      %tm_nk = llvm.add %tm_k, %tm_c1  : i64
      llvm.br ^tm_kloop(%tm_nk, %tm_nacc : i64, f32)

    ^tm_kend(%tm_facc: f32):
      nvvm.barrier0
      %tm_nt2 = llvm.add %tm_t, %tm_c1  : i64
      llvm.br ^tm_tloop(%tm_nt2, %tm_facc : i64, f32)

    ^tm_store:
      %tm_s_r_ok = llvm.icmp "ult" %tm_gr, %tm_cM : i64
      %tm_s_c_ok = llvm.icmp "ult" %tm_gc, %tm_cN : i64
      %tm_s_ok = llvm.and %tm_s_r_ok, %tm_s_c_ok  : i1
      llvm.cond_br %tm_s_ok, ^tm_write, ^tm_done

    ^tm_write:
      %tm_o_off = llvm.mul %tm_gr, %oc_stride0  : i64
      %tm_o_off2 = llvm.add %tm_o_off, %tm_gc  : i64
      %tm_o_off3 = llvm.add %oc_offset, %tm_o_off2  : i64
      %tm_o_ptr = llvm.getelementptr %oc_aligned[%tm_o_off3] : (!llvm.ptr, i64) -> !llvm.ptr, f32
      llvm.store %tm_acc, %tm_o_ptr : f32, !llvm.ptr
      llvm.br ^tm_done

    ^tm_done:
      llvm.return
    }}
  }}
}}"""
    return GPUModuleScaffold(text, module_name, name)


# ---------------------------------------------------------------------------
# Sort / Grade kernels (serial single-thread)
# ---------------------------------------------------------------------------


def build_descriptor_abi_sort_gpu_module(
    function: HIRFunction,
    *,
    module_name: str = "remora_gpu",
    kernel_name: str | None = None,
) -> GPUModuleScaffold:
    """Build a descriptor-ABI GPU module for sorting (serial insertion sort)."""
    if not isinstance(function.body, HIRSort):
        raise GPUScaffoldError("GPU sort requires HIRSort body")
    if len(function.params) != 1:
        raise GPUScaffoldError("GPU sort requires exactly one parameter")
    param_type = function.params[0].type
    if not isinstance(param_type, ArrayType) or param_type.element != FLOAT:
        raise GPUScaffoldError("GPU sort supports rank-1 f32 only")
    if param_type.rank != 1:
        raise GPUScaffoldError("GPU sort supports rank-1 only")

    N = int(param_type.shape[0].value)
    name = kernel_name or f"remora_{function.name}_sort"
    _validate_scaffold_names(module_name, name)

    desc_lines = _descriptor_load_lines("in", "%input_desc", 1)
    desc_lines.extend(_descriptor_load_lines("out", "%output_desc", 1))

    text = f"""module {{
  gpu.module @{module_name} {{
    llvm.func @{name}(%input_desc: !llvm.ptr, %output_desc: !llvm.ptr) attributes {{gpu.kernel, nvvm.kernel}} {{
{chr(10).join(desc_lines)}
      %s_tid32 = nvvm.read.ptx.sreg.tid.x : i32
      %s_tid = llvm.sext %s_tid32 : i32 to i64
      %s_z = llvm.mlir.constant(0 : index) : i64
      %s_is_t0 = llvm.icmp "eq" %s_tid, %s_z : i64
      llvm.cond_br %s_is_t0, ^s_copy, ^s_done

    ^s_copy:
      %s_N = llvm.mlir.constant({N} : index) : i64
      %s_one = llvm.mlir.constant(1 : index) : i64
      llvm.br ^s_cp_loop(%s_z : i64)

    ^s_cp_loop(%s_ci: i64):
      %s_cp_done = llvm.icmp "uge" %s_ci, %s_N : i64
      llvm.cond_br %s_cp_done, ^s_sort, ^s_cp_body

    ^s_cp_body:
      %s_si = llvm.add %in_offset, %s_ci  : i64
      %s_sp = llvm.getelementptr %in_aligned[%s_si] : (!llvm.ptr, i64) -> !llvm.ptr, f32
      %s_sv = llvm.load %s_sp : !llvm.ptr -> f32
      %s_di = llvm.add %out_offset, %s_ci  : i64
      %s_dp = llvm.getelementptr %out_aligned[%s_di] : (!llvm.ptr, i64) -> !llvm.ptr, f32
      llvm.store %s_sv, %s_dp : f32, !llvm.ptr
      %s_cn = llvm.add %s_ci, %s_one : i64
      llvm.br ^s_cp_loop(%s_cn : i64)

    ^s_sort:
      llvm.br ^s_outer(%s_one : i64)

    ^s_outer(%s_i: i64):
      %s_od = llvm.icmp "uge" %s_i, %s_N : i64
      llvm.cond_br %s_od, ^s_done, ^s_load_key

    ^s_load_key:
      %s_ki = llvm.add %out_offset, %s_i  : i64
      %s_kp = llvm.getelementptr %out_aligned[%s_ki] : (!llvm.ptr, i64) -> !llvm.ptr, f32
      %s_key = llvm.load %s_kp : !llvm.ptr -> f32
      llvm.br ^s_inner(%s_i, %s_key : i64, f32)

    ^s_inner(%s_j: i64, %s_kv: f32):
      %s_jz = llvm.icmp "eq" %s_j, %s_z : i64
      llvm.cond_br %s_jz, ^s_place(%s_j, %s_kv : i64, f32), ^s_cmp

    ^s_cmp:
      %s_jm1 = llvm.sub %s_j, %s_one  : i64
      %s_pi = llvm.add %out_offset, %s_jm1  : i64
      %s_pp = llvm.getelementptr %out_aligned[%s_pi] : (!llvm.ptr, i64) -> !llvm.ptr, f32
      %s_pv = llvm.load %s_pp : !llvm.ptr -> f32
      %s_gt = llvm.fcmp "ogt" %s_pv, %s_kv : f32
      llvm.cond_br %s_gt, ^s_shift, ^s_place(%s_j, %s_kv : i64, f32)

    ^s_shift:
      %s_wi = llvm.add %out_offset, %s_j  : i64
      %s_wp = llvm.getelementptr %out_aligned[%s_wi] : (!llvm.ptr, i64) -> !llvm.ptr, f32
      llvm.store %s_pv, %s_wp : f32, !llvm.ptr
      llvm.br ^s_inner(%s_jm1, %s_kv : i64, f32)

    ^s_place(%s_pj: i64, %s_pk: f32):
      %s_fi = llvm.add %out_offset, %s_pj  : i64
      %s_fp = llvm.getelementptr %out_aligned[%s_fi] : (!llvm.ptr, i64) -> !llvm.ptr, f32
      llvm.store %s_pk, %s_fp : f32, !llvm.ptr
      %s_ni = llvm.add %s_i, %s_one : i64
      llvm.br ^s_outer(%s_ni : i64)

    ^s_done:
      llvm.return
    }}
  }}
}}"""
    return GPUModuleScaffold(text, module_name, name)


def build_descriptor_abi_grade_gpu_module(
    function: HIRFunction,
    *,
    module_name: str = "remora_gpu",
    kernel_name: str | None = None,
) -> GPUModuleScaffold:
    """Build a descriptor-ABI GPU module for grade (sort-indices, serial).

    Single-thread: initialise output indices 0..N-1, then sort by comparing
    input values at those indices (insertion sort on the permutation).
    Output element type is i32 (integer indices).
    """
    if not isinstance(function.body, HIRGrade):
        raise GPUScaffoldError("GPU grade requires HIRGrade body")
    if len(function.params) != 1:
        raise GPUScaffoldError("GPU grade requires exactly one parameter")
    param_type = function.params[0].type
    if not isinstance(param_type, ArrayType) or param_type.element != FLOAT:
        raise GPUScaffoldError("GPU grade supports rank-1 f32 input only")
    if param_type.rank != 1:
        raise GPUScaffoldError("GPU grade supports rank-1 only")

    N = int(param_type.shape[0].value)
    name = kernel_name or f"remora_{function.name}_grade"
    _validate_scaffold_names(module_name, name)

    desc_lines = _descriptor_load_lines("in", "%input_desc", 1)
    desc_lines.extend(_descriptor_load_lines("out", "%output_desc", 1))

    text = f"""module {{
  gpu.module @{module_name} {{
    llvm.func @{name}(%input_desc: !llvm.ptr, %output_desc: !llvm.ptr) attributes {{gpu.kernel, nvvm.kernel}} {{
{chr(10).join(desc_lines)}
      %g_tid32 = nvvm.read.ptx.sreg.tid.x : i32
      %g_tid = llvm.sext %g_tid32 : i32 to i64
      %g_z = llvm.mlir.constant(0 : index) : i64
      %g_is_t0 = llvm.icmp "eq" %g_tid, %g_z : i64
      llvm.cond_br %g_is_t0, ^g_init, ^g_done

    ^g_init:
      %g_N = llvm.mlir.constant({N} : index) : i64
      %g_one = llvm.mlir.constant(1 : index) : i64
      llvm.br ^g_init_loop(%g_z : i64)

    ^g_init_loop(%g_ii: i64):
      %g_id = llvm.icmp "uge" %g_ii, %g_N : i64
      llvm.cond_br %g_id, ^g_sort, ^g_init_body

    ^g_init_body:
      %g_oi = llvm.add %out_offset, %g_ii  : i64
      %g_op = llvm.getelementptr %out_aligned[%g_oi] : (!llvm.ptr, i64) -> !llvm.ptr, i32
      %g_ii32 = llvm.trunc %g_ii : i64 to i32
      llvm.store %g_ii32, %g_op : i32, !llvm.ptr
      %g_in = llvm.add %g_ii, %g_one : i64
      llvm.br ^g_init_loop(%g_in : i64)

    ^g_sort:
      llvm.br ^g_outer(%g_one : i64)

    ^g_outer(%g_i: i64):
      %g_od = llvm.icmp "uge" %g_i, %g_N : i64
      llvm.cond_br %g_od, ^g_done, ^g_load_key

    ^g_load_key:
      %g_ki = llvm.add %out_offset, %g_i  : i64
      %g_kp = llvm.getelementptr %out_aligned[%g_ki] : (!llvm.ptr, i64) -> !llvm.ptr, i32
      %g_kidx = llvm.load %g_kp : !llvm.ptr -> i32
      %g_kidx64 = llvm.sext %g_kidx : i32 to i64
      %g_kvi = llvm.add %in_offset, %g_kidx64  : i64
      %g_kvp = llvm.getelementptr %in_aligned[%g_kvi] : (!llvm.ptr, i64) -> !llvm.ptr, f32
      %g_kval = llvm.load %g_kvp : !llvm.ptr -> f32
      llvm.br ^g_inner(%g_i, %g_kidx, %g_kval : i64, i32, f32)

    ^g_inner(%g_j: i64, %g_ki32: i32, %g_kv: f32):
      %g_jz = llvm.icmp "eq" %g_j, %g_z : i64
      llvm.cond_br %g_jz, ^g_place(%g_j, %g_ki32 : i64, i32), ^g_cmp

    ^g_cmp:
      %g_jm1 = llvm.sub %g_j, %g_one  : i64
      %g_pi = llvm.add %out_offset, %g_jm1  : i64
      %g_pp = llvm.getelementptr %out_aligned[%g_pi] : (!llvm.ptr, i64) -> !llvm.ptr, i32
      %g_pidx = llvm.load %g_pp : !llvm.ptr -> i32
      %g_pidx64 = llvm.sext %g_pidx : i32 to i64
      %g_pvi = llvm.add %in_offset, %g_pidx64  : i64
      %g_pvp = llvm.getelementptr %in_aligned[%g_pvi] : (!llvm.ptr, i64) -> !llvm.ptr, f32
      %g_pval = llvm.load %g_pvp : !llvm.ptr -> f32
      %g_gt = llvm.fcmp "ogt" %g_pval, %g_kv : f32
      llvm.cond_br %g_gt, ^g_shift, ^g_place(%g_j, %g_ki32 : i64, i32)

    ^g_shift:
      %g_wi = llvm.add %out_offset, %g_j  : i64
      %g_wp = llvm.getelementptr %out_aligned[%g_wi] : (!llvm.ptr, i64) -> !llvm.ptr, i32
      llvm.store %g_pidx, %g_wp : i32, !llvm.ptr
      llvm.br ^g_inner(%g_jm1, %g_ki32, %g_kv : i64, i32, f32)

    ^g_place(%g_pj: i64, %g_pk: i32):
      %g_fi = llvm.add %out_offset, %g_pj  : i64
      %g_fp = llvm.getelementptr %out_aligned[%g_fi] : (!llvm.ptr, i64) -> !llvm.ptr, i32
      llvm.store %g_pk, %g_fp : i32, !llvm.ptr
      %g_ni = llvm.add %g_i, %g_one : i64
      llvm.br ^g_outer(%g_ni : i64)

    ^g_done:
      llvm.return
    }}
  }}
}}"""
    return GPUModuleScaffold(text, module_name, name)


def build_descriptor_abi_bitonic_sort_gpu_module(
    function: HIRFunction,
    *,
    module_name: str = "remora_gpu",
    kernel_name: str | None = None,
) -> GPUModuleScaffold:
    """Build a parallel bitonic sort kernel in shared memory (N ≤ 1024)."""
    from remora.hir import HIRSort
    if not isinstance(function.body, HIRSort):
        raise GPUScaffoldError("GPU bitonic sort requires HIRSort body")
    if len(function.params) != 1:
        raise GPUScaffoldError("GPU bitonic sort requires one parameter")
    param_type = function.params[0].type
    if not isinstance(param_type, ArrayType) or param_type.element != FLOAT:
        raise GPUScaffoldError("GPU bitonic sort supports rank-1 f32 only")
    if param_type.rank != 1:
        raise GPUScaffoldError("GPU bitonic sort supports rank-1 only")

    N = int(param_type.shape[0].value)
    NP = 1
    while NP < N:
        NP *= 2
    if NP > 1024:
        raise GPUScaffoldError("GPU bitonic sort requires N ≤ 1024")
    num_stages = NP.bit_length() - 1

    name = kernel_name or f"remora_{function.name}_sort"
    _validate_scaffold_names(module_name, name)

    desc_lines = _descriptor_load_lines("in", "%input_desc", 1)
    desc_lines.extend(_descriptor_load_lines("out", "%output_desc", 1))

    text = f"""module {{
  gpu.module @{module_name} {{
    llvm.mlir.global internal @bs_shmem() {{addr_space = 3 : i32}} : !llvm.array<{NP} x f32>
    llvm.func @{name}(%input_desc: !llvm.ptr, %output_desc: !llvm.ptr) attributes {{gpu.kernel, nvvm.kernel}} {{
{chr(10).join(desc_lines)}
      %bs_tid32 = nvvm.read.ptx.sreg.tid.x : i32
      %bs_tid = llvm.sext %bs_tid32 : i32 to i64
      %bs_N = llvm.mlir.constant({N} : index) : i64
      %bs_NP = llvm.mlir.constant({NP} : index) : i64
      %bs_ok = llvm.icmp "ult" %bs_tid, %bs_N : i64
      %bs_shmem = llvm.mlir.addressof @bs_shmem : !llvm.ptr<3>
      %bs_inf = llvm.mlir.constant(3.40282347e+38 : f32) : f32
      llvm.cond_br %bs_ok, ^bs_ld, ^bs_pad

    ^bs_ld:
      %bs_si = llvm.add %in_offset, %bs_tid  : i64
      %bs_sp = llvm.getelementptr %in_aligned[%bs_si] : (!llvm.ptr, i64) -> !llvm.ptr, f32
      %bs_val = llvm.load %bs_sp : !llvm.ptr -> f32
      %bs_mp = llvm.getelementptr %bs_shmem[0, %bs_tid] : (!llvm.ptr<3>, i64) -> !llvm.ptr<3>, !llvm.array<{NP} x f32>
      llvm.store %bs_val, %bs_mp : f32, !llvm.ptr<3>
      llvm.br ^bs_sync0

    ^bs_pad:
      %bs_in_np = llvm.icmp "ult" %bs_tid, %bs_NP : i64
      llvm.cond_br %bs_in_np, ^bs_pad_w, ^bs_sync0

    ^bs_pad_w:
      %bs_mp2 = llvm.getelementptr %bs_shmem[0, %bs_tid] : (!llvm.ptr<3>, i64) -> !llvm.ptr<3>, !llvm.array<{NP} x f32>
      llvm.store %bs_inf, %bs_mp2 : f32, !llvm.ptr<3>
      llvm.br ^bs_sync0

    ^bs_sync0:
      nvvm.barrier0
      %bs_c0 = llvm.mlir.constant(0 : index) : i64
      %bs_c1 = llvm.mlir.constant(1 : index) : i64
      %bs_c2 = llvm.mlir.constant(2 : index) : i64
      %bs_ns = llvm.mlir.constant({num_stages} : index) : i64
      llvm.br ^bs_outer(%bs_c0 : i64)

    ^bs_outer(%bs_k: i64):
      %bs_kd = llvm.icmp "uge" %bs_k, %bs_ns : i64
      llvm.cond_br %bs_kd, ^bs_write, ^bs_inner_start

    ^bs_inner_start:
      llvm.br ^bs_inner(%bs_c0 : i64)

    ^bs_inner(%bs_s: i64):
      %bs_sd = llvm.icmp "sgt" %bs_s, %bs_k : i64
      llvm.cond_br %bs_sd, ^bs_outer_next, ^bs_step

    ^bs_step:
      %bs_j = llvm.sub %bs_k, %bs_s  : i64
      %bs_stride = llvm.shl %bs_c1, %bs_j  : i64
      %bs_partner = llvm.xor %bs_tid, %bs_stride  : i64
      %bs_kp1 = llvm.add %bs_k, %bs_c1  : i64
      %bs_shifted = llvm.lshr %bs_tid, %bs_kp1  : i64
      %bs_bit = llvm.and %bs_shifted, %bs_c1  : i64
      %bs_asc = llvm.icmp "eq" %bs_bit, %bs_c0 : i64
      %bs_low_bit = llvm.and %bs_tid, %bs_stride  : i64
      %bs_is_lower = llvm.icmp "eq" %bs_low_bit, %bs_c0 : i64
      %bs_me_p = llvm.getelementptr %bs_shmem[0, %bs_tid] : (!llvm.ptr<3>, i64) -> !llvm.ptr<3>, !llvm.array<{NP} x f32>
      %bs_pa_p = llvm.getelementptr %bs_shmem[0, %bs_partner] : (!llvm.ptr<3>, i64) -> !llvm.ptr<3>, !llvm.array<{NP} x f32>
      %bs_me_v = llvm.load %bs_me_p : !llvm.ptr<3> -> f32
      %bs_pa_v = llvm.load %bs_pa_p : !llvm.ptr<3> -> f32
      nvvm.barrier0
      %bs_lt = llvm.fcmp "olt" %bs_me_v, %bs_pa_v : f32
      %bs_min = llvm.select %bs_lt, %bs_me_v, %bs_pa_v : i1, f32
      %bs_max = llvm.select %bs_lt, %bs_pa_v, %bs_me_v : i1, f32
      %bs_want_min = llvm.icmp "eq" %bs_is_lower, %bs_asc : i1
      %bs_res = llvm.select %bs_want_min, %bs_min, %bs_max : i1, f32
      llvm.store %bs_res, %bs_me_p : f32, !llvm.ptr<3>
      nvvm.barrier0
      %bs_ns2 = llvm.add %bs_s, %bs_c1  : i64
      llvm.br ^bs_inner(%bs_ns2 : i64)

    ^bs_outer_next:
      %bs_nk = llvm.add %bs_k, %bs_c1  : i64
      llvm.br ^bs_outer(%bs_nk : i64)

    ^bs_write:
      llvm.cond_br %bs_ok, ^bs_wb, ^bs_done

    ^bs_wb:
      %bs_rp = llvm.getelementptr %bs_shmem[0, %bs_tid] : (!llvm.ptr<3>, i64) -> !llvm.ptr<3>, !llvm.array<{NP} x f32>
      %bs_rv = llvm.load %bs_rp : !llvm.ptr<3> -> f32
      %bs_oi = llvm.add %out_offset, %bs_tid  : i64
      %bs_op = llvm.getelementptr %out_aligned[%bs_oi] : (!llvm.ptr, i64) -> !llvm.ptr, f32
      llvm.store %bs_rv, %bs_op : f32, !llvm.ptr
      llvm.br ^bs_done

    ^bs_done:
      llvm.return
    }}
  }}
}}"""
    return GPUModuleScaffold(text, module_name, name)


def build_descriptor_abi_bitonic_grade_gpu_module(
    function: HIRFunction,
    *,
    module_name: str = "remora_gpu",
    kernel_name: str | None = None,
) -> GPUModuleScaffold:
    """Build a parallel bitonic grade (argsort) kernel in shared memory (N ≤ 1024)."""
    from remora.hir import HIRGrade
    if not isinstance(function.body, HIRGrade):
        raise GPUScaffoldError("GPU bitonic grade requires HIRGrade body")
    if len(function.params) != 1:
        raise GPUScaffoldError("GPU bitonic grade requires one parameter")
    param_type = function.params[0].type
    if not isinstance(param_type, ArrayType) or param_type.element != FLOAT:
        raise GPUScaffoldError("GPU bitonic grade supports rank-1 f32 only")
    if param_type.rank != 1:
        raise GPUScaffoldError("GPU bitonic grade supports rank-1 only")

    N = int(param_type.shape[0].value)
    NP = 1
    while NP < N:
        NP *= 2
    if NP > 1024:
        raise GPUScaffoldError("GPU bitonic grade requires N ≤ 1024")
    num_stages = NP.bit_length() - 1

    name = kernel_name or f"remora_{function.name}_grade"
    _validate_scaffold_names(module_name, name)

    desc_lines = _descriptor_load_lines("in", "%input_desc", 1)
    desc_lines.extend(_descriptor_load_lines("out", "%output_desc", 1))

    text = f"""module {{
  gpu.module @{module_name} {{
    llvm.mlir.global internal @bg_vals() {{addr_space = 3 : i32}} : !llvm.array<{NP} x f32>
    llvm.mlir.global internal @bg_idxs() {{addr_space = 3 : i32}} : !llvm.array<{NP} x i32>
    llvm.func @{name}(%input_desc: !llvm.ptr, %output_desc: !llvm.ptr) attributes {{gpu.kernel, nvvm.kernel}} {{
{chr(10).join(desc_lines)}
      %bg_tid32 = nvvm.read.ptx.sreg.tid.x : i32
      %bg_tid = llvm.sext %bg_tid32 : i32 to i64
      %bg_N = llvm.mlir.constant({N} : index) : i64
      %bg_NP = llvm.mlir.constant({NP} : index) : i64
      %bg_ok = llvm.icmp "ult" %bg_tid, %bg_N : i64
      %bg_vals = llvm.mlir.addressof @bg_vals : !llvm.ptr<3>
      %bg_idxs = llvm.mlir.addressof @bg_idxs : !llvm.ptr<3>
      %bg_inf = llvm.mlir.constant(3.40282347e+38 : f32) : f32
      %bg_tid_i32 = llvm.trunc %bg_tid : i64 to i32
      %bg_neg1 = llvm.mlir.constant(-1 : i32) : i32
      llvm.cond_br %bg_ok, ^bg_ld, ^bg_pad

    ^bg_ld:
      %bg_si = llvm.add %in_offset, %bg_tid  : i64
      %bg_sp = llvm.getelementptr %in_aligned[%bg_si] : (!llvm.ptr, i64) -> !llvm.ptr, f32
      %bg_val = llvm.load %bg_sp : !llvm.ptr -> f32
      %bg_vp = llvm.getelementptr %bg_vals[0, %bg_tid] : (!llvm.ptr<3>, i64) -> !llvm.ptr<3>, !llvm.array<{NP} x f32>
      llvm.store %bg_val, %bg_vp : f32, !llvm.ptr<3>
      %bg_ip = llvm.getelementptr %bg_idxs[0, %bg_tid] : (!llvm.ptr<3>, i64) -> !llvm.ptr<3>, !llvm.array<{NP} x i32>
      llvm.store %bg_tid_i32, %bg_ip : i32, !llvm.ptr<3>
      llvm.br ^bg_sync0

    ^bg_pad:
      %bg_in_np = llvm.icmp "ult" %bg_tid, %bg_NP : i64
      llvm.cond_br %bg_in_np, ^bg_pad_w, ^bg_sync0

    ^bg_pad_w:
      %bg_vp2 = llvm.getelementptr %bg_vals[0, %bg_tid] : (!llvm.ptr<3>, i64) -> !llvm.ptr<3>, !llvm.array<{NP} x f32>
      llvm.store %bg_inf, %bg_vp2 : f32, !llvm.ptr<3>
      %bg_ip2 = llvm.getelementptr %bg_idxs[0, %bg_tid] : (!llvm.ptr<3>, i64) -> !llvm.ptr<3>, !llvm.array<{NP} x i32>
      llvm.store %bg_neg1, %bg_ip2 : i32, !llvm.ptr<3>
      llvm.br ^bg_sync0

    ^bg_sync0:
      nvvm.barrier0
      %bg_c0 = llvm.mlir.constant(0 : index) : i64
      %bg_c1 = llvm.mlir.constant(1 : index) : i64
      %bg_ns = llvm.mlir.constant({num_stages} : index) : i64
      llvm.br ^bg_outer(%bg_c0 : i64)

    ^bg_outer(%bg_k: i64):
      %bg_kd = llvm.icmp "uge" %bg_k, %bg_ns : i64
      llvm.cond_br %bg_kd, ^bg_write, ^bg_inner_start

    ^bg_inner_start:
      llvm.br ^bg_inner(%bg_c0 : i64)

    ^bg_inner(%bg_s: i64):
      %bg_sd = llvm.icmp "sgt" %bg_s, %bg_k : i64
      llvm.cond_br %bg_sd, ^bg_outer_next, ^bg_step

    ^bg_step:
      %bg_j = llvm.sub %bg_k, %bg_s  : i64
      %bg_stride = llvm.shl %bg_c1, %bg_j  : i64
      %bg_partner = llvm.xor %bg_tid, %bg_stride  : i64
      %bg_kp1 = llvm.add %bg_k, %bg_c1  : i64
      %bg_shifted = llvm.lshr %bg_tid, %bg_kp1  : i64
      %bg_bit = llvm.and %bg_shifted, %bg_c1  : i64
      %bg_asc = llvm.icmp "eq" %bg_bit, %bg_c0 : i64
      %bg_low_bit = llvm.and %bg_tid, %bg_stride  : i64
      %bg_is_lower = llvm.icmp "eq" %bg_low_bit, %bg_c0 : i64
      %bg_mvp = llvm.getelementptr %bg_vals[0, %bg_tid] : (!llvm.ptr<3>, i64) -> !llvm.ptr<3>, !llvm.array<{NP} x f32>
      %bg_pvp = llvm.getelementptr %bg_vals[0, %bg_partner] : (!llvm.ptr<3>, i64) -> !llvm.ptr<3>, !llvm.array<{NP} x f32>
      %bg_mv = llvm.load %bg_mvp : !llvm.ptr<3> -> f32
      %bg_pv = llvm.load %bg_pvp : !llvm.ptr<3> -> f32
      %bg_mip = llvm.getelementptr %bg_idxs[0, %bg_tid] : (!llvm.ptr<3>, i64) -> !llvm.ptr<3>, !llvm.array<{NP} x i32>
      %bg_pip = llvm.getelementptr %bg_idxs[0, %bg_partner] : (!llvm.ptr<3>, i64) -> !llvm.ptr<3>, !llvm.array<{NP} x i32>
      %bg_mi = llvm.load %bg_mip : !llvm.ptr<3> -> i32
      %bg_pi = llvm.load %bg_pip : !llvm.ptr<3> -> i32
      nvvm.barrier0
      %bg_lt = llvm.fcmp "olt" %bg_mv, %bg_pv : f32
      %bg_min_v = llvm.select %bg_lt, %bg_mv, %bg_pv : i1, f32
      %bg_max_v = llvm.select %bg_lt, %bg_pv, %bg_mv : i1, f32
      %bg_min_i = llvm.select %bg_lt, %bg_mi, %bg_pi : i1, i32
      %bg_max_i = llvm.select %bg_lt, %bg_pi, %bg_mi : i1, i32
      %bg_want_min = llvm.icmp "eq" %bg_is_lower, %bg_asc : i1
      %bg_rv = llvm.select %bg_want_min, %bg_min_v, %bg_max_v : i1, f32
      %bg_ri = llvm.select %bg_want_min, %bg_min_i, %bg_max_i : i1, i32
      llvm.store %bg_rv, %bg_mvp : f32, !llvm.ptr<3>
      llvm.store %bg_ri, %bg_mip : i32, !llvm.ptr<3>
      nvvm.barrier0
      %bg_ns2 = llvm.add %bg_s, %bg_c1  : i64
      llvm.br ^bg_inner(%bg_ns2 : i64)

    ^bg_outer_next:
      %bg_nk = llvm.add %bg_k, %bg_c1  : i64
      llvm.br ^bg_outer(%bg_nk : i64)

    ^bg_write:
      llvm.cond_br %bg_ok, ^bg_wb, ^bg_done

    ^bg_wb:
      %bg_rip = llvm.getelementptr %bg_idxs[0, %bg_tid] : (!llvm.ptr<3>, i64) -> !llvm.ptr<3>, !llvm.array<{NP} x i32>
      %bg_riv = llvm.load %bg_rip : !llvm.ptr<3> -> i32
      %bg_oi = llvm.add %out_offset, %bg_tid  : i64
      %bg_op = llvm.getelementptr %out_aligned[%bg_oi] : (!llvm.ptr, i64) -> !llvm.ptr, i32
      llvm.store %bg_riv, %bg_op : i32, !llvm.ptr
      llvm.br ^bg_done

    ^bg_done:
      llvm.return
    }}
  }}
}}"""
    return GPUModuleScaffold(text, module_name, name)


def build_descriptor_abi_multiblock_bitonic_sort_gpu_module(
    function: HIRFunction,
    *,
    module_name: str = "remora_gpu",
    kernel_name: str | None = None,
    block_size: int = 1024,
) -> GPUModuleScaffold:
    """Build a multi-block bitonic sort using local sort + global merge steps.

    Phase 1: each block sorts its 1024 elements in shared memory.
    Phase 2+: global compare-swap kernels for cross-block merge stages.
    Double buffering avoids race conditions in global steps.

    Returns a gpu.module with ``{name}_local`` and ``{name}_gstep_{i}``
    kernels.  Supports N up to block_size * 1024.
    """
    from remora.hir import HIRSort
    if not isinstance(function.body, HIRSort):
        raise GPUScaffoldError("GPU multi-block bitonic sort requires HIRSort body")
    if len(function.params) != 1:
        raise GPUScaffoldError("GPU multi-block sort requires one parameter")
    param_type = function.params[0].type
    if not isinstance(param_type, ArrayType) or param_type.element != FLOAT:
        raise GPUScaffoldError("GPU multi-block sort supports rank-1 f32 only")
    if param_type.rank != 1:
        raise GPUScaffoldError("GPU multi-block sort supports rank-1 only")

    N = int(param_type.shape[0].value)
    BS = block_size
    NP = 1
    while NP < N:
        NP *= 2
    if NP <= BS:
        raise GPUScaffoldError("Use single-block sort for N <= block_size")
    num_blocks = (NP + BS - 1) // BS
    if num_blocks > 1024:
        raise GPUScaffoldError(f"N={N} too large for multi-block bitonic sort")
    num_stages = NP.bit_length() - 1
    local_stages = BS.bit_length() - 1

    base = kernel_name or f"remora_{function.name}_sort"
    local_name = f"{base}_local"
    _validate_scaffold_names(module_name, local_name)

    local_desc = _descriptor_load_lines("sl_in", "%input_desc", 1)
    local_desc.extend(_descriptor_load_lines("sl_out", "%output_desc", 1))

    local_NP = BS
    local_num_stages = local_stages

    kernels_mlir = []

    kernels_mlir.append(f"""    llvm.mlir.global internal @mbs_shmem() {{addr_space = 3 : i32}} : !llvm.array<{BS} x f32>

    llvm.func @{local_name}(%input_desc: !llvm.ptr, %output_desc: !llvm.ptr) attributes {{gpu.kernel, nvvm.kernel}} {{
{chr(10).join(local_desc)}
      %sl_tid32 = nvvm.read.ptx.sreg.tid.x : i32
      %sl_tid = llvm.sext %sl_tid32 : i32 to i64
      %sl_bid32 = nvvm.read.ptx.sreg.ctaid.x : i32
      %sl_bid = llvm.sext %sl_bid32 : i32 to i64
      %sl_BS = llvm.mlir.constant({BS} : index) : i64
      %sl_N = llvm.mlir.constant({N} : index) : i64
      %sl_NP = llvm.mlir.constant({NP} : index) : i64
      %sl_base = llvm.mul %sl_bid, %sl_BS  : i64
      %sl_gidx = llvm.add %sl_base, %sl_tid  : i64
      %sl_ok = llvm.icmp "ult" %sl_gidx, %sl_N : i64
      %sl_shmem = llvm.mlir.addressof @mbs_shmem : !llvm.ptr<3>
      %sl_inf = llvm.mlir.constant(3.40282347e+38 : f32) : f32
      llvm.cond_br %sl_ok, ^sl_ld, ^sl_pad

    ^sl_ld:
      %sl_si = llvm.add %sl_in_offset, %sl_gidx  : i64
      %sl_sp = llvm.getelementptr %sl_in_aligned[%sl_si] : (!llvm.ptr, i64) -> !llvm.ptr, f32
      %sl_val = llvm.load %sl_sp : !llvm.ptr -> f32
      %sl_mp = llvm.getelementptr %sl_shmem[0, %sl_tid] : (!llvm.ptr<3>, i64) -> !llvm.ptr<3>, !llvm.array<{BS} x f32>
      llvm.store %sl_val, %sl_mp : f32, !llvm.ptr<3>
      llvm.br ^sl_sync0

    ^sl_pad:
      %sl_mp2 = llvm.getelementptr %sl_shmem[0, %sl_tid] : (!llvm.ptr<3>, i64) -> !llvm.ptr<3>, !llvm.array<{BS} x f32>
      llvm.store %sl_inf, %sl_mp2 : f32, !llvm.ptr<3>
      llvm.br ^sl_sync0

    ^sl_sync0:
      nvvm.barrier0
      %sl_c0 = llvm.mlir.constant(0 : index) : i64
      %sl_c1 = llvm.mlir.constant(1 : index) : i64
      %sl_ns = llvm.mlir.constant({local_num_stages} : index) : i64
      llvm.br ^sl_outer(%sl_c0 : i64)

    ^sl_outer(%sl_k: i64):
      %sl_kd = llvm.icmp "uge" %sl_k, %sl_ns : i64
      llvm.cond_br %sl_kd, ^sl_write, ^sl_inner_s
    ^sl_inner_s:
      llvm.br ^sl_inner(%sl_c0 : i64)
    ^sl_inner(%sl_s: i64):
      %sl_sd = llvm.icmp "sgt" %sl_s, %sl_k : i64
      llvm.cond_br %sl_sd, ^sl_outer_n, ^sl_step
    ^sl_step:
      %sl_j = llvm.sub %sl_k, %sl_s  : i64
      %sl_stride = llvm.shl %sl_c1, %sl_j  : i64
      %sl_partner = llvm.xor %sl_tid, %sl_stride  : i64
      %sl_kp1 = llvm.add %sl_k, %sl_c1  : i64
      %sl_shifted = llvm.lshr %sl_tid, %sl_kp1  : i64
      %sl_bit = llvm.and %sl_shifted, %sl_c1  : i64
      %sl_asc = llvm.icmp "eq" %sl_bit, %sl_c0 : i64
      %sl_low = llvm.and %sl_tid, %sl_stride  : i64
      %sl_is_l = llvm.icmp "eq" %sl_low, %sl_c0 : i64
      %sl_mep = llvm.getelementptr %sl_shmem[0, %sl_tid] : (!llvm.ptr<3>, i64) -> !llvm.ptr<3>, !llvm.array<{BS} x f32>
      %sl_pap = llvm.getelementptr %sl_shmem[0, %sl_partner] : (!llvm.ptr<3>, i64) -> !llvm.ptr<3>, !llvm.array<{BS} x f32>
      %sl_mev = llvm.load %sl_mep : !llvm.ptr<3> -> f32
      %sl_pav = llvm.load %sl_pap : !llvm.ptr<3> -> f32
      nvvm.barrier0
      %sl_lt = llvm.fcmp "olt" %sl_mev, %sl_pav : f32
      %sl_min = llvm.select %sl_lt, %sl_mev, %sl_pav : i1, f32
      %sl_max = llvm.select %sl_lt, %sl_pav, %sl_mev : i1, f32
      %sl_wm = llvm.icmp "eq" %sl_is_l, %sl_asc : i1
      %sl_res = llvm.select %sl_wm, %sl_min, %sl_max : i1, f32
      llvm.store %sl_res, %sl_mep : f32, !llvm.ptr<3>
      nvvm.barrier0
      %sl_ns2 = llvm.add %sl_s, %sl_c1  : i64
      llvm.br ^sl_inner(%sl_ns2 : i64)
    ^sl_outer_n:
      %sl_nk = llvm.add %sl_k, %sl_c1  : i64
      llvm.br ^sl_outer(%sl_nk : i64)

    ^sl_write:
      %sl_rp = llvm.getelementptr %sl_shmem[0, %sl_tid] : (!llvm.ptr<3>, i64) -> !llvm.ptr<3>, !llvm.array<{BS} x f32>
      %sl_rv = llvm.load %sl_rp : !llvm.ptr<3> -> f32
      %sl_oi = llvm.add %sl_out_offset, %sl_gidx  : i64
      %sl_op = llvm.getelementptr %sl_out_aligned[%sl_oi] : (!llvm.ptr, i64) -> !llvm.ptr, f32
      llvm.store %sl_rv, %sl_op : f32, !llvm.ptr
      llvm.br ^sl_done

    ^sl_done:
      llvm.return
    }}""")

    step_idx = 0
    global_steps_info: list[tuple[str, int, int]] = []
    for k in range(local_stages, num_stages):
        for j in range(k, -1, -1):
            stride = 1 << j
            sort_block = 1 << (k + 1)
            sname = f"{base}_gstep_{step_idx}"
            global_steps_info.append((sname, stride, sort_block))

            gs_desc = _descriptor_load_lines(f"g{step_idx}_in", "%input_desc", 1)
            gs_desc.extend(_descriptor_load_lines(f"g{step_idx}_out", "%output_desc", 1))

            kernels_mlir.append(f"""
    llvm.func @{sname}(%input_desc: !llvm.ptr, %output_desc: !llvm.ptr) attributes {{gpu.kernel, nvvm.kernel}} {{
{chr(10).join(gs_desc)}
      %g{step_idx}_tid32 = nvvm.read.ptx.sreg.tid.x : i32
      %g{step_idx}_tid = llvm.sext %g{step_idx}_tid32 : i32 to i64
      %g{step_idx}_bid32 = nvvm.read.ptx.sreg.ctaid.x : i32
      %g{step_idx}_bid = llvm.sext %g{step_idx}_bid32 : i32 to i64
      %g{step_idx}_bdim32 = nvvm.read.ptx.sreg.ntid.x : i32
      %g{step_idx}_bdim = llvm.sext %g{step_idx}_bdim32 : i32 to i64
      %g{step_idx}_base = llvm.mul %g{step_idx}_bid, %g{step_idx}_bdim  : i64
      %g{step_idx}_gid = llvm.add %g{step_idx}_base, %g{step_idx}_tid  : i64
      %g{step_idx}_NP = llvm.mlir.constant({NP} : index) : i64
      %g{step_idx}_ok = llvm.icmp "ult" %g{step_idx}_gid, %g{step_idx}_NP : i64
      llvm.cond_br %g{step_idx}_ok, ^g{step_idx}_work, ^g{step_idx}_done
    ^g{step_idx}_work:
      %g{step_idx}_c0 = llvm.mlir.constant(0 : index) : i64
      %g{step_idx}_c1 = llvm.mlir.constant(1 : index) : i64
      %g{step_idx}_stride = llvm.mlir.constant({stride} : index) : i64
      %g{step_idx}_sb = llvm.mlir.constant({sort_block} : index) : i64
      %g{step_idx}_partner = llvm.xor %g{step_idx}_gid, %g{step_idx}_stride  : i64
      %g{step_idx}_div = llvm.udiv %g{step_idx}_gid, %g{step_idx}_sb  : i64
      %g{step_idx}_bit = llvm.and %g{step_idx}_div, %g{step_idx}_c1  : i64
      %g{step_idx}_asc = llvm.icmp "eq" %g{step_idx}_bit, %g{step_idx}_c0 : i64
      %g{step_idx}_low = llvm.and %g{step_idx}_gid, %g{step_idx}_stride  : i64
      %g{step_idx}_is_l = llvm.icmp "eq" %g{step_idx}_low, %g{step_idx}_c0 : i64
      %g{step_idx}_me_si = llvm.add %g{step_idx}_in_offset, %g{step_idx}_gid  : i64
      %g{step_idx}_me_sp = llvm.getelementptr %g{step_idx}_in_aligned[%g{step_idx}_me_si] : (!llvm.ptr, i64) -> !llvm.ptr, f32
      %g{step_idx}_me_v = llvm.load %g{step_idx}_me_sp : !llvm.ptr -> f32
      %g{step_idx}_pa_si = llvm.add %g{step_idx}_in_offset, %g{step_idx}_partner  : i64
      %g{step_idx}_pa_sp = llvm.getelementptr %g{step_idx}_in_aligned[%g{step_idx}_pa_si] : (!llvm.ptr, i64) -> !llvm.ptr, f32
      %g{step_idx}_pa_v = llvm.load %g{step_idx}_pa_sp : !llvm.ptr -> f32
      %g{step_idx}_lt = llvm.fcmp "olt" %g{step_idx}_me_v, %g{step_idx}_pa_v : f32
      %g{step_idx}_min = llvm.select %g{step_idx}_lt, %g{step_idx}_me_v, %g{step_idx}_pa_v : i1, f32
      %g{step_idx}_max = llvm.select %g{step_idx}_lt, %g{step_idx}_pa_v, %g{step_idx}_me_v : i1, f32
      %g{step_idx}_wm = llvm.icmp "eq" %g{step_idx}_is_l, %g{step_idx}_asc : i1
      %g{step_idx}_res = llvm.select %g{step_idx}_wm, %g{step_idx}_min, %g{step_idx}_max : i1, f32
      %g{step_idx}_oi = llvm.add %g{step_idx}_out_offset, %g{step_idx}_gid  : i64
      %g{step_idx}_op = llvm.getelementptr %g{step_idx}_out_aligned[%g{step_idx}_oi] : (!llvm.ptr, i64) -> !llvm.ptr, f32
      llvm.store %g{step_idx}_res, %g{step_idx}_op : f32, !llvm.ptr
      llvm.br ^g{step_idx}_done
    ^g{step_idx}_done:
      llvm.return
    }}""")
            step_idx += 1

    all_kernels = "\n".join(kernels_mlir)
    text = f"""module {{
  gpu.module @{module_name} {{
{all_kernels}
  }}
}}"""
    last_kernel = global_steps_info[-1][0] if global_steps_info else local_name
    return GPUModuleScaffold(text, module_name, last_kernel)


def build_descriptor_abi_multiblock_bitonic_grade_gpu_module(
    function: HIRFunction,
    *,
    module_name: str = "remora_gpu",
    kernel_name: str | None = None,
    block_size: int = 1024,
) -> GPUModuleScaffold:
    """Build a multi-block bitonic grade (argsort) using value-lookup merge.

    Phase 1: pad input values to next power-of-2 with +inf.
    Phase 2: each block sorts (value, index) pairs in shared memory.
    Phase 3+: global merge steps compare values[idx] and swap indices.
    Double-buffered i32 index arrays avoid race conditions.
    """
    from remora.hir import HIRGrade
    if not isinstance(function.body, HIRGrade):
        raise GPUScaffoldError("GPU multi-block grade requires HIRGrade body")
    if len(function.params) != 1:
        raise GPUScaffoldError("GPU multi-block grade requires one parameter")
    param_type = function.params[0].type
    if not isinstance(param_type, ArrayType) or param_type.element != FLOAT:
        raise GPUScaffoldError("GPU multi-block grade supports rank-1 f32 only")
    if param_type.rank != 1:
        raise GPUScaffoldError("GPU multi-block grade supports rank-1 only")

    N = int(param_type.shape[0].value)
    BS = block_size
    NP = 1
    while NP < N:
        NP *= 2
    if NP <= BS:
        raise GPUScaffoldError("Use single-block grade for N <= block_size")
    num_blocks = NP // BS
    if num_blocks > 1024:
        raise GPUScaffoldError(f"N={N} too large for multi-block grade")
    num_stages = NP.bit_length() - 1
    local_stages = BS.bit_length() - 1

    base = kernel_name or f"remora_{function.name}_grade"
    pad_name = f"{base}_pad"
    local_name = f"{base}_local"
    _validate_scaffold_names(module_name, pad_name)

    # Descriptor load helpers
    pad_in = _descriptor_load_lines("pi", "%input_desc", 1)
    pad_out = _descriptor_load_lines("po", "%output_desc", 1)
    lg_vals = _descriptor_load_lines("lv", "%values_desc", 1)
    lg_idx = _descriptor_load_lines("li", "%indices_desc", 1)

    kernels_mlir = []

    # ── Kernel 1: pad values ──
    kernels_mlir.append(f"""
    llvm.func @{pad_name}(%input_desc: !llvm.ptr, %output_desc: !llvm.ptr) attributes {{gpu.kernel, nvvm.kernel}} {{
{chr(10).join(pad_in + pad_out)}
      %p_tid32 = nvvm.read.ptx.sreg.tid.x : i32
      %p_tid = llvm.sext %p_tid32 : i32 to i64
      %p_bid32 = nvvm.read.ptx.sreg.ctaid.x : i32
      %p_bid = llvm.sext %p_bid32 : i32 to i64
      %p_bdim32 = nvvm.read.ptx.sreg.ntid.x : i32
      %p_bdim = llvm.sext %p_bdim32 : i32 to i64
      %p_base = llvm.mul %p_bid, %p_bdim  : i64
      %p_gid = llvm.add %p_base, %p_tid  : i64
      %p_NP = llvm.mlir.constant({NP} : index) : i64
      %p_N = llvm.mlir.constant({N} : index) : i64
      %p_ok = llvm.icmp "ult" %p_gid, %p_NP : i64
      %p_inf = llvm.mlir.constant(3.40282347e+38 : f32) : f32
      llvm.cond_br %p_ok, ^p_work, ^p_done
    ^p_work:
      %p_in_range = llvm.icmp "ult" %p_gid, %p_N : i64
      llvm.cond_br %p_in_range, ^p_copy, ^p_pad
    ^p_copy:
      %p_si = llvm.add %pi_offset, %p_gid  : i64
      %p_sp = llvm.getelementptr %pi_aligned[%p_si] : (!llvm.ptr, i64) -> !llvm.ptr, f32
      %p_val = llvm.load %p_sp : !llvm.ptr -> f32
      %p_oi = llvm.add %po_offset, %p_gid  : i64
      %p_op = llvm.getelementptr %po_aligned[%p_oi] : (!llvm.ptr, i64) -> !llvm.ptr, f32
      llvm.store %p_val, %p_op : f32, !llvm.ptr
      llvm.br ^p_done
    ^p_pad:
      %p_oi2 = llvm.add %po_offset, %p_gid  : i64
      %p_op2 = llvm.getelementptr %po_aligned[%p_oi2] : (!llvm.ptr, i64) -> !llvm.ptr, f32
      llvm.store %p_inf, %p_op2 : f32, !llvm.ptr
      llvm.br ^p_done
    ^p_done:
      llvm.return
    }}""")

    # ── Kernel 2: local grade (per-block bitonic sort on value+index pairs) ──
    kernels_mlir.append(f"""
    llvm.mlir.global internal @mbg_vals() {{addr_space = 3 : i32}} : !llvm.array<{BS} x f32>
    llvm.mlir.global internal @mbg_idxs() {{addr_space = 3 : i32}} : !llvm.array<{BS} x i32>
    llvm.func @{local_name}(%values_desc: !llvm.ptr, %indices_desc: !llvm.ptr) attributes {{gpu.kernel, nvvm.kernel}} {{
{chr(10).join(lg_vals + lg_idx)}
      %lg_tid32 = nvvm.read.ptx.sreg.tid.x : i32
      %lg_tid = llvm.sext %lg_tid32 : i32 to i64
      %lg_bid32 = nvvm.read.ptx.sreg.ctaid.x : i32
      %lg_bid = llvm.sext %lg_bid32 : i32 to i64
      %lg_BS = llvm.mlir.constant({BS} : index) : i64
      %lg_base = llvm.mul %lg_bid, %lg_BS  : i64
      %lg_gid = llvm.add %lg_base, %lg_tid  : i64
      %lg_sv = llvm.mlir.addressof @mbg_vals : !llvm.ptr<3>
      %lg_si = llvm.mlir.addressof @mbg_idxs : !llvm.ptr<3>
      %lg_vi = llvm.add %lv_offset, %lg_gid  : i64
      %lg_vp = llvm.getelementptr %lv_aligned[%lg_vi] : (!llvm.ptr, i64) -> !llvm.ptr, f32
      %lg_val = llvm.load %lg_vp : !llvm.ptr -> f32
      %lg_svp = llvm.getelementptr %lg_sv[0, %lg_tid] : (!llvm.ptr<3>, i64) -> !llvm.ptr<3>, !llvm.array<{BS} x f32>
      llvm.store %lg_val, %lg_svp : f32, !llvm.ptr<3>
      %lg_gid32 = llvm.trunc %lg_gid : i64 to i32
      %lg_sip = llvm.getelementptr %lg_si[0, %lg_tid] : (!llvm.ptr<3>, i64) -> !llvm.ptr<3>, !llvm.array<{BS} x i32>
      llvm.store %lg_gid32, %lg_sip : i32, !llvm.ptr<3>
      nvvm.barrier0
      %lg_c0 = llvm.mlir.constant(0 : index) : i64
      %lg_c1 = llvm.mlir.constant(1 : index) : i64
      %lg_ns = llvm.mlir.constant({local_stages} : index) : i64
      llvm.br ^lg_outer(%lg_c0 : i64)
    ^lg_outer(%lg_k: i64):
      %lg_kd = llvm.icmp "uge" %lg_k, %lg_ns : i64
      llvm.cond_br %lg_kd, ^lg_write, ^lg_is
    ^lg_is:
      llvm.br ^lg_inner(%lg_c0 : i64)
    ^lg_inner(%lg_s: i64):
      %lg_sd = llvm.icmp "sgt" %lg_s, %lg_k : i64
      llvm.cond_br %lg_sd, ^lg_on, ^lg_step
    ^lg_step:
      %lg_j = llvm.sub %lg_k, %lg_s  : i64
      %lg_stride = llvm.shl %lg_c1, %lg_j  : i64
      %lg_partner = llvm.xor %lg_tid, %lg_stride  : i64
      %lg_kp1 = llvm.add %lg_k, %lg_c1  : i64
      %lg_sh = llvm.lshr %lg_tid, %lg_kp1  : i64
      %lg_bit = llvm.and %lg_sh, %lg_c1  : i64
      %lg_asc = llvm.icmp "eq" %lg_bit, %lg_c0 : i64
      %lg_low = llvm.and %lg_tid, %lg_stride  : i64
      %lg_is_l = llvm.icmp "eq" %lg_low, %lg_c0 : i64
      %lg_mvp = llvm.getelementptr %lg_sv[0, %lg_tid] : (!llvm.ptr<3>, i64) -> !llvm.ptr<3>, !llvm.array<{BS} x f32>
      %lg_pvp = llvm.getelementptr %lg_sv[0, %lg_partner] : (!llvm.ptr<3>, i64) -> !llvm.ptr<3>, !llvm.array<{BS} x f32>
      %lg_mv = llvm.load %lg_mvp : !llvm.ptr<3> -> f32
      %lg_pv = llvm.load %lg_pvp : !llvm.ptr<3> -> f32
      %lg_mip = llvm.getelementptr %lg_si[0, %lg_tid] : (!llvm.ptr<3>, i64) -> !llvm.ptr<3>, !llvm.array<{BS} x i32>
      %lg_pip = llvm.getelementptr %lg_si[0, %lg_partner] : (!llvm.ptr<3>, i64) -> !llvm.ptr<3>, !llvm.array<{BS} x i32>
      %lg_mi = llvm.load %lg_mip : !llvm.ptr<3> -> i32
      %lg_pi = llvm.load %lg_pip : !llvm.ptr<3> -> i32
      nvvm.barrier0
      %lg_lt = llvm.fcmp "olt" %lg_mv, %lg_pv : f32
      %lg_minv = llvm.select %lg_lt, %lg_mv, %lg_pv : i1, f32
      %lg_maxv = llvm.select %lg_lt, %lg_pv, %lg_mv : i1, f32
      %lg_mini = llvm.select %lg_lt, %lg_mi, %lg_pi : i1, i32
      %lg_maxi = llvm.select %lg_lt, %lg_pi, %lg_mi : i1, i32
      %lg_wm = llvm.icmp "eq" %lg_is_l, %lg_asc : i1
      %lg_rv = llvm.select %lg_wm, %lg_minv, %lg_maxv : i1, f32
      %lg_ri = llvm.select %lg_wm, %lg_mini, %lg_maxi : i1, i32
      llvm.store %lg_rv, %lg_mvp : f32, !llvm.ptr<3>
      llvm.store %lg_ri, %lg_mip : i32, !llvm.ptr<3>
      nvvm.barrier0
      %lg_ns2 = llvm.add %lg_s, %lg_c1  : i64
      llvm.br ^lg_inner(%lg_ns2 : i64)
    ^lg_on:
      %lg_nk = llvm.add %lg_k, %lg_c1  : i64
      llvm.br ^lg_outer(%lg_nk : i64)
    ^lg_write:
      %lg_rip = llvm.getelementptr %lg_si[0, %lg_tid] : (!llvm.ptr<3>, i64) -> !llvm.ptr<3>, !llvm.array<{BS} x i32>
      %lg_riv = llvm.load %lg_rip : !llvm.ptr<3> -> i32
      %lg_oi = llvm.add %li_offset, %lg_gid  : i64
      %lg_op = llvm.getelementptr %li_aligned[%lg_oi] : (!llvm.ptr, i64) -> !llvm.ptr, i32
      llvm.store %lg_riv, %lg_op : i32, !llvm.ptr
      llvm.br ^lg_done
    ^lg_done:
      llvm.return
    }}""")

    # ── Global merge step kernels ──
    step_idx = 0
    global_steps_info: list[str] = []
    for k in range(local_stages, num_stages):
        for j in range(k, -1, -1):
            stride = 1 << j
            sort_block = 1 << (k + 1)
            sname = f"{base}_gstep_{step_idx}"
            global_steps_info.append(sname)

            gs_v = _descriptor_load_lines(f"gv{step_idx}", "%values_desc", 1)
            gs_i = _descriptor_load_lines(f"gi{step_idx}", "%indices_in_desc", 1)
            gs_o = _descriptor_load_lines(f"go{step_idx}", "%indices_out_desc", 1)

            kernels_mlir.append(f"""
    llvm.func @{sname}(%values_desc: !llvm.ptr, %indices_in_desc: !llvm.ptr, %indices_out_desc: !llvm.ptr) attributes {{gpu.kernel, nvvm.kernel}} {{
{chr(10).join(gs_v + gs_i + gs_o)}
      %g{step_idx}_tid32 = nvvm.read.ptx.sreg.tid.x : i32
      %g{step_idx}_tid = llvm.sext %g{step_idx}_tid32 : i32 to i64
      %g{step_idx}_bid32 = nvvm.read.ptx.sreg.ctaid.x : i32
      %g{step_idx}_bid = llvm.sext %g{step_idx}_bid32 : i32 to i64
      %g{step_idx}_bdim32 = nvvm.read.ptx.sreg.ntid.x : i32
      %g{step_idx}_bdim = llvm.sext %g{step_idx}_bdim32 : i32 to i64
      %g{step_idx}_base = llvm.mul %g{step_idx}_bid, %g{step_idx}_bdim  : i64
      %g{step_idx}_gid = llvm.add %g{step_idx}_base, %g{step_idx}_tid  : i64
      %g{step_idx}_NP = llvm.mlir.constant({NP} : index) : i64
      %g{step_idx}_ok = llvm.icmp "ult" %g{step_idx}_gid, %g{step_idx}_NP : i64
      llvm.cond_br %g{step_idx}_ok, ^g{step_idx}_work, ^g{step_idx}_done
    ^g{step_idx}_work:
      %g{step_idx}_c0 = llvm.mlir.constant(0 : index) : i64
      %g{step_idx}_c1 = llvm.mlir.constant(1 : index) : i64
      %g{step_idx}_stride = llvm.mlir.constant({stride} : index) : i64
      %g{step_idx}_sb = llvm.mlir.constant({sort_block} : index) : i64
      %g{step_idx}_partner = llvm.xor %g{step_idx}_gid, %g{step_idx}_stride  : i64
      %g{step_idx}_div = llvm.udiv %g{step_idx}_gid, %g{step_idx}_sb  : i64
      %g{step_idx}_bit = llvm.and %g{step_idx}_div, %g{step_idx}_c1  : i64
      %g{step_idx}_asc = llvm.icmp "eq" %g{step_idx}_bit, %g{step_idx}_c0 : i64
      %g{step_idx}_low = llvm.and %g{step_idx}_gid, %g{step_idx}_stride  : i64
      %g{step_idx}_is_l = llvm.icmp "eq" %g{step_idx}_low, %g{step_idx}_c0 : i64
      %g{step_idx}_me_ii = llvm.add %gi{step_idx}_offset, %g{step_idx}_gid  : i64
      %g{step_idx}_me_ip = llvm.getelementptr %gi{step_idx}_aligned[%g{step_idx}_me_ii] : (!llvm.ptr, i64) -> !llvm.ptr, i32
      %g{step_idx}_me_idx32 = llvm.load %g{step_idx}_me_ip : !llvm.ptr -> i32
      %g{step_idx}_me_idx = llvm.sext %g{step_idx}_me_idx32 : i32 to i64
      %g{step_idx}_pa_ii = llvm.add %gi{step_idx}_offset, %g{step_idx}_partner  : i64
      %g{step_idx}_pa_ip = llvm.getelementptr %gi{step_idx}_aligned[%g{step_idx}_pa_ii] : (!llvm.ptr, i64) -> !llvm.ptr, i32
      %g{step_idx}_pa_idx32 = llvm.load %g{step_idx}_pa_ip : !llvm.ptr -> i32
      %g{step_idx}_pa_idx = llvm.sext %g{step_idx}_pa_idx32 : i32 to i64
      %g{step_idx}_me_vi = llvm.add %gv{step_idx}_offset, %g{step_idx}_me_idx  : i64
      %g{step_idx}_me_vp = llvm.getelementptr %gv{step_idx}_aligned[%g{step_idx}_me_vi] : (!llvm.ptr, i64) -> !llvm.ptr, f32
      %g{step_idx}_me_v = llvm.load %g{step_idx}_me_vp : !llvm.ptr -> f32
      %g{step_idx}_pa_vi = llvm.add %gv{step_idx}_offset, %g{step_idx}_pa_idx  : i64
      %g{step_idx}_pa_vp = llvm.getelementptr %gv{step_idx}_aligned[%g{step_idx}_pa_vi] : (!llvm.ptr, i64) -> !llvm.ptr, f32
      %g{step_idx}_pa_v = llvm.load %g{step_idx}_pa_vp : !llvm.ptr -> f32
      %g{step_idx}_lt = llvm.fcmp "olt" %g{step_idx}_me_v, %g{step_idx}_pa_v : f32
      %g{step_idx}_min_i = llvm.select %g{step_idx}_lt, %g{step_idx}_me_idx32, %g{step_idx}_pa_idx32 : i1, i32
      %g{step_idx}_max_i = llvm.select %g{step_idx}_lt, %g{step_idx}_pa_idx32, %g{step_idx}_me_idx32 : i1, i32
      %g{step_idx}_wm = llvm.icmp "eq" %g{step_idx}_is_l, %g{step_idx}_asc : i1
      %g{step_idx}_res = llvm.select %g{step_idx}_wm, %g{step_idx}_min_i, %g{step_idx}_max_i : i1, i32
      %g{step_idx}_oi = llvm.add %go{step_idx}_offset, %g{step_idx}_gid  : i64
      %g{step_idx}_op = llvm.getelementptr %go{step_idx}_aligned[%g{step_idx}_oi] : (!llvm.ptr, i64) -> !llvm.ptr, i32
      llvm.store %g{step_idx}_res, %g{step_idx}_op : i32, !llvm.ptr
      llvm.br ^g{step_idx}_done
    ^g{step_idx}_done:
      llvm.return
    }}""")
            step_idx += 1

    all_kernels = "\n".join(kernels_mlir)
    text = f"""module {{
  gpu.module @{module_name} {{
{all_kernels}
  }}
}}"""
    last_kernel = global_steps_info[-1] if global_steps_info else local_name
    return GPUModuleScaffold(text, module_name, last_kernel)


# ---------------------------------------------------------------------------
# IndicesOf kernel
# ---------------------------------------------------------------------------


def build_descriptor_abi_indices_of_gpu_module(
    function: HIRFunction,
    *,
    module_name: str = "remora_gpu",
    kernel_name: str | None = None,
) -> GPUModuleScaffold:
    """Build a descriptor-ABI GPU module for coordinate tensor generation.

    For an input of shape [D0, D1, ..., Dk], the output has shape
    [D0, D1, ..., Dk, k+1] where out[i0,i1,...,ik, d] = i_d.
    One thread per input-shape position; each thread writes k+1 values.
    """
    if not isinstance(function.body, HIRIndicesOf):
        raise GPUScaffoldError("GPU indices-of requires HIRIndicesOf body")
    if len(function.params) != 1:
        raise GPUScaffoldError("GPU indices-of requires exactly one parameter")
    param_type = function.params[0].type
    if not isinstance(param_type, ArrayType):
        raise GPUScaffoldError("GPU indices-of requires an array parameter")

    input_shape = tuple(int(d.value) for d in param_type.shape)
    input_rank = len(input_shape)
    input_total = 1
    for d in input_shape:
        input_total *= d

    result_type = function.body.result_type
    out_shape = tuple(int(d.value) for d in result_type.shape)
    out_rank = len(out_shape)

    name = kernel_name or f"remora_{function.name}_indices"
    _validate_scaffold_names(module_name, name)

    desc_lines = _descriptor_load_lines("in", "%input_desc", input_rank)
    desc_lines.extend(_descriptor_load_lines("out", "%output_desc", out_rank))

    planes: list[int] = []
    for k in range(input_rank):
        p = 1
        for d in input_shape[k + 1:]:
            p *= d
        planes.append(p)

    out_planes: list[int] = []
    for k in range(out_rank):
        p = 1
        for d in out_shape[k + 1:]:
            p *= d
        out_planes.append(p)

    body_lines = desc_lines + [
        "      %io_tid32 = nvvm.read.ptx.sreg.tid.x : i32",
        "      %io_tid = llvm.sext %io_tid32 : i32 to i64",
        "      %io_bid32 = nvvm.read.ptx.sreg.ctaid.x : i32",
        "      %io_bid = llvm.sext %io_bid32 : i32 to i64",
        "      %io_bdim32 = nvvm.read.ptx.sreg.ntid.x : i32",
        "      %io_bdim = llvm.sext %io_bdim32 : i32 to i64",
        "      %io_blk = llvm.mul %io_bid, %io_bdim  : i64",
        "      %io_flat = llvm.add %io_blk, %io_tid  : i64",
        f"      %io_total = llvm.mlir.constant({input_total} : index) : i64",
        '      %io_ok = llvm.icmp "ult" %io_flat, %io_total : i64',
        "      llvm.cond_br %io_ok, ^io_body, ^io_done",
        "    ^io_body:",
    ]

    remainder = "%io_flat"
    for k in range(input_rank):
        if k < input_rank - 1:
            plane_ssa = f"%io_plane{k}"
            body_lines.append(f"      {plane_ssa} = llvm.mlir.constant({planes[k]} : index) : i64")
            coord_ssa = f"%io_c{k}"
            body_lines.append(f"      {coord_ssa} = llvm.udiv {remainder}, {plane_ssa}  : i64")
            next_rem = f"%io_rem{k}"
            body_lines.append(f"      {next_rem} = llvm.urem {remainder}, {plane_ssa}  : i64")
            remainder = next_rem
        else:
            coord_ssa = f"%io_c{k}"
            body_lines.append(f"      {coord_ssa} = llvm.add {remainder}, %io_zero_c  : i64")
            body_lines.append(f"      %io_zero_c = llvm.mlir.constant(0 : index) : i64")

    for k in range(input_rank):
        base_offset = "%out_offset"
        for ax in range(input_rank):
            oplane_ssa = f"%io_oplane_{k}_{ax}"
            body_lines.append(f"      {oplane_ssa} = llvm.mlir.constant({out_planes[ax]} : index) : i64")
            term = f"%io_oterm_{k}_{ax}"
            body_lines.append(f"      {term} = llvm.mul %io_c{ax}, {oplane_ssa}  : i64")
            noff = f"%io_noff_{k}_{ax}"
            body_lines.append(f"      {noff} = llvm.add {base_offset}, {term}  : i64")
            base_offset = noff

        last_plane = f"%io_olast_{k}"
        body_lines.append(f"      {last_plane} = llvm.mlir.constant({out_planes[input_rank]} : index) : i64")
        k_const = f"%io_kc_{k}"
        body_lines.append(f"      {k_const} = llvm.mlir.constant({k} : index) : i64")
        k_term = f"%io_kt_{k}"
        body_lines.append(f"      {k_term} = llvm.mul {k_const}, {last_plane}  : i64")
        final_off = f"%io_foff_{k}"
        body_lines.append(f"      {final_off} = llvm.add {base_offset}, {k_term}  : i64")

        coord_i32 = f"%io_ci32_{k}"
        body_lines.append(f"      {coord_i32} = llvm.trunc %io_c{k} : i64 to i32")
        ptr = f"%io_ptr_{k}"
        body_lines.append(f"      {ptr} = llvm.getelementptr %out_aligned[{final_off}] : (!llvm.ptr, i64) -> !llvm.ptr, i32")
        body_lines.append(f"      llvm.store {coord_i32}, {ptr} : i32, !llvm.ptr")

    body_lines.extend([
        "      llvm.br ^io_done",
        "    ^io_done:",
    ])

    text = f"""module {{
  gpu.module @{module_name} {{
    llvm.func @{name}(%input_desc: !llvm.ptr, %output_desc: !llvm.ptr) attributes {{gpu.kernel, nvvm.kernel}} {{
{chr(10).join(body_lines)}
      llvm.return
    }}
  }}
}}"""
    return GPUModuleScaffold(text, module_name, name)


# ---------------------------------------------------------------------------
# Filter kernel (serial)
# ---------------------------------------------------------------------------


def build_descriptor_abi_filter_gpu_module(
    function: HIRFunction,
    *,
    module_name: str = "remora_gpu",
    kernel_name: str | None = None,
) -> GPUModuleScaffold:
    """Build a descriptor-ABI GPU module for filter (serial, single-thread).

    Thread 0 iterates the input, evaluates a comparison predicate, and
    writes matching elements contiguously to the output.  Output is
    allocated at the input size (upper bound).  Unused trailing positions
    are zeroed.

    A production implementation would use a parallel prefix-sum + scatter
    with multi-kernel orchestration (see docs/FUTURE_WORK.md).
    """
    if not isinstance(function.body, HIRFilter):
        raise GPUScaffoldError("GPU filter requires HIRFilter body")
    if len(function.params) != 1:
        raise GPUScaffoldError("GPU filter requires exactly one array parameter")
    param_type = function.params[0].type
    if not isinstance(param_type, ArrayType) or param_type.element != FLOAT:
        raise GPUScaffoldError("GPU filter supports rank-1 f32 only")
    if param_type.rank != 1:
        raise GPUScaffoldError("GPU filter supports rank-1 only")

    pred = function.body.predicate
    cmp_op = None
    cmp_const = None
    cmp_side = "right"

    if isinstance(pred, HIRPrimCallable):
        cmp_op = pred.op
        if pred.right_arg is not None and isinstance(pred.right_arg, HIRLit):
            cmp_const = float(pred.right_arg.value)
            cmp_side = "right"
        elif pred.left_arg is not None and isinstance(pred.left_arg, HIRLit):
            cmp_const = float(pred.left_arg.value)
            cmp_side = "left"
    elif isinstance(pred, HIRLambda):
        from remora.hir import HIRPrimOp as _PO2
        body = pred.body
        if isinstance(body, _PO2) and len(body.args) == 2:
            raw_op = body.op
            for sfx in ("f", "i", "b"):
                if raw_op.endswith(sfx):
                    raw_op = raw_op[:-1]
                    break
            if raw_op in {"<", "<=", ">", ">=", "==", "!="}:
                cmp_op = raw_op
                a0, a1 = body.args
                if isinstance(a1, HIRLit):
                    cmp_const = float(a1.value)
                    cmp_side = "right"
                elif isinstance(a0, HIRLit):
                    cmp_const = float(a0.value)
                    cmp_side = "left"

    if cmp_op is None or cmp_op not in {"<", "<=", ">", ">=", "==", "!="}:
        raise GPUScaffoldError("GPU filter requires a comparison predicate (primitive or simple lambda)")
    if cmp_const is None:
        raise GPUScaffoldError("GPU filter predicate requires a literal constant")

    N = int(param_type.shape[0].value)
    name = kernel_name or f"remora_{function.name}_filter"
    _validate_scaffold_names(module_name, name)

    pred_map = {"<": "olt", "<=": "ole", ">": "ogt", ">=": "oge", "==": "oeq", "!=": "one"}
    pred_str = pred_map[cmp_op]

    if cmp_side == "left":
        cmp_line = f'      %fl_cmp = llvm.fcmp "{pred_str}" %fl_const, %fl_val : f32'
    else:
        cmp_line = f'      %fl_cmp = llvm.fcmp "{pred_str}" %fl_val, %fl_const : f32'

    desc_lines = _descriptor_load_lines("in", "%input_desc", 1)
    desc_lines.extend(_descriptor_load_lines("out", "%output_desc", 1))

    text = f"""module {{
  gpu.module @{module_name} {{
    llvm.func @{name}(%input_desc: !llvm.ptr, %output_desc: !llvm.ptr) attributes {{gpu.kernel, nvvm.kernel}} {{
{chr(10).join(desc_lines)}
      %fl_tid32 = nvvm.read.ptx.sreg.tid.x : i32
      %fl_tid = llvm.sext %fl_tid32 : i32 to i64
      %fl_z = llvm.mlir.constant(0 : index) : i64
      %fl_is_t0 = llvm.icmp "eq" %fl_tid, %fl_z : i64
      llvm.cond_br %fl_is_t0, ^fl_work, ^fl_done

    ^fl_work:
      %fl_N = llvm.mlir.constant({N} : index) : i64
      %fl_one = llvm.mlir.constant(1 : index) : i64
      %fl_const = llvm.mlir.constant({cmp_const:.6e} : f32) : f32
      %fl_zerof = llvm.mlir.constant(0.000000e+00 : f32) : f32
      llvm.br ^fl_loop(%fl_z, %fl_z : i64, i64)

    ^fl_loop(%fl_i: i64, %fl_wp: i64):
      %fl_ld = llvm.icmp "uge" %fl_i, %fl_N : i64
      llvm.cond_br %fl_ld, ^fl_zero(%fl_wp : i64), ^fl_body

    ^fl_body:
      %fl_si = llvm.add %in_offset, %fl_i  : i64
      %fl_sp = llvm.getelementptr %in_aligned[%fl_si] : (!llvm.ptr, i64) -> !llvm.ptr, f32
      %fl_val = llvm.load %fl_sp : !llvm.ptr -> f32
{cmp_line}
      llvm.cond_br %fl_cmp, ^fl_write, ^fl_skip

    ^fl_write:
      %fl_di = llvm.add %out_offset, %fl_wp  : i64
      %fl_dp = llvm.getelementptr %out_aligned[%fl_di] : (!llvm.ptr, i64) -> !llvm.ptr, f32
      llvm.store %fl_val, %fl_dp : f32, !llvm.ptr
      %fl_nwp = llvm.add %fl_wp, %fl_one : i64
      %fl_ni = llvm.add %fl_i, %fl_one : i64
      llvm.br ^fl_loop(%fl_ni, %fl_nwp : i64, i64)

    ^fl_skip:
      %fl_ni2 = llvm.add %fl_i, %fl_one : i64
      llvm.br ^fl_loop(%fl_ni2, %fl_wp : i64, i64)

    ^fl_zero(%fl_wp2: i64):
      %fl_zd = llvm.icmp "uge" %fl_wp2, %fl_N : i64
      llvm.cond_br %fl_zd, ^fl_done, ^fl_zbody

    ^fl_zbody:
      %fl_zi = llvm.add %out_offset, %fl_wp2  : i64
      %fl_zp = llvm.getelementptr %out_aligned[%fl_zi] : (!llvm.ptr, i64) -> !llvm.ptr, f32
      llvm.store %fl_zerof, %fl_zp : f32, !llvm.ptr
      %fl_znext = llvm.add %fl_wp2, %fl_one : i64
      llvm.br ^fl_zero(%fl_znext : i64)

    ^fl_done:
      llvm.return
    }}
  }}
}}"""
    return GPUModuleScaffold(text, module_name, name)


def build_descriptor_abi_parallel_filter_gpu_module(
    function: HIRFunction,
    *,
    module_name: str = "remora_gpu",
    kernel_name: str | None = None,
) -> GPUModuleScaffold:
    """Build a three-kernel parallel filter using prefix-sum + scatter.

    Returns a gpu.module with three entry points:
    - ``{name}_pred``: evaluate predicate per element → i32 (0/1)
    - ``{name}_scan``: i32 inclusive prefix sum (Hillis-Steele)
    - ``{name}_scatter``: scatter matching elements to output

    Only supports rank-1 f32 inputs with N ≤ 1024.
    """
    if not isinstance(function.body, HIRFilter):
        raise GPUScaffoldError("GPU parallel filter requires HIRFilter body")
    if len(function.params) != 1:
        raise GPUScaffoldError("GPU parallel filter requires one array parameter")
    param_type = function.params[0].type
    if not isinstance(param_type, ArrayType) or param_type.element != FLOAT:
        raise GPUScaffoldError("GPU parallel filter supports rank-1 f32 only")
    if param_type.rank != 1:
        raise GPUScaffoldError("GPU parallel filter supports rank-1 only")

    N = int(param_type.shape[0].value)
    if N > 1024:
        raise GPUScaffoldError("GPU parallel filter requires N ≤ 1024")

    pred = function.body.predicate
    cmp_op = None
    cmp_const = None
    cmp_side = "right"

    if isinstance(pred, HIRPrimCallable):
        cmp_op = pred.op
        if pred.right_arg is not None and isinstance(pred.right_arg, HIRLit):
            cmp_const = float(pred.right_arg.value)
            cmp_side = "right"
        elif pred.left_arg is not None and isinstance(pred.left_arg, HIRLit):
            cmp_const = float(pred.left_arg.value)
            cmp_side = "left"
    elif isinstance(pred, HIRLambda):
        from remora.hir import HIRPrimOp as _PO2
        body = pred.body
        if isinstance(body, _PO2) and len(body.args) == 2:
            raw_op = body.op
            for sfx in ("f", "i", "b"):
                if raw_op.endswith(sfx):
                    raw_op = raw_op[:-1]
                    break
            if raw_op in {"<", "<=", ">", ">=", "==", "!="}:
                cmp_op = raw_op
                a0, a1 = body.args
                if isinstance(a1, HIRLit):
                    cmp_const = float(a1.value)
                    cmp_side = "right"
                elif isinstance(a0, HIRLit):
                    cmp_const = float(a0.value)
                    cmp_side = "left"

    if cmp_op is None or cmp_op not in {"<", "<=", ">", ">=", "==", "!="}:
        raise GPUScaffoldError("GPU parallel filter requires a comparison predicate")
    if cmp_const is None:
        raise GPUScaffoldError("GPU parallel filter predicate requires a literal constant")

    base = kernel_name or f"remora_{function.name}_filter"
    pred_name = f"{base}_pred"
    scan_name = f"{base}_scan"
    scatter_name = f"{base}_scatter"
    _validate_scaffold_names(module_name, pred_name)

    pred_map = {"<": "olt", "<=": "ole", ">": "ogt", ">=": "oge", "==": "oeq", "!=": "one"}
    pred_str = pred_map[cmp_op]

    if cmp_side == "left":
        cmp_line = f'      %fp_cmp = llvm.fcmp "{pred_str}" %fp_const, %fp_val : f32'
    else:
        cmp_line = f'      %fp_cmp = llvm.fcmp "{pred_str}" %fp_val, %fp_const : f32'

    max_d = (N - 1).bit_length() if N > 1 else 1

    k1_desc = _descriptor_load_lines("fp_in", "%input_desc", 1)
    k1_desc.extend(_descriptor_load_lines("fp_pred", "%pred_desc", 1))

    k2_desc = _descriptor_load_lines("fs_pred", "%pred_desc", 1)
    k2_desc.extend(_descriptor_load_lines("fs_scan", "%scan_desc", 1))

    k3_desc = _descriptor_load_lines("fx_in", "%input_desc", 1)
    k3_desc.extend(_descriptor_load_lines("fx_pred", "%pred_desc", 1))
    k3_desc.extend(_descriptor_load_lines("fx_scan", "%scan_desc", 1))
    k3_desc.extend(_descriptor_load_lines("fx_out", "%output_desc", 1))

    text = f"""module {{
  gpu.module @{module_name} {{
    llvm.mlir.global internal @pf_scan_shmem() {{addr_space = 3 : i32}} : !llvm.array<1024 x i32>

    llvm.func @{pred_name}(%input_desc: !llvm.ptr, %pred_desc: !llvm.ptr) attributes {{gpu.kernel, nvvm.kernel}} {{
{chr(10).join(k1_desc)}
      %fp_tid32 = nvvm.read.ptx.sreg.tid.x : i32
      %fp_tid = llvm.sext %fp_tid32 : i32 to i64
      %fp_bid32 = nvvm.read.ptx.sreg.ctaid.x : i32
      %fp_bid = llvm.sext %fp_bid32 : i32 to i64
      %fp_bdim32 = nvvm.read.ptx.sreg.ntid.x : i32
      %fp_bdim = llvm.sext %fp_bdim32 : i32 to i64
      %fp_base = llvm.mul %fp_bid, %fp_bdim  : i64
      %fp_idx = llvm.add %fp_base, %fp_tid  : i64
      %fp_N = llvm.mlir.constant({N} : index) : i64
      %fp_in_bounds = llvm.icmp "ult" %fp_idx, %fp_N : i64
      llvm.cond_br %fp_in_bounds, ^fp_work, ^fp_done

    ^fp_work:
      %fp_const = llvm.mlir.constant({cmp_const:.6e} : f32) : f32
      %fp_si = llvm.add %fp_in_offset, %fp_idx  : i64
      %fp_sp = llvm.getelementptr %fp_in_aligned[%fp_si] : (!llvm.ptr, i64) -> !llvm.ptr, f32
      %fp_val = llvm.load %fp_sp : !llvm.ptr -> f32
{cmp_line}
      %fp_one = llvm.mlir.constant(1 : i32) : i32
      %fp_zero = llvm.mlir.constant(0 : i32) : i32
      %fp_result = llvm.select %fp_cmp, %fp_one, %fp_zero : i1, i32
      %fp_di = llvm.add %fp_pred_offset, %fp_idx  : i64
      %fp_dp = llvm.getelementptr %fp_pred_aligned[%fp_di] : (!llvm.ptr, i64) -> !llvm.ptr, i32
      llvm.store %fp_result, %fp_dp : i32, !llvm.ptr
      llvm.br ^fp_done

    ^fp_done:
      llvm.return
    }}

    llvm.func @{scan_name}(%pred_desc: !llvm.ptr, %scan_desc: !llvm.ptr) attributes {{gpu.kernel, nvvm.kernel}} {{
{chr(10).join(k2_desc)}
      %fs_tid32 = nvvm.read.ptx.sreg.tid.x : i32
      %fs_tid = llvm.sext %fs_tid32 : i32 to i64
      %fs_N = llvm.mlir.constant({N} : index) : i64
      %fs_in_bounds = llvm.icmp "ult" %fs_tid, %fs_N : i64
      %fs_shmem_base = llvm.mlir.addressof @pf_scan_shmem : !llvm.ptr<3>
      %fs_zero_val = llvm.mlir.constant(0 : i32) : i32
      llvm.cond_br %fs_in_bounds, ^fs_load, ^fs_load_oob

    ^fs_load:
      %fs_si = llvm.add %fs_pred_offset, %fs_tid  : i64
      %fs_sp = llvm.getelementptr %fs_pred_aligned[%fs_si] : (!llvm.ptr, i64) -> !llvm.ptr, i32
      %fs_val = llvm.load %fs_sp : !llvm.ptr -> i32
      %fs_me = llvm.getelementptr %fs_shmem_base[0, %fs_tid] : (!llvm.ptr<3>, i64) -> !llvm.ptr<3>, !llvm.array<1024 x i32>
      llvm.store %fs_val, %fs_me : i32, !llvm.ptr<3>
      llvm.br ^fs_after_load

    ^fs_load_oob:
      %fs_me_oob = llvm.getelementptr %fs_shmem_base[0, %fs_tid] : (!llvm.ptr<3>, i64) -> !llvm.ptr<3>, !llvm.array<1024 x i32>
      llvm.store %fs_zero_val, %fs_me_oob : i32, !llvm.ptr<3>
      llvm.br ^fs_after_load

    ^fs_after_load:
      nvvm.barrier0
      %fs_c0 = llvm.mlir.constant(0 : index) : i64
      %fs_c1 = llvm.mlir.constant(1 : index) : i64
      %fs_c2 = llvm.mlir.constant(2 : index) : i64
      %fs_max_d = llvm.mlir.constant({max_d} : index) : i64
      llvm.br ^fs_loop(%fs_c0, %fs_c1 : i64, i64)

    ^fs_loop(%fs_d: i64, %fs_stride: i64):
      %fs_loop_done = llvm.icmp "uge" %fs_d, %fs_max_d : i64
      llvm.cond_br %fs_loop_done, ^fs_write, ^fs_step

    ^fs_step:
      %fs_active = llvm.icmp "uge" %fs_tid, %fs_stride : i64
      %fs_partner_raw = llvm.sub %fs_tid, %fs_stride  : i64
      %fs_safe_partner = llvm.select %fs_active, %fs_partner_raw, %fs_c0 : i1, i64
      %fs_pptr = llvm.getelementptr %fs_shmem_base[0, %fs_safe_partner] : (!llvm.ptr<3>, i64) -> !llvm.ptr<3>, !llvm.array<1024 x i32>
      %fs_pval = llvm.load %fs_pptr : !llvm.ptr<3> -> i32
      %fs_temp = llvm.select %fs_active, %fs_pval, %fs_zero_val : i1, i32
      nvvm.barrier0
      %fs_me2 = llvm.getelementptr %fs_shmem_base[0, %fs_tid] : (!llvm.ptr<3>, i64) -> !llvm.ptr<3>, !llvm.array<1024 x i32>
      %fs_cur = llvm.load %fs_me2 : !llvm.ptr<3> -> i32
      %fs_new = llvm.add %fs_cur, %fs_temp  : i32
      %fs_result = llvm.select %fs_active, %fs_new, %fs_cur : i1, i32
      llvm.store %fs_result, %fs_me2 : i32, !llvm.ptr<3>
      nvvm.barrier0
      %fs_next_d = llvm.add %fs_d, %fs_c1  : i64
      %fs_next_stride = llvm.mul %fs_stride, %fs_c2  : i64
      llvm.br ^fs_loop(%fs_next_d, %fs_next_stride : i64, i64)

    ^fs_write:
      llvm.cond_br %fs_in_bounds, ^fs_write_body, ^fs_done

    ^fs_write_body:
      %fs_me3 = llvm.getelementptr %fs_shmem_base[0, %fs_tid] : (!llvm.ptr<3>, i64) -> !llvm.ptr<3>, !llvm.array<1024 x i32>
      %fs_final = llvm.load %fs_me3 : !llvm.ptr<3> -> i32
      %fs_oi = llvm.add %fs_scan_offset, %fs_tid  : i64
      %fs_op = llvm.getelementptr %fs_scan_aligned[%fs_oi] : (!llvm.ptr, i64) -> !llvm.ptr, i32
      llvm.store %fs_final, %fs_op : i32, !llvm.ptr
      llvm.br ^fs_done

    ^fs_done:
      llvm.return
    }}

    llvm.func @{scatter_name}(%input_desc: !llvm.ptr, %pred_desc: !llvm.ptr, %scan_desc: !llvm.ptr, %output_desc: !llvm.ptr) attributes {{gpu.kernel, nvvm.kernel}} {{
{chr(10).join(k3_desc)}
      %fx_tid32 = nvvm.read.ptx.sreg.tid.x : i32
      %fx_tid = llvm.sext %fx_tid32 : i32 to i64
      %fx_bid32 = nvvm.read.ptx.sreg.ctaid.x : i32
      %fx_bid = llvm.sext %fx_bid32 : i32 to i64
      %fx_bdim32 = nvvm.read.ptx.sreg.ntid.x : i32
      %fx_bdim = llvm.sext %fx_bdim32 : i32 to i64
      %fx_base = llvm.mul %fx_bid, %fx_bdim  : i64
      %fx_idx = llvm.add %fx_base, %fx_tid  : i64
      %fx_N = llvm.mlir.constant({N} : index) : i64
      %fx_in_bounds = llvm.icmp "ult" %fx_idx, %fx_N : i64
      llvm.cond_br %fx_in_bounds, ^fx_work, ^fx_done

    ^fx_work:
      %fx_pi = llvm.add %fx_pred_offset, %fx_idx  : i64
      %fx_pp = llvm.getelementptr %fx_pred_aligned[%fx_pi] : (!llvm.ptr, i64) -> !llvm.ptr, i32
      %fx_pred_val = llvm.load %fx_pp : !llvm.ptr -> i32
      %fx_one_i32 = llvm.mlir.constant(1 : i32) : i32
      %fx_is_match = llvm.icmp "eq" %fx_pred_val, %fx_one_i32 : i32
      llvm.cond_br %fx_is_match, ^fx_scatter, ^fx_done

    ^fx_scatter:
      %fx_sci = llvm.add %fx_scan_offset, %fx_idx  : i64
      %fx_scp = llvm.getelementptr %fx_scan_aligned[%fx_sci] : (!llvm.ptr, i64) -> !llvm.ptr, i32
      %fx_pos_i32 = llvm.load %fx_scp : !llvm.ptr -> i32
      %fx_pos = llvm.sext %fx_pos_i32 : i32 to i64
      %fx_one64 = llvm.mlir.constant(1 : index) : i64
      %fx_out_idx = llvm.sub %fx_pos, %fx_one64  : i64
      %fx_ii = llvm.add %fx_in_offset, %fx_idx  : i64
      %fx_ip = llvm.getelementptr %fx_in_aligned[%fx_ii] : (!llvm.ptr, i64) -> !llvm.ptr, f32
      %fx_val = llvm.load %fx_ip : !llvm.ptr -> f32
      %fx_oi = llvm.add %fx_out_offset, %fx_out_idx  : i64
      %fx_op = llvm.getelementptr %fx_out_aligned[%fx_oi] : (!llvm.ptr, i64) -> !llvm.ptr, f32
      llvm.store %fx_val, %fx_op : f32, !llvm.ptr
      llvm.br ^fx_done

    ^fx_done:
      llvm.return
    }}
  }}
}}"""
    return GPUModuleScaffold(text, module_name, scatter_name)


def build_descriptor_abi_replicate_gpu_module(
    function: HIRFunction,
    *,
    module_name: str = "remora_gpu",
    kernel_name: str | None = None,
    output_upper_bound: int | None = None,
) -> GPUModuleScaffold:
    """Build a descriptor-ABI GPU module for replicate (serial, single-thread).

    Thread 0 iterates the counts array, writing each corresponding value
    element ``counts[i]`` times contiguously to the output.  The output
    descriptor must be pre-allocated at an upper-bound size.

    A production implementation would use a parallel prefix-sum + scatter
    with multi-kernel orchestration (see docs/FUTURE_WORK.md).
    """
    if not isinstance(function.body, HIRReplicate):
        raise GPUScaffoldError("GPU replicate requires HIRReplicate body")
    if len(function.params) != 2:
        raise GPUScaffoldError("GPU replicate requires exactly two parameters (counts, values)")

    counts_type = function.params[0].type
    values_type = function.params[1].type
    if not isinstance(counts_type, ArrayType) or not isinstance(values_type, ArrayType):
        raise GPUScaffoldError("GPU replicate requires array parameters")
    if values_type.element != FLOAT:
        raise GPUScaffoldError("GPU replicate supports f32 values only")
    if counts_type.rank != 1 or values_type.rank != 1:
        raise GPUScaffoldError("GPU replicate supports rank-1 only")

    N = int(values_type.shape[0].value)
    out_N = output_upper_bound or N * N
    name = kernel_name or f"remora_{function.name}_replicate"
    _validate_scaffold_names(module_name, name)

    desc_lines = _descriptor_load_lines("cnt", "%input0_desc", 1)
    desc_lines.extend(_descriptor_load_lines("val", "%input1_desc", 1))
    desc_lines.extend(_descriptor_load_lines("out", "%output_desc", 1))

    text = f"""module {{
  gpu.module @{module_name} {{
    llvm.func @{name}(%input0_desc: !llvm.ptr, %input1_desc: !llvm.ptr, %output_desc: !llvm.ptr) attributes {{gpu.kernel, nvvm.kernel}} {{
{chr(10).join(desc_lines)}
      %rp_tid32 = nvvm.read.ptx.sreg.tid.x : i32
      %rp_tid = llvm.sext %rp_tid32 : i32 to i64
      %rp_z = llvm.mlir.constant(0 : index) : i64
      %rp_is_t0 = llvm.icmp "eq" %rp_tid, %rp_z : i64
      llvm.cond_br %rp_is_t0, ^rp_work, ^rp_done

    ^rp_work:
      %rp_N = llvm.mlir.constant({N} : index) : i64
      %rp_one = llvm.mlir.constant(1 : index) : i64
      llvm.br ^rp_outer(%rp_z, %rp_z : i64, i64)

    ^rp_outer(%rp_i: i64, %rp_wp: i64):
      %rp_od = llvm.icmp "uge" %rp_i, %rp_N : i64
      llvm.cond_br %rp_od, ^rp_done, ^rp_load

    ^rp_load:
      %rp_ci = llvm.add %cnt_offset, %rp_i  : i64
      %rp_cp = llvm.getelementptr %cnt_aligned[%rp_ci] : (!llvm.ptr, i64) -> !llvm.ptr, i32
      %rp_count = llvm.load %rp_cp : !llvm.ptr -> i32
      %rp_count64 = llvm.sext %rp_count : i32 to i64
      %rp_vi = llvm.add %val_offset, %rp_i  : i64
      %rp_vp = llvm.getelementptr %val_aligned[%rp_vi] : (!llvm.ptr, i64) -> !llvm.ptr, f32
      %rp_val = llvm.load %rp_vp : !llvm.ptr -> f32
      llvm.br ^rp_inner(%rp_z, %rp_wp : i64, i64)

    ^rp_inner(%rp_j: i64, %rp_iwp: i64):
      %rp_jd = llvm.icmp "uge" %rp_j, %rp_count64 : i64
      llvm.cond_br %rp_jd, ^rp_next(%rp_iwp : i64), ^rp_store

    ^rp_store:
      %rp_oi = llvm.add %out_offset, %rp_iwp  : i64
      %rp_op = llvm.getelementptr %out_aligned[%rp_oi] : (!llvm.ptr, i64) -> !llvm.ptr, f32
      llvm.store %rp_val, %rp_op : f32, !llvm.ptr
      %rp_nj = llvm.add %rp_j, %rp_one : i64
      %rp_nwp = llvm.add %rp_iwp, %rp_one : i64
      llvm.br ^rp_inner(%rp_nj, %rp_nwp : i64, i64)

    ^rp_next(%rp_fwp: i64):
      %rp_ni = llvm.add %rp_i, %rp_one : i64
      llvm.br ^rp_outer(%rp_ni, %rp_fwp : i64, i64)

    ^rp_done:
      llvm.return
    }}
  }}
}}"""
    return GPUModuleScaffold(text, module_name, name)


def build_descriptor_abi_parallel_replicate_gpu_module(
    function: HIRFunction,
    *,
    module_name: str = "remora_gpu",
    kernel_name: str | None = None,
    output_upper_bound: int | None = None,
) -> GPUModuleScaffold:
    """Build a two-kernel parallel replicate using prefix-sum + scatter.

    Returns a gpu.module with two entry points:
    - ``{name}_scan``: i32 inclusive prefix sum on counts
    - ``{name}_scatter``: scatter-replicate values to output positions

    Only supports rank-1 with N ≤ 1024.
    """
    if not isinstance(function.body, HIRReplicate):
        raise GPUScaffoldError("GPU parallel replicate requires HIRReplicate body")
    if len(function.params) != 2:
        raise GPUScaffoldError("GPU parallel replicate requires two parameters")

    counts_type = function.params[0].type
    values_type = function.params[1].type
    if not isinstance(counts_type, ArrayType) or not isinstance(values_type, ArrayType):
        raise GPUScaffoldError("GPU parallel replicate requires array parameters")
    if values_type.element != FLOAT:
        raise GPUScaffoldError("GPU parallel replicate supports f32 values only")
    if counts_type.rank != 1 or values_type.rank != 1:
        raise GPUScaffoldError("GPU parallel replicate supports rank-1 only")

    N = int(values_type.shape[0].value)
    if N > 1024:
        raise GPUScaffoldError("GPU parallel replicate requires N ≤ 1024")

    out_N = output_upper_bound or N * N
    base = kernel_name or f"remora_{function.name}_replicate"
    scan_name = f"{base}_scan"
    scatter_name = f"{base}_scatter"
    _validate_scaffold_names(module_name, scan_name)

    max_d = (N - 1).bit_length() if N > 1 else 1

    k1_desc = _descriptor_load_lines("rs_cnt", "%counts_desc", 1)
    k1_desc.extend(_descriptor_load_lines("rs_scan", "%scan_desc", 1))

    k2_desc = _descriptor_load_lines("rx_cnt", "%counts_desc", 1)
    k2_desc.extend(_descriptor_load_lines("rx_val", "%values_desc", 1))
    k2_desc.extend(_descriptor_load_lines("rx_scan", "%scan_desc", 1))
    k2_desc.extend(_descriptor_load_lines("rx_out", "%output_desc", 1))

    text = f"""module {{
  gpu.module @{module_name} {{
    llvm.mlir.global internal @pr_scan_shmem() {{addr_space = 3 : i32}} : !llvm.array<1024 x i32>

    llvm.func @{scan_name}(%counts_desc: !llvm.ptr, %scan_desc: !llvm.ptr) attributes {{gpu.kernel, nvvm.kernel}} {{
{chr(10).join(k1_desc)}
      %rs_tid32 = nvvm.read.ptx.sreg.tid.x : i32
      %rs_tid = llvm.sext %rs_tid32 : i32 to i64
      %rs_N = llvm.mlir.constant({N} : index) : i64
      %rs_in_bounds = llvm.icmp "ult" %rs_tid, %rs_N : i64
      %rs_shmem_base = llvm.mlir.addressof @pr_scan_shmem : !llvm.ptr<3>
      %rs_zero_val = llvm.mlir.constant(0 : i32) : i32
      llvm.cond_br %rs_in_bounds, ^rs_load, ^rs_load_oob

    ^rs_load:
      %rs_si = llvm.add %rs_cnt_offset, %rs_tid  : i64
      %rs_sp = llvm.getelementptr %rs_cnt_aligned[%rs_si] : (!llvm.ptr, i64) -> !llvm.ptr, i32
      %rs_val = llvm.load %rs_sp : !llvm.ptr -> i32
      %rs_me = llvm.getelementptr %rs_shmem_base[0, %rs_tid] : (!llvm.ptr<3>, i64) -> !llvm.ptr<3>, !llvm.array<1024 x i32>
      llvm.store %rs_val, %rs_me : i32, !llvm.ptr<3>
      llvm.br ^rs_after_load

    ^rs_load_oob:
      %rs_me_oob = llvm.getelementptr %rs_shmem_base[0, %rs_tid] : (!llvm.ptr<3>, i64) -> !llvm.ptr<3>, !llvm.array<1024 x i32>
      llvm.store %rs_zero_val, %rs_me_oob : i32, !llvm.ptr<3>
      llvm.br ^rs_after_load

    ^rs_after_load:
      nvvm.barrier0
      %rs_c0 = llvm.mlir.constant(0 : index) : i64
      %rs_c1 = llvm.mlir.constant(1 : index) : i64
      %rs_c2 = llvm.mlir.constant(2 : index) : i64
      %rs_max_d = llvm.mlir.constant({max_d} : index) : i64
      llvm.br ^rs_loop(%rs_c0, %rs_c1 : i64, i64)

    ^rs_loop(%rs_d: i64, %rs_stride: i64):
      %rs_loop_done = llvm.icmp "uge" %rs_d, %rs_max_d : i64
      llvm.cond_br %rs_loop_done, ^rs_write, ^rs_step

    ^rs_step:
      %rs_active = llvm.icmp "uge" %rs_tid, %rs_stride : i64
      %rs_partner_raw = llvm.sub %rs_tid, %rs_stride  : i64
      %rs_safe_partner = llvm.select %rs_active, %rs_partner_raw, %rs_c0 : i1, i64
      %rs_pptr = llvm.getelementptr %rs_shmem_base[0, %rs_safe_partner] : (!llvm.ptr<3>, i64) -> !llvm.ptr<3>, !llvm.array<1024 x i32>
      %rs_pval = llvm.load %rs_pptr : !llvm.ptr<3> -> i32
      %rs_temp = llvm.select %rs_active, %rs_pval, %rs_zero_val : i1, i32
      nvvm.barrier0
      %rs_me2 = llvm.getelementptr %rs_shmem_base[0, %rs_tid] : (!llvm.ptr<3>, i64) -> !llvm.ptr<3>, !llvm.array<1024 x i32>
      %rs_cur = llvm.load %rs_me2 : !llvm.ptr<3> -> i32
      %rs_new = llvm.add %rs_cur, %rs_temp  : i32
      %rs_result = llvm.select %rs_active, %rs_new, %rs_cur : i1, i32
      llvm.store %rs_result, %rs_me2 : i32, !llvm.ptr<3>
      nvvm.barrier0
      %rs_next_d = llvm.add %rs_d, %rs_c1  : i64
      %rs_next_stride = llvm.mul %rs_stride, %rs_c2  : i64
      llvm.br ^rs_loop(%rs_next_d, %rs_next_stride : i64, i64)

    ^rs_write:
      llvm.cond_br %rs_in_bounds, ^rs_write_body, ^rs_done

    ^rs_write_body:
      %rs_me3 = llvm.getelementptr %rs_shmem_base[0, %rs_tid] : (!llvm.ptr<3>, i64) -> !llvm.ptr<3>, !llvm.array<1024 x i32>
      %rs_final = llvm.load %rs_me3 : !llvm.ptr<3> -> i32
      %rs_oi = llvm.add %rs_scan_offset, %rs_tid  : i64
      %rs_op = llvm.getelementptr %rs_scan_aligned[%rs_oi] : (!llvm.ptr, i64) -> !llvm.ptr, i32
      llvm.store %rs_final, %rs_op : i32, !llvm.ptr
      llvm.br ^rs_done

    ^rs_done:
      llvm.return
    }}

    llvm.func @{scatter_name}(%counts_desc: !llvm.ptr, %values_desc: !llvm.ptr, %scan_desc: !llvm.ptr, %output_desc: !llvm.ptr) attributes {{gpu.kernel, nvvm.kernel}} {{
{chr(10).join(k2_desc)}
      %rx_tid32 = nvvm.read.ptx.sreg.tid.x : i32
      %rx_tid = llvm.sext %rx_tid32 : i32 to i64
      %rx_bid32 = nvvm.read.ptx.sreg.ctaid.x : i32
      %rx_bid = llvm.sext %rx_bid32 : i32 to i64
      %rx_bdim32 = nvvm.read.ptx.sreg.ntid.x : i32
      %rx_bdim = llvm.sext %rx_bdim32 : i32 to i64
      %rx_base = llvm.mul %rx_bid, %rx_bdim  : i64
      %rx_idx = llvm.add %rx_base, %rx_tid  : i64
      %rx_N = llvm.mlir.constant({N} : index) : i64
      %rx_in_bounds = llvm.icmp "ult" %rx_idx, %rx_N : i64
      llvm.cond_br %rx_in_bounds, ^rx_work, ^rx_done

    ^rx_work:
      %rx_c0 = llvm.mlir.constant(0 : index) : i64
      %rx_c1 = llvm.mlir.constant(1 : index) : i64
      %rx_sci = llvm.add %rx_scan_offset, %rx_idx  : i64
      %rx_scp = llvm.getelementptr %rx_scan_aligned[%rx_sci] : (!llvm.ptr, i64) -> !llvm.ptr, i32
      %rx_end_i32 = llvm.load %rx_scp : !llvm.ptr -> i32
      %rx_end = llvm.sext %rx_end_i32 : i32 to i64
      %rx_is_first = llvm.icmp "eq" %rx_idx, %rx_c0 : i64
      %rx_prev_idx = llvm.sub %rx_idx, %rx_c1  : i64
      %rx_prev_safe = llvm.select %rx_is_first, %rx_c0, %rx_prev_idx : i1, i64
      %rx_prev_sci = llvm.add %rx_scan_offset, %rx_prev_safe  : i64
      %rx_prev_scp = llvm.getelementptr %rx_scan_aligned[%rx_prev_sci] : (!llvm.ptr, i64) -> !llvm.ptr, i32
      %rx_prev_i32 = llvm.load %rx_prev_scp : !llvm.ptr -> i32
      %rx_prev = llvm.sext %rx_prev_i32 : i32 to i64
      %rx_start = llvm.select %rx_is_first, %rx_c0, %rx_prev : i1, i64
      %rx_vi = llvm.add %rx_val_offset, %rx_idx  : i64
      %rx_vp = llvm.getelementptr %rx_val_aligned[%rx_vi] : (!llvm.ptr, i64) -> !llvm.ptr, f32
      %rx_val = llvm.load %rx_vp : !llvm.ptr -> f32
      llvm.br ^rx_inner(%rx_start : i64)

    ^rx_inner(%rx_pos: i64):
      %rx_inner_done = llvm.icmp "uge" %rx_pos, %rx_end : i64
      llvm.cond_br %rx_inner_done, ^rx_done, ^rx_store

    ^rx_store:
      %rx_oi = llvm.add %rx_out_offset, %rx_pos  : i64
      %rx_op = llvm.getelementptr %rx_out_aligned[%rx_oi] : (!llvm.ptr, i64) -> !llvm.ptr, f32
      llvm.store %rx_val, %rx_op : f32, !llvm.ptr
      %rx_next_pos = llvm.add %rx_pos, %rx_c1  : i64
      llvm.br ^rx_inner(%rx_next_pos : i64)

    ^rx_done:
      llvm.return
    }}
  }}
}}"""
    return GPUModuleScaffold(text, module_name, scatter_name)


# ---------------------------------------------------------------------------
# Scatter-add kernel
# ---------------------------------------------------------------------------


def build_descriptor_abi_scatter_add_gpu_module(
    function: HIRFunction,
    *,
    module_name: str = "remora_gpu",
    kernel_name: str | None = None,
) -> GPUModuleScaffold:
    """Build a descriptor-ABI GPU module for a scalar scatter-add.

    Single-thread kernel: copies target to output, then adds update at
    the given index position.
    """
    if not isinstance(function.body, HIRScatterAdd):
        raise GPUScaffoldError("GPU scatter-add requires HIRScatterAdd body")

    sa = function.body
    if not isinstance(sa.result_type, ArrayType) or sa.result_type.element != FLOAT:
        raise GPUScaffoldError("GPU scatter-add currently supports f32 output only")

    shape = tuple(int(d.value) for d in sa.result_type.shape)
    if len(shape) != 1:
        raise GPUScaffoldError("GPU scatter-add currently supports rank-1 output only")
    N = shape[0]

    name = kernel_name or f"remora_{function.name}_scatter"
    _validate_scaffold_names(module_name, name)

    target_param_name = None
    idx_param_idx = None
    upd_param_idx = None
    scalar_count = 0

    for pidx, p in enumerate(function.params):
        if isinstance(p.type, ArrayType):
            if target_param_name is None:
                target_param_name = p.name
        else:
            if isinstance(sa.index, HIRVar) and sa.index.name == p.name:
                idx_param_idx = scalar_count
            if isinstance(sa.update, HIRVar) and sa.update.name == p.name:
                upd_param_idx = scalar_count
            scalar_count += 1

    desc_lines = _descriptor_load_lines("tgt", "%input0_desc", 1)
    desc_lines.extend(_descriptor_load_lines("out", "%output_desc", 1))

    scalar_params: list[str] = []
    for si in range(scalar_count):
        scalar_params.append(f"%scalar{si}: f32")

    idx_load = ""
    if idx_param_idx is not None:
        idx_load = f"      %sa_idx = llvm.fptosi %scalar{idx_param_idx} : f32 to i64"
    elif isinstance(sa.index, HIRLit):
        idx_load = f"      %sa_idx = llvm.mlir.constant({int(sa.index.value)} : index) : i64"
    else:
        raise GPUScaffoldError("GPU scatter-add: index must be a param or literal")

    upd_load = ""
    if upd_param_idx is not None:
        upd_load = f"      %sa_upd = llvm.mlir.constant(0.0 : f32) : f32\n      %sa_upd2 = llvm.fadd %scalar{upd_param_idx}, %sa_upd  : f32"
        upd_ssa = f"%sa_upd2"
    elif isinstance(sa.update, HIRLit):
        upd_load = f"      %sa_upd = llvm.mlir.constant({float(sa.update.value):.6e} : f32) : f32"
        upd_ssa = "%sa_upd"
    else:
        raise GPUScaffoldError("GPU scatter-add: update must be a param or literal")

    all_params = ["%input0_desc: !llvm.ptr"] + scalar_params + ["%output_desc: !llvm.ptr"]

    text = f"""module {{
  gpu.module @{module_name} {{
    llvm.func @{name}({", ".join(all_params)}) attributes {{gpu.kernel, nvvm.kernel}} {{
{chr(10).join(desc_lines)}
      %sa_tid32 = nvvm.read.ptx.sreg.tid.x : i32
      %sa_tid = llvm.sext %sa_tid32 : i32 to i64
      %sa_zero = llvm.mlir.constant(0 : index) : i64
      %sa_is_t0 = llvm.icmp "eq" %sa_tid, %sa_zero : i64
      llvm.cond_br %sa_is_t0, ^sa_work, ^sa_done

    ^sa_work:
      %sa_N = llvm.mlir.constant({N} : index) : i64
      %sa_one = llvm.mlir.constant(1 : index) : i64
      llvm.br ^sa_copy(%sa_zero : i64)

    ^sa_copy(%sa_ci: i64):
      %sa_cdone = llvm.icmp "uge" %sa_ci, %sa_N : i64
      llvm.cond_br %sa_cdone, ^sa_add, ^sa_cbody

    ^sa_cbody:
      %sa_src_off = llvm.add %tgt_offset, %sa_ci  : i64
      %sa_src_p = llvm.getelementptr %tgt_aligned[%sa_src_off] : (!llvm.ptr, i64) -> !llvm.ptr, f32
      %sa_val = llvm.load %sa_src_p : !llvm.ptr -> f32
      %sa_dst_off = llvm.add %out_offset, %sa_ci  : i64
      %sa_dst_p = llvm.getelementptr %out_aligned[%sa_dst_off] : (!llvm.ptr, i64) -> !llvm.ptr, f32
      llvm.store %sa_val, %sa_dst_p : f32, !llvm.ptr
      %sa_cnext = llvm.add %sa_ci, %sa_one : i64
      llvm.br ^sa_copy(%sa_cnext : i64)

    ^sa_add:
{idx_load}
{upd_load}
      %sa_aoff = llvm.add %out_offset, %sa_idx  : i64
      %sa_ap = llvm.getelementptr %out_aligned[%sa_aoff] : (!llvm.ptr, i64) -> !llvm.ptr, f32
      %sa_old = llvm.load %sa_ap : !llvm.ptr -> f32
      %sa_new = llvm.fadd %sa_old, {upd_ssa}  : f32
      llvm.store %sa_new, %sa_ap : f32, !llvm.ptr
      llvm.br ^sa_done

    ^sa_done:
      llvm.return
    }}
  }}
}}"""
    return GPUModuleScaffold(text, module_name, name)


def build_descriptor_abi_parallel_scatter_add_gpu_module(
    function: HIRFunction,
    *,
    module_name: str = "remora_gpu",
    kernel_name: str | None = None,
) -> GPUModuleScaffold:
    """Build a parallel scatter-add kernel (N ≤ 1024).

    All threads copy target to output in parallel, then thread 0
    performs the scalar add after a barrier.
    """
    if not isinstance(function.body, HIRScatterAdd):
        raise GPUScaffoldError("GPU parallel scatter-add requires HIRScatterAdd body")

    sa = function.body
    if not isinstance(sa.result_type, ArrayType) or sa.result_type.element != FLOAT:
        raise GPUScaffoldError("GPU parallel scatter-add supports f32 only")

    shape = tuple(int(d.value) for d in sa.result_type.shape)
    if len(shape) != 1:
        raise GPUScaffoldError("GPU parallel scatter-add supports rank-1 only")
    N = shape[0]
    if N > 1024:
        raise GPUScaffoldError("GPU parallel scatter-add requires N ≤ 1024")

    name = kernel_name or f"remora_{function.name}_scatter"
    _validate_scaffold_names(module_name, name)

    target_param_name = None
    idx_param_idx = None
    upd_param_idx = None
    scalar_count = 0

    for pidx, p in enumerate(function.params):
        if isinstance(p.type, ArrayType):
            if target_param_name is None:
                target_param_name = p.name
        else:
            if isinstance(sa.index, HIRVar) and sa.index.name == p.name:
                idx_param_idx = scalar_count
            if isinstance(sa.update, HIRVar) and sa.update.name == p.name:
                upd_param_idx = scalar_count
            scalar_count += 1

    desc_lines = _descriptor_load_lines("tgt", "%input0_desc", 1)
    desc_lines.extend(_descriptor_load_lines("out", "%output_desc", 1))

    scalar_params: list[str] = []
    for si in range(scalar_count):
        scalar_params.append(f"%scalar{si}: f32")

    idx_load = ""
    if idx_param_idx is not None:
        idx_load = f"      %pa_idx = llvm.fptosi %scalar{idx_param_idx} : f32 to i64"
    elif isinstance(sa.index, HIRLit):
        idx_load = f"      %pa_idx = llvm.mlir.constant({int(sa.index.value)} : index) : i64"
    else:
        raise GPUScaffoldError("GPU parallel scatter-add: index must be a param or literal")

    upd_load = ""
    if upd_param_idx is not None:
        upd_load = f"      %pa_upd = llvm.mlir.constant(0.0 : f32) : f32\n      %pa_upd2 = llvm.fadd %scalar{upd_param_idx}, %pa_upd  : f32"
        upd_ssa = "%pa_upd2"
    elif isinstance(sa.update, HIRLit):
        upd_load = f"      %pa_upd = llvm.mlir.constant({float(sa.update.value):.6e} : f32) : f32"
        upd_ssa = "%pa_upd"
    else:
        raise GPUScaffoldError("GPU parallel scatter-add: update must be a param or literal")

    all_params = ["%input0_desc: !llvm.ptr"] + scalar_params + ["%output_desc: !llvm.ptr"]

    text = f"""module {{
  gpu.module @{module_name} {{
    llvm.func @{name}({", ".join(all_params)}) attributes {{gpu.kernel, nvvm.kernel}} {{
{chr(10).join(desc_lines)}
      %pa_tid32 = nvvm.read.ptx.sreg.tid.x : i32
      %pa_tid = llvm.sext %pa_tid32 : i32 to i64
      %pa_N = llvm.mlir.constant({N} : index) : i64
      %pa_in_bounds = llvm.icmp "ult" %pa_tid, %pa_N : i64
      llvm.cond_br %pa_in_bounds, ^pa_copy, ^pa_after_copy

    ^pa_copy:
      %pa_src_off = llvm.add %tgt_offset, %pa_tid  : i64
      %pa_src_p = llvm.getelementptr %tgt_aligned[%pa_src_off] : (!llvm.ptr, i64) -> !llvm.ptr, f32
      %pa_val = llvm.load %pa_src_p : !llvm.ptr -> f32
      %pa_dst_off = llvm.add %out_offset, %pa_tid  : i64
      %pa_dst_p = llvm.getelementptr %out_aligned[%pa_dst_off] : (!llvm.ptr, i64) -> !llvm.ptr, f32
      llvm.store %pa_val, %pa_dst_p : f32, !llvm.ptr
      llvm.br ^pa_after_copy

    ^pa_after_copy:
      nvvm.barrier0
      %pa_zero = llvm.mlir.constant(0 : index) : i64
      %pa_is_t0 = llvm.icmp "eq" %pa_tid, %pa_zero : i64
      llvm.cond_br %pa_is_t0, ^pa_add, ^pa_done

    ^pa_add:
{idx_load}
{upd_load}
      %pa_aoff = llvm.add %out_offset, %pa_idx  : i64
      %pa_ap = llvm.getelementptr %out_aligned[%pa_aoff] : (!llvm.ptr, i64) -> !llvm.ptr, f32
      %pa_old = llvm.load %pa_ap : !llvm.ptr -> f32
      %pa_new = llvm.fadd %pa_old, {upd_ssa}  : f32
      llvm.store %pa_new, %pa_ap : f32, !llvm.ptr
      llvm.br ^pa_done

    ^pa_done:
      llvm.return
    }}
  }}
}}"""
    return GPUModuleScaffold(text, module_name, name)


# ---------------------------------------------------------------------------
# General GPU expression emitter and kernel scaffold
# ---------------------------------------------------------------------------


def _unwrap_view_op(expr):
    """Peel view-op wrappers from a map array expression.

    Returns ``(base_var_name, coord_offsets, coord_transforms)`` if the
    expression is a chain of view ops wrapping an HIRVar, or
    ``(None, (), ())`` if not.
    """
    offsets: list[int] = []
    transforms: list[str] = []

    while True:
        if isinstance(expr, HIRVar):
            return expr.name, tuple(offsets), tuple(transforms)
        if isinstance(expr, HIRTake):
            expr = expr.array
            continue
        if isinstance(expr, HIRDrop):
            while len(offsets) < 1:
                offsets.append(0)
            offsets[0] += expr.count
            expr = expr.array
            continue
        if isinstance(expr, HIRSubarray):
            for k, o in enumerate(expr.offsets):
                while len(offsets) <= k:
                    offsets.append(0)
                offsets[k] += int(o.value)
            expr = expr.array
            continue
        if isinstance(expr, HIRReverse):
            N = int(expr.result_type.shape[0].value)
            while len(transforms) < 1:
                transforms.append("")
            transforms[0] = f"reverse:{N}"
            expr = expr.array
            continue
        if isinstance(expr, HIRRotate):
            N = int(expr.result_type.shape[0].value)
            shift = int(expr.shift.value)
            while len(transforms) < 1:
                transforms.append("")
            transforms[0] = f"mod:{N}:{shift}"
            expr = expr.array
            continue
        if isinstance(expr, HIRTranspose):
            expr = expr.array
            continue
        if isinstance(expr, HIRReshape):
            expr = expr.array
            continue
        if isinstance(expr, HIRRavel):
            expr = expr.array
            continue
        if isinstance(expr, HIRWithShape):
            expr = expr.source
            continue
        return None, (), ()


def build_descriptor_abi_general_map_gpu_module(
    function: HIRFunction,
    *,
    module_name: str = "remora_gpu",
    kernel_name: str | None = None,
) -> GPUModuleScaffold:
    """Build a descriptor-ABI GPU module for a general map kernel.

    Handles map callable bodies with compound expressions: folds, index
    operations, nested maps, conditionals, casts, and let bindings.
    Uses recursive expression compilation to LLVM dialect + scf.for loops.
    """
    from remora._gpu_expr_lowering import gpu_expr_from_hir
    from remora.hir import HIRLambda

    name = kernel_name or f"remora_{function.name}_general"
    _validate_scaffold_names(module_name, name)

    if not isinstance(function.body, (HIRMap, HIRApply)):
        inner = function.body
        while isinstance(inner, HIRLet):
            inner = inner.body
        if isinstance(inner, HIRVar):
            raise GPUScaffoldError(
                "general GPU map requires a HIRMap or HIRApply top-level body "
                "(CSE reduced the body to a variable reference; the expression "
                "is likely too complex for a single GPU kernel)"
            )
        raise GPUScaffoldError(
            "general GPU map requires a HIRMap or HIRApply top-level body"
        )

    body_map = function.body
    map_func = body_map.func
    if isinstance(map_func, HIRPrimCallable):
        from remora.hir import HIRLambda as _L, HIRParam as _P, HIRPrimOp as _PO, HIRVar as _V
        from remora.types import FuncType as _FT
        _op = map_func.op
        _rt = map_func.result_type
        _elem_rt = _rt.element if isinstance(_rt, ArrayType) else _rt
        _suffix = "f" if _elem_rt == FLOAT else ("i" if str(getattr(_elem_rt, 'name', '')) == 'int' else "f")
        _typed_op = f"{_op}{_suffix}"
        if map_func.left_arg is not None:
            _p = _P("_gx", map_func.params[0] if map_func.params else FLOAT)
            map_func = _L([_p], _PO(_typed_op, [map_func.left_arg, _V("_gx", _p.type)], _rt), _FT((_p.type,), _rt))
        elif map_func.right_arg is not None:
            _p = _P("_gx", map_func.params[0] if map_func.params else FLOAT)
            map_func = _L([_p], _PO(_typed_op, [_V("_gx", _p.type), map_func.right_arg], _rt), _FT((_p.type,), _rt))
        elif len(map_func.params) == 2:
            _p0 = _P("_gx", map_func.params[0])
            _p1 = _P("_gy", map_func.params[1])
            map_func = _L([_p0, _p1], _PO(_typed_op, [_V("_gx", _p0.type), _V("_gy", _p1.type)], _rt), _FT((_p0.type, _p1.type), _rt))
        body_map = HIRMap(body_map.frame_shape, body_map.cell_shape, map_func, body_map.arrays, body_map.result_type)

    if isinstance(body_map.func, HIRVar):
        var_name = body_map.func.name
        is_param = any(p.name == var_name for p in function.params)
        if is_param:
            raise GPUScaffoldError(
                f"GPU backend does not support higher-order function parameters as "
                f"map callables ('{var_name}' is a function parameter); use a lambda "
                f"or inline the function definition"
            )
        raise GPUScaffoldError(
            f"general GPU map: unresolved function reference '{var_name}' as callable; "
            f"defunctionalization should have inlined this"
        )

    if not isinstance(body_map.func, HIRLambda):
        raise GPUScaffoldError(
            f"general GPU map requires a HIRLambda callable (got {type(body_map.func).__name__})"
        )

    result_type = body_map.result_type
    if not isinstance(result_type, ArrayType):
        raise GPUScaffoldError(
            "general GPU map requires an array result type"
        )

    result_shape = tuple(int(d.value) for d in result_type.shape)
    _validate_shape(result_shape)

    # Determine rank and total output size
    rank = len(result_shape)
    total_size = 1
    for d in result_shape:
        total_size *= d

    # Build input mapping: each function param gets a descriptor slot index
    input_map: dict[str, int] = {}
    scalar_params: dict[str, int] = {}
    input_elem_types: list[str] = []
    num_array_inputs = 0
    num_scalar_inputs = 0

    for param in function.params:
        if isinstance(param.type, ArrayType):
            input_map[param.name] = num_array_inputs
            num_array_inputs += 1
            elem = "f32"
            if param.type.element.name == "int":
                elem = "i32"
            elif param.type.element.name == "bool":
                elem = "i1"
            input_elem_types.append(elem)
        else:
            scalar_params[param.name] = num_scalar_inputs
            num_scalar_inputs += 1

    # Frame rank: number of independent computations.
    # For a map with frame [N] producing output [N, 3], frame_rank=1, output_rank=2.
    # Each thread computes one frame element, producing all extra output dims.
    frame_rank = len(body_map.frame_shape)
    frame_size = 1
    for d in body_map.frame_shape:
        frame_size *= int(d.value)

    # Coords: thread's multi-dimensional indices over FRAME dimensions only
    coords = [f"%i{axis}" for axis in range(frame_rank)]

    # Compile the map lambda body to a GpuExpr
    lambda_body = body_map.func.body

    # Build coord_map for iota-based outer maps.
    # When the outer map's array is HIRIota, the lambda param maps to
    # the thread coordinate rather than an input descriptor slot.
    coord_map: dict[str, str] = {}
    input_adjustments: dict[str, tuple[tuple[int, ...], tuple[str, ...]]] = {}
    input_flat_shapes: dict[str, tuple[int, ...]] = {}
    input_desc_ranks: dict[int, int] = {}
    append_inputs: dict[str, tuple[int, int, int]] = {}
    input_broadcast_skip: dict[str, int] = {}
    input_element_types: dict[str, str] = {}

    def _detect_reshape(ae):
        """Check if array_expr is a reshape/ravel wrapping an input."""
        if isinstance(ae, HIRReshape):
            return ae, tuple(int(d.value) for d in ae.result_type.shape)
        if isinstance(ae, HIRRavel):
            return ae, tuple(int(d.value) for d in ae.result_type.shape)
        return None, None

    def _detect_append(ae):
        """Check if array_expr is an append of two inputs."""
        if isinstance(ae, HIRAppend):
            return ae
        return None

    def _detect_withshape(ae):
        """Check if array_expr is a withshape broadcast."""
        if isinstance(ae, HIRWithShape):
            return ae
        return None

    # Bind lambda params to input slots or thread coords.
    for idx, array_expr in enumerate(body_map.arrays):
        if idx < len(body_map.func.params):
            param_name = body_map.func.params[idx].name
            if isinstance(array_expr, HIRVar) and array_expr.name in input_map:
                input_map[param_name] = input_map[array_expr.name]
            elif isinstance(array_expr, HIRLit) and array_expr.type == FLOAT:
                scalar_params[param_name] = num_scalar_inputs
                num_scalar_inputs += 1
            else:
                reshape_node, flat_shape = _detect_reshape(array_expr)
                append_node = _detect_append(array_expr)
                withshape_node = _detect_withshape(array_expr)

                if reshape_node is not None and flat_shape is not None:
                    base_name, _, _ = _unwrap_view_op(reshape_node)
                    if base_name is not None and base_name in input_map:
                        slot = input_map[base_name]
                        input_map[param_name] = slot
                        input_flat_shapes[param_name] = flat_shape
                        param_type = None
                        for p in function.params:
                            if p.name == base_name:
                                param_type = p.type
                                break
                        if isinstance(param_type, ArrayType):
                            input_desc_ranks[slot] = param_type.rank
                    else:
                        for axis_idx in range(len(body_map.func.params)):
                            pname = body_map.func.params[axis_idx].name
                            if axis_idx < len(coords):
                                coord_map[pname] = coords[axis_idx]

                elif append_node is not None:
                    left_name, _, _ = _unwrap_view_op(append_node.left)
                    right_name, _, _ = _unwrap_view_op(append_node.right)
                    if (left_name is not None and left_name in input_map
                            and right_name is not None and right_name in input_map):
                        left_slot = input_map[left_name]
                        right_slot = input_map[right_name]
                        left_type = None
                        for p in function.params:
                            if p.name == left_name:
                                left_type = p.type
                                break
                        left_size = int(left_type.shape[0].value) if isinstance(left_type, ArrayType) else 0
                        if isinstance(left_type, ArrayType) and left_type.rank > 1:
                            raise GPUScaffoldError(
                                f"GPU append supports rank-1 arrays only (got rank {left_type.rank})"
                            )
                        input_map[param_name] = left_slot
                        append_inputs[param_name] = (left_slot, right_slot, left_size)
                    else:
                        for axis_idx in range(len(body_map.func.params)):
                            pname = body_map.func.params[axis_idx].name
                            if axis_idx < len(coords):
                                coord_map[pname] = coords[axis_idx]

                elif withshape_node is not None:
                    base_name, view_offsets, view_transforms = _unwrap_view_op(withshape_node)
                    if base_name is not None and base_name in input_map:
                        input_map[param_name] = input_map[base_name]
                        src_type = None
                        for p in function.params:
                            if p.name == base_name:
                                src_type = p.type
                                break
                        src_rank = src_type.rank if isinstance(src_type, ArrayType) else 0
                        target_rank = withshape_node.result_type.rank
                        dims_to_skip = target_rank - src_rank
                        input_broadcast_skip[param_name] = dims_to_skip
                        if isinstance(src_type, ArrayType):
                            input_desc_ranks[input_map[base_name]] = src_rank
                    else:
                        for axis_idx in range(len(body_map.func.params)):
                            pname = body_map.func.params[axis_idx].name
                            if axis_idx < len(coords):
                                coord_map[pname] = coords[axis_idx]
                else:
                    base_name, view_offsets, view_transforms = _unwrap_view_op(array_expr)
                    if base_name is not None and base_name in input_map:
                        input_map[param_name] = input_map[base_name]
                        if view_offsets or view_transforms:
                            input_adjustments[param_name] = (view_offsets, view_transforms)
                    elif hasattr(array_expr, 'result_type') and array_expr.result_type is not None:
                        from remora.hir import HIRLet as _HL
                        from remora.hir_opt import hir_optimize as _hopt
                        _optimized_arr = _hopt(array_expr)
                        _body_rt = getattr(lambda_body, 'result_type', body_map.result_type)
                        lambda_body = _HL(
                            name=param_name,
                            value_type=array_expr.result_type,
                            value=_optimized_arr,
                            body=lambda_body,
                            result_type=_body_rt,
                        )
                    else:
                        for axis_idx in range(len(body_map.func.params)):
                            pname = body_map.func.params[axis_idx].name
                            if axis_idx < len(coords):
                                coord_map[pname] = coords[axis_idx]

    from remora.types import BOOL as _BOOL, INT as _INT
    _slot_etypes: dict[int, str] = {}
    for _p in function.params:
        if isinstance(_p.type, ArrayType):
            _s = input_map.get(_p.name)
            if _s is not None:
                _e = _p.type.element
                if _e == FLOAT:
                    _slot_etypes[_s] = "f32"
                elif _e == _INT:
                    _slot_etypes[_s] = "i32"
                elif _e == _BOOL:
                    _slot_etypes[_s] = "i8"
    for _n, _s in input_map.items():
        if _s in _slot_etypes:
            input_element_types[_n] = _slot_etypes[_s]

    result_elem_type = "f32"
    if isinstance(result_type, ArrayType):
        _re = result_type.element
        if _re == _INT:
            result_elem_type = "i32"
        elif _re == _BOOL:
            result_elem_type = "i8"

    expr = gpu_expr_from_hir(
        lambda_body,
        input_map=input_map,
        scalar_env=scalar_params,
        coords=coords,
        coord_map=coord_map,
        context="general GPU map kernel",
        input_adjustments=input_adjustments,
        input_flat_shapes=input_flat_shapes,
        input_broadcast_skip=input_broadcast_skip,
        input_element_types=input_element_types,
    )

    # Emit the kernel body lines
    body_lines: list[str] = []

    # Descriptor load lines — use per-input ranks when available
    prefixes = []
    for idx in range(num_array_inputs):
        prefix = f"in{idx}"
        prefixes.append(prefix)
        desc_name = f"%input{idx}_desc"
        desc_rank = input_desc_ranks.get(idx, rank)
        body_lines.extend(_descriptor_load_lines(prefix, desc_name, desc_rank))

    # Output descriptor
    out_prefix = "out"
    prefixes.append(out_prefix)
    body_lines.extend(_descriptor_load_lines(out_prefix, "%output_desc", rank))

    # Thread/block index boilerplate + multi-index decomposition
    body_lines.extend([
        "      %tid32 = nvvm.read.ptx.sreg.tid.x : i32",
        "      %tid = llvm.sext %tid32 : i32 to i64",
        "      %bid32 = nvvm.read.ptx.sreg.ctaid.x : i32",
        "      %bid = llvm.sext %bid32 : i32 to i64",
        "      %bdim32 = nvvm.read.ptx.sreg.ntid.x : i32",
        "      %bdim = llvm.sext %bdim32 : i32 to i64",
        "      %block_base = llvm.mul %bid, %bdim  : i64",
        "      %idx = llvm.add %block_base, %tid  : i64",
    ])

    # Compute plane sizes for multi-index decomposition over frame dimensions
    if frame_rank > 1:
        body_lines.append("      %plane_last = llvm.mlir.constant(1 : index) : i64")
        prev_plane = "%plane_last"
        frame_shape_vals = [int(d.value) for d in body_map.frame_shape]
        for axis in range(frame_rank - 1, 0, -1):
            plane_name = f"%plane{axis - 1}"
            val = 1
            for d in frame_shape_vals[axis:]:
                val *= d
            body_lines.append(
                f"      {plane_name} = llvm.mlir.constant({val} : index) : i64"
            )
            prev_plane = plane_name

    total_name = "%total_size"
    body_lines.append(
        f"      {total_name} = llvm.mlir.constant({frame_size} : index) : i64"
    )

    body_lines.extend([
        f"      %inside = llvm.icmp \"ult\" %idx, {total_name} : i64",
        "      llvm.cond_br %inside, ^bb_body, ^bb_done",
        "    ^bb_body:",
    ])

    # Multi-index decomposition over frame dimensions
    body_lines.extend(_multi_index_lines(frame_rank))

    # Linear index computation for output store position (frame coords only)
    body_lines.extend(_linear_index_lines(out_prefix, frame_rank))

    # Emit the expression tree
    _initial_env = {c: c for c in coords}
    result_ssa = _gpu_emit_expr(expr, body_lines, _initial_env)

    # Store result — handle both scalar and array-valued results
    if isinstance(result_ssa, list):
        # Array-valued result: store each component at successive output offsets
        K = len(result_ssa)
        store_lines: list[str] = []
        for k in range(K):
            if k == 0:
                k_offset = "%out_linear"
            else:
                kc = f"%out_kconst{k}"
                koff = f"%out_off{k}"
                store_lines.append(f"      {kc} = llvm.mlir.constant({k} : index) : i64")
                store_lines.append(f"      {koff} = llvm.add %out_linear, {kc}  : i64")
                k_offset = koff
            ptr = f"%out_elem_ptr{k}" if k > 0 else "%out_elem_ptr"
            # Always emit getelementptr for each component
            store_lines.append(f"      {ptr} = llvm.getelementptr %out_aligned[{k_offset}] : (!llvm.ptr, i64) -> !llvm.ptr, {result_elem_type}")
            store_lines.append(f"      llvm.store {result_ssa[k]}, {ptr} : f32, !llvm.ptr")
        body_lines.extend(store_lines)
    else:
        body_lines.extend([
            f"      %out_elem_ptr = llvm.getelementptr %out_aligned[%out_linear] : (!llvm.ptr, i64) -> !llvm.ptr, {result_elem_type}",
            f"      llvm.store {result_ssa}, %out_elem_ptr : f32, !llvm.ptr",
        ])

    body_lines.extend([
        "      llvm.br ^bb_done",
        "    ^bb_done:",
    ])

    # Build the module
    param_parts: list[str] = []
    for idx in range(num_array_inputs):
        param_parts.append(f"%input{idx}_desc: !llvm.ptr")
    for idx in range(num_scalar_inputs):
        param_parts.append(f"%scalar{idx}: f32")
    param_parts.append("%output_desc: !llvm.ptr")
    all_params = ", ".join(param_parts)

    text = f"""module {{
  gpu.module @{module_name} {{
    llvm.func @llvm.nvvm.sqrt.f(f32) -> f32
    llvm.func @llvm.nvvm.ex2.approx.f(f32) -> f32
    llvm.func @llvm.nvvm.lg2.approx.f(f32) -> f32
    llvm.func @{name}({all_params}) attributes {{gpu.kernel, nvvm.kernel}} {{
{chr(10).join(body_lines)}
      llvm.return
    }}
  }}
}}"""

    return GPUModuleScaffold(text, module_name, name)


def _gpu_emit_expr(
    expr: "GpuExpr",
    lines: list[str],
    env: dict[str, str],
    *,
    temp_counter: int = 0,
) -> str:
    """Recursively emit MLIR LLVM dialect lines for a GpuExpr tree.

    Returns the SSA name (str) for scalar expressions, or a list[str]
    of SSA names for array-valued expressions (GpuArrayExpr, multi-component
    GpuReduce).

    Parameters
    ----------
    expr : GpuExpr
        The expression to emit.
    lines : list[str]
        MLIR line accumulator.
    env : dict[str, str]
        Maps GpuLetBinding and coordinator names to SSA names.
    temp_counter : int
        Starting temp counter for unique SSA names.
    """
    from remora._gpu_expr_lowering import (
        GpuAppendLoad,
        GpuArrayExpr,
        GpuBinaryOp,
        GpuCast,
        GpuCompareOp,
        GpuConstant,
        GpuExtractComponent,
        GpuFlatLoad,
        GpuIndexCoordinate,
        GpuInputLoad,
        GpuIntrinsic,
        GpuLetBinding,
        GpuReduce,
        GpuScalarParam,
        GpuSelect,
        _GpuLetExpr,
    )

    def _fresh() -> str:
        nonlocal temp_counter
        name = f"%gen_expr{temp_counter}"
        temp_counter += 1
        return name

    tcounter = [temp_counter]

    def _fresh_name() -> str:
        name = f"%gen_expr{tcounter[0]}"
        tcounter[0] += 1
        return name

    _emit_counter = [0]
    _block_counter = [0]

    def _fresh_ssa() -> str:
        _emit_counter[0] += 1
        return f"%gen{_emit_counter[0]}"

    def _fresh_block() -> str:
        _block_counter[0] += 1
        return f"bb_gen{_block_counter[0]}"

    def emit(expr: "GpuExpr", env: dict[str, str]) -> str:
        # GpuConstant
        if isinstance(expr, GpuConstant):
            ssa = _fresh_ssa()
            if expr.element_type == "f32":
                val_str = f"{float(expr.value):.6e}"
            elif expr.element_type == "i32":
                val_str = str(int(expr.value))
            elif expr.element_type == "i1":
                val_str = "1" if expr.value else "0"
            else:
                val_str = str(expr.value)
            lines.append(
                f"      {ssa} = llvm.mlir.constant({val_str} : {expr.element_type}) : {expr.element_type}"
            )
            return ssa

        # GpuScalarParam
        if isinstance(expr, GpuScalarParam):
            return f"%scalar{expr.index}"

        # GpuLetBinding
        if isinstance(expr, GpuLetBinding):
            if expr.name in env:
                return env[expr.name]
            raise GPUScaffoldError(
                f"unresolved let binding '{expr.name}'"
            )

        # GpuIndexCoordinate
        if isinstance(expr, GpuIndexCoordinate):
            # Resolve from env (set inside reduce loop or by outer coords)
            if expr.name in env:
                return env[expr.name]
            # Fall back to first coordinate if available
            raise GPUScaffoldError(
                f"unresolved coordinate '{expr.name}'"
            )

        # GpuInputLoad
        if isinstance(expr, GpuInputLoad):
            return _emit_input_load(expr, env)

        # GpuFlatLoad — compute flat index from coords, load from base+offset+flat
        if isinstance(expr, GpuFlatLoad):
            return _emit_flat_load(expr, env)

        # GpuAppendLoad — conditional load from left or right descriptor
        if isinstance(expr, GpuAppendLoad):
            return _emit_append_load(expr, env)

        # GpuBinaryOp
        if isinstance(expr, GpuBinaryOp):
            left = emit(expr.left, env)
            right = emit(expr.right, env)
            ssa = _fresh_ssa()
            et = getattr(expr, 'element_type', 'f32')
            llvm = llvm_op(expr.op, et)
            lines.append(f"      {ssa} = {llvm} {left}, {right}  : {et}")
            return ssa

        # GpuCompareOp
        if isinstance(expr, GpuCompareOp):
            left = emit(expr.left, env)
            right = emit(expr.right, env)
            ssa = _fresh_ssa()
            et = getattr(expr, 'dtype', getattr(expr, 'element_type', 'f32'))
            if et in ("i32", "i64"):
                int_preds = {"<": "slt", "<=": "sle", ">": "sgt",
                             ">=": "sge", "==": "eq", "!=": "ne"}
                pred = int_preds.get(expr.op, "eq")
                lines.append(
                    f'      {ssa} = llvm.icmp "{pred}" {left}, {right} : {et}'
                )
            else:
                pred = {"<": "olt", "<=": "ole", ">": "ogt",
                        ">=": "oge", "==": "oeq", "!=": "one"}.get(expr.op, "olt")
                lines.append(
                    f'      {ssa} = llvm.fcmp "{pred}" {left}, {right} : {et}'
                )
            return ssa

        # GpuSelect (branchless)
        if isinstance(expr, GpuSelect):
            cond = emit(expr.condition, env)
            true_v = emit(expr.true_val, env)
            false_v = emit(expr.false_val, env)
            ssa = _fresh_ssa()
            val_type = "f32"
            if isinstance(expr.true_val, GpuConstant):
                val_type = expr.true_val.element_type
            elif isinstance(expr.true_val, GpuBinaryOp):
                val_type = getattr(expr.true_val, 'element_type', 'f32')
            lines.append(
                f"      {ssa} = llvm.select {cond}, {true_v}, {false_v} : i1, {val_type}"
            )
            return ssa

        # GpuCast
        if isinstance(expr, GpuCast):
            inner = emit(expr.expr, env)
            ssa = _fresh_ssa()
            if expr.from_type == "i32" and expr.to_type == "f32":
                lines.append(
                    f"      {ssa} = llvm.sitofp {inner} : i32 to f32"
                )
            elif expr.from_type == "f32" and expr.to_type == "i32":
                lines.append(
                    f"      {ssa} = llvm.fptosi {inner} : f32 to i32"
                )
            elif expr.from_type == "i32" and expr.to_type == "i64":
                lines.append(
                    f"      {ssa} = llvm.sext {inner} : i32 to i64"
                )
            elif expr.from_type == "i64" and expr.to_type == "i32":
                lines.append(
                    f"      {ssa} = llvm.trunc {inner} : i64 to i32"
                )
            elif expr.from_type == "i1" and expr.to_type == "i32":
                lines.append(
                    f"      {ssa} = llvm.zext {inner} : i1 to i32"
                )
            elif expr.from_type == "i32" and expr.to_type == "i1":
                lines.append(
                    f"      {ssa} = llvm.trunc {inner} : i32 to i1"
                )
            elif expr.from_type == "i64" and expr.to_type == "f32":
                lines.append(
                    f"      {ssa} = llvm.sitofp {inner} : i64 to f32"
                )
            else:
                raise GPUScaffoldError(
                    f"unsupported cast: {expr.from_type} → {expr.to_type}"
                )
            return ssa

        # GpuReduce (per-thread scf.for)
        if isinstance(expr, GpuReduce):
            return _emit_reduce(expr, env)

        # _GpuLetExpr
        if isinstance(expr, _GpuLetExpr):
            val_ssa = emit(expr.value, env)
            new_env = dict(env)
            # If multi-value, store only the first component name
            if isinstance(val_ssa, list):
                new_env[expr.name] = val_ssa[0]
            else:
                new_env[expr.name] = val_ssa
            return emit(expr.body, new_env)

        # GpuArrayExpr: emit all components, return list of SSA names
        if isinstance(expr, GpuArrayExpr):
            result: list[str] = []
            for comp in expr.components:
                r = emit(comp, env)
                if isinstance(r, list):
                    result.extend(r)
                else:
                    result.append(r)
            return result

        # GpuExtractComponent: emit the array, return the k-th component
        if isinstance(expr, GpuExtractComponent):
            arr_result = emit(expr.array, env)
            if isinstance(arr_result, list):
                if expr.index < len(arr_result):
                    return arr_result[expr.index]
                raise GPUScaffoldError(
                    f"GpuExtractComponent index {expr.index} out of range (size {len(arr_result)})"
                )
            return arr_result

        # GpuIntrinsic: call NV device function via native PTX instructions
        if isinstance(expr, GpuIntrinsic):
            inner = emit(expr.arg, env)
            ssa = _fresh_ssa()
            if expr.intrinsic == "sqrt":
                lines.append(
                    f"      {ssa} = llvm.call @llvm.nvvm.sqrt.f({inner}) : (f32) -> f32"
                )
            elif expr.intrinsic == "exp":
                # exp(x) = ex2(x * log2(e)), log2(e) ≈ 1.44269504089
                scale = _fresh_ssa()
                scaled = _fresh_ssa()
                lines.append(f"      {scale} = llvm.mlir.constant(0x3fb8aa3b : f32) : f32")
                lines.append(f"      {scaled} = llvm.fmul {inner}, {scale}  : f32")
                lines.append(f"      {ssa} = llvm.call @llvm.nvvm.ex2.approx.f({scaled}) : (f32) -> f32")
            elif expr.intrinsic == "log":
                # log(x) = lg2(x) * ln(2), ln(2) ≈ 0.69314718056
                lg2 = _fresh_ssa()
                scale = _fresh_ssa()
                lines.append(f"      {lg2} = llvm.call @llvm.nvvm.lg2.approx.f({inner}) : (f32) -> f32")
                lines.append(f"      {scale} = llvm.mlir.constant(0x3f317218 : f32) : f32")
                lines.append(f"      {ssa} = llvm.fmul {lg2}, {scale}  : f32")
            else:
                raise GPUScaffoldError(f"unsupported intrinsic: {expr.intrinsic}")
            return ssa

        raise GPUScaffoldError(
            f"unhandled GpuExpr node: {type(expr).__name__}"
        )

    def _emit_input_load(expr: GpuInputLoad, env: dict[str, str]) -> str:
        """Emit a strided descriptor load for GpuInputLoad.

        Coords may be literal integers, SSA names, or placeholder names
        that resolve through env (e.g., "_iota_coord" → loop variable).

        Supports coord_offsets (per-axis additive offset) and
        coord_transforms (per-axis: "reverse:N" or "mod:N:S").
        Transforms are applied first, then offsets, then stride multiply.
        """
        prefix = f"in{expr.index}"
        et = getattr(expr, 'element_type', 'f32')
        resolved_coords: list[str] = []
        for c in expr.coords:
            if c in env:
                resolved_coords.append(env[c])
            else:
                resolved_coords.append(c)

        has_adjustments = bool(expr.coord_offsets) or bool(expr.coord_transforms)

        if not has_adjustments:
            all_literals = all(
                c.lstrip("-").isdigit() for c in resolved_coords
            )
            if all_literals:
                terms: list[str] = []
                for axis, coord_str in enumerate(resolved_coords):
                    coord_val = int(coord_str)
                    coord_name = _fresh_ssa()
                    lines.append(
                        f"      {coord_name} = llvm.mlir.constant({coord_val} : index) : i64"
                    )
                    term_name = _fresh_ssa()
                    lines.append(
                        f"      {term_name} = llvm.mul {coord_name}, %{prefix}_stride{axis}  : i64"
                    )
                    terms.append(term_name)

                current = f"%{prefix}_offset"
                for term in terms:
                    sum_name = _fresh_ssa()
                    lines.append(
                        f"      {sum_name} = llvm.add {current}, {term}  : i64"
                    )
                    current = sum_name

                linear_name = _fresh_ssa()
                zidx_name = _fresh_ssa()
                lines.append(
                    f"      {linear_name} = llvm.add {current}, {zidx_name} : i64"
                )
                lines.insert(
                    -2 if len(lines) >= 2 else len(lines),
                    f"      {zidx_name} = llvm.mlir.constant(0 : index) : i64",
                )

                ptr_name = _fresh_ssa()
                lines.append(
                    f"      {ptr_name} = llvm.getelementptr %{prefix}_aligned[{linear_name}] : (!llvm.ptr, i64) -> !llvm.ptr, {et}"
                )
                load_name = _fresh_ssa()
                lines.append(
                    f"      {load_name} = llvm.load {ptr_name} : !llvm.ptr -> {et}"
                )
                return load_name

        terms: list[str] = []
        for axis, coord_ssa in enumerate(resolved_coords):
            if coord_ssa.lstrip("-").isdigit():
                lit_name = _fresh_ssa()
                lines.append(
                    f"      {lit_name} = llvm.mlir.constant({int(coord_ssa)} : index) : i64"
                )
                coord_ssa = lit_name

            if expr.coord_transforms and axis < len(expr.coord_transforms):
                t = expr.coord_transforms[axis]
                if t.startswith("reverse:"):
                    N = int(t.split(":")[1])
                    n_ssa = _fresh_ssa()
                    lines.append(f"      {n_ssa} = llvm.mlir.constant({N - 1} : index) : i64")
                    rev_ssa = _fresh_ssa()
                    lines.append(f"      {rev_ssa} = llvm.sub {n_ssa}, {coord_ssa}  : i64")
                    coord_ssa = rev_ssa
                elif t.startswith("mod:"):
                    parts = t.split(":")
                    N = int(parts[1])
                    S = int(parts[2])
                    s_ssa = _fresh_ssa()
                    lines.append(f"      {s_ssa} = llvm.mlir.constant({S} : index) : i64")
                    shifted_ssa = _fresh_ssa()
                    lines.append(f"      {shifted_ssa} = llvm.add {coord_ssa}, {s_ssa}  : i64")
                    n_ssa = _fresh_ssa()
                    lines.append(f"      {n_ssa} = llvm.mlir.constant({N} : index) : i64")
                    mod_ssa = _fresh_ssa()
                    lines.append(f"      {mod_ssa} = llvm.urem {shifted_ssa}, {n_ssa}  : i64")
                    coord_ssa = mod_ssa

            if expr.coord_offsets and axis < len(expr.coord_offsets):
                off = expr.coord_offsets[axis]
                if off != 0:
                    off_ssa = _fresh_ssa()
                    lines.append(f"      {off_ssa} = llvm.mlir.constant({off} : index) : i64")
                    adj_ssa = _fresh_ssa()
                    lines.append(f"      {adj_ssa} = llvm.add {coord_ssa}, {off_ssa}  : i64")
                    coord_ssa = adj_ssa

            term_name = _fresh_ssa()
            lines.append(
                f"      {term_name} = llvm.mul {coord_ssa}, %{prefix}_stride{axis}  : i64"
            )
            terms.append(term_name)

        current = f"%{prefix}_offset"
        for term in terms:
            sum_name = _fresh_ssa()
            lines.append(
                f"      {sum_name} = llvm.add {current}, {term}  : i64"
            )
            current = sum_name

        ptr_name = _fresh_ssa()
        lines.append(
            f"      {ptr_name} = llvm.getelementptr %{prefix}_aligned[{current}] : (!llvm.ptr, i64) -> !llvm.ptr, {et}"
        )
        load_name = _fresh_ssa()
        lines.append(
            f"      {load_name} = llvm.load {ptr_name} : !llvm.ptr -> {et}"
        )
        return load_name

    def _emit_flat_load(expr: "GpuFlatLoad", env: dict[str, str]) -> str:
        """Emit a flat-index load for reshape/ravel."""
        prefix = f"in{expr.index}"
        et = getattr(expr, 'element_type', 'f32')
        resolved: list[str] = []
        for c in expr.coords:
            resolved.append(env[c] if c in env else c)

        shape = expr.output_shape
        rank = len(shape)
        planes: list[int] = []
        for k in range(rank):
            p = 1
            for d in shape[k + 1:]:
                p *= d
            planes.append(p)

        flat_ssa = None
        for k, coord_ssa in enumerate(resolved):
            if coord_ssa.lstrip("-").isdigit():
                lit = _fresh_ssa()
                lines.append(f"      {lit} = llvm.mlir.constant({int(coord_ssa)} : index) : i64")
                coord_ssa = lit
            if planes[k] != 1:
                plane_ssa = _fresh_ssa()
                lines.append(f"      {plane_ssa} = llvm.mlir.constant({planes[k]} : index) : i64")
                term = _fresh_ssa()
                lines.append(f"      {term} = llvm.mul {coord_ssa}, {plane_ssa}  : i64")
            else:
                term = coord_ssa
            if flat_ssa is None:
                flat_ssa = term
            else:
                s = _fresh_ssa()
                lines.append(f"      {s} = llvm.add {flat_ssa}, {term}  : i64")
                flat_ssa = s

        if flat_ssa is None:
            flat_ssa = _fresh_ssa()
            lines.append(f"      {flat_ssa} = llvm.mlir.constant(0 : index) : i64")

        linear = _fresh_ssa()
        lines.append(f"      {linear} = llvm.add %{prefix}_offset, {flat_ssa}  : i64")
        ptr = _fresh_ssa()
        lines.append(f"      {ptr} = llvm.getelementptr %{prefix}_aligned[{linear}] : (!llvm.ptr, i64) -> !llvm.ptr, {et}")
        load = _fresh_ssa()
        lines.append(f"      {load} = llvm.load {ptr} : !llvm.ptr -> {et}")
        return load

    def _emit_append_load(expr: "GpuAppendLoad", env: dict[str, str]) -> str:
        """Emit a branchless conditional load for append."""
        left_pf = f"in{expr.left_index}"
        right_pf = f"in{expr.right_index}"
        et = getattr(expr, 'element_type', 'f32')
        coord_ssa = resolved = None
        if expr.coords:
            c = expr.coords[0]
            coord_ssa = env[c] if c in env else c
            if coord_ssa.lstrip("-").isdigit():
                lit = _fresh_ssa()
                lines.append(f"      {lit} = llvm.mlir.constant({int(coord_ssa)} : index) : i64")
                coord_ssa = lit
        if coord_ssa is None:
            coord_ssa = "%idx"

        ls = _fresh_ssa()
        lines.append(f"      {ls} = llvm.mlir.constant({expr.left_size} : index) : i64")
        is_left = _fresh_ssa()
        lines.append(f'      {is_left} = llvm.icmp "ult" {coord_ssa}, {ls} : i64')
        right_idx = _fresh_ssa()
        lines.append(f"      {right_idx} = llvm.sub {coord_ssa}, {ls}  : i64")

        sel_idx = _fresh_ssa()
        lines.append(f"      {sel_idx} = llvm.select {is_left}, {coord_ssa}, {right_idx} : i1, i64")
        sel_base = _fresh_ssa()
        lines.append(f"      {sel_base} = llvm.select {is_left}, %{left_pf}_aligned, %{right_pf}_aligned : i1, !llvm.ptr")
        sel_off = _fresh_ssa()
        lines.append(f"      {sel_off} = llvm.select {is_left}, %{left_pf}_offset, %{right_pf}_offset : i1, i64")

        linear = _fresh_ssa()
        lines.append(f"      {linear} = llvm.add {sel_off}, {sel_idx}  : i64")
        ptr = _fresh_ssa()
        lines.append(f"      {ptr} = llvm.getelementptr {sel_base}[{linear}] : (!llvm.ptr, i64) -> !llvm.ptr, {et}")
        load = _fresh_ssa()
        lines.append(f"      {load} = llvm.load {ptr} : !llvm.ptr -> {et}")
        return load

    def _emit_reduce(expr: GpuReduce, env: dict[str, str]) -> str | list[str]:
        """Emit a per-thread scf.for reduction loop (scalar or array-valued)."""
        components = getattr(expr, "components", None) or []
        is_multi = len(components) > 0

        if is_multi and isinstance(expr.init, list) and len(expr.init) > 1:
            return _emit_multi_reduce(expr, components, env)
        elif is_multi:
            # Scalar fold with array body: iterate components into ONE accumulator
            return _emit_scalar_reduce(expr, env, components=components)
        return _emit_scalar_reduce(expr, env)

    def _emit_scalar_reduce(expr: GpuReduce, env: dict[str, str], components: list | None = None) -> str:
        """Emit a scalar (single-accumulator) scf.for loop."""
        init_raw = expr.init
        if isinstance(init_raw, list):
            init_ssa = emit(init_raw[0], env)
        else:
            init_ssa = emit(init_raw, env)

        dim = expr.dimension
        blk = _fresh_block()
        loop_label = f"{blk}_loop"
        c0_ssa = _fresh_ssa()
        c1_ssa = _fresh_ssa()
        cN_ssa = _fresh_ssa()
        idx_ssa = _fresh_ssa()
        acc_in_ssa = _fresh_ssa()

        lines.extend([
            f"      {c0_ssa} = llvm.mlir.constant(0 : index) : i64",
            f"      {c1_ssa} = llvm.mlir.constant(1 : index) : i64",
            f"      {cN_ssa} = llvm.mlir.constant({dim} : index) : i64",
        ])

        lines.append(
            f"      llvm.br ^{loop_label}({c0_ssa}, {init_ssa} : i64, f32)"
        )

        body_label = f"{loop_label}_body"
        done_label = f"{loop_label}_done"
        done_cond_ssa = _fresh_ssa()
        lines.append(f"    ^{loop_label}({idx_ssa}: i64, {acc_in_ssa}: f32):")
        lines.append(
            f"      {done_cond_ssa} = llvm.icmp \"uge\" {idx_ssa}, {cN_ssa} : i64"
        )
        lines.append(
            f"      llvm.cond_br {done_cond_ssa}, ^{done_label}({acc_in_ssa} : f32), ^{body_label}"
        )

        lines.append(f"    ^{body_label}:")

        body_env = dict(env)
        loop_var_name = getattr(expr, "loop_var_name", "_reduction_idx")
        is_reverse = getattr(expr, "reverse", False)
        if is_reverse:
            cNm1 = _fresh_ssa()
            lines.append(f"      {cNm1} = llvm.mlir.constant({dim - 1} : index) : i64")
            rev_idx = _fresh_ssa()
            lines.append(f"      {rev_idx} = llvm.sub {cNm1}, {idx_ssa}  : i64")
            body_env[loop_var_name] = rev_idx
            if loop_var_name == "_iota_coord":
                body_env["_iota_coord"] = rev_idx
        else:
            body_env[loop_var_name] = idx_ssa
            if loop_var_name == "_iota_coord":
                body_env["_iota_coord"] = idx_ssa

        if components:
            # Emit the component indexed by the loop variable
            # Use a select chain: if idx==0 emit comps[0], elif idx==1 emit comps[1], ...
            elem_ssa = None
            for k in range(len(components)):
                k_ssa = _fresh_ssa()
                lines.append(f"      {k_ssa} = llvm.mlir.constant({k} : index) : i64")
                eq_ssa = _fresh_ssa()
                lines.append(f"      {eq_ssa} = llvm.icmp \"eq\" {idx_ssa}, {k_ssa} : i64")
                comp_ssa = emit(components[k], body_env)
                if elem_ssa is None:
                    elem_ssa = comp_ssa
                else:
                    sel_ssa = _fresh_ssa()
                    lines.append(f"      {sel_ssa} = llvm.select {eq_ssa}, {comp_ssa}, {elem_ssa} : i1, f32")
                    elem_ssa = sel_ssa
        else:
            elem_ssa = emit(expr.body_expr, body_env)

        acc_next_ssa = _fresh_ssa()
        fold_llvm = llvm_op(expr.op, "f32")
        lines.append(
            f"      {acc_next_ssa} = {fold_llvm} {acc_in_ssa}, {elem_ssa}  : f32"
        )
        idx_next_ssa = _fresh_ssa()
        lines.append(
            f"      {idx_next_ssa} = llvm.add {idx_ssa}, {c1_ssa} : i64"
        )
        lines.append(
            f"      llvm.br ^{loop_label}({idx_next_ssa}, {acc_next_ssa} : i64, f32)"
        )

        result_ssa = _fresh_ssa()
        lines.append(
            f"    ^{done_label}({result_ssa}: f32):"
        )
        return result_ssa

    def _emit_multi_reduce(
        expr: GpuReduce, components: list, env: dict[str, str]
    ) -> list[str]:
        """Emit a multi-component (array-valued) reduction with K accumulators."""
        K = len(components)
        init_exprs = expr.init
        if not isinstance(init_exprs, list):
            init_exprs = [init_exprs] * K

        # Emit init values
        init_ssas: list[str] = []
        for k in range(K):
            init_ssas.append(emit(init_exprs[k], env))

        dim = expr.dimension
        blk = _fresh_block()
        loop_label = f"{blk}_loop"
        c0_ssa = _fresh_ssa()
        c1_ssa = _fresh_ssa()
        cN_ssa = _fresh_ssa()
        idx_ssa = _fresh_ssa()
        acc_in_ssas = [_fresh_ssa() for _ in range(K)]

        lines.extend([
            f"      {c0_ssa} = llvm.mlir.constant(0 : index) : i64",
            f"      {c1_ssa} = llvm.mlir.constant(1 : index) : i64",
            f"      {cN_ssa} = llvm.mlir.constant({dim} : index) : i64",
        ])

        # Loop branch: (idx, acc0, acc1, ..., accK-1)
        iter_types = "i64, " + ", ".join("f32" for _ in range(K))
        iter_args = f"{c0_ssa}, " + ", ".join(init_ssas)
        lines.append(
            f"      llvm.br ^{loop_label}({iter_args} : {iter_types})"
        )

        body_label = f"{loop_label}_body"
        done_label = f"{loop_label}_done"
        done_cond_ssa = _fresh_ssa()
        iter_params = f"{idx_ssa}: i64, " + ", ".join(f"{a}: f32" for a in acc_in_ssas)
        # Format: %v1, %v2 : type1, type2
        done_vals = ", ".join(acc_in_ssas)
        done_types = ", ".join("f32" for _ in range(K))
        lines.append(f"    ^{loop_label}({iter_params}):")
        lines.append(
            f"      {done_cond_ssa} = llvm.icmp \"uge\" {idx_ssa}, {cN_ssa} : i64"
        )
        lines.append(
            f"      llvm.cond_br {done_cond_ssa}, ^{done_label}({done_vals} : {done_types}), ^{body_label}"
        )

        # Loop body
        body_env = dict(env)
        loop_var_name = getattr(expr, "loop_var_name", "_reduction_idx")
        body_env[loop_var_name] = idx_ssa
        if loop_var_name == "_iota_coord":
            body_env["_iota_coord"] = idx_ssa

        lines.append(f"    ^{body_label}:")
        elem_ssas: list[str] = []
        for k in range(K):
            s = emit(components[k], body_env)
            elem_ssas.append(s)

        # Accumulate each component
        acc_next_ssas: list[str] = []
        fold_llvm = llvm_op(expr.op, "f32")
        for k in range(K):
            nxt = _fresh_ssa()
            lines.append(
                f"      {nxt} = {fold_llvm} {acc_in_ssas[k]}, {elem_ssas[k]}  : f32"
            )
            acc_next_ssas.append(nxt)

        idx_next_ssa = _fresh_ssa()
        lines.append(
            f"      {idx_next_ssa} = llvm.add {idx_ssa}, {c1_ssa} : i64"
        )

        # Loop back
        next_iter_args = f"{idx_next_ssa}, " + ", ".join(acc_next_ssas)
        next_iter_types = "i64, " + ", ".join("f32" for _ in range(K))
        lines.append(
            f"      llvm.br ^{loop_label}({next_iter_args} : {next_iter_types})"
        )

        # Done
        result_ssas = [_fresh_ssa() for _ in range(K)]
        result_params = ", ".join(f"{r}: f32" for r in result_ssas)
        lines.append(
            f"    ^{done_label}({result_params}):"
        )
        return result_ssas

    result = emit(expr, env)
    return result


def _gpu_descriptor_info(
    rank: int, prefixes: list[str]
) -> dict[str, dict[str, str]]:
    """Return SSA name mappings for descriptor fields.

    For each prefix (e.g. ``"in0"``, ``"out"``), returns a dict with keys:
    ``aligned``, ``offset``, ``size{axis}``, ``stride{axis}``.
    """
    info: dict[str, dict[str, str]] = {}
    for prefix in prefixes:
        entry: dict[str, str] = {
            "aligned": f"%{prefix}_aligned",
            "offset": f"%{prefix}_offset",
        }
        for axis in range(rank):
            entry[f"size{axis}"] = f"%{prefix}_size{axis}"
            entry[f"stride{axis}"] = f"%{prefix}_stride{axis}"
        info[prefix] = entry
    return info
