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
from remora.hir import HIRFold, HIRFunction, HIRLit, HIRMap, HIRPrimCallable, HIRVar
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

    Uses a single-thread serial scan within a GPU kernel — correct for any
    size but potentially slow. A production implementation would use shared-
    memory Blelloch or Kogge-Stone scan.
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

    rank = 1
    desc_lines = _descriptor_load_lines("in", "%input_desc", rank)
    desc_lines.extend(_descriptor_load_lines("out", "%output_desc", rank))

    text = f"""module {{
  gpu.module @{module_name} {{
    llvm.func @{name}(%input_desc: !llvm.ptr, %output_desc: !llvm.ptr) attributes {{gpu.kernel, nvvm.kernel}} {{
{chr(10).join(desc_lines)}
      %tid32 = nvvm.read.ptx.sreg.tid.x : i32
      %tid = llvm.sext %tid32 : i32 to i64
      %c0_i64 = llvm.mlir.constant(0 : index) : i64
      %is_main = llvm.icmp "eq" %tid, %c0_i64 : i64
      llvm.cond_br %is_main, ^scan, ^done

    ^scan:
      %cN = llvm.mlir.constant({N} : index) : i64
      %init = llvm.mlir.constant(0.000000e+00 : f32) : f32
      %c1 = llvm.mlir.constant(1 : index) : i64
      %c0 = llvm.mlir.constant(0 : index) : i64
      llvm.br ^loop(%c0, %init : i64, f32)

    ^loop(%i: i64, %acc: f32):
      %loop_done = llvm.icmp "uge" %i, %cN : i64
      llvm.cond_br %loop_done, ^done, ^body

    ^body:
      %in_linear = llvm.add %in_offset, %i  : i64
      %in_ptr = llvm.getelementptr %in_aligned[%in_linear] : (!llvm.ptr, i64) -> !llvm.ptr, f32
      %elem = llvm.load %in_ptr : !llvm.ptr -> f32
      %next_acc = llvm.fadd %acc, %elem  : f32
      %out_linear = llvm.add %out_offset, %i  : i64
      %out_ptr = llvm.getelementptr %out_aligned[%out_linear] : (!llvm.ptr, i64) -> !llvm.ptr, f32
      llvm.store %next_acc, %out_ptr : f32, !llvm.ptr
      %next_i = llvm.add %i, %c1 : i64
      llvm.br ^loop(%next_i, %next_acc : i64, f32)

    ^done:
      llvm.return
    }}
  }}
}}"""
    return GPUModuleScaffold(text, module_name, name)


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
    """Validate rank 1-10 with positive dimensions; return normalized int tuple."""
    if not 1 <= len(shape) <= 10:
        raise GPUScaffoldError("GPU scaffold currently supports rank-1 through rank-10 shapes only")
    if any(dim <= 0 for dim in shape):
        raise GPUScaffoldError("GPU scaffold shape dimensions must be positive")
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
                f"      %{pf}_elem_ptr = llvm.getelementptr %{pf}_aligned[{linadj}] : (!llvm.ptr, i64) -> !llvm.ptr, f32",
            ])
        else:
            lines.extend(
                [
                    f"      %{pf}_elem_ptr = llvm.getelementptr %{pf}_aligned[%{pf}_linear] : (!llvm.ptr, i64) -> !llvm.ptr, f32",
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


# ---------------------------------------------------------------------------
# GPU append kernel
# ---------------------------------------------------------------------------


def build_descriptor_abi_f32_append_gpu_module(
    function: HIRFunction,
    *,
    module_name: str = "remora_gpu",
    kernel_name: str | None = None,
) -> GPUModuleScaffold:
    """Build a descriptor-ABI GPU module for f32 array append (concatenation)."""
    if len(function.params) != 2:
        raise GPUScaffoldError("GPU append supports two-parameter functions only")
    left_type = function.params[0].type
    right_type = function.params[1].type
    if not isinstance(left_type, ArrayType) or not isinstance(right_type, ArrayType):
        raise GPUScaffoldError("GPU append requires array inputs")
    if left_type.element != FLOAT or right_type.element != FLOAT:
        raise GPUScaffoldError("GPU append currently supports f32 input only")
    if left_type.rank != 1 or right_type.rank != 1:
        raise GPUScaffoldError("GPU append currently supports rank-1 input only")

    left_N = int(left_type.shape[0].value)
    right_N = int(right_type.shape[0].value)
    total_N = left_N + right_N
    name = kernel_name or f"remora_{function.name}_f32_append"
    _validate_scaffold_names(module_name, name)

    rank = 1
    desc_lines = _descriptor_load_lines("left", "%input0_desc", rank)
    desc_lines.extend(_descriptor_load_lines("right", "%input1_desc", rank))
    desc_lines.extend(_descriptor_load_lines("out", "%output_desc", rank))

    text = f"""module {{
  gpu.module @{module_name} {{
    llvm.func @{name}(%input0_desc: !llvm.ptr, %input1_desc: !llvm.ptr, %output_desc: !llvm.ptr) attributes {{gpu.kernel, nvvm.kernel}} {{
{chr(10).join(desc_lines)}
      %tid32 = nvvm.read.ptx.sreg.tid.x : i32
      %tid = llvm.sext %tid32 : i32 to i64
      %bid32 = nvvm.read.ptx.sreg.ctaid.x : i32
      %bid = llvm.sext %bid32 : i32 to i64
      %bdim32 = nvvm.read.ptx.sreg.ntid.x : i32
      %bdim = llvm.sext %bdim32 : i32 to i64
      %block_offset = llvm.mul %bid, %bdim : i64
      %idx = llvm.add %block_offset, %tid : i64
      %cN = llvm.mlir.constant({total_N} : index) : i64
      %inside = llvm.icmp "ult" %idx, %cN : i64
      llvm.cond_br %inside, ^body, ^done

    ^body:
      %cLeft = llvm.mlir.constant({left_N} : index) : i64
      %is_left = llvm.icmp "ult" %idx, %cLeft : i64
      llvm.cond_br %is_left, ^left_elem, ^right_elem

    ^left_elem:
      %left_linear = llvm.add %left_offset, %idx : i64
      %left_ptr = llvm.getelementptr %left_aligned[%left_linear] : (!llvm.ptr, i64) -> !llvm.ptr, f32
      %elem = llvm.load %left_ptr : !llvm.ptr -> f32
      llvm.br ^store

    ^right_elem:
      %right_idx = llvm.sub %idx, %cLeft : i64
      %right_linear = llvm.add %right_offset, %right_idx : i64
      %right_ptr = llvm.getelementptr %right_aligned[%right_linear] : (!llvm.ptr, i64) -> !llvm.ptr, f32
      %elem = llvm.load %right_ptr : !llvm.ptr -> f32
      llvm.br ^store

    ^store:
      %out_linear = llvm.add %out_offset, %idx : i64
      %out_ptr = llvm.getelementptr %out_aligned[%out_linear] : (!llvm.ptr, i64) -> !llvm.ptr, f32
      llvm.store %elem, %out_ptr : f32, !llvm.ptr
      llvm.br ^done

    ^done:
      llvm.return
    }}
  }}
}}"""
    return GPUModuleScaffold(text, module_name, name)


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


def _detect_nbody_gpu_kernel(function: HIRFunction) -> tuple[int, int] | None:
    """Return ``(N, M)`` if *function* matches the N-body force pattern.

    Pattern: ``map (lambda i -> fold + [0..0] (map (lambda j -> force_expr) (iota N))) (iota N)``
    where ``force_expr`` references param ``pos`` of shape ``[N, M]``.
    """
    from remora.hir import (
        HIRFold, HIRIota, HIRLambda, HIRMap, HIRPrimCallable, HIRReduce,
    )

    body = function.body
    if not isinstance(body, HIRMap) or body.cell_shape != ():
        return None
    if not isinstance(body.func, HIRLambda):
        return None
    outer_lambda = body.func
    outer_body = outer_lambda.body
    if not isinstance(outer_body, (HIRFold, HIRReduce)):
        return None
    if not isinstance(outer_body.func, HIRPrimCallable) or outer_body.func.left_arg is not None:
        return None
    if outer_body.func.right_arg is not None:
        return None
    fold_array = outer_body.array
    if not isinstance(fold_array, HIRMap) or fold_array.cell_shape != ():
        return None
    if not isinstance(fold_array.func, HIRLambda):
        return None
    if not isinstance(body.array, HIRIota) or not isinstance(fold_array.array, HIRIota):
        return None
    N_body = body.frame_shape[0].value
    N_fold = fold_array.frame_shape[0].value
    if N_body != N_fold:
        return None

    if len(function.params) != 1:
        return None
    from remora.types import ArrayType
    ptype = function.params[0].type
    if not isinstance(ptype, ArrayType) or ptype.rank != 2:
        return None
    M = ptype.shape[1].value
    return (N_body, M)


def build_descriptor_abi_nbody_gpu_module(
    function: HIRFunction,
    *,
    module_name: str = "remora_gpu",
    kernel_name: str | None = None,
    N: int,
    M: int,
) -> GPUModuleScaffold:
    """Build a descriptor-ABI GPU module for an N-body force kernel."""
    name = kernel_name or f"remora_{function.name}_nbody"
    shape = (N * M,)

    eps = 0.01  # softening parameter
    base_prefixes = ["in", "out"]

    # stride = M (contiguous components), offset = 0
    lines = [
        f"module {{",
        f"  gpu.module @{module_name} {{",
        f"    llvm.func @{name}(%input_desc: !llvm.ptr, %output_desc: !llvm.ptr) attributes {{gpu.kernel, nvvm.kernel}} {{",
        *[f"      {l}" for l in _descriptor_load_lines("in", "%input_desc", 2)],
        *[f"      {l}" for l in _descriptor_load_lines("out", "%output_desc", 2)],
        "",
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
        f"      %start = llvm.add %tid, %block_offset : i64",
        f"      %total = llvm.mlir.constant({N} : index) : i64",
        "      llvm.br ^bb_outer_loop(%start : i64)",
        "",
        "    ^bb_outer_loop(%i: i64):",
        "      %keep_going = llvm.icmp \"ult\" %i, %total : i64",
        "      llvm.cond_br %keep_going, ^bb_outer_body, ^bb_done",
        "",
        "    ^bb_outer_body:",
        f"      %M = llvm.mlir.constant({M} : index) : i64",
        f"      %eps = llvm.mlir.constant({eps:.6e} : f32) : f32",
        "      %zero_acc = llvm.mlir.constant(0.000000e+00 : f32) : f32",
        "",
        f"      %N_val = llvm.mlir.constant({N} : index) : i64",
        "      %c0 = llvm.mlir.constant(0 : index) : i64",
        "      %c1 = llvm.mlir.constant(1 : index) : i64",
        "      %c2 = llvm.mlir.constant(2 : index) : i64",
        "",
        "      llvm.br ^bb_inner_loop(%c0, %zero_acc, %zero_acc, %zero_acc : i64, f32, f32, f32)",
        "",
        "    ^bb_inner_loop(%j: i64, %acc0: f32, %acc1: f32, %acc2: f32):",
        "      %inner_cont = llvm.icmp \"ult\" %j, %N_val : i64",
        "      llvm.cond_br %inner_cont, ^bb_inner_body, ^bb_write",
        "",
        "    ^bb_inner_body:",
        "",
    ]

    # Load pos[i, k] for k=0,1,2
    for k in range(M):
        kc = f"{k}"
        lines.extend([
            f"      %in_offset_i{k} = llvm.mul %i, %in_stride0 : i64",
            f"      %in_linear_i{k} = llvm.add %in_offset, %in_offset_i{k} : i64",
            f"      %in_linear_i{k}_k = llvm.add %in_linear_i{k}, %c{k} : i64",
            f"      %in_ptr_i{k} = llvm.getelementptr %in_aligned[%in_linear_i{k}_k] : (!llvm.ptr, i64) -> !llvm.ptr, f32",
            f"      %pos_i{k} = llvm.load %in_ptr_i{k} : !llvm.ptr -> f32",
        ])

    # Load pos[j, k]
    for k in range(M):
        lines.extend([
            f"      %in_offset_j{k} = llvm.mul %j, %in_stride0 : i64",
            f"      %in_linear_j{k} = llvm.add %in_offset, %in_offset_j{k} : i64",
            f"      %in_linear_j{k}_k = llvm.add %in_linear_j{k}, %c{k} : i64",
            f"      %in_ptr_j{k} = llvm.getelementptr %in_aligned[%in_linear_j{k}_k] : (!llvm.ptr, i64) -> !llvm.ptr, f32",
            f"      %pos_j{k} = llvm.load %in_ptr_j{k} : !llvm.ptr -> f32",
        ])

    # D = pos[j] - pos[i]
    for k in range(M):
        lines.append(f"      %D{k} = llvm.fsub %pos_j{k}, %pos_i{k} : f32")

    # dsq = D0*D0 + D1*D1 + D2*D2 + eps
    sq_parts = []
    for k in range(M):
        lines.append(f"      %D{k}_sq = llvm.fmul %D{k}, %D{k} : f32")
        sq_parts.append(f"%D{k}_sq")
    lines.append(f"      %dsq_temp = llvm.fadd {sq_parts[0]}, {sq_parts[1]} : f32")
    for k in range(2, M):
        lines.append(f"      %dsq_temp_{k} = llvm.fadd %dsq_temp{'_'+str(k-1) if k>2 else ''}, {sq_parts[k]} : f32")
    dsq_name = f"%dsq_temp_{M-1}" if M > 2 else "%dsq_temp"
    lines.append(f"      %dsq = llvm.fadd {dsq_name}, %eps : f32")

    # sd = dsq (simplified; omits sqrt to avoid requiring math intrinsics)
    lines.extend([
        "      %sd_denom = llvm.fmul %dsq, %dsq : f32",
    ])

    # force[k] = D[k] / sd
    for k in range(M):
        lines.append(f"      %force{k} = llvm.fdiv %D{k}, %sd_denom : f32")

    # acc[k] += force[k]
    for k in range(M):
        lines.append(f"      %next_acc{k} = llvm.fadd %acc{k}, %force{k} : f32")

    lines.extend([
        f"      %next_j = llvm.add %j, %c1 : i64",
        f"      llvm.br ^bb_inner_loop(%next_j, %next_acc0, %next_acc1, %next_acc2 : i64, f32, f32, f32)",
        "",
        "    ^bb_write:",
        # Write accumulated forces to output
    ])

    for k in range(M):
        lines.extend([
            f"      %out_offset_i{k} = llvm.mul %i, %in_stride0 : i64",
            f"      %out_linear_i{k} = llvm.add %out_offset, %out_offset_i{k} : i64",
            f"      %out_linear_i{k}_k = llvm.add %out_linear_i{k}, %c{k} : i64",
            f"      %out_ptr_i{k} = llvm.getelementptr %out_aligned[%out_linear_i{k}_k] : (!llvm.ptr, i64) -> !llvm.ptr, f32",
            f"      llvm.store %acc{k}, %out_ptr_i{k} : f32, !llvm.ptr",
        ])

    lines.extend([
        "      %next_i = llvm.add %i, %grid_stride : i64",
        "      llvm.br ^bb_outer_loop(%next_i : i64)",
        "",
        "    ^bb_done:",
        "      llvm.return",
        "    }}",
        "  }}",
        "}}",
    ])

    return GPUModuleScaffold("\n".join(lines), module_name, name)
