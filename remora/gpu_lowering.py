"""Experimental MLIR GPU module scaffolds for the production NVIDIA path."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from remora._gpu_map_support import (
    F32MapKernel,
    F32MapOperation,
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


def _scan_kernel(function: HIRFunction) -> HIRFunction:
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
    if kernel.num_inputs == 2:
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
        lines.extend(
            [
                f"      %{prefixes[index]}_elem_ptr = llvm.getelementptr %{prefixes[index]}_aligned[%{prefixes[index]}_linear] : (!llvm.ptr, i64) -> !llvm.ptr, f32",
                f"      %x{index} = llvm.load %{prefixes[index]}_elem_ptr : !llvm.ptr -> {element_type}",
            ]
        )
        lines[-2] = lines[-2].replace(", f32", f", {element_type}")
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
