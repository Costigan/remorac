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


def _validate_scaffold_names(module_name: str, kernel_name: str) -> None:
    if not module_name.isidentifier() or not kernel_name.isidentifier():
        raise GPUScaffoldError("GPU scaffold names must be valid identifiers")


def _build_f32_map_gpu_scaffold(
    kernel: F32MapKernel,
    *,
    module_name: str,
    kernel_name: str,
) -> GPUModuleScaffold:
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
    if not 1 <= len(shape) <= 3:
        raise GPUScaffoldError("GPU scaffold currently supports rank-1 through rank-3 shapes only")
    if any(dim <= 0 for dim in shape):
        raise GPUScaffoldError("GPU scaffold shape dimensions must be positive")
    return tuple(int(dim) for dim in shape)


def _memref_type(shape: tuple[int, ...]) -> str:
    return f"memref<{'x'.join(str(dim) for dim in shape)}xf32>"


def _product(shape: tuple[int, ...]) -> int:
    total = 1
    for dim in shape:
        total *= dim
    return total


def _indexing_lines(shape: tuple[int, ...]) -> tuple[str, list[str]]:
    if len(shape) == 1:
        return "", ["%idx"]
    if len(shape) == 2:
        return "\n".join(
            [
                f"        %dim1 = arith.constant {shape[1]} : index",
                "        %i0 = arith.divui %idx, %dim1 : index",
                "        %i1 = arith.remui %idx, %dim1 : index",
            ]
        ), ["%i0", "%i1"]
    plane = shape[1] * shape[2]
    return "\n".join(
        [
            f"        %dim2 = arith.constant {shape[2]} : index",
            f"        %plane = arith.constant {plane} : index",
            "        %i0 = arith.divui %idx, %plane : index",
            "        %rem0 = arith.remui %idx, %plane : index",
            "        %i1 = arith.divui %rem0, %dim2 : index",
            "        %i2 = arith.remui %rem0, %dim2 : index",
        ]
    ), ["%i0", "%i1", "%i2"]


def _operation_lines(kernel: F32MapKernel, memref_type: str, indices: list[str]) -> str:
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
    left = "%x0"
    right = "%c"
    if operation.constant_side == "left":
        left, right = right, left
    if operation.op == "*":
        return f"arith.mulf {left}, {right} : f32"
    if operation.op == "+":
        return f"arith.addf {left}, {right} : f32"
    if operation.op == "-":
        return f"arith.subf {left}, {right} : f32"
    if operation.op == "/":
        return f"arith.divf {left}, {right} : f32"
    raise GPUScaffoldError(f"GPU scaffold does not support operator {operation.op}")


def _binary_op_expr(operation: F32MapOperation) -> str:
    if operation.op == "*":
        return "arith.mulf %x0, %x1 : f32"
    if operation.op == "+":
        return "arith.addf %x0, %x1 : f32"
    if operation.op == "-":
        return "arith.subf %x0, %x1 : f32"
    if operation.op == "/":
        return "arith.divf %x0, %x1 : f32"
    raise GPUScaffoldError(f"GPU scaffold does not support operator {operation.op}")


def _f32_map_kernel(function: HIRFunction) -> F32MapKernel:
    return analyze_supported_f32_map_function(
        function,
        on_unsupported=GPUScaffoldError,
        context="GPU scaffold",
    )


def _i32_map_kernel(function: HIRFunction) -> I32MapKernel:
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
    total_name = "%out_size0"
    for axis in range(1, rank):
        name = f"%total{axis}"
        lines.append(f"      {name} = llvm.mul {total_name}, %out_size{axis}  : i64")
        total_name = name
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
    fields = ["ptr", "ptr", "i64", *(["i64"] * rank), *(["i64"] * rank)]
    return f"!llvm.struct<({', '.join(fields)})>"


def _descriptor_load_lines(prefix: str, descriptor_name: str, rank: int) -> list[str]:
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
    if rank == 1:
        return [
            "      %index_zero = llvm.mlir.constant(0 : index) : i64",
            "      %i0 = llvm.add %idx, %index_zero  : i64",
        ]
    if rank == 2:
        return [
            "      %i0 = llvm.udiv %idx, %out_size1  : i64",
            "      %i1 = llvm.urem %idx, %out_size1  : i64",
        ]
    if rank == 3:
        return [
            "      %plane = llvm.mul %out_size1, %out_size2  : i64",
            "      %i0 = llvm.udiv %idx, %plane  : i64",
            "      %rem0 = llvm.urem %idx, %plane  : i64",
            "      %i1 = llvm.udiv %rem0, %out_size2  : i64",
            "      %i2 = llvm.urem %rem0, %out_size2  : i64",
        ]
    raise GPUScaffoldError("descriptor ABI GPU module currently supports rank-1 through rank-3 shapes only")


def _linear_index_lines(prefix: str, rank: int) -> list[str]:
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
    if kernel.num_inputs == 2:
        return [f"      %y = {_descriptor_binary_op_expr(kernel.operation)}"]
    assert kernel.operation.constant is not None
    lines = [f"      %c = llvm.mlir.constant({kernel.operation.constant:.6e} : f32) : f32"]
    lines.append(f"      %y = {_descriptor_unary_op_expr(kernel.operation)}")
    return lines


def _descriptor_unary_op_expr(operation: F32MapOperation) -> str:
    left = "%x0"
    right = "%c"
    if operation.constant_side == "left":
        left, right = right, left
    if operation.op == "*":
        return f"llvm.fmul {left}, {right}  : f32"
    if operation.op == "+":
        return f"llvm.fadd {left}, {right}  : f32"
    if operation.op == "-":
        return f"llvm.fsub {left}, {right}  : f32"
    if operation.op == "/":
        return f"llvm.fdiv {left}, {right}  : f32"
    raise GPUScaffoldError(f"descriptor ABI GPU module does not support operator {operation.op}")


def _descriptor_binary_op_expr(operation: F32MapOperation) -> str:
    if operation.op == "*":
        return "llvm.fmul %x0, %x1  : f32"
    if operation.op == "+":
        return "llvm.fadd %x0, %x1  : f32"
    if operation.op == "-":
        return "llvm.fsub %x0, %x1  : f32"
    if operation.op == "/":
        return "llvm.fdiv %x0, %x1  : f32"
    raise GPUScaffoldError(f"descriptor ABI GPU module does not support operator {operation.op}")


def _descriptor_i32_operation_lines(kernel: I32MapKernel) -> list[str]:
    if kernel.num_inputs == 2:
        return [f"      %y = {_descriptor_i32_binary_op_expr(kernel.operation)}"]
    assert kernel.operation.constant is not None
    lines = [f"      %c = llvm.mlir.constant({kernel.operation.constant} : i32) : i32"]
    lines.append(f"      %y = {_descriptor_i32_unary_op_expr(kernel.operation)}")
    return lines


def _descriptor_i32_unary_op_expr(operation: I32MapOperation) -> str:
    left = "%x0"
    right = "%c"
    if operation.constant_side == "left":
        left, right = right, left
    if operation.op == "*":
        return f"llvm.mul {left}, {right}  : i32"
    if operation.op == "+":
        return f"llvm.add {left}, {right}  : i32"
    if operation.op == "-":
        return f"llvm.sub {left}, {right}  : i32"
    if operation.op == "/":
        return f"llvm.sdiv {left}, {right}  : i32"
    raise GPUScaffoldError(f"descriptor ABI GPU module does not support operator {operation.op}")


def _descriptor_i32_binary_op_expr(operation: I32MapOperation) -> str:
    if operation.op == "*":
        return "llvm.mul %x0, %x1  : i32"
    if operation.op == "+":
        return "llvm.add %x0, %x1  : i32"
    if operation.op == "-":
        return "llvm.sub %x0, %x1  : i32"
    if operation.op == "/":
        return "llvm.sdiv %x0, %x1  : i32"
    raise GPUScaffoldError(f"descriptor ABI GPU module does not support operator {operation.op}")


@dataclass(frozen=True)
class F32ReductionKernel:
    size: int
    fold_op: str
    init: float
    input_op: str | None = None
    num_inputs: int = 1


def _f32_reduction_kernel(function: HIRFunction) -> F32ReductionKernel:
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
        _require_rank1_f32_param(function.params[0].type)
        size = function.params[0].type.shape[0].value  # type: ignore[union-attr]
        return F32ReductionKernel(size, function.body.func.op, float(function.body.init.value))

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
        return F32ReductionKernel(
            first_type.shape[0].value,
            function.body.func.op,
            float(function.body.init.value),
            input_op=mapped.func.op,
            num_inputs=2,
        )

    raise GPUScaffoldError("descriptor ABI GPU reduction input must be a parameter or binary map over parameters")


def _require_rank1_f32_param(param_type: object) -> ArrayType:
    if not (
        isinstance(param_type, ArrayType)
        and param_type.element == FLOAT
        and param_type.rank == 1
    ):
        raise GPUScaffoldError("descriptor ABI GPU reduction currently supports rank-1 float inputs only")
    return param_type


def _build_descriptor_abi_f32_reduction_gpu_module(
    kernel: F32ReductionKernel,
    *,
    module_name: str,
    kernel_name: str,
) -> GPUModuleScaffold:
    params = [
        *(f"%input{index}_desc: !llvm.ptr" for index in range(kernel.num_inputs)),
        "%output_desc: !llvm.ptr",
    ]
    body_lines = _reduction_kernel_body_lines(kernel)
    text = f"""module {{
  gpu.module @{module_name} {{
    llvm.func @{kernel_name}({", ".join(params)}) attributes {{gpu.kernel, nvvm.kernel}} {{
{chr(10).join(body_lines)}
      llvm.return
    }}
  }}
}}"""
    return GPUModuleScaffold(text, module_name, kernel_name)


def _reduction_kernel_body_lines(kernel: F32ReductionKernel) -> list[str]:
    prefixes = [f"in{index}" for index in range(kernel.num_inputs)]
    lines: list[str] = []
    for index, prefix in enumerate(prefixes):
        lines.extend(_descriptor_load_lines(prefix, f"%input{index}_desc", 1))
    lines.extend(_descriptor_load_lines("out", "%output_desc", 0))
    lines.extend(
        [
            f"      %init = llvm.mlir.constant({kernel.init:.6e} : f32) : f32",
            "      %zero = llvm.mlir.constant(0 : index) : i64",
            "      llvm.br ^bb1(%zero, %init : i64, f32)",
            "    ^bb1(%i: i64, %acc: f32):",
            "      %inside = llvm.icmp \"ult\" %i, %in0_size0 : i64",
            "      llvm.cond_br %inside, ^bb2, ^bb3(%acc : f32)",
            "    ^bb2:",
        ]
    )
    for prefix in prefixes:
        lines.extend(
            [
                f"      %{prefix}_term = llvm.mul %i, %{prefix}_stride0  : i64",
                f"      %{prefix}_linear = llvm.add %{prefix}_offset, %{prefix}_term  : i64",
                f"      %{prefix}_elem_ptr = llvm.getelementptr %{prefix}_aligned[%{prefix}_linear] : (!llvm.ptr, i64) -> !llvm.ptr, f32",
                f"      %{prefix}_x = llvm.load %{prefix}_elem_ptr : !llvm.ptr -> f32",
            ]
        )
    if kernel.num_inputs == 2:
        lines.append(f"      %item = {_reduction_binary_input_expr(kernel.input_op)}")
    else:
        lines.append("      %item = llvm.fadd %in0_x, %zero_f  : f32")
        lines.insert(-1, "      %zero_f = llvm.mlir.constant(0.000000e+00 : f32) : f32")
    lines.extend(
        [
            f"      %next_acc = {_reduction_fold_expr(kernel.fold_op)}",
            "      %one = llvm.mlir.constant(1 : index) : i64",
            "      %next_i = llvm.add %i, %one  : i64",
            "      llvm.br ^bb1(%next_i, %next_acc : i64, f32)",
            "    ^bb3(%result: f32):",
            "      %out_elem_ptr = llvm.getelementptr %out_aligned[%out_offset] : (!llvm.ptr, i64) -> !llvm.ptr, f32",
            "      llvm.store %result, %out_elem_ptr : f32, !llvm.ptr",
        ]
    )
    return lines


def _reduction_binary_input_expr(operation: str | None) -> str:
    if operation == "*":
        return "llvm.fmul %in0_x, %in1_x  : f32"
    if operation == "+":
        return "llvm.fadd %in0_x, %in1_x  : f32"
    if operation == "-":
        return "llvm.fsub %in0_x, %in1_x  : f32"
    if operation == "/":
        return "llvm.fdiv %in0_x, %in1_x  : f32"
    raise GPUScaffoldError(f"descriptor ABI GPU reduction does not support map operator {operation}")


def _reduction_fold_expr(operation: str) -> str:
    if operation == "+":
        return "llvm.fadd %acc, %item  : f32"
    if operation == "*":
        return "llvm.fmul %acc, %item  : f32"
    raise GPUScaffoldError(f"descriptor ABI GPU reduction does not support fold operator {operation}")
