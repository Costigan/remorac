"""Experimental MLIR GPU module scaffolds for the production NVIDIA path."""

from __future__ import annotations

from dataclasses import dataclass

from remora.errors import RemoraError
from remora.hir import HIRFunction, HIRLit, HIRMap, HIRPrimCallable, HIRVar
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
    if size <= 0:
        raise GPUScaffoldError("rank-1 GPU scaffold size must be positive")
    if not module_name.isidentifier() or not kernel_name.isidentifier():
        raise GPUScaffoldError("GPU scaffold names must be valid identifiers")

    memref_type = f"memref<{size}xf32>"
    text = f"""module {{
  gpu.module @{module_name} {{
    gpu.func @{kernel_name}(%input: {memref_type}, %output: {memref_type}) kernel {{
      %tid = gpu.thread_id x
      %bid = gpu.block_id x
      %bdim = gpu.block_dim x
      %block_base = arith.muli %bid, %bdim : index
      %idx = arith.addi %block_base, %tid : index
      %size = arith.constant {size} : index
      %inside = arith.cmpi ult, %idx, %size : index
      scf.if %inside {{
        %x = memref.load %input[%idx] : {memref_type}
        %c = arith.constant {multiplier:.6e} : f32
        %y = arith.mulf %x, %c : f32
        memref.store %y, %output[%idx] : {memref_type}
      }}
      gpu.return
    }}
  }}
}}"""
    return GPUModuleScaffold(text, module_name, kernel_name)


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
    size, multiplier = _rank1_f32_scale_map(function)
    return build_rank1_f32_unary_map_gpu_scaffold(
        size=size,
        multiplier=multiplier,
        module_name=module_name,
        kernel_name=kernel_name or f"remora_{function.name}_rank1_f32",
    )


def _rank1_f32_scale_map(function: HIRFunction) -> tuple[int, float]:
    if len(function.params) != 1:
        raise GPUScaffoldError("GPU scaffold currently supports one input parameter")
    param = function.params[0]
    if not (
        isinstance(param.type, ArrayType)
        and param.type.element == FLOAT
        and param.type.rank == 1
    ):
        raise GPUScaffoldError("GPU scaffold currently supports rank-1 float inputs only")
    if not (
        isinstance(function.return_type, ArrayType)
        and function.return_type.element == FLOAT
        and function.return_type.shape == param.type.shape
    ):
        raise GPUScaffoldError("GPU scaffold output must match the rank-1 float input")
    if not (
        isinstance(function.body, HIRMap)
        and len(function.body.arrays) == 1
        and isinstance(function.body.array, HIRVar)
        and function.body.array.name == param.name
        and isinstance(function.body.func, HIRPrimCallable)
    ):
        raise GPUScaffoldError("GPU scaffold currently supports primitive maps over the parameter only")

    callable_ = function.body.func
    if callable_.op != "*":
        raise GPUScaffoldError("GPU scaffold currently supports scale maps only")
    multiplier = _literal_f32_section_multiplier(callable_)
    return param.type.shape[0].value, multiplier


def _literal_f32_section_multiplier(callable_: HIRPrimCallable) -> float:
    if isinstance(callable_.left_arg, HIRLit) and callable_.left_arg.type == FLOAT:
        return float(callable_.left_arg.value)
    if isinstance(callable_.right_arg, HIRLit) and callable_.right_arg.type == FLOAT:
        return float(callable_.right_arg.value)
    raise GPUScaffoldError("GPU scaffold scale map requires a literal float section")
