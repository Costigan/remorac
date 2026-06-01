"""Experimental MLIR GPU module scaffolds for the production NVIDIA path."""

from __future__ import annotations

from dataclasses import dataclass

from remora.errors import RemoraError


class GPUScaffoldError(RemoraError):
    """Raised when an experimental GPU scaffold cannot be built."""


@dataclass(frozen=True)
class GPUModuleScaffold:
    text: str
    module_name: str
    kernel_name: str


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
