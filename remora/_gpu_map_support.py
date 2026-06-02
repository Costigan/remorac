"""Shared support for the current narrow GPU map slice."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from remora.hir import HIRFunction, HIRLit, HIRMap, HIRPrimCallable, HIRVar
from remora.types import FLOAT, ArrayType


@dataclass(frozen=True)
class F32MapOperation:
    op: str
    constant: float | None = None
    constant_side: str | None = None


@dataclass(frozen=True)
class F32MapKernel:
    shape: tuple[int, ...]
    operation: F32MapOperation
    num_inputs: int


def analyze_supported_f32_map_function(
    function: HIRFunction,
    *,
    on_unsupported: Callable[[str], Exception],
    context: str,
) -> F32MapKernel:
    if len(function.params) not in (1, 2):
        raise on_unsupported(f"{context} currently supports one or two input parameters")

    input_types: list[ArrayType] = []
    for param in function.params:
        if not (
            isinstance(param.type, ArrayType)
            and param.type.element == FLOAT
            and 1 <= param.type.rank <= 3
        ):
            raise on_unsupported(f"{context} currently supports rank-1 through rank-3 float inputs only")
        input_types.append(param.type)

    if not (
        isinstance(function.return_type, ArrayType)
        and function.return_type.element == FLOAT
        and 1 <= function.return_type.rank <= 3
    ):
        raise on_unsupported(f"{context} currently supports rank-1 through rank-3 float outputs only")

    if any(input_type.shape != function.return_type.shape for input_type in input_types):
        raise on_unsupported(f"{context} input and output shapes must match")

    if not (
        isinstance(function.body, HIRMap)
        and len(function.body.arrays) == len(function.params)
        and all(isinstance(array, HIRVar) for array in function.body.arrays)
        and [array.name for array in function.body.arrays] == [param.name for param in function.params]
        and isinstance(function.body.func, HIRPrimCallable)
    ):
        raise on_unsupported(f"{context} currently supports primitive maps over function parameters only")

    callable_ = function.body.func
    if len(function.params) == 1:
        if isinstance(callable_.left_arg, HIRLit) and callable_.left_arg.type == FLOAT:
            operation = F32MapOperation(callable_.op, float(callable_.left_arg.value), "left")
        elif isinstance(callable_.right_arg, HIRLit) and callable_.right_arg.type == FLOAT:
            operation = F32MapOperation(callable_.op, float(callable_.right_arg.value), "right")
        else:
            raise on_unsupported(f"{context} unary map requires a literal float section")
    elif callable_.left_arg is None and callable_.right_arg is None:
        operation = F32MapOperation(callable_.op)
    else:
        raise on_unsupported(f"{context} binary map does not support operator sections")

    return F32MapKernel(
        tuple(dim.value for dim in function.return_type.shape),
        operation,
        len(function.params),
    )
