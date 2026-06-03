"""Shared support for the current narrow GPU map slice."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from remora.hir import HIRFunction, HIRLit, HIRMap, HIRPrimCallable, HIRVar
from remora.types import BOOL, FLOAT, INT, ArrayType, ScalarType


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


@dataclass(frozen=True)
class I32MapOperation:
    op: str
    constant: int | None = None
    constant_side: str | None = None


@dataclass(frozen=True)
class I32MapKernel:
    shape: tuple[int, ...]
    operation: I32MapOperation
    num_inputs: int


def analyze_supported_map_function(
    function: HIRFunction,
    *,
    on_unsupported: Callable[[str], Exception],
    context: str,
    element_type: ScalarType,
) -> F32MapKernel | I32MapKernel:
    type_name = "float" if element_type == FLOAT else "int" if element_type == INT else "bool"

    if len(function.params) not in (1, 2):
        raise on_unsupported(f"{context} currently supports one or two input parameters")

    input_types: list[ArrayType] = []
    for param in function.params:
        if not (
            isinstance(param.type, ArrayType)
            and param.type.element == element_type
            and 1 <= param.type.rank <= 10
        ):
            raise on_unsupported(f"{context} currently supports rank-1 through rank-10 {type_name} inputs only")
        input_types.append(param.type)

    if not (
        isinstance(function.return_type, ArrayType)
        and function.return_type.element == element_type
        and 1 <= function.return_type.rank <= 10
    ):
        raise on_unsupported(f"{context} currently supports rank-1 through rank-10 {type_name} outputs only")

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
    if element_type == FLOAT:
        allowed_ops = {"+", "-", "*", "/"}
    elif element_type == INT:
        allowed_ops = {"+", "-", "*", "/"}
    else: # BOOL
        allowed_ops = {"&&", "||", "==", "!="}
        
    if callable_.op not in allowed_ops:
         raise on_unsupported(f"{context} does not support operator {callable_.op} for {type_name}")

    if len(function.params) == 1:
        if isinstance(callable_.left_arg, HIRLit) and callable_.left_arg.type == element_type:
            operation: F32MapOperation | I32MapOperation = F32MapOperation(callable_.op, callable_.left_arg.value, "left") if element_type == FLOAT else I32MapOperation(callable_.op, int(callable_.left_arg.value), "left")
        elif isinstance(callable_.right_arg, HIRLit) and callable_.right_arg.type == element_type:
            operation = F32MapOperation(callable_.op, callable_.right_arg.value, "right") if element_type == FLOAT else I32MapOperation(callable_.op, int(callable_.right_arg.value), "right")
        else:
            raise on_unsupported(f"{context} unary map requires a literal {type_name} section")
    elif callable_.left_arg is None and callable_.right_arg is None:
        operation = F32MapOperation(callable_.op) if element_type == FLOAT else I32MapOperation(callable_.op)
    else:
        raise on_unsupported(f"{context} binary map does not support operator sections")

    KernelClass = F32MapKernel if element_type == FLOAT else I32MapKernel
    return KernelClass(
        tuple(dim.value for dim in function.return_type.shape),
        operation,
        len(function.params),
    )


def analyze_supported_f32_map_function(
    function: HIRFunction,
    *,
    on_unsupported: Callable[[str], Exception],
    context: str,
) -> F32MapKernel:
    return analyze_supported_map_function(
        function,
        on_unsupported=on_unsupported,
        context=context,
        element_type=FLOAT,
    ) # type: ignore


def analyze_supported_i32_map_function(
    function: HIRFunction,
    *,
    on_unsupported: Callable[[str], Exception],
    context: str,
) -> I32MapKernel:
    return analyze_supported_map_function(
        function,
        on_unsupported=on_unsupported,
        context=context,
        element_type=INT,
    ) # type: ignore


def analyze_supported_bool_map_function(
    function: HIRFunction,
    *,
    on_unsupported: Callable[[str], Exception],
    context: str,
) -> I32MapKernel:
    return analyze_supported_map_function(
        function,
        on_unsupported=on_unsupported,
        context=context,
        element_type=BOOL,
    ) # type: ignore


def _require_scalar_array_params(
    function: HIRFunction,
    element_type: ScalarType,
    context: str,
    on_unsupported: Callable[[str], Exception],
) -> list[ArrayType]:
    input_types: list[ArrayType] = []
    type_name = "float" if element_type == FLOAT else "int" if element_type == INT else "bool"
    for param in function.params:
        if not (
            isinstance(param.type, ArrayType)
            and param.type.element == element_type
            and 1 <= param.type.rank <= 10
        ):
            raise on_unsupported(
                f"{context} currently supports rank-1 through rank-10 {type_name} inputs only"
            )
        input_types.append(param.type)
    return input_types
