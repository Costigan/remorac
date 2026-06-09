"""MLIR lowering for view operations: index, slice, transpose, reshape, etc."""

from __future__ import annotations

from typing import Any

from remora.hir import (
    HIRDrop,
    HIRExpr,
    HIRIndex,
    HIRRavel,
    HIRReshape,
    HIRReverse,
    HIRSlice,
    HIRSubarray,
    HIRTake,
    HIRTranspose,
)
from remora.types import ArrayType, StaticDim

from remora.lowering.types import (
    RemoraLoweringError,
    TensorEnv,
    _expr_result_type,
    _join_prefix,
    type_to_mlir,
)


def _lower_view_module(
    node: HIRIndex
    | HIRSlice
    | HIRTranspose
    | HIRReshape
    | HIRRavel
    | HIRTake
    | HIRDrop,
    functions: dict[str, Any],
    tensor_env: TensorEnv | None = None,
) -> str:
    from remora.lowering.module import _MLIRMainModuleBuilder

    code, result_value, result_type = _lower_view_result(
        node, functions, tensor_env
    )
    builder = _MLIRMainModuleBuilder(result_type)
    builder.add_block(code)
    return builder.render(result_value)


def _lower_view_result(
    node: HIRIndex
    | HIRSlice
    | HIRTranspose
    | HIRReshape
    | HIRRavel
    | HIRReverse
    | HIRTake
    | HIRDrop
    | HIRSubarray,
    functions: dict[str, Any],
    tensor_env: TensorEnv | None = None,
    prefix: str = "view",
) -> tuple[str, str, str]:
    if isinstance(node, HIRIndex):
        from remora.lowering.indexing import _lower_index_result

        return _lower_index_result(node, functions, tensor_env, prefix)

    from remora.lowering.tensor_ops import _lower_tensor_input

    input_code, input_name, input_type, _input_element_type = (
        _lower_tensor_input(
            node.array,
            _join_prefix(prefix, "in"),
            functions,
            tensor_env,
        )
    )
    result_type = type_to_mlir(node.result_type)
    result_name = f"%{_join_prefix(prefix, 'result')}"

    if isinstance(node, HIRTranspose):
        rank = node.result_type.rank
        if rank < 2:
            raise RemoraLoweringError(
                "transpose expects an array of rank at least 2"
            )
        permutation = [1, 0, *range(2, rank)]
        from remora.lowering.tensor_ops import (
            _identity_affine_map,
            _parallel_iterators,
        )
        transposed_map = _transpose_affine_map(rank, permutation)
        identity = _identity_affine_map(rank)
        iterators = _parallel_iterators(rank)
        empty_name = f"%{_join_prefix(prefix, 'empty')}"
        elem_type = type_to_mlir(node.result_type.element)
        result_line = f"""    {empty_name} = tensor.empty() : {result_type}
    {result_name} = linalg.generic {{
      indexing_maps = [{transposed_map}, {identity}],
      iterator_types = {iterators}
    }} ins({input_name} : {input_type}) outs({empty_name} : {result_type}) {{
    ^bb0(%in: {elem_type}, %out: {elem_type}):
      linalg.yield %in : {elem_type}
    }} -> {result_type}"""

    elif isinstance(node, HIRReshape):
        rank = node.result_type.rank
        shape_vals = ", ".join(str(d.value) for d in node.result_type.shape)
        shape_name = f"%{_join_prefix(prefix, 'reshape_shape')}"
        shape_code = f"    {shape_name} = arith.constant dense<[{shape_vals}]> : tensor<{rank}xindex>"
        input_code = f"{input_code}\n{shape_code}"
        result_line = f"    {result_name} = tensor.reshape {input_name}({shape_name}) : ({input_type}, tensor<{rank}xindex>) -> {result_type}"

    elif isinstance(node, HIRRavel):
        total = node.result_type.shape[0].value
        shape_name = f"%{_join_prefix(prefix, 'ravel_shape')}"
        shape_code = f"    {shape_name} = arith.constant dense<[{total}]> : tensor<1xindex>"
        input_code = f"{input_code}\n{shape_code}"
        result_line = f"    {result_name} = tensor.reshape {input_name}({shape_name}) : ({input_type}, tensor<1xindex>) -> {result_type}"

    elif isinstance(node, HIRReverse):
        rank = node.result_type.rank
        if rank < 1:
            raise RemoraLoweringError(
                "reverse expects an array of rank at least 1"
            )
        array_type = _expr_result_type(node.array)
        if not isinstance(array_type, ArrayType):
            raise RemoraLoweringError("reverse expects an array input")
        from remora.lowering.tensor_ops import (
            _identity_affine_map,
            _parallel_iterators,
            _reverse_first_axis_affine_map,
        )

        reverse_map = _reverse_first_axis_affine_map(array_type)
        identity = _identity_affine_map(rank)
        iterators = _parallel_iterators(rank)
        empty_name = f"%{_join_prefix(prefix, 'empty')}"
        elem_type = type_to_mlir(node.result_type.element)
        result_line = f"""    {empty_name} = tensor.empty() : {result_type}
    {result_name} = linalg.generic {{
      indexing_maps = [{reverse_map}, {identity}],
      iterator_types = {iterators}
    }} ins({input_name} : {input_type}) outs({empty_name} : {result_type}) {{
    ^bb0(%in: {elem_type}, %out: {elem_type}):
      linalg.yield %in : {elem_type}
    }} -> {result_type}"""

    elif isinstance(node, (HIRTake, HIRDrop)):
        array_type = _expr_result_type(node.array)
        if not isinstance(array_type, ArrayType):
            raise RemoraLoweringError("take/drop expects an array input")
        rank = array_type.rank

        if isinstance(node, HIRTake):
            offsets = ["0"] * rank
            sizes = [str(node.count)] + [
                str(d.value) for d in array_type.shape[1:]
            ]
        else:
            offsets = [str(node.count)] + ["0"] * (rank - 1)
            sizes = [
                str(array_type.shape[0].value - node.count)
            ] + [str(d.value) for d in array_type.shape[1:]]

        strides = ["1"] * rank
        result_line = (
            f"    {result_name} = tensor.extract_slice {input_name}"
            f"[{', '.join(offsets)}] [{', '.join(sizes)}] [{', '.join(strides)}] : "
            f"{input_type} to {result_type}"
        )
    elif isinstance(node, HIRSubarray):
        array_type = _expr_result_type(node.array)
        if not isinstance(array_type, ArrayType):
            raise RemoraLoweringError("subarray expects an array input")
        rank = array_type.rank
        offsets = [str(o.value) for o in node.offsets]
        sizes = [str(s.value) for s in node.sizes]
        strides = ["1"] * rank
        result_line = (
            f"    {result_name} = tensor.extract_slice {input_name}"
            f"[{', '.join(offsets)}] [{', '.join(sizes)}] [{', '.join(strides)}] : "
            f"{input_type} to {result_type}"
        )
    else:
        raise AssertionError(
            f"unhandled view node type {type(node).__name__}"
        )

    if "linalg.generic" in result_line:
        body = f"""{input_code}
{result_line}
"""
    else:
        body = f"""{input_code}
{result_line}
"""
    return body.rstrip(), result_name, result_type


def _transpose_affine_map(rank: int, permutation: list[int]) -> str:
    dims = ", ".join(f"d{axis}" for axis in range(rank))
    results = ", ".join(f"d{p}" for p in permutation)
    return f"affine_map<({dims}) -> ({results})>"


def _slice_affine_map(input_rank: int, result_rank: int, offset: int) -> str:
    dims = ", ".join(f"d{axis}" for axis in range(result_rank))
    results = ", ".join(f"d{axis} + {offset if axis == 0 else 0}" for axis in range(result_rank))
    # Pad with unused dims if input_rank > result_rank
    if input_rank > result_rank:
        results += ", " + ", ".join("0" for _ in range(input_rank - result_rank))
    return f"affine_map<({dims}) -> ({results})>"


def _lower_view_input(
    node: HIRIndex
    | HIRSlice
    | HIRTranspose
    | HIRReshape
    | HIRRavel
    | HIRReverse
    | HIRTake
    | HIRDrop
    | HIRSubarray,
    functions: dict[str, Any],
    prefix: str,
    tensor_env: TensorEnv | None = None,
) -> tuple[str, str, str, str]:
    code, result_value, result_type = _lower_view_result(
        node,
        functions,
        tensor_env,
        prefix,
    )
    element_type = type_to_mlir(node.result_type.element)
    return code, result_value, result_type, element_type


def _lower_transpose_module(
    node: HIRTranspose,
    functions: dict[str, Any],
    tensor_env: TensorEnv | None = None,
) -> str:
    from remora.lowering.module import _MLIRMainModuleBuilder

    code, result_value, result_type = _lower_transpose_result(
        node, functions, tensor_env
    )
    builder = _MLIRMainModuleBuilder(result_type)
    builder.add_block(code)
    return builder.render(result_value)


def _lower_transpose_result(
    node: HIRTranspose,
    functions: dict[str, Any],
    tensor_env: TensorEnv | None = None,
) -> tuple[str, str, str]:
    from remora.lowering.tensor_ops import _lower_tensor_input

    input_code, input_name, input_type, _input_element_type = (
        _lower_tensor_input(
            node.array,
            "trans_in",
            functions,
            tensor_env,
        )
    )
    result_type = type_to_mlir(node.result_type)
    rank = node.result_type.rank
    if rank < 2:
        raise RemoraLoweringError(
            "transpose expects an array of rank at least 2"
        )

    permutation = [1, 0, *range(2, rank)]
    perm_attr = "[" + ", ".join(map(str, permutation)) + "]"

    body = f"""{input_code}
    %trans_empty = tensor.empty() : {result_type}
    %transposed = linalg.transpose ins({input_name} : {input_type}) outs(%trans_empty : {result_type}) permutation = {perm_attr}
"""
    return body.rstrip(), "%transposed", result_type
