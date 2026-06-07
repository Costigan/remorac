"""MLIR lowering for indexing operations."""

from __future__ import annotations

from typing import Any

from remora.hir import (
    HIRExpr,
    HIRIndex,
    HIRLit,
    HIRSlice,
)
from remora.types import ArrayType, ScalarType

from remora.lowering.scalar import _Operand, _RegionEmitter
from remora.lowering.types import (
    RemoraLoweringError,
    TensorEnv,
    _expr_result_type,
    _join_prefix,
    type_to_mlir,
)


def _lower_index_module(
    node: HIRIndex,
    functions: dict[str, Any],
    tensor_env: TensorEnv | None = None,
) -> str:
    from remora.lowering.module import _MLIRMainModuleBuilder

    code, result_value, result_type = _lower_index_result(
        node, functions, tensor_env
    )
    builder = _MLIRMainModuleBuilder(result_type)
    builder.add_block(code)
    return builder.render(result_value)


def _lower_index_result(
    node: HIRIndex,
    functions: dict[str, Any],
    tensor_env: TensorEnv | None = None,
    prefix: str = "idx",
) -> tuple[str, str, str]:
    from remora.lowering.tensor_ops import _lower_tensor_input

    array_type = _expr_result_type(node.array)
    if not isinstance(array_type, ArrayType):
        raise RemoraLoweringError("indexing expects a tensor input")

    array_code, array_value, array_mlir_type, _element_type = (
        _lower_tensor_input(
            node.array,
            _join_prefix(prefix, "in"),
            functions,
            tensor_env,
        )
    )

    all_scalars = all(
        not isinstance(idx, HIRSlice) for idx in node.indices
    )
    full_indexing = all_scalars and len(node.indices) == array_type.rank

    index_lines: list[str] = []

    if full_indexing:
        index_names: list[str] = []
        for position, index in enumerate(node.indices):
            name = f"%{prefix}_{position}"
            index_names.append(name)
            if isinstance(index, HIRLit):
                index_lines.append(
                    f"    {name} = arith.constant {index.value} : index"
                )
            else:
                index_code, index_value_name = _lower_scalar_index_expr(
                    index, name, functions
                )
                index_lines.append(index_code)
                index_names[-1] = index_value_name

        result_type = type_to_mlir(node.result_type)
        indices = ", ".join(index_names)
        result_name = f"%{prefix}_result"
        result_line = f"    {result_name} = tensor.extract {array_value}[{indices}] : {array_mlir_type}"
        body = "\n".join(
            part
            for part in (
                array_code,
                "\n".join(index_lines),
                result_line,
            )
            if part
        )
        return body, result_name, result_type

    offsets: list[str] = []
    sizes: list[str] = []
    strides: list[str] = []
    extra_lines: list[str] = []

    for position, index in enumerate(node.indices):
        if isinstance(index, HIRSlice):
            offsets.append(str(index.start))
            sizes.append(str(index.end - index.start))
            strides.append("1")
        elif isinstance(index, HIRLit):
            offsets.append(str(index.value))
            sizes.append("1")
            strides.append("1")
        else:
            idx_name = f"%{prefix}_idx{position}"
            code, val_name = _lower_scalar_index_expr(
                index, idx_name, functions
            )
            extra_lines.append(code)
            offsets.append(val_name)
            sizes.append("1")
            strides.append("1")

    for position in range(len(node.indices), array_type.rank):
        offsets.append("0")
        sizes.append(str(array_type.shape[position].value))
        strides.append("1")

    result_mlir_type = type_to_mlir(node.result_type)

    result_name = f"%{prefix}_result"
    result_line = (
        f"    {result_name} = tensor.extract_slice {array_value}"
        f"[{', '.join(offsets)}] [{', '.join(sizes)}] [{', '.join(strides)}] : "
        f"{array_mlir_type} to {result_mlir_type}"
    )

    body = "\n".join(
        part
        for part in (array_code, *extra_lines, result_line)
        if part
    )
    return body, result_name, result_mlir_type


def _lower_scalar_index_expr(
    expr: HIRExpr,
    name_hint: str,
    functions: dict[str, Any],
    env: dict[str, _Operand] | None = None,
) -> tuple[str, str]:
    """Lower an index expression to a scalar index SSA value."""
    if isinstance(expr, HIRLit):
        value = str(int(expr.value))
        return (
            f"    {name_hint} = arith.constant {value} : index",
            name_hint,
        )
    if isinstance(expr, HIRIndex) and isinstance(
        expr.result_type, ScalarType
    ):
        inner_prefix = name_hint.lstrip("%")
        index_code, index_val, _index_type = _lower_index_result(
            expr, functions, None, prefix=inner_prefix
        )
        lines = [index_code]
        lines.append(
            f"    {name_hint} = arith.index_cast {index_val} : i32 to index"
        )
        return "\n".join(lines), name_hint
    emitter = _RegionEmitter(
        input_name="", input_type="", functions=functions
    )
    value = emitter.emit_expr(expr, env or {})
    cast_name = f"{name_hint}_idx"
    lines = emitter.lines
    if value.type != "index":
        lines.append(
            f"    {cast_name} = arith.index_cast {value.value} : {value.type} to index"
        )
        return "\n".join(lines), cast_name
    if value.value != name_hint:
        lines.append(
            f"    {name_hint} = arith.index_cast {value.value} : {value.type} to index"
        )
        return "\n".join(lines), name_hint
    return "\n".join(lines), value.value
