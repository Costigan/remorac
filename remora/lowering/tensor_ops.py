"""Tensor operation lowering for MLIR: maps, folds, iota, array literals."""

from __future__ import annotations

from typing import Any

from remora.hir import (
    HIRApply,
    HIRAppend,
    HIRArrayLit,
    HIRCall,
    HIRCast,
    HIRDrop,
    HIRExpr,
    HIRFilter,
    HIRFold,
    HIRFoldRight,
    HIRFunction,
    HIRGrade,
    HIRIf,
    HIRIndex,
    HIRIota,
    HIRLambda,
    HIRLet,
    HIRLit,
    HIRMap,
    HIRPrimCallable,
    HIRPrimOp,
    HIRRavel,
    HIRReduce,
    HIRReplicate,
    HIRReshape,
    HIRReverse,
    HIRRotate,
    HIRScan,
    HIRSlice,
    HIRSort,
    HIRSubarray,
    HIRScatterAdd,
    HIRTake,
    HIRTranspose,
    HIRVar,
    HIRWithShape,
    HIRIndicesOf,
)
from remora.types import ArrayType, BOOL, FLOAT, INT, ScalarType, SigmaType, StaticDim

from remora.lowering.scalar import (
    _Operand,
    _RegionEmitter,
    _arith_op,
    _cast_if_needed,
    _literal_value,
    _lower_callable_operand,
    _lower_scalar_value_for_fold_init,
)
from remora.lowering.types import (
    RemoraLoweringError,
    TensorEnv,
    _TensorValue,
    _expr_result_type,
    _is_scalar_type,
    _join_prefix,
    type_to_mlir,
)


# ---------------------------------------------------------------------------
# Affine map helpers
# ---------------------------------------------------------------------------


def _identity_affine_map(rank: int) -> str:
    dims = ", ".join(f"d{axis}" for axis in range(rank))
    results = ", ".join(f"d{axis}" for axis in range(rank))
    return f"affine_map<({dims}) -> ({results})>"


def _constant_affine_map(rank: int) -> str:
    """Affine map projecting from *rank* dimensions to a scalar (no results)."""
    dims = ", ".join(f"d{axis}" for axis in range(rank))
    return f"affine_map<({dims}) -> ()>"


def _parallel_iterators(rank: int) -> str:
    return "[" + ", ".join('"parallel"' for _axis in range(rank)) + "]"


def _tensor_rank_from_mlir_type(mlir_type: str) -> int:
    """Extract the rank from a MLIR tensor type string like 'tensor<2x3xi32>'."""
    if not mlir_type.startswith("tensor<"):
        return 0  # scalar type like 'i32', 'f32', etc.
    inner = mlir_type[len("tensor<") : -1]  # remove 'tensor<' and trailing '>'
    # Find the element type by looking for the last 'x' followed by non-digit
    # Simple: count dimensions by splitting on 'x' and checking if parts are digits
    parts = inner.split("x")
    dim_count = 0
    for p in parts:
        if p and p[0].isdigit():
            dim_count += 1
        else:
            break
    return dim_count


def _fold_iterators(rank: int) -> str:
    if rank < 1:
        raise RemoraLoweringError("fold rank must be at least 1")
    iterators = ['"reduction"', *('"parallel"' for _axis in range(rank - 1))]
    return "[" + ", ".join(iterators) + "]"


def _drop_first_affine_map(rank: int) -> str:
    if rank < 2:
        raise RemoraLoweringError("array-cell fold rank must be at least 2")
    dims = ", ".join(f"d{axis}" for axis in range(rank))
    results = ", ".join(f"d{axis}" for axis in range(1, rank))
    return f"affine_map<({dims}) -> ({results})>"


def _take_first_affine_map(rank: int, count: int) -> str:
    if count < 1 or count > rank:
        raise RemoraLoweringError("invalid affine map result rank")
    dims = ", ".join(f"d{axis}" for axis in range(rank))
    results = ", ".join(f"d{axis}" for axis in range(count))
    return f"affine_map<({dims}) -> ({results})>"


def _reverse_first_axis_affine_map(array_type: ArrayType) -> str:
    if array_type.rank < 1:
        raise RemoraLoweringError("reverse expects an array of rank at least 1")
    if not isinstance(array_type.shape[0], StaticDim):
        raise RemoraLoweringError("reverse requires a static leading dimension")
    dims = ", ".join(f"d{axis}" for axis in range(array_type.rank))
    results = [f"d{axis}" for axis in range(array_type.rank)]
    results[0] = f"{array_type.shape[0].value - 1} - d0"
    return f"affine_map<({dims}) -> ({', '.join(results)})>"


def _map_cell_iterators(frame_rank: int, cell_rank: int) -> str:
    if cell_rank < 1:
        raise RemoraLoweringError(
            "cell maps require at least one cell dimension"
        )
    iterators = [
        *('"parallel"' for _axis in range(frame_rank)),
        *('"reduction"' for _axis in range(cell_rank)),
    ]
    return "[" + ", ".join(iterators) + "]"


def _cell_element_affine_map(frame_rank: int, position: int) -> str:
    frame_dims = ", ".join(f"d{axis}" for axis in range(frame_rank))
    results = ", ".join(
        [f"d{axis}" for axis in range(frame_rank)] + [str(position)]
    )
    return f"affine_map<({frame_dims}) -> ({results})>"


# ---------------------------------------------------------------------------
# Iota / array literal lowering
# ---------------------------------------------------------------------------


def _lower_iota_module(node: HIRIota) -> str:
    from remora.lowering.module import _MLIRMainModuleBuilder

    result_type = type_to_mlir(node.result_type)
    element_type = type_to_mlir(node.result_type.element)
    if element_type != "i32":
        raise RemoraLoweringError(
            "iota lowering currently supports i32 results only"
        )

    body = f"""    %empty = tensor.empty() : {result_type}
    %result = linalg.generic {{
      indexing_maps = [affine_map<(d0) -> (d0)>],
      iterator_types = [\"parallel\"]
    }} outs(%empty : {result_type}) {{
    ^bb0(%out: {element_type}):
      %idx = linalg.index 0 : index
      %cast = arith.index_cast %idx : index to {element_type}
      linalg.yield %cast : {element_type}
    }} -> {result_type}
"""
    builder = _MLIRMainModuleBuilder(result_type)
    builder.add_block(body)
    return builder.render("%result")


def _lower_array_literal_module(node: HIRArrayLit) -> str:
    from remora.lowering.module import _MLIRMainModuleBuilder

    code, name, result_type, _element_type = _lower_tensor_input(
        node, "literal", {}
    )
    builder = _MLIRMainModuleBuilder(result_type)
    builder.add_block(code)
    return builder.render(name)


# ---------------------------------------------------------------------------
# Tensor input lowering (entry point for turning HIR exprs into SSA values)
# ---------------------------------------------------------------------------


def _lower_tensor_input(
    node: HIRExpr,
    prefix: str,
    functions: dict[str, HIRFunction],
    tensor_env: TensorEnv | None = None,
) -> tuple[str, str, str, str]:
    if isinstance(node, HIRVar):
        if tensor_env is None or node.name not in tensor_env:
            raise RemoraLoweringError(
                "only tensor literals, iota values, and descriptor inputs lower as tensor inputs so far"
            )
        value = tensor_env[node.name]
        return "", value.name, value.type, value.element_type

    if isinstance(node, HIRIota):
        result_type = type_to_mlir(node.result_type)
        element_type = type_to_mlir(node.result_type.element)
        code = f"""    %{prefix}_empty = tensor.empty() : {result_type}
    %{prefix} = linalg.generic {{
      indexing_maps = [affine_map<(d0) -> (d0)>],
      iterator_types = [\"parallel\"]
    }} outs(%{prefix}_empty : {result_type}) {{
    ^bb0(%{prefix}_out: {element_type}):
      %{prefix}_idx = linalg.index 0 : index
      %{prefix}_cast = arith.index_cast %{prefix}_idx : index to {element_type}
      linalg.yield %{prefix}_cast : {element_type}
    }} -> {result_type}"""
        return code, f"%{prefix}", result_type, element_type

    if isinstance(node, HIRArrayLit):
        result_type = type_to_mlir(node.result_type)
        element_type = type_to_mlir(node.result_type.element)
        flat = _flatten_array_literal(node)
        if not flat:
            return (
                f"    %{prefix} = tensor.empty() : {result_type}",
                f"%{prefix}",
                result_type,
                element_type,
            )
        lines = []
        names = []
        for index, literal in enumerate(flat):
            name = f"%{prefix}_c{index}"
            names.append(name)
            lines.append(
                f"    {name} = arith.constant {_literal_value(literal, element_type)} : {element_type}"
            )
        values = ", ".join(names)
        lines.append(
            f"    %{prefix} = tensor.from_elements {values} : {result_type}"
        )
        return "\n".join(lines), f"%{prefix}", result_type, element_type

    if isinstance(node, HIRAppend):
        return _lower_append_input(
            node, prefix, functions, tensor_env=tensor_env
        )

    if isinstance(node, (HIRFold, HIRReduce)) and isinstance(
        node.result_type, ArrayType
    ):
        code, name, result_type = _lower_fold_result(
            node, functions, tensor_env, prefix=prefix
        )
        return code, name, result_type, type_to_mlir(node.result_type.element)

    if isinstance(node, HIRScatterAdd):
        from remora.lowering.scalar import _lower_scalar_module

        target_code, target_name, target_type, target_elem = _lower_tensor_input(
            node.target, _join_prefix(prefix, "target"), functions, tensor_env
        )
        # Lower the index: literal → constant, non-literal → scalar module
        if isinstance(node.index, HIRLit) and node.index.type == INT:
            idx_val = int(node.index.value)
            idx_code = f"    %{_join_prefix(prefix, 'idx')} = arith.constant {idx_val} : index"
            idx_name = f"%{_join_prefix(prefix, 'idx')}"
        else:
            raise RemoraLoweringError(
                "scatter-add fold input only supports literal index values"
            )
        # Lower update
        if isinstance(node.update, HIRLit):
            lit_val = _literal_value(node.update, target_elem)
            update_name = f"%{_join_prefix(prefix, 'update')}"
            update_code = f"    {update_name} = arith.constant {lit_val} : {target_elem}"
        elif isinstance(node.update, HIRIndex) and len(node.update.indices) == 1:
            idx_item = node.update.indices[0]
            if isinstance(idx_item, HIRLit):
                idx_val = int(idx_item.value)
                arr_code, arr_name, arr_type, arr_elem = _lower_tensor_input(
                    node.update.array, _join_prefix(prefix, "idx_arr"), functions, tensor_env
                )
                update_name = f"%{_join_prefix(prefix, 'update')}"
                update_code = f"""{arr_code}
    {update_name}_pos = arith.constant {idx_val} : index
    {update_name} = tensor.extract {arr_name}[{update_name}_pos] : {arr_type}"""
            else:
                raise RemoraLoweringError(
                    "scatter-add cannot lower non-literal index in fold input"
                )
        else:
            raise RemoraLoweringError(
                f"scatter-add cannot lower update of type {type(node.update).__name__}"
            )
        result_type = type_to_mlir(node.result_type)
        result_name = f"%{prefix}"
        code = f"""{target_code}
{idx_code}
{update_code}
    {result_name}_extracted = tensor.extract {target_name}[{idx_name}] : {target_type}
    {result_name}_added = arith.addf {result_name}_extracted, {update_name} : {target_elem}
    {result_name} = tensor.insert {result_name}_added into {target_name}[{idx_name}] : {target_type}"""
        return code, result_name, result_type, target_elem

    if isinstance(node, (HIRMap, HIRApply)):
        if node.cell_shape:
            code, name, result_type = _lower_map_cell_result(
                node, functions, tensor_env
            )
            if not isinstance(node.result_type, ArrayType):
                raise RemoraLoweringError("cell-map tensor input must be an array")
            return code, name, result_type, type_to_mlir(node.result_type.element)
        return _lower_fold_input(
            node, functions, prefix, tensor_env=tensor_env
        )

    if isinstance(
        node,
        (
            HIRIndex,
            HIRSlice,
            HIRTranspose,
            HIRReshape,
            HIRRavel,
            HIRReverse,
            HIRTake,
            HIRDrop,
            HIRSubarray,
            HIRRotate,
            HIRScatterAdd,
        ),
    ):
        from remora.lowering.view_ops import _lower_view_input

        return _lower_view_input(node, functions, prefix, tensor_env=tensor_env)

    if isinstance(node, HIRWithShape):
        result_type = type_to_mlir(node.result_type)
        result_elem = type_to_mlir(node.result_type.element)
        rank = node.result_type.rank

        if isinstance(node.source, HIRLit):
            lit_val = _literal_value(node.source, result_elem)
            identity = _identity_affine_map(rank)
            iterators = _parallel_iterators(rank)
            val_name = f"%{prefix}_val"
            empty_name = f"%{prefix}_empty"
            target_name = f"%{prefix}"
            code = f"""    {val_name} = arith.constant {lit_val} : {result_elem}
    {empty_name} = tensor.empty() : {result_type}
    {target_name} = linalg.generic {{
      indexing_maps = [{identity}],
      iterator_types = {iterators}
    }} outs({empty_name} : {result_type}) {{
    ^bb0(%out: {result_elem}):
      linalg.yield {val_name} : {result_elem}
    }} -> {result_type}"""
            return code, target_name, result_type, result_elem

        # Non-literal source: recursively lower, then broadcast
        source_remora = _expr_result_type(node.source)
        if isinstance(source_remora, ArrayType):
            src_code, src_name, src_type, src_elem = _lower_tensor_input(
                node.source, f"{prefix}_src", functions, tensor_env
            )
            source_rank = source_remora.rank
            # Broadcast: source maps to last source_rank dims of target
            all_dims = ", ".join(f"d{a}" for a in range(rank))
            src_dims = ", ".join(f"d{a}" for a in range(rank - source_rank, rank))
            src_map = f"affine_map<({all_dims}) -> ({src_dims})>"
            tgt_map = _identity_affine_map(rank)
            iterators = _parallel_iterators(rank)
            empty_name = f"%{prefix}_empty"
            target_name = f"%{prefix}"
            code = f"""{src_code}
    {empty_name} = tensor.empty() : {result_type}
    {target_name} = linalg.generic {{
      indexing_maps = [{src_map}, {tgt_map}],
      iterator_types = {iterators}
    }} ins({src_name} : {src_type}) outs({empty_name} : {result_type}) {{
    ^bb0(%in: {src_elem}, %out: {result_elem}):
      linalg.yield %in : {result_elem}
    }} -> {result_type}"""
            return code, target_name, result_type, result_elem

        raise RemoraLoweringError(
            "only scalar-literal or array-source with-shape lowers as tensor input so far"
        )

    raise RemoraLoweringError(
        "only tensor literals and iota values lower as tensor inputs so far"
    )


def _lower_transpose_input(
    node: HIRTranspose,
    functions: dict[str, HIRFunction],
    prefix: str,
    tensor_env: TensorEnv | None = None,
) -> tuple[str, str, str, str]:
    from remora.lowering.view_ops import _lower_transpose_result

    code, result_value, result_type = _lower_transpose_result(
        node, functions, tensor_env
    )
    element_type = type_to_mlir(node.result_type.element)
    return code, result_value, result_type, element_type


# ---------------------------------------------------------------------------
# Scalar map / binary map modules (top-level entry points)
# ---------------------------------------------------------------------------


def _lower_scalar_map_module(
    node: HIRMap | HIRApply, functions: dict[str, HIRFunction]
) -> str:
    from remora.lowering.module import _MLIRMainModuleBuilder

    if node.frame_shape or node.cell_shape:
        raise RemoraLoweringError(
            "only rank-0 scalar maps lower as scalar MLIR so far"
        )

    result_type = type_to_mlir(node.result_type)
    emitter = _RegionEmitter(input_name="", input_type="")
    input_value = emitter.emit_expr(node.array, {})
    callable_lines, result_value = _lower_map_callable_result(
        node.func,
        functions,
        input_name=input_value.value,
        input_type=input_value.type,
        result_type=result_type,
        next_temp=emitter.next_temp,
    )
    builder = _MLIRMainModuleBuilder(result_type, functions=functions)
    builder.add_block("\n".join([*emitter.lines, *callable_lines]))
    return builder.render(result_value)


def _lower_scalar_map_binary_module(
    node: HIRMap | HIRApply, functions: dict[str, HIRFunction]
) -> str:
    from remora.lowering.module import _MLIRMainModuleBuilder

    if node.frame_shape or node.cell_shape:
        raise RemoraLoweringError("only rank-0 binary maps lower as scalar MLIR")
    if len(node.arrays) != 2:
        raise RemoraLoweringError(
            "binary scalar map requires exactly two inputs"
        )

    result_type = type_to_mlir(node.result_type)
    emitter = _RegionEmitter(
        input_name="", input_type="", functions=functions
    )
    left = emitter.emit_expr(node.arrays[0], {})
    right = emitter.emit_expr(node.arrays[1], {})
    callable_lines, result_value = _lower_map_binary_callable_result(
        node.func,
        functions,
        left_name=left.value,
        left_type=left.type,
        right_name=right.value,
        right_type=right.type,
        result_type=result_type,
        next_temp=emitter.next_temp,
    )
    builder = _MLIRMainModuleBuilder(result_type, functions=functions)
    builder.add_block("\n".join([*emitter.lines, *callable_lines]))
    return builder.render(result_value)


# ---------------------------------------------------------------------------
# Tensor map lowering
# ---------------------------------------------------------------------------


def _lower_iota_scalar_map_module(
    node: HIRMap | HIRApply,
    functions: dict[str, HIRFunction],
    tensor_env: TensorEnv | None = None,
) -> str:
    from remora.lowering.module import _MLIRMainModuleBuilder

    body, result_value, result_type = _lower_iota_scalar_map_result(
        node,
        functions,
        tensor_env,
    )
    builder = _MLIRMainModuleBuilder(result_type)
    builder.add_block(body)
    return builder.render(result_value)


def _lower_iota_scalar_map_result(
    node: HIRMap | HIRApply,
    functions: dict[str, HIRFunction],
    tensor_env: TensorEnv | None = None,
) -> tuple[str, str, str]:
    if node.cell_shape:
        return _lower_map_cell_result(node, functions, tensor_env)
    if not isinstance(node.result_type, ArrayType):
        raise RemoraLoweringError("map lowering requires an array result")

    input_code, input_name, input_type, input_element_type = (
        _lower_tensor_input(
            node.array,
            "input",
            functions,
            tensor_env,
        )
    )
    result_type = type_to_mlir(node.result_type)
    result_element_type = type_to_mlir(node.result_type.element)
    rank = node.result_type.rank
    identity = _identity_affine_map(rank)
    iterators = _parallel_iterators(rank)
    op_lines = _lower_map_callable_body(
        node.func,
        functions,
        input_name="%in",
        input_type=input_element_type,
        result_type=result_element_type,
    )

    body = f"""{input_code}
    %map_empty = tensor.empty() : {result_type}
    %mapped = linalg.generic {{
      indexing_maps = [{identity}, {identity}],
      iterator_types = {iterators}
    }} ins({input_name} : {input_type}) outs(%map_empty : {result_type}) {{
    ^bb0(%in: {input_element_type}, %out: {result_element_type}):
{op_lines}
    }} -> {result_type}
"""
    return body.rstrip(), "%mapped", result_type


def _lower_binary_map_module(
    node: HIRMap | HIRApply,
    functions: dict[str, HIRFunction],
    tensor_env: TensorEnv | None = None,
) -> str:
    from remora.lowering.module import _MLIRMainModuleBuilder

    body, result_value, result_type = _lower_binary_map_result(
        node, functions, tensor_env
    )
    builder = _MLIRMainModuleBuilder(result_type)
    builder.add_block(body)
    return builder.render(result_value)


def _lower_binary_map_result(
    node: HIRMap | HIRApply,
    functions: dict[str, HIRFunction],
    tensor_env: TensorEnv | None = None,
) -> tuple[str, str, str]:
    if node.cell_shape:
        raise RemoraLoweringError("binary cell-map MLIR lowering is deferred")
    if len(node.arrays) != 2:
        raise RemoraLoweringError("binary map requires exactly two inputs")
    if not isinstance(node.result_type, ArrayType):
        raise RemoraLoweringError(
            "ranked binary map lowering requires an array result"
        )

    result_type = type_to_mlir(node.result_type)
    result_element_type = type_to_mlir(node.result_type.element)

    def _lower_input(arg: HIRExpr, prefix: str):
        """Lower a single map input, promoting scalars to tensors."""
        if isinstance(arg, HIRLit) and _is_scalar_type(arg.type):
            scalar_code = (
                f"    %{prefix}_scalar = arith.constant "
                f"{_literal_value(arg, result_element_type)} : {result_element_type}"
            )
            rank = node.result_type.rank
            splat_identity = _identity_affine_map(rank)
            splat_iterators = _parallel_iterators(rank)
            splat_code = f"""    %{prefix}_empty = tensor.empty() : {result_type}
    %{prefix} = linalg.generic {{
      indexing_maps = [{splat_identity}],
      iterator_types = {splat_iterators}
    }} outs(%{prefix}_empty : {result_type}) {{
    ^bb0(%{prefix}_out: {result_element_type}):
      linalg.yield %{prefix}_scalar : {result_element_type}
    }} -> {result_type}"""
            return f"{scalar_code}\n{splat_code}", f"%{prefix}", result_type, result_element_type
        return _lower_tensor_input(arg, prefix, functions, tensor_env)

    left_code, left_name, left_type, left_element_type = _lower_input(
        node.arrays[0], "left"
    )
    right_code, right_name, right_type, right_element_type = _lower_input(
        node.arrays[1], "right"
    )

    result_rank = node.result_type.rank
    left_rank = _tensor_rank_from_mlir_type(left_type)
    right_rank = _tensor_rank_from_mlir_type(right_type)

    # Broadcasting indexing maps: each input projects from the principal
    # (result) rank down to its own rank, keeping the first k dimensions.
    def _broadcast_map(input_rank: int) -> str:
        if input_rank == result_rank:
            return _identity_affine_map(result_rank)
        if input_rank == 0:
            return _constant_affine_map(result_rank)
        dims = ", ".join(f"d{i}" for i in range(result_rank))
        kept = ", ".join(f"d{i}" for i in range(input_rank))
        return f"affine_map<({dims}) -> ({kept})>"

    left_map = _broadcast_map(left_rank)
    right_map = _broadcast_map(right_rank)
    identity = _identity_affine_map(result_rank)
    iterators = _parallel_iterators(result_rank)
    op_lines = _lower_map_binary_callable_body(
        node.func,
        functions,
        left_name="%left_map_in",
        left_type=left_element_type,
        right_name="%right_map_in",
        right_type=right_element_type,
        result_type=result_element_type,
    )

    body = f"""{left_code}
{right_code}
    %map_empty = tensor.empty() : {result_type}
    %mapped = linalg.generic {{
      indexing_maps = [{left_map}, {right_map}, {identity}],
      iterator_types = {iterators}
    }} ins({left_name}, {right_name} : {left_type}, {right_type}) outs(%map_empty : {result_type}) {{
    ^bb0(%left_map_in: {left_element_type}, %right_map_in: {right_element_type}, %out: {result_element_type}):
{op_lines}
    }} -> {result_type}
"""
    return body.rstrip(), "%mapped", result_type


# ---------------------------------------------------------------------------
# Cell map lowering
# ---------------------------------------------------------------------------


def _lower_map_cell_module(
    node: HIRMap | HIRApply,
    functions: dict[str, HIRFunction],
    tensor_env: TensorEnv | None = None,
) -> str:
    from remora.lowering.module import _MLIRMainModuleBuilder

    body, result_value, result_type = _lower_map_cell_result(
        node, functions, tensor_env
    )
    builder = _MLIRMainModuleBuilder(result_type)
    builder.add_block(body)
    return builder.render(result_value)


def _lower_map_cell_result(
    node: HIRMap | HIRApply,
    functions: dict[str, HIRFunction],
    tensor_env: TensorEnv | None = None,
) -> tuple[str, str, str]:
    if not isinstance(node.result_type, ArrayType):
        raise RemoraLoweringError("cell-map lowering requires an array result")
    if len(node.cell_shape) != 1:
        raise RemoraLoweringError("only rank-1 cell maps lower to MLIR so far")
    if node.result_type.rank != len(node.frame_shape):
        raise RemoraLoweringError(
            "only scalar-result cell maps lower to MLIR so far"
        )
    if not isinstance(node.func, HIRVar):
        raise RemoraLoweringError(
            "only lifted lambda cell maps lower to MLIR so far"
        )

    function = functions.get(node.func.name)
    if function is None:
        raise RemoraLoweringError(
            f"unknown cell-map function {node.func.name}"
        )
    if len(function.params) != 1:
        raise RemoraLoweringError(
            "only unary cell-map functions lower to MLIR so far"
        )

    param_name = function.params[0].name

    if isinstance(function.body, (HIRFold, HIRReduce)):
        return _lower_map_cell_fold_result(
            node, function, param_name, functions, tensor_env
        )

    return _lower_map_cell_index_result(
        node, function, param_name, functions, tensor_env
    )


def _lower_map_cell_fold_result(
    node: HIRMap | HIRApply,
    function: HIRFunction,
    param_name: str,
    functions: dict[str, HIRFunction],
    tensor_env: TensorEnv | None = None,
) -> tuple[str, str, str]:
    body_fold = function.body
    if not isinstance(body_fold.init, HIRLit):
        raise RemoraLoweringError(
            "only literal cell-fold initial values lower to MLIR so far"
        )
    # The cell-map parameter must be reduced somewhere. For nested folds like
    # (fold + 0 (fold + init m)), allow the inner fold to reduce the parameter.
    if not _reduces_param(body_fold.array, param_name):
        raise RemoraLoweringError(
            "cell-map fold must reduce the cell-map parameter (directly or via nested fold)"
        )
    input_remora_type = _expr_result_type(node.array)
    if not isinstance(input_remora_type, ArrayType):
        raise RemoraLoweringError("cell-map input must be an array")
    input_rank = input_remora_type.rank
    frame_rank = len(node.frame_shape)
    if input_rank != frame_rank + len(node.cell_shape):
        raise RemoraLoweringError(
            "cell-map frame and cell ranks do not match input rank"
        )

    input_code, input_name, input_type, input_element_type = (
        _lower_tensor_input(
            node.array,
            "input",
            functions,
            tensor_env,
        )
    )
    result_type = type_to_mlir(node.result_type)
    result_element_type = type_to_mlir(node.result_type.element)
    if input_element_type != result_element_type:
        raise RemoraLoweringError(
            "cell-map fold element type must match result element type"
        )
    init_value = _literal_value(body_fold.init, result_element_type)
    if not isinstance(body_fold.func, HIRPrimCallable):
        raise RemoraLoweringError(
            "only primitive cell-fold callables lower to MLIR so far"
        )
    fold_body = _lower_fold_callable_body(
        body_fold.func,
        functions,
        input_name="%in",
        input_type=input_element_type,
        acc_name="%acc",
        acc_type=result_element_type,
        result_type=result_element_type,
    )
    body = f"""{input_code}
    %map_empty = tensor.empty() : {result_type}
    %init = arith.constant {init_value} : {result_element_type}
    %filled = linalg.fill ins(%init : {result_element_type}) outs(%map_empty : {result_type}) -> {result_type}
    %mapped = linalg.generic {{
      indexing_maps = [{_identity_affine_map(input_rank)}, {_take_first_affine_map(input_rank, frame_rank)}],
      iterator_types = {_map_cell_iterators(frame_rank, len(node.cell_shape))}
    }} ins({input_name} : {input_type}) outs(%filled : {result_type}) {{
    ^bb0(%in: {input_element_type}, %acc: {result_element_type}):
{fold_body}
    }} -> {result_type}
"""
    return body.rstrip(), "%mapped", result_type


def _reduces_param(expr: HIRExpr, param_name: str) -> bool:
    """Check if *expr* directly or indirectly reduces *param_name*."""
    if isinstance(expr, HIRVar) and expr.name == param_name:
        return True
    if isinstance(expr, (HIRFold, HIRReduce)):
        return _reduces_param(expr.array, param_name)
    return False


def _collect_cell_indices(expr: HIRExpr, param_name: str) -> set[int]:
    """Collect all literal index values used on *param_name* in *expr*."""
    indices: set[int] = set()
    if (
        isinstance(expr, HIRIndex)
        and isinstance(expr.array, HIRVar)
        and expr.array.name == param_name
    ):
        for idx in expr.indices:
            if isinstance(idx, HIRLit):
                indices.add(int(idx.value))
        return indices
    for field_name in (
        "condition",
        "then_branch",
        "else_branch",
        "value",
        "body",
        "init",
        "array",
    ):
        child = getattr(expr, field_name, None)
        if isinstance(child, HIRExpr):
            indices |= _collect_cell_indices(child, param_name)
    if isinstance(expr, (HIRPrimOp, HIRMap, HIRApply)):
        for child_expr in getattr(expr, "args", []):
            if isinstance(child_expr, HIRExpr):
                indices |= _collect_cell_indices(child_expr, param_name)
        for child_expr in getattr(expr, "arrays", []):
            if isinstance(child_expr, HIRExpr):
                indices |= _collect_cell_indices(child_expr, param_name)
    if isinstance(expr, (HIRFold, HIRReduce)):
        if isinstance(expr.init, HIRExpr):
            indices |= _collect_cell_indices(expr.init, param_name)
        if isinstance(expr.array, HIRExpr):
            indices |= _collect_cell_indices(expr.array, param_name)
        if isinstance(expr.func, HIRVar):
            pass
    if isinstance(expr, HIRCall):
        for arg in expr.args:
            if isinstance(arg, HIRExpr):
                indices |= _collect_cell_indices(arg, param_name)
    if isinstance(expr, HIRLet):
        indices |= _collect_cell_indices(expr.value, param_name)
        indices |= _collect_cell_indices(expr.body, param_name)
    if isinstance(expr, HIRIndex):
        if isinstance(expr.array, HIRExpr):
            indices |= _collect_cell_indices(expr.array, param_name)
        for idx in expr.indices:
            if isinstance(idx, HIRExpr):
                indices |= _collect_cell_indices(idx, param_name)
    return indices


def _lower_map_cell_index_result(
    node: HIRMap | HIRApply,
    function: HIRFunction,
    param_name: str,
    functions: dict[str, HIRFunction],
    tensor_env: TensorEnv | None = None,
) -> tuple[str, str, str]:
    cell_indices = sorted(
        _collect_cell_indices(function.body, param_name)
    )
    if not cell_indices:
        raise RemoraLoweringError(
            "cell-map body must reference the cell parameter via indexing or fold"
        )
    cell_size = cell_indices[-1] + 1
    input_remora_type = _expr_result_type(node.array)
    if not isinstance(input_remora_type, ArrayType):
        raise RemoraLoweringError("cell-map input must be an array")
    input_rank = input_remora_type.rank
    frame_rank = len(node.frame_shape)
    if input_rank != frame_rank + 1:
        raise RemoraLoweringError("cell-map requires rank-1 cells")
    if cell_size > input_remora_type.shape[frame_rank].value:
        raise RemoraLoweringError("cell index out of bounds for cell size")

    input_code, input_name, input_type, input_element_type = (
        _lower_tensor_input(
            node.array,
            "input",
            functions,
            tensor_env,
        )
    )
    result_type = type_to_mlir(node.result_type)
    result_element_type = type_to_mlir(node.result_type.element)

    cell_maps = [
        _cell_element_affine_map(frame_rank, pos)
        for pos in range(cell_size)
    ]
    output_map = _identity_affine_map(frame_rank)
    ins_types = f"{input_type}"
    ins_names = input_name
    if cell_size > 1:
        ins_types = ", ".join([input_type] * cell_size)
        ins_names = ", ".join([input_name] * cell_size)
    map_str = ", ".join([*cell_maps, output_map])
    iterators = _parallel_iterators(frame_rank)
    cell_param_names = [f"%cell_{pos}" for pos in range(cell_size)]

    emitter = _RegionEmitter(
        input_name="", input_type="", functions=functions
    )
    env: dict[str, _Operand] = {}
    for pos in range(cell_size):
        env[f"{param_name}_{pos}"] = _Operand(
            cell_param_names[pos], [], input_element_type
        )

    rewritten_body = _rewrite_cell_indices(function.body, param_name, env)
    value = emitter.emit_expr(rewritten_body, env)
    cast_lines = _cast_if_needed(
        value.value, value.type, result_element_type, "%cell_result_cast"
    )
    result_value_name = (
        "%cell_result_cast" if cast_lines else value.value
    )
    region_body = "\n".join(
        [
            *emitter.lines,
            *cast_lines,
            f"      linalg.yield {result_value_name} : {result_element_type}",
        ]
    )

    body = f"""{input_code}
    %map_empty = tensor.empty() : {result_type}
    %mapped = linalg.generic {{
      indexing_maps = [{map_str}],
      iterator_types = {iterators}
    }} ins({ins_names} : {ins_types}) outs(%map_empty : {result_type}) {{
    ^bb0({', '.join(f'{name}: {input_element_type}' for name in cell_param_names)}, %out: {result_element_type}):
{region_body}
    }} -> {result_type}
"""
    return body.rstrip(), "%mapped", result_type


def _rewrite_cell_indices(
    expr: HIRExpr,
    param_name: str,
    env: dict[str, _Operand],
) -> HIRExpr:
    """Replace HIRIndex(cell_param, [HIRLit(pos)]) with HIRVar(param_name_pos)."""
    if (
        isinstance(expr, HIRIndex)
        and isinstance(expr.array, HIRVar)
        and expr.array.name == param_name
    ):
        if len(expr.indices) == 1 and isinstance(expr.indices[0], HIRLit):
            pos = int(expr.indices[0].value)
            var_name = f"{param_name}_{pos}"
            if var_name in env:
                return HIRVar(var_name, expr.result_type)
    if isinstance(expr, (HIRPrimOp, HIRMap, HIRApply)):
        return HIRPrimOp(
            expr.op,
            [
                _rewrite_cell_indices(arg, param_name, env)
                for arg in expr.args
            ],
            expr.result_type,
        )
    if isinstance(expr, HIRLet):
        return HIRLet(
            expr.name,
            expr.value_type,
            _rewrite_cell_indices(expr.value, param_name, env),
            _rewrite_cell_indices(expr.body, param_name, env),
            expr.result_type,
        )
    if isinstance(expr, HIRIf):
        return HIRIf(
            _rewrite_cell_indices(expr.condition, param_name, env),
            _rewrite_cell_indices(expr.then_branch, param_name, env),
            _rewrite_cell_indices(expr.else_branch, param_name, env),
            expr.result_type,
        )
    if isinstance(expr, HIRCast):
        return HIRCast(
            _rewrite_cell_indices(expr.value, param_name, env),
            expr.from_type,
            expr.to_type,
            expr.result_type,
        )
    return expr


# ---------------------------------------------------------------------------
# Fold lowering
# ---------------------------------------------------------------------------


def _lower_fold_module(
    node: HIRFold | HIRReduce | HIRFoldRight,
    functions: dict[str, HIRFunction],
    tensor_env: TensorEnv | None = None,
) -> str:
    from remora.lowering.module import _MLIRMainModuleBuilder

    body, result_value, result_type = _lower_fold_result(
        node, functions, tensor_env
    )
    builder = _MLIRMainModuleBuilder(result_type)
    builder.add_block(body)
    return builder.render(result_value)


def _lower_fold_result(
    node: HIRFold | HIRReduce,
    functions: dict[str, HIRFunction],
    tensor_env: TensorEnv | None = None,
    *,
    prefix: str = "",
) -> tuple[str, str, str]:
    if isinstance(node.result_type, ArrayType):
        return _lower_array_fold_result(
            node, functions, tensor_env, prefix=prefix
        )
    return _lower_scalar_fold_result(node, functions, tensor_env)


def _lower_scalar_fold_module(
    node: HIRFold | HIRReduce,
    functions: dict[str, HIRFunction],
    tensor_env: TensorEnv | None = None,
) -> str:
    from remora.lowering.module import _MLIRMainModuleBuilder

    body, result_value, result_type = _lower_scalar_fold_result(
        node, functions, tensor_env
    )
    builder = _MLIRMainModuleBuilder(result_type)
    builder.add_block(body)
    return builder.render(result_value)


def _lower_scalar_fold_result(
    node: HIRFold | HIRReduce,
    functions: dict[str, HIRFunction],
    tensor_env: TensorEnv | None = None,
) -> tuple[str, str, str]:
    input_code, input_name, input_type, input_element_type = _lower_fold_input(
        node.array,
        functions,
        tensor_env=tensor_env,
    )
    result_type = type_to_mlir(node.result_type)
    # Promote i32 accumulator to i64 for primitive fold callables
    # to avoid overflow on large reductions. HIRLambda bodies retain
    # their original types since the lambda operates in the Remora type system.
    acc_type = result_type
    truncate = False
    if result_type == "i32" and isinstance(node.func, HIRPrimCallable):
        acc_type = "i64"
        truncate = True

    init_code, init_value = _lower_scalar_value_for_fold_init(
        node.init,
        acc_type,
        functions=functions,
        env={},
        result_prefix="init_scalar",
    )
    fold_body = _lower_fold_callable_body(
        node.func,
        functions,
        input_name="%in",
        input_type=input_element_type,
        acc_name="%acc",
        acc_type=acc_type,
        result_type=acc_type,
    )

    trunc_block = ""
    if truncate:
        body = f"""{input_code}
{init_code}
    %init = tensor.from_elements {init_value} : tensor<{acc_type}>
    %folded = linalg.generic {{
      indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> ()>],
      iterator_types = [\"reduction\"]
    }} ins({input_name} : {input_type}) outs(%init : tensor<{acc_type}>) {{
    ^bb0(%in: {input_element_type}, %acc: {acc_type}):
{fold_body}
    }} -> tensor<{acc_type}>
    %wide = tensor.extract %folded[] : tensor<{acc_type}>
    %result = arith.trunci %wide : {acc_type} to {result_type}
"""
    else:
        body = f"""{input_code}
{init_code}
    %init = tensor.from_elements {init_value} : tensor<{acc_type}>
    %folded = linalg.generic {{
      indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> ()>],
      iterator_types = [\"reduction\"]
    }} ins({input_name} : {input_type}) outs(%init : tensor<{acc_type}>) {{
    ^bb0(%in: {input_element_type}, %acc: {acc_type}):
{fold_body}
    }} -> tensor<{acc_type}>
    %result = tensor.extract %folded[] : tensor<{acc_type}>
"""
    return body.rstrip(), "%result", result_type


def _lower_array_fold_module(
    node: HIRFold | HIRReduce,
    functions: dict[str, HIRFunction],
    tensor_env: TensorEnv | None = None,
) -> str:
    from remora.lowering.module import _MLIRMainModuleBuilder

    body, result_value, result_type = _lower_array_fold_result(
        node, functions, tensor_env
    )
    builder = _MLIRMainModuleBuilder(result_type)
    builder.add_block(body)
    return builder.render(result_value)


def _lower_array_fold_result(
    node: HIRFold | HIRReduce,
    functions: dict[str, HIRFunction],
    tensor_env: TensorEnv | None = None,
    *,
    prefix: str = "",
) -> tuple[str, str, str]:
    if not isinstance(node.result_type, ArrayType):
        raise RemoraLoweringError(
            "array fold lowering requires an array result"
        )

    input_remora_type = _expr_result_type(node.array)
    if (
        not isinstance(input_remora_type, ArrayType)
        or input_remora_type.rank < 2
    ):
        raise RemoraLoweringError(
            "array-cell fold lowering requires rank-2 or rank-3 input"
        )

    input_code, input_name, input_type, input_element_type = _lower_fold_input(
        node.array,
        functions,
        _join_prefix(prefix, "fold_input"),
        tensor_env=tensor_env,
    )
    init_code, init_name, init_type, init_element_type = _lower_tensor_input(
        node.init,
        _join_prefix(prefix, "fold_init"),
        functions,
        tensor_env,
    )
    result_type = type_to_mlir(node.result_type)
    result_element_type = type_to_mlir(node.result_type.element)
    if init_type != result_type:
        raise RemoraLoweringError(
            "array-cell fold init type must match result type"
        )
    if (
        input_element_type != result_element_type
        or init_element_type != result_element_type
    ):
        raise RemoraLoweringError(
            "array-cell fold element types must match"
        )

    rank = input_remora_type.rank
    fold_body = _lower_fold_callable_body(
        node.func,
        functions,
        input_name="%in",
        input_type=input_element_type,
        acc_name="%acc",
        acc_type=result_element_type,
        result_type=result_element_type,
    )

    folded = f"%{_join_prefix(prefix, 'folded')}"
    body = f"""{input_code}
{init_code}
    {folded} = linalg.generic {{
      indexing_maps = [{_identity_affine_map(rank)}, {_drop_first_affine_map(rank)}],
      iterator_types = {_fold_iterators(rank)}
    }} ins({input_name} : {input_type}) outs({init_name} : {result_type}) {{
    ^bb0(%in: {input_element_type}, %acc: {result_element_type}):
{fold_body}
    }} -> {result_type}
"""
    return body.rstrip(), folded, result_type


def _lower_fold_input(
    node: HIRExpr,
    functions: dict[str, HIRFunction],
    prefix: str = "",
    *,
    tensor_env: TensorEnv | None = None,
) -> tuple[str, str, str, str]:
    if isinstance(node, (HIRIota, HIRArrayLit, HIRWithShape)):
        return _lower_tensor_input(
            node, _join_prefix(prefix, "input"), functions, tensor_env
        )
    if isinstance(node, HIRVar):
        return _lower_tensor_input(
            node, _join_prefix(prefix, "input"), functions, tensor_env
        )

    if isinstance(
        node,
        (
            HIRIndex,
            HIRSlice,
            HIRTranspose,
            HIRReshape,
            HIRRavel,
            HIRTake,
            HIRDrop,
            HIRSubarray,
            HIRRotate,
        ),
    ):
        return _lower_tensor_input(
            node, _join_prefix(prefix, "input"), functions, tensor_env
        )

    if isinstance(node, HIRAppend):
        return _lower_tensor_input(
            node, _join_prefix(prefix, "input"), functions, tensor_env
        )

    if isinstance(node, (HIRMap, HIRApply)):
        if node.cell_shape:
            code, name, result_type = _lower_map_cell_result(
                node, functions, tensor_env
            )
            if not isinstance(node.result_type, ArrayType):
                raise RemoraLoweringError("cell-map fold input must be an array")
            return code, name, result_type, type_to_mlir(node.result_type.element)
        if not isinstance(node.result_type, ArrayType):
            raise RemoraLoweringError(
                "map fold input must have array type"
            )
        if len(node.arrays) == 2:
            return _lower_binary_map_fold_input(
                node, functions, prefix, tensor_env
            )
        if len(node.arrays) != 1:
            raise RemoraLoweringError(
                "only unary and binary scalar maps lower to fold MLIR so far"
            )

        input_code, input_name, input_type, input_element_type = (
            _lower_fold_input(
                node.array,
                functions,
                _join_prefix(prefix, "input"),
                tensor_env=tensor_env,
            )
        )
        map_type = type_to_mlir(node.result_type)
        map_element_type = type_to_mlir(node.result_type.element)
        rank = node.result_type.rank
        identity = _identity_affine_map(rank)
        iterators = _parallel_iterators(rank)
        map_empty = f"%{_join_prefix(prefix, 'map_empty')}"
        mapped = f"%{_join_prefix(prefix, 'mapped')}"
        map_in = f"%{_join_prefix(prefix, 'map_in')}"
        map_out = f"%{_join_prefix(prefix, 'map_out')}"
        map_body = _lower_map_callable_body(
            node.func,
            functions,
            input_name=map_in,
            input_type=input_element_type,
            result_type=map_element_type,
        )
        code = f"""{input_code}
    {map_empty} = tensor.empty() : {map_type}
    {mapped} = linalg.generic {{
      indexing_maps = [{identity}, {identity}],
      iterator_types = {iterators}
    }} ins({input_name} : {input_type}) outs({map_empty} : {map_type}) {{
    ^bb0({map_in}: {input_element_type}, {map_out}: {map_element_type}):
{map_body}
    }} -> {map_type}"""
        return code, mapped, map_type, map_element_type

    if isinstance(node, (HIRFold, HIRReduce)) and isinstance(
        node.result_type, ArrayType
    ):
        code, name, result_type = _lower_fold_result(
            node, functions, tensor_env, prefix=prefix
        )
        return code, name, result_type, type_to_mlir(node.result_type.element)

    raise RemoraLoweringError(
        "only folds over tensor literals, iota, or direct scalar maps lower to MLIR so far"
    )


def _lower_binary_map_fold_input(
    node: HIRMap | HIRApply,
    functions: dict[str, HIRFunction],
    prefix: str,
    tensor_env: TensorEnv | None = None,
) -> tuple[str, str, str, str]:
    left_code, left_name, left_type, left_element_type = _lower_fold_input(
        node.arrays[0],
        functions,
        _join_prefix(prefix, "left"),
        tensor_env=tensor_env,
    )
    right_code, right_name, right_type, right_element_type = (
        _lower_fold_input(
            node.arrays[1],
            functions,
            _join_prefix(prefix, "right"),
            tensor_env=tensor_env,
        )
    )
    map_type = type_to_mlir(node.result_type)
    map_element_type = type_to_mlir(node.result_type.element)
    rank = node.result_type.rank
    identity = _identity_affine_map(rank)
    iterators = _parallel_iterators(rank)
    map_empty = f"%{_join_prefix(prefix, 'map_empty')}"
    mapped = f"%{_join_prefix(prefix, 'mapped')}"
    map_left = f"%{_join_prefix(prefix, 'map_left')}"
    map_right = f"%{_join_prefix(prefix, 'map_right')}"
    map_out = f"%{_join_prefix(prefix, 'map_out')}"
    map_body = _lower_map_binary_callable_body(
        node.func,
        functions,
        left_name=map_left,
        left_type=left_element_type,
        right_name=map_right,
        right_type=right_element_type,
        result_type=map_element_type,
    )
    code = f"""{left_code}
{right_code}
    {map_empty} = tensor.empty() : {map_type}
    {mapped} = linalg.generic {{
      indexing_maps = [{identity}, {identity}, {identity}],
      iterator_types = {iterators}
    }} ins({left_name}, {right_name} : {left_type}, {right_type}) outs({map_empty} : {map_type}) {{
    ^bb0({map_left}: {left_element_type}, {map_right}: {right_element_type}, {map_out}: {map_element_type}):
{map_body}
    }} -> {map_type}"""
    return code, mapped, map_type, map_element_type


# ---------------------------------------------------------------------------
# Callable body lowering (map / fold)
# ---------------------------------------------------------------------------


def _lower_map_callable_body(
    callable_: object,
    functions: dict[str, HIRFunction],
    input_name: str,
    input_type: str,
    result_type: str,
) -> str:
    lines, result_value = _lower_map_callable_result(
        callable_,
        functions,
        input_name=input_name,
        input_type=input_type,
        result_type=result_type,
    )
    lines.append(f"      linalg.yield {result_value} : {result_type}")
    return "\n".join(lines)


def _lower_map_callable_result(
    callable_: object,
    functions: dict[str, HIRFunction],
    input_name: str,
    input_type: str,
    result_type: str,
    next_temp: int = 0,
) -> tuple[list[str], str]:
    if isinstance(callable_, HIRPrimCallable):
        return _lower_primitive_callable_result(
            callable_,
            input_name=input_name,
            input_type=input_type,
            result_type=result_type,
        )
    if isinstance(callable_, HIRVar):
        function = functions.get(callable_.name)
        if function is None:
            raise RemoraLoweringError(
                f"unknown map function {callable_.name}"
            )
        if len(function.params) != 1:
            raise RemoraLoweringError(
                "only unary map functions lower to MLIR so far"
            )
        emitter = _RegionEmitter(
            input_name=input_name,
            input_type=input_type,
            next_temp=next_temp,
            functions=functions,
        )
        value = emitter.emit_expr(
            function.body,
            {function.params[0].name: _Operand(input_name, [])},
        )
        lines = [
            *emitter.lines,
            *_cast_if_needed(
                value.value, value.type, result_type, "%result_cast"
            ),
        ]
        result_value = (
            "%result_cast" if value.type != result_type else value.value
        )
        return lines, result_value
    if isinstance(callable_, HIRLambda):
        if len(callable_.params) != 1:
            raise RemoraLoweringError(
                "only unary lambda map functions lower to MLIR so far"
            )
        emitter = _RegionEmitter(
            input_name=input_name,
            input_type=input_type,
            next_temp=next_temp,
            functions=functions,
        )
        value = emitter.emit_expr(
            callable_.body,
            {callable_.params[0].name: _Operand(input_name, [], input_type)},
        )
        lines = [
            *emitter.lines,
            *_cast_if_needed(
                value.value, value.type, result_type, "%result_cast"
            ),
        ]
        result_value = (
            "%result_cast" if value.type != result_type else value.value
        )
        return lines, result_value
    raise RemoraLoweringError(
        "only primitive and lifted function map callables lower to MLIR so far"
    )


def _lower_map_binary_callable_body(
    callable_: object,
    functions: dict[str, HIRFunction],
    left_name: str,
    left_type: str,
    right_name: str,
    right_type: str,
    result_type: str,
) -> str:
    lines, result_value = _lower_map_binary_callable_result(
        callable_,
        functions,
        left_name=left_name,
        left_type=left_type,
        right_name=right_name,
        right_type=right_type,
        result_type=result_type,
    )
    lines.append(f"      linalg.yield {result_value} : {result_type}")
    return "\n".join(lines)


def _lower_map_binary_callable_result(
    callable_: object,
    functions: dict[str, HIRFunction],
    left_name: str,
    left_type: str,
    right_name: str,
    right_type: str,
    result_type: str,
    next_temp: int = 0,
) -> tuple[list[str], str]:
    if isinstance(callable_, HIRPrimCallable):
        if callable_.left_arg is not None or callable_.right_arg is not None:
            raise RemoraLoweringError(
                "binary map operator sections are deferred"
            )
        return _lower_binary_primitive_callable_result(
            callable_,
            left_name=left_name,
            left_type=left_type,
            right_name=right_name,
            right_type=right_type,
            result_type=result_type,
        )
    if isinstance(callable_, HIRVar):
        function = functions.get(callable_.name)
        if function is None:
            raise RemoraLoweringError(
                f"unknown map function {callable_.name}"
            )
        if len(function.params) != 2:
            raise RemoraLoweringError(
                "binary map functions must take two parameters"
            )
        emitter = _RegionEmitter(
            input_name="",
            input_type="",
            next_temp=next_temp,
            functions=functions,
        )
        value = emitter.emit_expr(
            function.body,
            {
                function.params[0].name: _Operand(
                    left_name, [], left_type
                ),
                function.params[1].name: _Operand(
                    right_name, [], right_type
                ),
            },
        )
        lines = [
            *emitter.lines,
            *_cast_if_needed(
                value.value, value.type, result_type, "%result_cast"
            ),
        ]
        result_value = (
            "%result_cast" if value.type != result_type else value.value
        )
        return lines, result_value
    if isinstance(callable_, HIRLambda):
        if len(callable_.params) != 2:
            raise RemoraLoweringError(
                "binary map lambda functions must take two parameters"
            )
        emitter = _RegionEmitter(
            input_name="",
            input_type="",
            next_temp=next_temp,
            functions=functions,
        )
        value = emitter.emit_expr(
            callable_.body,
            {
                callable_.params[0].name: _Operand(
                    left_name, [], left_type
                ),
                callable_.params[1].name: _Operand(
                    right_name, [], right_type
                ),
            },
        )
        lines = [
            *emitter.lines,
            *_cast_if_needed(
                value.value, value.type, result_type, "%result_cast"
            ),
        ]
        result_value = (
            "%result_cast" if value.type != result_type else value.value
        )
        return lines, result_value
    raise RemoraLoweringError(
        "only primitive and lifted function binary map callables lower to MLIR so far"
    )


def _lower_binary_primitive_callable_result(
    callable_: HIRPrimCallable,
    left_name: str,
    left_type: str,
    right_name: str,
    right_type: str,
    result_type: str,
) -> tuple[list[str], str]:
    left_lines = _cast_if_needed(
        left_name, left_type, result_type, "%left_cast"
    )
    right_lines = _cast_if_needed(
        right_name, right_type, result_type, "%right_cast"
    )
    left_value = "%left_cast" if left_lines else left_name
    right_value = "%right_cast" if right_lines else right_name
    op = _arith_op(callable_.op, result_type)
    lines = [
        *left_lines,
        *right_lines,
        f"      %result = {op} {left_value}, {right_value} : {result_type}",
    ]
    return lines, "%result"


def _lower_primitive_callable_body(
    callable_: HIRPrimCallable,
    input_name: str,
    input_type: str,
    result_type: str,
) -> str:
    lines, result_value = _lower_primitive_callable_result(
        callable_,
        input_name=input_name,
        input_type=input_type,
        result_type=result_type,
    )
    return "\n".join(
        [*lines, f"      linalg.yield {result_value} : {result_type}"]
    )


def _lower_primitive_callable_result(
    callable_: HIRPrimCallable,
    input_name: str,
    input_type: str,
    result_type: str,
) -> tuple[list[str], str]:
    op_type = result_type
    if callable_.op in {"==", "!=", "<", "<="}:
        op_type = input_type

    left = _lower_callable_operand(callable_.left_arg, "%left", op_type)
    right = _lower_callable_operand(callable_.right_arg, "%right", op_type)
    if callable_.left_arg is None:
        left.value = input_name
        left.lines = _cast_if_needed(
            input_name, input_type, op_type, "%left_cast"
        )
        if left.lines:
            left.value = "%left_cast"
    if callable_.right_arg is None:
        right.value = input_name
        right.lines = _cast_if_needed(
            input_name, input_type, op_type, "%right_cast"
        )
        if right.lines:
            right.value = "%right_cast"

    op = _arith_op(callable_.op, op_type)

    sep = ", " if "cmp" in op else " "

    lines = [
        *left.lines,
        *right.lines,
        f"      %result = {op}{sep}{left.value}, {right.value} : {op_type}",
    ]
    return lines, "%result"


def _lower_fold_callable_body(
    callable_: object,
    functions: dict[str, HIRFunction],
    input_name: str,
    input_type: str,
    acc_name: str,
    acc_type: str,
    result_type: str,
) -> str:
    if isinstance(callable_, HIRPrimCallable):
        if callable_.left_arg is not None or callable_.right_arg is not None:
            raise RemoraLoweringError("fold operator sections are deferred")
        left_lines = _cast_if_needed(
            acc_name, acc_type, result_type, "%fold_left"
        )
        right_lines = _cast_if_needed(
            input_name, input_type, result_type, "%fold_right"
        )
        left_value = "%fold_left" if left_lines else acc_name
        right_value = "%fold_right" if right_lines else input_name
        op = _arith_op(callable_.op, result_type)
        sep = ", " if "cmp" in op else " "
        lines = [
            *left_lines,
            *right_lines,
            f"      %fold_result = {op}{sep}{left_value}, {right_value} : {result_type}",
            f"      linalg.yield %fold_result : {result_type}",
        ]
        return "\n".join(lines)

    if isinstance(callable_, HIRVar):
        function = functions.get(callable_.name)
        if function is None:
            raise RemoraLoweringError(
                f"unknown fold function {callable_.name}"
            )
        if len(function.params) != 2:
            raise RemoraLoweringError(
                "fold functions must take two parameters"
            )
        param_names = {function.params[0].name, function.params[1].name}
        if isinstance(function.body, HIRPrimOp) and all(
            isinstance(arg, HIRVar) and arg.name in param_names
            for arg in function.body.args
        ):
            left_lines = _cast_if_needed(
                acc_name, acc_type, result_type, "%fold_left"
            )
            right_lines = _cast_if_needed(
                input_name, input_type, result_type, "%fold_right"
            )
            left_value = "%fold_left" if left_lines else acc_name
            right_value = "%fold_right" if right_lines else input_name
            op = _arith_op(function.body.op[:-1], result_type)
            sep = ", " if "cmp" in op else " "
            lines = [
                *left_lines,
                *right_lines,
                f"      %fold_result = {op}{sep}{left_value}, {right_value} : {result_type}",
                f"      linalg.yield %fold_result : {result_type}",
            ]
            return "\n".join(lines)
        if isinstance(function.body, (HIRMap, HIRApply)) and not function.body.cell_shape:
            if (
                isinstance(function.body.func, HIRPrimCallable)
                and function.body.func.left_arg is None
                and function.body.func.right_arg is None
            ):
                op = _arith_op(function.body.func.op, result_type)
                sep = ", " if "cmp" in op else " "
                left_lines = _cast_if_needed(
                    acc_name, acc_type, result_type, "%fold_left"
                )
                right_lines = _cast_if_needed(
                    input_name, input_type, result_type, "%fold_right"
                )
                left_value = "%fold_left" if left_lines else acc_name
                right_value = (
                    "%fold_right" if right_lines else input_name
                )
                lines = [
                    *left_lines,
                    *right_lines,
                    f"      %fold_result = {op}{sep}{left_value}, {right_value} : {result_type}",
                    f"      linalg.yield %fold_result : {result_type}",
                ]
                return "\n".join(lines)
        emitter = _RegionEmitter(
            input_name="", input_type="", functions=functions
        )
        value = emitter.emit_expr(
            function.body,
            {
                function.params[0].name: _Operand(
                    acc_name, [], acc_type
                ),
                function.params[1].name: _Operand(
                    input_name, [], input_type
                ),
            },
        )
        cast_lines = _cast_if_needed(
            value.value, value.type, result_type, "%fold_result_cast"
        )
        result_value = (
            "%fold_result_cast" if cast_lines else value.value
        )
        lines = [
            *emitter.lines,
            *cast_lines,
            f"      linalg.yield {result_value} : {result_type}",
        ]
        return "\n".join(lines)

    if isinstance(callable_, HIRLambda):
        if len(callable_.params) != 2:
            raise RemoraLoweringError(
                "fold lambda functions must take two parameters"
            )
        emitter = _RegionEmitter(
            input_name="", input_type="", functions=functions
        )
        value = emitter.emit_expr(
            callable_.body,
            {
                callable_.params[0].name: _Operand(
                    acc_name, [], acc_type
                ),
                callable_.params[1].name: _Operand(
                    input_name, [], input_type
                ),
            },
        )
        cast_lines = _cast_if_needed(
            value.value, value.type, result_type, "%fold_result_cast"
        )
        result_value = (
            "%fold_result_cast" if cast_lines else value.value
        )
        lines = [
            *emitter.lines,
            *cast_lines,
            f"      linalg.yield {result_value} : {result_type}",
        ]
        return "\n".join(lines)

    raise RemoraLoweringError(
        "only primitive and lifted scalar fold callables lower to MLIR so far"
    )


# ---------------------------------------------------------------------------
# Array literal flattening
# ---------------------------------------------------------------------------


def _flatten_array_literal(node: HIRArrayLit) -> list[HIRLit]:
    flat: list[HIRLit] = []
    for element in node.elements:
        if isinstance(element, HIRLit):
            flat.append(element)
        elif isinstance(element, HIRArrayLit):
            flat.extend(_flatten_array_literal(element))
        else:
            raise RemoraLoweringError(
                "only scalar literal elements lower in tensor literals so far"
            )
    return flat


# ---------------------------------------------------------------------------
# Scan lowering
# ---------------------------------------------------------------------------


def _lower_rotate_module(
    node: HIRRotate, functions: dict[str, HIRFunction]
) -> str:
    from remora.lowering.module import _MLIRMainModuleBuilder

    if not isinstance(node.result_type, ArrayType):
        raise RemoraLoweringError("rotate lowering requires an array result")
    result_rank = node.result_type.rank
    if result_rank < 1:
        raise RemoraLoweringError("rotate lowering requires at least rank-1 array")

    input_code, input_name, input_type, input_element_type = _lower_tensor_input(
        node.array, "input", functions
    )
    result_type = type_to_mlir(node.result_type)
    result_element_type = type_to_mlir(node.result_type.element)
    shift = node.shift.value
    N = node.result_type.shape[0].value

    # Build N-D affine map and iterator types
    dims = ", ".join(f"d{i}" for i in range(result_rank))
    affine_map = f"affine_map<({dims}) -> ({dims})>"
    iterator_types = ", ".join(['"parallel"'] * result_rank)

    # Build trailing dimension indices and extract indices
    trailing_indices = ""
    extract_indices = "%wrapped"
    if result_rank > 1:
        trailing_defs = "\n".join(
            f"      %d{i} = linalg.index {i} : index" for i in range(1, result_rank)
        )
        trailing_indices = "\n" + trailing_defs
        extract_indices = "%wrapped, " + ", ".join(f"%d{i}" for i in range(1, result_rank))

    body = f"""{input_code}
    %rot_zero = arith.constant 0 : index
    %rot_N = arith.constant {N} : index
    %rot_shift = arith.constant {shift} : index
    %empty = tensor.empty() : {result_type}
    %rotated = linalg.generic {{
      indexing_maps = [{affine_map}],
      iterator_types = [{iterator_types}]
    }} outs(%empty : {result_type}) {{
    ^bb0(%out: {result_element_type}):
      %idx = linalg.index 0 : index
      %shifted = arith.addi %idx, %rot_shift : index
      %wrapped = arith.remsi %shifted, %rot_N : index{trailing_indices}
      %elem = tensor.extract {input_name}[{extract_indices}] : {input_type}
      linalg.yield %elem : {result_element_type}
    }} -> {result_type}"""

    builder = _MLIRMainModuleBuilder(result_type)
    builder.add_block(body)
    return builder.render("%rotated")


# ---------------------------------------------------------------------------
# Subarray lowering
# ---------------------------------------------------------------------------


def _lower_subarray_module(
    node: HIRSubarray, functions: dict[str, HIRFunction]
) -> str:
    from remora.lowering.module import _MLIRMainModuleBuilder

    input_code, input_name, input_type, _input_elem = _lower_tensor_input(
        node.array, "input", functions
    )
    result_type_mlir = type_to_mlir(node.result_type)
    input_type_mlir = type_to_mlir(_expr_result_type(node.array))

    offsets = ", ".join(str(o.value) for o in node.offsets)
    sizes = ", ".join(str(s.value) for s in node.sizes)
    strides = ", ".join("1" for _ in node.offsets)

    body = f"""{input_code}
    %extracted = tensor.extract_slice {input_name}[{offsets}] [{sizes}] [{strides}] : {input_type_mlir} to {result_type_mlir}"""

    builder = _MLIRMainModuleBuilder(result_type_mlir)
    builder.add_block(body)
    return builder.render("%extracted")


# ---------------------------------------------------------------------------
# Indices-of lowering
# ---------------------------------------------------------------------------


def _lower_indices_of_module(
    node: HIRIndicesOf, functions: dict[str, HIRFunction]
) -> str:
    from remora.lowering.module import _MLIRMainModuleBuilder

    result_type = type_to_mlir(node.result_type)
    result_elem = type_to_mlir(node.result_type.element)
    rank = node.result_type.rank
    input_rank = node.result_type.rank - 1 if rank > 1 else 0

    identity = _identity_affine_map(rank)
    iterators = _parallel_iterators(rank)

    # Generate the conditional chain: for each coordinate dim k, yield linalg.index (k+1)
    yield_val = f"%idx{input_rank}" if input_rank >= 1 else "%c0_i32"
    if rank == 2:  # rank-1 input → rank-2 result [1, N]
        body = f"""    %c0_i32 = arith.constant 0 : i32
    %empty = tensor.empty() : {result_type}
    %indices = linalg.generic {{
      indexing_maps = [{identity}],
      iterator_types = {iterators}
    }} outs(%empty : {result_type}) {{
    ^bb0(%out: {result_elem}):
      %d1 = linalg.index 1 : index
      %cast1 = arith.index_cast %d1 : index to {result_elem}
      linalg.yield %cast1 : {result_elem}
    }} -> {result_type}"""
    elif rank == 3:  # rank-2 input → rank-3 result [2, R, C]
        body = f"""    %empty = tensor.empty() : {result_type}
    %indices = linalg.generic {{
      indexing_maps = [{identity}],
      iterator_types = {iterators}
    }} outs(%empty : {result_type}) {{
    ^bb0(%out: {result_elem}):
      %d0 = linalg.index 0 : index
      %d1 = linalg.index 1 : index
      %d2 = linalg.index 2 : index
      %c0_idx = arith.constant 0 : index
      %is_row = arith.cmpi eq, %d0, %c0_idx : index
      %row_val = arith.index_cast %d1 : index to {result_elem}
      %col_val = arith.index_cast %d2 : index to {result_elem}
      %val = arith.select %is_row, %row_val, %col_val : {result_elem}
      linalg.yield %val : {result_elem}
    }} -> {result_type}"""
    else:
        raise RemoraLoweringError(f"unsupported rank {rank} for indices-of lowering")

    builder = _MLIRMainModuleBuilder(result_type)
    builder.add_block(body)
    return builder.render("%indices")


# ---------------------------------------------------------------------------
# With-shape lowering
# ---------------------------------------------------------------------------


def _lower_with_shape_module(
    node: HIRWithShape, functions: dict[str, HIRFunction]
) -> str:
    from remora.lowering.module import _MLIRMainModuleBuilder

    result_type = type_to_mlir(node.result_type)
    result_elem = type_to_mlir(node.result_type.element)
    rank = node.result_type.rank

    identity = _identity_affine_map(rank)
    iterators = _parallel_iterators(rank)

    # For scalar→tensor: splat the value
    source_remora = _expr_result_type(node.source)
    if isinstance(source_remora, ScalarType):
        if not isinstance(node.source, HIRLit):
            raise RemoraLoweringError(
                "only scalar-literal with-shape lowers as top-level module"
            )
        lit_val = _literal_value(node.source, result_elem)
        body = f"""    %val = arith.constant {lit_val} : {result_elem}
    %empty = tensor.empty() : {result_type}
    %result = linalg.generic {{
      indexing_maps = [{identity}],
      iterator_types = {iterators}
    }} outs(%empty : {result_type}) {{
    ^bb0(%out: {result_elem}):
      linalg.yield %val : {result_elem}
    }} -> {result_type}"""
        builder = _MLIRMainModuleBuilder(result_type)
        builder.add_block(body)
        return builder.render("%result")

    # Array→tensor: broadcast source tensor to target shape
    if isinstance(source_remora, ArrayType):
        source_rank = source_remora.rank
        src_code, src_name, src_type, src_elem = _lower_tensor_input(
            node.source, "src", functions, tensor_env=None
        )
        all_dims = ", ".join(f"d{a}" for a in range(rank))
        src_dims = ", ".join(f"d{a}" for a in range(rank - source_rank, rank))
        src_map = f"affine_map<({all_dims}) -> ({src_dims})>"
        tgt_map = _identity_affine_map(rank)
        body = f"""{src_code}
    %empty = tensor.empty() : {result_type}
    %result = linalg.generic {{
      indexing_maps = [{src_map}, {tgt_map}],
      iterator_types = {iterators}
    }} ins({src_name} : {src_type}) outs(%empty : {result_type}) {{
    ^bb0(%in: {src_elem}, %out: {result_elem}):
      linalg.yield %in : {result_elem}
    }} -> {result_type}"""
        builder = _MLIRMainModuleBuilder(result_type)
        builder.add_block(body)
        return builder.render("%result")

    raise RemoraLoweringError("only scalar→tensor with-shape lowers to MLIR so far")


def _lower_scatter_add_module(
    node: HIRScatterAdd, functions: dict[str, HIRFunction]
) -> str:
    from remora.lowering.module import _MLIRMainModuleBuilder

    target_code, target_name, target_type, target_elem = _lower_tensor_input(
        node.target, "target", functions
    )
    from remora.lowering.scalar import _lower_scalar_module

    if isinstance(node.update, HIRLit):
        lit_val = _literal_value(node.update, target_elem)
        update_code = f"    %update = arith.constant {lit_val} : {target_elem}"
        update_name = "%update"
    else:
        update_code = _lower_scalar_module(node.update, functions)
        update_name = "%result"

    if isinstance(node.index, HIRLit) and node.index.type == INT:
        idx_code = f"    %idx = arith.constant {int(node.index.value)} : index"
        idx_name = "%idx"
    else:
        idx_code = _lower_scalar_module(node.index, functions)
        idx_name = "%result"

    result_type = type_to_mlir(node.result_type)
    body = f"""{target_code}
{update_code}
{idx_code}
    %extracted = tensor.extract {target_name}[{idx_name}] : {target_type}
    %added = arith.addf %extracted, {update_name} : {target_elem}
    %result = tensor.insert %added into {target_name}[{idx_name}] : {target_type}"""
    builder = _MLIRMainModuleBuilder(result_type)
    builder.add_block(body)
    return builder.render("%result")


# ---------------------------------------------------------------------------
# Append lowering
# ---------------------------------------------------------------------------


def _lower_append_input(
    node: HIRAppend,
    prefix: str,
    functions: dict[str, HIRFunction],
    tensor_env: TensorEnv | None = None,
) -> tuple[str, str, str, str]:
    left_code, left_name, left_type, _left_elem = _lower_tensor_input(
        node.left, f"{prefix}_left", functions, tensor_env=tensor_env
    )
    right_code, right_name, right_type, _right_elem = _lower_tensor_input(
        node.right, f"{prefix}_right", functions, tensor_env=tensor_env
    )
    result_type_mlir = type_to_mlir(node.result_type)
    result_element_type = type_to_mlir(node.result_type.element)
    result_rank = node.result_type.rank

    left_remora = _expr_result_type(node.left)
    right_remora = _expr_result_type(node.right)
    left_shape = left_remora.shape if isinstance(left_remora, ArrayType) else ()
    right_shape = right_remora.shape if isinstance(right_remora, ArrayType) else ()

    left_dim = left_shape[0].value

    zero_offsets = ", ".join(["0"] * result_rank)
    left_sizes = ", ".join(str(d.value) for d in left_shape)
    right_sizes = ", ".join(str(d.value) for d in right_shape)
    strides = ", ".join(["1"] * result_rank)
    right_offsets = f"{left_dim}" + (
        ", 0" * (result_rank - 1) if result_rank > 1 else ""
    )

    empty_name = f"%{prefix}_empty"
    tmp_name = f"%{prefix}_tmp"
    result_name = f"%{prefix}"
    code = f"""{left_code}
{right_code}
    {empty_name} = tensor.empty() : {result_type_mlir}
    {tmp_name} = tensor.insert_slice {left_name} into {empty_name}[{zero_offsets}] [{left_sizes}] [{strides}] : {left_type} into {result_type_mlir}
    {result_name} = tensor.insert_slice {right_name} into {tmp_name}[{right_offsets}] [{right_sizes}] [{strides}] : {right_type} into {result_type_mlir}"""
    return code, result_name, result_type_mlir, result_element_type


def _lower_append_module(
    node: HIRAppend, functions: dict[str, HIRFunction]
) -> str:
    from remora.lowering.module import _MLIRMainModuleBuilder

    body, result_name, result_type_mlir, _result_element_type = (
        _lower_append_input(node, "result", functions)
    )
    builder = _MLIRMainModuleBuilder(result_type_mlir)
    builder.add_block(body)
    return builder.render(result_name)


# ---------------------------------------------------------------------------
# Scan lowering
# ---------------------------------------------------------------------------


def _lower_scan_module(
    node: HIRScan, functions: dict[str, HIRFunction]
) -> str:
    from remora.lowering.module import _MLIRMainModuleBuilder

    if not isinstance(node.result_type, ArrayType):
        raise RemoraLoweringError("scan lowering requires an array result")
    result_rank = node.result_type.rank
    if result_rank < 1:
        raise RemoraLoweringError("scan lowering requires at least rank 1")

    if result_rank == 1:
        return _lower_scan_rank1(node, functions)

    return _lower_scan_multirank(node, functions, result_rank)


def _lower_scan_rank1(
    node: HIRScan, functions: dict[str, HIRFunction]
) -> str:
    from remora.lowering.module import _MLIRMainModuleBuilder

    result_type = type_to_mlir(node.result_type)
    result_element_type = type_to_mlir(node.result_type.element)
    init_value_str = _literal_value(node.init, result_element_type)
    op_name = _arith_op(node.func.op, result_element_type)
    N = node.reduction_dim.value

    input_code, input_name, input_type, input_element_type = _lower_tensor_input(
        node.array, "input", functions
    )

    if node.right:
        body = f"""{input_code}
    %init = arith.constant {init_value_str} : {result_element_type}
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %cN = arith.constant {N} : index
    %cNminus1 = arith.constant {N - 1} : index
    %empty = tensor.empty() : {result_type}
    %filled = linalg.fill ins(%init : {result_element_type}) outs(%empty : {result_type}) -> {result_type}
    %scanned, %_carry = \"scf.for\"(%c0, %cN, %c1, %filled, %init) ({{
    ^bb0(%i: index, %acc_tensor: {result_type}, %carry: {result_element_type}):
      %rev_idx = arith.subi %cNminus1, %i : index
      %elem = tensor.extract {input_name}[%rev_idx] : {input_type}
      %next_carry = {op_name} %carry, %elem : {result_element_type}
      %stored = tensor.insert %next_carry into %acc_tensor[%rev_idx] : {result_type}
      \"scf.yield\"(%stored, %next_carry) : ({result_type}, {result_element_type}) -> ()
    }}) : (index, index, index, {result_type}, {result_element_type}) -> ({result_type}, {result_element_type})"""
    elif node.exclusive:
        body = f"""{input_code}
    %init = arith.constant {init_value_str} : {result_element_type}
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %cN = arith.constant {N} : index
    %empty = tensor.empty() : {result_type}
    %filled = linalg.fill ins(%init : {result_element_type}) outs(%empty : {result_type}) -> {result_type}
    %scanned, %_carry = \"scf.for\"(%c0, %cN, %c1, %filled, %init) ({{
    ^bb0(%i: index, %acc_tensor: {result_type}, %carry: {result_element_type}):
      %stored = tensor.insert %carry into %acc_tensor[%i] : {result_type}
      %elem = tensor.extract {input_name}[%i] : {input_type}
      %next_carry = {op_name} %carry, %elem : {result_element_type}
      \"scf.yield\"(%stored, %next_carry) : ({result_type}, {result_element_type}) -> ()
    }}) : (index, index, index, {result_type}, {result_element_type}) -> ({result_type}, {result_element_type})"""
    else:
        body = f"""{input_code}
    %init = arith.constant {init_value_str} : {result_element_type}
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %cN = arith.constant {N} : index
    %empty = tensor.empty() : {result_type}
    %filled = linalg.fill ins(%init : {result_element_type}) outs(%empty : {result_type}) -> {result_type}
    %scanned, %_carry = \"scf.for\"(%c0, %cN, %c1, %filled, %init) ({{
    ^bb0(%i: index, %acc_tensor: {result_type}, %carry: {result_element_type}):
      %elem = tensor.extract {input_name}[%i] : {input_type}
      %next_carry = {op_name} %carry, %elem : {result_element_type}
      %stored = tensor.insert %next_carry into %acc_tensor[%i] : {result_type}
      \"scf.yield\"(%stored, %next_carry) : ({result_type}, {result_element_type}) -> ()
    }}) : (index, index, index, {result_type}, {result_element_type}) -> ({result_type}, {result_element_type})"""

    builder = _MLIRMainModuleBuilder(result_type)
    builder.add_block(body)
    return builder.render("%scanned")


def _lower_scan_multirank(
    node: HIRScan, functions: dict[str, HIRFunction], rank: int
) -> str:
    """Lower a scan over rank >= 2 using nested loops.

    Outer loop over the leading dimension; inner loops over all trailing
    dimensions do element-wise carry updates via flat indexing.
    """
    from remora.lowering.module import _MLIRMainModuleBuilder

    N = node.reduction_dim.value
    result_type = type_to_mlir(node.result_type)
    result_element_type = type_to_mlir(node.result_type.element)

    if node.exclusive or node.right:
        raise RemoraLoweringError(
            "exclusive and right scans with rank >= 2 are deferred"
        )

    # Trailing dimensions
    trailing_dims = [d.value for d in node.result_type.shape[1:]]
    trailing_total = 1
    for d in trailing_dims:
        trailing_total *= d
    trailing_type = _tensor_type_mlir(trailing_dims, result_element_type)

    init_code, init_name, _init_type, _ielem = _lower_tensor_input(
        node.init, "scan_init", functions
    )
    input_code, input_name, input_type, _ielem2 = _lower_tensor_input(
        node.array, "input", functions
    )
    op_name = _arith_op(node.func.op, result_element_type)

    # Build constant definitions for all dimensions
    dim_consts = "".join(
        f"    %cD{di} = arith.constant {d} : index\n"
        for di, d in enumerate(trailing_dims)
    )

    # Build product-of-suffixes for flat-index decomposition
    # For trailing dims [d0, d1, d2], products: d1*d2, d2, 1
    suffix_products = []
    for i in range(1, len(trailing_dims)):
        prod = 1
        for d in trailing_dims[i:]:
            prod *= d
        suffix_products.append(prod)

    # Build offset index list for row extraction: [%i, 0, 0, ...]
    row_offsets = "%i" + ", %c0" * (rank - 1)
    # Row sizes: [1, d0, d1, ...]
    row_sizes = "1, " + ", ".join(str(d) for d in trailing_dims)
    # Row strides: all 1s
    row_strides = ", ".join(["1"] * rank)

    # Build flat-index decomposition into multi-index
    if len(trailing_dims) == 1:
        # Single trailing dim: just use %k directly
        multi_idx = "%k"
        multi_idx_compute = ""
    else:
        # Multiple trailing dims: decompose flat index into (j0, j1, ...)
        parts = []
        compute = ""
        remaining = "%k"
        for di in range(len(trailing_dims)):
            if di == len(trailing_dims) - 1:
                idx = remaining
            else:
                suffix_var = f"%s{di}"
                div_op = f"{suffix_var} = arith.divui {remaining}, %cS{di} : index"
                rem_op = f"%r{di} = arith.remui {remaining}, %cS{di} : index"
                compute += f"        {div_op}\n        {rem_op}\n"
                idx = suffix_var
                remaining = f"%r{di}"
            parts.append(idx)
        multi_idx = ", ".join(parts)
        multi_idx_compute = compute

    # Define suffix product constants
    suffix_consts = "".join(
        f"    %cS{si} = arith.constant {s} : index\n"
        for si, s in enumerate(suffix_products)
    )

    # The carry update loop: for k in 0..trailing_total:
    #   extract carry[k] and input[i, k0, k1, ...]
    #   apply op, insert into carry[k]
    body = f"""{init_code}
{input_code}
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %cN = arith.constant {N} : index
    %cTotal = arith.constant {trailing_total} : index
{dim_consts}{suffix_consts}
    %empty = tensor.empty() : {result_type}
    %scanned, %_carry = "scf.for"(%c0, %cN, %c1, %empty, {init_name}) ({{
    ^bb0(%i: index, %acc_tensor: {result_type}, %carry: {trailing_type}):
      %new_carry = "scf.for"(%c0, %cTotal, %c1, %carry) ({{
      ^bb1(%k: index, %c: {trailing_type}):
{multi_idx_compute}        %c_elem = tensor.extract %c[{multi_idx}] : {trailing_type}
        %in_elem = tensor.extract {input_name}[%i, {multi_idx}] : {input_type}
        %added = {op_name} %c_elem, %in_elem : {result_element_type}
        %c_next = tensor.insert %added into %c[{multi_idx}] : {trailing_type}
        "scf.yield"(%c_next) : ({trailing_type}) -> ()
      }}) : (index, index, index, {trailing_type}) -> {trailing_type}
      %acc_next = tensor.insert_slice %new_carry into %acc_tensor[{row_offsets}] [{row_sizes}] [{row_strides}] : {trailing_type} into {result_type}
      "scf.yield"(%acc_next, %new_carry) : ({result_type}, {trailing_type}) -> ()
    }}) : (index, index, index, {result_type}, {trailing_type}) -> ({result_type}, {trailing_type})"""

    builder = _MLIRMainModuleBuilder(result_type)
    builder.add_block(body)
    return builder.render("%scanned")


def _tensor_type_mlir(dims: list[int], elem: str) -> str:
    if not dims:
        return elem
    ds = "x".join(str(d) for d in dims)
    return f"tensor<{ds}x{elem}>"


# ---------------------------------------------------------------------------
# Sort / Grade lowering (C runtime qsort)
# ---------------------------------------------------------------------------


def _sort_runtime_func(result_elem: str) -> str:
    if result_elem == "i32":
        return "remora_sort_i32"
    if result_elem == "f32":
        return "remora_sort_f32"
    raise RemoraLoweringError(f"sort not supported for type {result_elem}")


def _lower_sort_module(node: HIRSort, functions: dict[str, HIRFunction]) -> str:
    from remora.lowering.module import _MLIRMainModuleBuilder

    if not isinstance(node.result_type, ArrayType):
        raise RemoraLoweringError("sort lowering requires array result type")
    rank = node.result_type.rank
    if rank < 1 or rank > 2:
        raise RemoraLoweringError("sort lowering supports ranks 1 and 2")

    input_code, input_name, input_type, input_element_type = _lower_tensor_input(
        node.array, "sort_input", functions, tensor_env=None
    )
    result_type = type_to_mlir(node.result_type)
    result_elem = type_to_mlir(node.result_type.element)
    rt_func = _sort_runtime_func(result_elem)

    if rank == 1:
        n = node.result_type.shape[0].value
        sort_body = f"""{input_code}
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %cN{n} = arith.constant {n} : index
    %buf = memref.alloc() : memref<{n}x{result_elem}>
    scf.for %i = %c0 to %cN{n} step %c1 {{
      %val = tensor.extract {input_name}[%i] : {result_type}
      memref.store %val, %buf[%i] : memref<{n}x{result_elem}>
    }}
    func.call @{rt_func}(%buf) : (memref<{n}x{result_elem}>) -> ()
    %sorted = bufferization.to_tensor %buf restrict writable : memref<{n}x{result_elem}>"""
        builder = _MLIRMainModuleBuilder(result_type)
        builder.add_extern(f"  func.func private @{rt_func}(memref<{n}x{result_elem}>)")
        builder.add_block(sort_body)
        return builder.render("%sorted")

    # Rank 2: per-row sort using memref operations
    R = node.result_type.shape[0].value
    C = node.result_type.shape[1].value
    sort_body = f"""{input_code}
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %cR = arith.constant {R} : index
    %cC = arith.constant {C} : index
    %in_mem = memref.alloc() : memref<{R}x{C}x{result_elem}>
    scf.for %i = %c0 to %cR step %c1 {{
      scf.for %j = %c0 to %cC step %c1 {{
        %v = tensor.extract {input_name}[%i, %j] : {result_type}
        memref.store %v, %in_mem[%i, %j] : memref<{R}x{C}x{result_elem}>
      }}
    }}
    scf.for %r = %c0 to %cR step %c1 {{
      %row_buf = memref.alloc() : memref<{C}x{result_elem}>
      scf.for %j = %c0 to %cC step %c1 {{
        %v = memref.load %in_mem[%r, %j] : memref<{R}x{C}x{result_elem}>
        memref.store %v, %row_buf[%j] : memref<{C}x{result_elem}>
      }}
      func.call @remora_sort_1d_{result_elem}(%row_buf) : (memref<{C}x{result_elem}>) -> ()
      scf.for %j = %c0 to %cC step %c1 {{
        %v = memref.load %row_buf[%j] : memref<{C}x{result_elem}>
        memref.store %v, %in_mem[%r, %j] : memref<{R}x{C}x{result_elem}>
      }}
    }}
    %sorted = bufferization.to_tensor %in_mem restrict writable : memref<{R}x{C}x{result_elem}>"""
    builder = _MLIRMainModuleBuilder(result_type)
    builder.add_extern(f"  func.func private @remora_sort_1d_{result_elem}(memref<{C}x{result_elem}>)")
    builder.add_block(sort_body)
    return builder.render("%sorted")


def _lower_grade_module(node: HIRGrade, functions: dict[str, HIRFunction]) -> str:
    from remora.lowering.module import _MLIRMainModuleBuilder

    if not isinstance(node.result_type, ArrayType):
        raise RemoraLoweringError("grade result must be array type")
    rank = node.result_type.rank
    if rank < 1 or rank > 2:
        raise RemoraLoweringError("grade lowering supports ranks 1 and 2")

    input_code, input_name, input_type, input_element_type = _lower_tensor_input(
        node.array, "grade_input", functions, tensor_env=None
    )
    del input_type
    n = node.result_type.shape[0].value
    result_type = type_to_mlir(node.result_type)
    result_elem = type_to_mlir(node.result_type.element)

    if rank == 1:
        rt_func = "remora_grade_i32" if input_element_type == "i32" else "remora_grade_f32"
        grade_body = f"""{input_code}
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %cN{n} = arith.constant {n} : index
    %buf_in = memref.alloc() : memref<{n}x{input_element_type}>
    %buf_out = memref.alloc() : memref<{n}x{result_elem}>
    scf.for %i = %c0 to %cN{n} step %c1 {{
      %val = tensor.extract {input_name}[%i] : tensor<{n}x{input_element_type}>
      memref.store %val, %buf_in[%i] : memref<{n}x{input_element_type}>
    }}
    func.call @{rt_func}(%buf_in, %buf_out) : (memref<{n}x{input_element_type}>, memref<{n}x{result_elem}>) -> ()
    %sorted_indices = bufferization.to_tensor %buf_out restrict writable : memref<{n}x{result_elem}>
    memref.dealloc %buf_in : memref<{n}x{input_element_type}>"""
        builder = _MLIRMainModuleBuilder(result_type)
        builder.add_extern(
            f"  func.func private @{rt_func}(memref<{n}x{input_element_type}>, memref<{n}x{result_elem}>)"
        )
        builder.add_block(grade_body)
        return builder.render("%sorted_indices")

    # Rank 2: per-row grade
    R = node.result_type.shape[0].value
    C = node.result_type.shape[1].value
    rt_1d = f"remora_grade_1d_{input_element_type}"
    grade_body = f"""{input_code}
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %cR = arith.constant {R} : index
    %cC = arith.constant {C} : index
    %in_mem = memref.alloc() : memref<{R}x{C}x{input_element_type}>
    %out_mem = memref.alloc() : memref<{R}x{C}x{result_elem}>
    scf.for %i = %c0 to %cR step %c1 {{
      scf.for %j = %c0 to %cC step %c1 {{
        %v = tensor.extract {input_name}[%i, %j] : tensor<{R}x{C}x{input_element_type}>
        memref.store %v, %in_mem[%i, %j] : memref<{R}x{C}x{input_element_type}>
      }}
    }}
    scf.for %r = %c0 to %cR step %c1 {{
      %row_in = memref.alloc() : memref<{C}x{input_element_type}>
      %row_out = memref.alloc() : memref<{C}x{result_elem}>
      scf.for %j = %c0 to %cC step %c1 {{
        %v = memref.load %in_mem[%r, %j] : memref<{R}x{C}x{input_element_type}>
        memref.store %v, %row_in[%j] : memref<{C}x{input_element_type}>
      }}
      func.call @{rt_1d}(%row_in, %row_out) : (memref<{C}x{input_element_type}>, memref<{C}x{result_elem}>) -> ()
      scf.for %j = %c0 to %cC step %c1 {{
        %v = memref.load %row_out[%j] : memref<{C}x{result_elem}>
        memref.store %v, %out_mem[%r, %j] : memref<{R}x{C}x{result_elem}>
      }}
    }}
    %sorted_indices = bufferization.to_tensor %out_mem restrict writable : memref<{R}x{C}x{result_elem}>"""
    builder = _MLIRMainModuleBuilder(result_type)
    builder.add_extern(
        f"  func.func private @{rt_1d}(memref<{C}x{input_element_type}>, memref<{C}x{result_elem}>)"
    )
    builder.add_block(grade_body)
    return builder.render("%sorted_indices")


# ---------------------------------------------------------------------------
# Filter / Replicate lowering (C runtime with dynamic sizing)
# ---------------------------------------------------------------------------


def _cmp_op_to_mlir(op: str, elem_type: str) -> str:
    """Map Remora comparison operator to MLIR arith.cmpi/cmpf predicate."""
    int_preds = {">": "sgt", "<": "slt", ">=": "sge", "<=": "sle", "==": "eq", "!=": "ne"}
    flt_preds = {">": "ogt", "<": "olt", ">=": "oge", "<=": "ole", "==": "oeq", "!=": "one"}
    preds = int_preds if elem_type == "i32" else flt_preds
    op_base = op[:-1] if op.endswith("b") else op
    return preds.get(op_base, "sgt")


def _lower_filter_module(node: HIRFilter, functions: dict[str, HIRFunction]) -> str:
    from remora.lowering.module import _MLIRMainModuleBuilder

    if not isinstance(node.result_type, SigmaType):
        raise RemoraLoweringError("filter result must be SigmaType")
    body_type = node.result_type.body
    if not isinstance(body_type, ArrayType) or body_type.rank != 1:
        raise RemoraLoweringError("filter only supports rank-1 arrays")

    input_code, input_name, input_type, input_elem = _lower_tensor_input(
        node.array, "flt_in", functions, tensor_env=None
    )
    n = body_type.shape[0].value
    result_elem = type_to_mlir(body_type.element)

    # Generate mask via linalg.generic
    pred = node.predicate
    if not isinstance(pred, HIRPrimCallable):
        raise RemoraLoweringError("filter predicate must be a primitive operator")
    if pred.right_arg is None or not isinstance(pred.right_arg, HIRLit):
        raise RemoraLoweringError("filter predicate must be a left section with literal")
    rhs_val = _literal_value(pred.right_arg, result_elem)
    cmp_op = _cmp_op_to_mlir(pred.op, input_elem)
    cmp_kind = "arith.cmpi" if input_elem == "i32" else "arith.cmpf"

    rt = "remora_filter_i32" if input_elem == "i32" else "remora_filter_f32"

    filter_body = f"""{input_code}
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %cN{n} = arith.constant {n} : index
    %cBig = arith.constant {n+2} : index
    %mask_empty = tensor.empty() : tensor<{n}xi32>
    %mask = linalg.generic {{
      indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>],
      iterator_types = ["parallel"]
    }} ins({input_name} : {input_type}) outs(%mask_empty : tensor<{n}xi32>) {{
    ^bb0(%in: {input_elem}, %out: i32):
      %rhs = arith.constant {rhs_val} : {input_elem}
      %cond = {cmp_kind} {cmp_op}, %in, %rhs : {input_elem}
      %intv = arith.extui %cond : i1 to i32
      linalg.yield %intv : i32
    }} -> tensor<{n}xi32>
    %buf_src = memref.alloc() : memref<{n}x{input_elem}>
    %buf_mask = memref.alloc() : memref<{n}xi32>
    %buf_dst = memref.alloc() : memref<{n+2}x{result_elem}>
    scf.for %i = %c0 to %cN{n} step %c1 {{
      %v = tensor.extract {input_name}[%i] : {input_type}
      memref.store %v, %buf_src[%i] : memref<{n}x{input_elem}>
      %m = tensor.extract %mask[%i] : tensor<{n}xi32>
      memref.store %m, %buf_mask[%i] : memref<{n}xi32>
    }}
    %count = func.call @{rt}(%buf_src, %buf_mask, %buf_dst) : (memref<{n}x{input_elem}>, memref<{n}xi32>, memref<{n+2}x{result_elem}>) -> i64
    %count_idx = arith.index_cast %count : i64 to index
    scf.for %k = %c0 to %count_idx step %c1 {{
      %offset = arith.subi %count_idx, %k : index
      %src_minus_1 = arith.subi %offset, %c1 : index
      %val = memref.load %buf_dst[%src_minus_1] : memref<{n+2}x{result_elem}>
      memref.store %val, %buf_dst[%offset] : memref<{n+2}x{result_elem}>
    }}
    %count_i32 = arith.trunci %count : i64 to i32
    memref.store %count_i32, %buf_dst[%c0] : memref<{n+2}x{result_elem}>
    %count_p1 = arith.addi %count_idx, %c1 : index
    %view = memref.subview %buf_dst[0] [%count_p1] [1] : memref<{n+2}x{result_elem}> to memref<?x{result_elem}, strided<[1]>>
    memref.dealloc %buf_src : memref<{n}x{input_elem}>
    memref.dealloc %buf_mask : memref<{n}xi32>
    %result = bufferization.to_tensor %view restrict writable : memref<?x{result_elem}, strided<[1]>>"""
    result_type_str = f"tensor<?x{result_elem}>"
    builder = _MLIRMainModuleBuilder(result_type_str)
    builder.add_extern(
        f"  func.func private @{rt}(memref<{n}x{input_elem}>, memref<{n}xi32>, memref<{n+2}x{result_elem}>) -> i64"
    )
    builder.add_block(filter_body)
    return builder.render("%result")


def _lower_replicate_module(node: HIRReplicate, functions: dict[str, HIRFunction]) -> str:
    from remora.lowering.module import _MLIRMainModuleBuilder

    if not isinstance(node.result_type, SigmaType):
        raise RemoraLoweringError("replicate result must be SigmaType")
    body_type = node.result_type.body
    if not isinstance(body_type, ArrayType) or body_type.rank != 1:
        raise RemoraLoweringError("replicate only supports rank-1 arrays")

    arr_code, arr_name, arr_type, arr_elem = _lower_tensor_input(
        node.array, "rep_arr", functions, tensor_env=None
    )
    cnt_code, cnt_name, cnt_type, _cnt_elem = _lower_tensor_input(
        node.counts, "rep_cnt", functions, tensor_env=None
    )
    n = body_type.shape[0].value
    big_n = n * 100
    result_elem = type_to_mlir(body_type.element)

    rt = "remora_replicate_i32" if arr_elem == "i32" else "remora_replicate_f32"

    replicate_body = f"""{arr_code}
{cnt_code}
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %cN{n} = arith.constant {n} : index
    %buf_src = memref.alloc() : memref<{n}x{arr_elem}>
    %buf_cnt = memref.alloc() : memref<{n}xi32>
    scf.for %i = %c0 to %cN{n} step %c1 {{
      %v = tensor.extract {arr_name}[%i] : {arr_type}
      memref.store %v, %buf_src[%i] : memref<{n}x{arr_elem}>
      %c = tensor.extract {cnt_name}[%i] : {cnt_type}
      memref.store %c, %buf_cnt[%i] : memref<{n}xi32>
    }}
    %count = func.call @{rt}_count(%buf_src, %buf_cnt) : (memref<{n}x{arr_elem}>, memref<{n}xi32>) -> i64
    %count_idx = arith.index_cast %count : i64 to index
    %count_p1 = arith.addi %count_idx, %c1 : index
    %c{big_n} = arith.constant {big_n} : index
    %buf_dst = memref.alloc() : memref<{big_n}x{result_elem}>
    func.call @{rt}_fill(%buf_src, %buf_cnt, %buf_dst) : (memref<{n}x{arr_elem}>, memref<{n}xi32>, memref<{big_n}x{result_elem}>) -> ()
    scf.for %k = %c0 to %count_idx step %c1 {{
      %offset = arith.subi %count_idx, %k : index
      %src_minus_1 = arith.subi %offset, %c1 : index
      %val = memref.load %buf_dst[%src_minus_1] : memref<{big_n}x{result_elem}>
      memref.store %val, %buf_dst[%offset] : memref<{big_n}x{result_elem}>
    }}
    %count_i32 = arith.trunci %count : i64 to i32
    memref.store %count_i32, %buf_dst[%c0] : memref<{big_n}x{result_elem}>
    %view = memref.subview %buf_dst[0] [%count_p1] [1] : memref<{big_n}x{result_elem}> to memref<?x{result_elem}, strided<[1]>>
    %result = bufferization.to_tensor %view restrict writable : memref<?x{result_elem}, strided<[1]>>
    memref.dealloc %buf_src : memref<{n}x{arr_elem}>
    memref.dealloc %buf_cnt : memref<{n}xi32>"""
    result_type_str = f"tensor<?x{result_elem}>"
    builder = _MLIRMainModuleBuilder(result_type_str)
    builder.add_extern(
        f"  func.func private @{rt}_count(memref<{n}x{arr_elem}>, memref<{n}xi32>) -> i64"
    )
    builder.add_extern(
        f"  func.func private @{rt}_fill(memref<{n}x{arr_elem}>, memref<{n}xi32>, memref<{big_n}x{result_elem}>)"
    )
    builder.add_block(replicate_body)
    return builder.render("%result")


def _lower_rank2_c_unary(node, functions, c_base_name):
    """Per-row rank-2 lowering for unary array→array ops via C _1d wrappers."""
    from remora.lowering.module import _MLIRMainModuleBuilder

    R = node.result_type.shape[0].value
    C = node.result_type.shape[1].value
    result_type = type_to_mlir(node.result_type)
    result_elem = type_to_mlir(node.result_type.element)

    input_code, input_name, input_type, input_elem = _lower_tensor_input(
        node.array, "input", functions, tensor_env=None
    )

    rt = f"{c_base_name}_{input_elem}_1d"

    body = f"""{input_code}
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %cR = arith.constant {R} : index
    %cC = arith.constant {C} : index
    %in_mem = memref.alloc() : memref<{R}x{C}x{input_elem}>
    %out_mem = memref.alloc() : memref<{R}x{C}x{result_elem}>
    scf.for %i = %c0 to %cR step %c1 {{
      scf.for %j = %c0 to %cC step %c1 {{
        %v = tensor.extract {input_name}[%i, %j] : {input_type}
        memref.store %v, %in_mem[%i, %j] : memref<{R}x{C}x{input_elem}>
      }}
    }}
    scf.for %r = %c0 to %cR step %c1 {{
      %row_in = memref.alloc() : memref<{C}x{input_elem}>
      %row_out = memref.alloc() : memref<{C}x{result_elem}>
      scf.for %j = %c0 to %cC step %c1 {{
        %v = memref.load %in_mem[%r, %j] : memref<{R}x{C}x{input_elem}>
        memref.store %v, %row_in[%j] : memref<{C}x{input_elem}>
      }}
      func.call @{rt}(%row_in, %row_out) : (memref<{C}x{input_elem}>, memref<{C}x{result_elem}>) -> ()
      scf.for %j = %c0 to %cC step %c1 {{
        %v = memref.load %row_out[%j] : memref<{C}x{result_elem}>
        memref.store %v, %out_mem[%r, %j] : memref<{R}x{C}x{result_elem}>
      }}
    }}
    %result = bufferization.to_tensor %out_mem restrict writable : memref<{R}x{C}x{result_elem}>"""
    builder = _MLIRMainModuleBuilder(result_type)
    builder.add_extern(f"  func.func private @{rt}(memref<{C}x{input_elem}>, memref<{C}x{result_elem}>)")
    builder.add_block(body)
    return builder.render("%result")
