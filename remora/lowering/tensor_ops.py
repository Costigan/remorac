"""Tensor operation lowering for MLIR: maps, folds, iota, array literals."""

from __future__ import annotations

from typing import Any

from remora.hir import (
    HIRApply,
    HIRArrayLit,
    HIRCall,
    HIRCast,
    HIRDrop,
    HIRExpr,
    HIRFold,
    HIRFunction,
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
    HIRReshape,
    HIRReverse,
    HIRScan,
    HIRSlice,
    HIRTake,
    HIRTranspose,
    HIRVar,
)
from remora.types import ArrayType, ScalarType, StaticDim

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

    if isinstance(node, (HIRMap, HIRApply)):
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
        ),
    ):
        from remora.lowering.view_ops import _lower_view_input

        return _lower_view_input(node, functions, prefix, tensor_env=tensor_env)

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
        left_name="%left_in",
        left_type=left_element_type,
        right_name="%right_in",
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
    ^bb0(%left_in: {left_element_type}, %right_in: {right_element_type}, %out: {result_element_type}):
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
) -> tuple[str, str, str]:
    if isinstance(node.result_type, ArrayType):
        return _lower_array_fold_result(node, functions, tensor_env)
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
    init_code, init_value = _lower_scalar_value_for_fold_init(
        node.init,
        result_type,
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
        acc_type=result_type,
        result_type=result_type,
    )

    body = f"""{input_code}
{init_code}
    %init = tensor.from_elements {init_value} : tensor<{result_type}>
    %folded = linalg.generic {{
      indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> ()>],
      iterator_types = [\"reduction\"]
    }} ins({input_name} : {input_type}) outs(%init : tensor<{result_type}>) {{
    ^bb0(%in: {input_element_type}, %acc: {result_type}):
{fold_body}
    }} -> tensor<{result_type}>
    %result = tensor.extract %folded[] : tensor<{result_type}>
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
        tensor_env=tensor_env,
    )
    init_code, init_name, init_type, init_element_type = _lower_tensor_input(
        node.init,
        "init",
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

    body = f"""{input_code}
{init_code}
    %folded = linalg.generic {{
      indexing_maps = [{_identity_affine_map(rank)}, {_drop_first_affine_map(rank)}],
      iterator_types = {_fold_iterators(rank)}
    }} ins({input_name} : {input_type}) outs({init_name} : {result_type}) {{
    ^bb0(%in: {input_element_type}, %acc: {result_element_type}):
{fold_body}
    }} -> {result_type}
"""
    return body.rstrip(), "%folded", result_type


def _lower_fold_input(
    node: HIRExpr,
    functions: dict[str, HIRFunction],
    prefix: str = "",
    *,
    tensor_env: TensorEnv | None = None,
) -> tuple[str, str, str, str]:
    if isinstance(node, (HIRIota, HIRArrayLit)):
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
        ),
    ):
        return _lower_tensor_input(
            node, _join_prefix(prefix, "input"), functions, tensor_env
        )

    if isinstance(node, (HIRMap, HIRApply)):
        if node.cell_shape:
            raise RemoraLoweringError(
                "only scalar-cell map inputs lower to fold MLIR so far"
            )
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


def _lower_scan_module(
    node: HIRScan, functions: dict[str, HIRFunction]
) -> str:
    from remora.lowering.module import _MLIRMainModuleBuilder

    if not isinstance(node.result_type, ArrayType):
        raise RemoraLoweringError("scan lowering requires an array result")
    if node.result_type.rank != 1:
        raise RemoraLoweringError("only rank-1 scan lowers to MLIR so far")
    if not isinstance(node.func, HIRPrimCallable):
        raise RemoraLoweringError("only primitive scan callables lower to MLIR so far")

    input_code, input_name, input_type, input_element_type = _lower_tensor_input(
        node.array, "input", functions
    )
    result_type = type_to_mlir(node.result_type)
    result_element_type = type_to_mlir(node.result_type.element)
    init_value_str = _literal_value(node.init, result_element_type)
    op_name = _arith_op(node.func.op, result_element_type)
    N = node.reduction_dim.value

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
