"""Initial HIR to MLIR lowering support.

This phase starts with validated textual MLIR because the installed
``iree-compiler`` package exposes core IR parsing, while importing some dialect
builder modules currently requires extra packages not in the project metadata.
"""

from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
from typing import Any

from remora.errors import RemoraError
from remora.hir import (
    HIRArrayLit,
    HIRCast,
    HIRCall,
    HIRExpr,
    HIRFold,
    HIRFunction,
    HIRIndex,
    HIRIota,
    HIRLit,
    HIRLet,
    HIRMap,
    HIRPrimCallable,
    HIRPrimOp,
    HIRProgram,
    HIRVar,
)
from remora.types import BOOL, FLOAT, INT, ArrayType, FuncType, RemoraType, ScalarType


class RemoraLoweringError(RemoraError):
    """Raised when the current MLIR lowering slice cannot handle a program."""


@dataclass(frozen=True)
class LoweredModule:
    text: str
    module: Any


class MLIRLowering:
    """Minimal MLIR lowering context for the Phase 5 spike."""

    def __init__(self) -> None:
        self.ir = _load_iree_ir()
        self.context = self.ir.Context()
        self.context.allow_unregistered_dialects = True

    def lower_type(self, value_type: RemoraType) -> Any:
        """Lower a Remora type to an MLIR type object."""
        with self.context, self.ir.Location.unknown(self.context):
            return self.ir.Type.parse(type_to_mlir(value_type), self.context)

    def lower_program(self, program: HIRProgram) -> LoweredModule:
        functions = {function.name: function for function in program.functions}
        main = _prepare_main_expr(program.main)
        if not isinstance(
            main,
            (
                HIRLit,
                HIRCast,
                HIRPrimOp,
                HIRLet,
                HIRCall,
                HIRIndex,
                HIRIota,
                HIRArrayLit,
                HIRMap,
                HIRFold,
            ),
        ):
            raise RemoraLoweringError(
                "only scalar expressions, scalar lets/calls, full-rank indexing, iota, array literals, scalar maps, and scalar folds lower to MLIR so far"
            )

        text = _lower_main_module(main, functions)
        with self.context, self.ir.Location.unknown(self.context):
            module = self.ir.Module.parse(text)
        return LoweredModule(str(module), module)


def type_to_mlir(value_type: RemoraType) -> str:
    if value_type == FLOAT:
        return "f32"
    if value_type == INT:
        return "i32"
    if value_type == BOOL:
        return "i1"
    if isinstance(value_type, ArrayType):
        dims = "x".join(str(dim.value) for dim in value_type.shape)
        element = type_to_mlir(value_type.element)
        if dims:
            return f"tensor<{dims}x{element}>"
        return f"tensor<{element}>"
    if isinstance(value_type, FuncType):
        params = ", ".join(type_to_mlir(param) for param in value_type.params)
        return f"({params}) -> {type_to_mlir(value_type.result)}"
    if isinstance(value_type, ScalarType):
        raise RemoraLoweringError(f"unknown scalar type {value_type.name}")
    raise RemoraLoweringError(f"cannot lower type {value_type}")


def _prepare_main_expr(expr: HIRExpr | None) -> HIRExpr | None:
    if _can_lower_as_scalar_expr(expr):
        return expr
    return _inline_lets(expr) if expr is not None else None


def _can_lower_as_scalar_expr(expr: HIRExpr | None) -> bool:
    if expr is None:
        return False
    if isinstance(expr, (HIRLit, HIRCast, HIRPrimOp, HIRCall)):
        return _is_scalar_type(expr.result_type if not isinstance(expr, HIRLit) else expr.type)
    if isinstance(expr, HIRLet):
        return (
            _is_scalar_type(expr.value_type)
            and _is_scalar_type(expr.result_type)
            and _can_lower_as_scalar_expr(expr.value)
            and _can_lower_as_scalar_expr(expr.body)
        )
    return False


def _inline_lets(expr: HIRExpr | None, env: dict[str, HIRExpr] | None = None) -> HIRExpr | None:
    if expr is None:
        return None
    env = dict(env or {})
    if isinstance(expr, HIRLet):
        value = _inline_lets(expr.value, env)
        if value is None:
            raise RemoraLoweringError("let value cannot be empty")
        return _inline_lets(expr.body, {**env, expr.name: value})
    if isinstance(expr, HIRVar):
        return env.get(expr.name, expr)
    if isinstance(expr, HIRMap):
        arrays = [_inline_lets(array, env) for array in expr.arrays]
        if any(array is None for array in arrays):
            raise RemoraLoweringError("map array cannot be empty")
        return HIRMap(
            expr.frame_shape,
            expr.cell_shape,
            _inline_callable(expr.func, env),
            arrays,
            expr.result_type,
        )  # type: ignore[arg-type]
    if isinstance(expr, HIRFold):
        init = _inline_lets(expr.init, env)
        array = _inline_lets(expr.array, env)
        if init is None or array is None:
            raise RemoraLoweringError("fold operands cannot be empty")
        return HIRFold(
            expr.reduction_dim,
            _inline_callable(expr.func, env),
            init,
            array,
            expr.result_type,
        )
    if isinstance(expr, HIRPrimOp):
        args = [_inline_lets(arg, env) for arg in expr.args]
        if any(arg is None for arg in args):
            raise RemoraLoweringError("primitive operands cannot be empty")
        return HIRPrimOp(expr.op, args, expr.result_type)  # type: ignore[arg-type]
    if isinstance(expr, HIRIndex):
        array = _inline_lets(expr.array, env)
        indices = [_inline_lets(index, env) for index in expr.indices]
        if array is None or any(index is None for index in indices):
            raise RemoraLoweringError("index operands cannot be empty")
        return HIRIndex(array, indices, expr.result_type)  # type: ignore[arg-type]
    if isinstance(expr, HIRCast):
        value = _inline_lets(expr.value, env)
        if value is None:
            raise RemoraLoweringError("cast value cannot be empty")
        return HIRCast(value, expr.from_type, expr.to_type, expr.result_type)
    if isinstance(expr, HIRArrayLit):
        elements = [_inline_lets(element, env) for element in expr.elements]
        if any(element is None for element in elements):
            raise RemoraLoweringError("array elements cannot be empty")
        return HIRArrayLit(elements, expr.result_type)  # type: ignore[arg-type]
    if isinstance(expr, HIRCall):
        args = [_inline_lets(arg, env) for arg in expr.args]
        if any(arg is None for arg in args):
            raise RemoraLoweringError("call arguments cannot be empty")
        return HIRCall(expr.func_name, args, expr.result_type)  # type: ignore[arg-type]
    return expr


def _inline_callable(callable_: object, env: dict[str, HIRExpr]) -> object:
    if isinstance(callable_, HIRPrimCallable):
        left = (
            _inline_lets(callable_.left_arg, env)
            if callable_.left_arg is not None
            else None
        )
        right = (
            _inline_lets(callable_.right_arg, env)
            if callable_.right_arg is not None
            else None
        )
        return HIRPrimCallable(
            callable_.op,
            callable_.params,
            callable_.result_type,
            left_arg=left,
            right_arg=right,
        )
    if isinstance(callable_, HIRVar):
        inlined = env.get(callable_.name)
        return inlined if inlined is not None else callable_
    return callable_


def _lower_main_module(
    node: HIRLit | HIRCast | HIRPrimOp | HIRLet | HIRCall | HIRIndex | HIRIota | HIRArrayLit | HIRMap | HIRFold,
    functions: dict[str, HIRFunction],
) -> str:
    if isinstance(node, (HIRLit, HIRCast, HIRPrimOp, HIRLet, HIRCall)):
        return _lower_scalar_module(node, functions)
    if isinstance(node, HIRIndex):
        return _lower_index_module(node, functions)
    if isinstance(node, HIRIota):
        return _lower_iota_module(node)
    if isinstance(node, HIRArrayLit):
        return _lower_array_literal_module(node)
    if isinstance(node, HIRFold):
        return _lower_fold_module(node, functions)
    if len(node.arrays) != 1:
        raise RemoraLoweringError("binary map MLIR lowering is deferred")
    if not isinstance(node.result_type, ArrayType):
        return _lower_scalar_map_module(node, functions)
    return _lower_iota_scalar_map_module(node, functions)


def _lower_scalar_module(
    node: HIRLit | HIRCast | HIRPrimOp | HIRLet | HIRCall,
    functions: dict[str, HIRFunction] | None = None,
) -> str:
    functions = functions or {}
    if isinstance(node, HIRPrimOp):
        result_type = type_to_mlir(node.result_type)
    elif isinstance(node, HIRCast):
        result_type = type_to_mlir(node.result_type)
    elif isinstance(node, HIRLet):
        result_type = type_to_mlir(node.result_type)
    elif isinstance(node, HIRCall):
        result_type = type_to_mlir(node.result_type)
    else:
        result_type = type_to_mlir(node.type)
    emitter = _RegionEmitter(input_name="", input_type="", functions=functions)
    value = emitter.emit_expr(node, {})
    lines = [
        *emitter.lines,
        *_cast_if_needed(value.value, value.type, result_type, "%result_cast"),
    ]
    result_value = "%result_cast" if value.type != result_type else value.value
    body = "\n".join(lines)
    function_text = _lower_functions(functions)
    function_prefix = f"\n{function_text}\n" if function_text else ""

    return f"""module {{
{function_prefix}\
  func.func @main() -> {result_type} {{
{body}
    return {result_value} : {result_type}
  }}
}}"""


def _lower_iota_module(node: HIRIota) -> str:
    result_type = type_to_mlir(node.result_type)
    element_type = type_to_mlir(node.result_type.element)
    if element_type != "i32":
        raise RemoraLoweringError("iota lowering currently supports i32 results only")

    return f"""module {{
  func.func @main() -> {result_type} {{
    %empty = tensor.empty() : {result_type}
    %result = linalg.generic {{
      indexing_maps = [affine_map<(d0) -> (d0)>],
      iterator_types = [\"parallel\"]
    }} outs(%empty : {result_type}) {{
    ^bb0(%out: {element_type}):
      %idx = linalg.index 0 : index
      %cast = arith.index_cast %idx : index to {element_type}
      linalg.yield %cast : {element_type}
    }} -> {result_type}
    return %result : {result_type}
  }}
}}"""


def _lower_array_literal_module(node: HIRArrayLit) -> str:
    code, name, result_type, _element_type = _lower_tensor_input(node, "literal", {})
    return f"""module {{
  func.func @main() -> {result_type} {{
{code}
    return {name} : {result_type}
  }}
}}"""


def _lower_index_module(node: HIRIndex, functions: dict[str, HIRFunction]) -> str:
    if not isinstance(node.result_type, ScalarType):
        raise RemoraLoweringError("only full-rank scalar indexing lowers to MLIR so far")
    array_type = _expr_result_type(node.array)
    if not isinstance(array_type, ArrayType):
        raise RemoraLoweringError("indexing expects a tensor input")
    if len(node.indices) != array_type.rank:
        raise RemoraLoweringError("partial indexing lowering is deferred")

    array_code, array_value, array_mlir_type, _element_type = _lower_tensor_input(
        node.array,
        "idx_in",
        functions,
    )
    result_type = type_to_mlir(node.result_type)
    index_lines: list[str] = []
    index_names: list[str] = []
    for position, index in enumerate(node.indices):
        if not isinstance(index, HIRLit) or index.type != INT:
            raise RemoraLoweringError("only literal integer indices lower to MLIR so far")
        name = f"%idx_{position}"
        index_names.append(name)
        index_lines.append(f"    {name} = arith.constant {index.value} : index")
    indices = ", ".join(index_names)
    index_body = "\n".join(index_lines)
    return f"""module {{
  func.func @main() -> {result_type} {{
{array_code}
{index_body}
    %result = tensor.extract {array_value}[{indices}] : {array_mlir_type}
    return %result : {result_type}
  }}
}}"""


def _lower_scalar_map_module(node: HIRMap, functions: dict[str, HIRFunction]) -> str:
    if node.frame_shape or node.cell_shape:
        raise RemoraLoweringError("only rank-0 scalar maps lower as scalar MLIR so far")

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
    body = "\n".join([*emitter.lines, *callable_lines])

    return f"""module {{
  func.func @main() -> {result_type} {{
{body}
    return {result_value} : {result_type}
  }}
}}"""


def _lower_functions(functions: dict[str, HIRFunction]) -> str:
    lowered = [_lower_function(function) for function in functions.values()]
    return "\n\n".join(lowered)


def _lower_function(function: HIRFunction) -> str:
    if not _is_scalar_type(function.return_type):
        raise RemoraLoweringError(
            "only scalar-returning HIR functions lower to MLIR functions so far"
        )
    for param in function.params:
        if not _is_scalar_type(param.type):
            raise RemoraLoweringError(
                "only scalar HIR function parameters lower to MLIR functions so far"
            )

    result_type = type_to_mlir(function.return_type)
    args = [
        f"%arg{index}: {type_to_mlir(param.type)}"
        for index, param in enumerate(function.params)
    ]
    env = {
        param.name: _Operand(
            f"%arg{index}",
            [],
            type_to_mlir(param.type),
        )
        for index, param in enumerate(function.params)
    }
    emitter = _RegionEmitter(
        input_name="",
        input_type="",
        functions={function.name: function},
    )
    value = emitter.emit_expr(function.body, env)
    lines = [
        *emitter.lines,
        *_cast_if_needed(value.value, value.type, result_type, "%result_cast"),
    ]
    result_value = "%result_cast" if value.type != result_type else value.value
    body = "\n".join(lines)

    return f"""  func.func private @{function.name}({", ".join(args)}) -> {result_type} {{
{body}
    return {result_value} : {result_type}
  }}"""


def _lower_iota_scalar_map_module(node: HIRMap, functions: dict[str, HIRFunction]) -> str:
    if node.cell_shape:
        return _lower_map_cell_module(node, functions)
    if not isinstance(node.result_type, ArrayType):
        raise RemoraLoweringError("map lowering requires an array result")

    input_code, input_name, input_type, input_element_type = _lower_tensor_input(
        node.array,
        "input",
        functions,
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

    return f"""module {{
  func.func @main() -> {result_type} {{
{input_code}
    %map_empty = tensor.empty() : {result_type}
    %mapped = linalg.generic {{
      indexing_maps = [{identity}, {identity}],
      iterator_types = {iterators}
    }} ins({input_name} : {input_type}) outs(%map_empty : {result_type}) {{
    ^bb0(%in: {input_element_type}, %out: {result_element_type}):
{op_lines}
    }} -> {result_type}
    return %mapped : {result_type}
  }}
}}"""


def _lower_map_cell_module(node: HIRMap, functions: dict[str, HIRFunction]) -> str:
    if not isinstance(node.result_type, ArrayType):
        raise RemoraLoweringError("cell-map lowering requires an array result")
    if len(node.cell_shape) != 1:
        raise RemoraLoweringError("only rank-1 cell maps lower to MLIR so far")
    if node.result_type.rank != len(node.frame_shape):
        raise RemoraLoweringError("only scalar-result cell maps lower to MLIR so far")
    if not isinstance(node.func, HIRVar):
        raise RemoraLoweringError("only lifted lambda cell maps lower to MLIR so far")

    function = functions.get(node.func.name)
    if function is None:
        raise RemoraLoweringError(f"unknown cell-map function {node.func.name}")
    if len(function.params) != 1:
        raise RemoraLoweringError("only unary cell-map functions lower to MLIR so far")
    if not isinstance(function.body, HIRFold):
        raise RemoraLoweringError("only cell maps whose body is a fold lower to MLIR so far")
    if not isinstance(function.body.init, HIRLit):
        raise RemoraLoweringError("only literal cell-fold initial values lower to MLIR so far")
    if not isinstance(function.body.array, HIRVar):
        raise RemoraLoweringError("only folds over the cell-map parameter lower to MLIR so far")
    if function.body.array.name != function.params[0].name:
        raise RemoraLoweringError("cell-map fold must reduce the cell-map parameter")
    if not isinstance(function.body.func, HIRPrimCallable):
        raise RemoraLoweringError("only primitive cell-fold callables lower to MLIR so far")

    input_remora_type = _expr_result_type(node.array)
    if not isinstance(input_remora_type, ArrayType):
        raise RemoraLoweringError("cell-map input must be an array")
    input_rank = input_remora_type.rank
    frame_rank = len(node.frame_shape)
    if input_rank != frame_rank + len(node.cell_shape):
        raise RemoraLoweringError("cell-map frame and cell ranks do not match input rank")

    input_code, input_name, input_type, input_element_type = _lower_tensor_input(
        node.array,
        "input",
        functions,
    )
    result_type = type_to_mlir(node.result_type)
    result_element_type = type_to_mlir(node.result_type.element)
    if input_element_type != result_element_type:
        raise RemoraLoweringError("cell-map fold element type must match result element type")
    init_value = _literal_value(function.body.init, result_element_type)
    fold_body = _lower_fold_callable_body(
        function.body.func,
        input_name="%in",
        input_type=input_element_type,
        acc_name="%acc",
        acc_type=result_element_type,
        result_type=result_element_type,
    )

    return f"""module {{
  func.func @main() -> {result_type} {{
{input_code}
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
    return %mapped : {result_type}
  }}
}}"""


def _lower_fold_module(node: HIRFold, functions: dict[str, HIRFunction]) -> str:
    if isinstance(node.result_type, ArrayType):
        return _lower_array_fold_module(node, functions)
    return _lower_scalar_fold_module(node, functions)


def _lower_scalar_fold_module(node: HIRFold, functions: dict[str, HIRFunction]) -> str:
    if not isinstance(node.func, HIRPrimCallable):
        raise RemoraLoweringError("only primitive fold callables lower to MLIR so far")
    if not isinstance(node.init, HIRLit):
        raise RemoraLoweringError("only literal fold initial values lower to MLIR so far")

    input_code, input_name, input_type, input_element_type = _lower_fold_input(
        node.array,
        functions,
    )
    result_type = type_to_mlir(node.result_type)
    init_value = _literal_value(node.init, result_type)
    fold_body = _lower_fold_callable_body(
        node.func,
        input_name="%in",
        input_type=input_element_type,
        acc_name="%acc",
        acc_type=result_type,
        result_type=result_type,
    )

    return f"""module {{
  func.func @main() -> {result_type} {{
{input_code}
    %init_scalar = arith.constant {init_value} : {result_type}
    %init = tensor.from_elements %init_scalar : tensor<{result_type}>
    %folded = linalg.generic {{
      indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> ()>],
      iterator_types = [\"reduction\"]
    }} ins({input_name} : {input_type}) outs(%init : tensor<{result_type}>) {{
    ^bb0(%in: {input_element_type}, %acc: {result_type}):
{fold_body}
    }} -> tensor<{result_type}>
    %result = tensor.extract %folded[] : tensor<{result_type}>
    return %result : {result_type}
  }}
}}"""


def _lower_array_fold_module(node: HIRFold, functions: dict[str, HIRFunction]) -> str:
    if not isinstance(node.func, HIRPrimCallable):
        raise RemoraLoweringError("only primitive array-cell fold callables lower to MLIR so far")
    if not isinstance(node.init, HIRArrayLit):
        raise RemoraLoweringError("only array literal fold initial values lower to array-cell MLIR so far")
    if not isinstance(node.result_type, ArrayType):
        raise RemoraLoweringError("array fold lowering requires an array result")

    input_remora_type = _expr_result_type(node.array)
    if not isinstance(input_remora_type, ArrayType) or input_remora_type.rank < 2:
        raise RemoraLoweringError("array-cell fold lowering requires rank-2 or rank-3 input")

    input_code, input_name, input_type, input_element_type = _lower_fold_input(
        node.array,
        functions,
    )
    init_code, init_name, init_type, init_element_type = _lower_tensor_input(
        node.init,
        "init",
        functions,
    )
    result_type = type_to_mlir(node.result_type)
    result_element_type = type_to_mlir(node.result_type.element)
    if init_type != result_type:
        raise RemoraLoweringError("array-cell fold init type must match result type")
    if input_element_type != result_element_type or init_element_type != result_element_type:
        raise RemoraLoweringError("array-cell fold element types must match")

    rank = input_remora_type.rank
    fold_body = _lower_fold_callable_body(
        node.func,
        input_name="%in",
        input_type=input_element_type,
        acc_name="%acc",
        acc_type=result_element_type,
        result_type=result_element_type,
    )

    return f"""module {{
  func.func @main() -> {result_type} {{
{input_code}
{init_code}
    %folded = linalg.generic {{
      indexing_maps = [{_identity_affine_map(rank)}, {_drop_first_affine_map(rank)}],
      iterator_types = {_fold_iterators(rank)}
    }} ins({input_name} : {input_type}) outs({init_name} : {result_type}) {{
    ^bb0(%in: {input_element_type}, %acc: {result_element_type}):
{fold_body}
    }} -> {result_type}
    return %folded : {result_type}
  }}
}}"""


def _lower_fold_input(
    node: HIRExpr,
    functions: dict[str, HIRFunction],
    prefix: str = "",
) -> tuple[str, str, str, str]:
    if isinstance(node, (HIRIota, HIRArrayLit)):
        return _lower_tensor_input(node, _join_prefix(prefix, "input"), functions)

    if isinstance(node, HIRMap):
        if node.cell_shape:
            raise RemoraLoweringError("only scalar-cell map inputs lower to fold MLIR so far")
        if not isinstance(node.result_type, ArrayType):
            raise RemoraLoweringError("map fold input must have array type")

        input_code, input_name, input_type, input_element_type = _lower_fold_input(
            node.array,
            functions,
            _join_prefix(prefix, "input"),
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

    raise RemoraLoweringError("only folds over tensor literals, iota, or direct scalar maps lower to MLIR so far")


def _lower_tensor_input(
    node: HIRExpr,
    prefix: str,
    functions: dict[str, HIRFunction],
) -> tuple[str, str, str, str]:
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

    if isinstance(node, HIRMap):
        return _lower_fold_input(node, functions, prefix)

    raise RemoraLoweringError("only tensor literals and iota values lower as tensor inputs so far")


def _join_prefix(prefix: str, suffix: str) -> str:
    return f"{prefix}_{suffix}" if prefix else suffix


def _is_scalar_type(value_type: RemoraType) -> bool:
    return isinstance(value_type, ScalarType)


def _expr_result_type(expr: HIRExpr) -> RemoraType:
    if isinstance(expr, HIRLit):
        return expr.type
    if isinstance(expr, HIRVar):
        return expr.type
    if isinstance(expr, HIRIota):
        return expr.result_type
    if isinstance(expr, HIRArrayLit):
        return expr.result_type
    if isinstance(expr, HIRMap):
        return expr.result_type
    if isinstance(expr, HIRFold):
        return expr.result_type
    if isinstance(expr, HIRLet):
        return expr.result_type
    if isinstance(expr, HIRCall):
        return expr.result_type
    if isinstance(expr, HIRCast):
        return expr.result_type
    if isinstance(expr, HIRPrimOp):
        return expr.result_type
    if isinstance(expr, HIRIndex):
        return expr.result_type
    raise AssertionError(f"unknown HIR expression {type(expr).__name__}")


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


def _identity_affine_map(rank: int) -> str:
    dims = ", ".join(f"d{axis}" for axis in range(rank))
    results = ", ".join(f"d{axis}" for axis in range(rank))
    return f"affine_map<({dims}) -> ({results})>"


def _parallel_iterators(rank: int) -> str:
    return "[" + ", ".join('"parallel"' for _axis in range(rank)) + "]"


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


def _map_cell_iterators(frame_rank: int, cell_rank: int) -> str:
    if frame_rank < 1 or cell_rank < 1:
        raise RemoraLoweringError("cell maps require frame and cell dimensions")
    iterators = [
        *('"parallel"' for _axis in range(frame_rank)),
        *('"reduction"' for _axis in range(cell_rank)),
    ]
    return "[" + ", ".join(iterators) + "]"


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
            raise RemoraLoweringError(f"unknown map function {callable_.name}")
        if len(function.params) != 1:
            raise RemoraLoweringError("only unary map functions lower to MLIR so far")
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
            *_cast_if_needed(value.value, value.type, result_type, "%result_cast"),
        ]
        result_value = "%result_cast" if value.type != result_type else value.value
        return lines, result_value
    raise RemoraLoweringError("only primitive and lifted function map callables lower to MLIR so far")


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
    return "\n".join([*lines, f"      linalg.yield {result_value} : {result_type}"])


def _lower_primitive_callable_result(
    callable_: HIRPrimCallable,
    input_name: str,
    input_type: str,
    result_type: str,
) -> tuple[list[str], str]:
    left = _lower_callable_operand(
        callable_.left_arg, "%left", result_type
    )
    right = _lower_callable_operand(
        callable_.right_arg, "%right", result_type
    )
    if callable_.left_arg is None:
        left.value = input_name
        left.lines = _cast_if_needed(input_name, input_type, result_type, "%left_cast")
        if left.lines:
            left.value = "%left_cast"
    if callable_.right_arg is None:
        right.value = input_name
        right.lines = _cast_if_needed(input_name, input_type, result_type, "%right_cast")
        if right.lines:
            right.value = "%right_cast"

    op = _arith_op(callable_.op, result_type)
    lines = [
        *left.lines,
        *right.lines,
        f"      %result = {op} {left.value}, {right.value} : {result_type}",
    ]
    return lines, "%result"


def _lower_fold_callable_body(
    callable_: HIRPrimCallable,
    input_name: str,
    input_type: str,
    acc_name: str,
    acc_type: str,
    result_type: str,
) -> str:
    if callable_.left_arg is not None or callable_.right_arg is not None:
        raise RemoraLoweringError("fold operator sections are deferred")
    left_lines = _cast_if_needed(acc_name, acc_type, result_type, "%fold_left")
    right_lines = _cast_if_needed(input_name, input_type, result_type, "%fold_right")
    left_value = "%fold_left" if left_lines else acc_name
    right_value = "%fold_right" if right_lines else input_name
    op = _arith_op(callable_.op, result_type)
    lines = [
        *left_lines,
        *right_lines,
        f"      %fold_result = {op} {left_value}, {right_value} : {result_type}",
        f"      linalg.yield %fold_result : {result_type}",
    ]
    return "\n".join(lines)


@dataclass
class _Operand:
    value: str
    lines: list[str]
    type: str = ""


def _lower_callable_operand(
    expr: object,
    name: str,
    result_type: str,
) -> _Operand:
    if expr is None:
        return _Operand("", [], result_type)
    if not isinstance(expr, HIRLit):
        raise RemoraLoweringError("only literal operator section operands lower to MLIR so far")

    if result_type == "f32":
        value = f"{float(expr.value):.6e}"
    elif result_type == "i32":
        value = str(int(expr.value))
    else:
        raise RemoraLoweringError(f"cannot lower primitive operand of type {result_type}")
    return _Operand(name, [f"      {name} = arith.constant {value} : {result_type}"], result_type)


def _cast_if_needed(value_name: str, from_type: str, to_type: str, result_name: str) -> list[str]:
    if from_type == to_type:
        return []
    if from_type == "i32" and to_type == "f32":
        return [f"      {result_name} = arith.sitofp {value_name} : i32 to f32"]
    raise RemoraLoweringError(f"cannot cast {from_type} to {to_type} in map lowering")


def _arith_op(op: str, result_type: str) -> str:
    if result_type == "f32":
        ops = {"+": "arith.addf", "-": "arith.subf", "*": "arith.mulf", "/": "arith.divf"}
    elif result_type == "i32":
        ops = {"+": "arith.addi", "-": "arith.subi", "*": "arith.muli"}
    else:
        raise RemoraLoweringError(f"primitive result type {result_type} is deferred")
    try:
        return ops[op]
    except KeyError as exc:
        raise RemoraLoweringError(f"primitive operator {op} is deferred") from exc


class _RegionEmitter:
    def __init__(
        self,
        input_name: str,
        input_type: str,
        next_temp: int = 0,
        functions: dict[str, HIRFunction] | None = None,
    ) -> None:
        self.lines: list[str] = []
        self._next_temp = next_temp
        self._input_name = input_name
        self._input_type = input_type
        self._functions = functions or {}

    def emit_expr(self, expr: HIRExpr, env: dict[str, _Operand]) -> _Operand:
        if isinstance(expr, HIRVar):
            try:
                operand = env[expr.name]
            except KeyError as exc:
                raise RemoraLoweringError(f"unbound HIR variable {expr.name}") from exc
            return _Operand(operand.value, [], operand.type or self._input_type)
        if isinstance(expr, HIRLit):
            result_type = type_to_mlir(expr.type)
            name = self.temp()
            value = _literal_value(expr, result_type)
            self.lines.append(f"      {name} = arith.constant {value} : {result_type}")
            return _Operand(name, [], result_type)
        if isinstance(expr, HIRCast):
            value = self.emit_expr(expr.value, env)
            result_type = type_to_mlir(expr.result_type)
            cast_lines = _cast_if_needed(value.value, value.type, result_type, self.temp())
            if not cast_lines:
                return value
            self.lines.extend(cast_lines)
            return _Operand(cast_lines[-1].split(" = ", 1)[0].strip(), [], result_type)
        if isinstance(expr, HIRLet):
            return _lower_let(self, expr, env)
        if isinstance(expr, HIRCall):
            return self._emit_call(expr, env)
        if isinstance(expr, HIRPrimOp):
            args = [self.emit_expr(arg, env) for arg in expr.args]
            result_type = type_to_mlir(expr.result_type)
            return self._emit_prim_op(expr.op, args, result_type)
        raise RemoraLoweringError(f"cannot lower HIR expression {type(expr).__name__} in map body")

    def _emit_call(self, expr: HIRCall, env: dict[str, _Operand]) -> _Operand:
        function = self._functions.get(expr.func_name)
        if function is None:
            raise RemoraLoweringError(f"unknown HIR function {expr.func_name}")
        args = [self.emit_expr(arg, env) for arg in expr.args]
        if len(args) != len(function.params):
            raise RemoraLoweringError(f"function {expr.func_name} arity mismatch")
        result_type = type_to_mlir(expr.result_type)
        arg_values = ", ".join(arg.value for arg in args)
        arg_types = ", ".join(arg.type for arg in args)
        result = self.temp()
        self.lines.append(
            f"      {result} = func.call @{expr.func_name}({arg_values}) : ({arg_types}) -> {result_type}"
        )
        return _Operand(result, [], result_type)

    def _emit_prim_op(
        self,
        op: str,
        args: list[_Operand],
        result_type: str,
    ) -> _Operand:
        if len(args) != 2:
            raise RemoraLoweringError("only binary primitive operations lower to MLIR")

        if op in {"+f", "-f", "*f", "/f", "+i", "-i", "*i"}:
            coerced_args = [self._coerce(arg, result_type) for arg in args]
            result = self.temp()
            mlir_op = _hir_prim_op(op, result_type)
            self.lines.append(
                f"      {result} = {mlir_op} {coerced_args[0].value}, {coerced_args[1].value} : {result_type}"
            )
            return _Operand(result, [], result_type)

        if op in {"<b", "<=b", "==b", "!=b"}:
            left, right = args
            if left.type != right.type:
                raise RemoraLoweringError(
                    f"comparison operands must have the same lowered type, got {left.type} and {right.type}"
                )
            result = self.temp()
            mlir_op, predicate = _comparison_op(op, left.type)
            self.lines.append(
                f"      {result} = {mlir_op} {predicate}, {left.value}, {right.value} : {left.type}"
            )
            return _Operand(result, [], result_type)

        if op in {"&&b", "||b"}:
            left = self._coerce(args[0], "i1")
            right = self._coerce(args[1], "i1")
            result = self.temp()
            mlir_op = "arith.andi" if op == "&&b" else "arith.ori"
            self.lines.append(
                f"      {result} = {mlir_op} {left.value}, {right.value} : i1"
            )
            return _Operand(result, [], result_type)

        raise RemoraLoweringError(f"primitive HIR op {op} is deferred")

    def _coerce(self, operand: _Operand, result_type: str) -> _Operand:
        cast_lines = _cast_if_needed(operand.value, operand.type, result_type, self.temp())
        if not cast_lines:
            return operand
        self.lines.extend(cast_lines)
        return _Operand(cast_lines[-1].split(" = ", 1)[0].strip(), [], result_type)

    def temp(self) -> str:
        value = f"%v{self._next_temp}"
        self._next_temp += 1
        return value

    @property
    def next_temp(self) -> int:
        return self._next_temp


def _lower_let(
    emitter: _RegionEmitter,
    expr: HIRLet,
    env: dict[str, _Operand],
) -> _Operand:
    if not _is_scalar_type(expr.value_type) or not _is_scalar_type(expr.result_type):
        raise RemoraLoweringError("only scalar lets lower through the SSA environment so far")
    value = emitter.emit_expr(expr.value, env)
    return emitter.emit_expr(expr.body, {**env, expr.name: value})


def _literal_value(expr: HIRLit, result_type: str) -> str:
    if result_type == "f32":
        return f"{float(expr.value):.6e}"
    if result_type == "i32":
        return str(int(expr.value))
    if result_type == "i1":
        return "1" if expr.value else "0"
    raise RemoraLoweringError(f"cannot lower literal of type {result_type}")


def _hir_prim_op(op: str, result_type: str) -> str:
    if op in {"+f", "-f", "*f", "/f"}:
        return _arith_op(op[0], result_type)
    if op in {"+i", "-i", "*i"}:
        return _arith_op(op[0], result_type)
    raise RemoraLoweringError(f"primitive HIR op {op} is deferred")


def _comparison_op(op: str, operand_type: str) -> tuple[str, str]:
    if operand_type == "i32":
        predicates = {"<b": "slt", "<=b": "sle", "==b": "eq", "!=b": "ne"}
        try:
            return "arith.cmpi", predicates[op]
        except KeyError as exc:
            raise RemoraLoweringError(f"comparison HIR op {op} is deferred") from exc
    if operand_type == "f32":
        predicates = {"<b": "olt", "<=b": "ole", "==b": "oeq", "!=b": "one"}
        try:
            return "arith.cmpf", predicates[op]
        except KeyError as exc:
            raise RemoraLoweringError(f"comparison HIR op {op} is deferred") from exc
    raise RemoraLoweringError(f"comparison operand type {operand_type} is deferred")


def _load_iree_ir() -> Any:
    try:
        return import_module("iree.compiler.ir")
    except ModuleNotFoundError as exc:
        raise RemoraLoweringError(
            "IREE compiler MLIR bindings are required for MLIR lowering"
        ) from exc
