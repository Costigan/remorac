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
        main = _inline_lets(program.main) if program.main is not None else None
        if not isinstance(
            main,
            (HIRLit, HIRCast, HIRPrimOp, HIRIota, HIRArrayLit, HIRMap, HIRFold),
        ):
            raise RemoraLoweringError(
                "only scalar expressions, iota, array literals, scalar maps, and scalar folds lower to MLIR so far"
            )

        functions = {function.name: function for function in program.functions}
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
        array = _inline_lets(expr.array, env)
        if array is None:
            raise RemoraLoweringError("map array cannot be empty")
        return HIRMap(
            expr.frame_shape,
            expr.cell_shape,
            _inline_callable(expr.func, env),
            array,
            expr.result_type,
        )
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
    node: HIRLit | HIRCast | HIRPrimOp | HIRIota | HIRArrayLit | HIRMap | HIRFold,
    functions: dict[str, HIRFunction],
) -> str:
    if isinstance(node, (HIRLit, HIRCast, HIRPrimOp)):
        return _lower_scalar_module(node)
    if isinstance(node, HIRIota):
        return _lower_iota_module(node)
    if isinstance(node, HIRArrayLit):
        return _lower_array_literal_module(node)
    if isinstance(node, HIRFold):
        return _lower_scalar_fold_module(node, functions)
    return _lower_iota_scalar_map_module(node, functions)


def _lower_scalar_module(node: HIRLit | HIRCast | HIRPrimOp) -> str:
    if isinstance(node, HIRPrimOp):
        result_type = type_to_mlir(node.result_type)
    elif isinstance(node, HIRCast):
        result_type = type_to_mlir(node.result_type)
    else:
        result_type = type_to_mlir(node.type)
    emitter = _RegionEmitter(input_name="", input_type="")
    value = emitter.emit_expr(node, {})
    lines = [
        *emitter.lines,
        *_cast_if_needed(value.value, value.type, result_type, "%result_cast"),
    ]
    result_value = "%result_cast" if value.type != result_type else value.value
    body = "\n".join(lines)

    return f"""module {{
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


def _lower_iota_scalar_map_module(node: HIRMap, functions: dict[str, HIRFunction]) -> str:
    if node.cell_shape:
        raise RemoraLoweringError("only scalar-cell maps lower to MLIR so far")
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


def _lower_fold_input(
    node: HIRExpr,
    functions: dict[str, HIRFunction],
) -> tuple[str, str, str, str]:
    if isinstance(node, (HIRIota, HIRArrayLit)):
        return _lower_tensor_input(node, "input", functions)

    if isinstance(node, HIRMap):
        if node.cell_shape:
            raise RemoraLoweringError("only scalar-cell map inputs lower to fold MLIR so far")
        if not isinstance(node.result_type, ArrayType):
            raise RemoraLoweringError("map fold input must have array type")

        input_code, input_name, input_type, input_element_type = _lower_fold_input(
            node.array,
            functions,
        )
        map_type = type_to_mlir(node.result_type)
        map_element_type = type_to_mlir(node.result_type.element)
        rank = node.result_type.rank
        identity = _identity_affine_map(rank)
        iterators = _parallel_iterators(rank)
        map_body = _lower_map_callable_body(
            node.func,
            functions,
            input_name="%map_in",
            input_type=input_element_type,
            result_type=map_element_type,
        )
        code = f"""{input_code}
    %map_empty = tensor.empty() : {map_type}
    %mapped = linalg.generic {{
      indexing_maps = [{identity}, {identity}],
      iterator_types = {iterators}
    }} ins({input_name} : {input_type}) outs(%map_empty : {map_type}) {{
    ^bb0(%map_in: {input_element_type}, %map_out: {map_element_type}):
{map_body}
    }} -> {map_type}"""
        return code, "%mapped", map_type, map_element_type

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
        return _lower_fold_input(node, functions)

    raise RemoraLoweringError("only tensor literals and iota values lower as tensor inputs so far")


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


def _lower_map_callable_body(
    callable_: object,
    functions: dict[str, HIRFunction],
    input_name: str,
    input_type: str,
    result_type: str,
) -> str:
    if isinstance(callable_, HIRPrimCallable):
        return _lower_primitive_callable_body(
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
        emitter = _RegionEmitter(input_name=input_name, input_type=input_type)
        value = emitter.emit_expr(
            function.body,
            {function.params[0].name: _Operand(input_name, [])},
        )
        lines = [
            *emitter.lines,
            *_cast_if_needed(value.value, value.type, result_type, "%result_cast"),
        ]
        result_value = "%result_cast" if value.type != result_type else value.value
        lines.append(f"      linalg.yield {result_value} : {result_type}")
        return "\n".join(lines)
    raise RemoraLoweringError("only primitive and lifted function map callables lower to MLIR so far")


def _lower_primitive_callable_body(
    callable_: HIRPrimCallable,
    input_name: str,
    input_type: str,
    result_type: str,
) -> str:
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
        f"      linalg.yield %result : {result_type}",
    ]
    return "\n".join(lines)


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
    def __init__(self, input_name: str, input_type: str) -> None:
        self.lines: list[str] = []
        self._next_temp = 0
        self._input_name = input_name
        self._input_type = input_type

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
        if isinstance(expr, HIRPrimOp):
            args = [self.emit_expr(arg, env) for arg in expr.args]
            result_type = type_to_mlir(expr.result_type)
            return self._emit_prim_op(expr.op, args, result_type)
        raise RemoraLoweringError(f"cannot lower HIR expression {type(expr).__name__} in map body")

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
