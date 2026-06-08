"""Scalar region emission for MLIR lowering."""

from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
from typing import Any

from remora.hir import (
    HIRCall,
    HIRCast,
    HIRExpr,
    HIRIf,
    HIRLambda,
    HIRLet,
    HIRLit,
    HIRPrimCallable,
    HIRPrimOp,
    HIRVar,
)
from remora.operators import arith_op, comparison_mlir_op, comparison_predicate

from remora.lowering.types import (
    RemoraLoweringError,
    _is_scalar_type,
    type_to_mlir,
)


@dataclass
class _Operand:
    value: str
    lines: list[str]
    type: str = ""


def _literal_value(expr: HIRLit, result_type: str) -> str:
    if result_type == "f32":
        return f"{float(expr.value):.6e}"
    if result_type == "i32":
        return str(int(expr.value))
    if result_type == "i1":
        return "1" if expr.value else "0"
    raise RemoraLoweringError(f"cannot lower literal of type {result_type}")


def _cast_if_needed(
    value_name: str, from_type: str, to_type: str, result_name: str
) -> list[str]:
    if from_type == to_type:
        return []
    if from_type == "i32" and to_type == "f32":
        return [f"      {result_name} = arith.sitofp {value_name} : i32 to f32"]
    if from_type == "i32" and to_type == "i64":
        return [f"      {result_name} = arith.extsi {value_name} : i32 to i64"]
    if from_type == "i64" and to_type == "i32":
        return [f"      {result_name} = arith.trunci {value_name} : i64 to i32"]
    raise RemoraLoweringError(
        f"cannot cast {from_type} to {to_type} in map lowering"
    )


def _arith_op(op: str, result_type: str) -> str:
    try:
        return arith_op(op, result_type)
    except KeyError as exc:
        raise RemoraLoweringError(str(exc)) from exc


def _hir_prim_op(op: str, result_type: str) -> str:
    if op in {"+f", "-f", "*f", "/f"}:
        return _arith_op(op[0], result_type)
    if op in {"+i", "-i", "*i"}:
        return _arith_op(op[0], result_type)
    raise RemoraLoweringError(f"primitive HIR op {op} is deferred")


def _comparison_op(op: str, operand_type: str) -> tuple[str, str]:
    base_op = op[:-1] if op.endswith("b") else op
    try:
        return comparison_mlir_op(
            base_op, operand_type
        ), comparison_predicate(base_op, operand_type)
    except KeyError as exc:
        raise RemoraLoweringError(str(exc)) from exc


def _load_iree_ir() -> Any:
    try:
        return import_module("iree.compiler.ir")
    except ModuleNotFoundError as exc:
        raise RemoraLoweringError(
            "IREE compiler MLIR bindings are required for MLIR lowering"
        ) from exc


def _lower_callable_operand(
    expr: object,
    name: str,
    result_type: str,
) -> _Operand:
    if expr is None:
        return _Operand("", [], result_type)
    if not isinstance(expr, HIRLit):
        raise RemoraLoweringError(
            "only literal operator section operands lower to MLIR so far"
        )

    if result_type == "f32":
        value = f"{float(expr.value):.6e}"
    elif result_type == "i32":
        value = str(int(expr.value))
    elif result_type == "i1":
        value = "true" if expr.value else "false"
        return _Operand(name, [f"      {name} = arith.constant {value}"], result_type)
    else:
        raise RemoraLoweringError(
            f"cannot lower primitive operand of type {result_type}"
        )
    return _Operand(
        name,
        [f"      {name} = arith.constant {value} : {result_type}"],
        result_type,
    )


def _lower_scalar_value_for_fold_init(
    expr: HIRExpr,
    result_type: str,
    *,
    functions: dict[str, Any],
    env: dict[str, _Operand],
    result_prefix: str,
) -> tuple[str, str]:
    emitter = _RegionEmitter(input_name="", input_type="", functions=functions)
    value = emitter.emit_expr(expr, env)
    cast_name = f"%{result_prefix}_cast"
    cast_lines = _cast_if_needed(value.value, value.type, result_type, cast_name)
    result_value = cast_name if cast_lines else value.value
    return "\n".join([*emitter.lines, *cast_lines]), result_value


def _lower_scalar_module(
    node: HIRLit | HIRCast | HIRPrimOp | HIRLet | HIRCall | HIRIf,
    functions: dict[str, Any] | None = None,
) -> str:
    from remora.lowering.module import _MLIRMainModuleBuilder

    functions = functions or {}
    if isinstance(node, HIRPrimOp):
        result_type = type_to_mlir(node.result_type)
    elif isinstance(node, HIRCast):
        result_type = type_to_mlir(node.result_type)
    elif isinstance(node, HIRLet):
        result_type = type_to_mlir(node.result_type)
    elif isinstance(node, HIRCall):
        result_type = type_to_mlir(node.result_type)
    elif isinstance(node, HIRIf):
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
    builder = _MLIRMainModuleBuilder(result_type, functions=functions)
    builder.add_block("\n".join(lines))
    return builder.render(result_value)


def _lower_let(
    emitter: _RegionEmitter,
    expr: HIRLet,
    env: dict[str, _Operand],
) -> _Operand:
    if not _is_scalar_type(expr.value_type) or not _is_scalar_type(
        expr.result_type
    ):
        raise RemoraLoweringError(
            "only scalar lets lower through the SSA environment so far"
        )
    value = emitter.emit_expr(expr.value, env)
    return emitter.emit_expr(expr.body, {**env, expr.name: value})


class _RegionEmitter:
    def __init__(
        self,
        input_name: str,
        input_type: str,
        next_temp: int = 0,
        functions: dict[str, Any] | None = None,
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
                raise RemoraLoweringError(
                    f"unbound HIR variable {expr.name}"
                ) from exc
            return _Operand(operand.value, [], operand.type or self._input_type)
        if isinstance(expr, HIRLit):
            result_type = type_to_mlir(expr.type)
            name = self.temp()
            value = _literal_value(expr, result_type)
            self.lines.append(
                f"      {name} = arith.constant {value} : {result_type}"
            )
            return _Operand(name, [], result_type)
        if isinstance(expr, HIRCast):
            value = self.emit_expr(expr.value, env)
            result_type = type_to_mlir(expr.result_type)
            cast_lines = _cast_if_needed(
                value.value, value.type, result_type, self.temp()
            )
            if not cast_lines:
                return value
            self.lines.extend(cast_lines)
            return _Operand(
                cast_lines[-1].split(" = ", 1)[0].strip(), [], result_type
            )
        if isinstance(expr, HIRLet):
            return _lower_let(self, expr, env)
        if isinstance(expr, HIRIf):
            condition = self.emit_expr(expr.condition, env)
            then_branch = self.emit_expr(expr.then_branch, env)
            else_branch = self.emit_expr(expr.else_branch, env)
            condition = self._coerce(condition, "i1")
            result_type = type_to_mlir(expr.result_type)
            then_branch = self._coerce(then_branch, result_type)
            else_branch = self._coerce(else_branch, result_type)
            result = self.temp()
            self.lines.append(
                f"      {result} = arith.select {condition.value}, {then_branch.value}, {else_branch.value} : {result_type}"
            )
            return _Operand(result, [], result_type)
        if isinstance(expr, HIRCall):
            return self._emit_call(expr, env)
        if isinstance(expr, HIRPrimOp):
            args = [self.emit_expr(arg, env) for arg in expr.args]
            result_type = type_to_mlir(expr.result_type)
            return self._emit_prim_op(expr.op, args, result_type)
        raise RemoraLoweringError(
            f"cannot lower HIR expression {type(expr).__name__} in map body"
        )

    def _emit_call(
        self, expr: HIRCall, env: dict[str, _Operand]
    ) -> _Operand:
        function = self._functions.get(expr.func_name)
        if function is None:
            raise RemoraLoweringError(
                f"unknown HIR function {expr.func_name}"
            )
        args = [self.emit_expr(arg, env) for arg in expr.args]
        if len(args) != len(function.params):
            raise RemoraLoweringError(
                f"function {expr.func_name} arity mismatch"
            )
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
            raise RemoraLoweringError(
                "only binary primitive operations lower to MLIR"
            )

        if op in {"+f", "-f", "*f", "/f", "+i", "-i", "*i"}:
            coerced_args = [self._coerce(arg, result_type) for arg in args]
            result = self.temp()
            mlir_op = _hir_prim_op(op, result_type)
            self.lines.append(
                f"      {result} = {mlir_op} {coerced_args[0].value}, {coerced_args[1].value} : {result_type}"
            )
            return _Operand(result, [], result_type)

        if op in {"<b", "<=b", ">b", ">=b", "==b", "!=b"}:
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
        cast_lines = _cast_if_needed(
            operand.value, operand.type, result_type, self.temp()
        )
        if not cast_lines:
            return operand
        self.lines.extend(cast_lines)
        return _Operand(
            cast_lines[-1].split(" = ", 1)[0].strip(), [], result_type
        )

    def temp(self) -> str:
        value = f"%v{self._next_temp}"
        self._next_temp += 1
        return value

    @property
    def next_temp(self) -> int:
        return self._next_temp
