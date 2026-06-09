"""Builder-API-based scalar region emitter for MLIR lowering.

This module provides ``_BuilderRegionEmitter`` which constructs MLIR
regions using the ``iree.compiler.dialects`` builder API instead of
f-string text concatenation. It is a drop-in replacement for
``_RegionEmitter`` in ``remora/lowering/scalar.py``.
"""

from __future__ import annotations

from remora.hir import (
    HIRCall,
    HIRCast,
    HIRExpr,
    HIRIf,
    HIRLit,
    HIRLet,
    HIRPrimOp,
    HIRVar,
)
from remora.lowering.types import RemoraLoweringError, _is_scalar_type, type_to_mlir
from remora.lowering.scalar import (
    _Operand,
    _cast_if_needed,
    _comparison_op,
    _hir_prim_op,
    _literal_value,
    _lower_callable_operand,
)


class _BuilderRegionEmitter:
    """Emits MLIR operations using the builder API instead of text."""

    def __init__(
        self,
        input_name: str,
        input_type: str,
        next_temp: int = 0,
        functions: dict | None = None,
    ) -> None:
        import iree.compiler.ir as ir

        self.lines: list[str] = []
        self._next_temp = next_temp
        self._input_name = input_name
        self._input_type = input_type
        self._functions = functions or {}
        self._ir = ir
        self._ctx = ir.Context()
        self._ctx.allow_unregistered_dialects = True
        self._loc = ir.Location.unknown(self._ctx)

    def emit_expr(self, expr: HIRExpr, env: dict[str, _Operand]) -> _Operand:
        """Lower *expr* to text MLIR and return an _Operand."""
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
            return self._emit_let(expr, env)
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

    def _emit_let(self, expr: HIRLet, env: dict[str, _Operand]) -> _Operand:
        if not _is_scalar_type(expr.value_type) or not _is_scalar_type(
            expr.result_type
        ):
            raise RemoraLoweringError(
                "only scalar lets lower through the SSA environment so far"
            )
        value = self.emit_expr(expr.value, env)
        return self.emit_expr(expr.body, {**env, expr.name: value})

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
        self, op: str, args: list[_Operand], result_type: str
    ) -> _Operand:
        if op in {"expf", "logf"}:
            if len(args) != 1:
                raise RemoraLoweringError(f"{op[:-1]} expects one operand")
            operand = self._coerce(args[0], "f32")
            result = self.temp()
            mlir_op = "math.exp" if op == "expf" else "math.log"
            self.lines.append(
                f"      {result} = {mlir_op} {operand.value} : f32"
            )
            return _Operand(result, [], "f32")
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
