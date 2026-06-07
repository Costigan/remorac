"""MLIR builder API region emitter (Stream E2).

Provides ``_BuilderRegionEmitter``, a parallel implementation of
``_RegionEmitter`` that creates ``ir.Value`` objects via the IREE/MLIR
Python builder API instead of text concatenation.

This module is the first step of the builder API port.  Once all callers
are migrated, the text-based ``_RegionEmitter`` in ``scalar.py`` will be
retired.
"""

from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
from typing import Any

from remora.hir import (
    HIRCall,
    HIRCast,
    HIRExpr,
    HIRIf,
    HIRLet,
    HIRLit,
    HIRPrimOp,
    HIRVar,
)
from remora.operators import comparison_mlir_op, comparison_predicate

from remora.lowering.scalar import _lower_callable_operand
from remora.lowering.types import (
    RemoraLoweringError,
    _is_scalar_type,
    type_to_mlir,
)

__all__ = [
    "_BuilderOperand",
    "_BuilderRegionEmitter",
]


# ---------------------------------------------------------------------------
# Builder operand
# ---------------------------------------------------------------------------


@dataclass
class _BuilderOperand:
    """An SSA value produced by the builder API.

    Attributes
    ----------
    value : str
        SSA name in the MLIR text representation (e.g. ``%v0``).
    type : str
        MLIR type string (e.g. ``f32``, ``i32``, ``i1``).
    ir_value : ir.Value | None
        The MLIR ``Value`` object, set when the builder built this operand.
    lines : list[str]
        Accumulated text lines that produced this operand (for compatibility with
        the text-based ``_Operand`` interface).
    """

    value: str
    type: str = ""
    ir_value: Any = None
    lines: list[str] | None = None

    def __post_init__(self) -> None:
        if self.lines is None:
            self.lines = []


# ---------------------------------------------------------------------------
# Type / constant helpers
# ---------------------------------------------------------------------------


def _ir_type_for(text_type: str, context: Any) -> Any:
    """Convert an MLIR type string to an ``ir.Type``."""
    ir = import_module("iree.compiler.ir")
    if text_type == "f32":
        return ir.F32Type.get()
    if text_type == "i32":
        return ir.IntegerType.get_signless(32)
    if text_type == "i1":
        return ir.IntegerType.get_signless(1)
    if text_type == "index":
        return ir.IndexType.get()
    # For tensor / memref / other complex types, parse from string
    if "tensor" in text_type or "memref" in text_type or "<" in text_type:
        return ir.Type.parse(text_type, context)
    raise RemoraLoweringError(f"cannot create builder type for '{text_type}'")


def _constant_attr(value: Any, text_type: str) -> Any:
    """Create an ``ir.Attribute`` for a constant value."""
    ir = import_module("iree.compiler.ir")
    if text_type == "f32":
        return ir.FloatAttr.get(_ir_type_for(text_type, None), float(value))
    if text_type == "i32":
        return ir.IntegerAttr.get(_ir_type_for(text_type, None), int(value))
    if text_type == "i1":
        return ir.BoolAttr.get(bool(value))
    raise RemoraLoweringError(f"cannot create constant for type '{text_type}'")


def _literal_value_text(value: Any, result_type: str) -> str:
    """Format a literal value as an MLIR constant text fragment."""
    if result_type == "f32":
        return f"{float(value):.6e}"
    if result_type == "i32":
        return str(int(value))
    if result_type == "i1":
        return "true" if value else "false"
    raise RemoraLoweringError(f"cannot lower literal of type {result_type}")


# ---------------------------------------------------------------------------
# CmpI/CmpF predicate mapping
# ---------------------------------------------------------------------------

# Map from ``comparison_predicate`` output to ``arith.CmpIPredicate`` / ``CmpFPredicate``
_CMPI_PRED_MAP: dict[str, int] = {
    "eq": 0,
    "ne": 1,
    "slt": 2,
    "sle": 3,
    "sgt": 4,
    "sge": 5,
}

_CMPF_PRED_MAP: dict[str, int] = {
    "oeq": 1,
    "ogt": 2,
    "oge": 3,
    "olt": 4,
    "ole": 5,
    "one": 6,
}


def _cmp_mlir_op_name(operand_type: str) -> str:
    """Return ``arith.cmpi`` or ``arith.cmpf`` for *operand_type*."""
    return "arith.cmpi" if operand_type in ("i32", "i1") else "arith.cmpf"


def _cmp_predicate_attr(predicate_name: str, operand_type: str) -> Any:
    """Build an ``IntegerAttr`` holding the comparison predicate index."""
    ir = import_module("iree.compiler.ir")
    pred_map = _CMPI_PRED_MAP if operand_type in ("i32", "i1") else _CMPF_PRED_MAP
    idx = pred_map.get(predicate_name, 0)
    return ir.IntegerAttr.get(ir.IntegerType.get_signless(64), idx)


# ---------------------------------------------------------------------------
# Emitter
# ---------------------------------------------------------------------------


class _BuilderRegionEmitter:
    """Region emitter that uses the MLIR builder API.

    Mirrors the interface of ``_RegionEmitter`` in ``scalar.py`` but
    creates ``ir.Value`` objects via the IREE/MLIR Python builder API
    instead of text concatenation.

    Parameters
    ----------
    block :
        The ``ir.Block`` to build operations into.
    input_name : str
        SSA name of the input argument (for ``linalg.generic`` region emission).
    input_type : str
        MLIR type string of the input argument.
    next_temp : int
        Starting value for the temporary SSA name counter.
    functions :
        Dictionary of HIR function definitions for resolving call references.
    """

    def __init__(
        self,
        block: Any,
        *,
        input_name: str = "",
        input_type: str = "",
        next_temp: int = 0,
        functions: dict[str, Any] | None = None,
    ) -> None:
        self._ir = import_module("iree.compiler.ir")
        self._block = block
        self._input_name = input_name
        self._input_type = input_type
        self._next_temp = next_temp
        self._functions = functions or {}
        self.lines: list[str] = []

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def emit_expr(
        self, expr: HIRExpr, env: dict[str, _BuilderOperand]
    ) -> _BuilderOperand:
        """Emit an MLIR expression and return its SSA value."""
        if isinstance(expr, HIRVar):
            return self._emit_var(expr, env)
        if isinstance(expr, HIRLit):
            return self._emit_literal(expr)
        if isinstance(expr, HIRCast):
            return self._emit_cast(expr, env)
        if isinstance(expr, HIRLet):
            return self._emit_let(expr, env)
        if isinstance(expr, HIRIf):
            return self._emit_if(expr, env)
        if isinstance(expr, HIRCall):
            return self._emit_call(expr, env)
        if isinstance(expr, HIRPrimOp):
            return self._emit_prim_op(expr, env)
        raise RemoraLoweringError(
            f"cannot lower HIR expression {type(expr).__name__} in builder map body"
        )

    def temp(self) -> str:
        """Allocate a unique SSA name."""
        value = f"%v{self._next_temp}"
        self._next_temp += 1
        return value

    @property
    def next_temp(self) -> int:
        return self._next_temp

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @property
    def _ip(self) -> Any:
        """An insertion point at the end of the builder block."""
        return self._ir.InsertionPoint(self._block)

    @property
    def _ctx(self) -> Any:
        """The MLIR context from the containing block."""
        return self._block.owner.opview.context

    def _emit_text(self, text: str) -> None:
        """Record a text line (input and result SSA names are our
        managed names, not MLIR autogenerated names)."""
        self.lines.append(text)

    def _create_op(
        self,
        op_name: str,
        operands: list[Any] | None = None,
        results: list[Any] | None = None,
        attributes: dict[str, Any] | None = None,
    ) -> Any:
        """Create an MLIR operation in the builder block.

        Returns the ``Operation`` object.
        """
        kwargs: dict[str, Any] = {"ip": self._ip}
        if operands is not None:
            kwargs["operands"] = operands
        if results is not None:
            kwargs["results"] = results
        if attributes is not None:
            kwargs["attributes"] = attributes
        return self._ir.Operation.create(op_name, **kwargs)

    # ------------------------------------------------------------------
    # Expression emitters
    # ------------------------------------------------------------------

    def _emit_var(
        self, expr: HIRVar, env: dict[str, _BuilderOperand]
    ) -> _BuilderOperand:
        try:
            operand = env[expr.name]
        except KeyError as exc:
            raise RemoraLoweringError(
                f"unbound HIR variable {expr.name}"
            ) from exc
        return _BuilderOperand(
            operand.value,
            operand.type or self._input_type,
            ir_value=operand.ir_value,
        )

    def _emit_literal(self, expr: HIRLit) -> _BuilderOperand:
        result_type = type_to_mlir(expr.type)
        name = self.temp()
        value_text = _literal_value_text(expr.value, result_type)

        ir_type = _ir_type_for(result_type, self._ctx)
        attr = _constant_attr(expr.value, result_type)

        const_op = self._create_op(
            "arith.constant",
            results=[ir_type],
            attributes={"value": attr},
        )
        line = f"      {name} = arith.constant {value_text} : {result_type}"
        self._emit_text(line)
        return _BuilderOperand(
            value=name, type=result_type, ir_value=const_op.result, lines=[line]
        )

    def _emit_cast(
        self, expr: HIRCast, env: dict[str, _BuilderOperand]
    ) -> _BuilderOperand:
        value = self.emit_expr(expr.value, env)
        from_type = value.type
        to_type = type_to_mlir(expr.result_type)
        if from_type == to_type:
            return value

        if from_type == "i32" and to_type == "f32":
            result_name = self.temp()
            to_ir_type = _ir_type_for(to_type, self._ctx)
            ir_value = None
            if value.ir_value is not None:
                cast_op = self._create_op(
                    "arith.sitofp",
                    operands=[value.ir_value],
                    results=[to_ir_type],
                )
                ir_value = cast_op.result
            line = (
                f"      {result_name} = arith.sitofp {value.value} : "
                f"{from_type} to {to_type}"
            )
            self._emit_text(line)
            return _BuilderOperand(
                value=result_name, type=to_type, ir_value=ir_value, lines=[line]
            )

        raise RemoraLoweringError(
            f"cannot cast {from_type} to {to_type} in builder lowering"
        )

    def _emit_let(
        self, expr: HIRLet, env: dict[str, _BuilderOperand]
    ) -> _BuilderOperand:
        if not _is_scalar_type(expr.value_type) or not _is_scalar_type(
            expr.result_type
        ):
            raise RemoraLoweringError(
                "only scalar lets lower through the builder SSA environment so far"
            )
        value = self.emit_expr(expr.value, env)
        return self.emit_expr(expr.body, {**env, expr.name: value})

    def _emit_if(
        self, expr: HIRIf, env: dict[str, _BuilderOperand]
    ) -> _BuilderOperand:
        condition = self.emit_expr(expr.condition, env)
        then_branch = self.emit_expr(expr.then_branch, env)
        else_branch = self.emit_expr(expr.else_branch, env)
        condition = self._coerce(condition, "i1")
        result_type = type_to_mlir(expr.result_type)
        then_branch = self._coerce(then_branch, result_type)
        else_branch = self._coerce(else_branch, result_type)

        result_name = self.temp()
        ir_type = _ir_type_for(result_type, self._ctx)
        ir_value = None

        if (
            condition.ir_value is not None
            and then_branch.ir_value is not None
            and else_branch.ir_value is not None
        ):
            select_op = self._create_op(
                "arith.select",
                operands=[
                    condition.ir_value,
                    then_branch.ir_value,
                    else_branch.ir_value,
                ],
                results=[ir_type],
            )
            ir_value = select_op.result

        line = (
            f"      {result_name} = arith.select {condition.value}, "
            f"{then_branch.value}, {else_branch.value} : {result_type}"
        )
        self._emit_text(line)
        return _BuilderOperand(
            value=result_name, type=result_type, ir_value=ir_value, lines=[line]
        )

    def _emit_call(
        self, expr: HIRCall, env: dict[str, _BuilderOperand]
    ) -> _BuilderOperand:
        func_def = self._functions.get(expr.func_name)
        if func_def is None:
            raise RemoraLoweringError(f"unknown HIR function {expr.func_name}")

        args = [self.emit_expr(arg, env) for arg in expr.args]
        if len(args) != len(func_def.params):
            raise RemoraLoweringError(f"function {expr.func_name} arity mismatch")

        result_type = type_to_mlir(expr.result_type)
        arg_values = ", ".join(arg.value for arg in args)
        arg_types = ", ".join(arg.type for arg in args)
        result_name = self.temp()

        ir_type = _ir_type_for(result_type, self._ctx)
        arg_ir_values = [a.ir_value for a in args]
        ir_value = None

        if all(v is not None for v in arg_ir_values):
            call_op = self._create_op(
                "func.call",
                operands=arg_ir_values,
                results=[ir_type],
                attributes={
                    "callee": self._ir.FlatSymbolRefAttr.get(expr.func_name),
                },
            )
            ir_value = call_op.result

        line = (
            f"      {result_name} = func.call @{expr.func_name}({arg_values}) "
            f": ({arg_types}) -> {result_type}"
        )
        self._emit_text(line)
        return _BuilderOperand(
            value=result_name, type=result_type, ir_value=ir_value, lines=[line]
        )

    def _emit_prim_op(
        self, expr: HIRPrimOp, env: dict[str, _BuilderOperand]
    ) -> _BuilderOperand:
        args = [self.emit_expr(arg, env) for arg in expr.args]
        result_type = type_to_mlir(expr.result_type)
        op = expr.op

        if len(args) != 2:
            raise RemoraLoweringError(
                "only binary primitive operations lower to builder"
            )

        # Arithmetic ops
        if op in {"+f", "-f", "*f", "/f", "+i", "-i", "*i"}:
            coerced = [self._coerce(arg, result_type) for arg in args]
            return self._emit_arith_op(op, coerced, result_type)

        # Comparison ops
        if op in {"<b", "<=b", "==b", "!=b"}:
            left, right = args
            if left.type != right.type:
                raise RemoraLoweringError(
                    f"comparison operands must have the same lowered type, "
                    f"got {left.type} and {right.type}"
                )
            return self._emit_comparison_op(op, left, right, result_type)

        # Boolean logic ops
        if op in {"&&b", "||b"}:
            left = self._coerce(args[0], "i1")
            right = self._coerce(args[1], "i1")
            return self._emit_logic_op(op, left, right, result_type)

        raise RemoraLoweringError(f"primitive HIR op {op} is deferred")

    # ------------------------------------------------------------------
    # Sub-operation emitters
    # ------------------------------------------------------------------

    _ARITH_OP_MAP: dict[str, str] = {
        "+f": "arith.addf",
        "-f": "arith.subf",
        "*f": "arith.mulf",
        "/f": "arith.divf",
        "+i": "arith.addi",
        "-i": "arith.subi",
        "*i": "arith.muli",
    }

    def _emit_arith_op(
        self, op: str, args: list[_BuilderOperand], result_type: str
    ) -> _BuilderOperand:
        mlir_op_name = self._ARITH_OP_MAP.get(op)
        if mlir_op_name is None:
            raise RemoraLoweringError(f"unknown arithmetic op '{op}'")

        result_name = self.temp()
        ir_type = _ir_type_for(result_type, self._ctx)
        ir_value = None

        if args[0].ir_value is not None and args[1].ir_value is not None:
            arith_op = self._create_op(
                mlir_op_name,
                operands=[args[0].ir_value, args[1].ir_value],
                results=[ir_type],
            )
            ir_value = arith_op.result

        line = (
            f"      {result_name} = {mlir_op_name} "
            f"{args[0].value}, {args[1].value} : {result_type}"
        )
        self._emit_text(line)
        return _BuilderOperand(
            value=result_name, type=result_type, ir_value=ir_value, lines=[line]
        )

    def _emit_comparison_op(
        self, op: str, left: _BuilderOperand, right: _BuilderOperand, result_type: str
    ) -> _BuilderOperand:
        base_op = op[:-1]  # strip trailing 'b'
        operand_type = left.type
        result_name = self.temp()

        mlir_op_name = _cmp_mlir_op_name(operand_type)
        predicate_name = comparison_predicate(base_op, operand_type)
        predicate_attr = _cmp_predicate_attr(predicate_name, operand_type)

        ir_type = _ir_type_for(result_type, self._ctx)
        ir_value = None

        if left.ir_value is not None and right.ir_value is not None:
            cmp_op = self._create_op(
                mlir_op_name,
                operands=[left.ir_value, right.ir_value],
                results=[ir_type],
                attributes={"predicate": predicate_attr},
            )
            ir_value = cmp_op.result

        # Use the text-based helper for the text line format
        mlir_op_text, predicate_text = comparison_mlir_op(base_op, operand_type), predicate_name
        line = (
            f"      {result_name} = {mlir_op_text} {predicate_text}, "
            f"{left.value}, {right.value} : {operand_type}"
        )
        self._emit_text(line)
        return _BuilderOperand(
            value=result_name, type=result_type, ir_value=ir_value, lines=[line]
        )

    def _emit_logic_op(
        self,
        op: str,
        left: _BuilderOperand,
        right: _BuilderOperand,
        result_type: str,
    ) -> _BuilderOperand:
        mlir_op_name = "arith.andi" if op == "&&b" else "arith.ori"
        result_name = self.temp()
        ir_type = _ir_type_for(result_type, self._ctx)
        ir_value = None

        if left.ir_value is not None and right.ir_value is not None:
            logic_op = self._create_op(
                mlir_op_name,
                operands=[left.ir_value, right.ir_value],
                results=[ir_type],
            )
            ir_value = logic_op.result

        line = (
            f"      {result_name} = {mlir_op_name} "
            f"{left.value}, {right.value} : {result_type}"
        )
        self._emit_text(line)
        return _BuilderOperand(
            value=result_name, type=result_type, ir_value=ir_value, lines=[line]
        )

    # ------------------------------------------------------------------
    # Coercion
    # ------------------------------------------------------------------

    def _coerce(self, operand: _BuilderOperand, target_type: str) -> _BuilderOperand:
        from_type = operand.type
        if from_type == target_type:
            return operand

        if from_type == "i32" and target_type == "f32":
            to_ir_type = _ir_type_for(target_type, self._ctx)
            result_name = self.temp()
            ir_value = None
            if operand.ir_value is not None:
                cast_op = self._create_op(
                    "arith.sitofp",
                    operands=[operand.ir_value],
                    results=[to_ir_type],
                )
                ir_value = cast_op.result
            line = (
                f"      {result_name} = arith.sitofp {operand.value} : "
                f"{from_type} to {target_type}"
            )
            self._emit_text(line)
            return _BuilderOperand(
                value=result_name, type=target_type, ir_value=ir_value, lines=[line]
            )

        raise RemoraLoweringError(
            f"cannot coerce {from_type} to {target_type} in builder lowering"
        )


# ---------------------------------------------------------------------------
# Convenience: build a standalone scalar expression module
# ---------------------------------------------------------------------------


def _build_scalar_module(expr: HIRExpr) -> str:
    """Lower a single scalar HIR expression to MLIR using the builder API.

    This produces the same output format as ``_lower_scalar_module`` in
    ``scalar.py`` but builds operations through the MLIR Python builder API
    instead of text concatenation.
    """
    ir = import_module("iree.compiler.ir")
    from iree.compiler.dialects import func as func_dialect

    ctx = ir.Context()
    ctx.allow_unregistered_dialects = True
    loc = ir.Location.unknown(ctx)

    result_type = type_to_mlir(
        expr.result_type
        if hasattr(expr, "result_type")
        else getattr(expr, "type", None)
    )

    with ctx, loc:
        module = ir.Module.create(loc)
        main_type = ir.FunctionType.get([], [_ir_type_for(result_type, ctx)])
        main_op = func_dialect.FuncOp(
            "main", main_type, ip=ir.InsertionPoint(module.body)
        )
        entry_block = main_op.add_entry_block()

        emitter = _BuilderRegionEmitter(entry_block)
        result = emitter.emit_expr(expr, {})
        from remora.lowering.scalar import _cast_if_needed

        cast_lines = _cast_if_needed(
            result.value, result.type, result_type, "%result_cast"
        )
        for cl in cast_lines:
            emitter._emit_text(cl)
        result_value = "%result_cast" if result.type != result_type else result.value

        if result.ir_value is not None:
            func_dialect.ReturnOp(
                [result.ir_value], ip=ir.InsertionPoint(entry_block)
            )

        # Wrap in the expected `module { func.func @main ... }` format
        return str(module)
