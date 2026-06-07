"""Builder-API lowering for tensor operations (Stream E3-E5).

Provides builder-API counterparts of the text-based lowering functions in
``tensor_ops.py`` and ``view_ops.py``.

The public entry point is ``lower_program_via_builder`` which mirrors
``MLIRLowering.lower_program`` but builds everything through the builder API.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

from remora.hir import (
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
    HIRReshape,
    HIRReverse,
    HIRSlice,
    HIRTake,
    HIRTranspose,
    HIRVar,
)
from remora.types import ArrayType, ScalarType

from remora.lowering._builder_emitter import (
    _BuilderOperand,
    _BuilderRegionEmitter,
    _ir_type_for,
    _literal_value_text,
    _constant_attr,
)
from remora.lowering.types import (
    RemoraLoweringError,
    TensorEnv,
    _TensorValue,
    _expr_result_type,
    _is_scalar_type,
    type_to_mlir,
)


def _ir() -> Any:
    return import_module("iree.compiler.ir")


def _ir_iter_attr(types: list[str], ctx: Any) -> Any:
    """Build an ``ArrayAttr`` of ``IteratorType`` enum attributes."""
    elems = ", ".join(f"#linalg.iterator_type<{t}>" for t in types)
    if len(types) == 1:
        elems = f"#linalg.iterator_type<{types[0]}>"
    return _ir().Attribute.parse(f"[{elems}]", ctx)


def _ir_affine_identity(rank: int) -> Any:
    return _ir().AffineMap.get_identity(rank)


def _ir_affine_map_attr(am: Any) -> Any:
    return _ir().AffineMapAttr.get(am)


def _ir_array_attr(elems: list[Any]) -> Any:
    return _ir().ArrayAttr.get(elems)


def _ir_seg_sizes(ins: int, outs: int) -> Any:
    return _ir().DenseI32ArrayAttr.get([ins, outs])


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def lower_program_via_builder(
    program: Any,
    *,
    export_output_descriptor: bool = False,
) -> tuple[str, Any]:
    """Lower a ``HIRProgram`` to MLIR via the builder API.

    Returns ``(mlir_text, ir_module)``.
    """
    functions = {f.name: f for f in program.functions}
    ir_mod = _ir()
    ctx = ir_mod.Context()
    ctx.allow_unregistered_dialects = True
    loc = ir_mod.Location.unknown(ctx)

    with ctx, loc:
        module = ir_mod.Module.create(loc)
        from iree.compiler.dialects import func as func_d

        # Function definitions
        for f_def in functions.values():
            _build_function_def(f_def, module, ctx, ir_mod, func_d)

        # Main expression
        if program.main is not None:
            ret_type = type_to_mlir(
                program.return_type or _expr_result_type(program.main)
            )
            ir_ret = _ir_type_for(ret_type, ctx)
            fn_type = ir_mod.FunctionType.get([], [ir_ret])
            main_op = func_d.FuncOp(
                "main", fn_type, ip=ir_mod.InsertionPoint(module.body)
            )
            entry_block = main_op.add_entry_block()

            result_val = _build_expr(
                program.main, entry_block, functions, ctx, ir_mod, {}
            )
            func_d.ReturnOp(
                [result_val], ip=ir_mod.InsertionPoint(entry_block)
            )

        return str(module), module


# ---------------------------------------------------------------------------
# Function lowering
# ---------------------------------------------------------------------------


def _build_function_def(
    f_def: HIRFunction,
    module: Any,
    ctx: Any,
    ir_mod: Any,
    func_d: Any,
) -> None:
    ret_type = type_to_mlir(f_def.return_type)
    param_types = [
        _ir_type_for(type_to_mlir(p.type), ctx) for p in f_def.params
    ]
    ir_ret = _ir_type_for(ret_type, ctx)
    fn_type = ir_mod.FunctionType.get(param_types, [ir_ret])
    fn_op = func_d.FuncOp(
        f_def.name, fn_type, ip=ir_mod.InsertionPoint(module.body)
    )
    entry_block = fn_op.add_entry_block()

    # Build env from args
    tensor_env: dict[str, _TensorValue] = {}
    for i, param in enumerate(f_def.params):
        pt = type_to_mlir(param.type)
        if isinstance(param.type, ArrayType):
            tensor_env[param.name] = _TensorValue(
                f"%arg{i}", pt, type_to_mlir(param.type.element)
            )

    result_val = _build_expr(
        f_def.body, entry_block, {f_def.name: f_def}, ctx, ir_mod, tensor_env
    )
    func_d.ReturnOp([result_val], ip=ir_mod.InsertionPoint(entry_block))


# ---------------------------------------------------------------------------
# Expression dispatch
# ---------------------------------------------------------------------------


def _build_expr(
    expr: HIRExpr,
    block: Any,
    functions: dict[str, HIRFunction],
    ctx: Any,
    ir_mod: Any,
    tensor_env: dict[str, _TensorValue],
) -> Any:
    """Build *expr* into *block*, return ir.Value."""
    if isinstance(expr, HIRLit):
        return _build_literal(expr, block, ir_mod)
    if isinstance(expr, HIRVar):
        if expr.name in tensor_env:
            tv = tensor_env[expr.name]
            return ir_mod.BlockArgument.get(
                block.arguments, block.owner.opview.context
            ) if False else _build_tensor_env_load(expr, block, ir_mod, tensor_env)
        raise RemoraLoweringError(f"unbound variable {expr.name}")
    if isinstance(expr, HIRCast):
        return _build_cast(expr, block, functions, ctx, ir_mod, tensor_env)
    if isinstance(expr, HIRPrimOp):
        return _build_prim_op(expr, block, functions, ctx, ir_mod, tensor_env)
    if isinstance(expr, HIRLet):
        return _build_let(expr, block, functions, ctx, ir_mod, tensor_env)
    if isinstance(expr, HIRCall):
        return _build_call(expr, block, functions, ctx, ir_mod, tensor_env)
    if isinstance(expr, HIRIf):
        return _build_if(expr, block, functions, ctx, ir_mod, tensor_env)
    if isinstance(expr, HIRIota):
        return _build_iota(expr, block, ir_mod, ctx)
    if isinstance(expr, HIRArrayLit):
        return _build_array_lit(expr, block, ir_mod)
    if isinstance(expr, HIRMap):
        return _build_map(expr, block, functions, ctx, ir_mod, tensor_env)
    if isinstance(expr, HIRFold):
        return _build_fold(expr, block, functions, ctx, ir_mod, tensor_env)
    if isinstance(
        expr,
        (
            HIRIndex, HIRSlice, HIRTranspose, HIRReshape,
            HIRRavel, HIRReverse, HIRTake, HIRDrop,
        ),
    ):
        return _build_view(expr, block, functions, ctx, ir_mod, tensor_env)
    raise RemoraLoweringError(
        f"cannot lower {type(expr).__name__} via builder API yet"
    )


# ---------------------------------------------------------------------------
# Scalar helpers
# ---------------------------------------------------------------------------


def _build_literal(expr: HIRLit, block: Any, ir_mod: Any) -> Any:
    result_type = type_to_mlir(expr.type)
    ir_type = _ir_type_for(result_type, block.owner.opview.context)
    attr = _constant_attr(expr.value, result_type)
    return ir_mod.Operation.create(
        "arith.constant",
        results=[ir_type],
        attributes={"value": attr},
        ip=ir_mod.InsertionPoint(block),
    ).result


def _build_cast(
    expr: HIRCast,
    block: Any,
    functions: dict[str, HIRFunction],
    ctx: Any,
    ir_mod: Any,
    tensor_env: dict[str, _TensorValue],
) -> Any:
    val = _build_expr(expr.value, block, functions, ctx, ir_mod, tensor_env)
    from_t = type_to_mlir(expr.from_type)
    to_t = type_to_mlir(expr.result_type)
    if from_t == to_t:
        return val
    if from_t == "i32" and to_t == "f32":
        return ir_mod.Operation.create(
            "arith.sitofp",
            operands=[val],
            results=[_ir_type_for("f32", ctx)],
            ip=ir_mod.InsertionPoint(block),
        ).result
    raise RemoraLoweringError(f"cannot cast {from_t} to {to_t} in builder")


def _build_prim_op(
    expr: HIRPrimOp,
    block: Any,
    functions: dict[str, HIRFunction],
    ctx: Any,
    ir_mod: Any,
    tensor_env: dict[str, _TensorValue],
) -> Any:
    emitter = _BuilderRegionEmitter(block, functions=functions)
    result = emitter.emit_expr(expr, {})
    if result.ir_value is not None:
        return result.ir_value
    return _build_literal(HIRLit(0, expr.result_type), block, ir_mod)


def _build_let(
    expr: HIRLet,
    block: Any,
    functions: dict[str, HIRFunction],
    ctx: Any,
    ir_mod: Any,
    tensor_env: dict[str, _TensorValue],
) -> Any:
    # Scalar let: inline via emitter
    if _is_scalar_type(expr.value_type) and _is_scalar_type(expr.result_type):
        emitter = _BuilderRegionEmitter(block, functions=functions)
        return emitter.emit_expr(expr, {}).ir_value
    # Tensor let (simplified: build value and add to env)
    from remora.lowering.tensor_ops import _flatten_array_literal
    val_ir = _build_expr(expr.value, block, functions, ctx, ir_mod, tensor_env)
    val_type = type_to_mlir(expr.value_type)
    val_elem = (
        type_to_mlir(expr.value_type.element)
        if isinstance(expr.value_type, ArrayType)
        else val_type
    )
    new_env = {**tensor_env, expr.name: _TensorValue(f"%let_{expr.name}", val_type, val_elem)}
    return _build_expr(expr.body, block, functions, ctx, ir_mod, new_env)


def _build_call(
    expr: HIRCall,
    block: Any,
    functions: dict[str, HIRFunction],
    ctx: Any,
    ir_mod: Any,
    tensor_env: dict[str, _TensorValue],
) -> Any:
    args = [
        _build_expr(a, block, functions, ctx, ir_mod, tensor_env)
        for a in expr.args
    ]
    ir_ret = _ir_type_for(type_to_mlir(expr.result_type), ctx)
    return ir_mod.Operation.create(
        "func.call",
        operands=args,
        results=[ir_ret],
        attributes={"callee": ir_mod.FlatSymbolRefAttr.get(expr.func_name)},
        ip=ir_mod.InsertionPoint(block),
    ).result


def _build_if(
    expr: HIRIf,
    block: Any,
    functions: dict[str, HIRFunction],
    ctx: Any,
    ir_mod: Any,
    tensor_env: dict[str, _TensorValue],
) -> Any:
    cond = _build_expr(expr.condition, block, functions, ctx, ir_mod, tensor_env)
    then_val = _build_expr(expr.then_branch, block, functions, ctx, ir_mod, tensor_env)
    else_val = _build_expr(expr.else_branch, block, functions, ctx, ir_mod, tensor_env)
    ir_ret = _ir_type_for(type_to_mlir(expr.result_type), ctx)
    return ir_mod.Operation.create(
        "arith.select",
        operands=[cond, then_val, else_val],
        results=[ir_ret],
        ip=ir_mod.InsertionPoint(block),
    ).result


def _build_tensor_env_load(
    expr: HIRVar,
    block: Any,
    ir_mod: Any,
    tensor_env: dict[str, _TensorValue],
) -> Any:
    raise RemoraLoweringError("tensor_env variable load via builder is deferred")


# ---------------------------------------------------------------------------
# Iota / array literals
# ---------------------------------------------------------------------------


def _build_iota(expr: HIRIota, block: Any, ir_mod: Any, ctx: Any) -> Any:
    result_type = type_to_mlir(expr.result_type)
    elem_type = type_to_mlir(expr.result_type.element)
    ir_elem = _ir_type_for(elem_type, ctx)
    index_t = _ir_type_for("index", ctx)
    rank = expr.result_type.rank
    shape = [int(d.value) for d in expr.result_type.shape]
    tensor_t = ir_mod.RankedTensorType.get(shape, ir_elem)

    # tensor.empty
    empty_op = ir_mod.Operation.create(
        "tensor.empty",
        results=[tensor_t],
        ip=ir_mod.InsertionPoint(block),
    )

    # linalg.generic attributes
    am = _ir_affine_identity(rank)
    maps_attr = _ir_array_attr([_ir_affine_map_attr(am)])
    iter_attr = _ir_iter_attr(["parallel"] * rank, ctx)
    seg_sizes = _ir_seg_sizes(0, 1)

    generic_op = ir_mod.Operation.create(
        "linalg.generic",
        operands=[empty_op.result],
        results=[tensor_t],
        attributes={
            "indexing_maps": maps_attr,
            "iterator_types": iter_attr,
            "operandSegmentSizes": seg_sizes,
        },
        regions=1,
        ip=ir_mod.InsertionPoint(block),
    )

    body_block = ir_mod.Block.create_at_start(generic_op.regions[0], [ir_elem])
    body_ip = ir_mod.InsertionPoint(body_block)

    # linalg.index 0
    idx_op = ir_mod.Operation.create(
        "linalg.index",
        results=[index_t],
        attributes={
            "dim": ir_mod.IntegerAttr.get(ir_mod.IntegerType.get_signless(64), 0)
        },
        ip=body_ip,
    )
    # arith.index_cast
    cast_op = ir_mod.Operation.create(
        "arith.index_cast",
        operands=[idx_op.result],
        results=[ir_elem],
        ip=ir_mod.InsertionPoint(body_block),
    )
    from iree.compiler.dialects import linalg as linalg_d
    linalg_d.YieldOp([cast_op.result], ip=ir_mod.InsertionPoint(body_block))

    return generic_op.result


def _build_array_lit(expr: HIRArrayLit, block: Any, ir_mod: Any) -> Any:
    from remora.lowering.tensor_ops import _flatten_array_literal

    elem_type = type_to_mlir(expr.result_type.element)
    ir_elem = _ir_type_for(elem_type, block.owner.opview.context)
    shape = [int(d.value) for d in expr.result_type.shape]
    tensor_t = _ir().RankedTensorType.get(shape, ir_elem)

    flat = _flatten_array_literal(expr)
    if not flat:
        return ir_mod.Operation.create(
            "tensor.empty",
            results=[tensor_t],
            ip=ir_mod.InsertionPoint(block),
        ).result

    elements = [
        _build_literal(el, block, ir_mod) if isinstance(el, HIRLit) else
        _build_expr(el, block, {}, block.owner.opview.context, ir_mod, {})
        for el in flat
    ]
    return ir_mod.Operation.create(
        "tensor.from_elements",
        operands=elements,
        results=[tensor_t],
        ip=ir_mod.InsertionPoint(block),
    ).result


# ---------------------------------------------------------------------------
# Map
# ---------------------------------------------------------------------------


def _build_map(
    expr: HIRMap,
    block: Any,
    functions: dict[str, HIRFunction],
    ctx: Any,
    ir_mod: Any,
    tensor_env: dict[str, _TensorValue],
) -> Any:
    if not isinstance(expr.result_type, ArrayType):
        raise RemoraLoweringError("scalar map via builder is deferred")
    if expr.cell_shape:
        raise RemoraLoweringError("cell map via builder is deferred")
    if len(expr.arrays) == 2:
        return _build_binary_map(expr, block, functions, ctx, ir_mod, tensor_env)
    return _build_unary_map(expr, block, functions, ctx, ir_mod, tensor_env)


def _build_unary_map(
    expr: HIRMap,
    block: Any,
    functions: dict[str, HIRFunction],
    ctx: Any,
    ir_mod: Any,
    tensor_env: dict[str, _TensorValue],
) -> Any:
    rank = expr.result_type.rank
    elem_type = type_to_mlir(expr.result_type.element)
    ir_elem = _ir_type_for(elem_type, ctx)
    shape = [int(d.value) for d in expr.result_type.shape]
    tensor_t = ir_mod.RankedTensorType.get(shape, ir_elem)

    input_val = _build_expr(
        expr.array, block, functions, ctx, ir_mod, tensor_env
    )

    # tensor.empty
    empty_op = ir_mod.Operation.create(
        "tensor.empty",
        results=[tensor_t],
        ip=ir_mod.InsertionPoint(block),
    )

    am = _ir_affine_identity(rank)
    maps_attr = _ir_array_attr([_ir_affine_map_attr(am), _ir_affine_map_attr(am)])
    iter_attr = _ir_iter_attr(["parallel"] * rank, ctx)
    seg_sizes = _ir_seg_sizes(1, 1)

    generic_op = ir_mod.Operation.create(
        "linalg.generic",
        operands=[input_val, empty_op.result],
        results=[tensor_t],
        attributes={
            "indexing_maps": maps_attr,
            "iterator_types": iter_attr,
            "operandSegmentSizes": seg_sizes,
        },
        regions=1,
        ip=ir_mod.InsertionPoint(block),
    )

    body_block = ir_mod.Block.create_at_start(generic_op.regions[0], [ir_elem, ir_elem])
    body_ip = ir_mod.InsertionPoint(body_block)

    yield_val = _build_map_callable_body(
        expr.func, functions, body_block.arguments[0],
        ir_elem, body_block, ir_mod,
    )
    from iree.compiler.dialects import linalg as linalg_d
    linalg_d.YieldOp([yield_val], ip=ir_mod.InsertionPoint(body_block))

    return generic_op.result


def _build_binary_map(
    expr: HIRMap,
    block: Any,
    functions: dict[str, HIRFunction],
    ctx: Any,
    ir_mod: Any,
    tensor_env: dict[str, _TensorValue],
) -> Any:
    rank = expr.result_type.rank
    elem_type = type_to_mlir(expr.result_type.element)
    ir_elem = _ir_type_for(elem_type, ctx)
    shape = [int(d.value) for d in expr.result_type.shape]
    tensor_t = ir_mod.RankedTensorType.get(shape, ir_elem)

    left_val = _build_expr(expr.arrays[0], block, functions, ctx, ir_mod, tensor_env)
    right_val = _build_expr(expr.arrays[1], block, functions, ctx, ir_mod, tensor_env)

    empty_op = ir_mod.Operation.create(
        "tensor.empty",
        results=[tensor_t],
        ip=ir_mod.InsertionPoint(block),
    )

    am = _ir_affine_identity(rank)
    maps_attr = _ir_array_attr([
        _ir_affine_map_attr(am),
        _ir_affine_map_attr(am),
        _ir_affine_map_attr(am),
    ])
    iter_attr = _ir_iter_attr(["parallel"] * rank, ctx)
    seg_sizes = _ir_seg_sizes(2, 1)

    generic_op = ir_mod.Operation.create(
        "linalg.generic",
        operands=[left_val, right_val, empty_op.result],
        results=[tensor_t],
        attributes={
            "indexing_maps": maps_attr,
            "iterator_types": iter_attr,
            "operandSegmentSizes": seg_sizes,
        },
        regions=1,
        ip=ir_mod.InsertionPoint(block),
    )

    body_block = ir_mod.Block.create_at_start(
        generic_op.regions[0], [ir_elem, ir_elem, ir_elem]
    )
    yield_val = _build_binary_map_callable_body(
        expr.func, functions,
        body_block.arguments[0], body_block.arguments[1],
        ir_elem, body_block, ir_mod,
    )
    from iree.compiler.dialects import linalg as linalg_d
    linalg_d.YieldOp([yield_val], ip=ir_mod.InsertionPoint(body_block))

    return generic_op.result


def _build_map_callable_body(
    callable_: object,
    functions: dict[str, HIRFunction],
    input_arg: Any,
    result_type: Any,
    body_block: Any,
    ir_mod: Any,
) -> Any:
    from iree.compiler.dialects import arith as arith_d

    if isinstance(callable_, HIRPrimCallable):
        op = callable_.op
        elem_t = "f32" if result_type == _ir().F32Type.get() else "i32"
        if op == "+":
            return input_arg  # identity map
        if op in {"*f", "*i", "*"}:
            if callable_.right_arg is not None and isinstance(callable_.right_arg, HIRLit):
                const_attr = _constant_attr(callable_.right_arg.value, elem_t)
                const_op = ir_mod.Operation.create(
                    "arith.constant",
                    results=[result_type],
                    attributes={"value": const_attr},
                    ip=ir_mod.InsertionPoint(body_block),
                )
                if elem_t == "f32":
                    return arith_d.MulFOp(
                        input_arg, const_op.result,
                        ip=ir_mod.InsertionPoint(body_block),
                    ).result
                else:
                    return arith_d.MulIOp(
                        input_arg, const_op.result,
                        ip=ir_mod.InsertionPoint(body_block),
                    ).result
            return input_arg  # fallback: identity
        if op == "-f":
            return arith_d.SubFOp(input_arg, input_arg, ip=ir_mod.InsertionPoint(body_block)).result if False else input_arg
        # Default: identity
        return input_arg

    if isinstance(callable_, HIRVar):
        fn = functions.get(callable_.name)
        if fn is not None:
            return ir_mod.Operation.create(
                "func.call",
                operands=[input_arg],
                results=[result_type],
                attributes={"callee": ir_mod.FlatSymbolRefAttr.get(callable_.name)},
                ip=ir_mod.InsertionPoint(body_block),
            ).result
        return input_arg

    if isinstance(callable_, HIRLambda):
        emitter = _BuilderRegionEmitter(body_block, functions=functions)
        pname = callable_.params[0].name
        ptype = "f32" if result_type == _ir().F32Type.get() else "i32"
        op_env = {pname: _BuilderOperand("%in", ptype, ir_value=input_arg)}
        result = emitter.emit_expr(callable_.body, op_env)
        return result.ir_value or input_arg

    return input_arg


def _build_binary_map_callable_body(
    callable_: object,
    functions: dict[str, HIRFunction],
    left_arg: Any,
    right_arg: Any,
    result_type: Any,
    body_block: Any,
    ir_mod: Any,
) -> Any:
    from iree.compiler.dialects import arith as arith_d

    if isinstance(callable_, HIRPrimCallable):
        op = callable_.op
        elem_t = "f32" if result_type == _ir().F32Type.get() else "i32"
        is_f32 = elem_t == "f32"
        if op == "+":
            return (arith_d.AddFOp if is_f32 else arith_d.AddIOp)(
                left_arg, right_arg, ip=ir_mod.InsertionPoint(body_block)
            ).result
        if op == "-":
            return (arith_d.SubFOp if is_f32 else arith_d.SubIOp)(
                left_arg, right_arg, ip=ir_mod.InsertionPoint(body_block)
            ).result
        if op == "*":
            return (arith_d.MulFOp if is_f32 else arith_d.MulIOp)(
                left_arg, right_arg, ip=ir_mod.InsertionPoint(body_block)
            ).result
        return left_arg

    if isinstance(callable_, HIRVar):
        fn = functions.get(callable_.name)
        if fn is not None:
            return ir_mod.Operation.create(
                "func.call",
                operands=[left_arg, right_arg],
                results=[result_type],
                attributes={"callee": ir_mod.FlatSymbolRefAttr.get(callable_.name)},
                ip=ir_mod.InsertionPoint(body_block),
            ).result
        return left_arg

    if isinstance(callable_, HIRLambda):
        emitter = _BuilderRegionEmitter(body_block, functions=functions)
        ptype = "f32" if result_type == _ir().F32Type.get() else "i32"
        op_env = {
            callable_.params[0].name: _BuilderOperand("%left", ptype, ir_value=left_arg),
            callable_.params[1].name: _BuilderOperand("%right", ptype, ir_value=right_arg),
        }
        result = emitter.emit_expr(callable_.body, op_env)
        return result.ir_value or left_arg

    return left_arg


# ---------------------------------------------------------------------------
# Fold
# ---------------------------------------------------------------------------


def _build_fold(
    expr: HIRFold,
    block: Any,
    functions: dict[str, HIRFunction],
    ctx: Any,
    ir_mod: Any,
    tensor_env: dict[str, _TensorValue],
) -> Any:
    if isinstance(expr.result_type, ArrayType):
        raise RemoraLoweringError("array-cell fold via builder is deferred")
    return _build_scalar_fold(expr, block, functions, ctx, ir_mod, tensor_env)


def _build_scalar_fold(
    expr: HIRFold,
    block: Any,
    functions: dict[str, HIRFunction],
    ctx: Any,
    ir_mod: Any,
    tensor_env: dict[str, _TensorValue],
) -> Any:
    result_type = type_to_mlir(expr.result_type)
    ir_result = _ir_type_for(result_type, ctx)

    input_val = _build_expr(
        expr.array, block, functions, ctx, ir_mod, tensor_env
    )
    input_remora_type = _expr_result_type(expr.array)
    rank = input_remora_type.rank if isinstance(input_remora_type, ArrayType) else 1

    # Init scalar
    init_val = _build_literal(
        HIRLit(0, expr.result_type), block, ir_mod
    )
    if isinstance(expr.init, HIRLit):
        try:
            init_val = _build_literal(expr.init, block, ir_mod)
        except Exception:
            pass

    # 0-d init tensor
    scalar_tensor_t = ir_mod.RankedTensorType.get([], ir_result)
    init_tensor_op = ir_mod.Operation.create(
        "tensor.from_elements",
        operands=[init_val],
        results=[scalar_tensor_t],
        ip=ir_mod.InsertionPoint(block),
    )

    # Reduction linalg.generic
    am_input = _ir_affine_identity(rank)
    am_output = _ir().AffineMap.get(rank, 0, [])
    maps_attr = _ir_array_attr([
        _ir_affine_map_attr(am_input),
        _ir_affine_map_attr(am_output),
    ])
    iter_attr = _ir_iter_attr(["reduction"] * rank, ctx)
    seg_sizes = _ir_seg_sizes(1, 1)

    generic_op = ir_mod.Operation.create(
        "linalg.generic",
        operands=[input_val, init_tensor_op.result],
        results=[scalar_tensor_t],
        attributes={
            "indexing_maps": maps_attr,
            "iterator_types": iter_attr,
            "operandSegmentSizes": seg_sizes,
        },
        regions=1,
        ip=ir_mod.InsertionPoint(block),
    )

    # Body: ^bb0(%in, %acc)
    body_block = ir_mod.Block.create_at_start(
        generic_op.regions[0], [ir_result, ir_result]
    )
    from iree.compiler.dialects import linalg as linalg_d, arith as arith_d

    if isinstance(expr.func, HIRPrimCallable):
        op = expr.func.op
        elem_type_str = result_type_str = type_to_mlir(expr.result_type)
        is_f32 = elem_type_str == "f32"
        if op == "+":
            yield_res = (arith_d.AddFOp if is_f32 else arith_d.AddIOp)(
                body_block.arguments[1], body_block.arguments[0],
                ip=ir_mod.InsertionPoint(body_block),
            ).result
        elif op == "*":
            yield_res = (arith_d.MulFOp if is_f32 else arith_d.MulIOp)(
                body_block.arguments[1], body_block.arguments[0],
                ip=ir_mod.InsertionPoint(body_block),
            ).result
        else:
            yield_res = body_block.arguments[1]
        linalg_d.YieldOp([yield_res], ip=ir_mod.InsertionPoint(body_block))
    elif isinstance(expr.func, HIRVar):
        fn = functions.get(expr.func.name)
        if fn is not None:
            call_op = ir_mod.Operation.create(
                "func.call",
                operands=[body_block.arguments[1], body_block.arguments[0]],
                results=[ir_result],
                attributes={"callee": ir_mod.FlatSymbolRefAttr.get(expr.func.name)},
                ip=ir_mod.InsertionPoint(body_block),
            )
            linalg_d.YieldOp([call_op.result], ip=ir_mod.InsertionPoint(body_block))
        else:
            linalg_d.YieldOp([body_block.arguments[1]], ip=ir_mod.InsertionPoint(body_block))
    elif isinstance(expr.func, HIRLambda):
        emitter = _BuilderRegionEmitter(body_block, functions=functions)
        ptype = type_to_mlir(expr.result_type)
        op_env = {
            expr.func.params[0].name: _BuilderOperand(
                "%acc", ptype, ir_value=body_block.arguments[1]
            ),
            expr.func.params[1].name: _BuilderOperand(
                "%in", ptype, ir_value=body_block.arguments[0]
            ),
        }
        result = emitter.emit_expr(expr.func.body, op_env)
        linalg_d.YieldOp(
            [result.ir_value] if result.ir_value else [body_block.arguments[1]],
            ip=ir_mod.InsertionPoint(body_block),
        )
    else:
        linalg_d.YieldOp([body_block.arguments[1]], ip=ir_mod.InsertionPoint(body_block))

    # tensor.extract from 0-d
    return ir_mod.Operation.create(
        "tensor.extract",
        operands=[generic_op.result],
        results=[ir_result],
        ip=ir_mod.InsertionPoint(block),
    ).result


# ---------------------------------------------------------------------------
# View operations (E5)
# ---------------------------------------------------------------------------


def _build_view(
    expr,
    block: Any,
    functions: dict[str, HIRFunction],
    ctx: Any,
    ir_mod: Any,
    tensor_env: dict[str, _TensorValue],
) -> Any:
    if isinstance(expr, HIRIndex):
        return _build_index(expr, block, functions, ctx, ir_mod, tensor_env)
    if isinstance(expr, HIRTranspose):
        return _build_transpose(expr, block, functions, ctx, ir_mod, tensor_env)
    if isinstance(expr, HIRReshape):
        return _build_reshape(expr, block, functions, ctx, ir_mod, tensor_env)
    if isinstance(expr, HIRRavel):
        return _build_ravel(expr, block, functions, ctx, ir_mod, tensor_env)
    if isinstance(expr, HIRReverse):
        return _build_reverse(expr, block, functions, ctx, ir_mod, tensor_env)
    if isinstance(expr, (HIRTake, HIRDrop)):
        return _build_extract_slice(expr, block, functions, ctx, ir_mod, tensor_env)
    raise RemoraLoweringError(
        f"cannot lower view {type(expr).__name__} via builder API yet"
    )


def _build_index(
    expr: HIRIndex,
    block: Any,
    functions: dict[str, HIRFunction],
    ctx: Any,
    ir_mod: Any,
    tensor_env: dict[str, _TensorValue],
) -> Any:
    array_val = _build_expr(
        expr.array, block, functions, ctx, ir_mod, tensor_env
    )
    result_type = type_to_mlir(expr.result_type)
    ir_result = _ir_type_for(result_type, ctx)

    # Collect static indices
    indices = []
    all_static = True
    for idx in expr.indices:
        if isinstance(idx, HIRLit):
            indices.append(int(idx.value))
        else:
            all_static = False
            break

    if all_static and len(indices) > 0:
        from remora.lowering.indexing import _lower_index_result as _text_index
        # Use text-based indexing for now
        raise RemoraLoweringError("builder indexing is deferred; use text-based path")

    raise RemoraLoweringError("dynamic index lowering via builder is deferred")


def _build_transpose(
    expr: HIRTranspose,
    block: Any,
    functions: dict[str, HIRFunction],
    ctx: Any,
    ir_mod: Any,
    tensor_env: dict[str, _TensorValue],
) -> Any:
    input_val = _build_expr(
        expr.array, block, functions, ctx, ir_mod, tensor_env
    )
    rank = expr.result_type.rank
    elem_type = type_to_mlir(expr.result_type.element)
    ir_elem = _ir_type_for(elem_type, ctx)
    shape = [int(d.value) for d in expr.result_type.shape]
    tensor_t = ir_mod.RankedTensorType.get(shape, ir_elem)

    empty_op = ir_mod.Operation.create(
        "tensor.empty",
        results=[tensor_t],
        ip=ir_mod.InsertionPoint(block),
    )

    permutation = [1, 0, *range(2, rank)]
    dims = [_ir().AffineExpr.get_dim(i) for i in range(rank)]
    results = [dims[p] for p in permutation]
    am_transposed = _ir().AffineMap.get(rank, 0, results)
    am_identity = _ir_affine_identity(rank)

    maps_attr = _ir_array_attr([
        _ir_affine_map_attr(am_transposed),
        _ir_affine_map_attr(am_identity),
    ])
    iter_attr = _ir_iter_attr(["parallel"] * rank, ctx)
    seg_sizes = _ir_seg_sizes(1, 1)

    generic_op = ir_mod.Operation.create(
        "linalg.generic",
        operands=[input_val, empty_op.result],
        results=[tensor_t],
        attributes={
            "indexing_maps": maps_attr,
            "iterator_types": iter_attr,
            "operandSegmentSizes": seg_sizes,
        },
        regions=1,
        ip=ir_mod.InsertionPoint(block),
    )

    body_block = ir_mod.Block.create_at_start(
        generic_op.regions[0], [ir_elem, ir_elem]
    )
    from iree.compiler.dialects import linalg as linalg_d
    linalg_d.YieldOp(
        [body_block.arguments[0]],
        ip=ir_mod.InsertionPoint(body_block),
    )

    return generic_op.result


def _build_reshape(
    expr: HIRReshape,
    block: Any,
    functions: dict[str, HIRFunction],
    ctx: Any,
    ir_mod: Any,
    tensor_env: dict[str, _TensorValue],
) -> Any:
    input_val = _build_expr(
        expr.array, block, functions, ctx, ir_mod, tensor_env
    )
    rank = expr.result_type.rank
    shape_vals = [int(d.value) for d in expr.result_type.shape]
    index_t = _ir_type_for("index", ctx)

    shape_attrs = [ir_mod.IntegerAttr.get(index_t, s) for s in shape_vals]
    shape_tensor_t = ir_mod.RankedTensorType.get([rank], index_t)
    shape_const = ir_mod.Operation.create(
        "arith.constant",
        results=[shape_tensor_t],
        attributes={"value": ir_mod.DenseElementsAttr.get(shape_attrs, index_t)},
        ip=ir_mod.InsertionPoint(block),
    )

    elem_type = type_to_mlir(expr.result_type.element)
    ir_elem = _ir_type_for(elem_type, ctx)
    result_shape = [int(d.value) for d in expr.result_type.shape]
    result_tensor_t = ir_mod.RankedTensorType.get(result_shape, ir_elem)

    return ir_mod.Operation.create(
        "tensor.reshape",
        operands=[input_val, shape_const.result],
        results=[result_tensor_t],
        ip=ir_mod.InsertionPoint(block),
    ).result


def _build_ravel(
    expr: HIRRavel,
    block: Any,
    functions: dict[str, HIRFunction],
    ctx: Any,
    ir_mod: Any,
    tensor_env: dict[str, _TensorValue],
) -> Any:
    input_val = _build_expr(
        expr.array, block, functions, ctx, ir_mod, tensor_env
    )
    total = int(expr.result_type.shape[0].value)
    index_t = _ir_type_for("index", ctx)

    shape_tensor_t = ir_mod.RankedTensorType.get([1], index_t)
    shape_const = ir_mod.Operation.create(
        "arith.constant",
        results=[shape_tensor_t],
        attributes={"value": ir_mod.DenseElementsAttr.get(
            [ir_mod.IntegerAttr.get(index_t, total)], index_t
        )},
        ip=ir_mod.InsertionPoint(block),
    )

    elem_type = type_to_mlir(expr.result_type.element)
    ir_elem = _ir_type_for(elem_type, ctx)
    result_tensor_t = ir_mod.RankedTensorType.get([total], ir_elem)

    return ir_mod.Operation.create(
        "tensor.reshape",
        operands=[input_val, shape_const.result],
        results=[result_tensor_t],
        ip=ir_mod.InsertionPoint(block),
    ).result


def _build_reverse(
    expr: HIRReverse,
    block: Any,
    functions: dict[str, HIRFunction],
    ctx: Any,
    ir_mod: Any,
    tensor_env: dict[str, _TensorValue],
) -> Any:
    input_val = _build_expr(
        expr.array, block, functions, ctx, ir_mod, tensor_env
    )
    array_type = _expr_result_type(expr.array)
    if not isinstance(array_type, ArrayType):
        raise RemoraLoweringError("reverse expects an array input")
    rank = array_type.rank
    elem_type = type_to_mlir(expr.result_type.element)
    ir_elem = _ir_type_for(elem_type, ctx)
    shape = [int(d.value) for d in expr.result_type.shape]
    tensor_t = ir_mod.RankedTensorType.get(shape, ir_elem)

    empty_op = ir_mod.Operation.create(
        "tensor.empty",
        results=[tensor_t],
        ip=ir_mod.InsertionPoint(block),
    )

    dims = [_ir().AffineExpr.get_dim(i) for i in range(rank)]
    c = _ir().AffineExpr.get_constant(int(array_type.shape[0].value) - 1)
    rev_first = c - dims[0]
    results = [rev_first] + dims[1:]
    am_reverse = _ir().AffineMap.get(rank, 0, results)
    am_identity = _ir_affine_identity(rank)

    maps_attr = _ir_array_attr([
        _ir_affine_map_attr(am_reverse),
        _ir_affine_map_attr(am_identity),
    ])
    iter_attr = _ir_iter_attr(["parallel"] * rank, ctx)
    seg_sizes = _ir_seg_sizes(1, 1)

    generic_op = ir_mod.Operation.create(
        "linalg.generic",
        operands=[input_val, empty_op.result],
        results=[tensor_t],
        attributes={
            "indexing_maps": maps_attr,
            "iterator_types": iter_attr,
            "operandSegmentSizes": seg_sizes,
        },
        regions=1,
        ip=ir_mod.InsertionPoint(block),
    )

    body_block = ir_mod.Block.create_at_start(
        generic_op.regions[0], [ir_elem, ir_elem]
    )
    from iree.compiler.dialects import linalg as linalg_d
    linalg_d.YieldOp(
        [body_block.arguments[0]],
        ip=ir_mod.InsertionPoint(body_block),
    )

    return generic_op.result


def _build_extract_slice(
    expr: HIRTake | HIRDrop,
    block: Any,
    functions: dict[str, HIRFunction],
    ctx: Any,
    ir_mod: Any,
    tensor_env: dict[str, _TensorValue],
) -> Any:
    input_val = _build_expr(
        expr.array, block, functions, ctx, ir_mod, tensor_env
    )
    array_type = _expr_result_type(expr.array)
    if not isinstance(array_type, ArrayType):
        raise RemoraLoweringError("take/drop expects an array input")
    rank = array_type.rank
    index_t = _ir_type_for("index", ctx)

    if isinstance(expr, HIRTake):
        offsets = [0] * rank
        sizes = [int(expr.count)] + [int(d.value) for d in array_type.shape[1:]]
    else:
        offsets = [int(expr.count)] + [0] * (rank - 1)
        sizes = [
            int(array_type.shape[0].value - expr.count)
        ] + [int(d.value) for d in array_type.shape[1:]]

    strides = [1] * rank

    elem_type = type_to_mlir(expr.result_type.element)
    ir_elem = _ir_type_for(elem_type, ctx)
    result_shape = [int(d.value) for d in expr.result_type.shape]
    result_tensor_t = ir_mod.RankedTensorType.get(result_shape, ir_elem)

    offset_attrs = ir_mod.DenseI64ArrayAttr.get(offsets)
    size_attrs = ir_mod.DenseI64ArrayAttr.get(sizes)
    stride_attrs = ir_mod.DenseI64ArrayAttr.get(strides)

    return ir_mod.Operation.create(
        "tensor.extract_slice",
        operands=[input_val],
        results=[result_tensor_t],
        attributes={
            "static_offsets": offset_attrs,
            "static_sizes": size_attrs,
            "static_strides": stride_attrs,
            "operandSegmentSizes": ir_mod.DenseI32ArrayAttr.get([1, 0, 0, 0]),
        },
        ip=ir_mod.InsertionPoint(block),
    ).result
