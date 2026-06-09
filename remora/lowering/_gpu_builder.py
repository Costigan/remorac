"""Builder-API lowering for GPU operations (Stream E6).

Provides builder-API equivalents of the text-based GPU scaffold
functions in ``gpu_lowering.py`` for the ``gpu.module`` + ``gpu.func``
kernel path.

The LLVM descriptor-ABI path (``llvm.func`` with ``gpu.kernel`` /
``nvvm.kernel``) is deferred.  The text-based generation in
``gpu_lowering.py`` correctly handles the complex struct-descriptor
accesses, PTX register reads, multi-dimensional index decomposition,
and shared-memory tree reductions that the descriptor ABI requires.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

from remora._gpu_map_support import (
    F32BinaryExpr,
    F32CmpExpr,
    F32ConstantExpr,
    F32InputExpr,
    F32SelectExpr,
    F32MapKernel,
    F32MapOperation,
)
from remora.errors import RemoraError


class GPUScaffoldError(RemoraError):
    """Raised when a GPU scaffold cannot be built."""


def _ir() -> Any:
    return import_module("iree.compiler.ir")


# ---------------------------------------------------------------------------
# Simple gpu.module + gpu.func kernel (non-descriptor path)
# ---------------------------------------------------------------------------


def build_f32_map_gpu_scaffold(
    kernel: F32MapKernel,
    *,
    module_name: str,
    kernel_name: str,
) -> str:
    """Build a ``gpu.module`` with a mapping kernel using the builder API.

    Equivalent to ``_build_f32_map_gpu_scaffold`` in ``gpu_lowering.py``
    but constructs all MLIR via ``ir.Operation.create`` instead of f-string
    concatenation.

    Supports rank-1 through rank-3 shapes with multi-dimensional index
    decomposition.
    """
    ir_mod = _ir()
    ctx = ir_mod.Context()
    ctx.allow_unregistered_dialects = True
    loc = ir_mod.Location.unknown(ctx)

    shape = tuple(int(d) for d in kernel.shape)
    rank = len(shape)
    total_size = 1
    for d in shape:
        total_size *= d

    with ctx, loc:
        module = ir_mod.Module.create(loc)

        # ---- gpu.module ----
        gpu_op = ir_mod.Operation.create(
            "gpu.module",
            attributes={"sym_name": ir_mod.StringAttr.get(module_name)},
            regions=1,
            ip=ir_mod.InsertionPoint(module.body),
        )
        gpu_block = ir_mod.Block.create_at_start(gpu_op.regions[0], [])

        # ---- types ----
        memref_shape = "x".join(str(d) for d in shape)
        memref_t = ir_mod.Type.parse(f"memref<{memref_shape}xf32>", ctx)
        index_t = ir_mod.IndexType.get()
        f32_t = ir_mod.F32Type.get()
        i1_t = ir_mod.IntegerType.get_signless(1)
        dim_x = ir_mod.Attribute.parse("#gpu<dim x>", ctx)

        # ---- gpu.func ----
        arg_types = [memref_t] * (kernel.num_inputs + 1)
        fn_type = ir_mod.FunctionType.get(arg_types, [])

        gpu_func_op = ir_mod.Operation.create(
            "gpu.func",
            attributes={
                "sym_name": ir_mod.StringAttr.get(kernel_name),
                "function_type": ir_mod.TypeAttr.get(fn_type),
                "gpu.kernel": ir_mod.UnitAttr.get(),
            },
            regions=1,
            ip=ir_mod.InsertionPoint(gpu_block),
        )
        func_block = ir_mod.Block.create_at_start(
            gpu_func_op.regions[0], arg_types
        )

        # ---- thread / block indexing ----
        tid = _op(ir_mod, "gpu.thread_id", index_t, func_block,
                  attributes={"dimension": dim_x})
        bid = _op(ir_mod, "gpu.block_id", index_t, func_block,
                  attributes={"dimension": dim_x})
        bdim = _op(ir_mod, "gpu.block_dim", index_t, func_block,
                   attributes={"dimension": dim_x})

        block_base = _op(ir_mod, "arith.muli", index_t, func_block,
                         operands=[bid, bdim])
        idx = _op(ir_mod, "arith.addi", index_t, func_block,
                  operands=[block_base, tid])

        size_c = _op(ir_mod, "arith.constant", index_t, func_block,
                     attributes={"value": ir_mod.IntegerAttr.get(index_t, total_size)})

        # ult = 6 in arith.CmpIPredicate
        inside = _op(ir_mod, "arith.cmpi", i1_t, func_block,
                     operands=[idx, size_c],
                     attributes={"predicate": ir_mod.IntegerAttr.get(
                         ir_mod.IntegerType.get_signless(64), 6)})

        # ---- scf.if %inside { then } else { } ----
        scf_if = ir_mod.Operation.create(
            "scf.if", operands=[inside], regions=2,
            ip=ir_mod.InsertionPoint(func_block),
        )

        # Then region: load -> compute -> store
        then_blk = ir_mod.Block.create_at_start(scf_if.regions[0], [])

        index_vals = _index_decomp(shape, idx, then_blk, ir_mod, index_t)

        loaded = []
        for inp_idx in range(kernel.num_inputs):
            r = ir_mod.Operation.create(
                "memref.load",
                operands=[func_block.arguments[inp_idx]] + index_vals,
                results=[f32_t],
                ip=ir_mod.InsertionPoint(then_blk),
            ).result
            loaded.append(r)

        result = _compute(kernel, loaded, f32_t, then_blk, ir_mod)

        ir_mod.Operation.create(
            "memref.store",
            operands=[result, func_block.arguments[-1]] + index_vals,
            ip=ir_mod.InsertionPoint(then_blk),
        )
        ir_mod.Operation.create("scf.yield", ip=ir_mod.InsertionPoint(then_blk))

        # Else region: empty
        else_blk = ir_mod.Block.create_at_start(scf_if.regions[1], [])
        ir_mod.Operation.create("scf.yield", ip=ir_mod.InsertionPoint(else_blk))

        # gpu.return
        ir_mod.Operation.create("gpu.return", ip=ir_mod.InsertionPoint(func_block))

        return str(module)


def _index_decomp(
    shape: tuple[int, ...],
    idx: Any,
    block: Any,
    ir_mod: Any,
    index_t: Any,
) -> list[Any]:
    """Decompose flat thread index into multi-dimensional indices.

    Handles rank 1–10 via a general row-major decomposition algorithm.
    """
    rank = len(shape)
    if rank == 1:
        return [idx]
    if rank == 2:
        dim1_c = _op(ir_mod, "arith.constant", index_t, block,
                     attributes={"value": ir_mod.IntegerAttr.get(index_t, shape[1])})
        i0 = _op(ir_mod, "arith.divui", index_t, block,
                 operands=[idx, dim1_c])
        i1 = _op(ir_mod, "arith.remui", index_t, block,
                 operands=[idx, dim1_c])
        return [i0, i1]
    # General case: rank >= 3
    # plane[k] = product of dimensions k+1 .. rank-1
    plane_consts = []
    for axis in range(1, rank):
        plane = 1
        for d in shape[axis:]:
            plane *= d
        pc = _op(ir_mod, "arith.constant", index_t, block,
                 attributes={"value": ir_mod.IntegerAttr.get(index_t, plane)})
        plane_consts.append(pc)

    current = idx
    indices: list[Any] = []
    for axis in range(rank - 1):
        i = _op(ir_mod, "arith.divui", index_t, block,
                operands=[current, plane_consts[axis]])
        rem = _op(ir_mod, "arith.remui", index_t, block,
                  operands=[current, plane_consts[axis]])
        indices.append(i)
        current = rem
    # Last axis
    indices.append(current)
    return indices


def _compute(
    kernel: F32MapKernel,
    loaded: list[Any],
    f32_t: Any,
    block: Any,
    ir_mod: Any,
) -> Any:
    """Apply the map operation to loaded inputs."""
    if kernel.expression is not None:
        return _compute_expression(kernel.expression, loaded, f32_t, block, ir_mod)
    if kernel.num_inputs == 2:
        return _arith_op(ir_mod, kernel.operation.op, loaded[0], loaded[1],
                         f32_t, block)
    c = float(kernel.operation.constant) if kernel.operation.constant else 0.0
    c_val = _op(ir_mod, "arith.constant", f32_t, block,
                attributes={"value": ir_mod.FloatAttr.get(f32_t, c)})
    lhs = c_val if kernel.operation.constant_side == "left" else loaded[0]
    rhs = loaded[0] if kernel.operation.constant_side == "left" else c_val
    return _arith_op(ir_mod, kernel.operation.op, lhs, rhs, f32_t, block)


def _compute_expression(
    expression: F32Expr,
    loaded: list[Any],
    f32_t: Any,
    block: Any,
    ir_mod: Any,
) -> Any:
    if isinstance(expression, F32InputExpr):
        return loaded[expression.index]
    if isinstance(expression, F32ConstantExpr):
        return _op(
            ir_mod,
            "arith.constant",
            f32_t,
            block,
            attributes={"value": ir_mod.FloatAttr.get(f32_t, expression.value)},
        )
    if isinstance(expression, F32SelectExpr):
        cond = _compute_expression(expression.condition, loaded, f32_t, block, ir_mod)
        then_v = _compute_expression(expression.then_expr, loaded, f32_t, block, ir_mod)
        else_v = _compute_expression(expression.else_expr, loaded, f32_t, block, ir_mod)
        return _op(
            ir_mod, "arith.select", f32_t, block,
            operands=[cond, then_v, else_v],
        )
    if isinstance(expression, F32CmpExpr):
        left = _compute_expression(expression.left, loaded, f32_t, block, ir_mod)
        right = _compute_expression(expression.right, loaded, f32_t, block, ir_mod)
        return _cmpf_op(ir_mod, expression.op, left, right, f32_t, block)
    assert isinstance(expression, F32BinaryExpr)
    left = _compute_expression(expression.left, loaded, f32_t, block, ir_mod)
    right = _compute_expression(expression.right, loaded, f32_t, block, ir_mod)
    return _arith_op(ir_mod, expression.op, left, right, f32_t, block)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _op(
    ir_mod: Any, name: str, result_type: Any, block: Any,
    *, operands: list[Any] | None = None,
    attributes: dict[str, Any] | None = None,
) -> Any:
    """Create a single-result operation in *block* and return its result."""
    kwargs: dict[str, Any] = {
        "results": [result_type],
        "ip": ir_mod.InsertionPoint(block),
    }
    if operands:
        kwargs["operands"] = operands
    if attributes:
        kwargs["attributes"] = attributes
    return ir_mod.Operation.create(name, **kwargs).result


def _arith_op(
    ir_mod: Any, op: str, lhs: Any, rhs: Any,
    result_t: Any, block: Any,
) -> Any:
    """Create an arith binary op and return the result."""
    name = {"+": "arith.addf", "-": "arith.subf",
            "*": "arith.mulf", "/": "arith.divf"}.get(op, "arith.addf")
    return ir_mod.Operation.create(
        name, operands=[lhs, rhs], results=[result_t],
        ip=ir_mod.InsertionPoint(block),
    ).result


def _cmpf_op(
    ir_mod: Any, op: str, lhs: Any, rhs: Any,
    f32_t: Any, block: Any,
) -> Any:
    predicate = {
        ">": 2, "<": 4, ">=": 6, "<=": 3,
        "==": 1, "!=": 5,
    }.get(op, 2)
    i1_t = ir_mod.IntegerType.get_signless(1)
    return ir_mod.Operation.create(
        "arith.cmpf",
        operands=[lhs, rhs],
        results=[i1_t],
        attributes={"predicate": ir_mod.IntegerAttr.get(ir_mod.IntegerType.get_signless(64), predicate)},
        ip=ir_mod.InsertionPoint(block),
    ).result
