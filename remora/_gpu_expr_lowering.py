"""GPU expression IR and HIR -> GPUExpr compiler for general GPU lowering.

This module defines a simple expression IR (GpuExpr) that bridges HIR
and MLIR LLVM dialect emission.  The compiler walks arbitrary HIR tree
expressions inside map callable bodies and produces GpuExpr trees.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TypeAlias

from remora.gpu_lowering import GPUScaffoldError
from remora.hir import (
    HIRApply,
    HIRCast,
    HIRExpr,
    HIRFold,
    HIRIf,
    HIRIndex,
    HIRIota,
    HIRLambda,
    HIRLet,
    HIRLit,
    HIRMap,
    HIRPrimCallable,
    HIRPrimOp,
    HIRReduce,
    HIRVar,
)
from remora.types import BOOL, FLOAT, INT, ArrayType, ScalarType, StaticDim


# ---------------------------------------------------------------------------
# GPU expression IR dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GpuInputLoad:
    """Load from an input descriptor at coordinates.

    ``index`` is the descriptor slot index.
    ``coords`` is the list of coordinate specifiers (SSA names or integers).
    """

    index: int
    coords: list[str]  # SSA names or literal integers as strings


@dataclass(frozen=True)
class GpuConstant:
    """Float, int, or bool literal."""

    value: int | float | bool
    element_type: str  # "f32", "i32", "i1"


@dataclass(frozen=True)
class GpuBinaryOp:
    """Binary arithmetic on f32 or i32."""

    op: str  # "+", "-", "*", "/"
    left: "GpuExpr"
    right: "GpuExpr"


@dataclass(frozen=True)
class GpuCompareOp:
    """Comparison producing i1."""

    op: str  # "<", ">", "<=", ">=", "==", "!="
    left: "GpuExpr"
    right: "GpuExpr"


@dataclass(frozen=True)
class GpuSelect:
    """Branchless select."""

    condition: "GpuExpr"
    true_val: "GpuExpr"
    false_val: "GpuExpr"


@dataclass(frozen=True)
class GpuCast:
    """Integer/float conversion."""

    expr: "GpuExpr"
    from_type: str
    to_type: str


@dataclass(frozen=True)
class GpuReduce:
    """Per-thread scalar reduction (scf.for loop).

    ``op``: reduction operator (``"+"``, ``"*"``).
    ``init``: initial value.
    ``body_expr``: per-element expression.
    ``dimension``: loop bound.
    ``loop_var_name``: name for the loop index variable inside body_expr.
    """

    op: str
    init: "GpuExpr"
    body_expr: "GpuExpr"
    dimension: int
    loop_var_name: str = "_reduction_idx"


@dataclass(frozen=True)
class GpuScalarParam:
    """Scalar kernel parameter."""

    index: int


@dataclass(frozen=True)
class GpuLetBinding:
    """Let-bound value reference."""

    name: str


@dataclass(frozen=True)
class GpuIndexCoordinate:
    """A coordinate expression for GpuInputLoad: either a thread coord or a
    let-bound variable that will be resolved during emission."""

    name: str


@dataclass(frozen=True)
class _GpuLetExpr:
    """Internal: sequential let binding."""

    name: str
    value: "GpuExpr"
    body: "GpuExpr"


GpuExpr: TypeAlias = (
    GpuInputLoad
    | GpuConstant
    | GpuBinaryOp
    | GpuCompareOp
    | GpuSelect
    | GpuCast
    | GpuReduce
    | GpuScalarParam
    | GpuLetBinding
    | GpuIndexCoordinate
    | _GpuLetExpr
)


# ---------------------------------------------------------------------------
# Compilation context
# ---------------------------------------------------------------------------


@dataclass
class _CompileCtx:
    input_map: dict[str, int] = field(default_factory=dict)
    scalar_env: dict[str, int] = field(default_factory=dict)
    let_env: dict[str, GpuExpr] = field(default_factory=dict)
    coords: list[str] = field(default_factory=list)
    coord_map: dict[str, str] = field(default_factory=dict)
    context: str = "general GPU lowering"


def _scalar_type_to_mlir(t: ScalarType) -> str:
    if t == FLOAT:
        return "f32"
    if t == INT:
        return "i32"
    if t == BOOL:
        return "i1"
    raise GPUScaffoldError(f"unsupported scalar type: {t.name}")


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def gpu_expr_from_hir(
    expr: HIRExpr,
    *,
    input_map: dict[str, int],
    scalar_env: dict[str, int],
    coords: list[str],
    coord_map: dict[str, str] | None = None,
    context: str = "general GPU lowering",
) -> GpuExpr:
    """Compile a HIR expression to a GpuExpr."""
    ctx = _CompileCtx(
        input_map=input_map,
        scalar_env=scalar_env,
        coords=list(coords),
        coord_map=dict(coord_map or {}),
        context=context,
    )
    return _lower_hir(expr, ctx)


def _lower_hir(expr: HIRExpr, ctx: _CompileCtx) -> GpuExpr:
    # HIRVar
    if isinstance(expr, HIRVar):
        if expr.name in ctx.let_env:
            return GpuLetBinding(expr.name)
        if expr.name in ctx.input_map:
            return GpuInputLoad(ctx.input_map[expr.name], list(ctx.coords))
        if expr.name in ctx.coord_map:
            return GpuIndexCoordinate(expr.name)
        if expr.name in ctx.scalar_env:
            return GpuScalarParam(ctx.scalar_env[expr.name])
        raise GPUScaffoldError(
            f"{ctx.context}: unbound variable '{expr.name}'"
        )

    # HIRLit
    if isinstance(expr, HIRLit):
        return GpuConstant(expr.value, _scalar_type_to_mlir(expr.type))

    # HIRCast
    if isinstance(expr, HIRCast):
        from_type = _scalar_type_to_mlir(expr.from_type)
        to_type = _scalar_type_to_mlir(expr.to_type)
        return GpuCast(_lower_hir(expr.value, ctx), from_type, to_type)

    # HIRPrimOp
    if isinstance(expr, HIRPrimOp):
        return _lower_prim_op(expr, ctx)

    # HIRMap / HIRApply
    if isinstance(expr, (HIRMap, HIRApply)):
        return _lower_map_apply(expr, ctx)

    # HIRIf
    if isinstance(expr, HIRIf):
        return GpuSelect(
            _lower_hir(expr.condition, ctx),
            _lower_hir(expr.then_branch, ctx),
            _lower_hir(expr.else_branch, ctx),
        )

    # HIRIndex
    if isinstance(expr, HIRIndex):
        return _lower_index(expr, ctx)

    # HIRFold / HIRReduce
    if isinstance(expr, (HIRFold, HIRReduce)):
        return _lower_scalar_fold_to_gpu(expr, ctx)

    # HIRLet
    if isinstance(expr, HIRLet):
        value_expr = _lower_hir(expr.value, ctx)
        saved_env = dict(ctx.let_env)
        ctx.let_env[expr.name] = _placeholder(expr.name)
        body_expr = _lower_hir(expr.body, ctx)
        ctx.let_env = saved_env
        return _GpuLetExpr(expr.name, value_expr, body_expr)

    # HIRIota (standalone, inside a fold or map body)
    if isinstance(expr, HIRIota):
        return _lower_iota(expr, ctx)

    raise GPUScaffoldError(
        f"{ctx.context}: unsupported HIR node {type(expr).__name__}"
    )


def _placeholder(name: str) -> GpuLetBinding:
    return GpuLetBinding(name)


def _lower_iota(expr: HIRIota, ctx: _CompileCtx) -> GpuExpr:
    """Lower HIRIota to a GpuExpr.

    In the GPU lowering, iota represents a range that needs to be
    iterated.  Standalone iota (e.g. as a map array) resolves to
    the current coordinate.
    """
    dim = int(expr.size.value) if isinstance(expr.size, StaticDim) else 0
    if len(ctx.coords) >= 1 and dim > 0:
        return GpuIndexCoordinate("_iota_coord")
    raise GPUScaffoldError(
        f"{ctx.context}: iota with size {dim} cannot be lowered standalone"
    )


def _lower_prim_op(expr: HIRPrimOp, ctx: _CompileCtx) -> GpuExpr:
    op = expr.op
    base_op = op
    for suffix in ("f", "i", "b"):
        if base_op.endswith(suffix):
            base_op = base_op[:-1]
            break

    lowered_args = [_lower_hir(a, ctx) for a in expr.args]

    if base_op in {"+", "-", "*", "/"}:
        if len(lowered_args) != 2:
            raise GPUScaffoldError(f"{ctx.context}: binary op needs 2 args")
        return GpuBinaryOp(base_op, lowered_args[0], lowered_args[1])

    if base_op in {"<", ">", "<=", ">=", "==", "!="}:
        if len(lowered_args) != 2:
            raise GPUScaffoldError(f"{ctx.context}: comparison needs 2 args")
        return GpuCompareOp(base_op, lowered_args[0], lowered_args[1])

    raise GPUScaffoldError(f"{ctx.context}: unsupported prim op '{op}'")


def _lower_map_apply(expr: HIRMap | HIRApply, ctx: _CompileCtx) -> GpuExpr:
    callable_expr = expr.func
    args = expr.arrays

    if isinstance(callable_expr, HIRPrimCallable):
        return _lower_prim_callable(callable_expr, args, ctx)

    if isinstance(callable_expr, HIRLambda):
        # Lambda call inside an expression: inline it
        if len(callable_expr.params) != len(args):
            raise GPUScaffoldError(f"{ctx.context}: lambda arity mismatch")

        lowered_args = [_lower_hir(a, ctx) for a in args]

        # Check for the pattern: map over iota → bind param to loop variable
        # This is used inside fold bodies
        if len(args) == 1 and isinstance(args[0], HIRIota):
            iota_dim = int(args[0].size.value) if isinstance(args[0].size, StaticDim) else 0
            if iota_dim > 0:
                param_name = callable_expr.params[0].name
                # Return a GpuReduce placeholder: the outer fold handles the loop
                # We embed the lambda body into a GpuIndexCoordinate chain
                body_ctx = _copy_ctx(ctx)
                body_ctx.coord_map[param_name] = "_iota_coord"
                body_expr = _lower_hir(callable_expr.body, body_ctx)
                # Wrap in placeholder let for the param
                return _GpuLetExpr(param_name, GpuIndexCoordinate("_iota_coord"), body_expr)

        # General lambda inlining: bind params to lowered args
        body_expr = _lower_hir(callable_expr.body, ctx)
        for param, arg_expr in reversed(list(zip(callable_expr.params, lowered_args))):
            body_expr = _GpuLetExpr(param.name, arg_expr, body_expr)

        return body_expr

    raise GPUScaffoldError(
        f"{ctx.context}: unsupported callable {type(callable_expr).__name__}"
    )


def _copy_ctx(ctx: _CompileCtx) -> _CompileCtx:
    return _CompileCtx(
        input_map=dict(ctx.input_map),
        scalar_env=dict(ctx.scalar_env),
        let_env=dict(ctx.let_env),
        coords=list(ctx.coords),
        coord_map=dict(ctx.coord_map),
        context=ctx.context,
    )


def _lower_prim_callable(
    callable_expr: HIRPrimCallable,
    args: list[HIRExpr],
    ctx: _CompileCtx,
) -> GpuExpr:
    lowered_args = [_lower_hir(a, ctx) for a in args]

    full_args = list(lowered_args)
    if callable_expr.left_arg is not None:
        full_args = [_lower_hir(callable_expr.left_arg, ctx)] + full_args
    if callable_expr.right_arg is not None:
        full_args = full_args + [_lower_hir(callable_expr.right_arg, ctx)]

    op = callable_expr.op

    if op in {"+", "-", "*", "/"}:
        if len(full_args) != 2:
            raise GPUScaffoldError(f"{ctx.context}: op {op} needs 2 operands")
        return GpuBinaryOp(op, full_args[0], full_args[1])

    if op in {"<", "<=", ">", ">=", "==", "!="}:
        if len(full_args) != 2:
            raise GPUScaffoldError(f"{ctx.context}: comparison needs 2 operands")
        return GpuCompareOp(op, full_args[0], full_args[1])

    raise GPUScaffoldError(f"{ctx.context}: unsupported prim op '{op}'")


def _lower_index(expr: HIRIndex, ctx: _CompileCtx) -> GpuExpr:
    """Lower HIRIndex to a GpuInputLoad with literal or SSA coords."""
    if isinstance(expr.array, HIRVar) and expr.array.name in ctx.input_map:
        slot = ctx.input_map[expr.array.name]
    else:
        raise GPUScaffoldError(
            f"{ctx.context}: index on non-input or complex array"
        )

    index_coords: list[str] = []
    for idx in expr.indices:
        if isinstance(idx, HIRLit) and idx.type == INT:
            index_coords.append(str(int(idx.value)))
        elif isinstance(idx, HIRVar):
            if idx.name in ctx.coord_map:
                index_coords.append("_iota_coord")
            elif idx.name in ctx.let_env:
                index_coords.append("_iota_coord")
            elif idx.name in ctx.input_map:
                index_coords.append("%i0")
            else:
                raise GPUScaffoldError(
                    f"{ctx.context}: index variable '{idx.name}' not resolved"
                )
        else:
            raise GPUScaffoldError(
                f"{ctx.context}: non-literal index in HIRIndex"
            )
    return GpuInputLoad(slot, index_coords)


def _lower_scalar_fold_to_gpu(
    fold: HIRFold | HIRReduce, ctx: _CompileCtx
) -> GpuExpr:
    """Lower a fold to a GpuReduce, handling map-over-iota patterns."""
    from remora.types import ArrayType, StaticDim

    result_type = fold.result_type
    if isinstance(result_type, ArrayType):
        raise GPUScaffoldError(
            f"{ctx.context}: array-valued fold not supported"
        )

    if not isinstance(fold.func, HIRPrimCallable):
        raise GPUScaffoldError(
            f"{ctx.context}: only primitive fold callables supported"
        )
    op = fold.func.op
    if op not in {"+", "*"}:
        raise GPUScaffoldError(
            f"{ctx.context}: fold op '{op}' not supported"
        )

    init_expr = _lower_hir(fold.init, ctx)

    # Handle the fold array
    array_expr = fold.array

    # Pattern: fold over HIRMap(HIRLambda, [HIRIota(N)])
    if isinstance(array_expr, (HIRMap, HIRApply)):
        if isinstance(array_expr.func, HIRLambda) and len(array_expr.arrays) == 1:
            inner_array = array_expr.arrays[0]
            if isinstance(inner_array, HIRIota):
                iota_dim = int(inner_array.size.value) if isinstance(inner_array.size, StaticDim) else 0
                if iota_dim > 0:
                    param_name = array_expr.func.params[0].name
                    # Build a context where param_name maps to _iota_coord
                    inner_ctx = _copy_ctx(ctx)
                    inner_ctx.coord_map[param_name] = "_iota_coord"
                    body_expr = _lower_hir(array_expr.func.body, inner_ctx)
                    # Wrap in a let binding the param
                    body_with_let = _GpuLetExpr(
                        param_name,
                        GpuIndexCoordinate("_iota_coord"),
                        body_expr,
                    )
                    return GpuReduce(
                        op=op,
                        init=init_expr,
                        body_expr=body_with_let,
                        dimension=iota_dim,
                        loop_var_name="_iota_coord",
                    )

    # Pattern: fold over HIRIota(N) directly
    if isinstance(array_expr, HIRIota):
        iota_dim = int(array_expr.size.value) if isinstance(array_expr.size, StaticDim) else 0
        if iota_dim > 0:
            # Body is just the loop index value (GpuIndexCoordinate)
            return GpuReduce(
                op=op,
                init=init_expr,
                body_expr=GpuIndexCoordinate("_iota_coord"),
                dimension=iota_dim,
                loop_var_name="_iota_coord",
            )

    # General case: lower the array directly
    dim = int(fold.reduction_dim.value) if isinstance(fold.reduction_dim, StaticDim) else 1
    body_expr = _lower_hir(array_expr, ctx)
    return GpuReduce(
        op=op,
        init=init_expr,
        body_expr=body_expr,
        dimension=dim,
        loop_var_name="_reduction_idx",
    )
