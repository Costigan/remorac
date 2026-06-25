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
    HIRAppend,
    HIRApply,
    HIRArrayLit,
    HIRCall,
    HIRCast,
    HIRDrop,
    HIRExpr,
    HIRFold,
    HIRFoldRight,
    HIRIf,
    HIRIndex,
    HIRIota,
    HIRLambda,
    HIRLet,
    HIRLit,
    HIRMap,
    HIRPair,
    HIRFirst,
    HIRSecond,
    HIRPrimCallable,
    HIRPrimOp,
    HIRRavel,
    HIRReduce,
    HIRReshape,
    HIRReverse,
    HIRRotate,
    HIRScatterAdd,
    HIRSlice,
    HIRSubarray,
    HIRTake,
    HIRTranspose,
    HIRVar,
    HIRWithShape,
)
from remora.types import BOOL, FLOAT, FLOAT64, INT, ArrayType, ScalarType, StaticDim


# ---------------------------------------------------------------------------
# GPU expression IR dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GpuInputLoad:
    """Load from an input descriptor at coordinates.

    ``index`` is the descriptor slot index.
    ``coords`` is the list of coordinate specifiers (SSA names or integers).
    ``coord_offsets`` is an optional per-axis additive offset (for drop/subarray).
    ``coord_transforms`` is an optional per-axis transform string:
      - ``""`` — identity
      - ``"reverse:N"`` — replace coord with ``N - 1 - coord``
      - ``"mod:N:S"`` — replace coord with ``(coord + S) % N``
    Transforms are applied before offsets during emission.
    """

    index: int
    coords: list[str]
    coord_offsets: tuple[int, ...] = ()
    coord_transforms: tuple[str, ...] = ()
    element_type: str = "f32"


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
    element_type: str = "f32"


@dataclass(frozen=True)
class GpuCompareOp:
    """Comparison producing i1."""

    op: str  # "<", ">", "<=", ">=", "==", "!="
    left: "GpuExpr"
    right: "GpuExpr"
    element_type: str = "f32"


@dataclass(frozen=True)
class GpuSelect:
    """Branchless select."""

    condition: "GpuExpr"
    true_val: "GpuExpr"
    false_val: "GpuExpr"
    value_type: str | None = None  # explicit MLIR type for true/false values


@dataclass(frozen=True)
class GpuCast:
    """Integer/float conversion."""

    expr: "GpuExpr"
    from_type: str
    to_type: str


@dataclass(frozen=True)
class GpuReduce:
    """Per-thread scalar or array reduction (scf.for loop).

    ``op``: reduction operator (``"+"``, ``"*"``).
    ``init``: initial value (single GpuExpr for scalar, list for array-valued).
    ``body_expr``: per-element expression (single GpuExpr for scalar).
    ``components``: when non-empty, per-component body expressions for
      array-valued folds.  Each component gets its own accumulator but
      shares the same loop.  When set, ``init`` is a list of per-component
      init expressions and ``body_expr`` is ignored.
    ``dimension``: loop bound.
    ``loop_var_name``: name for the loop index variable inside body_expr.
    """

    op: str
    init: "GpuExpr | list[GpuExpr]"
    body_expr: "GpuExpr"
    dimension: int
    loop_var_name: str = "_reduction_idx"
    components: list["GpuExpr"] = field(default_factory=list)
    reverse: bool = False


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


@dataclass(frozen=True)
class GpuArrayExpr:
    """An array-typed value: K scalar GpuExpr components.

    Created when lowering a HIRIndex that produces an ArrayType result
    (fewer indices than the source array's rank).  Each component is a
    separate GpuExpr (typically GpuInputLoad at different coordinates).

    During emission, all K components are emitted and returned as a list
    of SSA names.
    """

    components: list["GpuExpr"]
    element_type: str = "f32"


@dataclass(frozen=True)
class GpuExtractComponent:
    """Extract the k-th scalar from an array-valued expression.

    At emission time, the ``array`` must resolve to a GpuArrayExpr
    (or a multi-value list), and the k-th SSA name is returned.
    """

    array: "GpuExpr"
    index: int


@dataclass(frozen=True)
class GpuIntrinsic:
    """Call an LLVM intrinsic (sqrt, exp, log, etc.)."""

    intrinsic: str  # "sqrt", "exp", "log"
    arg: "GpuExpr"


@dataclass(frozen=True)
class GpuFlatLoad:
    """Load from an input descriptor using a flat (linear) index.

    Used for reshape/ravel where thread coords correspond to the output
    shape but the load uses the source descriptor's base pointer + offset.
    ``output_shape`` is the shape used to compute the flat index from coords.
    """

    index: int
    coords: list[str]
    output_shape: tuple[int, ...]
    element_type: str = "f32"


@dataclass(frozen=True)
class GpuAppendLoad:
    """Load from one of two input descriptors based on the thread index.

    If the first coordinate < ``left_size``, load from ``left_index`` at
    coordinate; otherwise load from ``right_index`` at ``coord - left_size``.
    """

    left_index: int
    right_index: int
    left_size: int
    coords: list[str]
    element_type: str = "f32"


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
    | GpuArrayExpr
    | GpuExtractComponent
    | GpuIntrinsic
    | GpuFlatLoad
    | GpuAppendLoad
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
    input_adjustments: dict[str, tuple[tuple[int, ...], tuple[str, ...]]] = field(default_factory=dict)
    input_flat_shapes: dict[str, tuple[int, ...]] = field(default_factory=dict)
    input_broadcast_skip: dict[str, int] = field(default_factory=dict)
    input_element_types: dict[str, str] = field(default_factory=dict)
    # Phase 6.3: function definitions for HIRCall inlining.
    # Populated by the general map builder from the containing HIRFunction.
    functions: dict = field(default_factory=dict)


def _scalar_type_to_mlir(t: ScalarType) -> str:
    if t == FLOAT:
        return "f32"
    if t == FLOAT64:
        return "f64"
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
    input_adjustments: dict[str, tuple[tuple[int, ...], tuple[str, ...]]] | None = None,
    input_flat_shapes: dict[str, tuple[int, ...]] | None = None,
    input_broadcast_skip: dict[str, int] | None = None,
    input_element_types: dict[str, str] | None = None,
    functions: dict | None = None,
) -> GpuExpr:
    """Compile a HIR expression to a GpuExpr."""
    ctx = _CompileCtx(
        input_map=input_map,
        scalar_env=scalar_env,
        coords=list(coords),
        coord_map=dict(coord_map or {}),
        context=context,
        input_adjustments=dict(input_adjustments or {}),
        input_flat_shapes=dict(input_flat_shapes or {}),
        input_broadcast_skip=dict(input_broadcast_skip or {}),
        input_element_types=dict(input_element_types or {}),
        functions=dict(functions or {}),
    )
    return _lower_hir(expr, ctx)


def _lower_hir(expr: HIRExpr, ctx: _CompileCtx) -> GpuExpr:
    # HIRVar
    if isinstance(expr, HIRVar):
        if expr.name in ctx.let_env:
            return GpuLetBinding(expr.name)
        if expr.name in ctx.input_map:
            slot = ctx.input_map[expr.name]
            et = ctx.input_element_types.get(expr.name, "f32")
            flat_shape = ctx.input_flat_shapes.get(expr.name)
            if flat_shape is not None:
                return GpuFlatLoad(slot, list(ctx.coords), flat_shape, et)
            bskip = ctx.input_broadcast_skip.get(expr.name, 0)
            use_coords = list(ctx.coords)[bskip:] if bskip > 0 else list(ctx.coords)
            adj = ctx.input_adjustments.get(expr.name)
            if adj is not None:
                offsets, transforms = adj
                return GpuInputLoad(slot, use_coords, offsets, transforms, et)
            return GpuInputLoad(slot, use_coords, element_type=et)
        if expr.name in ctx.coord_map:
            return GpuIndexCoordinate(ctx.coord_map[expr.name])
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
        cond = _lower_hir(expr.condition, ctx)
        then_val = _lower_hir(expr.then_branch, ctx)
        else_val = _lower_hir(expr.else_branch, ctx)
        if isinstance(then_val, GpuArrayExpr) and isinstance(else_val, GpuArrayExpr):
            K = len(then_val.components)
            comps = [GpuSelect(cond, then_val.components[k], else_val.components[k]) for k in range(K)]
            return GpuArrayExpr(components=comps, element_type=then_val.element_type)
        return GpuSelect(cond, then_val, else_val,
                         value_type=_scalar_type_to_mlir(expr.result_type) if isinstance(expr.result_type, ScalarType) else None)

    # HIRIndex
    if isinstance(expr, HIRIndex):
        return _lower_index(expr, ctx)

    # HIRFold / HIRReduce / HIRFoldRight
    if isinstance(expr, (HIRFold, HIRReduce, HIRFoldRight)):
        return _lower_fold_to_gpu(expr, ctx)

    # HIRLet
    if isinstance(expr, HIRLet):
        saved_input_map = None
        if isinstance(expr.value, HIRVar) and expr.value.name in ctx.input_map:
            saved_input_map = dict(ctx.input_map)
            ctx.input_map[expr.name] = ctx.input_map[expr.value.name]
        value_expr = _lower_hir(expr.value, ctx)
        saved_env = dict(ctx.let_env)
        ctx.let_env[expr.name] = _placeholder(expr.name)
        body_expr = _lower_hir(expr.body, ctx)
        ctx.let_env = saved_env
        if saved_input_map is not None:
            ctx.input_map = saved_input_map
        return _GpuLetExpr(expr.name, value_expr, body_expr)

    # HIRIota (standalone, inside a fold or map body)
    if isinstance(expr, HIRIota):
        return _lower_iota(expr, ctx)

    # HIRTake — the outer map already constrains iteration to count elements
    if isinstance(expr, HIRTake):
        return _lower_hir(expr.array, ctx)

    # HIRDrop — offset first coordinate by count
    if isinstance(expr, HIRDrop):
        return _lower_view_offset(expr.array, ctx, dim0_offset=expr.count)

    # HIRSubarray — offset each coordinate by the subarray offsets
    if isinstance(expr, HIRSubarray):
        offsets = tuple(int(o.value) for o in expr.offsets)
        return _lower_view_offset(expr.array, ctx, per_axis_offsets=offsets)

    # HIRSlice (standalone) — offset first coordinate by start
    if isinstance(expr, HIRSlice):
        return _lower_view_offset_slice(expr, ctx)

    # HIRReverse — reverse first coordinate
    if isinstance(expr, HIRReverse):
        N = int(expr.result_type.shape[0].value)
        return _lower_view_transform(expr.array, ctx, dim0_transform=f"reverse:{N}")

    # HIRRotate — modular shift on first coordinate
    if isinstance(expr, HIRRotate):
        N = int(expr.result_type.shape[0].value)
        shift = int(expr.shift.value)
        return _lower_view_transform(expr.array, ctx, dim0_transform=f"mod:{N}:{shift}")

    # HIRTranspose — swap first two coordinates
    if isinstance(expr, HIRTranspose):
        return _lower_transpose(expr, ctx)

    # HIRArrayLit — lower each element to GpuArrayExpr
    if isinstance(expr, HIRArrayLit):
        components = [_lower_hir(e, ctx) for e in expr.elements]
        elem_type = _scalar_type_to_mlir(expr.result_type.element)
        return GpuArrayExpr(components=components, element_type=elem_type)

    # HIRReshape — flat-index load from contiguous source
    if isinstance(expr, HIRReshape):
        return _lower_reshape(expr, ctx)

    # HIRRavel — special case of reshape to rank-1
    if isinstance(expr, HIRRavel):
        return _lower_ravel(expr, ctx)

    # HIRAppend — conditional load from left or right
    if isinstance(expr, HIRAppend):
        return _lower_append(expr, ctx)

    # HIRWithShape — broadcast: drop leading coordinates
    if isinstance(expr, HIRWithShape):
        return _lower_with_shape(expr, ctx)

    if isinstance(expr, HIRScatterAdd):
        target_expr = _lower_hir(expr.target, ctx)
        update_expr = _lower_hir(expr.update, ctx)
        index_val = None
        if isinstance(expr.index, HIRLit):
            index_val = int(expr.index.value)
        if isinstance(target_expr, GpuArrayExpr) and index_val is not None:
            components = list(target_expr.components)
            if 0 <= index_val < len(components):
                components[index_val] = GpuBinaryOp(
                    "+", components[index_val], update_expr, "f32",
                )
            return GpuArrayExpr(components, element_type=target_expr.element_type)
        if index_val is not None and ctx.coords:
            coord = GpuIndexCoordinate(ctx.coords[0])
            index_const = GpuConstant(index_val, "i64")
            cond = GpuCompareOp("eq", coord, index_const, "i64")
            sum_val = GpuBinaryOp("+", target_expr, update_expr, "f32")
            return GpuSelect(cond, sum_val, target_expr)
        raise GPUScaffoldError(
            f"{ctx.context}: HIRScatterAdd requires compile-time constant index"
        )

    # HIRCall — inline callee body for non-recursive calls (Phase 6.3)
    if isinstance(expr, HIRCall):
        callee_name = expr.func_name if hasattr(expr, 'func_name') else None
        if callee_name and callee_name in ctx.functions:
            from remora.compiler import _substitute_hir
            callee_fn = ctx.functions[callee_name]
            subs = {}
            for i, p in enumerate(callee_fn.params):
                if i < len(expr.args):
                    subs[p.name] = expr.args[i]
            inlined = _substitute_hir(callee_fn.body, subs)
            return _lower_hir(inlined, ctx)
        raise GPUScaffoldError(
            f"{ctx.context}: function calls are not supported on GPU"
        )

    # HIRPair / HIRFirst / HIRSecond — pair construction and projection
    if isinstance(expr, HIRPair):
        left = _lower_hir(expr.left, ctx)
        right = _lower_hir(expr.right, ctx)
        # Represent pairs as 2-component arrays for GPU emission
        if isinstance(left, GpuArrayExpr) and isinstance(right, GpuArrayExpr):
            return GpuArrayExpr(
                components=list(left.components) + list(right.components),
                element_type=left.element_type,
            )
        left_comps = left.components if isinstance(left, GpuArrayExpr) else [left]
        right_comps = right.components if isinstance(right, GpuArrayExpr) else [right]
        elem_type = getattr(left_comps[0], 'element_type', 'f32') if left_comps else 'f32'
        return GpuArrayExpr(components=left_comps + right_comps, element_type=elem_type)

    if isinstance(expr, HIRFirst):
        inner = _lower_hir(expr.pair, ctx)
        if isinstance(inner, GpuArrayExpr):
            return inner.components[0] if len(inner.components) > 0 else inner
        return inner

    if isinstance(expr, HIRSecond):
        inner = _lower_hir(expr.pair, ctx)
        if isinstance(inner, GpuArrayExpr):
            return inner.components[1] if len(inner.components) > 1 else inner
        return inner

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


def _maybe_expand_rank1_cell(inner: GpuExpr, array: HIRExpr) -> "GpuArrayExpr | None":
    """Expand a per-cell rank-1 array reference into component loads.

    Inside a general GPU map, a reference to an array-typed *cell* parameter
    lowers to a single ``GpuInputLoad`` carrying only the *frame* coordinates.
    A view op (reverse/rotate/drop) applied to it must act on the *cell* axis,
    not the frame axis.  This helper expands that single load into a
    ``GpuArrayExpr`` of ``K`` component loads — one per cell element, each
    appending the cell index as a trailing coordinate — so the view op can be
    applied component-wise (the correct, frame-agnostic path).

    Returns ``None`` when expansion does not apply (not a plain input load,
    already carries coord offsets/transforms, or the cell is not rank-1).
    """
    if not isinstance(inner, GpuInputLoad):
        return None
    if inner.coord_offsets or inner.coord_transforms:
        return None
    cell_type = getattr(array, "type", None)
    if cell_type is None:
        cell_type = getattr(array, "result_type", None)
    if not isinstance(cell_type, ArrayType) or cell_type.rank != 1:
        return None
    K = int(cell_type.shape[0].value)
    if K <= 0:
        return None
    comps: list[GpuExpr] = [
        GpuInputLoad(
            inner.index,
            list(inner.coords) + [str(k)],
            element_type=inner.element_type,
        )
        for k in range(K)
    ]
    return GpuArrayExpr(components=comps, element_type=inner.element_type)


def _apply_offset_to_load(
    inner: GpuExpr,
    offsets: tuple[int, ...],
    ctx: _CompileCtx,
) -> GpuExpr:
    if isinstance(inner, GpuInputLoad):
        raise GPUScaffoldError(
            f"{ctx.context}: view-with-offset (drop/subarray) inside a map body is "
            f"not supported on a per-cell array reference; the offset would target "
            f"the frame axis instead of the cell axis"
        )
    if isinstance(inner, GpuArrayExpr):
        start = offsets[0] if offsets else 0
        return GpuArrayExpr(
            components=inner.components[start:],
            element_type=inner.element_type,
        )
    raise GPUScaffoldError(
        f"{ctx.context}: view offset on non-input-load expression ({type(inner).__name__})"
    )


def _lower_view_offset(
    array: HIRExpr,
    ctx: _CompileCtx,
    *,
    dim0_offset: int = 0,
    per_axis_offsets: tuple[int, ...] | None = None,
) -> GpuExpr:
    inner = _lower_hir(array, ctx)
    if per_axis_offsets is not None:
        return _apply_offset_to_load(inner, per_axis_offsets, ctx)
    expanded = _maybe_expand_rank1_cell(inner, array)
    if expanded is not None:
        inner = expanded
    return _apply_offset_to_load(inner, (dim0_offset,), ctx)


def _lower_view_offset_slice(expr: HIRSlice, ctx: _CompileCtx) -> GpuExpr:
    raise GPUScaffoldError(
        f"{ctx.context}: standalone HIRSlice is not supported in GPU lowering"
    )


def _apply_transform_to_load(
    inner: GpuExpr,
    dim0_transform: str,
    ctx: _CompileCtx,
) -> GpuExpr:
    if isinstance(inner, GpuInputLoad):
        raise GPUScaffoldError(
            f"{ctx.context}: view-transform (reverse/rotate) inside a map body is "
            f"not supported on a per-cell array reference; the transform would target "
            f"the frame axis instead of the cell axis"
        )
    if isinstance(inner, GpuArrayExpr):
        if dim0_transform.startswith("reverse:"):
            return GpuArrayExpr(
                components=list(reversed(inner.components)),
                element_type=inner.element_type,
            )
        if dim0_transform.startswith("mod:"):
            parts = dim0_transform.split(":")
            N = int(parts[1])
            shift = int(parts[2])
            comps = inner.components
            rotated = comps[shift % N:] + comps[:shift % N]
            return GpuArrayExpr(
                components=rotated,
                element_type=inner.element_type,
            )
    raise GPUScaffoldError(
        f"{ctx.context}: view transform on non-input-load expression ({type(inner).__name__})"
    )


def _lower_view_transform(
    array: HIRExpr,
    ctx: _CompileCtx,
    *,
    dim0_transform: str,
) -> GpuExpr:
    inner = _lower_hir(array, ctx)
    expanded = _maybe_expand_rank1_cell(inner, array)
    if expanded is not None:
        inner = expanded
    return _apply_transform_to_load(inner, dim0_transform, ctx)


def _lower_transpose(expr: HIRTranspose, ctx: _CompileCtx) -> GpuExpr:
    inner = _lower_hir(expr.array, ctx)
    if isinstance(inner, GpuInputLoad):
        raise GPUScaffoldError(
            f"{ctx.context}: transpose inside a map body is not supported on a "
            f"per-cell array reference; the axis swap would target the frame axes "
            f"instead of the cell axes"
        )
    raise GPUScaffoldError(
        f"{ctx.context}: transpose on non-input-load expression ({type(inner).__name__})"
    )


def _lower_reshape(expr: HIRReshape, ctx: _CompileCtx) -> GpuExpr:
    inner = _lower_hir(expr.array, ctx)
    if isinstance(inner, GpuInputLoad):
        output_shape = tuple(int(d.value) for d in expr.result_type.shape)
        return GpuFlatLoad(inner.index, list(ctx.coords), output_shape)
    raise GPUScaffoldError(
        f"{ctx.context}: reshape on non-input-load expression ({type(inner).__name__})"
    )


def _lower_ravel(expr: HIRRavel, ctx: _CompileCtx) -> GpuExpr:
    inner = _lower_hir(expr.array, ctx)
    if isinstance(inner, GpuInputLoad):
        output_shape = tuple(int(d.value) for d in expr.result_type.shape)
        return GpuFlatLoad(inner.index, list(ctx.coords), output_shape)
    raise GPUScaffoldError(
        f"{ctx.context}: ravel on non-input-load expression ({type(inner).__name__})"
    )


def _lower_append(expr: HIRAppend, ctx: _CompileCtx) -> GpuExpr:
    left = _lower_hir(expr.left, ctx)
    right = _lower_hir(expr.right, ctx)
    if isinstance(left, GpuInputLoad) and isinstance(right, GpuInputLoad):
        left_type = getattr(expr.left, 'type', None)
        if left_type is None:
            left_type = getattr(expr.left, 'result_type', None)
        if not isinstance(left_type, ArrayType):
            raise GPUScaffoldError(
                f"{ctx.context}: append left operand has no array type"
            )
        if left_type.rank > 1:
            raise GPUScaffoldError(
                f"{ctx.context}: GPU append supports rank-1 arrays only (got rank {left_type.rank})"
            )
        left_size = int(left_type.shape[0].value)
        return GpuAppendLoad(left.index, right.index, left_size, list(ctx.coords))
    raise GPUScaffoldError(
        f"{ctx.context}: append on non-input-load expressions"
    )


def _lower_with_shape(expr: HIRWithShape, ctx: _CompileCtx) -> GpuExpr:
    inner = _lower_hir(expr.source, ctx)
    if isinstance(inner, GpuInputLoad):
        source_type = getattr(expr.source, 'type', None)
        if source_type is None:
            source_type = getattr(expr.source, 'result_type', None)
        source_rank = source_type.rank if isinstance(source_type, ArrayType) else 0
        target_rank = expr.result_type.rank
        dims_to_skip = target_rank - source_rank
        trailing_coords = list(ctx.coords)[dims_to_skip:]
        return GpuInputLoad(inner.index, trailing_coords)
    raise GPUScaffoldError(
        f"{ctx.context}: with-shape on non-input-load expression ({type(inner).__name__})"
    )


def _lower_prim_op(expr: HIRPrimOp, ctx: _CompileCtx) -> GpuExpr:
    op = expr.op
    base_op = op
    elem_type = "f32"
    for suffix, etype in (("f", "f32"), ("i", "i32"), ("b", "i1")):
        if base_op.endswith(suffix):
            base_op = base_op[:-1]
            elem_type = etype
            break

    lowered_args = [_lower_hir(a, ctx) for a in expr.args]

    if base_op in {"+", "-", "*", "/"}:
        if len(lowered_args) != 2:
            raise GPUScaffoldError(f"{ctx.context}: binary op needs 2 args")
        return _gpu_element_wise_binary(base_op, lowered_args[0], lowered_args[1], elem_type)

    if base_op in {"<", ">", "<=", ">=", "==", "!="}:
        if len(lowered_args) != 2:
            raise GPUScaffoldError(f"{ctx.context}: comparison needs 2 args")
        # Infer element type from operand HIR types (not GpuExpr attrs or op suffix).
        # GpuIndexCoordinate has no element_type; fall back to f32 for float operands.
        _hir_types = set()
        for a in expr.args:
            at = getattr(a, 'type', None)
            if at is not None:
                _hir_types.add(_scalar_type_to_mlir(at))
        _comp_et = next(iter(_hir_types)) if len(_hir_types) == 1 else "f32"
        return GpuCompareOp(base_op, lowered_args[0], lowered_args[1], _comp_et)

    if base_op in {"exp", "log", "sqrt"}:
        if len(lowered_args) != 1:
            raise GPUScaffoldError(f"{ctx.context}: math intrinsic needs 1 arg")
        return GpuIntrinsic(base_op, lowered_args[0])

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

        # Check for element-wise map: all args are GpuArrayExpr of same size
        all_arrays = len(lowered_args) >= 1 and all(
            isinstance(a, GpuArrayExpr) for a in lowered_args
        )
        if all_arrays:
            K = len(lowered_args[0].components)
            if all(len(a.components) == K for a in lowered_args[1:]):
                comps: list[GpuExpr] = []
                for k in range(K):
                    comp_args = [a.components[k] for a in lowered_args]
                    inner_ctx = _copy_ctx(ctx)
                    for param in callable_expr.params:
                        inner_ctx.let_env[param.name] = _placeholder(param.name)
                    body_expr = _lower_hir(callable_expr.body, inner_ctx)
                    for param, arg in reversed(list(zip(callable_expr.params, comp_args))):
                        body_expr = _GpuLetExpr(param.name, arg, body_expr)
                    comps.append(body_expr)
                return GpuArrayExpr(components=comps, element_type="f32")

        # General lambda inlining: bind params to lowered args, then lower body
        inner_ctx = _copy_ctx(ctx)
        for param in callable_expr.params:
            inner_ctx.let_env[param.name] = _placeholder(param.name)
        body_expr = _lower_hir(callable_expr.body, inner_ctx)
        # Wrap in let bindings so each param resolves to its lowered arg
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
        input_adjustments=dict(ctx.input_adjustments),
        input_flat_shapes=dict(ctx.input_flat_shapes),
        input_broadcast_skip=dict(ctx.input_broadcast_skip),
        input_element_types=dict(ctx.input_element_types),
    )


def _gpu_element_wise_binary(op: str, left: GpuExpr, right: GpuExpr, element_type: str = "f32") -> GpuExpr:
    """Create a binary op, promoting to element-wise GpuArrayExpr if needed."""
    if isinstance(left, GpuArrayExpr) or isinstance(right, GpuArrayExpr):
        left_comps = left.components if isinstance(left, GpuArrayExpr) else [left]
        right_comps = right.components if isinstance(right, GpuArrayExpr) else [right]
        if len(left_comps) != len(right_comps):
            raise GPUScaffoldError(
                f"element-wise op on mismatched sizes: {len(left_comps)} vs {len(right_comps)}"
            )
        comps = [GpuBinaryOp(op, l, r, element_type) for l, r in zip(left_comps, right_comps)]
        return GpuArrayExpr(components=comps, element_type=element_type)
    return GpuBinaryOp(op, left, right, element_type)


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
    rt = callable_expr.result_type
    if isinstance(rt, ArrayType):
        rt = rt.element
    if isinstance(rt, ScalarType):
        elem_type = _scalar_type_to_mlir(rt)
    else:
        elem_type = "f32"

    if op in {"+", "-", "*", "/"}:
        if len(full_args) != 2:
            raise GPUScaffoldError(f"{ctx.context}: op {op} needs 2 operands")
        return _gpu_element_wise_binary(op, full_args[0], full_args[1], elem_type)

    if op in {"<", "<=", ">", ">=", "==", "!="}:
        if len(full_args) != 2:
            raise GPUScaffoldError(f"{ctx.context}: comparison needs 2 operands")
        param_type = "f32"
        if callable_expr.params:
            p0 = callable_expr.params[0]
            if isinstance(p0, ScalarType):
                param_type = _scalar_type_to_mlir(p0)
            elif isinstance(p0, ArrayType):
                param_type = _scalar_type_to_mlir(p0.element)
        return GpuCompareOp(op, full_args[0], full_args[1], param_type)

    raise GPUScaffoldError(f"{ctx.context}: unsupported prim op '{op}'")


def _lower_index(expr: HIRIndex, ctx: _CompileCtx) -> GpuExpr:
    """Lower HIRIndex to a GpuInputLoad (scalar) or GpuArrayExpr (sub-array)."""
    if not isinstance(expr.array, HIRVar):
        raise GPUScaffoldError(
            f"{ctx.context}: index on non-variable array"
        )

    array_name = expr.array.name
    slot = ctx.input_map.get(array_name)
    if slot is None and array_name in ctx.let_env:
        bound = ctx.let_env[array_name]
        if hasattr(bound, 'expr') and isinstance(bound.expr, GpuInputLoad):
            slot = bound.expr.descriptor_index
        elif isinstance(bound, GpuInputLoad):
            slot = bound.descriptor_index
    if slot is None and isinstance(expr.array.type, ArrayType):
        raise GPUScaffoldError(
            f"{ctx.context}: index on non-input array '{array_name}'"
        )
    if slot is None:
        raise GPUScaffoldError(
            f"{ctx.context}: index on non-input variable '{array_name}'"
        )

    # Resolve index coordinates
    index_coords: list[str] = []
    let_exprs: list[tuple[str, GpuExpr]] = []  # let bindings for computed indices
    for idx in expr.indices:
        if isinstance(idx, HIRLit) and idx.type == INT:
            index_coords.append(str(int(idx.value)))
        elif isinstance(idx, HIRVar):
            if idx.name in ctx.coord_map:
                index_coords.append(ctx.coord_map[idx.name])
            elif idx.name in ctx.let_env:
                index_coords.append(f"_let_{idx.name}")
            elif idx.name in ctx.input_map:
                if len(ctx.coords) > 0:
                    index_coords.append(ctx.coords[0])
                else:
                    index_coords.append("%i0")
            else:
                raise GPUScaffoldError(
                    f"{ctx.context}: index variable '{idx.name}' not resolved"
                )
        else:
            # Computed (non-literal) index expression — lower it recursively
            # and bind to a fresh let variable, cast to i64 for addressing.
            # Phase 1.4: GPU index-in-map.
            computed = _lower_hir(idx, ctx)
            fresh_name = f"_cidx_{len(let_exprs)}_{id(idx)}"
            coord_name = fresh_name  # same name — resolved via _GpuLetExpr env
            # The expression compiler produces i32 for INT arithmetic, but
            # thread coordinates and GpuInputLoad addressing require i64.
            let_exprs.append((fresh_name, GpuCast(computed, "i32", "i64")))
            ctx.let_env[fresh_name] = _placeholder(fresh_name)
            index_coords.append(coord_name)

    # Determine the result rank from the HIRIndex result_type
    result_type = expr.result_type
    if isinstance(result_type, ScalarType):
        result = GpuInputLoad(slot, index_coords, element_type=_scalar_type_to_mlir(result_type))
    elif isinstance(result_type, ArrayType):
        import itertools
        dims = [int(d.value) for d in result_type.shape]
        if not dims or any(d <= 0 for d in dims):
            raise GPUScaffoldError(
                f"{ctx.context}: array index result has zero-size dimension"
            )
        elem_type = _scalar_type_to_mlir(result_type.element)
        components: list[GpuExpr] = []
        for multi in itertools.product(*(range(d) for d in dims)):
            full_coords = list(index_coords) + [str(x) for x in multi]
            components.append(GpuInputLoad(slot, full_coords, element_type=elem_type))
        result: GpuExpr = GpuArrayExpr(components=components, element_type=elem_type)
    else:
        raise GPUScaffoldError(
            f"{ctx.context}: unexpected index result type {type(result_type).__name__}"
        )

    # Wrap in _GpuLetExpr for any computed index bindings
    for name, value_expr in reversed(let_exprs):
        result = _GpuLetExpr(name, value_expr, result)
    return result


def _lower_fold_to_gpu(
    fold: HIRFold | HIRReduce | HIRFoldRight, ctx: _CompileCtx
) -> GpuExpr:
    """Lower a fold (scalar or array-valued) to a GpuReduce."""
    from remora.types import ArrayType, StaticDim

    is_reverse = isinstance(fold, HIRFoldRight)
    result_type = fold.result_type

    if not isinstance(fold.func, HIRPrimCallable):
        raise GPUScaffoldError(
            f"{ctx.context}: only primitive fold callables supported"
        )
    op = fold.func.op
    if op not in {"+", "*"}:
        raise GPUScaffoldError(
            f"{ctx.context}: fold op '{op}' not supported"
        )

    if not isinstance(fold.reduction_dim, StaticDim):
        raise GPUScaffoldError(
            f"{ctx.context}: fold dimension must be static"
        )
    dim = int(fold.reduction_dim.value)

    array_expr = fold.array

    # ── Resolve map-over-iota patterns into body+loop_var ──
    body_expr: GpuExpr | None = None
    loop_var_name = "_reduction_idx"

    # Pattern: fold over HIRMap(HIRLambda, [HIRIota(N)])
    if isinstance(array_expr, (HIRMap, HIRApply)):
        if isinstance(array_expr.func, HIRLambda) and len(array_expr.arrays) == 1:
            inner_array = array_expr.arrays[0]
            if isinstance(inner_array, HIRIota):
                iota_dim = int(inner_array.size.value) if isinstance(inner_array.size, StaticDim) else 0
                if iota_dim > 0:
                    param_name = array_expr.func.params[0].name
                    inner_ctx = _copy_ctx(ctx)
                    inner_ctx.coord_map[param_name] = "_iota_coord"
                    body_expr = _lower_hir(array_expr.func.body, inner_ctx)
                    loop_var_name = "_iota_coord"
                    dim = iota_dim

    # Pattern: fold over HIRIota(N) directly
    if isinstance(array_expr, HIRIota):
        iota_dim = int(array_expr.size.value) if isinstance(array_expr.size, StaticDim) else 0
        if iota_dim > 0:
            body_raw: GpuExpr = GpuIndexCoordinate("_iota_coord")
            body_expr = GpuCast(body_raw, "i64", "f32")
            dim = iota_dim
            loop_var_name = "_iota_coord"

    # Fallback: lower the array directly
    if body_expr is None:
        body_expr = _lower_hir(array_expr, ctx)

    # ── Handle result type: scalar vs array-valued ──
    if isinstance(result_type, ScalarType):
        init_expr = _lower_hir(fold.init, ctx)
        # Scalar fold with array body: use components for per-element iteration
        if isinstance(body_expr, GpuArrayExpr):
            return GpuReduce(
                op=op,
                init=init_expr,
                body_expr=body_expr.components[0],  # dummy
                dimension=dim,
                loop_var_name=loop_var_name,
                components=list(body_expr.components),
                reverse=is_reverse,
            )
        return GpuReduce(
            op=op,
            init=init_expr,
            body_expr=body_expr,
            dimension=dim,
            loop_var_name=loop_var_name,
            reverse=is_reverse,
        )

    # Array-valued result: decompose into K components via GpuExtractComponent
    assert isinstance(result_type, ArrayType)
    K = 1
    for d in result_type.shape:
        K *= int(d.value)
    if K <= 0:
        raise GPUScaffoldError(
            f"{ctx.context}: array-valued fold has zero-size result"
        )

    # Lower init to per-component expressions
    init_exprs: list[GpuExpr] = _flatten_gpu_exprs(
        _lower_fold_init_components(fold.init, K, ctx)
    )

    # Decompose body into K per-component scalar expressions
    # If the fold body came from a map-over-iota pattern producing an array,
    # unwind the _GpuLetExpr to extract the GpuArrayExpr components.
    component_bodies: list[GpuExpr] = []
    inner = body_expr
    if isinstance(inner, _GpuLetExpr):
        # Unwrap single let layer (map-over-iota wraps param in _GpuLetExpr)
        inner = inner.body

    # Materialized rank->=2 array fold: the body is a fully-materialized
    # GpuArrayExpr holding dim*K elements (e.g. folding (index m i) where the
    # cell is rank->=2). Folding reduces the *leading* axis, so reduce it at
    # compile time into K grouped accumulators — no scf.for loop is needed since
    # every element is already known. result[b] = init[b] op_a flat[a*K + b]
    # (row-major: the reduced leading axis has stride K in the flat layout).
    if isinstance(inner, GpuArrayExpr):
        flat = _flatten_gpu_exprs(list(inner.components))
        if dim >= 1 and len(flat) == dim * K and len(flat) > K:
            elem_type = _scalar_type_to_mlir(result_type.element)
            grouped: list[GpuExpr] = []
            for b in range(K):
                acc: GpuExpr = init_exprs[b] if b < len(init_exprs) else init_exprs[0]
                for a in range(dim):
                    acc = GpuBinaryOp(op, acc, flat[a * K + b], elem_type)
                grouped.append(acc)
            return GpuArrayExpr(components=grouped, element_type=elem_type)
        component_bodies = flat
    else:
        for k in range(K):
            component_bodies.append(GpuExtractComponent(body_expr, k))

    if len(component_bodies) < K:
        for k in range(len(component_bodies), K):
            component_bodies.append(GpuExtractComponent(body_expr, k))

    return GpuReduce(
        op=op,
        init=init_exprs,
        body_expr=component_bodies[0],
        dimension=dim,
        loop_var_name=loop_var_name,
        components=component_bodies,
        reverse=is_reverse,
    )


def _flatten_gpu_exprs(exprs: list[GpuExpr]) -> list[GpuExpr]:
    """Recursively flatten nested GpuArrayExpr into scalar components."""
    result: list[GpuExpr] = []
    for e in exprs:
        if isinstance(e, GpuArrayExpr):
            result.extend(_flatten_gpu_exprs(list(e.components)))
        else:
            result.append(e)
    return result


def _lower_fold_init_components(
    init: HIRExpr, K: int, ctx: _CompileCtx
) -> list[GpuExpr]:
    """Lower a fold init value into K per-component GpuExpr values."""
    if isinstance(init, HIRArrayLit):
        exprs: list[GpuExpr] = []
        for elem in init.elements:
            exprs.append(_lower_hir(elem, ctx))
        return exprs
    if isinstance(init, HIRLit):
        return [_lower_hir(init, ctx)] * K
    # General case: index into each component
    result: list[GpuExpr] = []
    init_type = getattr(init, "result_type", getattr(init, "type", None))
    elem_type = init_type.element if isinstance(init_type, ArrayType) else FLOAT
    for k in range(K):
        idx_k = HIRIndex(
            array=init,
            indices=[HIRLit(k, INT)],
            result_type=elem_type,
        )
        result.append(_lower_hir(idx_k, ctx))
    return result


def _iota_loop_name(array_expr: HIRExpr) -> str:
    """Extract the loop variable name from a map-over-iota pattern, if any."""
    if isinstance(array_expr, (HIRMap, HIRApply)):
        if (isinstance(array_expr.func, HIRLambda)
                and len(array_expr.arrays) == 1
                and isinstance(array_expr.arrays[0], HIRIota)):
            return array_expr.func.params[0].name
    return "_reduction_idx"
