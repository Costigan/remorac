"""HIR optimization passes: CSE, DCE, duplicate analysis.

Phase 4: Common-subexpression elimination on typed HIR before descriptor
MLIR generation.  Repeated pure array-valued subexpressions are lifted into
shared bindings so they are lowered once and referenced by SSA value.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import fields, is_dataclass, replace
from typing import Iterable

from remora.hir import (
    HIRApply,
    HIRAppend,
    HIRArrayLit,
    HIRBox,
    HIRCall,
    HIRCast,
    HIRDrop,
    HIRExpr,
    HIRFilter,
    HIRFold,
    HIRFoldRight,
    HIRGrade,
    HIRIf,
    HIRIm2col,
    HIRCol2im,
    HIRIndex,
    HIRIndicesOf,
    HIRIota,
    HIRFirst,
    HIRSecond,
    HIRPair,
    HIRLambda,
    HIRLet,
    HIRLit,
    HIRMap,
    HIRParam,
    HIRPrimCallable,
    HIRPrimOp,
    HIRRavel,
    HIRReduce,
    HIRReplicate,
    HIRReshape,
    HIRReverse,
    HIRRotate,
    HIRScan,
    HIRScatterAdd,
    HIRSlice,
    HIRSort,
    HIRSubarray,
    HIRTake,
    HIRTranspose,
    HIRUnbox,
    HIRVar,
    HIRWithShape,
)
from remora.types import ArrayType, RemoraType, ScalarType


# ---------------------------------------------------------------------------
# Purity / hoistability
# ---------------------------------------------------------------------------

# Nodes that are *never* hoistable (unreliable effects, unknown purity,
# closure-dependent behaviour, or already-trivial leaves).
#
# HIRVar and HIRLit are leaves — hoisting them adds indirection for no gain.
# HIRCall invokes an external function with unknown effects.
# HIRBox / HIRUnbox / HIRFilter / HIRReplicate / HIRSort / HIRGrade involve
# type-erasure, allocation, or opaque out-of-band behaviour.
# HIRScan is stateful by nature (prefix-sum).
# HIRPrimOp is a scalar operation — handled by _hoist_closed_scalar_folds.
# HIRCast is a scalar type conversion — too trivial to hoist.
_UNHOISTABLE_TYPES: tuple[type, ...] = (
    HIRVar,
    HIRLit,
    HIRCall,
    HIRBox,
    HIRUnbox,
    HIRFilter,
    HIRReplicate,
    HIRSort,
    HIRGrade,
    HIRScan,
    HIRPrimOp,
    HIRCast,
)


def _is_hoistable(expr: HIRExpr) -> bool:
    """Return *True* if *expr* is a pure node worth commoning up.

    Only array-typed expressions are hoisted because scalar subexpressions
    are already handled by ``_hoist_closed_scalar_folds``, and hoisting
    scalar leaves adds indirection without meaningful sharing.
    """
    if isinstance(expr, _UNHOISTABLE_TYPES):
        return False
    if _is_array_type(expr):
        return True
    return False


def _is_array_type(expr: HIRExpr) -> bool:
    rt = _result_type_of(expr)
    return isinstance(rt, ArrayType)


# ---------------------------------------------------------------------------
# Free variable names (local to this module so it stays self-contained)
# ---------------------------------------------------------------------------

def _free_var_names(
    expr: HIRExpr,
    bound: frozenset[str] = frozenset(),
) -> frozenset[str]:
    """Return the set of free variable names in *expr*.

    *bound* is the set of names that are lexically bound in enclosing scopes
    (let, lambda param, unbox value) and are therefore *not* free.
    """
    if isinstance(expr, HIRVar):
        return frozenset() if expr.name in bound else frozenset({expr.name})
    if isinstance(expr, HIRLet):
        return _free_var_names(expr.value, bound) | _free_var_names(
            expr.body, bound | {expr.name}
        )
    if isinstance(expr, HIRLambda):
        lambda_bound = bound | {p.name for p in expr.params}
        return _free_var_names(expr.body, lambda_bound)
    if isinstance(expr, HIRUnbox):
        return _free_var_names(expr.box_value, bound) | _free_var_names(
            expr.body, bound | {expr.value_name}
        )
    # Generic fallback: walk all dataclass fields
    free: set[str] = set()
    _collect_free_vars(expr, bound, free)
    return frozenset(free)


def _collect_free_vars(obj: object, bound: frozenset[str], out: set[str]) -> None:
    """Walk *obj* and collect free variable names into *out*."""
    if isinstance(obj, HIRVar):
        if obj.name not in bound:
            out.add(obj.name)
        return
    if isinstance(obj, HIRLet):
        _collect_free_vars(obj.value, bound, out)
        _collect_free_vars(obj.body, bound | {obj.name}, out)
        return
    if isinstance(obj, HIRLambda):
        _collect_free_vars(obj.body, bound | {p.name for p in obj.params}, out)
        return
    if isinstance(obj, HIRUnbox):
        _collect_free_vars(obj.box_value, bound, out)
        _collect_free_vars(obj.body, bound | {obj.value_name}, out)
        return
    if isinstance(obj, (str, int, float, bool, type(None), HIRParam)):
        return
    if isinstance(obj, HIRSlice):
        return
    if is_dataclass(obj):
        for f in fields(obj):
            _collect_free_vars(getattr(obj, f.name), bound, out)
        return
    if isinstance(obj, (list, tuple)):
        for item in obj:
            _collect_free_vars(item, bound, out)


# ---------------------------------------------------------------------------
# Structural key for CSE deduplication
# ---------------------------------------------------------------------------

def _to_cse_key(obj: object) -> object:
    """Convert *obj* to a hashable, structurally-comparable representation.

    HIR nodes may contain unhashable containers (e.g. ``list``), so this
    function recursively projects every value into a ``tuple`` form that
    Python can hash natively.
    """
    if isinstance(obj, (int, float, bool, str, type(None))):
        return obj
    if isinstance(obj, HIRLit):
        return ("Lit", obj.value, str(obj.type))
    if isinstance(obj, HIRVar):
        return ("Var", obj.name, str(obj.type))
    if isinstance(obj, HIRIota):
        return ("Iota", obj.size, str(obj.result_type))
    if isinstance(obj, HIRPrimOp):
        return ("PrimOp", obj.op, tuple(_to_cse_key(a) for a in obj.args), str(obj.result_type))
    if isinstance(obj, HIRCast):
        return ("Cast", _to_cse_key(obj.value), str(obj.from_type), str(obj.to_type), str(obj.result_type))
    if isinstance(obj, HIRArrayLit):
        return ("ArrayLit", tuple(_to_cse_key(e) for e in obj.elements), str(obj.result_type))
    if isinstance(obj, HIRMap):
        return (
            "Map",
            obj.frame_shape,
            obj.cell_shape,
            _to_cse_key(obj.func),
            tuple(_to_cse_key(a) for a in obj.arrays),
            str(obj.result_type),
        )
    if isinstance(obj, HIRApply):
        return (
            "Apply",
            obj.frame_shape,
            obj.cell_shape,
            _to_cse_key(obj.func),
            tuple(_to_cse_key(a) for a in obj.arrays),
            str(obj.result_type),
        )
    if isinstance(obj, HIRFold):
        return (
            "Fold",
            obj.reduction_dim,
            _to_cse_key(obj.func),
            _to_cse_key(obj.init),
            _to_cse_key(obj.array),
            str(obj.result_type),
        )
    if isinstance(obj, HIRReduce):
        return (
            "Reduce",
            obj.reduction_dim,
            _to_cse_key(obj.func),
            _to_cse_key(obj.init),
            _to_cse_key(obj.array),
            str(obj.result_type),
        )
    if isinstance(obj, HIRFoldRight):
        return (
            "FoldRight",
            obj.reduction_dim,
            _to_cse_key(obj.func),
            _to_cse_key(obj.init),
            _to_cse_key(obj.array),
            str(obj.result_type),
        )
    if isinstance(obj, HIRScan):
        return (
            "Scan",
            obj.reduction_dim,
            _to_cse_key(obj.func),
            _to_cse_key(obj.init),
            _to_cse_key(obj.array),
            obj.exclusive,
            obj.right,
            str(obj.result_type),
        )
    if isinstance(obj, HIRTranspose):
        return ("Transpose", _to_cse_key(obj.array), str(obj.result_type))
    if isinstance(obj, HIRReshape):
        return ("Reshape", _to_cse_key(obj.array), str(obj.result_type))
    if isinstance(obj, HIRRavel):
        return ("Ravel", _to_cse_key(obj.array), str(obj.result_type))
    if isinstance(obj, HIRReverse):
        return ("Reverse", _to_cse_key(obj.array), str(obj.result_type))
    if isinstance(obj, HIRTake):
        return ("Take", obj.count, _to_cse_key(obj.array), str(obj.result_type))
    if isinstance(obj, HIRDrop):
        return ("Drop", obj.count, _to_cse_key(obj.array), str(obj.result_type))
    if isinstance(obj, HIRRotate):
        return ("Rotate", _to_cse_key(obj.array), obj.shift, str(obj.result_type))
    if isinstance(obj, HIRSubarray):
        return ("Subarray", _to_cse_key(obj.array), obj.offsets, obj.sizes, str(obj.result_type))
    if isinstance(obj, HIRIndicesOf):
        return ("IndicesOf", _to_cse_key(obj.array), str(obj.result_type))
    if isinstance(obj, HIRWithShape):
        return ("WithShape", _to_cse_key(obj.source), str(obj.result_type))
    if isinstance(obj, HIRScatterAdd):
        return (
            "ScatterAdd",
            _to_cse_key(obj.target),
            _to_cse_key(obj.index),
            _to_cse_key(obj.update),
            str(obj.result_type),
        )
    if isinstance(obj, HIRIm2col):
        return (
            "Im2col",
            _to_cse_key(obj.image),
            obj.kernel_shape,
            obj.stride,
            str(obj.result_type),
        )
    if isinstance(obj, HIRCol2im):
        return (
            "Col2im",
            _to_cse_key(obj.columns),
            obj.image_shape,
            obj.kernel_shape,
            obj.stride,
            str(obj.result_type),
        )
    if isinstance(obj, HIRAppend):
        return ("Append", _to_cse_key(obj.left), _to_cse_key(obj.right), str(obj.result_type))
    if isinstance(obj, HIRPair):
        return ("Pair", _to_cse_key(obj.left), _to_cse_key(obj.right), str(obj.result_type))
    if isinstance(obj, HIRFirst):
        return ("First", _to_cse_key(obj.pair), str(obj.result_type))
    if isinstance(obj, HIRSecond):
        return ("Second", _to_cse_key(obj.pair), str(obj.result_type))
    if isinstance(obj, HIRIf):
        return (
            "If",
            _to_cse_key(obj.condition),
            _to_cse_key(obj.then_branch),
            _to_cse_key(obj.else_branch),
            str(obj.result_type),
        )
    if isinstance(obj, HIRIndex):
        return (
            "Index",
            _to_cse_key(obj.array),
            tuple(
                _to_cse_key(i) if isinstance(i, HIRExpr) else ("Slice", i.start, i.end, str(i.result_type))
                for i in obj.indices
            ),
            str(obj.result_type),
        )
    if isinstance(obj, HIRSlice):
        return ("Slice", obj.start, obj.end, str(obj.result_type))
    if isinstance(obj, HIRPrimCallable):
        return (
            "PrimCallable",
            obj.op,
            obj.params,
            str(obj.result_type),
            _to_cse_key(obj.left_arg) if obj.left_arg is not None else None,
            _to_cse_key(obj.right_arg) if obj.right_arg is not None else None,
        )
    if isinstance(obj, HIRLambda):
        return (
            "Lambda",
            tuple((p.name, str(p.type)) for p in obj.params),
            _to_cse_key(obj.body),
            str(obj.result_type),
        )
    if isinstance(obj, HIRLet):
        return (
            "Let",
            obj.name,
            str(obj.value_type),
            _to_cse_key(obj.value),
            _to_cse_key(obj.body),
            str(obj.result_type),
        )
    if isinstance(obj, HIRCall):
        return ("Call", obj.func_name, tuple(_to_cse_key(a) for a in obj.args), str(obj.result_type))
    if isinstance(obj, HIRBox):
        return ("Box", _to_cse_key(obj.value), str(obj.result_type))
    if isinstance(obj, HIRUnbox):
        return (
            "Unbox",
            _to_cse_key(obj.box_value),
            tuple(obj.hidden_names),
            obj.value_name,
            _to_cse_key(obj.body),
            str(obj.result_type),
        )
    if isinstance(obj, HIRFilter):
        return ("Filter", _to_cse_key(obj.predicate), _to_cse_key(obj.array), str(obj.result_type))
    if isinstance(obj, HIRReplicate):
        return ("Replicate", _to_cse_key(obj.counts), _to_cse_key(obj.array), str(obj.result_type))
    if isinstance(obj, HIRSort):
        return ("Sort", _to_cse_key(obj.array), str(obj.result_type))
    if isinstance(obj, HIRGrade):
        return ("Grade", _to_cse_key(obj.array), str(obj.result_type))
    if isinstance(obj, RemoraType):
        return ("Type", str(obj))
    if isinstance(obj, (list, tuple)):
        return tuple(_to_cse_key(item) for item in obj)
    if is_dataclass(obj):
        return (type(obj).__name__,) + tuple(
            (f.name, _to_cse_key(getattr(obj, f.name))) for f in fields(obj)
        )
    return ("?", repr(obj))


# ---------------------------------------------------------------------------
# Common-subexpression elimination
# ---------------------------------------------------------------------------

def hir_cse(expr: HIRExpr) -> tuple[HIRExpr, list[tuple[str, HIRExpr]]]:
    """Eliminate common pure subexpressions in *expr*.

    Returns ``(rewritten_expr, bindings)`` where *bindings* is a list of
    ``(name, shared_subexpr)`` pairs that must be lowered once each in the
    order given.  *rewritten_expr* references shared subexpressions via
    ``HIRVar(name)``.
    """

    hoisted: list[tuple[str, HIRExpr]] = []
    canonical: dict[object, str] = {}  # cse_key -> let name
    counter = 0

    def rewrite(node: object, local_names: frozenset[str]) -> object:
        nonlocal counter

        # ---- leaves --------------------------------------------------------
        if isinstance(node, (HIRLit, HIRSlice)):
            return node

        if isinstance(node, HIRVar):
            return node

        # ---- scoping constructs (update local_names) -----------------------
        if isinstance(node, HIRLet):
            new_value = rewrite(node.value, local_names)
            assert isinstance(new_value, HIRExpr)
            new_body = rewrite(node.body, local_names | {node.name})
            assert isinstance(new_body, HIRExpr)
            return HIRLet(node.name, node.value_type, new_value, new_body, node.result_type)

        if isinstance(node, HIRLambda):
            lambda_locals = local_names | {p.name for p in node.params}
            new_body = rewrite(node.body, lambda_locals)
            assert isinstance(new_body, HIRExpr)
            return HIRLambda(node.params, new_body, node.result_type)

        if isinstance(node, HIRUnbox):
            new_box = rewrite(node.box_value, local_names)
            assert isinstance(new_box, HIRExpr)
            new_body = rewrite(node.body, local_names | {node.value_name})
            assert isinstance(new_body, HIRExpr)
            return HIRUnbox(new_box, node.hidden_names, node.value_name, new_body, node.result_type)

        # ---- compound nodes (rewrite children, then consider hoisting) ------
        if isinstance(node, HIRExpr):
            rewritten = _rewrite_children(node, local_names, rewrite)
            # Only hoist when:
            #   1) the node is pure AND
            #   2) it does not reference a name that is locally bound
            if _is_hoistable(rewritten):
                free_in_rewritten = _free_var_names(rewritten)
                if not (free_in_rewritten & local_names):
                    key = _to_cse_key(rewritten)
                    if key in canonical:
                        return HIRVar(canonical[key], _result_type_of(rewritten))
                    name = f"__cse_{counter}"
                    canonical[key] = name
                    counter += 1
                    hoisted.append((name, rewritten))
                    return HIRVar(name, _result_type_of(rewritten))
            return rewritten

        # ---- non-HIRExpr dataclasses / lists / tuples ---------------------
        if is_dataclass(node):
            updates = {
                f.name: rewrite(getattr(node, f.name), local_names)
                for f in fields(node)
            }
            return replace(node, **updates)  # type: ignore[type-var]
        if isinstance(node, list):
            return [rewrite(item, local_names) for item in node]
        if isinstance(node, tuple):
            return tuple(rewrite(item, local_names) for item in node)
        return node

    rewritten = rewrite(expr, frozenset())
    assert isinstance(rewritten, HIRExpr)
    return rewritten, hoisted


def _rewrite_children(
    node: HIRExpr,
    local_names: frozenset[str],
    rewrite,
) -> HIRExpr:
    """Return a copy of *node* with every child rewritten bottom-up."""
    if isinstance(node, (HIRMap, HIRApply)):
        return type(node)(
            node.frame_shape,
            node.cell_shape,
            rewrite(node.func, local_names),  # type: ignore[arg-type]
            [rewrite(a, local_names) for a in node.arrays],  # type: ignore[arg-type]
            node.result_type,
        )
    if isinstance(node, (HIRFold, HIRReduce)):
        return type(node)(
            node.reduction_dim,
            rewrite(node.func, local_names),  # type: ignore[arg-type]
            rewrite(node.init, local_names),  # type: ignore[arg-type]
            rewrite(node.array, local_names),  # type: ignore[arg-type]
            node.result_type,
        )
    if isinstance(node, HIRFoldRight):
        return HIRFoldRight(
            node.reduction_dim,
            rewrite(node.func, local_names),  # type: ignore[arg-type]
            rewrite(node.init, local_names),  # type: ignore[arg-type]
            rewrite(node.array, local_names),  # type: ignore[arg-type]
            node.result_type,
        )
    if isinstance(node, HIRScan):
        return HIRScan(
            node.reduction_dim,
            rewrite(node.func, local_names),  # type: ignore[arg-type]
            rewrite(node.init, local_names),  # type: ignore[arg-type]
            rewrite(node.array, local_names),  # type: ignore[arg-type]
            node.exclusive,
            node.right,
            node.result_type,
        )
    if isinstance(node, HIRPrimOp):
        return HIRPrimOp(
            node.op,
            [rewrite(a, local_names) for a in node.args],  # type: ignore[arg-type]
            node.result_type,
        )
    if isinstance(node, HIRIf):
        return HIRIf(
            rewrite(node.condition, local_names),  # type: ignore[arg-type]
            rewrite(node.then_branch, local_names),  # type: ignore[arg-type]
            rewrite(node.else_branch, local_names),  # type: ignore[arg-type]
            node.result_type,
        )
    if isinstance(node, HIRCast):
        return HIRCast(
            rewrite(node.value, local_names),  # type: ignore[arg-type]
            node.from_type,
            node.to_type,
            node.result_type,
        )
    if isinstance(node, HIRIndex):
        return HIRIndex(
            rewrite(node.array, local_names),  # type: ignore[arg-type]
            [rewrite(i, local_names) if isinstance(i, HIRExpr) else i for i in node.indices],  # type: ignore[arg-type]
            node.result_type,
        )
    if isinstance(node, HIRArrayLit):
        return HIRArrayLit(
            [rewrite(e, local_names) for e in node.elements],  # type: ignore[arg-type]
            node.result_type,
        )
    if isinstance(node, HIRTranspose):
        return HIRTranspose(rewrite(node.array, local_names), node.result_type)  # type: ignore[arg-type]
    if isinstance(node, HIRReshape):
        return HIRReshape(rewrite(node.array, local_names), node.result_type)  # type: ignore[arg-type]
    if isinstance(node, HIRRavel):
        return HIRRavel(rewrite(node.array, local_names), node.result_type)  # type: ignore[arg-type]
    if isinstance(node, HIRReverse):
        return HIRReverse(rewrite(node.array, local_names), node.result_type)  # type: ignore[arg-type]
    if isinstance(node, HIRTake):
        return HIRTake(node.count, rewrite(node.array, local_names), node.result_type)  # type: ignore[arg-type]
    if isinstance(node, HIRDrop):
        return HIRDrop(node.count, rewrite(node.array, local_names), node.result_type)  # type: ignore[arg-type]
    if isinstance(node, HIRRotate):
        return HIRRotate(rewrite(node.array, local_names), node.shift, node.result_type)  # type: ignore[arg-type]
    if isinstance(node, HIRSubarray):
        return HIRSubarray(
            rewrite(node.array, local_names), node.offsets, node.sizes, node.result_type  # type: ignore[arg-type]
        )
    if isinstance(node, HIRIndicesOf):
        return HIRIndicesOf(rewrite(node.array, local_names), node.result_type)  # type: ignore[arg-type]
    if isinstance(node, HIRWithShape):
        return HIRWithShape(rewrite(node.source, local_names), node.result_type)  # type: ignore[arg-type]
    if isinstance(node, HIRScatterAdd):
        return HIRScatterAdd(
            rewrite(node.target, local_names),  # type: ignore[arg-type]
            rewrite(node.index, local_names),  # type: ignore[arg-type]
            rewrite(node.update, local_names),  # type: ignore[arg-type]
            node.result_type,
        )
    if isinstance(node, HIRIm2col):
        return HIRIm2col(
            rewrite(node.image, local_names), node.kernel_shape, node.stride, node.result_type  # type: ignore[arg-type]
        )
    if isinstance(node, HIRCol2im):
        return HIRCol2im(
            rewrite(node.columns, local_names), node.image_shape, node.kernel_shape, node.stride, node.result_type  # type: ignore[arg-type]
        )
    if isinstance(node, HIRAppend):
        return HIRAppend(
            rewrite(node.left, local_names), rewrite(node.right, local_names), node.result_type  # type: ignore[arg-type]
        )
    if isinstance(node, HIRPair):
        return HIRPair(
            rewrite(node.left, local_names), rewrite(node.right, local_names), node.result_type  # type: ignore[arg-type]
        )
    if isinstance(node, HIRFirst):
        return HIRFirst(rewrite(node.pair, local_names), node.result_type)  # type: ignore[arg-type]
    if isinstance(node, HIRSecond):
        return HIRSecond(rewrite(node.pair, local_names), node.result_type)  # type: ignore[arg-type]
    if isinstance(node, HIRCall):
        return HIRCall(
            node.func_name,
            [rewrite(a, local_names) for a in node.args],  # type: ignore[arg-type]
            node.result_type,
        )
    if isinstance(node, HIRBox):
        return HIRBox(rewrite(node.value, local_names), node.result_type)  # type: ignore[arg-type]
    if isinstance(node, HIRFilter):
        return HIRFilter(
            rewrite(node.predicate, local_names),  # type: ignore[arg-type]
            rewrite(node.array, local_names),  # type: ignore[arg-type]
            node.result_type,
        )
    if isinstance(node, HIRReplicate):
        return HIRReplicate(
            rewrite(node.counts, local_names),  # type: ignore[arg-type]
            rewrite(node.array, local_names),  # type: ignore[arg-type]
            node.result_type,
        )
    if isinstance(node, HIRSort):
        return HIRSort(rewrite(node.array, local_names), node.result_type)  # type: ignore[arg-type]
    if isinstance(node, HIRGrade):
        return HIRGrade(rewrite(node.array, local_names), node.result_type)  # type: ignore[arg-type]
    # Let and Lambda are handled before this point; Unbox too.
    # Primitive callable fields
    if isinstance(node, HIRPrimCallable):
        return HIRPrimCallable(
            node.op,
            node.params,  # type: ignore[arg-type]
            node.result_type,
            left_arg=rewrite(node.left_arg, local_names) if node.left_arg is not None else None,  # type: ignore[arg-type]
            right_arg=rewrite(node.right_arg, local_names) if node.right_arg is not None else None,  # type: ignore[arg-type]
        )
    raise AssertionError(f"unexpected HIR node type in CSE: {type(node).__name__}")


def _result_type_of(expr: HIRExpr) -> RemoraType:
    """Extract the result type of any HIRExpr."""
    if isinstance(expr, HIRLit):
        return expr.type
    if isinstance(expr, HIRVar):
        return expr.type
    if hasattr(expr, "result_type"):
        return expr.result_type  # type: ignore[return-value]
    raise AssertionError(f"cannot determine result type of {type(expr).__name__}")


# ---------------------------------------------------------------------------
# Duplicate-subtree analysis
# ---------------------------------------------------------------------------

def hir_duplicate_analysis(expr: HIRExpr) -> dict[str, int]:
    """Count structurally-duplicated subexpressions in *expr*.

    Returns a dictionary with keys:

    * ``total_subtrees`` — number of non-leaf HIRExpr nodes
    * ``unique_subtrees`` — number of structurally distinct subtrees
    * ``duplicated_subtrees`` — subtrees that appear more than once
    * ``max_duplication`` — highest occurrence count for one subtree shape
    """
    counts: Counter = Counter()

    def walk(node: object) -> None:
        if isinstance(node, (str, int, float, bool, type(None))):
            return
        if isinstance(node, (HIRLit, HIRVar, HIRSlice)):
            return
        if isinstance(node, HIRExpr):
            if not isinstance(node, (HIRLet, HIRLambda, HIRUnbox)):
                key = _to_cse_key(node)
                counts[key] += 1
            for f in fields(node):
                walk(getattr(node, f.name))
            return
        if is_dataclass(node):
            for f in fields(node):
                walk(getattr(node, f.name))
            return
        if isinstance(node, (list, tuple)):
            for item in node:
                walk(item)

    walk(expr)

    if not counts:
        return {
            "total_subtrees": 0,
            "unique_subtrees": 0,
            "duplicated_subtrees": 0,
            "max_duplication": 0,
        }

    return {
        "total_subtrees": sum(counts.values()),
        "unique_subtrees": len(counts),
        "duplicated_subtrees": sum(1 for c in counts.values() if c > 1),
        "max_duplication": max(counts.values()),
    }


# ---------------------------------------------------------------------------
# Dead-code elimination (basic: remove unused let-bindings)
# ---------------------------------------------------------------------------

def hir_dce(expr: HIRExpr) -> HIRExpr:
    """Remove ``HIRLet`` bindings whose *name* is never referenced in *body*.

    This is a simple single-pass DCE that only eliminates let-bindings.
    It does not remove unused lambda bodies or if-branches.
    """

    def rewrite(node: object, live: frozenset[str]) -> object:
        if isinstance(node, (HIRLit, HIRSlice)):
            return node
        if isinstance(node, HIRVar):
            return node
        if isinstance(node, HIRLet):
            # Check if the let-bound name is actually used in the body
            body_free = _free_var_names(node.body)
            if node.name not in body_free:
                # This let is dead — skip it
                return rewrite(node.body, live)
            # Keep the let; the value might reference live names
            new_value = rewrite(node.value, live)
            new_body = rewrite(node.body, live)
            assert isinstance(new_value, HIRExpr)
            assert isinstance(new_body, HIRExpr)
            return HIRLet(node.name, node.value_type, new_value, new_body, node.result_type)
        if isinstance(node, HIRLambda):
            return node  # Don't descend into lambdas (they have their own scope)
        if isinstance(node, HIRUnbox):
            return node  # Don't descend into unbox (complex scoping)
        if isinstance(node, HIRExpr):
            return _rewrite_children(node, live, rewrite)
        if is_dataclass(node):
            updates = {
                f.name: rewrite(getattr(node, f.name), live)
                for f in fields(node)
            }
            return replace(node, **updates)  # type: ignore[type-var]
        if isinstance(node, list):
            return [rewrite(item, live) for item in node]
        if isinstance(node, tuple):
            return tuple(rewrite(item, live) for item in node)
        return node

    rewritten = rewrite(expr, frozenset())
    assert isinstance(rewritten, HIRExpr)
    return rewritten


# ---------------------------------------------------------------------------
# Convenience: combine CSE + DCE
# ---------------------------------------------------------------------------

def hir_optimize(expr: HIRExpr) -> HIRExpr:
    """Run CSE followed by DCE on *expr*.

    Returns an expression where repeated pure subexpressions have been
    lifted into ``HIRLet`` bindings and dead bindings have been removed.
    """
    rewritten, _bindings = hir_cse(expr)
    return hir_dce(rewritten)
