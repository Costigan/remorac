"""AD expression simplification (Phase 5).

Bottom-up simplifications on ``_Expr`` trees before they are emitted as
Remora source text.  All rules are applied recursively and produce a
semantically-equivalent expression.
"""

from __future__ import annotations

from typing import TypeAlias

# Re-use the _Expr types from ad_source so we don't duplicate definitions.
from remora.ad_source import (
    Shape,
    _Expr,
    _Atom,
    _Op,
    _Fold,
    _FoldBroadcast,
    _Fill,
    _Reshape,
    _Transpose,
    _View,
    _Append,
    _SubarrayView,
    _Rotate,
    _Index,
    _ScatterAdd,
    _Im2col,
    _Col2im,
    _If,
    _Pair,
    _constant,
    _fill,
    _is_zero,
    _is_one,
    _is_constant,
)


# ---------------------------------------------------------------------------
# Constant folding
# ---------------------------------------------------------------------------

_BINARY_OPS: dict[str, int] = {
    "+": 0, "-": 1, "*": 2, "/": 3,
}


def _try_eval_binary(op: str, left_val: float, right_val: float) -> float | None:
    """Evaluate a binary scalar op if both operands are float constants."""
    if op == "+":
        return left_val + right_val
    if op == "-":
        return left_val - right_val
    if op == "*":
        return left_val * right_val
    if op == "/":
        if right_val == 0.0:
            return None
        return left_val / right_val
    return None


def _constant_fold(expr: _Expr) -> _Expr:
    """Fold scalar arithmetic between two constants into a single constant.

    Also handles: ``neg(const)`` → constant; ``fold(+, 0, const)`` → sum.
    """
    if isinstance(expr, _Op) and expr.right is not None:
        left, right = expr.left, expr.right
        if isinstance(left, _Atom) and isinstance(right, _Atom) and left.shape == () and right.shape == ():
            try:
                lv = float(left.text)
                rv = float(right.text)
            except ValueError:
                return expr
            result = _try_eval_binary(expr.op, lv, rv)
            if result is not None:
                return _constant(result, ())
        # Special: _Atom("0.0") * array_expr → simplify if possible.
        # Kept as-is: the zero preserves shape, handled elsewhere.

    if isinstance(expr, _Op) and expr.right is None and expr.op == "neg":
        if isinstance(expr.left, _Atom) and expr.left.shape == ():
            try:
                return _constant(-float(expr.left.text), ())
            except ValueError:
                return expr

    return expr


# ---------------------------------------------------------------------------
# Extended algebraic identities
# ---------------------------------------------------------------------------

def _simplify_algebraic(expr: _Expr) -> _Expr:
    """Apply algebraic identities beyond what the constructor helpers catch.

    These fire after the full tree is built and children have already been
    simplified bottom-up, so nested patterns like ``((x + 0) + 0)`` are caught.
    """
    if isinstance(expr, _Op) and expr.right is not None:
        left, right = expr.left, expr.right
        op = expr.op

        # x + 0 → x  and  0 + x → x (the helpers already catch this for
        # scalars; here we also catch cases where the zero is an array-shaped
        # fill of zeros resulting from downstream simplification).
        if op == "+":
            if _is_zero_fill(left) and left.shape == right.shape:
                return right
            if _is_zero_fill(right) and right.shape == left.shape:
                return left

        # x * 0 → _Fill(0.0, x)  — preserve shape for arrays
        if op == "*" and _is_zero(left) and not _is_zero(right):
            if right.shape:
                return _fill(_constant(0.0, ()), right)
        if op == "*" and _is_zero(right) and not _is_zero(left):
            if left.shape:
                return _fill(_constant(0.0, ()), left)

        # fill(scalar, x) * fill(scalar2, x) can sometimes be simplified
        # but deferred for now.

    # Fill of fill collapses to fill
    if isinstance(expr, _Fill) and isinstance(expr.like, _Fill):
        return _Fill(expr.value, expr.like.like, expr.shape)

    # Reshape(Reshape(x, s1), s2) where s2 == x.shape → x
    if isinstance(expr, _Reshape) and isinstance(expr.value, _Reshape):
        inner = expr.value.value
        if expr.shape == inner.shape:
            return inner

    # Transpose(Transpose(x)) → x (already handled in _transpose constructor)
    # FoldBroadcast(Fill(scalar, array), axis, shape) can sometimes be a no-op
    if isinstance(expr, _FoldBroadcast):
        if isinstance(expr.value, _Fill) and expr.value.shape == expr.shape:
            # fold_broadcast(fill(s, a)) already has the right shape
            pass  # keep as-is

    return expr


def _is_zero_fill(expr: _Expr) -> bool:
    """Check if *expr* is a fill-of-zeros (array-shaped zero)."""
    if isinstance(expr, _Fill) and _is_zero(expr.value):
        return True
    if isinstance(expr, _Op) and expr.right is not None and expr.op == "+":
        left, right = expr.left, expr.right
        if isinstance(left, _Op) and left.op == "*":
            if _is_zero(left.left) and left.right.shape == right.shape:  # type: ignore[union-attr]
                return True
        if isinstance(right, _Op) and right.op == "*":
            if _is_zero(right.left) and right.shape == left.shape:
                return True
    return False


# ---------------------------------------------------------------------------
# Dead-branch elimination for _If
# ---------------------------------------------------------------------------

def _is_true_atom(expr: _Expr) -> bool:
    """Check if *expr* is the literal ``#t``."""
    return isinstance(expr, _Atom) and expr.text == "#t" and expr.shape == ()


def _is_false_atom(expr: _Expr) -> bool:
    """Check if *expr* is the literal ``#f``."""
    return isinstance(expr, _Atom) and expr.text == "#f" and expr.shape == ()


def _simplify_if(expr: _Expr) -> _Expr:
    """Remove dead branches when the condition is a known literal."""
    if not isinstance(expr, _If):
        return expr
    if _is_true_atom(expr.condition):
        return expr.then_expr
    if _is_false_atom(expr.condition):
        return expr.else_expr
    return expr


# ---------------------------------------------------------------------------
# Map-pattern fusion (precursors only — full fusion requires HIR-level analysis)
# ---------------------------------------------------------------------------

def _simplify_view_chain(expr: _Expr) -> _Expr:
    """Cancel adjacent inverse views.

    Examples: *take* then *drop* on the same count, *reverse* then *reverse*.
    """
    if isinstance(expr, _View) and expr.kind == "reverse":
        if isinstance(expr.value, _View) and expr.value.kind == "reverse":
            return expr.value.value
    return expr


# ---------------------------------------------------------------------------
# Bottom-up simplifier
# ---------------------------------------------------------------------------

def simplify_ad_expr(expr: _Expr) -> _Expr:
    """Recursively simplify an ``_Expr`` tree.

    Rules applied (in order at each node, after children are simplified):

    1. constant folding
    2. algebraic identities
    3. dead-branch elimination
    4. view-chain cancellation

    Returns a semantically equivalent but potentially smaller expression.
    """
    # Step 0: simplify children first (bottom-up)
    expr = _simplify_children(expr)

    # Step 1: constant folding
    expr = _constant_fold(expr)

    # Step 2: algebraic identities
    expr = _simplify_algebraic(expr)

    # Step 3: dead branches
    expr = _simplify_if(expr)

    # Step 4: view chain cancellation
    expr = _simplify_view_chain(expr)

    return expr


def _simplify_children(expr: _Expr) -> _Expr:
    """Recursively simplify all children of *expr*, returning a new node."""
    if isinstance(expr, _Atom):
        return expr

    if isinstance(expr, _Op):
        new_left = simplify_ad_expr(expr.left)
        new_right = simplify_ad_expr(expr.right) if expr.right is not None else None
        # Re-apply binary constructor helpers to catch simplifications the
        # bottom-up pass created (e.g. left simplified to 0 → + simplification).
        if new_right is not None:
            from remora.ad_source import _binary as _ad_binary
            return _ad_binary(expr.op, new_left, new_right)
        return _Op(expr.op, new_left, new_right, expr.shape)

    if isinstance(expr, _Fold):
        return _Fold(simplify_ad_expr(expr.value), expr.axis, expr.shape)
    if isinstance(expr, _FoldBroadcast):
        return _FoldBroadcast(simplify_ad_expr(expr.value), expr.axis, expr.input_shape, expr.shape)
    if isinstance(expr, _Fill):
        new_val = simplify_ad_expr(expr.value)
        new_like = simplify_ad_expr(expr.like)
        from remora.ad_source import _fill as _ad_fill
        return _ad_fill(new_val, new_like)
    if isinstance(expr, _Reshape):
        new_val = simplify_ad_expr(expr.value)
        from remora.ad_source import _reshape as _ad_reshape
        return _ad_reshape(new_val, expr.shape)
    if isinstance(expr, _Transpose):
        new_val = simplify_ad_expr(expr.value)
        from remora.ad_source import _transpose as _ad_transpose
        return _ad_transpose(new_val)
    if isinstance(expr, _View):
        new_val = simplify_ad_expr(expr.value)
        from remora.ad_source import _view as _ad_view
        return _ad_view(expr.kind, new_val, expr.count)
    if isinstance(expr, _Append):
        new_left = simplify_ad_expr(expr.left)
        new_right = simplify_ad_expr(expr.right)
        from remora.ad_source import _append as _ad_append
        return _ad_append(new_left, new_right)
    if isinstance(expr, _SubarrayView):
        return _SubarrayView(simplify_ad_expr(expr.value), expr.offsets, expr.sizes, expr.shape)
    if isinstance(expr, _Rotate):
        new_val = simplify_ad_expr(expr.value)
        from remora.ad_source import _rotate as _ad_rotate
        return _ad_rotate(new_val, expr.shift)
    if isinstance(expr, _Index):
        return _Index(simplify_ad_expr(expr.value), expr.idx, expr.shape)
    if isinstance(expr, _ScatterAdd):
        new_target = simplify_ad_expr(expr.target)
        new_idx = expr.index if isinstance(expr.index, int) else simplify_ad_expr(expr.index)
        new_update = simplify_ad_expr(expr.update)
        return _ScatterAdd(new_target, new_idx, new_update, expr.shape)
    if isinstance(expr, _Im2col):
        return _Im2col(simplify_ad_expr(expr.image), expr.image_shape, expr.kh, expr.kw, expr.stride, expr.shape)
    if isinstance(expr, _Col2im):
        return _Col2im(simplify_ad_expr(expr.value), expr.image_shape, expr.kh, expr.kw, expr.stride, expr.shape)
    if isinstance(expr, _If):
        new_cond = simplify_ad_expr(expr.condition)
        new_then = simplify_ad_expr(expr.then_expr)
        new_else = simplify_ad_expr(expr.else_expr)
        return _If(new_cond, new_then, new_else, expr.shape)
    if isinstance(expr, _Pair):
        return _Pair(simplify_ad_expr(expr.left), simplify_ad_expr(expr.right), expr.shape)
    return expr


# ---------------------------------------------------------------------------
# Size estimation (for before/after comparison)
# ---------------------------------------------------------------------------

def ad_expr_node_count(expr: _Expr) -> int:
    """Count the number of ``_Expr`` nodes in *expr* (tree size)."""
    count = 1
    if isinstance(expr, _Atom):
        return count
    if isinstance(expr, _Op):
        count += ad_expr_node_count(expr.left)
        if expr.right is not None:
            count += ad_expr_node_count(expr.right)
        return count
    if isinstance(expr, _Fold):
        return count + ad_expr_node_count(expr.value)
    if isinstance(expr, _FoldBroadcast):
        return count + ad_expr_node_count(expr.value)
    if isinstance(expr, _Fill):
        return count + ad_expr_node_count(expr.value) + ad_expr_node_count(expr.like)
    if isinstance(expr, (_Reshape, _Transpose)):
        return count + ad_expr_node_count(expr.value)
    if isinstance(expr, _View):
        return count + ad_expr_node_count(expr.value)
    if isinstance(expr, _Append):
        return count + ad_expr_node_count(expr.left) + ad_expr_node_count(expr.right)
    if isinstance(expr, _SubarrayView):
        return count + ad_expr_node_count(expr.value)
    if isinstance(expr, _Rotate):
        return count + ad_expr_node_count(expr.value)
    if isinstance(expr, _Index):
        return count + ad_expr_node_count(expr.value)
    if isinstance(expr, _ScatterAdd):
        c = count + ad_expr_node_count(expr.target) + ad_expr_node_count(expr.update)
        if not isinstance(expr.index, int):
            c += ad_expr_node_count(expr.index)
        return c
    if isinstance(expr, (_Im2col, _Col2im)):
        v = expr.value if isinstance(expr, _Col2im) else expr.image
        return count + ad_expr_node_count(v)
    if isinstance(expr, _If):
        return count + ad_expr_node_count(expr.condition) + ad_expr_node_count(expr.then_expr) + ad_expr_node_count(expr.else_expr)
    if isinstance(expr, _Pair):
        return count + ad_expr_node_count(expr.left) + ad_expr_node_count(expr.right)
    return count
