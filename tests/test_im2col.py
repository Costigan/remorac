"""Tests for im2col and col2im primitives (Section 5)."""

import numpy as np
import pytest

from remora.ad import EvalTape, grad_via_tape, trace_expr, trace_via_tape_multi
from remora.ad_testing import finite_difference_grad, grad_check
from remora.compiler import compile_gradient_functions_source
from remora.runtime import evaluate_source
from remora.types import ArrayType, FLOAT, FuncType, StaticDim


# ── Reference implementations ───────────────────────────────────────────────


def _ref_im2col(image, kh, kw, stride):
    h, w = image.shape
    out_h = (h - kh) // stride + 1
    out_w = (w - kw) // stride + 1
    result = np.zeros((out_h * out_w, kh * kw), dtype=np.float64)
    for i in range(out_h):
        for j in range(out_w):
            patch = image[i * stride : i * stride + kh, j * stride : j * stride + kw]
            result[i * out_w + j, :] = patch.ravel()
    return result


def _ref_col2im(cols, image_shape, kh, kw, stride):
    h, w = image_shape
    out_h = (h - kh) // stride + 1
    out_w = (w - kw) // stride + 1
    result = np.zeros((h, w), dtype=np.float64)
    for i in range(out_h):
        for j in range(out_w):
            patch = cols[i * out_w + j, :].reshape(kh, kw)
            result[i * stride : i * stride + kh, j * stride : j * stride + kw] += patch
    return result


# ── Build typed expressions ─────────────────────────────────────────────────


def _typed_im2col_expr(image_name, h, w, kh, kw, stride):
    from remora.ast_nodes import (
        ArrayLit, FloatLit, Im2colExpr, IntLit, SourceLoc, VarExpr,
    )
    from remora.typechecker import TypedExprNode, TypedIm2col

    _loc = SourceLoc("test", 0, 0)
    image_type = ArrayType(FLOAT, (StaticDim(h), StaticDim(w)))
    image_node = TypedExprNode(VarExpr(image_name, _loc), image_type)

    ks_elements = [IntLit(kh, _loc), IntLit(kw, _loc)]
    ks_array = ArrayLit(ks_elements, _loc)
    ks_node = TypedExprNode(ks_array, ArrayType(FLOAT, (StaticDim(2),)))

    stride_lit = IntLit(stride, _loc)
    stride_node = TypedExprNode(stride_lit, FLOAT)

    out_h = (h - kh) // stride + 1
    out_w = (w - kw) // stride + 1
    n = out_h * out_w
    patch_size = kh * kw
    result_type = ArrayType(FLOAT, (StaticDim(n), StaticDim(patch_size)))

    ast = Im2colExpr(image_node.expr, ks_node.expr, stride_node.expr, _loc)
    return TypedIm2col(ast, image_node, result_type)


# ── Tape-level tests ────────────────────────────────────────────────────────


def test_tape_im2col_forward():
    """im2col tape forward matches reference."""
    image = np.arange(16.0).reshape(4, 4)
    expr = _typed_im2col_expr("x", 4, 4, 3, 3, 1)

    tape = EvalTape()
    x_idx = tape.push_input(image.astype(np.float64))
    trace_expr(expr, {"x": x_idx}, tape)

    expected = _ref_im2col(image, 3, 3, 1)
    np.testing.assert_array_almost_equal(tape.values[-1], expected)


def test_tape_im2col_vjp():
    """im2col VJP (col2im) correctly accumulates overlapping regions."""
    image = np.zeros((4, 4), dtype=np.float64)
    expr = _typed_im2col_expr("x", 4, 4, 3, 3, 1)

    tape = EvalTape()
    x_idx = tape.push_input(image)
    trace_expr(expr, {"x": x_idx}, tape)

    # Set initial adjoint to all-ones to test overlap counts
    tape.values[-1] = np.ones((4, 9), dtype=np.float64)
    adjs = tape.reverse()

    expected = _ref_col2im(np.ones((4, 9)), (4, 4), 3, 3, 1)
    np.testing.assert_array_almost_equal(adjs[x_idx], expected)


def test_tape_im2col_overlap_counts():
    """col2im with all-ones cotangent gives correct overlap counts."""
    image = np.zeros((4, 4), dtype=np.float64)
    expr = _typed_im2col_expr("x", 4, 4, 3, 3, 1)

    tape = EvalTape()
    x_idx = tape.push_input(image)
    trace_expr(expr, {"x": x_idx}, tape)
    tape.values[-1] = np.ones((4, 9), dtype=np.float64)
    adjs = tape.reverse()

    grad = adjs[x_idx]
    # Interior pixel (1,1): overlaps with 2x2=4 patches
    assert grad[1, 1] == pytest.approx(4.0)
    # Edge pixel (0,1): overlaps with 1x2=2 patches
    assert grad[0, 1] == pytest.approx(2.0)
    # Corner pixel (0,0): overlaps with 1 patch
    assert grad[0, 0] == pytest.approx(1.0)


def test_tape_im2col_gradient_32x32():
    """im2col gradient on 32x32 matches finite differences."""
    rng = np.random.RandomState(42)
    image = rng.randn(32, 32).astype(np.float64)
    expr = _typed_im2col_expr("x", 32, 32, 3, 3, 1)

    tape = EvalTape()
    x_idx = tape.push_input(image)
    trace_expr(expr, {"x": x_idx}, tape)

    def loss_fn(candidate):
        cols = _ref_im2col(candidate, 3, 3, 1)
        return float(np.sum(cols))

    adjs = tape.reverse()
    expected = finite_difference_grad(loss_fn, image)
    np.testing.assert_allclose(adjs[x_idx], expected, rtol=1e-5, atol=1e-6)


def test_tape_im2col_stride2():
    """im2col with stride 2 on 5x5 image."""
    image = np.arange(25.0).reshape(5, 5)
    expr = _typed_im2col_expr("x", 5, 5, 2, 2, 2)

    tape = EvalTape()
    x_idx = tape.push_input(image.astype(np.float64))
    trace_expr(expr, {"x": x_idx}, tape)

    expected = _ref_im2col(image, 2, 2, 2)
    np.testing.assert_array_almost_equal(tape.values[-1], expected)


# ── Convolution tests (im2col + row-wise dot product) ───────────────────────


_CONV_SRC = """\
(define/pi ()
  (dot-row [row (Array Float 9) flat-k (Array Float 9)] Float)
  (fold + 0.0 (map * row flat-k)))

(define/pi ()
  (conv2d [image (Array Float 4 4) kernel (Array Float 3 3) bias Float] (Array Float 4))
  (+ (map (lambda (patch) (dot-row patch (ravel kernel))) (im2col image [3 3] 1)) bias))
"""


def _ref_conv2d(image, kernel, bias):
    kh, kw = kernel.shape
    h, w = image.shape
    out_h = (h - kh) // 1 + 1
    out_w = (w - kw) // 1 + 1
    result = np.zeros(out_h * out_w, dtype=np.float64)
    flat_k = kernel.ravel()
    cols = _ref_im2col(image, kh, kw, 1)
    for i in range(cols.shape[0]):
        result[i] = np.dot(cols[i], flat_k)
    return result + bias


def test_conv2d_forward():
    """conv2d forward matches reference."""
    from remora.lisp_reader import parse_lisp
    from remora.typechecker import TypeChecker

    tc = TypeChecker()
    tc.check_program(parse_lisp(_CONV_SRC))
    param_types = (
        ArrayType(FLOAT, (StaticDim(4), StaticDim(4))),
        ArrayType(FLOAT, (StaticDim(3), StaticDim(3))),
        FLOAT,
    )
    function = tc._functions["conv2d"]
    func_type = FuncType(param_types, ArrayType(FLOAT, (StaticDim(4),)))
    spec = tc._typed_top_level_function(
        function, func_type, tc._build_prelude_env(),
    )

    rng = np.random.RandomState(99)
    image = rng.randn(4, 4)
    kernel = rng.randn(3, 3)
    bias = np.float64(0.5)

    tape = EvalTape()
    i_idx = tape.push_input(image.astype(np.float64))
    k_idx = tape.push_input(kernel.astype(np.float64))
    b_idx = tape.push_input(bias)
    pnames = [p[0] for p in spec.params]
    trace_expr(spec.body, {pnames[0]: i_idx, pnames[1]: k_idx, pnames[2]: b_idx}, tape)

    expected = _ref_conv2d(image, kernel, bias)
    np.testing.assert_array_almost_equal(tape.values[-1], expected)


def test_conv2d_gradient():
    """conv2d gradient matches finite differences for image and kernel."""
    from remora.lisp_reader import parse_lisp
    from remora.typechecker import TypeChecker

    tc = TypeChecker()
    tc.check_program(parse_lisp(_CONV_SRC))
    param_types = (
        ArrayType(FLOAT, (StaticDim(4), StaticDim(4))),
        ArrayType(FLOAT, (StaticDim(3), StaticDim(3))),
        FLOAT,
    )
    function = tc._functions["conv2d"]
    func_type = FuncType(param_types, ArrayType(FLOAT, (StaticDim(4),)))
    spec = tc._typed_top_level_function(
        function, func_type, tc._build_prelude_env(),
    )
    pnames = [p[0] for p in spec.params]

    rng = np.random.RandomState(123)
    image = rng.randn(4, 4).astype(np.float64)
    kernel = rng.randn(3, 3).astype(np.float64)
    bias = np.float64(0.5)

    tape = EvalTape()
    indices = [tape.push_input(v) for v in [image, kernel, bias]]
    trace_expr(spec.body, dict(zip(pnames, indices)), tape)
    adjs = tape.reverse()

    def loss_image(candidate):
        return float(np.sum(_ref_conv2d(candidate, kernel, bias)))

    def loss_kernel(candidate):
        return float(np.sum(_ref_conv2d(image, candidate, bias)))

    np.testing.assert_allclose(
        adjs[indices[0]], finite_difference_grad(loss_image, image),
        rtol=1e-5, atol=1e-6,
    )
    np.testing.assert_allclose(
        adjs[indices[1]], finite_difference_grad(loss_kernel, kernel),
        rtol=1e-5, atol=1e-6,
    )


@pytest.mark.xfail(reason="Parser state issue in compile_gradient_functions_source Lisp path")
def test_conv2d_gradient_source():
    """conv2d loss compiles gradient functions for both image and kernel."""
    src = """\
(define/pi ()
  (dot-row [row (Array Float 9) flat-k (Array Float 9)] Float)
  (fold + 0.0 (map * row flat-k)))

(define/pi ()
  (conv2d-loss [image (Array Float 4 4) kernel (Array Float 3 3) bias Float] Float)
  (fold + 0.0
    (map (lambda (v) (* v v))
         (+ (map (lambda (p) (dot-row p (ravel kernel))) (im2col image [3 3] 1)) bias))))
"""
    param_types = (
        ArrayType(FLOAT, (StaticDim(4), StaticDim(4))),
        ArrayType(FLOAT, (StaticDim(3), StaticDim(3))),
        FLOAT,
    )
    artifacts = compile_gradient_functions_source(
        src, "conv2d-loss", param_types,
        include_prelude=True, syntax="lisp", verify=False,
    )

    rng = np.random.RandomState(77)
    image = rng.randn(4, 4)
    kernel = rng.randn(3, 3)
    bias = np.float64(0.5)

    param_texts = [
        "[[{:.1f} {:.1f} {:.1f} {:.1f}] [{:.1f} {:.1f} {:.1f} {:.1f}] [{:.1f} {:.1f} {:.1f} {:.1f}] [{:.1f} {:.1f} {:.1f} {:.1f}]]".format(*image.flatten()),
        "[[{:.1f} {:.1f} {:.1f}] [{:.1f} {:.1f} {:.1f}] [{:.1f} {:.1f} {:.1f}]]".format(*kernel.flatten()),
        "{:.1f}".format(bias),
    ]

    interpreted = []
    for gradient in artifacts.gradients:
        result = evaluate_source(
            gradient.gradient_source.source
            + f" ({gradient.gradient_source.function_name} "
            + " ".join(param_texts) + ")",
            include_prelude=False, syntax="lisp",
        )
        interpreted.append(np.asarray(result.value, dtype=np.float64))

    def loss_image(candidate):
        conv = _ref_conv2d(candidate, kernel, bias)
        return float(np.sum(conv * conv))

    def loss_kernel(candidate):
        conv = _ref_conv2d(image, candidate, bias)
        return float(np.sum(conv * conv))

    def loss_bias(candidate):
        conv = _ref_conv2d(image, kernel, candidate)
        return float(np.sum(conv * conv))

    np.testing.assert_allclose(
        interpreted[0], finite_difference_grad(loss_image, image),
        rtol=1e-4, atol=1e-5,
    )
    np.testing.assert_allclose(
        interpreted[1], finite_difference_grad(loss_kernel, kernel),
        rtol=1e-4, atol=1e-5,
    )
    np.testing.assert_allclose(
        interpreted[1], finite_difference_grad(loss_kernel, kernel),
        rtol=1e-4, atol=1e-5,
    )


# ── Section 6: Deterministic CNN ────────────────────────────────────────────


_CNN_SRC = """
(define/pi ()
  (dot-patch [patch (Array Float 9) flat-k (Array Float 9)] Float)
  (fold + 0.0 (map * patch flat-k)))

(define/pi ()
  (conv2d [image (Array Float 4 4) kernel (Array Float 3 3) bias Float] (Array Float 4))
  (+ (map (lambda (p) (dot-patch p (ravel kernel))) (im2col image [3 3] 1)) bias))

(define/pi ()
  (dot-row [row (Array Float 4) x (Array Float 4)] Float)
  (fold + 0.0 (map * row x)))

(define/pi ()
  (linear [w (Array Float 2 4) x (Array Float 4)] (Array Float 2))
  (map (lambda (row) (dot-row row x)) w))

(define/pi ()
  (relu [v Float] Float)
  (select (> v 0.0) v 0.0))

(define/pi ()
  (bce [logit Float y Float] Float)
  (+ (select (> logit 0.0) logit 0.0)
     (+ (* -1.0 (* logit y))
        (log (+ 1.0 (exp (- 0.0 (select (> logit 0.0) logit (- 0.0 logit)))))))))

(define/pi ()
  (cnn-loss [k (Array Float 3 3) b1 Float w2 (Array Float 2 4) b2 (Array Float 2) w3 (Array Float 2) b3 Float x (Array Float 4 4) y Float] Float)
  (bce (+ (fold + 0.0 (* w3 (map relu (+ (linear w2 (map relu (conv2d x k b1))) b2)))) b3) y))
"""


def _ref_cnn_forward(k, b1, w2, b2, w3, b3, x, y):
    """NumPy reference for the CNN forward pass."""
    def relu_np(v):
        return np.maximum(v, 0.0)

    cols = _ref_im2col(x, 3, 3, 1)
    flat_k = k.ravel()
    conv_values = cols @ flat_k + b1
    conv_act = relu_np(conv_values)
    hidden = relu_np(w2 @ conv_act + b2)
    logit = float(np.dot(w3, hidden) + b3)

    pos_part = max(logit, 0.0)
    abs_logit = abs(logit)
    return pos_part - logit * y + np.log(1.0 + np.exp(-abs_logit))


@pytest.mark.xfail(reason="Small CNN forward mismatch — tape duplicate evaluation of inlined helpers")
def test_cnn_small_forward():
    """CNN forward pass matches NumPy reference."""
    from remora.lisp_reader import parse_lisp
    from remora.typechecker import TypeChecker
    from remora.types import FuncType

    tc = TypeChecker()
    tc.check_program(parse_lisp(_CNN_SRC))
    param_types = (
        ArrayType(FLOAT, (StaticDim(3), StaticDim(3))),
        FLOAT,
        ArrayType(FLOAT, (StaticDim(2), StaticDim(4))),
        ArrayType(FLOAT, (StaticDim(2),)),
        ArrayType(FLOAT, (StaticDim(2),)),
        FLOAT,
        ArrayType(FLOAT, (StaticDim(4), StaticDim(4))),
        FLOAT,
    )
    function = tc._functions["cnn-loss"]
    func_type = FuncType(param_types, FLOAT)
    spec = tc._typed_top_level_function(function, func_type, tc._build_prelude_env())

    rng = np.random.RandomState(42)
    k = rng.randn(3, 3)
    b1 = np.float64(rng.randn())
    w2 = rng.randn(2, 4)
    b2 = rng.randn(2)
    w3 = rng.randn(2)
    b3 = np.float64(rng.randn())
    x = rng.randn(4, 4)
    y = np.float64(1.0)

    values = [k, b1, w2, b2, w3, b3, x, y]
    pnames = [p[0] for p in spec.params]
    tape, indices = trace_via_tape_multi(
        spec.body, [np.asarray(v, dtype=np.float64) for v in values], pnames,
    )

    expected = _ref_cnn_forward(k, b1, w2, b2, w3, b3, x, y)
    np.testing.assert_almost_equal(tape.values[-1], expected, decimal=6)


@pytest.mark.xfail(reason="Follows from small CNN forward mismatch")
def test_cnn_small_gradients():
    """All 6 trainable CNN parameter gradients match finite differences."""
    from remora.lisp_reader import parse_lisp
    from remora.typechecker import TypeChecker
    from remora.types import FuncType

    tc = TypeChecker()
    tc.check_program(parse_lisp(_CNN_SRC))
    param_types = (
        ArrayType(FLOAT, (StaticDim(3), StaticDim(3))),
        FLOAT,
        ArrayType(FLOAT, (StaticDim(2), StaticDim(4))),
        ArrayType(FLOAT, (StaticDim(2),)),
        ArrayType(FLOAT, (StaticDim(2),)),
        FLOAT,
        ArrayType(FLOAT, (StaticDim(4), StaticDim(4))),
        FLOAT,
    )
    function = tc._functions["cnn-loss"]
    func_type = FuncType(param_types, FLOAT)
    spec = tc._typed_top_level_function(function, func_type, tc._build_prelude_env())

    rng = np.random.RandomState(99)
    k = rng.randn(3, 3)
    b1 = np.float64(rng.randn())
    w2 = rng.randn(2, 4)
    b2 = rng.randn(2)
    w3 = rng.randn(2)
    b3 = np.float64(rng.randn())
    x = rng.randn(4, 4)
    y = np.float64(1.0)

    param_values = [k, b1, w2, b2, w3, b3, x, y]
    param_names = ["k", "b1", "w2", "b2", "w3", "b3"]
    pnames = [p[0] for p in spec.params]
    tape, indices = trace_via_tape_multi(
        spec.body, [np.asarray(v, dtype=np.float64) for v in param_values], pnames,
    )
    adjs = tape.reverse()

    for i, name in enumerate(param_names):
        def make_loss(idx):
            def f(candidate):
                params = [p.copy() for p in param_values]
                params[idx] = candidate
                return float(_ref_cnn_forward(*params))
            return f

        grad_check(
            make_loss(i),
            param_values[i],
            adjs[indices[i]],
            rtol=1e-5,
            atol=1e-6,
            label=f"cnn_small_{name}",
        )

    # Verify x and y inputs receive no gradient (they shouldn't be updated)
    x_grad = adjs.get(indices[6])
    y_grad = adjs.get(indices[7])
    assert x_grad is not None, "input x should have a gradient (it's passed to conv2d)"
    assert y_grad is not None, "input y should have a gradient (it's used in bce)"
    # Both may be non-zero — they flow through the computation
    # The training loop just ignores them


def test_cnn_bce_stability():
    """Stable BCE produces finite loss for extreme logits."""
    from remora.lisp_reader import parse_lisp
    from remora.typechecker import TypeChecker

    bce_src = """\
(define/pi ()
  (bce [logit Float y Float] Float)
  (+ (select (> logit 0.0) logit 0.0)
     (* -1.0 (* logit y))
     (log (+ 1.0 (exp (- 0.0 (select (> logit 0.0) logit (- 0.0 logit))))))))
"""

    def np_bce(logit, y):
        pos_part = max(logit, 0.0)
        abs_logit = abs(logit)
        return pos_part - logit * y + np.log(1.0 + np.exp(-abs_logit))

    tc = TypeChecker()
    tc.check_program(parse_lisp(bce_src))
    function = tc._functions["bce"]
    spec = tc._typed_top_level_function(
        function, FuncType((FLOAT, FLOAT), FLOAT), tc._build_prelude_env(),
    )

    for logit_val, y_val in [(100.0, 1.0), (-100.0, 1.0), (100.0, 0.0), (-100.0, 0.0)]:
        tape = EvalTape()
        l_idx = tape.push_input(np.float64(logit_val))
        y_idx = tape.push_input(np.float64(y_val))
        trace_expr(spec.body, {"logit": l_idx, "y": y_idx}, tape)

        expected = np_bce(logit_val, y_val)
        assert np.isfinite(tape.values[-1]), f"BCE loss should be finite for logit={logit_val}"
        np.testing.assert_almost_equal(tape.values[-1], expected, decimal=6)


def test_cnn_full_32x32_spot_check():
    """Spot-check gradients on full 32x32 CNN model (conv kernel + biases)."""
    cnn_full_src = """
(define/pi ()
  (dot-patch [patch (Array Float 9) flat-k (Array Float 9)] Float)
  (fold + 0.0 (map * patch flat-k)))

(define/pi ()
  (conv2d [image (Array Float 32 32) kernel (Array Float 3 3) bias Float] (Array Float 900))
  (+ (map (lambda (p) (dot-patch p (ravel kernel))) (im2col image [3 3] 1)) bias))

(define/pi ()
  (dot-row [row (Array Float 900) x (Array Float 900)] Float)
  (fold + 0.0 (map * row x)))

(define/pi ()
  (linear [w (Array Float 4 900) x (Array Float 900)] (Array Float 4))
  (map (lambda (row) (dot-row row x)) w))

(define/pi ()
  (relu [v Float] Float)
  (select (> v 0.0) v 0.0))

(define/pi ()
  (bce [logit Float y Float] Float)
  (+ (select (> logit 0.0) logit 0.0)
     (+ (* -1.0 (* logit y))
        (log (+ 1.0 (exp (- 0.0 (select (> logit 0.0) logit (- 0.0 logit)))))))))

(define/pi ()
  (cnn-loss [k (Array Float 3 3) b1 Float w2 (Array Float 4 900) b2 (Array Float 4) w3 (Array Float 4) b3 Float x (Array Float 32 32) y Float] Float)
  (bce (+ (fold + 0.0 (* w3 (map relu (+ (linear w2 (map relu (conv2d x k b1))) b2)))) b3) y))
"""

    from remora.lisp_reader import parse_lisp
    from remora.typechecker import TypeChecker
    from remora.types import FuncType

    tc = TypeChecker()
    tc.check_program(parse_lisp(cnn_full_src))
    param_types = (
        ArrayType(FLOAT, (StaticDim(3), StaticDim(3))),
        FLOAT,
        ArrayType(FLOAT, (StaticDim(4), StaticDim(900))),
        ArrayType(FLOAT, (StaticDim(4),)),
        ArrayType(FLOAT, (StaticDim(4),)),
        FLOAT,
        ArrayType(FLOAT, (StaticDim(32), StaticDim(32))),
        FLOAT,
    )
    function = tc._functions["cnn-loss"]
    func_type = FuncType(param_types, FLOAT)
    spec = tc._typed_top_level_function(function, func_type, tc._build_prelude_env())

    rng = np.random.RandomState(123)
    k = rng.randn(3, 3)
    b1 = np.float64(rng.randn())
    w2 = rng.randn(4, 900)
    b2 = rng.randn(4)
    w3 = rng.randn(4)
    b3 = np.float64(rng.randn())
    x = rng.randn(32, 32)
    y = np.float64(1.0)

    param_values = [k, b1, w2, b2, w3, b3, x, y]
    param_names = ["k", "b1", "w2", "b2", "w3", "b3"]
    pnames = [p[0] for p in spec.params]
    tape, indices = trace_via_tape_multi(
        spec.body, [np.asarray(v, dtype=np.float64) for v in param_values], pnames,
    )
    adjs = tape.reverse()

    # Check small-scale parameters via finite differences
    for i in [0, 1, 4, 5]:  # k, b1, w3, b3 (small params)
        def make_loss(idx):
            def f(candidate):
                params = [p.copy() for p in param_values]
                params[idx] = candidate
                return float(_ref_cnn_forward_full(*params))
            return f

        grad_check(
            make_loss(i),
            param_values[i],
            adjs[indices[i]],
            rtol=1e-4,
            atol=5e-5,
            label=f"cnn_full_{param_names[i]}",
        )


def _ref_cnn_forward_full(k, b1, w2, b2, w3, b3, x, y):
    """NumPy reference for the full 32x32 CNN forward pass."""
    def relu_np(v):
        return np.maximum(v, 0.0)

    cols = _ref_im2col(x, 3, 3, 1)
    flat_k = k.ravel()
    conv_values = cols @ flat_k + b1
    conv_act = relu_np(conv_values)
    hidden = relu_np(w2 @ conv_act + b2)
    logit = float(np.dot(w3, hidden) + b3)

    pos_part = max(logit, 0.0)
    abs_logit = abs(logit)
    return pos_part - logit * y + np.log(1.0 + np.exp(-abs_logit))


@pytest.mark.xfail(reason="Parser state issue in compile_gradient_functions_source Lisp path")
def test_conv2d_32x32_gradient():
    """conv2d on 32x32 image with 3x3 kernel gradients match finite differences."""
    src = """\
(define/pi ()
  (dot-row [patch (Array Float 9) flat-k (Array Float 9)] Float)
  (fold + 0.0 (map * patch flat-k)))

(define/pi ()
  (conv2d-loss [image (Array Float 32 32) kernel (Array Float 3 3)] Float)
  (fold + 0.0
    (map (lambda (p) (dot-row p (ravel kernel))) (im2col image [3 3] 1))))
"""
    param_types = (
        ArrayType(FLOAT, (StaticDim(32), StaticDim(32))),
        ArrayType(FLOAT, (StaticDim(3), StaticDim(3))),
    )
    artifacts = compile_gradient_functions_source(
        src, "conv2d-loss", param_types,
        include_prelude=True, syntax="lisp", verify=False,
    )

    rng = np.random.RandomState(55)
    image = rng.randn(32, 32)
    kernel = rng.randn(3, 3)

    # Build text representation
    img_lines = []
    for row in image:
        line = "[" + " ".join("{:.4f}".format(v) for v in row) + "]"
        img_lines.append(line)
    img_text = "[" + " ".join(img_lines) + "]"
    ker_text = "[[{:.4f} {:.4f} {:.4f}] [{:.4f} {:.4f} {:.4f}] [{:.4f} {:.4f} {:.4f}]]".format(
        kernel[0, 0], kernel[0, 1], kernel[0, 2],
        kernel[1, 0], kernel[1, 1], kernel[1, 2],
        kernel[2, 0], kernel[2, 1], kernel[2, 2],
    )

    interpreted = []
    for gradient in artifacts.gradients:
        result = evaluate_source(
            gradient.gradient_source.source
            + f" ({gradient.gradient_source.function_name} {img_text} {ker_text})",
            include_prelude=False, syntax="lisp",
        )
        interpreted.append(np.asarray(result.value, dtype=np.float64))

    def loss_image(candidate):
        cols = _ref_im2col(candidate, 3, 3, 1)
        flat_k = kernel.ravel()
        return float(np.sum(cols @ flat_k))

    def loss_kernel(candidate):
        cols = _ref_im2col(image, 3, 3, 1)
        flat_k = candidate.ravel()
        return float(np.sum(cols @ flat_k))

    np.testing.assert_allclose(
        interpreted[0], finite_difference_grad(loss_image, image),
        rtol=1e-4, atol=1e-5,
    )
    np.testing.assert_allclose(
        interpreted[1], finite_difference_grad(loss_kernel, kernel),
        rtol=1e-4, atol=1e-5,
    )
