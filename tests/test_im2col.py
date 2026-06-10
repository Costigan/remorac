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
        interpreted[2], finite_difference_grad(loss_bias, bias),
        rtol=1e-4, atol=1e-5,
    )


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
