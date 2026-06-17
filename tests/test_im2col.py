"""Tests for im2col and col2im primitives (Section 5)."""

import numpy as np
import pytest

from remora.ad import EvalTape, grad_via_tape, trace_expr, trace_via_tape_multi
from remora.ad_source import generate_gradient_function_source
from remora.ad_testing import finite_difference_grad, grad_check
from remora.compiler import compile_function_source, compile_gradient_functions_source
from remora.runtime import CPUFunctionExecutor, evaluate_source
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


def _im2col_source(size):
    patches = (size - 3 + 1) ** 2
    return f"""\
(define/pi ()
  (patches [image (Array Float {size} {size})] (Array Float {patches} 9))
  (im2col image [3 3] 1))
"""


def test_im2col_lowering_size_does_not_scale_with_image_area():
    artifacts = [
        compile_function_source(
            _im2col_source(size),
            "patches",
            (ArrayType(FLOAT, (StaticDim(size), StaticDim(size))),),
            include_prelude=False,
            syntax="lisp",
            verify=False,
        )
        for size in (4, 32)
    ]

    small, large = (artifact.mlir_text for artifact in artifacts)
    assert len(large) < len(small) * 1.5
    assert small.count("tensor.extract") == large.count("tensor.extract")
    assert large.count("tensor.extract") <= 2
    assert "tensor.insert" not in small
    assert "tensor.insert" not in large
    assert small.count("scf.for") == large.count("scf.for")
    assert large.count("scf.for") <= 4


@pytest.mark.parametrize(
    ("size", "kernel_size", "stride"),
    [(4, 3, 1), (5, 2, 2), (32, 3, 1)],
)
def test_compiled_im2col_and_col2im_preserve_overlap_counts(
    size, kernel_size, stride
):
    patches_per_axis = (size - kernel_size) // stride + 1
    patch_count = patches_per_axis ** 2
    patch_size = kernel_size ** 2
    source = f"""\
(define/pi ()
  (overlap-counts [image (Array Float {size} {size})] (Array Float {size} {size}))
  (col2im
    (im2col image [{kernel_size} {kernel_size}] {stride})
    [{size} {size}]
    [{kernel_size} {kernel_size}]
    {stride}))
"""
    param_types = (ArrayType(FLOAT, (StaticDim(size), StaticDim(size))),)
    artifact = CPUFunctionExecutor.compile_source(
        source,
        "overlap-counts",
        param_types,
        include_prelude=False,
        syntax="lisp",
    )
    try:
        result = CPUFunctionExecutor(artifact).execute(
            np.ones((size, size), dtype=np.float32)
        )
    finally:
        artifact.close()

    expected = _ref_col2im(
        np.ones((patch_count, patch_size), dtype=np.float32),
        (size, size),
        kernel_size,
        kernel_size,
        stride,
    )
    np.testing.assert_array_equal(result.value, expected.astype(np.float32))


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


def test_conv2d_gradient_source():
    """conv2d loss gradients verified against finite differences via tape."""
    from remora.lisp_reader import parse_lisp
    from remora.typechecker import TypeChecker

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

    tc = TypeChecker()
    tc.check_program(parse_lisp(src))
    function = tc._functions["conv2d-loss"]
    spec = tc._typed_top_level_function(function, FuncType(param_types, FLOAT), tc._build_prelude_env())

    rng = np.random.RandomState(77)
    image = rng.randn(4, 4)
    kernel = rng.randn(3, 3)
    bias = np.float64(0.5)

    pnames = [p[0] for p in spec.params]
    tape, indices = trace_via_tape_multi(
        spec.body, [np.asarray(v, dtype=np.float64) for v in [image, kernel, bias]], pnames,
    )
    adjs = tape.reverse()

    def loss_image(candidate):
        conv = _ref_conv2d(candidate, kernel, bias)
        return float(np.sum(conv * conv))

    def loss_kernel(candidate):
        conv = _ref_conv2d(image, candidate, bias)
        return float(np.sum(conv * conv))

    def loss_bias(candidate):
        conv = _ref_conv2d(image, kernel, candidate)
        return float(np.sum(conv * conv))

    grad_check(loss_image, image, adjs[indices[0]], rtol=1e-5, atol=1e-6, label="conv2d_grad_image")
    grad_check(loss_kernel, kernel, adjs[indices[1]], rtol=1e-5, atol=1e-6, label="conv2d_grad_kernel")
    grad_check(loss_bias, bias, adjs[indices[2]], rtol=1e-5, atol=1e-6, label="conv2d_grad_bias")


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
        rtol=1e-3, atol=1e-5,
    )
    np.testing.assert_allclose(
        interpreted[1], finite_difference_grad(loss_kernel, kernel),
        rtol=1e-3, atol=1e-5,
    )


# ── map + fold over im2col (cell-map lowering for inline lambdas) ────────────
#
# Regression for a lowering gap that previously produced empty MLIR (and thus a
# .so with no `_mlir_ciface_remora_call` entry point) whenever a `map` lambda
# containing a `fold` was applied to `im2col` output.  The cell-map lowerer only
# accepted lifted (named) functions; inline lambdas were rejected with
# "only lifted lambda cell maps lower to MLIR so far" and the error was
# swallowed by `compile_prepared_function`.


def _ref_patch_sums(image, kh, kw, stride):
    h, w = image.shape
    out_h = (h - kh) // stride + 1
    out_w = (w - kw) // stride + 1
    out = np.empty(out_h * out_w, dtype=np.float64)
    idx = 0
    for i in range(out_h):
        for j in range(out_w):
            out[idx] = image[
                i * stride : i * stride + kh, j * stride : j * stride + kw
            ].sum()
            idx += 1
    return out


@pytest.mark.parametrize(
    ("size", "kernel_size", "stride"),
    [(8, 3, 1), (32, 3, 1), (8, 3, 2), (7, 2, 2)],
)
def test_map_fold_over_im2col_lowers_and_matches_reference(
    size, kernel_size, stride
):
    patches_per_axis = (size - kernel_size) // stride + 1
    patch_count = patches_per_axis ** 2
    source = f"""\
(define/pi ()
  (f [image (Array Float {size} {size})] (Array Float {patch_count}))
  (map (lambda (p) (fold + 0.0 p)) (im2col image [{kernel_size} {kernel_size}] {stride})))
"""
    param_types = (ArrayType(FLOAT, (StaticDim(size), StaticDim(size))),)
    artifact = compile_function_source(
        source,
        "f",
        param_types,
        verify=True,
        include_prelude=False,
        syntax="lisp",
    )
    # The lowering gap produced 0-byte MLIR; guard against regression.
    assert len(artifact.mlir_text.strip()) > 0
    assert artifact.mlir_module is not None

    compiled = CPUFunctionExecutor.compile_source(
        source, "f", param_types, include_prelude=False, syntax="lisp"
    )
    try:
        rng = np.random.default_rng(0)
        image = rng.standard_normal((size, size)).astype(np.float32)
        result = CPUFunctionExecutor(compiled).execute(image)
        out = np.asarray(result.value, dtype=np.float32)
    finally:
        compiled.close()

    expected = _ref_patch_sums(image, kernel_size, kernel_size, stride).astype(
        np.float32
    )
    np.testing.assert_allclose(out, expected, rtol=1e-5, atol=1e-5)


def test_map_fold_over_im2col_value_and_grad_compiles():
    """The value-and-grad path for a scalar loss built from a map+fold over
    im2col must generate and compile (this was the Phase 2 detector blocker)."""
    from remora.ad_source import generate_value_and_grad_function_source

    # loss = sum over patches of (patch_sum + bias).  Two params, Float result.
    # The map lambda contains a fold over the cell plus a free scalar; AD
    # restructures the cell-map away so the value-and-grad lowers cleanly.
    source = """\
(define/pi ()
  (loss [image (Array Float 8 8) bias Float] Float)
  (fold + 0.0 (map (lambda (p) (+ (fold + 0.0 p) bias)) (im2col image [3 3] 1))))
"""
    param_types = (
        ArrayType(FLOAT, (StaticDim(8), StaticDim(8))),
        FLOAT,
    )
    grad_artifact = generate_value_and_grad_function_source(
        source,
        "loss",
        param_types,
        include_prelude=False,
        syntax="lisp",
    )
    compiled = compile_function_source(
        grad_artifact.source,
        grad_artifact.function_name,
        param_types,
        verify=True,
        include_prelude=False,
        syntax="lisp",
    )
    assert len(compiled.mlir_text.strip()) > 0
    assert compiled.mlir_module is not None

    # Execute and sanity-check the bias gradient (analytically = patch count).
    executable = CPUFunctionExecutor.compile_source(
        grad_artifact.source,
        grad_artifact.function_name,
        param_types,
        include_prelude=False,
        syntax="lisp",
    )
    try:
        rng = np.random.default_rng(1)
        image = rng.standard_normal((8, 8)).astype(np.float32)
        bias = np.float32(0.7)
        result = CPUFunctionExecutor(executable).execute(image, bias)
    finally:
        executable.close()

    def _flatten(value):
        if isinstance(value, (list, tuple)):
            flat = []
            for part in value:
                flat.extend(_flatten(part))
            return flat
        return [value]

    outputs = _flatten(result.value)
    grads = [np.asarray(o) for o in outputs if np.asarray(o).ndim > 0]
    scalars = [float(np.asarray(o)) for o in outputs if np.asarray(o).ndim == 0]
    assert len(grads) == 1 and grads[0].shape == (8, 8)
    assert scalars, "expected the bias gradient among the outputs"
    # 36 patches, d(loss)/d(bias) = 36.
    assert scalars[-1] == pytest.approx(36.0, abs=1e-4)


# ── dot-product cell-fold (fold + 0.0 (map * p k)) and the detector forward ──
#
# These exercise the cell-map fold whose per-element value is a producer map
# (the ``dot-patch`` pattern used by conv2d and the crater detector), the
# HIRLet inlining for named helpers, rank-polymorphic ``+``/``*`` with scalar
# variables, and the runtime's handling of multiple scalar (Float) parameters.


def _ref_dot_patch(image, kernel, stride=1):
    kh, kw = kernel.shape
    h, w = image.shape
    out_h = (h - kh) // stride + 1
    out_w = (w - kw) // stride + 1
    out = np.empty(out_h * out_w, dtype=np.float64)
    idx = 0
    for i in range(out_h):
        for j in range(out_w):
            patch = image[i * stride : i * stride + kh, j * stride : j * stride + kw]
            out[idx] = (patch * kernel).sum()
            idx += 1
    return out


@pytest.mark.parametrize("size", [8, 16])
def test_dot_product_cell_fold_over_im2col_compiles_and_matches(size):
    source = f"""\
(define/pi ()
  (f [image (Array Float {size} {size}) k (Array Float 3 3)] (Array Float {((size - 3) // 1 + 1) ** 2}))
  (map (lambda (p) (fold + 0.0 (map * p (ravel k)))) (im2col image [3 3] 1)))
"""
    param_types = (
        ArrayType(FLOAT, (StaticDim(size), StaticDim(size))),
        ArrayType(FLOAT, (StaticDim(3), StaticDim(3))),
    )
    artifact = compile_function_source(
        source, "f", param_types, verify=True,
        include_prelude=False, syntax="lisp",
    )
    assert len(artifact.mlir_text.strip()) > 0

    compiled = CPUFunctionExecutor.compile_source(
        source, "f", param_types, include_prelude=False, syntax="lisp"
    )
    try:
        rng = np.random.default_rng(0)
        image = rng.standard_normal((size, size)).astype(np.float32)
        kernel = rng.standard_normal((3, 3)).astype(np.float32)
        out = np.asarray(
            CPUFunctionExecutor(compiled).execute(image, kernel).value,
            dtype=np.float32,
        )
    finally:
        compiled.close()
    expected = _ref_dot_patch(image, kernel).astype(np.float32)
    np.testing.assert_allclose(out, expected, rtol=1e-5, atol=1e-5)


def test_classifier_conv2d_pattern_compiles_and_matches():
    """The classifier's exact conv2d source (named dot-patch + bias) must
    compile natively and match the reference.  This was always interpreted
    before the cell-fold / HIRLet / scalar-splat fixes."""
    source = """\
(define/pi ()
  (dot-patch [patch (Array Float 9) flat-k (Array Float 9)] Float)
  (fold + 0.0 (map * patch flat-k)))
(define/pi ()
  (conv2d [image (Array Float 8 8) kernel (Array Float 3 3) bias Float] (Array Float 36))
  (+ (map (lambda (p) (dot-patch p (ravel kernel))) (im2col image [3 3] 1)) bias))
"""
    param_types = (
        ArrayType(FLOAT, (StaticDim(8), StaticDim(8))),
        ArrayType(FLOAT, (StaticDim(3), StaticDim(3))),
        FLOAT,
    )
    artifact = compile_function_source(
        source, "conv2d", param_types, verify=True,
        include_prelude=False, syntax="lisp",
    )
    assert len(artifact.mlir_text.strip()) > 0

    compiled = CPUFunctionExecutor.compile_source(
        source, "conv2d", param_types, include_prelude=False, syntax="lisp"
    )
    try:
        rng = np.random.default_rng(1)
        image = rng.standard_normal((8, 8)).astype(np.float32)
        kernel = rng.standard_normal((3, 3)).astype(np.float32)
        bias = np.float32(0.3)
        out = np.asarray(
            CPUFunctionExecutor(compiled).execute(image, kernel, bias).value,
            dtype=np.float32,
        )
    finally:
        compiled.close()
    expected = (_ref_dot_patch(image, kernel) + bias).astype(np.float32)
    np.testing.assert_allclose(out, expected, rtol=1e-5, atol=1e-5)


def test_multiple_scalar_parameters_do_not_alias():
    """Regression for a runtime bug: with 2+ scalar (Float) parameters, the
    descriptor for each scalar captured a raw pointer to a temporary 0-d numpy
    array; the temporaries were GC'd and their addresses reused, so every
    scalar received the last one's value."""
    source = """\
(define/pi ()
  (f [arr (Array Float 4) w Float b Float] (Array Float 4))
  (+ (* w arr) b))
"""
    param_types = (ArrayType(FLOAT, (StaticDim(4),)), FLOAT, FLOAT)
    compiled = CPUFunctionExecutor.compile_source(
        source, "f", param_types, include_prelude=False, syntax="lisp"
    )
    try:
        rng = np.random.default_rng(2)
        arr = rng.standard_normal((4,)).astype(np.float32)
        w = np.float32(2.0)
        b = np.float32(0.5)
        out = np.asarray(
            CPUFunctionExecutor(compiled).execute(arr, w, b).value,
            dtype=np.float32,
        )
    finally:
        compiled.close()
    np.testing.assert_allclose(out, (w * arr + b).astype(np.float32), rtol=1e-6, atol=1e-6)


def test_detector_forward_compiles_and_matches_reference():
    """The Phase 2 detector forward (conv + relu + scalar head) compiles
    natively and matches the reference."""
    source = """\
(define/pi ()
  (relu [v Float] Float)
  (select (> v 0.0) v 0.0))
(define/pi ()
  (detect [k (Array Float 3 3) b1 Float w2 Float b2 Float image (Array Float 8 8)] (Array Float 36))
  (+ (* w2 (map relu (+ (map (lambda (p) (fold + 0.0 (map * p (ravel k)))) (im2col image [3 3] 1)) b1))) b2))
"""
    param_types = (
        ArrayType(FLOAT, (StaticDim(3), StaticDim(3))),
        FLOAT, FLOAT, FLOAT,
        ArrayType(FLOAT, (StaticDim(8), StaticDim(8))),
    )
    artifact = compile_function_source(
        source, "detect", param_types, verify=True,
        include_prelude=False, syntax="lisp",
    )
    assert len(artifact.mlir_text.strip()) > 0

    compiled = CPUFunctionExecutor.compile_source(
        source, "detect", param_types, include_prelude=False, syntax="lisp"
    )
    try:
        rng = np.random.default_rng(3)
        kernel = rng.standard_normal((3, 3)).astype(np.float32)
        b1 = np.float32(0.1)
        w2 = np.float32(2.0)
        b2 = np.float32(0.5)
        image = rng.standard_normal((8, 8)).astype(np.float32)
        out = np.asarray(
            CPUFunctionExecutor(compiled).execute(kernel, b1, w2, b2, image).value,
            dtype=np.float32,
        )
    finally:
        compiled.close()
    dots = _ref_dot_patch(image, kernel)
    expected = (w2 * np.maximum(dots + b1, 0.0) + b2).astype(np.float32)
    np.testing.assert_allclose(out, expected, rtol=1e-5, atol=1e-5)


def test_inlined_detector_loss_forward_compiles_and_matches():
    """The scalar detector loss (sum of squared logits) with the detect body
    inlined compiles natively and matches the NumPy reference.  Exercises the
    full chain: cell-fold dot-product over im2col, scalar splat for +b1, named
    relu map, scalar splat for *w2 and +b2, fold+map reduction to a scalar."""
    source = """\
(define/pi ()
  (relu [v Float] Float)
  (select (> v 0.0) v 0.0))
(define/pi ()
  (detect-loss [k (Array Float 3 3) b1 Float w2 Float b2 Float image (Array Float 8 8)] Float)
  (fold + 0.0 (map (lambda (l) (* l l)) (+ (* w2 (map relu (+ (map (lambda (p) (fold + 0.0 (map * p (ravel k)))) (im2col image [3 3] 1)) b1))) b2))))
"""
    param_types = (
        ArrayType(FLOAT, (StaticDim(3), StaticDim(3))),
        FLOAT, FLOAT, FLOAT,
        ArrayType(FLOAT, (StaticDim(8), StaticDim(8))),
    )
    artifact = compile_function_source(
        source, "detect-loss", param_types, verify=True,
        include_prelude=False, syntax="lisp",
    )
    assert len(artifact.mlir_text.strip()) > 0

    compiled = CPUFunctionExecutor.compile_source(
        source, "detect-loss", param_types, include_prelude=False, syntax="lisp"
    )
    try:
        rng = np.random.default_rng(4)
        kernel = rng.standard_normal((3, 3)).astype(np.float32)
        b1 = np.float32(0.1)
        w2 = np.float32(2.0)
        b2 = np.float32(0.5)
        image = rng.standard_normal((8, 8)).astype(np.float32)
        loss = float(np.asarray(
            CPUFunctionExecutor(compiled).execute(kernel, b1, w2, b2, image).value
        ))
    finally:
        compiled.close()
    dots = _ref_dot_patch(image, kernel)
    logits = (w2 * np.maximum(dots + b1, 0.0) + b2).astype(np.float32)
    expected = float((logits * logits).sum())
    assert loss == pytest.approx(expected, rel=1e-5, abs=1e-5)


@pytest.mark.parametrize(
    ("section_src", "ref_fn"),
    [
        ("(map (* w) arr)", lambda arr, w: w * arr),
        ("(map (+ b) arr)", lambda arr, b, w=None: arr + b),
        ("(map (- w) arr)", lambda arr, w: arr - w),  # right section: x - w
    ],
)
def test_variable_operator_section_in_map_compiles(section_src, ref_fn):
    """Regression for gap #8: operator sections with a *variable* operand
    (e.g. ``(map (* w) arr)``) must compile and match the reference.  The
    AD-generated value-and-grad emits these; previously only literal sections
    like ``(* 2.0)`` lowered."""
    uses_w = "w" in section_src
    if uses_w:
        source = f"""\
(define/pi ()
  (f [arr (Array Float 6) w Float] (Array Float 6))
  {section_src})
"""
        param_types = (ArrayType(FLOAT, (StaticDim(6),)), FLOAT)
    else:
        source = f"""\
(define/pi ()
  (f [arr (Array Float 6) b Float] (Array Float 6))
  {section_src})
"""
        param_types = (ArrayType(FLOAT, (StaticDim(6),)), FLOAT)

    compiled = CPUFunctionExecutor.compile_source(
        source, "f", param_types, include_prelude=False, syntax="lisp"
    )
    try:
        rng = np.random.default_rng(0)
        arr = rng.standard_normal((6,)).astype(np.float32)
        scalar = np.float32(3.0)
        out = np.asarray(
            CPUFunctionExecutor(compiled).execute(arr, scalar).value,
            dtype=np.float32,
        )
    finally:
        compiled.close()
    expected = (ref_fn(arr, scalar) if uses_w else ref_fn(arr, scalar, None)).astype(np.float32)
    np.testing.assert_allclose(out, expected, rtol=1e-6, atol=1e-6)
