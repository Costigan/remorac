"""Tests for compiled non-AD image kernels (Sobel, threshold, blur)."""

import numpy as np
import pytest

from examples.image_filters import (
    _compile_and_run,
    _ref_blur,
    _ref_sobel_sq,
    _ref_threshold,
    run_blur,
    run_sobel,
    run_threshold,
)
from remora.types import ArrayType, FLOAT, StaticDim


def _ptype(size, *extra):
    t = (ArrayType(FLOAT, (StaticDim(size), StaticDim(size))),)
    return t + extra


def test_sobel_matches_reference():
    rng = np.random.default_rng(0)
    image = rng.standard_normal((32, 32)).astype(np.float32)
    out = _compile_and_run(
        "sobel",
        _ptype(32, ArrayType(FLOAT, (StaticDim(3), StaticDim(3))),
               ArrayType(FLOAT, (StaticDim(3), StaticDim(3)))),
        (image, examples.image_filters._SOBEL_X, examples.image_filters._SOBEL_Y),
    )
    patches = (32 - 2) ** 2
    ref = _ref_sobel_sq(image)
    np.testing.assert_array_almost_equal(out[:patches].reshape(-1), ref, decimal=4)


@pytest.mark.parametrize("t", [0.0, -0.5, 0.5])
def test_threshold_matches_reference(t: float):
    rng = np.random.default_rng(0)
    image = rng.standard_normal((32, 32)).astype(np.float32)
    out = _compile_and_run(
        "threshold",
        _ptype(32, FLOAT),
        (image, np.float32(t)),
    )
    ref = _ref_threshold(image, t)
    np.testing.assert_array_equal(out, ref)


def test_blur_matches_reference():
    rng = np.random.default_rng(0)
    image = rng.standard_normal((32, 32)).astype(np.float32)
    out = _compile_and_run(
        "blur",
        _ptype(32, ArrayType(FLOAT, (StaticDim(3), StaticDim(3)))),
        (image, examples.image_filters._BOX_KERNEL),
    )
    patches = (32 - 2) ** 2
    ref = _ref_blur(image)
    np.testing.assert_array_almost_equal(out[:patches].reshape(-1), ref, decimal=5)


def test_blur_of_uniform_region_is_unchanged():
    image = np.ones((32, 32), dtype=np.float32)
    out = _compile_and_run(
        "blur",
        _ptype(32, ArrayType(FLOAT, (StaticDim(3), StaticDim(3)))),
        (image, examples.image_filters._BOX_KERNEL),
    )
    interior = out.reshape(32 - 2, 32 - 2)
    np.testing.assert_array_almost_equal(interior, np.ones((30, 30), dtype=np.float32), decimal=5)


def test_threshold_binary_output():
    rng = np.random.default_rng(42)
    image = rng.standard_normal((32, 32)).astype(np.float32)
    out = _compile_and_run(
        "threshold",
        _ptype(32, FLOAT),
        (image, np.float32(0.0)),
    )
    unique = np.unique(out)
    assert len(unique) <= 2
    for val in unique:
        assert val in (0.0, 1.0), f"unexpected threshold output value {val}"


# Register the examples module for import, needed above
import examples.image_filters
