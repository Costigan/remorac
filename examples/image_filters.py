"""Non-AD dense image kernels: Sobel edge magnitude, thresholding, and blur.

These examples isolate the compiler's dense non-AD path from neural-net
complexity — pure convolution, elementwise arithmetic, and comparisons.
Compiled filters are verified against NumPy references.

Usage:
  python examples/image_filters.py --size 128 --filter sobel
  python examples/image_filters.py --size 64  --filter blur
  python examples/image_filters.py                 # run all
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from time import perf_counter
from typing import Callable

import numpy as np

from remora.runtime import CPUFunctionExecutor
from remora.types import ArrayType, FLOAT, StaticDim

# ── Remora source ───────────────────────────────────────────────────────────

_IMAGE_FILTERS_LISP_SRC = """
(define/pi ()
  (relu [v Float] Float)
  (select (> v 0.0) v 0.0))

;; ── Sobel edge magnitude (squared, no sqrt) ────────────────────────────────
;; valid convolution → flat [W-2, H-2] output.
(define/pi ()
  (sobel [image (Array Float 32 32) kx (Array Float 3 3) ky (Array Float 3 3)] (Array Float 900))
  (+ (* (map (lambda (p) (fold + 0.0 (map * p (ravel kx)))) (im2col image [3 3] 1))
        (map (lambda (p) (fold + 0.0 (map * p (ravel kx)))) (im2col image [3 3] 1)))
     (* (map (lambda (p) (fold + 0.0 (map * p (ravel ky)))) (im2col image [3 3] 1))
        (map (lambda (p) (fold + 0.0 (map * p (ravel ky)))) (im2col image [3 3] 1)))))

;; ── Binary threshold (elementwise, same shape) ─────────────────────────────
(define/pi ()
  (threshold [image (Array Float 32 32) t Float] (Array Float 32 32))
  (map (lambda (v) (select (> v t) 1.0 0.0)) image))

;; ── Box blur (3×3 average, valid convolution → flat [W-2, H-2]) ───────────
(define/pi ()
  (blur [image (Array Float 32 32) kb (Array Float 3 3)] (Array Float 900))
  (map (lambda (p) (fold + 0.0 (map * p (ravel kb)))) (im2col image [3 3] 1)))
"""


# ── Compiled filter runners ─────────────────────────────────────────────────


def _compile_and_run(
    function_name: str,
    param_types: tuple,
    inputs: tuple,
) -> np.ndarray:
    """Compile *function_name* from the shared source and run it once."""
    compiled = CPUFunctionExecutor.compile_source(
        _IMAGE_FILTERS_LISP_SRC,
        function_name,
        param_types,
        include_prelude=False,
        syntax="lisp",
    )
    try:
        result = CPUFunctionExecutor(compiled).execute(*inputs)
        return np.asarray(result.value, dtype=np.float32)
    finally:
        compiled.close()


_SOBEL_X = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)
_SOBEL_Y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float32)
_BOX_KERNEL = np.full((3, 3), 1.0 / 9.0, dtype=np.float32)


def _conv2d_valid(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """2-D valid convolution (no padding, stride 1)."""
    kh, kw = kernel.shape
    h, w = image.shape
    out_h, out_w = h - kh + 1, w - kw + 1
    out = np.empty((out_h, out_w), dtype=np.float32)
    for i in range(out_h):
        for j in range(out_w):
            out[i, j] = (image[i:i+kh, j:j+kw] * kernel).sum()
    return out


def _ref_sobel_sq(image: np.ndarray) -> np.ndarray:
    """Squared Sobel edge magnitude via valid convolution (no sqrt)."""
    gx = _conv2d_valid(image, _SOBEL_X)
    gy = _conv2d_valid(image, _SOBEL_Y)
    return (gx * gx + gy * gy).ravel().astype(np.float32)


def _ref_threshold(image: np.ndarray, t: float) -> np.ndarray:
    return (image > t).astype(np.float32)


def _ref_blur(image: np.ndarray) -> np.ndarray:
    return _conv2d_valid(image, _BOX_KERNEL).ravel().astype(np.float32)


def run_sobel(size: int, *, verbose: bool = True) -> np.ndarray:
    image = np.random.default_rng(0).standard_normal((size, size)).astype(np.float32)
    ptype = (
        ArrayType(FLOAT, (StaticDim(size), StaticDim(size))),
        ArrayType(FLOAT, (StaticDim(3), StaticDim(3))),
        ArrayType(FLOAT, (StaticDim(3), StaticDim(3))),
    )
    if verbose:
        print(f"Sobel {size}×{size} …", end=" ", flush=True)
    t0 = perf_counter()
    out = _compile_and_run("sobel", ptype, (image, _SOBEL_X, _SOBEL_Y))
    elapsed = perf_counter() - t0
    patches = (size - 2) ** 2
    ref = _ref_sobel_sq(image)
    np.testing.assert_array_almost_equal(out[:patches].reshape(-1), ref, decimal=4)
    if verbose:
        print(f"OK ({elapsed:.3f}s)")
    return out


def run_threshold(size: int, t: float = 0.0, *, verbose: bool = True) -> np.ndarray:
    image = np.random.default_rng(1).standard_normal((size, size)).astype(np.float32)
    ptype = (
        ArrayType(FLOAT, (StaticDim(size), StaticDim(size))),
        FLOAT,
    )
    if verbose:
        print(f"Threshold {size}×{size} …", end=" ", flush=True)
    t0 = perf_counter()
    out = _compile_and_run("threshold", ptype, (image, np.float32(t)))
    elapsed = perf_counter() - t0
    ref = _ref_threshold(image, t)
    np.testing.assert_array_equal(out, ref)
    if verbose:
        print(f"OK ({elapsed:.3f}s)")
    return out


def run_blur(size: int, *, verbose: bool = True) -> np.ndarray:
    image = np.random.default_rng(2).standard_normal((size, size)).astype(np.float32)
    ptype = (
        ArrayType(FLOAT, (StaticDim(size), StaticDim(size))),
        ArrayType(FLOAT, (StaticDim(3), StaticDim(3))),
    )
    if verbose:
        print(f"Blur {size}×{size} …", end=" ", flush=True)
    t0 = perf_counter()
    out = _compile_and_run("blur", ptype, (image, _BOX_KERNEL))
    elapsed = perf_counter() - t0
    patches = (size - 2) ** 2
    ref = _ref_blur(image)
    np.testing.assert_array_almost_equal(out[:patches].reshape(-1), ref, decimal=5)
    if verbose:
        print(f"OK ({elapsed:.3f}s)")
    return out


# ── CLI ──────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--size", type=int, default=32)
    parser.add_argument("--filter", choices=["sobel", "threshold", "blur", "all"], default="all")
    parser.add_argument("--threshold", type=float, default=0.0)
    args = parser.parse_args()

    if args.filter in ("sobel", "all"):
        run_sobel(args.size)
    if args.filter in ("threshold", "all"):
        run_threshold(args.size, args.threshold)
    if args.filter in ("blur", "all"):
        run_blur(args.size)


if __name__ == "__main__":
    main()
