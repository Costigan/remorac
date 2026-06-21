#!/usr/bin/env python
"""Mandelbrot set computed with Remora array functions.

Defines three compiled Remora kernels using ``remora.define()`` — one
for the real part, one for the imaginary part of z = z² + c, and one
for the squared magnitude — then drives the iteration from Python
and renders the result with matplotlib.

Run::

    uv run python examples/mandelbrot.py
"""

import numpy as np
import remora

W, H = 800, 600
N = W * H

step_real = remora.define(
    f"(define/pi () (step_real"
    f" [zr (Array Float {N}) zi (Array Float {N}) cr (Array Float {N})]"
    f" (Array Float {N}))"
    f" (+ (- (* zr zr) (* zi zi)) cr))",
)

step_imag = remora.define(
    f"(define/pi () (step_imag"
    f" [zr (Array Float {N}) zi (Array Float {N}) ci (Array Float {N})]"
    f" (Array Float {N}))"
    f" (+ (* 2.0 (* zr zi)) ci))",
)

mag_sq = remora.define(
    f"(define/pi () (mag_sq"
    f" [zr (Array Float {N}) zi (Array Float {N})]"
    f" (Array Float {N}))"
    f" (+ (* zr zr) (* zi zi)))",
)

re = np.linspace(-2.0, 1.0, W, dtype=np.float32)
im = np.linspace(-1.2, 1.2, H, dtype=np.float32)
cr = np.tile(re, H)
ci = np.repeat(im, W)

zr = np.zeros(N, dtype=np.float32)
zi = np.zeros(N, dtype=np.float32)
counts = np.zeros(N, dtype=np.int32)
escaped = np.zeros(N, dtype=bool)

MAX_ITER = 80
for i in range(MAX_ITER):
    zr_new = step_real(zr, zi, cr)
    zi_new = step_imag(zr, zi, ci)
    zr, zi = zr_new, zi_new
    just_escaped = (mag_sq(zr, zi) > 4.0) & ~escaped
    escaped |= just_escaped
    counts[~escaped] = i + 1
    zr = np.where(escaped, 0.0, zr).astype(np.float32)
    zi = np.where(escaped, 0.0, zi).astype(np.float32)

import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(10, 6))
ax.imshow(counts.reshape(H, W), cmap="inferno", extent=[-2, 1, -1.2, 1.2], aspect="auto")
ax.set_xlabel("Re")
ax.set_ylabel("Im")
ax.set_title("Mandelbrot set (Remora-compiled kernels)")
out = "examples/mandelbrot.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Saved {out}")
