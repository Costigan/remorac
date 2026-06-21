#!/usr/bin/env python
"""Remora functions called from Python.

Defines three compiled Remora functions using ``remora.define()`` and
calls them with NumPy arrays.

Run::

    uv run python examples/embedded_remora.py
"""

import numpy as np
import remora

scale = remora.define(
    "(define/pi () (scale [xs (Array Float 4)] (Array Float 4))"
    "  (map (* 2.0) xs))",
)

dot = remora.define(
    "(define/pi () (dot [a (Array Float 4) b (Array Float 4)] Float)"
    "  (fold + 0.0 (map * a b)))",
)

negate = remora.define(
    "(define/pi () (negate [xs (Array Float 4)] (Array Float 4))"
    "  (map (* -1.0) xs))",
)

xs = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
ys = np.array([4.0, 3.0, 2.0, 1.0], dtype=np.float32)

print("Remora functions called from Python")
print("=" * 40)
print(f"xs        = {xs}")
print(f"scale(xs) = {scale(xs)}")
print(f"negate(xs)= {negate(xs)}")
print(f"dot(xs,ys)= {dot(xs, ys)}")
