"""Numerical finite-difference utilities for testing AD correctness.

Uses central finite differences at default step h = 1e-5.
All functions operate on numpy arrays.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

DEFAULT_H: float = 1e-5


def finite_difference_grad(
    f, x: NDArray[np.floating], h: float = DEFAULT_H
) -> NDArray[np.floating]:
    """Compute the gradient of scalar-valued f at x via central differences.

    f: NDArray[float] -> float
    x: NDArray[float]  (flattened internally)
    Returns: gradient as NDArray with same shape as x
    """
    x_flat = np.asarray(x, dtype=np.float64).ravel()
    grad = np.zeros_like(x_flat, dtype=np.float64)
    for i in range(len(x_flat)):
        x_plus = x_flat.copy()
        x_minus = x_flat.copy()
        x_plus[i] += h
        x_minus[i] -= h
        grad[i] = (f(x_plus.reshape(x.shape)) - f(x_minus.reshape(x.shape))) / (2.0 * h)
    return grad.reshape(np.asarray(x).shape)


def directional_derivative(
    f, x: NDArray[np.floating], direction: NDArray[np.floating], h: float = DEFAULT_H
) -> float:
    """Compute the directional derivative D_v f(x) via central difference."""
    d = np.asarray(direction, dtype=np.float64).ravel()
    norm = np.linalg.norm(d)
    if norm == 0:
        return 0.0
    v = d / norm
    x_flat = np.asarray(x, dtype=np.float64).ravel()
    f_plus = f((x_flat + h * v).reshape(x.shape))
    f_minus = f((x_flat - h * v).reshape(x.shape))
    return (f_plus - f_minus) / (2.0 * h * norm)


def grad_check(
    f,
    x: NDArray[np.floating],
    grad_x: NDArray[np.floating],
    *,
    rtol: float = 1e-4,
    atol: float = 1e-6,
    h: float = DEFAULT_H,
    label: str = "grad",
) -> None:
    """Assert that grad_x matches finite-difference gradient of f at x."""
    num_grad = finite_difference_grad(f, x, h=h)
    np.testing.assert_allclose(
        num_grad, grad_x, rtol=rtol, atol=atol,
        err_msg=f"{label}: finite-difference gradient mismatch"
    )
