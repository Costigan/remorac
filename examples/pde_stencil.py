"""PDE stencil: one-step heat equation via explicit finite differences.

Uses ``subarray`` views to express the 5-point Laplacian on a 2-D grid.
The Remora function returns the updated *interior* (size − 2 per axis);
Python composes it back into the full grid (Dirichlet boundaries).

Usage:
  python examples/pde_stencil.py --size 64 --steps 10
  python examples/pde_stencil.py --size 32 --alpha 0.2
"""

from __future__ import annotations

import argparse
from time import perf_counter

import numpy as np

from remora.runtime import CPUFunctionExecutor
from remora.types import ArrayType, FLOAT, StaticDim


def _build_source(size: int, alpha: float) -> str:
    interior = size - 2
    return f"""
(define/pi ()
  (heat_step [image (Array Float {size} {size})] (Array Float {interior} {interior}))
  (+ (subarray image [1 1] [{interior} {interior}])
     (* {alpha}
        (- (+ (subarray image [0 1] [{interior} {interior}])
              (subarray image [2 1] [{interior} {interior}])
              (subarray image [1 0] [{interior} {interior}])
              (subarray image [1 2] [{interior} {interior}]))
           (* 4.0 (subarray image [1 1] [{interior} {interior}]))))))
"""


def _ref_heat_step(grid: np.ndarray, alpha: float) -> np.ndarray:
    """NumPy reference: one explicit heat-equation step."""
    lap = (
        grid[:-2, 1:-1] + grid[2:, 1:-1] + grid[1:-1, :-2] + grid[1:-1, 2:]
        - 4.0 * grid[1:-1, 1:-1]
    )
    interior = grid[1:-1, 1:-1] + alpha * lap
    result = grid.copy()
    result[1:-1, 1:-1] = interior
    return result


def run_heat_steps(
    size: int,
    steps: int = 1,
    alpha: float = 0.2,
    *,
    verbose: bool = True,
) -> np.ndarray:
    interior = size - 2
    ptype = (ArrayType(FLOAT, (StaticDim(size), StaticDim(size))),)
    source = _build_source(size, alpha)

    compiled = CPUFunctionExecutor.compile_source(
        source, "heat_step", ptype, include_prelude=False, syntax="lisp",
    )
    try:
        exe = CPUFunctionExecutor(compiled)

        # Initial condition: Gaussian hot spot.
        yy, xx = np.ogrid[:size, :size]
        grid = np.exp(-((xx - size / 2) ** 2 + (yy - size / 2) ** 2) / (size / 4) ** 2).astype(np.float32)
        grid_ref = grid.copy()

        if verbose:
            t0 = perf_counter()

        for _ in range(steps):
            interior_arr = np.asarray(exe.execute(grid).value, dtype=np.float32)
            grid[1:-1, 1:-1] = interior_arr.reshape(interior, interior)
            grid_ref = _ref_heat_step(grid_ref, alpha)

        if verbose:
            elapsed = perf_counter() - t0
            print(f"Heat {size}x{size} {steps} steps: {elapsed:.3f}s "
                  f"max diff {float(np.abs(grid - grid_ref).max()):.2e}")

        np.testing.assert_array_almost_equal(grid, grid_ref, decimal=4,
                                              err_msg="Heat step mismatch")
    finally:
        compiled.close()

    return grid


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--size", type=int, default=32)
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--alpha", type=float, default=0.2)
    args = parser.parse_args()

    run_heat_steps(args.size, args.steps, args.alpha)


if __name__ == "__main__":
    main()
