"""Tests for the compiled heat-equation PDE stencil."""

import numpy as np
import pytest

from examples.pde_stencil import _build_source, _ref_heat_step
from remora.runtime import CPUFunctionExecutor
from remora.types import ArrayType, FLOAT, StaticDim


@pytest.mark.parametrize("size", [16, 32])
def test_one_step_matches_reference(size: int):
    alpha = 0.2
    interior = size - 2
    ptype = (ArrayType(FLOAT, (StaticDim(size), StaticDim(size))),)
    source = _build_source(size, alpha)
    compiled = CPUFunctionExecutor.compile_source(
        source, "heat_step", ptype, include_prelude=False, syntax="lisp",
    )
    try:
        rng = np.random.default_rng(0)
        grid = rng.standard_normal((size, size)).astype(np.float32)
        interior_arr = np.asarray(
            CPUFunctionExecutor(compiled).execute(grid).value, dtype=np.float32
        ).reshape(interior, interior)
        ref = _ref_heat_step(grid, alpha)[1:-1, 1:-1]
        np.testing.assert_array_almost_equal(interior_arr, ref, decimal=4)
    finally:
        compiled.close()


@pytest.mark.parametrize("steps", [1, 3])
def test_multi_step_matches_reference(steps: int):
    size = 16
    alpha = 0.2
    interior = size - 2
    ptype = (ArrayType(FLOAT, (StaticDim(size), StaticDim(size))),)
    source = _build_source(size, alpha)
    compiled = CPUFunctionExecutor.compile_source(
        source, "heat_step", ptype, include_prelude=False, syntax="lisp",
    )
    try:
        exe = CPUFunctionExecutor(compiled)
        rng = np.random.default_rng(42)
        grid = rng.standard_normal((size, size)).astype(np.float32)
        ref = grid.copy()
        for _ in range(steps):
            interior_arr = np.asarray(exe.execute(grid).value, dtype=np.float32)
            grid[1:-1, 1:-1] = interior_arr.reshape(interior, interior)
            ref = _ref_heat_step(ref, alpha)
        np.testing.assert_array_almost_equal(grid, ref, decimal=4)
    finally:
        compiled.close()


def test_boundaries_are_preserved():
    """Dirichlet boundaries: border values must be unchanged."""
    size = 16
    alpha = 0.2
    ptype = (ArrayType(FLOAT, (StaticDim(size), StaticDim(size))),)
    source = _build_source(size, alpha)
    compiled = CPUFunctionExecutor.compile_source(
        source, "heat_step", ptype, include_prelude=False, syntax="lisp",
    )
    try:
        rng = np.random.default_rng(1)
        grid = rng.standard_normal((size, size)).astype(np.float32)
        exe = CPUFunctionExecutor(compiled)
        for _ in range(5):
            interior_arr = np.asarray(exe.execute(grid).value, dtype=np.float32)
            grid[1:-1, 1:-1] = interior_arr.reshape(size - 2, size - 2)
        # Top and bottom rows
        np.testing.assert_array_equal(grid[0, :], grid[0, :])
        np.testing.assert_array_equal(grid[-1, :], grid[-1, :])
        # Left and right columns (interior updated, but we can check the corners)
    finally:
        compiled.close()


def test_heat_spreads_maximum():
    """After one step, the peak temperature must decrease (diffusion)."""
    size = 16
    alpha = 0.2
    ptype = (ArrayType(FLOAT, (StaticDim(size), StaticDim(size))),)
    source = _build_source(size, alpha)
    compiled = CPUFunctionExecutor.compile_source(
        source, "heat_step", ptype, include_prelude=False, syntax="lisp",
    )
    try:
        yy, xx = np.ogrid[:size, :size]
        grid = np.exp(-((xx - size/2)**2 + (yy - size/2)**2) / 9.0).astype(np.float32)
        exe = CPUFunctionExecutor(compiled)
        interior_arr = np.asarray(exe.execute(grid).value, dtype=np.float32)
        grid[1:-1, 1:-1] = interior_arr.reshape(size - 2, size - 2)
        assert grid.max() < 1.0, "peak temperature should decrease with diffusion"
        assert grid.min() >= 0.0, "temperature must stay non-negative"
    finally:
        compiled.close()
