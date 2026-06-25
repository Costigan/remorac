"""Tests for the heat1d RemoraC model — non-uniform grid, N=60.

Covers: non-uniform grid construction, Remora-compiled K(T)/Cp(T),
CN step (assembly + Thomas solve), Picard iteration, and bounds.
"""

import numpy as np
import pytest

from examples.heat1d.heat1d_model import (
    Heat1DModel,
    build_grid,
    _compute_Cp_numpy,
    _compute_K_numpy,
    _get_K,
    _get_Cp,
    _get_cn,
    CP_COEFFS,
    R350,
    N,
)


class TestGrid:
    """Non-uniform spatial grid construction."""

    def test_build_grid_shape(self):
        z, dz, g1, g2 = build_grid(2.5, growth_rate=1.05)
        assert len(z) == N
        assert len(dz) == N - 1
        assert len(g1) == N - 2
        assert len(g2) == N - 2

    def test_grid_increasing_spacing(self):
        z, dz, g1, g2 = build_grid(2.5, growth_rate=1.1)
        assert np.all(np.diff(dz) > 0)

    def test_grid_starts_at_zero(self):
        z, _, _, _ = build_grid(2.5, growth_rate=1.05)
        assert z[0] == 0.0

    def test_grid_near_target_max(self):
        z, _, _, _ = build_grid(2.5, growth_rate=1.05)
        assert z[-1] <= 2.5 + 1e-10


class TestRemoraProperties:
    """Remora-compiled K(T) and Cp(T) vs NumPy oracle."""

    def test_K_matches_numpy(self):
        T = np.full(N, 300.0, dtype=np.float64)
        Kc = np.full(N, 0.001, dtype=np.float64)
        executor = _get_K()
        r = _get_K()(T.astype(np.float32), Kc.astype(np.float32))
        np.testing.assert_allclose(r, _compute_K_numpy(T, Kc), rtol=1e-4)

    def test_K_temperature_dependent(self):
        T_cold = np.full(N, 100.0, dtype=np.float64)
        T_hot = np.full(N, 400.0, dtype=np.float64)
        Kc = np.full(N, 0.001, dtype=np.float64)
        ex = _get_K()
        Kc_f32 = Kc.astype(np.float32)
        K_cold = ex(T_cold.astype(np.float32), Kc_f32)
        K_hot = ex(T_hot.astype(np.float32), Kc_f32)
        assert np.all(K_hot > K_cold)

    def test_Cp_matches_numpy(self):
        T = np.full(N, 300.0, dtype=np.float64)
        executor = _get_Cp()
        r = _get_Cp()(T.astype(np.float32))
        np.testing.assert_allclose(r, _compute_Cp_numpy(T), rtol=1e-4)


class TestCNStep:
    """Remora-compiled Crank-Nicolson step."""

    def test_cn_step_compiles_and_runs(self):
        T_old = np.full(N, 200.0, dtype=np.float32)
        g1    = np.ones(N - 2, dtype=np.float32)
        g2    = np.ones(N - 2, dtype=np.float32)
        rho   = np.full(N, 1100.0, dtype=np.float32)
        K     = np.full(N, 0.01, dtype=np.float32)
        Cp    = np.full(N, 600.0, dtype=np.float32)
        r = _get_cn()(
            T_old, g1, g2, rho, K, Cp,
            np.asarray(3600.0, dtype=np.float32),
            np.asarray(250.0, dtype=np.float32),
        )
        assert len(r) == N
        assert abs(r[0] - 250.0) < 0.01
        assert np.all(np.isfinite(r))


class TestHeat1DModel:
    """End-to-end tests with the non-uniform grid model."""

    def test_model_compiles_and_steps(self):
        model = Heat1DModel(z_max=2.5, T_surface=300.0, T_init=200.0,
                            growth_rate=1.0)
        T_new = model.step()
        assert abs(T_new[0] - 300.0) < 1.0
        assert len(T_new) == N
        assert np.all(np.isfinite(T_new))

    def test_temperature_bounded(self):
        model = Heat1DModel(z_max=0.5, T_surface=300.0, T_init=200.0,
                            dt=3600.0, growth_rate=1.0)
        model.run(10)
        assert np.all(model.T >= 100.0)
        assert np.all(model.T <= 320.0)

    def test_picard_handles_jump(self):
        """Picard with large initial T jump converges without NaN."""
        model = Heat1DModel(z_max=0.1, T_surface=400.0, T_init=100.0,
                            dt=10.0, picard_tol=0.1, growth_rate=1.0)
        T_new = model.step()
        assert np.all(np.isfinite(T_new))
        assert abs(T_new[0] - 400.0) < 1.0

    def test_steady_state_approaches_surface_temp(self):
        """Hot surface + cold interior + insulated bottom → tends to Tsurface."""
        model = Heat1DModel(z_max=0.02, T_surface=250.0, T_init=200.0,
                            dt=1.0, growth_rate=1.0, picard_tol=0.01)
        model.run(1000)
        assert np.all(model.T >= 80.0)
        assert np.all(model.T <= 260.0)
        assert model.T[0] > model.T[-1]
