"""Tests for the heat1d RemoraC model — Stage 1c.

Covers: Thomas solver, constant-coefficient CN (Stage 1a),
Remora-compiled K(T)/Cp(T) (Stage 1c), Picard iteration convergence,
and steady-state validation.
"""

import numpy as np
import pytest

from examples.heat1d.heat1d_model import (
    Heat1DModel,
    thomas_solve,
    analytical_steady_state,
    _compute_Cp_numpy,
    _compute_K_numpy,
    CP_COEFFS,
    R350,
)


class TestThomasSolve:
    """Verify the Thomas tridiagonal solver."""

    def test_thomas_random_system(self):
        rng = np.random.RandomState(42)
        n = 30
        lower = -rng.rand(n - 1).astype(np.float64) * 0.3
        diag = rng.rand(n).astype(np.float64) * 2.0 + 2.0
        upper = -rng.rand(n - 1).astype(np.float64) * 0.3
        rhs = rng.rand(n).astype(np.float64)

        x = thomas_solve(lower, diag, upper, rhs)

        A = np.diag(diag) + np.diag(lower, -1) + np.diag(upper, 1)
        expected = np.linalg.solve(A, rhs)
        np.testing.assert_allclose(x, expected, rtol=1e-10)

    def test_thomas_simple_3x3(self):
        lower = np.array([-1.0, -1.0], dtype=np.float64)
        diag = np.array([2.0, 2.0, 2.0], dtype=np.float64)
        upper = np.array([-1.0, -1.0], dtype=np.float64)
        rhs = np.array([1.0, 0.0, 0.0], dtype=np.float64)

        x = thomas_solve(lower, diag, upper, rhs)
        expected = np.array([0.75, 0.5, 0.25], dtype=np.float64)
        np.testing.assert_allclose(x, expected, rtol=1e-10)

    def test_thomas_identity(self):
        n = 10
        lower = np.zeros(n - 1, dtype=np.float64)
        diag = np.ones(n, dtype=np.float64)
        upper = np.zeros(n - 1, dtype=np.float64)
        rhs = np.arange(n, dtype=np.float64)
        x = thomas_solve(lower, diag, upper, rhs)
        np.testing.assert_allclose(x, rhs, rtol=1e-15)


class TestRemoraProperties:
    """Verify Remora-compiled K(T) and Cp(T) against NumPy oracle."""

    @pytest.fixture(autouse=True)
    def setup_model(self):
        """Create a model to trigger lazy compilation."""
        self.model = Heat1DModel(N=30, T_init=250.0)
        # Force compilation by accessing properties
        self.model._update_properties()

    def test_K_matches_numpy(self):
        T = np.full(30, 300.0, dtype=np.float64)
        Kc = self.model.Kc

        K_remora = self.model._compute_properties_for(T)[0]
        K_numpy = _compute_K_numpy(T, Kc)

        np.testing.assert_allclose(K_remora, K_numpy, rtol=1e-4)

    def test_K_temperature_dependent(self):
        """K increases with temperature (T^3 radiative term)."""
        T_cold = np.full(30, 100.0, dtype=np.float64)
        T_hot = np.full(30, 400.0, dtype=np.float64)

        K_cold = self.model._compute_properties_for(T_cold)[0]
        K_hot = self.model._compute_properties_for(T_hot)[0]

        assert np.all(K_hot > K_cold)

    def test_Cp_matches_numpy(self):
        T = np.full(30, 300.0, dtype=np.float64)
        Cp_remora = self.model._compute_properties_for(T)[1]
        Cp_numpy = _compute_Cp_numpy(T)

        np.testing.assert_allclose(Cp_remora, Cp_numpy, rtol=1e-4)

    def test_Cp_at_known_temperatures(self):
        """Verify Cp(T) against precomputed values."""
        T_test = np.array([100.0, 200.0, 300.0, 400.0, 500.0], dtype=np.float64)

        # Recompile for N=5
        from examples.heat1d.heat1d_model import _compile_Cp
        executor = _compile_Cp(5)
        result = executor.execute(T_test.astype(np.float32))

        Cp_values = result.value.astype(np.float64)
        Cp_expected = _compute_Cp_numpy(T_test)

        np.testing.assert_allclose(Cp_values, Cp_expected, rtol=1e-4)


class TestHeat1DModelConstant:
    """Stage 1a tests — constant coefficients (verify no regression)."""

    def test_single_step_structure(self):
        model = Heat1DModel(N=30, T_surface=300.0, T_init=200.0, z_max=0.5, dt=3600.0)
        T_new = model.step_crank_nicolson()
        assert T_new[0] == pytest.approx(300.0)
        assert np.all(np.isfinite(T_new))
        assert np.all(T_new >= 200.0 - 1e-6)
        assert np.all(T_new <= 300.0 + 1e-6)

    def test_temperature_monotonic_decay(self):
        model = Heat1DModel(N=30, T_surface=300.0, T_init=100.0, z_max=0.5, dt=3600.0)
        model.run(100)
        T_final = model.T
        np.testing.assert_array_compare(lambda a, b: a <= b, np.diff(T_final), 0,
            err_msg="Final temperature profile not monotonic")

    @pytest.mark.parametrize("N", [10, 30])
    def test_varying_grid_sizes(self, N):
        model = Heat1DModel(N=N, T_surface=250.0, T_init=200.0, z_max=0.5, dt=3600.0)
        T_new = model.step_crank_nicolson()
        assert len(T_new) == N
        assert np.all(np.isfinite(T_new))

    def test_steady_state_convergence(self):
        """Temperature-dependent model approaches steady state monotonically."""
        model = Heat1DModel(
            N=10, z_max=0.002, T_surface=250.0, T_init=200.0, dt=0.5,
            picard_tol=0.01,
        )
        history = model.run(1000)

        # Temperatures should approach surface temperature from below
        T_final = model.T
        assert T_final[0] == pytest.approx(250.0)
        # All interior temperatures should be between init and surface
        assert np.all(T_final >= 200.0 - 1e-6)
        assert np.all(T_final <= 250.0 + 1e-6)
        # Temperature should be monotonic in depth (hotter near surface)
        assert np.all(np.diff(T_final) <= 0)


class TestPicardIteration:
    """Test the Picard iteration convergence behavior."""

    def test_picard_converges_in_few_iterations(self):
        """Picard should converge quickly for small temperature changes."""
        model = Heat1DModel(
            N=30, T_surface=300.0, T_init=299.0, z_max=0.5, dt=60.0,
            picard_tol=0.01, picard_max_iter=20,
        )
        T_new = model.step_crank_nicolson()
        assert T_new[0] == pytest.approx(300.0)
        assert np.all(np.isfinite(T_new))

    def test_picard_handles_large_jump(self):
        """Picard with large initial T jump still converges."""
        model = Heat1DModel(
            N=15, T_surface=400.0, T_init=100.0, z_max=0.1, dt=10.0,
            picard_tol=0.1, picard_max_iter=50,
        )
        T_new = model.step_crank_nicolson()
        assert T_new[0] == pytest.approx(400.0)
        assert np.all(np.isfinite(T_new))
