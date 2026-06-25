"""Fourier-matrix solver for the 1D periodic heat equation.

Solves the diurnal steady state in the frequency domain via thermal
transmission matrices (thermal quadrupole formalism, Pipes 1957 /
Maillet et al. 2000).  Nonlinear surface radiation is handled via
Newton iteration on a circulant admittance matrix.  Thermal rectification
(solid-state greenhouse effect) is computed from the exact time-domain
product <k(T)·∂T/∂z> and iterated to convergence.

Reference: Hayne et al. (2017), JGR Planets 122, 2371–2400.
"""

from __future__ import annotations

import math
import numpy as np
from numpy.fft import rfft, irfft


# ═══════════════════════════════════════════════════════════════════════════
# Physical constants (shared with heat1d_model.py)
# ═══════════════════════════════════════════════════════════════════════════

S0 = 1361.0
A0 = 0.12
ALBEDO_A = 0.06
ALBEDO_B = 0.25
EMISSIVITY = 0.95
SIGMA = 5.670367e-8
LUNAR_DAY = 29.53059 * 24.0 * 3600.0
Kcs = 0.00074
Kcd = 0.0034
CHI = 2.7
R350 = CHI / 350.0**3
rhos = 1100.0
rhod = 1800.0
H_SCALE = 0.07
Q_GEO = 0.018
CP_COEFFS = (-3.6125, 2.7431, 2.3616e-3, -1.2340e-5, 8.9093e-9)


def _cp(T: np.ndarray) -> np.ndarray:
    c0, c1, c2, c3, c4 = CP_COEFFS
    return c0 + c1*T + c2*T**2 + c3*T**3 + c4*T**4


def _k_cond(T: np.ndarray, kc: np.ndarray) -> np.ndarray:
    return kc * (1.0 + R350 * T**3)


def _rho(z: np.ndarray) -> np.ndarray:
    return rhod - (rhod - rhos) * np.exp(-z / H_SCALE)


def _kc(z: np.ndarray) -> np.ndarray:
    rho_z = _rho(z)
    return Kcs + (Kcd - Kcs) * (rho_z - rhos) / (rhod - rhos)


# ═══════════════════════════════════════════════════════════════════════════
# Solar insolation (circular orbit approximation)
# ═══════════════════════════════════════════════════════════════════════════

def _albedo(incidence: np.ndarray) -> np.ndarray:
    x = incidence / (math.pi / 4.0)
    y = incidence / (math.pi / 2.0)
    return A0 + ALBEDO_A * x**3 + ALBEDO_B * y**8


def compute_flux_series(
    lat: float, n_steps: int, dec: float = 0.0, r_au: float = 1.0,
) -> np.ndarray:
    """Absorbed solar flux over one diurnal cycle."""
    t = np.linspace(0, LUNAR_DAY, n_steps, endpoint=False)
    hour_angle = 2.0 * math.pi * t / LUNAR_DAY

    cos_z = np.sin(lat) * np.sin(dec) + np.cos(lat) * np.cos(dec) * np.cos(hour_angle)
    cos_z = np.maximum(cos_z, 0.0)

    incidence = np.where(cos_z > 0, np.arccos(cos_z), 0.0)
    A_i = _albedo(incidence)
    f = (1.0 - A_i) / (1.0 - A0)

    return f * S0 * (1.0 - A0) * (r_au ** -2) * cos_z


# ═══════════════════════════════════════════════════════════════════════════
# Mean surface temperature (Newton on radiative balance)
# ═══════════════════════════════════════════════════════════════════════════

def _mean_surface_temp(F_mean: float, J_geo: float, emissivity: float) -> float:
    T_subsolar = ((F_mean * math.pi + J_geo) / (emissivity * SIGMA)) ** 0.25
    T = T_subsolar / math.sqrt(2.0)
    for _ in range(50):
        f = emissivity * SIGMA * T**4 - (F_mean + J_geo)
        fp = 4.0 * emissivity * SIGMA * T**3
        dt = -f / fp
        T += dt
        if abs(dt) < 1e-8:
            break
    return T


# ═══════════════════════════════════════════════════════════════════════════
# Equilibrium profile (RK4 integration of dT/dz = (J_geo - J_pump) / k(T))
# ═══════════════════════════════════════════════════════════════════════════

def _equilibrium_profile(
    T_surf: float, z: np.ndarray, kc: np.ndarray,
    J_geo: float, J_pump: np.ndarray | None = None,
    k_eff: np.ndarray | None = None,
) -> np.ndarray:
    """Integrate dT/dz = J_geo / k(T) from surface downward via RK4."""
    if J_pump is None:
        J_pump = np.zeros_like(z)
    N = len(z)
    T = np.empty(N)
    T[0] = T_surf

    for i in range(N - 1):
        dz_i = z[i + 1] - z[i]
        J = J_geo - 0.5 * (J_pump[i] + J_pump[i + 1])

        def f(T_val: float) -> float:
            k_val = kc[i] * (1.0 + R350 * T_val**3) if k_eff is None else k_eff[i]
            return J / max(k_val, 1e-30)

        k1 = f(T[i])
        k2 = f(T[i] + 0.5 * dz_i * k1)
        k3 = f(T[i] + 0.5 * dz_i * k2)
        k4 = f(T[i] + dz_i * k3)
        T[i + 1] = T[i] + dz_i / 6.0 * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

    return T


# ═══════════════════════════════════════════════════════════════════════════
# Thermal transmission matrices (quadrupole formalism)
# ═══════════════════════════════════════════════════════════════════════════

def _layer_matrices(
    omega: np.ndarray, dz: np.ndarray, k_eq: np.ndarray, kappa_eq: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute surface impedance and depth transfer functions.

    Parameters
    ----------
    omega : (M,) angular frequencies [rad/s]
    dz : (L,) layer thicknesses [m]
    k_eq : (L+1,) conductivity at layer midpoints
    kappa_eq : (L+1,) diffusivity at layer midpoints

    Returns
    -------
    Z : (M,) complex surface impedance
    depth_ratio : (M, L+1) complex temperature ratio T(z) / T_surf
    flux_ratio : (M, L+1) complex flux ratio q(z) / T_surf
    """
    M = len(omega)
    L = len(dz)
    Nz = L + 1

    gamma = np.sqrt(1j * omega[:, None] / kappa_eq[None, :])  # (M, Nz)
    k = k_eq[None, :]  # (1, Nz)

    # Layer matrices P_{00}, P_{10} at each depth
    P00 = np.ones((M, Nz), dtype=complex)
    P10 = np.zeros((M, Nz), dtype=complex)

    # Cumulate from bottom upward
    for j in range(L - 1, -1, -1):
        gd = gamma[:, j] * dz[j]
        # Guard against overflow
        safe = np.where(gd.real < 20.0)
        ch = np.ones(M, dtype=complex)
        sh = np.zeros(M, dtype=complex)
        idx = safe[0]
        ch[idx] = np.cosh(gd[idx])
        sh[idx] = np.sinh(gd[idx])

        # Asymptotic for large |gd|
        big = np.where(gd.real >= 20.0)[0]
        if len(big) > 0:
            ch[big] = 0.5 * np.exp(gd[big])
            sh[big] = 0.5 * np.exp(gd[big])

        kg = k[:, j] * gamma[:, j]
        kg_safe = np.where(np.abs(kg) < 1e-30, 1e-30, kg)

        P00_new = ch * P00[:, j] + sh / kg_safe * P10[:, j]
        P10_new = kg * sh * P00[:, j] + ch * P10[:, j]

        P00[:, j] = P00_new
        P10[:, j] = P10_new
        if j > 0:
            P00[:, j - 1] = P00_new
            P10[:, j - 1] = P10_new

    # Surface impedance
    Z = np.where(np.abs(P10[:, 0]) > 1e-30, P00[:, 0] / P10[:, 0], 1e10)

    # Depth ratios
    depth_ratio = P00 / np.where(np.abs(P00[:, 0:1]) > 1e-30, P00[:, 0:1], 1.0)
    flux_ratio = P10 / np.where(np.abs(P00[:, 0:1]) > 1e-30, P00[:, 0:1], 1.0)

    return Z, depth_ratio, flux_ratio


# ═══════════════════════════════════════════════════════════════════════════
# Circulant admittance matrix
# ═══════════════════════════════════════════════════════════════════════════

def _circulant_from_admittance(Y: np.ndarray, N: int) -> np.ndarray:
    """Build circulant matrix from frequency-domain admittance spectrum."""
    Y_full = np.zeros(N // 2 + 1, dtype=complex)
    Y_full[1:len(Y) + 1] = Y
    c0 = irfft(Y_full, n=N)
    C = np.empty((N, N))
    for i in range(N):
        C[i] = np.roll(c0, i)
    return C


# ═══════════════════════════════════════════════════════════════════════════
# Newton iteration with circulant admittance
# ═══════════════════════════════════════════════════════════════════════════

def _newton_surface_circulant(
    T_surf: np.ndarray,
    F_abs: np.ndarray,
    J_geo: float,
    Z: np.ndarray,
    emissivity: float,
    max_iter: int = 40,
    tol: float = 0.5,
) -> np.ndarray:
    """Newton iteration for T_surf(t) with circulant conductive response."""
    N = len(T_surf)
    # Admittance at AC frequencies only (skip DC)
    omega = 2.0 * math.pi * np.arange(1, N // 2 + 1) / LUNAR_DAY
    Y_ac = np.where(np.abs(Z) > 1e-30, 1.0 / Z, 0.0)

    C = _circulant_from_admittance(Y_ac, N)

    T = T_surf.copy()
    for _ in range(max_iter):
        rad = emissivity * SIGMA * T**4
        drad = 4.0 * emissivity * SIGMA * T**3

        residual = F_abs + J_geo - rad - C @ T
        J_diag = np.diag(drad) + C

        try:
            dT = np.linalg.solve(J_diag, residual)
        except np.linalg.LinAlgError:
            dT = residual / (drad + np.diag(C).mean())

        T = np.maximum(T + dT, 2.0)
        if np.max(np.abs(dT)) < tol:
            break

    return T


# ═══════════════════════════════════════════════════════════════════════════
# Thermal rectification (pumping)
# ═══════════════════════════════════════════════════════════════════════════

def _compute_rectification(
    T_surf: np.ndarray,
    depth_ratio: np.ndarray,
    flux_ratio: np.ndarray,
    T_eq: np.ndarray,
    kc: np.ndarray,
    k_eq: np.ndarray,
    dz: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute net downward rectification flux and effective conductivity.

    J_pump(z) = <k(T) · dT/dz>  (time-averaged conductive flux)
    """
    N = len(T_surf)
    Nz = len(T_eq)

    # Reconstruct T(z, t) at all depths
    T_hat_surf = rfft(T_surf)
    T_all = np.empty((N, Nz))
    for j in range(Nz):
        spectrum = np.zeros(N // 2 + 1, dtype=complex)
        spectrum[0] = T_eq[j] * N  # DC
        if j < depth_ratio.shape[1]:
            spectrum[1:len(T_hat_surf)] = T_hat_surf[1:] * depth_ratio[:len(T_hat_surf) - 1, j]
        T_all[:, j] = irfft(spectrum, n=N)

    # Reconstruct dT/dz from flux
    dTdz = np.zeros((N, Nz - 1))
    for j in range(Nz - 1):
        spectrum = np.zeros(N // 2 + 1, dtype=complex)
        if j < flux_ratio.shape[1] - 1:
            spectrum[1:len(T_hat_surf)] = -T_hat_surf[1:] * flux_ratio[:len(T_hat_surf) - 1, j] / max(k_eq[j], 1e-30)
        dTdz[:, j] = irfft(spectrum, n=N)

    # k(T) at every (t, z) and time-averaged flux
    k_T = _k_cond(T_all, kc[None, :])
    J_pump = np.zeros(Nz)
    k_mean = np.zeros(Nz)

    for j in range(Nz):
        if j < Nz - 1:
            J_pump[j] = np.mean(k_T[:, j] * dTdz[:, j])
        else:
            J_pump[j] = J_pump[j - 1]
        k_mean[j] = np.mean(k_T[:, j])

    return J_pump, k_mean


# ═══════════════════════════════════════════════════════════════════════════
# Master solver
# ═══════════════════════════════════════════════════════════════════════════

def solve_fourier(
    lat: float,
    z: np.ndarray,
    dz: np.ndarray,
    kc: np.ndarray,
    rho: np.ndarray,
    n_steps: int = 256,
    dec: float = 0.0,
    max_outer: int = 5,
    max_inner: int = 40,
    outer_tol: float = 0.1,
    inner_tol: float = 0.5,
) -> tuple[np.ndarray, np.ndarray]:
    """Solve the periodic steady state via Fourier matrix method.

    Parameters
    ----------
    lat : float
        Latitude [rad].
    z : (Nz,) node positions [m].
    dz : (Nz-1,) layer thicknesses [m].
    kc : (Nz,) contact conductivity [W/m/K].
    rho : (Nz,) density [kg/m³].
    n_steps : int
        Number of time steps per cycle (must be even for rfft).
    max_outer : int
        Max rectification outer iterations.
    max_inner : int
        Max Newton iterations per outer loop.
    outer_tol : float
        Outer convergence tolerance on mean T_surf [K].
    inner_tol : float
        Newton convergence tolerance [K].

    Returns
    -------
    T_surf : (n_steps,) converged surface temperature
    T_eq : (Nz,) equilibrium depth profile
    """
    Nz = len(z)

    # ── Solar flux ──
    F_abs = compute_flux_series(lat, n_steps, dec=dec)
    F_mean = np.mean(F_abs)

    # ── Phase 1: mean surface temperature ──
    T_mean = _mean_surface_temp(F_mean, Q_GEO, EMISSIVITY)

    # ── Outer iteration ──
    J_pump = np.zeros(Nz)
    k_mean_eff: np.ndarray | None = None
    T_surf_prev_mean = T_mean

    omega = 2.0 * math.pi * np.arange(1, n_steps // 2 + 1) / LUNAR_DAY

    for outer in range(max_outer):
        # Phase 2: equilibrium profile
        T_eq = _equilibrium_profile(
            T_mean, z, kc, Q_GEO, J_pump=J_pump, k_eff=k_mean_eff,
        )

        # Frozen properties at T_eq
        k_eq = _k_cond(T_eq, kc)
        cp_eq = _cp(T_eq)
        kappa_eq = k_eq / (rho * cp_eq + 1e-30)
        k_mid = 0.5 * (k_eq[:-1] + k_eq[1:])
        kappa_mid = 0.5 * (kappa_eq[:-1] + kappa_eq[1:])

        # Phase 3: transmission matrices
        Z, depth_ratio, flux_ratio = _layer_matrices(omega, dz, k_mid, kappa_mid)

        h_r = 4.0 * EMISSIVITY * SIGMA * T_mean**3

        # Phase 4: initial guess (linearized)
        if outer == 0:
            F_hat = rfft(F_abs)
            T_hat = np.zeros(n_steps // 2 + 1, dtype=complex)
            for n in range(1, len(F_hat)):
                Z_n = Z[n - 1]
                T_hat[n] = Z_n / (1.0 + h_r * Z_n) * F_hat[n] if abs(Z_n) < 1e10 else F_hat[n] / h_r
            T_hat[0] = T_mean * n_steps
            T_surf_guess = irfft(T_hat, n=n_steps)
            T_surf_guess = np.maximum(T_surf_guess, 2.0)
        else:
            T_surf_guess = T_surf

        # Phase 5: Newton iteration
        T_surf = _newton_surface_circulant(
            T_surf_guess, F_abs, Q_GEO, Z, EMISSIVITY,
            max_iter=max_inner, tol=inner_tol,
        )

        # Phase 5b: check outer convergence
        T_mean_new = np.mean(T_surf)
        if abs(T_mean_new - T_surf_prev_mean) < outer_tol:
            T_mean = T_mean_new
            break
        T_surf_prev_mean = T_mean_new
        T_mean = T_mean_new

        # Phase 5c: rectification for next iteration
        if outer < max_outer - 1:
            J_pump, k_mean_eff = _compute_rectification(
                T_surf, depth_ratio, flux_ratio, T_eq, kc, k_eq, dz,
            )

    # ── Phase 6: final depth profile ──
    T_eq = _equilibrium_profile(
        T_mean, z, kc, Q_GEO,
        J_pump=J_pump if outer < max_outer else np.zeros(Nz),
        k_eff=k_mean_eff,
    )

    return T_surf, T_eq
