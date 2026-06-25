"""Driver and plotting for the 1D lunar heat flow model.

Runs a simulation for a specified latitude/longitude, recording
surface temperature and depth profiles over one lunar day, then
generates standard diagnostic plots.

Usage
-----
    uv run python examples/heat1d/driver.py --lat 85 --z-max 2.5
    uv run python examples/heat1d/driver.py --lat 0 --z-max 2.5 --n-days 3 --output equator
    uv run python examples/heat1d/driver.py --lat 0 --dt 13289 --n-days 1 --no-show

Options
-------
    --lat        Latitude in degrees (default 0)
    --lon        Longitude in degrees (default 0)
    --z-max      Domain depth in metres (default 2.5)
    --dt         Time step in seconds (default 3600 = 1 hr)
    --n-days     Number of lunar days (default 1)
    --T-init     Initial temperature in K (default 200)
    --growth-rate  Grid growth ratio (default 1.05, use 1.0 for uniform)
    --month   Month 0–12 (0 = southern summer solstice, dec ≈ −1.54°)
    --output     Output filename prefix (default "heat1d")
    --show       Display plots interactively (default: save to file only)

A lunar day is ~29.5 Earth days (~2.55e6 s).  With the default
3600 s time step this gives ~709 steps per day.

Generated plots: diurnal temperature curves, min/max/avg depth profile,
and a time-vs-depth heatmap.

The lunar highland albedo (A0=0.12) and emissivity (0.95) are used.
Only one solver is implemented: Crank-Nicolson with operator-split
nonlinear surface BC (Newton with Volterra predictor).
"""

from __future__ import annotations

import math
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np

from examples.heat1d.heat1d_model import (
    Heat1DModel, N, _get_K, Kcs, R350,
)

# ═══════════════════════════════════════════════════════════════════════════
# Physical constants
# ═══════════════════════════════════════════════════════════════════════════

S0 = 1361.0           # solar constant at 1 AU [W/m²]
A0 = 0.12             # Bond albedo (highland)
ALBEDO_A = 0.06       # Keihm (1984) albedo coefficient a
ALBEDO_B = 0.25       # Keihm (1984) albedo coefficient b
EMISSIVITY = 0.95     # lunar surface emissivity
SIGMA = 5.670367e-8   # Stefan-Boltzmann constant
LUNAR_DAY = 29.53059 * 24.0 * 3600.0  # synodic month [s]
OBLIQUITY = math.radians(1.54)         # lunar obliquity [rad]
Q_GEO = 0.018         # geothermal heat flux [W/m²]

# ── Orbital parameters ──────────────────────────────────────────────────

OBLIQUITY_RAD = math.radians(1.54)
LUNAR_ORBIT = 27.321661 * 24.0 * 3600.0  # sidereal month [s]


def _declination(month: float) -> float:
    """Solar declination at a given month (0–12).

    The Moon's orbit is ~27.3 days; we approximate the seasonal cycle
    as a sinusoidal variation between ±1.54° (lunar obliquity).
    month=0 → equinox (dec ≈ 0°), month=3 → southern summer (dec ≈ −1.54°).
    """
    return -OBLIQUITY_RAD * math.sin(2.0 * math.pi * month / 12.0)


def equilibrium_temp(lat: float) -> float:
    """Mean equilibrium temperature for a rapidly rotating airless body.

    Hayne et al. (2017): the subsurface temperature at a given latitude
    is well approximated by the radiative-equilibrium noontime temperature
    divided by sqrt(2), following the classic analytic solution for a
    semi-infinite solid with a periodic surface boundary condition.

    T = [(1 - A) · S · cos(lat) / (ε · σ)]^(1/4)  /  sqrt(2)
    """
    T_noon = ((1.0 - A0) * S0 * max(math.cos(lat), 0.0)
              / (EMISSIVITY * SIGMA)) ** 0.25
    return T_noon / math.sqrt(2.0)
DTSURF = 0.1          # Newton surface-temp convergence tolerance [K]
MAX_NEWTON_ITER = 100


# ═══════════════════════════════════════════════════════════════════════════
# Solar insolation
# ═══════════════════════════════════════════════════════════════════════════

def cos_solar_zenith(lat: float, dec: float, hour_angle: float) -> float:
    """Cosine of the solar zenith angle."""
    c = math.sin(lat) * math.sin(dec) + math.cos(lat) * math.cos(dec) * math.cos(hour_angle)
    return max(c, 0.0)


def angle_dependent_albedo(incidence: float) -> float:
    """Keihm (1984) angle-dependent albedo."""
    if incidence <= 0.0:
        return A0
    x = incidence / (math.pi / 4.0)
    y = incidence / (math.pi / 2.0)
    return A0 + ALBEDO_A * x ** 3 + ALBEDO_B * y ** 8


def absorbed_flux(
    lat: float, hour_angle: float, dec: float = 0.0, r_au: float = 1.0,
) -> float:
    """Absorbed solar flux at the surface [W/m²].

    Parameters
    ----------
    lat : float
        Latitude [rad].
    hour_angle : float
        Hour angle [rad] (0 = local noon).
    dec : float
        Solar declination [rad].
    r_au : float
        Heliocentric distance [AU].
    """
    c = cos_solar_zenith(lat, dec, hour_angle)
    if c <= 0.0:
        return 0.0
    i = math.acos(c)
    A_i = angle_dependent_albedo(i)
    f = (1.0 - A_i) / (1.0 - A0)
    return f * S0 * (1.0 - A0) * (r_au ** -2) * c


# ═══════════════════════════════════════════════════════════════════════════
# Surface energy balance (Newton solver)
# ═══════════════════════════════════════════════════════════════════════════

def surface_temp_newton(
    Ts_guess: float,
    T1: float,
    T2: float,
    dz0: float,
    Kc0: float,
    rho0: float,
    cp0: float,
    Qs: float,
    Qs_prev: float | None = None,
    dt: float | None = None,
) -> float:
    """Solve the surface energy balance via Newton iteration.

    epsilon * sigma * Ts^4 - Qs - K(Ts) * dT/dz|_surface = 0
    """
    # Volterra predictor for the initial guess
    if Qs_prev is not None and dt is not None and dt > 0 and Qs_prev > 1e-6:
        k0 = Kc0 * (1.0 + R350 * Ts_guess ** 3)
        H0 = k0 * (-3.0 * Ts_guess + 4.0 * T1 - T2) / (2.0 * dz0)
        Gamma = math.sqrt(rho0 * cp0 * k0)
        Q_avg = (2.0 * Qs + Qs_prev) / 3.0
        rad = EMISSIVITY * SIGMA * Ts_guess ** 4
        denom = math.sqrt(math.pi / (4.0 * dt)) * Gamma + (8.0 / 3.0) * EMISSIVITY * SIGMA * Ts_guess ** 3
        if denom > 1e-30:
            delta_T = (-H0 - rad + Q_avg) / denom
            Ts_guess = max(min(Ts_guess + delta_T, 450.0), 50.0)

    # Newton refinement
    Ts = max(Ts_guess, 1.0)
    for _ in range(MAX_NEWTON_ITER):
        Ts3 = Ts ** 3
        x = EMISSIVITY * SIGMA * Ts3
        kT = Kc0 * (1.0 + R350 * Ts3)
        y = 0.5 * kT / dz0

        f_val = x * Ts - Qs - y * (-3.0 * Ts + 4.0 * T1 - T2)
        fp_val = 4.0 * x - 3.0 * Kc0 * R350 * Ts ** 2 * 0.5 * (4.0 * T1 - 3.0 * Ts - T2) / dz0 + 3.0 * y

        if abs(fp_val) < 1e-30:
            break
        delta_T = -f_val / fp_val
        Ts_new = Ts + delta_T
        if Ts_new < 1.0 or Ts_new > 700.0 or abs(delta_T) > 100.0:
            break
        Ts = Ts_new
        if abs(delta_T) <= DTSURF:
            break

    return max(Ts, 1.0)


# ═══════════════════════════════════════════════════════════════════════════
# Simulation runner
# ═══════════════════════════════════════════════════════════════════════════

def run_simulation(
    lat: float = math.radians(0.0),
    lon: float = math.radians(0.0),
    z_max: float = 2.5,
    dt: float = 3600.0,
    n_days: int = 1,
    equil_days: int = 0,
    T_init: float | None = None,
    growth_rate: float = 1.05,
    dec: float = 0.0,
    fourier_equil: bool = True,
) -> dict:
    """Run the 1D heat flow model and return results.

    Parameters
    ----------
    equil_days : int
        Number of days to time-step before recording output.
    T_init : float or None
        Initial temperature.  If None, uses the latitude-dependent
        equilibrium temperature from Hayne et al. (2017).
    fourier_equil : bool
        If True, use the Fourier-matrix solver to compute the periodic
        steady state, then initialize the time-stepping model from it.
        This replaces multi-orbit spin-up and is ~1000x faster.
    """
    from examples.heat1d.heat1d_model import build_grid, N, rhos, rhod, H_SCALE, Kcs, Kcd
    from examples.heat1d.fourier_solver import solve_fourier, _kc

    if T_init is None:
        T_init = equilibrium_temp(lat)
    steps_per_day = int(LUNAR_DAY / dt)
    model = Heat1DModel(z_max=z_max, dt=dt, T_init=T_init,
                        growth_rate=growth_rate)

    # ── Fourier equilibration (jump to periodic steady state) ──
    if fourier_equil:
        kc = _kc(model.z)
        rho_z = rhod - (rhod - rhos) * np.exp(-model.z / H_SCALE)
        T_surf_fourier, T_eq_fourier = solve_fourier(lat, model.z, model.dz, kc, rho_z, dec=dec)
        model.T[:] = T_eq_fourier
        model._update_properties()

    Qs_prev = 0.0
    month_offset = dec / OBLIQUITY_RAD * 6.0 / math.pi if dec != 0.0 else 0.0  # approx

    # ── Equilibration ──
    for _ in range(equil_days):
        for _ in range(steps_per_day):
            t_hour = 2.0 * math.pi * (_ * dt % LUNAR_DAY) / LUNAR_DAY + lon
            Qs = absorbed_flux(lat, t_hour, dec=dec)
            T_surf = surface_temp_newton(
                model.T[0], model.T[1], model.T[2],
                model.dz[0], Kcs, model.rho[0], model.Cp[0],
                Qs, Qs_prev=Qs_prev, dt=dt,
            )
            model._T_surface = T_surf
            model.T[-1] = model.T[-2] + (Q_GEO / model.K[-2]) * model.dz[-1]
            model.step()
            Qs_prev = Qs

    # ── Output recording ──
    n_steps = n_days * steps_per_day

    T_history = np.empty((n_steps + 1, len(model.T)), dtype=np.float64)
    T_history[0] = model.T.copy()

    for step in range(n_steps):
        t = step * dt
        hour_angle = 2.0 * math.pi * t / LUNAR_DAY + lon
        # Declination cycles with the lunar orbit period (~27.3 days)
        current_dec = _declination(month_offset + t / (27.3 * 24.0 * 3600.0) * 12.0)
        Qs = absorbed_flux(lat, hour_angle, dec=current_dec)

        T_surf = surface_temp_newton(
            model.T[0], model.T[1], model.T[2],
            model.dz[0], Kcs, model.rho[0], model.Cp[0],
            Qs, Qs_prev=Qs_prev, dt=dt,
        )

        model.T[-1] = model.T[-2] + (Q_GEO / model.K[-2]) * model.dz[-1]
        model._T_surface = T_surf
        model.step()

        T_history[step + 1] = model.T
        Qs_prev = Qs

    hours = np.arange(n_steps + 1) * dt / 3600.0

    return {
        "time_hours": hours,
        "T_history": T_history,
        "z": model.z,
        "dz": model.dz,
        "lat": lat,
        "lon": lon,
    }


# ═══════════════════════════════════════════════════════════════════════════
# Plotting
# ═══════════════════════════════════════════════════════════════════════════

def plot_diurnal_curves(
    time_hours: np.ndarray,
    T_history: np.ndarray,
    z: np.ndarray,
    depths: Sequence[float] = (0.0, 0.02, 0.05, 0.10, 0.20, 0.50, 1.0),
    lat: float = 0.0,
    ax: plt.Axes | None = None,
) -> plt.Axes:
    """Plot temperature vs time at selected depths."""
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 5))

    cmap = plt.cm.magma
    for i, target_depth in enumerate(depths):
        if target_depth > z[-1]:
            continue
        idx = np.searchsorted(z, target_depth)
        if idx >= len(z):
            idx = len(z) - 1
        color = cmap(i / max(len(depths) - 1, 1))
        ax.plot(time_hours, T_history[:, idx], color=color,
                label=f"{z[idx]:.3f} m", linewidth=0.8)

    ax.set_xlabel("Local Time [hours]")
    ax.set_ylabel("Temperature [K]")
    ax.set_title(f"Diurnal Temperature Curves (lat = {math.degrees(lat):.0f}$^\\circ$)")
    ax.legend(fontsize=7, loc="upper right")
    ax.grid(True, alpha=0.3)
    return ax


def plot_depth_profile(
    T_history: np.ndarray,
    z: np.ndarray,
    lat: float = 0.0,
    ax: plt.Axes | None = None,
    z_min: float | None = None,
    z_max: float | None = None,
) -> plt.Axes:
    """Plot min, max, and average temperature vs depth."""
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 8))

    T_max = np.max(T_history[1:], axis=0)  # skip initial
    T_min = np.min(T_history[1:], axis=0)
    T_avg = np.mean(T_history[1:], axis=0)

    ax.plot(T_max, z, "r-", label="$T_{max}$", linewidth=1.2)
    ax.plot(T_min, z, "b--", label="$T_{min}$", linewidth=1.2)
    ax.plot(T_avg, z, "k-", label="$T_{avg}$", linewidth=1.2)

    ax.set_xlabel("Temperature [K]")
    ax.set_ylabel("Depth [m]")
    ax.set_title(f"Temperature vs Depth (lat = {math.degrees(lat):.0f}$^\\circ$)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.invert_yaxis()
    top = 0 if z_min is None else z_min
    bot = z[-1] if z_max is None else z_max
    ax.set_ylim(bot, top)
    ax.margins(y=0)
    return ax


def plot_heatmap(
    time_hours: np.ndarray,
    T_history: np.ndarray,
    z: np.ndarray,
    lat: float = 0.0,
    ax: plt.Axes | None = None,
) -> plt.Axes:
    """Plot temperature heatmap (time vs depth)."""
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 6))

    # Use only the first day
    day_mask = time_hours <= 24.0
    t_plot = time_hours[day_mask]
    T_plot = T_history[day_mask]

    mesh = ax.pcolormesh(t_plot, z, T_plot.T, cmap="magma", shading="auto")
    plt.colorbar(mesh, ax=ax, label="Temperature [K]")

    ax.set_xlabel("Local Time [hours]")
    ax.set_ylabel("Depth [m]")
    ax.set_title(f"Temperature Heatmap (lat = {math.degrees(lat):.0f}$^\\circ$)")
    ax.invert_yaxis()
    return ax


def generate_plots(
    results: dict,
    output_prefix: str = "heat1d",
    show: bool = True,
) -> None:
    """Generate and save all diagnostic plots."""
    time_hours = results["time_hours"]
    T_history = results["T_history"]
    z = results["z"]
    lat = results.get("lat", 0.0)

    # 1. Diurnal curves
    fig1, ax1 = plt.subplots(figsize=(10, 5))
    plot_diurnal_curves(time_hours, T_history, z, lat=lat, ax=ax1)
    fig1.tight_layout()
    fig1.savefig(f"{output_prefix}_diurnal.png", dpi=150)

    # 2. Depth profile
    fig2, ax2 = plt.subplots(figsize=(6, 8))
    plot_depth_profile(T_history, z, lat=lat, ax=ax2,
                       z_min=args.plot_y_min, z_max=args.plot_y_max)
    fig2.tight_layout()
    fig2.savefig(f"{output_prefix}_depth_profile.png", dpi=150)

    # 3. Heatmap
    fig3, ax3 = plt.subplots(figsize=(10, 6))
    plot_heatmap(time_hours, T_history, z, lat=lat, ax=ax3)
    fig3.tight_layout()
    fig3.savefig(f"{output_prefix}_heatmap.png", dpi=150)

    if show:
        plt.show()
    else:
        plt.close("all")


# ═══════════════════════════════════════════════════════════════════════════
# CLI entry point
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="1D lunar heat flow simulation")
    p.add_argument("--lat", type=float, default=0.0,
                   help="Latitude [degrees]")
    p.add_argument("--lon", type=float, default=0.0,
                   help="Longitude [degrees]")
    p.add_argument("--z-max", type=float, default=None,
                   help="Override domain depth [m] (default: skin-depth grid)")
    p.add_argument("--dt", type=float, default=3600.0,
                   help="Time step [s] (default 3600 = 1 hr)")
    p.add_argument("--n-days", type=int, default=20,
                   help="Number of lunar days (default 20)")
    p.add_argument("--equil-days", type=int, default=20,
                   help="Equilibration days before output (default 20)")
    p.add_argument("--growth-rate", type=float, default=1.05,
                   help="Grid growth ratio (only with --z-max override)")
    p.add_argument("--T-init", type=float, default=None,
                   help="Initial temperature [K] (default: T_eq(lat))")
    p.add_argument("--month", type=int, default=0,
                   help="Starting month (0 = equinox, 3 = southern summer)")
    p.add_argument("--dec", type=float, default=None,
                   help="Solar declination [deg] (overrides --month)")
    p.add_argument("--csv", type=str, default=None,
                   help="Write temperature data to CSV file")
    p.add_argument("--plot-y-min", type=float, default=None,
                   help="Min depth shown in profile plot [m]")
    p.add_argument("--plot-y-max", type=float, default=None,
                   help="Max depth shown in profile plot [m]")
    p.add_argument("--show", action="store_true",
                   help="Display plots interactively")
    p.add_argument("--output", type=str, default="heat1d",
                   help="Output file prefix")
    args = p.parse_args()

    lat_rad = math.radians(args.lat)
    if args.dec is not None:
        dec_rad = math.radians(args.dec)
    else:
        dec_rad = _declination(args.month)

    print(f"Running {args.n_days} lunar day(s) at lat = {args.lat}°")
    print(f"  dt = {args.dt:.0f} s, z_max = {args.z_max} m")
    print(f"  solar declination = {math.degrees(dec_rad):.2f}°")
    results = run_simulation(
        lat=lat_rad, lon=math.radians(args.lon),
        z_max=args.z_max, dt=args.dt,
        n_days=args.n_days, equil_days=args.equil_days,
        T_init=args.T_init, growth_rate=args.growth_rate,
        dec=dec_rad,
    )

    T_surf = results["T_history"][:, 0]
    print(f"  T_surf min = {T_surf.min():.1f} K, max = {T_surf.max():.1f} K")
    print(f"  T_surf mean = {T_surf.mean():.1f} K")

    if args.csv:
        import csv as _csv
        z = results["z"]
        T = results["T_history"]
        with open(args.csv, "w", newline="") as f:
            w = _csv.writer(f)
            w.writerow(["time_hr"] + [f"z_{zv:.6f}" for zv in z])
            for i, t in enumerate(results["time_hours"]):
                w.writerow([f"{t:.4f}"] + [f"{T[i,j]:.4f}" for j in range(len(z))])
        print(f"  Wrote {args.csv}")

    print(f"Generating plots...")
    generate_plots(results, output_prefix=args.output, show=args.show)
