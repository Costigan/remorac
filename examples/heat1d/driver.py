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
    --dec        Solar declination in degrees (default 0)
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
    if Qs < 1e-6:
        # No insolation — surface cools radiatively to a floor.
        # In steady state with subsurface geothermal flux this floor
        # is ~30-50 K; during transient cooling from a warm initial
        # condition we step downward gently.
        return max(Ts_guess * 0.999, 30.0)

    # Use Volterra predictor for sunlit periods
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
    T_init: float = 200.0,
    growth_rate: float = 1.05,
    dec: float = 0.0,
) -> dict:
    """Run the 1D heat flow model and return results.

    Parameters
    ----------
    equil_days : int
        Number of days to run before recording output.  The model
        starts from a uniform *T_init* profile; equilibration spins
        it up to a periodic steady state.
    """
    steps_per_day = int(LUNAR_DAY / dt)
    model = Heat1DModel(z_max=z_max, dt=dt, T_init=T_init,
                        growth_rate=growth_rate)

    Qs_prev = 0.0

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
        Qs = absorbed_flux(lat, hour_angle, dec=dec)

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
    if z[-1] > 0.02:
        ax.set_yscale("log")
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
    plot_depth_profile(T_history, z, lat=lat, ax=ax2)
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
    p.add_argument("--z-max", type=float, default=2.5,
                   help="Domain depth [m]")
    p.add_argument("--dt", type=float, default=3600.0,
                   help="Time step [s] (default 3600 = 1 hr)")
    p.add_argument("--n-days", type=int, default=20,
                   help="Number of lunar days (default 20)")
    p.add_argument("--equil-days", type=int, default=20,
                   help="Equilibration days before output (default 20)")
    p.add_argument("--growth-rate", type=float, default=1.05,
                   help="Grid growth ratio (1.0 = uniform)")
    p.add_argument("--T-init", type=float, default=200.0,
                   help="Initial temperature [K]")
    p.add_argument("--dec", type=float, default=0.0,
                   help="Solar declination [degrees]")
    p.add_argument("--show", action="store_true",
                   help="Display plots interactively")
    p.add_argument("--output", type=str, default="heat1d",
                   help="Output file prefix")
    args = p.parse_args()

    lat_rad = math.radians(args.lat)
    dec_rad = math.radians(args.dec)

    print(f"Running {args.n_days} lunar day(s) at lat = {args.lat}°")
    print(f"  dt = {args.dt:.0f} s, z_max = {args.z_max} m")
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
    print(f"Generating plots...")
    generate_plots(results, output_prefix=args.output, show=args.show)
