# 1D Lunar Heat Flow Model (Hayne et al. 2017)

RemoraC implementation of the one-dimensional heat conduction model for the
lunar regolith described in Hayne et al. (2017, *JGR Planets* **122**, 2371–2400).
Push as much numerical work as possible into Remora (CPU-compiled); Python
handles grid setup, Picard convergence loops, time stepping, and I/O.

## Physics

The model solves the 1-D heat equation in lunar regolith with depth- and
temperature-dependent thermophysical properties:

```
rho * Cp(T) * dT/dt = d/dz [ K(T) * dT/dz ] + Q_geo
```

**Thermal conductivity** (Mitchell & de Pater 1994):
```
K(T) = Kc(z) * (1 + chi/350^3 * T^3)
```
where the contact conductivity follows an exponential depth profile
`Kc(z) = K_cd - (K_cd - K_cs) * exp(-z/H)` linking the surface value
(7.4e-4 W/m/K) to the deep regolith value (3.4e-3 W/m/K).

**Heat capacity** (Hemingway et al. 1981, Ledlow et al. 1992):
a 4th-order polynomial in T.

**Angle-dependent albedo** (Keihm 1984):
`A(i) = A0 + a*(i/(pi/4))^3 + b*(i/(pi/2))^8`.

Parameters for the lunar highlands (default): A0=0.12, emissivity=0.95,
geothermal flux Q_geo=0.018 W/m².

**Orbit model** (`orbit.py`): Earth's eccentric orbit (e=0.0167) modulates
heliocentric distance (±1.7%), while the lunar obliquity (1.54°) drives
declination.  Kepler's equation is solved for true anomaly ν(t), from
which r_au(t) and dec(t) are derived.

## Spatial grid

The default grid follows Hayne Eqs. A31–A33: a geometric progression of
layer thicknesses.  The skin depth `z_s = sqrt(kappa * P / pi)` determines
the fundamental scale; m=10 layers fit inside one skin depth, n=4 controls
the growth rate `r = 1 + 1/n`, and b=20 skin depths define the domain bottom.
With lunar parameters this gives **N=19 nodes**, z_max ≈ 0.66 m.

An alternative uniform-growth grid is available via `--z-max` or the
`build_grid()` function.

## Solvers

| Solver | Where | Description |
|--------|-------|-------------|
| Fourier-matrix (equilibration) | `fourier_solver.py` | Frequency-domain steady state via thermal transmission matrices (Pipes 1957 / Maillet et al. 2000). Solves the nonlinear surface radiation with Newton iteration on a circulant admittance matrix. ~1000× faster than time-stepping. |
| Crank-Nicolson (time-stepping) | Remora-compiled in `heat1d_model.py` | Semi-implicit finite-difference with Picard iteration for nonlinear k(T)/Cp(T). Tridiagonal system solved via Thomas algorithm (TDMA) implemented as `iscan`/`trace-right` in Remora. |
| Surface energy balance | `driver.py` | Newton solver with Volterra predictor (Schorghofer & Khatiwala 2024, Eq. 62) for the nonlinear Stefan-Boltzmann boundary condition, applied via operator splitting before each CN step. |

## Architecture

```
orbit.py ──→ r_au(t), dec(t)
               │
driver.py ──→ run_simulation()   ← CLI, plotting, CSV
               │
               ├─ fourier_solver.py  (equilibration)
               ├─ Heat1DModel        (time-stepping)
               │    ├─ Remora: compute_K, compute_Cp, cn_step
               │    └─ Python: Picard loop, geothermal BC
               └─ surface_temp_newton()  (surface BC)
```

### Remora-compiled functions

Three functions are compiled from a single `_SOURCE` f-string with N
hardcoded from the grid:

| Function | Signature | Purpose |
|----------|-----------|---------|
| `compute_K` | `Float[N] Float[N] → Float[N]` | Radiative+contact conductivity k(T) |
| `compute_Cp` | `Float[N] → Float[N]` | Polynomial heat capacity cp(T) |
| `cn_step` | `Float[N] Float[N-2] Float[N-2] Float[N] Float[N] Float[N] Float Float → Float[N]` | Crank-Nicolson assembly + Thomas tridiagonal solve |

The CN step constructs the full N×N tridiagonal system including surface
(Dirichlet) and bottom (Neumann) boundary conditions, then solves via
forward `iscan` + backward `trace-right`.

## Running

```bash
# Default: south pole, 2 lunar days, 360 s step
uv run python examples/heat1d/driver.py --lat -85 --dt 360 --n-days 2

# Equator, 20 days, custom depth, show plots
uv run python examples/heat1d/driver.py --lat 0 --n-days 20 --z-max 2.5 --show

# 3 months, CSV output, skip warmup
uv run python examples/heat1d/driver.py --lat -85 --n-days 24 --dt 360 \
    --csv data.csv --no-warmup

# Start at southern summer (month=3), 5-day run
uv run python examples/heat1d/driver.py --lat -85 --month 3 --n-days 5 --show
```

### CLI options

| Flag | Default | Description |
|------|---------|-------------|
| `--lat` | 0 | Latitude [deg] |
| `--lon` | 0 | Longitude [deg] |
| `--z-max` | None | Domain depth [m] (None = skin-depth grid) |
| `--dt` | 3600 | Time step [s] |
| `--n-days` | 20 | Output lunar days |
| `--equil-days` | 20 | Equilibration days before output |
| `--month` | 0 | Starting phase (0=equinox, 3=S.summer) |
| `--dec` | None | Solar declination [deg] (overrides --month) |
| `--no-warmup` | False | Skip CFL warmup phase |
| `--n-warmup` | 3 | Warmup diurnal cycles |
| `--csv` | None | Write T(time, depth) to CSV |
| `--show` | False | Display plots interactively |

### Plots

Three plots are generated (saved to `<prefix>_*.png` unless `--show`):
- **Diurnal curves**: T vs time at selected depths
- **Depth profile**: T_min, T_max, T_avg vs depth (linear scale, surface at 0)
- **Heatmap**: time vs depth with magma colormap

CSV format: first column = time [hr], remaining columns = T at each depth
node.  Depth values appear in the header row.

## Comparison with Hayne et al. (2017) reference

Our implementation was validated against the open-source reference model
([github.com/pog1990/heat1d](https://github.com/pog1990/heat1d)) by P. O.
Hayne.  The comparison was run at latitude −85° (lunar south pole), dt=360 s,
with Fourier-matrix equilibration followed by 3-cycle CFL warmup and 24
lunar days (~708 Earth days, ~23 calendar months) of Crank-Nicolson output.

### Surface temperature

| Metric | Reference | Our model | Δ |
|--------|-----------|-----------|-----|
| T_surf min | 56.6 K | 56.5 K | −0.1 K |
| T_surf max | 177.9 K | 177.9 K | 0.0 K |
| T_surf mean | 99.3 K | 99.2 K | −0.1 K |
| Diurnal swing | 121.3 K | 121.4 K | +0.1 K |

### Depth profile

| Depth [m] | Ref swing [K] | Our swing [K] | Δ [K] | Ref mean [K] | Our mean [K] | Δ [K] |
|-----------|---------------|---------------|-------|--------------|--------------|-------|
| 0.00 | 121.3 | 121.4 | 0.1 | 99.3 | 99.2 | 0.1 |
| 0.01 | 94.7 | 94.8 | 0.1 | 100.4 | 100.1 | 0.2 |
| 0.06 | 47.6 | 47.5 | 0.1 | 101.3 | 100.9 | 0.4 |
| 0.10 | 31.2 | 31.0 | 0.2 | 101.5 | 101.0 | 0.6 |
| 0.21 | 13.9 | 13.6 | 0.4 | 101.8 | 101.0 | 0.8 |
| 0.33 | 8.0 | 7.6 | 0.5 | 102.1 | 101.1 | 1.0 |
| 0.42 | 6.6 | 6.2 | 0.4 | 102.4 | 101.2 | 1.2 |
| 0.52 | 6.2 | 5.7 | 0.5 | 102.9 | 101.4 | 1.5 |
| 0.66 | 6.2 | 5.7 | 0.5 | 103.6 | 102.1 | 1.5 |

The diurnal swing agrees to within 0.5 K at all depths.  Mean subsurface
temperature runs ~1.5 K cooler at the bottom of our model, attributable to
a small orbit-phase offset accumulated during the warmup phase (our single
CFL-dt computation vs. the reference's per-step CFL recomputation).

### Validated components

- **Crank-Nicolson step**: single-step comparison between our Remora TDMA
  and the reference's NumPy Thomas solver shows agreement to 3×10⁻⁵ K.
- **Surface Newton solver**: standalone comparison to 3×10⁻⁵ K.
- **Absorbed solar flux**: identical to machine precision (matching
  Keihm 1984 angle-dependent albedo and cosine solar zenith).
- **Fourier-matrix solver**: surface temperature extrema match to 0.1 K.
- **Kcs consistency**: the contact conductivity at the surface is now
  7.4×10⁻⁴ W/m/K everywhere (was 7.0×10⁻⁴ in an earlier `fourier_solver.py`
  — this 5.4% discrepancy has been corrected).

## File map

```
examples/heat1d/
├── README.md            ← this file
├── heat1d_model.py      Remora sources, grid builders, Heat1DModel class
├── driver.py            CLI, run_simulation(), surface BC Newton, plotting
├── fourier_solver.py    Fourier-matrix equilibration (frequency domain)
├── orbit.py             Earth-Sun orbit model (Kepler, r_au, declination)
└── test_heat1d.py       12 tests (grid, properties, CN step, model)
```

## Future work

### Move more computation into Remora

Currently Remora handles k(T), Cp(T), and the CN step (assembly + TDMA).
The following could be moved into Remora to reduce Python overhead:

- **Bottom geothermal BC**: the line `T[-1] = T[-2] + Q_geo*dz[-1]/K[-2]`
  is applied in Python after each CN step.  This could be folded into the
  Remora `cn_step` function as an optional final pass.
- **Surface energy balance Newton iteration**: the Python
  `surface_temp_newton()` loops over Newton steps calling scalar arithmetic.
  Each step reads T[0], T[1], T[2] from NumPy arrays.  A Remora-compiled
  scalar Newton solver would eliminate the Python loop overhead.
- **Picard convergence loop**: currently in Python, iterating up to 20
  times per CN step.  Moving this loop into Remora would reduce
  Python→Remora call overhead (currently 1–20 calls per time step).
- **Property updates**: K and Cp are recomputed in Remora at each Picard
  iteration.  If the Picard loop were internal to Remora, properties could
  be updated in-place without returning to Python.

### Numba acceleration of Python portions

The Python portions that remain after moving work to Remora could benefit
from Numba JIT compilation:

- **`surface_temp_newton()`**: the Newton iteration is a tight scalar loop
  ideal for `@njit`.  The reference model already does this for its
  `_newton_surface` function.
- **Grid building** (`build_grid_skin_depth`, `build_grid`): vectorized
  NumPy already, but Numba could accelerate the cumsum and arithmetic.
- **Equilibration and output loops** in `run_simulation()`: the inner
  time-stepping loop iterates ~170k times for a 24-day run.  Moving the
  step body to a `@njit` function that operates on arrays in-place would
  eliminate Python interpreter overhead.

### FFT primitives in Remora

The Fourier-matrix solver (`fourier_solver.py`) is implemented in pure
NumPy and shares no code with the Remora path.  It uses `np.fft.rfft` and
`np.fft.irfft` extensively for:

- Computing the frequency-domain solar flux spectrum
- Building circulant admittance matrices from complex impedance
- Reconstructing T(t,z) from surface T_hat × depth transfer functions

If Remora exposed an FFT primitive (similar to APL's special-purpose array
algorithms), the entire Fourier equilibration could be expressed in Remora:

```
remora.fft.rfft  — real-to-complex forward transform
remora.fft.irfft — inverse real transform
```

This would unify the codebase, allow the Fourier solver to benefit from
Remora's optimization pipeline (potentially generating GPU-accelerated
FFTs), and make the transmission-matrix multiplication (`P00 * depth_ratio`)
operate on Remora arrays rather than NumPy.

### GPU path

The CN step and property functions already compile through the Remora
pipeline.  Extending the GPU lowering path to cover `iscan`, `trace-right`,
and `index-item` within map bodies would allow the entire time-stepping
phase to run on GPU.  Combined with an FFT primitive, the Fourier
equilibration could also target GPU, enabling full-GPU simulations.

### Adaptive time-stepping

The reference model supports Richardson-extrapolation adaptive time-stepping
for implicit and CN solvers (step doubling with error estimation).  Our
implementation uses a fixed dt throughout.  Adding adaptive stepping would
allow automatic resolution of sunrise/sunset transitions while taking larger
steps during the isothermal night.

### Multi-latitude sweeps and validation

The reference model includes a validation suite against Diviner radiometer
data and Apollo heat flow measurements.  Adding a similar validation
pipeline (reading Diviner CSV time series, running parameter sweeps across
latitudes) would close the loop on model accuracy.
