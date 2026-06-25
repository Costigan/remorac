"""1D lunar heat flow model — non-uniform grid, Remora-compiled throughout.

Wherever possible, computation is done in Remora (CPU-compiled).  Python
handles only what Remora cannot: spatial grid setup (time-independent),
Picard convergence loops, time stepping, and I/O.  This example also
serves as a driver for gap discovery in the Remora lowering pipeline.

The grid size N=60 is a constant in the Remora source.  The spatial grid
is non-uniform — geometric progression of layer thickness following
Hayne et al. (2017), Eqs. A31-A33.
"""

from __future__ import annotations

import numpy as np

import remora

# ═══════════════════════════════════════════════════════════════════════════
# Physical constants
# ═══════════════════════════════════════════════════════════════════════════

CP_COEFFS = (-3.6125, 2.7431, 2.3616e-3, -1.2340e-5, 8.9093e-9)
Kcs = 0.0007
Kcd = 0.0034
CHI = 2.7
R350 = CHI / 350.0**3
rhos = 1100.0
rhod = 1800.0
H_SCALE = 0.07
Q_GEO = 0.018          # geothermal heat flux [W/m²]

N = 60

# ═══════════════════════════════════════════════════════════════════════════
# Remora source — all functions in one string, N=60 hardcoded
# ═══════════════════════════════════════════════════════════════════════════

_SOURCE = f"""
(define/pi ()
  (compute_K
    [T  (Array Float 60) Kc (Array Float 60)]
    (Array Float 60))
  (let ((R350 {repr(R350)}))
    (map (lambda (t kcz) (* kcz (+ 1.0 (* R350 (* t (* t t))))))  T Kc)))

(define/pi ()
  (compute_Cp [T (Array Float 60)] (Array Float 60))
  (let ((c0 {repr(CP_COEFFS[0])}) (c1 {repr(CP_COEFFS[1])})
        (c2 {repr(CP_COEFFS[2])}) (c3 {repr(CP_COEFFS[3])})
        (c4 {repr(CP_COEFFS[4])}))
    (map (lambda (t)
      (+ c0 (* c1 t) (* c2 (* t t))
         (* c3 (* t (* t t))) (* c4 (* t (* t (* t t))))))
      T)))

(define/pi ()
  (cn_step
    [T_old  (Array Float 60)    g1   (Array Float 58)
     g2     (Array Float 58)    rho  (Array Float 60)
     K      (Array Float 60)    Cp   (Array Float 60)
     dt     Float               T_surf Float]
    (Array Float 60))

  (let* (
         (ha (map (lambda (i)
                (/ (* 0.5 dt (index-item g1 i) (index-item K i))
                   (* (index-item rho (+ i 1)) (index-item Cp (+ i 1)))))
              (iota 58)))
         (hb (map (lambda (i)
                (/ (* 0.5 dt (index-item g2 i) (index-item K (+ i 1)))
                   (* (index-item rho (+ i 1)) (index-item Cp (+ i 1)))))
              (iota 58)))

         (lower (map (lambda (i)
                  (if (< i 58) (- 0.0 (index-item ha i)) -1.0))
                (iota 59)))
         (diag (map (lambda (i)
                  (if (< i 1) 1.0
                    (if (< i 59)
                        (+ 1.0 (index-item ha (- i 1)) (index-item hb (- i 1)))
                        1.0)))
                (iota 60)))
         (upper (map (lambda (i)
                  (if (< i 1) 0.0 (- 0.0 (index-item hb (- i 1)))))
                (iota 59)))
         (rhs (map (lambda (i)
                  (if (< i 1) T_surf
                    (if (< i 59)
                        (let ((ha_i (index-item ha (- i 1)))
                              (hb_i (index-item hb (- i 1))))
                          (+ (* ha_i (index-item T_old (- i 1)))
                             (* (- 1.0 ha_i hb_i) (index-item T_old i))
                             (* hb_i (index-item T_old (+ i 1)))))
                        0.0)))
                (iota 60)))

         (cp (iscan (lambda (prev i)
                (let ((u (index-item upper i))
                      (d (index-item diag i))
                      (l (if (< i 1) 0.0 (index-item lower (- i 1)))))
                  (/ u (- d (* l prev)))))
              0.0 (iota 59)))
         (m  (map (lambda (i)
                (let ((d  (index-item diag i))
                      (l  (if (< i 1) 0.0 (index-item lower (- i 1))))
                      (pv (if (< i 1) 0.0 (index-item cp (- i 1)))))
                  (- d (* l pv))))
              (iota 60)))
         (dp (iscan (lambda (prev i)
                (let ((r  (index-item rhs i))
                      (mi (index-item m i))
                      (l  (if (< i 1) 0.0 (index-item lower (- i 1)))))
                  (/ (- r (* l prev)) mi)))
              0.0 (iota 60))))

    (trace-right (lambda (xnext i)
      (let ((dpi (index-item dp i))
            (cpi (if (< i 59) (index-item cp i) 0.0)))
        (- dpi (* cpi xnext))))
      0.0 (iota 60))))
"""

# ═══════════════════════════════════════════════════════════════════════════
# Compilation — compile once per session, reuse
# ═══════════════════════════════════════════════════════════════════════════

_K_FN: remora.RemoraFunction | None = None
_CP_FN: remora.RemoraFunction | None = None
_CN_FN: remora.RemoraFunction | None = None


def _get_K() -> remora.RemoraFunction:
    global _K_FN
    if _K_FN is None:
        _K_FN = remora.compile_function(_SOURCE, "compute_K",
                                         include_prelude=False)
    return _K_FN


def _get_Cp() -> remora.RemoraFunction:
    global _CP_FN
    if _CP_FN is None:
        _CP_FN = remora.compile_function(_SOURCE, "compute_Cp",
                                          include_prelude=False)
    return _CP_FN


def _get_cn() -> remora.RemoraFunction:
    global _CN_FN
    if _CN_FN is None:
        _CN_FN = remora.compile_function(_SOURCE, "cn_step",
                                          include_prelude=False)
    return _CN_FN


# ═══════════════════════════════════════════════════════════════════════════
# Non-uniform spatial grid
# ═══════════════════════════════════════════════════════════════════════════

def build_grid(
    z_max: float, growth_rate: float = 1.05,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build a non-uniform 1D grid with geometrically increasing spacing."""
    if growth_rate == 1.0:
        dz = np.full(N - 1, z_max / (N - 1), dtype=np.float64)
        z = np.linspace(0, z_max, N, dtype=np.float64)
    else:
        i = np.arange(N - 1, dtype=np.float64)
        dz0 = z_max * (growth_rate - 1) / (growth_rate ** (N - 1) - 1)
        dz = dz0 * growth_rate ** i
        z = np.zeros(N, dtype=np.float64)
        z[1:] = np.cumsum(dz)

    d3z = dz[1:] * dz[:-1] * (dz[1:] + dz[:-1])
    g1 = 2.0 * dz[1:] / d3z
    g2 = 2.0 * dz[:-1] / d3z
    return z, dz, g1, g2


# ═══════════════════════════════════════════════════════════════════════════
# 1D Heat model
# ═══════════════════════════════════════════════════════════════════════════

class Heat1DModel:
    """1D heat conduction model with temperature-dependent properties.

    Remora-compiled: K(T), Cp(T), CN step (interior assembly + Thomas solve).
    Python: grid setup, Picard loop, time stepping.
    """

    def __init__(
        self,
        z_max: float = 2.5,
        dt: float = 3600.0,
        T_surface: float = 250.0,
        T_init: float = 200.0,
        growth_rate: float = 1.05,
        picard_tol: float = 0.01,
        picard_max_iter: int = 20,
    ):
        self.dt = dt
        self.picard_tol = picard_tol
        self.picard_max_iter = picard_max_iter
        self._T_surface = T_surface

        self.z, self.dz, self.g1, self.g2 = build_grid(z_max, growth_rate)
        self.rho = rhod - (rhod - rhos) * np.exp(-self.z / H_SCALE)
        self.Kc = Kcs + (Kcd - Kcs) * (self.rho - rhos) / (rhod - rhos)
        self.T = np.full(N, T_init, dtype=np.float64)
        self.K: np.ndarray = np.empty(0)
        self.Cp: np.ndarray = np.empty(0)
        self._update_properties()

    def _update_properties(self) -> None:
        T_f32 = self.T.astype(np.float32)
        Kc_f32 = self.Kc.astype(np.float32)
        self.K = _get_K()(T_f32, Kc_f32).astype(np.float64)           # type: ignore[union-attr]
        self.Cp = _get_Cp()(T_f32).astype(np.float64)                  # type: ignore[union-attr]

    def step(self) -> np.ndarray:
        """Advance one Crank-Nicolson time step with Picard iteration."""
        T_old = self.T.copy()

        T_guess = T_old.copy()
        T_guess[0] = self._T_surface
        T_guess[-1] = T_guess[-2]

        for _ in range(self.picard_max_iter):
            K_guess, Cp_guess = self._props_for(T_guess)

            T_new = _get_cn()(                                               # type: ignore[union-attr]
                T_old.astype(np.float32),
                self.g1.astype(np.float32),
                self.g2.astype(np.float32),
                self.rho.astype(np.float32),
                K_guess.astype(np.float32),
                Cp_guess.astype(np.float32),
                np.asarray(self.dt, dtype=np.float32),
                np.asarray(self._T_surface, dtype=np.float32),
            ).astype(np.float64)

            # Bottom BC: geothermal flux (operator splitting)
            T_new[-1] = T_new[-2] + (Q_GEO / max(K_guess[-2], 1e-30)) * self.dz[-1]

            if np.max(np.abs(T_new - T_guess)) < self.picard_tol:
                self.T = T_new  # type: ignore[assignment]
                self._update_properties()
                return self.T.copy()

            T_guess = T_new

        self.T = T_new  # type: ignore[assignment]
        self._update_properties()
        return self.T.copy()

    def _props_for(self, T: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        T_f32 = T.astype(np.float32)
        Kc_f32 = self.Kc.astype(np.float32)
        K = _get_K()(T_f32, Kc_f32).astype(np.float64)               # type: ignore[union-attr]
        Cp = _get_Cp()(T_f32).astype(np.float64)                      # type: ignore[union-attr]
        return K, Cp

    def run(self, n_steps: int) -> list[np.ndarray]:
        history = [self.T.copy()]
        for _ in range(n_steps):
            self.step()
            history.append(self.T.copy())
        return history


# ═══════════════════════════════════════════════════════════════════════════
# NumPy reference oracle (for test validation)
# ═══════════════════════════════════════════════════════════════════════════

def _compute_Cp_numpy(T: np.ndarray) -> np.ndarray:
    c0, c1, c2, c3, c4 = CP_COEFFS
    return c0 + c1 * T + c2 * T**2 + c3 * T**3 + c4 * T**4


def _compute_K_numpy(T: np.ndarray, Kc: np.ndarray) -> np.ndarray:
    return Kc * (1.0 + R350 * T**3)
