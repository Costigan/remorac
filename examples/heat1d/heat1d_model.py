"""1D lunar heat flow model — Stage 1c: temperature-dependent K(T), Cp(T).

Uses Remora-compiled functions for element-wise K(T) and Cp(T) computation
via map.  Python handles the Thomas solve, CN assembly, boundary conditions,
Picard iteration, and time stepping.

Stage 1a (constant coefficients, uniform grid) → Stage 1c (temperature-dependent).
"""

from __future__ import annotations

import numpy as np

from remora.runtime import CPUFunctionExecutor
from remora.types import ArrayType, FLOAT, INT, StaticDim


# ── Physical constants ──────────────────────────────────────────────────

# Moon coefficients for Cp(T) polynomial (Ledlow et al. 1992)
# Cp(T) = c0 + c1*T + c2*T^2 + c3*T^3 + c4*T^4   [J/kg/K], T in Kelvin
CP_COEFFS = (-3.6125, 2.7431, 2.3616e-3, -1.2340e-5, 8.9093e-9)

# Thermal conductivity parameters
Kcs = 0.0007   # surface contact conductivity [W/m/K]
Kcd = 0.0034   # deep contact conductivity [W/m/K]
CHI = 2.7      # radiative conductivity parameter (Mitchell & de Pater 1994)
R350 = CHI / 350.0 ** 3  # radiative factor = chi / 350^3

# Density
rhos = 1100.0  # surface density [kg/m^3]
rhod = 1800.0  # deep regolith density [kg/m^3]
H = 0.07       # scale height [m]


# ── Remora source for K(T) ──────────────────────────────────────────────

_K_SOURCE = """
(define/pi ()
  (compute_K
    [T (Array Float {N}) Kc (Array Float {N})]
    (Array Float {N}))
  (map (lambda (t kcz)
    (* kcz (+ 1.0 (* {R350} (* t (* t t))))))
    T Kc))
"""

# ── Remora source for Cp(T) ─────────────────────────────────────────────

_CP_SOURCE = """
(define/pi ()
  (compute_Cp [T (Array Float {N})] (Array Float {N}))
  (map (lambda (t)
    (+ {c0}
       (* {c1} t)
       (* {c2} (* t t))
       (* {c3} (* t (* t t)))
       (* {c4} (* t (* t (* t t))))))
    T))
"""


def _compile_K(N: int) -> CPUFunctionExecutor:
    """Compile compute_K(T, Kc) for grid size N."""
    source = _K_SOURCE.format(N=N, R350=repr(R350))
    artifact = CPUFunctionExecutor.compile_source(
        source, "compute_K",
        (ArrayType(FLOAT, (StaticDim(N),)), ArrayType(FLOAT, (StaticDim(N),))),
        syntax="lisp", include_prelude=False,
    )
    return CPUFunctionExecutor(artifact)


def _compile_Cp(N: int) -> CPUFunctionExecutor:
    """Compile compute_Cp(T) for grid size N."""
    c0, c1, c2, c3, c4 = CP_COEFFS
    source = _CP_SOURCE.format(N=N, c0=repr(c0), c1=repr(c1), c2=repr(c2),
                                c3=repr(c3), c4=repr(c4))
    artifact = CPUFunctionExecutor.compile_source(
        source, "compute_Cp",
        (ArrayType(FLOAT, (StaticDim(N),)),),
        syntax="lisp", include_prelude=False,
    )
    return CPUFunctionExecutor(artifact)


# ── Thomas Tridiagonal Solver (Python) ───────────────────────────────────

def thomas_solve(
    lower: np.ndarray,
    diag: np.ndarray,
    upper: np.ndarray,
    rhs: np.ndarray,
) -> np.ndarray:
    """Solve tridiagonal system using the Thomas algorithm (TDMA).  O(n)."""
    n = len(rhs)
    cp = np.empty(n - 1, dtype=np.float64)
    dp = np.empty(n, dtype=np.float64)

    cp[0] = upper[0] / diag[0]
    dp[0] = rhs[0] / diag[0]

    for i in range(1, n):
        m = diag[i] - lower[i - 1] * cp[i - 1]
        if i < n - 1:
            cp[i] = upper[i] / m
        dp[i] = (rhs[i] - lower[i - 1] * dp[i - 1]) / m

    x = np.empty(n, dtype=np.float64)
    x[-1] = dp[-1]
    for i in range(n - 2, -1, -1):
        x[i] = dp[i] - cp[i] * x[i + 1]

    return x


# ── 1D Heat Model with Temperature-Dependent Properties ──────────────────

class Heat1DModel:
    """1D heat conduction model with temperature-dependent K(T), Cp(T).

    Uses Remora-compiled functions for element-wise material property
    computation.  Python orchestrates the Picard iteration, CN assembly,
    Thomas solve, and time stepping.
    """

    def __init__(
        self,
        N: int = 30,
        z_max: float = 2.5,
        dt: float = 3600.0,
        T_surface: float = 250.0,
        T_init: float = 200.0,
        # Picard convergence
        picard_tol: float = 0.01,
        picard_max_iter: int = 20,
    ):
        self.N = N
        self.z_max = z_max
        self.dt = dt
        self.picard_tol = picard_tol
        self.picard_max_iter = picard_max_iter

        # Uniform spatial grid
        self.dz = z_max / (N - 1) if N > 1 else z_max
        self.z = np.linspace(0, z_max, N, dtype=np.float64)

        # Precompute depth-dependent arrays (time-independent)
        self.rng = np.random.RandomState(12345)  # for testing only

        # Density rho(z)
        self.rho = self._compute_rho()

        # Contact conductivity Kc(z)
        self.Kc = self._compute_Kc()

        # Temperature profile
        self.T = np.full(N, T_init, dtype=np.float64)

        # Current material properties (computed from T)
        self.K: np.ndarray = np.empty(N, dtype=np.float64)  # conductivity at nodes
        self.Cp: np.ndarray = np.empty(N, dtype=np.float64)  # heat capacity at nodes

        # Lazy-compiled Remora functions
        self._K_executor: CPUFunctionExecutor | None = None
        self._Cp_executor: CPUFunctionExecutor | None = None

        # Boundary conditions
        self._T_surface = T_surface

        # Initialize properties
        self._update_properties()

    def _compute_rho(self) -> np.ndarray:
        """Density profile: rho(z) = rhod - (rhod - rhos) * exp(-z/H)."""
        return rhod - (rhod - rhos) * np.exp(-self.z / H)

    def _compute_Kc(self) -> np.ndarray:
        """Contact conductivity: Kc(z) = Kcs + (Kcd-Kcs)*(rho(z)-rhos)/(rhod-rhos)."""
        return Kcs + (Kcd - Kcs) * (self.rho - rhos) / (rhod - rhos)

    def _get_K_executor(self) -> CPUFunctionExecutor:
        if self._K_executor is None:
            self._K_executor = _compile_K(self.N)
        return self._K_executor

    def _get_Cp_executor(self) -> CPUFunctionExecutor:
        if self._Cp_executor is None:
            self._Cp_executor = _compile_Cp(self.N)
        return self._Cp_executor

    def _update_properties(self):
        """Update K(T) and Cp(T) from current temperature using Remora."""
        T_f32 = self.T.astype(np.float32)

        # Compute K(T) via Remora
        Kc_f32 = self.Kc.astype(np.float32)
        result = self._get_K_executor().execute(T_f32, Kc_f32)
        self.K = result.value.astype(np.float64)

        # Compute Cp(T) via Remora
        result = self._get_Cp_executor().execute(T_f32)
        self.Cp = result.value.astype(np.float64)

    def step_crank_nicolson(self) -> np.ndarray:
        """Advance one time step using Crank-Nicolson with Picard iteration.

        Uses Remora-compiled K(T) and Cp(T) inside the Picard loop.
        Returns the new temperature profile T^{t+1}.
        """
        N = self.N
        T_old = self.T.copy()
        dt = self.dt
        dz_sq = self.dz ** 2

        # Initial guess for T^{n+1}: T^n (explicit extrapolation)
        T_guess = T_old.copy()
        T_guess[0] = self._T_surface
        if N > 1:
            T_guess[-1] = T_guess[-2]

        # Picard iteration
        for _it in range(self.picard_max_iter):
            # Compute K(T_guess) and Cp(T_guess) using Remora
            K_guess, Cp_guess = self._compute_properties_for(T_guess)

            # ── Build tridiagonal system ──
            lower = np.zeros(N - 1, dtype=np.float64)
            diag = np.zeros(N, dtype=np.float64)
            upper = np.zeros(N - 1, dtype=np.float64)
            rhs = np.zeros(N, dtype=np.float64)

            # Row 0: Dirichlet BC at surface
            diag[0] = 1.0
            rhs[0] = self._T_surface

            # Interior rows [1, N-2]: CN scheme
            # For node i: a = dt*K[i-1]/(rho[i]*Cp[i]*dz^2), b = dt*K[i]/(...)
            if N > 2:
                rho_int = self.rho[1:-1]
                cp_int = Cp_guess[1:-1]
                denom = rho_int * cp_int * dz_sq + 1e-30
                ai = dt * K_guess[0:N-2] / denom  # coupling to T[i-1], size N-2
                bi = dt * K_guess[1:N-1] / denom  # coupling to T[i+1], size N-2
                ha = 0.5 * ai
                hb = 0.5 * bi

                for i in range(1, N - 1):
                    j = i - 1  # 0-based index into ha/hb
                    lower[i - 1] = -ha[j]
                    diag[i] = 1.0 + ha[j] + hb[j]
                    upper[i] = -hb[j]
                    rhs[i] = (
                        ha[j] * T_old[i - 1]
                        + (1.0 - ha[j] - hb[j]) * T_old[i]
                        + hb[j] * T_old[i + 1]
                    )

            # Row N-1: zero-flux bottom BC
            if N > 1:
                lower[N - 2] = -1.0
                diag[N - 1] = 1.0

            # ── Solve ──
            T_new = thomas_solve(lower, diag, upper, rhs)

            # ── Check convergence ──
            if np.max(np.abs(T_new - T_guess)) < self.picard_tol:
                self.T = T_new
                self._update_properties()
                return self.T.copy()

            T_guess = T_new

        # Max iterations reached — accept last result
        self.T = T_new
        self._update_properties()
        return self.T.copy()

    def _compute_properties_for(
        self, T: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Compute K(T) and Cp(T) for a given temperature array using Remora."""
        T_f32 = T.astype(np.float32)
        Kc_f32 = self.Kc.astype(np.float32)

        K_result = self._get_K_executor().execute(T_f32, Kc_f32)
        K_vals = K_result.value.astype(np.float64)

        Cp_result = self._get_Cp_executor().execute(T_f32)
        Cp_vals = Cp_result.value.astype(np.float64)

        return K_vals, Cp_vals

    def run(self, n_steps: int) -> list[np.ndarray]:
        """Run for n_steps.  Returns history of temperature profiles."""
        history = [self.T.copy()]
        for _ in range(n_steps):
            self.step_crank_nicolson()
            history.append(self.T.copy())
        return history

    def run_to_steady_state(
        self, tol: float = 1e-4, max_steps: int = 10000
    ) -> tuple[int, list[np.ndarray]]:
        """Run until max|dT| < tol.  Returns (n_steps, history)."""
        history = [self.T.copy()]
        for step in range(max_steps):
            T_prev = self.T.copy()
            self.step_crank_nicolson()
            history.append(self.T.copy())
            if np.max(np.abs(self.T - T_prev)) < tol:
                return step + 1, history
        return max_steps, history


# ── Analytical solution ─────────────────────────────────────────────────

def analytical_steady_state(
    z: np.ndarray,
    T_surface: float,
    T_bottom: float | None = None,
) -> np.ndarray:
    """Analytical steady-state solution for uniform 1D heat conduction.

    Zero-flux bottom (insulated): T(z) = T_surface (constant).
    Fixed bottom temperature: linear profile.
    """
    if T_bottom is None:
        return np.full_like(z, T_surface)
    else:
        return T_surface + (T_bottom - T_surface) * z / z[-1]


# ── NumPy reference for validation ──────────────────────────────────────

def _compute_Cp_numpy(T: np.ndarray) -> np.ndarray:
    """Pure-NumPy Cp(T) computation (oracle for Remora comparison)."""
    c0, c1, c2, c3, c4 = CP_COEFFS
    return c0 + c1 * T + c2 * T ** 2 + c3 * T ** 3 + c4 * T ** 4


def _compute_K_numpy(T: np.ndarray, Kc: np.ndarray) -> np.ndarray:
    """Pure-NumPy K(T) computation (oracle for Remora comparison)."""
    return Kc * (1.0 + R350 * T ** 3)
