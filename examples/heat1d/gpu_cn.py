"""GPU-accelerated Crank-Nicolson step.

Decomposes the CN assembly + Thomas tridiagonal solve into individual
Remora GPU kernels and chains them on the host.

Phase 1 — GPU Dense-Subset Completion Plan.
"""

from __future__ import annotations

import numpy as np

from remora.compiler import compile_function_source_to_mlir_gpu_ptx
from remora.runtime import CUDARuntime
from remora.executor import RemoraExecutor
from remora.types import FLOAT, ArrayType, StaticDim


def _arr(*dims: int) -> ArrayType:
    return ArrayType(FLOAT, tuple(StaticDim(d) for d in dims))


# ═══════════════════════════════════════════════════════════════════════════
# Remora source templates — {N}/{N1}/{N2} interpolated at compile time
# ═══════════════════════════════════════════════════════════════════════════

_T_HA = """(define/pi ()
 (compute_ha [g1 (Array Float {N2}) K (Array Float {N2})
              rho (Array Float {N}) Cp (Array Float {N}) dt Float]
  (Array Float {N2}))
 (map (lambda (i) (/ (* 0.5 dt (index-item g1 i) (index-item K i))
                     (* (index-item rho (+ i 1)) (index-item Cp (+ i 1)))))
      (iota {N2})))
"""

_T_HB = """(define/pi ()
 (compute_hb [g2 (Array Float {N2}) K (Array Float {N}) Cp (Array Float {N})
              rho (Array Float {N}) dt Float]
  (Array Float {N2}))
 (map (lambda (i) (/ (* 0.5 dt (index-item g2 i) (index-item K (+ i 1)))
                     (* (index-item rho (+ i 1)) (index-item Cp (+ i 1)))))
      (iota {N2})))
"""

_T_LOWER = """(define/pi ()
 (compute_lower [ha (Array Float {N2})] (Array Float {N1}))
 (map (lambda (i) (if (< i {N2}) (- 0.0 (index-item ha i)) -1.0))
      (iota {N1})))
"""

_T_UPPER = """(define/pi ()
 (compute_upper [hb (Array Float {N2})] (Array Float {N1}))
 (map (lambda (i) (if (< i 1) 0.0 (- 0.0 (index-item hb (- i 1)))))
      (iota {N1})))
"""

_T_DIAG = """(define/pi ()
 (compute_diag [ha (Array Float {N2}) hb (Array Float {N2})] (Array Float {N}))
 (map (lambda (i)
   (if (< i 1) 1.0
     (if (< i {N1}) (+ 1.0 (index-item ha (- i 1)) (index-item hb (- i 1)))
       1.0)))
   (iota {N})))
"""

_T_RHS = """(define/pi ()
 (compute_rhs [ha (Array Float {N2}) hb (Array Float {N2})
               T_old (Array Float {N}) T_surf Float]
  (Array Float {N}))
 (map (lambda (i)
   (if (< i 1) T_surf
     (if (< i {N1})
         (let ((ha_i (index-item ha (- i 1)))
               (hb_i (index-item hb (- i 1))))
           (+ (* ha_i (index-item T_old (- i 1)))
              (* (- 1.0 ha_i hb_i) (index-item T_old i))
              (* hb_i (index-item T_old (+ i 1)))))
         0.0)))
   (iota {N})))
"""

# ── Combined-input maps for scan access (avoid multi-parameter scan limitation) ──

_T_CP_TABLE = """(define/pi ()
 (compute_cp_table [upper (Array Float {N1}) diag (Array Float {N})
                    lower (Array Float {N1})]
  (Array Float {N1}))
 (map (lambda (i)
   (let ((u (index-item upper i))
         (d (index-item diag i))
         (l (if (< i 1) 0.0 (index-item lower (- i 1)))))
     (/ u d)))
   (iota {N1})))
"""

_T_DP_TABLE = """(define/pi ()
 (compute_dp_table [rhs (Array Float {N}) m (Array Float {N})
                    lower (Array Float {N1})]
  (Array Float {N}))
 (map (lambda (i)
   (let ((r  (index-item rhs i))
         (mi (index-item m i))
         (l  (if (< i 1) 0.0 (index-item lower (- i 1)))))
     (/ r mi)))
   (iota {N})))
"""

# ── M (diag - lower[i-1] * cp[i-1]) ──

_T_M = """(define/pi ()
 (compute_m [diag (Array Float {N}) lower (Array Float {N1})
             cp (Array Float {N1})]
  (Array Float {N}))
 (map (lambda (i)
   (let ((d  (index-item diag i))
         (l  (if (< i 1) 0.0 (index-item lower (- i 1))))
         (pv (if (< i 1) 0.0 (index-item cp (- i 1)))))
     (- d (* l pv))))
   (iota {N})))
"""

# ── Scans: single array, simple arithmetic (no captured arrays) ──

_T_CP_SCAN = """(define/pi ()
 (compute_cp_scan [cp_table (Array Float {N1}) lower (Array Float {N1})]
  (Array Float {N1}))
 (iscan (lambda (prev i)
   (let ((ct (index-item cp_table i))
         (l  (if (< i 1) 0.0 (index-item lower (- i 1)))))
     (* ct (- 1.0 (* l prev)))))
   0.0 (iota {N1})))
"""

_T_DP_SCAN = """(define/pi ()
 (compute_dp_scan [dp_table (Array Float {N}) lower (Array Float {N1})]
  (Array Float {N}))
 (iscan (lambda (prev i)
   (let ((dt (index-item dp_table i))
         (l  (if (< i 1) 0.0 (index-item lower (- i 1)))))
     (- dt (* (* l prev) dt))))  ; simplified: (r - l*prev)/mi pre-computed
   0.0 (iota {N})))
"""

_T_BACKSUB = """(define/pi ()
 (compute_backsub [dp (Array Float {N}) cp (Array Float {N1})]
  (Array Float {N}))
 (trace-right (lambda (xnext i)
   (let ((dpi (index-item dp i))
         (cpi (if (< i {N1}) (index-item cp i) 0.0)))
     (- dpi (* cpi xnext))))
   0.0 (iota {N})))
"""

_SOURCES = {
    "compute_ha": _T_HA, "compute_hb": _T_HB,
    "compute_lower": _T_LOWER, "compute_upper": _T_UPPER,
    "compute_diag": _T_DIAG, "compute_rhs": _T_RHS,
    "compute_cp_table": _T_CP_TABLE,
    "compute_dp_table": _T_DP_TABLE,
    "compute_m": _T_M,
    "compute_cp_scan": _T_CP_SCAN,
    "compute_dp_scan": _T_DP_SCAN,
    "compute_backsub": _T_BACKSUB,
}


# ═══════════════════════════════════════════════════════════════════════════
# GPU CN step
# ═══════════════════════════════════════════════════════════════════════════

def gpu_cn_step(
    T_old: np.ndarray,
    g1: np.ndarray,
    g2: np.ndarray,
    rho: np.ndarray,
    K: np.ndarray,
    Cp: np.ndarray,
    dt: float,
    T_surf: float,
) -> np.ndarray:
    """Run one Crank-Nicolson time step on GPU.

    Parameters
    ----------
    T_old, rho, K, Cp : ndarray[N], float
    g1, g2 : ndarray[N-2], float
    dt, T_surf : float

    Returns
    -------
    ndarray[N], float64
    """
    N = len(T_old)
    N1 = N - 1
    N2 = N - 2
    runtime = CUDARuntime()

    def _f32(a):
        return np.asarray(a, dtype=np.float32)
    def _s(v):
        return np.float32(v)

    Tf = _f32(T_old); g1f = _f32(g1); g2f = _f32(g2)
    rf = _f32(rho); Kf = _f32(K); Cpf = _f32(Cp)

    def _k(name: str, ptypes: tuple, inputs: list) -> np.ndarray:
        src = _SOURCES[name].format(N=N, N1=N1, N2=N2)
        ptx, kernels, _ = compile_function_source_to_mlir_gpu_ptx(
            src, name, ptypes, include_prelude=False, kernel_name=name, syntax="lisp")
        ex = RemoraExecutor(ptx, kernels, runtime=runtime)
        try:
            return np.asarray(ex.execute(name, list(inputs)), dtype=np.float32)
        finally:
            ex.close()

    # Assembly
    ha = _k("compute_ha", (_arr(N2), _arr(N2), _arr(N), _arr(N), FLOAT),
            [g1f, Kf, rf, Cpf, _s(dt)])
    hb = _k("compute_hb", (_arr(N2), _arr(N), _arr(N), _arr(N), FLOAT),
            [g2f, Kf, Cpf, rf, _s(dt)])
    lower = _k("compute_lower", (_arr(N2),), [ha])
    upper = _k("compute_upper", (_arr(N2),), [hb])
    diag = _k("compute_diag", (_arr(N2), _arr(N2)), [ha, hb])
    rhs = _k("compute_rhs", (_arr(N2), _arr(N2), _arr(N), FLOAT),
             [ha, hb, Tf, _s(T_surf)])

    # Thomas sweeps
    cp_table = _k("compute_cp_table", (_arr(N1), _arr(N), _arr(N1)),
                  [upper, diag, lower])
    cp_scan = _k("compute_cp_scan", (_arr(N1), _arr(N1)),
                  [cp_table, lower])

    m = _k("compute_m", (_arr(N), _arr(N1), _arr(N1)),
           [diag, lower, cp_scan])

    dp_table = _k("compute_dp_table", (_arr(N), _arr(N), _arr(N1)),
                  [rhs, m, lower])
    dp_scan = _k("compute_dp_scan", (_arr(N), _arr(N1)),
                  [dp_table, lower])

    T_new = _k("compute_backsub", (_arr(N), _arr(N1)),
               [dp_scan, cp_scan])

    return T_new.astype(np.float64)
