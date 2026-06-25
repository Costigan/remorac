"""Earth-Sun orbit model for lunar diurnal cycle forcing.

Provides the heliocentric distance r_au(t) and solar declination dec(t)
as functions of time, following the orbital mechanics used in the
reference heat1d model (Hayne et al. 2017, JGR Planets).

The orbit follows Earth's eccentricity (e=0.0167) with a period of one
sidereal year.  Declination varies sinusoidally with the lunar obliquity
(1.54 deg) as the true anomaly advances.
"""

from __future__ import annotations

import math

# ── Earth orbital constants ──────────────────────────────────────────────

_YEAR = 365.256 * 24.0 * 3600.0   # sidereal year [s]
_ECC  = 0.0167                     # orbital eccentricity (Earth)
_OBLIQUITY = 0.026878              # lunar obliquity [rad]  (≈ 1.54°)
_LP = 0.0                          # longitude of perihelion [rad]


# ── Kepler solver ───────────────────────────────────────────────────────

def _eccentric_anomaly(M: float, ecc: float = _ECC) -> float:
    """Solve Kepler's equation  M = E - e*sin(E)  for eccentric anomaly E."""
    E = M
    for _ in range(50):
        dE = (M - E + ecc * math.sin(E)) / (1.0 - ecc * math.cos(E))
        E += dE
        if abs(dE) < 1e-14:
            break
    return E


def _true_from_mean(M: float, ecc: float = _ECC) -> float:
    """Compute true anomaly nu from mean anomaly M."""
    E = _eccentric_anomaly(M, ecc)
    cos_nu = (math.cos(E) - ecc) / (1.0 - ecc * math.cos(E))
    sin_nu = (math.sqrt(1.0 - ecc**2) * math.sin(E)) / (1.0 - ecc * math.cos(E))
    return math.atan2(sin_nu, cos_nu)


def _mean_from_true(nu: float, ecc: float = _ECC) -> float:
    """Compute mean anomaly M from true anomaly nu."""
    cos_nu = math.cos(nu)
    E = math.atan2(math.sqrt(1.0 - ecc**2) * math.sin(nu), ecc + cos_nu)
    return E - ecc * math.sin(E)


# ── Public API ──────────────────────────────────────────────────────────

def orbit_state(t: float, lon: float = 0.0) -> tuple[float, float]:
    """Return (r_au, declination_rad) at time *t* [s] from epoch.

    Parameters
    ----------
    t : float
        Time since epoch [s].  The epoch is defined such that
        at t=0 the true anomaly nu = *lon* (observer longitude).
    lon : float
        Observer longitude [rad].  Ensures nu(0) = lon.

    Returns
    -------
    r_au : float
        Heliocentric distance [AU].
    dec : float
        Solar declination [rad].
    """
    M0 = _mean_from_true(lon, _ECC)
    M = M0 + 2.0 * math.pi * t / _YEAR
    nu = _true_from_mean(M, _ECC)

    r_au = (1.0 - _ECC**2) / (1.0 + _ECC * math.cos(nu))
    dec = math.asin(math.sin(_OBLIQUITY) * math.sin(nu + _LP))

    return r_au, dec
