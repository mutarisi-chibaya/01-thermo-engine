"""Least-squares Shomate fit for C (graphite) to the NIST-JANAF C-002 table.

The NIST WebBook gives no Shomate coefficients for graphite, so this project fits
its own. These coefficients are NOT published by NIST; species.yaml labels them so.

Method (two ranges, split at 1000 K, 298.15-4000 K overall):
  1. A..E: linear least squares on Cp, each row weighted by 1/Cp (relative error).
  2. Range 1: F and G fixed so H - H298 = 0 and S = S_JANAF exactly at 298.15 K.
     Range 2: F and G fixed so H and S are continuous with range 1 at 1000 K.
  3. H = 0 (graphite is the reference state of carbon).

Note: the validation rows for graphite come from the same JANAF table that was
fitted, so those tests check the quality of the fit, not independent agreement.
"""

from __future__ import annotations

import numpy as np
from nist_parse import JanafRow

SPLIT_K = 1000.0
T_LO_K = 298.15
T_HI_K = 4000.0


def _cp_basis(t: np.ndarray) -> np.ndarray:
    return np.column_stack([np.ones_like(t), t, t**2, t**3, 1.0 / t**2])


def _h_poly(a: np.ndarray, t: float) -> float:
    return float(a[0] * t + a[1] * t**2 / 2 + a[2] * t**3 / 3 + a[3] * t**4 / 4 - a[4] / t)


def _s_poly(a: np.ndarray, t: float) -> float:
    return float(
        a[0] * np.log(t) + a[1] * t + a[2] * t**2 / 2 + a[3] * t**3 / 3 - a[4] / (2 * t**2)
    )


def fit_graphite(rows: list[JanafRow]) -> list[tuple[float, float, dict[str, float]]]:
    """Return ``[(Tmin, Tmax, {A..H}), ...]`` fitted to JANAF graphite rows."""
    T = np.array([r.T for r in rows])
    cp = np.array([r.Cp for r in rows])
    ref = {r.T: r for r in rows}
    out = []
    anchor: tuple[float, float, float] | None = None  # (t, H formation-scale, S)
    for lo, hi in ((T_LO_K, SPLIT_K), (SPLIT_K, T_HI_K)):
        m = (lo <= T) & (hi >= T)
        t = T[m] / 1000.0
        w = 1.0 / cp[m]
        a = np.linalg.lstsq(_cp_basis(t) * w[:, None], cp[m] * w, rcond=None)[0]
        if anchor is None:
            t0 = T_LO_K / 1000.0
            F = -_h_poly(a, t0)
            G = ref[T_LO_K].S - _s_poly(a, t0)
        else:
            tb, hb, sb = anchor
            F = hb - _h_poly(a, tb)
            G = sb - _s_poly(a, tb)
        tb = hi / 1000.0
        anchor = (tb, _h_poly(a, tb) + F, _s_poly(a, tb) + G)
        coeffs = dict(zip("ABCDE", (float(x) for x in a), strict=True))
        coeffs |= {"F": float(F), "G": float(G), "H": 0.0}
        out.append((lo, hi, coeffs))
    return out
