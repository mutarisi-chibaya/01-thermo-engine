"""Pure Shomate-equation evaluators (NIST WebBook form).

With t = T / 1000 (T in K), the NIST WebBook gives::

    Cp          = A + B t + C t^2 + D t^3 + E / t^2                    [J/(mol K)]
    H - H298    = A t + B t^2/2 + C t^3/3 + D t^4/4 - E/t + F - H      [kJ/mol]
    S           = A ln(t) + B t + C t^2/2 + D t^3/3 - E/(2 t^2) + G    [J/(mol K)]

The functions here work in NIST's own units for one coefficient set. Unit
conversion to SI and range selection happen in :mod:`pyrothermo.species`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float64]


@dataclass(frozen=True)
class ShomateRange:
    """One set of Shomate coefficients, valid for ``Tmin <= T <= Tmax`` (K).

    ``transition`` is True when a real phase change happens at ``Tmax``, so a
    jump in H between this range and the next one is expected.
    ``phase`` optionally overrides the species-level phase for this range.
    """

    Tmin: float
    Tmax: float
    A: float
    B: float
    C: float
    D: float
    E: float
    F: float
    G: float
    H: float
    transition: bool = False
    phase: str | None = None

    def cp(self, T: FloatArray) -> FloatArray:
        """Heat capacity in J/(mol K) for temperatures ``T`` in K."""
        t = T / 1000.0
        return self.A + self.B * t + self.C * t**2 + self.D * t**3 + self.E / t**2

    def h_formation_scale(self, T: FloatArray) -> FloatArray:
        """Enthalpy on the formation scale, ``(H - H298) + H_coef``, in kJ/mol.

        This is ``A t + B t^2/2 + C t^3/3 + D t^4/4 - E/t + F``. At 298.15 K it
        equals the coefficient ``H`` (the NIST standard enthalpy of formation of
        this phase), so differences of this quantity are enthalpy changes.
        """
        t = T / 1000.0
        return (
            self.A * t
            + self.B * t**2 / 2.0
            + self.C * t**3 / 3.0
            + self.D * t**4 / 4.0
            - self.E / t
            + self.F
        )

    def h_minus_h298(self, T: FloatArray) -> FloatArray:
        """``H(T) - H(298.15)`` for this coefficient set alone, in kJ/mol."""
        return self.h_formation_scale(T) - self.H

    def s(self, T: FloatArray) -> FloatArray:
        """Absolute (third-law) entropy in J/(mol K) for temperatures ``T`` in K."""
        t = T / 1000.0
        return (
            self.A * np.log(t)
            + self.B * t
            + self.C * t**2 / 2.0
            + self.D * t**3 / 3.0
            - self.E / (2.0 * t**2)
            + self.G
        )
