"""Species thermochemistry from piecewise NIST Shomate coefficients.

Enthalpy convention (see :mod:`pyrothermo.shomate`)::

    H(T) - H298 = [poly(T) + F]_range(T) - H_coef[first range]     [kJ/mol]
    H(T)        = dHf298 + (H(T) - H298)

``poly + F`` is on NIST's formation scale in every range, so differences are
valid across ranges and phases (e.g. solid -> liquid Fe), and the enthalpy
increment is measured from the 298.15 K reference phase as in JANAF. For the
first range this is exactly NIST's ``poly + F - H``.

All public methods take T in K (scalar or NumPy array) and return SI units:
J/(mol K) for Cp and S, J/mol for H and G. Scalars in give floats out.
"""

from __future__ import annotations

from pathlib import Path
from typing import overload

import numpy as np
import numpy.typing as npt

from pyrothermo.data import SpeciesData, load_species_data
from pyrothermo.formula import molar_mass, parse_formula
from pyrothermo.shomate import FloatArray, ShomateRange

T_REF = 298.15  # K


class TemperatureRangeError(ValueError):
    """Raised when a temperature lies outside a species' valid ranges."""


class Species:
    """One species with piecewise Shomate thermochemistry (standard state, pure)."""

    def __init__(self, data: SpeciesData) -> None:
        """Wrap validated species data from :func:`pyrothermo.data.load_species_data`."""
        self.data = data
        self.name = data.name
        self.formula = data.formula
        self.composition: dict[str, float] = parse_formula(data.formula)
        self._tmax = np.array([r.Tmax for r in data.ranges], dtype=float)
        self._h_ref_kj = data.ranges[0].H  # formation-scale H of the 298 K phase

    def __repr__(self) -> str:
        return f"Species({self.name!r}, {self.t_min:g}-{self.t_max:g} K)"

    @property
    def ranges(self) -> tuple[ShomateRange, ...]:
        """Shomate ranges in increasing temperature."""
        return self.data.ranges

    @property
    def t_min(self) -> float:
        """Lowest valid temperature in K."""
        return self.data.ranges[0].Tmin

    @property
    def t_max(self) -> float:
        """Highest valid temperature in K."""
        return self.data.ranges[-1].Tmax

    @property
    def molar_mass(self) -> float:
        """Molar mass in kg/mol."""
        return molar_mass(self.formula)

    def range_phase(self, r: ShomateRange) -> str:
        """Phase label of a range (its own ``phase`` or the species-level one)."""
        return r.phase if r.phase is not None else self.data.phase

    def phase_at(self, T: float) -> str:
        """Phase label of the range used at temperature ``T`` (K)."""
        idx = int(self._range_index(np.asarray([T], dtype=float))[0])
        return self.range_phase(self.data.ranges[idx])

    def phase_transitions(self) -> list[tuple[float, str, str]]:
        """Boundaries marked as real phase changes.

        A boundary counts if the lower range has ``transition: true`` or the
        phase label changes across it.

        Returns:
            List of ``(T in K, phase below, phase above)``.
        """
        out = []
        for a, b in zip(self.ranges, self.ranges[1:], strict=False):
            pa, pb = self.range_phase(a), self.range_phase(b)
            if a.transition or pa != pb:
                out.append((a.Tmax, pa, pb))
        return out

    # -- internals -------------------------------------------------------

    def _range_index(self, T: FloatArray) -> npt.NDArray[np.intp]:
        if T.size == 0:
            return np.zeros(T.shape, dtype=np.intp)
        if not np.all(np.isfinite(T)):
            raise TemperatureRangeError(f"{self.name}: temperatures must be finite")
        lo, hi = float(np.min(T)), float(np.max(T))
        if lo < self.t_min or hi > self.t_max:
            spans = ", ".join(f"{r.Tmin:g}-{r.Tmax:g}" for r in self.ranges)
            raise TemperatureRangeError(
                f"{self.name}: T must be within {self.t_min:g}-{self.t_max:g} K "
                f"(ranges: {spans} K); got {lo:g}-{hi:g} K"
            )
        # First range with Tmax >= T, so an exact shared boundary uses the lower range.
        return np.searchsorted(self._tmax, T, side="left")

    def _piecewise(self, T: npt.ArrayLike, which: str) -> FloatArray:
        Tarr = np.asarray(T, dtype=float)
        idx = self._range_index(Tarr)
        out = np.empty_like(Tarr)
        for i, r in enumerate(self.ranges):
            mask = idx == i
            if np.any(mask):
                out[mask] = getattr(r, which)(Tarr[mask])
        return out

    @staticmethod
    def _out(values: FloatArray, T: npt.ArrayLike) -> float | FloatArray:
        return float(values) if np.ndim(T) == 0 else values

    # -- properties ------------------------------------------------------

    @overload
    def cp(self, T: float) -> float: ...
    @overload
    def cp(self, T: FloatArray) -> FloatArray: ...
    def cp(self, T: float | FloatArray) -> float | FloatArray:
        """Heat capacity Cp in J/(mol K) at ``T`` in K."""
        return self._out(self._piecewise(T, "cp"), T)

    @overload
    def h_minus_h298(self, T: float) -> float: ...
    @overload
    def h_minus_h298(self, T: FloatArray) -> FloatArray: ...
    def h_minus_h298(self, T: float | FloatArray) -> float | FloatArray:
        """Enthalpy increment H(T) - H(298.15 K) in J/mol at ``T`` in K.

        Measured from the reference phase at 298.15 K, so it includes latent
        heats of any phase changes below T (as in the JANAF tables).
        """
        kj = self._piecewise(T, "h_formation_scale") - self._h_ref_kj
        return self._out(kj * 1000.0, T)

    @overload
    def h(self, T: float) -> float: ...
    @overload
    def h(self, T: FloatArray) -> FloatArray: ...
    def h(self, T: float | FloatArray) -> float | FloatArray:
        """Absolute enthalpy H(T) = dHf298 + (H(T) - H298) in J/mol at ``T`` in K."""
        return self._out(self.data.dHf298 * 1000.0 + np.asarray(self.h_minus_h298(T)), T)

    @overload
    def s(self, T: float) -> float: ...
    @overload
    def s(self, T: FloatArray) -> FloatArray: ...
    def s(self, T: float | FloatArray) -> float | FloatArray:
        """Absolute (third-law) entropy S in J/(mol K) at ``T`` in K."""
        return self._out(self._piecewise(T, "s"), T)

    @overload
    def g(self, T: float) -> float: ...
    @overload
    def g(self, T: FloatArray) -> FloatArray: ...
    def g(self, T: float | FloatArray) -> float | FloatArray:
        """Gibbs energy G = H - T S in J/mol at ``T`` in K."""
        Tarr = np.asarray(T, dtype=float)
        return self._out(np.asarray(self.h(Tarr)) - Tarr * np.asarray(self.s(Tarr)), T)


def load_species(path: str | Path | None = None) -> dict[str, Species]:
    """Load every species in a YAML file as :class:`Species` objects.

    Args:
        path: YAML file; defaults to ``01-thermo-engine/data/species.yaml``.

    Returns:
        Mapping of species name to :class:`Species`.
    """
    return {name: Species(d) for name, d in load_species_data(path).items()}
