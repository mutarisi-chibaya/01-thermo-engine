"""Reactions: parsing, element balance, and dH, dS, dG, K as functions of T.

Equations are strings such as ``"2 Fe + O2 = 2 FeO"`` or
``"4/3 Fe + O2 = 2/3 Fe2O3"`` (``->`` also works). Coefficients may be
integers, decimals or fractions and are stored as exact
:class:`fractions.Fraction`. Names must match keys of the species mapping
(normally ``species.yaml``), and ``+`` must be surrounded by whitespace.
Every species is in its standard state: pure condensed phases, ideal gases
at 1 bar. Results are per mole of reaction as written.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from fractions import Fraction
from typing import overload

import numpy as np
import numpy.typing as npt

from pyrothermo.shomate import FloatArray
from pyrothermo.species import Species

R = 8.314462618  # J/(mol K), molar gas constant (CODATA 2018, exact by SI definition)

_TERM = re.compile(r"^\s*(\d+(?:\.\d+)?(?:/\d+)?)?\s*([A-Za-z(]\S*)\s*$")
_BALANCE_TOL = 1e-9  # mol of element; formulas may have decimal subscripts


class ReactionError(ValueError):
    """Raised for unparseable, unbalanced or unknown-species reactions."""


def _parse_side(side: str, equation: str) -> dict[str, Fraction]:
    terms: dict[str, Fraction] = {}
    for raw in re.split(r"\s\+\s", f" {side} "):
        m = _TERM.match(raw)
        if m is None:
            raise ReactionError(f"cannot parse term {raw.strip()!r} in {equation!r}")
        coef = Fraction(m.group(1)) if m.group(1) else Fraction(1)
        if coef <= 0:
            raise ReactionError(f"coefficients must be positive in {equation!r}")
        terms[m.group(2)] = terms.get(m.group(2), Fraction(0)) + coef
    return terms


def _fmt(coef: Fraction) -> str:
    return "" if coef == 1 else f"{coef} "


class Reaction:
    """A balanced reaction between species in their standard states."""

    def __init__(
        self,
        reactants: Mapping[str, Fraction | int],
        products: Mapping[str, Fraction | int],
        species: Mapping[str, Species],
    ) -> None:
        """Build and check a reaction.

        Args:
            reactants: Species name -> positive stoichiometric coefficient.
            products: Species name -> positive stoichiometric coefficient.
            species: Species lookup, e.g. from :func:`pyrothermo.load_species`.

        Raises:
            ReactionError: if a species is unknown or the elements do not balance.
        """
        self.reactants = {k: Fraction(v) for k, v in reactants.items()}
        self.products = {k: Fraction(v) for k, v in products.items()}
        if not self.reactants or not self.products:
            raise ReactionError("a reaction needs at least one reactant and one product")
        missing = sorted({n for n in (*self.reactants, *self.products) if n not in species})
        if missing:
            raise ReactionError(
                f"species not in species data: {', '.join(missing)} (add them to species.yaml)"
            )
        self.species = {n: species[n] for n in (*self.reactants, *self.products)}
        # Signed coefficients: products positive, reactants negative.
        self.nu: dict[str, Fraction] = {}
        for n, c in self.reactants.items():
            self.nu[n] = self.nu.get(n, Fraction(0)) - c
        for n, c in self.products.items():
            self.nu[n] = self.nu.get(n, Fraction(0)) + c
        self._check_balance()

    @classmethod
    def from_string(cls, equation: str, species: Mapping[str, Species]) -> Reaction:
        """Parse an equation like ``"2 Fe + O2 = 2 FeO"`` and check its balance."""
        sides = re.split(r"=|->", equation)
        if len(sides) != 2:
            raise ReactionError(f"need exactly one '=' (or '->') in {equation!r}")
        return cls(_parse_side(sides[0], equation), _parse_side(sides[1], equation), species)

    def element_imbalance(self) -> dict[str, float]:
        """Element totals, products minus reactants (mol); zero for a balanced reaction."""
        total: dict[str, float] = {}
        for name, nu in self.nu.items():
            for el, n in self.species[name].composition.items():
                total[el] = total.get(el, 0.0) + float(nu) * n
        return total

    def _check_balance(self) -> None:
        off = {el: v for el, v in self.element_imbalance().items() if abs(v) > _BALANCE_TOL}
        if off:
            detail = ", ".join(f"{el}: {v:+g}" for el, v in sorted(off.items()))
            raise ReactionError(f"unbalanced reaction {self}: {detail} (products - reactants)")

    def __str__(self) -> str:
        lhs = " + ".join(f"{_fmt(c)}{n}" for n, c in self.reactants.items())
        rhs = " + ".join(f"{_fmt(c)}{n}" for n, c in self.products.items())
        return f"{lhs} = {rhs}"

    def __repr__(self) -> str:
        return f"Reaction({str(self)!r})"

    @property
    def t_range(self) -> tuple[float, float]:
        """Temperature interval (K) where every species has data."""
        return (
            max(s.t_min for s in self.species.values()),
            min(s.t_max for s in self.species.values()),
        )

    def phase_transitions(self) -> list[tuple[float, str, str, str]]:
        """Phase changes of participating species strictly inside :attr:`t_range`.

        Returns:
            Sorted list of ``(T in K, species, phase below, phase above)``.
        """
        lo, hi = self.t_range
        return sorted(
            (T, name, a, b)
            for name, sp in self.species.items()
            for T, a, b in sp.phase_transitions()
            if lo < T < hi
        )

    def _sum(self, prop: str, T: npt.ArrayLike) -> FloatArray:
        Tarr = np.asarray(T, dtype=float)
        total = np.zeros_like(Tarr)
        for name, nu in self.nu.items():
            total = total + float(nu) * np.asarray(getattr(self.species[name], prop)(Tarr))
        return total

    @staticmethod
    def _out(values: FloatArray, T: npt.ArrayLike) -> float | FloatArray:
        return float(values) if np.ndim(T) == 0 else values

    @overload
    def dCp(self, T: float) -> float: ...
    @overload
    def dCp(self, T: FloatArray) -> FloatArray: ...
    def dCp(self, T: float | FloatArray) -> float | FloatArray:
        """Reaction heat-capacity change in J/(mol K) at ``T`` in K."""
        return self._out(self._sum("cp", T), T)

    @overload
    def dH(self, T: float) -> float: ...
    @overload
    def dH(self, T: FloatArray) -> FloatArray: ...
    def dH(self, T: float | FloatArray) -> float | FloatArray:
        """Standard reaction enthalpy in J/mol at ``T`` in K."""
        return self._out(self._sum("h", T), T)

    @overload
    def dS(self, T: float) -> float: ...
    @overload
    def dS(self, T: FloatArray) -> FloatArray: ...
    def dS(self, T: float | FloatArray) -> float | FloatArray:
        """Standard reaction entropy in J/(mol K) at ``T`` in K."""
        return self._out(self._sum("s", T), T)

    @overload
    def dG(self, T: float) -> float: ...
    @overload
    def dG(self, T: FloatArray) -> FloatArray: ...
    def dG(self, T: float | FloatArray) -> float | FloatArray:
        """Standard reaction Gibbs energy in J/mol at ``T`` in K."""
        return self._out(self._sum("g", T), T)

    @overload
    def log10K(self, T: float) -> float: ...
    @overload
    def log10K(self, T: FloatArray) -> FloatArray: ...
    def log10K(self, T: float | FloatArray) -> float | FloatArray:
        """Base-10 log of the equilibrium constant, ``-dG / (R T ln 10)``, at ``T`` in K.

        Use this for plotting; K itself overflows for strongly favoured reactions.
        """
        Tarr = np.asarray(T, dtype=float)
        return self._out(-self._sum("g", Tarr) / (R * Tarr * np.log(10.0)), T)

    @overload
    def K(self, T: float) -> float: ...
    @overload
    def K(self, T: FloatArray) -> FloatArray: ...
    def K(self, T: float | FloatArray) -> float | FloatArray:
        """Equilibrium constant ``K = exp(-dG / (R T))`` (dimensionless) at ``T`` in K."""
        Tarr = np.asarray(T, dtype=float)
        return self._out(np.exp(-self._sum("g", Tarr) / (R * Tarr)), T)
