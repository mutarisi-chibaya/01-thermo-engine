"""Plain-language interpretation of reaction thermodynamics.

Turns dG(T) and K(T) into statements a non-specialist can act on:

- direction: does the reaction run forward at T with every gas at 1 bar?
- turning points: temperatures where dG = 0 and the direction reverses;
- gas requirement: for one gas in and one gas out (rest pure condensed phases),
  the equilibrium gas composition at a total pressure (e.g. the minimum H2 share
  needed to reduce FeO);
- limiting pressure: for a single gas (e.g. O2 in an oxidation), the equilibrium
  partial pressure above/below which the reaction runs forward.

Assumptions (also stated to the user): pure condensed phases (activity 1), ideal
gases, equilibrium only - nothing about rates. Temperatures in K, pressures in bar.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import numpy as np

from pyrothermo.reactions import R, Reaction

GAS = "gas"


@dataclass(frozen=True)
class TurningPoint:
    """A temperature (K) where dG changes sign.

    ``forward_above`` is True if the reaction runs forward above this temperature.
    """

    T: float
    forward_above: bool


@dataclass(frozen=True)
class GasRequirement:
    """Equilibrium gas composition for one gas reactant and one gas product.

    ``product_fraction`` is the equilibrium mole fraction of ``product`` in the
    two-gas mixture at ``pressure_bar``. The reaction runs forward while the
    product share is below it, i.e. while the reactant share exceeds
    ``1 - product_fraction``.
    """

    reactant: str
    product: str
    product_fraction: float
    pressure_bar: float

    @property
    def min_reactant_fraction(self) -> float:
        """Smallest reactant-gas mole fraction for which the reaction runs forward."""
        return 1.0 - self.product_fraction


@dataclass(frozen=True)
class GasPressureLimit:
    """Equilibrium partial pressure (bar) of the only gas in the reaction.

    ``side`` says where the gas is. A reactant gas (e.g. O2 in an oxidation) drives
    the reaction forward when its pressure is *above* ``pressure_bar``; a product
    gas (e.g. CO2 from a carbonate) allows it forward when *below*.
    """

    gas: str
    side: Literal["reactant", "product"]
    pressure_bar: float


def _gases(rx: Reaction, T: float) -> dict[str, float]:
    """Signed coefficients of the gas-phase species at T (products +)."""
    return {n: float(nu) for n, nu in rx.nu.items() if rx.species[n].phase_at(T) == GAS}


def ln_k(rx: Reaction, T: float) -> float:
    """Natural log of K at ``T`` (K); avoids overflow of K itself."""
    return -float(rx.dG(T)) / (R * T)


def turning_points(rx: Reaction, n: int = 2000) -> list[TurningPoint]:
    """Temperatures within the reaction's valid range where dG crosses zero.

    Args:
        rx: Reaction.
        n: Number of grid points used to bracket sign changes before bisection.

    Returns:
        Turning points in increasing temperature, each refined to ~1e-6 K.
    """
    lo, hi = rx.t_range
    T = np.linspace(lo, hi, n)
    g = np.asarray(rx.dG(T))
    out = []
    for i in np.nonzero(np.sign(g[:-1]) * np.sign(g[1:]) < 0)[0]:
        a, b = float(T[i]), float(T[i + 1])
        ga = float(g[i])
        for _ in range(60):
            m = 0.5 * (a + b)
            gm = float(rx.dG(m))
            if (gm < 0) == (ga < 0):
                a, ga = m, gm
            else:
                b = m
        out.append(TurningPoint(0.5 * (a + b), forward_above=float(g[i]) > 0))
    return out


def gas_requirement(rx: Reaction, T: float, pressure_bar: float = 1.0) -> GasRequirement | None:
    """Equilibrium gas split for reactions with exactly one gas on each side.

    Solves ``x^a / (1 - x)^b * P^(a - b) = K`` for the product-gas fraction x, where
    a and b are the product and reactant gas coefficients and P is the total pressure.

    Returns:
        :class:`GasRequirement`, or None if the reaction is not of this form at T.
    """
    gases = _gases(rx, T)
    reactants = [(n, -nu) for n, nu in gases.items() if nu < 0]
    products = [(n, nu) for n, nu in gases.items() if nu > 0]
    if len(reactants) != 1 or len(products) != 1:
        return None
    (r_name, b), (p_name, a) = reactants[0], products[0]
    target = ln_k(rx, T) - (a - b) * math.log(pressure_bar)

    def f(y: float) -> float:  # y = logit(x); f increases monotonically with y
        log_x = -math.log1p(math.exp(-y)) if y > 0 else y - math.log1p(math.exp(y))
        log_1mx = -math.log1p(math.exp(y)) if y < 0 else -y - math.log1p(math.exp(-y))
        return a * log_x - b * log_1mx - target

    lo, hi = -700.0, 700.0
    if f(lo) > 0:
        x = 0.0
    elif f(hi) < 0:
        x = 1.0
    else:
        for _ in range(200):
            mid = 0.5 * (lo + hi)
            if f(mid) < 0:
                lo = mid
            else:
                hi = mid
        y = 0.5 * (lo + hi)
        x = 1.0 / (1.0 + math.exp(-y))
    return GasRequirement(r_name, p_name, x, pressure_bar)


def gas_pressure_limit(rx: Reaction, T: float) -> GasPressureLimit | None:
    """Equilibrium partial pressure for reactions with a single gas species.

    With all condensed phases pure, ``K = p^nu`` (p in bar, nu signed), so
    ``p = K^(1/nu)``.

    Returns:
        :class:`GasPressureLimit`, or None if the reaction does not have exactly one gas.
    """
    gases = _gases(rx, T)
    if len(gases) != 1:
        return None
    ((name, nu),) = gases.items()
    p = math.exp(ln_k(rx, T) / nu)
    return GasPressureLimit(name, "product" if nu > 0 else "reactant", p)
