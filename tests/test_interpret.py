from __future__ import annotations

import math
from typing import Any

import numpy as np
import pytest

from pyrothermo import R, Reaction, Species
from pyrothermo.data import default_data_path, parse_species_mapping
from pyrothermo.interpret import (
    gas_pressure_limit,
    gas_requirement,
    ln_k,
    turning_points,
)


def _const(name: str, formula: str, phase: str, s298: float, h_kj: float) -> dict[str, Any]:
    """FAKE species with Cp = 0 so H and S are constant: G = h - T s exactly."""
    return {
        name: {
            "formula": formula,
            "phase": phase,
            "dHf298": h_kj,
            "source": "fake-test-data",
            "ranges": [
                {"Tmin": 300, "Tmax": 3000, "A": 0, "B": 0, "C": 0, "D": 0, "E": 0,
                 "F": h_kj, "G": s298, "H": h_kj}
            ],
        }
    }  # fmt: skip


@pytest.fixture
def toy() -> dict[str, Species]:
    """FAKE species: X, Y gases (both 'H2'), Z gas 'H4', S solid; constant H, S."""
    raw = (
        _const("X", "H2", "gas", 100.0, 0.0)
        | _const("Y", "H2", "gas", 90.0, -20.0)  # X = Y: dH=-20 kJ, dS=-10 J/K
        | _const("Z", "H4", "gas", 150.0, -30.0)
        | _const("S", "H2", "solid", 99.0, -50.0)  # X = S: dS = -1 J/K, dG < 0 to 50000 K
    )
    return {n: Species(d) for n, d in parse_species_mapping(raw).items()}


def test_turning_point_analytic(toy: dict[str, Species]) -> None:
    rx = Reaction.from_string("X = Y", toy)  # dG = -20000 + 10 T -> zero at 2000 K
    (tp,) = turning_points(rx)
    assert pytest.approx(2000.0, abs=1e-4) == tp.T
    assert tp.forward_above is False  # dG < 0 below 2000 K


def test_no_turning_point(toy: dict[str, Species]) -> None:
    assert turning_points(Reaction.from_string("X = S", toy)) == []


def test_gas_requirement_equal_coefficients(toy: dict[str, Species]) -> None:
    rx = Reaction.from_string("X = Y", toy)
    T = 1000.0
    K = math.exp(ln_k(rx, T))
    req = gas_requirement(rx, T)
    assert req is not None
    assert (req.reactant, req.product) == ("X", "Y")
    assert req.product_fraction == pytest.approx(K / (1 + K))
    assert req.min_reactant_fraction == pytest.approx(1 / (1 + K))
    # Equal gas moles on both sides: total pressure does not matter.
    assert gas_requirement(rx, T, 10.0).product_fraction == pytest.approx(req.product_fraction)  # type: ignore[union-attr]


@pytest.mark.parametrize("P", [0.1, 1.0, 5.0])
def test_gas_requirement_unequal_coefficients(toy: dict[str, Species], P: float) -> None:
    rx = Reaction.from_string("2 X = Z", toy)  # K = x_Z / x_X^2 / P
    T = 1500.0
    req = gas_requirement(rx, T, P)
    assert req is not None
    x = req.product_fraction
    assert math.log(x) - 2 * math.log(1 - x) - math.log(P) == pytest.approx(ln_k(rx, T))


def test_gas_requirement_not_applicable(toy: dict[str, Species]) -> None:
    assert gas_requirement(Reaction.from_string("X = S", toy), 1000.0) is None


def test_gas_pressure_limit(toy: dict[str, Species]) -> None:
    rx = Reaction.from_string("X = S", toy)  # K = 1 / p_X
    lim = gas_pressure_limit(rx, 1000.0)
    assert lim is not None
    assert lim.gas == "X"
    assert lim.side == "reactant"
    assert lim.pressure_bar == pytest.approx(math.exp(-ln_k(rx, 1000.0)))
    rev = gas_pressure_limit(Reaction.from_string("S = X", toy), 1000.0)
    assert rev is not None
    assert rev.side == "product"
    assert rev.pressure_bar == pytest.approx(lim.pressure_bar)
    assert gas_pressure_limit(Reaction.from_string("X = Y", toy), 1000.0) is None


def test_ln_k_matches_k(toy: dict[str, Species]) -> None:
    rx = Reaction.from_string("X = Y", toy)
    assert math.exp(ln_k(rx, 900.0)) == pytest.approx(rx.K(900.0))
    assert ln_k(rx, 900.0) == pytest.approx(-float(rx.dG(900.0)) / (R * 900.0))


# --- the same functions on the real data -------------------------------------------

needs_data = pytest.mark.skipif(not default_data_path().is_file(), reason="no real data")


@needs_data
def test_boudouard_turns_near_700C() -> None:
    from pyrothermo import load_species

    rx = Reaction.from_string("C + CO2 = 2 CO", load_species())
    (tp,) = turning_points(rx)
    assert pytest.approx(973.2, abs=0.5) == tp.T
    assert tp.forward_above is True


@needs_data
def test_hydrogen_reduction_of_wustite_needs_mostly_h2() -> None:
    from pyrothermo import load_species

    rx = Reaction.from_string("FeO + H2 = Fe + H2O", load_species())
    req = gas_requirement(rx, 1273.15)
    assert req is not None
    K = 10 ** float(rx.log10K(1273.15))
    assert req.min_reactant_fraction == pytest.approx(1 / (1 + K))
    assert 0.7 < req.min_reactant_fraction < 0.8


@needs_data
def test_iron_oxidation_oxygen_pressure_is_tiny() -> None:
    from pyrothermo import load_species

    rx = Reaction.from_string("2 Fe + O2 = 2 FeO", load_species())
    lim = gas_pressure_limit(rx, 1273.15)
    assert lim is not None
    assert lim.side == "reactant"
    assert lim.pressure_bar == pytest.approx(float(np.exp(rx.dG(1273.15) / (R * 1273.15))))
    assert lim.pressure_bar < 1e-10
