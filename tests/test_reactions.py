"""Logic tests for Reaction using FAKE closed-form coefficients (tests/fixtures)."""

from __future__ import annotations

from fractions import Fraction

import numpy as np
import pytest

from pyrothermo import R, Reaction, ReactionError, Species


def test_parse_and_coefficients(fake_db: dict[str, Species]) -> None:
    rx = Reaction.from_string("2 Fe + O2 = 2 FeO", fake_db)
    assert rx.reactants == {"Fe": 2, "O2": 1}
    assert rx.products == {"FeO": 2}
    assert rx.nu == {"Fe": -2, "O2": -1, "FeO": 2}
    assert str(rx) == "2 Fe + O2 = 2 FeO"


@pytest.mark.parametrize(
    "eq", ["Fe + 0.5 O2 = FeO", "Fe + 1/2 O2 -> FeO", "  Fe +  1/2 O2=FeO  ", "2Fe + O2 = 2FeO"]
)
def test_coefficient_formats(fake_db: dict[str, Species], eq: str) -> None:
    rx = Reaction.from_string(eq, fake_db)
    assert rx.nu["FeO"] / -rx.nu["O2"] == 2


def test_fraction_is_exact(fake_db: dict[str, Species]) -> None:
    rx = Reaction.from_string("Fe + 1/2 O2 = FeO", fake_db)
    assert rx.reactants["O2"] == Fraction(1, 2)


@pytest.mark.parametrize("eq", ["Fe + O2 = FeO", "2 Fe + O2 = FeO", "FeO = Fe"])
def test_unbalanced_raises(fake_db: dict[str, Species], eq: str) -> None:
    with pytest.raises(ReactionError, match="unbalanced"):
        Reaction.from_string(eq, fake_db)


def test_unbalanced_message_names_element(fake_db: dict[str, Species]) -> None:
    with pytest.raises(ReactionError, match=r"O: -1"):
        Reaction.from_string("Fe + O2 = FeO", fake_db)


@pytest.mark.parametrize(
    ("eq", "match"),
    [
        ("Fe + O2", "exactly one"),
        ("Fe = FeO = Fe", "exactly one"),
        ("2 Fe + O2 = 2 Fe2O3", "Fe2O3"),
        ("0 Fe + O2 = O2", "positive"),
        ("Fe + = FeO", "cannot parse"),
    ],
)
def test_parse_errors(fake_db: dict[str, Species], eq: str, match: str) -> None:
    with pytest.raises(ReactionError, match=match):
        Reaction.from_string(eq, fake_db)


def test_dh_ds_dg_from_species(fake_db: dict[str, Species]) -> None:
    rx = Reaction.from_string("2 Fe + O2 = 2 FeO", fake_db)
    fe, o2, feo = fake_db["Fe"], fake_db["O2"], fake_db["FeO"]
    T = 800.0
    assert rx.dH(T) == pytest.approx(2 * feo.h(T) - 2 * fe.h(T) - o2.h(T))
    assert rx.dS(T) == pytest.approx(2 * feo.s(T) - 2 * fe.s(T) - o2.s(T))
    assert rx.dCp(T) == pytest.approx(2 * 50 - 2 * 25 - 30)
    assert rx.dG(T) == pytest.approx(rx.dH(T) - T * rx.dS(T))


def test_dh_at_298_is_formation_enthalpy(fake_db: dict[str, Species]) -> None:
    rx = Reaction.from_string("Fe + 1/2 O2 = FeO", fake_db)
    assert rx.dH(298.15) == pytest.approx(-270_000.0)


def test_k_definition(fake_db: dict[str, Species]) -> None:
    rx = Reaction.from_string("2 FeO = 2 Fe + O2", fake_db)
    T = np.array([500.0, 1000.0, 1500.0])
    np.testing.assert_allclose(rx.K(T), np.exp(-rx.dG(T) / (R * T)))
    np.testing.assert_allclose(rx.log10K(T), np.log10(rx.K(T)))


def test_reverse_reaction_inverts_k(fake_db: dict[str, Species]) -> None:
    fwd = Reaction.from_string("2 Fe + O2 = 2 FeO", fake_db)
    rev = Reaction.from_string("2 FeO = 2 Fe + O2", fake_db)
    assert fwd.dG(1200.0) == pytest.approx(-rev.dG(1200.0))
    assert fwd.log10K(1200.0) == pytest.approx(-rev.log10K(1200.0))


def test_vectorised_and_scalar(fake_db: dict[str, Species]) -> None:
    rx = Reaction.from_string("2 Fe + O2 = 2 FeO", fake_db)
    assert isinstance(rx.dG(1000.0), float)
    out = rx.dG(np.linspace(300.0, 1900.0, 7))
    assert isinstance(out, np.ndarray)
    assert out.shape == (7,)


def test_t_range_and_transitions(fake_db: dict[str, Species]) -> None:
    rx = Reaction.from_string("2 Fe + O2 = 2 FeO", fake_db)
    assert rx.t_range == (298.0, 2000.0)
    assert rx.phase_transitions() == [(1000.0, "Fe", "solid", "liquid")]


def test_outside_range_propagates(fake_db: dict[str, Species]) -> None:
    rx = Reaction.from_string("2 Fe + O2 = 2 FeO", fake_db)
    with pytest.raises(ValueError, match="Fe"):
        rx.dG(2500.0)
