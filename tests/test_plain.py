"""Tests for the plain-language layer (app/plain.py)."""

from __future__ import annotations

import pytest

from app.plain import (
    PRESETS,
    PROPERTY_HELP,
    describe_reaction,
    fmt_pct,
    fmt_pressure,
    fmt_T,
    from_display,
    heat_per_tonne_kwh,
    to_display,
)
from pyrothermo import Reaction, Species, default_data_path, load_species

needs_data = pytest.mark.skipif(not default_data_path().is_file(), reason="no real data")


def test_unit_round_trip() -> None:
    assert float(to_display(1273.15, "°C")) == pytest.approx(1000.0)
    assert float(to_display(1273.15, "K")) == pytest.approx(1273.15)
    assert from_display(1000.0, "°C") == pytest.approx(1273.15)
    assert from_display(1273.15, "K") == pytest.approx(1273.15)


def test_fmt_T() -> None:
    assert fmt_T(1273.15, "°C") == "1000 °C (1273 K)"
    assert fmt_T(1273.15, "K") == "1273 K (1000 °C)"


@pytest.mark.parametrize(
    ("x", "text"),
    [
        (0.763, "76.3 %"),
        (0.99999, "more than 99.99 %"),
        (0.00001, "less than 0.01 %"),
        (0.9993, "99.93 %"),
    ],
)
def test_fmt_pct(x: float, text: str) -> None:
    assert fmt_pct(x) == text


def test_fmt_pressure() -> None:
    assert fmt_pressure(0.5) == "0.5 bar"
    assert fmt_pressure(1.23e-12) == "1.2 × 10⁻¹² bar"


def test_property_help_covers_all_properties() -> None:
    from app.streamlit_app import PROPERTIES

    assert set(PROPERTY_HELP) == set(PROPERTIES)


def test_describe_on_fake_data_mentions_direction_and_heat(fake_db: dict[str, Species]) -> None:
    rx = Reaction.from_string("2 Fe + O2 = 2 FeO", fake_db)
    lines = describe_reaction(rx, 800.0, "°C")
    text = " ".join(lines)
    assert "Goes forward" in text  # FAKE data: strongly negative dG
    assert "O2 level" in text  # single gas -> pressure statement
    assert "Heat" in text


@needs_data
def test_every_preset_is_balanced_and_described() -> None:
    db = load_species()
    for p in PRESETS:
        rx = Reaction.from_string(p.equation, db)
        lo, hi = rx.t_range
        T = min(max(1273.15, lo), hi)
        assert describe_reaction(rx, T, "°C"), p.label


@needs_data
def test_hydrogen_preset_wording() -> None:
    db = load_species()
    rx = Reaction.from_string("FeO + H2 = Fe + H2O", db)
    text = " ".join(describe_reaction(rx, 1273.15, "°C"))
    assert "Does not go forward on its own" in text
    assert "more than 76." in text and "H2" in text  # min H2 share ~76 %
    assert "endothermic" in text


@needs_data
def test_boudouard_wording() -> None:
    db = load_species()
    rx = Reaction.from_string("C + CO2 = 2 CO", db)
    text = " ".join(describe_reaction(rx, 1273.15, "°C"))
    assert "Goes forward" in text
    assert "700 °C" in text  # turning point 973 K = 700 °C
    assert "runs forward above" in text


@needs_data
def test_heat_per_tonne_iron() -> None:
    fe = load_species()["Fe"]
    kwh = heat_per_tonne_kwh(fe, 1873.15)  # 25 -> 1600 °C, includes melting
    expected = fe.h_minus_h298(1873.15) / fe.molar_mass * 1000 / 3.6e6
    assert kwh == pytest.approx(expected)
    assert 350 < kwh < 400  # ~375 kWh/t: textbook order for heating + melting steel
