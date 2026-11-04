"""Logic tests for Species using FAKE closed-form coefficients (tests/fixtures)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from pyrothermo import Species, TemperatureRangeError

T0 = 0.29815  # 298.15 K / 1000


def test_cp_constant(fake_db: dict[str, Species]) -> None:
    fe = fake_db["Fe"]
    assert fe.cp(500.0) == pytest.approx(25.0)
    assert fe.cp(1500.0) == pytest.approx(40.0)


def test_h_minus_h298_and_absolute_h(fake_db: dict[str, Species]) -> None:
    feo = fake_db["FeO"]
    assert feo.h_minus_h298(298.15) == pytest.approx(0.0, abs=1e-9)
    assert feo.h_minus_h298(1000.0) == pytest.approx(50.0 * (1.0 - T0) * 1000.0)
    assert feo.h(298.15) == pytest.approx(-270_000.0)
    assert feo.h(1000.0) == pytest.approx(-270_000.0 + 50.0 * (1.0 - T0) * 1000.0)


def test_h_across_phase_change_includes_latent_heat(fake_db: dict[str, Species]) -> None:
    fe = fake_db["Fe"]
    # solid 298->1000 K, +10 kJ jump, then liquid Cp=40 from 1000->1500 K.
    expected_kj = 25.0 * (1.0 - T0) + 10.0 + 40.0 * 0.5
    assert fe.h_minus_h298(1500.0) == pytest.approx(expected_kj * 1000.0)
    jump = fe.h_minus_h298(1000.0 + 1e-9) - fe.h_minus_h298(1000.0)
    assert jump == pytest.approx(10_000.0, rel=1e-6)


def test_s_and_g(fake_db: dict[str, Species]) -> None:
    o2 = fake_db["O2"]
    T = 1200.0
    s = 30.0 * math.log(T / 1000.0) + 205.0
    assert o2.s(T) == pytest.approx(s)
    assert o2.g(T) == pytest.approx(o2.h(T) - T * s)


def test_boundary_uses_lower_range(fake_db: dict[str, Species]) -> None:
    fe = fake_db["Fe"]
    assert fe.cp(1000.0) == pytest.approx(25.0)
    assert fe.phase_at(1000.0) == "solid"
    assert fe.phase_at(1000.1) == "liquid"


def test_scalar_returns_float(fake_db: dict[str, Species]) -> None:
    fe = fake_db["Fe"]
    for fn in (fe.cp, fe.h, fe.h_minus_h298, fe.s, fe.g):
        assert isinstance(fn(500.0), float)
        assert isinstance(fn(np.float64(500.0)), float)


def test_array_input_matches_scalar(fake_db: dict[str, Species]) -> None:
    fe = fake_db["Fe"]
    T = np.array([[300.0, 999.0], [1000.0, 1999.0]])
    for fn in (fe.cp, fe.h, fe.s, fe.g):
        out = fn(T)
        assert isinstance(out, np.ndarray)
        assert out.shape == T.shape
        expected = np.array([[fn(float(x)) for x in row] for row in T])
        np.testing.assert_allclose(out, expected)


def test_empty_array(fake_db: dict[str, Species]) -> None:
    out = fake_db["Fe"].cp(np.array([]))
    assert isinstance(out, np.ndarray)
    assert out.shape == (0,)


@pytest.mark.parametrize("T", [297.0, 2000.5, 5000.0, float("nan"), float("inf")])
def test_outside_range_raises(fake_db: dict[str, Species], T: float) -> None:
    with pytest.raises(TemperatureRangeError, match="Fe"):
        fake_db["Fe"].cp(T)


def test_error_message_lists_ranges(fake_db: dict[str, Species]) -> None:
    with pytest.raises(TemperatureRangeError, match=r"298-1000, 1000-2000"):
        fake_db["Fe"].h(np.array([500.0, 2500.0]))


def test_metadata(fake_db: dict[str, Species]) -> None:
    fe = fake_db["Fe"]
    assert (fe.t_min, fe.t_max) == (298.0, 2000.0)
    assert fe.composition == {"Fe": 1.0}
    assert fe.molar_mass == pytest.approx(0.055845)
    assert fe.phase_transitions() == [(1000.0, "solid", "liquid")]
    assert fake_db["O2"].phase_transitions() == []
    assert "Fe" in repr(fe)
