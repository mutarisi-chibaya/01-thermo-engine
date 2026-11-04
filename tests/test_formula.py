from __future__ import annotations

import pytest

from pyrothermo import FormulaError, molar_mass, molar_mass_g_per_mol, parse_formula


@pytest.mark.parametrize(
    ("formula", "expected"),
    [
        ("Fe", {"Fe": 1}),
        ("O2", {"O": 2}),
        ("Fe2O3", {"Fe": 2, "O": 3}),
        ("Fe3O4", {"Fe": 3, "O": 4}),
        ("CO2", {"C": 1, "O": 2}),
        ("Ca(OH)2", {"Ca": 1, "O": 2, "H": 2}),
        ("Mg2(SiO4)", {"Mg": 2, "Si": 1, "O": 4}),
        ("CaMg(CO3)2", {"Ca": 1, "Mg": 1, "C": 2, "O": 6}),
        ("((CH3)2)3", {"C": 6, "H": 18}),
        ("HOH", {"H": 2, "O": 1}),
        (" Fe2O3 ", {"Fe": 2, "O": 3}),
    ],
)
def test_parse_formula(formula: str, expected: dict[str, float]) -> None:
    assert parse_formula(formula) == pytest.approx(expected)


def test_parse_decimal_subscript() -> None:
    assert parse_formula("Fe0.947O") == pytest.approx({"Fe": 0.947, "O": 1.0})


@pytest.mark.parametrize("bad", ["", "   ", "fe2O3", "Fe2O3)", "(FeO", "2Fe", "Fe-O", "Fe0"])
def test_parse_formula_errors(bad: str) -> None:
    with pytest.raises(FormulaError):
        parse_formula(bad)


def test_molar_mass_fe2o3() -> None:
    # PLAN.md target: 159.69 g/mol.
    assert round(molar_mass_g_per_mol("Fe2O3"), 2) == 159.69


def test_molar_mass_is_si() -> None:
    assert molar_mass("Fe2O3") == pytest.approx(0.159687, rel=1e-6)
    assert molar_mass("H2O") == pytest.approx(0.018015, rel=1e-6)


def test_molar_mass_unknown_element() -> None:
    with pytest.raises(FormulaError, match="Zr"):
        molar_mass("ZrO2")
