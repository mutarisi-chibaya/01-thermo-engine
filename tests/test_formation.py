"""Reaction-level validation: formation dH and dG from elements vs NIST-JANAF.

JANAF tabulates delta-f H and delta-f G for each compound, computed from its own
element reference tables. Reproducing them with Reaction checks the whole chain
(species data, H/S/G evaluation, stoichiometry) independently of the per-species
Cp/S/H comparison. Reference rows: tests/data/janaf_formation.csv (kJ/mol).
"""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

from pyrothermo import Reaction, Species, default_data_path, load_species

TOL_KJ = 0.05  # kJ/mol, absolute

FORMATION = {
    "FeO": "Fe + 1/2 O2 = FeO",
    "Fe3O4": "3 Fe + 2 O2 = Fe3O4",
    "Fe2O3": "2 Fe + 3/2 O2 = Fe2O3",
    "CO": "C + 1/2 O2 = CO",
    "CO2": "C + O2 = CO2",
    "H2O": "H2 + 1/2 O2 = H2O",
}

# The NIST alpha-Fe fit error near the Curie point (~0.35 kJ/mol per Fe atom at
# 1000 K, see test_reference_data.KNOWN_DEVIATIONS) carries into every iron oxide.
KNOWN = {
    ("FeO", 1000.0): "alpha-Fe fit error at 1000 K (1 Fe): ~0.36 kJ/mol",
    ("Fe3O4", 1000.0): "alpha-Fe fit error at 1000 K (3 Fe): ~1.06 kJ/mol",
    ("Fe2O3", 1000.0): "alpha-Fe fit error at 1000 K (2 Fe): ~0.71 kJ/mol",
}

CSV_PATH = Path(__file__).parent / "data" / "janaf_formation.csv"
ROWS = list(csv.DictReader(CSV_PATH.open(encoding="utf-8"))) if CSV_PATH.is_file() else []


def _param(row: dict[str, str]) -> object:
    key = (row["species"], float(row["T"]))
    rid = f"{key[0]}-{row['T']}K"
    if key in KNOWN:
        return pytest.param(row, id=rid, marks=pytest.mark.xfail(strict=True, reason=KNOWN[key]))
    return pytest.param(row, id=rid)


@pytest.fixture(scope="module")
def real_db() -> dict[str, Species]:
    return load_species()


@pytest.mark.skipif(not ROWS or not default_data_path().is_file(), reason="data not provided")
@pytest.mark.parametrize("row", [_param(r) for r in ROWS])
def test_formation_matches_janaf(real_db: dict[str, Species], row: dict[str, str]) -> None:
    rx = Reaction.from_string(FORMATION[row["species"]], real_db)
    T = float(row["T"])
    assert rx.dH(T) / 1000.0 == pytest.approx(float(row["dfH"]), abs=TOL_KJ), "dfH"
    assert rx.dG(T) / 1000.0 == pytest.approx(float(row["dfG"]), abs=TOL_KJ), "dfG"
