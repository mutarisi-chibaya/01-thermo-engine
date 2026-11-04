"""Validation of the REAL data in data/species.yaml against NIST-JANAF.

These tests are skipped until the human provides:
  - data/species.yaml               (Shomate coefficients from the NIST WebBook)
  - tests/data/janaf_reference.csv  (rows copied from the NIST-JANAF tables)

CSV columns: species,T,Cp,S,H_minus_H298
  species       key in species.yaml (e.g. Fe2O3)
  T             K
  Cp, S         J/(mol K)
  H_minus_H298  kJ/mol   (JANAF column "H-H(Tr)")
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pytest

from pyrothermo import Species, default_data_path, load_species

REL_TOL = 1e-3  # 0.1 %, from PLAN.md
# JANAF prints H - H298 to 0.001 kJ/mol and it is ~0 near 298 K, so also allow a small
# absolute tolerance there; and Cp/S to 0.001 J/(mol K).
ABS_TOL_H_KJ = 0.005
ABS_TOL_CP_S = 0.005
# Allowed mismatch in H where two ranges meet without a phase change (kJ/mol).
CONTINUITY_TOL_KJ = 0.1
# Allowed mismatch between dHf298 and the first range's H coefficient (kJ/mol).
DHF_COEF_TOL_KJ = 0.05

# Rows where the NIST Shomate fit itself departs from JANAF by more than REL_TOL.
# Verified 2026-11-04 by evaluating the unmodified NIST coefficients; they are listed
# here instead of loosening the tolerance for every species. strict=True means the
# suite flags it if one of these starts passing (e.g. after a data change).
KNOWN_DEVIATIONS: dict[tuple[str, float], str] = {
    ("Fe", 1000.0): "NIST alpha-Fe fits near the Curie point: H-H298 off ~0.33 kJ/mol",
    ("Fe", 1100.0): "NIST alpha-Fe fits near the Curie point: S off ~0.5 %, H ~0.37 kJ/mol",
    ("Fe", 1700.0): "NIST delta-Fe (alpha-delta fit) Cp ~0.14 % high",
    ("Fe", 1800.0): "NIST delta-Fe (alpha-delta fit) Cp ~0.23 % high",
    ("Fe2O3", 500.0): "NIST hematite fit Cp 0.11 % low",
    # NIST's printed coefficients give H - H298 slightly non-zero at 298.15 K itself.
    ("Fe2O3", 298.15): "NIST coefficients give H-H298(298.15 K) = -0.026 kJ/mol, not 0",
    ("Fe3O4", 298.15): "NIST coefficients give H-H298(298.15 K) = -0.007 kJ/mol, not 0",
    # Added with the project-02 species; measured 2026-11-04 with the unmodified NIST
    # coefficients. These NIST fits (reviewed 1967-1979) predate JANAF's 4th edition.
    ("Al2O3", 298.15): "NIST alpha-Al2O3 fit: Cp -0.27 %, S -0.24 %, H-H298 -0.026 kJ/mol",
    ("Al2O3", 500.0): "NIST alpha-Al2O3 fit: S +0.12 %, H-H298 +0.066 kJ/mol",
    ("Cr", 1000.0): "NIST Cr fit Cp +0.33 %",
    ("Cr", 1500.0): "NIST Cr fit Cp -0.11 %",
    ("Cr2O3", 298.15): "NIST Cr2O3 S298 80.65 vs JANAF 81.154 J/(mol K) (-0.62 %); "
    "H-H298(298.15 K) = +0.065 kJ/mol",
    ("Cr2O3", 2000.0): "NIST Cr2O3 fit Cp -0.22 %",
    ("Mn", 298.15): "NIST alpha-Mn fit Cp -0.13 %",
    ("Mn", 500.0): "NIST alpha-Mn fit Cp +0.40 %, H-H298 -0.008 kJ/mol",
}
# Fit boundaries (no phase change) where consecutive NIST fits do not join in H.
# Value: largest jump accepted (kJ/mol), slightly above the observed one.
KNOWN_JUMPS_KJ: dict[tuple[str, float], float] = {
    ("Fe", 700.0): 0.15,  # observed +0.141
    ("Fe", 1042.0): 0.25,  # observed -0.212 (Curie point)
    ("Fe", 1100.0): 0.40,  # observed -0.365
    ("Ni", 600.0): 0.20,  # observed -0.164 (near the 631 K Curie point)
    ("Ni", 700.0): 0.30,  # observed -0.266 (near the 631 K Curie point)
}

REQUIRED_FORMULAS = {"Fe", "FeO", "Fe3O4", "Fe2O3", "C", "CO", "CO2", "H2", "H2O", "O2"}

SPECIES_YAML = default_data_path()
JANAF_CSV = Path(__file__).parent / "data" / "janaf_reference.csv"

needs_yaml = pytest.mark.skipif(
    not SPECIES_YAML.is_file(), reason=f"real data not provided yet: {SPECIES_YAML}"
)


def _janaf_rows() -> list[dict[str, str]]:
    if not JANAF_CSV.is_file():
        return []
    with JANAF_CSV.open(newline="", encoding="utf-8-sig") as fh:
        return [row for row in csv.DictReader(fh) if row.get("species", "").strip()]


JANAF_ROWS = _janaf_rows()


def _param(row: dict[str, str]) -> object:
    key = (row["species"].strip(), float(row["T"]))
    rid = f"{key[0]}-{row['T']}K"
    if key in KNOWN_DEVIATIONS:
        mark = pytest.mark.xfail(
            strict=True, reason=f"known NIST-fit deviation: {KNOWN_DEVIATIONS[key]}"
        )
        return pytest.param(row, marks=mark, id=rid)
    return pytest.param(row, id=rid)


@pytest.fixture(scope="module")
def real_db() -> dict[str, Species]:
    return load_species(SPECIES_YAML)


@needs_yaml
def test_required_species_present(real_db: dict[str, Species]) -> None:
    formulas = {sp.formula for sp in real_db.values()}
    missing = REQUIRED_FORMULAS - formulas
    assert not missing, f"species.yaml is missing formulas: {sorted(missing)}"


@needs_yaml
def test_dhf298_matches_first_range_h_coefficient(real_db: dict[str, Species]) -> None:
    # In NIST's Shomate tables the H coefficient equals dHf298 of the 298 K phase,
    # so a mismatch usually means a copying error.
    bad = {
        name: (sp.data.dHf298, sp.ranges[0].H)
        for name, sp in real_db.items()
        if abs(sp.data.dHf298 - sp.ranges[0].H) > DHF_COEF_TOL_KJ
    }
    assert not bad, f"dHf298 vs first-range H coefficient (kJ/mol): {bad}"


@needs_yaml
def test_h_continuous_at_range_boundaries(real_db: dict[str, Species]) -> None:
    bad = []
    for name, sp in real_db.items():
        transitions = {T for T, _, _ in sp.phase_transitions()}
        for a, b in zip(sp.ranges, sp.ranges[1:], strict=False):
            if a.Tmax in transitions:
                continue
            T = np.array([a.Tmax])
            jump = float(b.h_formation_scale(T)[0] - a.h_formation_scale(T)[0])
            tol = KNOWN_JUMPS_KJ.get((name, a.Tmax), CONTINUITY_TOL_KJ)
            if abs(jump) > tol:
                bad.append(f"{name} at {a.Tmax:g} K: jump {jump:+.3f} kJ/mol")
    assert not bad, (
        "H jumps where no phase change is marked (add `transition: true` if real):\n"
        + "\n".join(bad)
    )


@pytest.mark.skipif(not JANAF_ROWS, reason=f"JANAF reference not provided yet: {JANAF_CSV}")
@needs_yaml
@pytest.mark.parametrize("row", [_param(r) for r in JANAF_ROWS])
def test_matches_janaf(real_db: dict[str, Species], row: dict[str, str]) -> None:
    name = row["species"].strip()
    assert name in real_db, f"{name} in janaf_reference.csv but not in species.yaml"
    sp = real_db[name]
    T = float(row["T"])
    cp_ref, s_ref, dh_ref = float(row["Cp"]), float(row["S"]), float(row["H_minus_H298"])

    assert sp.cp(T) == pytest.approx(cp_ref, rel=REL_TOL, abs=ABS_TOL_CP_S), "Cp"
    assert sp.s(T) == pytest.approx(s_ref, rel=REL_TOL, abs=ABS_TOL_CP_S), "S"
    assert sp.h_minus_h298(T) / 1000.0 == pytest.approx(dh_ref, rel=REL_TOL, abs=ABS_TOL_H_KJ), (
        "H - H298"
    )
