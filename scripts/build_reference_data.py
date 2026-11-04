"""Build data/species.yaml and tests/data/janaf_reference.csv from saved NIST sources.

Run from 01-thermo-engine (after downloading with: bash scripts/fetch_sources.sh):

    .venv\\Scripts\\python.exe -I scripts/build_reference_data.py

Every coefficient is read by code from the saved pages in data/raw/nist/ and every
reference value from data/raw/janaf/. Nothing is typed by hand. The only choices made
here are *which* NIST columns to use and, for Fe, the temperatures at which to switch
between NIST's alpha-delta and gamma fits; those come from the JANAF Fe-003 table.

Further documented choices (added for the Ellingham diagram, project 02):
- Al: NIST's solid fit ends at 933 K but its liquid fit starts at 933.45 K (the JANAF
  Al-001 melting point). The solid fit is extended by 0.45 K to close the gap; this is
  labelled in species.yaml.
- SiO2: solid fits come from the WebBook *quartz* page (CAS 14808-60-7) and the liquid
  fit from the generic SiO2 page (CAS 7631-86-9). NIST's quartz fit runs to 1996 K,
  i.e. metastable quartz above ~1140 K (the stable phases are tridymite, then
  cristobalite). Reference rows are taken from JANAF O-037 (quartz) at <= 1500 K.
- Mn, Cr, Ni, Si, Al, Al2O3, Cr2O3: phase changes are marked where JANAF *-001 /
  table notes list a first-order transition; Cp lambda points (Cr 311.5 K, Ni 631 K,
  Cr2O3 305 K) are fit boundaries only.
- MnO and NiO are not included: neither the WebBook nor NIST-JANAF has data for them.
"""

from __future__ import annotations

import csv
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from fit_graphite import fit_graphite
from nist_parse import (
    JanafRow,
    ShomateTable,
    parse_janaf,
    parse_one_dimensional,
    parse_shomate_tables,
    read_source,
)

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
ACCESSED = "2026-11-04"
WEBBOOK = "https://webbook.nist.gov/cgi/cbook.cgi?ID={id}&Units=SI&Mask={mask}"
JANAF_URL = "https://janaf.nist.gov/tables/{table}.html"


@dataclass(frozen=True)
class Piece:
    """One output range: NIST table (label substring) and column, optionally clipped."""

    table: str
    column: int
    Tmin: float | None = None  # override (clip) the NIST range
    Tmax: float | None = None
    phase: str | None = None
    transition: bool = False
    # Read this piece from data/raw/nist/<page>.html instead of the recipe's own page.
    page: str | None = None
    page_id: str | None = None  # WebBook ID of that page, for the source line
    # Allow Tmax to exceed NIST's stated Tmax by at most this many K (documented gaps).
    extend_K: float = 0.0


@dataclass(frozen=True)
class Recipe:
    key: str
    formula: str
    phase: str
    nist_id: str
    mask: int
    dhf_quantity: str | None  # None => element in its reference state, 0 by definition
    pieces: tuple[Piece, ...]
    janaf_table: str
    janaf_T: tuple[float, ...]


SOLID = "Solid Phase"
LIQUID = "Liquid Phase"
GAS = "Gas Phase"

RECIPES: tuple[Recipe, ...] = (
    Recipe(
        "Fe", "Fe", "solid", "C7439896", 2, None,
        (
            # alpha (bcc) up to the alpha->gamma transition at 1184 K (JANAF Fe-003).
            # 1042 K is the magnetic (Curie) point: a fit boundary, H is continuous.
            Piece(SOLID, 0, phase="alpha"),
            Piece(SOLID, 1, phase="alpha"),
            Piece(SOLID, 2, phase="alpha"),
            Piece(SOLID, 3, Tmax=1184.0, phase="alpha", transition=True),
            # gamma (fcc) 1184-1665 K, from NIST's separate gamma-phase fit.
            Piece(SOLID, 4, Tmin=1184.0, Tmax=1665.0, phase="gamma", transition=True),
            # delta (bcc) 1665-1809 K, NIST alpha-delta fit again.
            Piece(SOLID, 3, Tmin=1665.0, phase="delta", transition=True),
            Piece(LIQUID, 0, phase="liquid"),
        ),
        "Fe-003", (298.15, 500, 1000, 1100, 1200, 1500, 1700, 1800, 2000, 2500),
    ),
    Recipe(
        "FeO", "FeO", "solid", "C1345251", 2, "ΔfH°solid",
        (Piece(SOLID, 0, transition=True), Piece(LIQUID, 0, phase="liquid")),
        "Fe-020", (298.15, 500, 1000, 1500, 1700, 2000),
    ),
    Recipe(
        "Fe3O4", "Fe3O4", "solid", "C1309382", 2, "ΔfH°solid",
        # 900 K is a lambda (Curie) point in JANAF, not a first-order change.
        (Piece(SOLID, 0), Piece(SOLID, 1)),
        "Fe-032", (298.15, 500, 1000, 1500),
    ),
    Recipe(
        "Fe2O3", "Fe2O3", "solid", "C1317608", 2, "ΔfH°solid",
        (
            # JANAF Fe-030: I->II at 950 K (H jumps), II->III at 1050 K.
            Piece(SOLID, 0, transition=True),
            Piece(SOLID, 1, transition=True),
            Piece(SOLID, 2),
        ),
        "Fe-030", (298.15, 500, 1000, 1500, 2000),
    ),
    Recipe(
        "CO", "CO", "gas", "C630080", 1, "ΔfH°gas",
        (Piece(GAS, 0), Piece(GAS, 1)),
        "C-093", (298.15, 500, 1000, 1500, 2000),
    ),
    Recipe(
        "CO2", "CO2", "gas", "C124389", 1, "ΔfH°gas",
        (Piece(GAS, 0), Piece(GAS, 1)),
        "C-095", (298.15, 500, 1000, 1500, 2000),
    ),
    Recipe(
        "H2", "H2", "gas", "C1333740", 1, None,
        (Piece(GAS, 0), Piece(GAS, 1), Piece(GAS, 2)),
        "H-050", (298.15, 500, 1000, 1500, 2000),
    ),
    Recipe(
        "H2O", "H2O", "gas", "C7732185", 1, "ΔfH°gas",
        # NIST's gas-phase fit starts at 500 K, so no 298.15 K reference row.
        (Piece(GAS, 0), Piece(GAS, 1)),
        "H-064", (500, 1000, 1500, 2000),
    ),
    Recipe(
        "O2", "O2", "gas", "C7782447", 1, None,
        (Piece(GAS, 0), Piece(GAS, 1), Piece(GAS, 2)),
        "O-029", (298.15, 500, 1000, 1500, 2000),
    ),
    # ---- added for the Ellingham diagram (project 02) ----
    Recipe(
        "Al", "Al", "solid", "C7429905", 2, None,
        (
            # NIST solid fit ends at 933 K, liquid starts at 933.45 K (JANAF Al-001
            # melting point): extend the solid fit by 0.45 K to close the gap.
            Piece(SOLID, 0, Tmax=933.45, extend_K=0.45, transition=True),
            Piece(LIQUID, 0, phase="liquid"),
        ),
        "Al-001", (298.15, 500, 900, 1000, 1500, 2000),
    ),
    Recipe(
        "Al2O3", "Al2O3", "solid", "C1344281", 2, "ΔfH°solid",
        # Column 1 of the solid table is the alpha (corundum) phase.
        (Piece(SOLID, 0, phase="alpha", transition=True), Piece(LIQUID, 0, phase="liquid")),
        "Al-101", (298.15, 500, 1000, 1500, 2000),
    ),
    Recipe(
        "Cr", "Cr", "solid", "C7440473", 2, None,
        # 311.5 K is a Cp lambda point (JANAF Cr-001), not a first-order change.
        (Piece(SOLID, 0), Piece(SOLID, 1, transition=True), Piece(LIQUID, 0, phase="liquid")),
        "Cr-001", (298.15, 500, 1000, 1500, 2000),
    ),
    Recipe(
        "Cr2O3", "Cr2O3", "solid", "C1308389", 2, "ΔfH°solid",
        # 305 K is a Cp lambda point (JANAF Cr-014): fit boundary only.
        (Piece(SOLID, 0), Piece(SOLID, 1, transition=True), Piece(LIQUID, 0, phase="liquid")),
        "Cr-014", (298.15, 500, 1000, 1500, 2000),
    ),
    Recipe(
        "Mn", "Mn", "solid", "C7439965", 2, None,
        (
            # JANAF Mn-001: alpha->beta 980 K, beta->gamma 1361 K, gamma->delta 1412 K,
            # delta->liquid 1519 K; NIST's ranges break at the same temperatures.
            Piece(SOLID, 0, phase="alpha", transition=True),
            Piece(SOLID, 1, phase="beta", transition=True),
            Piece(SOLID, 2, phase="gamma", transition=True),
            Piece(SOLID, 3, phase="delta", transition=True),
            Piece(LIQUID, 0, phase="liquid"),
        ),
        "Mn-001", (298.15, 500, 1000, 1400, 1500, 2000),
    ),
    Recipe(
        "Ni", "Ni", "solid", "C7440020", 2, None,
        # 631 K is the Curie (lambda) point (JANAF Ni-001): fit boundaries only.
        (
            Piece(SOLID, 0),
            Piece(SOLID, 1),
            Piece(SOLID, 2, transition=True),
            Piece(LIQUID, 0, phase="liquid"),
        ),
        "Ni-001", (298.15, 500, 1000, 1500, 2000),
    ),
    Recipe(
        "Si", "Si", "solid", "C7440213", 2, None,
        (Piece(SOLID, 0, transition=True), Piece(LIQUID, 0, phase="liquid")),
        "Si-001", (298.15, 500, 1000, 1500, 2000),
    ),
    Recipe(
        "SiO2", "SiO2", "solid", "C14808607", 2, "ΔfH°solid",
        (
            # Quartz page: I (alpha) to 847 K, II (beta) to 1996 K (JANAF O-037: I->II
            # at 847 K). Above ~1140 K this is metastable quartz; see module docstring.
            Piece(SOLID, 0, phase="quartz-I", transition=True),
            Piece(SOLID, 1, phase="quartz-II", transition=True),
            # Liquid fit from the generic SiO2 page (CAS 7631-86-9), from 1996 K.
            Piece(LIQUID, 0, phase="liquid", page="SiO2_liquid", page_id="C7631869"),
        ),
        "O-037", (298.15, 500, 1000, 1500),
    ),
)  # fmt: skip


def _table(tables: list[ShomateTable], label: str, key: str) -> ShomateTable:
    found = [t for t in tables if label in t.label]
    if len(found) != 1:
        raise SystemExit(f"{key}: expected one '{label}' Shomate table, found {len(found)}")
    return found[0]


def _num(x: float) -> str:
    return repr(x)


def build_species(r: Recipe) -> tuple[list[str], tuple[float, float]]:
    page = read_source(RAW / "nist" / f"{r.key}.html")
    tables = parse_shomate_tables(page)
    if r.dhf_quantity is None:
        dhf, dhf_note = 0.0, "element in its reference state: 0 by definition"
    else:
        dhf = parse_one_dimensional(page)[r.dhf_quantity]
        dhf_note = f"NIST {r.dhf_quantity}"
    url = WEBBOOK.format(id=r.nist_id, mask=r.mask)
    for p in r.pieces:
        if p.page is not None:
            if p.page_id is None:
                raise SystemExit(f"{r.key}: piece from page {p.page!r} needs page_id")
            # No ": " here - it would break the unquoted YAML scalar.
            url += f" ; {p.table.lower()} fit from {WEBBOOK.format(id=p.page_id, mask=r.mask)}"
    other_pages = {
        p.page: parse_shomate_tables(read_source(RAW / "nist" / f"{p.page}.html"))
        for p in r.pieces
        if p.page is not None
    }

    lines = [
        f"{r.key}:",
        f"  formula: {r.formula}",
        f"  phase: {r.phase}",
        f"  dHf298: {_num(dhf)}   # kJ/mol, {dhf_note}",
        f"  source: {url}",
        f"  janaf: {JANAF_URL.format(table=r.janaf_table)}",
        "  ranges:",
    ]
    prev_tmax: float | None = None
    for p in r.pieces:
        source_tables = tables if p.page is None else other_pages[p.page]
        col = _table(source_tables, p.table, r.key).columns[p.column]
        tmin = p.Tmin if p.Tmin is not None else col.Tmin
        tmax = p.Tmax if p.Tmax is not None else col.Tmax
        if not (col.Tmin <= tmin < tmax <= col.Tmax + p.extend_K):
            raise SystemExit(
                f"{r.key}: clip {tmin}-{tmax} outside NIST range {col.Tmin}-{col.Tmax}"
                + (f" (+{p.extend_K} K allowed)" if p.extend_K else "")
            )
        if prev_tmax is not None and tmin != prev_tmax:
            raise SystemExit(f"{r.key}: ranges not contiguous at {prev_tmax}/{tmin}")
        prev_tmax = tmax
        c = col.coeffs
        coeffs = ", ".join(f"{k}: {_num(c[k])}" for k in "ABCDEFGH")
        extra = ""
        if p.phase:
            extra += f", phase: {p.phase}"
        if p.transition:
            extra += ", transition: true"
        if (tmin, tmax) == (col.Tmin, col.Tmax):
            clip = ""
        elif tmax > col.Tmax:
            clip = (
                f" (NIST range {col.Tmin:g}-{col.Tmax:g} K; EXTENDED by this project "
                f"by {tmax - col.Tmax:g} K to {tmax:g} K to meet the next range)"
            )
        else:
            clip = f" (clipped from {col.Tmin:g}-{col.Tmax:g} K)"
        where = "" if p.page is None else f" of page {p.page_id}"
        lines.append(f"    # NIST '{p.table}' column {p.column + 1}{where}{clip}: {col.comment}")
        lines.append(f"    - {{Tmin: {_num(tmin)}, Tmax: {_num(tmax)}, {coeffs}{extra}}}")
    first = r.pieces[0]
    t0 = _table(tables, first.table, r.key).columns[first.column].Tmin
    return lines, (t0 if first.Tmin is None else first.Tmin, prev_tmax or 0.0)


def janaf_rows(r: Recipe, t_range: tuple[float, float]) -> list[JanafRow]:
    rows = parse_janaf(read_source(RAW / "janaf" / f"{r.janaf_table}.txt"))
    out = []
    for T in r.janaf_T:
        matches = [row for row in rows if row.T == T]
        if len(matches) != 1:
            raise SystemExit(f"{r.key}: JANAF {r.janaf_table} has {len(matches)} rows at {T} K")
        if not (t_range[0] <= T <= t_range[1]):
            raise SystemExit(f"{r.key}: {T} K outside data range {t_range}")
        out.append(matches[0])
    return out


GRAPHITE = Recipe(
    "C", "C", "solid", "", 0, None, (), "C-002", (298.15, 500, 1000, 1500, 2000, 3000)
)


def build_graphite() -> tuple[list[str], tuple[float, float]]:
    rows = parse_janaf(read_source(RAW / "janaf" / "C-002.txt"))
    pieces = fit_graphite(rows)
    lines = [
        "C:",
        "  formula: C",
        "  phase: solid",
        "  dHf298: 0.0   # kJ/mol, element in its reference state: 0 by definition",
        f"  source: FITTED by this project to {JANAF_URL.format(table='C-002')} "
        "(scripts/fit_graphite.py); NIST WebBook has no graphite Shomate fit",
        f"  janaf: {JANAF_URL.format(table='C-002')}",
        "  ranges:",
    ]
    for tmin, tmax, c in pieces:
        coeffs = ", ".join(f"{k}: {_num(c[k])}" for k in "ABCDEFGH")
        lines.append(f"    # least-squares fit to JANAF C-002, {tmin:g}-{tmax:g} K")
        lines.append(f"    - {{Tmin: {_num(tmin)}, Tmax: {_num(tmax)}, {coeffs}, phase: graphite}}")
    return lines, (pieces[0][0], pieces[-1][1])


def main() -> None:
    header = [
        "# GENERATED by scripts/build_reference_data.py - do not edit by hand.",
        "# Coefficients read by code from NIST Chemistry WebBook pages saved in data/raw/nist/",
        f"# (accessed {ACCESSED}); the WebBook Shomate fits are from Chase (1998),",
        "# NIST-JANAF Thermochemical Tables, 4th ed., J. Phys. Chem. Ref. Data Monograph 9.",
        "# Units as NIST: K, kJ/mol (dHf298, H - H298), J/(mol K) (Cp, S).",
        "#",
        "#",
        "# EXCEPTION: C (graphite) has no WebBook Shomate fit. Its coefficients are FITTED BY",
        "# THIS PROJECT to the JANAF C-002 table (scripts/fit_graphite.py), not published by NIST.",
        "",
    ]
    yaml_lines = list(header)
    csv_rows: list[list[str]] = []
    for r in RECIPES:
        lines, t_range = build_species(r)
        yaml_lines += [*lines, ""]
        for row in janaf_rows(r, t_range):
            csv_rows.append([r.key, f"{row.T:g}", f"{row.Cp}", f"{row.S}", f"{row.H_minus_H298}"])

    lines, t_range = build_graphite()
    yaml_lines += [*lines, ""]
    for row in janaf_rows(GRAPHITE, t_range):
        csv_rows.append(["C", f"{row.T:g}", f"{row.Cp}", f"{row.S}", f"{row.H_minus_H298}"])

    (ROOT / "data" / "species.yaml").write_text("\n".join(yaml_lines), encoding="utf-8")
    csv_path = ROOT / "tests" / "data" / "janaf_reference.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["species", "T", "Cp", "S", "H_minus_H298"])
        w.writerows(csv_rows)
    # Formation properties of compounds (independent check of the Reaction class).
    form_path = ROOT / "tests" / "data" / "janaf_formation.csv"
    with form_path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["species", "T", "dfH", "dfG"])
        for r in RECIPES:
            if r.dhf_quantity is None:
                continue  # elements: formation properties are zero by definition
            rows = parse_janaf(read_source(RAW / "janaf" / f"{r.janaf_table}.txt"))
            for T in r.janaf_T:
                if T == 298.15:
                    continue
                (row,) = [x for x in rows if x.T == T]
                w.writerow([r.key, f"{row.T:g}", f"{row.dfH}", f"{row.dfG}"])
    print(f"wrote {len(RECIPES) + 1} species and {len(csv_rows)} JANAF rows")


if __name__ == "__main__":
    main()
