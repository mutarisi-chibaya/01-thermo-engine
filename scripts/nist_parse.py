"""Deterministic parsers for saved NIST WebBook pages and NIST-JANAF text tables.

No values are typed by hand or generated: numbers are read verbatim from the
saved source files in data/raw/ so every value can be traced to its source.
"""

from __future__ import annotations

import contextlib
import html
import re
from dataclasses import dataclass, field
from pathlib import Path

_TABLE = re.compile(r'<table class="data" aria-label="([^"]*)"[^>]*>(.*?)</table>', re.S)
_ROW = re.compile(r"<tr>\s*<th[^>]*>(.*?)</th>(.*?)</tr>", re.S)
_CELL = re.compile(r"<td[^>]*>(.*?)</td>", re.S)
_TAG = re.compile(r"<[^>]+>")
_SCI = re.compile(r"^(-?[\d.]+)\s*&times;10<sup>([-\d]+)</sup>$")
_RANGE = re.compile(r"^([\d.]+)\s+to\s+([\d.]+)$")


def read_source(path: Path) -> str:
    """Read a saved source file (NIST pages are not always UTF-8)."""
    data = path.read_bytes()
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError:
        return data.decode("latin-1")


def _text(cell: str) -> str:
    return html.unescape(_TAG.sub("", cell)).strip()


def nist_number(cell: str) -> float:
    """Convert a NIST table cell like ``-1.884667&times;10<sup>-8</sup>`` to float."""
    raw = cell.strip()
    m = _SCI.match(raw)
    if m:
        return float(f"{m.group(1)}e{m.group(2)}")
    return float(_text(raw))


@dataclass
class ShomateColumn:
    Tmin: float
    Tmax: float
    coeffs: dict[str, float]
    comment: str


@dataclass
class ShomateTable:
    label: str
    columns: list[ShomateColumn] = field(default_factory=list)


def parse_shomate_tables(page: str) -> list[ShomateTable]:
    """All Shomate tables on a WebBook page, columns in page order."""
    tables = []
    for label, body in _TABLE.findall(page):
        if "Shomate" not in label:
            continue
        rows = {_text(name): _CELL.findall(cells) for name, cells in _ROW.findall(body)}
        temps = [_RANGE.match(_text(c)) for c in rows["Temperature (K)"]]
        cols = []
        for i, m in enumerate(temps):
            if m is None:
                raise ValueError(f"cannot parse temperature range in {label!r}")
            coeffs = {k: nist_number(rows[k][i]) for k in "ABCDEFGH"}
            comment = _text(rows["Comment"][i]) if "Comment" in rows else ""
            cols.append(ShomateColumn(float(m.group(1)), float(m.group(2)), coeffs, comment))
        tables.append(ShomateTable(html.unescape(label), cols))
    return tables


def parse_one_dimensional(page: str) -> dict[str, float]:
    """Quantity -> value (NIST units) from the 'One dimensional data' tables."""
    out: dict[str, float] = {}
    for label, body in _TABLE.findall(page):
        if label != "One dimensional data":
            continue
        for row in re.findall(r'<tr class="[^"]*">(.*?)</tr>', body, re.S):
            cells = _CELL.findall(row)
            if len(cells) >= 2:
                name = _text(cells[0])
                try:
                    out.setdefault(name, nist_number(cells[1]))
                except ValueError:
                    continue
    return out


@dataclass
class JanafRow:
    T: float
    Cp: float
    S: float
    H_minus_H298: float  # kJ/mol
    note: str
    dfH: float = float("nan")  # kJ/mol, standard enthalpy of formation at T
    dfG: float = float("nan")  # kJ/mol, standard Gibbs energy of formation at T


def parse_janaf(text: str) -> list[JanafRow]:
    """Rows of a NIST-JANAF ``.txt`` table (tab separated; first two lines are headers)."""
    rows = []
    for line in text.splitlines()[2:]:
        parts = line.split("\t")
        if len(parts) < 5:
            continue
        try:
            T, cp, s, dh = (float(parts[i]) for i in (0, 1, 2, 4))
        except ValueError:
            continue  # e.g. 'INFINITE' or blank cells at transitions
        note = parts[5].strip() if len(parts) > 5 and re.search("[A-Za-z]", parts[5]) else ""
        dfh = dfg = float("nan")
        if not note and len(parts) > 6:
            with contextlib.suppress(ValueError):
                dfh, dfg = float(parts[5]), float(parts[6])
        rows.append(JanafRow(T, cp, s, dh, note, dfh, dfg))
    return rows


if __name__ == "__main__":
    import sys

    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
    for arg in sys.argv[1:]:
        p = Path(arg)
        content = read_source(p)
        print(f"=== {p.name}")
        if p.suffix == ".html":
            for name, val in parse_one_dimensional(content).items():
                print(f"  {name} = {val}")
            for t in parse_shomate_tables(content):
                print(f"  [{t.label}]")
                for c in t.columns:
                    print(f"    {c.Tmin:g}-{c.Tmax:g}  H={c.coeffs['H']}  {c.comment}")
        else:
            for r in parse_janaf(content):
                if r.note:
                    print(f"  {r.T:g} K  {r.note}")
