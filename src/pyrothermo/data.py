"""Loader and validator for ``data/species.yaml`` (NIST Shomate coefficients).

Schema (one top-level key per species; the key is the name used in reactions)::

    Fe2O3:
      formula: Fe2O3
      phase: solid
      dHf298: -825.50          # kJ/mol, standard enthalpy of formation at 298.15 K
      source: https://webbook.nist.gov/...
      ranges:                  # ascending, contiguous, non-overlapping
        - {Tmin: 298, Tmax: 950, A: ..., B: ..., C: ..., D: ..., E: ..., F: ..., G: ..., H: ...,
           transition: true}   # optional: a real phase change happens at this Tmax
        - {Tmin: 950, Tmax: 1050, ..., phase: solid-beta}   # optional per-range phase

This module only reads and checks the file. It never supplies or alters values.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from pyrothermo.shomate import ShomateRange

COEFFICIENT_KEYS: tuple[str, ...] = ("A", "B", "C", "D", "E", "F", "G", "H")
REQUIRED_SPECIES_KEYS: tuple[str, ...] = ("formula", "phase", "dHf298", "source", "ranges")

# Ranges that meet within this many kelvin are treated as contiguous.
_BOUNDARY_TOL_K = 1e-6


class SpeciesDataError(ValueError):
    """Raised when ``species.yaml`` is missing, malformed or inconsistent."""


@dataclass(frozen=True)
class SpeciesData:
    """Raw validated data for one species.

    ``dHf298`` is stored in kJ/mol exactly as written in the YAML (NIST units);
    :class:`pyrothermo.species.Species` converts to J/mol.
    """

    name: str
    formula: str
    phase: str
    dHf298: float
    source: str
    ranges: tuple[ShomateRange, ...]


def default_data_path() -> Path:
    """Path to ``01-thermo-engine/data/species.yaml``.

    This resolves relative to the source tree, so it works for an editable
    install (``pip install -e``), which is how the other projects use this package.
    """
    return Path(__file__).resolve().parents[2] / "data" / "species.yaml"


def _to_float(value: Any, where: str) -> float:
    # PyYAML (YAML 1.1) reads "1e-3" (no dot) as a string, so accept numeric strings.
    if isinstance(value, bool):
        raise SpeciesDataError(f"{where}: expected a number, got {value!r}")
    try:
        result = float(value)
    except (TypeError, ValueError):
        raise SpeciesDataError(f"{where}: expected a number, got {value!r}") from None
    if not math.isfinite(result):
        raise SpeciesDataError(f"{where}: value must be finite, got {value!r}")
    return result


def _parse_range(raw: Any, where: str) -> ShomateRange:
    if not isinstance(raw, Mapping):
        raise SpeciesDataError(f"{where}: each range must be a mapping")
    missing = [k for k in ("Tmin", "Tmax", *COEFFICIENT_KEYS) if k not in raw]
    if missing:
        raise SpeciesDataError(f"{where}: missing keys {missing}")
    known = {"Tmin", "Tmax", "transition", "phase", *COEFFICIENT_KEYS}
    unknown = sorted(set(raw) - known)
    if unknown:
        raise SpeciesDataError(f"{where}: unknown keys {unknown}")

    tmin = _to_float(raw["Tmin"], f"{where}.Tmin")
    tmax = _to_float(raw["Tmax"], f"{where}.Tmax")
    if tmin <= 0 or tmin >= tmax:
        raise SpeciesDataError(f"{where}: need 0 < Tmin < Tmax, got Tmin={tmin}, Tmax={tmax}")

    transition = raw.get("transition", False)
    if not isinstance(transition, bool):
        raise SpeciesDataError(f"{where}.transition: expected true/false, got {transition!r}")
    phase = raw.get("phase")
    if phase is not None and not isinstance(phase, str):
        raise SpeciesDataError(f"{where}.phase: expected a string, got {phase!r}")

    coeffs = {k: _to_float(raw[k], f"{where}.{k}") for k in COEFFICIENT_KEYS}
    return ShomateRange(Tmin=tmin, Tmax=tmax, transition=transition, phase=phase, **coeffs)


def _parse_species(name: str, raw: Any) -> SpeciesData:
    where = f"species '{name}'"
    if not isinstance(raw, Mapping):
        raise SpeciesDataError(f"{where}: entry must be a mapping")
    missing = [k for k in REQUIRED_SPECIES_KEYS if k not in raw]
    if missing:
        raise SpeciesDataError(f"{where}: missing keys {missing}")

    for key in ("formula", "phase", "source"):
        if not isinstance(raw[key], str) or not raw[key].strip():
            raise SpeciesDataError(f"{where}.{key}: expected a non-empty string")

    raw_ranges = raw["ranges"]
    if not isinstance(raw_ranges, list) or not raw_ranges:
        raise SpeciesDataError(f"{where}.ranges: expected a non-empty list")
    ranges = tuple(_parse_range(r, f"{where}.ranges[{i}]") for i, r in enumerate(raw_ranges))

    for i in range(1, len(ranges)):
        prev, cur = ranges[i - 1], ranges[i]
        gap = cur.Tmin - prev.Tmax
        if gap < -_BOUNDARY_TOL_K:
            raise SpeciesDataError(
                f"{where}: ranges[{i - 1}] ({prev.Tmin}-{prev.Tmax} K) and ranges[{i}] "
                f"({cur.Tmin}-{cur.Tmax} K) overlap or are out of order"
            )
        if gap > _BOUNDARY_TOL_K:
            raise SpeciesDataError(
                f"{where}: gap between {prev.Tmax} K and {cur.Tmin} K "
                f"(ranges[{i - 1}] and ranges[{i}])"
            )

    return SpeciesData(
        name=name,
        formula=raw["formula"].strip(),
        phase=raw["phase"].strip(),
        dHf298=_to_float(raw["dHf298"], f"{where}.dHf298"),
        source=raw["source"].strip(),
        ranges=ranges,
    )


def parse_species_mapping(raw: Any) -> dict[str, SpeciesData]:
    """Validate an already-parsed YAML document and return ``{name: SpeciesData}``."""
    if not isinstance(raw, Mapping) or not raw:
        raise SpeciesDataError("species file must be a non-empty mapping of species entries")
    return {str(name): _parse_species(str(name), entry) for name, entry in raw.items()}


def load_species_data(path: str | Path | None = None) -> dict[str, SpeciesData]:
    """Read and validate a species YAML file.

    Args:
        path: YAML file to read. Defaults to :func:`default_data_path`.

    Returns:
        Mapping of species name to :class:`SpeciesData` (kJ/mol and K, as in NIST).

    Raises:
        SpeciesDataError: if the file is missing or any entry is invalid.
    """
    p = Path(path) if path is not None else default_data_path()
    if not p.is_file():
        raise SpeciesDataError(
            f"species data file not found: {p}. Fill it in from the NIST WebBook "
            "(run scripts/build_reference_data.py, see README)."
        )
    with p.open(encoding="utf-8") as fh:
        try:
            raw = yaml.safe_load(fh)
        except yaml.YAMLError as exc:
            raise SpeciesDataError(f"could not parse {p}: {exc}") from exc
    return parse_species_mapping(raw)
