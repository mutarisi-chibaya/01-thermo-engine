from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import pytest

from pyrothermo.data import (
    SpeciesDataError,
    default_data_path,
    load_species_data,
    parse_species_mapping,
)
from tests.conftest import FAKE_YAML


def _good() -> dict[str, Any]:
    rng = {"A": 1, "B": 0, "C": 0, "D": 0, "E": 0, "F": 0, "G": 0, "H": 0}
    return {
        "X": {
            "formula": "Fe",
            "phase": "solid",
            "dHf298": 0,
            "source": "test",
            "ranges": [
                {"Tmin": 298, "Tmax": 500, **rng},
                {"Tmin": 500, "Tmax": 900, **rng},
            ],
        }
    }


def test_load_fixture() -> None:
    data = load_species_data(FAKE_YAML)
    assert set(data) == {"Fe", "O2", "FeO", "CO"}
    fe = data["Fe"]
    assert fe.formula == "Fe"
    assert len(fe.ranges) == 2
    assert fe.ranges[0].transition is True
    assert fe.ranges[1].phase == "liquid"
    assert fe.ranges[1].Tmin == 1000


def test_numeric_strings_are_accepted() -> None:
    # PyYAML reads "1e-3" style numbers as strings; the loader must convert them.
    assert load_species_data(FAKE_YAML)["CO"].dHf298 == -110.0


def test_valid_mapping_parses() -> None:
    assert parse_species_mapping(_good())["X"].ranges[1].Tmax == 900


def test_default_path_points_into_package_folder() -> None:
    p = default_data_path()
    assert p.name == "species.yaml"
    assert p.parent.name == "data"
    assert (p.parent.parent / "pyproject.toml").is_file()


def test_missing_file(tmp_path: Path) -> None:
    with pytest.raises(SpeciesDataError, match="not found"):
        load_species_data(tmp_path / "nope.yaml")


def test_empty_file(tmp_path: Path) -> None:
    p = tmp_path / "empty.yaml"
    p.write_text("", encoding="utf-8")
    with pytest.raises(SpeciesDataError, match="non-empty"):
        load_species_data(p)


def test_bad_yaml(tmp_path: Path) -> None:
    p = tmp_path / "bad.yaml"
    p.write_text("X: [unclosed", encoding="utf-8")
    with pytest.raises(SpeciesDataError, match="could not parse"):
        load_species_data(p)


@pytest.mark.parametrize("key", ["formula", "phase", "dHf298", "source", "ranges"])
def test_missing_species_key(key: str) -> None:
    raw = _good()
    del raw["X"][key]
    with pytest.raises(SpeciesDataError, match=key):
        parse_species_mapping(raw)


@pytest.mark.parametrize("key", ["Tmin", "Tmax", "A", "H"])
def test_missing_range_key(key: str) -> None:
    raw = _good()
    del raw["X"]["ranges"][0][key]
    with pytest.raises(SpeciesDataError, match=key):
        parse_species_mapping(raw)


def _mutate(path: list[Any], value: Any) -> dict[str, Any]:
    raw = copy.deepcopy(_good())
    target: Any = raw["X"]
    for p in path[:-1]:
        target = target[p]
    target[path[-1]] = value
    return raw


@pytest.mark.parametrize(
    ("path", "value", "match"),
    [
        (["ranges", 0, "A"], "...", "expected a number"),
        (["ranges", 0, "A"], None, "expected a number"),
        (["ranges", 0, "A"], True, "expected a number"),
        (["ranges", 0, "A"], float("nan"), "finite"),
        (["ranges", 0, "Tmax"], 298, "Tmin < Tmax"),
        (["ranges", 1, "Tmin"], 450, "overlap"),
        (["ranges", 1, "Tmin"], 600, "gap"),
        (["ranges", 0, "transition"], "yes", "true/false"),
        (["ranges", 0, "typo"], 1, "unknown keys"),
        (["ranges"], [], "non-empty list"),
        (["formula"], "", "non-empty string"),
        (["dHf298"], "abc", "expected a number"),
    ],
)
def test_invalid_entries(path: list[Any], value: Any, match: str) -> None:
    with pytest.raises(SpeciesDataError, match=match):
        parse_species_mapping(_mutate(path, value))


def test_out_of_order_ranges() -> None:
    raw = _good()
    raw["X"]["ranges"].reverse()
    with pytest.raises(SpeciesDataError, match="overlap or are out of order"):
        parse_species_mapping(raw)
