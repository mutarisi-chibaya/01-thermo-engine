"""Tests for the Streamlit interface: helper functions and full-app smoke tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("streamlit")
pytest.importorskip("plotly")

from streamlit.testing.v1 import AppTest

from app.streamlit_app import (
    PROJECT_ROOT,
    _names,
    display_path,
    reaction_figure,
    resolve_path,
    species_figure,
    species_table,
)
from pyrothermo import Reaction, Species, default_data_path

APP = str(Path(__file__).parents[1] / "app" / "streamlit_app.py")
needs_data = pytest.mark.skipif(not default_data_path().is_file(), reason="no real data")
H_LABEL = "Heat needed from 25 °C (H − H298)"


def test_species_table_nan_outside_range(fake_db: dict[str, Species]) -> None:
    T = np.array([500.0, 2500.0])
    df = species_table([fake_db["Fe"], fake_db["O2"]], "Heat capacity (Cp)", T)
    col = "Heat capacity (Cp) [J/(mol·K)]"
    fe = df[df.species == "Fe"][col].to_numpy()
    o2 = df[df.species == "O2"][col].to_numpy()
    assert fe[0] == pytest.approx(25.0)
    assert np.isnan(fe[1])
    np.testing.assert_allclose(o2, [30.0, 30.0])


@pytest.mark.parametrize("unit", ["K", "°C"])
def test_figures_build(fake_db: dict[str, Species], unit: str) -> None:
    T = np.linspace(300.0, 1900.0, 50)
    fig = species_figure([fake_db["Fe"]], H_LABEL, T, unit)  # type: ignore[arg-type]
    assert len(fig.data) == 1
    assert np.isnan(np.asarray(fig.data[0].y, dtype=float)).sum() == 1  # break at 1000 K
    x0 = float(fig.data[0].x[0])
    assert x0 == pytest.approx(300.0 if unit == "K" else 300.0 - 273.15)
    rx = Reaction.from_string("2 Fe + O2 = 2 FeO", fake_db)
    fig_g = reaction_figure(rx, T, "dG", unit, mark_T=1000.0)  # type: ignore[arg-type]
    assert len(fig_g.data) == 2  # line + chosen-temperature marker
    assert len(reaction_figure(rx, T, "logK", unit).data) == 1  # type: ignore[arg-type]


def test_names_from_equation() -> None:
    assert _names("3 Fe2O3 + CO = 2 Fe3O4 + CO2") == ["Fe2O3", "CO", "Fe3O4", "CO2"]


def test_paths_shown_relative_to_project() -> None:
    assert display_path(default_data_path()) == "data/species.yaml"
    assert resolve_path("data/species.yaml") == PROJECT_ROOT / "data" / "species.yaml"


def _app() -> AppTest:
    at = AppTest.from_file(APP, default_timeout=30)
    at.run()
    assert not at.exception
    return at


@needs_data
def test_app_preset_reaction_gives_plain_language() -> None:
    at = _app()
    assert str(at.selectbox(key="preset").value).startswith(
        "Reduce wüstite (FeO) to iron with hydrogen"
    )
    text = " ".join(m.value for m in at.markdown)
    assert "Gas needed" in text
    assert "H2" in text
    labels = [m.label for m in at.metric]
    assert "ΔG° (kJ/mol)" in labels
    assert "log₁₀ K" in labels


@needs_data
def test_app_custom_reaction_and_unbalanced_error() -> None:
    at = _app()
    at.selectbox(key="preset").set_value("Custom reaction (type your own)…").run()
    at.text_input(key="reaction_eq").set_value("C + CO2 = 2 CO").run()
    assert not at.exception
    assert any("Balanced" in s.value for s in at.success)
    assert "Turning point" in " ".join(m.value for m in at.markdown)
    at.text_input(key="reaction_eq").set_value("Fe + O2 = FeO").run()
    assert any("unbalanced" in e.value for e in at.error)


@needs_data
def test_app_unit_switch_to_kelvin() -> None:
    at = _app()
    at.radio(key="unit").set_value("K").run()
    assert not at.exception
    assert any("(K)" in n.label for n in at.number_input)


@needs_data
def test_app_every_preset_runs() -> None:
    at = _app()
    for option in at.selectbox(key="preset").options:
        if option.startswith("Custom"):
            continue
        at.selectbox(key="preset").set_value(option).run()
        assert not at.exception, option
        assert not at.error, (option, [e.value for e in at.error])


@needs_data
def test_app_has_no_demo_toggle() -> None:
    at = _app()
    assert all("FAKE" not in t.label for t in at.toggle)
    assert all("FAKE" not in w.value for w in at.warning)


def test_app_without_data_shows_instructions(tmp_path: Path) -> None:
    at = AppTest.from_file(APP, default_timeout=30)
    at.run()
    at.text_input(key="data_path").set_value(str(tmp_path / "missing.yaml")).run()
    assert not at.exception
    assert any("not found" in e.value for e in at.error)


def test_app_imports_when_started_like_streamlit_cli(tmp_path: Path) -> None:
    """`streamlit run` adds only app/ to sys.path; the app must still import."""
    import subprocess
    import sys

    code = (
        "import runpy, sys; sys.path[:] = [p for p in sys.path if p not in ('', '.')]; "
        f"sys.path.insert(0, {str(Path(APP).parent)!r}); "
        f"runpy.run_path({APP!r}, run_name='not_main')"
    )
    result = subprocess.run(
        [sys.executable, "-I", "-c", code], cwd=tmp_path, capture_output=True, text=True
    )
    assert "No module named" not in result.stderr, result.stderr
    assert result.returncode == 0, result.stderr
