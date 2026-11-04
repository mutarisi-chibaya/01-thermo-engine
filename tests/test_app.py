"""Smoke tests for the Streamlit interface (FAKE demo data only)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("streamlit")
pytest.importorskip("plotly")

from streamlit.testing.v1 import AppTest

from app.streamlit_app import reaction_figure, species_figure, species_table
from pyrothermo import Reaction, Species

APP = str(Path(__file__).parents[1] / "app" / "streamlit_app.py")


def test_species_table_nan_outside_range(fake_db: dict[str, Species]) -> None:
    T = np.array([500.0, 2500.0])
    df = species_table([fake_db["Fe"], fake_db["O2"]], "Cp", T)
    fe = df[df.species == "Fe"]["Cp [J/(mol·K)]"].to_numpy()
    o2 = df[df.species == "O2"]["Cp [J/(mol·K)]"].to_numpy()
    assert fe[0] == pytest.approx(25.0)
    assert np.isnan(fe[1])
    np.testing.assert_allclose(o2, [30.0, 30.0])


def test_figures_build(fake_db: dict[str, Species]) -> None:
    T = np.linspace(300.0, 1900.0, 50)
    fig = species_figure([fake_db["Fe"]], "H − H298", T)
    assert len(fig.data) == 1
    # Line is broken (NaN inserted) at the 1000 K phase change.
    assert np.isnan(np.asarray(fig.data[0].y, dtype=float)).sum() == 1
    rx = Reaction.from_string("2 Fe + O2 = 2 FeO", fake_db)
    assert len(reaction_figure(rx, T, "dG").data) == 1
    assert len(reaction_figure(rx, T, "logK").data) == 1


def test_app_demo_mode_runs_reaction() -> None:
    at = AppTest.from_file(APP, default_timeout=30)
    at.run()
    at.toggle(key="demo").set_value(True).run()
    assert not at.exception
    assert any("FAKE" in w.value for w in at.warning)
    at.text_input(key="reaction_eq").set_value("2 Fe + O2 = 2 FeO").run()
    assert not at.exception
    assert any("Balanced" in s.value for s in at.success)
    labels = [m.label for m in at.metric]
    assert "ΔG° (kJ/mol)" in labels


def test_app_reports_unbalanced_reaction() -> None:
    at = AppTest.from_file(APP, default_timeout=30)
    at.run()
    at.toggle(key="demo").set_value(True).run()
    at.text_input(key="reaction_eq").set_value("Fe + O2 = FeO").run()
    assert not at.exception
    assert any("unbalanced" in e.value for e in at.error)


def test_app_without_data_shows_instructions(tmp_path: Path) -> None:
    at = AppTest.from_file(APP, default_timeout=30)
    at.run()
    at.text_input(key="data_path").set_value(str(tmp_path / "missing.yaml")).run()
    assert not at.exception
    assert any("not found" in e.value for e in at.error)
