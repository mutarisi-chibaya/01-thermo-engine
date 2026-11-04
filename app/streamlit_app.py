"""Streamlit interface for pyrothermo.

Run from the 01-thermo-engine folder:

    streamlit run app/streamlit_app.py

Temperatures are entered and computed in K; degC is shown alongside for reading.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from pyrothermo import (
    FormulaError,
    Reaction,
    ReactionError,
    Species,
    SpeciesDataError,
    TemperatureRangeError,
    default_data_path,
    load_species,
    molar_mass_g_per_mol,
    parse_formula,
)

FAKE_YAML = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "fake_species.yaml"

# Categorical slots in fixed order (validated reference palette); max 4 series per chart.
SERIES_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
MAX_SERIES = len(SERIES_COLORS)
N_POINTS = 400

PROPERTIES: dict[str, tuple[str, str, float]] = {
    # label: (Species method, unit, display scale from SI)
    "Cp": ("cp", "J/(mol·K)", 1.0),
    "H − H298": ("h_minus_h298", "kJ/mol", 1e-3),
    "H (formation scale)": ("h", "kJ/mol", 1e-3),
    "S": ("s", "J/(mol·K)", 1.0),
    "G = H − TS": ("g", "kJ/mol", 1e-3),
}


@st.cache_resource(show_spinner=False)
def _load(path: str, mtime: float) -> dict[str, Species]:
    # mtime is part of the cache key so edits to the YAML are picked up.
    return load_species(path)


def species_table(species: list[Species], prop: str, T: np.ndarray) -> pd.DataFrame:
    """Long-format table of one property for several species, NaN outside valid ranges."""
    method, unit, scale = PROPERTIES[prop]
    frames = []
    for sp in species:
        mask = (sp.t_min <= T) & (sp.t_max >= T)
        values = np.full_like(T, np.nan)
        if np.any(mask):
            values[mask] = np.asarray(getattr(sp, method)(T[mask])) * scale
        frames.append(
            pd.DataFrame(
                {"species": sp.name, "T_K": T, "T_C": T - 273.15, f"{prop} [{unit}]": values}
            )
        )
    return pd.concat(frames, ignore_index=True)


def _base_layout(fig: go.Figure, y_title: str) -> go.Figure:
    fig.update_layout(
        xaxis_title="Temperature (K)",
        yaxis_title=y_title,
        hovermode="x unified",
        margin={"l": 10, "r": 10, "t": 30, "b": 10},
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "x": 0},
        height=420,
    )
    fig.update_xaxes(showgrid=False)
    return fig


def species_figure(species: list[Species], prop: str, T: np.ndarray) -> go.Figure:
    """Line chart of one property vs T, with phase changes marked by dotted lines."""
    method, unit, scale = PROPERTIES[prop]
    fig = go.Figure()
    for i, sp in enumerate(species):
        mask = (sp.t_min <= T) & (sp.t_max >= T)
        if not np.any(mask):
            continue
        Tm = T[mask]
        y = np.asarray(getattr(sp, method)(Tm)) * scale
        # Break the line at phase changes so jumps are not drawn as steep slopes.
        for Tt, _, _ in sp.phase_transitions():
            if Tm[0] < Tt < Tm[-1]:
                k = int(np.searchsorted(Tm, Tt, side="right"))
                Tm = np.insert(Tm, k, Tt)
                y = np.insert(y, k, np.nan)
        fig.add_trace(
            go.Scatter(
                x=Tm,
                y=y,
                customdata=Tm - 273.15,
                mode="lines",
                name=sp.name,
                line={"width": 2, "color": SERIES_COLORS[i]},
                hovertemplate=f"{sp.name}: %{{y:.3f}} {unit}<extra></extra>",
            )
        )
        for Tt, below, above in sp.phase_transitions():
            if T[0] <= Tt <= T[-1]:
                fig.add_vline(
                    x=Tt,
                    line={"width": 1, "dash": "dot", "color": "gray"},
                    annotation_text=f"{sp.name}: {below}→{above} ({Tt:g} K)",
                    annotation_font_size=11,
                )
    return _base_layout(fig, f"{prop} ({unit})")


def reaction_figure(rx: Reaction, T: np.ndarray, which: str) -> go.Figure:
    """ΔG (kJ/mol) or log10 K vs T for a reaction, as a single-axis line chart."""
    if which == "dG":
        y, title = np.asarray(rx.dG(T)) / 1000.0, "ΔG° (kJ/mol of reaction)"
    else:
        y, title = np.asarray(rx.log10K(T)), "log₁₀ K"
    fig = go.Figure(
        go.Scatter(
            x=T,
            y=y,
            mode="lines",
            name=title,
            line={"width": 2, "color": SERIES_COLORS[0]},
            hovertemplate="%{y:.3f}<extra></extra>",
        )
    )
    fig.add_hline(y=0, line={"width": 1, "color": "gray"})
    for Tt, name, below, above in rx.phase_transitions():
        fig.add_vline(
            x=Tt,
            line={"width": 1, "dash": "dot", "color": "gray"},
            annotation_text=f"{name}: {below}→{above}",
            annotation_font_size=11,
        )
    fig.update_layout(showlegend=False)
    return _base_layout(fig, title)


def _t_slider(label: str, lo: float, hi: float, key: str) -> tuple[float, float]:
    t0, t1 = st.slider(
        label,
        min_value=float(lo),
        max_value=float(hi),
        value=(float(lo), float(hi)),
        step=1.0,
        key=key,
    )
    st.caption(f"{t0:g}–{t1:g} K  =  {t0 - 273.15:.0f}–{t1 - 273.15:.0f} °C")
    return t0, t1


def species_tab(db: dict[str, Species]) -> None:
    names = sorted(db)
    chosen = st.multiselect(
        f"Species (up to {MAX_SERIES})",
        names,
        default=names[:1],
        max_selections=MAX_SERIES,
    )
    prop = st.radio("Property", list(PROPERTIES), horizontal=True)
    if not chosen:
        st.info("Pick at least one species.")
        return
    sel = [db[n] for n in chosen]
    lo, hi = min(s.t_min for s in sel), max(s.t_max for s in sel)
    t0, t1 = _t_slider("Temperature range (K)", lo, hi, "sp_T")
    T = np.linspace(t0, t1, N_POINTS)
    st.plotly_chart(species_figure(sel, prop, T), width="stretch")

    with st.expander("Data table, sources and ranges"):
        df = species_table(sel, prop, T)
        st.dataframe(df, width="stretch", hide_index=True)
        st.download_button("Download CSV", df.to_csv(index=False), "species_properties.csv")
        for sp in sel:
            spans = ", ".join(f"{r.Tmin:g}–{r.Tmax:g}" for r in sp.ranges)
            st.markdown(
                f"**{sp.name}** ({sp.formula}, {sp.molar_mass * 1000:.3f} g/mol): "
                f"ranges {spans} K · ΔfH°298 = {sp.data.dHf298:g} kJ/mol · source: {sp.data.source}"
            )


def reaction_tab(db: dict[str, Species]) -> None:
    st.caption(
        "Write coefficients before names and put spaces around '+', e.g. "
        "`2 Fe + O2 = 2 FeO` or `Fe2O3 + 3 CO = 2 Fe + 3 CO2`. "
        f"Available species: {', '.join(sorted(db))}"
    )
    eq = st.text_input("Reaction", value="", key="reaction_eq")
    if not eq.strip():
        return
    try:
        rx = Reaction.from_string(eq, db)
    except (ReactionError, FormulaError) as exc:
        st.error(str(exc))
        return
    lo, hi = rx.t_range
    if lo >= hi:
        st.error("The species in this reaction have no common temperature range.")
        return
    st.success(f"Balanced: {rx}  ·  valid {lo:g}–{hi:g} K")

    t0, t1 = _t_slider("Temperature range (K)", lo, hi, "rx_T")
    T = np.linspace(t0, t1, N_POINTS)
    c1, c2 = st.columns(2)
    c1.plotly_chart(reaction_figure(rx, T, "dG"), width="stretch")
    c2.plotly_chart(reaction_figure(rx, T, "logK"), width="stretch")

    st.subheader("At a single temperature")
    Tq = st.number_input(
        "T (K)", min_value=float(lo), max_value=float(hi), value=float(min(max(1273.15, lo), hi))
    )
    m = st.columns(4)
    m[0].metric("ΔH° (kJ/mol)", f"{rx.dH(Tq) / 1000:.2f}")
    m[1].metric("ΔS° (J/mol·K)", f"{rx.dS(Tq):.2f}")
    m[2].metric("ΔG° (kJ/mol)", f"{rx.dG(Tq) / 1000:.2f}")
    m[3].metric("log₁₀ K", f"{rx.log10K(Tq):.3f}")
    st.caption(
        f"T = {Tq:g} K = {Tq - 273.15:.1f} °C. "
        "Standard states: pure condensed phases, ideal gases at 1 bar."
    )

    with st.expander("Data table"):
        df = pd.DataFrame(
            {
                "T_K": T,
                "T_C": T - 273.15,
                "dH_kJ_per_mol": np.asarray(rx.dH(T)) / 1000,
                "dS_J_per_molK": rx.dS(T),
                "dG_kJ_per_mol": np.asarray(rx.dG(T)) / 1000,
                "log10K": rx.log10K(T),
            }
        )
        st.dataframe(df, width="stretch", hide_index=True)
        st.download_button("Download CSV", df.to_csv(index=False), "reaction.csv")


def formula_tab() -> None:
    f = st.text_input("Formula", value="Fe2O3", key="formula")
    if not f.strip():
        return
    try:
        comp = parse_formula(f)
        mm = molar_mass_g_per_mol(f)
    except FormulaError as exc:
        st.error(str(exc))
        return
    st.metric("Molar mass", f"{mm:.3f} g/mol")
    st.dataframe(
        pd.DataFrame({"element": list(comp), "mol per mol formula": list(comp.values())}),
        hide_index=True,
    )
    st.caption("Atomic weights: CIAAW abridged standard atomic weights 2024.")


def main() -> None:
    st.set_page_config(page_title="pyrothermo", layout="wide")
    st.title("pyrothermo")
    st.caption("Thermochemistry from NIST Shomate coefficients · SI units internally")

    with st.sidebar:
        st.header("Data")
        path_text = st.text_input("species.yaml", value=str(default_data_path()), key="data_path")
        demo = st.toggle("Use FAKE test data (demo only)", value=False, key="demo")
        with st.expander("Limitations"):
            st.markdown(
                "- Pure species in standard states; gases ideal at 1 bar.\n"
                "- No solutions, slags or activities; no kinetics.\n"
                "- Accuracy is that of the NIST Shomate fits; no extrapolation "
                "outside each species' listed ranges."
            )

    path = FAKE_YAML if demo else Path(path_text)
    if demo:
        st.warning("Showing FAKE coefficients from tests/fixtures. Numbers are meaningless.")
    try:
        db = _load(str(path), path.stat().st_mtime if path.is_file() else 0.0)
    except SpeciesDataError as exc:
        st.error(str(exc))
        st.info("Fill in data/species.yaml from the NIST WebBook, or turn on the demo toggle.")
        return

    t1, t2, t3 = st.tabs(["Species", "Reaction", "Molar mass"])
    try:
        with t1:
            species_tab(db)
        with t2:
            reaction_tab(db)
        with t3:
            formula_tab()
    except TemperatureRangeError as exc:
        st.error(str(exc))


if __name__ == "__main__":
    main()
