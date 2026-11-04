"""Streamlit interface for pyrothermo.

Run from the 01-thermo-engine folder:

    streamlit run app/streamlit_app.py

Serves two audiences at once: a plain-language summary first, the specialist numbers
(ΔH°, ΔS°, ΔG°, log K, Cp, S, G) alongside. Temperatures are computed in K; the user
chooses °C or K for display.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from app.plain import (
    ASSUMPTIONS,
    BASIC_PROPERTIES,
    CUSTOM,
    GLOSSARY,
    PRESETS,
    PROPERTY_HELP,
    Unit,
    describe_reaction,
    fmt_T,
    from_display,
    heat_per_tonne_kwh,
    to_display,
)
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

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEMO_VIDEO = PROJECT_ROOT / "demo" / "output" / "pyrothermo_demo.mp4"

# Categorical slots in fixed order (validated reference palette); max 4 series per chart.
SERIES_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
MAX_SERIES = len(SERIES_COLORS)
N_POINTS = 400
DEFAULT_T_K = 1273.15  # 1000 °C

PROPERTIES: dict[str, tuple[str, str, float]] = {
    # label: (Species method, unit, display scale from SI)
    "Heat needed from 25 °C (H − H298)": ("h_minus_h298", "kJ/mol", 1e-3),
    "Heat capacity (Cp)": ("cp", "J/(mol·K)", 1.0),
    "Entropy (S)": ("s", "J/(mol·K)", 1.0),
    "Gibbs energy (G = H − TS)": ("g", "kJ/mol", 1e-3),
    "Enthalpy, formation scale (H)": ("h", "kJ/mol", 1e-3),
}


def display_path(p: Path) -> str:
    """Path relative to the project folder for display (absolute if outside it)."""
    try:
        return p.resolve().relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return str(p)


def resolve_path(text: str) -> Path:
    """Interpret a user-entered path; relative paths are relative to the project folder."""
    p = Path(text.strip())
    return p if p.is_absolute() else PROJECT_ROOT / p


@st.cache_resource(show_spinner=False)
def _load(path: str, mtime: float) -> dict[str, Species]:
    # mtime is part of the cache key so edits to the YAML are picked up.
    return load_species(path)


# ---------------------------------------------------------------- tables & charts


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


def _base_layout(fig: go.Figure, y_title: str, unit: Unit) -> go.Figure:
    fig.update_layout(
        xaxis_title=f"Temperature ({unit})",
        yaxis_title=y_title,
        hovermode="x unified",
        margin={"l": 10, "r": 10, "t": 30, "b": 10},
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "x": 0},
        height=400,
    )
    fig.update_xaxes(showgrid=False)
    return fig


def mark_transitions(
    fig: go.Figure, items: list[tuple[float, str]], lo: float, hi: float, unit: Unit
) -> None:
    """Dotted vertical line per phase-change temperature (K) in [lo, hi], one vertical label.

    Labels at the same temperature are merged so they never stack on top of each other.
    """
    grouped: dict[float, list[str]] = {}
    for Tt, label in items:
        if lo <= Tt <= hi and label not in grouped.setdefault(Tt, []):
            grouped[Tt].append(label)
    for Tt, labels in sorted(grouped.items()):
        fig.add_vline(
            x=float(to_display(Tt, unit)),
            line={"width": 1, "dash": "dot", "color": "gray"},
            annotation_text=" / ".join(labels),
            annotation_textangle=-90,
            annotation_position="bottom left",
            annotation_font={"size": 10, "color": "gray"},
        )


def species_figure(species: list[Species], prop: str, T: np.ndarray, unit: Unit = "K") -> go.Figure:
    """Line chart of one property vs T (K in, ``unit`` on the axis), phase changes marked."""
    method, punit, scale = PROPERTIES[prop]
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
                x=to_display(Tm, unit),
                y=y,
                mode="lines",
                name=sp.name,
                line={"width": 2, "color": SERIES_COLORS[i]},
                hovertemplate=f"{sp.name}: %{{y:.3f}} {punit}<extra></extra>",
            )
        )
    transitions = [
        (Tt, f"{sp.name} {below}→{above}")
        for sp in species
        for Tt, below, above in sp.phase_transitions()
    ]
    mark_transitions(fig, transitions, float(T[0]), float(T[-1]), unit)
    return _base_layout(fig, f"{prop.split(' (')[0]} ({punit})", unit)


def reaction_figure(
    rx: Reaction, T: np.ndarray, which: str, unit: Unit = "K", mark_T: float | None = None
) -> go.Figure:
    """ΔG° (kJ/mol) or log10 K vs T (K in) as a single-axis line chart, with a T marker."""
    title = "ΔG° (kJ/mol of reaction)" if which == "dG" else "log₁₀ K"

    def fn(t: np.ndarray) -> np.ndarray:
        return np.asarray(rx.dG(t)) / 1000.0 if which == "dG" else np.asarray(rx.log10K(t))

    fig = go.Figure(
        go.Scatter(
            x=to_display(T, unit),
            y=fn(T),
            mode="lines",
            name=title,
            line={"width": 2, "color": SERIES_COLORS[0]},
            hovertemplate="%{y:.3f}<extra></extra>",
        )
    )
    fig.add_hline(y=0, line={"width": 1, "color": "gray"})
    if mark_T is not None and T[0] <= mark_T <= T[-1]:
        fig.add_trace(
            go.Scatter(
                x=to_display([mark_T], unit),
                y=fn(np.array([mark_T])),
                mode="markers",
                name="chosen temperature",
                marker={
                    "size": 10,
                    "color": SERIES_COLORS[0],
                    "line": {"width": 2, "color": "white"},
                },
                hoverinfo="skip",
            )
        )
    transitions = [(Tt, f"{n} {b}→{a}") for Tt, n, b, a in rx.phase_transitions()]
    mark_transitions(fig, transitions, float(T[0]), float(T[-1]), unit)
    fig.update_layout(showlegend=False)
    return _base_layout(fig, title, unit)


# ---------------------------------------------------------------- inputs


def _t_range_slider(label: str, lo: float, hi: float, unit: Unit, key: str) -> tuple[float, float]:
    """Range slider in the display unit; returns (lo, hi) in K."""
    dlo, dhi = float(to_display(lo, unit)), float(to_display(hi, unit))
    a, b = st.slider(
        label, min_value=dlo, max_value=dhi, value=(dlo, dhi), step=1.0, key=f"{key}_{unit}"
    )
    return from_display(a, unit), from_display(b, unit)


def _t_input(label: str, lo: float, hi: float, unit: Unit, key: str) -> float:
    """Single temperature in the display unit, clipped to [lo, hi] K; returns K."""
    default = min(max(DEFAULT_T_K, lo), hi)
    value = st.number_input(
        f"{label} ({unit})",
        min_value=float(to_display(lo, unit)),
        max_value=float(to_display(hi, unit)),
        value=float(to_display(default, unit)),
        step=10.0,
        key=f"{key}_{unit}",
    )
    return from_display(float(value), unit)


# ---------------------------------------------------------------- tabs


def start_tab() -> None:
    st.markdown(
        """
### What this app does
It answers **"can this reaction happen, and under what conditions?"** for iron- and
steelmaking reactions, using published thermodynamic data (NIST). Each answer comes in
two forms: **a plain-language summary** first, and **the specialist numbers** next to it.

### Try it in three steps
1. Open **Reactions** and pick *"Reduce wüstite (FeO) to iron with hydrogen"*.
2. Set the **temperature** (try 1000 °C). Read the **In plain language** box: it tells you
   whether the reaction goes forward, where it turns around, and **how much hydrogen the gas
   needs** for it to work.
3. Scroll down to the charts: below the zero line the reaction runs forward.
   Specialists will find ΔH°, ΔS°, ΔG° and log K beside the summary.

The **Substances** tab shows how a material behaves when heated, including the energy to
heat one tonne. **Molar mass** converts a formula into g/mol.

### What it cannot tell you
"""
        + ASSUMPTIONS
        + """ Use it for "is this possible, and roughly where?", not for plant design.
"""
    )
    if DEMO_VIDEO.is_file():
        with st.expander("Watch the 3-minute demo video"):
            st.video(str(DEMO_VIDEO))
    with st.expander("Glossary of terms"):
        for term, text in GLOSSARY.items():
            st.markdown(f"**{term}** — {text}")


def reaction_tab(db: dict[str, Species], unit: Unit) -> None:
    available = [p for p in PRESETS if all(n in db for n in _names(p.equation))]
    labels = [p.label for p in available] + [CUSTOM]
    choice = st.selectbox("Reaction", labels, key="preset")
    if choice == CUSTOM:
        st.caption(
            "Write coefficients before names and put spaces around '+', e.g. "
            f"`2 Fe + O2 = 2 FeO`. Available species: {', '.join(sorted(db))}"
        )
        eq = st.text_input("Reaction equation", value="", key="reaction_eq")
        if not eq.strip():
            return
    else:
        preset = available[labels.index(choice)]
        eq = preset.equation
        st.caption(f"`{eq}` — {preset.note}")
    try:
        rx = Reaction.from_string(eq, db)
    except (ReactionError, FormulaError) as exc:
        st.error(str(exc))
        return
    lo, hi = rx.t_range
    if lo >= hi:
        st.error("The species in this reaction have no common temperature range.")
        return
    st.success(f"Balanced: {rx}  ·  data valid {fmt_T(lo, unit)} to {fmt_T(hi, unit)}")

    Tq = _t_input("Temperature", lo, hi, unit, "rx_T")

    left, right = st.columns([3, 2], gap="large")
    with left:
        st.subheader("In plain language")
        for line in describe_reaction(rx, Tq, unit):
            st.markdown(f"- {line}")
        st.caption(ASSUMPTIONS)
    with right:
        st.subheader("For specialists")
        m1, m2 = st.columns(2)
        m1.metric("ΔH° (kJ/mol)", f"{rx.dH(Tq) / 1000:.2f}")
        m2.metric("ΔS° (J/mol·K)", f"{rx.dS(Tq):.2f}")
        m3, m4 = st.columns(2)
        m3.metric("ΔG° (kJ/mol)", f"{rx.dG(Tq) / 1000:.2f}")
        m4.metric("log₁₀ K", f"{rx.log10K(Tq):.3f}")
        st.caption(
            f"At {fmt_T(Tq, unit)}. Standard states: pure condensed phases, each gas at 1 bar."
        )

    st.subheader("How it changes with temperature")
    t0, t1 = _t_range_slider("Chart temperature range", lo, hi, unit, "rx_range")
    T = np.linspace(t0, t1, N_POINTS)
    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(reaction_figure(rx, T, "dG", unit, Tq), width="stretch")
        st.caption(
            "Below the zero line the reaction runs forward; above it, backwards. "
            "The dot is your chosen temperature."
        )
    with c2:
        st.plotly_chart(reaction_figure(rx, T, "logK", unit, Tq), width="stretch")
        st.caption("log₁₀ K above 0 means products are favoured; each step of 1 is a factor of 10.")

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


def _names(equation: str) -> list[str]:
    out = []
    for side in equation.replace("->", "=").split("="):
        for term in side.split(" + "):
            out.append(term.strip().split()[-1])
    return out


def species_tab(db: dict[str, Species], unit: Unit) -> None:
    names = sorted(db)
    chosen = st.multiselect(
        f"Substances (up to {MAX_SERIES})",
        names,
        default=["Fe"] if "Fe" in names else names[:1],
        max_selections=MAX_SERIES,
    )
    if not chosen:
        st.info("Pick at least one substance.")
        return
    sel = [db[n] for n in chosen]
    lo, hi = min(s.t_min for s in sel), max(s.t_max for s in sel)

    st.subheader("Energy to heat one tonne")
    Tq = _t_input("Heat up to", max(lo, 298.15), hi, unit, "sp_T")
    cols = st.columns(len(sel))
    for col, sp in zip(cols, sel, strict=True):
        if sp.t_min <= Tq <= sp.t_max and sp.t_min <= 298.15:
            kwh = heat_per_tonne_kwh(sp, Tq)
            col.metric(
                f"{sp.name}: 25 °C → {fmt_T(Tq, unit).split(' (')[0]}",
                f"{kwh:,.0f} kWh/t",
                f"{kwh * 3.6e-3:.2f} GJ/t",
                delta_color="off",
            )
        else:
            col.metric(sp.name, "—", "outside data range", delta_color="off")
    st.caption(
        "Theoretical minimum heat (sensible heat plus any melting or phase-change heat), with "
        "no furnace losses. Gases are taken as starting as gas at 25 °C."
    )

    st.subheader("Properties against temperature")
    advanced = st.toggle("Show advanced properties (entropy, Gibbs energy, enthalpy)", key="adv")
    options = list(PROPERTIES) if advanced else list(BASIC_PROPERTIES)
    prop = st.radio("Property", options, horizontal=True, key=f"prop_{advanced}")
    t0, t1 = _t_range_slider("Chart temperature range", lo, hi, unit, "sp_range")
    T = np.linspace(t0, t1, N_POINTS)
    st.plotly_chart(species_figure(sel, prop, T, unit), width="stretch")
    st.caption(PROPERTY_HELP[prop] + " Dotted lines mark phase changes.")

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
    st.caption(
        "Iron- and steelmaking thermodynamics from NIST data · "
        "plain-language answers and specialist numbers"
    )

    with st.sidebar:
        st.header("Settings")
        unit: Unit = st.radio("Temperature unit", ["°C", "K"], horizontal=True, key="unit")
        with st.expander("Data file"):
            path_text = st.text_input(
                "species.yaml", value=display_path(default_data_path()), key="data_path"
            )
        with st.expander("Limitations"):
            st.markdown(
                "- Pure substances in standard states; gases ideal, 1 bar.\n"
                "- No solutions, slags or activities; no kinetics; no heat losses.\n"
                "- Accuracy is that of the NIST fits (documented in the README); no "
                "extrapolation outside each substance's data range."
            )

    path = resolve_path(path_text)
    try:
        db = _load(str(path), path.stat().st_mtime if path.is_file() else 0.0)
    except SpeciesDataError as exc:
        st.error(str(exc).replace(str(path), display_path(path)))
        st.info("Build the data with scripts/build_reference_data.py (see README).")
        return

    t0, t1, t2, t3 = st.tabs(["Start here", "Reactions", "Substances", "Molar mass"])
    try:
        with t0:
            start_tab()
        with t1:
            reaction_tab(db, unit)
        with t2:
            species_tab(db, unit)
        with t3:
            formula_tab()
    except TemperatureRangeError as exc:
        st.error(str(exc))


if __name__ == "__main__":
    main()
