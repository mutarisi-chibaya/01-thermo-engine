"""Plain-language wording, presets and unit handling for the Streamlit app.

Pure functions (no Streamlit), so they are unit tested. Temperatures are K in code;
the user's display unit is only applied when formatting or reading inputs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import numpy.typing as npt

from pyrothermo import Reaction, Species
from pyrothermo.interpret import gas_pressure_limit, gas_requirement, turning_points

Unit = Literal["°C", "K"]
ZERO_C = 273.15
T_REF = 298.15
J_PER_KWH = 3.6e6


# ---------------------------------------------------------------- units & formatting


def to_display(T_K: npt.ArrayLike, unit: Unit) -> npt.NDArray[np.float64]:
    """Convert K to the display unit."""
    T = np.asarray(T_K, dtype=float)
    return T - ZERO_C if unit == "°C" else T


def from_display(value: float, unit: Unit) -> float:
    """Convert a value in the display unit to K."""
    return value + ZERO_C if unit == "°C" else value


def fmt_T(T_K: float, unit: Unit) -> str:
    """'1000 °C (1273 K)' or '1273 K (1000 °C)'."""
    c, k = T_K - ZERO_C, T_K
    return f"{c:.0f} °C ({k:.0f} K)" if unit == "°C" else f"{k:.0f} K ({c:.0f} °C)"


def fmt_pct(x: float) -> str:
    """Fraction as a percentage that stays readable at the extremes."""
    if x >= 0.9999:
        return "more than 99.99 %"
    if x <= 0.0001:
        return "less than 0.01 %"
    return f"{x * 100:.1f} %" if 0.001 < x < 0.999 else f"{x * 100:.2f} %"


_SUP = str.maketrans("-0123456789", "⁻⁰¹²³⁴⁵⁶⁷⁸⁹")


def fmt_pressure(p_bar: float) -> str:
    """Pressure in bar, using '1.2 × 10⁻¹² bar' for very small or large values."""
    if 1e-3 <= p_bar < 1e4:
        return f"{p_bar:.3g} bar"
    mant, exp = f"{p_bar:.1e}".split("e")
    return f"{mant} × 10{str(int(exp)).translate(_SUP)} bar"


# ---------------------------------------------------------------- presets


@dataclass(frozen=True)
class Preset:
    label: str
    equation: str
    note: str


WUSTITE_NOTE = (
    "Real wüstite is slightly iron-deficient and is unstable below about 570 °C; "
    "the model treats FeO as a pure, stoichiometric phase."
)

PRESETS: tuple[Preset, ...] = (
    Preset(
        "Reduce wüstite (FeO) to iron with hydrogen — hydrogen DRI",
        "FeO + H2 = Fe + H2O",
        "The last and hardest step of hydrogen-based direct reduction. " + WUSTITE_NOTE,
    ),
    Preset(
        "Reduce wüstite (FeO) to iron with carbon monoxide — blast furnace",
        "FeO + CO = Fe + CO2",
        "The last reduction step in a blast furnace or gas-based DRI. " + WUSTITE_NOTE,
    ),
    Preset(
        "Reduce hematite (Fe2O3) to magnetite with CO",
        "3 Fe2O3 + CO = 2 Fe3O4 + CO2",
        "The first, easy step of iron-ore reduction, in the cool top of a blast furnace.",
    ),
    Preset(
        "Reduce magnetite (Fe3O4) to wüstite with CO",
        "Fe3O4 + CO = 3 FeO + CO2",
        "The middle step of iron-ore reduction. " + WUSTITE_NOTE,
    ),
    Preset(
        "Reduce magnetite (Fe3O4) to wüstite with hydrogen",
        "Fe3O4 + H2 = 3 FeO + H2O",
        "The middle step of hydrogen-based reduction. " + WUSTITE_NOTE,
    ),
    Preset(
        "Boudouard reaction: CO2 + carbon → carbon monoxide",
        "C + CO2 = 2 CO",
        "Coke gasification. It regenerates the CO that reduces the ore, and only runs at "
        "high temperature.",
    ),
    Preset(
        "Reduce hematite (Fe2O3) directly with carbon",
        "2 Fe2O3 + 3 C = 4 Fe + 3 CO2",
        "Overall carbothermic reduction, written with CO2 as the only gas.",
    ),
    Preset(
        "Oxidation of iron: iron + oxygen → wüstite",
        "2 Fe + O2 = 2 FeO",
        "How easily iron rusts at high temperature, e.g. scale on hot steel. Also the basis "
        "of an Ellingham diagram. " + WUSTITE_NOTE,
    ),
    Preset(
        "Water-gas shift: CO + steam → CO2 + hydrogen",
        "CO + H2O = CO2 + H2",
        "Converts CO and steam into hydrogen; it sets the H2/CO balance in mixed gases.",
    ),
)

CUSTOM = "Custom reaction (type your own)…"


# ---------------------------------------------------------------- plain-language text


def _side(names: list[str]) -> str:
    return " + ".join(names)


def describe_reaction(rx: Reaction, T_K: float, unit: Unit) -> list[str]:
    """Plain-language statements (Markdown) about ``rx`` at temperature ``T_K``.

    Every statement is derived from dG, dH and K; nothing is hard-coded per reaction.
    """
    out: list[str] = []
    dG = float(rx.dG(T_K)) / 1000.0
    dH = float(rx.dH(T_K)) / 1000.0
    products = _side(list(rx.products))
    reactants = _side(list(rx.reactants))
    at = fmt_T(T_K, unit)

    if abs(dG) < 1.0:
        out.append(
            f"**Close to balance.** At {at}, ΔG° is almost zero ({dG:+.1f} kJ/mol), so "
            "small changes in gas composition decide which way it goes."
        )
    elif dG < 0:
        out.append(
            f"**Goes forward.** At {at}, with every gas at 1 bar, the reaction runs towards "
            f"**{products}** (ΔG° = {dG:.1f} kJ/mol, negative)."
        )
    else:
        out.append(
            f"**Does not go forward on its own** at {at} with every gas at 1 bar; the reverse "
            f"(back to **{reactants}**) is favoured (ΔG° = +{dG:.1f} kJ/mol, positive). "
            "Changing the gas composition can still drive it forward — see below."
        )

    tps = turning_points(rx)
    lo, hi = rx.t_range
    if tps:
        for tp in tps:
            side = "above" if tp.forward_above else "below"
            out.append(
                f"**Turning point:** the direction reverses at about **{fmt_T(tp.T, unit)}** "
                f"— with all gases at 1 bar it runs forward {side} that temperature."
            )
    else:
        always = "favourable" if dG < 0 else "unfavourable"
        out.append(
            f"**No turning point** between {fmt_T(lo, unit)} and {fmt_T(hi, unit)}: with all "
            f"gases at 1 bar it is {always} across the whole data range."
        )

    req = gas_requirement(rx, T_K)
    if req is not None:
        out.append(
            f"**Gas needed:** at equilibrium (1 bar total) the gas holds "
            f"{fmt_pct(req.min_reactant_fraction)} {req.reactant} and "
            f"{fmt_pct(req.product_fraction)} {req.product}. The reaction runs forward only "
            f"while the gas contains **more than {fmt_pct(req.min_reactant_fraction)} "
            f"{req.reactant}**."
        )
    lim = gas_pressure_limit(rx, T_K)
    if lim is not None:
        if lim.side == "reactant":
            out.append(
                f"**{lim.gas} level:** it runs forward whenever the {lim.gas} partial pressure "
                f"is above **{fmt_pressure(lim.pressure_bar)}**, and backwards below that."
            )
        else:
            out.append(
                f"**{lim.gas} level:** it runs forward only while the {lim.gas} partial "
                f"pressure is kept below **{fmt_pressure(lim.pressure_bar)}**."
            )

    kind = "absorbs" if dH > 0 else "releases"
    label = "endothermic — it needs heat supplied" if dH > 0 else "exothermic — it gives off heat"
    out.append(
        f"**Heat:** it {kind} about **{abs(dH):.0f} kJ** per mole of reaction as written ({label})."
    )
    return out


ASSUMPTIONS = (
    "Assumes pure solids and liquids, ideal gases and full equilibrium. It says whether a "
    "reaction *can* happen, not how fast, and ignores slags, impurities and heat losses."
)


def heat_per_tonne_kwh(sp: Species, T_K: float) -> float:
    """Theoretical energy (kWh per tonne) to heat ``sp`` from 25 °C to ``T_K``.

    Includes latent heat of phase changes on the way; no losses. Uses H - H298.
    """
    return float(sp.h_minus_h298(T_K)) / sp.molar_mass * 1000.0 / J_PER_KWH


PROPERTY_HELP: dict[str, str] = {
    "Heat needed from 25 °C (H − H298)": (
        "How much heat it takes to warm one mole from 25 °C up to each temperature, "
        "including melting or other phase changes on the way (the steps in the line)."
    ),
    "Heat capacity (Cp)": (
        "Heat needed to raise one mole by one degree at that temperature. Spikes and "
        "jumps show where the material changes structure."
    ),
    "Entropy (S)": "A measure of disorder; it always rises with temperature.",
    "Gibbs energy (G = H − TS)": (
        "The quantity that decides which way reactions go: reactions run towards lower G."
    ),
    "Enthalpy, formation scale (H)": (
        "Absolute heat content, measured from the elements at 25 °C. Differences between "
        "products and reactants give the reaction heat."
    ),
}
BASIC_PROPERTIES = ("Heat needed from 25 °C (H − H298)", "Heat capacity (Cp)")

GLOSSARY: dict[str, str] = {
    "ΔG° (Gibbs energy change)": (
        "Negative → the reaction can run forward; positive → it runs backwards unless "
        "conditions are changed. The ° means standard conditions: pure solids, each gas at 1 bar."
    ),
    "K (equilibrium constant)": (
        "How far the reaction goes. K ≫ 1: mostly products. K ≪ 1: mostly reactants. For "
        "gas reductions such as FeO + H2 = Fe + H2O, K equals the H2O/H2 ratio at equilibrium."
    ),
    "log₁₀ K": "K on a log scale: 0 means K = 1, +2 means 100 : 1 for products, −2 means 1 : 100.",
    "ΔH° (reaction enthalpy)": "Heat absorbed (positive, endothermic) or released (negative).",
    "Equilibrium": (
        "The state a reaction settles into if given enough time. Real furnaces may not reach it."
    ),
    "Partial pressure": (
        "The share of the total pressure due to one gas: 20 % O2 in air at 1 bar is 0.2 bar."
    ),
    "Phase change": "A change of structure or state, e.g. solid → liquid, or α-iron → γ-iron.",
}
