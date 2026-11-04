"""Chemical formula parsing and molar mass.

Atomic weights are the CIAAW/IUPAC *abridged* standard atomic weights
(table "Abridged Standard Atomic Weights 2024", based on Atomic Weights 2021),
https://www.ciaaw.org/abridged-atomic-weights.htm, accessed 2026-11-04.
Only elements relevant to iron/steelmaking and slags are included; add more
from the same table if needed.
"""

from __future__ import annotations

import re
from collections import defaultdict

# g/mol, CIAAW abridged standard atomic weights 2024 (uncertainty in comment).
ATOMIC_WEIGHTS_G_PER_MOL: dict[str, float] = {
    "H": 1.0080,  # +/- 0.0002
    "C": 12.011,  # +/- 0.002
    "N": 14.007,  # +/- 0.001
    "O": 15.999,  # +/- 0.001
    "Na": 22.990,  # +/- 0.001
    "Mg": 24.305,  # +/- 0.002
    "Al": 26.982,  # +/- 0.001
    "Si": 28.085,  # +/- 0.001
    "P": 30.974,  # +/- 0.001
    "S": 32.06,  # +/- 0.02
    "Ca": 40.078,  # +/- 0.004
    "Ti": 47.867,  # +/- 0.001
    "Cr": 51.996,  # +/- 0.001
    "Mn": 54.938,  # +/- 0.001
    "Fe": 55.845,  # +/- 0.002
    "Ni": 58.693,  # +/- 0.001
}

_TOKEN = re.compile(r"([A-Z][a-z]?)|(\()|(\))|(\d+(?:\.\d+)?|\.\d+)")


class FormulaError(ValueError):
    """Raised for a formula that cannot be parsed or has unknown elements."""


def parse_formula(formula: str) -> dict[str, float]:
    """Parse a formula such as ``"Fe2O3"`` or ``"Ca(OH)2"`` into element counts.

    Decimal subscripts are allowed (e.g. ``"Fe0.947O"`` for wustite).

    Args:
        formula: Chemical formula without charges or phase labels.

    Returns:
        Mapping of element symbol to amount (mol of element per mol of formula).

    Raises:
        FormulaError: on syntax errors or empty formulas.
    """
    text = formula.strip()
    if not text:
        raise FormulaError("empty formula")

    stack: list[defaultdict[str, float]] = [defaultdict(float)]
    last: dict[str, float] | None = None  # the group/element a number would multiply
    pos = 0
    while pos < len(text):
        m = _TOKEN.match(text, pos)
        if m is None:
            raise FormulaError(f"unexpected character {text[pos]!r} in formula {formula!r}")
        element, open_p, close_p, number = m.groups()
        if element:
            last = {element: 1.0}
            stack[-1][element] += 1.0
        elif open_p:
            stack.append(defaultdict(float))
            last = None
        elif close_p:
            if len(stack) == 1:
                raise FormulaError(f"unmatched ')' in formula {formula!r}")
            group = stack.pop()
            for el, n in group.items():
                stack[-1][el] += n
            last = dict(group)
        else:
            if last is None:
                raise FormulaError(f"number without element or group in formula {formula!r}")
            mult = float(number)
            if mult <= 0:
                raise FormulaError(f"subscript must be positive in formula {formula!r}")
            for el, n in last.items():
                stack[-1][el] += n * (mult - 1.0)
            last = None
        pos = m.end()

    if len(stack) != 1:
        raise FormulaError(f"unmatched '(' in formula {formula!r}")
    return {el: n for el, n in stack[0].items() if n != 0.0}


def molar_mass(formula: str) -> float:
    """Molar mass of ``formula`` in kg/mol (SI).

    Raises:
        FormulaError: if the formula is invalid or contains an element not in
            :data:`ATOMIC_WEIGHTS_G_PER_MOL`.
    """
    return molar_mass_g_per_mol(formula) / 1000.0


def molar_mass_g_per_mol(formula: str) -> float:
    """Molar mass of ``formula`` in g/mol (for display)."""
    total = 0.0
    for el, n in parse_formula(formula).items():
        if el not in ATOMIC_WEIGHTS_G_PER_MOL:
            raise FormulaError(f"no atomic weight for element {el!r}; add it from the CIAAW table")
        total += n * ATOMIC_WEIGHTS_G_PER_MOL[el]
    return total
