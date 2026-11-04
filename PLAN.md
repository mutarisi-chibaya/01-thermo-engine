# 01 – Thermochemistry engine (`pyrothermo`)

**Time:** about 4–5 evenings
**Depends on:** nothing

The goal is a package that returns heat capacity, enthalpy, entropy and Gibbs energy for any species at any temperature, plus reaction ΔH, ΔG and equilibrium constant K. All later projects depend on this.

## Folder layout

```
01-thermo-engine/
  pyproject.toml
  README.md
  data/species.yaml          # you fill this in from NIST
  src/pyrothermo/
    __init__.py
    data.py                  # YAML loader
    species.py               # Species class
    formula.py               # formula parser, molar mass
    reactions.py             # Reaction class
  tests/
    data/janaf_reference.csv # you fill this in from JANAF
  notebooks/01_examples.ipynb
```

## The science

The NIST WebBook gives Shomate coefficients (A–H) per species over temperature ranges, with t = T/1000:

```
Cp           = A + B t + C t^2 + D t^3 + E / t^2                         [J/(mol K)]
H(T) - H298  = A t + B t^2/2 + C t^3/3 + D t^4/4 - E/t + F - H           [kJ/mol]
S(T)         = A ln(t) + B t + C t^2/2 + D t^3/3 - E/(2 t^2) + G         [J/(mol K)]
```

- Absolute enthalpy: H(T) = ΔfH°(298) + (H(T) − H298)
- Gibbs energy: G(T) = H(T) − T·S(T)
- Reaction: ΔG = Σ ν·G(products) − Σ ν·G(reactants), and K(T) = exp(−ΔG / RT)

## Your manual task first

Fill `data/species.yaml` for: Fe, FeO, Fe3O4, Fe2O3, C (graphite), CO, CO2, H2, H2O (gas), O2. Iron has several solid phases and melts at 1811 K, so record every temperature range NIST lists. Put the source URL next to each species.

Suggested schema:

```yaml
Fe2O3:
  phase: solid
  formula: Fe2O3
  dHf298: -825.50        # kJ/mol, copy from NIST
  source: https://webbook.nist.gov/...
  ranges:
    - {Tmin: 298, Tmax: 950, A: ..., B: ..., C: ..., D: ..., E: ..., F: ..., G: ..., H: ...}
```

(The dHf298 value above is only a placeholder for the format. Use NIST's number.)

Then fill `tests/data/janaf_reference.csv` with a few rows per species (T, Cp, S, H−H298) copied from the NIST-JANAF tables.

## Tasks

1. YAML schema and loader.
2. `Species` class with `cp(T)`, `h(T)`, `s(T)`, `g(T)` that picks the right range and raises a clear error outside the valid ranges. Accept scalars and NumPy arrays.
3. Formula parser ("Fe2O3" → {Fe: 2, O: 3}) and molar mass.
4. `Reaction` class built from a string like `"2 Fe + O2 = 2 FeO"` that checks the element balance and gives `dH(T)`, `dG(T)` and `K(T)`.
5. Notebook showing Cp of Fe across its phase changes, and ΔG of a few reactions.

## Tests

- Cp, H − H298 and S at 298.15, 500, 1000 and 1500 K match JANAF within 0.1%.
- An unbalanced reaction raises an error.
- H is continuous where ranges meet (except across a real phase change).
- Molar mass of Fe2O3 is 159.69 g/mol.

## Prompt for Claude Code

```
Read CLAUDE.md and 01-thermo-engine/PLAN.md. Set up 01-thermo-engine as an installable package called pyrothermo (src layout, pyproject.toml, pytest/ruff/mypy). Implement: a YAML loader for data/species.yaml (Shomate coefficients, multiple temperature ranges per species), a Species class with cp/h/s/g (vectorised, SI units, clear error outside valid ranges), a formula parser with molar mass, and a Reaction class parsed from strings like "2 Fe + O2 = 2 FeO" that checks element balance and returns dH, dG and K as functions of T. Write pytest tests comparing against tests/data/janaf_reference.csv (tolerance 0.1%). Propose the plan first, then implement. Do not add or change any coefficients in species.yaml.
```
