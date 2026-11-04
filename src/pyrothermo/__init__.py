"""pyrothermo: Shomate-based species and reaction thermochemistry.

Units are SI throughout (J, mol, K, Pa); temperatures are in K.
"""

from pyrothermo.data import (
    SpeciesData,
    SpeciesDataError,
    default_data_path,
    load_species_data,
)
from pyrothermo.formula import FormulaError, molar_mass, molar_mass_g_per_mol, parse_formula
from pyrothermo.reactions import R, Reaction, ReactionError
from pyrothermo.shomate import ShomateRange
from pyrothermo.species import Species, TemperatureRangeError, load_species

__all__ = [
    "FormulaError",
    "R",
    "Reaction",
    "ReactionError",
    "ShomateRange",
    "Species",
    "SpeciesData",
    "SpeciesDataError",
    "TemperatureRangeError",
    "default_data_path",
    "load_species",
    "load_species_data",
    "molar_mass",
    "molar_mass_g_per_mol",
    "parse_formula",
]

__version__ = "0.1.0"
