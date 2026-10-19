"""Structured inorganic additions, adapted to the shared chemistry normalizer."""
from __future__ import annotations

from decimal import Decimal
import math
from typing import Any

from adsorbent_recommender.catalog import clean_text
from normalization_rules import _species_to_ions, _species_molar_mass, _to_molar

ADDITIVE_INPUT_VERSION = 1
ADDITIVE_UNITS = ("mM", "µM", "mol/L", "mg/L", "µg/L", "g/L")
COMMON_SPECIES = ("NaCl", "CaCl2", "MgCl2", "KCl", "NaHCO3", "KHCO3", "Na2SO4", "MgSO4",
                  "CaSO4", "NaNO3", "KNO3", "Ca(NO3)2", "Mg(NO3)2", "Na2CO3", "NaH2PO4",
                  "Na2HPO4", "KH2PO4", "K2HPO4")


class AdditiveInputError(ValueError):
    pass


def additive_unit(value: Any) -> str:
    key = str(value or "").strip().replace("µ", "u").replace("μ", "u").replace(" ", "").casefold()
    aliases = {"mm": "mM", "mmol/l": "mM", "um": "µM", "umol/l": "µM", "m": "mol/L", "mol/l": "mol/L",
               "mg/l": "mg/L", "ug/l": "µg/L", "g/l": "g/L"}
    if key not in aliases:
        raise AdditiveInputError(f"Unit must be one of {', '.join(ADDITIVE_UNITS)}.")
    return aliases[key]


def validate_additives(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list) or len(value) > 50:
        raise AdditiveInputError("Inorganic additives must be a list of at most 50 species/concentration/unit entries.")
    result, identities = [], set()
    for index, row in enumerate(value, 1):
        if not isinstance(row, dict):
            raise AdditiveInputError(f"Additive {index} must contain species, concentration, and unit fields.")
        species = clean_text(row.get("species"))
        ions = _species_to_ions(species)
        if not species or len(species) > 200 or not ions or any(token in species for token in ("_", ";", "&&", "\n")):
            raise AdditiveInputError(f"Additive {index}: choose a supported species or enter a recognized chemical name/formula.")
        amount = row.get("concentration")
        if isinstance(amount, bool) or not isinstance(amount, (int, float)) or not math.isfinite(amount) or amount <= 0:
            raise AdditiveInputError(f"Additive {index}: concentration must be finite and greater than zero. Remove an entry to represent no addition.")
        unit = additive_unit(row.get("unit"))
        molar = _to_molar(str(amount), unit, species)
        if molar is None or not math.isfinite(molar) or molar <= 0:
            raise AdditiveInputError(f"Additive {index}: {unit} cannot be converted for {species}. Choose a molar unit or a species with known molar mass.")
        signature = (tuple(sorted(ions)), _species_molar_mass(species))
        if signature in identities:
            raise AdditiveInputError(f"Additive {index}: {species} duplicates an existing species or alias. Keep one total concentration per species.")
        identities.add(signature)
        result.append({"species": species, "concentration": float(amount), "unit": unit})
    return result


def database_additive_text(entries: list[dict[str, Any]]) -> str:
    """The database parser receives unambiguous tokens constructed from fields."""
    entries = validate_additives(entries)
    return "; ".join(f"{row['species']}_{format(Decimal(str(row['concentration'])), 'f')}_{row['unit']}" for row in entries)
