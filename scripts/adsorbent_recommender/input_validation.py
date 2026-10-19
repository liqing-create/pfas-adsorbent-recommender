"""Shared numeric rules for submitted screening inputs; no UI/model imports."""
from __future__ import annotations

import math
from numbers import Real
from typing import Any

SCREENING_BEHAVIOR_VERSION = 2
DEFAULT_TEMPERATURE_C = 25.0
NONNEGATIVE_WATER_FIELDS = frozenset({"organic_carbon_mg_l", "tds_mg_l", "ionic_strength_mol_l"})
WATER_NUMBER_ALIASES = {
    "pH": ("pH", "ph"),
    "organic_carbon_mg_l": ("organic_carbon_mg_l", "organic_carbon_mg_L", "doc_mg_l", "DOC_mg_L", "toc_mg_l", "TOC_mg_L"),
    "tds_mg_l": ("tds_mg_l", "TDS_mg_L"),
    "ionic_strength_mol_l": ("ionic_strength_mol_l", "ionic_strength_mol_L"),
}


class InputValidationError(ValueError):
    pass


def finite_number(value: Any, field: str, *, optional: bool = False,
                  nonnegative: bool = False, positive: bool = False,
                  allow_text: bool = False) -> float | None:
    if value is None or (allow_text and isinstance(value, str) and not value.strip()):
        if optional:
            return None
    if isinstance(value, bool) or not (isinstance(value, Real) or (allow_text and isinstance(value, str))):
        raise InputValidationError(f"{field} must be a finite number{' or left blank' if optional else ''}.")
    try:
        number = float(value)
    except (ValueError, TypeError, OverflowError) as exc:
        raise InputValidationError(f"{field} must be a finite number{' or left blank' if optional else ''}.") from exc
    if not math.isfinite(number):
        raise InputValidationError(f"{field} must be a finite number{' or left blank' if optional else ''}.")
    if positive and number <= 0:
        raise InputValidationError(f"{field} must be greater than zero.")
    if nonnegative and number < 0:
        raise InputValidationError(f"{field} must be zero or greater.")
    return number


def validate_water_numbers(water: Any) -> dict:
    if not isinstance(water, dict):
        raise InputValidationError("scenario.water must be an object when supplied.")
    checked = dict(water)
    for field, aliases in WATER_NUMBER_ALIASES.items():
        for alias in aliases:
            if alias in checked:
                checked[alias] = finite_number(checked[alias], f"water.{alias}", optional=True,
                    nonnegative=field in NONNEGATIVE_WATER_FIELDS, allow_text=True)
    return checked


def validate_scenario_numbers(scenario: Any) -> dict:
    if not isinstance(scenario, dict):
        raise InputValidationError("Each PFAS scenario must be an object.")
    checked = dict(scenario)
    checked["initial_concentration_mg_l"] = finite_number(scenario.get("initial_concentration_mg_l"),
        "scenario.initial_concentration_mg_l", positive=True, allow_text=True)
    checked["temperature_c"] = finite_number(scenario.get("temperature_c"), "Temperature", optional=True, allow_text=True)
    checked["water"] = validate_water_numbers(scenario.get("water") if scenario.get("water") is not None else {})
    return checked


def effective_temperature(scenario: dict, fixed_conditions: dict) -> float:
    value = finite_number(scenario.get("temperature_c"), "Temperature", optional=True, allow_text=True)
    if value is None:
        value = next((value for key, value in fixed_conditions.items() if key.startswith("Temperature_(")), DEFAULT_TEMPERATURE_C)
    return finite_number(value, "Effective temperature")
