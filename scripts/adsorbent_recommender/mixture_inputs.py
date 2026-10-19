"""Concentration conversion and transactional CSV mixture preparation."""
from __future__ import annotations

import csv
import io
import math
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Callable

from adsorbent_recommender.pfas_descriptors import prepare_structure

CONCENTRATION_UNITS = ("ng/L", "µg/L", "mg/L")
FACTORS_MG_L = {"ng/L": Decimal("0.000001"), "µg/L": Decimal("0.001"), "mg/L": Decimal("1")}
MAX_CSV_BYTES = 1_000_000
MAX_CSV_ROWS = 100
EXAMPLE_CSV_PATH = Path(__file__).parent / "examples" / "pfas_mixture_example.csv"


class MixtureInputError(ValueError):
    pass


def normalize_unit(unit: Any) -> str:
    key = str(unit or "").strip().replace("μ", "u").replace("µ", "u").replace(" ", "").casefold()
    aliases = {"ng/l": "ng/L", "ug/l": "µg/L", "mg/l": "mg/L"}
    if key not in aliases:
        raise MixtureInputError("Use ng/L, ug/L (or µg/L), or mg/L; every concentration needs a unit.")
    return aliases[key]


def to_mg_l(value: Any, unit: Any, *, allow_zero: bool = False) -> float:
    unit = normalize_unit(unit)
    try:
        number = Decimal(str(value).strip())
        converted = float(number * FACTORS_MG_L[unit])
    except (InvalidOperation, ValueError, TypeError, OverflowError) as exc:
        raise MixtureInputError("Enter a quantified numeric concentration; ND and < or > qualifiers cannot be imported as numbers.") from exc
    if not number.is_finite() or not math.isfinite(converted) or converted < 0 or (converted == 0 and not allow_zero):
        raise MixtureInputError("Concentration must be finite and greater than zero. ND and zero values are not substituted automatically.")
    return converted


def from_mg_l(value: Any, unit: Any) -> float:
    result = float(Decimal(str(value)) / FACTORS_MG_L[normalize_unit(unit)])
    if not math.isfinite(result):
        raise MixtureInputError("The concentration cannot be represented in this unit.")
    return result


def read_mixture_csv(payload: bytes) -> list[dict[str, Any]]:
    if len(payload) > MAX_CSV_BYTES:
        raise MixtureInputError("CSV files must be smaller than 1 MB.")
    try:
        text = payload.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise MixtureInputError("Save the file as CSV UTF-8 and upload it again. The example file uses this encoding.") from exc
    reader = csv.DictReader(io.StringIO(text, newline=""), strict=True)
    try:
        columns = [str(name).strip().casefold() for name in reader.fieldnames or []]
        if len(columns) != len(set(columns)):
            raise MixtureInputError("CSV column names must be unique, including capitalization and spaces.")
        missing = sorted({"pfas", "concentration", "unit"} - set(columns))
        if missing:
            raise MixtureInputError("Missing CSV columns: " + ", ".join(missing) + ". Download the example for the required format.")
        reader.fieldnames = columns
        rows = []
        for row in reader:
            if None in row or any(value is None for value in row.values()):
                raise MixtureInputError(f"CSV line {reader.line_num} does not have the same number of fields as the header.")
            if not any(str(value).strip() for value in row.values()):
                continue
            rows.append({**{key: str(value).strip() for key, value in row.items()}, "csv_row": reader.line_num})
            if len(rows) > MAX_CSV_ROWS:
                raise MixtureInputError(f"Import at most {MAX_CSV_ROWS} PFAS rows at a time.")
    except csv.Error as exc:
        raise MixtureInputError("The CSV is malformed. Check commas and quoted fields, or start with the example file.") from exc
    if not rows:
        raise MixtureInputError("The CSV has no PFAS rows. Add at least one row below the header.")
    return rows


def _existing_structure_keys(entries: list[dict], resolver: Callable) -> set[str]:
    keys = set()
    for entry in entries:
        key = entry.get("structure_key")
        if not key:
            pfas = {name: entry[name] for name in ("identity", "identity_type", "confirmed_smiles", "structure_confirmed") if name in entry}
            try:
                lookup = resolver(pfas)
                if lookup.ready:
                    _, key = prepare_structure(lookup.canonical_smiles)
            except (ValueError, RuntimeError):
                pass
        if key:
            keys.add(key)
    return keys


def _first_text(row: dict, *columns: str) -> str:
    for column in columns:
        value = str(row.get(column) or "").strip()
        if value.casefold() not in {"", "nan", "none", "<na>"}:
            return value
    return ""


def prepare_csv_import(rows: list[dict], resolver: Callable, existing: list[dict]) -> list[dict]:
    """Resolve locally, preserve every row's error, and never query public identities implicitly."""
    keys = _existing_structure_keys(existing, resolver)
    prepared = []
    for row in rows:
        kind = row.get("identity_type", "").strip().casefold() or "name"
        item = {"csv_row": row["csv_row"], "pfas": row.get("pfas", ""), "entry": None,
                "identity_type": kind,
                "concentration": row.get("concentration", ""), "unit": row.get("unit", ""),
                "error": "", "structure": ""}
        try:
            identity = row.get("pfas", "").strip()
            if not identity:
                raise MixtureInputError("Enter a PFAS identifier.")
            unit = normalize_unit(row.get("unit"))
            mg_l = to_mg_l(row.get("concentration"), unit)
            amount = float(Decimal(row.get("concentration", "")))
            if kind not in {"name", "cas", "abbreviation", "smiles"}:
                raise MixtureInputError("identity_type must be name, cas, abbreviation, or smiles.")
            parent_smiles = row.get("parent_smiles", "").strip()
            legacy_smiles = row.get("smiles", "").strip()
            if parent_smiles and legacy_smiles and prepare_structure(parent_smiles)[1] != prepare_structure(legacy_smiles)[1]:
                raise MixtureInputError("parent_smiles and the legacy smiles column describe different structures. Keep one intended parent structure.")
            supplied_smiles = parent_smiles or legacy_smiles
            if kind == "smiles" and supplied_smiles and prepare_structure(identity)[1] != prepare_structure(supplied_smiles)[1]:
                raise MixtureInputError("When identity_type is smiles, put the intended structure in pfas and leave parent_smiles empty; the supplied structures disagree.")
            lookup = resolver({"identity": supplied_smiles or identity, "identity_type": "smiles" if supplied_smiles else kind})
            if not lookup.ready:
                raise MixtureInputError(" ".join(lookup.warnings) or "Resolve this identity in the manual editor, or supply its parent SMILES in the CSV.")
            _, key = prepare_structure(lookup.canonical_smiles)
            item["structure"] = lookup.canonical_smiles
            if key in keys:
                raise MixtureInputError("This compound duplicates an existing mixture entry or an earlier CSV row. Keep one concentration per PFAS.")
            keys.add(key)
            feature_row = lookup.feature_row or {}
            entry = {"identity": identity, "identity_type": kind,
                "label": lookup.cache_key or row.get("label") or identity,
                "cached_choice": lookup.status == "cached", "structure_key": key,
                "canonical_smiles": lookup.canonical_smiles, "normalized_identity": key,
                "concentration_value": amount, "concentration_unit": unit, "initial_concentration_mg_l": mg_l,
                "abbreviation": _first_text(feature_row, "Abbreviation"),
                "full_name": _first_text(feature_row, "PFAS_name", "Compound Full Name"),
                "cas": _first_text(feature_row, "CAS Number", "CAS"),
                "import_source": {"kind": "csv", "row": row["csv_row"]}}
            if supplied_smiles or kind == "smiles":
                entry.update({"confirmed_smiles": supplied_smiles or identity,
                    "resolution": {"source": "csv_smiles", "query": identity, "identity_type": kind}})
            item.update({"entry": entry, "requires_confirmation": bool(supplied_smiles or kind == "smiles" or lookup.status == "generated"),
                         "feature_source": lookup.feature_source, "warnings": list(lookup.warnings)})
        except (MixtureInputError, InvalidOperation, ValueError, TypeError, OverflowError) as exc:
            item["error"] = str(exc) or "Enter a quantified positive numeric concentration."
        prepared.append(item)
    return prepared


def append_prepared_import(existing: list[dict], prepared: list[dict], next_id: int) -> tuple[list[dict], int]:
    """Apply the complete reviewed import or raise without modifying the mixture."""
    if not prepared or any(row.get("error") or not row.get("entry") for row in prepared):
        raise MixtureInputError("Correct every CSV row before importing. No rows have been added.")
    next_id = max(next_id, max((entry.get("id", 0) for entry in existing), default=0) + 1)
    keys = {entry.get("structure_key") for entry in existing if entry.get("structure_key")}
    additions = []
    for row in prepared:
        entry = dict(row["entry"])
        if entry["structure_key"] in keys:
            raise MixtureInputError("The mixture changed and now contains a duplicate CSV compound. Review the preview again.")
        keys.add(entry["structure_key"])
        entry.update({"id": next_id, "structure_confirmed": True})
        additions.append(entry)
        next_id += 1
    return [*existing, *additions], next_id
