"""Internal raw-record assembly stage for ``build_database.py``.

This module intentionally has no standalone command-line configuration.
Run ``build_database.py`` to execute the database pipeline.
"""

if __name__ == "__main__":
    raise SystemExit(
        "merge.py is an internal pipeline stage. Run build_database.py instead."
    )

import json
import math
import os
import re
from collections import OrderedDict
from typing import Any

import pandas as pd

from adsorbent_property_overrides import (
    load_elemental_average_overrides,
    load_expected_missing_adsorbent_property_studies,
    normalized_adsorbent_property_study_key,
)
from normalization_rules import (
    PREMERGE_NORMALIZED_ADSORBENT_FRACTION_COLUMNS,
    normalize_adsorbent_fraction_fields,
)


# ----------------------- Stable output schema -----------------------

TEMP_COL = "Temperature_(\u00b0C)"

META_SCHEMA = [
    "extraction_dataset",
    "study_no",
    "DOI",
    "authors",
    "article_title",
    "source_title",
    "publication_year",
]

ID_SCHEMA = ["performance_id", "adsorbent_id"]

ADS_NAME_SCHEMA = [
    "name_full",
    "name_commercial",
    "name_abbreviation",
]

PFAS_TEST_SCHEMA = [
    "PFAS_name",
    "Test_mode",
    "Differentiating_Condition",
    "Equilibrium_reached",
]

ISOTHERM_SCHEMA = [
    "Freundlich_equation_form",
    "Freundlich_KF_value",
    "Freundlich_KF_unit",
    "Freundlich_n_or_1/n_verbatim",
    "Freundlich_R2",
    "Langmuir_Qm_value",
    "Langmuir_Qm_unit",
    "Langmuir_KL_value",
    "Langmuir_KL_unit",
    "Langmuir_R2",
    "Kd_value",
    "Kd_unit",
    "Removal_rate",
    "Qe_value",
    "Qe_unit",
]

KINETIC_SCHEMA = [
    "PFO_k1_value",
    "PFO_k1_unit",
    "PFO_Qe_value",
    "PFO_Qe_unit",
    "PFO_R2",
    "PSO_k2_value",
    "PSO_k2_unit",
    "PSO_v0_value",
    "PSO_v0_unit",
    "PSO_Qe_value",
    "PSO_Qe_unit",
    "PSO_R2",
]

EXPERIMENT_SCHEMA = [
    "pH",
    TEMP_COL,
    "Contact_time_(h)",
    "Solution_volume_(mL)",
    "Mixing_speed_(rpm)",
    "Water_type",
    "TOC_(mg/L)",
    "DOC_(mg/L)",
    "TDS_(mg/L)",
    "PFAS_C0_value",
    "PFAS_C0_unit",
    "Adsorbent_dosage_value",
    "Adsorbent_dosage_unit",
    "Inorganic_matter",
    "Organic_matter",
    "Ionic_strength",
]

ADS_PROP_SCHEMA = [
    "adsorbent_category",
    "adsorbent_subcategory",
    "vendor",
    "AC_raw_material",
    "AC_activation_method",
    "polymer_matrix",
    "functional_group",
    "elemental_composition_method",
    "pore_class",
    "average_pore_diameter_(angstrom)",
    "ssa_(m2/g)",
    "phpzc",
]

ADS_DERIVED_SCHEMA = [
    "porosity_total_value",
    "porosity_micro_value",
    "porosity_meso_value",
    "porosity_macro_value",
    "pore_volume_total_value",
    "pore_volume_micro_value",
    "pore_volume_meso_value",
    "pore_volume_macro_value",
    "pore_volume_unit",
    "element_C_value",
    "element_N_value",
    "element_O_value",
    "zeta_potential_value",
    "zeta_potential_pH",
    "Nominal_grain_size_value",
    "Nominal_grain_size_unit",
    "ion_exchange_capacity_value",
    "ion_exchange_capacity_unit",
    "iodine_number_mg/g",
]

# Retain the reported property lists and the one-time normalization marker when
# a clean adsorbent property is copied to performance rows. The marker is
# consumed and removed by the performance-stage normalizer.
ADS_PREMERGE_NORMALIZATION_COLUMNS = [
    "element_C_value_raw",
    "element_N_value_raw",
    "element_O_value_raw",
    "porosity_total_value_raw",
    "porosity_micro_value_raw",
    "porosity_meso_value_raw",
    "porosity_macro_value_raw",
    "porosity_micro_value_avg",
    "porosity_meso_value_avg",
    "porosity_macro_value_avg",
    PREMERGE_NORMALIZED_ADSORBENT_FRACTION_COLUMNS,
    "normalization_trace",
]

REVIEW_SCHEMA = ["data_provenance", "require_review", "review_reason"]
PROV_SCHEMA = ["performance_source", "adsorbent_source"]
NOTE_SCHEMA = ["warning_message"]

INTERNAL_ONLY_COLUMNS = {
    "PFAS_C0",
    "Adsorbent_dosage",
    "Nominal_grain_size",
    "particle_size",
    "pore_percentage",
    "pore_volume_(cm3/g)",
    "zeta_potential_(mv)",
    "ion_exchange_capacity",
}

PERFORMANCE_SCHEMA = (
    META_SCHEMA
    + ID_SCHEMA
    + ADS_NAME_SCHEMA
    + PFAS_TEST_SCHEMA
    + ISOTHERM_SCHEMA
    + KINETIC_SCHEMA
    + EXPERIMENT_SCHEMA
    + ADS_PROP_SCHEMA
    + ADS_DERIVED_SCHEMA
    + REVIEW_SCHEMA
    + PROV_SCHEMA
    + NOTE_SCHEMA
)

ADSORBENT_SCHEMA = (
    META_SCHEMA
    + ["adsorbent_id"]
    + ADS_NAME_SCHEMA
    + ADS_PROP_SCHEMA
    + ADS_DERIVED_SCHEMA
    + REVIEW_SCHEMA
    + ["adsorbent_source"]
    + NOTE_SCHEMA
)


# ----------------------- Canonicalization -----------------------

DIRECT_FIELD_MAP = {
    # IDs and task fields
    "performance_id": "performance_id",
    "adsorbent_id": "adsorbent_id",
    "pfas_name": "PFAS_name",
    "test_mode": "Test_mode",
    "differentiating_condition": "Differentiating_Condition",
    # Embedded experiment fields
    "pfas_c0": "PFAS_C0",
    "pfas_initial_concentration": "PFAS_C0",
    # The extraction schema owns these two names. Accept the previous
    # input aliases at the merge boundary, but never emit them in a raw
    # merge file.
    "inorganic_matter": "Inorganic_matter",
    "added_inorganic_matter": "Inorganic_matter",
    "organic_matter": "Organic_matter",
    "added_organic_matter": "Organic_matter",
    "ionic_strength": "Ionic_strength",
    # Adsorbent fields
    "particle_size": "Nominal_grain_size",
    "nominal_grain_size": "Nominal_grain_size",
    "ac_raw_material": "AC_raw_material",
    "ac_activation_method": "AC_activation_method",
    "polymer_matrix": "polymer_matrix",
    "c%": "element_C_value",
    "n%": "element_N_value",
    "o%": "element_O_value",
    "pore_percentage": "pore_percentage",
    "total_pore_percentage": "porosity_total_value",
    "micro_pore_percentage": "porosity_micro_value",
    "meso_pore_percentage": "porosity_meso_value",
    "macro_pore_percentage": "porosity_macro_value",
    "total_pore_volume_(cm3/g)": "pore_volume_total_value",
    "micro_pore_volume_(cm3/g)": "pore_volume_micro_value",
    "meso_pore_volume_(cm3/g)": "pore_volume_meso_value",
    "macro_pore_volume_(cm3/g)": "pore_volume_macro_value",
    # Review fields
    "data_provenance": "data_provenance",
    "require_review": "require_review",
    "review_reason": "review_reason",
    # Non-blocking post-processing provenance retained in the final trace.
    "normalization_trace": "normalization_trace",
}

CANONICAL_FIELDS = (
    PERFORMANCE_SCHEMA
    + ADSORBENT_SCHEMA
    + list(INTERNAL_ONLY_COLUMNS)
    + [
        "Water_sample_source",
        "Water_sample_location",
        "Freundlich_R2",
        "Langmuir_R2",
    ]
)


def _norm_key(key: Any) -> str:
    text = str(key or "").strip()
    text = text.replace("\u03bc", "\u00b5")
    text = text.replace("\u00c2\u00b5", "\u00b5")
    text = text.replace("\u00b5", "u")
    text = text.replace("\u00c2\u00b0", "deg")
    text = text.replace("\u00b0", "deg")
    text = text.replace("\u2013", "-").replace("\u2014", "-")
    text = re.sub(r"\s+", " ", text)
    return text.lower()


CANONICAL_BY_NORM = {_norm_key(name): name for name in CANONICAL_FIELDS}
DIRECT_FIELD_MAP_NORM = {_norm_key(key): value for key, value in DIRECT_FIELD_MAP.items()}

# These fields were retired when the extraction contract became explicitly
# activated-carbon scoped.  Do not alias them: stale artifacts must not
# reintroduce the old database columns.
RETIRED_EXTRACTION_FIELDS = {
    "raw_material": "AC_raw_material",
    "activation_method": "AC_activation_method",
}


def _canonical_field_name(key: Any) -> str:
    norm = _norm_key(key)
    if norm in DIRECT_FIELD_MAP_NORM:
        return DIRECT_FIELD_MAP_NORM[norm]
    if norm in CANONICAL_BY_NORM:
        return CANONICAL_BY_NORM[norm]
    return str(key or "").strip()


def _is_extraction_missing_placeholder(value: Any) -> bool:
    if not isinstance(value, str):
        return False
    text = value.strip().lower()
    text = text.removeprefix("[").removesuffix("]").strip()
    return text in {
        "value not provided in chunk",
        "value not available in chunk",
        "not provided in chunk",
    }


def _is_empty(value: Any) -> bool:
    if value is None:
        return True
    try:
        if isinstance(value, float) and math.isnan(value):
            return True
        if pd.isna(value):
            return True
    except Exception:
        pass
    if isinstance(value, str) and value.strip().lower() in {
        "",
        "na",
        "n/a",
        "nan",
        "null",
        "none",
        "not found",
        "not reported",
    }:
        return True
    if _is_extraction_missing_placeholder(value):
        return True
    return False


def _clean_value(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return ""
    return value


def _coerce_note_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item) for item in value if str(item).strip()]
    if isinstance(value, str):
        return [part.strip() for part in re.split(r";|\n", value) if part.strip()]
    return [str(value)]


def _merge_notes(*parts: Any) -> str:
    seen = OrderedDict()
    for part in parts:
        for item in _coerce_note_list(part):
            if item:
                seen.setdefault(item, None)
    return "; ".join(seen.keys())


def _set_first_nonempty(row: dict[str, Any], key: str, value: Any) -> None:
    if _is_empty(row.get(key)) and not _is_empty(value):
        row[key] = _clean_value(value)
    elif key not in row:
        row[key] = ""


def _canonicalize_record(record: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    warnings: list[str] = []
    errors: list[str] = []

    def consume_items(items):
        for raw_key, value in items:
            if raw_key in ("_warnings", "warnings"):
                warnings.extend(_coerce_note_list(value))
                continue
            if raw_key in ("_errors", "errors"):
                errors.extend(_coerce_note_list(value))
                continue
            if str(raw_key).startswith("_"):
                continue
            replacement = RETIRED_EXTRACTION_FIELDS.get(_norm_key(raw_key))
            if replacement:
                errors.append(
                    f"Unsupported retired field '{raw_key}'; use '{replacement}'"
                )
                continue
            key = _canonical_field_name(raw_key)
            if not key:
                continue
            if key == "error_message":
                errors.extend(_coerce_note_list(value))
                continue

            if key in out and not _is_empty(out.get(key)) and not _is_empty(value):
                if str(out[key]).strip() != str(value).strip():
                    warnings.append(f"Conflicting repeated field '{key}' kept first value")
                continue
            _set_first_nonempty(out, key, value)

    consume_items(record.items())
    extras = record.get("__extras__")
    if isinstance(extras, dict):
        consume_items(extras.items())

    out["warning_message"] = _merge_notes(out.get("warning_message"), warnings)
    if errors:
        out["require_review"] = "Yes"
        out["review_reason"] = _merge_notes(
            out.get("review_reason"),
            [f"Ingest error: {message}" for message in errors],
        )
    return out


# ----------------------- Artifact loading -----------------------

def _artifact_root_for_dataset(dataset: dict[str, Any]) -> str:
    if dataset.get("artifact_root"):
        return str(dataset["artifact_root"])
    return os.path.join(str(dataset["extraction_output_dir"]), "_chain_artifacts")


def _sort_study_id(study_id: str) -> tuple[int, str]:
    match = re.search(r"(\d+)", study_id)
    if not match:
        return (10**9, study_id)
    return (int(match.group(1)), study_id)


def _discover_study_ids(artifact_root: str) -> list[str]:
    if not os.path.isdir(artifact_root):
        return []
    ids = [
        name
        for name in os.listdir(artifact_root)
        if os.path.isdir(os.path.join(artifact_root, name))
        and name.lower().startswith("study_")
    ]
    return sorted(ids, key=_sort_study_id)


def _selected_study_ids(
    artifact_root: str,
    selected_study_ids: list[str],
) -> list[str]:
    if selected_study_ids:
        return list(selected_study_ids)
    return _discover_study_ids(artifact_root)


def _artifact_path(artifact_root: str, study_id: str, chain_name: str) -> str:
    return os.path.join(artifact_root, study_id, f"{study_id}_{chain_name}.json")


def _load_artifact_payload(artifact_root: str, study_id: str, chain_name: str) -> tuple[dict, str]:
    path = _artifact_path(artifact_root, study_id, chain_name)
    if not os.path.exists(path):
        return {}, path
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh), path


def _format_source(file_type: Any, chunk_id: Any) -> str:
    ft = str(file_type or "").strip()
    if not ft:
        ft = "unknown"
    if chunk_id is None or str(chunk_id).strip() == "":
        return ft
    return f"{ft}_{chunk_id}"


def _iter_artifact_records(
    *,
    dataset_label: str,
    artifact_root: str,
    study_id: str,
    chain_name: str,
    task: str,
    source_col: str,
) -> list[dict[str, Any]]:
    payload, path = _load_artifact_payload(artifact_root, study_id, chain_name)
    if not payload:
        print(f"[{dataset_label}][{study_id}] missing {chain_name} artifact: {path}")
        return []

    status = str(payload.get("status") or "").strip().lower()
    outputs = payload.get("outputs") if isinstance(payload.get("outputs"), list) else []
    if status in {"failed", "skipped"} and not outputs:
        print(f"[{dataset_label}][{study_id}] {chain_name} artifact status={status}; no records")
        return []

    rows: list[dict[str, Any]] = []
    for output in outputs:
        if not isinstance(output, dict):
            continue
        extracted = output.get("extracted_data") or {}
        task_payload = extracted.get(task) or {}
        if isinstance(task_payload, dict):
            records = task_payload.get("records") or []
        elif isinstance(task_payload, list):
            records = task_payload
        else:
            records = []
        for record in records:
            if not isinstance(record, dict):
                continue
            row = _canonicalize_record(record)
            row["extraction_dataset"] = dataset_label
            row["study_no"] = output.get("study_folder") or study_id
            row[source_col] = _format_source(output.get("file_type"), output.get("chunk_id"))
            row["warning_message"] = _merge_notes(
                row.get("warning_message"),
                f"{chain_name} artifact status={status}" if status == "ran_with_errors" else "",
            )
            rows.append(row)
    return rows


# ----------------------- Study metadata -----------------------

def _study_num(value: Any) -> str:
    match = re.search(r"(\d+)", str(value or ""))
    if not match:
        return ""
    num = match.group(1).lstrip("0")
    return num or "0"


def _load_study_metadata(
    path: str,
    sheet_name: str,
) -> dict[str, dict[str, Any]]:
    if not os.path.exists(path):
        print(f"Warning: study metadata workbook not found: {path}")
        return {}

    try:
        df = pd.read_excel(path, sheet_name=sheet_name)
    except ValueError:
        df = pd.read_excel(path)

    by_num: dict[str, dict[str, Any]] = {}
    for _, row in df.iterrows():
        num = _study_num(row.get("study_no"))
        if not num:
            continue
        by_num[num] = {
            "DOI": row.get("DOI Link") or "",
            "authors": row.get("Authors") or "",
            "article_title": row.get("Article Title") or "",
            "source_title": row.get("Source Title") or "",
            "publication_year": row.get("Publication Year") or "",
        }
    return by_num


def _add_study_metadata(row: dict[str, Any], study_metadata: dict[str, dict[str, Any]]) -> None:
    meta = study_metadata.get(_study_num(row.get("study_no")), {})
    for key in ("DOI", "authors", "article_title", "source_title", "publication_year"):
        if key not in row or _is_empty(row.get(key)):
            row[key] = _clean_value(meta.get(key, ""))


# ----------------------- Value parsing -----------------------

NON_PFAS_EXACT = {
    "NOM",
    "SRNOM",
    "GENERAL",
    "NA",
    "N/A",
    "NAN",
    "NULL",
    "NONE",
    "-",
    "--",
}

NON_PFAS_REGEX = [
    r"^PFCA\s*\(\s*C?6\s*[-\u2013\u2014]\s*C?11\s*\)$",
]


def _is_non_pfas(name: Any) -> bool:
    if _is_empty(name):
        return True
    text = str(name).strip()
    if text.upper() in NON_PFAS_EXACT:
        return True
    return any(re.fullmatch(pattern, text, flags=re.IGNORECASE) for pattern in NON_PFAS_REGEX)


def _norm_id(value: Any) -> str:
    return str(value or "").strip().casefold()


def _norm_adsorbent_alias(value: Any) -> str:
    """Return a punctuation-insensitive key for study-local adsorbent aliases."""
    text = _norm_id(value)
    text = text.replace("×", "x")
    return re.sub(r"[^a-z0-9]+", "", text)


def _split_series_items(raw: Any) -> list[str]:
    if _is_empty(raw):
        return []
    return [
        part.strip()
        for part in re.split(r"\s*&&\s*|\s*;\s*", str(raw))
        if part.strip()
    ]


def _leading_token(raw: Any) -> str:
    return str(raw or "").strip().split("_", 1)[0].strip()


def _pick_series_item(raw: Any, name_hint: Any) -> str:
    items = _split_series_items(raw)
    if not items:
        return ""
    hint = _norm_id(name_hint)
    if hint:
        for item in items:
            if _norm_id(_leading_token(item)) == hint:
                return item
    return items[0]


_VALUE_UNIT_NUMBER = (
    r"[<>]?[+-]?(?:\d+(?:\.\d*)?|\.\d+)"
    r"(?:[eE][+-]?\d+)?"
)
_AND_JOINED_VALUE_UNIT_RE = re.compile(
    rf"^\s*(?:[^_]+_)?"
    rf"(?P<values>{_VALUE_UNIT_NUMBER}(?:\s*_?\s*and\s*_?\s*{_VALUE_UNIT_NUMBER})+)"
    rf"\s*[_ ]\s*(?P<unit>[A-Za-z0-9\u00b5\u03bc/().\-]+)\s*$",
    flags=re.IGNORECASE,
)


def _split_value_unit_token(token: Any) -> tuple[str, str]:
    text = str(token or "").strip()
    if not text:
        return "", ""

    def _looks_like_measurement(value: str) -> bool:
        return bool(re.fullmatch(r"\s*[<>\u2264\u2265~]?[0-9.+eE,\-\s]+\s*", value or ""))

    # An extraction can report several values for one analyte in the form
    # ``PFNA_8.5e4_and_1.7e5_ng/L``. Preserve every value with the shared
    # unit instead of treating ``and_1.7e5`` as part of the unit string.
    and_joined = _AND_JOINED_VALUE_UNIT_RE.match(text)
    if and_joined:
        values = re.findall(_VALUE_UNIT_NUMBER, and_joined.group("values"))
        if values:
            return ",".join(values), and_joined.group("unit").replace("\u03bc", "\u00b5")

    parts = text.split("_", 2)
    if len(parts) == 3 and _looks_like_measurement(parts[1]) and not _is_empty(parts[2]):
        return parts[1].strip(), parts[2].strip().replace("\u03bc", "\u00b5")

    two_parts = text.split("_", 1)
    if len(two_parts) == 2 and _looks_like_measurement(two_parts[0]) and not _is_empty(two_parts[1]):
        return two_parts[0].strip(), two_parts[1].strip().replace("\u03bc", "\u00b5")

    match = re.match(r"^\s*([<>]?[0-9.+eE,\-]+)\s*[_ ]\s*([A-Za-z0-9\u00b5\u03bc/().\-]+)\s*$", text)
    if match:
        return match.group(1).strip(), match.group(2).strip().replace("\u03bc", "\u00b5")
    return "", ""


def _add_split_value_unit_columns(row: dict[str, Any]) -> None:
    pfas_raw = row.get("PFAS_C0")
    if not _is_empty(pfas_raw):
        picked = _pick_series_item(pfas_raw, row.get("PFAS_name"))
        value, unit = _split_value_unit_token(picked)
        if value:
            row["PFAS_C0_value"] = value
        if unit:
            row["PFAS_C0_unit"] = unit

    dosage_raw = row.get("Adsorbent_dosage")
    if not _is_empty(dosage_raw):
        picked = _pick_series_item(dosage_raw, row.get("adsorbent_id"))
        value, unit = _split_value_unit_token(picked)
        if value:
            row["Adsorbent_dosage_value"] = value
        if unit:
            row["Adsorbent_dosage_unit"] = unit


def _to_number(value: Any) -> float | None:
    if _is_empty(value):
        return None
    text = str(value).strip()
    if "," in text or ";" in text or "&&" in text:
        return None
    text = re.sub(r"^[<>~\u2264\u2265]+", "", text)
    try:
        return float(text)
    except ValueError:
        return None


def _percent_to_fraction(value: Any) -> str:
    if _is_empty(value):
        return ""
    text = str(value).strip()
    num = _to_number(text.replace("%", ""))
    if num is None:
        return text
    if "%" in text or num > 1:
        num = num / 100.0
    return f"{num:.6g}"


def _parse_descriptor_values(raw: Any, default_desc: str = "total") -> dict[str, str]:
    if _is_empty(raw):
        return {}
    out: dict[str, str] = {}
    for item in _split_series_items(raw):
        if "_" in item:
            desc, value = item.split("_", 1)
        else:
            desc, value = default_desc, item
        desc = desc.strip().lower()
        value = value.strip()
        if desc and value:
            out[desc] = value
    return out


def _canonical_pore_desc(desc: str) -> str:
    mapping = {
        "total": "total",
        "micro": "micro",
        "micropore": "micro",
        "micropores": "micro",
        "meso": "meso",
        "mesopore": "meso",
        "mesopores": "meso",
        "macro": "macro",
        "macropore": "macro",
        "macropores": "macro",
    }
    return mapping.get(desc.strip().lower(), "")


def _parse_particle_size(raw: Any) -> tuple[str, str]:
    values: list[str] = []
    units: list[str] = []
    for item in _split_series_items(raw):
        match = re.match(r"^\s*(.+?)\s*_\s*([A-Za-z0-9\u00b5\u03bc/().\-]+)\s*$", item)
        if not match:
            continue
        values.append(match.group(1).strip())
        units.append(match.group(2).strip().replace("\u03bc", "\u00b5"))
    return "&&".join(values), "&&".join(units)


_CAPACITY_VALUE_PATTERN = (
    r"[<>]?[+-]?(?:\d+(?:\.\d*)?|\.\d+)"
    r"(?:[eE][+-]?\d+)?"
)
_CAPACITY_PAIR_RE = re.compile(
    rf"(?P<value>{_CAPACITY_VALUE_PATTERN})\s*[_ ]\s*"
    r"(?P<unit>[A-Za-z0-9\u00b5\u03bc/().\-]+)"
)


def _parse_capacity(raw: Any) -> tuple[str, str]:
    """Parse one or more IEC value/unit pairs while preserving pair alignment.

    A material may report solution-basis and mass-basis capacities together,
    for example ``0.6_eq/L, 124.58_meq/100g``.  The normalization stage needs
    both values so it can populate its separate meq/L and meq/g outputs.  We
    serialize aligned pairs with ``&&``, which is already the merged-row list
    convention used by other adsorbent properties.

    Commas delimit complete value/unit pairs. Scalar values use plain digits
    without thousands separators.
    """
    if _is_empty(raw):
        return "", ""
    text = str(raw).strip()
    matches = list(_CAPACITY_PAIR_RE.finditer(text))
    if not matches:
        return "", ""

    # Accept only complete value/unit pairs separated by the established list
    # delimiters.  This avoids silently extracting a valid fragment from a
    # malformed capacity expression.
    if text[:matches[0].start()].strip():
        return "", ""
    for previous, current in zip(matches, matches[1:]):
        delimiter = text[previous.end():current.start()]
        if not re.fullmatch(r"\s*(?:&&|[;,])\s*", delimiter):
            return "", ""
    if text[matches[-1].end():].strip():
        return "", ""

    values = [match.group("value").strip() for match in matches]
    units = [
        match.group("unit").strip().replace("\u03bc", "\u00b5")
        for match in matches
    ]
    return "&&".join(values), "&&".join(units)


def _add_adsorbent_derived_fields(row: dict[str, Any]) -> None:
    for raw_col, out_col in (
        ("element_C_value", "element_C_value"),
        ("element_N_value", "element_N_value"),
        ("element_O_value", "element_O_value"),
    ):
        if not _is_empty(row.get(raw_col)):
            row[out_col] = _percent_to_fraction(row.get(raw_col))

    for desc, value in _parse_descriptor_values(row.get("pore_percentage")).items():
        canon = _canonical_pore_desc(desc)
        if canon:
            row[f"porosity_{canon}_value"] = _percent_to_fraction(value)

    for col in (
        "porosity_total_value",
        "porosity_micro_value",
        "porosity_meso_value",
        "porosity_macro_value",
    ):
        if not _is_empty(row.get(col)):
            row[col] = _percent_to_fraction(row.get(col))

    pore_volume = _parse_descriptor_values(row.get("pore_volume_(cm3/g)"))
    for desc, value in pore_volume.items():
        canon = _canonical_pore_desc(desc)
        if canon:
            row[f"pore_volume_{canon}_value"] = value

    has_pore_volume = pore_volume or any(
        not _is_empty(row.get(col))
        for col in (
            "pore_volume_total_value",
            "pore_volume_micro_value",
            "pore_volume_meso_value",
            "pore_volume_macro_value",
        )
    )
    if has_pore_volume:
        row["pore_volume_unit"] = "cm3/g"

    grain_size_raw = row.get("Nominal_grain_size")
    if _is_empty(grain_size_raw):
        grain_size_raw = row.get("particle_size")
    grain_size_value, grain_size_unit = _parse_particle_size(grain_size_raw)
    if grain_size_value:
        row["Nominal_grain_size_value"] = grain_size_value
        row["Nominal_grain_size_unit"] = grain_size_unit

    zeta_raw = row.get("zeta_potential_(mv)")
    if not _is_empty(zeta_raw):
        zeta_values: list[str] = []
        zeta_phs: list[str] = []
        for item in _split_series_items(zeta_raw):
            match = re.match(r"^\s*([+-]?[0-9.+eE\-]+)\s*@\s*pH[_ ]?([0-9.+eE\-]+)\s*$", item, flags=re.I)
            if match:
                zeta_values.append(match.group(1))
                zeta_phs.append(match.group(2))
                continue
            value, _unit = _parse_capacity(item)
            if value:
                zeta_values.append(value)
        if zeta_values:
            row["zeta_potential_value"] = "&&".join(zeta_values)
        if zeta_phs:
            row["zeta_potential_pH"] = "&&".join(zeta_phs)

    capacity_value, capacity_unit = _parse_capacity(row.get("ion_exchange_capacity"))
    if capacity_value:
        row["ion_exchange_capacity_value"] = capacity_value
        row["ion_exchange_capacity_unit"] = capacity_unit


# ----------------------- Row de-duplication and merge -----------------------

def _combine_sources(*sources: Any) -> str:
    seen = OrderedDict()
    for source in sources:
        for part in _coerce_note_list(source):
            if part:
                seen.setdefault(part, None)
    return "; ".join(seen.keys())


def _dedupe_rows(
    rows: list[dict[str, Any]],
    *,
    id_col: str,
    source_col: str,
    duplicate_warning: str,
) -> list[dict[str, Any]]:
    deduped: OrderedDict[tuple[str, str, str], dict[str, Any]] = OrderedDict()
    no_id_rows: list[dict[str, Any]] = []

    for row in rows:
        ident = _norm_id(row.get(id_col))
        if not ident:
            row["warning_message"] = _merge_notes(row.get("warning_message"), f"Missing {id_col}")
            no_id_rows.append(row)
            continue

        key = (_norm_id(row.get("extraction_dataset")), _norm_id(row.get("study_no")), ident)
        if key not in deduped:
            deduped[key] = row
            continue

        base = deduped[key]
        for col, value in row.items():
            if col == source_col:
                base[source_col] = _combine_sources(base.get(source_col), value)
            elif col == "warning_message":
                base[col] = _merge_notes(base.get(col), value)
            else:
                _set_first_nonempty(base, col, value)
        base["warning_message"] = _merge_notes(base.get("warning_message"), duplicate_warning)

    return list(deduped.values()) + no_id_rows


def _adsorbent_index(rows: list[dict[str, Any]]) -> dict[tuple[str, str, str], dict[str, Any]]:
    index: dict[tuple[str, str, str], dict[str, Any]] = {}
    for row in rows:
        ads_id = _norm_id(row.get("adsorbent_id"))
        if not ads_id:
            continue
        key = (_norm_id(row.get("extraction_dataset")), _norm_id(row.get("study_no")), ads_id)
        index[key] = row
    return index


ADSORBENT_IDENTITY_FIELDS = (
    "adsorbent_id",
    "name_abbreviation",
    "name_commercial",
    "name_full",
)

# The source table labels this reactivated material ``F400-R`` while the
# study-level property extraction identifies it as ``R-F400``. This is a
# study-specific ordering difference, so keep it explicit rather than adding
# an unsafe global rule that could collapse genuinely distinct variants.
STUDY_SCOPED_ADSORBENT_ALIASES = {
    ("training", "study_45", "f400r"): "rf400",
}


def _study_adsorbent_key(row: dict[str, Any]) -> tuple[str, str]:
    return (
        _norm_id(row.get("extraction_dataset")),
        _norm_id(row.get("study_no")),
    )


def _append_unique_candidate(
    index: dict[tuple[str, str, str], list[dict[str, Any]]],
    key: tuple[str, str, str],
    row: dict[str, Any],
) -> None:
    candidates = index.setdefault(key, [])
    if not any(candidate is row for candidate in candidates):
        candidates.append(row)


def _adsorbent_alias_index(
    rows: list[dict[str, Any]],
) -> dict[tuple[str, str, str], list[dict[str, Any]]]:
    """Index study-local IDs plus structured name aliases.

    The alias index intentionally retains every candidate. An alias is used
    only when it resolves to exactly one property record, avoiding guesses
    across particle-size or other material variants.
    """
    index: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for row in rows:
        dataset, study = _study_adsorbent_key(row)
        for field in ADSORBENT_IDENTITY_FIELDS:
            alias = _norm_adsorbent_alias(row.get(field))
            if alias:
                _append_unique_candidate(index, (dataset, study, alias), row)
    return index


def _vendor_alias(value: Any) -> str:
    """Use only the leading vendor token for safe manufacturer aliases."""
    first_token = re.split(r"[\s,(/]+", str(value or "").strip(), maxsplit=1)[0]
    return _norm_adsorbent_alias(first_token)


def _adsorbent_vendor_index(
    rows: list[dict[str, Any]],
) -> dict[tuple[str, str, str], list[dict[str, Any]]]:
    index: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for row in rows:
        vendor = _vendor_alias(row.get("vendor"))
        if vendor:
            dataset, study = _study_adsorbent_key(row)
            _append_unique_candidate(index, (dataset, study, vendor), row)
    return index


def _only_candidate(
    candidates: list[dict[str, Any]] | None,
) -> dict[str, Any] | None:
    return candidates[0] if candidates and len(candidates) == 1 else None


def _resolve_adsorbent_record(
    perf: dict[str, Any],
    ads_index: dict[tuple[str, str, str], dict[str, Any]],
    alias_index: dict[tuple[str, str, str], list[dict[str, Any]]],
    vendor_index: dict[tuple[str, str, str], list[dict[str, Any]]],
) -> dict[str, Any] | None:
    """Resolve a performance adsorbent to one unambiguous property record."""
    dataset, study = _study_adsorbent_key(perf)
    ads_id = _norm_id(perf.get("adsorbent_id"))
    direct = ads_index.get((dataset, study, ads_id))
    if direct:
        return direct

    alias = _norm_adsorbent_alias(perf.get("adsorbent_id"))
    if not alias:
        return None

    explicit_alias = STUDY_SCOPED_ADSORBENT_ALIASES.get((dataset, study, alias))
    if explicit_alias:
        resolved = _only_candidate(alias_index.get((dataset, study, explicit_alias)))
        if resolved:
            return resolved

    resolved = _only_candidate(alias_index.get((dataset, study, alias)))
    if resolved:
        return resolved

    # A vendor label is a valid fallback only when that manufacturer identifies
    # one adsorbent record in the current study. This repairs cases such as
    # ``Jacobi`` without assigning every material sold by a multi-product vendor.
    return _only_candidate(vendor_index.get((dataset, study, alias)))


def _copy_adsorbent_fields(perf: dict[str, Any], ads: dict[str, Any]) -> None:
    perf["adsorbent_source"] = ads.get("adsorbent_source", "")
    for col in (
        ADS_NAME_SCHEMA
        + ADS_PROP_SCHEMA
        + ADS_DERIVED_SCHEMA
        + ADS_PREMERGE_NORMALIZATION_COLUMNS
    ):
        if col in ads:
            if col == "normalization_trace":
                perf[col] = _merge_notes(perf.get(col), ads.get(col))
            else:
                _set_first_nonempty(perf, col, ads.get(col))


def _build_performance_rows(
    performance_rows: list[dict[str, Any]],
    ads_rows: list[dict[str, Any]],
    study_metadata: dict[str, dict[str, Any]],
    *,
    expected_missing_property_study_keys: set[tuple[str, str]] | None = None,
) -> list[dict[str, Any]]:
    ads_index = _adsorbent_index(ads_rows)
    alias_index = _adsorbent_alias_index(ads_rows)
    vendor_index = _adsorbent_vendor_index(ads_rows)
    expected_missing_property_study_keys = expected_missing_property_study_keys or set()
    out: list[dict[str, Any]] = []

    for perf in performance_rows:
        _add_study_metadata(perf, study_metadata)
        _add_split_value_unit_columns(perf)

        ads_id = _norm_id(perf.get("adsorbent_id"))
        if not ads_id:
            perf["warning_message"] = _merge_notes(perf.get("warning_message"), "Performance row lacks adsorbent_id")
        else:
            ads = _resolve_adsorbent_record(
                perf,
                ads_index,
                alias_index,
                vendor_index,
            )
            if ads:
                _copy_adsorbent_fields(perf, ads)
            elif normalized_adsorbent_property_study_key(
                perf.get("extraction_dataset"), perf.get("study_no")
            ) in expected_missing_property_study_keys:
                perf["normalization_trace"] = _merge_notes(
                    perf.get("normalization_trace"),
                    "Adsorbent properties intentionally absent (reviewed study decision)",
                )
            else:
                perf["warning_message"] = _merge_notes(
                    perf.get("warning_message"),
                    f"No matching adsorbent property record for adsorbent_id={perf.get('adsorbent_id')}",
                )

        if "PFAS_name" in perf and _is_non_pfas(perf.get("PFAS_name")):
            continue
        out.append(perf)
    return out


def _build_adsorbent_rows(
    ads_rows: list[dict[str, Any]],
    study_metadata: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in ads_rows:
        _add_study_metadata(row, study_metadata)
        _add_adsorbent_derived_fields(row)
        out.append(row)
    return out


def _normalize_adsorbent_property_rows(
    ads_rows: list[dict[str, Any]],
    adsorbent_property_review_decisions_file: str | None,
) -> list[dict[str, Any]]:
    """Normalize copied adsorbent fractions once, before performance merging."""
    if not ads_rows:
        return []
    ads_df = pd.DataFrame(ads_rows)
    approved_elemental_average_keys = load_elemental_average_overrides(
        adsorbent_property_review_decisions_file
    )
    normalize_adsorbent_fraction_fields(
        ads_df,
        approved_elemental_average_keys=approved_elemental_average_keys,
        mark_premerge_normalization=True,
    )
    return ads_df.to_dict(orient="records")


# ----------------------- Excel output -----------------------

def _ordered_dataframe(rows: list[dict[str, Any]], schema: list[str]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    if df.empty:
        df = pd.DataFrame(columns=schema)

    df = df.drop(columns=[col for col in INTERNAL_ONLY_COLUMNS if col in df.columns])


    for col in schema:
        if col not in df.columns:
            df[col] = ""

    extra_cols = [col for col in df.columns if col not in schema]
    ordered = schema + sorted(extra_cols, key=lambda c: str(c).lower())
    return df[ordered].copy()


def _write_workbook(path: str, df: pd.DataFrame) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with pd.ExcelWriter(path, engine="xlsxwriter") as writer:
        df.to_excel(writer, sheet_name="Sheet1", index=False)
        worksheet = writer.sheets["Sheet1"]
        columns = list(df.columns)
        if "DOI" in columns:
            doi_col = columns.index("DOI")
            for row_idx, url in enumerate(df["DOI"], start=1):
                if isinstance(url, str) and url.startswith("http"):
                    worksheet.write_url(row_idx, doi_col, url, string=url)


# ----------------------- Pipeline API -----------------------

def collect_rows(
    datasets: list[dict[str, Any]],
    selected_study_ids: list[str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    performance_rows: list[dict[str, Any]] = []
    adsorbent_rows: list[dict[str, Any]] = []

    for dataset in datasets:
        label = str(dataset["label"])
        artifact_root = _artifact_root_for_dataset(dataset)
        study_ids = _selected_study_ids(artifact_root, selected_study_ids)
        if not study_ids:
            print(f"[{label}] no study artifact folders found under {artifact_root}")
            continue

        for study_id in study_ids:
            ads = _iter_artifact_records(
                dataset_label=label,
                artifact_root=artifact_root,
                study_id=study_id,
                chain_name="adsorbent_study",
                task="adsorbent",
                source_col="adsorbent_source",
            )
            perf = _iter_artifact_records(
                dataset_label=label,
                artifact_root=artifact_root,
                study_id=study_id,
                chain_name="performance",
                task="performance",
                source_col="performance_source",
            )
            print(f"[{label}][{study_id}] adsorbent={len(ads)} performance={len(perf)}")
            adsorbent_rows.extend(ads)
            performance_rows.extend(perf)

    adsorbent_rows = _dedupe_rows(
        adsorbent_rows,
        id_col="adsorbent_id",
        source_col="adsorbent_source",
        duplicate_warning="Duplicate adsorbent_id within study; coalesced fields",
    )
    performance_rows = _dedupe_rows(
        performance_rows,
        id_col="performance_id",
        source_col="performance_source",
        duplicate_warning="Duplicate performance_id within study; coalesced fields",
    )
    return performance_rows, adsorbent_rows


def run(
    *,
    datasets: list[dict[str, Any]],
    selected_study_ids: list[str],
    study_metadata_xlsx: str,
    study_metadata_sheet: str,
    performance_output_file: str,
    adsorbent_output_file: str,
    adsorbent_normalized_output_file: str | None = None,
    adsorbent_property_review_decisions_file: str | None = None,
) -> None:
    """Assemble raw records and one normalized adsorbent-property workbook."""
    study_metadata = _load_study_metadata(
        study_metadata_xlsx,
        study_metadata_sheet,
    )
    raw_perf_rows, raw_ads_rows = collect_rows(datasets, selected_study_ids)

    raw_ads_rows = _build_adsorbent_rows(raw_ads_rows, study_metadata)
    normalized_ads_rows = _normalize_adsorbent_property_rows(
        raw_ads_rows,
        adsorbent_property_review_decisions_file,
    )
    expected_missing_property_study_keys = load_expected_missing_adsorbent_property_studies(
        adsorbent_property_review_decisions_file
    )
    perf_rows = _build_performance_rows(
        raw_perf_rows,
        normalized_ads_rows,
        study_metadata,
        expected_missing_property_study_keys=expected_missing_property_study_keys,
    )

    perf_df = _ordered_dataframe(perf_rows, PERFORMANCE_SCHEMA)
    ads_df = _ordered_dataframe(raw_ads_rows, ADSORBENT_SCHEMA)
    normalized_ads_df = _ordered_dataframe(normalized_ads_rows, ADSORBENT_SCHEMA)

    _write_workbook(performance_output_file, perf_df)
    _write_workbook(adsorbent_output_file, ads_df)
    if adsorbent_normalized_output_file:
        _write_workbook(adsorbent_normalized_output_file, normalized_ads_df)

    print(f"Written {len(perf_df)} performance rows to {performance_output_file}")
    print(f"Written {len(ads_df)} adsorbent rows to {adsorbent_output_file}")
    if adsorbent_normalized_output_file:
        print(
            f"Written {len(normalized_ads_df)} normalized adsorbent rows to "
            f"{adsorbent_normalized_output_file}"
        )
