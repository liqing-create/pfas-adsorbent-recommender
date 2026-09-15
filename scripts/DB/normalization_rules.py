"""Semantic normalization helpers for the adsorption database pipeline."""

from __future__ import annotations

import csv
import math
import os
import re
import unicodedata
from typing import Any, Optional

import pandas as pd

from performance_normalization_overrides import (
    normalized_adsorbent_scalar_normalization_key,
    normalized_expected_raw,
    normalized_performance_normalization_key,
)

from adsorbent_property_overrides import normalized_adsorbent_property_key

MAPPINGS_DIR = os.path.dirname(__file__)
PFAS_MW_OUTPUT_COLUMN = "Mw (g/mol)"
PFAS_MW_SOURCE_COLUMNS = ("rdkit_exact_mw", "Mw (g/mol)")
PFAS_SECOND_CLASS_OUTPUT_COLUMN = "Second_Class"
DEBUG_PFAS_C0 = False
DEBUG_PFAS_C0_STUDIES = ["study_02"]

ADDITIVE_ML_FEATURE_COLUMNS = [
    "organic_matter_present",
    "organic_matter_class",
    "inorganic_matter_present",
    "contains_Na",
    "contains_K",
    "contains_Ca",
    "contains_Mg",
    "contains_Cl",
    "contains_HCO3",
    "contains_SO4",
    "contains_phosphate",
]

def _norm_pfas_label(s: str) -> str:
    if not isinstance(s, str):
        return ""
    t = unicodedata.normalize("NFKC", s).lower().strip()
    return re.sub(r"[^a-z0-9]+", "", t)

# ---------- debug PFAS Co and dosage----------
def _debug_pfas_c0_row(idx, row, raw_val, raw_unit, u_norm, mw, stage, extra=""):
    """
    Lightweight helper to print out why PFAS_C0 conversions did
    or did not happen. This NEVER changes computation – it only
    logs to stdout when DEBUG_PFAS_C0 is True.
    Optionally respects DEBUG_PFAS_C0_STUDIES to limit output
    to specific study_no values.
    """
    if not DEBUG_PFAS_C0:
        return
    try:
        study = row.get("study_no", "")
    except Exception:
        study = ""

    # If a study filter is provided, only emit logs when the current
    # row's study_no matches one of the requested studies (string match).
    if DEBUG_PFAS_C0_STUDIES:
        study_str = str(study)
        allowed = {str(s) for s in DEBUG_PFAS_C0_STUDIES}
        if study_str not in allowed:
            return

    try:
        pfas = row.get("PFAS_name", row.get("name_abbreviation", ""))
    except Exception:
        pfas = ""
    print(
        "[PFAS_C0 DEBUG] "
        f"idx={idx}; study_no={study!r}; PFAS_name={pfas!r}; "
        f"raw_val={raw_val!r}; unit_raw={raw_unit!r}; "
        f"u_norm={u_norm!r}; mw={mw!r}; stage={stage}; {extra}"
    )

# ---------- Mapping ----------
def load_mapping_csv(filename, norm_key_fn=None):
    """
    Load a two-column CSV [key, canonical] into a dict.
    Optionally normalize the keys with `norm_key_fn`.

    Tries several common encodings (UTF-8, UTF-8 with BOM, cp1252, latin-1)
    so that older 'ANSI' CSVs won't crash with UnicodeDecodeError.
    """
    path = os.path.join(MAPPINGS_DIR, filename)
    mapping = {}

    encodings_to_try = ("utf-8", "utf-8-sig", "cp1252", "latin-1")
    last_err = None

    for enc in encodings_to_try:
        try:
            with open(path, newline="", encoding=enc) as f:
                reader = csv.DictReader(f)
                for row in reader:
                    key = row.get("key", "")
                    canon = row.get("canonical", "")
                    if not key:
                        continue
                    if norm_key_fn:
                        key = norm_key_fn(key)
                    mapping[key] = canon
            # If we got here, this encoding worked; stop trying others
            break
        except UnicodeDecodeError as e:
            last_err = e
            continue

    else:
        # Only reached if *all* encodings fail
        print(
            f"[ERROR] Failed to read mapping CSV {path!r} with "
            f"encodings {encodings_to_try}: {last_err}"
        )

    return mapping


def apply_value_map(
    df: pd.DataFrame,
    target_col: str,
    mapping: dict,
    norm_key_fn,
    source_cols=None,
    keep_raw_col: str | None = None,
):
    """
    Generic “standardize values” helper.

    - mapping: dict[normalized_key -> canonical_value]
    - norm_key_fn: function that takes a string and returns a normalized key
    - source_cols:
        * None        -> use target_col as both source & destination
        * list[str]   -> try these columns in order, first match wins
    - keep_raw_col:
        * if provided and not in df, it will be created as a copy of target_col
    """
    if source_cols is None:
        source_cols = [target_col]

    if keep_raw_col and keep_raw_col not in df.columns:
        df[keep_raw_col] = df[target_col]

    if target_col not in df.columns:
        df[target_col] = ""

    for idx, row in df.iterrows():
        canon = None
        for col in source_cols:
            val = row.get(col, "")
            if not isinstance(val, str) or not val.strip():
                continue
            key = norm_key_fn(val)
            canon = mapping.get(key)
            if canon:
                break
        if canon:
            df.at[idx, target_col] = canon

def standardize_pfas_names(df: pd.DataFrame) -> None:
    source_cols = ["PFAS_name", "PFAS_abbrev"]

    if "PFAS_name" not in df.columns:
        df["PFAS_name"] = ""
    if "PFAS_name_raw" not in df.columns:
        df["PFAS_name_raw"] = df["PFAS_name"]

    for idx, row in df.iterrows():
        saw_non_compound_name = False
        for col in source_cols:
            canon = _canonical_pfas_name(row.get(col, ""))
            if canon is None:
                continue
            if canon == "":
                saw_non_compound_name = True
                continue
            if canon:
                df.at[idx, "PFAS_name"] = canon
                break
        else:
            if saw_non_compound_name:
                df.at[idx, "PFAS_name"] = ""

# ---------- Water type canonicalization ----------
def _norm_water_key(s: str) -> str:
    if not isinstance(s, str):
        return ""
    t = unicodedata.normalize("NFKC", s).strip()
    t = (t.replace("—", "-").replace("–", "-")
           .replace("\u2212", "-").replace("\u2011", "-"))
    t = t.replace("_", " ").replace("/", " ").replace("-", " ")
    t = " ".join(t.split())
    return t.lower()

WATER_TYPE_MAP = load_mapping_csv("water_type_map.csv", norm_key_fn=_norm_water_key)
WATER_TYPE_MAP_NORM = {_norm_water_key(k): v for k, v in WATER_TYPE_MAP.items()}
WATER_TYPE_CLASSES = {
    "ultrapure water",
    "synthetic water",
    "groundwater",
    "AFFF solution",
    "surface water",
    "wastewater",
    "landfill leachate",
    "tap water",
    "others",
}


def _replace_water_type(val: str) -> str:
    if not isinstance(val, str) or not val.strip():
        return val
    key = _norm_water_key(val)
    hit = WATER_TYPE_MAP_NORM.get(key)
    if hit is not None:
        val = hit
        key = _norm_water_key(hit)
    class_aliases = {
        "ultrapure water": "ultrapure water",
        "synthetic water": "synthetic water",
        "groundwater": "groundwater",
        "ground water": "groundwater",
        "afff": "AFFF solution",
        "afff solution": "AFFF solution",
        "surface water": "surface water",
        "wastewater": "wastewater",
        "waste water": "wastewater",
        "landfill leachate": "landfill leachate",
        "leachate": "landfill leachate",
        "tap water": "tap water",
        "others": "others",
        "other": "others",
        "nom spiked di water": "synthetic water",
        "salt spiked di water": "synthetic water",
    }
    if key in class_aliases:
        return class_aliases[key]
    # The water-type enumerator deliberately keeps the source-faithful
    # ``Full_name`` alongside a controlled ``Class``.  The performance table
    # only retains one model-facing field, so classify common detailed labels
    # here and preserve the original text in Water_type_raw for audit.
    if any(token in key for token in ("scrubber", "retentate", "raw water")):
        return "others"
    if "contaminated well" in key or "well water" in key:
        return "groundwater"
    if "type 1 water" in key:
        return "synthetic water" if any(
            token in key for token in ("buffer", "nacl", "nahco3", "nano3", "na2so4")
        ) else "ultrapure water"
    if any(token in key for token in ("demineralized", "distilled deionized")):
        return "ultrapure water"
    if "deionized water" in key or "ultrapure water" in key:
        if any(
            token in key
            for token in (
                "amended",
                "buffer",
                "nacl",
                "nahco3",
                "nano3",
                "na2so4",
                "nah2po4",
                "hepes",
                "dom",
                "humic",
                "srdom",
            )
        ):
            return "synthetic water"
        return "ultrapure water"
    if "sulfuric acid" in key:
        # Acid used only for pH adjustment does not make an ultrapure matrix
        # synthetic under the extraction vocabulary.
        return "ultrapure water"
    if "solution" in key:
        if any(token in key for token in ("natural organic", "inorganic matter", "dom", "humic")):
            return "synthetic water"
        if any(token in key for token in ("pfas", "pfoa", "pfos", "pfba")):
            # Per the enumeration prompt, ultrapure water amended only with
            # PFAS is not a synthetic water matrix.
            return "ultrapure water"
        if any(
            token in key
            for token in (
                "salt",
                "nacl",
                "nahco3",
                "nano3",
                "na2so4",
                "nah2po4",
                "cacl2",
                "carbonate",
                "phosphate",
                "buffer",
                "sodium chloride",
            )
        ):
            return "synthetic water"
    if "salt solution" in key or "salt solutions" in key:
        return "synthetic water"
    if "river" in key or "stream" in key:
        return "surface water"
    if "lake" in key:
        return "surface water"
    if "milli q" in key or "milliq" in key or "millipore" in key:
        return "ultrapure water"
    if key.startswith("di ") or key == "di":
        return "ultrapure water"
    if "synthetic" in key or "simulated" in key:
        return "synthetic water"
    if "afff" in key:
        return "AFFF solution"
    if "waste water" in key or "wastewater" in key or key.startswith("ww"):
        return "wastewater"
    if "landfill" in key and "leachate" in key:
        return "landfill leachate"
    if "tap water" in key:
        return "tap water"
    if "ground water" in key or "groundwater" in key or key.startswith("gw"):
        return "groundwater"
    # Do not let source wording become an unbounded model category. The raw
    # wording remains available in Standardized_Full for review.
    return "others"


def standardize_water_type(df_in: pd.DataFrame) -> None:
    candidates = ["Water_type", "water_type", "Water Type", "Matrix", "matrix"]
    col = next((c for c in candidates if c in df_in.columns), "Water_type")
    if col not in df_in.columns:
        df_in[col] = ""

    if "Water_type_raw" not in df_in.columns:
        df_in["Water_type_raw"] = df_in[col]

    df_in["Water_type"] = df_in[col].astype("object").apply(_replace_water_type)


def _norm_shorthand_key(s: str) -> str:
    if not isinstance(s, str):
        return ""
    t = unicodedata.normalize("NFKC", s).strip()
    t = (t.replace("—", "-").replace("–", "-")
           .replace("\u2212", "-").replace("\u2011", "-"))
    t = t.replace("\u00A0", " ")
    t = " ".join(t.split())
    return t.upper()

def _canonical_fluorotelomer_name(s: str) -> Optional[str]:
    if not isinstance(s, str):
        return None
    t = unicodedata.normalize("NFKC", s).strip()
    t = (t.replace("â€”", "-").replace("â€“", "-")
           .replace("\u2212", "-").replace("\u2011", "-"))
    t = t.replace("_", " ").replace("-", " ")
    t = " ".join(t.split()).upper()

    match = re.fullmatch(r"(\d{1,2})\s*:?\s*(\d)\s*(FTS|FTSA)", t)
    if match:
        return f"{int(match.group(1))}:{match.group(2)} {match.group(3)}"

    compact = re.sub(r"[^A-Z0-9]+", "", t)
    match = re.fullmatch(r"(\d{2,3})(FTS|FTSA)", compact)
    if match:
        digits = match.group(1)
        return f"{int(digits[:-1])}:{digits[-1]} {match.group(2)}"
    return None


PFAS_SHORTHAND_MAP = load_mapping_csv("pfas_map.csv", norm_key_fn=_norm_shorthand_key)
PFAS_SHORTHAND_MAP_NORM = {_norm_shorthand_key(k): v for k, v in PFAS_SHORTHAND_MAP.items()}
CANON_NAME_BY_LABEL = {
    _norm_pfas_label(canon): canon
    for canon in PFAS_SHORTHAND_MAP.values()
}
NON_COMPOUND_PFAS_NAME_LABELS = {
    "pfas",
    "pfca",
    "pfsa",
    "aixr",
    "cac",
}


def _is_non_compound_pfas_name(val) -> bool:
    if not isinstance(val, str) or not val.strip():
        return False
    return _norm_pfas_label(val) in NON_COMPOUND_PFAS_NAME_LABELS

def _canonical_pfas_name(val) -> Optional[str]:
    if not isinstance(val, str) or not val.strip():
        return None
    if _is_non_compound_pfas_name(val):
        return ""
    key = _norm_shorthand_key(val)
    hit = PFAS_SHORTHAND_MAP_NORM.get(key)
    if hit is not None:
        return hit

    hit = _canonical_fluorotelomer_name(val)
    if hit is not None:
        return hit

    m = re.match(r"^\s*[Ll]\s*[-\s]\s*(.+)$", val.strip())
    if m:
        rest = m.group(1).strip()
        rest_label = _norm_pfas_label(rest)
        canon = CANON_NAME_BY_LABEL.get(rest_label)
        if canon:
            return canon
    return None

def _replace_pfas_shorthand(val):
    canon = _canonical_pfas_name(val)
    if canon is not None:
        return canon
    return val

# ======== Small generic helpers (new) =========

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


def _blank(x) -> bool:
    return (
        x is None
        or (isinstance(x, float) and math.isnan(x))
        or (isinstance(x, str) and not x.strip())
        or _is_extraction_missing_placeholder(x)
    )

def _reported_value_text(x) -> str:
    if _blank(x):
        return ""
    text = str(x).strip()
    if text.lower() in {"na", "n/a", "nan", "none", "not reported", "not applicable", "unclassified"}:
        return ""
    return text

def _missing_or_unclassified(x) -> bool:
    if _blank(x):
        return True
    if isinstance(x, str) and x.strip().lower() == "unclassified":
        return True
    return False


def blank_extraction_missing_placeholders(df_in: pd.DataFrame) -> None:
    object_cols = df_in.select_dtypes(include=["object"]).columns
    for col in object_cols:
        mask = df_in[col].apply(_is_extraction_missing_placeholder)
        if mask.any():
            df_in.loc[mask, col] = ""

def _as_float_or_blank(v):
    if _blank(v):
        return None
    try:
        return _coerce_number(v)
    except Exception:
        return None

_NORMALIZATION_ISSUE_RE = re.compile(
    r"(missing|mismatch|not\s+converted|unrecognized|invalid|failed|skipped|cannot\s+convert|≤\s*0|negative)",
    flags=re.I,
)

def _ensure_columns(df: pd.DataFrame, cols, default=""):
    """
    Ensure each column in `cols` exists, initializing missing ones with `default`.
    """
    missing = []
    for c in cols:
        if c not in df.columns and c not in missing:
            missing.append(c)

    if missing:
        additions = pd.DataFrame(
            {c: pd.Series(default, index=df.index, dtype="object") for c in missing}
        )
        # Keep the caller-visible DataFrame object, but add columns as one block.
        df._update_inplace(pd.concat([df, additions], axis=1))
    for c in cols:
        if c in df.columns:
            df[c] = df[c].astype("object")

def _append_normalization_notes(
    df: pd.DataFrame,
    notes,
    info_col="normalization_trace",
    issue_col="normalization_issues",
):
    """
    Split notes into successful-conversion info versus normalization issues.
    Behavior matches the previous repeated blocks.
    """
    infos, errs = [], []
    for n in notes:
        if not n:
            infos.append("")
            errs.append("")
        elif _NORMALIZATION_ISSUE_RE.search(n):
            infos.append("")
            errs.append(n)
        else:
            infos.append(n)
            errs.append("")

    _ensure_columns(df, [info_col, issue_col], default="")

    df[info_col] = (
        df[info_col].fillna("").astype(str).str.rstrip()
        + ["; " + i if i else "" for i in infos]
    ).str.strip("; ").str.strip()

    df[issue_col] = (
        df[issue_col].fillna("").astype(str).str.rstrip()
        + ["; " + e if e else "" for e in errs]
    ).str.strip("; ").str.strip()


# ======== Bounded adsorbent fractions =========

ELEMENTAL_COMPOSITION_COLUMNS = (
    "element_C_value",
    "element_N_value",
    "element_O_value",
)
ELEMENTAL_COMPOSITION_TOTAL_TOLERANCE = 1e-3

# (reported-source column, normalized model-facing column)
POROSITY_FRACTION_COLUMNS = (
    ("porosity_total_value", "porosity_total_value"),
    ("porosity_micro_value", "porosity_micro_value_avg"),
    ("porosity_meso_value", "porosity_meso_value_avg"),
    ("porosity_macro_value", "porosity_macro_value_avg"),
)

# A narrow spread accepts repeated/coherent characterizations (for example,
# Table S2 in study_110) but rejects values combined from distinct materials
# (for example, the two H2O2/Fe points in study_174).
MAX_FRACTION_RELATIVE_SPREAD = 0.10
_FRACTION_NUMBER_RE = re.compile(r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?")
PREMERGE_NORMALIZED_ADSORBENT_FRACTION_COLUMNS = (
    "premerge_normalized_adsorbent_fraction_columns"
)


def _fraction_numbers(value: Any) -> list[float]:
    """Return numeric values from a scalar, list, or range-like cell."""
    if _blank(value):
        return []
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return [float(value)] if math.isfinite(float(value)) else []

    text = unicodedata.normalize("NFKC", str(value))
    text = text.replace("−", "-").replace("–", "-").replace("—", "-")
    # Keep the reported central value and discard an attached uncertainty.
    text = re.sub(
        rf"({_FRACTION_NUMBER_RE.pattern})\s*(?:±|\+/-)\s*{_FRACTION_NUMBER_RE.pattern}",
        r"\1",
        text,
    )
    text = re.sub(r"(?<=\d)\s*(?:-|to)\s*(?=-?\d)", " ", text, flags=re.I)
    values: list[float] = []
    for match in _FRACTION_NUMBER_RE.findall(text):
        number = float(match)
        if math.isfinite(number):
            values.append(number)
    return values


ZETA_POTENTIAL_VALUE_COLUMN = "zeta_potential_value"
ZETA_POTENTIAL_PH_COLUMN = "zeta_potential_pH"
ZETA_POTENTIAL_NEAR_PH_7_COLUMN = "zeta_potential_near_pH_7_(mV)"
ZETA_POTENTIAL_ACCEPTED_PH_MIN = 6.5
ZETA_POTENTIAL_TARGET_PH = 7.0
ZETA_POTENTIAL_ACCEPTED_PH_MAX = 7.5
_ZETA_NUMBER_PATTERN = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?"
_ZETA_VALUE_RE = re.compile(
    rf"^\s*({_ZETA_NUMBER_PATTERN})\s*(?:mV)?\s*$",
    flags=re.I,
)
_ZETA_PH_RE = re.compile(
    rf"^\s*(?:pH\s*[_:=]?\s*)?({_ZETA_NUMBER_PATTERN})\s*$",
    flags=re.I,
)
_ZETA_PH_RANGE_RE = re.compile(
    rf"^\s*(?:pH\s*[_:=]?\s*)?({_ZETA_NUMBER_PATTERN})\s*(?:-|\u2013|\u2014|to)\s*({_ZETA_NUMBER_PATTERN})\s*$",
    flags=re.I,
)

def _zeta_series_tokens(value: Any) -> list[Any]:
    """Split explicitly paired zeta-potential series without inventing pairs."""
    if _blank(value):
        return []
    if isinstance(value, (list, tuple)):
        return list(value)
    return [part.strip() for part in re.split(r"\s*(?:&&|;)\s*", str(value)) if part.strip()]


def _normalized_zeta_text(value: Any) -> str:
    return (
        unicodedata.normalize("NFKC", str(value))
        .replace("\u2212", "-")
        .replace("\u2013", "-")
        .replace("\u2014", "-")
        .strip()
    )


def _zeta_value_number(value: Any) -> float | None:
    """Parse one reported zeta-potential value, retaining only clean scalars."""
    if _blank(value):
        return None
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        number = float(value)
        return number if math.isfinite(number) else None
    match = _ZETA_VALUE_RE.fullmatch(_normalized_zeta_text(value))
    if not match:
        return None
    number = float(match.group(1))
    return number if math.isfinite(number) else None


def _zeta_ph_interval(value: Any) -> tuple[float, float] | None:
    """Return a scalar pH or reported pH interval as finite ordered bounds."""
    if _blank(value):
        return None
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        number = float(value)
        return (number, number) if math.isfinite(number) else None
    text = _normalized_zeta_text(value)
    range_match = _ZETA_PH_RANGE_RE.fullmatch(text)
    if range_match:
        low, high = (float(range_match.group(1)), float(range_match.group(2)))
        if math.isfinite(low) and math.isfinite(high):
            return min(low, high), max(low, high)
        return None
    scalar_match = _ZETA_PH_RE.fullmatch(text)
    if not scalar_match:
        return None
    number = float(scalar_match.group(1))
    return (number, number) if math.isfinite(number) else None


def _zeta_distance_from_ph_7(value: Any) -> float | None:
    """Return a candidate's closest in-window distance from pH 7.0.

    A reported interval is retained when it overlaps the accepted window. This
    lets a value reported at ``pH 7-8`` use its pH-7 edge, while an interval
    wholly outside 6.5-7.5 remains unavailable to the model.
    """
    interval = _zeta_ph_interval(value)
    if interval is None:
        return None
    low, high = interval
    eligible_low = max(low, ZETA_POTENTIAL_ACCEPTED_PH_MIN)
    eligible_high = min(high, ZETA_POTENTIAL_ACCEPTED_PH_MAX)
    if eligible_low > eligible_high:
        return None
    closest_ph = min(max(ZETA_POTENTIAL_TARGET_PH, eligible_low), eligible_high)
    return abs(closest_ph - ZETA_POTENTIAL_TARGET_PH)


def zeta_potential_near_ph_7(df_in: pd.DataFrame) -> pd.Series:
    """Derive one numeric zeta potential paired to the pH nearest 7.0.

    Source pH/value lists must have the same number of explicitly paired
    entries. Unpaired, malformed, and out-of-window reports produce missing
    values rather than an inferred model feature. Ties retain source order.
    """
    result = pd.Series(float("nan"), index=df_in.index, dtype="float64")
    if (
        ZETA_POTENTIAL_PH_COLUMN not in df_in.columns
        or ZETA_POTENTIAL_VALUE_COLUMN not in df_in.columns
    ):
        return result

    for idx, row in df_in.iterrows():
        pH_values = _zeta_series_tokens(row.get(ZETA_POTENTIAL_PH_COLUMN))
        zeta_values = _zeta_series_tokens(row.get(ZETA_POTENTIAL_VALUE_COLUMN))
        if not pH_values or len(pH_values) != len(zeta_values):
            continue
        candidates: list[tuple[float, int, float]] = []
        for position, (pH_value, zeta_value) in enumerate(zip(pH_values, zeta_values)):
            distance = _zeta_distance_from_ph_7(pH_value)
            numeric_zeta = _zeta_value_number(zeta_value)
            if distance is not None and numeric_zeta is not None:
                candidates.append((distance, position, numeric_zeta))
        if candidates:
            result.at[idx] = min(candidates, key=lambda item: (item[0], item[1]))[2]
    return result


def _aggregate_fraction_values(
    values: list[float],
    *,
    field: str,
    scale_all_as_percent: bool = False,
) -> tuple[float | None, str]:
    """Scale, validate, and aggregate a bounded [0, 1] property."""
    if not values:
        return None, ""

    scaled = [value / 100.0 if scale_all_as_percent else (value / 100.0 if value > 1 else value) for value in values]
    if any(value < 0 or value > 1 for value in scaled):
        return None, f"{field}: invalid bounded-fraction value; normalized value blanked"

    mean = sum(scaled) / len(scaled)
    if len(scaled) > 1:
        spread = max(scaled) - min(scaled)
        if math.isclose(mean, 0.0, abs_tol=1e-12):
            coherent = math.isclose(spread, 0.0, abs_tol=1e-12)
        else:
            coherent = (spread / abs(mean)) <= MAX_FRACTION_RELATIVE_SPREAD
        if not coherent:
            return (
                None,
                f"{field}: invalid inconsistent multiple values; normalized value blanked",
            )

    return mean, ""


def normalize_bounded_fraction_values(
    values: list[float],
    *,
    field: str,
) -> tuple[float | None, str]:
    """Normalize one porosity-like bounded fraction value list."""
    return _aggregate_fraction_values(values, field=field)


def normalize_elemental_fraction_value_lists(
    values_by_column: dict[str, list[float]],
    *,
    allow_inconsistent_average: bool = False,
) -> tuple[dict[str, float | None], str]:
    """Normalize elemental composition reported as fractions or wt%.

    A row containing any value above one is treated as a wt% table, so every
    element in that row is divided by 100. This correctly handles rows such
    as C=50.05875, N=0.325, O=0.5. If all raw values are already below one,
    they are treated as fractions. A small combined overage (up to 0.1
    percentage point) is retained to accommodate measurement and rounding
    error; larger totals are considered ambiguous and are not used for
    modeling.
    """
    all_values = [value for values in values_by_column.values() for value in values]
    if not all_values:
        return {column: None for column in values_by_column}, ""

    scale_all_as_percent = any(value > 1 for value in all_values)
    normalized: dict[str, float | None] = {}
    notes: list[str] = []
    for column, values in values_by_column.items():
        value, note = _aggregate_fraction_values(
            values,
            field=column,
            scale_all_as_percent=scale_all_as_percent,
        )
        if (
            allow_inconsistent_average
            and value is None
            and note == f"{column}: invalid inconsistent multiple values; normalized value blanked"
        ):
            scaled = [
                raw_value / 100.0
                if scale_all_as_percent
                else (raw_value / 100.0 if raw_value > 1 else raw_value)
                for raw_value in values
            ]
            value = sum(scaled) / len(scaled)
            note = (
                f"{column}: manually approved average of {len(scaled)} reported values "
                "(adsorbent_property_review_decisions.csv)"
            )
        normalized[column] = value
        if note:
            notes.append(note)

    if notes:
        return normalized, "; ".join(dict.fromkeys(notes))

    total = sum(value for value in normalized.values() if value is not None)
    if total > 1 + ELEMENTAL_COMPOSITION_TOTAL_TOLERANCE:
        return (
            {column: None for column in values_by_column},
            "elemental composition: invalid combined fraction above one; normalized values blanked",
        )
    return normalized, ""


def _preserve_raw_field(df_in: pd.DataFrame, source_column: str) -> str:
    raw_column = f"{source_column}_raw"
    if source_column in df_in.columns and raw_column not in df_in.columns:
        df_in[raw_column] = df_in[source_column].astype("object").copy()
    return raw_column


def _premerge_fraction_columns(value: Any) -> set[str]:
    """Parse the internal marker that avoids re-normalizing copied properties."""
    return {part.strip() for part in str(value or "").split(";") if part.strip()}


def _adsorbent_property_identity_key(row: pd.Series) -> tuple[str, str, str]:
    return normalized_adsorbent_property_key(
        row.get("extraction_dataset"),
        row.get("study_no"),
        row.get("adsorbent_id"),
    )


def normalize_adsorbent_fraction_fields(
    df_in: pd.DataFrame,
    *,
    approved_elemental_average_keys: set[tuple[str, str, str]] | None = None,
    mark_premerge_normalization: bool = False,
) -> None:
    """Materialize model-ready elemental composition and porosity fractions.

    Raw cells stay in ``*_raw`` columns in Standardized_Full. The normalized
    values replace only the fields projected to Equilibrium_Data. Invalid or
    incoherent lists are blanked and recorded in the existing unit-error trace.

    The raw-adsorbent stage calls this function before copying properties to
    performance records.  It marks successfully normalized fields so the
    performance stage preserves their source lists and does not normalize them
    once per copied record.
    """
    approved_elemental_average_keys = approved_elemental_average_keys or set()
    if mark_premerge_normalization:
        _ensure_columns(df_in, [PREMERGE_NORMALIZED_ADSORBENT_FRACTION_COLUMNS], default="")
    element_raw_columns = {
        column: _preserve_raw_field(df_in, column)
        for column in ELEMENTAL_COMPOSITION_COLUMNS
        if column in df_in.columns
    }
    porosity_specs = [
        (source, normalized, _preserve_raw_field(df_in, source))
        for source, normalized in POROSITY_FRACTION_COLUMNS
        if source in df_in.columns
    ]
    for column in set(element_raw_columns).union(
        normalized for _, normalized, _ in porosity_specs
    ):
        if column not in df_in.columns:
            df_in[column] = ""
        df_in[column] = df_in[column].astype("object")

    notes = [""] * len(df_in)
    for pos, (idx, row) in enumerate(df_in.iterrows()):
        premerged_columns = (
            set()
            if mark_premerge_normalization
            else _premerge_fraction_columns(
                row.get(PREMERGE_NORMALIZED_ADSORBENT_FRACTION_COLUMNS, "")
            )
        )
        newly_premerged_columns: set[str] = set()
        reported_by_column = {
            column: column not in premerged_columns and not _blank(row.get(raw_column, ""))
            for column, raw_column in element_raw_columns.items()
        }
        values_by_column = {
            column: (
                []
                if column in premerged_columns
                else _fraction_numbers(row.get(raw_column, ""))
            )
            for column, raw_column in element_raw_columns.items()
        }
        invalid_element_columns = [
            column
            for column, reported in reported_by_column.items()
            if reported and not values_by_column[column]
        ]
        for column in invalid_element_columns:
            df_in.at[idx, column] = ""
        if invalid_element_columns:
            notes[pos] = "; ".join(
                f"{column}: invalid nonnumeric reported value; normalized value blanked"
                for column in invalid_element_columns
            )
        if any(values_by_column.values()):
            normalized, note = normalize_elemental_fraction_value_lists(
                values_by_column,
                allow_inconsistent_average=(
                    _adsorbent_property_identity_key(row)
                    in approved_elemental_average_keys
                ),
            )
            for column, value in normalized.items():
                if values_by_column[column]:
                    df_in.at[idx, column] = "" if value is None else value
                    if mark_premerge_normalization and value is not None:
                        newly_premerged_columns.add(column)
            if note:
                notes[pos] = note

        porosity_notes: list[str] = []
        for source, normalized_column, raw_column in porosity_specs:
            if normalized_column in premerged_columns:
                continue
            raw_value = row.get(raw_column, "")
            values = _fraction_numbers(raw_value)
            if not values:
                if not _blank(raw_value):
                    df_in.at[idx, normalized_column] = ""
                    porosity_notes.append(
                        f"{normalized_column}: invalid nonnumeric reported value; "
                        "normalized value blanked"
                    )
                continue
            value, note = _aggregate_fraction_values(values, field=normalized_column)
            df_in.at[idx, normalized_column] = "" if value is None else value
            if mark_premerge_normalization and value is not None:
                newly_premerged_columns.add(normalized_column)
            if note:
                porosity_notes.append(note)
        if porosity_notes:
            notes[pos] = "; ".join(part for part in (notes[pos], *porosity_notes) if part)
        if mark_premerge_normalization and newly_premerged_columns:
            existing = _premerge_fraction_columns(
                row.get(PREMERGE_NORMALIZED_ADSORBENT_FRACTION_COLUMNS, "")
            )
            df_in.at[idx, PREMERGE_NORMALIZED_ADSORBENT_FRACTION_COLUMNS] = "; ".join(
                sorted(existing | newly_premerged_columns)
            )

    if any(notes):
        _append_normalization_notes(df_in, notes)
    if (
        not mark_premerge_normalization
        and PREMERGE_NORMALIZED_ADSORBENT_FRACTION_COLUMNS in df_in.columns
    ):
        df_in.drop(columns=[PREMERGE_NORMALIZED_ADSORBENT_FRACTION_COLUMNS], inplace=True)


# ======== Additive parsing and ionic-strength features =========

MISSING_SPECIES_TOKENS = {"", "none", "na", "n/a", "nan", "null", "not reported", "not found"}


def _fmt_float(value: float) -> str:
    return f"{value:.6g}"


def _to_molar(val: str, unit: str, species_name: str = "") -> Optional[float]:
    n = _coerce_number(val)
    if n is None:
        return None
    raw = str(unit or "").strip().replace("\u03bc", "\u00b5").replace("\u00c2\u00b5", "\u00b5")
    from normalization_units import normalize_unit

    normalized = normalize_unit(unit or "")
    def _unit_key(text: str) -> str:
        return str(text or "").casefold().replace("\u03bc", "\u00b5").replace("\u00c2\u00b5", "\u00b5")
    candidates = {
        _unit_key(raw),
        _unit_key(normalized),
    }
    if candidates & {"mol/l", "m"} or raw == "M":
        return float(n)
    if candidates & {"mmol/l", "mm"}:
        return float(n) * 1e-3
    if candidates & {"\u00b5mol/l", "umol/l", "\u00b5m", "um"}:
        return float(n) * 1e-6

    mass_conc_factors_g_per_l = {
        "g/l": 1.0,
        "mg/l": 1e-3,
        "\u00b5g/l": 1e-6,
        "ug/l": 1e-6,
        "ng/l": 1e-9,
        "ppm": 1e-3,
        "ppb": 1e-6,
        "ppt": 1e-9,
    }
    for candidate in candidates:
        factor = mass_conc_factors_g_per_l.get(candidate)
        if factor is None:
            continue
        mw = _species_molar_mass(species_name)
        if mw is None:
            return None
        return float(n) * factor / mw

    equiv_conc_factors_eq_per_l = {
        "eq/l": 1.0,
        "meq/l": 1e-3,
        "\u00b5eq/l": 1e-6,
        "ueq/l": 1e-6,
    }
    for candidate in candidates:
        factor = equiv_conc_factors_eq_per_l.get(candidate)
        if factor is None:
            continue
        charge = _species_equivalent_charge(species_name)
        if charge is None or charge <= 0:
            return None
        return float(n) * factor / charge
    return None


def _clean_species_label(name: str) -> str:
    text = unicodedata.normalize("NFKC", str(name or "")).strip()
    text = text.replace("\u2212", "-").replace("\u2013", "-").replace("\u2014", "-")
    text = text.replace("\uff0b", "+").replace("\ufe62", "+")
    text = re.sub(r"_(?:[<>]=?|~)?\d+(?:\.\d+)?_or$", "", text, flags=re.I)
    return re.sub(r"\s+", " ", text)


def _split_ambiguous_species_prefix(name: str) -> tuple[str, str] | None:
    match = re.match(
        r"^(?P<name>.+)_(?P<value>[<>]=?|~)?(?P<num>\d+(?:\.\d+)?)_or$",
        str(name or "").strip(),
        flags=re.I,
    )
    if not match:
        return None
    first_value = (match.group("value") or "") + match.group("num")
    return _clean_species_label(match.group("name")), first_value


def _species_key(name: str) -> str:
    text = _clean_species_label(name).casefold()
    text = text.replace(" ", "").replace("^", "")
    text = text.replace("-", "minus").replace("+", "plus")
    return re.sub(r"[^a-z0-9]+", "", text)


def _canonical_charge_label(formula: str, charge: int) -> str:
    sign = "+" if charge > 0 else "-"
    magnitude = abs(charge)
    suffix = f"^{magnitude}{sign}" if magnitude != 1 else f"^{sign}"
    return f"{formula}{suffix}"


def _parse_explicit_ion(name: str) -> tuple[str, int] | None:
    text = _clean_species_label(name).replace(" ", "")
    match = re.match(r"^(?P<formula>[A-Za-z][A-Za-z0-9()]*)\^(?P<num>\d*)(?P<sign>[+-])$", text)
    if not match:
        match = re.match(r"^(?P<formula>[A-Za-z][A-Za-z0-9()]*)(?P<sign>[+-])$", text)
    if not match:
        return None
    magnitude = int(match.groupdict().get("num") or "1")
    charge = magnitude if match.group("sign") == "+" else -magnitude
    return _canonical_charge_label(match.group("formula"), charge), charge


def _canonicalize_additive_unit(unit: str) -> str:
    """Return the canonical spelling for supported additive concentration units."""
    text = str(unit or "").strip()
    if re.fullmatch(r"mg\s*_?\s*C\s*/\s*L", text, flags=re.IGNORECASE):
        return "mgC/L"
    return text


def _split_species_cell(s: str) -> list[dict]:
    out = []
    if not isinstance(s, str) or not s.strip():
        return out
    # Commas can delimit multiple additives, but they are also part of common
    # chemical names (for example, 1,4-dioxane) and numeric values.  Treat a
    # comma as a delimiter only when the next token begins with a letter.
    for chunk in re.split(r"\s*(?:&&|;|,(?=\s*[A-Za-z]))\s*", s):
        t = chunk.strip()
        if not t or t.casefold() in MISSING_SPECIES_TOKENS:
            continue
        # Preserve a source-reported range such as ``K2SO4_1_mM_or_2_mM`` as
        # an explicitly ambiguous concentration.  The generic parser below
        # would otherwise treat ``K2SO4_1_mM_or`` as the chemical name and
        # incorrectly report a known salt as unrecognized.
        range_token = re.match(
            r"^(?P<name>.+?)_(?P<first>[<>]=?|~)?(?P<first_num>\d+(?:\.\d+)?)_"
            r"(?P<first_unit>.+?)_or_(?P<second>[<>]=?|~)?(?P<second_num>\d+(?:\.\d+)?)_"
            r"(?P<second_unit>.+)$",
            t,
        )
        if range_token:
            name_clean = _clean_species_label(range_token.group("name"))
            first_value = (range_token.group("first") or "") + range_token.group("first_num")
            second_value = (range_token.group("second") or "") + range_token.group("second_num")
            first_unit = _canonicalize_additive_unit(range_token.group("first_unit"))
            second_unit = _canonicalize_additive_unit(range_token.group("second_unit"))
            out.append(
                {
                    "raw": t,
                    "name": name_clean,
                    "value": f"{first_value}_{first_unit}_or_{second_value}_{second_unit}",
                    "unit": "",
                    "molar": None,
                    "ambiguous_concentration": True,
                }
            )
            continue
        # A unit can include an underscore (for example ``mg_C/L``).  Parse
        # the last numeric value marker rather than blindly splitting on the
        # last two underscores, so the unit remains intact and the species
        # name is available for de-duplication.
        explicit_token = re.match(
            r"^(?P<name>.+)_(?P<value>[<>]=?|~)?(?P<num>\d+(?:\.\d+)?)_(?P<unit>.+)$",
            t,
        )
        if explicit_token:
            name = explicit_token.group("name")
            value = (explicit_token.group("value") or "") + explicit_token.group("num")
            unit = explicit_token.group("unit")
        else:
            parts = t.rsplit("_", 2)
            name, value, unit = parts if len(parts) == 3 else ("", "", "")
        if name:
            ambiguous = _split_ambiguous_species_prefix(name)
            if ambiguous:
                name_clean, first_value = ambiguous
                value_clean = f"{first_value}_or_{value.strip()}"
                molar = None
            else:
                name_clean = _clean_species_label(name)
                value_clean = value.strip()
                unit = _canonicalize_additive_unit(unit)
                molar = _to_molar(value_clean, unit, name_clean)
            out.append(
                {
                    "raw": t,
                    "name": name_clean,
                    "value": value_clean,
                    "unit": _canonicalize_additive_unit(unit),
                    "molar": molar,
                    "ambiguous_concentration": bool(ambiguous),
                }
            )
        else:
            out.append({"raw": t, "name": _clean_species_label(t), "value": "", "unit": "", "molar": None})
    return out


def _split_species_cells(txt: str):
    return [
        {"name": item["name"], "value": item["value"], "unit": item["unit"]}
        for item in _split_species_cell(txt)
    ]


def normalize_organic_matter_units(df_in: pd.DataFrame) -> None:
    """Canonicalize organic-carbon concentration units in Organic_matter."""
    _ensure_columns(df_in, ["Organic_matter"], default="")
    for idx, value in df_in["Organic_matter"].items():
        if _blank(value):
            continue
        raw = str(value)
        canonical = re.sub(
            r"(?<![A-Za-z0-9])mg\s*_\s*C\s*/\s*L\b",
            "mgC/L",
            raw,
            flags=re.IGNORECASE,
        )
        if canonical != raw:
            df_in.at[idx, "Organic_matter"] = canonical


CHEM_ION_LIBRARY_RAW: dict[str, list[tuple[str, int, float]]] = {
    "HCl": [("H^+", +1, 1.0), ("Cl^-", -1, 1.0)],
    "hydrochloric acid": [("H^+", +1, 1.0), ("Cl^-", -1, 1.0)],
    "HNO3": [("H^+", +1, 1.0), ("NO3^-", -1, 1.0)],
    "nitric acid": [("H^+", +1, 1.0), ("NO3^-", -1, 1.0)],
    "HClO4": [("H^+", +1, 1.0), ("ClO4^-", -1, 1.0)],
    "perchloric acid": [("H^+", +1, 1.0), ("ClO4^-", -1, 1.0)],
    "H2SO4": [("H^+", +1, 2.0), ("SO4^2-", -2, 1.0)],
    "sulfuric acid": [("H^+", +1, 2.0), ("SO4^2-", -2, 1.0)],
    "NaOH": [("Na^+", +1, 1.0), ("OH^-", -1, 1.0)],
    "sodium hydroxide": [("Na^+", +1, 1.0), ("OH^-", -1, 1.0)],
    "KOH": [("K^+", +1, 1.0), ("OH^-", -1, 1.0)],
    "potassium hydroxide": [("K^+", +1, 1.0), ("OH^-", -1, 1.0)],
    "Ca(OH)2": [("Ca^2+", +2, 1.0), ("OH^-", -1, 2.0)],
    "calcium hydroxide": [("Ca^2+", +2, 1.0), ("OH^-", -1, 2.0)],
    "NaCl": [("Na^+", +1, 1.0), ("Cl^-", -1, 1.0)],
    "sodium chloride": [("Na^+", +1, 1.0), ("Cl^-", -1, 1.0)],
    "KCl": [("K^+", +1, 1.0), ("Cl^-", -1, 1.0)],
    "potassium chloride": [("K^+", +1, 1.0), ("Cl^-", -1, 1.0)],
    "CaCl2": [("Ca^2+", +2, 1.0), ("Cl^-", -1, 2.0)],
    "calcium chloride": [("Ca^2+", +2, 1.0), ("Cl^-", -1, 2.0)],
    "MgCl2": [("Mg^2+", +2, 1.0), ("Cl^-", -1, 2.0)],
    "magnesium chloride": [("Mg^2+", +2, 1.0), ("Cl^-", -1, 2.0)],
    "NaNO3": [("Na^+", +1, 1.0), ("NO3^-", -1, 1.0)],
    "sodium nitrate": [("Na^+", +1, 1.0), ("NO3^-", -1, 1.0)],
    "Na2SO4": [("Na^+", +1, 2.0), ("SO4^2-", -2, 1.0)],
    "sodium sulfate": [("Na^+", +1, 2.0), ("SO4^2-", -2, 1.0)],
    "sodium sulphate": [("Na^+", +1, 2.0), ("SO4^2-", -2, 1.0)],
    "K2SO4": [("K^+", +1, 2.0), ("SO4^2-", -2, 1.0)],
    "potassium sulfate": [("K^+", +1, 2.0), ("SO4^2-", -2, 1.0)],
    "potassium sulphate": [("K^+", +1, 2.0), ("SO4^2-", -2, 1.0)],
    "CaSO4": [("Ca^2+", +2, 1.0), ("SO4^2-", -2, 1.0)],
    "calcium sulfate": [("Ca^2+", +2, 1.0), ("SO4^2-", -2, 1.0)],
    "calcium sulphate": [("Ca^2+", +2, 1.0), ("SO4^2-", -2, 1.0)],
    "MgSO4": [("Mg^2+", +2, 1.0), ("SO4^2-", -2, 1.0)],
    "magnesium sulfate": [("Mg^2+", +2, 1.0), ("SO4^2-", -2, 1.0)],
    "magnesium sulphate": [("Mg^2+", +2, 1.0), ("SO4^2-", -2, 1.0)],
    "(NH4)2SO4": [("NH4^+", +1, 2.0), ("SO4^2-", -2, 1.0)],
    "ammonium sulfate": [("NH4^+", +1, 2.0), ("SO4^2-", -2, 1.0)],
    "ammonium sulphate": [("NH4^+", +1, 2.0), ("SO4^2-", -2, 1.0)],
    "KNO3": [("K^+", +1, 1.0), ("NO3^-", -1, 1.0)],
    "potassium nitrate": [("K^+", +1, 1.0), ("NO3^-", -1, 1.0)],
    "Ca(NO3)2": [("Ca^2+", +2, 1.0), ("NO3^-", -1, 2.0)],
    "calcium nitrate": [("Ca^2+", +2, 1.0), ("NO3^-", -1, 2.0)],
    "Mg(NO3)2": [("Mg^2+", +2, 1.0), ("NO3^-", -1, 2.0)],
    "magnesium nitrate": [("Mg^2+", +2, 1.0), ("NO3^-", -1, 2.0)],
    "NaHCO3": [("Na^+", +1, 1.0), ("HCO3^-", -1, 1.0)],
    "sodium bicarbonate": [("Na^+", +1, 1.0), ("HCO3^-", -1, 1.0)],
    "sodium hydrogen carbonate": [("Na^+", +1, 1.0), ("HCO3^-", -1, 1.0)],
    "KHCO3": [("K^+", +1, 1.0), ("HCO3^-", -1, 1.0)],
    "potassium bicarbonate": [("K^+", +1, 1.0), ("HCO3^-", -1, 1.0)],
    "potassium hydrogen carbonate": [("K^+", +1, 1.0), ("HCO3^-", -1, 1.0)],
    "Na2CO3": [("Na^+", +1, 2.0), ("CO3^2-", -2, 1.0)],
    "sodium carbonate": [("Na^+", +1, 2.0), ("CO3^2-", -2, 1.0)],
    "CaCO3": [("Ca^2+", +2, 1.0), ("CO3^2-", -2, 1.0)],
    "calcium carbonate": [("Ca^2+", +2, 1.0), ("CO3^2-", -2, 1.0)],
    "Na2HPO4": [("Na^+", +1, 2.0), ("HPO4^2-", -2, 1.0)],
    "disodium hydrogen phosphate": [("Na^+", +1, 2.0), ("HPO4^2-", -2, 1.0)],
    "NaH2PO4": [("Na^+", +1, 1.0), ("H2PO4^-", -1, 1.0)],
    "sodium dihydrogen phosphate": [("Na^+", +1, 1.0), ("H2PO4^-", -1, 1.0)],
    "K2HPO4": [("K^+", +1, 2.0), ("HPO4^2-", -2, 1.0)],
    "dipotassium hydrogen phosphate": [("K^+", +1, 2.0), ("HPO4^2-", -2, 1.0)],
    "KH2PO4": [("K^+", +1, 1.0), ("H2PO4^-", -1, 1.0)],
    "potassium dihydrogen phosphate": [("K^+", +1, 1.0), ("H2PO4^-", -1, 1.0)],
    "NH4NO3": [("NH4^+", +1, 1.0), ("NO3^-", -1, 1.0)],
    "ammonium nitrate": [("NH4^+", +1, 1.0), ("NO3^-", -1, 1.0)],
    "NaN3": [("Na^+", +1, 1.0), ("N3^-", -1, 1.0)],
    "sodium azide": [("Na^+", +1, 1.0), ("N3^-", -1, 1.0)],
    "K2Cr2O7": [("K^+", +1, 2.0), ("Cr2O7^2-", -2, 1.0)],
    "potassium dichromate": [("K^+", +1, 2.0), ("Cr2O7^2-", -2, 1.0)],
    "Ca^2+": [("Ca^2+", +2, 1.0)],
    "Ca": [("Ca^2+", +2, 1.0)],
    "calcium": [("Ca^2+", +2, 1.0)],
    "Mg^2+": [("Mg^2+", +2, 1.0)],
    "Mg": [("Mg^2+", +2, 1.0)],
    "magnesium": [("Mg^2+", +2, 1.0)],
    "Na^+": [("Na^+", +1, 1.0)],
    "Na": [("Na^+", +1, 1.0)],
    "sodium": [("Na^+", +1, 1.0)],
    "K^+": [("K^+", +1, 1.0)],
    "K": [("K^+", +1, 1.0)],
    "potassium": [("K^+", +1, 1.0)],
    "Cl^-": [("Cl^-", -1, 1.0)],
    "chloride": [("Cl^-", -1, 1.0)],
    "NO3^-": [("NO3^-", -1, 1.0)],
    "nitrate": [("NO3^-", -1, 1.0)],
    "NO2^-": [("NO2^-", -1, 1.0)],
    "nitrite": [("NO2^-", -1, 1.0)],
    "SO4^2-": [("SO4^2-", -2, 1.0)],
    "sulfate": [("SO4^2-", -2, 1.0)],
    "sulphate": [("SO4^2-", -2, 1.0)],
    "HCO3^-": [("HCO3^-", -1, 1.0)],
    "bicarbonate": [("HCO3^-", -1, 1.0)],
    "CO3^2-": [("CO3^2-", -2, 1.0)],
    "carbonate": [("CO3^2-", -2, 1.0)],
    "PO4^3-": [("PO4^3-", -3, 1.0)],
    "phosphate": [("PO4^3-", -3, 1.0)],
    "HPO4^2-": [("HPO4^2-", -2, 1.0)],
    "hydrogen phosphate": [("HPO4^2-", -2, 1.0)],
    "H2PO4^-": [("H2PO4^-", -1, 1.0)],
    "dihydrogen phosphate": [("H2PO4^-", -1, 1.0)],
    "NH4^+": [("NH4^+", +1, 1.0)],
    "ammonium": [("NH4^+", +1, 1.0)],
    "F^-": [("F^-", -1, 1.0)],
    "fluoride": [("F^-", -1, 1.0)],
    "Br^-": [("Br^-", -1, 1.0)],
    "bromide": [("Br^-", -1, 1.0)],
    "Li^+": [("Li^+", +1, 1.0)],
    "lithium": [("Li^+", +1, 1.0)],
    "Cd^2+": [("Cd^2+", +2, 1.0)],
    "cadmium": [("Cd^2+", +2, 1.0)],
    "Mn^2+": [("Mn^2+", +2, 1.0)],
    "Mn": [("Mn^2+", +2, 1.0)],
    "manganese": [("Mn^2+", +2, 1.0)],
    "Cr2O7^2-": [("Cr2O7^2-", -2, 1.0)],
    "dichromate": [("Cr2O7^2-", -2, 1.0)],
    "S^2-": [("S^2-", -2, 1.0)],
    "sulfide": [("S^2-", -2, 1.0)],
    "sulphide": [("S^2-", -2, 1.0)],
    "NO3^--N": [("NO3^-", -1, 1.0)],
    "H^+": [("H^+", +1, 1.0)],
    "hydrogen": [("H^+", +1, 1.0)],
    "OH^-": [("OH^-", -1, 1.0)],
    "hydroxide": [("OH^-", -1, 1.0)],
}

CHEM_ION_LIBRARY = {_species_key(name): ions for name, ions in CHEM_ION_LIBRARY_RAW.items()}
CHEM_MOLAR_MASS_RAW: dict[str, float] = {
    "H^+": 1.00794,
    "OH^-": 17.00734,
    "Na^+": 22.98977,
    "Na": 22.98977,
    "sodium": 22.98977,
    "K^+": 39.0983,
    "K": 39.0983,
    "potassium": 39.0983,
    "Li^+": 6.94,
    "lithium": 6.94,
    "Ca^2+": 40.078,
    "Ca": 40.078,
    "calcium": 40.078,
    "Mg^2+": 24.305,
    "Mg": 24.305,
    "magnesium": 24.305,
    "Cd^2+": 112.414,
    "cadmium": 112.414,
    "Mn^2+": 54.93804,
    "Mn": 54.93804,
    "manganese": 54.93804,
    "Cr2O7^2-": 215.988,
    "dichromate": 215.988,
    "S^2-": 32.065,
    "sulfide": 32.065,
    "sulphide": 32.065,
    "Cl^-": 35.453,
    "chloride": 35.453,
    "F^-": 18.9984,
    "fluoride": 18.9984,
    "Br^-": 79.904,
    "bromide": 79.904,
    "NO3^-": 62.0049,
    "nitrate": 62.0049,
    "NO3^--N": 14.0067,
    "NO2^-": 46.0055,
    "nitrite": 46.0055,
    "SO4^2-": 96.0636,
    "sulfate": 96.0636,
    "sulphate": 96.0636,
    "HCO3^-": 61.0168,
    "bicarbonate": 61.0168,
    "CO3^2-": 60.0089,
    "carbonate": 60.0089,
    "PO4^3-": 94.9714,
    "phosphate": 94.9714,
    "HPO4^2-": 95.9793,
    "hydrogen phosphate": 95.9793,
    "H2PO4^-": 96.9872,
    "dihydrogen phosphate": 96.9872,
    "PO3^-": 78.9710,
    "NH4^+": 18.0385,
    "ammonium": 18.0385,
    "NaCl": 58.4428,
    "sodium chloride": 58.4428,
    "KCl": 74.5513,
    "potassium chloride": 74.5513,
    "CaCl2": 110.984,
    "calcium chloride": 110.984,
    "MgCl2": 95.211,
    "magnesium chloride": 95.211,
    "NaNO3": 84.9947,
    "sodium nitrate": 84.9947,
    "Na2SO4": 142.043,
    "sodium sulfate": 142.043,
    "sodium sulphate": 142.043,
    "K2SO4": 174.2592,
    "potassium sulfate": 174.2592,
    "potassium sulphate": 174.2592,
    "CaSO4": 136.1406,
    "calcium sulfate": 136.1406,
    "calcium sulphate": 136.1406,
    "MgSO4": 120.369,
    "magnesium sulfate": 120.369,
    "magnesium sulphate": 120.369,
    "(NH4)2SO4": 132.1395,
    "ammonium sulfate": 132.1395,
    "ammonium sulphate": 132.1395,
    "KNO3": 101.1032,
    "potassium nitrate": 101.1032,
    "Ca(NO3)2": 164.0878,
    "calcium nitrate": 164.0878,
    "Mg(NO3)2": 148.3148,
    "magnesium nitrate": 148.3148,
    "NaHCO3": 84.0066,
    "sodium bicarbonate": 84.0066,
    "sodium hydrogen carbonate": 84.0066,
    "KHCO3": 100.1151,
    "potassium bicarbonate": 100.1151,
    "potassium hydrogen carbonate": 100.1151,
    "Na2CO3": 105.9888,
    "sodium carbonate": 105.9888,
    "CaCO3": 100.0869,
    "calcium carbonate": 100.0869,
    "Na2HPO4": 141.9588,
    "disodium hydrogen phosphate": 141.9588,
    "NaH2PO4": 119.977,
    "sodium dihydrogen phosphate": 119.977,
    "K2HPO4": 174.176,
    "dipotassium hydrogen phosphate": 174.176,
    "KH2PO4": 136.086,
    "potassium dihydrogen phosphate": 136.086,
    "NH4NO3": 80.0434,
    "ammonium nitrate": 80.0434,
    "NaN3": 65.0099,
    "sodium azide": 65.0099,
    "K2Cr2O7": 294.185,
    "potassium dichromate": 294.185,
}
CHEM_MOLAR_MASS = {_species_key(name): mass for name, mass in CHEM_MOLAR_MASS_RAW.items()}
PH_ADJUSTER_KEYS = {
    _species_key(name)
    for name in (
        "HCl", "HNO3", "H2SO4", "NaOH", "KOH", "Ca(OH)2",
        "hydrochloric acid", "nitric acid", "sulfuric acid",
        "sodium hydroxide", "potassium hydroxide", "calcium hydroxide",
    )
}


# Explicit chemistry embedded in Water_type_raw is useful experimental context,
# but it previously only informed the categorical Water_type field.  Restrict
# recovery to named species and stated concentrations; generic labels such as
# "synthetic water" and "aqueous solution" remain unexpanded.
_WATER_CONTEXT_NUMBER = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?"
_WATER_CONTEXT_UNIT = (
    r"(?:millimolar|micromolar|mmol\s*/\s*[lL]|mol\s*/\s*[lL]|"
    r"[mun]?M|mg(?:-C)?\s*/\s*[lL]|[mun]?g\s*/\s*[lL])"
)
_WATER_CONTEXT_INORGANIC_PATTERNS = (
    ("NaHCO3", r"\b(?:NaHCO3|sodium\s+bicarbonate|sodium\s+hydrogen\s+carbonate)\b"),
    ("NaCl", r"\b(?:NaCl|sodium\s+chloride)\b"),
    ("CaCl2", r"\b(?:CaCl2|calcium\s+chloride)\b"),
    ("MgCl2", r"\b(?:MgCl2|magnesium\s+chloride)\b"),
    ("KCl", r"\b(?:KCl|potassium\s+chloride)\b"),
    ("NaNO3", r"\b(?:NaNO3|sodium\s+nitrate)\b"),
    ("Na2SO4", r"\b(?:Na2SO4|sodium\s+sul(?:f|ph)ate)\b"),
    ("MgSO4", r"\b(?:MgSO4|magnesium\s+sul(?:f|ph)ate)\b"),
    ("NaH2PO4", r"\b(?:NaH2PO4|sodium\s+dihydrogen\s+phosphate)\b"),
    ("Na2HPO4", r"\b(?:Na2HPO4|disodium\s+hydrogen\s+phosphate)\b"),
    ("KH2PO4", r"\b(?:KH2PO4|potassium\s+dihydrogen\s+phosphate)\b"),
    ("K2HPO4", r"\b(?:K2HPO4|dipotassium\s+hydrogen\s+phosphate)\b"),
    ("H2SO4", r"\b(?:H2SO4|sulfuric\s+acid|sulphuric\s+acid)\b"),
    ("HCl", r"\b(?:HCl|hydrochloric\s+acid)\b"),
    ("HNO3", r"\b(?:HNO3|nitric\s+acid)\b"),
)
_WATER_CONTEXT_ORGANIC_PATTERNS = (
    ("SRNOM", r"\b(?:SRNOM\d*|Suwannee\s+River\s+natural\s+organic\s+matter)\b"),
    ("SRDOM", r"\bSRDOM\d*\b"),
    ("humic acid", r"\bhumic\s+acid\b"),
    ("fulvic acid", r"\bfulvic\s+acid\b"),
    ("HEPES", r"\bHEPES\b"),
)
_WATER_CONTEXT_GENERIC_ORGANIC_RE = re.compile(
    r"\b(?:added\s+)?organic\s+compounds?\b|"
    r"\b(?:DOM|NOM|EfOM)\b|\bnatural\s+organic\s+matter\b",
    flags=re.IGNORECASE,
)
_WATER_CONTEXT_PH_RE = re.compile(
    rf"\bp\s*h\s*(?:[=:~≈]\s*)?(?P<value>{_WATER_CONTEXT_NUMBER})\b",
    flags=re.IGNORECASE,
)

# Water-context recovery must not append a generic chemical name when the
# extracted additive already reports the same humic/fulvic material by a common
# shorthand.  Keep distinct humic and fulvic substances distinct; these keys
# are only aliases within each substance family.
_ORGANIC_RECOVERY_ALIAS_KEYS = {
    "ha": "humic acid",
    "srha": "humic acid",
    "ppha": "humic acid",
    "humic": "humic acid",
    "humicacid": "humic acid",
    "fa": "fulvic acid",
    "srfa": "fulvic acid",
    "plfa": "fulvic acid",
    "fulvic": "fulvic acid",
    "fulvicacid": "fulvic acid",
}


def _recovery_species_key(name: str) -> str:
    key = _species_key(name)
    return _ORGANIC_RECOVERY_ALIAS_KEYS.get(key, key)


def _normalized_water_context_unit(unit: str) -> str:
    key = re.sub(r"\s+", "", str(unit or "")).casefold()
    if key == "millimolar":
        return "mM"
    if key == "micromolar":
        return "µM"
    return str(unit or "").replace(" ", "")


def _water_context_species_with_concentrations(
    text: str,
    patterns: tuple[tuple[str, str], ...],
) -> list[str]:
    """Return source-explicit species as additive-parser-compatible tokens."""
    if not text:
        return []
    # In statements such as "SO4 replacing NaHCO3 (ionic-strength equivalent
    # to 10 mM NaHCO3)", the named NaHCO3 is a comparison basis, not an added
    # chemical.  Retain pH/organic context but avoid inventing a salt mixture.
    skip_inorganic_comparison = "replacing" in text.casefold() and "equivalent to" in text.casefold()
    out: list[str] = []
    for canonical, pattern in patterns:
        if skip_inorganic_comparison and patterns is _WATER_CONTEXT_INORGANIC_PATTERNS:
            continue
        if not re.search(pattern, text, flags=re.IGNORECASE):
            continue

        concentration = None
        leading = re.search(
            rf"(?P<value>{_WATER_CONTEXT_NUMBER})\s*(?P<unit>{_WATER_CONTEXT_UNIT})"
            rf"\s+(?:of\s+)?{pattern}",
            text,
            flags=re.IGNORECASE,
        )
        trailing = re.search(
            rf"{pattern}(?:\s+(?:matrix|solution|buffer))?\s*\(\s*"
            rf"(?P<value>{_WATER_CONTEXT_NUMBER})\s*(?P<unit>{_WATER_CONTEXT_UNIT})\s*\)",
            text,
            flags=re.IGNORECASE,
        )
        match = leading or trailing
        if match:
            concentration = (
                match.group("value"),
                _normalized_water_context_unit(match.group("unit")),
            )

        token = canonical
        if concentration:
            token = f"{canonical}_{concentration[0]}_{concentration[1]}"
        out.append(token)
    return out


def _water_type_raw_has_generic_organic_additive(value) -> bool:
    """Return whether the matrix label reports a bulk organic category."""
    if _blank(value):
        return False
    text = unicodedata.normalize("NFKC", str(value))
    return (
        not re.search(r"\borganic[-\s]?free\b", text, flags=re.IGNORECASE)
        and bool(_WATER_CONTEXT_GENERIC_ORGANIC_RE.search(text))
    )


def _water_type_raw_generic_organic_class(value) -> str:
    """Classify qualitative organic evidence without inventing a species."""
    if not _water_type_raw_has_generic_organic_additive(value):
        return ""
    text = unicodedata.normalize("NFKC", str(value))
    if re.search(r"\b(?:DOM|NOM|EfOM)\b|\bnatural\s+organic\s+matter\b", text, flags=re.IGNORECASE):
        return "natural organic matter"
    return "other organic"


def _merge_recovered_species(existing, recovered: list[str]) -> str:
    """Append recovered additive species without duplicating extracted aliases."""
    current = "" if _blank(existing) or str(existing).strip().casefold() == "none" else str(existing).strip()
    current_tokens = [part.strip() for part in current.split("&&") if part.strip()]
    existing_keys = {
        _recovery_species_key(spec.get("name", ""))
        for spec in _split_species_cell(current)
    }
    for token in recovered:
        specs = _split_species_cell(token)
        key = _recovery_species_key(specs[0].get("name", "")) if specs else _recovery_species_key(token)
        if key and key in existing_keys:
            continue
        current_tokens.append(token)
        if key:
            existing_keys.add(key)
    return "&&".join(current_tokens)


def recover_water_type_raw_context(df_in: pd.DataFrame) -> None:
    """Recover explicitly reported chemistry and pH from Water_type_raw.

    Recovery never replaces extracted additive or pH values.  It records only
    named chemical components; an inorganic component without a stated
    concentration is retained as a species but cannot yield ionic strength.
    """
    _ensure_columns(
        df_in,
        ["Water_type_raw", "Inorganic_matter", "Organic_matter", "pH", "normalization_trace"],
        default="",
    )
    for idx, row in df_in.iterrows():
        raw = row.get("Water_type_raw", "")
        if _blank(raw):
            continue
        text = unicodedata.normalize("NFKC", str(raw)).strip()
        inorganic = _water_context_species_with_concentrations(
            text, _WATER_CONTEXT_INORGANIC_PATTERNS
        )
        organic = [] if re.search(r"\borganic[-\s]?free\b", text, flags=re.IGNORECASE) else _water_context_species_with_concentrations(
            text, _WATER_CONTEXT_ORGANIC_PATTERNS
        )
        generic_organic = _water_type_raw_has_generic_organic_additive(text)

        recovered_parts: list[str] = []
        if inorganic:
            existing = row.get("Inorganic_matter", "")
            merged = _merge_recovered_species(existing, inorganic)
            if merged != ("" if _blank(existing) or str(existing).strip().casefold() == "none" else str(existing).strip()):
                df_in.at[idx, "Inorganic_matter"] = merged
                recovered_parts.append("inorganic=" + ", ".join(inorganic))
        if organic:
            existing = row.get("Organic_matter", "")
            merged = _merge_recovered_species(existing, organic)
            if merged != ("" if _blank(existing) or str(existing).strip().casefold() == "none" else str(existing).strip()):
                df_in.at[idx, "Organic_matter"] = merged
                recovered_parts.append("organic=" + ", ".join(organic))
        if generic_organic:
            recovered_parts.append("bulk organic category reported without a named species")

        ph_match = _WATER_CONTEXT_PH_RE.search(text)
        if ph_match and _blank(row.get("pH", "")):
            try:
                df_in.at[idx, "pH"] = float(ph_match.group("value").replace(",", ""))
                recovered_parts.append(f"pH={df_in.at[idx, 'pH']:g}")
            except ValueError:
                pass
        if recovered_parts:
            _append_text_cell(
                df_in,
                idx,
                "normalization_trace",
                "Recovered water context from Water_type_raw: " + "; ".join(recovered_parts),
            )


def _species_to_ions(name: str) -> list[tuple[str, int, float]]:
    key = _species_key(name)
    if key in CHEM_ION_LIBRARY:
        return CHEM_ION_LIBRARY[key]
    parsed = _parse_explicit_ion(name)
    if parsed:
        ion_name, charge = parsed
        return [(ion_name, charge, 1.0)]
    return []


def _species_molar_mass(name: str) -> float | None:
    return CHEM_MOLAR_MASS.get(_species_key(name))


def _species_equivalent_charge(name: str) -> int | None:
    ions = _species_to_ions(name)
    if len(ions) != 1:
        return None
    _ion_name, charge, _stoich = ions[0]
    return abs(int(charge))


def _join_preserve_blanks(values: list[Any]) -> str:
    if not values or all(_blank(v) for v in values):
        return ""
    return "&&".join("" if _blank(v) else str(v) for v in values)


def _append_text_cell(df_in: pd.DataFrame, idx, col: str, text: str) -> None:
    if not text:
        return
    _ensure_columns(df_in, [col], default="")
    current = "" if _blank(df_in.at[idx, col]) else str(df_in.at[idx, col]).strip()
    df_in.at[idx, col] = f"{current}; {text}" if current else text


def _expand_ions(specs: list[dict]) -> tuple[list[dict], list[str], list[str], list[str]]:
    totals: dict[tuple[str, int], float] = {}
    unknowns = []
    missing_molar = []
    ambiguous = []
    for spec in specs:
        molar = spec.get("molar")
        name = spec.get("name", "")
        if spec.get("ambiguous_concentration"):
            ambiguous.append(name)
            continue
        if molar is None:
            if _species_to_ions(name):
                missing_molar.append(name)
            else:
                unknowns.append(name)
            continue
        if molar <= 0:
            continue
        ions = _species_to_ions(name)
        if not ions:
            unknowns.append(name)
            continue
        for ion_name, charge, stoich in ions:
            key = (ion_name, charge)
            totals[key] = totals.get(key, 0.0) + molar * stoich
    ion_rows = [
        {"species": ion_name, "charge": charge, "molar": molar}
        for (ion_name, charge), molar in totals.items()
    ]
    return ion_rows, unknowns, missing_molar, ambiguous


def _ionic_strength_from_ions(ions: list[dict]) -> float | None:
    if not ions:
        return None
    return 0.5 * sum(float(ion["molar"]) * (int(ion["charge"]) ** 2) for ion in ions)


def _ion_charge_balance(ions: list[dict]) -> tuple[bool, float, float]:
    if not ions:
        return False, 0.0, 0.0
    net_charge = sum(float(ion["molar"]) * int(ion["charge"]) for ion in ions)
    total_charge = sum(abs(float(ion["molar"]) * int(ion["charge"])) for ion in ions)
    if total_charge <= 0:
        return False, net_charge, total_charge
    tolerance = max(1e-12, 0.05 * total_charge)
    return abs(net_charge) <= tolerance, net_charge, total_charge


def _complete_ionic_strength(ions: list[dict], unknowns: list[str], missing_molar: list[str], ambiguous: list[str]) -> float | None:
    if unknowns or missing_molar or ambiguous:
        return None
    balanced, _net_charge, _total_charge = _ion_charge_balance(ions)
    if not balanced:
        return None
    return _ionic_strength_from_ions(ions)


def _ion_species_labels(ions: list[dict]) -> list[str]:
    return sorted({str(ion.get("species", "")) for ion in ions if ion.get("species")})


def _raw_ionic_strength_to_mol_L(raw) -> tuple[float | None, str]:
    if _blank(raw):
        return None, ""
    text = str(raw).strip()
    match = _NUM_RE.search(text)
    if not match:
        return None, f"Ionic_strength raw value not numeric: {text}"
    unit = text[match.end():].strip().lstrip("_").strip()
    if not unit:
        return None, "Ionic_strength unit missing; raw value not converted"
    molar = _to_molar(match.group(0), unit)
    if molar is None:
        return None, f"Ionic_strength unit not recognized: {unit}"
    return molar, ""


def compute_ionic_strength(df_in: pd.DataFrame) -> None:
    out_I = "Ionic_strength_(mol/L)"
    base_cols = [
        out_I, "normalization_trace", "normalization_issues", "Ionic_strength", "Inorganic_matter",
        "Organic_matter",
    ]
    _ensure_columns(df_in, base_cols, default="")

    for i, row in df_in.iterrows():
        raw_I, raw_I_error = _raw_ionic_strength_to_mol_L(row.get("Ionic_strength", ""))
        added_inorg = _split_species_cell(str(row.get("Inorganic_matter", "")))
        added_org = _split_species_cell(str(row.get("Organic_matter", "")))


        inorg_filtered = [
            spec for spec in added_inorg
            if _species_key(spec.get("name", "")) not in PH_ADJUSTER_KEYS
        ]
        ignored = [
            spec.get("name", "")
            for spec in added_inorg
            if _species_key(spec.get("name", "")) in PH_ADJUSTER_KEYS
        ]

        added_ions, added_unknowns, added_missing_molar, added_ambiguous = _expand_ions(inorg_filtered)
        added_I = _complete_ionic_strength(added_ions, added_unknowns, added_missing_molar, added_ambiguous)

        total_ions, total_unknowns, total_missing_molar, total_ambiguous = _expand_ions(inorg_filtered)
        total_I = _complete_ionic_strength(total_ions, total_unknowns, total_missing_molar, total_ambiguous)
        if raw_I is not None:
            df_in.at[i, out_I] = _fmt_float(raw_I)
            _append_text_cell(df_in, i, "normalization_trace", "Ionic strength converted from extracted Ionic_strength")
            if total_I is not None and not math.isclose(raw_I, total_I, rel_tol=0.05, abs_tol=1e-12):
                _append_text_cell(
                    df_in,
                    i,
                    "normalization_issues",
                    "Ionic strength from parsed inorganic ions differs from extracted Ionic_strength: "
                    f"parsed={_fmt_float(total_I)} mol/L, extracted={_fmt_float(raw_I)} mol/L; kept extracted value",
                )
        elif total_I is not None:
            df_in.at[i, out_I] = _fmt_float(total_I)
            _append_text_cell(df_in, i, "normalization_trace", "Ionic strength from parsed inorganic ions: 0.5*sum(c*z^2)")
            if raw_I_error:
                _append_text_cell(df_in, i, "normalization_issues", raw_I_error)
        elif raw_I_error:
            _append_text_cell(df_in, i, "normalization_issues", raw_I_error)

        if ignored:
            _append_text_cell(
                df_in,
                i,
                "normalization_trace",
                "Strong acid/base pH-adjusters ignored for ionic strength: "
                + _join_preserve_blanks(sorted(set(ignored))),
            )

        unknowns = sorted(set([u for u in added_unknowns + total_unknowns if u]))
        if unknowns:
            _append_text_cell(
                df_in,
                i,
                "normalization_issues",
                "Ionic strength skipped unrecognized inorganic species: "
                + _join_preserve_blanks(unknowns),
            )

        ambiguous = sorted(set([u for u in added_ambiguous + total_ambiguous if u]))
        if ambiguous:
            _append_text_cell(
                df_in,
                i,
                "normalization_issues",
                "Ionic strength skipped ambiguous inorganic concentration: "
                + _join_preserve_blanks(ambiguous),
            )

        missing_molar = sorted(set([u for u in added_missing_molar + total_missing_molar if u]))
        if missing_molar:
            _append_text_cell(
                df_in,
                i,
                "normalization_issues",
                "Ionic strength missing molar concentration for inorganic species: "
                + _join_preserve_blanks(missing_molar),
            )

        if (
            total_ions
            and not (total_unknowns or total_missing_molar or total_ambiguous)
            and not _ion_charge_balance(total_ions)[0]
        ):
            _append_text_cell(
                df_in,
                i,
                "normalization_issues",
                "Ionic strength skipped charge-imbalanced inorganic ions: "
                + _join_preserve_blanks(_ion_species_labels(total_ions)),
            )


def _is_trace_methanol(entry) -> bool:
    nm = str(entry.get("name", "")).lower()
    if "methanol" not in nm:
        return False
    unit = str(entry.get("unit", "")).replace(" ", "").lower()
    val = entry.get("value", "")
    v = _coerce_number(val)
    return (unit in {"v/v%", "vv%"} and (v is not None) and v <= 1.0)


_ORG_PATTERNS = [
    r"\bsrfa\b", r"\bsrha\b", r"\bsrnom\d*\b", r"\bsrdom\d*\b", r"\bdom\b", r"\bnom\b",
    r"\befom\b", r"\bha\b", r"\bfa\b", r"\bppha\b", r"\bplfa\b", r"\bhumic\b", r"\bfulvic\b",
    r"\btannic\b", r"\bhepes\b",
    r"\borganic\s+compounds?\b", r"\bnatural\s+organic\s+matter\b",
    r"\bcoexisting\s*organic\s*matter\b", r"\bpreloaded\s*organic\s*matter\b"
]
_ORG_REGEX = re.compile("|".join(_ORG_PATTERNS), flags=re.I)

_INORG_PATTERNS = [
    r"\bsodium\s*chloride\b", r"\bnacl\b",
    r"\bcalcium\b", r"\bcalcium\s*chloride\b", r"\bca(?::?|\s*)\+?\+?\b",
    r"\bmagnesium\b", r"\bmg(?::?|\s*)\+?\+?\b",
    r"\bpotassium\b", r"\bk\b",
    r"\bsulfate\b", r"\bsulphate\b", r"\bso4\b",
    r"\bnitrate\b", r"\bno3\b",
    r"\bchloride\b",
    r"\bphosphate\b", r"\bpo4\b",
    r"\bcarbonate\b", r"\bbicarbonate\b",
    r"\bsodium\s*hydroxide\b", r"\bnaoh\b",
    r"\bammonium\b", r"\bnh4\b"
]
_INORG_REGEX = re.compile("|".join(_INORG_PATTERNS), flags=re.I)


def _has_organic(entries) -> bool:
    for e in entries:
        if _is_trace_methanol(e) or _is_zero_concentration_species(e):
            continue
        nm = re.sub(r"[_\-]+", " ", str(e.get("name", "")).lower())
        if _ORG_REGEX.search(nm):
            return True
    return False


def _has_inorganic(entries) -> bool:
    for e in entries:
        nm = re.sub(r"[_\-]+", " ", str(e.get("name", "")).lower())
        if _species_key(e.get("name", "")) in CHEM_ION_LIBRARY or _INORG_REGEX.search(nm):
            return True
    return False

_UNKNOWN_ADDITIVE_TOKENS = {
    "", "na", "n/a", "nan", "null", "not reported", "not provided",
    "value not provided in chunk",
}
_EXPLICIT_NO_ADDITIVE_TOKENS = {"none", "no", "not detected", "not applicable"}
_NO_ADDITIVE_TOKENS = _UNKNOWN_ADDITIVE_TOKENS | _EXPLICIT_NO_ADDITIVE_TOKENS


def _normalized_additive_status(value) -> str:
    if _blank(value):
        return ""
    text = unicodedata.normalize("NFKC", str(value)).strip().casefold()
    return text.removeprefix("[").removesuffix("]").strip()


def _additive_reporting_known(value) -> bool:
    """Return whether an additive field reports composition or explicit absence."""
    return _normalized_additive_status(value) not in _UNKNOWN_ADDITIVE_TOKENS


def _has_reported_additive(value) -> bool:
    return _normalized_additive_status(value) not in _NO_ADDITIVE_TOKENS


def _join_classes(classes: list[str]) -> str:
    return "; ".join(dict.fromkeys([c for c in classes if c]))


def _is_zero_concentration_species(spec: dict) -> bool:
    """Return whether a parsed additive explicitly reports a zero concentration."""
    value = _coerce_number(spec.get("value", ""))
    return value is not None and value == 0.0


def _effective_organic_species(value) -> list[dict]:
    """Keep named organic species except those explicitly reported at zero."""
    if not _has_reported_additive(value):
        return []
    return [
        spec
        for spec in _split_species_cell(str(value))
        if not _is_zero_concentration_species(spec)
    ]


def _organic_species_class(name: str) -> str:
    """Return the broad class for one effective organic additive species."""
    key = _species_key(name)
    text = re.sub(r"[_-]+", " ", _clean_species_label(name).casefold())

    if (
        key in _ORGANIC_RECOVERY_ALIAS_KEYS
        or re.fullmatch(r"(?:srnom|srdom)\d*", key)
        or key in {"nom", "dom", "efom", "om", "naturalorganicmatter"}
        or re.search(r"\b(?:humic|fulvic|tannic|natural organic matter)\b", text)
    ):
        # Humic, fulvic, and tannic material are all modeled as natural organic
        # matter because the extracted data do not support reliable subtypes.
        return "natural organic matter"
    if re.search(r"\b(?:toc|doc|cod)\b", text):
        return "organic carbon metric"
    if re.search(
        r"\b(?:methanol|ethanol|acetone|dioxane|dgbe|diisopropyl ether|"
        r"butanone|sec butyl alcohol|methyl 2 pentanone)\b",
        text,
    ):
        return "co-solvent"
    if re.search(r"\b(?:cmc|pam|ctac|surfactant|polymer)\b", text):
        return "surfactant/polymer"
    if re.search(r"\b(?:diesel|dro|tce|benzene|toluene|hydrocarbon)\b", text):
        return "hydrocarbon/organic contaminant"
    return "other organic"


def _organic_matter_class(species: list[dict]) -> str:
    """Classify effective organic additives into one model-facing broad class."""
    classes = {_organic_species_class(spec.get("name", "")) for spec in species}
    if not classes:
        return ""
    if classes <= {"organic carbon metric", "natural organic matter"}:
        return "natural organic matter" if "natural organic matter" in classes else "organic carbon metric"
    return next(iter(classes)) if len(classes) == 1 else "mixed organic additives"


_BULK_ORGANIC_DESCRIPTOR_KEYS = {
    "dom", "nom", "efom", "om", "organicmatter", "naturalorganicmatter",
    "organiccompounds", "organiccompound", "doc", "toc", "cod",
}


def _is_detailed_organic_species(spec: dict) -> bool:
    """Accept named organic constituents but reject bulk carbon/matter labels."""
    name = str(spec.get("name", "") or "").strip()
    if not name or _species_key(name) in _BULK_ORGANIC_DESCRIPTOR_KEYS:
        return False
    return _organic_species_class(name) in {
        "natural organic matter",
        "co-solvent",
        "surfactant/polymer",
        "hydrocarbon/organic contaminant",
    }


def _parsed_species_token(spec: dict) -> str:
    """Render a parsed species in the additive field's source-compatible form."""
    name = str(spec.get("name", "") or "").strip()
    value = str(spec.get("value", "") or "").strip()
    unit = str(spec.get("unit", "") or "").strip()
    return f"{name}_{value}_{unit}" if value and unit else name


def recover_named_organic_conditions(df_in: pd.DataFrame) -> None:
    """Recover named organic additives from Differentiating_Condition.

    This field often carries the controlled condition for an individual
    performance result.  Only identifiable organic additives are transferred;
    bulk labels such as DOM, NOM, EfOM, DOC, and TOC remain qualitative context
    rather than pseudo-species in Organic_matter.
    """
    _ensure_columns(
        df_in,
        ["Differentiating_Condition", "Organic_matter", "normalization_trace"],
        default="",
    )
    for idx, row in df_in.iterrows():
        condition = row.get("Differentiating_Condition", "")
        if _blank(condition):
            continue
        recovered = [
            _parsed_species_token(spec)
            for spec in _split_species_cell(str(condition))
            if _is_detailed_organic_species(spec)
        ]
        if not recovered:
            continue
        existing = row.get("Organic_matter", "")
        merged = _merge_recovered_species(existing, recovered)
        normalized_existing = "" if _blank(existing) or str(existing).strip().casefold() == "none" else str(existing).strip()
        if merged != normalized_existing:
            df_in.at[idx, "Organic_matter"] = merged
            _append_text_cell(
                df_in,
                idx,
                "normalization_trace",
                "Recovered named organic additive from Differentiating_Condition: "
                + ", ".join(recovered),
            )


def _non_ph_adjuster_inorganic_specs(value) -> list[dict]:
    return [
        spec for spec in _split_species_cell(str(value or ""))
        if _species_key(spec.get("name", "")) not in PH_ADJUSTER_KEYS
    ]


def _inorganic_feature_text(specs: list[dict]) -> str:
    return " ; ".join(str(spec.get("name", "")) for spec in specs if spec.get("name"))


def _inorganic_contains_flags(row: pd.Series, specs: list[dict]) -> dict[str, bool]:
    text = _inorganic_feature_text(specs)
    text_nfkc = unicodedata.normalize("NFKC", text)
    text_lower = text_nfkc.casefold().replace("_", " ").replace("-", " ")

    ion_labels = []
    for spec in specs:
        ion_labels.extend([ion for ion, _charge, _stoich in _species_to_ions(spec.get("name", ""))])
    ion_blob = " ".join(ion_labels)
    return {
        "contains_Na": bool(re.search(r"\bsodium\b|\bnacl\b|\bnahco3\b|\bna2|\bnah2|\bna\^?\+", text_lower))
            or any(ion.startswith("Na") for ion in ion_labels),
        "contains_K": bool(re.search(r"\bpotassium\b|\bkcl\b|\bkhco3\b|\bkh2|\bk2|\bk\^?\+", text_lower))
            or any(ion.startswith("K") for ion in ion_labels),
        "contains_Ca": bool(re.search(r"\bcalcium\b|\bcacl2\b|\bcaco3\b|\bca\^?2?\+", text_lower))
            or any(ion.startswith("Ca") for ion in ion_labels),
        "contains_Mg": bool(re.search(r"\bmagnesium\b|\bmgcl2\b|\bmgso4\b|\bmg\^?2?\+", text_lower))
            or any(ion.startswith("Mg") for ion in ion_labels),
        "contains_Cl": bool(re.search(r"\bchloride\b|\bnacl\b|\bkcl\b|\bcacl2\b|\bmgcl2\b|\bcl\^?-", text_lower))
            or any(ion.startswith("Cl") for ion in ion_labels),
        "contains_HCO3": bool(re.search(r"\bbicarbonate\b|\bhydrogen carbonate\b|\bhco3\b|\bnahco3\b|\bkhco3\b", text_lower))
            or "HCO3" in ion_blob,
        "contains_SO4": bool(re.search(r"\bsulfate\b|\bsulphate\b|\bso4\b|\bna2so4\b|\bmgso4\b", text_lower))
            or "SO4" in ion_blob,
        "contains_phosphate": bool(re.search(r"\bphosphate\b|\bpo4\b|\bhpo4\b|\bh2po4\b|\bnah2po4\b|\bna2hpo4\b|\bkh2po4\b|\bk2hpo4\b", text_lower))
            or any(any(tok in ion for tok in ("PO4", "HPO4", "H2PO4")) for ion in ion_labels),
    }


def materialize_additive_ml_features(df_in: pd.DataFrame) -> None:
    """
    Add compact ML-facing additive descriptors while keeping the raw additive
    strings and parsed chemistry audit columns available separately.

    For inorganic additives, strong acid/base pH adjusters are ignored here,
    matching the ionic-strength calculation convention. Ion indicators use
    three states: true when the ion is listed, false when a reported inorganic
    field omits it, and blank when the inorganic field itself is unreported.
    """
    _ensure_columns(
        df_in,
        ["Organic_matter", "Inorganic_matter"] + ADDITIVE_ML_FEATURE_COLUMNS,
        default="",
    )

    ion_flag_cols = [c for c in ADDITIVE_ML_FEATURE_COLUMNS if c.startswith("contains_")]
    df_in["organic_matter_present"] = False
    # These indicators are deliberately three-state: True, False, or blank when
    # the additive field itself is unreported.  A bare "" assignment gives the
    # column pandas 3's string dtype, which then rejects the booleans written
    # below, so the object dtype _ensure_columns established is restated here.
    for col in ("inorganic_matter_present", *ion_flag_cols):
        df_in[col] = pd.Series("", index=df_in.index, dtype="object")
    df_in["organic_matter_class"] = ""

    for i, row in df_in.iterrows():
        organic_raw = row.get("Organic_matter", "")
        generic_organic_from_water_type = _water_type_raw_has_generic_organic_additive(
            row.get("Water_type_raw", "")
        )
        organic_specs = _effective_organic_species(organic_raw)
        organic_present = bool(organic_specs) or generic_organic_from_water_type
        df_in.at[i, "organic_matter_present"] = organic_present
        if organic_present:
            df_in.at[i, "organic_matter_class"] = (
                _organic_matter_class(organic_specs)
                if organic_specs
                else _water_type_raw_generic_organic_class(row.get("Water_type_raw", ""))
            )

        inorganic_raw = row.get("Inorganic_matter", "")
        if not _additive_reporting_known(inorganic_raw):
            continue
        inorganic_specs = (
            _non_ph_adjuster_inorganic_specs(inorganic_raw)
            if _has_reported_additive(inorganic_raw)
            else []
        )
        inorganic_present = bool(inorganic_specs)
        df_in.at[i, "inorganic_matter_present"] = inorganic_present
        for col, present in _inorganic_contains_flags(row, inorganic_specs).items():
            df_in.at[i, col] = bool(present)


def _reported_background_additives(row: pd.Series) -> tuple[bool, bool]:
    organic_entries = _split_species_cells(row.get("Organic_matter", ""))
    inorganic_entries = [
        entry for entry in _split_species_cells(row.get("Inorganic_matter", ""))
        if _species_key(entry.get("name", "")) not in PH_ADJUSTER_KEYS
    ]
    return _has_organic(organic_entries), _has_inorganic(inorganic_entries)


def apply_synthetic_water_from_reported_additives(df_in: pd.DataFrame) -> None:
    """
    Align water-type cleanup with the enumerated class vocabulary.

    Only explicit non-PFAS background additives can promote a blank/ultrapure
    matrix to synthetic water. Reported TOC, DOC, or ionic strength alone may
    describe natural water background, so those values are not used as water-
    type evidence.
    """
    _ensure_columns(
        df_in,
        ["Water_type", "Organic_matter", "Inorganic_matter"],
        default="",
    )

    for i, row in df_in.iterrows():
        wt = _norm_water_key(row.get("Water_type", ""))
        if wt and wt != "ultrapure water":
            continue
        has_org, has_inorg = _reported_background_additives(row)
        if has_org or has_inorg:
            df_in.at[i, "Water_type"] = "synthetic water"


def _has_positive_water_quality(row: pd.Series) -> bool:
    for col in ("TOC_(mg/L)", "DOC_(mg/L)", "Ionic_strength", "Ionic_strength_(mol/L)"):
        value = _coerce_number(row.get(col, ""))
        if value is not None and value > 0:
            return True
    return False


def apply_default_water_context(df_in: pd.DataFrame) -> None:
    """
    Defaults for confirmed ultrapure water, plus blank water-type rows with no
    reported background chemistry.
    """
    _ensure_columns(
        df_in,
        [
            "Water_type",
            "pH",
            "TOC_(mg/L)",
            "DOC_(mg/L)",
            "Ionic_strength",
            "Ionic_strength_(mol/L)",
            "Inorganic_matter",
            "Organic_matter",
        ],
        default="",
    )

    for i, row in df_in.iterrows():
        wt = _norm_water_key(row.get("Water_type", ""))
        if not wt:
            has_org, has_inorg = _reported_background_additives(row)
            if has_org or has_inorg or _has_positive_water_quality(row):
                continue
            wt = "ultrapure water"
            df_in.at[i, "Water_type"] = wt
        if wt != "ultrapure water":
            continue

        needs_default = (
            _blank(df_in.at[i, "pH"]) or
            (
                _blank(df_in.at[i, "Ionic_strength"]) and
                _blank(df_in.at[i, "Ionic_strength_(mol/L)"])
            ) or
            _blank(df_in.at[i, "Inorganic_matter"]) or
            _blank(df_in.at[i, "Organic_matter"])
        )
        if not needs_default:
            continue
        if _blank(df_in.at[i, "pH"]):
            df_in.at[i, "pH"] = 7.0
        if _blank(df_in.at[i, "Ionic_strength"]) and _blank(df_in.at[i, "Ionic_strength_(mol/L)"]):
            df_in.at[i, "Ionic_strength"] = "0_mol/L"
        for col in ("Inorganic_matter", "Organic_matter"):
            if _blank(df_in.at[i, col]):
                df_in.at[i, col] = "none"

PFAS_PROPS_IDENTIFIER_COLUMNS = (
    "PFAS_Abbreviation",
    "PFAS_name",
    "Abbreviation",
    "Name",
    "Compound Full Name",
)


def _pfas_property_lookup_key(value) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except Exception:
        pass
    text = str(value).strip()
    if text.lower() in {"", "nan", "none", "na", "n/a", "null", "not reported"}:
        return ""
    return _norm_pfas_label(text)


def inject_pfas_properties(
    df_in: pd.DataFrame,
    pfas_props_path: str,
    pfas_props_sheet: str,
    enabled: bool = True,
) -> None:
    """
    Inject PFAS molecular weight and Second_Class from the authoritative
    catalog. Molecular weight supports unit standardization; Second_Class is
    retained as database-level provenance for coverage figures.

    pfas_features.xlsx contains classification, OPERA, RDKit descriptors, and
    fingerprints, but this standardizer only needs molecular weight for
    molar-to-mass concentration conversions.

    Expected:
      - pfas_props_path points to an Excel file with a column like
        'Abbreviation' or 'PFAS_name' and a molecular-weight column such as
        'rdkit_exact_mw' or 'Mw (g/mol)'.
      - df_in['PFAS_name'] has already been standardized.
    """
    if not enabled:
        return
    if not pfas_props_path:
        return
    if not os.path.exists(pfas_props_path):
        print(f"[WARN] PFAS properties file not found: {pfas_props_path}")
        return

    header = pd.read_excel(pfas_props_path, sheet_name=pfas_props_sheet, nrows=0)
    header_lookup = {
        str(col).strip() if isinstance(col, str) else col: col
        for col in header.columns
    }
    id_source_cols = [
        header_lookup[cand]
        for cand in PFAS_PROPS_IDENTIFIER_COLUMNS
        if cand in header_lookup
    ]
    mw_source_name = next(
        (cand for cand in PFAS_MW_SOURCE_COLUMNS if cand in header_lookup),
        None,
    )
    second_class_source_name = header_lookup.get(PFAS_SECOND_CLASS_OUTPUT_COLUMN)
    if mw_source_name is None:
        print(
            f"[WARN] PFAS properties file {pfas_props_path} has no molecular-weight "
            f"column; expected one of {list(PFAS_MW_SOURCE_COLUMNS)}. "
            "PFAS property injection skipped."
        )
        return

    source_columns = [header_lookup[mw_source_name]]
    if second_class_source_name is not None:
        source_columns.append(second_class_source_name)
    props = pd.read_excel(
        pfas_props_path,
        sheet_name=pfas_props_sheet,
        usecols=id_source_cols + source_columns,
    )
    props.columns = [str(c).strip() if isinstance(c, str) else c for c in props.columns]
    if props.empty:
        return

    id_cols = [cand for cand in PFAS_PROPS_IDENTIFIER_COLUMNS if cand in props.columns]
    if not id_cols:
        print(
            f"[WARN] PFAS properties file {pfas_props_path} has no "
            "PFAS identifier column like 'PFAS_Abbreviation' or 'PFAS_name'; "
            "PFAS property injection skipped."
        )
        return

    if "PFAS_name" not in df_in.columns:
        return

    props = props.copy()
    props[PFAS_MW_OUTPUT_COLUMN] = props[mw_source_name]
    prop_cols = [PFAS_MW_OUTPUT_COLUMN]
    if PFAS_SECOND_CLASS_OUTPUT_COLUMN in props.columns:
        prop_cols.append(PFAS_SECOND_CLASS_OUTPUT_COLUMN)

    lookup_rows = []
    for _, prop_row in props.iterrows():
        seen_keys = set()
        for id_col in id_cols:
            key = _pfas_property_lookup_key(prop_row.get(id_col))
            if not key or key in seen_keys:
                continue
            seen_keys.add(key)
            lookup_rows.append(
                {"__pfas_key": key, **{col: prop_row.get(col) for col in prop_cols}}
            )
    if not lookup_rows:
        return

    props_lookup = (
        pd.DataFrame(lookup_rows)
          .drop_duplicates("__pfas_key", keep="first")
          .set_index("__pfas_key")
    )

    keys = df_in["PFAS_name"].map(_pfas_property_lookup_key)
    aligned = props_lookup.reindex(keys)[prop_cols].reset_index(drop=True)
    aligned.index = df_in.index

    missing_cols = [c for c in prop_cols if c not in df_in.columns]
    existing_cols = [c for c in prop_cols if c in df_in.columns]

    if missing_cols:
        df_in[missing_cols] = aligned[missing_cols]

    for col in existing_cols:
        blank_mask = df_in[col].map(_blank)
        if blank_mask.any():
            df_in.loc[blank_mask, col] = aligned.loc[blank_mask, col]

def get_pfasmw(row):
    """
    Return PFAS molecular weight (g/mol) from the standardized row.
    """
    mw_cols = list(dict.fromkeys([PFAS_MW_OUTPUT_COLUMN, *PFAS_MW_SOURCE_COLUMNS]))
    for col in mw_cols:
        if col in row.index:
            mw = _coerce_number(row.get(col))
            if mw is not None:
                return mw
    return None


# Parse plain numeric scalars in ranges, uncertainties, and comma-separated lists.
# Commas are list delimiters; extracted scalar values must not use digit grouping.
_NUM_RE = re.compile(r"[-+]?(?:\d+(?:\.\d+)?|\.\d+)(?:[eE][-+]?\d+)?")
_SEP_RE = re.compile(r"\s*(?:–|-|—|to|±|\+/-|\+⁄-)\s*")


def _coerce_number(v):
    if _blank(v):
        return None
    if isinstance(v, (int, float)):
        return float(v)
    s = str(v).strip()
    full_num = _NUM_RE.fullmatch(s)
    if full_num:
        try:
            return float(full_num.group(0))
        except Exception:
            pass

    if "," in s or ";" in s:
        parts = [p.strip() for p in re.split(r"[;,]", s) if p.strip()]
        nums_list = []
        for p in parts:
            if _SEP_RE.search(p):
                nums_list = []
                break
            m = _NUM_RE.search(p)
            if m:
                try:
                    nums_list.append(float(m.group(0)))
                except Exception:
                    pass
        if len(nums_list) >= 2:
            return sum(nums_list) / len(nums_list)

    parts = _SEP_RE.split(s)
    nums = []
    for p in parts:
        m = _NUM_RE.search(p)
        if m:
            try:
                nums.append(float(m.group(0)))
            except Exception:
                pass
    if not nums:
        return None
    if len(nums) == 1:
        return nums[0]
    return (nums[0] + nums[1]) / 2.0

# --------- Multi-value C0 / D expansion for isotherms ----------

_RANGE_SEP_RE = re.compile(r"(?:–|-|—|to)", flags=re.I)
_RANGE_EXPR_RE = re.compile(
    rf"^\s*(?P<low>{_NUM_RE.pattern})\s*(?:–|-|—|to)\s*(?P<high>{_NUM_RE.pattern})\s*$",
    flags=re.I,
)


def _range_endpoints(low: float, high: float) -> list[float]:
    """Return the reported endpoints of a numeric range, in ascending order."""
    if low is None or high is None:
        return []
    if high < low:
        low, high = high, low
    return sorted({round(float(low), 12), round(float(high), 12)})


def _parse_list_or_range_cell(raw) -> list[float]:
    """
    Parse a cell that may contain:
      - a single number
      - a comma/semicolon-separated list of numbers
      - ranges like '0.01-1.00' or '0.01 to 1.00' (reported endpoints only)

    Returns a list of floats. If parsing fails, returns [].
    NOTE: this is specialized for PFAS_C0_value and Adsorbent_dosage_value,
    and avoids the averaging behavior of _coerce_number.
    """
    if _blank(raw):
        return []
    if isinstance(raw, (int, float)):
        try:
            return [float(raw)]
        except Exception:
            return []

    s = str(raw).strip()
    if not s:
        return []
    m_single = _NUM_RE.fullmatch(s)
    if m_single:
        try:
            return [float(m_single.group(0).replace(",", ""))]
        except Exception:
            return []

    # split on top-level commas / semicolons
    parts = [p.strip() for p in re.split(r"[;,]", s) if p.strip()]
    out_vals: list[float] = []

    for p in parts:
        # If it is exactly a numeric range, expand. This avoids mistaking the
        # exponent sign in values like 2.12e-6 for a range separator.
        m_range = _RANGE_EXPR_RE.match(p)
        if m_range:
            try:
                low = float(m_range.group("low").replace(",", ""))
                high = float(m_range.group("high").replace(",", ""))
                out_vals.extend(_range_endpoints(low, high))
                continue
            except Exception:
                # If anything goes wrong, fall through to single-value parsing
                pass

        # Otherwise, treat as a single value, ignoring '±' style uncertainty
        m = _NUM_RE.search(p)
        if m:
            try:
                out_vals.append(float(m.group(0).replace(",", "")))
            except Exception:
                continue

    # unique + sorted
    out_vals = sorted({round(float(v), 12) for v in out_vals})
    return out_vals


def _drop_nonpositive_expanded_values(values: list[float]) -> list[float]:
    """
    Range/list expansion is used to create modelable isotherm operating points.
    Generated C0 or dosage values at/below zero are not physically usable for
    Kd conversion, so skip them instead of creating guaranteed-blocked rows.
    """
    if len(values) <= 1:
        return values
    return [v for v in values if v is not None and v > 0]


def _append_review_text(existing, message: str) -> str:
    text = "" if _blank(existing) else str(existing).strip()
    if not text:
        return message
    if message in text:
        return text
    return f"{text}; {message}"


ISOTHERM_EXPANDED_INHERITED_KD_FLAG_COLUMN = (
    "Kd_value_inherited_from_isotherm_expansion_flag"
)


def explode_isotherm_c0_d(df_in: pd.DataFrame) -> pd.DataFrame:
    """
    For rows that clearly correspond to isotherms (Langmuir/Freundlich present),
    split records when PFAS_C0_value and/or Adsorbent_dosage_value contain
    lists or ranges. A reported range contributes only its two endpoints. If
    both axes are multi-valued, keep the source row and route it to review
    rather than inventing every C0 x dosage combination.

    Example:
      PFAS_C0_value = '0.01,0.05,0.10,0.50,1.00' mg/L
      Adsorbent_dosage_value = 10 mg/L
    becomes 5 rows with C0 = 0.01, 0.05, 0.10, 0.50, 1.00 respectively
    (same dosage and isotherm parameters).
    """
    c0_col = "PFAS_C0_value"
    d_col = "Adsorbent_dosage_value"
    if c0_col not in df_in.columns and d_col not in df_in.columns:
        return df_in

    _ensure_columns(df_in, ["require_review", "review_reason", "warning_message"], default="")
    _ensure_columns(df_in, [ISOTHERM_EXPANDED_INHERITED_KD_FLAG_COLUMN], default=False)
    for col in (c0_col, d_col):
        if col in df_in.columns:
            df_in[col] = df_in[col].astype("object")

    rows = []
    for _, row in df_in.iterrows():
        # Only split rows that have isotherm information
        has_iso = False
        for iso_col in (
            "Langmuir_Qm_value", "Langmuir_KL_value",
            "Freundlich_KF_value", "Freundlich_n_or_1/n_verbatim",
        ):
            if iso_col in df_in.columns and _coerce_number(row.get(iso_col)) is not None:
                has_iso = True
                break
        if not has_iso:
            rows.append(row)
            continue

        raw_c0 = row.get(c0_col, None)
        raw_d = row.get(d_col, None)

        c0_vals = _parse_list_or_range_cell(raw_c0) if c0_col in df_in.columns else []
        d_vals = _parse_list_or_range_cell(raw_d) if d_col in df_in.columns else []
        c0_vals = _drop_nonpositive_expanded_values(c0_vals)
        d_vals = _drop_nonpositive_expanded_values(d_vals)

        # If neither side is multi-valued, keep the row as-is
        if len(c0_vals) <= 1 and len(d_vals) <= 1:
            rows.append(row)
            continue

        if len(c0_vals) > 1 and len(d_vals) > 1:
            new_row = row.astype("object").copy()
            msg = (
                "ambiguous isotherm design: both PFAS_C0 and adsorbent dosage "
                "are multi-valued; range/list expansion suppressed to avoid "
                "inventing C0 x dosage combinations"
            )
            new_row["require_review"] = "TRUE"
            new_row["review_reason"] = _append_review_text(new_row.get("review_reason"), msg)
            new_row["warning_message"] = _append_review_text(new_row.get("warning_message"), msg)
            rows.append(new_row)
            continue

        if not c0_vals:
            c0_vals = [None]
        if not d_vals:
            d_vals = [None]

        has_inherited_reported_kd = any(
            _coerce_number(row.get(column)) is not None
            for column in ("Kd_value", "Kd_value_L/g")
        )
        for c0 in c0_vals:
            for d in d_vals:
                new_row = row.astype("object").copy()
                new_row[ISOTHERM_EXPANDED_INHERITED_KD_FLAG_COLUMN] = bool(
                    has_inherited_reported_kd
                )
                if c0 is not None and c0_col in df_in.columns:
                    new_row[c0_col] = c0
                if d is not None and d_col in df_in.columns:
                    new_row[d_col] = d
                rows.append(new_row)

    return pd.DataFrame(rows, columns=df_in.columns)

# --------- Snapshot range/list context handling ----------

RANGE_TRACE_COLUMNS = [
    "PFAS_C0_raw",
    "PFAS_C0_context",
    "Adsorbent_dosage_raw",
    "Adsorbent_dosage_context",
    "Contact_time_raw_(h)",
    "Contact_time_context_(h)",
    "pH_raw",
    "pH_context",
    "Solution_volume_raw_(mL)",
    "Solution_volume_context_(mL)",
    "Temperature_raw_(°C)",
    "Mixing_speed_raw_(rpm)",
    "TOC_raw_(mg/L)",
    "DOC_raw_(mg/L)",
    "TDS_raw_(mg/L)",
    "average_pore_diameter_raw_(angstrom)",
    "phpzc_raw",
]

PH_SINGLE_CONDITION_MAX_WIDTH = 1.0
TEMPERATURE_SINGLE_CONDITION_MAX_WIDTH_C = 5.0


def _raw_with_optional_unit(value, unit_value="") -> str:
    if _blank(value):
        return ""
    unit = "" if _blank(unit_value) else str(unit_value).strip()
    return f"{str(value).strip()} {unit}".strip()


def preserve_raw_reported_range_fields(df_in: pd.DataFrame) -> None:
    """
    Preserve raw reported values before standardization overwrites or blanks
    the ML-ready field.

    These raw/context fields are intended for Standardized_Full only. They are
    not included in Equilibrium_Data.
    """
    specs = [
        ("PFAS_C0_value", "PFAS_C0_raw", "PFAS_C0_unit"),
        ("Adsorbent_dosage_value", "Adsorbent_dosage_raw", "Adsorbent_dosage_unit"),
        ("Contact_time_(h)", "Contact_time_raw_(h)", ""),
        ("pH", "pH_raw", ""),
        ("Solution_volume_(mL)", "Solution_volume_raw_(mL)", ""),
        ("Temperature_(°C)", "Temperature_raw_(°C)", ""),
        ("Mixing_speed_(rpm)", "Mixing_speed_raw_(rpm)", ""),
        ("TOC_(mg/L)", "TOC_raw_(mg/L)", ""),
        ("DOC_(mg/L)", "DOC_raw_(mg/L)", ""),
        ("TDS_(mg/L)", "TDS_raw_(mg/L)", ""),
    ]

    _ensure_columns(df_in, [raw_col for _src, raw_col, _unit in specs], default="")

    for idx, row in df_in.iterrows():
        for src_col, raw_col, unit_col in specs:
            if src_col not in df_in.columns:
                continue
            if not _blank(row.get(raw_col, "")):
                continue
            unit_val = row.get(unit_col, "") if unit_col and unit_col in df_in.columns else ""
            raw_text = _raw_with_optional_unit(row.get(src_col, ""), unit_val)
            if raw_text:
                df_in.at[idx, raw_col] = raw_text


# Some extracted measurements encode a fixed denominator in the unit field,
# e.g. ``15_mL/241_mL`` becomes value ``15`` and unit ``mL/241_mL`` after
# merge. The numeric denominator belongs in the value, yielding ``15 / 241``
# with unit ``mL/mL``. This also applies to mass-based forms such as
# ``10_mg/100_mL`` -> ``0.1 mg/mL``.
_FIXED_DENOMINATOR_UNIT_RE = re.compile(
    r"^\s*"
    r"(?P<numerator_unit>[A-Za-z\u00b5\u03bc][A-Za-z0-9\u00b5\u03bc^().*\-]*)"
    r"\s*/\s*"
    r"(?P<denominator>[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)"
    r"\s*_?\s*"
    r"(?P<denominator_unit>[A-Za-z\u00b5\u03bc][A-Za-z0-9\u00b5\u03bc^().*\-]*)"
    r"\s*$"
)
_NUMERIC_EXPRESSION_TOKEN_RE = re.compile(
    r"(?P<prefix>[<>\u2264\u2265~]?\s*)"
    r"(?P<number>[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)"
)
_AND_JOINED_UNIT_RE = re.compile(
    r"^\s*_?and_?\s*"
    r"(?P<values>[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"
    r"(?:\s*_?and_?\s*[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)*?)"
    r"\s*[_ ]\s*(?P<unit>[A-Za-z0-9\u00b5\u03bc/().\-]+)\s*$",
    flags=re.IGNORECASE,
)
_AND_JOINED_VALUE_RE = re.compile(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?")


def normalize_and_joined_value_units(df_in: pd.DataFrame) -> None:
    """Repair legacy ``value``/``and_value_unit`` splits from raw merges.

    Earlier merges split a source expression such as
    ``PFNA_8.5e4_and_1.7e5_ng/L`` into ``8.5e4`` and
    ``and_1.7e5_ng/L``. Restore it as the comma-separated list
    ``8.5e4,1.7e5`` with unit ``ng/L`` so isotherm expansion can retain each
    reported concentration.
    """
    unit_columns = [
        col for col in df_in.columns
        if isinstance(col, str) and col.endswith("_unit")
    ]
    for unit_col in unit_columns:
        value_col = f"{unit_col[:-5]}_value"
        if value_col not in df_in.columns:
            continue

        df_in[value_col] = df_in[value_col].astype("object")
        df_in[unit_col] = df_in[unit_col].astype("object")
        for idx, unit_raw in df_in[unit_col].items():
            if _blank(unit_raw) or _blank(df_in.at[idx, value_col]):
                continue
            match = _AND_JOINED_UNIT_RE.match(str(unit_raw))
            if not match:
                continue
            trailing_values = _AND_JOINED_VALUE_RE.findall(match.group("values"))
            if not trailing_values:
                continue
            df_in.at[idx, value_col] = (
                f"{str(df_in.at[idx, value_col]).strip()},"
                + ",".join(trailing_values)
            )
            df_in.at[idx, unit_col] = match.group("unit").replace("\u03bc", "\u00b5")


def _scale_numeric_expression_by_denominator(value, denominator: float) -> str | None:
    """Divide each number in a scalar/range/list expression by a fixed basis."""
    if _blank(value) or denominator == 0:
        return None

    text = str(value).strip()
    if not text:
        return None

    converted = 0

    def replace(match: re.Match) -> str:
        nonlocal converted
        try:
            numeric = float(match.group("number"))
        except (TypeError, ValueError):
            return match.group(0)
        converted += 1
        return f"{match.group('prefix')}{numeric / denominator:.12g}"

    # Scale each explicitly separated list item before preserving its delimiters.
    if "," in text or ";" in text:
        result = "".join(
            segment if segment in {",", ";"}
            else _NUMERIC_EXPRESSION_TOKEN_RE.sub(replace, segment)
            for segment in re.split(r"([,;])", text)
        )
    else:
        result = _NUMERIC_EXPRESSION_TOKEN_RE.sub(replace, text)
    return result if converted else None


def normalize_fixed_denominator_value_units(df_in: pd.DataFrame) -> None:
    """Normalize explicit fixed denominators in aligned ``*_value``/``*_unit`` fields.

    A unit such as ``mg/100_mL`` means the paired reported value is expressed
    per 100 mL, not per one mL. Converting it to ``mg/mL`` and dividing every
    scalar, range endpoint, list item, or uncertainty term by 100 produces a
    standard value/unit pair without changing the recorded raw text.
    """
    unit_columns = [
        col for col in df_in.columns
        if isinstance(col, str) and col.endswith("_unit")
    ]
    for unit_col in unit_columns:
        value_col = f"{unit_col[:-5]}_value"
        if value_col not in df_in.columns:
            continue

        df_in[value_col] = df_in[value_col].astype("object")
        df_in[unit_col] = df_in[unit_col].astype("object")
        for idx, unit_raw in df_in[unit_col].items():
            if _blank(unit_raw):
                continue
            match = _FIXED_DENOMINATOR_UNIT_RE.match(str(unit_raw))
            if not match:
                continue
            try:
                denominator = float(match.group("denominator"))
            except (TypeError, ValueError):
                continue
            if denominator == 0:
                continue

            value_scaled = _scale_numeric_expression_by_denominator(
                df_in.at[idx, value_col], denominator
            )
            if value_scaled is None:
                continue
            df_in.at[idx, value_col] = value_scaled
            df_in.at[idx, unit_col] = (
                f"{match.group('numerator_unit')}/{match.group('denominator_unit')}"
            )

def _range_list_info(raw) -> dict:
    """
    Classify a reported numeric cell without using _coerce_number's averaging.

    Returns a compact dict with:
      kind: single | range | list | uncertainty | none
      values: numeric values/endpoints in the reported order
      center: representative center only when one is scientifically meaningful
    """
    if _blank(raw):
        return {"kind": "none", "values": [], "center": None}
    if isinstance(raw, (int, float)) and not isinstance(raw, bool):
        try:
            return {"kind": "single", "values": [float(raw)], "center": float(raw)}
        except Exception:
            return {"kind": "none", "values": [], "center": None}

    text = unicodedata.normalize("NFKC", str(raw)).strip()
    if not text:
        return {"kind": "none", "values": [], "center": None}

    # Explicit uncertainty: use the reported center, not the average of center
    # and uncertainty width. Example: 25 ± 2 -> center 25.
    if re.search(r"(?:±|\+/-|\+⁄-)", text):
        nums = []
        for m in _NUM_RE.finditer(text):
            try:
                nums.append(float(m.group(0)))
            except Exception:
                pass
        if nums:
            return {"kind": "uncertainty", "values": nums, "center": nums[0]}

    # Commas, semicolons, and && identify explicit lists.
    if "&&" in text or ";" in text or "," in text:
        parts = [p.strip() for p in re.split(r"&&|[;,]", text) if p.strip()]
        vals = []
        for part in parts:
            m_range = _RANGE_EXPR_RE.match(part)
            if m_range:
                for name in ("low", "high"):
                    try:
                        vals.append(float(m_range.group(name)))
                    except Exception:
                        pass
                continue
            m = _NUM_RE.search(part)
            if m:
                try:
                    vals.append(float(m.group(0)))
                except Exception:
                    pass
        if len(vals) >= 2:
            return {"kind": "list", "values": vals, "center": None}
        if len(vals) == 1:
            return {"kind": "single", "values": vals, "center": vals[0]}

    # Explicit range. This avoids treating exponent signs as range separators.
    m_range = _RANGE_EXPR_RE.match(text)
    if m_range:
        try:
            low = float(m_range.group("low"))
            high = float(m_range.group("high"))
            return {"kind": "range", "values": [low, high], "center": (low + high) / 2.0}
        except Exception:
            pass

    m = _NUM_RE.fullmatch(text)
    if m:
        try:
            val = float(m.group(0))
            return {"kind": "single", "values": [val], "center": val}
        except Exception:
            pass

    # Last resort: if a text cell contains exactly one number plus words/units,
    # treat it as a single numeric report; otherwise it is not machine-resolved.
    nums = []
    for m in _NUM_RE.finditer(text):
        try:
            nums.append(float(m.group(0)))
        except Exception:
            pass
    if len(nums) == 1:
        return {"kind": "single", "values": nums, "center": nums[0]}
    if len(nums) >= 2 and _RANGE_SEP_RE.search(text):
        return {"kind": "range", "values": nums[:2], "center": sum(nums[:2]) / 2.0}
    return {"kind": "none", "values": [], "center": None}


WATER_QUALITY_SCALAR_SPECS = (
    ("TOC_(mg/L)", "TOC_raw_(mg/L)", "mg/L"),
    ("DOC_(mg/L)", "DOC_raw_(mg/L)", "mg/L"),
    ("TDS_(mg/L)", "TDS_raw_(mg/L)", "mg/L"),
)

ADSORBENT_SCALAR_SPECS = (
    (
        "average_pore_diameter_(angstrom)",
        "average_pore_diameter_raw_(angstrom)",
        "angstrom",
    ),
    ("phpzc", "phpzc_raw", "pH"),
)

# Treat close measurements as repeated characterizations of one water matrix.
# This accepts the observed 2.24-2.52 mg/L interval (11.8% relative spread)
# while keeping materially different conditions such as 3.1, 9.7, and 12 mg/L
# out of a scalar model-facing field.
MAX_WATER_QUALITY_RELATIVE_SPREAD = 0.15
_WATER_QUALITY_SINGLE_NUMBER_RE = re.compile(
    r"^[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?$"
)
# DOC values may be reported as detection-limit scalars, such as ``<0.3``.
# Preserve the source expression in ``DOC_raw_(mg/L)`` and use the numeric
# component in the normalized scalar field, matching grain-size treatment.
_DOC_CENSORED_SINGLE_NUMBER_RE = re.compile(
    r"^<=?\s*(?P<value>[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)$"
)


def _water_quality_expression(raw) -> str:
    """Normalize range delimiters used in already-unitized mg/L fields."""
    text = unicodedata.normalize("NFKC", str(raw)).strip()
    return re.sub(
        r"(?<=\d)\s*[~\uff5e]\s*(?=[+-]?(?:\d|\.\d))",
        " to ",
        text,
    )


def _normalize_coherent_scalar_fields(
    df_in: pd.DataFrame,
    specs: tuple[tuple[str, str, str], ...],
    *,
    property_kind: str,
    manual_range_average_overrides: dict[tuple[str, str, str, str], str] | None = None,
    manual_adsorbent_scalar_overrides: dict[
        tuple[str, str, str, str, str], tuple[str, float]
    ]
    | None = None,
) -> None:
    """Average close numeric reports and retain broad reports as raw context."""
    _ensure_columns(df_in, ["normalization_trace", "normalization_issues"], default="")

    manual_range_average_overrides = manual_range_average_overrides or {}
    manual_adsorbent_scalar_overrides = manual_adsorbent_scalar_overrides or {}

    for source_column, raw_column, unit_label in specs:
        if source_column not in df_in.columns:
            continue
        if raw_column not in df_in.columns:
            df_in[raw_column] = df_in[source_column].astype("object").copy()
        df_in[source_column] = df_in[source_column].astype("object")

        for idx, raw in df_in[raw_column].items():
            if _blank(raw):
                continue
            row = df_in.loc[idx]

            text = _water_quality_expression(raw)
            adsorbent_manual_key = normalized_adsorbent_scalar_normalization_key(
                row.get("extraction_dataset", ""),
                row.get("study_no", ""),
                row.get("adsorbent_id", ""),
                row.get("Water_type", ""),
                source_column,
            )
            approved_adsorbent_scalar = manual_adsorbent_scalar_overrides.get(
                adsorbent_manual_key
            )
            if (
                approved_adsorbent_scalar is not None
                and approved_adsorbent_scalar[0] == normalized_expected_raw(raw)
            ):
                df_in.at[idx, source_column] = approved_adsorbent_scalar[1]
                _append_text_cell(
                    df_in,
                    idx,
                    "normalization_trace",
                    f"{source_column}: manually approved scalar selected",
                )
                continue

            info = _range_list_info(text)
            kind = info.get("kind")
            values = [
                float(value)
                for value in (info.get("values") or [])
                if math.isfinite(float(value))
            ]
            differentiating_condition = row.get("Differentiating_Condition", "")
            if _blank(differentiating_condition):
                # Keep legacy pre-normalized inputs compatible with the
                # canonical extraction-schema spelling used by merged data.
                differentiating_condition = row.get(
                    "differentiating_condition", ""
                )
            manual_key = normalized_performance_normalization_key(
                row.get("extraction_dataset", ""),
                row.get("study_no", ""),
                differentiating_condition,
                source_column,
            )
            approved_expected_raw = manual_range_average_overrides.get(manual_key)

            plain_scalar_match = _WATER_QUALITY_SINGLE_NUMBER_RE.fullmatch(text)
            doc_censored_match = (
                _DOC_CENSORED_SINGLE_NUMBER_RE.fullmatch(text)
                if source_column == "DOC_(mg/L)"
                else None
            )
            if kind == "single" and (plain_scalar_match or doc_censored_match):
                value = float(
                    doc_censored_match.group("value")
                    if doc_censored_match
                    else text.replace(",", "")
                )
                if value >= 0:
                    df_in.at[idx, source_column] = value
                    if doc_censored_match:
                        _append_text_cell(
                            df_in,
                            idx,
                            "normalization_trace",
                            f"{source_column}: censored DOC scalar normalized using reported numeric value",
                        )
                    continue

            if kind == "uncertainty" and info.get("center") is not None:
                center = float(info["center"])
                if math.isfinite(center) and center >= 0:
                    df_in.at[idx, source_column] = center
                    _append_text_cell(
                        df_in,
                        idx,
                        "normalization_trace",
                        f"{source_column}: uncertainty expression normalized using reported center",
                    )
                    continue

            if kind in {"range", "list"} and len(values) >= 2 and all(
                value >= 0 for value in values
            ):
                mean = sum(values) / len(values)
                spread = max(values) - min(values)
                if approved_expected_raw == normalized_expected_raw(raw):
                    df_in.at[idx, source_column] = mean
                    _append_text_cell(
                        df_in,
                        idx,
                        "normalization_trace",
                        f"{source_column}: manually approved reported range average in {unit_label}",
                    )
                    continue
                if math.isclose(mean, 0.0, abs_tol=1e-12):
                    relative_spread = (
                        0.0
                        if math.isclose(spread, 0.0, abs_tol=1e-12)
                        else math.inf
                    )
                else:
                    relative_spread = spread / abs(mean)

                if relative_spread <= MAX_WATER_QUALITY_RELATIVE_SPREAD:
                    df_in.at[idx, source_column] = mean
                    _append_text_cell(
                        df_in,
                        idx,
                        "normalization_trace",
                        f"{source_column}: coherent range/list averaged in {unit_label}",
                    )
                    continue

                df_in.at[idx, source_column] = ""
                _append_text_cell(
                    df_in,
                    idx,
                    "normalization_issues",
                    f"{source_column}: inconsistent range/list; normalized scalar blanked",
                )
                continue

            df_in.at[idx, source_column] = ""
            _append_text_cell(
                df_in,
                idx,
                "normalization_issues",
                f"{source_column}: non-scalar or invalid {property_kind}; normalized scalar blanked",
            )


def normalize_water_quality_scalar_fields(
    df_in: pd.DataFrame,
    *,
    manual_range_average_overrides: dict[tuple[str, str, str, str], str] | None = None,
) -> None:
    """Normalize TOC, DOC, and TDS to defensible scalar mg/L values.

    Close ranges/lists are averaged. DOC detection-limit-style scalars use
    their reported numeric value. Broad or malformed expressions are blanked
    in the model-facing field while the reported text remains available in
    ``*_raw_(mg/L)`` on ``Standardized_Full``.
    """
    _normalize_coherent_scalar_fields(
        df_in,
        WATER_QUALITY_SCALAR_SPECS,
        property_kind="concentration",
        manual_range_average_overrides=manual_range_average_overrides,
    )


ORGANIC_CARBON_COLUMN = "organic_carbon_mg/L"


def _water_quality_was_reported(
    row: pd.Series,
    normalized_column: str,
    raw_column: str,
) -> bool:
    """Use the preserved raw report when available to enforce source priority."""
    if raw_column in row.index:
        return not _blank(row.get(raw_column, ""))
    return not _blank(row.get(normalized_column, ""))


def materialize_organic_carbon(df_in: pd.DataFrame) -> None:
    """Consolidate DOC and TOC into one model-facing organic-carbon field.

    Priority is strictly: reported DOC, reported TOC, confirmed/inferred
    ultrapure water (zero), otherwise missing. A preferred raw report that
    cannot be normalized remains missing rather than falling through to the
    lower-priority source. The separate normalized and raw DOC/TOC fields are
    retained for Standardized_Full traceability; the model-facing projection
    controls which of them reach Equilibrium_Data.
    """
    _ensure_columns(
        df_in,
        ["Water_type", "DOC_(mg/L)", "TOC_(mg/L)", ORGANIC_CARBON_COLUMN],
        default="",
    )
    values: list[float | str] = []
    for _, row in df_in.iterrows():
        doc_reported = _water_quality_was_reported(
            row,
            "DOC_(mg/L)",
            "DOC_raw_(mg/L)",
        )
        toc_reported = _water_quality_was_reported(
            row,
            "TOC_(mg/L)",
            "TOC_raw_(mg/L)",
        )
        doc = _coerce_number(row.get("DOC_(mg/L)", ""))
        toc = _coerce_number(row.get("TOC_(mg/L)", ""))

        if doc_reported:
            values.append(float(doc) if doc is not None else "")
        elif toc_reported:
            values.append(float(toc) if toc is not None else "")
        elif _norm_water_key(row.get("Water_type", "")) == "ultrapure water":
            values.append(0.0)
        else:
            values.append("")

    df_in[ORGANIC_CARBON_COLUMN] = pd.Series(values, index=df_in.index, dtype="object")


def normalize_adsorbent_scalar_fields(
    df_in: pd.DataFrame,
    *,
    manual_adsorbent_scalar_overrides: dict[
        tuple[str, str, str, str, str], tuple[str, float]
    ]
    | None = None,
) -> None:
    """Normalize scalar adsorbent properties with the coherent-value policy.

    This runs after adsorbent-property enrichment so directly extracted and
    injected properties receive identical close-list/range handling.
    """
    _normalize_coherent_scalar_fields(
        df_in,
        ADSORBENT_SCALAR_SPECS,
        property_kind="adsorbent property",
        manual_adsorbent_scalar_overrides=manual_adsorbent_scalar_overrides,
    )


def _format_context_range(raw, unit_raw="") -> str:
    info = _range_list_info(raw)
    vals = info.get("values") or []
    unit = str(unit_raw or "").strip()
    if info.get("kind") == "range" and len(vals) >= 2:
        text = f"[{_fmt_float(min(vals[0], vals[1]))}, {_fmt_float(max(vals[0], vals[1]))}]"
    elif info.get("kind") == "list" and vals:
        text = "[" + ", ".join(_fmt_float(v) for v in vals) + "]"
    else:
        text = str(raw).strip()
    return f"{text} {unit}".strip()


def _is_snapshot_test_mode(row) -> bool:
    text = str(row.get("Test_mode", "")).strip().lower()
    return text.startswith("snapshot")


def _convert_scalar_pfas_c0_or_dosage_to_mg_L(pname: str, scalar_value: float, row) -> float | None:
    """
    Convert a scalar PFAS_C0 or Adsorbent_dosage value to mg/L.

    Used only to correct explicit uncertainty reports such as 10 ± 1 mg/L,
    where _coerce_number would otherwise average the center and the uncertainty.
    Range/list cells are not converted here; their standardized value is blanked.
    """
    from normalization_units import (
        PARAM_SPECS,
        _conc_to_mg_per_L,
        _fixed_volume_conc_to_mg_per_L,
        compute_factor_against_signature,
        get_solution_volume_L,
        normalize_unit,
    )

    spec = PARAM_SPECS.get(pname)
    if not spec:
        return None
    unit_col = spec["unit_col"]
    u_norm = normalize_unit(row.get(unit_col, "")) if unit_col in row.index else ""
    if not u_norm:
        return None

    vol_L = get_solution_volume_L(row)
    if u_norm in {"ng", "µg", "ug", "mg", "g", "kg"}:
        if not vol_L:
            return None
        tok = "µg" if u_norm == "ug" else u_norm
        mg_factor = {"ng": 1e-6, "µg": 1e-3, "mg": 1.0, "g": 1e3, "kg": 1e6}.get(tok)
        if mg_factor is None:
            return None
        return float(scalar_value) * mg_factor / vol_L

    fixed = _fixed_volume_conc_to_mg_per_L(scalar_value, u_norm)
    if fixed is not None:
        return fixed

    mw = get_pfasmw(row) if pname == "PFAS_C0" else None
    conc_factor = _conc_to_mg_per_L(u_norm, mw)
    if conc_factor is not None:
        return float(scalar_value) * conc_factor

    u_fixed = (
        u_norm.replace("mL", "ml").replace("L", "l")
              .replace("µL", "µl")
              .replace("ug", "µg").replace("umol", "µmol")
    )
    if u_fixed.count("/") >= 2 and "·" not in u_fixed:
        parts = u_fixed.split("/")
        u_fixed = parts[0] + "/(" + "·".join(parts[1:]) + ")"
    factor, _why = compute_factor_against_signature(
        u_fixed,
        spec["dim"],
        spec["bases"],
        mw,
    )
    if factor is None:
        return None
    return float(scalar_value) * factor

def materialize_general_range_context(df_in: pd.DataFrame) -> None:
    """
    Resolve range/list handling that is not snapshot-specific.
    Trace fields are retained in Standardized_Full only.

    General policy:
      - narrow pH range -> midpoint in pH
      - broad pH range/list -> blank pH + pH_context
      - narrow temperature range -> midpoint
      - broad temperature range/list -> blank temperature; raw text remains available
      - solution volume/contact time/mixing speed ranges -> raw trace, not midpoint
      - explicit uncertainty expressions such as 25 ± 2 -> use center value

    This runs before unit conversion so unresolved solution-volume ranges are
    not used indirectly as midpoint volumes for mass-to-concentration conversion.
    """
    _ensure_columns(df_in, list(RANGE_TRACE_COLUMNS) + ["normalization_trace"], default="")
    mutable_condition_columns = [
        "pH",
        "Temperature_(°C)",
        "Contact_time_(h)",
        "Solution_volume_(mL)",
        "Mixing_speed_(rpm)",
    ]
    for col in mutable_condition_columns:
        if col in df_in.columns:
            df_in[col] = df_in[col].astype("object")

    for idx, row in df_in.iterrows():
        if "pH" in df_in.columns:
            info = _range_list_info(row.get("pH", ""))
            kind = info.get("kind")
            vals = info.get("values") or []
            if kind == "uncertainty" and info.get("center") is not None:
                df_in.at[idx, "pH"] = info["center"]
            elif kind in {"range", "list"} and vals:
                width = max(vals) - min(vals)
                if width <= PH_SINGLE_CONDITION_MAX_WIDTH:
                    df_in.at[idx, "pH"] = (max(vals) + min(vals)) / 2.0
                else:
                    df_in.at[idx, "pH_context"] = _format_context_range(row.get("pH", ""))
                    df_in.at[idx, "pH"] = ""
                    _append_text_cell(
                        df_in,
                        idx,
                        "normalization_trace",
                        "Broad pH range/list stored in pH_context; pH single-value field left blank",
                    )
        # Temperature: small room-temperature/control-window ranges are accepted.
        temp_col = "Temperature_(°C)"
        if temp_col in df_in.columns:
            info = _range_list_info(row.get(temp_col, ""))
            kind = info.get("kind")
            vals = info.get("values") or []
            if kind == "uncertainty" and info.get("center") is not None:
                df_in.at[idx, temp_col] = info["center"]
            elif kind in {"range", "list"} and vals:
                width = max(vals) - min(vals)
                if width <= TEMPERATURE_SINGLE_CONDITION_MAX_WIDTH_C:
                    df_in.at[idx, temp_col] = (max(vals) + min(vals)) / 2.0
                else:
                    df_in.at[idx, temp_col] = ""
                    _append_text_cell(
                        df_in,
                        idx,
                        "normalization_trace",
                        "Broad Temperature range/list retained in Temperature_raw_(°C); single-value field left blank",
                    )

        # Contact time: range/list is not a single endpoint time.
        ct_col = "Contact_time_(h)"
        if ct_col in df_in.columns:
            info = _range_list_info(row.get(ct_col, ""))
            kind = info.get("kind")
            if kind == "uncertainty" and info.get("center") is not None:
                df_in.at[idx, ct_col] = info["center"]
            elif kind in {"range", "list"}:
                df_in.at[idx, "Contact_time_context_(h)"] = _format_context_range(row.get(ct_col, ""))
                df_in.at[idx, ct_col] = ""
                _append_text_cell(
                    df_in,
                    idx,
                    "normalization_trace",
                    "Contact_time range/list stored in Contact_time_context_(h); single-value field left blank",
                )

        # Solution volume: protocol context, not a model-ready fixed condition.

        sv_col = "Solution_volume_(mL)"
        if sv_col in df_in.columns:
            info = _range_list_info(row.get(sv_col, ""))
            kind = info.get("kind")
            if kind == "uncertainty" and info.get("center") is not None:
                df_in.at[idx, sv_col] = info["center"]
            elif kind in {"range", "list"}:
                df_in.at[idx, "Solution_volume_context_(mL)"] = _format_context_range(row.get(sv_col, ""))
                df_in.at[idx, sv_col] = ""
                _append_text_cell(
                    df_in,
                    idx,
                    "normalization_trace",
                    "Solution_volume range/list stored in Solution_volume_context_(mL); single-value field left blank",
                )


        # Mixing speed: range/list is treated as unresolved operating context.
        rpm_col = "Mixing_speed_(rpm)"
        if rpm_col in df_in.columns:
            info = _range_list_info(row.get(rpm_col, ""))
            kind = info.get("kind")
            if kind == "uncertainty" and info.get("center") is not None:
                df_in.at[idx, rpm_col] = info["center"]
            elif kind in {"range", "list"}:
                df_in.at[idx, rpm_col] = ""
                _append_text_cell(
                    df_in,
                    idx,
                    "normalization_trace",
                    "Mixing_speed range/list retained in Mixing_speed_raw_(rpm); single-value field left blank",
                )

def materialize_snapshot_range_context(df_in: pd.DataFrame) -> None:
    """
    Preserve unresolved snapshot ranges/lists as context while keeping the
    standardized value columns as the only ML-ready single-condition fields.

    Strict snapshot design variables:
      - PFAS_C0_value
      - Adsorbent_dosage_value
      - Contact_time_(h)

    For these, a range/list means the endpoint is not mapped to one fixed
    condition. The raw value is preserved, a compact context column is filled,
    and the standardized single-value field is left blank.

    General pH, contact-time, solution-volume, temperature, and mixing-speed
    range handling is performed separately by
    materialize_general_range_context().
    """
    context_cols = list(RANGE_TRACE_COLUMNS) + ["normalization_trace"]
    _ensure_columns(df_in, context_cols, default="")

    strict_specs = [
        ("PFAS_C0", "PFAS_C0_value", "PFAS_C0_unit", "PFAS_C0_value_mg/L", "PFAS_C0_context"),
        ("Adsorbent_dosage", "Adsorbent_dosage_value", "Adsorbent_dosage_unit", "Adsorbent_dosage_value_mg/L", "Adsorbent_dosage_context"),
    ]
    for _pname, raw_col, _unit_col, out_col, context_col in strict_specs:
        for col in (raw_col, out_col, context_col):
            if col in df_in.columns:
                df_in[col] = df_in[col].astype("object")

    for idx, row in df_in.iterrows():
        if not _is_snapshot_test_mode(row):
            continue

        for pname, raw_col, unit_col, out_col, context_col in strict_specs:
            if raw_col not in df_in.columns:
                continue
            info = _range_list_info(row.get(raw_col, ""))
            kind = info.get("kind")
            if kind in {"range", "list"}:
                df_in.at[idx, context_col] = _format_context_range(row.get(raw_col, ""), row.get(unit_col, ""))
                if out_col in df_in.columns:
                    df_in.at[idx, out_col] = ""
                _append_text_cell(
                    df_in,
                    idx,
                    "normalization_trace",
                    f"Snapshot {pname} range/list stored in {context_col}; standardized single-value field left blank",
                )
            elif kind == "uncertainty" and info.get("center") is not None and out_col in df_in.columns:
                converted = _convert_scalar_pfas_c0_or_dosage_to_mg_L(pname, info["center"], row)
                if converted is not None:
                    df_in.at[idx, out_col] = converted
                    _append_text_cell(
                        df_in,
                        idx,
                        "normalization_trace",
                        f"Snapshot {pname} uncertainty expression converted using reported center value",
                    )

# ---------- Pore volume helpers ----------


def coalesce_renamed_input_columns(df_in: pd.DataFrame) -> None:
    rename_aliases = {
        "Inorganic_matter": ("Added_inorganic_matter",),
        "Organic_matter": ("Added_organic_matter",),
        # Preserve any legacy normalization diagnostics while emitting the
        # broader canonical name in all new output.
        "normalization_issues": ("unit_errors",),
        # Accept legacy merged workbooks while emitting only the canonical
        # Unicode degree-sign header in all new output.
        "Temperature_(°C)": ("Temperature_(Â°C)",),
        "Nominal_grain_size_value": ("particle_size_value",),
        "Nominal_grain_size_unit": ("particle_size_unit",),
        "Nominal_grain_size_value_mm": ("particle_size_value_mm",),
    }
    legacy_to_drop = []
    for new_col, old_cols in rename_aliases.items():
        present_old_cols = [col for col in old_cols if col in df_in.columns]
        if new_col not in df_in.columns and present_old_cols:
            df_in[new_col] = ""
        if new_col in df_in.columns:
            df_in[new_col] = df_in[new_col].astype("object")
            for old_col in present_old_cols:
                for idx, old_val in df_in[old_col].items():
                    if _blank(df_in.at[idx, new_col]) and not _blank(old_val):
                        df_in.at[idx, new_col] = old_val
                    elif new_col == "normalization_issues" and not _blank(old_val):
                        existing = str(df_in.at[idx, new_col]).strip()
                        legacy = str(old_val).strip()
                        existing_parts = {
                            part.strip() for part in existing.split(";") if part.strip()
                        }
                        if legacy and legacy not in existing_parts:
                            _append_text_cell(df_in, idx, new_col, legacy)
        legacy_to_drop.extend(present_old_cols)
    if legacy_to_drop:
        df_in.drop(columns=sorted(set(legacy_to_drop)), inplace=True, errors="ignore")
