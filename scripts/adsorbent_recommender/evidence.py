"""Database-first matching for normalized 25 mg/L adsorption evidence.

The matcher deliberately separates identity gates from condition tolerances.
PFAS, adsorbent, and standardized water type must match exactly after the
project's identifier normalization. Numeric conditions may vary only within a
documented tolerance. The selected database row is chosen by evidence quality
and condition distance, never by the magnitude of logKd.
"""

from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from adsorbent_recommender.catalog import clean_text, normalized_key


SCRIPT_DIR = Path(__file__).resolve().parent
AD_ROOT = SCRIPT_DIR.parents[1]
DEFAULT_EVIDENCE_PATH = (
    AD_ROOT / "output" / "Merge" / "pfas_adsorption_25mgL_search.xlsx"
)
DEFAULT_EVIDENCE_SHEET = "Dose_25mgL_Search"
EVIDENCE_SNAPSHOT_FILENAME = "reference_dose_evidence.csv"

# Concentrations and time span orders of magnitude, so multiplicative
# tolerances are more meaningful than a single absolute difference. The policy
# is exported with every bundle and returned in each recommendation payload.
DEFAULT_MATCH_POLICY: dict[str, Any] = {
    "version": 1,
    "reference_dose_mg_L": 25.0,
    "hard_exact_matches": ["PFAS", "adsorbent", "standardized water type"],
    "numeric_tolerances": {
        "initial_concentration_mg_L": {
            "kind": "log10_absolute",
            "tolerance": math.log10(2.0),
            "description": "within a factor of 2",
        },
        "pH": {"kind": "absolute", "tolerance": 0.5, "description": "±0.5 pH unit"},
        "temperature_C": {"kind": "absolute", "tolerance": 2.0, "description": "±2 °C"},
        "organic_carbon_mg_L": {
            "kind": "log10_absolute",
            "tolerance": math.log10(2.0),
            "description": "within a factor of 2",
        },
        "tds_mg_L": {
            "kind": "log10_absolute",
            "tolerance": math.log10(2.0),
            "description": "within a factor of 2",
        },
        "ionic_strength_mol_L": {
            "kind": "log10_absolute",
            "tolerance": math.log10(2.0),
            "description": "within a factor of 2",
        },
    },
    "selection_order": [
        "isotherm-recalculated result before near-dose observation",
        "smallest normalized condition distance",
        "highest qualifying isotherm R²",
        "stable standardized row identifier",
    ],
}

NUMERIC_FIELDS: dict[str, dict[str, Any]] = {
    "initial_concentration_mg_L": {
        "columns": ("PFAS_C0_value_mg/L",),
        "scenario_keys": ("initial_concentration_mg_l",),
    },
    "pH": {"columns": ("pH",), "water_keys": ("pH", "ph")},
    "temperature_C": {
        "columns": (
            "Temperature_(°C)",
            "Temperature_(Â°C)",
            "Temperature_(Ã‚Â°C)",
            "Temperature_(Ãƒâ€šÃ‚Â°C)",
        ),
        "scenario_keys": ("temperature_c",),
    },
    "organic_carbon_mg_L": {
        "columns": ("organic_carbon_mg/L",),
        "water_keys": ("organic_carbon_mg_l", "organic_carbon_mg_L", "doc_mg_l", "toc_mg_l"),
    },
    "tds_mg_L": {
        "columns": ("TDS_(mg/L)",),
        "water_keys": ("tds_mg_l", "TDS_mg_L"),
    },
    "ionic_strength_mol_L": {
        "columns": ("Ionic_strength_(mol/L)",),
        "water_keys": ("ionic_strength_mol_l", "ionic_strength_mol_L"),
    },
}

REQUIRED_EVIDENCE_COLUMNS = {
    "PFAS_name",
    "Water_type",
    "search_logKd_log10(L/g)",
    "search_dosage_mg/L",
}


def _first_present(mapping: dict[str, Any], keys: Iterable[str]) -> Any:
    for key in keys:
        value = mapping.get(key)
        if clean_text(value):
            return value
    return None


def _first_column(frame: pd.DataFrame, columns: Iterable[str]) -> str | None:
    return next((column for column in columns if column in frame.columns), None)


def _as_float(value: Any) -> float | None:
    number = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(number) or not math.isfinite(float(number)):
        return None
    return float(number)


def _truthy(value: Any) -> bool:
    return clean_text(value).casefold() in {"1", "1.0", "true", "yes", "y"}


def load_reference_dose_evidence(
    path: Path | None = None,
    sheet_name: str = DEFAULT_EVIDENCE_SHEET,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Load a standalone workbook or CSV and return an auditable status record."""
    configured = os.environ.get("ADSORBENT_EVIDENCE_PATH")
    source = Path(path or configured or DEFAULT_EVIDENCE_PATH).expanduser().resolve()
    info: dict[str, Any] = {
        "available": False,
        "source_path": str(source),
        "source_sheet": sheet_name,
        "rows": 0,
    }
    if not source.exists():
        info["reason"] = "reference-dose evidence file was not found"
        return pd.DataFrame(), info
    try:
        if source.suffix.casefold() == ".csv":
            frame = pd.read_csv(source)
        else:
            frame = pd.read_excel(source, sheet_name=sheet_name)
        missing = sorted(REQUIRED_EVIDENCE_COLUMNS - set(frame.columns))
        if missing:
            raise ValueError(f"missing required columns: {', '.join(missing)}")
    except Exception as exc:
        info["reason"] = f"{type(exc).__name__}: {exc}"
        return pd.DataFrame(), info
    info.update({"available": True, "rows": int(len(frame))})
    return frame, info


def requested_conditions(
    scenario: dict[str, Any],
    normalized_water: dict[str, Any],
    screening_conditions: dict[str, Any],
) -> dict[str, float]:
    """Return numeric constraints explicitly supplied for this request."""
    water_input = scenario.get("water", {}) if isinstance(scenario.get("water"), dict) else {}
    requested: dict[str, float] = {}
    for label, spec in NUMERIC_FIELDS.items():
        raw = _first_present(scenario, spec.get("scenario_keys", ()))
        if raw is None:
            water_keys = spec.get("water_keys", ())
            if _first_present(water_input, water_keys) is not None:
                raw = _first_present(normalized_water, spec["columns"] + tuple(water_keys))
        number = _as_float(raw)
        if number is not None:
            requested[label] = number

    return requested


def _candidate_keys(material: dict[str, Any]) -> set[str]:
    fields = (
        "Product_key",
        "Name_Commercial",
        "model_adsorbent_id",
        "Name_abbreviation",
        "Name_full",
    )
    return {normalized_key(material.get(field)) for field in fields if normalized_key(material.get(field))}


def _pfas_keys(feature_row: dict[str, Any], cache_key: str | None) -> set[str]:
    fields = ("PFAS_name", "Compound Full Name", "Name", "Abbreviation")
    values = [cache_key, *(feature_row.get(field) for field in fields)]
    return {normalized_key(value) for value in values if normalized_key(value)}


def _row_alias_keys(row: pd.Series, fields: Iterable[str]) -> set[str]:
    return {normalized_key(row.get(field)) for field in fields if normalized_key(row.get(field))}


def _numeric_distance(observed: float, requested: float, rule: dict[str, Any]) -> float | None:
    tolerance = float(rule["tolerance"])
    if rule["kind"] == "absolute":
        delta = abs(observed - requested)
    elif rule["kind"] == "log10_absolute":
        if observed <= 0 or requested <= 0:
            return None
        delta = abs(math.log10(observed) - math.log10(requested))
    else:
        raise ValueError(f"Unsupported numeric tolerance kind: {rule['kind']}")
    return delta / tolerance if tolerance > 0 else (0.0 if delta == 0 else None)


def _python_value(value: Any) -> Any:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    if isinstance(value, np.generic):
        return value.item()
    return value


def _doi_url(value: Any) -> str | None:
    doi = clean_text(value)
    if not doi:
        return None
    if doi.casefold().startswith(("http://", "https://")):
        return doi
    doi = doi.removeprefix("doi:").strip()
    return f"https://doi.org/{doi}" if doi else None


def _citation(row: pd.Series) -> dict[str, Any] | None:
    url = _doi_url(row.get("DOI"))
    title = clean_text(row.get("article_title"))
    authors = clean_text(row.get("authors"))
    year = clean_text(row.get("publication_year"))
    journal = clean_text(row.get("source_title"))
    if not any((url, title, authors, journal)):
        return None
    label_parts = [part for part in (authors, f"({year})" if year else "", title, journal) if part]
    return {"label": ". ".join(label_parts), "url": url, "doi": clean_text(row.get("DOI")) or None}


def _evidence_record(row: pd.Series, requested: dict[str, float]) -> dict[str, Any]:
    conditions: dict[str, Any] = {}
    for label, spec in NUMERIC_FIELDS.items():
        column = next((name for name in spec["columns"] if name in row.index), None)
        if column is not None:
            conditions[label] = _python_value(row.get(column))
    citation = _citation(row)
    adsorbent_name = next(
        (
            clean_text(row.get(column))
            for column in ("name_commercial", "name_abbreviation", "name_full")
            if clean_text(row.get(column))
        ),
        None,
    )
    return {
        "standardized_row_id": _python_value(row.get("standardized_row_id")),
        "performance_id": _python_value(row.get("performance_id")),
        "study_no": _python_value(row.get("study_no")),
        "PFAS": _python_value(row.get("PFAS_name")),
        "adsorbent_id": _python_value(row.get("adsorbent_id")),
        "adsorbent_name": adsorbent_name,
        "water_type": _python_value(row.get("Water_type")),
        "logKd_log10_L_g": _python_value(row.get("search_logKd_log10(L/g)")),
        "Kd_L_g": _python_value(row.get("search_Kd_L/g")),
        "dose_mg_L": _python_value(row.get("search_dosage_mg/L")),
        "reference_dose_target_mg_L": _python_value(row.get("reference_dosage_target_mg/L")),
        "reference_dose_basis": _python_value(row.get("reference_dose_basis")),
        "method": _python_value(row.get("reference_dose_method")),
        "is_exact_at_reference_dose": bool(_truthy(row.get("reference_dose_exact_flag"))),
        "isotherm_R2": _python_value(row.get("reference_isotherm_R2")),
        "conditions": conditions,
        "requested_conditions": requested,
        "normalized_condition_distance": _python_value(row.get("__match_distance")),
        "citation": citation,
    }


def match_candidate_evidence(
    evidence: pd.DataFrame,
    *,
    material: dict[str, Any],
    pfas_feature_row: dict[str, Any],
    pfas_cache_key: str | None,
    scenario: dict[str, Any],
    normalized_water: dict[str, Any],
    screening_conditions: dict[str, Any],
    policy: dict[str, Any] | None = None,
    max_records: int = 5,
) -> dict[str, Any]:
    """Find comparable rows for one catalog product under the request conditions."""
    policy = policy or DEFAULT_MATCH_POLICY
    product_keys = _candidate_keys(material)
    pfas_keys = _pfas_keys(pfas_feature_row, pfas_cache_key)
    water_key = normalized_key(normalized_water.get("Water_type"))
    missing_gates = []
    if not product_keys:
        missing_gates.append("adsorbent identity")
    if not pfas_keys:
        missing_gates.append("PFAS identity")
    if not water_key:
        missing_gates.append("water type")
    if evidence.empty:
        return {"status": "no_match", "reason": "reference-dose database is unavailable", "missing_exact_gates": missing_gates}
    if missing_gates:
        return {
            "status": "no_match",
            "reason": f"exact match requires {', '.join(missing_gates)}",
            "missing_exact_gates": missing_gates,
        }

    candidates = evidence.copy()
    adsorbent_fields = ("adsorbent_id", "name_abbreviation", "name_commercial", "name_full")
    candidates = candidates[
        candidates.apply(lambda row: bool(product_keys & _row_alias_keys(row, adsorbent_fields)), axis=1)
    ]
    candidates = candidates[candidates["PFAS_name"].map(normalized_key).isin(pfas_keys)]
    candidates = candidates[candidates["Water_type"].map(normalized_key).eq(water_key)]
    if candidates.empty:
        return {
            "status": "no_match",
            "reason": "no row passed the exact PFAS, adsorbent, and water-type gates",
            "missing_exact_gates": [],
        }

    requested = requested_conditions(scenario, normalized_water, screening_conditions)
    distances = pd.Series(0.0, index=candidates.index)
    compared_fields = 0
    for label, requested_value in requested.items():
        spec = NUMERIC_FIELDS[label]
        column = _first_column(candidates, spec["columns"])
        if column is None:
            candidates = candidates.iloc[0:0]
            break
        rule = policy["numeric_tolerances"][label]
        normalized = candidates[column].map(
            lambda value: (
                _numeric_distance(observed, requested_value, rule)
                if (observed := _as_float(value)) is not None
                else None
            )
        )
        keep = normalized.notna() & normalized.le(1.0 + 1e-12)
        candidates = candidates[keep]
        distances = distances.loc[candidates.index] + normalized.loc[candidates.index].astype(float)
        compared_fields += 1
        if candidates.empty:
            break

    water_input = scenario.get("water", {}) if isinstance(scenario.get("water"), dict) else {}
    required_true = []
    ions = water_input.get("ions", {}) if isinstance(water_input.get("ions"), dict) else {}
    required_true.extend(f"contains_{ion}" for ion, value in ions.items() if bool(value))
    if clean_text(water_input.get("inorganic_matter") or water_input.get("inorganic_additives")):
        required_true.append("inorganic_matter_present")
    if clean_text(water_input.get("organic_matter") or water_input.get("organic_additives")):
        required_true.append("organic_matter_present")
    for column in required_true:
        if column not in candidates.columns:
            candidates = candidates.iloc[0:0]
            break
        candidates = candidates[candidates[column].map(_truthy)]

    if candidates.empty:
        tolerance_text = {
            label: policy["numeric_tolerances"][label]["description"]
            for label in requested
        }
        return {
            "status": "no_match",
            "reason": "exact identities matched, but detailed conditions were outside tolerance or unreported",
            "missing_exact_gates": [],
            "requested_conditions": requested,
            "numeric_tolerances": tolerance_text,
        }

    candidates = candidates.copy()
    candidates["__match_distance"] = distances.loc[candidates.index] / max(1, compared_fields)
    exact = candidates.get("reference_dose_exact_flag", pd.Series(False, index=candidates.index)).map(_truthy)
    r2 = pd.to_numeric(candidates.get("reference_isotherm_R2", pd.Series(np.nan, index=candidates.index)), errors="coerce").fillna(-np.inf)
    candidates["__not_exact"] = ~exact
    candidates["__negative_r2"] = -r2
    candidates["__stable_id"] = candidates.get("standardized_row_id", pd.Series("", index=candidates.index)).map(clean_text)
    candidates = candidates.sort_values(
        ["__not_exact", "__match_distance", "__negative_r2", "__stable_id"],
        kind="stable",
    )
    primary = candidates.iloc[0]
    records = [_evidence_record(row, requested) for _, row in candidates.head(max_records).iterrows()]
    citations = []
    seen_citations: set[tuple[str | None, str]] = set()
    for _, citation_row in candidates.iterrows():
        citation = _citation(citation_row)
        if citation is None:
            continue
        key = (citation.get("url"), citation.get("label", ""))
        if key not in seen_citations:
            citations.append(citation)
            seen_citations.add(key)
        if len(citations) >= 10:
            break
    exact_mask = candidates.get(
        "reference_dose_exact_flag", pd.Series(False, index=candidates.index)
    ).map(_truthy)
    score_pool = candidates[exact_mask].copy() if exact_mask.any() else candidates.copy()
    score_pool["__logkd"] = pd.to_numeric(
        score_pool["search_logKd_log10(L/g)"], errors="coerce"
    )
    score_pool = score_pool[score_pool["__logkd"].notna()]
    if score_pool.empty:
        return {
            "status": "no_match",
            "reason": "matched rows do not contain a valid normalized logKd value",
            "missing_exact_gates": [],
        }
    score_pool["__study_key"] = score_pool.apply(
        lambda row: clean_text(row.get("DOI"))
        or clean_text(row.get("study_no"))
        or clean_text(row.get("standardized_row_id")),
        axis=1,
    )
    per_study = score_pool.groupby("__study_key", dropna=False)["__logkd"].median()
    database_logkd = float(per_study.median())
    values = pd.to_numeric(candidates["search_logKd_log10(L/g)"], errors="coerce").dropna()
    performance = {
        "source": "database_match",
        "mean_logKd": database_logkd,
        "Kd_L_g": float(10**database_logkd),
        "dose_mg_L": float(policy.get("reference_dose_mg_L", 25.0)),
        "reference_dose_target_mg_L": float(policy.get("reference_dose_mg_L", 25.0)),
        "basis": "database_matched_aggregate",
        "method": "study_balanced_median",
        "match_count": int(len(candidates)),
        "study_count": int(len(per_study)),
        "aggregation": "median within each study, then median across studies; exact 25 mg/L results preferred",
        "exact_reference_dose_match_count": int(exact_mask.sum()),
        "matched_logKd_min": float(values.min()) if not values.empty else None,
        "matched_logKd_max": float(values.max()) if not values.empty else None,
    }
    return {
        "status": "matched",
        "performance": performance,
        "primary_record": records[0],
        "records": records,
        "citations": citations,
        "match_policy": policy,
        "selection_note": "Primary evidence was selected without using logKd magnitude.",
    }
