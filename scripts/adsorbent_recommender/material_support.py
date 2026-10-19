"""Fitted evidence behind a catalog product, separate from property coverage."""
from __future__ import annotations

import re
import hashlib
from pathlib import Path
from typing import Any

import pandas as pd

from adsorbent_recommender.catalog import clean_text, normalized_key

SUPPORT_FILENAME = "adsorbent_training_support.csv"
SUPPORT_VERSION = 1
DEFAULT_POLICY = {
    "version": SUPPORT_VERSION,
    "minimum_independent_studies": 2,
    "minimum_source_records": 5,
    "study_identity": "distinct normalized DOI; unverified study IDs do not satisfy the independent-study floor",
    "source_identity": "distinct extraction_dataset/source_row_index; expanded siblings count once",
    "basis": "screening policy, not a calibrated accuracy or confidence threshold",
}
RECORD_ALIASES = ("adsorbent_id", "name_abbreviation", "name_commercial", "name_full")
PRODUCT_ALIASES = ("Product_key", "Name_Commercial", "model_adsorbent_id", "Name_abbreviation", "Name_full")
COUNT_COLUMNS = ("training_rows", "source_records", "study_ids", "independent_studies", "pfas_count", "missing_source_identifiers")


def doi_key(value: Any) -> str:
    text = clean_text(value).casefold().strip()
    text = re.sub(r"^(?:https?://(?:dx\.)?doi\.org/|doi:\s*)", "", text)
    return text if re.fullmatch(r"10\.\d{4,9}/\S+", text) else ""


def policy_for(metadata: dict[str, Any]) -> dict[str, Any]:
    policy = {**DEFAULT_POLICY, **metadata.get("policy", {})}
    for name in ("minimum_independent_studies", "minimum_source_records"):
        value = policy[name]
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"Adsorbent support policy {name} must be a positive integer.")
    return policy


def enrich_studies(records: pd.DataFrame, references: pd.DataFrame | None = None) -> pd.DataFrame:
    """Resolve study IDs to one DOI, refusing conflicting bibliography entries."""
    result = records.copy()
    if "study_no" not in result:
        result["study_no"] = ""
    result["DOI"] = result.get("DOI", pd.Series("", index=result.index)).map(doi_key)
    frames = [result]
    if references is not None and {"study_no", "DOI"}.issubset(references):
        frames.append(references[["study_no", "DOI"]].copy())
    registry = pd.concat([frame[["study_no", "DOI"]] for frame in frames if "study_no" in frame], ignore_index=True)
    registry["study_no"] = registry.study_no.map(clean_text).astype("string")
    registry["DOI"] = registry.DOI.map(doi_key)
    registry = registry[registry.study_no.ne("") & registry.DOI.ne("")].drop_duplicates()
    conflicts = set(registry.groupby("study_no").DOI.nunique().loc[lambda values: values.gt(1)].index)
    mapping = registry[~registry.study_no.isin(conflicts)].set_index("study_no").DOI.to_dict()
    study = result.get("study_no", pd.Series("", index=result.index)).map(clean_text).astype("string")
    result["DOI"] = study.map(mapping).fillna("")
    result["study_identity_conflict"] = study.isin(conflicts)
    return result


def support_table(catalog: pd.DataFrame, records: pd.DataFrame, references: pd.DataFrame | None = None) -> pd.DataFrame:
    """Count exact curated product matches from actual fitted records only."""
    records = enrich_studies(records, references)
    aliases = records.reindex(columns=RECORD_ALIASES).apply(lambda column: column.map(normalized_key))
    rows = []
    for _, product in catalog.iterrows():
        keys = {normalized_key(product.get(field)) for field in PRODUCT_ALIASES} - {""}
        hits = records[aliases.isin(keys).any(axis=1)].copy()
        study = hits.get("study_no", pd.Series("", index=hits.index)).map(clean_text).astype("string")
        hits["__study"] = hits.DOI.where(hits.DOI.ne(""), "unverified:" + study)
        hits.loc[study.eq("") & hits.DOI.eq(""), "__study"] = ""
        source = hits.get("source_row_index", pd.Series("", index=hits.index)).map(clean_text).astype("string")
        dataset = hits.get("extraction_dataset", pd.Series("", index=hits.index)).map(clean_text).astype("string")
        hits["__source"] = (dataset + "|" + source).where(source.ne(""), "")
        sources = hits[hits.__source.ne("")].drop_duplicates("__source")
        per_study = sources[sources.__study.ne("")].groupby("__study").size()
        rows.append({
            "Product_key": clean_text(product.Product_key), "training_rows": len(hits),
            "source_records": len(sources), "study_ids": study[study.ne("")].nunique(),
            "independent_studies": hits.loc[hits.DOI.ne(""), "DOI"].nunique(),
            "pfas_count": hits.get("PFAS_name", pd.Series(dtype=str)).map(normalized_key).replace("", pd.NA).nunique(),
            "missing_source_identifiers": int(source.eq("").sum()),
            "study_identity_conflicts": int(hits.study_identity_conflict.sum()),
            "largest_study_source_fraction": float(per_study.max() / len(sources)) if len(sources) and len(per_study) else None,
            "matched_training_aliases": "; ".join(sorted(set(aliases.loc[hits.index].to_numpy().ravel()) & keys)),
        })
    return pd.DataFrame(rows)


def validate_table(table: pd.DataFrame) -> None:
    required = {"Product_key", *COUNT_COLUMNS, "study_identity_conflicts"}
    if required - set(table):
        raise ValueError("Adsorbent support snapshot is missing: " + ", ".join(sorted(required - set(table))))
    if table.Product_key.map(clean_text).eq("").any() or table.Product_key.duplicated().any():
        raise ValueError("Adsorbent support snapshot needs unique nonempty product keys.")
    for name in (*COUNT_COLUMNS, "study_identity_conflicts"):
        values = pd.to_numeric(table[name], errors="coerce")
        if values.isna().any() or not values.between(0, 10**9).all() or not values.eq(values.round()).all():
            raise ValueError(f"Adsorbent support {name} must contain finite nonnegative counts.")
    if table.source_records.gt(table.training_rows).any() or table.independent_studies.gt(table.training_rows).any():
        raise ValueError("Adsorbent support counts contradict fitted record identities.")


def describe(table: pd.DataFrame, metadata: dict[str, Any], product_key: str) -> dict[str, Any]:
    policy = policy_for(metadata)
    hits = table[table.Product_key.eq(product_key)] if not table.empty else pd.DataFrame()
    result = {"available": False, "identity_verified": False, "status": "unknown", "meets_default_ml_policy": False,
              "policy": policy, "reasons": ["Fitted evidence support is unavailable; repackage this bundle to verify it."],
              **{name: None for name in COUNT_COLUMNS}}
    if not metadata.get("available") or hits.empty:
        return result
    row = hits.iloc[0]
    result.update({name: int(row[name]) for name in COUNT_COLUMNS})
    result.update({"available": True, "identity_verified": result["training_rows"] > 0,
                   "largest_study_source_fraction": None if pd.isna(row.get("largest_study_source_fraction")) else float(row.largest_study_source_fraction)})
    reasons = []
    if not result["identity_verified"]:
        reasons.append("No verified product identity match in the fitted records.")
    if result["independent_studies"] < policy["minimum_independent_studies"]:
        reasons.append(f"{result['independent_studies']} DOI-identified studies; at least {policy['minimum_independent_studies']} required.")
    if result["source_records"] < policy["minimum_source_records"]:
        reasons.append(f"{result['source_records']} distinct source records; at least {policy['minimum_source_records']} required.")
    if result["missing_source_identifiers"] or int(row.study_identity_conflicts):
        reasons.append("Some fitted source or study identities could not be verified.")
    result.update({"meets_default_ml_policy": not reasons, "reasons": reasons,
                   "status": "supported" if not reasons else "limited" if result["identity_verified"] else "unseen"})
    return result


def include_exploratory(scenario: dict[str, Any]) -> bool:
    adsorbent = scenario.get("adsorbent", {})
    if not isinstance(adsorbent, dict):
        raise ValueError("scenario.adsorbent must be an object.")
    value = adsorbent.get("include_exploratory", False)
    if not isinstance(value, bool):
        raise ValueError("scenario.adsorbent.include_exploratory must be true or false.")
    return value


def export_support(directory: Path, catalog: pd.DataFrame, records: pd.DataFrame | None,
                   references: pd.DataFrame | None = None) -> dict[str, Any]:
    if records is None or records.empty:
        return {"available": False, "version": SUPPORT_VERSION, "policy": dict(DEFAULT_POLICY),
                "reason": "Actual fitted records were not supplied."}
    table = support_table(catalog, records, references)
    validate_table(table)
    path = directory / SUPPORT_FILENAME
    table.to_csv(path, index=False)
    metadata = {"available": True, "version": SUPPORT_VERSION, "snapshot_filename": SUPPORT_FILENAME,
                "snapshot_sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "policy": dict(DEFAULT_POLICY),
                "fitted_rows": len(records), "identity_matching": "exact normalized curated product aliases against fitted record identities",
                "study_count_basis": "distinct DOI; study_no aliases for the same DOI count once",
                "source_count_basis": DEFAULT_POLICY["source_identity"]}
    metadata["default_ml_supported_products"] = sum(describe(table, metadata, key)["meets_default_ml_policy"] for key in table.Product_key)
    return metadata
