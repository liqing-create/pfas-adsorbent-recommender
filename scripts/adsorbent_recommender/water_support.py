"""Fitted-cohort coverage for the water matrix, separate from prediction accuracy."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

import pandas as pd

VERSION = 1
SNAPSHOT = "water_training_support.json"
# Qualitative display heuristics, not validated error or confidence thresholds.
POLICY = {"minimum_studies": 6, "minimum_source_records": 50,
          "minimum_crosswater_studies": 3, "dominant_study_share": .8}


def export_support(directory: Path, records: pd.DataFrame | None) -> dict:
    required = ["Water_type", "study_no", "source_row_index"]
    if records is None or records.empty or not set(required).issubset(records):
        return {"available": False, "reason": "fitted water/study/source identities unavailable"}
    frame = records.copy()
    if "data_split" in frame:
        split = frame.data_split.astype(str).str.casefold()
        partition = "all_data_refit_apparent" if split.eq("all_data_refit_apparent").any() else "training"
        frame = frame[split.eq(partition)].copy()
    if frame.empty or frame[required].isna().any().any():
        return {"available": False, "reason": "fitted partition or water/study/source identities unavailable"}
    frame[required] = frame[required].astype(str).apply(lambda column: column.str.strip())
    if frame[required].eq("").any().any():
        return {"available": False, "reason": "blank fitted water/study/source identities"}
    datasets = frame["extraction_dataset"].fillna("").astype(str) if "extraction_dataset" in frame else [""] * len(frame)
    frame["__source_key"] = list(zip(datasets, frame.source_row_index))
    source = frame.groupby("__source_key")
    if source.study_no.nunique().max() > 1 or source.Water_type.nunique().max() > 1:
        return {"available": False, "reason": "source identities span multiple studies or water types"}
    study_water = frame.groupby("study_no").Water_type.nunique()
    rows = []
    for water, part in frame.groupby("Water_type"):
        studies = sorted(part.study_no.unique())
        counts = part.groupby("study_no")["__source_key"].nunique().sort_values(ascending=False)
        crosswater = sorted(study_water.loc[studies][study_water.loc[studies].gt(1)].index)
        rows.append({"water_type": water, "model_rows": int(len(part)),
            "source_records": int(part["__source_key"].nunique()), "studies": len(studies),
            "study_ids": studies, "studies_with_other_water": len(crosswater), "crosswater_study_ids": crosswater,
            "largest_study": counts.index[0], "largest_study_source_records": int(counts.iloc[0]),
            "largest_study_source_share": float(counts.iloc[0] / part["__source_key"].nunique())})
    snapshot = {"version": VERSION, "population": "actual_fitted_records", "warning_policy": POLICY,
        "model_rows": int(len(frame)), "source_records": int(frame["__source_key"].nunique()),
        "study_count_basis": "study_no", "source_count_basis": "extraction_dataset + source_row_index", "water_types": rows}
    validate_snapshot(snapshot)
    payload = json.dumps(snapshot, indent=2, allow_nan=False).encode("utf-8")
    (directory / SNAPSHOT).write_bytes(payload)
    return {"available": True, "version": VERSION, "snapshot_filename": SNAPSHOT,
            "snapshot_sha256": hashlib.sha256(payload).hexdigest(), "population": "actual_fitted_records"}


def validate_snapshot(snapshot: dict) -> None:
    if snapshot.get("version") != VERSION or snapshot.get("population") != "actual_fitted_records":
        raise ValueError("Unsupported water-support snapshot. Repackage the bundle.")
    policy = snapshot.get("warning_policy", {})
    if set(policy) != set(POLICY) or any(not isinstance(value, (int, float)) or isinstance(value, bool)
        or not math.isfinite(value) or value <= 0 for value in policy.values()) or policy["dominant_study_share"] > 1:
        raise ValueError("Invalid water-support warning policy.")
    rows = snapshot.get("water_types")
    if not isinstance(rows, list) or not rows:
        raise ValueError("Empty water-support snapshot.")
    names = set()
    for row in rows:
        name = row.get("water_type")
        if not isinstance(name, str) or not name or name.casefold() in names:
            raise ValueError("Invalid water-support water identity.")
        names.add(name.casefold())
        if any(type(row.get(key)) is not int or row[key] < 0 for key in
               ("model_rows", "source_records", "studies", "studies_with_other_water", "largest_study_source_records")):
            raise ValueError("Invalid water-support counts.")
        if not 0 < row["studies"] <= row["source_records"] <= row["model_rows"]:
            raise ValueError("Contradictory water-support counts.")
        ids, cross = row.get("study_ids", []), row.get("crosswater_study_ids", [])
        if len(set(ids)) != row["studies"] or len(set(cross)) != row["studies_with_other_water"] or not set(cross) <= set(ids):
            raise ValueError("Contradictory water-support study identities.")
        largest = row["largest_study_source_records"]
        if row.get("largest_study") not in ids or not 0 < largest <= row["source_records"] or not math.isclose(
            row.get("largest_study_source_share", -1), largest / row["source_records"], rel_tol=1e-9):
            raise ValueError("Contradictory water-support study concentration.")
    for key in ("model_rows", "source_records"):
        if snapshot.get(key) != sum(row[key] for row in rows):
            raise ValueError("Water-support totals disagree.")


def load_support(directory: Path, manifest: dict) -> dict:
    metadata = manifest.get("water_support", {})
    if not metadata.get("available"):
        return {}
    payload = (directory / metadata["snapshot_filename"]).read_bytes()
    if hashlib.sha256(payload).hexdigest() != metadata.get("snapshot_sha256"):
        raise ValueError("The water-support snapshot does not match its bundle fingerprint. Repackage the bundle.")
    snapshot = json.loads(payload)
    validate_snapshot(snapshot)
    return snapshot


def diagnostics(snapshot: dict, water_type: Any) -> dict:
    name = str(water_type or "").strip()
    if not snapshot:
        return {"available": False, "water_type": name, "limited": False}
    row = next((row for row in snapshot["water_types"] if row["water_type"].casefold() == name.casefold()), None)
    if row is None:
        # The existing unseen-category guard handles unrepresented matrices.
        return {"available": True, "water_type": name, "seen_in_training": False, "limited": False}
    policy, reasons = snapshot["warning_policy"], []
    if row["studies"] < policy["minimum_studies"]:
        reasons.append("few_studies")
    if row["source_records"] < policy["minimum_source_records"]:
        reasons.append("few_source_records")
    if row["studies_with_other_water"] < policy["minimum_crosswater_studies"]:
        reasons.append("limited_crosswater_coverage")
    if row["largest_study_source_share"] >= policy["dominant_study_share"]:
        reasons.append("study_concentration")
    if row["water_type"].casefold() == "others":
        reasons.append("broad_water_category")
    studies = f"{row['studies']} {'study' if row['studies'] == 1 else 'studies'}"
    message = (f"Limited water-matrix evidence for {row['water_type']}: {studies}, {row['source_records']} source records. "
               "Validate shortlisted adsorbents in the intended water.") if reasons else ""
    return {**row, "available": True, "seen_in_training": True, "limited": bool(reasons),
            "reasons": reasons, "warning": message, "version": VERSION,
            "warning_policy": policy, "interpretation": "Training coverage only; not calibrated confidence or an error estimate."}


def caveats(support: dict) -> list[dict]:
    return [{"code": "limited_water_evidence", "severity": "info", "label": "Limited water-matrix evidence",
             "text": support["warning"]}] if support.get("limited") else []
