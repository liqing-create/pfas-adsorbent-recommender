"""Audit archived outer PFAS holdouts; never fit models or use refit errors."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd

if str(Path(__file__).resolve().parents[1]) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from adsorbent_recommender.pfas_descriptors import prepare_structure
from adsorbent_recommender.pfas_support import identity_key

def audit(archive: Path, cache_path: Path, output: Path) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    cache = pd.read_excel(cache_path)
    structures = {}
    for _, row in cache.iterrows():
        _, canonical = prepare_structure(str(row["Canonical SMILES"]))
        for column in ("Abbreviation", "PFAS_name", "Compound Full Name", "Name"):
            key = identity_key(row.get(column))
            if key:
                structures.setdefault(key, set()).add(canonical)
    split_rows = []
    for path in sorted((archive / "model_runs/pfas").rglob("split_assignments.csv")):
        frame = pd.read_csv(path)
        testing = frame["split"].eq("testing")
        train_keys = set(frame.loc[~testing, "PFAS_name"].map(identity_key))
        test_keys = set(frame.loc[testing, "PFAS_name"].map(identity_key))
        missing = sorted(key for key in train_keys | test_keys if len(structures.get(key, ())) != 1)
        train_structures = {s for key in train_keys for s in structures.get(key, ())}
        test_structures = {s for key in test_keys for s in structures.get(key, ())}
        split_rows.append({"allocation": path.parents[1].name, "repeat": path.parent.name,
            "training_compounds": len(train_structures), "testing_compounds": len(test_structures),
            "testing_source_rows": frame.loc[testing, "source_row_index"].nunique(),
            "unmapped_or_ambiguous_names": json.dumps(missing),
            "overlapping_names": len(train_keys & test_keys),
            "overlapping_structures": len(train_structures & test_structures),
            "overlapping_source_rows": len(set(frame.loc[testing, "source_row_index"]) & set(frame.loc[~testing, "source_row_index"]))})
    splits = pd.DataFrame(split_rows)
    if splits.empty:
        raise ValueError("No retained PFAS outer splits were found.")
    splits.to_csv(output / "chemical_separation.csv", index=False)
    predictions = pd.read_parquet(archive / "detail/testing_predictions.parquet")
    predictions = predictions[predictions["split_strategy"].eq("pfas")].copy()
    if predictions.empty or not predictions["data_split"].eq("testing").all():
        raise ValueError("Expected outer testing predictions only.")
    compound_rows = []
    for key, group in predictions.groupby(["outer_allocation_method", "outer_repeat_id", "PFAS_name"]):
        weights = group["source_row_weight"].to_numpy(float)
        error = group["y_pred"].to_numpy(float) - group["y_true"].to_numpy(float)
        compound_rows.append({"allocation": key[0], "repeat": int(key[1]), "PFAS": key[2],
            "rows": len(group), "source_rows": group["source_row_index"].nunique(),
            "studies": group["study_no"].nunique(), "mae": float(np.average(abs(error), weights=weights)),
            "rmse": float(np.sqrt(np.average(error**2, weights=weights))),
            "bias": float(np.average(error, weights=weights))})
    pd.DataFrame(compound_rows).to_csv(output / "per_compound_errors.csv", index=False)
    summary = pd.read_csv(archive / "summary/performance_summary.csv")
    summary = summary[summary["split_strategy"].isin(["pfas", "combination"])]
    summary.to_csv(output / "heldout_performance.csv", index=False)
    passed = bool((splits[["overlapping_names", "overlapping_structures", "overlapping_source_rows"]] == 0).all().all()
                  and splits["unmapped_or_ambiguous_names"].eq("[]").all())
    result = {"chemical_separation_passed": passed, "outer_splits": len(splits),
        "heldout_pfAS": predictions["PFAS_name"].nunique(), "testing_rows_including_repeats": len(predictions),
        "per_compound_rows": len(compound_rows), "archive": str(archive.resolve()),
        "cache": str(cache_path.resolve()),
        "limitations": ["The curated snapshot supplies structure mappings; the archive has no frozen structure snapshot.",
            "Population-level errors are not individual prediction intervals.",
            "Different allocation strategies have different test memberships; the metric differences are descriptive.",
            "Ranking among commercial adsorbents is not independently validated by aggregate logKd error."]}
    (output / "audit_summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("cache", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    print(json.dumps(audit(args.archive, args.cache, args.output), indent=2))
