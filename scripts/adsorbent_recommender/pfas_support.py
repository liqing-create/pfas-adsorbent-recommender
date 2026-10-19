"""Training-structure provenance and descriptive molecular resemblance; no confidence calibration."""
from __future__ import annotations
import json
from functools import lru_cache
from pathlib import Path
from typing import Any
import pandas as pd
import numpy as np
from rdkit import Chem, DataStructs, rdBase
from rdkit.Chem import rdFingerprintGenerator
from adsorbent_recommender.pfas_descriptors import DESCRIPTOR_VERSION, generate_features, prepare_structure

REFERENCE_FILENAME = "pfas_training_reference.csv"
UNSEEN_WARNING = (
    "Exploratory prediction: This PFAS was not represented in model training. "
    "Performance on held-out PFAS was weaker than on known-PFAS combinations. "
    "Validate shortlisted adsorbents experimentally."
)

def identity_key(value: Any) -> str:
    return "" if value is None or pd.isna(value) else "".join(c for c in str(value).casefold() if c.isalnum())

def export_runtime_support(directory: Path, manifest: dict, records: pd.DataFrame) -> dict:
    """Package structures linked to actual fitted rows without touching model weights."""
    info = manifest.get("pfas_features", {})
    filename = info.get("snapshot_filename")
    if not filename or records.empty or "PFAS_name" not in records:
        return {"available": False, "reason": "Training records and a reviewed feature snapshot are required."}
    cache = pd.read_excel(directory / filename, sheet_name=info.get("sheet_name", "Sheet1"))
    trained = records["PFAS_name"].map(identity_key)
    references: dict[str, dict] = {}
    mapped: set[str] = set()
    for _, row in cache.iterrows():
        aliases = {identity_key(row.get(k)) for k in ("Abbreviation", "PFAS_name", "Compound Full Name", "Name")} - {""}
        hits = records[trained.isin(aliases)]
        if hits.empty:
            continue
        smiles = row.get("Canonical SMILES") or row.get("SMILES")
        try:
            _, canonical = prepare_structure(str(smiles))
        except ValueError:
            continue
        mapped.update(trained.loc[hits.index])
        label = str(row.get("Abbreviation") or row.get("PFAS_name") or canonical)
        entry = references.setdefault(canonical, {"canonical_smiles": canonical, "label": label, "aliases": set(), "indices": set()})
        entry["aliases"].update(aliases)
        entry["indices"].update(hits.index)
    missing = sorted(set(trained) - mapped - {""})
    if not references or missing:
        return {"available": False, "reason": "Some fitted PFAS could not be linked to curated structures.", "unmapped_training_keys": missing}
    rows = []
    for entry in references.values():
        subset = records.loc[sorted(entry.pop("indices"))]
        rows.append({**entry, "aliases": json.dumps(sorted(entry["aliases"])), "records": len(subset),
                     "studies": int(subset["study_no"].nunique()) if "study_no" in subset else 0})
    pd.DataFrame(rows).to_csv(directory / REFERENCE_FILENAME, index=False)
    required = [f["model_feature"] for f in manifest["input_contract"]["fields"] if f.get("source") == "pfas_cache"]
    computable = generate_features("O=C(O)C(F)(F)F")
    supported = bool(required) and all(name in computable for name in required)
    mismatches = []
    if supported:
        for _, row in cache.iterrows():
            mol = Chem.MolFromSmiles(str(row.get("SMILES") or row.get("Canonical SMILES")))
            if mol is None:
                mismatches.append({"PFAS": str(row.get("Abbreviation")), "feature": "SMILES"})
                continue
            computed = generate_features(Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True))
            for name in required:
                value = pd.to_numeric(row.get(name), errors="coerce")
                if not np.isfinite(value) or not np.isclose(value, computed[name], rtol=1e-9, atol=1e-9):
                    mismatches.append({"PFAS": str(row.get("Abbreviation")), "feature": name})
    parity = {"rows_checked": len(cache) if supported else 0, "feature_count": len(required),
              "mismatch_count": len(mismatches), "mismatch_examples": mismatches[:10]}
    supported = supported and not mismatches
    return {"available": True, "enabled": supported, "snapshot_filename": REFERENCE_FILENAME,
            "descriptor_version": DESCRIPTOR_VERSION, "rdkit_version": rdBase.rdkitVersion,
            "required_features": required, "unsupported_features": [name for name in required if name not in computable],
            "descriptor_parity": parity,
            "training_structure_count": len(rows), "fingerprint": {"kind": "Morgan", "radius": 2, "bits": 2048,
            "include_chirality": True, "similarity": "Tanimoto"},
            "interpretation": "Structural resemblance only; not a confidence score or prediction interval."}

@lru_cache(maxsize=8)
def _references(path: str, mtime_ns: int) -> tuple:
    frame = pd.read_csv(path, keep_default_na=False)
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048, includeChirality=True)
    return tuple((row.to_dict(), generator.GetFingerprint(Chem.MolFromSmiles(row["canonical_smiles"]))) for _, row in frame.iterrows())

def describe_support(bundle: Any, lookup: Any) -> dict:
    metadata = bundle.manifest.get("pfas_runtime", {})
    filename = metadata.get("snapshot_filename")
    canonical = getattr(lookup, "canonical_smiles", "")
    result = {"seen_in_training": None, "nearest_training_pfas": [], "out_of_range_descriptors": [],
              "interpretation": "Structural resemblance only; not a confidence score or prediction interval."}
    if filename and (bundle.directory / filename).is_file() and canonical:
        path = bundle.directory / filename
        refs = _references(str(path), path.stat().st_mtime_ns)
        try:
            _, normalized = prepare_structure(canonical)
        except ValueError:
            return result
        generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048, includeChirality=True)
        query = generator.GetFingerprint(Chem.MolFromSmiles(normalized))
        exact = [row for row, _ in refs if row["canonical_smiles"] == normalized]
        result["seen_in_training"] = bool(exact)
        if exact:
            result["training_records"] = int(exact[0]["records"])
            result["training_studies"] = int(exact[0]["studies"])
        neighbors = [{"label": row["label"], "canonical_smiles": row["canonical_smiles"],
                      "tanimoto_similarity": float(DataStructs.TanimotoSimilarity(query, fp))} for row, fp in refs]
        result["nearest_training_pfas"] = sorted(neighbors, key=lambda row: (-row["tanimoto_similarity"], row["label"]))[:3]
    elif getattr(lookup, "feature_source", None) == "computed_at_runtime":
        result["seen_in_training"] = False
    else:
        known = set(bundle.bounds.get("known_pfas_keys", []))
        row = getattr(lookup, "feature_row", None) or {}
        aliases = {identity_key(row.get(k)) for k in ("Abbreviation", "PFAS_name", "Compound Full Name", "Name")}
        aliases.add(identity_key(getattr(lookup, "cache_key", None)))
        if known:
            result["seen_in_training"] = bool(known.intersection(aliases - {""}))
    required = {f["model_feature"] for f in bundle.manifest.get("input_contract", {}).get("fields", []) if f.get("source") == "pfas_cache"}
    for feature in required:
        value = (lookup.feature_row or {}).get(feature)
        limits = bundle.bounds.get("feature_support", {}).get(feature, {})
        if value is not None and limits.get("kind") == "numeric":
            low, high = limits.get("hard_min"), limits.get("hard_max")
            if low is not None and high is not None and not low <= float(value) <= high:
                result["out_of_range_descriptors"].append({"feature": feature, "value": float(value), "minimum": low, "maximum": high})
    return result
