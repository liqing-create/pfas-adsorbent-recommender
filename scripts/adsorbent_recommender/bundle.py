"""Versioned deployment-bundle export and validation for trained logKd models."""

from __future__ import annotations

import hashlib
import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from ML_model_training.backend import logkd_config as cfg
from adsorbent_recommender.catalog import (
    DEFAULT_CATALOG_PATH,
    DEFAULT_CATALOG_SHEET,
    catalog_eligibility,
    catalog_identity_map,
    load_authoritative_catalog,
    normalized_key,
)
from adsorbent_recommender.material_support import export_support
from adsorbent_recommender.water_support import export_support as export_water_support
from adsorbent_recommender.evidence import (
    ADSORBENT_RECORD_FIELDS,
    DEFAULT_EVIDENCE_PATH,
    DEFAULT_EVIDENCE_SHEET,
    DEFAULT_MATCH_POLICY,
    EVIDENCE_SNAPSHOT_FILENAME,
    load_reference_dose_evidence,
)


from adsorbent_recommender.pfas_support import export_runtime_support
from adsorbent_recommender.pfas_descriptors import DESCRIPTOR_VERSION
from rdkit import rdBase


BUNDLE_FILENAME = "recommender_bundle.json"
BUNDLE_FORMAT = "adsorbent_recommender_bundle"
BUNDLE_VERSION = 4
PFAS_FEATURES_SNAPSHOT_FILENAME = "pfas_features.xlsx"
TRAINING_SUPPORT_FILENAME = "training_support.csv"
TRAINING_PREDICTIONS_FILENAME = "all_predictions.csv"
TUNING_SUMMARY_FILENAME = "tuning_summary.json"

# The model keeps original dataset-column names.  This registry gives runtime
# code a stable source and human label without changing historic data columns.
SCENARIO_FIELD_SPECS: dict[str, dict[str, str]] = {
    "PFAS_C0_value_mg/L": {"source": "scenario", "key": "initial_concentration_mg_l", "label": "Initial PFAS concentration", "unit": "mg/L"},
    "pH": {"source": "water", "key": "pH", "label": "pH", "unit": ""},
    "organic_carbon_mg/L": {"source": "water", "key": "organic_carbon_mg_l", "label": "Organic carbon", "unit": "mg/L"},
    "TDS_(mg/L)": {"source": "water", "key": "tds_mg_l", "label": "TDS", "unit": "mg/L"},
    "Water_type": {"source": "water", "key": "water_type", "label": "Water type", "unit": ""},
    "Ionic_strength_(mol/L)": {"source": "water", "key": "ionic_strength_mol_l", "label": "Ionic strength", "unit": "mol/L"},
    "Inorganic_matter": {"source": "water", "key": "inorganic_matter", "label": "Inorganic matter", "unit": ""},
    "Organic_matter": {"source": "water", "key": "organic_matter", "label": "Organic matter", "unit": ""},
}
for _ion in ("Na", "K", "Ca", "Mg", "Cl", "HCO3", "SO4", "phosphate"):
    SCENARIO_FIELD_SPECS[f"contains_{_ion}"] = {
        "source": "water_ion", "key": _ion, "label": f"Contains {_ion}", "unit": "",
    }
for _feature in ("inorganic_matter_present", "organic_matter_present", "organic_matter_class"):
    SCENARIO_FIELD_SPECS[_feature] = {"source": "derived_water", "key": _feature, "label": _feature.replace("_", " "), "unit": ""}

# A recommender ranks adsorbents against one another, so the dose is held at a
# common value rather than optimized per candidate: a dose search would reward
# the material whose best dose happens to sit in a denser part of the training
# data instead of the material with the higher affinity.  The default is a
# treatment-realistic dose that every adsorbent class in the database supports.
DEFAULT_FIXED_DOSE_MG_L = 25.0

CONTROL_POLICIES: dict[str, dict[str, Any]] = {
    "Adsorbent_dosage_value_mg/L": {
        "role": "fixed",
        "default": DEFAULT_FIXED_DOSE_MG_L,
        "transform": "log10",
        "label": "Adsorbent dose",
        "unit": "mg/L",
        "default_basis": (
            "held common across candidates for a fair affinity comparison; "
            "25 mg/L is the published GAC carbon-usage feasibility threshold and "
            "sits inside the 2-50 mg/L full-scale PAC range, while keeping every "
            "adsorbent class represented in the training dose distribution"
        ),
    },
    "Temperature_(°C)": {"role": "fixed", "default": 25.0, "label": "Temperature", "unit": "°C"},
    "Temperature_(Â°C)": {"role": "fixed", "default": 25.0, "label": "Temperature", "unit": "°C"},
    "Temperature_(Ã‚Â°C)": {"role": "fixed", "default": 25.0, "label": "Temperature", "unit": "°C"},
    "Temperature_(Ãƒâ€šÃ‚Â°C)": {"role": "fixed", "default": 25.0, "label": "Temperature", "unit": "°C"},
    "Solution_volume_(mL)": {"role": "fixed", "default": 100.0, "label": "Solution volume", "unit": "mL"},
    "Mixing_speed_(rpm)": {"role": "fixed", "default": 150.0, "label": "Mixing speed", "unit": "rpm"},
}


def _support_record(values: pd.Series, kind: str) -> dict[str, Any]:
    if kind == "categorical":
        return {"kind": "categorical", "levels": sorted(set(filter(None, values.map(str))))}
    numeric = pd.to_numeric(values, errors="coerce").dropna().astype(float)
    if numeric.empty:
        return {"kind": "numeric", "count": 0}
    return {
        "kind": "numeric", "count": int(len(numeric)), "hard_min": float(numeric.min()), "hard_max": float(numeric.max()),
        "p01": float(numeric.quantile(0.01)), "p05": float(numeric.quantile(0.05)), "median": float(numeric.quantile(0.50)),
        "p95": float(numeric.quantile(0.95)), "p99": float(numeric.quantile(0.99)),
    }


def derive_feature_support_and_controls(
    frame: pd.DataFrame,
    selected: Iterable[str],
    numeric_features: Iterable[str],
    categorical_features: Iterable[str],
    control_defaults: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Derive deployment support and recipe-control bounds from development rows.

    ``control_defaults`` overrides the value a fixed control takes when the
    caller supplies none, so the deployed dose can be chosen per build without
    editing the policy table.  It changes scoring inputs only; the fitted model
    and its training rows are untouched.
    """
    control_defaults = control_defaults or {}
    numeric = set(numeric_features)
    categorical = set(categorical_features)
    controls: dict[str, Any] = {}
    support: dict[str, Any] = {}
    for feature in selected:
        if feature not in frame.columns:
            continue
        kind = "categorical" if feature in categorical else "numeric"
        record = _support_record(frame[feature], kind)
        support[feature] = record
        policy = CONTROL_POLICIES.get(feature)
        if policy is None or kind != "numeric" or record.get("count", 0) == 0:
            continue
        lower, upper = record["p05"], record["p95"]
        if "search_upper_cap" in policy:
            upper = min(upper, float(policy["search_upper_cap"]))
        default = float(control_defaults.get(feature, policy.get("default", record["median"])))
        controls[feature] = {
            **record,
            **policy,
            "default": default,
            "search_lower": float(lower),
            "search_upper": float(upper),
            # A fixed control set outside the development range makes every
            # prediction an extrapolation, so the choice is recorded rather than
            # assumed sound.
            "default_within_observed_support": bool(record["p05"] <= default <= record["p95"]),
        }
    return {"source_partition": "development_only", "controls": controls, "feature_support": support}


def _input_contract(selected: Iterable[str], selected_by_bucket: dict[str, list[str]], controls: dict[str, Any]) -> dict[str, Any]:
    material = set(selected_by_bucket.get("Adsorbent properties", []))
    pfas = set(selected_by_bucket.get("PFAS characteristics", []))
    fields, unsupported = [], []
    for feature in selected:
        if feature in material:
            fields.append({"model_feature": feature, "source": "catalog", "label": feature, "unit": ""})
        elif feature in pfas:
            fields.append({"model_feature": feature, "source": "pfas_cache", "label": feature, "unit": ""})
        elif feature in controls:
            fields.append({"model_feature": feature, "source": "test_control", "label": controls[feature].get("label", feature), "unit": controls[feature].get("unit", "")})
        elif feature in SCENARIO_FIELD_SPECS:
            fields.append({"model_feature": feature, **SCENARIO_FIELD_SPECS[feature]})
        else:
            fields.append({"model_feature": feature, "source": "unsupported", "label": feature, "unit": ""})
            unsupported.append(feature)
    return {"fields": fields, "unsupported_selected_features": unsupported}


def _relative_artifact(filename: str) -> str:
    return filename


def _snapshot_pfas_features(output_dir: Path, source: Path | None, sheet_name: str) -> dict[str, Any]:
    """Copy the PFAS feature cache the model was trained against into the bundle.

    Recommendation looks every PFAS up in this workbook at run time.  A bundle
    that only records where the workbook lived stops working wherever that path
    does not exist, which is everywhere except the training machine.
    """
    details: dict[str, Any] = {
        "available": False,
        "source_path": None if source is None else str(source),
        "sheet_name": sheet_name,
    }
    if source is None or not Path(source).is_file():
        details["reason"] = "no PFAS feature workbook was joined" if source is None else "PFAS feature workbook not found"
        return details
    target = output_dir / PFAS_FEATURES_SNAPSHOT_FILENAME
    if Path(source).resolve() != target.resolve():
        shutil.copy2(source, target)
    details.update({
        "available": True,
        "snapshot_filename": PFAS_FEATURES_SNAPSHOT_FILENAME,
        "sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
    })
    return details


def training_support_table(records: pd.DataFrame) -> pd.DataFrame:
    """Count the training records and studies behind each PFAS and adsorbent name.

    Keys are normalized the way the database matcher normalizes them, so a
    catalog product counts as seen in training exactly when one of its aliases
    names a training record.  Recording counts rather than a flag lets the app
    say how thin the support is, not only whether it exists.
    """
    studies = records["study_no"] if "study_no" in records.columns else pd.Series(range(len(records)))
    tables = []
    for kind, fields in (("pfas", ("PFAS_name",)), ("adsorbent", ADSORBENT_RECORD_FIELDS)):
        fields = [name for name in fields if name in records.columns]
        if not fields:
            continue
        keys = records[fields].apply(lambda column: column.map(normalized_key))
        keys["__study"] = studies.to_numpy()
        keys["__row"] = np.arange(len(records))
        long = keys.melt(id_vars=["__study", "__row"], value_name="key")
        long = long[long["key"].ne("")].drop_duplicates(["__row", "key"])
        table = long.groupby("key").agg(records=("__row", "nunique"), studies=("__study", "nunique")).reset_index()
        table.insert(0, "kind", kind)
        tables.append(table)
    if not tables:
        return pd.DataFrame(columns=["kind", "key", "records", "studies"])
    return pd.concat(tables, ignore_index=True)


def _validation_rmse(tuning_path: Path) -> float | None:
    if not tuning_path.is_file():
        return None
    value = json.loads(tuning_path.read_text(encoding="utf-8")).get("best_validation_rmse")
    return None if value is None else float(value)


def _snapshot_training_support(output_dir: Path, records: pd.DataFrame | None) -> dict[str, Any]:
    """Write which PFAS and adsorbents the deployed model was fitted on.

    The app uses it to warn when a PFAS or catalog product has no training
    record, and the tuned validation RMSE to say when two scores are too close
    to rank.  Both describe the model itself, so they travel with the bundle.
    """
    if records is None or records.empty:
        return {"available": False, "reason": "no training records were supplied"}
    table = training_support_table(records)
    table.to_csv(output_dir / TRAINING_SUPPORT_FILENAME, index=False)
    return {
        "available": True,
        "snapshot_filename": TRAINING_SUPPORT_FILENAME,
        "training_records": int(len(records)),
        "pfas_count": int(table["kind"].eq("pfas").sum()),
        "validation_rmse": _validation_rmse(output_dir / TUNING_SUMMARY_FILENAME),
        "validation_rmse_basis": "best cross-validated RMSE of the tuned configuration (tuning_summary.json)",
    }


def _snapshot_evidence(output_dir: Path, source: Path, sheet_name: str) -> dict[str, Any]:
    evidence, details = load_reference_dose_evidence(source, sheet_name)
    if details.get("available"):
        evidence.to_csv(output_dir / EVIDENCE_SNAPSHOT_FILENAME, index=False)
        details["snapshot_filename"] = EVIDENCE_SNAPSHOT_FILENAME
    details["match_policy"] = DEFAULT_MATCH_POLICY
    return details


def _snapshot_material_support(output_dir: Path, records: pd.DataFrame | None) -> dict[str, Any]:
    """Export fitted product support; bibliography is read only while packaging."""
    if records is None or records.empty:
        return export_support(output_dir, pd.DataFrame(), records)
    catalog_path = output_dir / "catalog_snapshot.csv"
    if not catalog_path.is_file():
        return {"available": False, "reason": "No catalog snapshot was supplied."}
    catalog = pd.read_csv(catalog_path)
    identity_path = output_dir / "catalog_product_model_identity_map.csv"
    if identity_path.is_file():
        catalog = catalog.merge(pd.read_csv(identity_path), on="Product_key", validate="one_to_one")
    # Study DOIs come from the database's Standardized_Full sheet, which lists every study. The path in
    # run_config is the one on the machine that trained the model, so the project's own copy of the
    # database is tried next. The reference-dose evidence snapshot only covers studies with records near
    # 25 mg/L and is a last resort; its use would undercount DOI-identified studies.
    candidates = []
    config_path = output_dir / "run_config.json"
    if config_path.is_file():
        config = json.loads(config_path.read_text(encoding="utf-8"))
        if config.get("input_path"):
            candidates.append(Path(config["input_path"]))
    candidates.append(Path(cfg.DEFAULT_INPUT))
    references, bibliography = None, None
    for source in candidates:
        if not source.is_file():
            continue
        try:
            references = pd.read_excel(source, sheet_name="Standardized_Full", usecols=lambda name: name in {"study_no", "DOI"})
            bibliography = f"Standardized_Full of {source.name}"
            break
        except (ValueError, OSError):
            continue
    if references is None:
        evidence = output_dir / EVIDENCE_SNAPSHOT_FILENAME
        if evidence.is_file():
            references = pd.read_csv(evidence, usecols=lambda name: name in {"study_no", "DOI"})
            bibliography = f"{EVIDENCE_SNAPSHOT_FILENAME} (partial: studies with reference-dose records only)"
    metadata = export_support(output_dir, catalog, records, references)
    if bibliography:
        metadata["bibliography_source"] = bibliography
    return metadata


def export_recommender_bundle(
    output_dir: Path,
    model_family: str,
    target: str,
    selected: list[str],
    numeric_features: list[str],
    categorical_features: list[str],
    selected_by_bucket: dict[str, list[str]],
    bounds: dict[str, Any],
    catalog_path: Path | None = None,
    catalog_sheet: str | None = None,
    model_adsorbent_ids: Iterable[Any] = (),
    minimum_material_feature_count: int | None = None,
    evidence_path: Path | None = None,
    evidence_sheet: str = DEFAULT_EVIDENCE_SHEET,
    pfas_features_path: Path | None = None,
    pfas_features_sheet: str = cfg.DEFAULT_PFAS_FEATURES_SHEET,
    training_records: pd.DataFrame | None = None,
) -> dict[str, Any]:
    """Write the stable runtime manifest and catalog artifacts for one run.

    Everything recommendation reads at run time is copied into ``output_dir``,
    so the bundle directory alone is enough to serve the model elsewhere.
    """
    catalog_details: dict[str, Any] = {"available": False, "category": None, "eligibility_filename": None}
    material_features = list(selected_by_bucket.get("Adsorbent properties", []))
    catalog_error = None
    # A model family screens the catalog categories it was actually fitted on.
    # Categories no family covers, such as Clay/mineral, are therefore never
    # scored by a model that has not seen them.
    screened_categories = list(cfg.MODEL_CATEGORIES.get(model_family, []))
    if screened_categories:
        try:
            source_path = catalog_path or DEFAULT_CATALOG_PATH
            source_sheet = catalog_sheet or DEFAULT_CATALOG_SHEET
            catalog = load_authoritative_catalog(source_path, source_sheet, category=screened_categories)
            eligibility = catalog_eligibility(catalog, material_features, minimum_material_feature_count)
            identity = catalog_identity_map(catalog, model_adsorbent_ids)
            catalog.to_csv(output_dir / "catalog_snapshot.csv", index=False)
            eligibility.to_csv(output_dir / "catalog_eligibility.csv", index=False)
            identity.to_csv(output_dir / "catalog_product_model_identity_map.csv", index=False)
            catalog_details = {
                "available": True,
                "category": screened_categories,
                "source_path": str(source_path),
                "source_sheet": source_sheet,
                "snapshot_filename": "catalog_snapshot.csv",
                "eligibility_filename": "catalog_eligibility.csv",
                "identity_map_filename": "catalog_product_model_identity_map.csv",
                "required_material_features": material_features,
                "minimum_material_feature_count": minimum_material_feature_count,
                "minimum_material_feature_basis": (
                    "median material-feature coverage of the fitted training rows"
                    if minimum_material_feature_count is not None
                    else "every selected adsorbent feature present"
                ),
                "screened_product_count": int(len(catalog)),
                "eligible_candidate_count": int(eligibility["eligible"].sum()),
            }
        except Exception as exc:  # Preserve a usable research run even when a catalog is unavailable.
            catalog_error = f"{type(exc).__name__}: {exc}"
            catalog_details["error"] = catalog_error

    evidence_details = _snapshot_evidence(output_dir, evidence_path or DEFAULT_EVIDENCE_PATH, evidence_sheet)
    pfas_details = _snapshot_pfas_features(output_dir, pfas_features_path, pfas_features_sheet)
    training_details = _snapshot_training_support(output_dir, training_records)
    material_support = _snapshot_material_support(output_dir, training_records)
    water_support = export_water_support(output_dir, training_records)

    contract = _input_contract(selected, selected_by_bucket, bounds.get("controls", {}))
    capabilities = {
        "catalog_screening": bool(catalog_details.get("available") and catalog_details.get("eligible_candidate_count", 0) > 0 and not contract["unsupported_selected_features"]),
        "database_evidence_lookup": bool(evidence_details.get("available")),
        "adsorbent_support_screening": bool(material_support.get("available")),
    }
    manifest = {
        "bundle_format": BUNDLE_FORMAT,
        "bundle_version": BUNDLE_VERSION,
        "model_family": model_family,
        "target": target,
        "artifacts": {
            "run_config": _relative_artifact("run_config.json"),
            "ensemble": _relative_artifact("model_ensemble.joblib"),
            "feature_support": _relative_artifact("optimizer_bounds.json"),
        },
        "features": {"selected": selected, "numeric": numeric_features, "categorical": categorical_features, "selected_by_bucket": selected_by_bucket},
        "input_contract": contract,
        "controls": bounds.get("controls", {}),
        "catalog": catalog_details,
        "database_evidence": evidence_details,
        "pfas_features": pfas_details,
        "training_support": training_details,
        "adsorbent_support": material_support,
        "water_support": water_support,
        "capabilities": capabilities,
        "status": "ready" if capabilities["catalog_screening"] else "not_ready",
    }
    if catalog_error:
        manifest["status_reason"] = catalog_error
    elif contract["unsupported_selected_features"]:
        manifest["status_reason"] = "Unsupported selected features require a runtime source mapping."
    elif not catalog_details.get("eligible_candidate_count", 0):
        manifest["status_reason"] = (
            "No catalog product reaches the required adsorbent-feature coverage "
            f"({catalog_details.get('minimum_material_feature_count')} of "
            f"{len(material_features)})."
        )
    runtime = export_runtime_support(output_dir, manifest, training_records if training_records is not None else pd.DataFrame())
    manifest["pfas_runtime"] = runtime
    manifest["capabilities"]["unseen_pfas_prediction"] = bool(runtime.get("enabled"))
    (output_dir / BUNDLE_FILENAME).write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def load_bundle_manifest(bundle_dir: Path) -> dict[str, Any]:
    path = bundle_dir / BUNDLE_FILENAME
    if not path.exists():
        raise FileNotFoundError(f"Missing recommender bundle manifest: {path}")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("bundle_format") != BUNDLE_FORMAT or manifest.get("bundle_version") != BUNDLE_VERSION:
        raise ValueError(
            "Legacy or unsupported recommender bundle; rebuild it with bundle schema version "
            f"{BUNDLE_VERSION}. Applicability-domain artifacts are no longer supported."
        )
    declared = dict(manifest.get("artifacts", {}))
    # A snapshot the manifest declares but the directory lacks would otherwise
    # fall back silently to a path on the training machine.
    for section in ("pfas_features", "database_evidence", "training_support", "pfas_runtime", "adsorbent_support", "water_support"):
        snapshot = manifest.get(section, {}).get("snapshot_filename")
        if snapshot is not None:
            declared[section] = snapshot
    for name, filename in declared.items():
        relative = Path(str(filename))
        if relative.is_absolute() or len(relative.parts) != 1 or not (bundle_dir / relative).exists():
            raise FileNotFoundError(f"Invalid or missing bundle artifact {name!r}: {filename!r}")
    return manifest


def make_bundle_self_contained(
    bundle_dir: Path,
    pfas_features_path: Path | None = None,
    evidence_path: Path | None = None,
    evidence_sheet: str = DEFAULT_EVIDENCE_SHEET,
    refresh_adsorbent_support: bool = False,
) -> dict[str, Any]:
    """Copy runtime data into a bundle exported before bundles carried it.

    The model, catalog snapshot and feature support are not touched.  The PFAS
    feature cache defaults to the workbook recorded in run_config; the evidence
    snapshot is taken from the workbook as it is now, which is also what such a
    bundle was reading at run time.  Training support is counted from the
    bundle's own all_predictions.csv, the rows the deployed model was fitted on.
    Sections already present are left alone, so running this twice changes
    nothing; ``refresh_adsorbent_support`` rebuilds the adsorbent training-support
    snapshot regardless, for example after its study bibliography was incomplete.
    """
    bundle_dir = Path(bundle_dir)
    manifest = load_bundle_manifest(bundle_dir)
    retrofitted = []
    if not manifest.get("pfas_features", {}).get("available"):
        run_config = json.loads((bundle_dir / manifest["artifacts"]["run_config"]).read_text(encoding="utf-8"))
        trained = run_config.get("pfas_features", {})
        source = pfas_features_path or trained.get("path")
        manifest["pfas_features"] = _snapshot_pfas_features(
            bundle_dir,
            Path(source) if source else None,
            trained.get("sheet_name") or cfg.DEFAULT_PFAS_FEATURES_SHEET,
        )
        retrofitted.append("pfas_features")
    if not manifest.get("database_evidence", {}).get("snapshot_filename"):
        evidence_details = _snapshot_evidence(
            bundle_dir, Path(evidence_path or DEFAULT_EVIDENCE_PATH), evidence_sheet,
        )
        if evidence_details != manifest.get("database_evidence"):
            retrofitted.append("database_evidence")
        manifest["database_evidence"] = evidence_details
        manifest.setdefault("capabilities", {})["database_evidence_lookup"] = bool(
            manifest["database_evidence"].get("available")
        )
    predictions_path = bundle_dir / TRAINING_PREDICTIONS_FILENAME
    if not manifest.get("water_support", {}).get("available") and predictions_path.is_file():
        manifest["water_support"] = export_water_support(bundle_dir, pd.read_csv(predictions_path, low_memory=False))
        if manifest["water_support"].get("available"):
            retrofitted.append("water_support")
    if (refresh_adsorbent_support or not manifest.get("adsorbent_support", {}).get("available")) and predictions_path.is_file():
        records = pd.read_csv(predictions_path, low_memory=False)
        if "data_split" in records and records.data_split.eq("all_data_refit_apparent").all():
            manifest["adsorbent_support"] = _snapshot_material_support(bundle_dir, records)
            manifest.setdefault("capabilities", {})["adsorbent_support_screening"] = bool(manifest["adsorbent_support"].get("available"))
            retrofitted.append("adsorbent_support")
    if not manifest.get("training_support", {}).get("available") and predictions_path.is_file():
        manifest["training_support"] = _snapshot_training_support(
            bundle_dir, pd.read_csv(predictions_path, low_memory=False),
        )
        retrofitted.append("training_support")
    previous_runtime = manifest.get("pfas_runtime", {})
    runtime_needs_refresh = (
        not previous_runtime.get("available")
        or previous_runtime.get("descriptor_version") != DESCRIPTOR_VERSION
        or previous_runtime.get("rdkit_version") != rdBase.rdkitVersion
        or "descriptor_parity" not in previous_runtime
    )
    if runtime_needs_refresh and predictions_path.is_file():
        runtime = export_runtime_support(bundle_dir, manifest, pd.read_csv(predictions_path, low_memory=False))
        if runtime != manifest.get("pfas_runtime"):
            retrofitted.append("pfas_runtime")
        manifest["pfas_runtime"] = runtime
        manifest.setdefault("capabilities", {})["unseen_pfas_prediction"] = bool(runtime.get("enabled"))
    if retrofitted:
        manifest.setdefault("retrofits", []).append({
            "sections": retrofitted,
            "at": datetime.now().isoformat(timespec="seconds"),
            "note": "copied in after export from the source files as they were at this time",
        })
        (bundle_dir / BUNDLE_FILENAME).write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Copy the PFAS feature cache, reference-dose evidence and training support into an existing bundle.",
    )
    parser.add_argument("bundle_dir", type=Path)
    parser.add_argument("--pfas-features-path", type=Path, help="defaults to the workbook recorded in run_config.json")
    parser.add_argument("--evidence-path", type=Path, help=f"defaults to {DEFAULT_EVIDENCE_PATH}")
    parser.add_argument("--refresh-adsorbent-support", action="store_true",
                        help="rebuild the adsorbent training-support snapshot even if the bundle already has one")
    arguments = parser.parse_args()
    result = make_bundle_self_contained(
        arguments.bundle_dir, arguments.pfas_features_path, arguments.evidence_path,
        refresh_adsorbent_support=arguments.refresh_adsorbent_support,
    )
    print(json.dumps({
        section: {key: result.get(section, {}).get(key) for key in ("available", "snapshot_filename", "reason")}
        for section in ("pfas_features", "database_evidence", "training_support", "adsorbent_support")
    }, indent=2))
