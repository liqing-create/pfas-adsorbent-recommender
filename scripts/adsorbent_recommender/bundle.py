"""Versioned deployment-bundle export and validation for trained logKd models."""

from __future__ import annotations

import json
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
)
from adsorbent_recommender.evidence import (
    DEFAULT_EVIDENCE_PATH,
    DEFAULT_EVIDENCE_SHEET,
    DEFAULT_MATCH_POLICY,
    EVIDENCE_SNAPSHOT_FILENAME,
    load_reference_dose_evidence,
)


BUNDLE_FILENAME = "recommender_bundle.json"
BUNDLE_FORMAT = "adsorbent_recommender_bundle"
BUNDLE_VERSION = 4

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
) -> dict[str, Any]:
    """Write the stable runtime manifest and catalog artifacts for one run."""
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

    evidence_source = evidence_path or DEFAULT_EVIDENCE_PATH
    evidence, evidence_details = load_reference_dose_evidence(
        evidence_source,
        evidence_sheet,
    )
    if evidence_details.get("available"):
        evidence.to_csv(output_dir / EVIDENCE_SNAPSHOT_FILENAME, index=False)
        evidence_details["snapshot_filename"] = EVIDENCE_SNAPSHOT_FILENAME
    evidence_details["match_policy"] = DEFAULT_MATCH_POLICY

    contract = _input_contract(selected, selected_by_bucket, bounds.get("controls", {}))
    capabilities = {
        "catalog_screening": bool(catalog_details.get("available") and catalog_details.get("eligible_candidate_count", 0) > 0 and not contract["unsupported_selected_features"]),
        "database_evidence_lookup": bool(evidence_details.get("available")),
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
    for name, filename in manifest.get("artifacts", {}).items():
        relative = Path(str(filename))
        if relative.is_absolute() or len(relative.parts) != 1 or not (bundle_dir / relative).exists():
            raise FileNotFoundError(f"Invalid or missing bundle artifact {name!r}: {filename!r}")
    return manifest
