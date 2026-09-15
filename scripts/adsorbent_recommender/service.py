"""Backend service for a trained adsorbent recommender bundle; it has no UI."""

from __future__ import annotations

import json
import os
import re
import sys
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path, PureWindowsPath
from typing import Any

import joblib
import numpy as np
import pandas as pd

from adsorbent_recommender.catalog import clean_text, present
from ML_model_training.backend.logkd_features import build_X
from ML_model_training.backend.logkd_config import (
    ADSORBENT_SPECIFIC_FEATURE_APPLICABILITY,
    MODEL_CATEGORIES,
)
from adsorbent_recommender.pfas_cache import PfasFeatureStore
from adsorbent_recommender.bundle import load_bundle_manifest
from adsorbent_recommender.evidence import (
    DEFAULT_EVIDENCE_PATH,
    DEFAULT_EVIDENCE_SHEET,
    DEFAULT_MATCH_POLICY,
    load_reference_dose_evidence,
    match_candidate_evidence,
)


SCRIPT_DIR = Path(__file__).resolve().parent
DB_DIR = SCRIPT_DIR.parent / "DB"
ML_DIR = SCRIPT_DIR.parent / "ML_model_training"
DEFAULT_RECOMMENDATION_COUNT = 20
if str(DB_DIR) not in sys.path:
    sys.path.insert(0, str(DB_DIR))
if str(ML_DIR) not in sys.path:
    sys.path.insert(0, str(ML_DIR))
from normalization_rules import (  # noqa: E402
    apply_default_water_context,
    compute_ionic_strength,
    materialize_additive_ml_features,
    materialize_organic_carbon,
    normalize_water_quality_scalar_fields,
    standardize_water_type,
)


# Curated catalog fields shown to users. Model features retain their standardized
# names internally; this registry provides readable labels and prevents
# family-specific properties from appearing for chemically unrelated materials.
ADSORBENT_PROPERTY_SPECS: tuple[dict[str, Any], ...] = (
    {"group": "Identity", "label": "Adsorbent subcategory", "sources": ("adsorbent_subcategory", "Adsorbent_subcategory"), "model_features": ("adsorbent_subcategory",)},
    {"group": "Composition", "label": "AC raw material", "sources": ("AC_raw_material",), "model_features": ("AC_raw_material",), "categories": ("Activated carbon",)},
    {"group": "Composition", "label": "AC activation method", "sources": ("AC_activation_method",), "model_features": ("AC_activation_method",), "categories": ("Activated carbon",)},
    {"group": "Composition", "label": "Polymer matrix", "sources": ("polymer_matrix", "Polymer_matrix"), "model_features": ("polymer_matrix",), "categories": ("Ion exchange resin", "Nonionic resin")},
    {"group": "Composition", "label": "Functional group", "sources": ("functional_group", "Functional_group"), "model_features": ("contains_quaternary_ammonium", "contains_sulfonic_acid", "contains_carboxyl", "contains_hydroxyl", "contains_pyridinic_nitrogen", "contains_amino_acid", "contains_amine", "contains_ester", "contains_benzyl_chloride")},
    {"group": "Porosity", "label": "Pore class", "sources": ("pore_class", "Pore_class"), "model_features": ("contains_microporous", "contains_mesoporous", "contains_macroporous", "contains_gel", "contains_nonporous")},
    {"group": "Capacity", "label": "Ion-exchange capacity", "unit": "meq/L", "sources": ("ion_exchange_capacity_value_meq/L",), "model_features": ("ion_exchange_capacity_value_meq/L",), "categories": ("Ion exchange resin",)},
    {"group": "Capacity", "label": "Ion-exchange capacity", "unit": "meq/g", "sources": ("ion_exchange_capacity_value_meq/g",), "model_features": ("ion_exchange_capacity_value_meq/g",), "categories": ("Ion exchange resin",)},
    {"group": "Capacity", "label": "Iodine number", "unit": "mg/g", "sources": ("iodine_number_mg/g",), "model_features": ("iodine_number_mg/g",), "categories": ("Activated carbon",)},
    {"group": "Physical", "label": "Nominal grain size", "unit": "mm", "sources": ("Nominal_grain_size_value_mm",), "model_features": ("Nominal_grain_size_value_mm",)},
    {"group": "Physical", "label": "Specific surface area", "unit": "m²/g", "sources": ("ssa_(m2/g)_avg", "SSA_(m2/g)"), "model_features": ("ssa_(m2/g)_avg",)},
    {"group": "Porosity", "label": "Average pore diameter", "unit": "Å", "sources": ("average_pore_diameter_(angstrom)", "Average_pore_diameter_(Angstrom)"), "model_features": ("average_pore_diameter_(angstrom)",)},
    {"group": "Porosity", "label": "Total pore volume", "unit": "cm³/g", "sources": ("pore_volume_total_value_cm3/g", "Total_pore_volume_(cm3/g)"), "model_features": ("pore_volume_total_value_cm3/g",)},
    {"group": "Porosity", "label": "Micropore volume", "unit": "cm³/g", "sources": ("pore_volume_micro_value_avg_cm3/g", "Micro_pore_volume_(cm3/g)"), "model_features": ("pore_volume_micro_value_avg_cm3/g",)},
    {"group": "Porosity", "label": "Mesopore volume", "unit": "cm³/g", "sources": ("pore_volume_meso_value_avg_cm3/g", "Meso_pore_volume_(cm3/g)"), "model_features": ("pore_volume_meso_value_avg_cm3/g",)},
    {"group": "Porosity", "label": "Macropore volume", "unit": "cm³/g", "sources": ("pore_volume_macro_value_avg_cm3/g", "Macro_pore_volume_(cm3/g)"), "model_features": ("pore_volume_macro_value_avg_cm3/g",)},
    {"group": "Surface chemistry", "label": "pH at point of zero charge", "sources": ("phpzc", "pHPZC"), "model_features": ("phpzc",)},
    {"group": "Surface chemistry", "label": "Zeta potential", "unit": "mV", "sources": ("zeta_potential_value", "Zeta_potential_(mV)"), "model_features": ("zeta_potential_value",)},
    {"group": "Composition", "label": "Carbon fraction", "sources": ("element_C_value",), "model_features": ("element_C_value",)},
    {"group": "Composition", "label": "Nitrogen fraction", "sources": ("element_N_value",), "model_features": ("element_N_value",)},
    {"group": "Composition", "label": "Oxygen fraction", "sources": ("element_O_value",), "model_features": ("element_O_value",)},
)
RESULT_SCHEMA_VERSION = 4

# The adsorbent families a screening can be restricted to.  The catalog labels
# are read from the model configuration rather than repeated here, so a category
# renamed upstream cannot quietly stop matching the family it belongs to.
ADSORBENT_CATEGORY_GROUPS: tuple[dict[str, Any], ...] = (
    {"key": "AC", "label": "Activated carbon (AC)", "categories": tuple(MODEL_CATEGORIES["AC"])},
    {"key": "Resin", "label": "Resin", "categories": tuple(MODEL_CATEGORIES["Resin"])},
    {"key": "CDP", "label": "Cyclodextrin polymer (CDP)", "categories": tuple(MODEL_CATEGORIES["CDP"])},
)
_CATEGORY_GROUP_KEYS = {
    clean_text(category).casefold(): group["key"]
    for group in ADSORBENT_CATEGORY_GROUPS
    for category in group["categories"]
}


class BundleNotReadyError(RuntimeError):
    """Raised when a selected model run cannot support catalog recommendation."""


@dataclass
class RecommenderBundle:
    directory: Path
    manifest: dict[str, Any]
    config: dict[str, Any]
    models: list[Any]
    bounds: dict[str, Any]
    catalog: pd.DataFrame
    evidence: pd.DataFrame = field(default_factory=pd.DataFrame)
    evidence_info: dict[str, Any] = field(default_factory=dict)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_recommender_bundle(bundle_dir: Path) -> RecommenderBundle:
    """Load one completed factory output without retraining or model selection."""
    directory = Path(bundle_dir).expanduser().resolve()
    manifest = load_bundle_manifest(directory)
    if not manifest.get("capabilities", {}).get("catalog_screening"):
        raise BundleNotReadyError(manifest.get("status_reason", "This model bundle cannot screen the catalog."))
    artifacts = manifest["artifacts"]
    config = _read_json(directory / artifacts["run_config"])
    bounds = _read_json(directory / artifacts["feature_support"])
    packet = joblib.load(directory / artifacts["ensemble"])
    if not isinstance(packet, dict) or "models" not in packet:
        raise ValueError("The deployment ensemble has an unsupported format.")
    models = list(packet["models"])
    catalog_info = manifest.get("catalog", {})
    snapshot = catalog_info.get("snapshot_filename")
    eligibility_name = catalog_info.get("eligibility_filename")
    if not snapshot or not eligibility_name:
        raise BundleNotReadyError("The bundle has no catalog snapshot or eligibility audit.")
    catalog = pd.read_csv(directory / snapshot)
    eligibility = pd.read_csv(directory / eligibility_name)
    if "eligible" not in eligibility.columns:
        raise BundleNotReadyError("The bundle's catalog eligibility audit is invalid.")
    catalog = catalog.merge(eligibility, on="Product_key", how="left", validate="one_to_one")
    identity_name = catalog_info.get("identity_map_filename")
    if identity_name and (directory / identity_name).exists():
        identity = pd.read_csv(directory / identity_name)
        catalog = catalog.merge(identity, on="Product_key", how="left", validate="one_to_one")
    catalog = catalog[catalog["eligible"].fillna(False)].copy()
    if catalog.empty:
        raise BundleNotReadyError("The chosen model has no eligible catalog products.")
    evidence_info = dict(manifest.get("database_evidence", {}))
    evidence = pd.DataFrame()
    evidence_snapshot = evidence_info.get("snapshot_filename")
    if evidence_snapshot and (directory / evidence_snapshot).exists():
        evidence, loaded_info = load_reference_dose_evidence(directory / evidence_snapshot)
        evidence_info.update(loaded_info)
        evidence_info["loaded_from"] = "bundle_snapshot"
    else:
        evidence_path = Path(os.environ.get("ADSORBENT_EVIDENCE_PATH", DEFAULT_EVIDENCE_PATH))
        evidence_sheet = os.environ.get("ADSORBENT_EVIDENCE_SHEET", DEFAULT_EVIDENCE_SHEET)
        evidence, loaded_info = load_reference_dose_evidence(evidence_path, evidence_sheet)
        evidence_info.update(loaded_info)
        evidence_info["loaded_from"] = "runtime_file" if loaded_info.get("available") else "unavailable"
    return RecommenderBundle(directory, manifest, config, models, bounds, catalog, evidence, evidence_info)


def _group_key(category: Any) -> str | None:
    """Return the family a catalog category belongs to, if it belongs to one."""
    return _CATEGORY_GROUP_KEYS.get(clean_text(category).casefold())


def catalog_category_groups(bundle: RecommenderBundle) -> list[dict[str, Any]]:
    """Report each adsorbent family with the products this bundle could screen.

    Families with no eligible product are reported with a count of zero rather
    than dropped: which families exist is a property of the project, and a list
    that silently shortens itself reads as if the missing family were never part
    of it.  The counts let the caller say which choices are actually available.
    """
    categories = bundle.catalog.get("adsorbent_category")
    tally: dict[str, int] = {}
    if categories is not None:
        for value in categories:
            key = _group_key(value)
            if key:
                tally[key] = tally.get(key, 0) + 1
    return [
        {
            "key": group["key"],
            "label": group["label"],
            "categories": list(group["categories"]),
            "product_count": tally.get(group["key"], 0),
        }
        for group in ADSORBENT_CATEGORY_GROUPS
    ]


def _requested_category_keys(scenario: dict[str, Any]) -> tuple[list[str], list[str]]:
    """Read the requested adsorbent families from a scenario.

    A family may be named by its key ("AC") or by any catalog category it covers
    ("Ion exchange resin"), because the caller holding a catalog label should not
    have to translate it first.  Anything unrecognised is returned separately
    instead of being discarded, so the screening can say the request was not
    honoured in full rather than appear to have applied it.
    """
    adsorbent = scenario.get("adsorbent")
    requested = (adsorbent or {}).get("categories") if isinstance(adsorbent, dict) else None
    if requested is None:
        requested = scenario.get("adsorbent_categories")
    if requested is None:
        return [], []
    if isinstance(requested, str):
        requested = [requested]
    keys: list[str] = []
    unknown: list[str] = []
    for value in requested:
        text = clean_text(value)
        if not text:
            continue
        key = next(
            (group["key"] for group in ADSORBENT_CATEGORY_GROUPS if group["key"].casefold() == text.casefold()),
            None,
        ) or _group_key(text)
        if key is None:
            unknown.append(text)
        elif key not in keys:
            keys.append(key)
    return keys, unknown


def filter_catalog_by_categories(
    catalog: pd.DataFrame, keys: list[str]
) -> tuple[pd.DataFrame, list[str]]:
    """Restrict a catalog to the requested adsorbent families.

    An empty request means every family, which is what makes the filter additive:
    a caller that does not know about it screens exactly what it screened before.
    """
    if not keys:
        return catalog, []
    if "adsorbent_category" not in catalog.columns:
        return catalog, [
            "The catalog snapshot records no adsorbent category, so the adsorbent "
            "class selection could not be applied and every product was screened."
        ]
    wanted = set(keys)
    mask = catalog["adsorbent_category"].map(lambda value: _group_key(value) in wanted)
    return catalog[mask].copy(), []


def _first(mapping: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in mapping and mapping[key] is not None:
            return mapping[key]
    return None


def normalize_water_scenario(scenario: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    """Normalize UI water inputs using the project's shared database rules."""
    water = scenario.get("water", {}) or {}
    if not isinstance(water, dict):
        raise ValueError("scenario.water must be an object when supplied.")
    direct_ionic_strength = _first(water, "ionic_strength_mol_l", "ionic_strength_mol_L")
    direct_organic_carbon = _first(water, "organic_carbon_mg_l", "organic_carbon_mg_L")
    frame = pd.DataFrame(
        [{
            "Water_type": _first(water, "water_type", "type") or "",
            "pH": _first(water, "pH", "ph") or "",
            # A consolidated runtime value is equivalent to the highest-priority
            # reported source. Legacy DOC/TOC keys remain accepted.
            "DOC_(mg/L)": direct_organic_carbon if direct_organic_carbon is not None else (_first(water, "doc_mg_l", "DOC_mg_L") or ""),
            "TOC_(mg/L)": _first(water, "toc_mg_l", "TOC_mg_L") or "",
            "TDS_(mg/L)": _first(water, "tds_mg_l", "TDS_mg_L") or "",
            "Ionic_strength": _first(water, "ionic_strength", "ionic_strength_with_unit") or "",
            "Ionic_strength_(mol/L)": direct_ionic_strength if direct_ionic_strength is not None else "",
            "Inorganic_matter": _first(water, "inorganic_matter", "added_inorganic_matter", "inorganic_additives") or "",
            "Organic_matter": _first(water, "organic_matter", "added_organic_matter", "organic_additives") or "",
        }]
    )
    standardize_water_type(frame)
    apply_default_water_context(frame)
    normalize_water_quality_scalar_fields(frame)
    materialize_organic_carbon(frame)
    compute_ionic_strength(frame)
    materialize_additive_ml_features(frame)
    row = frame.iloc[0].to_dict()
    if direct_ionic_strength is not None:
        row["Ionic_strength_(mol/L)"] = direct_ionic_strength
    ions = water.get("ions", {}) or {}
    if isinstance(ions, dict):
        for ion in ("Na", "K", "Ca", "Mg", "Cl", "HCO3", "SO4", "phosphate"):
            if ion in ions:
                row[f"contains_{ion}"] = bool(ions[ion])
    warnings = []
    if not any(clean_text(value) not in {"", "0", "False"} for value in row.values()):
        warnings.append("No water chemistry was supplied; missing water values are preserved and reported.")
    return row, warnings


def _scenario_identity(scenario: dict[str, Any]) -> tuple[str, str]:
    pfas = scenario.get("pfas") if isinstance(scenario.get("pfas"), dict) else {}
    identity = clean_text(_first(pfas, "identity", "value") or scenario.get("pfas_smiles"))
    identity_type = clean_text(_first(pfas, "identity_type", "type") or ("smiles" if scenario.get("pfas_smiles") else "smiles")).casefold()
    if not identity:
        raise ValueError("A PFAS identity is required.")
    return identity, identity_type


def pfas_features_source(bundle: RecommenderBundle) -> tuple[Path, str]:
    """Locate the PFAS feature cache the bundle was trained against.

    run_config records the absolute path on the training machine.  A hosted copy
    of the app has no such path, so it falls back to ADSORBENT_PFAS_FEATURES_PATH
    and then to a workbook of the same name shipped inside the bundle directory.
    """
    pfas_config = bundle.config.get("pfas_features", {})
    configured = Path(pfas_config.get("path", ""))
    sheet_name = pfas_config.get("sheet_name", "")
    # The recorded path is a Windows path; on a POSIX host its backslashes are
    # not separators, so the file name is taken with Windows parsing rules.
    filename = PureWindowsPath(pfas_config.get("path", "")).name
    for candidate in (
        configured,
        Path(os.environ.get("ADSORBENT_PFAS_FEATURES_PATH", "")),
        bundle.directory / filename,
    ):
        if candidate.name and candidate.is_file():
            return candidate, sheet_name
    return configured, sheet_name


def resolve_pfas(bundle: RecommenderBundle, scenario: dict[str, Any], rdkit_python: Path | None = None):
    identity, identity_type = _scenario_identity(scenario)
    path, sheet_name = pfas_features_source(bundle)
    store = PfasFeatureStore(path, sheet_name, rdkit_python)
    if identity_type == "smiles":
        return store.lookup(identity, scenario.get("pfas_abbreviation")), store.cache_version
    return store.lookup_identifier(identity, identity_type), store.cache_version


def _feature_aliases(feature: str) -> tuple[str, ...]:
    aliases = {
        "PFAS_C0_value_mg/L": ("initial_concentration_mg_l",),
        "organic_carbon_mg/L": ("organic_carbon_mg/L",),
        "TDS_(mg/L)": ("TDS_(mg/L)",),
        "Ionic_strength_(mol/L)": ("Ionic_strength_(mol/L)",),
    }
    if feature.startswith("Temperature_("):
        return ("temperature_c", feature)
    return aliases.get(feature, (feature,))


def _align_categorical_value(feature: str, value: Any, support: dict[str, Any]) -> Any:
    """Use a documented equivalent catalog/cache token when one is unambiguous.

    This does not impute a category: it only returns a level already observed
    during model development.  It also makes numeric Excel values such as 0
    match the training token ``0.0``.
    """
    if support.get("kind") != "categorical" or pd.isna(value) or clean_text(value) == "":
        return value
    levels = [clean_text(level) for level in support.get("levels", [])]
    text = clean_text(value)
    if text in levels:
        return text
    casefold = {level.casefold(): level for level in levels}
    if text.casefold() in casefold:
        return casefold[text.casefold()]
    numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.notna(numeric):
        for level in levels:
            level_number = pd.to_numeric(pd.Series([level]), errors="coerce").iloc[0]
            if pd.notna(level_number) and float(level_number) == float(numeric):
                return level
    if feature == "adsorbent_subcategory":
        without_abbreviation = re.sub(r"\s*\([^)]*(?:gac|pac)[^)]*\)", "", text, flags=re.IGNORECASE).strip()
        normalized = re.sub(r"[^a-z0-9]+", "", without_abbreviation.casefold())
        for level in levels:
            if re.sub(r"[^a-z0-9]+", "", level.casefold()) == normalized:
                return level
    return text


def _clear_family_inapplicable_values(row: dict[str, Any]) -> list[str]:
    """Express family-inapplicable properties the way the training data did.

    The training database records, for example, a nonionic resin's ion-exchange
    capacity as the literal "not applicable", which reads as missing, while the
    product catalog records the same fact as a numeric 0.  Passing the 0 to the
    model would describe the material differently than every row it was fitted
    on, and build_X rejects it outright as contradicting the configured family
    rule.  A zero is therefore converted to the missing representation; a
    non-zero value is a genuine disagreement with the rule and is reported
    instead of being dropped without notice.
    """
    category = clean_text(row.get("adsorbent_category"))
    if category not in set(MODEL_CATEGORIES["Global"]):
        return []
    warnings: list[str] = []
    for feature, applicable_categories in ADSORBENT_SPECIFIC_FEATURE_APPLICABILITY.items():
        if feature not in row or category in applicable_categories:
            continue
        value = row[feature]
        if not isinstance(value, (list, dict)) and pd.isna(value):
            continue
        if clean_text(value) == "":
            continue
        number = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
        if pd.isna(number) or float(number) != 0.0:
            warnings.append(
                f"The catalog reports {feature} as {clean_text(value)!r} for a "
                f"{category}, which the configured family rule says cannot have it. "
                "The value was not used for this prediction."
            )
        row[feature] = np.nan
    return warnings


def _model_row(
    bundle: RecommenderBundle,
    lookup_row: dict[str, Any],
    scenario: dict[str, Any],
    water: dict[str, Any],
    material: dict[str, Any],
) -> tuple[dict[str, Any], list[str]]:
    features = bundle.manifest["features"]["selected"]
    row: dict[str, Any] = {feature: np.nan for feature in features}
    values = dict(lookup_row)
    values.update(water)
    values["initial_concentration_mg_l"] = scenario["initial_concentration_mg_l"]
    values["temperature_c"] = scenario.get("temperature_c")
    values.update(material)
    controls = bundle.bounds.get("controls", {})
    missing_inputs: list[str] = []
    contract_by_feature = {field["model_feature"]: field for field in bundle.manifest["input_contract"]["fields"]}
    for feature in features:
        for alias in _feature_aliases(feature):
            if alias in values:
                row[feature] = values[alias]
                break
        control = controls.get(feature)
        if control and control.get("role") == "fixed" and pd.isna(row[feature]):
            row[feature] = control["default"]
        field = contract_by_feature.get(feature, {})
        if field.get("source") in {"scenario", "water", "water_ion", "derived_water"} and (pd.isna(row[feature]) or clean_text(row[feature]) == ""):
            missing_inputs.append(feature)
        row[feature] = _align_categorical_value(feature, row[feature], bundle.bounds.get("feature_support", {}).get(feature, {}))
    applicability_warnings = _clear_family_inapplicable_values(row)
    return row, [f"Missing scenario value: {feature}" for feature in missing_inputs] + applicability_warnings


def _input_support_warnings(row: dict[str, Any], bounds: dict[str, Any]) -> list[str]:
    """Report development-support differences; do not suppress a prediction."""
    warnings: list[str] = []
    for feature, support in bounds.get("feature_support", {}).items():
        value = row.get(feature)
        if pd.isna(value) or clean_text(value) == "":
            continue
        if support.get("kind") == "numeric":
            number = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
            if pd.notna(number) and (number < support.get("hard_min", -np.inf) or number > support.get("hard_max", np.inf)):
                warnings.append(f"{feature} is outside the development range [{support['hard_min']}, {support['hard_max']}].")
        elif support.get("kind") == "categorical" and clean_text(value) not in set(support.get("levels", [])):
            warnings.append(f"{feature} has a category not observed during development: {clean_text(value)!r}.")
    return warnings


def predict_logkd(bundle: RecommenderBundle, row: dict[str, Any]) -> dict[str, float]:
    features = bundle.manifest["features"]
    frame = pd.DataFrame([row])
    X = build_X(frame, features["selected"], features["numeric"], features["categorical"])
    values = np.asarray([float(model.predict(X)[0]) for model in bundle.models], dtype=float)
    mean, spread = float(values.mean()), float(values.std(ddof=0))
    return {"mean_logKd": mean, "ensemble_sd": spread}


def fixed_test_conditions(
    bundle: RecommenderBundle,
    scenario: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Describe the conditions every candidate was scored at.

    A recommender ranks materials against one another, so the controls it does
    not search are held common across candidates.  Reporting them is not
    optional detail: a ranking is only interpretable alongside the conditions
    that produced it.
    """
    controls = bundle.bounds.get("controls", {})
    fixed = {
        feature: control.get("default")
        for feature, control in controls.items()
        if control.get("role") == "fixed"
    }
    scenario = scenario or {}
    temperature = scenario.get("temperature_c")
    if temperature is not None:
        for feature in controls:
            if feature.startswith("Temperature_("):
                fixed[feature] = float(temperature)
    dose_feature = "Adsorbent_dosage_value_mg/L"
    dose = controls.get(dose_feature, {})
    return {
        "status": "fixed_conditions",
        "fixed_conditions": fixed,
        "dose_mg_L": fixed.get(dose_feature),
        "dose_within_development_support": dose.get("default_within_observed_support"),
        "basis": "every candidate is scored at the same conditions so the ranking compares affinity",
    }


def _material_characterization(material: pd.Series) -> dict[str, Any]:
    """Report how completely a product is characterized, and how it was matched.

    Eligibility is a floor, not a guarantee: products clear it with very
    different amounts of measured data, and one whose identity is absent from
    the training adsorbents is an extrapolation to an unseen material rather
    than a new observation of a familiar one.  Both belong beside the score.
    """
    present_count = material.get("present_material_feature_count")
    required_count = material.get("required_material_feature_count")
    minimum_count = material.get("minimum_present_material_features")
    warnings: list[str] = []
    if pd.notna(present_count) and pd.notna(minimum_count) and float(present_count) <= float(minimum_count):
        warnings.append(
            f"{material.get('Product_key')} is characterized on {int(present_count)} of "
            f"{int(required_count)} material features, at the eligibility floor."
        )
    if clean_text(material.get("mapping_status")) != "exact_normalized_key":
        warnings.append(
            f"{material.get('Product_key')} does not match a training adsorbent by identity, "
            "so its prediction extrapolates to an unseen material."
        )
    return {
        "material_features_present": None if pd.isna(present_count) else int(present_count),
        "material_features_required": None if pd.isna(required_count) else int(required_count),
        "material_features_minimum": None if pd.isna(minimum_count) else int(minimum_count),
        "missing_material_features": material.get("missing_required_material_features"),
        "seen_in_training": clean_text(material.get("mapping_status")) == "exact_normalized_key",
        "warnings": warnings,
    }


def _adsorbent_properties(material: pd.Series, selected_features: list[str]) -> list[dict[str, Any]]:
    """Return available, category-relevant catalog properties for presentation."""
    category = clean_text(material.get("adsorbent_category") or material.get("Adsorbent_category"))
    selected = set(selected_features)
    properties: list[dict[str, Any]] = []
    for spec in ADSORBENT_PROPERTY_SPECS:
        if spec.get("categories") and category not in spec["categories"]:
            continue
        source = next((name for name in spec["sources"] if present(material.get(name))), None)
        if source is None:
            continue
        value = material.get(source)
        if isinstance(value, np.generic):
            value = value.item()
        used_features = [feature for feature in spec["model_features"] if feature in selected]
        properties.append({
            "group": spec["group"],
            "property": spec["label"],
            "value": value,
            "unit": spec.get("unit", ""),
            "used_by_selected_model": bool(used_features),
            "model_features": used_features,
        })
    return properties


def _select_diverse(candidates: list[dict[str, Any]], count: int) -> list[dict[str, Any]]:
    # Asking for the complete result set is an explicit request for the true
    # global ranking.  The diversity policy is useful when curating a short
    # recommendation list, but applying it to every catalog product needlessly
    # reshuffles that complete ranking before the UI applies view-only filters.
    if count >= len(candidates):
        return list(candidates)
    selected, family_counts = [], {}
    for candidate in candidates:
        family = clean_text(candidate["product"].get("adsorbent_subcategory")) or "unclassified"
        if family_counts.get(family, 0) >= 2:
            continue
        selected.append(candidate)
        family_counts[family] = family_counts.get(family, 0) + 1
        if len(selected) == count:
            return selected
    return selected + [candidate for candidate in candidates if candidate not in selected][: max(0, count - len(selected))]


def recommend(
    bundle: RecommenderBundle,
    scenario: dict[str, Any],
    *,
    top_k: int = DEFAULT_RECOMMENDATION_COUNT,
    rdkit_python: Path | None = None,
) -> dict[str, Any]:
    """Screen the chosen bundle's catalog and return a stable result payload."""
    if scenario.get("initial_concentration_mg_l") is None:
        raise ValueError("scenario.initial_concentration_mg_l is required.")
    lookup, cache_version = resolve_pfas(bundle, scenario, rdkit_python)
    response: dict[str, Any] = {
        "result_schema_version": RESULT_SCHEMA_VERSION,
        "status": "pending",
        "bundle": {"directory": str(bundle.directory), "model_family": bundle.manifest["model_family"], "target": bundle.manifest["target"]},
        "pfas_lookup": {"status": lookup.status, "canonical_smiles": lookup.canonical_smiles, "cache_key": lookup.cache_key, "warnings": list(lookup.warnings)},
        "pfas_cache": cache_version,
        "scenario": scenario,
    }
    if lookup.status != "cached":
        response.update({"status": "pfas_not_ready", "offline_feature_request": lookup.offline_request})
        return response
    water, water_warnings = normalize_water_scenario(scenario)
    # Equilibrium comparisons use one common set of conditions for every candidate.
    shared_conditions = fixed_test_conditions(bundle, scenario)
    # Restricting the adsorbent families narrows what is screened rather than
    # what is reported: a product outside the requested classes is never scored,
    # so it cannot occupy a rank the user asked not to see.
    requested_keys, unknown_categories = _requested_category_keys(scenario)
    catalog, scope_warnings = filter_catalog_by_categories(bundle.catalog, requested_keys)
    if unknown_categories:
        scope_warnings.append(
            "Unrecognised adsorbent class(es) were ignored: " + ", ".join(unknown_categories) + "."
        )
    results, rejected = [], []
    for _, material in catalog.iterrows():
        model_row, input_warnings = _model_row(bundle, lookup.feature_row or {}, scenario, water, material.to_dict())
        support_warnings = _input_support_warnings(model_row, bundle.bounds)
        prediction = predict_logkd(bundle, model_row)
        database_evidence = match_candidate_evidence(
            bundle.evidence,
            material=material.to_dict(),
            pfas_feature_row=lookup.feature_row or {},
            pfas_cache_key=lookup.cache_key,
            scenario=scenario,
            normalized_water=water,
            screening_conditions=shared_conditions,
            policy=bundle.manifest.get("database_evidence", {}).get("match_policy", DEFAULT_MATCH_POLICY),
        )
        if database_evidence.get("status") == "matched":
            performance = database_evidence["performance"]
            score = float(performance["mean_logKd"])
            performance_source = "database_match"
        else:
            performance = {"source": "model_prediction", **prediction}
            score = prediction["mean_logKd"]
            performance_source = "model_prediction"
        characterization = _material_characterization(material)
        adsorbent_properties = _adsorbent_properties(
            material,
            bundle.manifest.get("features", {}).get("selected", []),
        )
        results.append({
            "product": {
                "Product_key": material["Product_key"], "Name_Commercial": material.get("Name_Commercial"),
                "adsorbent_category": material.get("adsorbent_category"),
                "adsorbent_subcategory": material.get("adsorbent_subcategory"), "AC_raw_material": material.get("AC_raw_material"),
                "catalog_match_status": material.get("catalog_match_status"), "catalog_data_sources": material.get("catalog_data_sources"),
                "mapping_status": material.get("mapping_status"),
            },
            "adsorbent_properties": adsorbent_properties,
            "characterization": characterization,
            "prediction": prediction,
            "performance": performance,
            "performance_source": performance_source,
            "database_evidence": database_evidence,
            "screening_conditions": shared_conditions,
            "ranking_score": score,
            "warnings": water_warnings + input_warnings + support_warnings + characterization["warnings"],
        })
    ranked = sorted(results, key=lambda item: item["ranking_score"], reverse=True)
    recommendations = _select_diverse(ranked, max(1, int(top_k)))
    for rank, recommendation in enumerate(recommendations, start=1):
        recommendation["rank"] = rank
    database_matches = sum(item["performance_source"] == "database_match" for item in results)
    response.update({
        "status": "recommendations_ready" if recommendations else "no_supported_candidates",
        "recommendations": recommendations,
        "rejected_candidates": rejected,
        "screening_conditions": shared_conditions,
        "candidates_screened": int(len(catalog)),
        "screening_scope": {
            # An empty request is the unrestricted screening, stated explicitly
            # so a reader of the payload need not infer it from a missing key.
            "requested_categories": requested_keys,
            "selected_categories": [
                group["label"]
                for group in ADSORBENT_CATEGORY_GROUPS
                if group["key"] in requested_keys
            ],
            "unrecognized_categories": unknown_categories,
            "catalog_product_count": int(len(bundle.catalog)),
            "screened_product_count": int(len(catalog)),
            "warnings": scope_warnings,
        },
        "evidence_routing": {
            "mode": "database_first_per_candidate",
            "database_matched_candidates": int(database_matches),
            "model_fallback_candidates": int(len(results) - database_matches),
            "database": bundle.evidence_info,
            "message": (
                "Database evidence was used wherever exact identities and all condition tolerances matched; "
                "the final model was used only for unmatched candidates."
                if database_matches
                else "No database row met the exact identity gates and condition tolerances; all candidates use the final model."
            ),
        },
    })
    return response


def recommend_mixture(
    bundle: RecommenderBundle,
    scenarios: list[dict[str, Any]],
    *,
    top_k: int = DEFAULT_RECOMMENDATION_COUNT,
    rdkit_python: Path | None = None,
) -> dict[str, Any]:
    """Rank products across every PFAS in one shared-water mixture.

    Each PFAS/concentration pair is routed independently through the same
    database-first logic as a single-compound screen.  Products are then ranked
    by their lowest logKd across the mixture (a maximin rule), so a high score
    for one PFAS cannot hide weak removal of another.
    """
    if not scenarios:
        raise ValueError("At least one PFAS scenario is required.")

    # Request the full eligible catalog for every PFAS.  Aggregating separate
    # top-k lists would silently omit a product that is merely mid-ranked for
    # one PFAS but is the best balanced choice across the complete mixture.
    all_count = max(1, int(len(bundle.catalog)))
    runs = [
        recommend(bundle, scenario, top_k=all_count, rdkit_python=rdkit_python)
        for scenario in scenarios
    ]
    lookups = [run.get("pfas_lookup", {}) for run in runs]
    response: dict[str, Any] = {
        "result_schema_version": RESULT_SCHEMA_VERSION,
        "status": "pending",
        "bundle": runs[0].get("bundle"),
        "scenarios": scenarios,
        "pfas_lookups": lookups,
        "mixture": {
            "pfas_count": len(scenarios),
            "aggregation": "maximin_logKd",
            "description": "Ranked by each adsorbent's lowest logKd across the PFAS mixture.",
            "competitive_adsorption_modeled": False,
        },
    }

    incomplete = [index for index, run in enumerate(runs) if run.get("status") == "pfas_not_ready"]
    if incomplete:
        response.update({
            "status": "pfas_not_ready",
            "unresolved_pfas": [
                {
                    "input": scenarios[index].get("pfas", {}),
                    "lookup": runs[index].get("pfas_lookup", {}),
                    "offline_feature_request": runs[index].get("offline_feature_request"),
                }
                for index in incomplete
            ],
        })
        return response

    nonready = [run for run in runs if run.get("status") != "recommendations_ready"]
    if nonready:
        first = nonready[0]
        response.update({
            "status": first.get("status", "no_supported_candidates"),
            "screening_scope": first.get("screening_scope", {}),
            "rejected_candidates": first.get("rejected_candidates", []),
        })
        return response

    by_run = [
        {item["product"]["Product_key"]: item for item in run["recommendations"]}
        for run in runs
    ]
    common_keys = set(by_run[0])
    for mapping in by_run[1:]:
        common_keys.intersection_update(mapping)

    combined: list[dict[str, Any]] = []
    for product_key in common_keys:
        source_items = [mapping[product_key] for mapping in by_run]
        scores = [float(item["ranking_score"]) for item in source_items]
        limiting_index = int(np.argmin(scores))
        limiting_item = source_items[limiting_index]
        per_pfas = []
        for scenario, run, item in zip(scenarios, runs, source_items):
            lookup = run.get("pfas_lookup", {})
            per_pfas.append({
                "pfas": lookup.get("cache_key") or scenario.get("pfas", {}).get("identity"),
                "input_identity": scenario.get("pfas", {}).get("identity"),
                "identity_type": scenario.get("pfas", {}).get("identity_type"),
                "initial_concentration_mg_l": scenario.get("initial_concentration_mg_l"),
                "performance": item.get("performance"),
                "performance_source": item.get("performance_source"),
                "prediction": item.get("prediction"),
                "database_evidence": item.get("database_evidence"),
                "warnings": item.get("warnings", []),
            })

        source_types = {item.get("performance_source") for item in source_items}
        if source_types == {"database_match"}:
            aggregate_source = "database_match"
        elif source_types == {"model_prediction"}:
            aggregate_source = "model_prediction"
        else:
            aggregate_source = "mixed_evidence"

        candidate = deepcopy(limiting_item)
        candidate.update({
            "performance_source": aggregate_source,
            "ranking_score": scores[limiting_index],
            "mixture_performance": {
                "worst_case_logKd": scores[limiting_index],
                "mean_logKd": float(np.mean(scores)),
                "limiting_pfas": per_pfas[limiting_index]["pfas"],
                "database_supported_pfas": sum(
                    row["performance_source"] == "database_match" for row in per_pfas
                ),
                "pfas_count": len(per_pfas),
            },
            "pfas_results": per_pfas,
            "warnings": list(dict.fromkeys(
                warning
                for item in source_items
                for warning in item.get("warnings", [])
            )),
        })
        combined.append(candidate)

    ranked = sorted(combined, key=lambda item: item["ranking_score"], reverse=True)
    recommendations = _select_diverse(ranked, max(1, int(top_k)))
    for rank, recommendation in enumerate(recommendations, start=1):
        recommendation["rank"] = rank

    database_pairs = sum(
        item.get("performance_source") == "database_match"
        for run in runs
        for item in run.get("recommendations", [])
    )
    total_pairs = sum(len(run.get("recommendations", [])) for run in runs)
    first = runs[0]
    response.update({
        "status": "recommendations_ready" if recommendations else "no_supported_candidates",
        "recommendations": recommendations,
        "rejected_candidates": first.get("rejected_candidates", []),
        "screening_conditions": first.get("screening_conditions", {}),
        "candidates_screened": first.get("candidates_screened", 0),
        "screening_scope": first.get("screening_scope", {}),
        "evidence_routing": {
            "mode": "database_first_per_candidate_pfas_pair",
            "database_matched_pairs": int(database_pairs),
            "model_fallback_pairs": int(total_pairs - database_pairs),
            "total_candidate_pfas_pairs": int(total_pairs),
            "database": first.get("evidence_routing", {}).get("database", {}),
            "message": (
                "Each candidate-PFAS pair used comparable database evidence where available; "
                "the final model filled unmatched pairs."
            ),
        },
    })
    return response
