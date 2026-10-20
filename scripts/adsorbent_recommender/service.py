"""Backend service for a trained adsorbent recommender bundle; it has no UI."""

from __future__ import annotations

import json
import hashlib
import os
import re
import sys
from copy import deepcopy
from dataclasses import dataclass, field, replace
from functools import lru_cache
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from adsorbent_recommender.catalog import clean_text, present
from adsorbent_recommender.water_support import load_support as load_water_support, diagnostics as water_diagnostics, caveats as water_caveats
from adsorbent_recommender.input_validation import effective_temperature, validate_scenario_numbers, validate_water_numbers
from adsorbent_recommender.additive_inputs import ADDITIVE_INPUT_VERSION, database_additive_text, validate_additives
from ML_model_training.backend.logkd_features import build_X
from ML_model_training.backend.logkd_config import (
    ADSORBENT_SPECIFIC_FEATURE_APPLICABILITY,
    MODEL_CATEGORIES,
)
from adsorbent_recommender.pfas_cache import PfasFeatureStore
from adsorbent_recommender.pfas_descriptors import DESCRIPTOR_VERSION
from rdkit import rdBase
from adsorbent_recommender.pfas_support import describe_support, UNSEEN_WARNING
from adsorbent_recommender.bundle import load_bundle_manifest
from adsorbent_recommender.material_support import (
    describe as describe_material_support, include_exploratory, policy_for, validate_table,
)
from adsorbent_recommender.evidence import (
    DEFAULT_EVIDENCE_PATH,
    DEFAULT_EVIDENCE_SHEET,
    DEFAULT_MATCH_POLICY,
    NUMERIC_FIELDS,
    candidate_keys,
    load_reference_dose_evidence,
    match_candidate_evidence,
    pfas_keys,
    requested_conditions,
)


SCRIPT_DIR = Path(__file__).resolve().parent
DB_DIR = SCRIPT_DIR.parent / "DB"
ML_DIR = SCRIPT_DIR.parent / "ML_model_training"
DEFAULT_RECOMMENDATION_COUNT = 20
LEADING_GROUP_MARGIN_FILENAME = "leading_group_margin.json"
if str(DB_DIR) not in sys.path:
    sys.path.insert(0, str(DB_DIR))
if str(ML_DIR) not in sys.path:
    sys.path.insert(0, str(ML_DIR))
from adsorbent_consolidation import (  # noqa: E402
    normalize_ac_raw_materials,
    normalize_adsorbent_subcategories,
    normalize_polymer_matrix,
)
from normalization_rules import (  # noqa: E402
    apply_default_water_context,
    apply_synthetic_water_from_reported_additives,
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
RESULT_SCHEMA_VERSION = 5

# Two tiers.  A critical caveat changes how a score should be read and is shown
# with the ranking; an informational one describes the evidence and is kept for
# the candidate details and the JSON.  General limits belong in the app's
# reading guide, not here.  Critical caveats sort first.
CAVEAT_SEVERITIES = ("critical", "info")
# Test predictions were least accurate outside this logKd range (L/g) in the
# split-strategy analysis, where the training data thin out.
WELL_SUPPORTED_LOGKD = (0.0, 4.0)
# Matched database values that span more than this many log units disagree too
# much to stand for one value without comment.
DATABASE_SPREAD_LIMIT_LOG_UNITS = 1.0
# Database conditions a user can leave blank, in the words the app uses.
DATABASE_CONDITION_LABELS = {
    "initial_concentration_mg_L": "C0",
    "pH": "pH",
    "temperature_C": "temperature",
    "organic_carbon_mg_L": "organic carbon",
    "tds_mg_L": "TDS",
    "ionic_strength_mol_L": "ionic strength",
}

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
    training_support: pd.DataFrame = field(default_factory=pd.DataFrame)
    training_support_info: dict[str, Any] = field(default_factory=dict)
    material_support: pd.DataFrame = field(default_factory=pd.DataFrame)
    material_support_info: dict[str, Any] = field(default_factory=dict)
    water_support: dict[str, Any] = field(default_factory=dict)


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
    catalog = canonicalize_catalog_categories(catalog)
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
    training_info = dict(manifest.get("training_support", {}))
    margin_path = directory / LEADING_GROUP_MARGIN_FILENAME
    if margin_path.is_file():
        margin = json.loads(margin_path.read_text(encoding="utf-8"))
        training_info["leading_group_margin"] = float(margin["difference_rmse"])
        training_info["leading_group_margin_basis"] = margin["basis"]
    training_support = pd.DataFrame()
    training_snapshot = training_info.get("snapshot_filename")
    if training_snapshot and (directory / training_snapshot).exists():
        training_support = pd.read_csv(directory / training_snapshot, dtype={"key": str}, keep_default_na=False)
    material_info = dict(manifest.get("adsorbent_support", {}))
    material_support = pd.DataFrame()
    material_snapshot = material_info.get("snapshot_filename")
    if material_info.get("available") and material_snapshot:
        material_path = directory / material_snapshot
        if material_info.get("snapshot_sha256") != hashlib.sha256(material_path.read_bytes()).hexdigest():
            raise ValueError("The adsorbent support snapshot does not match its bundle fingerprint. Repackage the bundle.")
        material_support = pd.read_csv(material_path)
        validate_table(material_support)
        policy_for(material_info)
    return RecommenderBundle(
        directory, manifest, config, models, bounds, catalog, evidence, evidence_info,
        training_support, training_info, material_support, material_info, load_water_support(directory, manifest),
    )


# Catalog columns the database canonicalizes before model training.
CANONICALIZED_CATALOG_COLUMNS = ("AC_raw_material", "adsorbent_subcategory", "polymer_matrix")


def canonicalize_catalog_categories(catalog: pd.DataFrame) -> pd.DataFrame:
    """Name catalog categories the way the training records name them.

    The model learned its categories from database records that passed through
    the database's model-facing canonicalization; the product catalog did not.
    A resin listed as "styrene-divinylbenzene" therefore reached the model as a
    category it had never seen instead of the training level
    "Polystyrene-divinylbenzene (PS-DVB)".  The database's own functions are
    applied here so that both describe a material in one vocabulary.  Cells
    the catalog leaves blank stay blank.
    """
    columns = [column for column in CANONICALIZED_CATALOG_COLUMNS if column in catalog.columns]
    if not columns or "adsorbent_category" not in catalog.columns:
        return catalog
    work = catalog[["adsorbent_category", *columns]].copy()
    if "AC_raw_material" in columns:
        normalize_ac_raw_materials(work)
    if "adsorbent_subcategory" in columns:
        normalize_adsorbent_subcategories(work)
    if "polymer_matrix" in columns:
        work["polymer_matrix"] = [
            normalize_polymer_matrix(value, category)
            for value, category in zip(work["polymer_matrix"], work["adsorbent_category"])
        ]
    result = catalog.copy()
    for column in columns:
        reported = catalog[column].map(present)
        result[column] = catalog[column].where(~reported, work[column])
    return result


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
    water = validate_water_numbers(scenario.get("water") if scenario.get("water") is not None else {})
    direct_ionic_strength = _first(water, "ionic_strength_mol_l", "ionic_strength_mol_L")
    direct_organic_carbon = _first(water, "organic_carbon_mg_l", "organic_carbon_mg_L")
    direct_ph = _first(water, "pH", "ph")
    direct_tds = _first(water, "tds_mg_l", "TDS_mg_L")
    legacy_doc = _first(water, "doc_mg_l", "DOC_mg_L")
    legacy_toc = _first(water, "toc_mg_l", "TOC_mg_L")
    organic_present = water.get("organic_matter_present", False)
    if not isinstance(organic_present, bool):
        raise ValueError("water.organic_matter_present must be true or false.")
    if any(clean_text(water.get(key)) for key in ("organic_matter", "added_organic_matter", "organic_additives")):
        raise ValueError("Use the organic_matter_present true/false flag instead of organic composition text.")
    if clean_text(water.get("inorganic_matter")) or clean_text(water.get("added_inorganic_matter")):
        raise ValueError("Use structured inorganic_additives with species, concentration, and unit fields instead of additive text.")
    additives = validate_additives(water.get("inorganic_additives", []))
    frame = pd.DataFrame(
        [{
            "Water_type": _first(water, "water_type", "type") or "",
            "pH": direct_ph if direct_ph is not None else "",
            # A consolidated runtime value is equivalent to the highest-priority
            # reported source. Legacy DOC/TOC keys remain accepted.
            "DOC_(mg/L)": direct_organic_carbon if direct_organic_carbon is not None else (legacy_doc if legacy_doc is not None else ""),
            "TOC_(mg/L)": legacy_toc if legacy_toc is not None else "",
            "TDS_(mg/L)": direct_tds if direct_tds is not None else "",
            "Ionic_strength": (f"{direct_ionic_strength}_mol/L" if direct_ionic_strength is not None
                               else _first(water, "ionic_strength", "ionic_strength_with_unit") or ""),
            "Ionic_strength_(mol/L)": direct_ionic_strength if direct_ionic_strength is not None else "",
            "Inorganic_matter": database_additive_text(additives),
            "Organic_matter": "",
        }]
    )
    standardize_water_type(frame)
    apply_synthetic_water_from_reported_additives(frame)
    if organic_present and clean_text(frame.at[0, "Water_type"]) in {"", "ultrapure water"}:
        frame.at[0, "Water_type"] = "synthetic water"
    apply_default_water_context(frame)
    normalize_water_quality_scalar_fields(frame)
    materialize_organic_carbon(frame)
    compute_ionic_strength(frame)
    materialize_additive_ml_features(frame)
    row = frame.iloc[0].to_dict()
    row["organic_matter_present"] = organic_present
    if direct_ionic_strength is not None:
        row["Ionic_strength_(mol/L)"] = direct_ionic_strength
    ions = water.get("ions", {}) or {}
    if isinstance(ions, dict):
        for ion in ("Na", "K", "Ca", "Mg", "Cl", "HCO3", "SO4", "phosphate"):
            if ion in ions:
                parsed = clean_text(row.get(f"contains_{ion}")).casefold() in {"true", "1", "1.0"}
                row[f"contains_{ion}"] = bool(ions[ion]) or parsed
    warnings = []
    if clean_text(row.get("normalization_issues")):
        warnings.append("Water chemistry: " + clean_text(row["normalization_issues"]))
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
    """Locate the PFAS feature cache the bundle's model was trained against.

    A bundle carries its own copy of the workbook, which is what lets it run
    away from the training machine.  ADSORBENT_PFAS_FEATURES_PATH overrides that
    copy; the absolute path in run_config serves bundles exported before copies
    existed (``python -m adsorbent_recommender.bundle <dir>`` adds one).
    """
    trained = bundle.config.get("pfas_features", {})
    bundled = bundle.manifest.get("pfas_features", {})
    sheet_name = bundled.get("sheet_name") or trained.get("sheet_name", "")
    candidates = [Path(os.environ["ADSORBENT_PFAS_FEATURES_PATH"])] if os.environ.get("ADSORBENT_PFAS_FEATURES_PATH") else []
    if bundled.get("snapshot_filename"):
        candidates.append(bundle.directory / bundled["snapshot_filename"])
    if trained.get("path"):
        candidates.append(Path(trained["path"]))
    for candidate in candidates:
        if candidate.is_file():
            return candidate, sheet_name
    return (candidates[0] if candidates else Path()), sheet_name


@lru_cache(maxsize=8)
def _feature_store(path: str, sheet: str, mtime_ns: int, required: tuple, numeric: tuple, enabled: bool):
    return PfasFeatureStore(Path(path), sheet, required_features=required,
                            numeric_features=numeric, allow_generation=enabled)


def resolve_pfas(bundle: RecommenderBundle, scenario: dict[str, Any], rdkit_python: Path | None = None):
    identity, identity_type = _scenario_identity(scenario)
    path, sheet_name = pfas_features_source(bundle)
    required = tuple(field["model_feature"] for field in bundle.manifest["input_contract"]["fields"]
                     if field.get("source") == "pfas_cache")
    numeric = tuple(name for name in required if name in bundle.manifest["features"]["numeric"])
    runtime = bundle.manifest.get("pfas_runtime", {})
    enabled = bool(bundle.manifest.get("capabilities", {}).get("unseen_pfas_prediction")
                   and runtime.get("enabled") and runtime.get("descriptor_version") == DESCRIPTOR_VERSION
                   and runtime.get("rdkit_version") == rdBase.rdkitVersion)
    store = _feature_store(str(path), sheet_name, path.stat().st_mtime_ns, required, numeric, enabled)
    pfas = scenario.get("pfas", {})
    # Public lookups happen in the editor, where a human confirms the structure.
    # The prediction service only uses the confirmed SMILES, never a supplied label as evidence.
    if pfas.get("confirmed_smiles"):
        if not pfas.get("structure_confirmed"):
            raise ValueError("Confirm the resolved molecular structure before screening.")
        lookup = store.lookup(str(pfas["confirmed_smiles"]))
        provenance = {key: value for key, value in pfas.get("resolution", {}).items()
                      if key in {"source", "cid", "url", "query", "identity_type", "inchikey", "title"}}
        lookup = replace(lookup, identity_resolution={**lookup.identity_resolution, **provenance,
                         "submitted_identity": identity, "submitted_identity_type": identity_type,
                         "structure_confirmed": True})
    elif identity_type == "smiles":
        lookup = store.lookup(identity)
    else:
        lookup = store.lookup_identifier(identity, identity_type)
    return lookup, store.cache_version


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
        # A parenthetical qualifies a subcategory without changing it: "(GAC)"
        # abbreviates granular activated carbon, and "(Type II)" names a kind
        # of strong base anion resin, which the training records never split.
        without_qualifier = re.sub(r"\s*\([^)]*\)", "", text).strip()
        normalized = re.sub(r"[^a-z0-9]+", "", without_qualifier.casefold())
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
) -> tuple[dict[str, Any], list[str], list[str]]:
    """Assemble one model input row.

    Returns the row, the scenario features the user left blank, and warnings
    about catalog values the adsorbent family rule discarded.
    """
    features = bundle.manifest["features"]["selected"]
    row: dict[str, Any] = {feature: np.nan for feature in features}
    values = dict(lookup_row)
    values.update(water)
    values["initial_concentration_mg_l"] = scenario["initial_concentration_mg_l"]
    values["temperature_c"] = effective_temperature(scenario, fixed_test_conditions(bundle, scenario)["fixed_conditions"])
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
    return row, missing_inputs, applicability_warnings


def _input_support_issues(row: dict[str, Any], bounds: dict[str, Any]) -> list[dict[str, str]]:
    """Report development-support differences; do not suppress a prediction.

    Each issue names its feature so the caller can tell a scenario input, which
    affects every candidate alike, from a catalog property of one product.
    """
    issues: list[dict[str, str]] = []
    for feature, support in bounds.get("feature_support", {}).items():
        value = row.get(feature)
        if pd.isna(value) or clean_text(value) == "":
            continue
        if support.get("kind") == "numeric":
            number = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
            if pd.notna(number) and (number < support.get("hard_min", -np.inf) or number > support.get("hard_max", np.inf)):
                issues.append({
                    "feature": feature, "kind": "range",
                    "text": f"{feature} is outside the development range [{support['hard_min']}, {support['hard_max']}].",
                })
        elif support.get("kind") == "categorical" and clean_text(value) not in set(support.get("levels", [])):
            issues.append({
                "feature": feature, "kind": "category",
                "text": f"{feature} has a category not observed during development: {clean_text(value)!r}.",
            })
    return issues


def _training_support(bundle: RecommenderBundle, kind: str, keys: set[str]) -> dict[str, int] | None:
    """Training records and studies behind a PFAS or product, or None if unknown.

    Aliases of one material can each name the same records, so the largest
    count over the aliases is reported rather than their sum.
    """
    table = bundle.training_support
    if table.empty or not keys:
        return None
    hits = table[table["kind"].eq(kind) & table["key"].isin(keys)]
    if hits.empty:
        return {"records": 0, "studies": 0}
    return {"records": int(hits["records"].max()), "studies": int(hits["studies"].max())}


def _caveat(code: str, severity: str, label: str, text: str) -> dict[str, str]:
    return {"code": code, "severity": severity, "label": label, "text": text}


def _by_severity(caveats: list[dict[str, Any]]) -> list[dict[str, Any]]:
    unique = list({(caveat["code"], caveat["text"]): caveat for caveat in caveats}.values())
    return sorted(unique, key=lambda caveat: CAVEAT_SEVERITIES.index(caveat["severity"]))


def _scenario_caveats(
    bundle: RecommenderBundle,
    lookup: Any,
    pfas_label: str,
    water_warnings: list[str],
    missing_inputs: list[str],
    scenario_issues: list[dict[str, Any]],
    pfas_support: dict[str, Any] | None = None,
) -> list[dict[str, str]]:
    """Caveats that apply to every candidate in one PFAS screening."""
    caveats = []
    support = _training_support(bundle, "pfas", pfas_keys(lookup.feature_row or {}, lookup.cache_key))
    seen = (pfas_support or {}).get("seen_in_training")
    if seen is False or (seen is None and support is not None and support["records"] == 0):
        caveats.append(_caveat(
            "unseen_pfas", "critical", "Unseen PFAS",
            f"{pfas_label}: {UNSEEN_WARNING}",
        ))
    elif support is not None and support["studies"] == 1:
        caveats.append(_caveat(
            "thin_pfas", "info", "PFAS from one study",
            f"{pfas_label} appears in the training data from one study only ({support['records']} records).",
        ))
    for issue in scenario_issues:
        caveats.append(_caveat(
            "scenario_out_of_range" if issue["kind"] == "range" else "scenario_unseen_category",
            "critical", "Input outside training data",
            f"{pfas_label}: {issue['text']}" if issue.get("pfas") else issue["text"],
        ))
    caveats.extend(_caveat("no_water_chemistry" if text.startswith("No water chemistry") else "water_chemistry", "info",
                          "No water chemistry" if text.startswith("No water chemistry") else "Water chemistry", text)
                   for text in water_warnings)
    if missing_inputs:
        labels = {
            field["model_feature"]: field.get("label") or field["model_feature"]
            for field in bundle.manifest["input_contract"]["fields"]
        }
        caveats.append(_caveat(
            "missing_inputs", "info", "Inputs left blank",
            "Not entered, so treated as unreported by the model: "
            + ", ".join(dict.fromkeys(labels.get(feature, feature) for feature in missing_inputs)) + ".",
        ))
    return caveats


def _unchecked_database_conditions(scenario: dict[str, Any], water: dict[str, Any], shared: dict[str, Any]) -> list[str]:
    compared = requested_conditions(scenario, water, shared)
    return [DATABASE_CONDITION_LABELS.get(label, label) for label in NUMERIC_FIELDS if label not in compared]


def _candidate_caveats(
    performance_source: str,
    performance: dict[str, Any],
    prediction: dict[str, Any],
    characterization: dict[str, Any],
    catalog_issues: list[dict[str, str]],
    applicability_warnings: list[str],
) -> list[dict[str, str]]:
    """Caveats for the value a candidate is actually ranked by.

    A database-backed score is a measurement, so model warnings computed for it
    as diagnostics are not its caveats; its own are about how much and how
    consistent the matched evidence is.
    """
    caveats = []
    if performance_source == "database_match":
        if performance.get("study_count") == 1:
            caveats.append(_caveat(
                "database_one_study", "info", "One study",
                f"Database value from one study ({performance.get('match_count', 0)} matched records).",
            ))
        low, high = performance.get("matched_logKd_min"), performance.get("matched_logKd_max")
        if low is not None and high is not None and high - low > DATABASE_SPREAD_LIMIT_LOG_UNITS:
            caveats.append(_caveat(
                "database_spread", "critical", "Records disagree",
                f"Matched records span {low:.2f} to {high:.2f} logKd; the score is their study-balanced median.",
            ))
        if (performance.get("match_count") or 0) > 0 and performance.get("exact_reference_dose_match_count") == 0:
            caveats.append(_caveat(
                "database_near_dose", "info", "Near-dose value",
                "Database value uses near-dose observations (20–30 mg/L) rather than a result at exactly 25 mg/L.",
            ))
        return _by_severity(caveats)

    if not characterization.get("seen_in_training"):
        caveats.append(_caveat(
            "unseen_material", "critical", "Unseen material",
            "This product does not appear in the training data, so its prediction extrapolates to an unseen material.",
        ))
    material_support = characterization.get("adsorbent_support", {})
    if material_support.get("exploratory_prediction"):
        caveats.append(_caveat(
            "exploratory_adsorbent", "critical", "Exploratory adsorbent",
            "Exploratory ML prediction: " + " ".join(material_support.get("reasons", []))
            + " Experimental validation is needed before selecting this product.",
        ))
    # How many studies or records a ranked product has is not a caveat: tier
    # placement was about as reliable for products with few studies or records
    # as for well-represented ones (by_support in tier_hidden_pairing_summary.json), because the model
    # predicts from adsorbent properties, not product identity. The counts are
    # reported as context instead.
    for issue in catalog_issues:
        caveats.append(_caveat(
            "property_out_of_range" if issue["kind"] == "range" else "property_unseen_category",
            "critical", "Property outside training data", issue["text"],
        ))
    mean = prediction.get("mean_logKd")
    low, high = WELL_SUPPORTED_LOGKD
    if mean is not None and not low <= mean <= high:
        caveats.append(_caveat(
            "extreme_logkd", "critical", "Extreme logKd",
            f"Predicted logKd {mean:.2f} lies outside {low:g}–{high:g}, where test predictions were least accurate.",
        ))
    if characterization.get("at_eligibility_floor"):
        caveats.append(_caveat(
            "thin_characterization", "info", "Few properties known",
            f"Characterized on {characterization.get('material_features_present')} of "
            f"{characterization.get('material_features_required')} adsorbent properties, the minimum for scoring.",
        ))
    caveats.extend(_caveat("inapplicable_value", "info", "Value not used", text) for text in applicability_warnings)
    return _by_severity(caveats)


def _leading_group_margin(bundle: RecommenderBundle) -> tuple[float | None, str]:
    """The score gap below which two candidates cannot be told apart.

    Two candidates are compared through the difference of two predictions, so the
    margin is the cross-validated RMSE of predicted differences between adsorbents
    tested under the same conditions (leading_group_margin.json, written by
    recommender_validation/cv_difference_error.py). Bundles without it fall back
    to the single-prediction validation RMSE.
    """
    info = bundle.training_support_info
    if info.get("leading_group_margin") is not None:
        return float(info["leading_group_margin"]), str(info.get("leading_group_margin_basis", ""))
    return info.get("validation_rmse"), "cross-validated RMSE of the deployed model configuration"


def _mark_rank_resolution(
    recommendations: list[dict[str, Any]],
    threshold: float | None,
    basis: str = "cross-validated RMSE of the deployed model configuration",
) -> dict[str, Any]:
    """Flag candidates whose scores the model's error cannot separate from the top."""
    if not recommendations or threshold is None:
        return {"available": False}
    top = float(recommendations[0]["ranking_score"])
    for item in recommendations:
        gap = top - float(item["ranking_score"])
        item["rank_resolution"] = {"gap_to_top": gap, "within_error_of_top": gap <= threshold}
    return {
        "available": True,
        "threshold_log_units": float(threshold),
        "basis": basis,
        "within_error_count": int(sum(item["rank_resolution"]["within_error_of_top"] for item in recommendations)),
    }


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
    temperature = effective_temperature(scenario, fixed)
    temperature_features = [feature for feature in fixed if feature.startswith("Temperature_(")] or ["Temperature_(°C)"]
    for feature in temperature_features:
        fixed[feature] = temperature
    dose_feature = "Adsorbent_dosage_value_mg/L"
    dose = controls.get(dose_feature, {})
    return {
        "status": "fixed_conditions",
        "fixed_conditions": fixed,
        "dose_mg_L": fixed.get(dose_feature),
        "dose_within_development_support": dose.get("default_within_observed_support"),
        "basis": "every candidate is scored at the same conditions so the ranking compares affinity",
    }


def _material_characterization(material: pd.Series, support: dict[str, int] | None = None) -> dict[str, Any]:
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
    at_floor = bool(pd.notna(present_count) and pd.notna(minimum_count) and float(present_count) <= float(minimum_count))
    # Training support counted from the deployed model's own rows decides
    # whether a product was seen; the identity map's mapping_status misses
    # products recorded under their commercial names.
    if support is not None:
        seen = support["records"] > 0
    else:
        seen = clean_text(material.get("mapping_status")) == "exact_normalized_key"
    if at_floor:
        warnings.append(
            f"{material.get('Product_key')} is characterized on {int(present_count)} of "
            f"{int(required_count)} material features, at the eligibility floor."
        )
    if not seen:
        warnings.append(
            f"{material.get('Product_key')} does not match a training adsorbent by identity, "
            "so its prediction extrapolates to an unseen material."
        )
    return {
        "material_features_present": None if pd.isna(present_count) else int(present_count),
        "material_features_required": None if pd.isna(required_count) else int(required_count),
        "material_features_minimum": None if pd.isna(minimum_count) else int(minimum_count),
        "missing_material_features": material.get("missing_required_material_features"),
        "seen_in_training": seen,
        "training_records": None if support is None else support["records"],
        "training_studies": None if support is None else support["studies"],
        "at_eligibility_floor": at_floor,
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
    scenario = validate_scenario_numbers(scenario)
    exploratory = include_exploratory(scenario)
    lookup, cache_version = resolve_pfas(bundle, scenario, rdkit_python)
    ready = getattr(lookup, "ready", lookup.status == "cached")
    pfas_support = describe_support(bundle, lookup) if ready else {}
    response: dict[str, Any] = {
        "result_schema_version": RESULT_SCHEMA_VERSION,
        "status": "pending",
        "bundle": {"directory": str(bundle.directory), "model_family": bundle.manifest["model_family"], "target": bundle.manifest["target"]},
        "pfas_lookup": {"status": lookup.status, "ready": ready, "canonical_smiles": lookup.canonical_smiles,
                        "cache_key": lookup.cache_key, "warnings": list(lookup.warnings),
                        "feature_source": getattr(lookup, "feature_source", "reviewed_cache"),
                        "descriptor_version": getattr(lookup, "descriptor_version", None),
                        "rdkit_version": getattr(lookup, "rdkit_version", None),
                        "identity_resolution": getattr(lookup, "identity_resolution", {}), "support": pfas_support,
                        "label": lookup.cache_key or scenario.get("pfas", {}).get("label") or scenario.get("pfas", {}).get("identity")},
        "pfas_support": pfas_support,
        "pfas_cache": cache_version,
        "scenario": scenario,
    }
    if not ready:
        response.update({"status": "pfas_not_ready", "offline_feature_request": lookup.offline_request})
        return response
    water, water_warnings = normalize_water_scenario(scenario)
    water_support = water_diagnostics(bundle.water_support, water.get("Water_type"))
    response["water_support"] = water_support
    response["water_chemistry"] = {
        "input_version": ADDITIVE_INPUT_VERSION, "normalized_water_type": water.get("Water_type"),
        "inorganic_additives": validate_additives((scenario.get("water") or {}).get("inorganic_additives", [])),
        "organic_matter_present": water.get("organic_matter_present", False),
        "organic_carbon_mg_l": water.get("organic_carbon_mg/L"),
        "ionic_strength_mol_l": water.get("Ionic_strength_(mol/L)"),
        "ions_present": [ion for ion in ("Na", "K", "Ca", "Mg", "Cl", "HCO3", "SO4", "phosphate")
                         if clean_text(water.get(f"contains_{ion}")).casefold() in {"true", "1", "1.0"}],
        "normalization_trace": water.get("normalization_trace"), "warnings": water_warnings,
    }
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
    sources = {field["model_feature"]: field.get("source") for field in bundle.manifest["input_contract"]["fields"]}
    pfas_label = response["pfas_lookup"]["label"] or "This PFAS"
    pfas_warnings = list(lookup.warnings)
    if pfas_support.get("seen_in_training") is False:
        pfas_warnings.append(UNSEEN_WARNING)
    pfas_caveats = []
    if pfas_support.get("seen_in_training") is False:
        pfas_caveats.append(_caveat("unseen_pfas", "critical", "Unseen PFAS", UNSEEN_WARNING))
    if pfas_support.get("out_of_range_descriptors"):
        text = "Molecular descriptors outside training ranges: " + ", ".join(v["feature"] for v in pfas_support["out_of_range_descriptors"]) + ". Treat this prediction with additional caution."
        pfas_warnings.append(text)
        pfas_caveats.append(_caveat("pfas_descriptors_out_of_range", "critical", "PFAS outside training ranges", text))
    results, rejected = [], []
    missing_inputs: list[str] = []
    scenario_issues: dict[str, dict[str, Any]] = {}
    for _, material in catalog.iterrows():
        material_support = describe_material_support(bundle.material_support, bundle.material_support_info, str(material["Product_key"]))
        database_evidence = match_candidate_evidence(
            bundle.evidence, material=material.to_dict(), pfas_feature_row=lookup.feature_row or {},
            pfas_cache_key=lookup.cache_key, scenario=scenario, normalized_water=water,
            screening_conditions=shared_conditions,
            policy=bundle.manifest.get("database_evidence", {}).get("match_policy", DEFAULT_MATCH_POLICY),
        )
        database_matched = database_evidence.get("status") == "matched"
        # The model is used for untested pairings of PFAS and adsorbents that are each in the training
        # data (the combination-held-out setting). A product with no training record lies outside that
        # setting, so it is ranked only through comparable database evidence or an explicit opt-in.
        if not database_matched and not material_support["identity_verified"] and not exploratory:
            rejected.append({"Product_key": material["Product_key"], "reason": "adsorbent_not_in_training_data",
                             "details": material_support["reasons"], "adsorbent_support": material_support,
                             "database_evidence": database_evidence})
            continue
        model_row, missing_inputs, applicability_warnings = _model_row(
            bundle, lookup.feature_row or {}, scenario, water, material.to_dict(),
        )
        input_warnings = [f"Missing scenario value: {feature}" for feature in missing_inputs] + applicability_warnings
        support_issues = _input_support_issues(model_row, bundle.bounds)
        support_warnings = [issue["text"] for issue in support_issues]
        catalog_issues = [issue for issue in support_issues if sources.get(issue["feature"]) == "catalog"]
        for issue in support_issues:
            if sources.get(issue["feature"]) != "catalog":
                scenario_issues[issue["text"]] = {**issue, "pfas": sources.get(issue["feature"]) == "pfas_cache"}
        prediction = predict_logkd(bundle, model_row)
        if database_evidence.get("status") == "matched":
            performance = database_evidence["performance"]
            score = float(performance["mean_logKd"])
            performance_source = "database_match"
        else:
            performance = {"source": "model_prediction", **prediction}
            score = prediction["mean_logKd"]
            performance_source = "model_prediction"
        characterization = _material_characterization(
            material, ({"records": material_support["training_rows"], "studies": material_support["study_ids"]}
                       if material_support["available"] else _training_support(bundle, "adsorbent", candidate_keys(material.to_dict()))),
        )
        material_support["exploratory_prediction"] = performance_source == "model_prediction" and not material_support["identity_verified"]
        material_support["ranking_basis"] = "database_evidence" if database_matched else "exploratory_ml" if material_support["exploratory_prediction"] else "default_ml"
        characterization["adsorbent_support"] = material_support
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
            "adsorbent_support": material_support,
            "prediction": prediction,
            "performance": performance,
            "performance_source": performance_source,
            "database_evidence": database_evidence,
            "screening_conditions": shared_conditions,
            "ranking_score": score,
            "warnings": pfas_warnings + water_warnings + input_warnings + support_warnings + characterization["warnings"],
            "pfas_support": pfas_support,
            "caveats": _by_severity(_candidate_caveats(
                performance_source, performance, prediction, characterization,
                catalog_issues, applicability_warnings,
            ) + (pfas_caveats + water_caveats(water_support) if performance_source == "model_prediction" else [])),
        })
    ranked = sorted(results, key=lambda item: item["ranking_score"], reverse=True)
    recommendations = _select_diverse(ranked, max(1, int(top_k)))
    for rank, recommendation in enumerate(recommendations, start=1):
        recommendation["rank"] = rank
    rank_resolution = _mark_rank_resolution(recommendations, *_leading_group_margin(bundle))
    database_matches = sum(item["performance_source"] == "database_match" for item in results)
    scenario_caveats = _scenario_caveats(
        bundle, lookup, pfas_label, water_warnings, missing_inputs, list(scenario_issues.values()), pfas_support,
    )
    if any(item["performance_source"] == "model_prediction" for item in results):
        scenario_caveats.extend(water_caveats(water_support))
    unchecked = _unchecked_database_conditions(scenario, water, shared_conditions)
    if database_matches and unchecked:
        scenario_caveats.append(_caveat(
            "database_conditions_unchecked", "info", "Database match not checked",
            "Database values were not checked against conditions left blank ("
            + ", ".join(unchecked) + "); they match on PFAS, product, water type and the entered conditions only.",
        ))
    response.update({
        "status": "recommendations_ready" if recommendations else "no_supported_candidates",
        "recommendations": recommendations,
        "rejected_candidates": rejected,
        "screening_conditions": shared_conditions,
        "scenario_caveats": _by_severity(scenario_caveats),
        "rank_resolution": rank_resolution,
        "candidates_screened": int(len(catalog)),
        "adsorbent_screening": {
            "include_exploratory": exploratory, "policy": policy_for(bundle.material_support_info),
            "support_snapshot_available": bool(bundle.material_support_info.get("available")),
            "property_eligible_products": len(catalog), "excluded_products": len(rejected),
            "ranked_products": len(results),
            "exploratory_products": sum(item["adsorbent_support"]["exploratory_prediction"] for item in results),
            "database_exception_products": sum(item["performance_source"] == "database_match" and not item["adsorbent_support"]["meets_default_ml_policy"] for item in results),
        },
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


def _mixture_caveats(per_pfas: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Merge per-PFAS caveats, naming the PFAS a caveat applies to.

    A caveat about the product itself is the same for every PFAS and is listed
    once without names; one about a score names the PFAS whose score it is.
    """
    merged: dict[tuple[str, str], dict[str, Any]] = {}
    for row in per_pfas:
        for caveat in row.get("caveats", []):
            entry = merged.setdefault((caveat["code"], caveat["text"]), {**caveat, "pfas": []})
            entry["pfas"].append(row.get("pfas"))
    for entry in merged.values():
        if len(per_pfas) == 1 or len(entry["pfas"]) == len(per_pfas):
            entry["pfas"] = []
    return _by_severity(list(merged.values()))


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
    # Validate the complete mixture before any identity lookup or scoring.
    scenarios = [validate_scenario_numbers(scenario) for scenario in scenarios]
    modes = {include_exploratory(scenario) for scenario in scenarios}
    if len(modes) != 1:
        raise ValueError("The exploratory adsorbent setting must be shared by every PFAS in a mixture.")

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

    by_run = [
        {item["product"]["Product_key"]: item for item in run["recommendations"]}
        for run in runs
    ]
    common_keys = set(by_run[0])
    for mapping in by_run[1:]:
        common_keys.intersection_update(mapping)
    rejected = {}
    for scenario, run in zip(scenarios, runs):
        pfas_label = run.get("pfas_lookup", {}).get("label") or scenario.get("pfas", {}).get("identity")
        for item in run.get("rejected_candidates", []):
            entry = rejected.setdefault(item["Product_key"], {"Product_key": item["Product_key"],
                "reason": item["reason"], "adsorbent_support": item.get("adsorbent_support"), "pfas_rejections": []})
            entry["pfas_rejections"].append({"pfas": pfas_label, "details": item.get("details", [])})

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
                "pfas": lookup.get("label") or lookup.get("cache_key") or scenario.get("pfas", {}).get("identity"),
                "pfas_support": lookup.get("support", {}),
                "pfas_lookup": lookup,
                "input_identity": scenario.get("pfas", {}).get("identity"),
                "identity_type": scenario.get("pfas", {}).get("identity_type"),
                "initial_concentration_mg_l": scenario.get("initial_concentration_mg_l"),
                "concentration_input": scenario.get("concentration_input"),
                "performance": item.get("performance"),
                "performance_source": item.get("performance_source"),
                "prediction": item.get("prediction"),
                "database_evidence": item.get("database_evidence"),
                "adsorbent_support": item.get("adsorbent_support", {}),
                "warnings": item.get("warnings", []),
                "caveats": item.get("caveats", []),
            })

        source_types = {item.get("performance_source") for item in source_items}
        if source_types == {"database_match"}:
            aggregate_source = "database_match"
        elif source_types == {"model_prediction"}:
            aggregate_source = "model_prediction"
        else:
            aggregate_source = "mixed_evidence"

        candidate = deepcopy(limiting_item)
        aggregate_support = deepcopy(limiting_item.get("adsorbent_support", {}))
        aggregate_support["exploratory_prediction"] = any(item.get("adsorbent_support", {}).get("exploratory_prediction") for item in source_items)
        candidate.update({
            "performance_source": aggregate_source,
            "adsorbent_support": aggregate_support,
            "ranking_score": scores[limiting_index],
            "mixture_performance": {
                "worst_case_logKd": scores[limiting_index],
                "mean_logKd": float(np.mean(scores)),
                "limiting_pfas": per_pfas[limiting_index]["pfas"],
                "limiting_pfas_unseen": per_pfas[limiting_index].get("pfas_support", {}).get("seen_in_training") is False,
                "limiting_score_exploratory_adsorbent": bool(limiting_item.get("adsorbent_support", {}).get("exploratory_prediction")),
                "database_supported_pfas": sum(
                    row["performance_source"] == "database_match" for row in per_pfas
                ),
                "pfas_count": len(per_pfas),
            },
            "pfas_results": per_pfas,
            "caveats": _mixture_caveats(per_pfas),
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
    rank_resolution = _mark_rank_resolution(recommendations, *_leading_group_margin(bundle))

    database_pairs = sum(
        item.get("performance_source") == "database_match"
        for run in runs
        for item in run.get("recommendations", [])
        if item["product"]["Product_key"] in common_keys
    )
    total_pairs = len(common_keys) * len(scenarios)
    first = runs[0]
    response.update({
        "status": "recommendations_ready" if recommendations else "no_supported_candidates",
        "recommendations": recommendations,
        "rejected_candidates": list(rejected.values()),
        "adsorbent_screening": {
            **first.get("adsorbent_screening", {}), "excluded_products": len(rejected), "ranked_products": len(combined),
            "exploratory_products": sum(bool(item["adsorbent_support"].get("exploratory_prediction")) for item in combined),
            "database_exception_products": sum(item["performance_source"] == "database_match" and not item["adsorbent_support"].get("meets_default_ml_policy") for item in combined),
        },
        "screening_conditions": first.get("screening_conditions", {}),
        "water_chemistry": first.get("water_chemistry", {}),
        "water_support": first.get("water_support", {}),
        "scenario_caveats": _by_severity([caveat for run in runs for caveat in run.get("scenario_caveats", [])]),
        "rank_resolution": rank_resolution,
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
