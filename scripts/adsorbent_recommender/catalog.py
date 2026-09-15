"""Authoritative adsorbent-catalog utilities for recommender deployment.

This module deliberately does not participate in feature selection or model
selection.  It only converts the maintained catalog into model column names
and audits whether a *chosen* model can be evaluated for each product.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
AD_ROOT = SCRIPT_DIR.parents[1]
DB_DIR = SCRIPT_DIR.parent / "DB"
if str(DB_DIR) not in sys.path:
    sys.path.insert(0, str(DB_DIR))
# The catalog must encode functional groups and pore classes exactly as the
# training database did, so the indicator columns are materialized by the same
# functions rather than by a second implementation that could drift from them.
from adsorbent_consolidation import (  # noqa: E402
    materialize_functional_group_ml_features,
    materialize_pore_class_ml_features,
    normalize_functional_groups,
)
DEFAULT_CATALOG_PATH = AD_ROOT / "output" / "Merge" / "commercial_adsorbent_properties_authoritative.xlsx"
DEFAULT_CATALOG_SHEET = "Adsorbent_Properties"
MISSING_TOKENS = {"", "na", "n/a", "nan", "none", "null", "not reported", "not applicable"}

# Canonical model column -> accepted catalog column names.  The canonical name
# itself is included so future catalog exports can adopt model names directly.
CATALOG_COLUMN_ALIASES: dict[str, tuple[str, ...]] = {
    "adsorbent_category": ("adsorbent_category", "Adsorbent_category"),
    "adsorbent_subcategory": ("adsorbent_subcategory", "Adsorbent_subcategory"),
    "polymer_matrix": ("polymer_matrix", "Polymer_matrix"),
    "AC_raw_material": ("AC_raw_material",),
    "AC_activation_method": ("AC_activation_method",),
    "pore_class": ("pore_class", "Pore_class", "Pore_Class"),
    "particle_size_value_mm": ("particle_size_value_mm", "Particle_size_value_mm"),
    "Nominal_grain_size_value_mm": ("Nominal_grain_size_value_mm",),
    "ssa_(m2/g)_avg": ("ssa_(m2/g)_avg", "SSA_(m2/g)"),
    "average_pore_diameter_(angstrom)": ("average_pore_diameter_(angstrom)", "Average_pore_diameter_(Angstrom)"),
    "phpzc": ("phpzc", "pHPZC"),
    "porosity_total_value": ("porosity_total_value", "Porosity_total_value"),
    "porosity_micro_value_avg": ("porosity_micro_value_avg", "Porosity_micro_value_avg"),
    "porosity_meso_value_avg": ("porosity_meso_value_avg", "Porosity_meso_value_avg"),
    "porosity_macro_value_avg": ("porosity_macro_value_avg", "Porosity_macro_value_avg"),
    "pore_volume_total_value_cm3/g": ("pore_volume_total_value_cm3/g", "Total_pore_volume_(cm3/g)"),
    "pore_volume_micro_value_avg_cm3/g": ("pore_volume_micro_value_avg_cm3/g", "Micro_pore_volume_(cm3/g)"),
    "pore_volume_meso_value_avg_cm3/g": ("pore_volume_meso_value_avg_cm3/g", "Meso_pore_volume_(cm3/g)"),
    "pore_volume_macro_value_avg_cm3/g": ("pore_volume_macro_value_avg_cm3/g", "Macro_pore_volume_(cm3/g)"),
    "element_C_value": ("element_C_value",),
    "element_N_value": ("element_N_value",),
    "element_O_value": ("element_O_value",),
    "zeta_potential_value": ("zeta_potential_value",),
    "zeta_potential_pH": ("zeta_potential_pH",),
    "ion_exchange_capacity_value_meq/g": ("ion_exchange_capacity_value_meq/g",),
    "ion_exchange_capacity_value_meq/L": ("ion_exchange_capacity_value_meq/L",),
    "functional_group": ("functional_group", "Functional_group", "Functional_Group"),
}


def clean_text(value: Any) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except TypeError:
        pass
    return str(value).strip()


def normalized_key(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "", clean_text(value).lower())


def present(value: Any) -> bool:
    return clean_text(value).lower() not in MISSING_TOKENS


def _first_existing(columns: Iterable[str], frame: pd.DataFrame) -> str | None:
    return next((column for column in columns if column in frame.columns), None)


def _catalog_provenance(path: Path) -> pd.DataFrame:
    try:
        audit = pd.read_excel(path, sheet_name="Merge_Audit")
    except ValueError:
        return pd.DataFrame(columns=["Product_key", "catalog_match_status", "catalog_data_sources"])
    required = {"resolved_Product_key", "match_status", "data_sources"}
    if not required.issubset(audit.columns):
        return pd.DataFrame(columns=["Product_key", "catalog_match_status", "catalog_data_sources"])
    audit = audit.copy()
    audit["Product_key"] = audit["resolved_Product_key"].map(clean_text)
    audit = audit[audit["Product_key"].ne("")]
    if audit.empty:
        return pd.DataFrame(columns=["Product_key", "catalog_match_status", "catalog_data_sources"])
    return audit.groupby("Product_key", dropna=False).agg(
        catalog_match_status=("match_status", lambda values: "; ".join(sorted(set(filter(None, map(clean_text, values))))),),
        catalog_data_sources=("data_sources", lambda values: "; ".join(sorted(set(filter(None, map(clean_text, values))))),),
    ).reset_index()


def load_authoritative_catalog(
    path: Path = DEFAULT_CATALOG_PATH,
    sheet_name: str = DEFAULT_CATALOG_SHEET,
    category: str | Iterable[str] | None = "activated carbon",
) -> pd.DataFrame:
    """Load products and materialize canonical model-property aliases.

    ``category`` accepts one label or several, because a model family can span
    more than one catalog category.  Passing ``None`` keeps every product,
    including categories no model family covers.
    """
    if not path.exists():
        raise FileNotFoundError(f"Authoritative catalog not found: {path}")
    catalog = pd.read_excel(path, sheet_name=sheet_name).copy()
    if "Product_key" not in catalog.columns:
        raise ValueError("Authoritative catalog requires unique, nonblank Product_key values.")
    catalog["Product_key"] = catalog["Product_key"].map(clean_text)
    if catalog["Product_key"].eq("").any() or catalog["Product_key"].duplicated().any():
        raise ValueError("Authoritative catalog requires unique, nonblank Product_key values.")
    if category is not None and "Adsorbent_category" in catalog.columns:
        wanted = [category] if isinstance(category, str) else list(category)
        keep = {clean_text(label).casefold() for label in wanted if clean_text(label)}
        if not keep:
            raise ValueError("A category filter must name at least one nonblank category.")
        catalog = catalog[catalog["Adsorbent_category"].map(clean_text).str.casefold().isin(keep)].copy()
    for canonical, aliases in CATALOG_COLUMN_ALIASES.items():
        source = _first_existing(aliases, catalog)
        if source is not None:
            catalog[canonical] = catalog[source]
        elif canonical not in catalog.columns:
            catalog[canonical] = np.nan
    _materialize_model_indicator_columns(catalog)
    return catalog.merge(_catalog_provenance(path), on="Product_key", how="left", validate="one_to_one")


def _materialize_model_indicator_columns(catalog: pd.DataFrame) -> None:
    """Expand catalog labels into the indicator columns the models consume.

    Functional groups and pore classes reach the model as per-label indicators,
    not as free text, so a catalog that carries only the parent label looks
    empty to a model that never saw that label.  The expansion is three-state:
    a reported parent yields True/False per label, and an unreported parent
    leaves every indicator blank rather than asserting absence.
    """
    catalog["functional_group"] = catalog["functional_group"].map(normalize_functional_groups)
    materialize_functional_group_ml_features(catalog)
    materialize_pore_class_ml_features(catalog)


def catalog_eligibility(
    catalog: pd.DataFrame,
    required_features: Iterable[str],
    minimum_present_features: int | None = None,
) -> pd.DataFrame:
    """Audit product eligibility for a chosen model's material features.

    ``minimum_present_features`` sets how many of the model's adsorbent
    features a product must carry to be scored.  It exists because demanding
    all of them is unreachable by construction: the models are fitted on rows
    whose adsorbent characterization is itself incomplete, and no training row
    carries the complete set either.  The caller derives the threshold from the
    fitted training rows, so the deployment bar is expressed in the same terms
    as the evidence the model actually learned from.

    Leaving it ``None`` keeps the original all-present rule.
    """
    required = list(dict.fromkeys(required_features))
    if minimum_present_features is not None:
        if not 0 <= minimum_present_features <= len(required):
            raise ValueError(
                f"minimum_present_features must fall between 0 and {len(required)}; "
                f"got {minimum_present_features}."
            )
    rows: list[dict[str, Any]] = []
    for _, product in catalog.iterrows():
        missing = [feature for feature in required if feature not in product.index or not present(product.get(feature))]
        n_present = len(required) - len(missing)
        eligible = not missing if minimum_present_features is None else n_present >= minimum_present_features
        rows.append(
            {
                "Product_key": clean_text(product["Product_key"]),
                "eligible": eligible,
                "required_material_feature_count": len(required),
                "minimum_present_material_features": minimum_present_features if minimum_present_features is not None else len(required),
                "present_material_feature_count": n_present,
                "missing_required_material_features": "; ".join(missing),
            }
        )
    columns = [
        "Product_key",
        "eligible",
        "required_material_feature_count",
        "minimum_present_material_features",
        "present_material_feature_count",
        "missing_required_material_features",
        "eligible_catalog_candidates",
    ]
    # An empty catalog still returns the full schema.  Callers read the
    # ``eligible`` column unconditionally, and a column-less frame would raise a
    # KeyError that reads like a corrupt catalog rather than an empty one.
    eligibility = pd.DataFrame(rows, columns=None if rows else columns)
    eligibility["eligible_catalog_candidates"] = int(eligibility["eligible"].sum()) if rows else 0
    return eligibility


def catalog_identity_map(catalog: pd.DataFrame, model_adsorbent_ids: Iterable[Any]) -> pd.DataFrame:
    """Record exact normalized identity matches without unsafe fuzzy matching."""
    known = {normalized_key(value): clean_text(value) for value in model_adsorbent_ids if normalized_key(value)}
    rows = []
    for product_key in catalog["Product_key"].map(clean_text):
        model_id = known.get(normalized_key(product_key), "")
        rows.append(
            {
                "Product_key": product_key,
                "model_adsorbent_id": model_id,
                "mapping_status": "exact_normalized_key" if model_id else "requires_curated_mapping",
                "mapping_method": "exact_normalized_key_only" if model_id else "no_fuzzy_match_attempted",
            }
        )
    return pd.DataFrame(rows)
