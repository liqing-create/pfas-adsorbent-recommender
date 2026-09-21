"""Internal normalization stage for ``build_database.py``.

This script is launched by the public ``build_database.py`` command so all
paths and run settings come from one place.
"""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

from adsorbent_consolidation import (
    FUNCTIONAL_GROUP_IDENTITY_CONFLICT_COLUMN,
    AdsorbentProcessingConfig,
    build_authoritative_adsorbent_database,
    build_consolidated_adsorbent_database,
    enrich_adsorbent_fields,
    infer_ac_subcategory_from_nominal_grain_size,
    materialize_nonionic_resin_iec_defaults,
    normalize_model_facing_adsorbent_categories,
)
from normalization_endpoints import derive_endpoint_columns, write_output_workbook
from normalization_rules import (
    RANGE_TRACE_COLUMNS,
    _ensure_columns,
    apply_default_water_context,
    apply_synthetic_water_from_reported_additives,
    blank_extraction_missing_placeholders,
    coalesce_renamed_input_columns,
    compute_ionic_strength,
    explode_isotherm_c0_d,
    inject_pfas_properties,
    materialize_additive_ml_features,
    materialize_organic_carbon,
    normalize_organic_matter_units,
    normalize_adsorbent_fraction_fields,
    normalize_adsorbent_scalar_fields,
    normalize_water_quality_scalar_fields,
    materialize_general_range_context,
    ZETA_POTENTIAL_NEAR_PH_7_COLUMN,
    zeta_potential_near_ph_7,
    materialize_snapshot_range_context,
    normalize_and_joined_value_units,
    normalize_fixed_denominator_value_units,
    preserve_raw_reported_range_fields,
    recover_named_organic_conditions,
    recover_water_type_raw_context,
    standardize_pfas_names,
    standardize_water_type,
)
from normalization_units import (
    _standardize_adsorbent_numeric_values,
    infer_kf_unit_from_langmuir,
    infer_pso_units,
    materialize_ssa_avg,
    normalize_pore_volumes,
    standardize_unit_parameters,
)
from performance_normalization_overrides import (
    load_manual_adsorbent_scalar_overrides,
    load_manual_range_average_overrides,
)

# Columns that identify or audit raw merged rows. Keep these visible even when
# external column-order CSVs lag behind the current merge schema.
PINNED_SHEET1_COLUMNS = [
    "standardized_row_id",
    "source_row_index",
    "extraction_dataset",
    "study_no",
    "DOI",
    "performance_id",
    "adsorbent_id",
    "adsorbent_identity_status",
    "adsorbent_identity_basis",
    "adsorbent_identity_key",
    "adsorbent_study_instance_key",
    "PFAS_name",
    "Second_Class",
    "Equilibrium_determination",
    "record_status",
    "status_reason",
    "review_outcome",
]

AUDIT_OUTPUT_COLUMNS = [
    "performance_source",
    "adsorbent_source",
    "data_provenance",
    "normalization_trace",
    # Names the materials whose studies reported different functional groups.
    # The pooled value is model-facing; the disagreement is for review only.
    FUNCTIONAL_GROUP_IDENTITY_CONFLICT_COLUMN,
]

OBSOLETE_OUTPUT_COLUMNS = [
    # Retired trace fields. Raw temperature/mixing-speed columns preserve the
    # source expressions; inferred pH and the parallel error channel were
    # never populated by the active pipeline.
    "pH_inferred_strong",
    "Temperature_context_(°C)",
    "Mixing_speed_context_(rpm)",
    "error_message",
    # Legacy alias; the canonical normalized field is Nominal_grain_size_value_mm.
    "particle_size_value_mm",
    "Particle_size_value_mm",
    "Background_species",
    "Conductivity_(??S/cm)",
    "Conductivity_(????S/cm)",
    "Conductivity_(???/cm)",
    # Derived aliases from a retired additive-parsing schema. The raw schema
    # fields are Inorganic_matter and Organic_matter; ionic strength remains
    # in its canonical normalized field, Ionic_strength_(mol/L).
    "Added_inorganic_species",
    "Added_inorganic_concentration_value",
    "Added_inorganic_concentration_unit",
    "Added_inorganic_concentration_mol/L",
    "Added_inorganic_species_count",
    "Added_inorganic_ionic_species",
    "Added_inorganic_ion_charge",
    "Added_inorganic_ion_concentration_mol/L",
    "Added_inorganic_ionic_strength_(mol/L)",
    "Added_organic_species",
    "Added_organic_concentration_value",
    "Added_organic_concentration_unit",
    "Added_organic_concentration_mol/L",
    "Added_organic_species_count",
]


def env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "y", "on"}


def env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    try:
        return float(raw)
    except ValueError as exc:
        raise RuntimeError(f"Pipeline setting {name} must be numeric, got {raw!r}") from exc


def required_env(name: str) -> str:
    value = os.getenv(name)
    if value is None or not value.strip():
        raise RuntimeError(
            f"Missing pipeline setting {name}. Run normalize.py through build_database.py."
        )
    return value


def _require_orchestrated() -> None:
    if os.getenv("BUILD_DATABASE_ORCHESTRATED") != "1":
        raise SystemExit(
            "normalize.py is an internal pipeline stage. "
            "Run build_database.py instead."
        )


def _adsorbent_config() -> AdsorbentProcessingConfig:
    return AdsorbentProcessingConfig(
        raw_input_path=Path(required_env("UNIT_STANDARDIZER_ADSORBENT_INPUT_PATH")),
        database_output_path=Path(required_env("UNIT_STANDARDIZER_ADSORBENT_OUTPUT_PATH")),
        raw_input_sheet=required_env("UNIT_STANDARDIZER_ADSORBENT_SHEET_NAME"),
        mapping_path=Path(required_env("UNIT_STANDARDIZER_ADSORBENT_MAP_PATH")),
        online_properties_path=Path(required_env("COMM_PROPS_ONLINE_PATH")),
        online_properties_sheet=required_env("COMM_PROPS_ONLINE_SHEET"),
        database_properties_path=Path(required_env("COMM_PROPS_DB_PATH")),
        database_properties_sheet=required_env("COMM_PROPS_DB_SHEET"),
        enable_online_injection=env_bool("ENABLE_COMM_PROPS_ONLINE_INJECTION", True),
        enable_database_injection=env_bool("ENABLE_COMM_PROPS_DB_INJECTION", False),
        authoritative_output_path=Path(required_env("COMM_PROPS_AUTHORITATIVE_PATH")),
        authoritative_properties_sheet=required_env("COMM_PROPS_AUTHORITATIVE_SHEET"),
    )


def load_input_workbook(input_path: str, sheet_name: str) -> pd.DataFrame:
    return pd.read_excel(input_path, sheet_name=sheet_name)


def normalize_dataframe(
    df: pd.DataFrame,
    adsorbent_config: AdsorbentProcessingConfig,
    *,
    manual_range_average_overrides: dict[tuple[str, str, str, str], str] | None = None,
    manual_adsorbent_scalar_overrides: dict[
        tuple[str, str, str, str, str], tuple[str, float]
    ]
    | None = None,
) -> pd.DataFrame:
    blank_extraction_missing_placeholders(df)
    # Coalesce legacy raw inputs before discarding retired derived aliases.
    # This lets an existing raw merged workbook rebuild cleanly while all new
    # output uses the extraction schema's canonical field names.
    coalesce_renamed_input_columns(df)
    df.drop(columns=[c for c in OBSOLETE_OUTPUT_COLUMNS if c in df.columns], inplace=True, errors="ignore")

    # Preserve raw reported experimental values before midpointing, blanking,
    # context routing, defaulting, or isotherm expansion changes the main fields.
    # These trace columns remain in Standardized_Full, not Equilibrium_Data.
    preserve_raw_reported_range_fields(df)

    # Extraction represents explicit numerator/denominator measurements as a
    # scalar value plus a unit such as ``mL/241_mL``.  Move the denominator
    # into the value before downstream range handling and unit conversion.
    # The raw trace above remains unchanged for auditability.
    normalize_and_joined_value_units(df)
    normalize_fixed_denominator_value_units(df)

    if "source_row_index" not in df.columns:
        df.insert(0, "source_row_index", range(len(df)))

    # Split isotherm records with multi-valued C0 and/or dosage into multiple rows.
    df = explode_isotherm_c0_d(df)
    df = df.reset_index(drop=True)
    df.drop(columns=["standardized_row_id"], inplace=True, errors="ignore")
    df.insert(0, "standardized_row_id", [f"std_{i + 1:06d}" for i in range(len(df))])

    standardize_pfas_names(df)
    inject_pfas_properties(
        df,
        required_env("PFAS_PROPS_PATH"),
        required_env("PFAS_PROPS_SHEET"),
        env_bool("ENABLE_PFAS_PROPS_INJECTION", True),
    )
    standardize_water_type(df)
    normalize_organic_matter_units(df)
    recover_named_organic_conditions(df)
    recover_water_type_raw_context(df)
    apply_synthetic_water_from_reported_additives(df)
    apply_default_water_context(df)
    # TOC, DOC, and TDS describe one water matrix per row. Average only close
    # repeated values; preserve and blank broad multi-condition reports.
    normalize_water_quality_scalar_fields(
        df,
        manual_range_average_overrides=manual_range_average_overrides,
    )
    materialize_organic_carbon(df)
    normalize_pore_volumes(df)
    materialize_ssa_avg(df)
    compute_ionic_strength(df)
    materialize_additive_ml_features(df)
    enrich_adsorbent_fields(df, adsorbent_config)
    normalize_adsorbent_scalar_fields(
        df,
        manual_adsorbent_scalar_overrides=manual_adsorbent_scalar_overrides,
    )
    # Preserve raw reported values in Standardized_Full, then convert the
    # model-facing elemental-composition and porosity fields to fractions.
    # This runs after enrichment so injected adsorbent properties receive the
    # same guardrails as directly extracted values.
    normalize_adsorbent_fraction_fields(df)
    # Keep the paired reported zeta-potential fields and one clean, pH-aware
    # converted value together in Standardized_Full for traceability.
    df[ZETA_POTENTIAL_NEAR_PH_7_COLUMN] = zeta_potential_near_ph_7(df)

    _ensure_columns(df, ["normalization_trace", "normalization_issues"], default="")
    df["normalization_trace"] = df["normalization_trace"].astype("object")
    df["normalization_issues"] = df["normalization_issues"].astype("object")

    infer_kf_unit_from_langmuir(df)
    infer_pso_units(df)

    # Resolve pH and solution-volume ranges before unit conversion. This keeps
    # narrow pH ranges as usable center values and prevents unresolved volume
    # ranges from being used as midpoint volumes in concentration conversions.
    materialize_general_range_context(df)

    standardize_unit_parameters(df)

    # Store unresolved snapshot C0/dosage/contact-time ranges/lists as context and
    # blank only the standardized ML-ready single-value fields before Kd/Ce derivation.
    materialize_snapshot_range_context(df)

    # After unit conversions (including Nominal_grain_size -> Nominal_grain_size_value_mm),
    # refine activated carbon subcategories (PAC vs GAC) using particle size.
    infer_ac_subcategory_from_nominal_grain_size(df)
    # Apply canonical categorical labels only after enrichment and size-based
    # inference have finalized the model-facing adsorbent fields. The raw
    # reported values remain available in Standardized_Full.
    normalize_model_facing_adsorbent_categories(df)
    # Nonionic resins share the Resin model family with ion-exchange resins,
    # but their IEC is structurally zero rather than missing/not applicable.
    materialize_nonionic_resin_iec_defaults(df)

    derive_endpoint_columns(df)
    return df


def main() -> None:
    _require_orchestrated()
    here = os.path.dirname(__file__)
    input_path = required_env("UNIT_STANDARDIZER_INPUT_PATH")
    output_path = required_env("UNIT_STANDARDIZER_OUTPUT_PATH")
    sheet_name = required_env("UNIT_STANDARDIZER_SHEET_NAME")
    adsorbent_config = _adsorbent_config()
    manual_range_average_overrides = load_manual_range_average_overrides(
        os.getenv("PERFORMANCE_NORMALIZATION_DECISIONS_PATH")
    )
    manual_adsorbent_scalar_overrides = load_manual_adsorbent_scalar_overrides(
        os.getenv("PERFORMANCE_NORMALIZATION_DECISIONS_PATH")
    )

    if env_bool("ENABLE_ADSORBENT_DATABASE_BUILD", True):
        build_consolidated_adsorbent_database(
            adsorbent_config,
            _standardize_adsorbent_numeric_values,
        )
        if env_bool("ENABLE_AUTHORITATIVE_ADSORBENT_DATABASE_BUILD", True):
            build_authoritative_adsorbent_database(adsorbent_config)

    df = load_input_workbook(input_path, sheet_name)
    df = normalize_dataframe(
        df,
        adsorbent_config,
        manual_range_average_overrides=manual_range_average_overrides,
        manual_adsorbent_scalar_overrides=manual_adsorbent_scalar_overrides,
    )

    out_dir = os.path.dirname(input_path)
    in_base, in_ext = os.path.splitext(os.path.basename(input_path))
    out_xlsx = output_path or os.path.join(out_dir, f"{in_base}_unit_standardized{in_ext}")
    write_output_workbook(
        df,
        out_xlsx,
        os.path.join(here, "column_order_sheet1.csv"),
        PINNED_SHEET1_COLUMNS,
        AUDIT_OUTPUT_COLUMNS,
        [],
        RANGE_TRACE_COLUMNS,
        reference_dosage_mg_l=env_float("REFERENCE_DOSAGE_MG_L", 25.0),
        reference_dosage_tolerance_mg_l=env_float(
            "REFERENCE_DOSAGE_TOLERANCE_MG_L",
            5.0,
        ),
    )


if __name__ == "__main__":
    main()
