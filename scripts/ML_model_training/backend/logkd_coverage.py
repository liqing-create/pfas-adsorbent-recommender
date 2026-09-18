"""Target-free evidence diagnostics for logKd model inputs.

The absolute evidence-coverage functions are retained for the whole-database
audit. Model runs use the reference-retention functions instead: they compare a
training partition with the complete eligible pre-split modelling cohort using
a fixed candidate-feature manifest and fixed value definitions. Neither family
ranks or constructs allocations, and neither uses the target, predictions, or
feature importance.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import numpy as np
import pandas as pd

from . import logkd_config as cfg
from .logkd_data import clean_text, normalize_pfas_key
from .logkd_features import adsorbent_unit_keys, feature_bucket, feature_value_mask


REFERENCE_RETENTION_BLOCKS = (
    "Experimental conditions",
    "Adsorbent properties",
    "PFAS characteristics",
)
REFERENCE_RETENTION_BLOCK_SLUGS = {
    "Experimental conditions": "experimental_conditions",
    "Adsorbent properties": "adsorbent_properties",
    "PFAS characteristics": "pfas_characteristics",
}
REFERENCE_RETENTION_SUMMARY_METRICS = tuple(
    f"{metric}_{slug}"
    for metric in (
        "database_contrast",
        "training_contrast",
        "contrast_retention",
    )
    for slug in REFERENCE_RETENTION_BLOCK_SLUGS.values()
)


def _clean_column(df: pd.DataFrame, column: str) -> pd.Series:
    if column not in df.columns:
        return pd.Series("", index=df.index, dtype="object")
    return df[column].map(clean_text)


def _study_keys(df: pd.DataFrame) -> pd.Series:
    return _clean_column(df, "study_no")


def _pfas_keys(df: pd.DataFrame) -> pd.Series:
    if "PFAS_name" not in df.columns:
        return pd.Series("", index=df.index, dtype="object")
    return df["PFAS_name"].map(normalize_pfas_key).map(clean_text)


def availability_unit_keys(df: pd.DataFrame, bucket: str) -> tuple[str, pd.Series]:
    """Return the independent reporting unit appropriate to a feature family."""

    study = _study_keys(df)
    if bucket == "Experimental conditions":
        return "study", study
    if bucket == "Adsorbent properties":
        return "study-specific adsorbent", adsorbent_unit_keys(df)
    if bucket == "PFAS characteristics":
        return "unique PFAS", _pfas_keys(df)
    return "model row", pd.Series([str(index) for index in df.index], index=df.index, dtype="object")


def _rounded_numeric_values(df: pd.DataFrame, feature: str) -> pd.Series:
    decimals = int(cfg.NUMERIC_FEATURE_ROUND_DECIMALS.get(feature, cfg.DEFAULT_NUMERIC_ROUND_DECIMALS))
    return pd.to_numeric(df.get(feature, pd.Series(np.nan, index=df.index)), errors="coerce").round(decimals)


def _categorical_values(df: pd.DataFrame, feature: str) -> pd.Series:
    values = _clean_column(df, feature).astype("object")
    return values.mask(values.eq(""), np.nan)


def _value_labels(
    df: pd.DataFrame,
    feature: str,
    kind: str,
    numeric_bins: int,
) -> tuple[pd.Series, int, str]:
    """Return stable labels for diversity calculations and their construction."""

    if kind == "categorical":
        values = _categorical_values(df, feature)
        return values.astype("object"), int(values.nunique(dropna=True)), "observed_categorical_levels"

    values = _rounded_numeric_values(df, feature)
    observed = values.dropna()
    unique = int(observed.nunique())
    if unique <= 1:
        return values.astype("object"), unique, "rounded_numeric_values"
    if unique <= numeric_bins:
        return values.astype("object"), unique, "rounded_numeric_values"
    quantiles = np.linspace(0.0, 1.0, numeric_bins + 1)
    edges = np.unique(np.quantile(observed.to_numpy(float), quantiles))
    if len(edges) <= 2:
        return values.astype("object"), unique, "rounded_numeric_values_due_to_collapsed_quantiles"
    labels = pd.cut(values, bins=edges, include_lowest=True, duplicates="drop").astype("object")
    return labels, int(labels.nunique(dropna=True)), "reference_quantile_bins"


def _entropy(probabilities: Iterable[float]) -> float:
    values = np.asarray(list(probabilities), dtype=float)
    values = values[values > 0]
    return float(-(values * np.log(values)).sum()) if len(values) else 0.0


def _study_weighted_distribution(study_value: pd.DataFrame) -> pd.Series:
    """Make every reporting study contribute one total unit of weight."""

    per_study = (
        study_value.groupby(["study", "value"], dropna=False)
        .size()
        .rename("count")
        .reset_index()
    )
    per_study["study_total"] = per_study.groupby("study", dropna=False)["count"].transform("sum")
    per_study["weight"] = per_study["count"] / per_study["study_total"]
    return per_study.groupby("value", dropna=False)["weight"].sum() / per_study["study"].nunique()


def evidence_coverage_by_feature(
    df: pd.DataFrame,
    selected_features: Iterable[str],
    numeric_features: Iterable[str],
    categorical_features: Iterable[str],
    feature_manifest: pd.DataFrame | None = None,
    *,
    numeric_bins: int = cfg.DEFAULT_EVIDENCE_COVERAGE_NUMERIC_BINS,
    scope: str = "training_refit",
) -> pd.DataFrame:
    """Measure availability and between/within-study variation for selected inputs."""

    selected = tuple(dict.fromkeys(str(feature) for feature in selected_features))
    numeric = set(str(feature) for feature in numeric_features)
    categorical = set(str(feature) for feature in categorical_features)
    if set(selected).difference(numeric | categorical):
        missing = sorted(set(selected).difference(numeric).difference(categorical))
        raise ValueError(f"Evidence-coverage features lack a declared kind: {missing}")
    if numeric_bins < 2:
        raise ValueError("numeric_bins must be at least two.")

    manifest = pd.DataFrame() if feature_manifest is None else feature_manifest.copy()
    if not manifest.empty and "feature" in manifest.columns:
        manifest["feature"] = manifest["feature"].map(clean_text)
        manifest = manifest.drop_duplicates("feature", keep="first").set_index("feature")

    records: list[dict[str, Any]] = []
    study = _study_keys(df)
    reporting_study_mask = study.ne("")
    for feature in selected:
        kind = "numeric" if feature in numeric else "categorical"
        manifest_row = manifest.loc[feature] if feature in manifest.index else pd.Series(dtype="object")
        bucket = clean_text(manifest_row.get("bucket", "")) or feature_bucket(feature)
        unit_label, unit_keys = availability_unit_keys(df, bucket)
        observed = feature_value_mask(df, feature, kind)
        values, value_count, value_definition = _value_labels(df, feature, kind, numeric_bins)
        observed = observed & values.notna()

        eligible_units = unit_keys.ne("")
        reporting_units = unit_keys.loc[eligible_units & observed].nunique(dropna=True)
        total_units = unit_keys.loc[eligible_units].nunique(dropna=True)
        unit_availability = reporting_units / total_units if total_units else np.nan
        row_availability = float(observed.mean()) if len(df) else np.nan

        study_values = pd.DataFrame({
            "study": study.loc[reporting_study_mask & observed].astype(str),
            "value": values.loc[reporting_study_mask & observed].astype(str),
        })
        reporting_studies = int(study_values["study"].nunique())
        total_studies = int(study.loc[reporting_study_mask].nunique())
        if value_count <= 1 or study_values.empty:
            overall_value_diversity = 0.0 if value_count == 1 else np.nan
            share_of_variation_within_studies = np.nan
            between_share = np.nan
        else:
            distribution = _study_weighted_distribution(study_values)
            total_entropy = _entropy(distribution.to_numpy(float))
            maximum_entropy = float(np.log(value_count))
            overall_value_diversity = total_entropy / maximum_entropy if maximum_entropy else 0.0
            within_entropies: list[float] = []
            for _, group in study_values.groupby("study", sort=False):
                proportions = group["value"].value_counts(normalize=True)
                within_entropies.append(_entropy(proportions.to_numpy(float)))
            mean_within_entropy = float(np.mean(within_entropies)) if within_entropies else 0.0
            share_of_variation_within_studies = mean_within_entropy / total_entropy if total_entropy else np.nan
            between_share = (
                1.0 - share_of_variation_within_studies
                if np.isfinite(share_of_variation_within_studies)
                else np.nan
            )

        if reporting_studies:
            varying_studies = int(study_values.groupby("study", dropna=False)["value"].nunique().ge(2).sum())
            studies_with_multiple_values = varying_studies / reporting_studies
        else:
            varying_studies = 0
            studies_with_multiple_values = np.nan

        records.append({
            "retention_scope": scope,
            "feature": feature,
            "bucket": bucket,
            "kind": kind,
            "availability_unit": unit_label,
            "total_rows": int(len(df)),
            "observed_rows": int(observed.sum()),
            "row_availability": row_availability,
            "total_availability_units": int(total_units),
            "reporting_availability_units": int(reporting_units),
            "unit_availability": unit_availability,
            "total_studies": total_studies,
            "reporting_studies": reporting_studies,
            "varying_reporting_studies": varying_studies,
            "studies_with_multiple_values": studies_with_multiple_values,
            "value_level_or_bin_count": int(value_count),
            "value_definition": value_definition,
            "overall_value_diversity": overall_value_diversity,
            "share_of_variation_within_studies": share_of_variation_within_studies,
            "between_study_variation_share": between_share,
        })
    return pd.DataFrame(records)


def _fixed_reference_value_labels(
    reference_df: pd.DataFrame,
    training_df: pd.DataFrame,
    feature: str,
    kind: str,
    numeric_bins: int,
) -> tuple[pd.Series, pd.Series, int, str]:
    """Label reference and training values with one reference-defined scale."""

    if kind == "categorical":
        reference = _categorical_values(reference_df, feature).astype("object")
        training = _categorical_values(training_df, feature).astype("object")
        return (
            reference,
            training,
            int(reference.nunique(dropna=True)),
            "reference_categorical_levels",
        )

    reference = _rounded_numeric_values(reference_df, feature)
    training = _rounded_numeric_values(training_df, feature)
    observed = reference.dropna()
    unique = int(observed.nunique())
    if unique <= 1:
        return (
            reference.astype("object"),
            training.astype("object"),
            unique,
            "reference_rounded_numeric_values",
        )
    if unique <= numeric_bins:
        return (
            reference.astype("object"),
            training.astype("object"),
            unique,
            "reference_rounded_numeric_values",
        )

    quantiles = np.linspace(0.0, 1.0, numeric_bins + 1)
    edges = np.unique(np.quantile(observed.to_numpy(float), quantiles))
    if len(edges) <= 2:
        return (
            reference.astype("object"),
            training.astype("object"),
            unique,
            "reference_rounded_numeric_values_due_to_collapsed_quantiles",
        )
    reference_labels = pd.cut(
        reference,
        bins=edges,
        include_lowest=True,
        duplicates="drop",
    ).astype("object")
    training_labels = pd.cut(
        training,
        bins=edges,
        include_lowest=True,
        duplicates="drop",
    ).astype("object")
    return (
        reference_labels,
        training_labels,
        int(reference_labels.nunique(dropna=True)),
        "fixed_reference_quantile_bins",
    )


def _unit_weighted_value_distribution(unit_keys: pd.Series, labels: pd.Series) -> pd.Series:
    """Give every reporting evidence unit one total unit of probability mass."""

    values = pd.DataFrame({"unit": unit_keys, "value": labels})
    values = values.loc[values["unit"].ne("") & values["value"].notna()]
    if values.empty:
        return pd.Series(dtype=float)
    counts = (
        values.groupby(["unit", "value"], dropna=False, observed=True)
        .size()
        .rename("count")
        .reset_index()
    )
    counts["unit_total"] = counts.groupby("unit", dropna=False)["count"].transform("sum")
    counts["weight"] = counts["count"] / counts["unit_total"]
    return (
        counts.groupby("value", dropna=False, observed=True)["weight"].sum()
        / counts["unit"].nunique()
    )


def _contrasting_studies(df: pd.DataFrame, labels: pd.Series) -> set[str]:
    study = _study_keys(df)
    values = pd.DataFrame({"study": study, "value": labels})
    values = values.loc[values["study"].ne("") & values["value"].notna()]
    if values.empty:
        return set()
    counts = values.groupby("study", dropna=False, observed=True)["value"].nunique()
    return set(counts.index[counts.ge(2)].astype(str))


def reference_retention_by_feature(
    reference_df: pd.DataFrame,
    training_df: pd.DataFrame,
    reference_manifest: pd.DataFrame,
    *,
    numeric_bins: int = cfg.DEFAULT_EVIDENCE_COVERAGE_NUMERIC_BINS,
    scope: str = "outer_training_vs_complete_reference",
) -> pd.DataFrame:
    """Measure evidence retained from a fixed, complete pre-split reference.

    The diagnostic feature set is fixed by ``reference_manifest`` and is not
    conditioned on which inputs survive training-scope feature selection.
    """

    if numeric_bins < 2:
        raise ValueError("numeric_bins must be at least two.")
    required = {"feature", "bucket", "kind"}
    missing_columns = sorted(required.difference(reference_manifest.columns))
    if missing_columns:
        raise ValueError(
            "Reference-retention manifest lacks required columns: "
            f"{missing_columns}"
        )

    manifest = reference_manifest.copy()
    manifest["feature"] = manifest["feature"].map(clean_text)
    manifest["bucket"] = manifest["bucket"].map(clean_text)
    manifest["kind"] = manifest["kind"].map(clean_text)
    manifest = manifest.loc[
        manifest["feature"].ne("")
        & manifest["feature"].isin(reference_df.columns)
        & manifest["bucket"].isin(REFERENCE_RETENTION_BLOCKS)
        & manifest["kind"].isin({"numeric", "categorical"})
    ].drop_duplicates("feature", keep="first")

    records: list[dict[str, Any]] = []
    for row in manifest.itertuples(index=False):
        feature = str(row.feature)
        bucket = str(row.bucket)
        kind = str(row.kind)
        unit_label, reference_units = availability_unit_keys(reference_df, bucket)
        _, training_units = availability_unit_keys(training_df, bucket)
        reference_unit_set = set(reference_units.loc[reference_units.ne("")].astype(str))
        training_unit_set = set(training_units.loc[training_units.ne("")].astype(str))
        retained_unit_count = len(reference_unit_set.intersection(training_unit_set))
        entity_retention = (
            retained_unit_count / len(reference_unit_set)
            if reference_unit_set
            else np.nan
        )

        reference_labels, training_labels, value_count, value_definition = (
            _fixed_reference_value_labels(
                reference_df,
                training_df,
                feature,
                kind,
                numeric_bins,
            )
        )
        reference_distribution = _unit_weighted_value_distribution(
            reference_units,
            reference_labels,
        )
        training_observed_values = set(training_labels.dropna().tolist())
        if reference_distribution.empty:
            value_support_retention = np.nan
        else:
            retained_values = reference_distribution.index.isin(training_observed_values)
            value_support_retention = float(reference_distribution.loc[retained_values].sum())

        reference_contrasts = _contrasting_studies(reference_df, reference_labels)
        training_contrasts = _contrasting_studies(training_df, training_labels)
        retained_contrast_count = len(reference_contrasts.intersection(training_contrasts))
        reference_studies = _study_keys(reference_df)
        reference_study_count = int(
            reference_studies.loc[reference_studies.ne("")].nunique()
        )
        database_contrast = (
            len(reference_contrasts) / reference_study_count
            if reference_study_count
            else np.nan
        )
        training_contrast = (
            retained_contrast_count / reference_study_count
            if reference_study_count
            else np.nan
        )
        contrast_retention = (
            retained_contrast_count / len(reference_contrasts)
            if reference_contrasts
            else np.nan
        )

        records.append({
            "coverage_scope": scope,
            "reference_scope": "complete_eligible_pre_split_modeling_cohort",
            "feature_scope": "fixed_reference_candidate_manifest",
            "feature": feature,
            "bucket": bucket,
            "kind": kind,
            "availability_unit": unit_label,
            "reference_total_units": int(len(reference_unit_set)),
            "training_retained_units": int(retained_unit_count),
            "entity_retention": entity_retention,
            "reference_value_level_or_bin_count": int(value_count),
            "training_observed_reference_value_level_or_bin_count": int(
                len(training_observed_values.intersection(set(reference_distribution.index)))
            ),
            "reference_value_definition": value_definition,
            "reference_value_support_mass_retention": value_support_retention,
            "reference_total_studies": reference_study_count,
            "reference_contrasting_studies": int(len(reference_contrasts)),
            "training_retained_contrasting_studies": int(retained_contrast_count),
            "database_contrast": database_contrast,
            "training_contrast": training_contrast,
            "contrast_retention": contrast_retention,
        })
    return pd.DataFrame(records)


def summarize_reference_retention(
    feature_retention: pd.DataFrame,
) -> dict[str, float]:
    """Return block contrast support and retention without a combined index.

    Database and training contrast use the same complete-reference denominator,
    so their difference directly shows the contrast support removed by the
    split. Contrast retention remains conditional on the contrasts that existed
    in the reference cohort.
    """

    records: dict[str, float] = {}
    for block in REFERENCE_RETENTION_BLOCKS:
        block_rows = feature_retention.loc[feature_retention["bucket"].eq(block)]
        if block_rows.empty:
            continue
        slug = REFERENCE_RETENTION_BLOCK_SLUGS[block]
        total_studies = pd.to_numeric(
            block_rows["reference_total_studies"], errors="coerce"
        )
        reference_contrasts = pd.to_numeric(
            block_rows["reference_contrasting_studies"], errors="coerce"
        )
        training_contrasts = pd.to_numeric(
            block_rows["training_retained_contrasting_studies"], errors="coerce"
        )
        valid = total_studies.notna() & reference_contrasts.notna() & training_contrasts.notna()
        denominator = float(total_studies.loc[valid].sum())
        reference_total = float(reference_contrasts.loc[valid].sum())
        training_total = float(training_contrasts.loc[valid].sum())
        records[f"database_contrast_{slug}"] = (
            reference_total / denominator if denominator else np.nan
        )
        records[f"training_contrast_{slug}"] = (
            training_total / denominator if denominator else np.nan
        )
        records[f"contrast_retention_{slug}"] = (
            training_total / reference_total if reference_total else np.nan
        )
    return records
