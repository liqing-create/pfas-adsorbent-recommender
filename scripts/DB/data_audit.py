"""Figure-oriented whole-database PFAS logKd quality audit.

Writes a compact workbook for manuscript figures and data-quality monitoring.
Feature availability is descriptive only; model training calculates feature
eligibility inside each training partition and never uses this workbook as a
selection input.  Feature sheets cover the equilibrium model input space, so
batch/kinetic protocol variables excluded from that space are not reported.
"""

from __future__ import annotations

import argparse
import sys
from copy import copy
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
TRAINING_DIR = SCRIPT_DIR.parent / "ML_model_training"
if str(TRAINING_DIR) not in sys.path:
    sys.path.insert(0, str(TRAINING_DIR))

from backend import logkd_config as cfg
from backend.logkd_coverage import evidence_coverage_by_feature
from backend.logkd_data import clean_text, load_model_rows
from backend.logkd_features import (
    add_feature_availability_summary,
    adsorbent_unit_keys,
    feature_availability_audit,
    joint_support_coverage,
)


MODELS = ("AC", "Resin", "CDP", "Global")
TARGET = cfg.TARGET
REVIEW_QUEUE_COLUMNS = (
    "extraction_dataset",
    "study_no",
    "standardized_row_id",
    "source_row_index",
    "audit_reason",
    "decision",
)
# These are the automated exclusion flags already handled by the modeling
# pipeline.  The human-review queue intentionally targets suspicious records
# that are still passing these automatic checks.
AUTOMATED_UNRELIABLE_FLAG_COLUMNS = (
    "logKd_conversion_spike_flag",
    "logKd_conversion_dip_flag",
    "low_Ce_detection_limit_flag",
    "small_concentration_difference_flag",
    "low_removal_rate_flag",
    "high_apparent_removal_flag",
)
QC_FLAG_SCOPE_LABELS = (
    ("logKd_conversion_spike_flag", "Conversion instability: spike"),
    ("logKd_conversion_dip_flag", "Conversion instability: dip"),
    ("low_Ce_detection_limit_flag", "Low Ce"),
    ("small_concentration_difference_flag", "Small concentration difference"),
    ("low_removal_rate_flag", "Low removal"),
    ("high_apparent_removal_flag", "High apparent removal"),
)
DEFAULT_REVIEW_HIGH_LOGKD_THRESHOLD = 8.0
DEFAULT_REVIEW_LOW_LOGKD_THRESHOLD = -5.0
DEFAULT_REVIEW_HIGH_C0_THRESHOLD_MG_L = 50_000.0
EQUILIBRIUM_DETERMINATIONS = {
    "equilibrium_endpoint_reported",
    "removal_explicit_equilibrium",
    "removal_single_timepoint_assumed",
    "removal_last_timepoint_assumed",
}
NON_EQUILIBRIUM_DETERMINATIONS = {
    "kinetic_model_reported",
    "removal_explicit_non_equilibrium",
    "removal_intermediate_timepoint",
}
FIGURE_1_ADSORBENT_LABELS = {
    "Activated carbon": "AC",
    "Ion exchange resin": "Resin",
    "Nonionic resin": "Resin",
    "Cyclodextrin polymers": "CDP",
    "Clay/mineral": "Clay",
    "Unclassified": "Unknown",
}
FIGURE_1_PFAS_CLASS_LABELS = {
    "Aromatic PFASs": "Aromatic PFAS",
    "PASF-based substances": "PASF",
    "PFCAs, cyclic": "Cyclic PFCAs",
    "PFSA derivatives": "PFSA deriv.",
    "PolyFCA derivatives": "PolyFCA deriv.",
    "others": "Other",
    "Unclassified": "Unknown",
}
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a compact PFAS logKd figure-data workbook.")
    parser.add_argument("--input-path", type=Path, default=cfg.DEFAULT_INPUT)
    parser.add_argument("--sheet-name", default=cfg.SHEET_NAME)
    parser.add_argument("--database-scope-sheet", default="Standardized_Full")
    parser.add_argument(
        "--output-path",
        type=Path,
        default=cfg.DEFAULT_INPUT.with_name("data_audit.xlsx"),
    )
    parser.add_argument("--pfas-features-path", type=Path, default=cfg.DEFAULT_PFAS_FEATURES)
    parser.add_argument("--pfas-features-sheet", default=cfg.DEFAULT_PFAS_FEATURES_SHEET)
    parser.add_argument(
        "--data-mode",
        choices=["baseline", "drop_unreliable"],
        default="drop_unreliable",
    )
    parser.add_argument("--corr-min-pairs", type=int, default=20)
    parser.add_argument("--corr-threshold", type=float, default=0.85)
    parser.add_argument("--max-categorical-levels", type=int, default=30)
    parser.add_argument(
        "--max-numeric-exact-levels",
        type=int,
        default=20,
        help="Use exact-value bars for numeric variables with no more than this many distinct reported values.",
    )
    parser.add_argument(
        "--max-histogram-bins",
        type=int,
        default=20,
        help="Maximum number of common histogram bins for a continuous numeric variable.",
    )
    parser.add_argument(
        "--review-decisions-path",
        type=Path,
        default=None,
        help=(
            "Append new human-review candidates to this CSV. Defaults to "
            "record_review_decisions.csv beside --input-path."
        ),
    )
    parser.add_argument(
        "--review-high-logkd-threshold",
        type=float,
        default=DEFAULT_REVIEW_HIGH_LOGKD_THRESHOLD,
        help="Queue unflagged records with logKd at or above this value.",
    )
    parser.add_argument(
        "--review-low-logkd-threshold",
        type=float,
        default=DEFAULT_REVIEW_LOW_LOGKD_THRESHOLD,
        help="Queue unflagged records with logKd at or below this value.",
    )
    parser.add_argument(
        "--review-high-c0-threshold-mg-l",
        type=float,
        default=DEFAULT_REVIEW_HIGH_C0_THRESHOLD_MG_L,
        help="Queue unflagged records with initial PFAS concentration at or above this value.",
    )
    return parser.parse_args()


def present(series: pd.Series) -> pd.Series:
    text = series.map(clean_text)
    return series.notna() & ~text.str.lower().isin(cfg.MISSING_TOKENS)


def numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series.where(present(series)), errors="coerce")


def column_or(df: pd.DataFrame, name: str, fallback: pd.Series) -> pd.Series:
    return df[name] if name in df.columns else fallback


def source_row_key(value: object) -> str:
    """Return a stable text representation for source-row identifiers."""
    if pd.isna(value):
        return ""
    if isinstance(value, (float, np.floating)) and float(value).is_integer():
        return str(int(value))
    return str(value).strip()


def truthy_flag(df: pd.DataFrame, column: str) -> pd.Series:
    """Read a boolean-like flag while allowing older input workbooks."""
    if column not in df.columns:
        return pd.Series(False, index=df.index)
    values = df[column]
    if values.dtype == bool:
        return values.fillna(False)
    return values.map(clean_text).str.casefold().isin(cfg.UNRELIABLE_FLAG_TRUTHY_VALUES)


def qc_flag_scope_rows(df: pd.DataFrame) -> list[dict[str, object]]:
    """Return non-exclusive QC counts for Equilibrium_Data Scope counts."""
    rows: list[dict[str, object]] = [
        {
            "Scope item": "QC flags below (Equilibrium_Data rows; non-exclusive)",
            "# of records": pd.NA,
        }
    ]
    any_flag = pd.Series(False, index=df.index)
    all_columns_present = True
    for column, label in QC_FLAG_SCOPE_LABELS:
        if column not in df.columns:
            all_columns_present = False
            rows.append(
                {
                    "Scope item": f"QC: {label} flagged (column unavailable)",
                    "# of records": pd.NA,
                }
            )
            continue
        flagged = truthy_flag(df, column)
        any_flag |= flagged
        rows.append(
            {
                "Scope item": f"QC: {label} flagged",
                "# of records": int(flagged.sum()),
            }
        )
    rows.append(
        {
            "Scope item": "QC: Any automated QC flag (unique rows)",
            "# of records": int(any_flag.sum()) if all_columns_present else pd.NA,
        }
    )
    return rows


def pre_expansion_scope_rows(
    raw: pd.DataFrame,
    determination: pd.Series,
) -> list[dict[str, object]]:
    """Count extracted performance records before isotherm rows were expanded.

    ``source_row_index`` is stamped on the merged extraction table before
    multi-valued isotherm designs are split into one row per C0/dosage
    condition, and that split only adds rows.  One source row therefore
    remains one extracted performance record.
    """
    if "source_row_index" not in raw.columns:
        return [
            {
                "Scope item": "Raw extracted records (source_row_index unavailable)",
                "# of records": pd.NA,
            }
        ]
    source_keys = raw["source_row_index"].map(source_row_key)
    first_row_per_source = source_keys.ne("") & ~source_keys.duplicated()
    kinetic = first_row_per_source & determination.isin(NON_EQUILIBRIUM_DETERMINATIONS)
    raw_records = int(first_row_per_source.sum())
    kinetic_records = int(kinetic.sum())
    return [
        {
            "Scope item": "Raw extracted performance records (before isotherm expansion)",
            "# of records": raw_records,
        },
        {
            "Scope item": "Raw records: clearly non-equilibrium (kinetic) excluded",
            "# of records": kinetic_records,
        },
        {
            "Scope item": "Raw performance records retained (before isotherm expansion)",
            "# of records": raw_records - kinetic_records,
        },
        {
            "Scope item": "Rows added by isotherm expansion",
            "# of records": len(raw) - raw_records,
        },
    ]


def first_text(group: pd.DataFrame, column: str) -> str:
    if column not in group.columns:
        return ""
    values = group[column].map(clean_text)
    values = values[values.ne("")]
    return values.iloc[0] if not values.empty else ""


def joined_text(group: pd.DataFrame, column: str) -> str:
    if column not in group.columns:
        return ""
    values = group[column].map(clean_text)
    values = values[values.ne("")].drop_duplicates()
    return "; ".join(values.tolist())


def format_number(value: float) -> str:
    return f"{value:.6g}"


def build_human_review_candidates(df: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    """Identify unflagged target and concentration tails deserving manual review.

    This is a pre-model input-data check.  It does not depend on predictions
    or make a reliability decision: candidates are emitted once per source
    row so a reviewer can decide on the original experimental record, not
    each expanded isotherm row.
    """
    if "source_row_index" not in df.columns:
        raise ValueError("Human-review candidate generation requires source_row_index.")
    if TARGET not in df.columns:
        raise ValueError(f"Human-review candidate generation requires {TARGET!r}.")

    target = numeric(df[TARGET])
    c0 = numeric(column_or(df, "PFAS_C0_value_mg/L", pd.Series(np.nan, index=df.index)))
    automated_unreliable = pd.Series(False, index=df.index)
    for column in AUTOMATED_UNRELIABLE_FLAG_COLUMNS:
        automated_unreliable |= truthy_flag(df, column)

    high_logkd = target.ge(args.review_high_logkd_threshold)
    low_logkd = target.le(args.review_low_logkd_threshold)
    high_c0 = c0.ge(args.review_high_c0_threshold_mg_l)
    candidate_mask = (
        target.notna()
        & np.isfinite(target)
        & ~automated_unreliable
        & (high_logkd | low_logkd | high_c0)
    )
    candidates = df.loc[candidate_mask].copy()
    if candidates.empty:
        return pd.DataFrame(columns=REVIEW_QUEUE_COLUMNS)

    candidates["__source_row_key"] = candidates["source_row_index"].map(source_row_key)
    candidates = candidates.loc[candidates["__source_row_key"].ne("")].copy()
    candidates["__target"] = target.loc[candidates.index]
    candidates["__c0"] = c0.loc[candidates.index]

    rows: list[dict[str, str]] = []
    for source_key, group in candidates.groupby("__source_row_key", sort=False):
        group_high = group["__target"].ge(args.review_high_logkd_threshold).any()
        group_low = group["__target"].le(args.review_low_logkd_threshold).any()
        group_high_c0 = group["__c0"].ge(args.review_high_c0_threshold_mg_l).any()
        reasons: list[str] = []
        if group_high:
            reasons.append(
                "CRITICAL unflagged high logKd="
                f"{format_number(float(group['__target'].max()))} "
                f">= {format_number(args.review_high_logkd_threshold)}"
            )
        if group_low:
            reasons.append(
                "CRITICAL unflagged low logKd="
                f"{format_number(float(group['__target'].min()))} "
                f"<= {format_number(args.review_low_logkd_threshold)}"
            )
        if group_high_c0:
            reasons.append(
                "HIGH initial PFAS concentration="
                f"{format_number(float(group['__c0'].max()))} mg/L "
                f">= {format_number(args.review_high_c0_threshold_mg_l)} mg/L"
            )
        rows.append(
            {
                "extraction_dataset": first_text(group, "extraction_dataset"),
                "study_no": first_text(group, "study_no"),
                "standardized_row_id": joined_text(group, "standardized_row_id"),
                "source_row_index": source_key,
                "audit_reason": "; ".join(reasons),
                "decision": "",
            }
        )
    return pd.DataFrame(rows, columns=REVIEW_QUEUE_COLUMNS)


def append_review_candidates(review_path: Path, candidates: pd.DataFrame) -> tuple[int, int]:
    """Append only previously unseen source-row candidates, preserving decisions."""
    review_path = review_path.expanduser()
    if review_path.exists() and review_path.stat().st_size:
        existing = pd.read_csv(review_path, dtype=str, keep_default_na=False)
        missing = [column for column in REVIEW_QUEUE_COLUMNS if column not in existing.columns]
        if missing:
            raise ValueError(
                f"Review CSV {review_path} is missing required columns: {missing}. "
                f"Expected {list(REVIEW_QUEUE_COLUMNS)}."
            )
        existing_keys = set(existing["source_row_index"].map(source_row_key))
    else:
        existing_keys = set()

    to_append = candidates.loc[~candidates["source_row_index"].isin(existing_keys)].copy()
    if not to_append.empty:
        review_path.parent.mkdir(parents=True, exist_ok=True)
        to_append.to_csv(
            review_path,
            mode="a",
            index=False,
            header=not review_path.exists() or review_path.stat().st_size == 0,
        )
    return len(candidates), len(to_append)


def identity_keys(df: pd.DataFrame) -> pd.Series:
    fallback = column_or(df, "study_no", pd.Series("missing-study", index=df.index)).map(clean_text)
    adsorbent = column_or(df, "adsorbent_id", pd.Series("missing-adsorbent", index=df.index)).map(clean_text)
    fallback = (fallback + "|" + adsorbent).replace("|", "missing")
    return column_or(df, "adsorbent_identity_key", fallback).map(clean_text).replace("", "missing")


def source_record_count(df: pd.DataFrame) -> int:
    """Count distinct extracted records behind a cohort, before isotherm expansion.

    ``source_row_index`` identifies one extracted performance record on the
    merged extraction table.  Multi-valued isotherm designs are split into one
    standardized row per C0/dosage condition afterwards, so distinct source
    keys recover the raw extracted-record count contributing to the cohort.
    """
    if "source_row_index" not in df.columns:
        return 0
    keys = df["source_row_index"].map(source_row_key)
    return int(keys[keys.ne("")].nunique())


def study_instance_keys(df: pd.DataFrame) -> pd.Series:
    """Count study-specific adsorbent instances exactly as the model audit does.

    ``adsorbent_study_instance_key`` keys on the study's own ``adsorbent_id``
    spelling, so it would not merge two labels a single study used for one
    material.  The shared helper keys on the resolved identity instead, which
    keeps this cohort headline consistent with the unit behind every
    availability statistic in the workbook.
    """
    return adsorbent_unit_keys(df).replace("", "missing")


def loaded_model_tables(args: argparse.Namespace) -> tuple[dict[str, pd.DataFrame], pd.DataFrame, dict[str, tuple[list[str], list[str], list[str]]]]:
    model_dfs: dict[str, pd.DataFrame] = {}
    coverage: list[pd.DataFrame] = []
    feature_sets: dict[str, tuple[list[str], list[str], list[str]]] = {}
    for model in MODELS:
        frame, _info, _catalog = load_model_rows(
            input_path=args.input_path,
            sheet_name=args.sheet_name,
            model=model,
            target=TARGET,
            pfas_features_path=args.pfas_features_path,
            pfas_features_sheet=args.pfas_features_sheet,
            data_mode=args.data_mode,
        )
        features, numeric_features, categorical_features, audit = feature_availability_audit(
            frame, model_name=model, pfas_structure_features="all"
        )
        audit = add_feature_availability_summary(audit, total_model_rows=len(frame))
        audit.insert(0, "model", model)
        summary_columns = [
            "model",
            "availability_plot_order",
            "feature",
            "total_model_rows",
            "nonmissing_rows",
            "missing_rows",
            "missing_fraction",
        ]
        audit = audit[summary_columns + [column for column in audit.columns if column not in summary_columns]]
        model_dfs[model] = frame
        coverage.append(audit)
        feature_sets[model] = (features, numeric_features, categorical_features)
    return model_dfs, pd.concat(coverage, ignore_index=True), feature_sets


def reported_endpoint(df: pd.DataFrame) -> pd.Series:
    """Assign concise Figure 1 reporting families, independent of convertibility."""
    source = column_or(df, "Kd_final_source", pd.Series("", index=df.index)).map(clean_text)
    langmuir = (
        source.eq("Langmuir")
        | present(column_or(df, "Langmuir_KL_value", pd.Series(np.nan, index=df.index)))
        | present(column_or(df, "Langmuir_Qm_value", pd.Series(np.nan, index=df.index)))
    )
    freundlich = (
        source.eq("Freundlich")
        | present(column_or(df, "Freundlich_KF_value", pd.Series(np.nan, index=df.index)))
        | present(column_or(df, "Freundlich_exponent_value", pd.Series(np.nan, index=df.index)))
    )
    masks = {
        "Kd": source.eq("Kd") | present(column_or(df, "Kd_value", pd.Series(np.nan, index=df.index))),
        "Removal": source.eq("Removal") | present(column_or(df, "Removal_rate", pd.Series(np.nan, index=df.index))),
        "Qe": source.eq("Qe") | present(column_or(df, "Qe_value", pd.Series(np.nan, index=df.index))),
        "Isotherm": langmuir | freundlich,
    }
    count = sum(mask.astype(int) for mask in masks.values())
    out = pd.Series("Other", index=df.index, dtype="object")
    for label, mask in masks.items():
        out.loc[mask] = label
    return out.mask(count.gt(1), "Multiple")


def concise_figure_labels(values: pd.Series, mapping: dict[str, str]) -> pd.Series:
    return values.map(clean_text).replace("", "Unclassified").replace(mapping)


def figure_1_tables(
    args: argparse.Namespace,
    qc_source: pd.DataFrame | None = None,
) -> dict[str, pd.DataFrame]:
    """Return the exact tables needed for Figure 1 without an abstract link table."""
    raw = pd.read_excel(args.input_path, sheet_name=args.database_scope_sheet)
    if qc_source is None:
        qc_source = pd.read_excel(args.input_path, sheet_name=args.sheet_name)
    determination = column_or(raw, "Equilibrium_determination", pd.Series("", index=raw.index)).map(clean_text)
    in_scope = determination.isin(EQUILIBRIUM_DETERMINATIONS)
    converted = in_scope & numeric(column_or(raw, TARGET, pd.Series(np.nan, index=raw.index))).notna()
    endpoint = reported_endpoint(raw)
    pfas_class = column_or(raw, "Second_Class", pd.Series("Unclassified", index=raw.index)).map(clean_text).replace("", "Unclassified")
    adsorbent_category = column_or(raw, "adsorbent_category", pd.Series("Unclassified", index=raw.index)).map(clean_text).replace("", "Unclassified")

    scope_summary = pd.DataFrame(
        [
            *pre_expansion_scope_rows(raw, determination),
            {"Scope item": "All standardized records", "# of records": len(raw)},
            {"Scope item": "Equilibrium or suspected-equilibrium records", "# of records": int(in_scope.sum())},
            {"Scope item": "Converted to logKd", "# of records": int(converted.sum())},
            {"Scope item": "In-scope records not converted", "# of records": int((in_scope & ~converted).sum())},
            {"Scope item": "Clearly non-equilibrium records excluded", "# of records": int(determination.isin(NON_EQUILIBRIUM_DETERMINATIONS).sum())},
            {"Scope item": "Unclear equilibrium status excluded", "# of records": int((~in_scope & ~determination.isin(NON_EQUILIBRIUM_DETERMINATIONS)).sum())},
            *qc_flag_scope_rows(qc_source),
        ]
    )
    flow = raw.loc[in_scope].copy()
    flow["PFAS class"] = concise_figure_labels(pfas_class.loc[in_scope], FIGURE_1_PFAS_CLASS_LABELS)
    flow["Adsorbent category"] = concise_figure_labels(
        adsorbent_category.loc[in_scope], FIGURE_1_ADSORBENT_LABELS
    )
    flow["Performance endpoint"] = endpoint.loc[in_scope]
    sankey = (
        flow.groupby(["PFAS class", "Adsorbent category", "Performance endpoint"], dropna=False)
        .size()
        .reset_index(name="# of records")
        .sort_values("# of records", ascending=False, kind="stable")
        .reset_index(drop=True)
    )
    status = pd.Series("Excluded from main flow", index=raw.index, dtype="object")
    status.loc[in_scope & ~converted] = "Not converted to logKd"
    status.loc[converted] = "Converted to logKd"
    conversion_outcomes = (
        pd.DataFrame({"endpoint": endpoint.loc[in_scope], "conversion_status": status.loc[in_scope]})
        .value_counts()
        .rename("record_count")
        .reset_index()
        .rename(
            columns={
                "endpoint": "Performance endpoint",
                "conversion_status": "Conversion status",
                "record_count": "# of records",
            }
        )
        .sort_values("# of records", ascending=False, kind="stable")
        .reset_index(drop=True)
    )
    reason_lookup = column_or(
        raw,
        "status_reason",
        pd.Series("", index=raw.index, dtype="object"),
    )
    reason_lookup.index = raw["standardized_row_id"]
    failure = raw.loc[in_scope & ~converted, ["standardized_row_id"]].copy()
    failure["conversion_reason"] = failure["standardized_row_id"].map(reason_lookup).fillna("Kd_final_missing")
    conversion_blockers = (
        failure["conversion_reason"]
        .str.split(";")
        .explode()
        .str.strip()
        .value_counts()
        .rename_axis("Conversion blocker")
        .reset_index(name="Affected unconverted records")
    )
    return {
        "Origin Sankey input": sankey,
        "Scope counts": scope_summary,
        "Conversion outcomes": conversion_outcomes,
        "Conversion blockers": conversion_blockers,
    }


def study_concentration(model_dfs: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Figure 2: contribution of each study to each model cohort."""
    output: list[pd.DataFrame] = []
    for model, df in model_dfs.items():
        studies = column_or(df, "study_no", pd.Series("missing", index=df.index)).map(clean_text).replace("", "missing")
        counts = studies.value_counts().rename_axis("study_no").reset_index(name="record_count")
        counts["record_fraction"] = counts["record_count"] / len(df)
        counts["cumulative_record_fraction"] = counts["record_fraction"].cumsum()
        counts.insert(0, "model", model)
        counts.insert(1, "rank", np.arange(1, len(counts) + 1))
        output.append(counts)
    return pd.concat(output, ignore_index=True)


def evidence_coverage(
    model_dfs: dict[str, pd.DataFrame],
    availability: pd.DataFrame,
    feature_sets: dict[str, tuple[list[str], list[str], list[str]]],
) -> pd.DataFrame:
    """Whole-database availability and variation diagnostics for all candidates."""

    frames: list[pd.DataFrame] = []
    audit_columns = [
        "feature", "bucket", "kind", "availability_plot_order",
    ]
    for model, df in model_dfs.items():
        features, numeric_features, categorical_features = feature_sets[model]
        manifest = availability.loc[
            availability["model"].eq(model), audit_columns
        ].copy()
        coverage = evidence_coverage_by_feature(
            df,
            features,
            numeric_features,
            categorical_features,
            manifest,
            scope="whole_database_audit",
        )
        if coverage.empty:
            continue

        coverage = coverage.merge(manifest, on=["feature", "bucket", "kind"], how="left")
        coverage.insert(0, "model", model)
        frames.append(coverage.sort_values("availability_plot_order", kind="stable"))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def correlation_pairs(model_dfs: dict[str, pd.DataFrame], feature_sets: dict[str, tuple[list[str], list[str], list[str]]], args: argparse.Namespace) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for model, df in model_dfs.items():
        numeric_features = feature_sets[model][1]
        for index, left in enumerate(numeric_features):
            for right in numeric_features[index + 1:]:
                pair = pd.DataFrame({"left": numeric(df[left]), "right": numeric(df[right])}).dropna()
                if len(pair) < args.corr_min_pairs or pair["left"].nunique() < 2 or pair["right"].nunique() < 2:
                    continue
                # Spearman rho is Pearson correlation of the two rank vectors;
                # this avoids making SciPy a runtime dependency for the audit.
                rho = float(pair["left"].rank().corr(pair["right"].rank()))
                rows.append({
                    "model": model,
                    "feature_1": left,
                    "feature_2": right,
                    "spearman_rho": rho,
                    "abs_spearman_rho": abs(rho),
                    "pairwise_n": len(pair),
                    "above_review_threshold": abs(rho) >= args.corr_threshold,
                })
    return pd.DataFrame(rows).sort_values(["model", "abs_spearman_rho"], ascending=[True, False])


def format_bin_value(value: float) -> str:
    return f"{value:.4g}"


def histogram_edges(reference_values: pd.Series, max_bins: int) -> np.ndarray:
    """Freedman-Diaconis bins, capped for a compact Origin-ready table."""
    values = pd.to_numeric(reference_values, errors="coerce").dropna().to_numpy(dtype=float)
    if values.size < 2 or np.isclose(values.min(), values.max()):
        return np.array([values.min(), values.max()])
    edges = np.histogram_bin_edges(values, bins="fd")
    bin_count = len(edges) - 1
    if bin_count < 4:
        bin_count = min(max_bins, max(4, int(np.ceil(np.sqrt(len(values))))))
    elif bin_count > max_bins:
        bin_count = max_bins
    return np.linspace(values.min(), values.max(), bin_count + 1)


def numeric_distribution_rows(
    *,
    model: str,
    feature: str,
    role: str,
    values: pd.Series,
    reference_values: pd.Series,
    total_rows: int,
    max_exact_levels: int,
    max_histogram_bins: int,
) -> list[dict[str, object]]:
    reported = numeric(values).dropna()
    if reported.empty:
        return []
    reference = numeric(reference_values).dropna()
    if reference.empty:
        reference = reported
    reported_rows = len(reported)
    missing_fraction = 1 - reported_rows / total_rows
    rows: list[dict[str, object]] = []
    if len(reference.unique()) <= max_exact_levels:
        counts = reported.value_counts().sort_index()
        for order, (value, count) in enumerate(counts.items(), start=1):
            rows.append(
                {
                    "model": model,
                    "feature": feature,
                    "role": role,
                    "distribution_type": "numeric_exact",
                    "plot_order": order,
                    "bin_left": float(value),
                    "bin_right": float(value),
                    "bin_center": float(value),
                    "bin_label": format_bin_value(float(value)),
                    "record_count": int(count),
                    "record_fraction": count / total_rows,
                    "reported_rows": reported_rows,
                    "missing_fraction": missing_fraction,
                }
            )
        return rows

    edges = histogram_edges(reference, max_histogram_bins)
    counts, _ = np.histogram(reported.to_numpy(dtype=float), bins=edges)
    for order, (left, right, count) in enumerate(zip(edges[:-1], edges[1:], counts), start=1):
        rows.append(
            {
                "model": model,
                "feature": feature,
                "role": role,
                "distribution_type": "numeric_histogram",
                "plot_order": order,
                "bin_left": float(left),
                "bin_right": float(right),
                "bin_center": float((left + right) / 2),
                "bin_label": f"{format_bin_value(float(left))}–{format_bin_value(float(right))}",
                "record_count": int(count),
                "record_fraction": count / total_rows,
                "reported_rows": reported_rows,
                "missing_fraction": missing_fraction,
            }
        )
    return rows


def feature_distributions(
    model_dfs: dict[str, pd.DataFrame],
    availability: pd.DataFrame,
    feature_sets: dict[str, tuple[list[str], list[str], list[str]]],
    max_levels: int,
    max_exact_levels: int,
    max_histogram_bins: int,
) -> pd.DataFrame:
    """Origin-ready distributions for every candidate input and target logKd."""
    rows: list[dict[str, object]] = []
    global_df = model_dfs["Global"]
    for model, df in model_dfs.items():
        _features, numeric_features, categorical_features = feature_sets[model]
        numeric_roles = [(TARGET, "logKd", "target"), *[(feature, feature, "candidate_input") for feature in numeric_features]]
        seen_source_features: set[str] = set()
        for source_feature, display_feature, role in numeric_roles:
            if source_feature in seen_source_features or source_feature not in df.columns:
                continue
            seen_source_features.add(source_feature)
            reference = global_df[source_feature] if source_feature in global_df.columns else df[source_feature]
            rows.extend(
                numeric_distribution_rows(
                    model=model,
                    feature=display_feature,
                    role=role,
                    values=df[source_feature],
                    reference_values=reference,
                    total_rows=len(df),
                    max_exact_levels=max_exact_levels,
                    max_histogram_bins=max_histogram_bins,
                )
            )
        categorical_audit = availability.loc[
            availability["model"].eq(model) & availability["kind"].eq("categorical")
        ].sort_values("availability_plot_order")
        categorical_candidates = set(categorical_features)
        for feature_info in categorical_audit.itertuples(index=False):
            feature = feature_info.feature
            if feature not in df.columns:
                continue
            values = df[feature].map(clean_text)
            reported = values[present(df[feature])]
            counts = reported.value_counts()
            role = "candidate_input" if feature in categorical_candidates else "categorical_candidate"
            if counts.empty:
                rows.append(
                    {
                        "model": model,
                        "feature": feature,
                        "role": role,
                        "distribution_type": "categorical",
                        "feature_plot_order": int(feature_info.availability_plot_order),
                        "plot_order": 1,
                        "bin_left": np.nan,
                        "bin_right": np.nan,
                        "bin_center": np.nan,
                        "bin_label": "No reported values",
                        "record_count": 0,
                        "record_fraction": 0.0,
                        "reported_rows": 0,
                        "missing_fraction": 1.0,
                    }
                )
                continue
            displayed = counts.head(max_levels)
            for order, (level, count) in enumerate(displayed.items(), start=1):
                rows.append(
                    {
                        "model": model,
                        "feature": feature,
                        "role": role,
                        "distribution_type": "categorical",
                        "feature_plot_order": int(feature_info.availability_plot_order),
                        "plot_order": order,
                        "bin_left": np.nan,
                        "bin_right": np.nan,
                        "bin_center": np.nan,
                        "bin_label": level,
                        "record_count": int(count),
                        "record_fraction": count / len(df),
                        "reported_rows": len(reported),
                        "missing_fraction": 1 - len(reported) / len(df),
                    }
                )
            if len(counts) > max_levels:
                other_count = int(counts.iloc[max_levels:].sum())
                rows.append(
                    {
                        "model": model,
                        "feature": feature,
                        "role": role,
                        "distribution_type": "categorical",
                        "feature_plot_order": int(feature_info.availability_plot_order),
                        "plot_order": max_levels + 1,
                        "bin_left": np.nan,
                        "bin_right": np.nan,
                        "bin_center": np.nan,
                        "bin_label": "Other levels",
                        "record_count": other_count,
                        "record_fraction": other_count / len(df),
                        "reported_rows": len(reported),
                        "missing_fraction": 1 - len(reported) / len(df),
                    }
                )
    columns = [
        "model",
        "feature",
        "role",
        "distribution_type",
        "feature_plot_order",
        "plot_order",
        "bin_left",
        "bin_right",
        "bin_center",
        "bin_label",
        "record_count",
        "record_fraction",
        "reported_rows",
        "missing_fraction",
    ]
    return pd.DataFrame(rows, columns=columns)


def joint_coverage(model_dfs: dict[str, pd.DataFrame], feature_sets: dict[str, tuple[list[str], list[str], list[str]]]) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for model, df in model_dfs.items():
        features, numeric_features, categorical_features = feature_sets[model]
        coverage = joint_support_coverage(df, features, numeric_features, categorical_features)
        coverage = coverage.rename(columns={"selected_feature_count": "candidate_feature_count"})
        coverage.insert(0, "table_type", "joint_feature_coverage")
        coverage.insert(1, "model", model)
        rows.append(coverage)
        combinations = (df["PFAS_name"].map(clean_text).replace("", "missing") + " | " + identity_keys(df)).value_counts()
        support = pd.DataFrame({"rows_per_combination": combinations})
        bins = pd.cut(support["rows_per_combination"], bins=[0, 1, 5, 10, np.inf], labels=["1", "2-5", "6-10", ">10"])
        summary = bins.value_counts(sort=False).rename_axis("combination_row_count_bin").reset_index(name="combination_count")
        summary.insert(0, "table_type", "PFAS_adsorbent_combination_support")
        summary.insert(1, "model", model)
        rows.append(summary)
    return pd.concat(rows, ignore_index=True, sort=False)


def start_here(model_dfs: dict[str, pd.DataFrame], feature_sets: dict[str, tuple[list[str], list[str], list[str]]]) -> pd.DataFrame:
    rows = [
        {"section": "Read this first", "item": "Figure 1", "value": "Fig 1 Data: Origin-ready Sankey paths plus conversion tables."},
        {"section": "Read this first", "item": "Figure 2", "value": "Study Concentration: one row per study within each model cohort."},
        {"section": "Read this first", "item": "Figure 3", "value": "Feature Availability: filter one model, sort availability_plot_order, then plot feature (X) against nonmissing_rows (Y) as simple columns."},
        {"section": "Read this first", "item": "Figure 4", "value": "Feature Relationships: filter by model: AC, Resin, CDP, or Global."},
        {"section": "Read this first", "item": "Figure 5", "value": "Feature Distributions: filter model and feature, then plot bin_label (X) against record_count (Y). It includes every candidate input and target logKd; role distinguishes inputs from the target."},
        {"section": "Read this first", "item": "Figure 6", "value": "Evidence Coverage: every candidate input. It separates row/unit availability, overall value diversity, studies with multiple values, and the share of variation occurring within studies. Use it to judge whether pooled database variation also contains within-study comparisons."},
        {"section": "Cohort definition", "item": "Model rows", "value": "Finite logKd rows after model inclusion, optional flag handling, and PFAS core-quality gating."},
        {"section": "Cohort definition", "item": "Raw extracted records", "value": "Distinct source_row_index values within the cohort: extracted performance records before multi-valued isotherm designs were split into one row per C0/dosage condition. Model cohort records counts standardized (post-expansion) rows."},
    ]
    for model, df in model_dfs.items():
        status = column_or(df, "adsorbent_identity_status", pd.Series("generic_study_local", index=df.index))
        source_records = source_record_count(df)
        rows.extend([
            {"section": "Model cohort", "item": f"{model}: raw extracted records (before isotherm expansion)", "value": source_records},
            {"section": "Model cohort", "item": f"{model}: rows added by isotherm expansion", "value": len(df) - source_records},
            {"section": "Model cohort", "item": f"{model}: records", "value": len(df)},
            {"section": "Model cohort", "item": f"{model}: PFAS", "value": df["PFAS_name"].nunique()},
            {"section": "Model cohort", "item": f"{model}: studies", "value": df["study_no"].nunique()},
            {"section": "Model cohort", "item": f"{model}: named material identities", "value": identity_keys(df)[status.eq("specific")].nunique()},
            {"section": "Model cohort", "item": f"{model}: study-specific adsorbent instances", "value": study_instance_keys(df).nunique()},
            {"section": "Global feature audit", "item": f"{model}: candidate features assessed", "value": len(feature_sets[model][0])},
        ])
    return pd.DataFrame(rows)


def definitions(args: argparse.Namespace) -> pd.DataFrame:
    rows = [
        ("adsorbent_identity_status", "specific = canonical commercial name; generic_study_local = label valid only within its study; unknown = no usable identifier."),
        ("adsorbent_identity_key", "Split key: named materials can recur across studies; generic labels cannot."),
        ("joint_feature_coverage", "Records simultaneously reporting every candidate feature in the specified group; diagnostic only, not an XGBoost complete-case filter."),
        ("Sankey scope", "Equilibrium or suspected-equilibrium records only; clearly kinetic records are excluded."),
        ("Model cohort raw extracted records", "Start Here reports each cohort's distinct source_row_index count, i.e. extracted performance records before isotherm expansion, alongside the standardized post-expansion record count. A source record is counted in a cohort when at least one of its expanded rows survives model inclusion, flag handling, and PFAS core-quality gating."),
        ("Pre-expansion scope counts", "The first Scope counts rows count extracted performance records, one per source_row_index, before multi-valued isotherm designs are split into one row per C0/dosage condition. Rows added by isotherm expansion is the difference between standardized records and source records. Every later row counts standardized (post-expansion) records."),
        ("Figure 1 endpoint labels", "Isotherm combines Langmuir and Freundlich reporting, including non-convertible records with Langmuir Qm but no KL. Multiple means more than one reporting family (Kd, Removal, Qe, or Isotherm)."),
        ("Figure 1 adsorbent labels", "AC = activated carbon; Resin = ion-exchange or nonionic resin; CDP = cyclodextrin polymer; Clay = clay/mineral; Unknown = no category."),
        ("Feature Availability", "One row per model and feature, sorted by nonmissing_rows. Plot nonmissing_rows directly for the all-feature availability figure; total_model_rows and missing_rows are retained as context."),
        ("Feature audit scope", "This workbook is whole-database QA. It assesses availability without applying feature-support, diversity, PFAS-family, or correlation screening; training applies those policies within its current training partition."),
        ("Excluded protocol variables", "Batch/kinetic protocol variables (contact time, mixing speed, solution volume) are not candidate model inputs for an equilibrium logKd model and are therefore absent from every feature sheet here. The standardized database still records them."),
        ("Feature Distributions", "Origin-ready count data. Candidate inputs and logKd use exact-value bars or common Global-reference histogram bins; category levels beyond max_categorical_levels are combined as Other levels. feature_plot_order follows Figure 3 availability order within each model."),
        ("Evidence Coverage", "A descriptive diagnostic for every candidate input. It is not used in feature selection, allocation, hyperparameter tuning, or model scoring."),
        ("Evidence Coverage: availability", "row_availability is the fraction of model rows reporting the feature. unit_availability uses study for experimental features, study-specific adsorbent for material features, and unique PFAS for PFAS characteristics."),
        ("Evidence Coverage: variability", "overall_value_diversity is study-weighted normalized entropy of reported categorical levels or reference quantile bins. studies_with_multiple_values is the fraction of reporting studies with at least two observed values. share_of_variation_within_studies is mean within-study entropy divided by total entropy; between_study_variation_share is its complement. These diagnose contrast, not causal identification."),
    ]
    return pd.DataFrame(rows, columns=["item", "definition_or_value"])


def style_header_row(worksheet, row: int, start_column: int, column_count: int) -> None:
    for column in range(start_column, start_column + column_count):
        cell = worksheet.cell(row=row, column=column)
        font = copy(cell.font)
        font.bold = True
        font.color = "FFFFFF"
        fill = copy(cell.fill)
        fill.fgColor = "1F4E78"
        fill.fill_type = "solid"
        cell.font = font
        cell.fill = fill


def write_figure_1_sheet(writer: pd.ExcelWriter, name: str, figure_1: dict[str, pd.DataFrame]) -> None:
    """Write the Origin table first, with only Figure 1 companion tables beside it."""
    sankey = figure_1["Origin Sankey input"]
    scope = figure_1["Scope counts"]
    outcomes = figure_1["Conversion outcomes"]
    blockers = figure_1["Conversion blockers"]
    sheet_name = name[:31]

    sankey.to_excel(writer, sheet_name=sheet_name, startrow=1, startcol=0, index=False)
    scope.to_excel(writer, sheet_name=sheet_name, startrow=1, startcol=5, index=False)
    outcome_startrow = 1 + len(scope) + 4
    outcomes.to_excel(writer, sheet_name=sheet_name, startrow=outcome_startrow, startcol=5, index=False)
    blockers.to_excel(writer, sheet_name=sheet_name, startrow=1, startcol=9, index=False)

    worksheet = writer.sheets[sheet_name]
    worksheet["A1"] = "Origin Sankey input (select columns A:D)"
    worksheet["F1"] = "Scope counts"
    worksheet.cell(row=outcome_startrow, column=6, value="Conversion outcomes (for a stacked bar chart)")
    worksheet["J1"] = "Conversion blockers (non-exclusive; use for a ranked bar chart)"
    worksheet.freeze_panes = "A2"
    worksheet.auto_filter.ref = f"A2:D{len(sankey) + 2}"
    worksheet.sheet_view.showGridLines = False

    style_header_row(worksheet, row=2, start_column=1, column_count=len(sankey.columns))
    style_header_row(worksheet, row=2, start_column=6, column_count=len(scope.columns))
    style_header_row(
        worksheet,
        row=outcome_startrow + 1,
        start_column=6,
        column_count=len(outcomes.columns),
    )
    style_header_row(worksheet, row=2, start_column=10, column_count=len(blockers.columns))
    for cell in (worksheet["A1"], worksheet["F1"], worksheet.cell(row=outcome_startrow, column=6), worksheet["J1"]):
        font = copy(cell.font)
        font.bold = True
        cell.font = font
    for column, width in {
        "A": 28,
        "B": 28,
        "C": 28,
        "D": 15,
        "F": 34,
        "G": 28,
        "H": 15,
        "J": 46,
        "K": 26,
    }.items():
        worksheet.column_dimensions[column].width = width


def write_workbook(
    path: Path,
    tables: dict[str, pd.DataFrame | dict[str, pd.DataFrame]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        for name, table in tables.items():
            if isinstance(table, dict):
                write_figure_1_sheet(writer, name, table)
                continue
            table.to_excel(writer, sheet_name=name[:31], index=False)
        for worksheet in writer.book.worksheets:
            if worksheet.title == "Fig 1 Data":
                continue
            worksheet.freeze_panes = "A2"
            worksheet.auto_filter.ref = worksheet.dimensions
            style_header_row(worksheet, row=1, start_column=1, column_count=worksheet.max_column)
            for column_cells in worksheet.iter_cols():
                width = max(len(str(cell.value or "")) for cell in column_cells[:200])
                worksheet.column_dimensions[column_cells[0].column_letter].width = min(max(width + 2, 12), 42)


def main() -> None:
    args = parse_args()
    review_path = args.review_decisions_path or args.input_path.with_name("record_review_decisions.csv")
    review_source = pd.read_excel(args.input_path, sheet_name=args.sheet_name)
    review_candidates = build_human_review_candidates(review_source, args)
    candidate_count, appended_count = append_review_candidates(review_path, review_candidates)
    model_dfs, availability, feature_sets = loaded_model_tables(args)
    tables = {
        "Start Here": start_here(model_dfs, feature_sets),
        "Fig 1 Data": figure_1_tables(args, review_source),
        "Study Concentration": study_concentration(model_dfs),
        "Feature Availability": availability,
        "Feature Relationships": correlation_pairs(model_dfs, feature_sets, args),
        "Feature Distributions": feature_distributions(
            model_dfs,
            availability,
            feature_sets,
            args.max_categorical_levels,
            args.max_numeric_exact_levels,
            args.max_histogram_bins,
        ),
        "Evidence Coverage": evidence_coverage(model_dfs, availability, feature_sets),
        "Joint Feature Coverage": joint_coverage(model_dfs, feature_sets),
        "Definitions": definitions(args),
    }
    write_workbook(args.output_path, tables)
    print(f"Wrote figure-oriented audit: {args.output_path}")
    print(
        "Human-review queue: "
        f"{candidate_count} candidate source records; {appended_count} appended to {review_path}"
    )


if __name__ == "__main__":
    main()
