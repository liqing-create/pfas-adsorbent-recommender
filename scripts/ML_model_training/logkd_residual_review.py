"""Create a study/source residual review queue.

The public CSV outputs retain two deliberately distinct kinds of human-review
candidate:

* ``review_systematic_bias``: a high-specificity random-row pattern that can
  justify checking the original study for a possible extraction or conversion
  error; and
* ``review_transfer_pattern``: a coherent grouped-holdout pattern that merits
  contextual source review, while remaining evidence of transfer failure rather
  than evidence that a source record is wrong.

Mixed-direction errors and insufficient evidence are counted in the manifest
but suppressed from the queue. The audit does *not* write to
``record_review_decisions.csv`` and never assigns a reliability decision to a
record. A queued pattern is only a prompt to inspect the original source and
context. If that review finds no data error, the data stay unchanged.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


DEFAULT_MIN_SOURCE_RECORDS = 3
DEFAULT_MIN_HELDOUT_SOURCE_ASSIGNMENTS = 6
DEFAULT_MIN_EXCESS_MEDIAN_ABSOLUTE_ERROR = 0.75
DEFAULT_MIN_SAME_SIGN_SOURCE_FRACTION = 0.75
DEFAULT_MIN_MEAN_SOURCE_MSE_RATIO = 2.0
DEFAULT_MIN_BASELINE_SOURCE_RECORDS = 3

HUMAN_REVIEW_AUDIT_CLASSES = frozenset({
    "review_systematic_bias",
    "review_transfer_pattern",
})

PREFERRED_INNER_ALLOCATION_METHODS = (
    "standard_random",
    "random_row",
    "random_group",
)

GROUP_KEYS = (
    "split_strategy",
    "outer_allocation_method",
    "study_no",
    "Kd_final_source",
    "reporting_subsource",
)

GROUP_AUDIT_COLUMNS = (
    "review_key",
    "audit_scope",
    "split_strategy",
    "outer_allocation_method",
    "study_no",
    "Kd_final_source",
    "apparent_removal_source",
    "reporting_subsource",
    "source_record_count",
    "heldout_source_assignment_count",
    "distinct_outer_assignment_count",
    "median_source_abs_error",
    "median_source_signed_residual",
    "positive_source_fraction",
    "negative_source_fraction",
    "same_sign_source_fraction",
    "mean_source_mse",
    "largest_source_mse_fraction",
    "baseline_source_record_count",
    "baseline_median_source_abs_error",
    "excess_median_abs_error",
    "baseline_mean_source_mse",
    "mean_source_mse_ratio_to_baseline",
    "audit_class",
    "manual_review_recommendation",
    "review_reason",
)

MEMBER_AUDIT_COLUMNS = (
    "review_key",
    "audit_class",
    "split_strategy",
    "outer_allocation_method",
    "study_no",
    "Kd_final_source",
    "apparent_removal_source",
    "reporting_subsource",
    "source_row_index",
    "standardized_row_id",
    "extraction_dataset",
    "PFAS_name",
    "adsorbent_id",
    "C0",
    "dosage",
    "pH",
    "temp",
    "heldout_outer_assignment_count",
    "median_y_true",
    "median_y_pred",
    "median_signed_residual",
    "median_abs_error",
    "mean_source_mse",
    "group_source_record_count",
    "group_excess_median_abs_error",
    "group_same_sign_source_fraction",
)


def source_row_key(value: object) -> str:
    """Return a stable text representation for a source-row identifier."""

    if pd.isna(value):
        return ""
    if isinstance(value, (float, np.floating)) and float(value).is_integer():
        return str(int(value))
    return str(value).strip()


def _text_column(
    frame: pd.DataFrame,
    column: str,
    *,
    default: str = "(not recorded)",
) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(default, index=frame.index, dtype="string")
    values = frame[column].astype("string").fillna("").str.strip()
    return values.mask(values.eq(""), default)


def _first_text(group: pd.DataFrame, column: str) -> str:
    if column not in group.columns:
        return ""
    values = group[column].astype("string").fillna("").str.strip()
    values = values.loc[values.ne("")]
    return str(values.iloc[0]) if not values.empty else ""


def _weighted_mean(values: pd.Series, weights: pd.Series) -> float:
    values = pd.to_numeric(values, errors="coerce")
    weights = pd.to_numeric(weights, errors="coerce")
    valid = np.isfinite(values) & np.isfinite(weights) & weights.gt(0)
    if not valid.any():
        return float("nan")
    return float(np.average(values.loc[valid], weights=weights.loc[valid]))


def _outer_assignment_keys(rows: pd.DataFrame) -> pd.Series:
    """Return scenario-qualified outer-assignment identifiers."""

    split_strategy = _text_column(rows, "split_strategy")
    outer_method = _text_column(rows, "outer_allocation_method")
    identifier = _text_column(rows, "outer_assignment_id", default="")
    blank_identifier = identifier.eq("")
    if blank_identifier.any():
        fallback_columns = ("outer_repeat_id", "random_seed")
        missing = [column for column in fallback_columns if column not in rows.columns]
        if missing:
            raise ValueError(
                "Residual group audit needs outer_assignment_id, or these fallback "
                f"columns: {', '.join(fallback_columns)}. Missing: {', '.join(missing)}."
            )
        fallback = rows.loc[:, list(fallback_columns)].fillna("").astype(str).agg("|".join, axis=1)
        identifier = identifier.mask(blank_identifier, fallback)
    return split_strategy + "|" + outer_method + "|" + identifier


def _select_model_predictions(predictions: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Keep held-out rows from one conventional inner method per outer assignment."""

    required = {"source_row_index", "absolute_error", "squared_error"}
    missing = required.difference(predictions.columns)
    if missing:
        raise ValueError(
            "Residual group audit predictions are missing required columns: "
            + ", ".join(sorted(missing))
        )

    rows = predictions.copy()
    input_rows = len(rows)
    if "data_split" in rows.columns:
        rows = rows.loc[rows["data_split"].eq("testing")].copy()
    testing_rows = len(rows)
    if rows.empty:
        return rows, {
            "input_prediction_rows": input_rows,
            "testing_prediction_rows": testing_rows,
            "selected_prediction_rows": 0,
            "inner_prediction_policy": "no testing rows",
            "outer_assignments_seen": 0,
            "outer_assignments_selected": 0,
            "outer_assignments_excluded_without_preferred_inner_method": 0,
        }

    rows["__outer_assignment"] = _outer_assignment_keys(rows)
    selection: dict[str, Any] = {
        "input_prediction_rows": input_rows,
        "testing_prediction_rows": testing_rows,
        "outer_assignments_seen": int(rows["__outer_assignment"].nunique()),
    }
    if "inner_allocation_method" not in rows.columns:
        selection.update({
            "selected_prediction_rows": len(rows),
            "inner_prediction_policy": "inner allocation method not recorded; all held-out rows retained",
            "outer_assignments_selected": int(rows["__outer_assignment"].nunique()),
            "outer_assignments_excluded_without_preferred_inner_method": 0,
        })
        return rows, selection

    rows["__inner_allocation_method"] = _text_column(
        rows, "inner_allocation_method", default=""
    ).str.casefold()
    selected_parts: list[pd.DataFrame] = []
    method_counts: dict[str, int] = {}
    excluded = 0
    for _, assignment_rows in rows.groupby("__outer_assignment", sort=False):
        chosen_method = next(
            (
                method
                for method in PREFERRED_INNER_ALLOCATION_METHODS
                if assignment_rows["__inner_allocation_method"].eq(method).any()
            ),
            None,
        )
        if chosen_method is None:
            excluded += 1
            continue
        selected_parts.append(
            assignment_rows.loc[
                assignment_rows["__inner_allocation_method"].eq(chosen_method)
            ].copy()
        )
        method_counts[chosen_method] = method_counts.get(chosen_method, 0) + 1
    selected = (
        pd.concat(selected_parts, ignore_index=True)
        if selected_parts
        else rows.iloc[0:0].copy()
    )
    selection.update({
        "selected_prediction_rows": len(selected),
        "inner_prediction_policy": (
            "one conventional inner-allocation method per frozen outer assignment; "
            "preference order=" + ", ".join(PREFERRED_INNER_ALLOCATION_METHODS)
        ),
        "selected_inner_methods_by_outer_assignment_count": method_counts,
        "outer_assignments_selected": int(selected["__outer_assignment"].nunique()),
        "outer_assignments_excluded_without_preferred_inner_method": excluded,
    })
    return selected, selection


def _prepared_prediction_rows(predictions: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    rows, selection = _select_model_predictions(predictions)
    if rows.empty:
        return rows, selection

    rows["source_row_index"] = rows["source_row_index"].map(source_row_key)
    rows["__absolute_error"] = pd.to_numeric(rows["absolute_error"], errors="coerce")
    rows["__squared_error"] = pd.to_numeric(rows["squared_error"], errors="coerce")
    if "residual" in rows.columns:
        rows["__residual"] = pd.to_numeric(rows["residual"], errors="coerce")
    else:
        rows["__residual"] = (
            pd.to_numeric(rows.get("y_pred"), errors="coerce")
            - pd.to_numeric(rows.get("y_true"), errors="coerce")
        )
    rows["__y_true"] = pd.to_numeric(rows.get("y_true"), errors="coerce")
    rows["__y_pred"] = pd.to_numeric(rows.get("y_pred"), errors="coerce")
    if "source_row_weight" in rows.columns:
        rows["__source_row_weight"] = pd.to_numeric(
            rows["source_row_weight"], errors="coerce"
        )
    else:
        counts = rows.groupby(
            ["__outer_assignment", "source_row_index"], dropna=False
        )["source_row_index"].transform("size")
        rows["__source_row_weight"] = 1.0 / counts

    rows["split_strategy"] = _text_column(rows, "split_strategy")
    rows["outer_allocation_method"] = _text_column(rows, "outer_allocation_method")
    rows["study_no"] = _text_column(rows, "study_no")
    rows["Kd_final_source"] = _text_column(rows, "Kd_final_source")
    rows["apparent_removal_source"] = _text_column(
        rows, "apparent_removal_source", default=""
    )
    for canonical, alternatives in {
        "C0": ("PFAS_C0_value_mg/L",),
        "dosage": ("Adsorbent_dosage_value_mg/L",),
        "temp": ("Temperature_(°C)",),
    }.items():
        if canonical not in rows.columns:
            source_column = next(
                (column for column in alternatives if column in rows.columns), None
            )
            rows[canonical] = rows[source_column] if source_column else pd.NA
    removal_source = rows["Kd_final_source"].str.contains("removal", case=False, na=False)
    rows["reporting_subsource"] = np.where(
        removal_source & rows["apparent_removal_source"].ne(""),
        rows["apparent_removal_source"],
        rows["Kd_final_source"],
    )
    valid = (
        rows["source_row_index"].ne("")
        & rows["__outer_assignment"].ne("")
        & np.isfinite(rows["__absolute_error"])
        & np.isfinite(rows["__squared_error"])
        & np.isfinite(rows["__residual"])
        & np.isfinite(rows["__source_row_weight"])
        & rows["__source_row_weight"].gt(0)
    )
    prepared = rows.loc[valid].copy()
    selection["valid_prediction_rows"] = len(prepared)
    selection["dropped_invalid_prediction_rows"] = len(rows) - len(prepared)
    return prepared, selection


def _source_assignment_summary(rows: pd.DataFrame) -> pd.DataFrame:
    source_columns = [
        "split_strategy",
        "outer_allocation_method",
        "study_no",
        "Kd_final_source",
        "apparent_removal_source",
        "reporting_subsource",
        "__outer_assignment",
        "source_row_index",
    ]
    context_columns = (
        "standardized_row_id", "extraction_dataset", "PFAS_name", "adsorbent_id",
        "C0", "dosage", "pH", "temp",
    )
    work = rows.copy()
    for column in context_columns:
        if column not in work.columns:
            work[column] = pd.NA
    for metric in ("absolute_error", "residual", "squared_error", "y_true", "y_pred"):
        work[f"__weighted_{metric}"] = work[f"__{metric}"] * work["__source_row_weight"]
    aggregation = {
        "__source_row_weight": "sum",
        "__weighted_absolute_error": "sum",
        "__weighted_residual": "sum",
        "__weighted_squared_error": "sum",
        "__weighted_y_true": "sum",
        "__weighted_y_pred": "sum",
        **{column: "first" for column in context_columns},
    }
    result = work.groupby(source_columns, dropna=False, as_index=False).agg(aggregation)
    denominator = result["__source_row_weight"].where(
        result["__source_row_weight"].gt(0), np.nan
    )
    result["source_assignment_abs_error"] = result["__weighted_absolute_error"] / denominator
    result["source_assignment_residual"] = result["__weighted_residual"] / denominator
    result["source_assignment_mse"] = result["__weighted_squared_error"] / denominator
    result["source_assignment_y_true"] = result["__weighted_y_true"] / denominator
    result["source_assignment_y_pred"] = result["__weighted_y_pred"] / denominator
    return result.drop(columns=[
        "__source_row_weight", "__weighted_absolute_error", "__weighted_residual",
        "__weighted_squared_error", "__weighted_y_true", "__weighted_y_pred",
    ])


def _source_summary(source_assignments: pd.DataFrame) -> pd.DataFrame:
    source_columns = [*GROUP_KEYS, "apparent_removal_source", "source_row_index"]
    context_columns = (
        "standardized_row_id", "extraction_dataset", "PFAS_name", "adsorbent_id",
        "C0", "dosage", "pH", "temp",
    )
    aggregation = {
        "__outer_assignment": "nunique",
        "source_assignment_y_true": "median",
        "source_assignment_y_pred": "median",
        "source_assignment_residual": "median",
        "source_assignment_abs_error": "median",
        "source_assignment_mse": "mean",
        **{column: "first" for column in context_columns},
    }
    result = source_assignments.groupby(
        source_columns, dropna=False, as_index=False
    ).agg(aggregation)
    return result.rename(columns={
        "__outer_assignment": "heldout_outer_assignment_count",
        "source_assignment_y_true": "median_y_true",
        "source_assignment_y_pred": "median_y_pred",
        "source_assignment_residual": "median_signed_residual",
        "source_assignment_abs_error": "median_abs_error",
        "source_assignment_mse": "mean_source_mse",
    })


def _review_key(row: pd.Series) -> str:
    return "|".join(str(row[column]) for column in GROUP_KEYS)


def _group_reason(
    *, audit_class: str, source_count: int, assignment_count: int,
    excess_error: float, same_sign_fraction: float, mse_ratio: float,
    largest_mse_fraction: float,
) -> str:
    evidence = (
        f"{source_count} source records across {assignment_count} held-out source/outer assignments; "
        f"median absolute error is {excess_error:.3g} above the leave-one-study-out endpoint baseline; "
        f"{same_sign_fraction:.0%} of source records have the same median residual direction; "
        f"mean source-level MSE is {mse_ratio:.2g}x baseline."
    )
    concentration = (
        f" The largest source record contributes {largest_mse_fraction:.0%} of the group's source-level MSE."
        if np.isfinite(largest_mse_fraction) else ""
    )
    if audit_class == "review_systematic_bias":
        return "Coherent study/source residual pattern; inspect reporting units and conversion inputs. " + evidence + concentration
    if audit_class == "review_transfer_pattern":
        return (
            "Coherent grouped-holdout transfer pattern; inspect the original "
            "study context, reporting inputs, and unrecorded conditions, but do "
            "not treat the residual alone as data-quality evidence. "
            + evidence + concentration
        )
    if audit_class == "high_error_mixed_direction":
        return "High-error group without a coherent residual direction; this is not a specific unit/conversion signal. " + evidence + concentration
    if audit_class == "extrapolation_diagnostic_only":
        return "Grouped holdout result retained as an extrapolation diagnostic only; do not treat it as data-quality evidence. " + evidence + concentration
    return "Insufficient repeated, comparative evidence for a study/source review recommendation. " + evidence + concentration


def _build_group_table(
    sources: pd.DataFrame, *, min_source_records: int,
    min_heldout_source_assignments: int, min_excess_median_absolute_error: float,
    min_same_sign_source_fraction: float, min_mean_source_mse_ratio: float,
    min_baseline_source_records: int, outer_assignment_counts: dict[tuple[Any, ...], int],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    baseline_keys = ("split_strategy", "outer_allocation_method", "Kd_final_source", "reporting_subsource")
    for _, group in sources.groupby(list(GROUP_KEYS), dropna=False, sort=False):
        first = group.iloc[0]
        same_endpoint = sources.loc[
            np.logical_and.reduce([sources[column].eq(first[column]) for column in baseline_keys])
        ]
        baseline = same_endpoint.loc[same_endpoint["study_no"].ne(first["study_no"])]
        baseline_source_count = int(len(baseline))
        baseline_median_error = float(baseline["median_abs_error"].median()) if not baseline.empty else float("nan")
        baseline_mean_mse = float(baseline["mean_source_mse"].mean()) if not baseline.empty else float("nan")
        median_error = float(group["median_abs_error"].median())
        median_residual = float(group["median_signed_residual"].median())
        mean_mse = float(group["mean_source_mse"].mean())
        excess_error = median_error - baseline_median_error
        mse_ratio = mean_mse / baseline_mean_mse if baseline_mean_mse > 0 else float("nan")
        source_count = int(len(group))
        assignment_count = int(group["heldout_outer_assignment_count"].sum())
        positive_fraction = float(group["median_signed_residual"].gt(0).mean())
        negative_fraction = float(group["median_signed_residual"].lt(0).mean())
        same_sign_fraction = max(positive_fraction, negative_fraction)
        total_mse = float(group["mean_source_mse"].sum())
        largest_mse_fraction = float(group["mean_source_mse"].max() / total_mse) if total_mse > 0 else float("nan")
        random_row_scope = str(first["split_strategy"]).casefold() == "random_row"
        enough_evidence = (
            source_count >= min_source_records
            and assignment_count >= min_heldout_source_assignments
            and baseline_source_count >= min_baseline_source_records
            and np.isfinite(excess_error) and np.isfinite(mse_ratio)
        )
        systematic_bias = (
            enough_evidence and excess_error >= min_excess_median_absolute_error
            and same_sign_fraction >= min_same_sign_source_fraction
            and mse_ratio >= min_mean_source_mse_ratio
        )
        high_error_mixed_direction = (
            enough_evidence and excess_error >= min_excess_median_absolute_error
            and mse_ratio >= min_mean_source_mse_ratio
        )
        if systematic_bias and random_row_scope:
            audit_class = "review_systematic_bias"
            recommendation = "Review the original study/source and its units or conversion inputs; change a record only when the source review identifies an actual error."
        elif systematic_bias:
            audit_class = "review_transfer_pattern"
            recommendation = (
                "Review the original study/source for reporting inputs and "
                "unrecorded study or adsorbent context. This is a transfer-failure "
                "pattern, not evidence to alter data; change a record only when "
                "the source review identifies an actual error."
            )
        elif not random_row_scope:
            audit_class = "extrapolation_diagnostic_only"
            recommendation = "Do not alter data from grouped-holdout residuals alone."
        elif high_error_mixed_direction:
            audit_class = "high_error_mixed_direction"
            recommendation = "No unit/conversion conclusion follows from this mixed-direction error pattern; leave data unchanged unless an independent source review finds an error."
        else:
            audit_class = "insufficient_evidence"
            recommendation = "No model-output-based data action is recommended."
        group_key = tuple(first[column] for column in GROUP_KEYS)
        distinct_outer_assignments = outer_assignment_counts[group_key]
        row = {
            "audit_scope": "study_reporting_source",
            "split_strategy": first["split_strategy"],
            "outer_allocation_method": first["outer_allocation_method"],
            "study_no": first["study_no"],
            "Kd_final_source": first["Kd_final_source"],
            "apparent_removal_source": _first_text(group, "apparent_removal_source"),
            "reporting_subsource": first["reporting_subsource"],
            "source_record_count": source_count,
            "heldout_source_assignment_count": assignment_count,
            "distinct_outer_assignment_count": distinct_outer_assignments,
            "median_source_abs_error": median_error,
            "median_source_signed_residual": median_residual,
            "positive_source_fraction": positive_fraction,
            "negative_source_fraction": negative_fraction,
            "same_sign_source_fraction": same_sign_fraction,
            "mean_source_mse": mean_mse,
            "largest_source_mse_fraction": largest_mse_fraction,
            "baseline_source_record_count": baseline_source_count,
            "baseline_median_source_abs_error": baseline_median_error,
            "excess_median_abs_error": excess_error,
            "baseline_mean_source_mse": baseline_mean_mse,
            "mean_source_mse_ratio_to_baseline": mse_ratio,
            "audit_class": audit_class,
            "manual_review_recommendation": recommendation,
        }
        row["review_key"] = _review_key(pd.Series(row))
        row["review_reason"] = _group_reason(
            audit_class=audit_class, source_count=source_count, assignment_count=assignment_count,
            excess_error=excess_error, same_sign_fraction=same_sign_fraction,
            mse_ratio=mse_ratio, largest_mse_fraction=largest_mse_fraction,
        )
        rows.append(row)
    result = pd.DataFrame(rows, columns=GROUP_AUDIT_COLUMNS)
    if result.empty:
        return result
    audit_order = {
        "review_systematic_bias": 0,
        "review_transfer_pattern": 1,
        "high_error_mixed_direction": 2,
        "insufficient_evidence": 3,
        "extrapolation_diagnostic_only": 4,
    }
    result["__audit_order"] = result["audit_class"].map(audit_order)
    return result.sort_values(
        ["__audit_order", "excess_median_abs_error", "mean_source_mse_ratio_to_baseline"],
        ascending=[True, False, False], na_position="last",
    ).drop(columns="__audit_order").reset_index(drop=True)


def _member_table(sources: pd.DataFrame, groups: pd.DataFrame) -> pd.DataFrame:
    if sources.empty or groups.empty:
        return pd.DataFrame(columns=MEMBER_AUDIT_COLUMNS)
    group_columns = [*GROUP_KEYS, "review_key", "audit_class", "source_record_count", "excess_median_abs_error", "same_sign_source_fraction"]
    merged = sources.merge(groups.loc[:, group_columns], on=list(GROUP_KEYS), how="inner")
    merged = merged.rename(columns={
        "source_record_count": "group_source_record_count",
        "excess_median_absolute_error": "group_excess_median_abs_error",
        "excess_median_abs_error": "group_excess_median_abs_error",
        "same_sign_source_fraction": "group_same_sign_source_fraction",
    })
    return merged.reindex(columns=MEMBER_AUDIT_COLUMNS).sort_values(
        ["audit_class", "review_key", "median_abs_error"], ascending=[True, True, False],
        na_position="last",
    ).reset_index(drop=True)


def build_residual_group_audit(
    predictions: pd.DataFrame, *,
    min_source_records: int = DEFAULT_MIN_SOURCE_RECORDS,
    min_heldout_source_assignments: int = DEFAULT_MIN_HELDOUT_SOURCE_ASSIGNMENTS,
    min_excess_median_absolute_error: float = DEFAULT_MIN_EXCESS_MEDIAN_ABSOLUTE_ERROR,
    min_same_sign_source_fraction: float = DEFAULT_MIN_SAME_SIGN_SOURCE_FRACTION,
    min_mean_source_mse_ratio: float = DEFAULT_MIN_MEAN_SOURCE_MSE_RATIO,
    min_baseline_source_records: int = DEFAULT_MIN_BASELINE_SOURCE_RECORDS,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Build a human-review residual queue without changing input data.

    All groups are screened internally so that endpoint baselines and manifest
    counts remain complete. Only high-specificity random-row review patterns
    and coherent grouped-holdout transfer-review patterns are returned for
    human review.
    """

    if min_source_records < 1 or min_heldout_source_assignments < 1 or min_baseline_source_records < 1:
        raise ValueError("Minimum source and assignment counts must be at least one.")
    if min_excess_median_absolute_error < 0 or min_mean_source_mse_ratio < 0:
        raise ValueError("Residual error thresholds cannot be negative.")
    if not 0 <= min_same_sign_source_fraction <= 1:
        raise ValueError("min_same_sign_source_fraction must be between zero and one.")
    prepared, selection = _prepared_prediction_rows(predictions)
    if prepared.empty:
        return pd.DataFrame(columns=GROUP_AUDIT_COLUMNS), pd.DataFrame(columns=MEMBER_AUDIT_COLUMNS), selection
    source_assignments = _source_assignment_summary(prepared)
    sources = _source_summary(source_assignments)
    outer_assignment_counts = {
        key: int(value)
        for key, value in source_assignments.groupby(
            list(GROUP_KEYS), dropna=False
        )["__outer_assignment"].nunique().items()
    }
    screened_groups = _build_group_table(
        sources, min_source_records=min_source_records,
        min_heldout_source_assignments=min_heldout_source_assignments,
        min_excess_median_absolute_error=min_excess_median_absolute_error,
        min_same_sign_source_fraction=min_same_sign_source_fraction,
        min_mean_source_mse_ratio=min_mean_source_mse_ratio,
        min_baseline_source_records=min_baseline_source_records,
        outer_assignment_counts=outer_assignment_counts,
    )
    screening_class_counts = {
        str(key): int(value)
        for key, value in screened_groups["audit_class"].value_counts(
            dropna=False
        ).items()
    }
    groups = screened_groups.loc[
        screened_groups["audit_class"].isin(HUMAN_REVIEW_AUDIT_CLASSES)
    ].copy().reset_index(drop=True)
    members = _member_table(sources, groups)
    selection.update({
        "source_outer_assignment_summaries": len(source_assignments),
        "source_record_summaries": len(sources),
        "study_reporting_source_groups_screened": len(screened_groups),
        "studies_screened": int(screened_groups["study_no"].nunique()),
        "screening_class_counts": screening_class_counts,
        "human_review_audit_classes": sorted(HUMAN_REVIEW_AUDIT_CLASSES),
        "human_review_study_reporting_source_groups": len(groups),
        "human_review_studies": int(groups["study_no"].nunique()),
        "non_actionable_groups_suppressed": len(screened_groups) - len(groups),
    })
    return groups, members, selection


def _read_predictions(prediction_path: Path) -> pd.DataFrame:
    return pd.read_parquet(prediction_path) if prediction_path.suffix.casefold() == ".parquet" else pd.read_csv(prediction_path)


def write_residual_group_audit(
    prediction_path: Path, audit_dir: Path, *,
    min_source_records: int = DEFAULT_MIN_SOURCE_RECORDS,
    min_heldout_source_assignments: int = DEFAULT_MIN_HELDOUT_SOURCE_ASSIGNMENTS,
    min_excess_median_absolute_error: float = DEFAULT_MIN_EXCESS_MEDIAN_ABSOLUTE_ERROR,
    min_same_sign_source_fraction: float = DEFAULT_MIN_SAME_SIGN_SOURCE_FRACTION,
    min_mean_source_mse_ratio: float = DEFAULT_MIN_MEAN_SOURCE_MSE_RATIO,
    min_baseline_source_records: int = DEFAULT_MIN_BASELINE_SOURCE_RECORDS,
) -> dict[str, Any]:
    """Write the actionable study/source review queue to an ``audit`` folder."""

    prediction_path = Path(prediction_path)
    audit_dir = Path(audit_dir)
    groups, members, selection = build_residual_group_audit(
        _read_predictions(prediction_path), min_source_records=min_source_records,
        min_heldout_source_assignments=min_heldout_source_assignments,
        min_excess_median_absolute_error=min_excess_median_absolute_error,
        min_same_sign_source_fraction=min_same_sign_source_fraction,
        min_mean_source_mse_ratio=min_mean_source_mse_ratio,
        min_baseline_source_records=min_baseline_source_records,
    )
    audit_dir.mkdir(parents=True, exist_ok=True)
    groups_path = audit_dir / "residual_group_audit.csv"
    members_path = audit_dir / "residual_group_members.csv"
    manifest_path = audit_dir / "residual_group_audit_manifest.json"
    groups.to_csv(groups_path, index=False)
    members.to_csv(members_path, index=False)
    class_counts = {str(key): int(value) for key, value in groups["audit_class"].value_counts(dropna=False).items()}
    manifest: dict[str, Any] = {
        "schema_version": 3,
        "purpose": "Model-residual queue for human source review. It contains high-specificity random-row review patterns and coherent grouped-holdout transfer-review patterns; it never writes to record_review_decisions.csv and never marks records reliable or unreliable.",
        "prediction_path": str(prediction_path),
        "selection": selection,
        "thresholds": {
            "min_source_records": min_source_records,
            "min_heldout_source_assignments": min_heldout_source_assignments,
            "min_excess_median_absolute_error": min_excess_median_absolute_error,
            "min_same_sign_source_fraction": min_same_sign_source_fraction,
            "min_mean_source_mse_ratio": min_mean_source_mse_ratio,
            "min_baseline_source_records": min_baseline_source_records,
        },
        "comparison": "Each study/reporting-source group is compared with source records from other studies that share its split scenario and reporting endpoint/subsource.",
        "eligibility": "A group enters the human-review queue only with sufficient repeated evidence, large error relative to the same-endpoint baseline, a coherent residual direction, and elevated MSE. random_row patterns are tagged review_systematic_bias; grouped-holdout patterns are tagged review_transfer_pattern.",
        "classification_guide": {"review_systematic_bias": "Random-row pattern suitable for checking reporting units or conversion inputs; it is still not a record-level reliability decision.", "review_transfer_pattern": "Grouped-holdout pattern suitable for contextual source review. It may reflect unrepresented study or adsorbent context and is not, by itself, evidence to alter data."},
        "output_policy": "The CSV outputs contain both human-review classes and their source-record members. Mixed-direction, extrapolation-only, and insufficient-evidence groups are suppressed and summarized only by count in this manifest.",
        "outputs": {"group_audit": groups_path.name, "group_members": members_path.name, "manifest": manifest_path.name},
        "counts": {
            "study_reporting_source_groups_screened": selection.get(
                "study_reporting_source_groups_screened", 0
            ),
            "studies_screened": selection.get("studies_screened", 0),
            "human_review_study_reporting_source_groups": len(groups),
            "human_review_studies": int(groups["study_no"].nunique()),
            "source_record_members": len(members),
            "non_actionable_groups_suppressed": selection.get(
                "non_actionable_groups_suppressed", 0
            ),
            "screening_class_counts": selection.get(
                "screening_class_counts", {}
            ),
            "audit_class_counts": class_counts,
        },
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Write a study/source-level residual diagnostic to a model batch audit folder.")
    parser.add_argument("prediction_path", type=Path)
    parser.add_argument("audit_dir", type=Path)
    parser.add_argument("--min-source-records", type=int, default=DEFAULT_MIN_SOURCE_RECORDS)
    parser.add_argument("--min-heldout-source-assignments", type=int, default=DEFAULT_MIN_HELDOUT_SOURCE_ASSIGNMENTS)
    parser.add_argument("--min-excess-median-absolute-error", type=float, default=DEFAULT_MIN_EXCESS_MEDIAN_ABSOLUTE_ERROR)
    parser.add_argument("--min-same-sign-source-fraction", type=float, default=DEFAULT_MIN_SAME_SIGN_SOURCE_FRACTION)
    parser.add_argument("--min-mean-source-mse-ratio", type=float, default=DEFAULT_MIN_MEAN_SOURCE_MSE_RATIO)
    parser.add_argument("--min-baseline-source-records", type=int, default=DEFAULT_MIN_BASELINE_SOURCE_RECORDS)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    manifest = write_residual_group_audit(
        args.prediction_path, args.audit_dir,
        min_source_records=args.min_source_records,
        min_heldout_source_assignments=args.min_heldout_source_assignments,
        min_excess_median_absolute_error=args.min_excess_median_absolute_error,
        min_same_sign_source_fraction=args.min_same_sign_source_fraction,
        min_mean_source_mse_ratio=args.min_mean_source_mse_ratio,
        min_baseline_source_records=args.min_baseline_source_records,
    )
    print(
            "Residual group review queue: "
        f"{manifest['counts']['human_review_study_reporting_source_groups']} human-review "
        f"group(s) from {manifest['counts']['human_review_studies']} study/studies; "
        f"suppressed {manifest['counts']['non_actionable_groups_suppressed']} "
        f"non-actionable group(s); wrote {manifest['outputs']['group_audit']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
