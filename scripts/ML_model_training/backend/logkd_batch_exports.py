"""Create consistent report- and audit-ready exports for logKd batch wrappers.

The model-run directories remain the complete, downstream-compatible records for
the recommender and SHAP workflows.  This module only builds convenient,
batch-level views under ``report/`` and ``audit/`` after a wrapper completes.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from .logkd_coverage import (
    REFERENCE_RETENTION_BLOCK_SLUGS,
    REFERENCE_RETENTION_SUMMARY_METRICS,
)

DEFAULT_REFERENCE_RETENTION_METRICS = REFERENCE_RETENTION_SUMMARY_METRICS
_REFERENCE_RETENTION_METRIC_DEFINITIONS = tuple(
    {
        "metric": f"{metric}_{slug}",
        "feature_block": block,
        "scope": scope,
        "lower_is_better": False,
        "definition": definition,
    }
    for metric, scope, definition in (
        (
            "database_contrast",
            "Complete eligible pre-split modeling cohort",
            "Fraction of eligible reference feature-study opportunities with at least two reference-defined values.",
        ),
        (
            "training_contrast",
            "Fitted training partition on the complete-reference scale",
            "Fraction of the same eligible reference feature-study opportunities that retain at least two reference-defined values in training.",
        ),
        (
            "contrast_retention",
            "Fitted training partition relative to complete eligible pre-split cohort",
            "Fraction of reference within-study contrasts retained in training; conditional on contrasts present in the reference cohort.",
        ),
    )
    for block, slug in REFERENCE_RETENTION_BLOCK_SLUGS.items()
)
DEFAULT_AUDIT_ARTIFACTS = (
    "run_config.json",
    "metrics_summary.csv",
    "all_predictions.csv",
    "split_assignments.csv",
    "feature_manifest.csv",
    "selected_feature_support.csv",
    "input_block_support.csv",
    "validation_fold_metrics.csv",
)
PREDICTION_COLUMNS = (
    "data_split",
    "standardized_row_id",
    "source_row_index",
    "dataset",
    "study_no",
    "PFAS_name",
    "adsorbent_id",
    "adsorbent_category",
    "adsorbent_subcategory",
    "y_true",
    "y_pred",
    "residual",
    "absolute_error",
    "squared_error",
)


def _completed_runs(runs: pd.DataFrame, status_column: str, completed_value: str) -> pd.DataFrame:
    if runs.empty or status_column not in runs:
        return pd.DataFrame(columns=runs.columns)
    return runs.loc[runs[status_column].eq(completed_value)].copy()


def _attach_run_metadata(
    frame: pd.DataFrame,
    run: pd.Series,
    metadata_columns: tuple[str, ...],
) -> pd.DataFrame:
    output = frame.copy()
    for column in reversed(metadata_columns):
        destination = column if column not in output.columns else f"run_{column}"
        output.insert(0, destination, run.get(column, pd.NA))
    return output


def _resolve_run_directory(batch_dir: Path, raw_directory: object) -> Path:
    """Return a usable run directory, including after a batch folder is moved.

    Batch CSV files deliberately retain the originally recorded absolute path for
    provenance.  When a complete batch is copied to a different Windows account
    or machine, recover the same directory by replacing the path through the
    batch-folder name with the current ``batch_dir``.
    """

    candidate = Path(str(raw_directory))
    if candidate.exists():
        return candidate
    if not candidate.is_absolute():
        within_batch = batch_dir / candidate
        if within_batch.exists():
            return within_batch
    batch_name = batch_dir.name.casefold()
    for index, part in enumerate(candidate.parts):
        if part.casefold() == batch_name:
            relocated = batch_dir.joinpath(*candidate.parts[index + 1:])
            if relocated.exists():
                return relocated
    return candidate


def _collect_run_csv(
    runs: pd.DataFrame,
    filename: str,
    *,
    batch_dir: Path,
    status_column: str,
    completed_value: str,
    run_directory_column: str,
    metadata_columns: tuple[str, ...],
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for _, run in _completed_runs(runs, status_column, completed_value).iterrows():
        raw_directory = run.get(run_directory_column)
        if pd.isna(raw_directory):
            continue
        path = _resolve_run_directory(batch_dir, raw_directory) / filename
        if path.exists() and path.stat().st_size:
            frames.append(_attach_run_metadata(pd.read_csv(path), run, metadata_columns))
    return pd.concat(frames, ignore_index=True, sort=False) if frames else pd.DataFrame()


def _reference_retention_feature_table(
    feature_retention: pd.DataFrame,
    metadata_columns: tuple[str, ...],
) -> pd.DataFrame:
    """Return the compact per-feature reference-retention diagnostic table."""

    columns = [
        *metadata_columns,
        "coverage_scope",
        "reference_scope",
        "feature_scope",
        "feature",
        "bucket",
        "kind",
        "availability_unit",
        "reference_total_units",
        "training_retained_units",
        "entity_retention",
        "reference_value_level_or_bin_count",
        "training_observed_reference_value_level_or_bin_count",
        "reference_value_definition",
        "reference_value_support_mass_retention",
        "reference_total_studies",
        "reference_contrasting_studies",
        "training_retained_contrasting_studies",
        "database_contrast",
        "training_contrast",
        "contrast_retention",
    ]
    return feature_retention.reindex(columns=columns).copy()


def _metric_plot_data(
    runs: pd.DataFrame,
    metrics: tuple[tuple[str, str, bool], ...],
    *,
    status_column: str,
    completed_value: str,
    metadata_columns: tuple[str, ...],
) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for _, run in _completed_runs(runs, status_column, completed_value).iterrows():
        for source_column, metric, lower_is_better in metrics:
            value = pd.to_numeric(pd.Series([run.get(source_column)]), errors="coerce").iloc[0]
            if pd.isna(value):
                continue
            records.append({
                **{column: run.get(column, pd.NA) for column in metadata_columns},
                "metric": metric,
                "value": float(value),
                "lower_is_better": lower_is_better,
            })
    return pd.DataFrame(records, columns=[*metadata_columns, "metric", "value", "lower_is_better"])


def _representative_run_catalog(
    runs: pd.DataFrame,
    *,
    status_column: str,
    completed_value: str,
    selection_columns: tuple[str, ...],
    selection_metric: str | None,
) -> pd.DataFrame:
    completed = _completed_runs(runs, status_column, completed_value)
    available_group_columns = [column for column in selection_columns if column in completed]
    groups = completed.groupby(available_group_columns, sort=True) if available_group_columns else [(None, completed)]
    selected_rows: list[pd.Series] = []
    for _, group in groups:
        candidates = group.copy()
        coverage = (
            pd.to_numeric(candidates[selection_metric], errors="coerce")
            if selection_metric and selection_metric in candidates
            else pd.Series(float("nan"), index=candidates.index, dtype=float)
        )
        candidates["_selection_metric"] = coverage
        tie_columns = [
            column
            for column in ("pair_id", "outer_repeat_id", "seed", "random_seed")
            if column in candidates
        ]
        if coverage.notna().any():
            median = float(coverage.median())
            candidates["_selection_distance"] = (coverage - median).abs()
            selected = candidates.sort_values(["_selection_distance", *tie_columns], kind="stable").iloc[0].copy()
            selected["representative_selection_rule"] = (
                f"median {selection_metric} within the report comparison cell; "
                "ties resolved by deterministic run identifiers"
            )
        else:
            selected = candidates.sort_values(tie_columns, kind="stable").iloc[0].copy() if tie_columns else candidates.iloc[0].copy()
            selected["representative_selection_rule"] = (
                "first completed run by deterministic run identifiers because the selection metric was unavailable"
            )
        selected_rows.append(selected.drop(labels=["_selection_metric", "_selection_distance"], errors="ignore"))
    return pd.DataFrame(selected_rows).reset_index(drop=True) if selected_rows else pd.DataFrame()


def _prediction_source(batch_dir: Path, run: pd.Series, run_directory_column: str) -> pd.DataFrame:
    run_dir = _resolve_run_directory(batch_dir, run[run_directory_column])
    all_predictions = run_dir / "all_predictions.csv"
    if all_predictions.exists() and all_predictions.stat().st_size:
        return pd.read_csv(all_predictions)
    frames = [
        pd.read_csv(path)
        for filename in ("training_predictions.csv", "testing_predictions.csv")
        if (path := run_dir / filename).exists() and path.stat().st_size
    ]
    return pd.concat(frames, ignore_index=True, sort=False) if frames else pd.DataFrame()


def _representative_prediction_plot_data(
    catalog: pd.DataFrame,
    *,
    batch_dir: Path,
    run_directory_column: str,
    metadata_columns: tuple[str, ...],
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for _, run in catalog.iterrows():
        predictions = _prediction_source(batch_dir, run, run_directory_column)
        if predictions.empty:
            continue
        annotated = _attach_run_metadata(predictions.reindex(columns=PREDICTION_COLUMNS), run, metadata_columns)
        annotated.insert(0, "representative_selection_rule", run["representative_selection_rule"])
        frames.append(annotated)
    columns = ["representative_selection_rule", *metadata_columns, *PREDICTION_COLUMNS]
    output = pd.concat(frames, ignore_index=True, sort=False) if frames else pd.DataFrame()
    return output.reindex(columns=columns)


def _all_prediction_plot_data(
    runs: pd.DataFrame,
    *,
    batch_dir: Path,
    status_column: str,
    completed_value: str,
    run_directory_column: str,
    metadata_columns: tuple[str, ...],
) -> pd.DataFrame:
    """Collect predictions from every completed run for later scenario selection."""

    frames: list[pd.DataFrame] = []
    for _, run in _completed_runs(runs, status_column, completed_value).iterrows():
        predictions = _prediction_source(batch_dir, run, run_directory_column)
        if predictions.empty:
            continue
        frames.append(
            _attach_run_metadata(predictions.reindex(columns=PREDICTION_COLUMNS), run, metadata_columns)
        )
    columns = [*metadata_columns, *PREDICTION_COLUMNS]
    output = pd.concat(frames, ignore_index=True, sort=False) if frames else pd.DataFrame()
    return output.reindex(columns=columns)


def _write_audit_outputs(
    batch_dir: Path,
    runs: pd.DataFrame,
    *,
    status_column: str,
    completed_value: str,
    run_directory_column: str,
    metadata_columns: tuple[str, ...],
) -> dict[str, str]:
    audit_dir = batch_dir / "audit"
    audit_dir.mkdir(exist_ok=True)
    completed = _completed_runs(runs, status_column, completed_value)
    completed.to_csv(audit_dir / "completed_run_summary.csv", index=False)
    for filename, output_name in (
        ("split_assignments.csv", "split_memberships.csv"),
        ("feature_manifest.csv", "feature_selection_by_run.csv"),
        ("selected_feature_support.csv", "selected_feature_support_by_run.csv"),
        ("input_block_support.csv", "input_block_support_by_run.csv"),
        ("validation_fold_metrics.csv", "validation_fold_metrics.csv"),
    ):
        _collect_run_csv(
            runs,
            filename,
            batch_dir=batch_dir,
            status_column=status_column,
            completed_value=completed_value,
            run_directory_column=run_directory_column,
            metadata_columns=metadata_columns,
        ).to_csv(audit_dir / output_name, index=False)
    inventory_rows: list[dict[str, Any]] = []
    for _, run in completed.iterrows():
        run_dir = _resolve_run_directory(batch_dir, run[run_directory_column])
        try:
            relative_run_directory = str(run_dir.relative_to(batch_dir))
        except ValueError:
            relative_run_directory = str(run_dir)
        for artifact in DEFAULT_AUDIT_ARTIFACTS:
            path = run_dir / artifact
            inventory_rows.append({
                **{column: run.get(column, pd.NA) for column in metadata_columns},
                "relative_run_directory": relative_run_directory,
                "artifact": artifact,
                "exists": path.exists(),
                "bytes": path.stat().st_size if path.exists() else 0,
            })
    inventory_columns = [
        *metadata_columns,
        "relative_run_directory",
        "artifact",
        "exists",
        "bytes",
    ]
    pd.DataFrame(inventory_rows, columns=inventory_columns).to_csv(
        audit_dir / "run_artifact_inventory.csv", index=False
    )
    manifest = {
        "completed_run_summary": "completed_run_summary.csv",
        "split_memberships": "split_memberships.csv",
        "feature_selection": "feature_selection_by_run.csv",
        "selected_feature_support": "selected_feature_support_by_run.csv",
        "input_block_support": "input_block_support_by_run.csv",
        "validation_fold_metrics": "validation_fold_metrics.csv",
        "artifact_inventory": "run_artifact_inventory.csv",
    }
    (audit_dir / "audit_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def write_standard_exports(
    batch_dir: Path,
    runs: pd.DataFrame,
    summary: pd.DataFrame,
    *,
    status_column: str,
    completed_value: str,
    run_directory_column: str,
    metadata_columns: tuple[str, ...],
    selection_columns: tuple[str, ...],
    selection_metric: str | None,
    performance_metrics: tuple[tuple[str, str, bool], ...],
    summary_filename: str,
    retention_metrics: tuple[str, ...] = DEFAULT_REFERENCE_RETENTION_METRICS,
    report_data_subdirectory: str | None = None,
    write_consolidated_predictions: bool = True,
) -> tuple[dict[str, Any], dict[str, str]]:
    """Write the shared ``report/`` and ``audit/`` contracts for a completed batch."""

    report_dir = batch_dir / "report"
    report_dir.mkdir(exist_ok=True)
    data_dir = report_dir / report_data_subdirectory if report_data_subdirectory else report_dir
    data_dir.mkdir(parents=True, exist_ok=True)
    relative_prefix = f"{report_data_subdirectory}/" if report_data_subdirectory else ""
    feature_retention = _collect_run_csv(
        runs,
        "selected_feature_support.csv",
        batch_dir=batch_dir,
        status_column=status_column,
        completed_value=completed_value,
        run_directory_column=run_directory_column,
        metadata_columns=metadata_columns,
    )
    _reference_retention_feature_table(feature_retention, metadata_columns).to_csv(
        data_dir / "reference_retention_by_feature.csv", index=False
    )
    _metric_plot_data(
        runs,
        tuple((metric, metric, False) for metric in retention_metrics),
        status_column=status_column,
        completed_value=completed_value,
        metadata_columns=metadata_columns,
    ).to_csv(data_dir / "reference_retention_metrics_by_run.csv", index=False)
    pd.DataFrame(_REFERENCE_RETENTION_METRIC_DEFINITIONS).to_csv(
        data_dir / "reference_retention_metric_definitions.csv", index=False
    )
    _metric_plot_data(
        runs,
        performance_metrics,
        status_column=status_column,
        completed_value=completed_value,
        metadata_columns=metadata_columns,
    ).to_csv(data_dir / "model_performance_metrics_by_run.csv", index=False)
    catalog = _representative_run_catalog(
        runs,
        status_column=status_column,
        completed_value=completed_value,
        selection_columns=selection_columns,
        selection_metric=selection_metric,
    )
    if write_consolidated_predictions:
        catalog.to_csv(data_dir / "prediction_scatter_representative_run_catalog.csv", index=False)
        _representative_prediction_plot_data(
            catalog,
            batch_dir=batch_dir,
            run_directory_column=run_directory_column,
            metadata_columns=metadata_columns,
        ).to_csv(data_dir / "predicted_vs_observed_representative_runs.csv", index=False)
        _all_prediction_plot_data(
            runs,
            batch_dir=batch_dir,
            status_column=status_column,
            completed_value=completed_value,
            run_directory_column=run_directory_column,
            metadata_columns=metadata_columns,
        ).to_csv(data_dir / "predicted_vs_observed_all_runs.csv", index=False)
    summary.to_csv(data_dir / summary_filename, index=False)
    report_manifest: dict[str, Any] = {
        "reference_retention": {
            "by_feature": f"{relative_prefix}reference_retention_by_feature.csv",
            "metrics_by_run": f"{relative_prefix}reference_retention_metrics_by_run.csv",
            "metric_definitions": f"{relative_prefix}reference_retention_metric_definitions.csv",
        },
        "model_evaluation": {
            "performance_metrics_by_run": f"{relative_prefix}model_performance_metrics_by_run.csv",
        },
        "summary_table": f"{relative_prefix}{summary_filename}",
    }
    if write_consolidated_predictions:
        report_manifest["model_evaluation"].update({
            "predicted_vs_observed_representative_runs": (
                f"{relative_prefix}predicted_vs_observed_representative_runs.csv"
            ),
            "predicted_vs_observed_all_runs": (
                f"{relative_prefix}predicted_vs_observed_all_runs.csv"
            ),
            "prediction_representative_run_catalog": (
                f"{relative_prefix}prediction_scatter_representative_run_catalog.csv"
            ),
            "prediction_selection": "one target-free representative run per report comparison cell",
        })
    (report_dir / "report_manifest.json").write_text(json.dumps(report_manifest, indent=2), encoding="utf-8")
    audit_manifest = _write_audit_outputs(
        batch_dir,
        runs,
        status_column=status_column,
        completed_value=completed_value,
        run_directory_column=run_directory_column,
        metadata_columns=metadata_columns,
    )
    return report_manifest, audit_manifest
