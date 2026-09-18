"""Shared consolidation backend for exploratory logKd experiments."""

from __future__ import annotations

import json
import shutil
from collections.abc import Mapping, MutableMapping, Sequence
from pathlib import Path
from typing import Any

import pandas as pd

from .logkd_coverage import REFERENCE_RETENTION_BLOCK_SLUGS
from .logkd_retention_diagnostics import (
    INNER_REFERENCE_RETENTION_SCORE_COLUMNS,
    summarize_inner_reference_retention,
)


# Fitted runs are model data, not batch bookkeeping: they are kept in a plainly
# named, reader-visible directory so a later post-hoc analysis (for example a
# revised SHAP calculation) never requires retraining the batch.
MODEL_RUNS_DIRNAME = "model_runs"
MODEL_RUN_INDEX_RELATIVE_PATH = Path("audit") / "model_runs.csv"
INCOMPLETE_MODEL_RUNS_RELATIVE_PATH = Path("debug") / "incomplete_model_runs"

PERFORMANCE_METRICS = ("mae", "rmse", "r2", "spearman")
LOGKD_STRATIFIED_PERFORMANCE_METRICS = tuple(
    f"{stratum}_{metric}"
    for stratum in ("low_logkd", "middle_logkd", "high_logkd")
    for metric in PERFORMANCE_METRICS
)
CONCISE_PERFORMANCE_METRICS = (
    *PERFORMANCE_METRICS,
    *LOGKD_STRATIFIED_PERFORMANCE_METRICS,
)
STANDARD_SCENARIO_METRICS = (
    *PERFORMANCE_METRICS,
    *INNER_REFERENCE_RETENTION_SCORE_COLUMNS,
)


def model_runs_root(batch_dir: Path) -> Path:
    """Return the retained fitted-run root of one exploratory batch."""

    return batch_dir / MODEL_RUNS_DIRNAME


def outer_repeat_directory_name(outer_repeat_id: int) -> str:
    """Return the shared directory name for one retained outer repeat."""

    return f"outer_repeat_{int(outer_repeat_id):03d}"


def model_run_directory(
    batch_dir: Path,
    *scenario_parts: str,
    outer_repeat_id: int,
) -> Path:
    """Return the retained directory for one fitted run.

    Experimental-scenario dimensions come first and the outer repeat last, so
    every repeat of one comparison cell is listed together. Each directory keeps
    the trainer's own artifacts (``model.joblib`` when the wrapper saves models,
    ``run_config.json``, ``split_assignments.csv``, ``feature_manifest.csv``,
    and the per-run tables), which is exactly what a post-hoc recalculation
    needs.
    """

    return model_runs_root(batch_dir).joinpath(
        *(str(part) for part in scenario_parts),
        outer_repeat_directory_name(outer_repeat_id),
    )


def quarantine_incomplete_model_runs(
    batch_dir: Path,
    rows: Sequence[MutableMapping[str, Any]],
    *,
    candidate_id: int,
    seed: int,
) -> None:
    """Move a discarded candidate's fitted runs out of the retained tree.

    Outer-repeat numbers are assigned only to retained candidates, so the next
    candidate reuses the number of a discarded one. Relocating the discarded
    directories keeps their console logs and artifacts available for debugging
    while guaranteeing that a retained repeat is never silently overwritten.
    Each moved row's ``run_directory`` is rewritten to its new location.
    """

    root = model_runs_root(batch_dir).resolve()
    destination_root = (
        batch_dir
        / INCOMPLETE_MODEL_RUNS_RELATIVE_PATH
        / f"candidate_{int(candidate_id):03d}_seed_{int(seed)}"
    )
    for row in rows:
        source = Path(str(row.get("run_directory", "")))
        if not str(row.get("run_directory", "")) or not source.exists():
            continue
        resolved = source.resolve()
        if root not in resolved.parents:
            continue
        destination = destination_root / resolved.relative_to(root)
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            shutil.rmtree(destination)
        shutil.move(str(source), str(destination))
        row["run_directory"] = str(destination)


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def summarize_numeric_distribution(values: pd.Series, prefix: str) -> dict[str, float | int]:
    """Return the common repeat-summary statistics used by every wrapper."""

    numeric = pd.to_numeric(values, errors="coerce").dropna()
    if numeric.empty:
        return {}
    count = len(numeric)
    std = float(numeric.std(ddof=1)) if count > 1 else 0.0
    sem = float(numeric.sem(ddof=1)) if count > 1 else 0.0
    return {
        f"{prefix}_mean": float(numeric.mean()),
        f"{prefix}_median": float(numeric.median()),
        f"{prefix}_std": std,
        f"{prefix}_sem": sem,
        f"{prefix}_normal_95ci_half_width": 1.96 * sem,
        f"{prefix}_min": float(numeric.min()),
        f"{prefix}_max": float(numeric.max()),
        f"{prefix}_n": int(count),
    }


def summarize_scenarios(
    runs: pd.DataFrame,
    *,
    group_columns: Sequence[str],
    comparison_group_column: str,
    metrics: Sequence[str] = STANDARD_SCENARIO_METRICS,
) -> pd.DataFrame:
    """Build the standard one-row-per-scenario exploratory summary.

    This legacy wide summary remains available as a helper for specialized
    audit tables. Reader-facing results use ``performance_summary.csv`` and
    ``coverage_summary.csv`` instead.
    """

    if runs.empty or "status" not in runs.columns:
        return pd.DataFrame()
    completed = runs.loc[runs["status"].eq("completed")].copy()
    if completed.empty:
        return pd.DataFrame()
    if comparison_group_column not in group_columns:
        raise ValueError("comparison_group_column must be one of group_columns.")

    records: list[dict[str, Any]] = []
    for keys, group in completed.groupby(list(group_columns), dropna=False, sort=True):
        record = dict(zip(group_columns, keys, strict=True))
        record["comparison_group"] = str(record[comparison_group_column])
        record["n_complete_runs"] = int(len(group))
        if "outer_testing_membership_hash" in group:
            record["n_distinct_outer_tests"] = int(
                group["outer_testing_membership_hash"].nunique()
            )
        for metric in metrics:
            if metric in group:
                record.update(summarize_numeric_distribution(group[metric], metric))
        records.append(record)
    return pd.DataFrame(records)


def _mean_and_sample_sd(values: pd.Series) -> tuple[float, float]:
    numeric = pd.to_numeric(values, errors="coerce").dropna()
    if numeric.empty:
        return float("nan"), float("nan")
    return (
        float(numeric.mean()),
        float(numeric.std(ddof=1)) if len(numeric) > 1 else 0.0,
    )


def summarize_concise_performance(
    runs: pd.DataFrame,
    *,
    group_columns: Sequence[str],
) -> pd.DataFrame:
    """Summarize overall and low/middle/high-logKd held-out performance with mean and SD only."""
    columns = [
        *group_columns,
        "n_complete_runs",
        *(
            column
            for metric in CONCISE_PERFORMANCE_METRICS
            for column in (f"{metric}_mean", f"{metric}_sd")
        ),
    ]
    if runs.empty or "status" not in runs:
        return pd.DataFrame(columns=columns)
    completed = runs.loc[runs["status"].eq("completed")].copy()
    if completed.empty:
        return pd.DataFrame(columns=columns)
    missing = [column for column in group_columns if column not in completed]
    if missing:
        raise ValueError(
            "Performance summary requires grouping columns: " + ", ".join(missing)
        )

    records: list[dict[str, Any]] = []
    for keys, group in completed.groupby(list(group_columns), dropna=False, sort=True):
        record = dict(zip(group_columns, keys, strict=True))
        record["n_complete_runs"] = int(len(group))
        for metric in CONCISE_PERFORMANCE_METRICS:
            mean, sd = _mean_and_sample_sd(
                group.get(metric, pd.Series(dtype=float))
            )
            record[f"{metric}_mean"] = mean
            record[f"{metric}_sd"] = sd
        records.append(record)
    return pd.DataFrame(records, columns=columns)


def summarize_concise_coverage(
    runs: pd.DataFrame,
    *,
    group_columns: Sequence[str],
    assignment_columns: Sequence[str] = ("outer_testing_membership_hash",),
) -> pd.DataFrame:
    """Summarize database contrast and split retention without derived contrast fields."""

    columns = [
        *group_columns,
        "feature_block",
        "n_outer_assignments",
        "database_contrast",
        "contrast_retention_mean",
        "contrast_retention_sd",
    ]
    if runs.empty or "status" not in runs:
        return pd.DataFrame(columns=columns)
    completed = runs.loc[runs["status"].eq("completed")].copy()
    if completed.empty:
        return pd.DataFrame(columns=columns)
    missing = [column for column in group_columns if column not in completed]
    if missing:
        raise ValueError(
            "Coverage summary requires grouping columns: " + ", ".join(missing)
        )
    available_assignment_columns = [
        column for column in assignment_columns if column in completed
    ]

    records: list[dict[str, Any]] = []
    for keys, group in completed.groupby(list(group_columns), dropna=False, sort=True):
        values = (
            group.drop_duplicates(available_assignment_columns, keep="first")
            if available_assignment_columns
            else group
        )
        context = dict(zip(group_columns, keys, strict=True))
        for block, slug in REFERENCE_RETENTION_BLOCK_SLUGS.items():
            database, _ = _mean_and_sample_sd(
                values.get(f"database_contrast_{slug}", pd.Series(dtype=float))
            )
            retention_mean, retention_sd = _mean_and_sample_sd(
                values.get(f"contrast_retention_{slug}", pd.Series(dtype=float))
            )
            records.append({
                **context,
                "feature_block": block,
                "n_outer_assignments": int(len(values)),
                "database_contrast": database,
                "contrast_retention_mean": retention_mean,
                "contrast_retention_sd": retention_sd,
            })
    return pd.DataFrame(records, columns=columns)


def write_concise_result_summaries(
    summary_dir: Path,
    runs: pd.DataFrame,
    *,
    performance_group_columns: Sequence[str],
    coverage_group_columns: Sequence[str],
) -> dict[str, str]:
    """Write the standard compact performance and coverage result tables."""

    summary_dir.mkdir(parents=True, exist_ok=True)
    performance = summarize_concise_performance(
        runs, group_columns=performance_group_columns
    )
    coverage = summarize_concise_coverage(
        runs, group_columns=coverage_group_columns
    )
    performance.to_csv(summary_dir / "performance_summary.csv", index=False)
    coverage.to_csv(summary_dir / "coverage_summary.csv", index=False)
    return {
        "performance_summary": "summary/performance_summary.csv",
        "coverage_summary": "summary/coverage_summary.csv",
    }


def _context_values(
    run: pd.Series,
    columns: Sequence[str],
    *,
    assignment_id_column: str | None,
    include_run_id: bool,
) -> dict[str, Any]:
    values: dict[str, Any] = {}
    if include_run_id:
        values["run_id"] = run["run_id"]
    if assignment_id_column:
        values[assignment_id_column] = run.get(assignment_id_column, pd.NA)
    for column in columns:
        values[column] = run.get(column, pd.NA)
    return values


def _with_context(
    frame: pd.DataFrame,
    run: pd.Series,
    columns: Sequence[str],
    *,
    assignment_id_column: str | None,
    include_run_id: bool = True,
) -> pd.DataFrame:
    output = frame.copy()
    for column, value in _context_values(
        run,
        columns,
        assignment_id_column=assignment_id_column,
        include_run_id=include_run_id,
    ).items():
        output[column] = value
    return output


def write_testing_prediction_detail(
    batch_dir: Path,
    runs: pd.DataFrame,
    *,
    run_context_columns: Sequence[str],
    output_relative_path: Path = Path("detail") / "testing_predictions.parquet",
) -> Path | None:
    """Write the canonical held-out prediction table for a completed batch.

    The table keeps the trainer's row-level prediction context and appends the
    wrapper's scenario and repeat identifiers.  It is the one canonical input
    for prediction diagnostics; no parallel CSV export is needed.
    """

    required = {"status", "run_directory"}
    missing = required.difference(runs.columns)
    if missing:
        raise ValueError(
            "Prediction export requires run columns: " + ", ".join(sorted(missing))
        )
    completed = runs.loc[runs["status"].eq("completed")].copy()
    if completed.empty:
        return None

    frames: list[pd.DataFrame] = []
    include_run_id = "run_id" in completed.columns
    for _, run in completed.iterrows():
        run_dir = Path(str(run["run_directory"]))
        predictions = _read_csv(run_dir / "all_predictions.csv")
        if predictions.empty:
            raise RuntimeError(
                f"Completed run did not write all_predictions.csv: {run_dir}"
            )
        required_prediction_columns = {"data_split", "y_true", "y_pred"}
        missing_prediction_columns = required_prediction_columns.difference(predictions.columns)
        if missing_prediction_columns:
            raise RuntimeError(
                "Completed run prediction output is missing columns "
                f"{sorted(missing_prediction_columns)}: {run_dir}"
            )
        testing = predictions.loc[predictions["data_split"].eq("testing")].copy()
        if testing.empty:
            raise RuntimeError(f"Completed run has no testing predictions: {run_dir}")
        frames.append(
            _with_context(
                testing,
                run,
                run_context_columns,
                assignment_id_column=None,
                include_run_id=include_run_id,
            )
        )

    output_path = batch_dir / output_relative_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.concat(frames, ignore_index=True, sort=False).to_parquet(output_path, index=False)
    return output_path


def _safe_remove(path: Path, batch_dir: Path) -> None:
    if not path.exists():
        return
    resolved = path.resolve()
    root = batch_dir.resolve()
    if root not in resolved.parents:
        raise RuntimeError(
            f"Refusing to remove path outside exploratory batch: {resolved}"
        )
    if resolved == root:
        raise RuntimeError("Refusing to remove the exploratory batch root.")
    if path.is_dir():
        shutil.rmtree(path)
    else:
        path.unlink()


def _reject_model_run_cleanup(cleanup_path: Path, batch_dir: Path) -> None:
    """Refuse to delete retained fitted runs through the cleanup hook."""

    resolved = cleanup_path.resolve()
    root = model_runs_root(batch_dir).resolve()
    if resolved == root or root in resolved.parents:
        raise ValueError(
            "Fitted model runs are retained for post-hoc analysis and cannot be "
            f"removed through extra_cleanup_paths: {cleanup_path}"
        )


def _batch_relative_path(path: Path, batch_dir: Path) -> str:
    try:
        return path.resolve().relative_to(batch_dir.resolve()).as_posix()
    except ValueError:
        return str(path)


def _model_run_index(
    batch_dir: Path,
    runs: pd.DataFrame,
    context_columns: Sequence[str],
) -> pd.DataFrame:
    """Map every run to its retained directory and the artifacts it kept.

    ``audit/run_summary.csv`` deliberately drops filesystem paths, so this is
    the one table that says where a given run's fitted artifacts now live.
    """

    available_context = [column for column in context_columns if column in runs.columns]
    columns = [
        "run_id",
        *available_context,
        "status",
        "model_run_path",
        "has_saved_model",
        "has_run_config",
        "has_split_assignments",
    ]
    records: list[dict[str, Any]] = []
    for _, run in runs.iterrows():
        directory_text = str(run.get("run_directory", ""))
        directory = Path(directory_text) if directory_text else None
        records.append({
            "run_id": run.get("run_id"),
            **{column: run.get(column) for column in available_context},
            "status": run.get("status"),
            "model_run_path": (
                _batch_relative_path(directory, batch_dir) if directory else ""
            ),
            "has_saved_model": bool(directory and (directory / "model.joblib").exists()),
            "has_run_config": bool(directory and (directory / "run_config.json").exists()),
            "has_split_assignments": bool(
                directory and (directory / "split_assignments.csv").exists()
            ),
        })
    return pd.DataFrame(records, columns=columns)


def write_model_runs_readme(batch_dir: Path) -> None:
    """Explain why the fitted runs are kept and how to reuse them."""

    root = model_runs_root(batch_dir)
    if not root.is_dir():
        return
    (root / "README.md").write_text(
        f"""# Retained model runs

Each directory here is one fitted evaluation run, laid out as
`<scenario dimensions>/{outer_repeat_directory_name(1)}/`. These are model
data, not batch bookkeeping: they are kept on purpose so post-hoc analyses can
be repeated or corrected without retraining the batch.

A run directory holds the trainer's own artifacts, including `run_config.json`
(with the `analysis_reconstruction` block that rebuilds the exact model frame),
`split_assignments.csv`, `feature_manifest.csv`, `all_predictions.csv`, and
`model.joblib` when the wrapper saved the fitted pipeline.

`{MODEL_RUN_INDEX_RELATIVE_PATH.as_posix()}` maps every run to its directory
here and records whether a saved model is present. Runs from candidates that
were discarded before retention are moved to
`{INCOMPLETE_MODEL_RUNS_RELATIVE_PATH.as_posix()}/` instead.

To recompute held-out SHAP for one scenario without retraining, pass that
scenario's run directories to `logkd_shap_outer_robustness.py`:

```
python logkd_shap_outer_robustness.py \\
  --run-dirs <scenario>/{outer_repeat_directory_name(1)} <scenario>/{outer_repeat_directory_name(2)} \\
  --output-dir <new output directory>
```
""",
        encoding="utf-8",
    )


def _configuration_records(
    completed: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    records: list[dict[str, Any]] = []
    summary_records: list[dict[str, Any]] = []
    for _, run in completed.iterrows():
        run_dir = Path(str(run["run_directory"]))
        config_path = run_dir / "run_config.json"
        config: dict[str, Any] = {}
        if config_path.exists():
            config = json.loads(config_path.read_text(encoding="utf-8"))
        selected_features = config.get("features", {}).get("selected", [])
        best_params = config.get("best_params", {})
        record = {
            "run_id": run["run_id"],
            "regressor": config.get("regressor"),
            "best_params_json": json.dumps(best_params, sort_keys=True),
            "selected_feature_count": len(selected_features),
            "selected_features_json": json.dumps(
                selected_features,
                ensure_ascii=False,
            ),
            "run_config_json": json.dumps(
                config,
                ensure_ascii=False,
                sort_keys=True,
            ),
        }
        record.update(summarize_inner_reference_retention(run_dir))
        records.append(record)
        summary_records.append({
            "run_id": record["run_id"],
            "regressor": record["regressor"],
            "best_params_json": record["best_params_json"],
            "selected_feature_count": record["selected_feature_count"],
            **{
                column: record[column]
                for column in INNER_REFERENCE_RETENTION_SCORE_COLUMNS
            },
        })
    columns = [
        "run_id",
        "regressor",
        "best_params_json",
        "selected_feature_count",
        "selected_features_json",
        "run_config_json",
        *INNER_REFERENCE_RETENTION_SCORE_COLUMNS,
    ]
    summary_columns = [
        "run_id",
        "regressor",
        "best_params_json",
        "selected_feature_count",
        *INNER_REFERENCE_RETENTION_SCORE_COLUMNS,
    ]
    return (
        pd.DataFrame(records, columns=columns),
        pd.DataFrame(summary_records, columns=summary_columns),
    )


def _merge_configuration_summary(
    completed: pd.DataFrame,
    configuration_summary: pd.DataFrame,
) -> pd.DataFrame:
    output = completed.copy()
    available = [
        column
        for column in configuration_summary.columns
        if column == "run_id" or column not in output.columns
    ]
    if available == ["run_id"]:
        return output
    return output.merge(
        configuration_summary[available],
        on="run_id",
        how="left",
        validate="one_to_one",
    )


def write_exploratory_outputs(
    batch_dir: Path,
    runs: pd.DataFrame,
    *,
    run_context_columns: Sequence[str],
    assignment_context_columns: Sequence[str] = (),
    assignment_id_column: str | None = None,
    assignment_path_column: str | None = None,
    assignment_filename: str = "split_assignments.parquet",
    debug_tables: Mapping[str, pd.DataFrame] | None = None,
    failures: Sequence[str] = (),
    extra_cleanup_paths: Sequence[Path] = (),
    performance_group_columns: Sequence[str] | None = None,
    coverage_group_columns: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Consolidate standard trainer artifacts for any exploratory wrapper.

    The caller owns experiment-specific identifiers, comparisons, README text,
    and configuration terminology. This backend owns the stable trainer-artifact
    collection, CSV/Parquet policy, failure-log retention, and preparation-work
    cleanup.

    Fitted runs under ``model_runs/`` are never removed here. Consolidating them
    into ``summary/``, ``detail/``, and ``audit/`` is an additional reading
    surface, not a replacement for the fitted models and their exact input
    contracts, which a later post-hoc analysis still needs.
    """

    required = {"status", "run_directory", "run_id"}
    missing = required.difference(runs.columns)
    if missing:
        raise ValueError(
            "Exploratory run table is missing required columns: "
            + ", ".join(sorted(missing))
        )
    if assignment_id_column and assignment_id_column not in runs.columns:
        raise ValueError(
            f"Exploratory run table is missing assignment ID column "
            f"{assignment_id_column!r}."
        )

    summary_dir = batch_dir / "summary"
    detail_dir = batch_dir / "detail"
    audit_dir = batch_dir / "audit"
    debug_dir = batch_dir / "debug"
    for directory in (
        summary_dir,
        detail_dir,
        audit_dir,
    ):
        directory.mkdir(parents=True, exist_ok=True)

    # Remove stale pre-cleanup residual exports when re-finalizing an older
    # batch so the summary remains a performance-only reading surface.
    for filename in (
        "CDP_residual_analysis_artifact.json",
        "CDP_residual_analysis_report.html",
        "residual_analysis_manifest.json",
        "residual_candidate_association_by_run.csv",
        "residual_candidate_association_summary.csv",
        "residual_candidate_screening_by_run.csv",
        "residual_candidate_screening_summary.csv",
        "residual_group_heterogeneity_by_run.csv",
        "residual_group_heterogeneity_summary.csv",
        "residual_model_evaluation_by_run.csv",
        "residual_model_evaluation_summary.csv",
    ):
        (summary_dir / filename).unlink(missing_ok=True)

    completed = runs.loc[runs["status"].eq("completed")].copy()
    configurations, configuration_summary = _configuration_records(completed)
    selected_feature_support_frames: list[pd.DataFrame] = []
    prediction_frames: list[pd.DataFrame] = []
    validation_frames: list[pd.DataFrame] = []
    assignment_frames: list[pd.DataFrame] = []
    seen_assignments: set[str] = set()

    for _, run in completed.iterrows():
        run_dir = Path(str(run["run_directory"]))
        assignment_key = (
            str(run[assignment_id_column])
            if assignment_id_column
            else str(run["run_id"])
        )

        if assignment_key not in seen_assignments:
            if assignment_path_column:
                assignment_path = Path(str(run.get(assignment_path_column, "")))
                assignment = _read_csv(assignment_path)
                if not assignment.empty:
                    assignment_frames.append(
                        _with_context(
                            assignment,
                            run,
                            assignment_context_columns,
                            assignment_id_column=assignment_id_column,
                            include_run_id=False,
                        )
                    )
            seen_assignments.add(assignment_key)

        predictions = _read_csv(run_dir / "all_predictions.csv")
        if not predictions.empty and "data_split" in predictions.columns:
            predictions = predictions.loc[
                predictions["data_split"].eq("testing")
            ].copy()
            prediction_frames.append(
                _with_context(
                    predictions,
                    run,
                    run_context_columns,
                    assignment_id_column=assignment_id_column,
                )
            )

        selected_feature_support = _read_csv(run_dir / "selected_feature_support.csv")
        if not selected_feature_support.empty:
            selected_feature_support_frames.append(
                _with_context(
                    selected_feature_support,
                    run,
                    run_context_columns,
                    assignment_id_column=assignment_id_column,
                )
            )

        validation = _read_csv(run_dir / "validation_fold_metrics.csv")
        if not validation.empty:
            validation_frames.append(
                _with_context(
                    validation,
                    run,
                    run_context_columns,
                    assignment_id_column=assignment_id_column,
                )
            )

    run_summary = _merge_configuration_summary(
        completed,
        configuration_summary,
    )
    path_columns = [
        column
        for column in run_summary.columns
        if "path" in column.casefold() or "directory" in column.casefold()
    ]
    run_summary.drop(columns=path_columns, errors="ignore").to_csv(
        audit_dir / "run_summary.csv",
        index=False,
    )
    (summary_dir / "run_summary.csv").unlink(missing_ok=True)

    selected_feature_support_detail = (
        pd.concat(selected_feature_support_frames, ignore_index=True, sort=False)
        if selected_feature_support_frames
        else pd.DataFrame()
    )
    for legacy_filename in (
        "coverage_feature_detail.csv",
        "observed_support_by_row.parquet",
    ):
        (detail_dir / legacy_filename).unlink(missing_ok=True)
    selected_feature_support_detail.to_csv(
        detail_dir / "selected_feature_support_detail.csv",
        index=False,
    )
    predictions = (
        pd.concat(prediction_frames, ignore_index=True, sort=False)
        if prediction_frames
        else pd.DataFrame()
    )
    predictions.to_parquet(
        detail_dir / "testing_predictions.parquet",
        index=False,
    )
    validation = (
        pd.concat(validation_frames, ignore_index=True, sort=False)
        if validation_frames
        else pd.DataFrame()
    )
    validation.to_parquet(
        detail_dir / "validation_detail.parquet",
        index=False,
    )
    configurations.to_parquet(
        audit_dir / "run_configurations.parquet",
        index=False,
    )

    assignments = (
        pd.concat(assignment_frames, ignore_index=True, sort=False)
        if assignment_frames
        else pd.DataFrame()
    )
    audit_manifest: dict[str, str] = {
        "run_configurations": "audit/run_configurations.parquet",
        "run_summary": "audit/run_summary.csv",
    }
    if assignment_path_column:
        assignments.to_parquet(
            audit_dir / assignment_filename,
            index=False,
        )
        audit_manifest["split_assignments"] = f"audit/{assignment_filename}"

    nonempty_debug_tables = {
        filename: frame
        for filename, frame in (debug_tables or {}).items()
        if not frame.empty
    }
    if nonempty_debug_tables or failures:
        debug_dir.mkdir(parents=True, exist_ok=True)
    for filename, frame in nonempty_debug_tables.items():
        frame.to_csv(debug_dir / filename, index=False)
    if failures:
        pd.DataFrame({"failure": list(failures)}).to_csv(
            debug_dir / "failures.csv",
            index=False,
        )

    failed = runs.loc[~runs["status"].eq("completed")]
    for _, run in failed.iterrows():
        run_dir = Path(str(run.get("run_directory", "")))
        log_path = run_dir / "console_output.txt"
        if log_path.exists():
            destination = debug_dir / "failed_runs"
            destination.mkdir(parents=True, exist_ok=True)
            shutil.copy2(
                log_path,
                destination / f"{run['run_id']}_console_output.txt",
            )

    for cleanup_path in extra_cleanup_paths:
        _reject_model_run_cleanup(cleanup_path, batch_dir)
        _safe_remove(cleanup_path, batch_dir)

    model_run_index = _model_run_index(batch_dir, runs, run_context_columns)
    model_run_index.to_csv(audit_dir / MODEL_RUN_INDEX_RELATIVE_PATH.name, index=False)
    audit_manifest["model_runs"] = MODEL_RUN_INDEX_RELATIVE_PATH.as_posix()
    write_model_runs_readme(batch_dir)

    summary_manifest: dict[str, str] = {}
    if performance_group_columns is not None or coverage_group_columns is not None:
        if performance_group_columns is None or coverage_group_columns is None:
            raise ValueError(
                "Both performance_group_columns and coverage_group_columns are required."
            )
        summary_manifest.update(write_concise_result_summaries(
            summary_dir,
            completed,
            performance_group_columns=performance_group_columns,
            coverage_group_columns=coverage_group_columns,
        ))
    return {
        "summary": summary_manifest,
        "detail": {
            "selected_feature_support_detail": "detail/selected_feature_support_detail.csv",
            "testing_predictions": "detail/testing_predictions.parquet",
            "validation_detail": "detail/validation_detail.parquet",
        },
        "audit": {
            **audit_manifest,
        },
        "_counts": {
            "completed_run_count": int(len(completed)),
            "distinct_assignment_count": int(len(seen_assignments)),
            "selected_feature_support_detail_rows": int(len(selected_feature_support_detail)),
            "testing_prediction_rows": int(len(predictions)),
            "validation_detail_rows": int(len(validation)),
        },
    }
