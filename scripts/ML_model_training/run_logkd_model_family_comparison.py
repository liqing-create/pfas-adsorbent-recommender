"""Run one native-XGBoost comparison across the configured model families.

The four scopes are fitted in one batch: Global, AC, Resin, and CDP.  Each
scope is filtered before a repeated random-source-row split is constructed, so
their held-out sets are intentionally distinct.  The batch is therefore a
consistent multi-scope benchmark, not a paired Global-versus-specialist test.

Each scope's completed outer repeats are also summarized with held-out SHAP
robustness outputs under ``shap/<model>/``, so feature importance is reported
per model family rather than pooled across scopes with different inputs.
"""

from __future__ import annotations

import argparse
import json
import sys
import zlib
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from backend import logkd_config as cfg
from backend.logkd_exploratory_exports import (
    model_run_directory,
    model_runs_root,
    quarantine_incomplete_model_runs,
    write_exploratory_outputs,
)
from backend.logkd_plotting import generate_aggregate_figure
from backend.logkd_progress import finalize_run_summary, write_live_run_summary
from logkd_shap_outer_robustness import generate_robustness_summary
import run_logkd_missing_strategy_comparison as paired_common


MODEL_FAMILIES = ("Global", "AC", "Resin", "CDP")
SPLIT_STRATEGY = "random_row"
OUTER_ALLOCATION_METHOD = "random_row"
INNER_ALLOCATION_METHOD = "random_row"
NUMERIC_MISSING_STRATEGY = "xgb_native"
DEFAULT_OUTER_REPEATS = cfg.DEFAULT_EXPERIMENT_OUTER_REPEATS
DEFAULT_N_TRIALS = cfg.DEFAULT_EXPERIMENT_N_TRIALS
DEFAULT_EARLY_STOP_WARMUP = cfg.DEFAULT_EXPERIMENT_EARLY_STOP_WARMUP
DEFAULT_EARLY_STOP_PATIENCE = cfg.DEFAULT_EXPERIMENT_EARLY_STOP_PATIENCE
DEFAULT_SHAP_TOP_K = 5
DEFAULT_SHAP_MAX_DISPLAY = 15
MAX_SEED_CANDIDATES_PER_REQUESTED_REPEAT = 20


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Fit Global, AC, Resin, and CDP XGBoost models in one batch. "
            "Every scope uses repeated random-source-row holdouts and XGBoost's "
            "native numeric-NaN handling."
        )
    )
    parser.add_argument(
        "--models",
        choices=sorted(cfg.MODEL_CATEGORIES),
        nargs="+",
        default=list(MODEL_FAMILIES),
        help="Model families to include (default: Global AC Resin CDP).",
    )
    parser.add_argument("--outer-repeats", type=int, default=DEFAULT_OUTER_REPEATS)
    parser.add_argument("--random-seed", type=int, default=cfg.DEFAULT_EXPERIMENT_RANDOM_SEED)
    parser.add_argument("--model-random-seed", type=int, default=cfg.DEFAULT_MODEL_RANDOM_SEED)
    parser.add_argument("--output-root", type=Path, default=cfg.DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--test-fraction", type=float, default=cfg.DEFAULT_TEST_FRACTION)
    parser.add_argument("--validation-folds", type=int, default=cfg.VALIDATION_FOLDS)
    parser.add_argument(
        "--data-mode",
        choices=("baseline", "drop_unreliable"),
        default=cfg.DEFAULT_DATA_MODE,
    )
    parser.add_argument(
        "--pfas-feature-family-policy",
        choices=cfg.PFAS_FEATURE_FAMILY_POLICIES,
        default=cfg.DEFAULT_PFAS_FEATURE_FAMILY_POLICY,
    )
    parser.add_argument(
        "--correlated-feature-handling",
        choices=cfg.CORRELATED_FEATURE_HANDLING_CHOICES,
        default=cfg.DEFAULT_CORRELATED_FEATURE_HANDLING,
    )
    parser.add_argument("--n-trials", type=int, default=DEFAULT_N_TRIALS)
    parser.add_argument("--early-stop-warmup", type=int, default=DEFAULT_EARLY_STOP_WARMUP)
    parser.add_argument("--early-stop-patience", type=int, default=DEFAULT_EARLY_STOP_PATIENCE)
    parser.add_argument(
        "--early-stop-se-multiplier",
        type=float,
        default=cfg.DEFAULT_EXPERIMENT_EARLY_STOP_SE_MULTIPLIER,
    )
    parser.add_argument(
        "--early-stop-min-delta-floor",
        type=float,
        default=cfg.DEFAULT_EXPERIMENT_EARLY_STOP_MIN_DELTA_FLOOR,
    )
    parser.add_argument("--n-jobs", type=int, default=cfg.DEFAULT_EXPERIMENT_N_JOBS)
    parser.add_argument(
        "--save-model",
        dest="save_model",
        action="store_true",
        default=True,
        help="Save each outer model for automatic held-out SHAP analysis (default: enabled).",
    )
    parser.add_argument(
        "--no-save-model",
        dest="save_model",
        action="store_false",
        help="Do not save fitted outer models; automatic SHAP will be skipped.",
    )
    parser.add_argument(
        "--no-shap",
        dest="run_shap",
        action="store_false",
        default=True,
        help="Disable automatic per-model-family held-out SHAP robustness summaries.",
    )
    parser.add_argument(
        "--shap-top-k",
        type=int,
        default=DEFAULT_SHAP_TOP_K,
        help="Top-rank threshold for SHAP stability summaries (default: 5).",
    )
    parser.add_argument(
        "--shap-max-display",
        type=int,
        default=DEFAULT_SHAP_MAX_DISPLAY,
        help="Maximum features displayed in each SHAP importance plot (default: 15).",
    )
    parser.add_argument(
        "--console-verbosity",
        choices=("quiet", "normal"),
        default=cfg.DEFAULT_EXPERIMENT_CONSOLE_VERBOSITY,
    )
    parser.add_argument(
        "--training-args",
        nargs=argparse.REMAINDER,
        default=[],
        help="Additional arguments forwarded to train_logkd_model.py.",
    )
    args = parser.parse_args()
    if args.outer_repeats < 1:
        parser.error("--outer-repeats must be at least one.")
    if not 0 < args.test_fraction < 1:
        parser.error("--test-fraction must be between zero and one.")
    if args.validation_folds < 2:
        parser.error("--validation-folds must be at least two.")
    if args.n_trials < 0:
        parser.error("--n-trials must be non-negative.")
    if args.shap_top_k < 1:
        parser.error("--shap-top-k must be at least one.")
    if args.shap_max_display < 1:
        parser.error("--shap-max-display must be at least one.")
    if len(set(args.models)) != len(args.models):
        parser.error("--models must not contain duplicates.")
    return args


def _candidate_seeds(root_seed: int, count: int) -> list[int]:
    salt = zlib.crc32(b"model_family_comparison:random_row")
    return [
        int(value)
        for value in np.random.SeedSequence([root_seed, salt]).generate_state(count)
    ]


def _prepare_command(
    train_script: Path,
    args: argparse.Namespace,
    *,
    model: str,
    seed: int,
    output_dir: Path,
) -> list[str]:
    return [
        sys.executable,
        "-u",
        str(train_script),
        *args.training_args,
        "--model",
        model,
        "--split-strategy",
        SPLIT_STRATEGY,
        "--outer-allocation-method",
        OUTER_ALLOCATION_METHOD,
        "--random-seed",
        str(seed),
        "--prepare-split-only",
        "--output-dir",
        str(output_dir),
        *paired_common._fixed_training_arguments(args),
    ]


def _training_command(
    train_script: Path,
    args: argparse.Namespace,
    *,
    model: str,
    seed: int,
    assignment_path: Path,
    output_dir: Path,
) -> list[str]:
    return [
        sys.executable,
        "-u",
        str(train_script),
        *args.training_args,
        "--model",
        model,
        "--split-strategy",
        SPLIT_STRATEGY,
        "--outer-allocation-method",
        OUTER_ALLOCATION_METHOD,
        "--inner-allocation-method",
        INNER_ALLOCATION_METHOD,
        "--random-seed",
        str(seed),
        "--split-assignment-path",
        str(assignment_path),
        "--numeric-missing-strategy",
        NUMERIC_MISSING_STRATEGY,
        "--output-dir",
        str(output_dir),
        *paired_common._fixed_training_arguments(args),
        *(["--save-model"] if args.save_model else []),
    ]


def _summary_row(
    run_dir: Path,
    *,
    model: str,
    outer_repeat_id: int,
    candidate_id: int,
    seed: int,
    assignment_path: Path,
    testing_membership_hash: str,
    testing_rows_from_assignment: int,
    status: str,
    message: str = "",
) -> dict[str, Any]:
    row = paired_common._summary_row(
        run_dir,
        split_strategy=SPLIT_STRATEGY,
        outer_repeat_id=outer_repeat_id,
        candidate_id=candidate_id,
        seed=seed,
        outer_allocation_method=OUTER_ALLOCATION_METHOD,
        assignment_path=assignment_path,
        testing_membership_hash=testing_membership_hash,
        testing_rows_from_assignment=testing_rows_from_assignment,
        numeric_missing_strategy=NUMERIC_MISSING_STRATEGY,
        status=status,
        message=message,
    )
    return {"model": model, **row}


def _run_id(run: pd.Series) -> str:
    return (
        f"{run['model']}__outer_{int(run['outer_repeat_id']):03d}_"
        f"seed_{int(run['random_seed'])}__{NUMERIC_MISSING_STRATEGY}"
    )


def _outer_assignment_id(run: pd.Series) -> str:
    digest = str(run.get("outer_testing_membership_hash", "unknown"))
    return f"{run['model']}__outer_{digest[:16]}"


def _testing_feature_availability_detail(runs: pd.DataFrame) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    context = (
        "model",
        "split_strategy",
        "outer_allocation_method",
        "outer_repeat_id",
        "candidate_id",
        "random_seed",
        "outer_testing_membership_hash",
        "numeric_missing_strategy",
        "run_id",
    )
    completed = runs.loc[runs.get("status", pd.Series(dtype=str)).eq("completed")]
    for _, run in completed.iterrows():
        path = Path(str(run["run_directory"])) / "testing_metrics_by_feature_availability.csv"
        try:
            detail = pd.read_csv(path) if path.exists() else pd.DataFrame()
        except (OSError, pd.errors.ParserError, UnicodeDecodeError):
            detail = pd.DataFrame()
        if detail.empty:
            continue
        for column in context:
            detail[column] = run.get(column)
        frames.append(detail)
    return pd.concat(frames, ignore_index=True, sort=False) if frames else pd.DataFrame()


def _completed_runs(runs: pd.DataFrame) -> pd.DataFrame:
    if runs.empty or "status" not in runs:
        return pd.DataFrame(columns=runs.columns)
    return runs.loc[runs["status"].eq("completed")].copy()


def _automatic_shap_outputs(
    batch_dir: Path,
    runs: pd.DataFrame,
    args: argparse.Namespace,
) -> dict[str, Any]:
    """Create held-out SHAP robustness summaries within each model family."""

    manifest: dict[str, Any] = {
        "enabled": bool(args.run_shap),
        "top_k": args.shap_top_k,
        "max_display": args.shap_max_display,
        "groups": [],
    }
    if not args.run_shap:
        manifest["status"] = "disabled"
        return manifest
    if not args.save_model:
        manifest.update(
            {
                "status": "skipped",
                "reason": (
                    "Automatic SHAP was skipped because --no-save-model disables the "
                    "outer model artifacts required for explanation."
                ),
            }
        )
        return manifest

    completed = _completed_runs(runs)
    if completed.empty:
        manifest.update(
            {
                "status": "skipped",
                "reason": "No completed outer runs were available for SHAP analysis.",
            }
        )
        return manifest

    failures: list[str] = []
    for model, group in completed.groupby("model", sort=False):
        model = str(model)
        run_dirs = [Path(str(value)) for value in group["run_directory"].tolist()]
        relative_output_dir = Path("shap") / model
        record: dict[str, Any] = {
            "model": model,
            "outer_runs": len(run_dirs),
            "output_dir": relative_output_dir.as_posix(),
        }
        if len(run_dirs) < 2:
            record.update(
                {
                    "status": "skipped",
                    "reason": "At least two completed outer runs are required.",
                }
            )
            manifest["groups"].append(record)
            continue
        print(f"Summarizing held-out SHAP: model={model}", flush=True)
        try:
            result = generate_robustness_summary(
                run_dirs,
                batch_dir / relative_output_dir,
                top_k=args.shap_top_k,
                max_display=args.shap_max_display,
            )
        except Exception as error:  # Preserve completed comparisons if SHAP fails.
            message = f"{type(error).__name__}: {error}"
            record.update({"status": "failed", "message": message})
            failures.append(f"SHAP failed for {model}: {message}")
        else:
            record.update(
                {
                    "status": "completed",
                    "features": result["features"],
                    "files": result["files"],
                    "representative_run": result["representative_run"],
                }
            )
        manifest["groups"].append(record)

    manifest["status"] = "completed_with_errors" if failures else "completed"
    if failures:
        manifest["failures"] = failures
    return manifest


def _write_readme(batch_dir: Path) -> None:
    (batch_dir / "START_HERE.md").write_text(
        """# Native-XGBoost model-family comparison

## Question

How does the Global XGBoost model perform relative to the AC, Resin, and CDP
specialist models when each scope uses repeated random-source-row holdouts and
native XGBoost handling of numeric missing values?

## Experimental design

This batch fits the requested model families sequentially under one frozen
protocol: `random_row` outer and inner allocation, XGBoost regression, and
`xgb_native` numeric missing-data handling. Every model family uses the same
root split seed, estimator seed, tuning budget, feature-selection policy, and
data-quality mode.

Each family is filtered before its split is created. Therefore, its held-out
rows, source-record count, target distribution, and feature availability can
differ from the other families. The aggregate metrics are a consistent
multi-scope benchmark, not paired Global-versus-specialist differences.

`random_row` samples whole `source_row_index` records, keeping all
isotherm-expanded records from one source row in the same outer partition and
inner validation fold.

## Automatic SHAP outputs

Held-out SHAP robustness summaries are generated by default for every model
family that retained at least two complete outer repeats, and are written to
`shap/<model>/`. Importance is aggregated only within a family, because each
family is filtered and split separately and can retain different input
features. Use `--no-shap` to disable this post-processing, or `--no-save-model`
when the per-repeat `model.joblib` artifacts are not needed; either flag skips
SHAP.

`shap_robustness_importance.png` is the cross-repeat aggregate.
`shap_representative_beeswarm.png` shows the signed per-row SHAP values of the
one repeat whose held-out metrics sit closest to that family's mean, because a
beeswarm needs a single model; `shap_representative_run_selection.csv` records
which repeat was chosen. Read the beeswarm as one typical repeat, not as an
aggregate.

## Read these files first

- `summary/performance_summary.csv`: held-out MAE, RMSE, R2, and Spearman by
  model family, summarized across retained outer repeats.
- `audit/run_summary.csv`: one row per fitted model-family/repeat run.
- `audit/outer_assignment_manifest.csv`: scope-specific outer split metadata.
- `detail/testing_metrics_by_feature_availability.csv`: diagnostics for present
  versus missing selected input features.
- `shap/<model>/shap_robustness_summary.csv`: held-out SHAP importance
  aggregated across that family's outer repeats.
- `audit/model_runs.csv`: locations of all retained fitted model artifacts.

For a strict Global-versus-specialist head-to-head test on the same category
holdouts, use a separate matched-cohort evaluation that excludes those holdout
source records from both models' training data.
""",
        encoding="utf-8",
    )


def _write_outputs(
    batch_dir: Path,
    runs: pd.DataFrame,
    assignments: pd.DataFrame,
    discarded_rows: list[dict[str, Any]],
    failures: list[str],
    args: argparse.Namespace,
    shap_manifest: dict[str, Any],
) -> dict[str, Any]:
    export_runs = runs.copy()
    if export_runs.empty:
        export_runs["run_id"] = pd.Series(dtype="string")
        export_runs["outer_assignment_id"] = pd.Series(dtype="string")
    else:
        export_runs["run_id"] = export_runs.apply(_run_id, axis=1)
        export_runs["outer_assignment_id"] = export_runs.apply(
            _outer_assignment_id, axis=1
        )

    availability_detail = _testing_feature_availability_detail(export_runs)
    manifest = write_exploratory_outputs(
        batch_dir,
        export_runs,
        run_context_columns=(
            "model",
            "split_strategy",
            "outer_allocation_method",
            "inner_allocation_method",
            "outer_repeat_id",
            "candidate_id",
            "random_seed",
            "numeric_missing_strategy",
        ),
        assignment_context_columns=(
            "model",
            "split_strategy",
            "outer_allocation_method",
            "inner_allocation_method",
            "outer_repeat_id",
            "candidate_id",
            "random_seed",
            "outer_testing_membership_hash",
        ),
        assignment_id_column="outer_assignment_id",
        assignment_path_column="outer_assignment_path",
        assignment_filename="outer_assignments.parquet",
        debug_tables={"discarded_outer_candidates.csv": pd.DataFrame(discarded_rows)},
        failures=failures,
        extra_cleanup_paths=(batch_dir / ".batch_metadata" / "outer_assignments",),
        performance_group_columns=("model",),
        coverage_group_columns=("model",),
    )
    counts = manifest.pop("_counts")
    detail_dir = batch_dir / "detail"
    audit_dir = batch_dir / "audit"
    availability_detail.to_csv(
        detail_dir / "testing_metrics_by_feature_availability.csv", index=False
    )
    assignment_manifest = assignments.drop(
        columns=[
            column
            for column in assignments.columns
            if "path" in column.casefold() or "directory" in column.casefold()
        ],
        errors="ignore",
    )
    assignment_manifest.to_csv(audit_dir / "outer_assignment_manifest.csv", index=False)
    (audit_dir / "comparison_config.json").write_text(
        json.dumps(
            {
                "comparison": "model_family_xgb_native_random_row",
                "model_families": args.models,
                "regressor": "xgb",
                "numeric_missing_strategy": NUMERIC_MISSING_STRATEGY,
                "split_strategy": SPLIT_STRATEGY,
                "outer_allocation_method": OUTER_ALLOCATION_METHOD,
                "inner_allocation_method": INNER_ALLOCATION_METHOD,
                "outer_repeats_requested_per_model": args.outer_repeats,
                "root_random_seed": args.random_seed,
                "model_random_seed": args.model_random_seed,
                "test_fraction": args.test_fraction,
                "validation_folds": args.validation_folds,
                "n_trials_requested": args.n_trials,
                "pairing": "none; model-family inputs are filtered before splitting",
                "selection_rule": (
                    "retain the first complete, distinct outer assignments in a "
                    "deterministic candidate-seed sequence for each model family"
                ),
                "runs_completed": counts["completed_run_count"],
                "distinct_outer_assignments": counts["distinct_assignment_count"],
                "automatic_shap": shap_manifest,
                "failures": failures,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    _write_readme(batch_dir)
    manifest["detail"]["testing_metrics_by_feature_availability"] = (
        "detail/testing_metrics_by_feature_availability.csv"
    )
    manifest["audit"].update(
        {
            "outer_assignment_manifest": "audit/outer_assignment_manifest.csv",
            "comparison_config": "audit/comparison_config.json",
        }
    )
    manifest["shap_robustness"] = shap_manifest
    return manifest


def _generate_figures(batch_dir: Path, models: list[str]) -> tuple[list[str], list[str]]:
    figure_paths: list[str] = []
    failures: list[str] = []
    try:
        figure_paths.extend(
            generate_aggregate_figure(
                batch_dir,
                plot_type="comparison",
                input_relative_path=Path("summary") / "performance_summary.csv",
                output_stem="model_family_performance_comparison",
                group_column="model",
                groups=tuple(models),
            )
        )
    except RuntimeError as error:
        failures.append(f"Performance figure generation failed: {error}")
    try:
        figure_paths.extend(
            generate_aggregate_figure(
                batch_dir,
                plot_type="predicted-vs-actual",
                input_relative_path=Path("detail") / "testing_predictions.parquet",
                output_stem="model_family_predicted_vs_actual",
                group_column="model",
                title="Native-XGBoost model-family comparison: predicted versus actual",
            )
        )
    except RuntimeError as error:
        failures.append(f"Prediction figure generation failed: {error}")
    return figure_paths, failures


def main() -> None:
    args = parse_args()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    batch_dir = args.output_root / f"model_family_xgb_native_random_row_{timestamp}"
    batch_dir.mkdir(parents=True, exist_ok=False)
    train_script = Path(__file__).with_name("train_logkd_model.py")
    preparation_root = batch_dir / ".batch_metadata" / "outer_assignments"
    model_runs_root(batch_dir).mkdir(parents=True, exist_ok=True)

    run_rows: list[dict[str, Any]] = []
    assignment_rows: list[dict[str, Any]] = []
    discarded_rows: list[dict[str, Any]] = []
    failures: list[str] = []
    max_candidates = args.outer_repeats * MAX_SEED_CANDIDATES_PER_REQUESTED_REPEAT
    candidate_seeds = _candidate_seeds(args.random_seed, max_candidates)
    write_live_run_summary(batch_dir, run_rows)

    for model in args.models:
        retained_hashes: set[str] = set()
        retained = 0
        for candidate_id, seed in enumerate(candidate_seeds, start=1):
            if retained >= args.outer_repeats:
                break
            prepare_dir = (
                preparation_root / model / f"candidate_{candidate_id:03d}_seed_{seed}"
            )
            print(
                f"Preparing outer split: model={model}, candidate={candidate_id}, seed={seed}",
                flush=True,
            )
            succeeded, message = paired_common._run(
                _prepare_command(
                    train_script,
                    args,
                    model=model,
                    seed=seed,
                    output_dir=prepare_dir,
                ),
                prepare_dir,
                console_verbosity=args.console_verbosity,
            )
            assignment_path = prepare_dir / "split_assignments.csv"
            if not succeeded or not assignment_path.exists():
                discarded_rows.append(
                    {
                        "model": model,
                        "candidate_id": candidate_id,
                        "random_seed": seed,
                        "status": "outer_preparation_failed",
                        "message": message or "Preparation did not write split_assignments.csv.",
                        "preparation_directory": str(prepare_dir),
                    }
                )
                continue
            try:
                testing_hash, testing_rows = paired_common._validate_outer_assignment(
                    assignment_path, expected_strategy=SPLIT_STRATEGY
                )
            except (OSError, ValueError, pd.errors.ParserError) as error:
                discarded_rows.append(
                    {
                        "model": model,
                        "candidate_id": candidate_id,
                        "random_seed": seed,
                        "status": "outer_assignment_validation_failed",
                        "message": str(error),
                        "preparation_directory": str(prepare_dir),
                    }
                )
                continue
            if testing_hash in retained_hashes:
                discarded_rows.append(
                    {
                        "model": model,
                        "candidate_id": candidate_id,
                        "random_seed": seed,
                        "status": "duplicate_outer_assignment",
                        "message": "Testing membership duplicates an already retained outer assignment.",
                        "preparation_directory": str(prepare_dir),
                    }
                )
                continue

            outer_repeat_id = retained + 1
            run_dir = model_run_directory(
                batch_dir,
                model,
                NUMERIC_MISSING_STRATEGY,
                outer_repeat_id=outer_repeat_id,
            )
            print(
                f"Training: model={model}, outer_repeat={outer_repeat_id}, seed={seed}",
                flush=True,
            )
            succeeded, message = paired_common._run(
                _training_command(
                    train_script,
                    args,
                    model=model,
                    seed=seed,
                    assignment_path=assignment_path,
                    output_dir=run_dir,
                ),
                run_dir,
                console_verbosity=args.console_verbosity,
            )
            row = _summary_row(
                run_dir,
                model=model,
                outer_repeat_id=outer_repeat_id,
                candidate_id=candidate_id,
                seed=seed,
                assignment_path=assignment_path,
                testing_membership_hash=testing_hash,
                testing_rows_from_assignment=testing_rows,
                status="completed" if succeeded else "training_failed",
                message=message,
            )
            write_live_run_summary(batch_dir, [*run_rows, row])
            if succeeded:
                run_rows.append(row)
                assignment_rows.append(
                    {
                        "model": model,
                        "split_strategy": SPLIT_STRATEGY,
                        "outer_repeat_id": outer_repeat_id,
                        "candidate_id": candidate_id,
                        "random_seed": seed,
                        "outer_allocation_method": OUTER_ALLOCATION_METHOD,
                        "outer_allocation_label": OUTER_ALLOCATION_METHOD,
                        "outer_assignment_path": str(assignment_path),
                        "outer_testing_membership_hash": testing_hash,
                        "testing_rows_from_assignment": testing_rows,
                        "inner_allocation_method": INNER_ALLOCATION_METHOD,
                        "coverage_used_for_allocation": False,
                    }
                )
                retained_hashes.add(testing_hash)
                retained += 1
            else:
                quarantine_incomplete_model_runs(
                    batch_dir,
                    [row],
                    candidate_id=candidate_id,
                    seed=seed,
                )
                run_rows.append(row)
                discarded_rows.append(
                    {
                        "model": model,
                        "candidate_id": candidate_id,
                        "random_seed": seed,
                        "status": "training_failed",
                        "message": message,
                        "preparation_directory": str(prepare_dir),
                    }
                )
        if retained < args.outer_repeats:
            failures.append(
                f"{model}: retained {retained} of {args.outer_repeats} complete repeats "
                f"after {max_candidates} candidate seeds"
            )

    runs = pd.DataFrame(run_rows)
    if runs.empty:
        runs = pd.DataFrame(
            columns=[
                "model",
                "split_strategy",
                "outer_repeat_id",
                "candidate_id",
                "random_seed",
                "outer_allocation_method",
                "inner_allocation_method",
                "numeric_missing_strategy",
                "outer_assignment_path",
                "outer_testing_membership_hash",
                "status",
                "run_directory",
            ]
        )
    shap_manifest = _automatic_shap_outputs(batch_dir, runs, args)
    failures.extend(shap_manifest.get("failures", []))
    manifest = _write_outputs(
        batch_dir,
        runs,
        pd.DataFrame(assignment_rows),
        discarded_rows,
        failures,
        args,
        shap_manifest,
    )
    figure_paths, figure_failures = _generate_figures(batch_dir, args.models)
    manifest["figures"] = figure_paths
    failures.extend(figure_failures)
    finalize_run_summary(batch_dir)
    paired_common._hide_internal_metadata_directory(batch_dir / ".batch_metadata")
    print(f"Completed model-family comparison: {batch_dir}", flush=True)
    if failures:
        print("WARNING: " + "; ".join(failures), file=sys.stderr)


if __name__ == "__main__":
    main()
