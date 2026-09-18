"""Run a paired, split-first comparison of selected logKd regressors.

Every candidate is evaluated on the same frozen outer assignments for each
split strategy.  Numeric inputs always use training-fold median imputation
with missingness indicators; coverage-aware allocation is intentionally
disabled.  The resulting differences therefore describe regressors, not a
change in held-out rows, allocation policy, or numeric missing-data policy.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import zlib
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from backend import logkd_config as cfg
from backend.logkd_plotting import generate_aggregate_figure
import run_logkd_missing_strategy_comparison as paired_common
from backend.logkd_exploratory_exports import (
    model_run_directory,
    model_runs_root,
    quarantine_incomplete_model_runs,
    summarize_scenarios,
    write_exploratory_outputs,
)
from backend.logkd_coverage import REFERENCE_RETENTION_SUMMARY_METRICS
from backend.logkd_progress import finalize_run_summary, write_live_run_summary
from backend.logkd_retention_diagnostics import (
    INNER_REFERENCE_RETENTION_SCORE_COLUMNS,
    summarize_inner_reference_retention,
)


PERFORMANCE_METRICS = paired_common.PERFORMANCE_METRICS
REFERENCE_RETENTION_METRICS = REFERENCE_RETENTION_SUMMARY_METRICS
# Keep one representative from each distinct modelling family: boosted trees,
# randomized trees, kernel regression, linear regression, and an interpretable
# additive model with a limited interaction budget.
SCREEN_REGRESSORS = ("xgb", "extra_trees", "rbf_svr", "ridge", "ebm")
NUMERIC_MISSING_STRATEGY = cfg.ALGORITHM_COMPARISON_NUMERIC_MISSING_STRATEGY
DEFAULT_OUTER_REPEATS = cfg.DEFAULT_EXPERIMENT_OUTER_REPEATS
DEFAULT_N_TRIALS = cfg.DEFAULT_EXPERIMENT_N_TRIALS
DEFAULT_EARLY_STOP_WARMUP = cfg.DEFAULT_EXPERIMENT_EARLY_STOP_WARMUP
DEFAULT_EARLY_STOP_PATIENCE = cfg.DEFAULT_EXPERIMENT_EARLY_STOP_PATIENCE
DEFAULT_GROUPED_OUTER_ALLOCATION_METHODS = cfg.DEFAULT_GROUPED_OUTER_ALLOCATION_METHODS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare selected logKd regressors on paired, repeated, unoptimized "
            "outer assignments. Numeric missing values always use training-fold "
            "median imputation plus indicators."
        )
    )
    parser.add_argument("--model", choices=sorted(cfg.MODEL_CATEGORIES), default="AC")
    parser.add_argument(
        "--split-strategies",
        choices=cfg.SPLIT_STRATEGIES,
        nargs="+",
        default=["random_row"],
        help=(
            "Modelling questions to retain as separate paired comparisons. "
            "Defaults to random_row; pass one or more strategies to override."
        ),
    )
    parser.add_argument(
        "--grouped-outer-allocation-methods",
        choices=cfg.DEFAULT_GROUPED_OUTER_ALLOCATION_METHODS,
        nargs="+",
        default=list(DEFAULT_GROUPED_OUTER_ALLOCATION_METHODS),
        help=(
            "Ordinary grouped outer allocation method(s). Coverage-aware allocation "
            "is not available in this runner."
        ),
    )
    parser.add_argument(
        "--regressors",
        choices=SCREEN_REGRESSORS,
        nargs="+",
        default=list(SCREEN_REGRESSORS),
        help="Focused candidate set, including Ridge as a diagnostic linear baseline.",
    )
    parser.add_argument(
        "--reference-regressor",
        choices=SCREEN_REGRESSORS,
        default="xgb",
        help="Incumbent against which every challenger is paired.",
    )
    parser.add_argument("--outer-repeats", type=int, default=DEFAULT_OUTER_REPEATS)
    parser.add_argument("--random-seed", type=int, default=cfg.DEFAULT_EXPERIMENT_RANDOM_SEED)
    parser.add_argument("--model-random-seed", type=int, default=cfg.DEFAULT_MODEL_RANDOM_SEED)
    parser.add_argument("--output-root", type=Path, default=cfg.DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--test-fraction", type=float, default=cfg.DEFAULT_TEST_FRACTION)
    parser.add_argument("--validation-folds", type=int, default=cfg.VALIDATION_FOLDS)
    parser.add_argument("--data-mode", choices=("baseline", "drop_unreliable"), default=cfg.DEFAULT_DATA_MODE)
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
    parser.add_argument(
        "--n-trials",
        type=int,
        default=DEFAULT_N_TRIALS,
        help="Total tuned parameter candidates per regressor and retained outer repeat.",
    )
    parser.add_argument("--early-stop-warmup", type=int, default=DEFAULT_EARLY_STOP_WARMUP)
    parser.add_argument("--early-stop-patience", type=int, default=DEFAULT_EARLY_STOP_PATIENCE)
    parser.add_argument("--early-stop-se-multiplier", type=float, default=cfg.DEFAULT_EXPERIMENT_EARLY_STOP_SE_MULTIPLIER)
    parser.add_argument("--early-stop-min-delta-floor", type=float, default=cfg.DEFAULT_EXPERIMENT_EARLY_STOP_MIN_DELTA_FLOOR)
    parser.add_argument("--n-jobs", type=int, default=cfg.DEFAULT_EXPERIMENT_N_JOBS)
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
    if args.n_trials < 0:
        parser.error("--n-trials must be non-negative.")
    if len(set(args.regressors)) != len(args.regressors):
        parser.error("--regressors must not contain duplicates.")
    if args.reference_regressor not in args.regressors:
        parser.error("--reference-regressor must appear in --regressors.")
    if len(set(args.grouped_outer_allocation_methods)) != len(args.grouped_outer_allocation_methods):
        parser.error("--grouped-outer-allocation-methods must not contain duplicates.")
    return args


def _require_optional_dependencies(regressors: list[str]) -> None:
    if "ebm" in regressors and importlib.util.find_spec("interpret") is None:
        raise RuntimeError(
            "InterpretML is required for --regressor ebm but is not installed in this "
            "Python environment. Install requirements-recommender.txt or run with a "
            "subset of --regressors that excludes ebm."
        )


def _candidate_seeds(root_seed: int, split_strategy: str, count: int) -> list[int]:
    key = zlib.crc32(f"algorithm_comparison:{split_strategy}".encode("utf-8"))
    return [
        int(value)
        for value in np.random.SeedSequence([root_seed, key]).generate_state(count)
    ]


def _fixed_training_arguments(args: argparse.Namespace) -> list[str]:
    """Arguments that must remain identical across every candidate model."""

    return [
        "--data-mode",
        args.data_mode,
        "--test-fraction",
        str(args.test_fraction),
        "--validation-folds",
        str(args.validation_folds),
        "--model-random-seed",
        str(args.model_random_seed),
        "--n-trials",
        str(args.n_trials),
        "--early-stop-warmup",
        str(args.early_stop_warmup),
        "--early-stop-patience",
        str(args.early_stop_patience),
        "--early-stop-se-multiplier",
        str(args.early_stop_se_multiplier),
        "--early-stop-min-delta-floor",
        str(args.early_stop_min_delta_floor),
        "--n-jobs",
        str(args.n_jobs),
        "--pfas-feature-family-policy",
        args.pfas_feature_family_policy,
        "--correlated-feature-handling",
        args.correlated_feature_handling,
    ]


def _prepare_command(
    train_script: Path,
    args: argparse.Namespace,
    *,
    strategy: str,
    outer_allocation_method: str,
    seed: int,
    output_dir: Path,
) -> list[str]:
    return [
        sys.executable,
        "-u",
        str(train_script),
        *args.training_args,
        "--model",
        args.model,
        "--split-strategy",
        strategy,
        "--outer-allocation-method",
        outer_allocation_method,
        "--random-seed",
        str(seed),
        "--prepare-split-only",
        "--output-dir",
        str(output_dir),
        *_fixed_training_arguments(args),
    ]


def _training_command(
    train_script: Path,
    args: argparse.Namespace,
    *,
    strategy: str,
    outer_allocation_method: str,
    seed: int,
    assignment_path: Path,
    regressor: str,
    output_dir: Path,
) -> list[str]:
    return [
        sys.executable,
        "-u",
        str(train_script),
        *args.training_args,
        "--model",
        args.model,
        "--split-strategy",
        strategy,
        "--outer-allocation-method",
        outer_allocation_method,
        "--inner-allocation-method",
        "random_row" if strategy == "random_row" else "random_group",
        "--random-seed",
        str(seed),
        "--split-assignment-path",
        str(assignment_path),
        "--regressor",
        regressor,
        "--numeric-missing-strategy",
        NUMERIC_MISSING_STRATEGY,
        "--output-dir",
        str(output_dir),
        *_fixed_training_arguments(args),
    ]


def _summary_row(
    run_dir: Path,
    *,
    split_strategy: str,
    outer_repeat_id: int,
    candidate_id: int,
    seed: int,
    outer_allocation_method: str,
    assignment_path: Path,
    testing_membership_hash: str,
    testing_rows_from_assignment: int,
    regressor: str,
    status: str,
    message: str = "",
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "split_strategy": split_strategy,
        "outer_repeat_id": outer_repeat_id,
        "candidate_id": candidate_id,
        "random_seed": seed,
        "outer_allocation_method": outer_allocation_method,
        "outer_allocation_label": paired_common._allocation_label(outer_allocation_method, split_strategy),
        "inner_allocation_method": "random_row" if split_strategy == "random_row" else "random_group",
        "outer_assignment_path": str(assignment_path),
        "outer_testing_membership_hash": testing_membership_hash,
        "testing_rows_from_assignment": testing_rows_from_assignment,
        "regressor": regressor,
        "numeric_missing_strategy": NUMERIC_MISSING_STRATEGY,
        "status": status,
        "run_directory": str(run_dir),
        "message": message,
    }
    if status != "completed":
        return row

    metrics_path = run_dir / "metrics_summary.csv"
    if metrics_path.exists():
        metrics = pd.read_csv(metrics_path)
        testing = metrics.loc[metrics["split"].eq("testing")]
        if not testing.empty:
            row.update(testing.iloc[0].drop(labels=["split"]).to_dict())
    row.update(summarize_inner_reference_retention(run_dir))

    config_path = run_dir / "run_config.json"
    if not config_path.exists():
        return row
    config = json.loads(config_path.read_text(encoding="utf-8"))
    selected = config.get("features", {}).get("selected")
    if isinstance(selected, list):
        row["selected_feature_count"] = len(selected)
    best_params = config.get("best_params")
    if isinstance(best_params, dict):
        row["best_params_json"] = json.dumps(best_params, sort_keys=True)
    tuning = config.get("hyperparameter_tuning")
    if isinstance(tuning, dict):
        row["hyperparameter_tuning_json"] = json.dumps(tuning, sort_keys=True)
    validation = config.get("split", {}).get("validation", {})
    for key in (
        *REFERENCE_RETENTION_METRICS,
        "testing_rows",
        "training_rows",
        "observed_test_fraction",
        "test_fraction_absolute_error",
        "evaluation_protocol",
        "coverage_objective_used_for_allocation",
    ):
        if key in validation:
            row[key] = validation[key]
    return row


def _completed_runs(runs: pd.DataFrame) -> pd.DataFrame:
    return runs.loc[runs.get("status", pd.Series(dtype=str)).eq("completed")].copy()


def _validate_frozen_outer_membership(runs: pd.DataFrame) -> None:
    """Ensure completed model variants retain their shared outer test set."""

    completed = _completed_runs(runs)
    grouping = (
        "split_strategy",
        "outer_allocation_method",
        "outer_repeat_id",
        "candidate_id",
        "random_seed",
    )
    for _, group in completed.groupby(list(grouping), dropna=False, sort=True):
        if group["regressor"].nunique() < 2:
            continue
        if group["outer_testing_membership_hash"].nunique() != 1:
            raise RuntimeError(
                "Completed regressor variants do not share a frozen outer testing membership."
            )


def _aggregate_performance(runs: pd.DataFrame) -> pd.DataFrame:
    grouping = ("split_strategy", "outer_allocation_method", "regressor")
    summary = summarize_scenarios(
        runs,
        group_columns=grouping,
        comparison_group_column="regressor",
        metrics=(
            *PERFORMANCE_METRICS,
            *INNER_REFERENCE_RETENTION_SCORE_COLUMNS,
            "selected_feature_count",
        ),
    )
    if summary.empty:
        return summary
    summary["outer_allocation_label"] = [
        paired_common._allocation_label(method, strategy)
        for strategy, method in zip(
            summary["split_strategy"],
            summary["outer_allocation_method"],
            strict=True,
        )
    ]
    summary["inner_allocation_method"] = summary["split_strategy"].map(
        lambda strategy: "random_row" if strategy == "random_row" else "random_group"
    )
    summary["numeric_missing_strategy"] = NUMERIC_MISSING_STRATEGY
    return summary


def _testing_feature_availability_detail(runs: pd.DataFrame) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    context = (
        "split_strategy",
        "outer_allocation_method",
        "outer_repeat_id",
        "candidate_id",
        "random_seed",
        "outer_testing_membership_hash",
        "regressor",
        "numeric_missing_strategy",
        "run_id",
    )
    for _, run in _completed_runs(runs).iterrows():
        detail = paired_common._read_csv(
            Path(str(run["run_directory"])) / "testing_metrics_by_feature_availability.csv"
        )
        if detail.empty:
            continue
        for column in context:
            detail[column] = run.get(column)
        frames.append(detail)
    return pd.concat(frames, ignore_index=True, sort=False) if frames else pd.DataFrame()


def _run_id(run: pd.Series) -> str:
    return (
        f"{run['split_strategy']}__outer_{int(run['outer_repeat_id']):03d}_"
        f"seed_{int(run['random_seed'])}__{run['outer_allocation_method']}__"
        f"{run['regressor']}"
    )


def _outer_assignment_id(run: pd.Series) -> str:
    digest = str(run.get("outer_testing_membership_hash", "unknown"))
    return f"{run['split_strategy']}__outer_{digest[:16]}"


def _write_readme(batch_dir: Path, reference_regressor: str) -> None:
    (batch_dir / "START_HERE.md").write_text(
        f"""# Paired algorithm comparison

## Question

Do the selected regressors improve held-out PFAS logKd predictions relative to
`{reference_regressor}` when the split design and numeric missing-data policy are fixed?

## Experimental design

For every split strategy and retained outer repeat, this runner creates one
frozen outer assignment and trains every configured regressor on it. Numeric
inputs use training-fold median imputation with a missingness indicator for
every model. Each model performs feature auditing and parameter tuning only
within the corresponding outer-training partition.

Only ordinary allocation is permitted: `random_row` for random-row splits and
the configured `random_group` method for grouped splits. Neither outer
allocation nor inner validation folds use coverage-aware
optimization. Coverage outputs written by the trainer are diagnostic only.

## Read these files first

- `audit/run_summary.csv`: one row per fitted model and outer repeat.
- `summary/performance_summary.csv`: held-out MAE, RMSE, R2, and Spearman,
  including low- and high-logKd subsets, summarized by mean and SD.
- `summary/coverage_summary.csv`: database contrast and split retention for
  each scientific feature block.
- `detail/testing_metrics_by_feature_availability.csv`: held-out metrics by
  selected-input availability.
- `audit/model_runs.csv`: where each fitted run is retained.

## Retained model runs

`model_runs/<split_strategy>/outer_<allocation_method>/<regressor>/outer_repeat_NNN/`
keeps every fitted run with its `run_config.json`, `split_assignments.csv`, and
per-run tables, so post-hoc analysis never requires retraining the batch. See
`model_runs/README.md`. Runs from candidates discarded before retention are kept
under `debug/incomplete_model_runs/`.

Use the aggregate figure or a visualization built from `audit/run_summary.csv` to
compare models with the configured reference `{reference_regressor}`.

Do not pool or directly compare metrics across different split strategies:
they intentionally represent different scientific generalization questions.
""",
        encoding="utf-8",
    )


def _write_outputs(
    batch_dir: Path,
    runs: pd.DataFrame,
    assignments: pd.DataFrame,
    discarded_rows: list[dict[str, Any]],
    failures: list[str],
    reference_regressor: str,
) -> dict[str, Any]:
    export_runs = runs.copy()
    if export_runs.empty:
        export_runs["run_id"] = pd.Series(dtype="string")
        export_runs["outer_assignment_id"] = pd.Series(dtype="string")
    else:
        export_runs["run_id"] = export_runs.apply(_run_id, axis=1)
        export_runs["outer_assignment_id"] = export_runs.apply(_outer_assignment_id, axis=1)

    availability_detail = _testing_feature_availability_detail(export_runs)
    manifest = write_exploratory_outputs(
        batch_dir,
        export_runs,
        run_context_columns=(
            "split_strategy",
            "outer_allocation_method",
            "inner_allocation_method",
            "outer_repeat_id",
            "candidate_id",
            "random_seed",
            "regressor",
            "numeric_missing_strategy",
        ),
        assignment_context_columns=(
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
        performance_group_columns=(
            "split_strategy",
            "outer_allocation_method",
            "regressor",
        ),
        coverage_group_columns=(
            "split_strategy",
            "outer_allocation_method",
        ),
    )
    counts = manifest.pop("_counts")
    summary_dir = batch_dir / "summary"
    detail_dir = batch_dir / "detail"
    audit_dir = batch_dir / "audit"

    for filename in (
        "aggregate_performance_and_time.csv",
        "outer_split_diagnostics.csv",
    ):
        (summary_dir / filename).unlink(missing_ok=True)
    _validate_frozen_outer_membership(export_runs)
    availability_detail.to_csv(detail_dir / "testing_metrics_by_feature_availability.csv", index=False)
    assignments.drop(
        columns=[
            column
            for column in assignments.columns
            if "path" in column.casefold() or "directory" in column.casefold()
        ],
        errors="ignore",
    ).to_csv(audit_dir / "outer_assignment_manifest.csv", index=False)

    config = {
        "comparison": "paired_regressor_comparison",
        "reference_regressor": reference_regressor,
        "numeric_missing_strategy": NUMERIC_MISSING_STRATEGY,
        "pairing": (
            "Within every split-strategy/outer-repeat/allocation cell, all regressors "
            "use the identical frozen outer assignment and random seed."
        ),
        "coverage_allocation": "disabled",
        "coverage_interpretation": "target-free split-support diagnostics only",
        "runs_completed": counts["completed_run_count"],
        "distinct_outer_assignments": counts["distinct_assignment_count"],
        "failures": failures,
    }
    (audit_dir / "comparison_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    _write_readme(batch_dir, reference_regressor)

    manifest["summary"].update(
        {
        }
    )
    manifest["detail"]["testing_metrics_by_feature_availability"] = "detail/testing_metrics_by_feature_availability.csv"
    manifest["audit"].update(
        {
            "outer_assignment_manifest": "audit/outer_assignment_manifest.csv",
            "comparison_config": "audit/comparison_config.json",
        }
    )
    return manifest


def _generate_figures(
    batch_dir: Path,
    args: argparse.Namespace,
) -> tuple[list[str], list[str]]:
    """Create one aggregate comparison figure for each split/allocation scenario."""

    figure_paths: list[str] = []
    failures: list[str] = []
    for strategy in args.split_strategies:
        outer_methods = paired_common._outer_allocation_methods_for_strategy(
            strategy,
            tuple(args.grouped_outer_allocation_methods),
        )
        for outer_method in outer_methods:
            stem = f"algorithm_comparison_{strategy}_{outer_method}"
            try:
                figure_paths.extend(
                    generate_aggregate_figure(
                        batch_dir,
                        plot_type="comparison",
                        input_relative_path=Path("summary") / "performance_summary.csv",
                        output_stem=stem,
                        group_column="regressor",
                        where={
                            "split_strategy": strategy,
                            "outer_allocation_method": outer_method,
                        },
                    )
                )
            except RuntimeError as error:
                failures.append(f"Figure generation ({strategy}/{outer_method}) failed: {error}")
            try:
                figure_paths.extend(
                    generate_aggregate_figure(
                        batch_dir,
                        plot_type="predicted-vs-actual",
                        input_relative_path=Path("detail") / "testing_predictions.parquet",
                        output_stem=f"algorithm_predicted_vs_actual_{strategy}_{outer_method}",
                        group_column="regressor",
                        where={
                            "split_strategy": strategy,
                            "outer_allocation_method": outer_method,
                        },
                        title=(
                            f"{strategy.replace('_', ' ').title()} outer split: "
                            "predicted versus actual across repeats"
                        ),
                    )
                )
            except RuntimeError as error:
                failures.append(
                    f"Prediction-figure generation ({strategy}/{outer_method}) failed: {error}"
                )
    return figure_paths, failures


def main() -> None:
    args = parse_args()
    _require_optional_dependencies(args.regressors)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    batch_dir = args.output_root / f"{args.model}_algorithm_comparison_{timestamp}"
    batch_dir.mkdir(parents=True, exist_ok=False)
    train_script = Path(__file__).with_name("train_logkd_model.py")
    metadata_dir = batch_dir / ".batch_metadata"
    preparation_root = metadata_dir / "outer_assignments"
    model_runs_root(batch_dir).mkdir(parents=True, exist_ok=True)

    run_rows: list[dict[str, Any]] = []
    assignment_rows: list[dict[str, Any]] = []
    discarded_rows: list[dict[str, Any]] = []
    failures: list[str] = []
    max_candidates = args.outer_repeats * paired_common.MAX_SEED_CANDIDATES_PER_REQUESTED_REPEAT
    write_live_run_summary(batch_dir, run_rows)

    for strategy in args.split_strategies:
        outer_methods = paired_common._outer_allocation_methods_for_strategy(
            strategy, tuple(args.grouped_outer_allocation_methods)
        )
        retained_assignments = {method: [] for method in outer_methods}
        retained_hashes = {method: set() for method in outer_methods}
        retained = 0
        for candidate_id, seed in enumerate(_candidate_seeds(args.random_seed, strategy, max_candidates), start=1):
            if retained >= args.outer_repeats:
                break
            candidate_root = preparation_root / strategy / f"candidate_{candidate_id:03d}_seed_{seed}"
            assignment_specs: dict[str, dict[str, Any]] = {}
            candidate_preparation_failed = False
            for outer_method in outer_methods:
                prepare_dir = candidate_root / outer_method
                command = _prepare_command(
                    train_script,
                    args,
                    strategy=strategy,
                    outer_allocation_method=outer_method,
                    seed=seed,
                    output_dir=prepare_dir,
                )
                print(
                    f"Preparing paired outer scenario: strategy={strategy}, candidate={candidate_id}, "
                    f"outer={outer_method}, seed={seed}",
                    flush=True,
                )
                succeeded, message = paired_common._run(command, prepare_dir, console_verbosity=args.console_verbosity)
                assignment_path = prepare_dir / "split_assignments.csv"
                if not succeeded or not assignment_path.exists():
                    discarded_rows.append(
                        {
                            "split_strategy": strategy,
                            "candidate_id": candidate_id,
                            "random_seed": seed,
                            "outer_allocation_method": outer_method,
                            "status": "outer_preparation_failed",
                            "message": message or "Preparation did not write split_assignments.csv.",
                            "preparation_directory": str(prepare_dir),
                        }
                    )
                    candidate_preparation_failed = True
                    continue
                try:
                    membership_hash, testing_rows = paired_common._validate_outer_assignment(
                        assignment_path, expected_strategy=strategy
                    )
                except (OSError, ValueError, pd.errors.ParserError) as error:
                    discarded_rows.append(
                        {
                            "split_strategy": strategy,
                            "candidate_id": candidate_id,
                            "random_seed": seed,
                            "outer_allocation_method": outer_method,
                            "status": "outer_assignment_validation_failed",
                            "message": str(error),
                            "preparation_directory": str(prepare_dir),
                        }
                    )
                    candidate_preparation_failed = True
                    continue
                assignment_specs[outer_method] = {
                    "assignment_path": assignment_path,
                    "testing_membership_hash": membership_hash,
                    "testing_rows_from_assignment": testing_rows,
                }
            if candidate_preparation_failed or len(assignment_specs) != len(outer_methods):
                continue
            if any(
                spec["testing_membership_hash"] in retained_hashes[method]
                for method, spec in assignment_specs.items()
            ):
                discarded_rows.append(
                    {
                        "split_strategy": strategy,
                        "candidate_id": candidate_id,
                        "random_seed": seed,
                        "status": "duplicate_outer_assignment",
                        "message": "Testing membership repeats an already retained outer assignment.",
                        "preparation_directory": str(candidate_root),
                    }
                )
                continue

            outer_repeat_id = retained + 1
            candidate_run_rows: list[dict[str, Any]] = []
            all_cells_completed = True
            for outer_method, spec in assignment_specs.items():
                for regressor in args.regressors:
                    run_dir = model_run_directory(
                        batch_dir,
                        strategy,
                        f"outer_{outer_method}",
                        regressor,
                        outer_repeat_id=outer_repeat_id,
                    )
                    command = _training_command(
                        train_script,
                        args,
                        strategy=strategy,
                        outer_allocation_method=outer_method,
                        seed=seed,
                        assignment_path=Path(spec["assignment_path"]),
                        regressor=regressor,
                        output_dir=run_dir,
                    )
                    print(
                        f"Training paired scenario: strategy={strategy}, outer_repeat={outer_repeat_id}, "
                        f"outer={outer_method}, regressor={regressor}, seed={seed}",
                        flush=True,
                    )
                    succeeded, message = paired_common._run(command, run_dir, console_verbosity=args.console_verbosity)
                    all_cells_completed = all_cells_completed and succeeded
                    candidate_run_rows.append(
                        _summary_row(
                            run_dir,
                            split_strategy=strategy,
                            outer_repeat_id=outer_repeat_id,
                            candidate_id=candidate_id,
                            seed=seed,
                            outer_allocation_method=outer_method,
                            assignment_path=Path(spec["assignment_path"]),
                            testing_membership_hash=str(spec["testing_membership_hash"]),
                            testing_rows_from_assignment=int(spec["testing_rows_from_assignment"]),
                            regressor=regressor,
                            status="completed" if succeeded else "training_failed",
                            message=message,
                        )
                    )
                    write_live_run_summary(batch_dir, [*run_rows, *candidate_run_rows])
            if all_cells_completed:
                run_rows.extend(candidate_run_rows)
                for outer_method, spec in assignment_specs.items():
                    retained_assignments[outer_method].append(Path(spec["assignment_path"]))
                    retained_hashes[outer_method].add(str(spec["testing_membership_hash"]))
                    assignment_rows.append(
                        {
                            "split_strategy": strategy,
                            "outer_repeat_id": outer_repeat_id,
                            "candidate_id": candidate_id,
                            "random_seed": seed,
                            "outer_allocation_method": outer_method,
                            "outer_allocation_label": paired_common._allocation_label(outer_method, strategy),
                            "outer_assignment_path": str(spec["assignment_path"]),
                            "outer_testing_membership_hash": str(spec["testing_membership_hash"]),
                            "testing_rows_from_assignment": int(spec["testing_rows_from_assignment"]),
                            "inner_allocation_method": "random_row" if strategy == "random_row" else "random_group",
                            "coverage_used_for_allocation": False,
                            "numeric_missing_strategy": NUMERIC_MISSING_STRATEGY,
                            "paired_regressors": ";".join(args.regressors),
                        }
                    )
                retained += 1
            else:
                # The next candidate reuses this outer repeat number, so move
                # the discarded candidate's fitted runs out of model_runs/.
                quarantine_incomplete_model_runs(
                    batch_dir, candidate_run_rows, candidate_id=candidate_id, seed=seed
                )
                run_rows.extend(row for row in candidate_run_rows if row["status"] != "completed")
                discarded_rows.append(
                    {
                        "split_strategy": strategy,
                        "candidate_id": candidate_id,
                        "random_seed": seed,
                        "status": "incomplete_regressor_set",
                        "message": "Outer assignment was excluded because at least one regressor training run failed.",
                        "preparation_directory": str(candidate_root),
                    }
                )
        if retained < args.outer_repeats:
            failures.append(
                f"{strategy}: retained {retained} of {args.outer_repeats} complete paired repeats "
                f"after {max_candidates} candidate seeds"
            )

    runs = pd.DataFrame(run_rows)
    if runs.empty:
        runs = pd.DataFrame(
            columns=[
                "split_strategy",
                "outer_repeat_id",
                "candidate_id",
                "random_seed",
                "outer_allocation_method",
                "regressor",
                "outer_assignment_path",
                "outer_testing_membership_hash",
                "status",
                "run_directory",
            ]
        )
    assignments = pd.DataFrame(assignment_rows)
    evidence_manifest = _write_outputs(
        batch_dir,
        runs,
        assignments,
        discarded_rows,
        failures,
        args.reference_regressor,
    )
    figure_paths, figure_failures = _generate_figures(batch_dir, args)
    evidence_manifest["figures"] = figure_paths
    failures.extend(figure_failures)
    finalize_run_summary(batch_dir)
    metadata_dir.mkdir(parents=True, exist_ok=True)
    (metadata_dir / "comparison_config.json").write_text(
        json.dumps(
            {
                "model": args.model,
                "regressors": args.regressors,
                "reference_regressor": args.reference_regressor,
                "numeric_missing_strategy": NUMERIC_MISSING_STRATEGY,
                "split_strategies": args.split_strategies,
                "outer_allocation_methods_by_split_strategy": {
                    strategy: list(
                        paired_common._outer_allocation_methods_for_strategy(
                            strategy, tuple(args.grouped_outer_allocation_methods)
                        )
                    )
                    for strategy in args.split_strategies
                },
                "inner_allocation_methods_by_split_strategy": {
                    strategy: "random_row" if strategy == "random_row" else "random_group"
                    for strategy in args.split_strategies
                },
                "coverage_allocation": "disabled",
                "outer_repeats_requested": args.outer_repeats,
                "root_random_seed": args.random_seed,
                "model_random_seed": args.model_random_seed,
                "test_fraction": args.test_fraction,
                "validation_folds": args.validation_folds,
                "n_trials_requested": args.n_trials,
                "console_verbosity": args.console_verbosity,
                "selection_rule": (
                    "retain the first complete, distinct paired scenario set in a deterministic "
                    "candidate-seed sequence; no target, prediction, performance, or coverage "
                    "metric selects assignments"
                ),
                "evidence_outputs": evidence_manifest,
                "failures": failures,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    paired_common._hide_internal_metadata_directory(metadata_dir)
    print(f"Completed algorithm comparison: {batch_dir}", flush=True)
    if failures:
        print("WARNING: " + "; ".join(failures), file=sys.stderr)


if __name__ == "__main__":
    main()
