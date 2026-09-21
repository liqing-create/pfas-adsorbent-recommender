"""Compare feature-screening policies on paired repeated random-row splits.

Every retained repeat begins with one frozen random-row outer assignment. Each
threshold policy then performs its own training- and inner-fold feature audit on
that same outer-training partition, and all policies are tested on the same
held-out rows.
"""

from __future__ import annotations

import argparse
import json
import sys
import zlib
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

from backend import logkd_config as cfg
from backend.logkd_exploratory_exports import (
    model_run_directory,
    model_runs_root,
    quarantine_incomplete_model_runs,
    summarize_concise_coverage,
    summarize_concise_performance,
    write_model_runs_readme,
    write_testing_prediction_detail,
)
from backend.logkd_plotting import generate_aggregate_figure
from backend.logkd_coverage import REFERENCE_RETENTION_SUMMARY_METRICS
import run_logkd_missing_strategy_comparison as paired_common
from backend.logkd_retention_diagnostics import (
    INNER_REFERENCE_RETENTION_SCORE_COLUMNS,
    summarize_inner_reference_retention,
)
from backend.logkd_progress import finalize_run_summary, write_live_run_summary


PERFORMANCE_METRICS = ("mae", "rmse", "r2", "spearman")
REFERENCE_RETENTION_METRICS = REFERENCE_RETENTION_SUMMARY_METRICS
DEFAULT_OUTER_REPEATS = cfg.DEFAULT_EXPERIMENT_OUTER_REPEATS
DEFAULT_N_TRIALS = cfg.DEFAULT_EXPERIMENT_N_TRIALS
MAX_SEED_CANDIDATES_PER_REQUESTED_REPEAT = 20

# Edit this policy grid for a different screening study. Every named policy
# inherits omitted threshold values from BASELINE_THRESHOLDS.
BASELINE_THRESHOLDS: dict[str, float | int] = {
    "min_nonmissing_rows": 50,
    "min_reporting_study_fraction": 0.20,
    "min_reporting_adsorbent_fraction": 0.15,
    "min_categorical_levels": 2,
    "min_categorical_level_rows": 20,
    "min_categorical_level_entities": 3,
    "min_numeric_distinct_values": 3,
    "min_numeric_iqr_bins": 1.0,
    "pfas_missing_rate_threshold": 0.20,
    "pfas_dominant_fraction_threshold": 0.95,
    "correlation_threshold": 0.90,
    "min_pairwise_observations": 20,
}
DEFAULT_CONFIGURATION_PATCHES: dict[str, dict[str, Any]] = {
    "baseline": {"feature_screening_mode": "none"},
    "light": {
        "feature_screening_mode": "standard",
    },
}
THRESHOLD_ARGUMENTS: tuple[tuple[str, str], ...] = (
    ("min_nonmissing_rows", "--min-nonmissing-rows"),
    ("min_reporting_study_fraction", "--min-reporting-study-fraction"),
    ("min_reporting_adsorbent_fraction", "--min-reporting-adsorbent-fraction"),
    ("min_categorical_levels", "--min-categorical-levels"),
    ("min_categorical_level_rows", "--min-categorical-level-rows"),
    ("min_categorical_level_entities", "--min-categorical-level-entities"),
    ("min_numeric_distinct_values", "--min-numeric-distinct-values"),
    ("min_numeric_iqr_bins", "--min-numeric-iqr-bins"),
    ("pfas_missing_rate_threshold", "--pfas-missing-rate-threshold"),
    ("pfas_dominant_fraction_threshold", "--pfas-dominant-fraction-threshold"),
    ("correlation_threshold", "--correlation-threshold"),
    ("min_pairwise_observations", "--min-pairwise-observations"),
)
THRESHOLD_KEYS = frozenset(name for name, _ in THRESHOLD_ARGUMENTS)
FEATURE_SCREENING_MODE_KEY = "feature_screening_mode"
FEATURE_SCREENING_MODES = frozenset(("standard", "none"))
EXCLUDED_FEATURE_BUCKETS_KEY = "excluded_feature_buckets"
CONFIGURATION_KEYS = THRESHOLD_KEYS | {FEATURE_SCREENING_MODE_KEY, EXCLUDED_FEATURE_BUCKETS_KEY}
SCREENING_THRESHOLD_KEYS: tuple[str, ...] = (
    "min_nonmissing_rows",
    "min_reporting_study_fraction",
    "min_reporting_adsorbent_fraction",
    "min_categorical_levels",
    "min_categorical_level_rows",
    "min_categorical_level_entities",
    "min_numeric_distinct_values",
    "min_numeric_iqr_bins",
    "pfas_missing_rate_threshold",
    "pfas_dominant_fraction_threshold",
)


def parse_args(description: str) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=description
    )
    parser.add_argument("--model", choices=sorted(cfg.MODEL_CATEGORIES), default="AC")
    parser.add_argument("--outer-repeats", type=int, default=DEFAULT_OUTER_REPEATS)
    parser.add_argument("--random-seed", type=int, default=cfg.DEFAULT_EXPERIMENT_RANDOM_SEED)
    parser.add_argument("--model-random-seed", type=int, default=cfg.DEFAULT_MODEL_RANDOM_SEED)
    parser.add_argument("--output-root", type=Path, default=cfg.DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--test-fraction", type=float, default=cfg.DEFAULT_TEST_FRACTION)
    parser.add_argument("--validation-folds", type=int, default=cfg.VALIDATION_FOLDS)
    parser.add_argument("--data-mode", choices=("baseline", "drop_unreliable"), default=cfg.DEFAULT_DATA_MODE)
    parser.add_argument(
        "--n-trials",
        type=int,
        default=DEFAULT_N_TRIALS,
        help="Optuna trials per policy/repeat; default 0 uses fixed XGBoost parameters for screening.",
    )
    parser.add_argument("--xgb-search-space", choices=("compact", "full"), default=cfg.DEFAULT_EXPERIMENT_XGB_SEARCH_SPACE)
    parser.add_argument("--early-stop-warmup", type=int, default=cfg.DEFAULT_EXPERIMENT_EARLY_STOP_WARMUP)
    parser.add_argument("--early-stop-patience", type=int, default=cfg.DEFAULT_EXPERIMENT_EARLY_STOP_PATIENCE)
    parser.add_argument("--early-stop-se-multiplier", type=float, default=cfg.DEFAULT_EXPERIMENT_EARLY_STOP_SE_MULTIPLIER)
    parser.add_argument("--early-stop-min-delta-floor", type=float, default=cfg.DEFAULT_EXPERIMENT_EARLY_STOP_MIN_DELTA_FLOOR)
    parser.add_argument("--n-jobs", type=int, default=cfg.DEFAULT_EXPERIMENT_N_JOBS)
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
        "--configuration-file",
        type=Path,
        help="JSON object of named policy overlays; omitted threshold keys inherit the baseline above.",
    )
    parser.add_argument(
        "--configurations",
        nargs="+",
        help="Named configurations to run; default runs every configuration in the selected set.",
    )
    parser.add_argument("--reference-configuration")
    parser.add_argument(
        "--reuse-outer-assignments-from",
        type=Path,
        help="Prior threshold-comparison output directory whose retained assignments should be reused.",
    )
    parser.add_argument("--console-verbosity", choices=("quiet", "normal"), default=cfg.DEFAULT_EXPERIMENT_CONSOLE_VERBOSITY)
    parser.add_argument(
        "--training-args",
        nargs=argparse.REMAINDER,
        default=[],
        help="Arguments forwarded unchanged to train_logkd_model.py. Place this argument last.",
    )
    args = parser.parse_args()
    if args.outer_repeats < 1:
        parser.error("--outer-repeats must be at least one.")
    if not 0 < args.test_fraction < 1:
        parser.error("--test-fraction must be between zero and one.")
    if args.validation_folds < 2:
        parser.error("--validation-folds must be at least two.")
    if args.n_trials < 0:
        parser.error("--n-trials must be nonnegative.")
    if args.early_stop_warmup < 0 or args.early_stop_patience < 0:
        parser.error("Early-stop warmup and patience must be nonnegative.")
    return args


def resolve_configurations(
    args: argparse.Namespace,
    *,
    default_configurations: dict[str, dict[str, Any]],
    default_reference_configuration: str,
) -> dict[str, dict[str, Any]]:
    raw: Any = default_configurations
    if args.configuration_file is not None:
        try:
            raw = json.loads(args.configuration_file.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError(f"Could not read --configuration-file: {error}") from error
    if not isinstance(raw, dict) or not raw:
        raise ValueError("Configuration data must be a nonempty JSON object.")

    requested = list(args.configurations) if args.configurations else list(raw)
    if len(set(requested)) != len(requested):
        raise ValueError("--configurations cannot contain duplicates.")
    reference_configuration = args.reference_configuration or default_reference_configuration
    if reference_configuration not in requested:
        raise ValueError("--reference-configuration must be included in --configurations.")
    args.reference_configuration = reference_configuration

    resolved: dict[str, dict[str, Any]] = {}
    for name in requested:
        if name not in raw:
            raise ValueError(f"Unknown configuration {name!r}.")
        patch = raw[name]
        if not isinstance(patch, dict):
            raise ValueError(f"Configuration {name!r} must be a JSON object.")
        unknown = sorted(set(patch).difference(CONFIGURATION_KEYS))
        if unknown:
            raise ValueError(f"Configuration {name!r} has unknown keys: {unknown}")
        numeric_values = [
            value for key, value in patch.items() if key in THRESHOLD_KEYS
        ]
        if any(isinstance(value, bool) or not isinstance(value, (int, float)) for value in numeric_values):
            raise ValueError(f"Configuration {name!r} contains a nonnumeric threshold.")
        screening_mode = patch.get(FEATURE_SCREENING_MODE_KEY, "standard")
        if screening_mode not in FEATURE_SCREENING_MODES:
            raise ValueError(
                f"Configuration {name!r} has unsupported feature_screening_mode "
                f"{screening_mode!r}; expected one of {sorted(FEATURE_SCREENING_MODES)}."
            )
        excluded_buckets = patch.get(EXCLUDED_FEATURE_BUCKETS_KEY, [])
        if not isinstance(excluded_buckets, list) or any(
            not isinstance(bucket, str) for bucket in excluded_buckets
        ):
            raise ValueError(
                f"Configuration {name!r} must define excluded_feature_buckets as a list of bucket names."
            )
        if len(set(excluded_buckets)) != len(excluded_buckets):
            raise ValueError(f"Configuration {name!r} repeats an excluded feature bucket.")
        unknown_buckets = sorted(set(excluded_buckets).difference(cfg.FEATURE_BUCKETS))
        if unknown_buckets:
            raise ValueError(
                f"Configuration {name!r} contains unknown feature buckets: {unknown_buckets}."
            )
        resolved[name] = {
            **BASELINE_THRESHOLDS,
            FEATURE_SCREENING_MODE_KEY: "standard",
            EXCLUDED_FEATURE_BUCKETS_KEY: [],
            **patch,
        }
    return resolved


def candidate_seeds(root_seed: int, count: int) -> list[int]:
    key = zlib.crc32(b"logkd_feature_threshold_comparison")
    return [
        int(seed)
        for seed in np.random.SeedSequence([root_seed, key]).generate_state(count)
    ]


def shared_arguments(args: argparse.Namespace) -> list[str]:
    return [
        "--data-mode", str(args.data_mode),
        "--test-fraction", str(args.test_fraction),
        "--validation-folds", str(args.validation_folds),
        "--model-random-seed", str(args.model_random_seed),
        "--early-stop-warmup", str(args.early_stop_warmup),
        "--early-stop-patience", str(args.early_stop_patience),
        "--early-stop-se-multiplier", str(args.early_stop_se_multiplier),
        "--early-stop-min-delta-floor", str(args.early_stop_min_delta_floor),
        "--n-jobs", str(args.n_jobs),
        "--pfas-feature-family-policy", args.pfas_feature_family_policy,
        "--correlated-feature-handling", args.correlated_feature_handling,
    ]


def policy_arguments(policy: dict[str, Any]) -> list[str]:
    arguments = [
        value
        for name, argument in THRESHOLD_ARGUMENTS
        for value in (argument, str(policy[name]))
    ] + ["--feature-screening-mode", str(policy[FEATURE_SCREENING_MODE_KEY])]
    excluded_buckets = policy.get(EXCLUDED_FEATURE_BUCKETS_KEY, [])
    if excluded_buckets:
        arguments.extend(["--exclude-feature-buckets", *[str(bucket) for bucket in excluded_buckets]])
    return arguments


def prepare_command(
    train_script: Path,
    args: argparse.Namespace,
    *,
    seed: int,
    output_dir: Path,
    policy: dict[str, Any],
) -> list[str]:
    return [
        sys.executable, "-u", str(train_script), *args.training_args,
        "--model", args.model,
        "--split-strategy", "random_row",
        "--outer-allocation-method", "random_row",
        "--random-seed", str(seed),
        "--prepare-split-only",
        "--output-dir", str(output_dir),
        *shared_arguments(args),
        *policy_arguments(policy),
    ]


def training_command(
    train_script: Path,
    args: argparse.Namespace,
    *,
    seed: int,
    assignment_path: Path,
    output_dir: Path,
    policy: dict[str, Any],
) -> list[str]:
    return [
        sys.executable, "-u", str(train_script), *args.training_args,
        "--model", args.model,
        "--split-strategy", "random_row",
        "--outer-allocation-method", "random_row",
        "--inner-allocation-method", "random_row",
        "--random-seed", str(seed),
        "--split-assignment-path", str(assignment_path),
        "--regressor", "xgb",
        "--n-trials", str(args.n_trials),
        "--xgb-search-space", args.xgb_search_space,
        "--output-dir", str(output_dir),
        *shared_arguments(args),
        *policy_arguments(policy),
    ]


def feature_manifest_summary(run_dir: Path) -> tuple[dict[str, Any], pd.DataFrame]:
    path = run_dir / "feature_manifest.csv"
    if not path.exists():
        return {}, pd.DataFrame()
    manifest = pd.read_csv(path)
    if "selected" not in manifest:
        return {}, manifest
    selected = manifest.loc[manifest["selected"].fillna(False).astype(bool)]
    summary: dict[str, Any] = {"selected_feature_count": int(len(selected))}
    if "kind" in selected:
        kinds = selected["kind"].fillna("").value_counts()
        summary["selected_numeric_feature_count"] = int(kinds.get("numeric", 0))
        summary["selected_categorical_feature_count"] = int(kinds.get("categorical", 0))
    if "bucket" in selected:
        for bucket, count in selected["bucket"].fillna("unknown").value_counts().items():
            summary[f"selected_{str(bucket).casefold().replace(' ', '_')}_count"] = int(count)
    return summary, manifest


def summary_row(
    run_dir: Path,
    *,
    configuration: str,
    policy: dict[str, Any],
    repeat_id: int,
    candidate_id: int,
    seed: int,
    assignment_path: Path,
    membership_hash: str,
    testing_rows: int,
    status: str,
    message: str = "",
) -> tuple[dict[str, Any], pd.DataFrame]:
    row: dict[str, Any] = {
        "threshold_configuration": configuration,
        "feature_screening_mode": policy[FEATURE_SCREENING_MODE_KEY],
        "excluded_feature_buckets": "; ".join(policy.get(EXCLUDED_FEATURE_BUCKETS_KEY, [])),
        **{f"threshold_{name}": policy[name] for name in THRESHOLD_KEYS},
        "thresholds_json": json.dumps(policy, sort_keys=True),
        "outer_repeat_id": repeat_id,
        "candidate_id": candidate_id,
        "random_seed": seed,
        "split_strategy": "random_row",
        "outer_allocation_method": "random_row",
        "inner_allocation_method": "random_row",
        "outer_assignment_path": str(assignment_path),
        "outer_testing_membership_hash": membership_hash,
        "testing_rows_from_assignment": testing_rows,
        "status": status,
        "run_directory": str(run_dir),
        "message": message,
    }
    if status != "completed":
        return row, pd.DataFrame()

    row.update(summarize_inner_reference_retention(run_dir))

    metrics_path = run_dir / "metrics_summary.csv"
    if metrics_path.exists():
        metrics = pd.read_csv(metrics_path)
        testing = metrics.loc[metrics["split"].eq("testing")]
        if not testing.empty:
            row.update(testing.iloc[0].drop(labels=["split"]).to_dict())
    config_path = run_dir / "run_config.json"
    if config_path.exists():
        config = json.loads(config_path.read_text(encoding="utf-8"))
        row["best_params_json"] = json.dumps(config.get("best_params", {}), sort_keys=True)
        row["hyperparameter_tuning_json"] = json.dumps(
            config.get("hyperparameter_tuning", {}), sort_keys=True
        )
        validation = config.get("split", {}).get("validation", {})
        for key in REFERENCE_RETENTION_METRICS:
            if key in validation:
                row[key] = validation[key]
    feature_summary, manifest = feature_manifest_summary(run_dir)
    row.update(feature_summary)
    return row, manifest


def _validate_frozen_outer_membership(runs: pd.DataFrame) -> None:
    """Ensure completed configurations reuse the same frozen outer test set."""

    if runs.empty or "status" not in runs:
        return
    completed = runs.loc[runs["status"].eq("completed")]
    context = ("outer_repeat_id", "candidate_id", "random_seed")
    for _, group in completed.groupby(list(context), dropna=False, sort=True):
        if group["threshold_configuration"].nunique() < 2:
            continue
        if group["outer_testing_membership_hash"].nunique() != 1:
            raise RuntimeError(
                "Completed threshold configurations do not share a frozen outer testing membership."
            )


def performance_summary(runs: pd.DataFrame) -> pd.DataFrame:
    """Aggregate completed runs into one row per threshold scenario."""

    summary = summarize_concise_performance(
        runs,
        group_columns=("threshold_configuration",),
    )
    if summary.empty:
        return summary
    summary["split_strategy"] = "random_row"
    summary["outer_allocation_method"] = "random_row"
    summary["inner_allocation_method"] = "random_row"
    return summary


def coverage_summary(runs: pd.DataFrame) -> pd.DataFrame:
    """Report one split-level coverage result because policies share outer assignments."""

    return summarize_concise_coverage(
        runs,
        group_columns=("split_strategy", "outer_allocation_method"),
    )


def feature_stability(feature_rows: pd.DataFrame) -> pd.DataFrame:
    if feature_rows.empty or "selected" not in feature_rows:
        return pd.DataFrame()
    values = feature_rows.copy()
    values["selected"] = values["selected"].fillna(False).astype(bool)
    rows: list[dict[str, Any]] = []
    for (configuration, feature), group in values.groupby(
        ["threshold_configuration", "feature"], dropna=False
    ):
        repeat_count = int(group["outer_repeat_id"].nunique())
        selected = group.loc[group["selected"]]
        rows.append({
            "threshold_configuration": configuration,
            "feature": feature,
            "outer_repeats_observed": repeat_count,
            "outer_repeats_selected": int(selected["outer_repeat_id"].nunique()),
            "selection_frequency": float(selected["outer_repeat_id"].nunique() / max(repeat_count, 1)),
            "bucket": selected["bucket"].iloc[0] if not selected.empty and "bucket" in selected else "",
            "kind": selected["kind"].iloc[0] if not selected.empty and "kind" in selected else "",
        })
    return pd.DataFrame(rows).sort_values(
        ["threshold_configuration", "selection_frequency", "feature"],
        ascending=[True, False, True],
        kind="stable",
    )


def screening_threshold_summary(
    configurations: dict[str, dict[str, Any]],
    *,
    include_excluded_feature_buckets: bool,
) -> pd.DataFrame:
    """Report only thresholds that are active for feature screening.

    Correlation pruning is shared across policies and is intentionally omitted
    here. Its exact settings remain available in each run's configuration.
    """
    rows: list[dict[str, Any]] = []
    for name, policy in configurations.items():
        screening_applied = policy[FEATURE_SCREENING_MODE_KEY] == "standard"
        row = {
            "threshold_configuration": name,
            "feature_screening_mode": policy[FEATURE_SCREENING_MODE_KEY],
            "screening_thresholds_applied": screening_applied,
            **{
                key: policy[key] if screening_applied else None
                for key in SCREENING_THRESHOLD_KEYS
            },
        }
        if include_excluded_feature_buckets:
            row["excluded_feature_buckets"] = "; ".join(
                policy.get(EXCLUDED_FEATURE_BUCKETS_KEY, [])
            )
        rows.append(row)
    return pd.DataFrame(rows)


def write_outputs(
    batch_dir: Path,
    *,
    runs: pd.DataFrame,
    assignments: pd.DataFrame,
    discarded: pd.DataFrame,
    feature_rows: pd.DataFrame,
    configurations: dict[str, dict[str, Any]],
    args: argparse.Namespace,
    failures: list[str],
    comparison_name: str,
    configuration_definition_filename: str,
    include_excluded_feature_buckets: bool,
    output_description: str,
    automatic_plot_group_column: str | None,
    automatic_plot_groups: tuple[str, ...],
) -> None:
    summary_dir, audit_dir, detail_dir = (
        batch_dir / "summary",
        batch_dir / "audit",
        batch_dir / "detail",
    )
    for directory in (summary_dir, audit_dir, detail_dir):
        directory.mkdir(parents=True, exist_ok=True)

    runs.to_csv(audit_dir / "run_summary.csv", index=False)
    write_model_runs_readme(batch_dir)
    _validate_frozen_outer_membership(runs)
    performance = performance_summary(runs)
    performance.to_csv(summary_dir / "performance_summary.csv", index=False)
    coverage_summary(runs).to_csv(summary_dir / "coverage_summary.csv", index=False)
    figure_paths: list[str] = []
    prediction_path: Path | None = None
    try:
        prediction_path = write_testing_prediction_detail(
            batch_dir,
            runs,
            run_context_columns=(
                "threshold_configuration",
                "feature_screening_mode",
                "outer_repeat_id",
                "candidate_id",
                "random_seed",
                "split_strategy",
                "outer_allocation_method",
                "inner_allocation_method",
            ),
        )
    except (RuntimeError, ValueError) as error:
        failures.append(f"Prediction export failed: {error}")
        print(f"WARNING: {error}", file=sys.stderr)
    try:
        figure_paths = generate_aggregate_figure(
            batch_dir,
        plot_type="comparison",
        input_relative_path=Path("summary") / "performance_summary.csv",
        output_stem="configuration_performance_comparison",
        group_column=automatic_plot_group_column,
        groups=automatic_plot_groups,
        )
    except RuntimeError as error:
        failures.append(f"Figure generation failed: {error}")
        print(f"WARNING: {error}", file=sys.stderr)
    if prediction_path is not None:
        try:
            figure_paths.extend(generate_aggregate_figure(
                batch_dir,
                plot_type="predicted-vs-actual",
                input_relative_path=Path("detail") / "testing_predictions.parquet",
                output_stem="configuration_predicted_vs_actual",
                group_column="threshold_configuration",
                title="Random-row outer split: predicted versus actual across repeats",
            ))
        except RuntimeError as error:
            failures.append(f"Prediction figure generation failed: {error}")
            print(f"WARNING: {error}", file=sys.stderr)
    configuration_definitions = screening_threshold_summary(
        configurations,
        include_excluded_feature_buckets=include_excluded_feature_buckets,
    )
    configuration_definitions.to_csv(
        summary_dir / configuration_definition_filename, index=False
    )
    feature_rows.to_csv(detail_dir / "feature_manifests_by_run.csv", index=False)
    feature_stability(feature_rows).to_csv(
        detail_dir / "feature_selection_stability.csv", index=False
    )
    assignments.to_csv(audit_dir / "outer_assignment_manifest.csv", index=False)
    discarded.to_csv(audit_dir / "discarded_outer_candidates.csv", index=False)
    (audit_dir / "comparison_config.json").write_text(
        json.dumps({
            "comparison": comparison_name,
            "model": args.model,
            "split_strategy": "random_row",
            "outer_allocation_method": "random_row",
            "inner_allocation_method": "random_row",
            "reference_configuration": args.reference_configuration,
            "configurations": configurations,
            "outer_repeats_requested": args.outer_repeats,
            "n_trials": args.n_trials,
            "xgb_search_space": args.xgb_search_space,
            "random_seed": args.random_seed,
            "model_random_seed": args.model_random_seed,
            "test_fraction": args.test_fraction,
            "validation_folds": args.validation_folds,
            "reuse_outer_assignments_from": (
                str(args.reuse_outer_assignments_from)
                if args.reuse_outer_assignments_from is not None else None
            ),
            "selection_rule": (
                "Retain the first complete policy set on distinct random-row assignments "
                "from a deterministic candidate-seed sequence. No target, prediction, "
                "performance, or feature-coverage measure selects assignments."
            ),
            "figures": figure_paths,
            "failures": failures,
        }, indent=2),
        encoding="utf-8",
    )
    (batch_dir / "START_HERE.md").write_text(
        f"""# Paired {output_description}

Every retained outer repeat uses a frozen random-row assignment. Each
configuration is trained and tested on the same outer rows, but reruns feature
selection within its own outer-training and inner-training partitions.

- summary/performance_summary.csv has one row per threshold configuration, with
  overall and low/high-logKd held-out performance summarized by mean and SD.
- summary/coverage_summary.csv has database contrast and split retention for
  each scientific feature block.
- summary/{configuration_definition_filename} records the exact policy for each model.
- audit/run_summary.csv contains held-out metrics, feature counts, tuned
  parameters, and individual run locations.
- figures/configuration_performance_comparison.png, .pdf, and .svg compare the
  aggregate held-out metrics and compact retention diagnostic.
- detail/testing_predictions.parquet is the canonical row-level held-out
  prediction table used to create the per-configuration predicted-versus-actual
  figures in figures/.
- detail/feature_selection_stability.csv reports how often each feature
  passes training-scope screening across the retained repeats.
- model_runs/<configuration>/outer_repeat_NNN/ retains every fitted run with its
  configuration, split assignment, and per-run tables, so post-hoc analysis never
  requires retraining; see model_runs/README.md. Runs from candidates discarded
  before retention are kept under debug/incomplete_model_runs/.

This is a random-row screen, not an estimate of generalization to unseen
studies or unseen material identities.
""",
        encoding="utf-8",
    )


def reused_assignments(directory: Path, count: int) -> list[dict[str, Any]]:
    path = directory / "audit" / "outer_assignment_manifest.csv"
    manifest = pd.read_csv(path).sort_values("outer_repeat_id", kind="stable")
    required = {
        "outer_repeat_id", "candidate_id", "random_seed", "outer_assignment_path",
        "outer_testing_membership_hash", "testing_rows_from_assignment",
    }
    missing = sorted(required.difference(manifest.columns))
    if missing:
        raise ValueError(f"Reuse manifest is missing required columns: {missing}")
    if len(manifest) < count:
        raise ValueError(f"Reuse manifest contains {len(manifest)} assignments, not {count}.")
    values: list[dict[str, Any]] = []
    seen: set[str] = set()
    for repeat_id, row in enumerate(manifest.iloc[:count].itertuples(), start=1):
        assignment_path = Path(str(row.outer_assignment_path))
        membership_hash, testing_rows = paired_common._validate_outer_assignment(
            assignment_path, expected_strategy="random_row"
        )
        if membership_hash != str(row.outer_testing_membership_hash):
            raise ValueError(f"Reused assignment membership changed: {assignment_path}")
        if membership_hash in seen:
            raise ValueError("Reuse manifest contains duplicate testing memberships.")
        seen.add(membership_hash)
        values.append({
            "outer_repeat_id": repeat_id,
            "candidate_id": int(row.candidate_id),
            "seed": int(row.random_seed),
            "assignment_path": assignment_path,
            "membership_hash": membership_hash,
            "testing_rows": testing_rows,
        })
    return values


def run_configuration_set(
    train_script: Path,
    args: argparse.Namespace,
    *,
    configurations: dict[str, dict[str, float | int]],
    batch_dir: Path,
    candidate: dict[str, Any],
    on_progress: Callable[[list[dict[str, Any]]], None] | None = None,
) -> tuple[list[dict[str, Any]], list[pd.DataFrame], bool]:
    rows: list[dict[str, Any]] = []
    manifests: list[pd.DataFrame] = []
    all_completed = True
    for name, policy in configurations.items():
        run_dir = model_run_directory(
            batch_dir,
            name,
            outer_repeat_id=int(candidate["outer_repeat_id"]),
        )
        print(
            f"Training configuration={name}, outer_repeat={candidate['outer_repeat_id']}, "
            f"seed={candidate['seed']}.",
            flush=True,
        )
        succeeded, message = paired_common._run(
            training_command(
                train_script,
                args,
                seed=int(candidate["seed"]),
                assignment_path=Path(candidate["assignment_path"]),
                output_dir=run_dir,
                policy=policy,
            ),
            run_dir,
            console_verbosity=args.console_verbosity,
        )
        all_completed = all_completed and succeeded
        row, manifest = summary_row(
            run_dir,
            configuration=name,
            policy=policy,
            repeat_id=int(candidate["outer_repeat_id"]),
            candidate_id=int(candidate["candidate_id"]),
            seed=int(candidate["seed"]),
            assignment_path=Path(candidate["assignment_path"]),
            membership_hash=str(candidate["membership_hash"]),
            testing_rows=int(candidate["testing_rows"]),
            status="completed" if succeeded else "training_failed",
            message=message,
        )
        rows.append(row)
        if on_progress is not None:
            on_progress(rows)
        if succeeded and not manifest.empty:
            manifest.insert(0, "threshold_configuration", name)
            manifest.insert(1, "outer_repeat_id", int(candidate["outer_repeat_id"]))
            manifest.insert(2, "candidate_id", int(candidate["candidate_id"]))
            manifest.insert(3, "random_seed", int(candidate["seed"]))
            manifests.append(manifest)
    return rows, manifests, all_completed


def main(
    *,
    default_configurations: dict[str, dict[str, Any]] | None = None,
    default_reference_configuration: str = "baseline",
    experiment_label: str = "feature_threshold_comparison",
    comparison_name: str = "paired_feature_selection_threshold_policies",
    configuration_definition_filename: str = "threshold_configurations.csv",
    include_excluded_feature_buckets: bool = False,
    output_description: str = "feature-threshold comparison",
    parser_description: str = "Compare paired feature-threshold policies on frozen random-row splits.",
    automatic_plot_group_column: str | None = None,
    automatic_plot_groups: tuple[str, ...] = (),
) -> None:
    args = parse_args(parser_description)
    try:
        configurations = resolve_configurations(
            args,
            default_configurations=(
                DEFAULT_CONFIGURATION_PATCHES
                if default_configurations is None
                else default_configurations
            ),
            default_reference_configuration=default_reference_configuration,
        )
    except ValueError as error:
        raise SystemExit(f"ERROR: invalid comparison configuration: {error}") from error

    batch_dir = args.output_root / f"{args.model}_{experiment_label}_{datetime.now():%Y%m%d_%H%M%S}"
    batch_dir.mkdir(parents=True, exist_ok=False)
    write_live_run_summary(batch_dir, [])
    train_script = Path(__file__).with_name("train_logkd_model.py")
    preparation_root = batch_dir / "outer_assignments"
    model_runs_root(batch_dir).mkdir(parents=True, exist_ok=True)
    run_rows: list[dict[str, Any]] = []
    assignment_rows: list[dict[str, Any]] = []
    discarded_rows: list[dict[str, Any]] = []
    manifests: list[pd.DataFrame] = []
    failures: list[str] = []
    retained = 0

    if args.reuse_outer_assignments_from is not None:
        try:
            candidates = reused_assignments(
                args.reuse_outer_assignments_from, args.outer_repeats
            )
        except (OSError, ValueError, pd.errors.ParserError) as error:
            raise SystemExit(f"ERROR: could not reuse outer assignments: {error}") from error
        candidate_source = iter(candidates)
        maximum_candidates = len(candidates)
    else:
        candidate_source = enumerate(
            candidate_seeds(
                args.random_seed,
                args.outer_repeats * MAX_SEED_CANDIDATES_PER_REQUESTED_REPEAT,
            ),
            start=1,
        )
        maximum_candidates = args.outer_repeats * MAX_SEED_CANDIDATES_PER_REQUESTED_REPEAT

    retained_paths: list[Path] = []
    retained_hashes: set[str] = set()
    for source_item in candidate_source:
        if retained >= args.outer_repeats:
            break
        if args.reuse_outer_assignments_from is not None:
            candidate = source_item
        else:
            candidate_id, seed = source_item
            prepare_dir = preparation_root / f"candidate_{candidate_id:03d}_seed_{seed}"
            print(f"Preparing outer candidate={candidate_id}, seed={seed}.", flush=True)
            succeeded, message = paired_common._run(
                prepare_command(
                    train_script,
                    args,
                    seed=seed,
                    output_dir=prepare_dir,
                    policy=configurations[args.reference_configuration],
                ),
                prepare_dir,
                console_verbosity=args.console_verbosity,
            )
            assignment_path = prepare_dir / "split_assignments.csv"
            if not succeeded or not assignment_path.exists():
                discarded_rows.append({
                    "candidate_id": candidate_id,
                    "random_seed": seed,
                    "status": "outer_preparation_failed",
                    "message": message or "Preparation did not write split_assignments.csv.",
                })
                continue
            try:
                membership_hash, testing_rows = paired_common._validate_outer_assignment(
                    assignment_path, expected_strategy="random_row"
                )
            except (OSError, ValueError, pd.errors.ParserError) as error:
                discarded_rows.append({
                    "candidate_id": candidate_id,
                    "random_seed": seed,
                    "status": "outer_assignment_validation_failed",
                    "message": str(error),
                })
                continue
            if membership_hash in retained_hashes:
                discarded_rows.append({
                    "candidate_id": candidate_id,
                    "random_seed": seed,
                    "status": "duplicate_outer_assignment",
                    "message": "Testing membership repeats an already retained outer assignment.",
                })
                continue
            candidate = {
                "outer_repeat_id": retained + 1,
                "candidate_id": candidate_id,
                "seed": seed,
                "assignment_path": assignment_path,
                "membership_hash": membership_hash,
                "testing_rows": testing_rows,
            }

        rows, new_manifests, complete = run_configuration_set(
            train_script,
            args,
            configurations=configurations,
            batch_dir=batch_dir,
            candidate=candidate,
            on_progress=lambda candidate_rows: write_live_run_summary(
                batch_dir, [*run_rows, *candidate_rows]
            ),
        )
        if not complete:
            # The next candidate reuses this outer repeat number, so move the
            # discarded candidate's fitted runs out of model_runs/.
            quarantine_incomplete_model_runs(
                batch_dir,
                rows,
                candidate_id=int(candidate["candidate_id"]),
                seed=int(candidate["seed"]),
            )
            run_rows.extend(row for row in rows if row["status"] != "completed")
            discarded_rows.append({
                "candidate_id": candidate["candidate_id"],
                "random_seed": candidate["seed"],
                "status": "incomplete_configuration_set",
                "message": "At least one threshold configuration failed; this assignment was excluded.",
            })
            continue
        run_rows.extend(rows)
        manifests.extend(new_manifests)
        retained += 1
        retained_paths.append(Path(candidate["assignment_path"]))
        retained_hashes.add(str(candidate["membership_hash"]))
        assignment_rows.append({
            "outer_repeat_id": candidate["outer_repeat_id"],
            "candidate_id": candidate["candidate_id"],
            "random_seed": candidate["seed"],
            "outer_assignment_path": str(candidate["assignment_path"]),
            "outer_testing_membership_hash": candidate["membership_hash"],
            "testing_rows_from_assignment": candidate["testing_rows"],
            "split_strategy": "random_row",
            "outer_allocation_method": "random_row",
            "inner_allocation_method": "random_row",
            "paired_configurations": ";".join(configurations),
        })

    if retained < args.outer_repeats:
        failures.append(
            f"Retained {retained} of {args.outer_repeats} complete paired repeats "
            f"after considering at most {maximum_candidates} candidates."
        )
    runs = pd.DataFrame(run_rows)
    if runs.empty:
        runs = pd.DataFrame(columns=["threshold_configuration", "outer_repeat_id", "status"])
    write_outputs(
        batch_dir,
        runs=runs,
        assignments=pd.DataFrame(assignment_rows),
        discarded=pd.DataFrame(discarded_rows),
        feature_rows=pd.concat(manifests, ignore_index=True) if manifests else pd.DataFrame(),
        configurations=configurations,
        args=args,
        failures=failures,
        comparison_name=comparison_name,
        configuration_definition_filename=configuration_definition_filename,
        include_excluded_feature_buckets=include_excluded_feature_buckets,
        output_description=output_description,
        automatic_plot_group_column=automatic_plot_group_column,
        automatic_plot_groups=automatic_plot_groups,
    )
    finalize_run_summary(batch_dir)
    print(f"Wrote {output_description} to: {batch_dir}", flush=True)
    if failures:
        print("WARNING: " + "; ".join(failures), file=sys.stderr)


if __name__ == "__main__":
    main()
