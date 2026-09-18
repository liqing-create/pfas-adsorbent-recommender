"""Run repeated, tuned comparisons of conventional split strategies.

This runner compares the random data-partition scenarios that define the
modelling questions:

* ``random_row`` with conventional random source-record holdouts;
* seeded random whole-group holdouts for each grouped strategy.

The trainer calculates reference-relative evidence-retention quantities after
every split. They are exported as diagnostics and related descriptively to
held-out performance, but never affect split selection, tuning, or model fitting.

All outer and inner split constructors operate on ``source_row_index``.  Thus
all isotherm-expanded records from one source row stay in one outer partition
and one inner validation fold.
"""

from __future__ import annotations

import argparse
from collections import deque
import ctypes
import hashlib
import json
import re
import subprocess
import sys
import zlib
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from backend.logkd_exploratory_exports import (
    model_run_directory,
    model_runs_root,
    quarantine_incomplete_model_runs,
    write_exploratory_outputs,
)
from backend.logkd_progress import finalize_run_summary, write_live_run_summary
from logkd_shap_outer_robustness import generate_robustness_summary
from backend import logkd_config as cfg
from backend.logkd_console_log import ConsoleLogSink
from backend.logkd_plotting import generate_aggregate_figure
from backend.logkd_coverage import REFERENCE_RETENTION_SUMMARY_METRICS
from backend.logkd_retention_diagnostics import summarize_inner_reference_retention


PERFORMANCE_METRICS = ("mae", "rmse", "r2", "spearman")
REFERENCE_RETENTION_METRICS = REFERENCE_RETENTION_SUMMARY_METRICS
DEFAULT_OUTER_REPEATS = cfg.DEFAULT_EXPERIMENT_OUTER_REPEATS
DEFAULT_N_TRIALS = cfg.DEFAULT_EXPERIMENT_N_TRIALS
DEFAULT_EARLY_STOP_WARMUP = cfg.DEFAULT_EXPERIMENT_EARLY_STOP_WARMUP
DEFAULT_EARLY_STOP_PATIENCE = cfg.DEFAULT_EXPERIMENT_EARLY_STOP_PATIENCE
DEFAULT_SHAP_TOP_K = 5
DEFAULT_SHAP_MAX_DISPLAY = 15
DEFAULT_GROUPED_OUTER_ALLOCATION_METHODS = (
    cfg.SPLIT_STRATEGY_GROUPED_OUTER_ALLOCATION_METHODS
)
MAX_SEED_CANDIDATES_PER_REQUESTED_REPEAT = 20
_CONSOLE_ALERT_PATTERN = re.compile(
    r"\b(?:traceback|exception|error|fatal|failed)\b", re.IGNORECASE
)
_MAX_CONSOLE_ALERT_LINES = 12
_MAX_CONSOLE_FAILURE_TAIL_LINES = 30


def _hide_internal_metadata_directory(path: Path) -> None:
    """Hide bookkeeping files in Windows Explorer without removing them."""

    if sys.platform != "win32" or not path.exists():
        return
    hidden_attribute = 0x2
    invalid_attributes = 0xFFFFFFFF
    attributes = ctypes.windll.kernel32.GetFileAttributesW(str(path))
    if attributes != invalid_attributes:
        ctypes.windll.kernel32.SetFileAttributesW(
            str(path), attributes | hidden_attribute
        )


def _allocation_unit_label(split_strategy: str) -> str:
    return "row" if split_strategy == "random_row" else "group"


def _allocation_label(allocation_method: str, split_strategy: str) -> str:
    if allocation_method in {
        "random_row",
        "random_group",
        "size_matched_random_group",
    }:
        return allocation_method
    return f"{allocation_method}_{_allocation_unit_label(split_strategy)}"


def _comparison_group(split_strategy: str, outer_allocation_method: str) -> str:
    """Return the unique plotting label for one outer evaluation scenario."""

    if split_strategy == "random_row":
        return "random_row"
    return f"{split_strategy}__{outer_allocation_method}"


def _outer_allocation_methods_for_strategy(
    split_strategy: str,
    grouped_outer_allocation_methods: tuple[str, ...] = (
        DEFAULT_GROUPED_OUTER_ALLOCATION_METHODS
    ),
) -> tuple[str, ...]:
    """Return scenario-valid allocation methods without coverage optimization."""

    if split_strategy == "random_row":
        return ("random_row",)
    return grouped_outer_allocation_methods


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run repeated, tuned split-strategy comparisons using random "
            "outer and inner allocation only. Reference retention is exported as diagnostics."
        )
    )
    parser.add_argument("--model", choices=sorted(cfg.MODEL_CATEGORIES), default="AC")
    parser.add_argument(
        "--prediction-target-label",
        default="logKd",
        help=(
            "Human-readable target label for automatic predicted-versus-actual figures "
            "(default: logKd)."
        ),
    )
    parser.add_argument(
        "--split-strategies",
        choices=cfg.SPLIT_STRATEGIES,
        nargs="+",
        default=list(cfg.SPLIT_STRATEGIES),
    )
    parser.add_argument(
        "--grouped-outer-allocation-methods",
        choices=cfg.SPLIT_STRATEGY_GROUPED_OUTER_ALLOCATION_METHODS,
        nargs="+",
        default=list(DEFAULT_GROUPED_OUTER_ALLOCATION_METHODS),
        help=(
            "Grouped outer allocation methods to compare. random_group targets a "
            "fraction of groups; size_matched_random_group targets the nearest "
            "feasible fraction of rows while keeping groups intact."
        ),
    )
    parser.add_argument(
        "--outer-repeats",
        type=int,
        default=DEFAULT_OUTER_REPEATS,
        help="Complete, distinct outer assignments retained for every scenario (default: 10).",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=cfg.DEFAULT_EXPERIMENT_RANDOM_SEED,
        help="Root seed used to generate candidate outer-split seeds.",
    )
    parser.add_argument(
        "--model-random-seed",
        type=int,
        default=cfg.DEFAULT_MODEL_RANDOM_SEED,
        help="Estimator seed held fixed across repeats; separate from split/tuning seeds.",
    )
    parser.add_argument("--output-root", type=Path, default=cfg.DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--finalize-existing-batch",
        type=Path,
        metavar="BATCH_DIR",
        help=(
            "Rebuild the aggregate outputs for an interrupted, already-fitted batch "
            "from its live run summary and retained run artifacts; does not retrain models."
        ),
    )
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
        help="Optuna trials per retained outer scenario (default: 200; 0 disables tuning).",
    )
    parser.add_argument("--early-stop-warmup", type=int, default=DEFAULT_EARLY_STOP_WARMUP)
    parser.add_argument("--early-stop-patience", type=int, default=DEFAULT_EARLY_STOP_PATIENCE)
    parser.add_argument("--early-stop-se-multiplier", type=float, default=cfg.DEFAULT_EXPERIMENT_EARLY_STOP_SE_MULTIPLIER)
    parser.add_argument("--early-stop-min-delta-floor", type=float, default=cfg.DEFAULT_EXPERIMENT_EARLY_STOP_MIN_DELTA_FLOOR)
    parser.add_argument("--n-jobs", type=int, default=cfg.DEFAULT_EXPERIMENT_N_JOBS)
    parser.add_argument(
        "--save-model",
        dest="save_model",
        action="store_true",
        default=True,
        help=(
            "Save each completed outer model for downstream analysis such as SHAP "
            "(default: enabled)."
        ),
    )
    parser.add_argument(
        "--no-save-model",
        dest="save_model",
        action="store_false",
        help="Disable saving completed outer models.",
    )
    parser.add_argument(
        "--no-shap",
        dest="run_shap",
        action="store_false",
        default=True,
        help=(
            "Disable automatic held-out SHAP robustness summaries. By default, one "
            "summary is generated for each split-strategy/outer-allocation scenario."
        ),
    )
    parser.add_argument(
        "--shap-top-k",
        type=int,
        default=DEFAULT_SHAP_TOP_K,
        help="Top-rank threshold used by automatic SHAP stability summaries (default: 5).",
    )
    parser.add_argument(
        "--shap-max-display",
        type=int,
        default=DEFAULT_SHAP_MAX_DISPLAY,
        help="Number of features shown in each automatic SHAP plot (default: 15).",
    )
    parser.add_argument(
        "--console-verbosity",
        choices=("quiet", "normal"),
        default=cfg.DEFAULT_EXPERIMENT_CONSOLE_VERBOSITY,
        help=(
            "quiet retains complete subprocess logs while printing wrapper progress and "
            "failure tails; normal streams all subprocess output."
        ),
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
    if args.shap_top_k < 1:
        parser.error("--shap-top-k must be at least one.")
    if args.shap_max_display < 1:
        parser.error("--shap-max-display must be at least one.")
    if len(set(args.grouped_outer_allocation_methods)) != len(
        args.grouped_outer_allocation_methods
    ):
        parser.error("--grouped-outer-allocation-methods must not contain duplicates.")
    return args


def _run(
    command: list[str],
    run_dir: Path,
    *,
    console_verbosity: str,
) -> tuple[bool, str]:
    """Run one subprocess and preserve its complete console output."""

    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / "console_output.txt"
    alert_lines: list[str] = []
    tail_lines: deque[str] = deque(maxlen=_MAX_CONSOLE_FAILURE_TAIL_LINES)
    with ConsoleLogSink(log_path) as log:
        process = subprocess.Popen(
            command,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            bufsize=1,
        )
        assert process.stdout is not None
        with process.stdout:
            for line in process.stdout:
                log.write(line)
                if console_verbosity == "normal":
                    print(line, end="")
                elif (
                    _CONSOLE_ALERT_PATTERN.search(line)
                    and len(alert_lines) < _MAX_CONSOLE_ALERT_LINES
                ):
                    alert_lines.append(line.rstrip())
                tail_lines.append(line.rstrip())
        return_code = process.wait()
    if log.degraded:
        print(f"ATTENTION: {log.error}; the run itself was unaffected.", flush=True)
    if console_verbosity == "quiet" and alert_lines:
        print(
            f"ATTENTION: subprocess emitted {len(alert_lines)} error/exception-like "
            f"line(s); full log: {log_path}",
            flush=True,
        )
        for line in alert_lines:
            print(f"  {line}", flush=True)
    if return_code == 0:
        return True, ""
    if console_verbosity == "quiet":
        print(
            f"FAILED: subprocess exited with code {return_code}; showing its final "
            f"{len(tail_lines)} log line(s): {log_path}",
            flush=True,
        )
        for line in tail_lines:
            print(f"  {line}", flush=True)
    return False, f"Subprocess exited with code {return_code}; see {log_path.name}."


def _testing_membership_hash(assignment_path: Path) -> str:
    assignments = pd.read_csv(assignment_path, usecols=["row_id", "split"])
    testing_ids = sorted(
        assignments.loc[assignments["split"].eq("testing"), "row_id"].astype(str)
    )
    return hashlib.sha256("\n".join(testing_ids).encode("utf-8")).hexdigest()


def _validate_outer_assignment(
    assignment_path: Path,
    *,
    expected_strategy: str,
) -> tuple[str, int]:
    """Validate frozen outer membership, including source-record cohesion."""

    assignments = pd.read_csv(assignment_path)
    required = {"row_id", "source_row_index", "split", "split_strategy", "split_unit"}
    missing = sorted(required.difference(assignments.columns))
    if missing:
        raise ValueError(
            f"Frozen assignment is missing required column(s): {', '.join(missing)}."
        )
    if assignments.empty or not assignments["split"].isin(("training", "testing")).all():
        raise ValueError("Frozen assignment must contain only nonempty training/testing memberships.")
    strategies = assignments["split_strategy"].dropna().astype(str).unique()
    if len(strategies) != 1 or strategies[0] != expected_strategy:
        raise ValueError(
            f"Frozen assignment strategy {strategies.tolist()} does not match "
            f"{expected_strategy!r}."
        )
    source_split_count = assignments.groupby("source_row_index", dropna=False)["split"].nunique()
    crossing_sources = int((source_split_count > 1).sum())
    if crossing_sources:
        raise ValueError(
            f"Frozen assignment separates {crossing_sources} isotherm-expanded source row(s) "
            "across outer partitions."
        )
    split_unit_count = assignments.groupby("split_unit", dropna=False)["split"].nunique()
    crossing_units = int((split_unit_count > 1).sum())
    if crossing_units:
        raise ValueError(
            f"Frozen assignment separates {crossing_units} split unit(s) across outer partitions."
        )
    testing_rows = int(assignments["split"].eq("testing").sum())
    training_rows = int(assignments["split"].eq("training").sum())
    if not testing_rows or not training_rows:
        raise ValueError("Frozen assignment must leave at least one training and testing row.")
    return _testing_membership_hash(assignment_path), testing_rows


def _candidate_seeds(root_seed: int, split_strategy: str, count: int) -> list[int]:
    key = zlib.crc32(f"split_strategy_comparison:{split_strategy}".encode("utf-8"))
    return [
        int(value)
        for value in np.random.SeedSequence([root_seed, key]).generate_state(count)
    ]


def _fixed_training_arguments(args: argparse.Namespace) -> list[str]:
    """Arguments that must be fixed across all retained evaluation scenarios."""

    return [
        "--data-mode", str(args.data_mode),
        "--test-fraction", str(args.test_fraction),
        "--validation-folds", str(args.validation_folds),
        "--model-random-seed", str(args.model_random_seed),
        "--n-trials", str(args.n_trials),
        "--early-stop-warmup", str(args.early_stop_warmup),
        "--early-stop-patience", str(args.early_stop_patience),
        "--early-stop-se-multiplier", str(args.early_stop_se_multiplier),
        "--early-stop-min-delta-floor", str(args.early_stop_min_delta_floor),
        "--n-jobs", str(args.n_jobs),
        "--pfas-feature-family-policy", args.pfas_feature_family_policy,
        "--correlated-feature-handling", args.correlated_feature_handling,
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
    command = [
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
    return command


def _training_command(
    train_script: Path,
    args: argparse.Namespace,
    *,
    strategy: str,
    outer_allocation_method: str,
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
        "--output-dir",
        str(output_dir),
        *_fixed_training_arguments(args),
        *(["--save-model"] if args.save_model else []),
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
    status: str,
    message: str = "",
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "split_strategy": split_strategy,
        "outer_repeat_id": outer_repeat_id,
        "candidate_id": candidate_id,
        "random_seed": seed,
        "outer_allocation_method": outer_allocation_method,
        "outer_allocation_label": _allocation_label(
            outer_allocation_method, split_strategy
        ),
        "inner_allocation_method": "random_row" if split_strategy == "random_row" else "random_group",
        "inner_allocation_label": "random_row" if split_strategy == "random_row" else "random_group",
        "outer_assignment_path": str(assignment_path),
        "outer_testing_membership_hash": testing_membership_hash,
        "testing_rows_from_assignment": testing_rows_from_assignment,
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
    validation = config.get("split", {}).get("validation", {})
    training_settings = config.get("training_settings", {})
    if "model_random_seed" in training_settings:
        row["model_random_seed"] = training_settings["model_random_seed"]
    for key in (
        *REFERENCE_RETENTION_METRICS,
        "testing_rows",
        "training_rows",
        "observed_test_fraction",
        "test_fraction_absolute_error",
        "test_row_target_requested_rows",
        "test_row_target_absolute_deviation_rows",
        "testing_size_policy",
        "evaluation_protocol",
        "coverage_objective_used_for_allocation",
    ):
        if key in validation:
            row[key] = validation[key]
    best_params = config.get("best_params")
    if isinstance(best_params, dict):
        row["best_params_json"] = json.dumps(best_params, sort_keys=True)
    selected = config.get("features", {}).get("selected")
    if isinstance(selected, list):
        row["selected_feature_count"] = len(selected)
    tuning = config.get("hyperparameter_tuning")
    if isinstance(tuning, dict):
        row["hyperparameter_tuning_json"] = json.dumps(tuning, sort_keys=True)
    return row


def _completed_runs(runs: pd.DataFrame) -> pd.DataFrame:
    return runs.loc[runs.get("status", pd.Series(dtype=str)).eq("completed")].copy()


def _numeric_distribution(values: pd.Series, prefix: str) -> dict[str, float | int]:
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


def _test_set_size_stability(runs: pd.DataFrame) -> pd.DataFrame:
    completed = _completed_runs(runs)
    records: list[dict[str, Any]] = []
    grouping = ("split_strategy", "outer_allocation_method")
    for keys, group in completed.groupby(list(grouping), dropna=False, sort=True):
        record = dict(zip(grouping, keys, strict=True))
        record["outer_allocation_label"] = _allocation_label(keys[1], keys[0])
        record["n_complete"] = int(len(group))
        for metric in (
            "testing_rows_from_assignment",
            "testing_rows",
            "training_rows",
            "observed_test_fraction",
            "test_fraction_absolute_error",
            "test_row_target_absolute_deviation_rows",
        ):
            if metric in group:
                record.update(_numeric_distribution(group[metric], metric))
        records.append(record)
    return pd.DataFrame(records)


def _retention_performance_associations(runs: pd.DataFrame) -> pd.DataFrame:
    """Report descriptive diagnostic associations; no causal interpretation is made."""

    completed = _completed_runs(runs)
    records: list[dict[str, Any]] = []
    grouping = ("split_strategy", "outer_allocation_method")
    for keys, group in completed.groupby(list(grouping), dropna=False, sort=True):
        for retention_metric in REFERENCE_RETENTION_METRICS:
            if retention_metric not in group:
                continue
            for performance_metric in PERFORMANCE_METRICS:
                if performance_metric not in group:
                    continue
                paired = pd.DataFrame({
                    "retention": pd.to_numeric(group[retention_metric], errors="coerce"),
                    "performance": pd.to_numeric(group[performance_metric], errors="coerce"),
                }).dropna()
                if len(paired) < 3:
                    continue
                if paired["retention"].nunique() < 2 or paired["performance"].nunique() < 2:
                    continue
                records.append({
                    "split_strategy": keys[0],
                    "outer_allocation_method": keys[1],
                    "outer_allocation_label": _allocation_label(keys[1], keys[0]),
                    "reference_retention_metric": retention_metric,
                    "performance_metric": performance_metric,
                    "performance_direction": (
                        "lower_is_better" if performance_metric in {"mae", "rmse"}
                        else "higher_is_better"
                    ),
                    "n_complete_pairs": int(len(paired)),
                    "pearson_correlation": float(paired["retention"].corr(paired["performance"])),
                    "spearman_correlation": float(
                        paired["retention"].corr(paired["performance"], method="spearman")
                    ),
                    "interpretation": (
                        "descriptive repeated-split association; reference retention was not used "
                        "to allocate, tune, or fit this model"
                    ),
                })
    return pd.DataFrame(records)


def _outer_repeat_scenarios(assignments: pd.DataFrame) -> pd.DataFrame:
    """Show paired seed bookkeeping without claiming different tests are paired models."""

    if assignments.empty:
        return pd.DataFrame()
    records: list[dict[str, Any]] = []
    grouping = ("split_strategy", "outer_repeat_id", "candidate_id", "random_seed")
    for keys, group in assignments.groupby(list(grouping), dropna=False, sort=True):
        record = dict(zip(grouping, keys, strict=True))
        record["comparison_interpretation"] = (
            "same candidate seed, but generally different testing memberships; "
            "not a paired model-performance comparison"
        )
        for _, assignment in group.iterrows():
            method = str(assignment["outer_allocation_method"])
            record[f"{method}_testing_membership_hash"] = assignment[
                "outer_testing_membership_hash"
            ]
            record[f"{method}_testing_rows"] = assignment[
                "testing_rows_from_assignment"
            ]
        records.append(record)
    return pd.DataFrame(records)


def _run_id(run: pd.Series) -> str:
    return (
        f"{run['split_strategy']}__outer_{int(run['outer_repeat_id']):03d}_"
        f"seed_{int(run['random_seed'])}__{run['outer_allocation_method']}"
    )


def _outer_assignment_id(run: pd.Series) -> str:
    digest = str(run.get("outer_testing_membership_hash", "unknown"))
    return f"{run['split_strategy']}__outer_{digest[:16]}"


def _automatic_shap_outputs(
    batch_dir: Path,
    runs: pd.DataFrame,
    args: argparse.Namespace,
) -> dict[str, Any]:
    """Generate SHAP robustness outputs separately for each fixed scenario."""

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
        manifest.update({
            "status": "skipped",
            "reason": (
                "Automatic SHAP was skipped because --no-save-model disables the "
                "outer model artifacts required for explanation."
            ),
        })
        return manifest

    completed = _completed_runs(runs)
    if completed.empty:
        manifest.update({
            "status": "skipped",
            "reason": "No completed outer runs were available for SHAP analysis.",
        })
        return manifest

    failures: list[str] = []
    for keys, group in completed.groupby(
        ["split_strategy", "outer_allocation_method"], sort=True
    ):
        split_strategy, outer_allocation_method = (str(value) for value in keys)
        run_dirs = [Path(str(value)) for value in group["run_directory"].tolist()]
        relative_output_dir = (
            Path("shap") / split_strategy / f"outer_{outer_allocation_method}"
        )
        record: dict[str, Any] = {
            "split_strategy": split_strategy,
            "outer_allocation_method": outer_allocation_method,
            "outer_runs": len(run_dirs),
            "output_dir": relative_output_dir.as_posix(),
        }
        if len(run_dirs) < 2:
            record.update({
                "status": "skipped",
                "reason": "At least two completed outer runs are required.",
            })
            manifest["groups"].append(record)
            continue
        try:
            result = generate_robustness_summary(
                run_dirs,
                batch_dir / relative_output_dir,
                top_k=args.shap_top_k,
                max_display=args.shap_max_display,
            )
        except Exception as error:  # Keep completed model comparisons usable if SHAP fails.
            message = f"{type(error).__name__}: {error}"
            record.update({"status": "failed", "message": message})
            failures.append(
                f"SHAP failed for {split_strategy}/{outer_allocation_method}: {message}"
            )
        else:
            record.update({
                "status": "completed",
                "features": result["features"],
                "files": result["files"],
                "representative_run": result["representative_run"],
            })
        manifest["groups"].append(record)

    manifest["status"] = "completed_with_errors" if failures else "completed"
    if failures:
        manifest["failures"] = failures
    return manifest


def _write_readme(batch_dir: Path) -> None:
    (batch_dir / "START_HERE.md").write_text(
        """# Repeated split-strategy comparison

This batch compares repeated, nested-CV fitted models across split strategies.
Every model receives random inner validation folds. Reference-relative evidence
retention is reported for each fitted training partition and never affects
allocation, tuning, or fitting.

## Split scenarios

- **random_row** samples whole `source_row_index` records.
  Isotherm-expanded siblings therefore cannot cross outer partitions or inner
  validation folds.
- **random_group** samples whole split groups with a seeded random draw. Its
  realized test-row count can vary because it targets a fraction of groups.
- **size_matched_random_group** samples whole groups using seeded random costs,
  but fixes the final testing-row count to the nearest feasible value to the
  requested row fraction. It is the row-size-controlled grouped comparator.

For the combination strategy, the configuration records whether held-out PFAS
and adsorbent components were required to remain represented in training.

## Automatic SHAP outputs

Held-out SHAP robustness summaries are generated by default for every scenario
with at least two completed outer runs. They are stored under
`shap/<split_strategy>/outer_<allocation_method>/`. Use `--no-shap` to disable
this post-processing, or `--no-save-model` when model files are not needed.

Each scenario folder also holds a signed beeswarm of one representative outer
repeat, because a beeswarm needs the per-row SHAP values of a single model.
That repeat is the one whose held-out MAE, RMSE, R2, and Spearman sit closest
to the scenario's repeat means, measured as the mean absolute standard score
across those metrics; `shap_representative_run_selection.csv` records the
comparison for every repeat. The beeswarm illustrates a typical repeat and is
not an aggregate: read `shap_robustness_importance.png` for the cross-repeat
result.

## Retained model runs

`model_runs/<split_strategy>/outer_<allocation_method>/outer_repeat_NNN/` keeps
every fitted run, including its saved model, `run_config.json`, and
`split_assignments.csv`. These directories are deliberately not deleted, so SHAP
can be recomputed or corrected post hoc without retraining. `audit/model_runs.csv`
indexes them; see `model_runs/README.md`. Runs from candidates discarded before
retention are kept under `debug/incomplete_model_runs/`.

## Start with these files

- `audit/run_summary.csv`: one row per retained fitted scenario.
- `summary/performance_summary.csv`: held-out MAE, RMSE, R2, and Spearman,
  including low- and high-logKd subsets; each is summarized by mean and SD only.
- `summary/coverage_summary.csv`: database-level contrast and split-retention
  evidence for the PFAS, adsorbent, and experimental-condition blocks.
- `summary/test_set_size_stability.csv`: diagnostic evidence for the practical
  difference between conventional and size-matched grouped test sets.
- `summary/reference_retention_performance_association.csv`: descriptive
  within-scenario correlations between target-free retention diagnostics and
  performance.
- `summary/outer_repeat_scenarios.csv`: same-seed bookkeeping for grouped
  scenarios. Different test memberships must not be read as paired performance
  effects.
- `shap/<split_strategy>/outer_<allocation_method>/shap_robustness_summary.csv`:
  held-out SHAP importance aggregated across that scenario's outer repeats.
- `shap/<split_strategy>/outer_<allocation_method>/shap_representative_beeswarm.png`:
  signed per-row SHAP values for the representative repeat, with
  `shap_representative_row_values.csv` holding every plotted value.

## Support diagnostics

The feature manifest records feature eligibility and data sufficiency. Selected
feature support records training-reference retention and, where applicable,
outer-test support for the features the fitted model used. Input-block support
summarizes completeness across PFAS, adsorbent, experimental-condition, and
combined inputs. These are target-free descriptors of the fitted data, not
row-level residual predictions or reliability classes.
""",
        encoding="utf-8",
    )


def _write_outputs(
    batch_dir: Path,
    runs: pd.DataFrame,
    assignments: pd.DataFrame,
    discarded_rows: list[dict[str, Any]],
    failures: list[str],
    shap_manifest: dict[str, Any],
) -> dict[str, Any]:
    """Consolidate generic artifacts and add split-comparison summaries."""

    export_runs = runs.copy()
    if export_runs.empty:
        export_runs["run_id"] = pd.Series(dtype="string")
        export_runs["outer_assignment_id"] = pd.Series(dtype="string")
    else:
        export_runs["run_id"] = export_runs.apply(_run_id, axis=1)
        export_runs["outer_assignment_id"] = export_runs.apply(
            _outer_assignment_id, axis=1
        )
        export_runs["comparison_group"] = [
            _comparison_group(str(strategy), str(method))
            for strategy, method in zip(
                export_runs["split_strategy"],
                export_runs["outer_allocation_method"],
                strict=True,
            )
        ]
    manifest = write_exploratory_outputs(
        batch_dir,
        export_runs,
        run_context_columns=(
            "comparison_group",
            "split_strategy",
            "outer_allocation_method",
            "outer_repeat_id",
            "candidate_id",
            "random_seed",
        ),
        assignment_context_columns=(
            "split_strategy",
            "outer_allocation_method",
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
            "comparison_group",
            "split_strategy",
            "outer_allocation_method",
        ),
        coverage_group_columns=(
            "split_strategy",
            "outer_allocation_method",
        ),
    )
    counts = manifest.pop("_counts")
    summary_dir = batch_dir / "summary"
    audit_dir = batch_dir / "audit"
    _test_set_size_stability(runs).to_csv(
        summary_dir / "test_set_size_stability.csv", index=False
    )
    _retention_performance_associations(runs).to_csv(
        summary_dir / "reference_retention_performance_association.csv", index=False
    )
    _outer_repeat_scenarios(assignments).to_csv(
        summary_dir / "outer_repeat_scenarios.csv", index=False
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
                "completed_run_count": counts["completed_run_count"],
                "distinct_outer_assignment_count": counts["distinct_assignment_count"],
                "selected_feature_support_detail_rows": counts[
                    "selected_feature_support_detail_rows"
                ],
                "testing_prediction_rows": counts["testing_prediction_rows"],
                "validation_detail_rows": counts["validation_detail_rows"],
                "shap_robustness": shap_manifest,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    _write_readme(batch_dir)
    manifest["summary"].update({
        "test_set_size_stability": "summary/test_set_size_stability.csv",
        "reference_retention_performance_association": (
            "summary/reference_retention_performance_association.csv"
        ),
        "outer_repeat_scenarios": "summary/outer_repeat_scenarios.csv",
    })
    manifest["audit"]["outer_assignments"] = manifest["audit"].pop(
        "split_assignments"
    )
    manifest["audit"].update({
        "outer_assignment_manifest": "audit/outer_assignment_manifest.csv",
        "comparison_config": "audit/comparison_config.json",
    })
    manifest["shap_robustness"] = shap_manifest
    return manifest


def finalize_existing_batch(
    batch_dir: Path,
    *,
    prediction_target_label: str,
) -> tuple[dict[str, Any], list[str]]:
    """Finish post-export comparison artifacts without rerunning model fitting.

    The generic exporter has already placed the durable run summary and outer
    assignment manifest in ``audit/`` by the time this recovery path is needed.
    Reusing those files avoids reading or replacing any fitted-run artifacts.
    """

    batch_dir = batch_dir.resolve()
    summary_dir = batch_dir / "summary"
    detail_dir = batch_dir / "detail"
    audit_dir = batch_dir / "audit"
    summary_path = audit_dir / "run_summary.csv"
    assignment_path = audit_dir / "outer_assignment_manifest.csv"
    required_paths = (
        summary_path,
        assignment_path,
        detail_dir / "selected_feature_support_detail.csv",
        detail_dir / "testing_predictions.parquet",
        detail_dir / "validation_detail.parquet",
    )
    missing_paths = [path for path in required_paths if not path.exists()]
    if missing_paths:
        raise FileNotFoundError(
            "Cannot finalize this batch because its completed export is missing: "
            + ", ".join(str(path) for path in missing_paths)
        )
    runs = pd.read_csv(summary_path)
    assignments = pd.read_csv(assignment_path)
    shap_manifest = {
        "status": "recovered_existing_outputs",
        "note": (
            "Existing SHAP artifacts were retained; SHAP was not recomputed during "
            "output recovery."
        ),
    }
    failures: list[str] = []
    _test_set_size_stability(runs).to_csv(
        summary_dir / "test_set_size_stability.csv", index=False
    )
    _retention_performance_associations(runs).to_csv(
        summary_dir / "reference_retention_performance_association.csv", index=False
    )
    _outer_repeat_scenarios(assignments).to_csv(
        summary_dir / "outer_repeat_scenarios.csv", index=False
    )
    support_detail = pd.read_csv(
        detail_dir / "selected_feature_support_detail.csv"
    )
    testing_predictions = pd.read_parquet(
        detail_dir / "testing_predictions.parquet"
    )
    validation_detail = pd.read_parquet(detail_dir / "validation_detail.parquet")
    manifest: dict[str, Any] = {
        "summary": {
            "performance": "summary/performance_summary.csv",
            "coverage": "summary/coverage_summary.csv",
            "test_set_size_stability": "summary/test_set_size_stability.csv",
            "reference_retention_performance_association": (
                "summary/reference_retention_performance_association.csv"
            ),
            "outer_repeat_scenarios": "summary/outer_repeat_scenarios.csv",
        },
        "detail": {
            "selected_feature_support_detail": (
                "detail/selected_feature_support_detail.csv"
            ),
            "testing_predictions": "detail/testing_predictions.parquet",
            "validation_detail": "detail/validation_detail.parquet",
        },
        "audit": {
            "run_configurations": "audit/run_configurations.parquet",
            "run_summary": "audit/run_summary.csv",
            "outer_assignments": "audit/outer_assignments.parquet",
            "outer_assignment_manifest": "audit/outer_assignment_manifest.csv",
        },
        "shap_robustness": shap_manifest,
    }
    figure_paths, figure_failures = _generate_figures(
        batch_dir,
        prediction_target_label=prediction_target_label,
    )
    manifest["figures"] = figure_paths
    failures.extend(figure_failures)
    (audit_dir / "comparison_config.json").write_text(
        json.dumps(
            {
                "recovery": (
                    "Final comparison artifacts rebuilt from completed exports after "
                    "an interrupted final metadata write; model fitting was not rerun."
                ),
                "completed_run_count": int(len(_completed_runs(runs))),
                "distinct_outer_assignment_count": int(len(assignments)),
                "selected_feature_support_detail_rows": int(len(support_detail)),
                "testing_prediction_rows": int(len(testing_predictions)),
                "validation_detail_rows": int(len(validation_detail)),
                "shap_robustness": shap_manifest,
                "evidence_outputs": manifest,
                "failures": failures,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    _write_readme(batch_dir)
    finalize_run_summary(batch_dir)
    return manifest, failures


def _generate_figures(
    batch_dir: Path,
    *,
    prediction_target_label: str,
) -> tuple[list[str], list[str]]:
    """Create allocation-specific figure folders for manuscript panel selection."""

    figure_paths: list[str] = []
    failures: list[str] = []
    performance_input_path = Path("summary") / "performance_summary.csv"
    prediction_input_path = Path("detail") / "testing_predictions.parquet"
    for output_subdirectory, outer_allocation_method, groups in (
        (
            Path("random_group"),
            "random_group",
            (
                "random_row",
                "adsorbent__random_group",
                "combination__random_group",
                "pfas__random_group",
                "study__random_group",
            ),
        ),
        (
            Path("size_matched_random_group"),
            "size_matched_random_group",
            (
                "random_row",
                "adsorbent__size_matched_random_group",
                "combination__size_matched_random_group",
                "pfas__size_matched_random_group",
                "study__size_matched_random_group",
            ),
        ),
    ):
        try:
            figure_paths.extend(
                generate_aggregate_figure(
                    batch_dir,
                    plot_type="comparison",
                    input_relative_path=performance_input_path,
                    output_stem="performance_comparison",
                    output_subdirectory=output_subdirectory,
                    groups=groups,
                )
            )
        except RuntimeError as error:
            failures.append(
                f"Figure generation ({output_subdirectory}/performance_comparison) "
                f"failed: {error}"
            )
        for allocation_filter in (outer_allocation_method, "random_row"):
            try:
                figure_paths.extend(
                    generate_aggregate_figure(
                        batch_dir,
                        plot_type="predicted-vs-actual",
                        input_relative_path=prediction_input_path,
                        output_stem="predicted_vs_actual",
                        output_subdirectory=output_subdirectory,
                        group_column="split_strategy",
                        where={"outer_allocation_method": allocation_filter},
                        target_label=prediction_target_label,
                    )
                )
            except RuntimeError as error:
                failures.append(
                    "Prediction-figure generation "
                    f"({output_subdirectory}/{allocation_filter}) failed: {error}"
                )
    return figure_paths, failures


def main() -> None:
    args = parse_args()
    if args.finalize_existing_batch is not None:
        _, failures = finalize_existing_batch(
            args.finalize_existing_batch,
            prediction_target_label=args.prediction_target_label,
        )
        if failures:
            print("WARNING: " + "; ".join(failures), file=sys.stderr)
        print(
            f"Finalized existing batch without retraining: "
            f"{args.finalize_existing_batch.resolve()}",
            flush=True,
        )
        return
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    label = args.split_strategies[0] if len(args.split_strategies) == 1 else "multi_strategy"
    batch_dir = args.output_root / f"{args.model}_split_strategy_comparison_{label}_{timestamp}"
    batch_dir.mkdir(parents=True, exist_ok=False)
    train_script = Path(__file__).with_name("train_logkd_model.py")
    metadata_dir = batch_dir / ".batch_metadata"
    preparation_root = metadata_dir / "outer_assignments"
    model_runs_root(batch_dir).mkdir(parents=True, exist_ok=True)

    run_rows: list[dict[str, Any]] = []
    assignment_rows: list[dict[str, Any]] = []
    discarded_rows: list[dict[str, Any]] = []
    failures: list[str] = []
    max_candidates = args.outer_repeats * MAX_SEED_CANDIDATES_PER_REQUESTED_REPEAT
    write_live_run_summary(batch_dir, run_rows)

    for strategy in args.split_strategies:
        outer_methods = _outer_allocation_methods_for_strategy(
            strategy, tuple(args.grouped_outer_allocation_methods)
        )
        retained_assignments = {method: [] for method in outer_methods}
        retained_hashes = {method: set() for method in outer_methods}
        retained = 0
        for candidate_id, seed in enumerate(
            _candidate_seeds(args.random_seed, strategy, max_candidates), start=1
        ):
            if retained >= args.outer_repeats:
                break
            candidate_root = (
                preparation_root / strategy / f"candidate_{candidate_id:03d}_seed_{seed}"
            )
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
                    f"Preparing outer scenario: strategy={strategy}, candidate={candidate_id}, "
                    f"outer={outer_method}, seed={seed}",
                    flush=True,
                )
                succeeded, message = _run(
                    command, prepare_dir, console_verbosity=args.console_verbosity
                )
                assignment_path = prepare_dir / "split_assignments.csv"
                if not succeeded or not assignment_path.exists():
                    discarded_rows.append({
                        "split_strategy": strategy,
                        "candidate_id": candidate_id,
                        "random_seed": seed,
                        "outer_allocation_method": outer_method,
                        "status": "outer_preparation_failed",
                        "message": message or "Preparation did not write split_assignments.csv.",
                        "preparation_directory": str(prepare_dir),
                    })
                    candidate_preparation_failed = True
                    continue
                try:
                    membership_hash, testing_rows = _validate_outer_assignment(
                        assignment_path, expected_strategy=strategy
                    )
                except (OSError, ValueError, pd.errors.ParserError) as error:
                    discarded_rows.append({
                        "split_strategy": strategy,
                        "candidate_id": candidate_id,
                        "random_seed": seed,
                        "outer_allocation_method": outer_method,
                        "status": "outer_assignment_validation_failed",
                        "message": str(error),
                        "preparation_directory": str(prepare_dir),
                    })
                    candidate_preparation_failed = True
                    continue
                assignment_specs[outer_method] = {
                    "assignment_path": assignment_path,
                    "testing_membership_hash": membership_hash,
                    "testing_rows_from_assignment": testing_rows,
                    "preparation_dir": prepare_dir,
                }
            if candidate_preparation_failed or len(assignment_specs) != len(outer_methods):
                continue
            if any(
                spec["testing_membership_hash"] in retained_hashes[method]
                for method, spec in assignment_specs.items()
            ):
                discarded_rows.append({
                    "split_strategy": strategy,
                    "candidate_id": candidate_id,
                    "random_seed": seed,
                    "status": "duplicate_outer_assignment",
                    "message": (
                        "Not retained because at least one allocation method repeated a "
                        "previously retained testing membership."
                    ),
                    "preparation_directory": str(candidate_root),
                })
                continue

            outer_repeat_id = retained + 1
            candidate_run_rows: list[dict[str, Any]] = []
            all_cells_completed = True
            for outer_method, spec in assignment_specs.items():
                run_dir = model_run_directory(
                    batch_dir,
                    strategy,
                    f"outer_{outer_method}",
                    outer_repeat_id=outer_repeat_id,
                )
                command = _training_command(
                    train_script,
                    args,
                    strategy=strategy,
                    outer_allocation_method=outer_method,
                    seed=seed,
                    assignment_path=Path(spec["assignment_path"]),
                    output_dir=run_dir,
                )
                print(
                    f"Training tuned scenario: strategy={strategy}, outer_repeat={outer_repeat_id}, "
                    f"outer={outer_method}, inner={'random_row' if strategy == 'random_row' else 'random_group'}, "
                    f"seed={seed}",
                    flush=True,
                )
                succeeded, message = _run(
                    command, run_dir, console_verbosity=args.console_verbosity
                )
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
                    assignment_rows.append({
                        "split_strategy": strategy,
                        "outer_repeat_id": outer_repeat_id,
                        "candidate_id": candidate_id,
                        "random_seed": seed,
                        "outer_allocation_method": outer_method,
                        "outer_allocation_label": _allocation_label(outer_method, strategy),
                        "outer_assignment_path": str(spec["assignment_path"]),
                        "outer_testing_membership_hash": str(spec["testing_membership_hash"]),
                        "testing_rows_from_assignment": int(spec["testing_rows_from_assignment"]),
                        "inner_allocation_method": (
                            "random_row" if strategy == "random_row" else "random_group"
                        ),
                        "coverage_used_for_allocation": False,
                        "combination_component_representation_enforced": strategy == "combination",
                    })
                retained += 1
            else:
                # The next candidate reuses this outer repeat number, so move
                # the discarded candidate's fitted runs out of model_runs/.
                quarantine_incomplete_model_runs(
                    batch_dir,
                    candidate_run_rows,
                    candidate_id=candidate_id,
                    seed=seed,
                )
                run_rows.extend(
                    row for row in candidate_run_rows if row["status"] != "completed"
                )
                discarded_rows.append({
                    "split_strategy": strategy,
                    "candidate_id": candidate_id,
                    "random_seed": seed,
                    "status": "incomplete_scenario_set",
                    "message": (
                        "Outer assignments were not retained because at least one tuned "
                        "scenario failed. Completed sibling cells were excluded from summaries."
                    ),
                    "preparation_directory": str(candidate_root),
                })
        if retained < args.outer_repeats:
            failures.append(
                f"{strategy}: retained {retained} of {args.outer_repeats} complete "
                f"repeats after {max_candidates} candidate seeds"
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
                "outer_assignment_path",
                "outer_testing_membership_hash",
                "status",
                "run_directory",
            ]
        )
    assignments = pd.DataFrame(assignment_rows)
    shap_manifest = _automatic_shap_outputs(batch_dir, runs, args)
    failures.extend(shap_manifest.get("failures", []))
    evidence_manifest = _write_outputs(
        batch_dir,
        runs,
        assignments,
        discarded_rows,
        failures,
        shap_manifest,
    )
    figure_paths, figure_failures = _generate_figures(
        batch_dir,
        prediction_target_label=args.prediction_target_label,
    )
    evidence_manifest["figures"] = figure_paths
    failures.extend(figure_failures)
    finalize_run_summary(batch_dir)
    (metadata_dir / "comparison_config.json").write_text(
        json.dumps(
            {
                "model": args.model,
                "split_strategies": args.split_strategies,
                "outer_allocation_methods_by_split_strategy": {
                    strategy: list(
                        _outer_allocation_methods_for_strategy(
                            strategy, tuple(args.grouped_outer_allocation_methods)
                        )
                    )
                    for strategy in args.split_strategies
                },
                "grouped_outer_allocation_methods_requested": (
                    args.grouped_outer_allocation_methods
                ),
                "inner_allocation_methods_by_split_strategy": {
                    strategy: (
                        "random_row" if strategy == "random_row" else "random_group"
                    )
                    for strategy in args.split_strategies
                },
                "coverage_allocation": "disabled",
                "coverage_interpretation": (
                    "target-free split-support diagnostics only; not an allocation "
                    "objective or a prediction-reliability classification"
                ),
                "outer_repeats_requested": args.outer_repeats,
                "root_random_seed": args.random_seed,
                "model_random_seed": args.model_random_seed,
                "test_fraction": args.test_fraction,
                "validation_folds": args.validation_folds,
                "n_trials_requested": args.n_trials,
                "console_verbosity": args.console_verbosity,
                "automatic_shap": shap_manifest,
                "combination_component_representation": (
                    "required for combination splits; each test PFAS and adsorbent "
                    "remains represented in outer training"
                ),
                "source_record_integrity": (
                    "outer assignments are explicitly validated for source_row_index "
                    "cohesion; the trainer validates the same constraint in every inner fold"
                ),
                "selection_rule": (
                    "retain the first complete, distinct scenario set in a deterministic "
                    "candidate-seed sequence; no target, prediction, performance, or "
                    "coverage metric selects assignments"
                ),
                "scenario_comparison_interpretation": (
                    "different split strategies and outer allocation methods generally "
                    "have different testing memberships, so compare their repeat "
                    "distributions rather than interpreting row-wise differences as paired effects"
                ),
                "evidence_outputs": evidence_manifest,
                "failures": failures,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    _hide_internal_metadata_directory(metadata_dir)
    if failures:
        print("WARNING: " + "; ".join(failures), file=sys.stderr)


if __name__ == "__main__":
    main()
