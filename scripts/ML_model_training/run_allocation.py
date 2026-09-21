"""Paired, inner-only reference-retention allocation experiment.

Every retained repeat starts with one seeded random outer testing assignment.
The paired cells differ only in how inner validation folds are allocated:
matching seeded-random folds versus target-free ``evidence_balanced`` folds.
The final outer test membership is therefore identical within every pair.
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
from time import perf_counter
from typing import Any, Sequence

import numpy as np
import pandas as pd

from backend import logkd_config as cfg
from backend.logkd_plotting import generate_aggregate_figure
from backend.logkd_retention_diagnostics import summarize_inner_reference_retention
from backend.logkd_coverage import REFERENCE_RETENTION_SUMMARY_METRICS
from backend.logkd_exploratory_exports import (
    CONCISE_PERFORMANCE_METRICS,
    model_run_directory,
    model_runs_root,
    quarantine_incomplete_model_runs,
    summarize_scenarios,
    write_exploratory_outputs,
)
from backend.logkd_progress import finalize_run_summary, write_live_run_summary


PERFORMANCE_METRICS = ("mae", "rmse", "r2", "spearman")
REFERENCE_RETENTION_METRICS = REFERENCE_RETENTION_SUMMARY_METRICS
RETENTION_COMPONENT_METRICS = tuple(
    f"reference_{family}_{block}"
    for family in (
        "entity_retention",
        "value_support_retention",
        "contrast_retention",
    )
    for block in ("experimental_conditions", "adsorbent_properties", "pfas_characteristics")
)
# These three quantities are the allocator's lexicographic objective in
# reader-facing score form: strengthen overall mean retention, then the weakest
# fold, then reduce the spread. The nine components remain in the fold-level
# detail file.
COMPOSITE_RETENTION_METRICS = (
    "min_reference_retention_score",
    "mean_reference_retention_score",
    "reference_retention_score_sd",
)
DEFAULT_OUTER_REPEATS = cfg.DEFAULT_EXPERIMENT_OUTER_REPEATS
DEFAULT_N_TRIALS = cfg.DEFAULT_EXPERIMENT_N_TRIALS
DEFAULT_EARLY_STOP_WARMUP = cfg.DEFAULT_EXPERIMENT_EARLY_STOP_WARMUP
DEFAULT_EARLY_STOP_PATIENCE = cfg.DEFAULT_EXPERIMENT_EARLY_STOP_PATIENCE
MAX_SEED_CANDIDATES_PER_REQUESTED_REPEAT = 20
_CONSOLE_ALERT_PATTERN = re.compile(r"\b(?:traceback|exception|error|fatal|failed)\b", re.IGNORECASE)
_MAX_CONSOLE_ALERT_LINES = 12
_MAX_CONSOLE_FAILURE_TAIL_LINES = 30


def _hide_internal_metadata_directory(path: Path) -> None:
    if sys.platform != "win32" or not path.exists():
        return
    attributes = ctypes.windll.kernel32.GetFileAttributesW(str(path))
    if attributes != 0xFFFFFFFF:
        ctypes.windll.kernel32.SetFileAttributesW(str(path), attributes | 0x2)


def _random_inner_method(strategy: str) -> str:
    return "random_row" if strategy == "random_row" else "random_group"


def _inner_methods(strategy: str) -> tuple[str, str]:
    return _random_inner_method(strategy), "evidence_balanced"


def _label(method: str, strategy: str) -> str:
    if method == "evidence_balanced":
        return "evidence_balanced_row" if strategy == "random_row" else "evidence_balanced_group"
    return method


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run paired inner-fold allocation on frozen seeded-random outer tests. "
            "evidence_balanced never changes the final testing split."
        )
    )
    parser.add_argument("--model", choices=sorted(cfg.MODEL_CATEGORIES), default="AC")
    parser.add_argument(
        "--split-strategies",
        choices=cfg.SPLIT_STRATEGIES,
        nargs="+",
        default=list(cfg.SPLIT_STRATEGIES),
        help=(
            "Split strategies to compare. Defaults to every configured strategy; "
            "pass one or more names to run a subset."
        ),
    )
    parser.add_argument(
        "--grouped-outer-allocation-methods",
        choices=cfg.SPLIT_STRATEGY_GROUPED_OUTER_ALLOCATION_METHODS,
        nargs="+",
        default=list(cfg.SPLIT_STRATEGY_GROUPED_OUTER_ALLOCATION_METHODS),
        help=(
            "Outer allocation scenarios for grouped split strategies. "
            "random_group targets a fraction of groups; "
            "size_matched_random_group targets the nearest feasible fraction "
            "of rows while preserving group integrity."
        ),
    )
    parser.add_argument("--outer-repeats", type=int, default=DEFAULT_OUTER_REPEATS)
    parser.add_argument("--random-seed", type=int, default=cfg.DEFAULT_EXPERIMENT_RANDOM_SEED)
    parser.add_argument("--model-random-seed", type=int, default=cfg.DEFAULT_MODEL_RANDOM_SEED)
    parser.add_argument("--output-root", type=Path, default=cfg.DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--test-fraction", type=float, default=cfg.DEFAULT_TEST_FRACTION)
    parser.add_argument("--validation-folds", type=int, default=cfg.VALIDATION_FOLDS)
    parser.add_argument("--data-mode", choices=("baseline", "drop_unreliable"), default=cfg.DEFAULT_DATA_MODE)
    parser.add_argument("--pfas-feature-family-policy", choices=cfg.PFAS_FEATURE_FAMILY_POLICIES, default=cfg.DEFAULT_PFAS_FEATURE_FAMILY_POLICY)
    parser.add_argument("--correlated-feature-handling", choices=cfg.CORRELATED_FEATURE_HANDLING_CHOICES, default=cfg.DEFAULT_CORRELATED_FEATURE_HANDLING)
    parser.add_argument("--n-trials", type=int, default=DEFAULT_N_TRIALS)
    parser.add_argument("--early-stop-warmup", type=int, default=DEFAULT_EARLY_STOP_WARMUP)
    parser.add_argument("--early-stop-patience", type=int, default=DEFAULT_EARLY_STOP_PATIENCE)
    parser.add_argument("--early-stop-se-multiplier", type=float, default=cfg.DEFAULT_EXPERIMENT_EARLY_STOP_SE_MULTIPLIER)
    parser.add_argument("--early-stop-min-delta-floor", type=float, default=cfg.DEFAULT_EXPERIMENT_EARLY_STOP_MIN_DELTA_FLOOR)
    parser.add_argument("--n-jobs", type=int, default=cfg.DEFAULT_EXPERIMENT_N_JOBS)
    parser.add_argument("--console-verbosity", choices=("quiet", "normal"), default=cfg.DEFAULT_EXPERIMENT_CONSOLE_VERBOSITY)
    parser.add_argument("--training-args", nargs=argparse.REMAINDER, default=[])
    args = parser.parse_args()
    if args.outer_repeats < 1:
        parser.error("--outer-repeats must be at least one.")
    if not 0 < args.test_fraction < 1:
        parser.error("--test-fraction must be between zero and one.")
    if args.validation_folds < 2:
        parser.error("--validation-folds must be at least two.")
    if args.n_trials < 0:
        parser.error("--n-trials must be non-negative.")
    if len(set(args.grouped_outer_allocation_methods)) != len(
        args.grouped_outer_allocation_methods
    ):
        parser.error("--grouped-outer-allocation-methods must not contain duplicates.")
    return args


def _run(command: list[str], run_dir: Path, *, console_verbosity: str) -> tuple[bool, str, float]:
    started = perf_counter()
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / "console_output.txt"
    alerts: list[str] = []
    tail: deque[str] = deque(maxlen=_MAX_CONSOLE_FAILURE_TAIL_LINES)
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(command, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, bufsize=1)
        assert process.stdout is not None
        with process.stdout:
            for line in process.stdout:
                log.write(line)
                if console_verbosity == "normal":
                    print(line, end="")
                elif _CONSOLE_ALERT_PATTERN.search(line) and len(alerts) < _MAX_CONSOLE_ALERT_LINES:
                    alerts.append(line.rstrip())
                tail.append(line.rstrip())
        code = process.wait()
    elapsed = perf_counter() - started
    if code == 0:
        return True, "", elapsed
    if console_verbosity == "quiet":
        print(f"FAILED: subprocess exited with code {code}; log: {log_path}", flush=True)
        for line in tail:
            print(f"  {line}", flush=True)
    return False, f"Subprocess exited with code {code}; see {log_path.name}.", elapsed


def _testing_membership_hash(path: Path) -> str:
    assignments = pd.read_csv(path, usecols=["row_id", "split"])
    values = sorted(assignments.loc[assignments["split"].eq("testing"), "row_id"].astype(str))
    return hashlib.sha256("\n".join(values).encode("utf-8")).hexdigest()


def _candidate_seeds(root_seed: int, strategy: str, count: int) -> list[int]:
    salt = zlib.crc32(f"reference_retention_allocation:{strategy}".encode("utf-8"))
    return [int(value) for value in np.random.SeedSequence([root_seed, salt]).generate_state(count)]


def _fixed_training_arguments(args: argparse.Namespace) -> list[str]:
    return [
        "--data-mode", args.data_mode,
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


def _outer_allocation_methods_for_strategy(
    strategy: str,
    grouped_outer_allocation_methods: Sequence[str],
) -> tuple[str, ...]:
    return (
        ("random_row",)
        if strategy == "random_row"
        else tuple(grouped_outer_allocation_methods)
    )


def _prepare_command(
    train_script: Path,
    args: argparse.Namespace,
    strategy: str,
    outer_allocation_method: str,
    seed: int,
    output_dir: Path,
) -> list[str]:
    return [
        sys.executable, "-u", str(train_script), *args.training_args,
        "--model", args.model,
        "--split-strategy", strategy,
        "--outer-allocation-method", outer_allocation_method,
        "--random-seed", str(seed),
        "--prepare-split-only",
        "--output-dir", str(output_dir),
        *_fixed_training_arguments(args),
    ]


def _read_inner_retention(run_dir: Path) -> dict[str, float]:
    return summarize_inner_reference_retention(run_dir)


def _summary_row(
    run_dir: Path, *, strategy: str, outer_allocation_method: str, repeat_id: int, seed: int,
    assignment_path: Path, testing_hash: str, inner_method: str, status: str,
    message: str, run_wall_seconds: float, preparation_wall_seconds: float,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "split_strategy": strategy,
        "outer_repeat_id": repeat_id,
        "random_seed": seed,
        "outer_allocation_method": outer_allocation_method,
        "inner_allocation_method": inner_method,
        "inner_allocation_label": _label(inner_method, strategy),
        "outer_assignment_path": str(assignment_path),
        "outer_testing_membership_hash": testing_hash,
        "status": status,
        "message": message,
        "run_directory": str(run_dir),
        "run_wall_seconds": float(run_wall_seconds),
        "outer_split_preparation_wall_seconds": float(preparation_wall_seconds),
    }
    if status != "completed":
        return row
    metrics_path = run_dir / "metrics_summary.csv"
    if metrics_path.exists():
        metrics = pd.read_csv(metrics_path)
        testing = metrics.loc[metrics["split"].eq("testing")]
        if not testing.empty:
            testing_row = testing.iloc[0]
            for column in ("metric_weighting", "n", *CONCISE_PERFORMANCE_METRICS):
                if column in testing_row:
                    row[column] = testing_row[column]
    row.update(_read_inner_retention(run_dir))
    config_path = run_dir / "run_config.json"
    if config_path.exists():
        config = json.loads(config_path.read_text(encoding="utf-8"))
        validation = config.get("split", {}).get("validation", {})
        for key in REFERENCE_RETENTION_METRICS:
            if key in validation:
                row[key] = validation[key]
        allocation = config.get("inner_validation", {}).get("reference_retention_allocation", {})
        row["inner_allocation_status"] = allocation.get("optimization_status", "random_baseline")
        for key in ("candidate_moves_considered", "moves_applied"):
            if key in allocation:
                row[f"inner_{key}"] = allocation[key]
    return row


def _completed(runs: pd.DataFrame) -> pd.DataFrame:
    return runs.loc[runs.get("status", pd.Series(dtype=str)).eq("completed")].copy()


def _validate_paired_outer_membership(runs: pd.DataFrame) -> None:
    """Verify completed paired methods still share each frozen outer test set."""

    completed = _completed(runs)
    keys = ("split_strategy", "outer_repeat_id", "random_seed", "outer_allocation_method")
    for values, group in completed.groupby(list(keys), sort=True):
        strategy = str(values[0])
        random_method, balanced_method = _inner_methods(strategy)
        if {random_method, balanced_method}.difference(group["inner_allocation_method"]):
            continue
        if group["outer_testing_membership_hash"].nunique() != 1:
            raise RuntimeError("Paired inner-allocation cells do not share an outer testing membership.")


def _inner_summary(runs: pd.DataFrame) -> pd.DataFrame:
    """Aggregate completed fitted cells into one row per allocation scenario."""

    scenario_keys = (
        "split_strategy",
        "outer_allocation_method",
        "inner_allocation_method",
        "inner_allocation_label",
    )
    return summarize_scenarios(
        runs,
        group_columns=scenario_keys,
        comparison_group_column="inner_allocation_label",
        metrics=(
            *PERFORMANCE_METRICS,
            *(f"inner_{metric}" for metric in COMPOSITE_RETENTION_METRICS),
            "run_wall_seconds",
        ),
    )


def _inner_fold_retention_diagnostics(runs: pd.DataFrame) -> pd.DataFrame:
    """Keep the nine component diagnostics at their natural fold-level grain."""

    frames: list[pd.DataFrame] = []
    context_columns = (
        "split_strategy",
        "outer_allocation_method",
        "outer_repeat_id",
        "random_seed",
        "inner_allocation_method",
        "inner_allocation_label",
        "outer_testing_membership_hash",
    )
    for _, run in _completed(runs).iterrows():
        path = Path(str(run["run_directory"])) / "validation_diagnostics.csv"
        if not path.exists():
            continue
        frame = pd.read_csv(path)
        retention_columns = [
            column for column in frame.columns
            if column.startswith("validation_reference_")
            and column != "validation_reference_retention_loss"
        ]
        if not retention_columns:
            continue
        columns = [
            column for column in (
                "validation_fold",
                "validation_rows",
                "validation_split_units",
                "training_rows",
                "training_only_rows",
                *retention_columns,
            )
            if column in frame.columns
        ]
        detail = frame.loc[:, columns].copy()
        for column in reversed(context_columns):
            detail.insert(0, column, run[column])
        frames.append(detail)
    return pd.concat(frames, ignore_index=True, sort=False) if frames else pd.DataFrame()


def _compact_run_summary(runs: pd.DataFrame) -> pd.DataFrame:
    """One concise results row per fitted allocation cell."""

    columns = (
        "split_strategy",
        "outer_allocation_method",
        "outer_repeat_id",
        "random_seed",
        "outer_testing_membership_hash",
        "inner_allocation_method",
        "inner_allocation_label",
        "status",
        *PERFORMANCE_METRICS,
        *(f"inner_{metric}" for metric in COMPOSITE_RETENTION_METRICS),
        "run_wall_seconds",
    )
    result = runs.copy()
    for column in columns:
        if column not in result:
            result[column] = pd.NA
    return result.loc[:, columns]


def _write_outputs(batch_dir: Path, runs: pd.DataFrame, assignments: pd.DataFrame, args: argparse.Namespace, failures: list[str]) -> dict[str, Any]:
    summary = batch_dir / "summary"
    summary.mkdir(parents=True, exist_ok=True)
    _validate_paired_outer_membership(runs)
    fold_retention = _inner_fold_retention_diagnostics(runs)
    export_runs = runs.copy()
    if export_runs.empty:
        export_runs["run_id"] = pd.Series(dtype="string")
        export_runs["outer_assignment_id"] = pd.Series(dtype="string")
    else:
        export_runs["run_id"] = export_runs.apply(
            lambda row: (
                f"{row['split_strategy']}__{row['outer_allocation_method']}__"
                f"outer_{int(row['outer_repeat_id']):03d}_"
                f"seed_{int(row['random_seed'])}__inner_{row['inner_allocation_method']}"
            ),
            axis=1,
        )
        export_runs["outer_assignment_id"] = [
            f"{strategy}__outer_{str(digest)[:16]}"
            for strategy, digest in zip(
                export_runs["split_strategy"],
                export_runs["outer_testing_membership_hash"],
                strict=True,
            )
        ]
    manifest = write_exploratory_outputs(
        batch_dir,
        export_runs,
        run_context_columns=(
            "split_strategy",
            "outer_allocation_method",
            "outer_repeat_id",
            "random_seed",
            "inner_allocation_method",
        ),
        assignment_context_columns=(
            "split_strategy",
            "outer_allocation_method",
            "outer_repeat_id",
            "random_seed",
            "outer_testing_membership_hash",
        ),
        assignment_id_column="outer_assignment_id",
        assignment_path_column="outer_assignment_path",
        assignment_filename="outer_assignments.parquet",
        failures=failures,
        performance_group_columns=(
            "split_strategy",
            "outer_allocation_method",
            "inner_allocation_method",
        ),
        coverage_group_columns=(
            "split_strategy",
            "outer_allocation_method",
        ),
    )
    # The exporter retains technical provenance in audit/. Keep a compact
    # per-cell result table, but remove its redundant outer/method aggregates.
    for filename in (
        "aggregate_performance_and_time.csv",
        "outer_split_diagnostics.csv",
    ):
        (summary / filename).unlink(missing_ok=True)
    _compact_run_summary(runs).to_csv(
        batch_dir / "audit" / "allocation_run_summary.csv", index=False
    )
    manifest["audit"]["allocation_run_summary"] = "audit/allocation_run_summary.csv"
    (batch_dir / "detail" / "selected_feature_support_detail.csv").unlink(missing_ok=True)
    manifest["detail"].pop("selected_feature_support_detail", None)
    fold_retention.to_csv(
        batch_dir / "detail" / "inner_fold_retention_diagnostics.csv",
        index=False,
    )
    (batch_dir / "START_HERE.md").write_text(
        "# Inner reference-retention allocation experiment\n\n"
        "Each paired row uses the same frozen seeded-random outer test membership. "
        "`evidence_balanced` changes only inner validation-fold allocation, using "
        "fixed outer-training reference entities, value support, and conditional "
        "within-study contrasts; it never uses targets or the final test rows.\n\n"
        "`summary/performance_summary.csv` contains the concise held-out performance "
        "summary, while `summary/coverage_summary.csv` contains the corresponding "
        "database contrast and split-retention evidence. The nine allocation components are retained only in "
        "`detail/inner_fold_retention_diagnostics.csv`, at their natural inner-fold "
        "grain. The aggregate allocation figure is written in PNG, PDF, and SVG "
        "formats under `figures/`. Frozen outer assignments and run configurations are in `audit/` "
        "for reproducibility. Grouped strategies are evaluated under both conventional "
        "`random_group` and row-size-controlled `size_matched_random_group` outer "
        "allocation by default; they are separate outer scenarios, not paired model "
        "performance comparisons.\n\n"
        "Every fitted run is retained under `model_runs/<split_strategy>/"
        "outer_<allocation_method>/inner_<allocation_method>/outer_repeat_NNN/` with "
        "its own configuration, split assignment, and per-run tables, so post-hoc "
        "analysis never requires retraining; `audit/model_runs.csv` indexes them. "
        "Runs from candidates discarded before retention are kept under "
        "`debug/incomplete_model_runs/`.\n",
        encoding="utf-8",
    )
    return manifest


def _generate_allocation_figures(
    batch_dir: Path,
    split_strategies: Sequence[str],
    grouped_outer_allocation_methods: Sequence[str],
) -> list[str]:
    """Generate one paired inner-allocation figure per outer allocation design."""

    figure_paths: list[str] = []
    outer_methods = ["random_row"]
    if any(strategy != "random_row" for strategy in split_strategies):
        outer_methods.extend(grouped_outer_allocation_methods)
    for outer_method in outer_methods:
        figure_paths.extend(
            generate_aggregate_figure(
                batch_dir,
                plot_type="comparison",
                input_relative_path=Path("summary") / "performance_summary.csv",
                output_stem=f"paired_inner_allocation_{outer_method}",
                group_column="inner_allocation_method",
                where={"outer_allocation_method": outer_method},
            )
        )
    return figure_paths


def _generate_prediction_figures(
    batch_dir: Path,
    split_strategies: Sequence[str],
    grouped_outer_allocation_methods: Sequence[str],
) -> list[str]:
    """Generate one repeat-colored prediction figure per split/allocation scenario."""

    figure_paths: list[str] = []
    for strategy in split_strategies:
        outer_methods = _outer_allocation_methods_for_strategy(
            strategy, grouped_outer_allocation_methods
        )
        for outer_method in outer_methods:
            figure_paths.extend(
                generate_aggregate_figure(
                    batch_dir,
                    plot_type="predicted-vs-actual",
                    input_relative_path=Path("detail") / "testing_predictions.parquet",
                    output_stem=f"predicted_vs_actual_{strategy}_{outer_method}",
                    group_column="inner_allocation_method",
                    where={
                        "split_strategy": strategy,
                        "outer_allocation_method": outer_method,
                    },
                    title=(
                        f"{strategy.replace('_', ' ').title()} / "
                        f"{outer_method.replace('_', ' ')} outer split: "
                        "predicted versus actual across repeats"
                    ),
                )
            )
    return figure_paths


def main() -> None:
    args = parse_args()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    label = args.split_strategies[0] if len(args.split_strategies) == 1 else "multi_strategy"
    batch_dir = args.output_root / f"{args.model}_reference_retention_allocation_{label}_{timestamp}"
    batch_dir.mkdir(parents=True, exist_ok=False)
    metadata = batch_dir / ".batch_metadata"
    model_runs_root(batch_dir).mkdir(parents=True, exist_ok=True)
    train_script = Path(__file__).with_name("train_logkd_model.py")
    run_rows: list[dict[str, Any]] = []
    assignment_rows: list[dict[str, Any]] = []
    failures: list[str] = []
    write_live_run_summary(batch_dir, _compact_run_summary(pd.DataFrame(run_rows)))

    for strategy in args.split_strategies:
        for outer_allocation_method in _outer_allocation_methods_for_strategy(
            strategy,
            args.grouped_outer_allocation_methods,
        ):
            retained_hashes: set[str] = set()
            retained = 0
            limit = args.outer_repeats * MAX_SEED_CANDIDATES_PER_REQUESTED_REPEAT
            for candidate_id, seed in enumerate(
                _candidate_seeds(args.random_seed, strategy, limit), start=1
            ):
                if retained >= args.outer_repeats:
                    break
                preparation_dir = (
                    metadata
                    / "outer_assignments"
                    / strategy
                    / f"outer_{outer_allocation_method}"
                    / f"candidate_{candidate_id:03d}_seed_{seed}"
                )
                print(
                    "Preparing frozen outer split: "
                    f"strategy={strategy}, outer={outer_allocation_method}, "
                    f"candidate={candidate_id}, seed={seed}",
                    flush=True,
                )
                succeeded, message, preparation_seconds = _run(
                    _prepare_command(
                        train_script,
                        args,
                        strategy,
                        outer_allocation_method,
                        seed,
                        preparation_dir,
                    ),
                    preparation_dir,
                    console_verbosity=args.console_verbosity,
                )
                assignment_path = preparation_dir / "split_assignments.csv"
                if not succeeded or not assignment_path.exists():
                    continue
                testing_hash = _testing_membership_hash(assignment_path)
                if testing_hash in retained_hashes:
                    continue
                repeat_id = retained + 1
                cell_rows: list[dict[str, Any]] = []
                all_completed = True
                for inner_method in _inner_methods(strategy):
                    run_dir = model_run_directory(
                        batch_dir,
                        strategy,
                        f"outer_{outer_allocation_method}",
                        f"inner_{inner_method}",
                        outer_repeat_id=repeat_id,
                    )
                    command = [
                        sys.executable, "-u", str(train_script), *args.training_args,
                        "--model", args.model,
                        "--split-strategy", strategy,
                        "--outer-allocation-method", outer_allocation_method,
                        "--inner-allocation-method", inner_method,
                        "--random-seed", str(seed),
                        "--split-assignment-path", str(assignment_path),
                        "--output-dir", str(run_dir),
                        *_fixed_training_arguments(args),
                    ]
                    print(
                        "Training paired cell: "
                        f"strategy={strategy}, outer={outer_allocation_method}, "
                        f"repeat={repeat_id}, inner={_label(inner_method, strategy)}",
                        flush=True,
                    )
                    completed, cell_message, seconds = _run(
                        command, run_dir, console_verbosity=args.console_verbosity
                    )
                    all_completed = all_completed and completed
                    cell_rows.append(_summary_row(
                        run_dir,
                        strategy=strategy,
                        outer_allocation_method=outer_allocation_method,
                        repeat_id=repeat_id,
                        seed=seed,
                        assignment_path=assignment_path,
                        testing_hash=testing_hash,
                        inner_method=inner_method,
                        status="completed" if completed else "training_failed",
                        message=cell_message,
                        run_wall_seconds=seconds,
                        preparation_wall_seconds=preparation_seconds,
                    ))
                    write_live_run_summary(
                        batch_dir,
                        _compact_run_summary(pd.DataFrame([*run_rows, *cell_rows])),
                    )
                run_rows.extend(cell_rows)
                if not all_completed:
                    # The next candidate reuses this outer repeat number, so
                    # move the discarded cells out of model_runs/ first.
                    quarantine_incomplete_model_runs(
                        batch_dir, cell_rows, candidate_id=candidate_id, seed=seed
                    )
                    continue
                retained += 1
                retained_hashes.add(testing_hash)
                assignments = pd.read_csv(assignment_path, usecols=["split"])
                assignment_rows.append({
                    "split_strategy": strategy,
                    "outer_repeat_id": repeat_id,
                    "random_seed": seed,
                    "outer_allocation_method": outer_allocation_method,
                    "outer_testing_membership_hash": testing_hash,
                    "outer_assignment_path": str(assignment_path),
                    "testing_rows": int(assignments["split"].eq("testing").sum()),
                    "training_rows": int(assignments["split"].eq("training").sum()),
                    "outer_split_preparation_wall_seconds": preparation_seconds,
                    "outer_protocol": "seeded_random_frozen_before_paired_inner_allocation",
                })
            if retained < args.outer_repeats:
                failures.append(
                    f"{strategy}/{outer_allocation_method}: retained {retained} of "
                    f"{args.outer_repeats} complete distinct outer repeats."
                )

    runs = pd.DataFrame(run_rows)
    assignments = pd.DataFrame(assignment_rows)
    manifest = _write_outputs(batch_dir, runs, assignments, args, failures)
    try:
        manifest["allocation_figures"] = _generate_allocation_figures(
            batch_dir,
            args.split_strategies,
            args.grouped_outer_allocation_methods,
        )
    except RuntimeError as error:
        failures.append(f"Figure generation failed: {error}")
        manifest["allocation_figures"] = []
        print(f"WARNING: {error}", file=sys.stderr)
    try:
        manifest["prediction_figures"] = _generate_prediction_figures(
            batch_dir,
            args.split_strategies,
            args.grouped_outer_allocation_methods,
        )
    except RuntimeError as error:
        failures.append(f"Prediction-figure generation failed: {error}")
        manifest["prediction_figures"] = []
        print(f"WARNING: {error}", file=sys.stderr)
    finalize_run_summary(batch_dir)
    (metadata / "allocation_config.json").write_text(json.dumps({
        "model": args.model,
        "split_strategies": args.split_strategies,
        "grouped_outer_allocation_methods": args.grouped_outer_allocation_methods,
        "outer_repeats_requested": args.outer_repeats,
        "outer_protocol": (
            "frozen seeded-random outer splits; grouped strategies include conventional "
            "random-group and row-size-matched random-group scenarios"
        ),
        "inner_methods": ["matching_seeded_random", "evidence_balanced"],
        "evidence_balanced_scope": "inner_validation_only_relative_to_complete_frozen_outer_training_cohort",
        "objective": {
            "components": ["entity_retention", "value_support_retention", "contrast_retention"],
            "aggregation": "mean_across_features_within_blocks; equal_blocks_and_components; lexicographic_mean_then_weakest_fold_then_sd",
            "retired_coverage_loss_terms_used": False,
        },
        "evidence_outputs": manifest,
        "failures": failures,
    }, indent=2), encoding="utf-8")
    _hide_internal_metadata_directory(metadata)
    if failures:
        print("WARNING: " + "; ".join(failures), file=sys.stderr)
    else:
        print(f"Completed paired allocation experiment: {batch_dir}", flush=True)


if __name__ == "__main__":
    main()
