"""Paired dosage / initial-concentration ablation for equilibrium logKd models.

The experiment asks how predictive performance and held-out SHAP importance
change when two scientifically plausible but Kd-conversion-coupled variables are
available or explicitly excluded:

* ``Adsorbent_dosage_value_mg/L``
* ``PFAS_C0_value_mg/L``

Four feature scenarios are fitted on the same frozen outer assignment within
each repeat:

1. ``both_included``: both variables remain available to the normal feature gate;
2. ``no_dosage``: adsorbent dosage is explicitly excluded;
3. ``no_initial_concentration``: initial PFAS concentration is explicitly excluded;
4. ``neither``: both variables are explicitly excluded.

The wrapper assumes ``backend/logkd_features.py`` centrally excludes the
non-equilibrium protocol variables ``Solution_volume_(mL)``, ``Contact_time_(h)``,
and ``Mixing_speed_(rpm)`` from equilibrium logKd models, and supports the
``--exclude-features`` argument used here.

Output consolidation uses the same shared exploratory-export infrastructure as
the other logKd comparison wrappers. Standard performance, coverage, prediction,
validation, configuration, and assignment artifacts are therefore written with a
consistent schema. Experiment-specific ablation-contract QA remains under
``audit/``, and SHAP robustness outputs remain under ``shap/<scenario>/``.

Every fitted run is retained under ``model_runs/<scenario>/outer_repeat_NNN/``
together with its saved model, configuration, and split assignment, so a revised
SHAP calculation can be run post hoc without retraining the batch.

No paired-delta CSVs are written. Pairing is an experimental-design property:
within one outer repeat, all feature scenarios use exactly the same held-out rows.
"""

from __future__ import annotations

import argparse
from collections import deque
from datetime import datetime
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from typing import Any, Sequence
import zlib

import numpy as np
import pandas as pd

from backend import logkd_config as cfg
from backend.logkd_console_log import ConsoleLogSink
from backend.logkd_exploratory_exports import (
    model_run_directory,
    model_runs_root,
    quarantine_incomplete_model_runs,
    write_exploratory_outputs,
)
from backend.logkd_plotting import generate_aggregate_figure
from backend.logkd_progress import finalize_run_summary, write_live_run_summary
from backend.logkd_retention_diagnostics import summarize_inner_reference_retention
from logkd_shap_outer_robustness import generate_robustness_summary


DOSAGE_FEATURE = "Adsorbent_dosage_value_mg/L"
C0_FEATURE = "PFAS_C0_value_mg/L"
ALWAYS_EXCLUDED_EQUILIBRIUM_FEATURES = (
    "Solution_volume_(mL)",
    "Contact_time_(h)",
    "Mixing_speed_(rpm)",
)

ABLATION_SCENARIOS: dict[str, dict[str, Any]] = {
    "both_included": {
        "label": "Dosage + initial concentration available",
        "excluded_features": (),
        "dosage_available": True,
        "c0_available": True,
    },
    "no_dosage": {
        "label": "Dosage excluded",
        "excluded_features": (DOSAGE_FEATURE,),
        "dosage_available": False,
        "c0_available": True,
    },
    "no_initial_concentration": {
        "label": "Initial concentration excluded",
        "excluded_features": (C0_FEATURE,),
        "dosage_available": True,
        "c0_available": False,
    },
    "neither": {
        "label": "Dosage + initial concentration excluded",
        "excluded_features": (DOSAGE_FEATURE, C0_FEATURE),
        "dosage_available": False,
        "c0_available": False,
    },
}
DEFAULT_SCENARIOS = tuple(ABLATION_SCENARIOS)
MAX_SEED_CANDIDATES_PER_REPEAT = 20
DEFAULT_SHAP_TOP_K = 5
DEFAULT_SHAP_MAX_DISPLAY = 15
_CONSOLE_ALERT_PATTERN = ("traceback", "exception", "error", "fatal", "failed")
_MAX_CONSOLE_FAILURE_TAIL_LINES = 30


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run a paired 2x2 ablation of adsorbent dosage and initial PFAS "
            "concentration using identical frozen outer test assignments."
        )
    )
    parser.add_argument("--input-path", type=Path, default=cfg.DEFAULT_INPUT)
    parser.add_argument("--sheet-name", default=cfg.SHEET_NAME)
    parser.add_argument("--pfas-features-path", type=Path, default=cfg.DEFAULT_PFAS_FEATURES)
    parser.add_argument("--pfas-features-sheet", default=cfg.DEFAULT_PFAS_FEATURES_SHEET)
    parser.add_argument("--skip-pfas-features-join", action="store_true")
    parser.add_argument("--model", choices=sorted(cfg.MODEL_CATEGORIES), default="AC")
    parser.add_argument("--target", default=cfg.TARGET)
    parser.add_argument("--output-root", type=Path, default=cfg.DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--scenarios",
        choices=tuple(ABLATION_SCENARIOS),
        nargs="+",
        default=list(DEFAULT_SCENARIOS),
        help="Ablation scenarios to fit; defaults to the complete 2x2 design.",
    )
    parser.add_argument(
        "--kd-final-sources",
        nargs="+",
        default=None,
        metavar="SOURCE",
        help=(
            "Optional Kd_final_source restriction applied identically to every ablation "
            "scenario. Omit for the current mixed-endpoint population."
        ),
    )
    parser.add_argument("--outer-repeats", type=int, default=cfg.DEFAULT_EXPERIMENT_OUTER_REPEATS)
    parser.add_argument("--random-seed", type=int, default=cfg.DEFAULT_EXPERIMENT_RANDOM_SEED)
    parser.add_argument("--model-random-seed", type=int, default=cfg.DEFAULT_MODEL_RANDOM_SEED)
    parser.add_argument("--split-strategy", choices=cfg.SPLIT_STRATEGIES, default="random_row")
    parser.add_argument(
        "--outer-allocation-method",
        choices=cfg.OUTER_ALLOCATION_METHODS,
        default=None,
        help=(
            "Outer allocation method. Omit to use random_row for random_row splits and "
            "random_group for grouped splits."
        ),
    )
    parser.add_argument("--test-fraction", type=float, default=cfg.DEFAULT_TEST_FRACTION)
    parser.add_argument("--validation-folds", type=int, default=cfg.VALIDATION_FOLDS)
    parser.add_argument(
        "--data-mode",
        choices=("baseline", "drop_unreliable"),
        default=cfg.DEFAULT_DATA_MODE,
    )
    parser.add_argument("--n-trials", type=int, default=cfg.DEFAULT_EXPERIMENT_N_TRIALS)
    parser.add_argument(
        "--xgb-search-space",
        choices=("compact", "full"),
        default=cfg.DEFAULT_EXPERIMENT_XGB_SEARCH_SPACE,
    )
    parser.add_argument("--early-stop-warmup", type=int, default=cfg.DEFAULT_EXPERIMENT_EARLY_STOP_WARMUP)
    parser.add_argument("--early-stop-patience", type=int, default=cfg.DEFAULT_EXPERIMENT_EARLY_STOP_PATIENCE)
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
        "--no-shap",
        dest="run_shap",
        action="store_false",
        default=True,
        help="Disable automatic source-row-weighted held-out SHAP robustness summaries.",
    )
    parser.add_argument("--shap-top-k", type=int, default=DEFAULT_SHAP_TOP_K)
    parser.add_argument("--shap-max-display", type=int, default=DEFAULT_SHAP_MAX_DISPLAY)
    parser.add_argument(
        "--console-verbosity",
        choices=("quiet", "normal"),
        default=cfg.DEFAULT_EXPERIMENT_CONSOLE_VERBOSITY,
    )
    parser.add_argument(
        "--training-args",
        nargs=argparse.REMAINDER,
        default=[],
        help=(
            "Additional arguments forwarded to train_logkd_model.py. Place this option "
            "last. Split, source, seed, output, and exact-feature ablation arguments are reserved."
        ),
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
    if args.shap_top_k < 1 or args.shap_max_display < 1:
        parser.error("SHAP rank/display limits must be positive.")
    if len(set(args.scenarios)) != len(args.scenarios):
        parser.error("--scenarios cannot contain duplicates.")

    _validate_forwarded_arguments(parser, args.training_args)
    args.outer_allocation_method = _resolve_outer_allocation_method(
        args.split_strategy, args.outer_allocation_method
    )
    return args


def _validate_forwarded_arguments(
    parser: argparse.ArgumentParser,
    values: Sequence[str],
) -> None:
    reserved = {
        "--input-path",
        "--sheet-name",
        "--pfas-features-path",
        "--pfas-features-sheet",
        "--skip-pfas-features-join",
        "--model",
        "--target",
        "--output-root",
        "--output-dir",
        "--data-mode",
        "--kd-final-sources",
        "--split-strategy",
        "--outer-allocation-method",
        "--inner-allocation-method",
        "--test-fraction",
        "--prepare-split-only",
        "--split-assignment-path",
        "--validation-folds",
        "--random-seed",
        "--model-random-seed",
        "--n-trials",
        "--xgb-search-space",
        "--n-jobs",
        "--save-model",
        "--exclude-features",
    }
    used = sorted({value.split("=", 1)[0] for value in values}.intersection(reserved))
    if used:
        parser.error(
            "These arguments are controlled by the ablation runner and cannot be "
            f"supplied through --training-args: {', '.join(used)}"
        )


def _resolve_outer_allocation_method(strategy: str, requested: str | None) -> str:
    expected = "random_row" if strategy == "random_row" else "random_group"
    method = requested or expected
    if strategy == "random_row" and method != "random_row":
        raise ValueError("random_row split_strategy requires outer-allocation-method=random_row.")
    if strategy != "random_row" and method == "random_row":
        raise ValueError("Grouped split strategies cannot use outer-allocation-method=random_row.")
    return method


def _inner_allocation_method(strategy: str) -> str:
    return "random_row" if strategy == "random_row" else "random_group"


def _source_arguments(args: argparse.Namespace) -> list[str]:
    return ["--kd-final-sources", *args.kd_final_sources] if args.kd_final_sources else []


def _input_arguments(args: argparse.Namespace) -> list[str]:
    values = [
        "--input-path", str(args.input_path),
        "--sheet-name", str(args.sheet_name),
        "--pfas-features-path", str(args.pfas_features_path),
        "--pfas-features-sheet", str(args.pfas_features_sheet),
    ]
    if args.skip_pfas_features_join:
        values.append("--skip-pfas-features-join")
    return values


def _fixed_training_arguments(args: argparse.Namespace) -> list[str]:
    return [
        "--data-mode", str(args.data_mode),
        "--test-fraction", str(args.test_fraction),
        "--validation-folds", str(args.validation_folds),
        "--model-random-seed", str(args.model_random_seed),
        "--n-trials", str(args.n_trials),
        "--xgb-search-space", str(args.xgb_search_space),
        "--early-stop-warmup", str(args.early_stop_warmup),
        "--early-stop-patience", str(args.early_stop_patience),
        "--early-stop-se-multiplier", str(args.early_stop_se_multiplier),
        "--early-stop-min-delta-floor", str(args.early_stop_min_delta_floor),
        "--n-jobs", str(args.n_jobs),
        "--pfas-feature-family-policy", str(args.pfas_feature_family_policy),
        "--correlated-feature-handling", str(args.correlated_feature_handling),
    ]


def _prepare_command(
    train_script: Path,
    args: argparse.Namespace,
    *,
    seed: int,
    output_dir: Path,
) -> list[str]:
    return [
        sys.executable,
        "-u",
        str(train_script),
        *args.training_args,
        *_input_arguments(args),
        *_source_arguments(args),
        "--model", args.model,
        "--target", args.target,
        "--split-strategy", args.split_strategy,
        "--outer-allocation-method", args.outer_allocation_method,
        "--random-seed", str(seed),
        "--prepare-split-only",
        "--output-dir", str(output_dir),
        *_fixed_training_arguments(args),
    ]


def _training_command(
    train_script: Path,
    args: argparse.Namespace,
    *,
    scenario: str,
    seed: int,
    assignment_path: Path,
    output_dir: Path,
) -> list[str]:
    exclusions = list(ABLATION_SCENARIOS[scenario]["excluded_features"])
    command = [
        sys.executable,
        "-u",
        str(train_script),
        *args.training_args,
        *_input_arguments(args),
        *_source_arguments(args),
        "--model", args.model,
        "--target", args.target,
        "--split-strategy", args.split_strategy,
        "--outer-allocation-method", args.outer_allocation_method,
        "--inner-allocation-method", _inner_allocation_method(args.split_strategy),
        "--random-seed", str(seed),
        "--split-assignment-path", str(assignment_path),
        "--output-dir", str(output_dir),
        *_fixed_training_arguments(args),
        "--save-model",
    ]
    if exclusions:
        command.extend(["--exclude-features", *exclusions])
    return command


def _run(
    command: list[str],
    run_dir: Path,
    *,
    console_verbosity: str,
) -> tuple[bool, str]:
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / "console_output.txt"
    tail_lines: deque[str] = deque(maxlen=_MAX_CONSOLE_FAILURE_TAIL_LINES)
    alert_lines: list[str] = []
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
                tail_lines.append(line.rstrip())
                if console_verbosity == "normal":
                    print(line, end="")
                elif (
                    any(token in line.casefold() for token in _CONSOLE_ALERT_PATTERN)
                    and len(alert_lines) < 12
                ):
                    alert_lines.append(line.rstrip())
        return_code = process.wait()
    if log.degraded:
        print(f"ATTENTION: {log.error}; the run itself was unaffected.", flush=True)
    if console_verbosity == "quiet" and alert_lines:
        print(f"ATTENTION: inspect {log_path}", flush=True)
        for line in alert_lines:
            print(f"  {line}", flush=True)
    if return_code == 0:
        return True, ""
    if console_verbosity == "quiet":
        print(f"FAILED: {log_path}", flush=True)
        for line in tail_lines:
            print(f"  {line}", flush=True)
    return False, f"Subprocess exited with code {return_code}; see {log_path}."


def _candidate_seeds(root_seed: int, count: int) -> list[int]:
    key = zlib.crc32(b"dosage_c0_ablation")
    return [
        int(value)
        for value in np.random.SeedSequence([root_seed, key]).generate_state(count)
    ]


def _validate_assignment(path: Path) -> tuple[str, int]:
    assignments = pd.read_csv(path)
    required = {"row_id", "source_row_index", "split"}
    missing = sorted(required.difference(assignments.columns))
    if missing:
        raise ValueError(f"Frozen assignment lacks required columns: {missing}")
    if assignments.empty or not assignments["split"].isin(("training", "testing")).all():
        raise ValueError("Frozen assignment must contain valid training/testing rows.")
    if assignments.groupby("source_row_index", dropna=False)["split"].nunique().gt(1).any():
        raise ValueError("A source_row_index crosses the frozen outer partition.")
    testing = assignments.loc[assignments["split"].eq("testing")]
    training = assignments.loc[assignments["split"].eq("training")]
    if testing.empty or training.empty:
        raise ValueError("Frozen assignment must contain both training and testing rows.")
    membership = "\n".join(sorted(testing["row_id"].astype(str)))
    return hashlib.sha256(membership.encode("utf-8")).hexdigest(), int(len(testing))


def _bool_value(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().casefold() in {"1", "true", "yes", "y"}


def _feature_contract_audit(run_dir: Path, scenario: str) -> dict[str, Any]:
    """Validate the ablation contract and return audit-only feature metadata."""

    manifest_path = run_dir / "feature_manifest.csv"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing feature manifest: {manifest_path}")
    manifest = pd.read_csv(manifest_path, keep_default_na=False)
    if not {"feature", "selected", "reason"}.issubset(manifest.columns):
        raise ValueError("feature_manifest.csv lacks feature/selected/reason columns.")
    by_feature = manifest.drop_duplicates("feature", keep="last").set_index("feature")

    for feature in ALWAYS_EXCLUDED_EQUILIBRIUM_FEATURES:
        if feature not in by_feature.index:
            raise ValueError(
                f"Central equilibrium feature {feature!r} is absent from the feature manifest."
            )
        if _bool_value(by_feature.at[feature, "selected"]):
            raise ValueError(
                f"Central equilibrium-domain exclusion is not active: {feature!r} was selected."
            )

    definition = ABLATION_SCENARIOS[scenario]
    for feature in definition["excluded_features"]:
        if feature not in by_feature.index:
            raise ValueError(f"Ablated feature {feature!r} is absent from the feature manifest.")
        if _bool_value(by_feature.at[feature, "selected"]):
            raise ValueError(
                f"Ablation contract failed: {feature!r} remained selected in {scenario!r}."
            )
        if str(by_feature.at[feature, "reason"]) != "feature_excluded_by_ablation":
            raise ValueError(
                f"Ablation contract failed: {feature!r} has reason "
                f"{by_feature.at[feature, 'reason']!r}, expected 'feature_excluded_by_ablation'."
            )

    result: dict[str, Any] = {
        "ablation_scenario": scenario,
        "run_directory": str(run_dir),
        "explicitly_excluded_features": "; ".join(definition["excluded_features"]),
    }
    for key, feature in (("dosage", DOSAGE_FEATURE), ("initial_concentration", C0_FEATURE)):
        if feature in by_feature.index:
            result[f"{key}_selected"] = _bool_value(by_feature.at[feature, "selected"])
            result[f"{key}_selection_reason"] = str(by_feature.at[feature, "reason"])
        else:
            result[f"{key}_selected"] = False
            result[f"{key}_selection_reason"] = "missing_from_manifest"

    config_path = run_dir / "run_config.json"
    if config_path.exists():
        config = json.loads(config_path.read_text(encoding="utf-8"))
        selected = config.get("features", {}).get("selected")
        if isinstance(selected, list):
            result["selected_feature_count"] = len(selected)
        result["model_artifact"] = config.get("model_artifacts", {}).get("model_filename")
    return result


def _summary_row(
    run_dir: Path,
    *,
    scenario: str,
    repeat_id: int,
    candidate_id: int,
    seed: int,
    assignment_path: Path,
    membership_hash: str,
    testing_rows: int,
    split_strategy: str,
    outer_allocation_method: str,
    status: str,
    message: str = "",
) -> dict[str, Any]:
    """Return one standard exploratory run row for the shared exporter."""

    definition = ABLATION_SCENARIOS[scenario]
    row: dict[str, Any] = {
        "ablation_scenario": scenario,
        "ablation_label": definition["label"],
        "explicitly_excluded_features": "; ".join(definition["excluded_features"]),
        "dosage_available_by_contract": bool(definition["dosage_available"]),
        "initial_concentration_available_by_contract": bool(definition["c0_available"]),
        "outer_repeat_id": repeat_id,
        "candidate_id": candidate_id,
        "random_seed": seed,
        "split_strategy": split_strategy,
        "outer_allocation_method": outer_allocation_method,
        "inner_allocation_method": _inner_allocation_method(split_strategy),
        "outer_assignment_path": str(assignment_path),
        "outer_testing_membership_hash": membership_hash,
        "testing_rows_from_assignment": testing_rows,
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
            # Copy the complete trainer testing row so the shared exporter owns
            # the definition of which metrics belong in performance_summary.csv.
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
    training_settings = config.get("training_settings", {})
    if isinstance(training_settings, dict) and "model_random_seed" in training_settings:
        row["model_random_seed"] = training_settings["model_random_seed"]
    return row


def _run_id(row: pd.Series) -> str:
    return (
        f"{row['ablation_scenario']}__outer_{int(row['outer_repeat_id']):03d}_"
        f"seed_{int(row['random_seed'])}"
    )


def _outer_assignment_id(row: pd.Series) -> str:
    # All ablation scenarios in one repeat deliberately share this assignment,
    # so the assignment ID must not include the scenario name.
    digest = str(row.get("outer_testing_membership_hash", "unknown"))
    return f"paired_outer_{digest[:16]}"


def _completed_runs(runs: pd.DataFrame) -> pd.DataFrame:
    return runs.loc[runs.get("status", pd.Series(dtype=str)).eq("completed")].copy()

def _generate_shap_outputs(
    batch_dir: Path,
    runs: pd.DataFrame,
    scenarios: Sequence[str],
    *,
    top_k: int,
    max_display: int,
) -> dict[str, Any]:
    """Write standard held-out SHAP artifacts under shap/<scenario>/ only."""

    manifest: dict[str, Any] = {"groups": []}
    failures: list[str] = []
    completed = _completed_runs(runs)
    for scenario in scenarios:
        group = completed.loc[completed["ablation_scenario"].eq(scenario)].sort_values(
            "outer_repeat_id"
        )
        record: dict[str, Any] = {
            "ablation_scenario": scenario,
            "outer_runs": int(len(group)),
            "output_dir": f"shap/{scenario}",
        }
        if len(group) < 2:
            record.update({
                "status": "skipped",
                "reason": "At least two completed outer runs are required.",
            })
            manifest["groups"].append(record)
            continue
        run_dirs = [Path(str(value)) for value in group["run_directory"].tolist()]
        try:
            result = generate_robustness_summary(
                run_dirs,
                batch_dir / "shap" / scenario,
                top_k=top_k,
                max_display=max_display,
            )
        except Exception as error:
            message = f"{type(error).__name__}: {error}"
            record.update({"status": "failed", "message": message})
            failures.append(f"SHAP failed for {scenario}: {message}")
        else:
            record.update({
                "status": "completed",
                "files": result["files"],
                "features": result["features"],
                "representative_run": result["representative_run"],
            })
        manifest["groups"].append(record)
    manifest["status"] = "completed_with_errors" if failures else "completed"
    if failures:
        manifest["failures"] = failures
    return manifest

def _generate_figures(
    batch_dir: Path,
    scenarios: Sequence[str],
) -> tuple[list[str], list[str]]:
    """Generate the same aggregate performance/prediction figures as other wrappers."""

    figures: list[str] = []
    failures: list[str] = []
    if len(scenarios) < 2:
        return [], ["Aggregate figures skipped because fewer than two scenarios were requested."]

    try:
        figures.extend(
            generate_aggregate_figure(
                batch_dir,
                plot_type="comparison",
                input_relative_path=Path("summary") / "performance_summary.csv",
                output_stem="dosage_c0_performance_comparison",
                group_column="ablation_scenario",
                groups=scenarios,
            )
        )
    except RuntimeError as error:
        failures.append(f"Performance comparison figure failed: {error}")

    try:
        figures.extend(
            generate_aggregate_figure(
                batch_dir,
                plot_type="predicted-vs-actual",
                input_relative_path=Path("detail") / "testing_predictions.parquet",
                output_stem="dosage_c0_predicted_vs_actual",
                group_column="ablation_scenario",
                groups=scenarios,
                title="Dosage / initial-concentration ablation: predicted versus actual logKd",
                target_label="logKd",
            )
        )
    except RuntimeError as error:
        failures.append(f"Predicted-versus-actual figure failed: {error}")

    return figures, failures

def _write_readme(batch_dir: Path) -> None:
    (batch_dir / "START_HERE.md").write_text(
        """# Dosage / initial-concentration ablation

This is a paired 2x2 feature ablation. Within each retained outer repeat, every
scenario uses the identical frozen test membership. The intended difference is
whether adsorbent dosage and/or initial PFAS concentration are available to the
feature-selection/modeling pipeline.

Solution volume, contact time, and mixing speed are expected to be excluded
centrally by `backend/logkd_features.py` for every equilibrium model. The runner
fails if those variables are selected.

## Standard exploratory outputs

The wrapper uses `backend/logkd_exploratory_exports.py`, the same consolidation
backend as the other logKd comparison runners. Read these files first:

- `summary/performance_summary.csv`: held-out overall, low-, middle-, and
  high-logKd performance by ablation scenario, summarized across outer repeats.
- `summary/coverage_summary.csv`: standard selected-feature support/retention
  summary by ablation scenario.
- `audit/run_summary.csv`: one retained fitted run per scenario and outer repeat.
- `detail/testing_predictions.parquet`: held-out predictions with ablation and
  repeat context.
- `detail/selected_feature_support_detail.csv`: selected-feature support detail.
- `detail/validation_detail.parquet`: inner-validation detail.
- `audit/run_configurations.parquet`: consolidated run configurations.
- `audit/outer_assignments.parquet`: deduplicated frozen outer assignments.
- `audit/model_runs.csv`: where each fitted run is retained, and whether it kept
  a saved model.

## Retained model runs

`model_runs/<scenario>/outer_repeat_NNN/` keeps every fitted run: `model.joblib`,
`run_config.json`, `split_assignments.csv`, `feature_manifest.csv`, and the
trainer's per-run tables. These directories are deliberately not deleted, so a
corrected or extended SHAP calculation can be run post hoc without retraining.
See `model_runs/README.md`. Runs from candidates that were discarded before
retention are kept separately under `debug/incomplete_model_runs/`.

## Ablation-specific audit

- `audit/ablation_feature_selection_status.csv`: verifies the central equilibrium
  exclusions and the requested dosage/C0 feature contract.
- `audit/outer_assignment_manifest.csv`: compact repeat-level assignment bookkeeping.

## Figures and SHAP

- `figures/dosage_c0_performance_comparison.png`: standard aggregate performance
  comparison.
- `figures/dosage_c0_predicted_vs_actual*.png`: held-out predicted-versus-actual
  figures generated from the consolidated prediction table.
- `shap/<scenario>/`: standard source-row-weighted held-out SHAP robustness
  outputs, plus `shap_representative_beeswarm.png` for the one repeat whose
  held-out metrics sit closest to that scenario's repeat means.

No paired-delta CSVs are written. Pairing is preserved in the experimental design
through identical outer test membership within each repeat.
""",
        encoding="utf-8",
    )

def main() -> None:
    args = parse_args()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    source_label = "all_sources" if not args.kd_final_sources else "_".join(args.kd_final_sources)
    batch_dir = args.output_root / f"{args.model}_dosage_c0_ablation_{source_label}_{timestamp}"
    batch_dir.mkdir(parents=True, exist_ok=False)

    audit_dir = batch_dir / "audit"
    audit_dir.mkdir(parents=True, exist_ok=True)
    preparation_root = batch_dir / ".batch_metadata" / "outer_assignments"
    model_runs_root(batch_dir).mkdir(parents=True, exist_ok=True)

    train_script = Path(__file__).with_name("train_logkd_model.py")
    if not train_script.exists():
        raise FileNotFoundError(f"Could not find trainer beside this wrapper: {train_script}")

    max_candidates = args.outer_repeats * MAX_SEED_CANDIDATES_PER_REPEAT
    retained_hashes: set[str] = set()
    run_rows: list[dict[str, Any]] = []
    feature_audit_rows: list[dict[str, Any]] = []
    assignment_rows: list[dict[str, Any]] = []
    discarded_rows: list[dict[str, Any]] = []
    failures: list[str] = []
    retained = 0
    write_live_run_summary(batch_dir, run_rows)

    for candidate_id, seed in enumerate(_candidate_seeds(args.random_seed, max_candidates), start=1):
        if retained >= args.outer_repeats:
            break

        prep_dir = preparation_root / f"candidate_{candidate_id:03d}_seed_{seed}"
        print(f"Preparing paired outer assignment: candidate={candidate_id}, seed={seed}", flush=True)
        succeeded, message = _run(
            _prepare_command(train_script, args, seed=seed, output_dir=prep_dir),
            prep_dir,
            console_verbosity=args.console_verbosity,
        )
        assignment_path = prep_dir / "split_assignments.csv"
        if not succeeded or not assignment_path.exists():
            discarded_rows.append({
                "candidate_id": candidate_id,
                "random_seed": seed,
                "status": "outer_preparation_failed",
                "message": message or "Preparation did not write split_assignments.csv.",
                "preparation_directory": str(prep_dir),
            })
            continue

        try:
            membership_hash, testing_rows = _validate_assignment(assignment_path)
        except Exception as error:
            discarded_rows.append({
                "candidate_id": candidate_id,
                "random_seed": seed,
                "status": "outer_assignment_validation_failed",
                "message": f"{type(error).__name__}: {error}",
                "preparation_directory": str(prep_dir),
            })
            continue
        if membership_hash in retained_hashes:
            discarded_rows.append({
                "candidate_id": candidate_id,
                "random_seed": seed,
                "status": "duplicate_outer_assignment",
                "message": "Testing membership repeats a previously retained paired assignment.",
                "preparation_directory": str(prep_dir),
            })
            continue

        repeat_id = retained + 1
        candidate_rows: list[dict[str, Any]] = []
        candidate_feature_audit: list[dict[str, Any]] = []
        all_scenarios_completed = True

        for scenario in args.scenarios:
            run_dir = model_run_directory(
                batch_dir, scenario, outer_repeat_id=repeat_id
            )
            print(
                f"Training ablation: repeat={repeat_id}, scenario={scenario}, seed={seed}",
                flush=True,
            )
            succeeded, message = _run(
                _training_command(
                    train_script,
                    args,
                    scenario=scenario,
                    seed=seed,
                    assignment_path=assignment_path,
                    output_dir=run_dir,
                ),
                run_dir,
                console_verbosity=args.console_verbosity,
            )
            row = _summary_row(
                run_dir,
                scenario=scenario,
                repeat_id=repeat_id,
                candidate_id=candidate_id,
                seed=seed,
                assignment_path=assignment_path,
                membership_hash=membership_hash,
                testing_rows=testing_rows,
                split_strategy=args.split_strategy,
                outer_allocation_method=args.outer_allocation_method,
                status="completed" if succeeded else "training_failed",
                message=message,
            )
            candidate_rows.append(row)
            write_live_run_summary(batch_dir, [*run_rows, *candidate_rows])

            if not succeeded:
                all_scenarios_completed = False
                continue
            try:
                feature_row = _feature_contract_audit(run_dir, scenario)
                feature_row.update({
                    "outer_repeat_id": repeat_id,
                    "candidate_id": candidate_id,
                    "random_seed": seed,
                    "outer_testing_membership_hash": membership_hash,
                    "dosage_available_by_contract": bool(
                        ABLATION_SCENARIOS[scenario]["dosage_available"]
                    ),
                    "initial_concentration_available_by_contract": bool(
                        ABLATION_SCENARIOS[scenario]["c0_available"]
                    ),
                })
                candidate_feature_audit.append(feature_row)
            except Exception as error:
                all_scenarios_completed = False
                row["status"] = "feature_contract_failed"
                row["message"] = f"{type(error).__name__}: {error}"

        if all_scenarios_completed and len(candidate_feature_audit) == len(args.scenarios):
            run_rows.extend(candidate_rows)
            feature_audit_rows.extend(candidate_feature_audit)
            assignment_rows.append({
                "outer_repeat_id": repeat_id,
                "candidate_id": candidate_id,
                "random_seed": seed,
                "split_strategy": args.split_strategy,
                "outer_allocation_method": args.outer_allocation_method,
                "inner_allocation_method": _inner_allocation_method(args.split_strategy),
                "outer_testing_membership_hash": membership_hash,
                "testing_rows_from_assignment": testing_rows,
                "outer_assignment_path": str(assignment_path.resolve()),
            })
            retained_hashes.add(membership_hash)
            retained += 1
        else:
            # Preserve failed cells for debugging, but exclude completed siblings
            # so the reader-facing comparison remains a complete paired design.
            # The outer repeat number is reused by the next candidate, so this
            # candidate's fitted runs leave the retained model_runs/ tree first.
            quarantine_incomplete_model_runs(
                batch_dir, candidate_rows, candidate_id=candidate_id, seed=seed
            )
            run_rows.extend(row for row in candidate_rows if row["status"] != "completed")
            discarded_rows.append({
                "candidate_id": candidate_id,
                "random_seed": seed,
                "status": "incomplete_paired_scenario_set",
                "message": (
                    "The outer assignment was not retained because at least one ablation "
                    "scenario failed training or the feature-contract audit. Completed sibling "
                    "cells were excluded from summaries."
                ),
                "preparation_directory": str(prep_dir),
            })

    if retained < args.outer_repeats:
        failures.append(
            f"Retained {retained} of {args.outer_repeats} requested complete paired repeats."
        )

    runs = pd.DataFrame(run_rows)
    if runs.empty:
        runs = pd.DataFrame(columns=(
            "ablation_scenario", "ablation_label", "outer_repeat_id", "candidate_id",
            "random_seed", "split_strategy", "outer_allocation_method",
            "inner_allocation_method", "outer_assignment_path",
            "outer_testing_membership_hash", "status", "run_directory",
        ))
    if _completed_runs(runs).empty:
        raise RuntimeError("No complete paired ablation repeats were retained.")

    runs["run_id"] = runs.apply(_run_id, axis=1) if not runs.empty else pd.Series(dtype="string")
    runs["outer_assignment_id"] = (
        runs.apply(_outer_assignment_id, axis=1) if not runs.empty else pd.Series(dtype="string")
    )

    if args.run_shap:
        shap_manifest = _generate_shap_outputs(
            batch_dir,
            runs,
            args.scenarios,
            top_k=args.shap_top_k,
            max_display=args.shap_max_display,
        )
        failures.extend(shap_manifest.get("failures", []))
    else:
        shap_manifest = {"status": "disabled", "groups": []}

    manifest = write_exploratory_outputs(
        batch_dir,
        runs,
        run_context_columns=(
            "ablation_scenario",
            "ablation_label",
            "outer_repeat_id",
            "candidate_id",
            "random_seed",
            "split_strategy",
            "outer_allocation_method",
            "inner_allocation_method",
        ),
        assignment_context_columns=(
            "outer_repeat_id",
            "candidate_id",
            "random_seed",
            "split_strategy",
            "outer_allocation_method",
            "outer_testing_membership_hash",
        ),
        assignment_id_column="outer_assignment_id",
        assignment_path_column="outer_assignment_path",
        assignment_filename="outer_assignments.parquet",
        debug_tables={"discarded_outer_candidates.csv": pd.DataFrame(discarded_rows)},
        failures=failures,
        extra_cleanup_paths=(preparation_root,),
        performance_group_columns=("ablation_scenario", "ablation_label"),
        coverage_group_columns=("ablation_scenario", "ablation_label"),
    )
    counts = manifest.pop("_counts")

    # Ablation-specific audit tables complement, rather than replace, the shared exports.
    pd.DataFrame(feature_audit_rows).to_csv(
        audit_dir / "ablation_feature_selection_status.csv", index=False
    )
    pd.DataFrame(assignment_rows).to_csv(
        audit_dir / "outer_assignment_manifest.csv", index=False
    )

    figure_paths, figure_failures = _generate_figures(batch_dir, args.scenarios)
    failures.extend(figure_failures)
    manifest["figures"] = figure_paths
    manifest["shap_robustness"] = shap_manifest
    manifest["audit"]["ablation_feature_selection_status"] = (
        "audit/ablation_feature_selection_status.csv"
    )
    manifest["audit"]["outer_assignment_manifest"] = "audit/outer_assignment_manifest.csv"

    config = {
        "comparison": "paired_dosage_initial_concentration_feature_ablation",
        "model": args.model,
        "target": args.target,
        "kd_final_sources": args.kd_final_sources,
        "scenarios": {key: ABLATION_SCENARIOS[key] for key in args.scenarios},
        "central_equilibrium_domain_exclusions_expected": list(
            ALWAYS_EXCLUDED_EQUILIBRIUM_FEATURES
        ),
        "pairing": {
            "outer_test_membership_shared_across_scenarios_within_repeat": True,
            "split_strategy": args.split_strategy,
            "outer_allocation_method": args.outer_allocation_method,
            "inner_allocation_method": _inner_allocation_method(args.split_strategy),
            "hyperparameters_tuned_independently_by_scenario": True,
            "completed_sibling_cells_from_incomplete_candidate_sets_excluded": True,
        },
        "outer_repeats_requested": args.outer_repeats,
        "outer_repeats_retained": retained,
        "completed_run_count": counts["completed_run_count"],
        "distinct_outer_assignment_count": counts["distinct_assignment_count"],
        "root_random_seed": args.random_seed,
        "model_random_seed": args.model_random_seed,
        "test_fraction": args.test_fraction,
        "validation_folds": args.validation_folds,
        "n_trials": args.n_trials,
        "evidence_outputs": manifest,
        "failures": failures,
    }
    (audit_dir / "comparison_config.json").write_text(
        json.dumps(config, indent=2), encoding="utf-8"
    )
    _write_readme(batch_dir)
    finalize_run_summary(batch_dir)

    print(f"Wrote paired dosage/C0 ablation outputs to: {batch_dir}", flush=True)
    performance_path = batch_dir / "summary" / "performance_summary.csv"
    if performance_path.exists():
        print(pd.read_csv(performance_path).to_string(index=False), flush=True)
    if failures:
        print("WARNING: " + "; ".join(failures), file=sys.stderr)


if __name__ == "__main__":
    main()
