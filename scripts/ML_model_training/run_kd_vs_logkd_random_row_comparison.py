"""Run a paired Kd-versus-log10(Kd) random-row comparison.

The existing split-strategy runner is executed once for each endpoint, with
``random_row`` as the only split strategy.  Both child runs use the same
deterministic candidate-seed sequence.  Their retained outer assignments are
then checked to confirm that every Kd/logKd pair has the identical held-out
rows.

The two native metric scales are intentionally kept separate: an MAE in
L/g cannot be compared numerically with an MAE in log10(L/g).  To compare the
two fitted models fairly, this wrapper also recalculates each model's testing
metrics on each common scale:

* Kd scale: inverse-transform logKd predictions with ``10 ** prediction``;
* log10(Kd) scale: log-transform Kd predictions, when every prediction is
  strictly positive.

The latter is left uncalculated for any repeat with a non-positive Kd
prediction, rather than silently dropping those predictions.

In addition to the endpoint-specific predicted-versus-actual figures, the
wrapper creates direct Kd-trained-versus-logKd-trained performance figures on
each common evaluation scale.
"""

from __future__ import annotations

import argparse
from collections import deque
import json
import math
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from backend import logkd_config as cfg
from backend.logkd_data import equal_source_row_weights
from backend.logkd_metrics import metrics_dict
from backend.logkd_plotting import generate_aggregate_figure
import run_logkd_split_strategy_comparison as split_runner


LOGKD_ENDPOINT = "logKd"
KD_ENDPOINT = "Kd"
DEFAULT_KD_TARGET = "Kd_final_L/g"
DEFAULT_LOGKD_TARGET = cfg.TARGET
PERFORMANCE_METRICS = ("mae", "rmse", "r2", "spearman")
FACTOR_2_LOGKD_TOLERANCE = math.log10(2.0)
DEFAULT_LOW_LOGKD_TAIL_THRESHOLD = 0.0
DEFAULT_HIGH_LOGKD_TAIL_THRESHOLD = 4.0
PAIR_KEYS = (
    "outer_repeat_id",
    "candidate_id",
    "random_seed",
    "outer_testing_membership_hash",
)
_MAX_FAILURE_TAIL_LINES = 30


def parse_args() -> argparse.Namespace:
    """Parse the split-runner defaults, restricted to random-row evaluation."""

    parser = argparse.ArgumentParser(
        description=(
            "Compare Kd and log10(Kd) endpoints on paired repeated random-row "
            "holdouts. All training defaults match "
            "run_logkd_split_strategy_comparison.py."
        )
    )
    parser.add_argument("--model", choices=sorted(cfg.MODEL_CATEGORIES), default="AC")
    parser.add_argument("--kd-target", default=DEFAULT_KD_TARGET)
    parser.add_argument("--logkd-target", default=DEFAULT_LOGKD_TARGET)
    parser.add_argument(
        "--low-logkd-tail-threshold",
        type=float,
        default=DEFAULT_LOW_LOGKD_TAIL_THRESHOLD,
        help=(
            "Diagnostic low-tail boundary. Held-out rows with true logKd below "
            "this value are summarized separately (default: 0)."
        ),
    )
    parser.add_argument(
        "--high-logkd-tail-threshold",
        type=float,
        default=DEFAULT_HIGH_LOGKD_TAIL_THRESHOLD,
        help=(
            "Diagnostic high-tail boundary. Held-out rows with true logKd above "
            "this value are summarized separately (default: 4)."
        ),
    )
    parser.add_argument(
        "--resume-batch",
        type=Path,
        default=None,
        help=(
            "Finish post-processing an existing wrapper batch after a prior run "
            "completed its two child experiments. No models are retrained."
        ),
    )
    parser.add_argument(
        "--outer-repeats",
        type=int,
        default=split_runner.DEFAULT_OUTER_REPEATS,
        help=(
            "Complete paired outer repeats to retain "
            f"(default: {split_runner.DEFAULT_OUTER_REPEATS})."
        ),
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=cfg.DEFAULT_EXPERIMENT_RANDOM_SEED,
    )
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
    parser.add_argument("--n-trials", type=int, default=split_runner.DEFAULT_N_TRIALS)
    parser.add_argument(
        "--early-stop-warmup", type=int, default=split_runner.DEFAULT_EARLY_STOP_WARMUP
    )
    parser.add_argument(
        "--early-stop-patience", type=int, default=split_runner.DEFAULT_EARLY_STOP_PATIENCE
    )
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
        help="Save completed outer models (default: enabled).",
    )
    parser.add_argument(
        "--no-save-model", dest="save_model", action="store_false", help="Do not save models."
    )
    parser.add_argument(
        "--no-shap",
        dest="run_shap",
        action="store_false",
        default=True,
        help="Disable automatic SHAP robustness summaries.",
    )
    parser.add_argument("--shap-top-k", type=int, default=split_runner.DEFAULT_SHAP_TOP_K)
    parser.add_argument(
        "--shap-max-display", type=int, default=split_runner.DEFAULT_SHAP_MAX_DISPLAY
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
        help=(
            "Additional arguments forwarded to train_logkd_model.py. Place last. "
            "Do not include --target; this wrapper assigns each paired target."
        ),
    )
    args = parser.parse_args()
    if args.outer_repeats < 1:
        parser.error("--outer-repeats must be at least one.")
    if not 0 < args.test_fraction < 1:
        parser.error("--test-fraction must be between zero and one.")
    if args.n_trials < 0:
        parser.error("--n-trials must be non-negative.")
    if args.shap_top_k < 1 or args.shap_max_display < 1:
        parser.error("--shap-top-k and --shap-max-display must be at least one.")
    if args.low_logkd_tail_threshold >= args.high_logkd_tail_threshold:
        parser.error("--low-logkd-tail-threshold must be below --high-logkd-tail-threshold.")
    if any(value == "--target" or value.startswith("--target=") for value in args.training_args):
        parser.error("Do not pass --target through --training-args; use --kd-target/--logkd-target.")
    return args


def _child_arguments(
    args: argparse.Namespace,
    target: str,
    output_root: Path,
    *,
    prediction_target_label: str,
) -> list[str]:
    """Return one constrained invocation of the existing split runner."""

    command = [
        sys.executable,
        "-u",
        str(Path(__file__).with_name("run_logkd_split_strategy_comparison.py")),
        "--model",
        args.model,
        "--prediction-target-label",
        prediction_target_label,
        "--split-strategies",
        "random_row",
        "--outer-repeats",
        str(args.outer_repeats),
        "--random-seed",
        str(args.random_seed),
        "--model-random-seed",
        str(args.model_random_seed),
        "--output-root",
        str(output_root),
        "--test-fraction",
        str(args.test_fraction),
        "--validation-folds",
        str(args.validation_folds),
        "--data-mode",
        args.data_mode,
        "--pfas-feature-family-policy",
        args.pfas_feature_family_policy,
        "--correlated-feature-handling",
        args.correlated_feature_handling,
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
        "--shap-top-k",
        str(args.shap_top_k),
        "--shap-max-display",
        str(args.shap_max_display),
        "--console-verbosity",
        args.console_verbosity,
    ]
    if not args.save_model:
        command.append("--no-save-model")
    if not args.run_shap:
        command.append("--no-shap")
    return [*command, "--training-args", *args.training_args, "--target", target]


def _run_child(command: list[str], log_path: Path, console_verbosity: str) -> None:
    """Run one child runner while retaining its full console log."""

    log_path.parent.mkdir(parents=True, exist_ok=True)
    tail: deque[str] = deque(maxlen=_MAX_FAILURE_TAIL_LINES)
    with log_path.open("w", encoding="utf-8") as log:
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
                log.flush()
                if console_verbosity == "normal":
                    print(line, end="")
                tail.append(line.rstrip())
        return_code = process.wait()
    if return_code == 0:
        return
    print(f"Child runner failed; final log lines from {log_path}:", file=sys.stderr)
    for line in tail:
        print(f"  {line}", file=sys.stderr)
    raise RuntimeError(f"Child runner exited with code {return_code}: {log_path}")


def _locate_child_batch(variant_root: Path, model: str) -> Path:
    candidates = sorted(path for path in variant_root.iterdir() if path.is_dir())
    if len(candidates) != 1:
        raise RuntimeError(
            f"Expected exactly one child batch below {variant_root}, found {len(candidates)}."
        )
    batch_dir = candidates[0]
    expected_prefix = f"{model}_split_strategy_comparison_random_row_"
    if not batch_dir.name.startswith(expected_prefix):
        raise RuntimeError(f"Unexpected child batch directory: {batch_dir}")
    return batch_dir


def _completed_summary(batch_dir: Path, endpoint: str) -> pd.DataFrame:
    summary_path = batch_dir / "audit" / "run_summary.csv"
    if not summary_path.exists():
        raise FileNotFoundError(f"Child runner did not write {summary_path}")
    summary = pd.read_csv(summary_path)
    completed = summary.loc[summary["status"].eq("completed")].copy()
    if completed.empty:
        raise RuntimeError(f"{endpoint} child run did not complete any outer repeats.")
    missing = [column for column in (*PAIR_KEYS, "run_id") if column not in completed]
    if missing:
        raise RuntimeError(f"{endpoint} summary is missing required columns: {missing}")
    if completed.duplicated(list(PAIR_KEYS)).any():
        raise RuntimeError(f"{endpoint} child run has duplicate completed outer-repeat keys.")
    return completed


def _paired_native_metrics(logkd: pd.DataFrame, kd: pd.DataFrame) -> pd.DataFrame:
    """Check matching holdouts and return the native-scale metrics in long form."""

    log_keys = set(map(tuple, logkd.loc[:, PAIR_KEYS].itertuples(index=False, name=None)))
    kd_keys = set(map(tuple, kd.loc[:, PAIR_KEYS].itertuples(index=False, name=None)))
    if log_keys != kd_keys:
        only_log = len(log_keys.difference(kd_keys))
        only_kd = len(kd_keys.difference(log_keys))
        raise RuntimeError(
            "Kd and logKd runs did not retain the same frozen outer assignments "
            f"(logKd-only={only_log}, Kd-only={only_kd})."
        )

    metric_columns = [metric for metric in PERFORMANCE_METRICS if metric in logkd and metric in kd]
    if len(metric_columns) != len(PERFORMANCE_METRICS):
        raise RuntimeError("At least one expected performance metric is absent from a child summary.")
    context_columns = [*PAIR_KEYS, "run_id", "testing_rows_from_assignment"]
    frames: list[pd.DataFrame] = []
    for endpoint, scale, frame in (
        (LOGKD_ENDPOINT, "log10(Kd [L/g])", logkd),
        (KD_ENDPOINT, "Kd [L/g]", kd),
    ):
        columns = [column for column in context_columns if column in frame] + metric_columns
        out = frame.loc[:, columns].copy()
        out.insert(len([column for column in context_columns if column in frame]), "model_endpoint", endpoint)
        out.insert(len([column for column in context_columns if column in frame]) + 1, "evaluation_scale", scale)
        out.insert(len([column for column in context_columns if column in frame]) + 2, "metric_status", "native_target_scale")
        frames.append(out)
    return pd.concat(frames, ignore_index=True, sort=False)


def _testing_predictions(
    batch_dir: Path,
    summary: pd.DataFrame,
    endpoint: str,
) -> pd.DataFrame:
    """Load exported held-out predictions and attach their repeat identifiers."""

    path = batch_dir / "detail" / "testing_predictions.parquet"
    if not path.exists():
        raise FileNotFoundError(f"{endpoint} child batch did not write {path}")
    testing = pd.read_parquet(path)
    required = {
        "run_id",
        "standardized_row_id",
        "source_row_index",
        "y_true",
        "y_pred",
    }
    missing = sorted(required.difference(testing.columns))
    if missing:
        raise RuntimeError(f"{endpoint} testing predictions are missing columns: {missing}")
    if testing.empty or testing["standardized_row_id"].isna().any():
        raise RuntimeError(f"{endpoint} testing predictions lack usable standardized row IDs.")
    if testing.duplicated(["run_id", "standardized_row_id"]).any():
        raise RuntimeError(
            f"{endpoint} testing predictions have duplicate standardized row IDs within a run."
        )
    if "source_row_weight" not in testing:
        weighted_runs: list[pd.DataFrame] = []
        for _, run in testing.groupby("run_id", sort=False, dropna=False):
            run = run.copy()
            run["source_row_weight"] = equal_source_row_weights(run).to_numpy(dtype=float)
            weighted_runs.append(run)
        testing = pd.concat(weighted_runs, ignore_index=True, sort=False)
    weights = pd.to_numeric(testing["source_row_weight"], errors="coerce")
    if weights.isna().any() or (~np.isfinite(weights)).any() or weights.le(0).any():
        raise RuntimeError(f"{endpoint} testing predictions contain invalid source-row weights.")
    testing["source_row_weight"] = weights
    run_context = summary.loc[:, [*PAIR_KEYS, "run_id"]]
    return testing.loc[:, [
        "run_id",
        "standardized_row_id",
        "source_row_index",
        "y_true",
        "y_pred",
        "source_row_weight",
    ]].merge(
        run_context,
        on="run_id",
        how="inner",
        validate="many_to_one",
    )


def _write_testing_prediction_detail(
    batch_dir: Path,
    logkd_batch: Path,
    kd_batch: Path,
    logkd: pd.DataFrame,
    kd: pd.DataFrame,
) -> Path:
    """Write the parent batch's canonical testing-prediction Parquet table."""

    frames: list[pd.DataFrame] = []
    for endpoint, child_batch, summary, target_label in (
        (LOGKD_ENDPOINT, logkd_batch, logkd, "logKd"),
        (KD_ENDPOINT, kd_batch, kd, "Kd (L/g)"),
    ):
        predictions = _testing_predictions(child_batch, summary, endpoint).copy()
        predictions.insert(0, "data_split", "testing")
        predictions.insert(1, "model_endpoint", endpoint)
        predictions.insert(2, "prediction_target_label", target_label)
        predictions = predictions.rename(columns={"run_id": "child_run_id"})
        predictions.insert(
            3,
            "run_id",
            endpoint
            + "__outer_"
            + predictions["outer_repeat_id"].astype(int).map("{:03d}".format)
            + "__candidate_"
            + predictions["candidate_id"].astype(int).map("{:03d}".format),
        )
        frames.append(predictions)

    if not frames:
        raise RuntimeError("No paired testing predictions were available to export.")
    output_path = batch_dir / "detail" / "testing_predictions.parquet"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.concat(frames, ignore_index=True, sort=False).to_parquet(output_path, index=False)
    return output_path


def _power10(values: np.ndarray) -> np.ndarray:
    with np.errstate(over="ignore", invalid="ignore"):
        return np.power(10.0, values)


def _metrics_record(
    context: dict[str, Any],
    *,
    endpoint: str,
    evaluation_scale: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    weights: np.ndarray,
    status: str,
    invalid_prediction_count: int = 0,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        **context,
        "model_endpoint": endpoint,
        "evaluation_scale": evaluation_scale,
        "metric_status": status,
        "invalid_prediction_count": invalid_prediction_count,
        "n_testing_rows": int(len(y_true)),
    }
    if status == "calculated":
        row.update(metrics_dict(y_true, y_pred, weights))
    else:
        row.update({metric: math.nan for metric in PERFORMANCE_METRICS})
        row["n"] = 0
    return row


def _common_scale_metrics(
    logkd_batch: Path,
    kd_batch: Path,
    logkd: pd.DataFrame,
    kd: pd.DataFrame,
) -> pd.DataFrame:
    """Re-score each paired model on raw Kd and log10(Kd) scales."""

    left = logkd.loc[:, [*PAIR_KEYS, "run_id"]].rename(
        columns={"run_id": "logkd_run_id"}
    )
    right = kd.loc[:, [*PAIR_KEYS, "run_id"]].rename(
        columns={"run_id": "kd_run_id"}
    )
    pairs = left.merge(right, on=list(PAIR_KEYS), how="inner", validate="one_to_one")
    log_predictions_all = _testing_predictions(logkd_batch, logkd, LOGKD_ENDPOINT)
    kd_predictions_all = _testing_predictions(kd_batch, kd, KD_ENDPOINT)
    rows: list[dict[str, Any]] = []

    for pair in pairs.to_dict(orient="records"):
        log_predictions = log_predictions_all.loc[
            log_predictions_all["run_id"].eq(pair["logkd_run_id"])
        ]
        kd_predictions = kd_predictions_all.loc[
            kd_predictions_all["run_id"].eq(pair["kd_run_id"])
        ]
        merged = log_predictions.merge(
            kd_predictions,
            on="standardized_row_id",
            suffixes=("_logkd", "_kd"),
            how="inner",
            validate="one_to_one",
        )
        if len(merged) != len(log_predictions) or len(merged) != len(kd_predictions):
            raise RuntimeError("Paired child runs have different held-out standardized row IDs.")
        merged = merged.sort_values("standardized_row_id").reset_index(drop=True)
        raw_true_from_log = _power10(merged["y_true_logkd"].to_numpy(dtype=float))
        raw_true = merged["y_true_kd"].to_numpy(dtype=float)
        if not np.isfinite(raw_true_from_log).all() or not np.isclose(
            raw_true_from_log, raw_true, rtol=1e-8, atol=1e-12
        ).all():
            raise RuntimeError(
                "Kd target values do not equal 10 ** logKd target values for a paired test set."
            )
        log_weights = merged["source_row_weight_logkd"].to_numpy(dtype=float)
        kd_weights = merged["source_row_weight_kd"].to_numpy(dtype=float)
        if not np.isclose(log_weights, kd_weights, rtol=1e-12, atol=1e-12).all():
            raise RuntimeError("Paired child runs produced unequal source-row weights.")
        if (raw_true <= 0).any():
            raise RuntimeError("Raw Kd testing targets must be strictly positive for log-scale comparison.")

        context = {column: pair[column] for column in PAIR_KEYS}
        context["logkd_run_id"] = pair["logkd_run_id"]
        context["kd_run_id"] = pair["kd_run_id"]
        log_pred = merged["y_pred_logkd"].to_numpy(dtype=float)
        kd_pred = merged["y_pred_kd"].to_numpy(dtype=float)
        log_true = merged["y_true_logkd"].to_numpy(dtype=float)
        raw_log_pred = _power10(log_pred)

        rows.append(
            _metrics_record(
                context,
                endpoint=LOGKD_ENDPOINT,
                evaluation_scale="Kd [L/g]",
                y_true=raw_true,
                y_pred=raw_log_pred,
                weights=log_weights,
                status="calculated" if np.isfinite(raw_log_pred).all() else "inverse_transform_nonfinite",
                invalid_prediction_count=int((~np.isfinite(raw_log_pred)).sum()),
            )
        )
        rows.append(
            _metrics_record(
                context,
                endpoint=KD_ENDPOINT,
                evaluation_scale="Kd [L/g]",
                y_true=raw_true,
                y_pred=kd_pred,
                weights=kd_weights,
                status="calculated" if np.isfinite(kd_pred).all() else "nonfinite_prediction",
                invalid_prediction_count=int((~np.isfinite(kd_pred)).sum()),
            )
        )
        rows.append(
            _metrics_record(
                context,
                endpoint=LOGKD_ENDPOINT,
                evaluation_scale="log10(Kd [L/g])",
                y_true=log_true,
                y_pred=log_pred,
                weights=log_weights,
                status="calculated" if np.isfinite(log_pred).all() else "nonfinite_prediction",
                invalid_prediction_count=int((~np.isfinite(log_pred)).sum()),
            )
        )
        invalid_kd_log = ~np.isfinite(kd_pred) | (kd_pred <= 0)
        rows.append(
            _metrics_record(
                context,
                endpoint=KD_ENDPOINT,
                evaluation_scale="log10(Kd [L/g])",
                y_true=log_true,
                y_pred=np.log10(kd_pred, where=~invalid_kd_log, out=np.full(len(kd_pred), np.nan)),
                weights=kd_weights,
                status=("calculated" if not invalid_kd_log.any() else "nonpositive_or_nonfinite_prediction"),
                invalid_prediction_count=int(invalid_kd_log.sum()),
            )
        )
    return pd.DataFrame(rows)


def _distribution(values: pd.Series, prefix: str) -> dict[str, float | int]:
    numeric = pd.to_numeric(values, errors="coerce").dropna()
    if numeric.empty:
        return {f"{prefix}_n": 0}
    standard_deviation = float(numeric.std(ddof=1)) if len(numeric) > 1 else 0.0
    standard_error = float(numeric.sem(ddof=1)) if len(numeric) > 1 else 0.0
    return {
        f"{prefix}_mean": float(numeric.mean()),
        f"{prefix}_median": float(numeric.median()),
        f"{prefix}_std": standard_deviation,
        f"{prefix}_sem": standard_error,
        f"{prefix}_normal_95ci_lower": float(numeric.mean() - 1.96 * standard_error),
        f"{prefix}_normal_95ci_upper": float(numeric.mean() + 1.96 * standard_error),
        f"{prefix}_min": float(numeric.min()),
        f"{prefix}_max": float(numeric.max()),
        f"{prefix}_n": int(len(numeric)),
    }


def _metric_summary(metrics: pd.DataFrame) -> pd.DataFrame:
    grouping = ("model_endpoint", "evaluation_scale", "metric_status")
    records: list[dict[str, Any]] = []
    for keys, group in metrics.groupby(list(grouping), dropna=False, sort=True):
        record = dict(zip(grouping, keys, strict=True))
        record["n_outer_repeats"] = int(len(group))
        record["invalid_prediction_count_total"] = (
            int(group["invalid_prediction_count"].sum())
            if "invalid_prediction_count" in group
            else 0
        )
        for metric in PERFORMANCE_METRICS:
            record.update(_distribution(group[metric], metric))
        records.append(record)
    return pd.DataFrame(records)


def _paired_differences(common_metrics: pd.DataFrame) -> pd.DataFrame:
    """Calculate logKd-trained minus Kd-trained errors on a shared scale."""

    index = [*PAIR_KEYS, "evaluation_scale"]
    logkd = common_metrics.loc[common_metrics["model_endpoint"].eq(LOGKD_ENDPOINT)].copy()
    kd = common_metrics.loc[common_metrics["model_endpoint"].eq(KD_ENDPOINT)].copy()
    keep = [*index, *PERFORMANCE_METRICS]
    merged = logkd.loc[:, keep].merge(
        kd.loc[:, keep], on=index, suffixes=("_logkd", "_kd"), validate="one_to_one"
    )
    for metric in PERFORMANCE_METRICS:
        merged[f"delta_logkd_minus_kd_{metric}"] = (
            merged[f"{metric}_logkd"] - merged[f"{metric}_kd"]
        )
    return merged


def _difference_summary(differences: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for scale, group in differences.groupby("evaluation_scale", dropna=False, sort=True):
        record: dict[str, Any] = {"evaluation_scale": scale, "n_outer_repeats": int(len(group))}
        for metric in PERFORMANCE_METRICS:
            record.update(_distribution(group[f"delta_logkd_minus_kd_{metric}"], f"delta_{metric}"))
        records.append(record)
    return pd.DataFrame(records)


def _write_readme(batch_dir: Path) -> None:
    (batch_dir / "README.md").write_text(
        """# Paired Kd versus logKd random-row comparison

Each outer repeat uses an identical frozen random-row testing set for both
endpoints. The wrapper stops if the two endpoints do not have the same
eligible rows, testing IDs, raw target values, or source-row weights.

## Read these files first

- `summary/native_target_metrics.csv`: each model's usual metrics on the
  endpoint it was trained to predict. These values have different units and
  must not be subtracted or ranked against one another.
- `summary/common_scale_metrics.csv`: both fitted models re-scored on the same
  Kd scale and, where the direct-Kd predictions are all positive, the same
  log10(Kd) scale.
- `summary/common_scale_paired_differences.csv`: per-repeat
  `logKd-trained minus Kd-trained` differences on a shared scale. Negative
  MAE/RMSE favors the logKd-trained model; positive R2/Spearman favors it.
- `summary/common_scale_difference_summary.csv`: means, uncertainty summaries,
  and ranges of those paired differences.
- `detail/testing_predictions.parquet`: canonical held-out predictions for the
  logKd and Kd endpoints. The automatic predicted-versus-actual figures keep
  the endpoint scales separate.
- `figures/common_scale_kd_performance_comparison.png`: direct comparison of
  both trained endpoints after scoring them on the raw Kd scale.
- `figures/common_scale_logkd_performance_comparison.png`: direct comparison
  on the log10(Kd) scale. This figure is skipped if the direct-Kd model has no
  repeat with entirely positive, finite predictions.

The direct-Kd model's log-scale metric is deliberately not calculated for any
repeat having a non-positive Kd prediction, since `log10` is undefined there.
""",
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    if args.resume_batch is not None:
        batch_dir = args.resume_batch.resolve()
        if not batch_dir.is_dir():
            raise FileNotFoundError(f"--resume-batch does not exist or is not a directory: {batch_dir}")
        print(f"Resuming post-processing for existing batch: {batch_dir}", flush=True)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        batch_dir = args.output_root / f"{args.model}_kd_vs_logkd_random_row_comparison_{timestamp}"
        batch_dir.mkdir(parents=True, exist_ok=False)
    variant_root = batch_dir / "variants"
    logkd_root = variant_root / "logkd"
    kd_root = variant_root / "kd"

    if args.resume_batch is None:
        print("Running logKd endpoint on repeated random-row splits.", flush=True)
        _run_child(
            _child_arguments(
                args,
                args.logkd_target,
                logkd_root,
                prediction_target_label="logKd",
            ),
            batch_dir / "audit" / "logkd_child_console.txt",
            args.console_verbosity,
        )
        print("Running Kd endpoint on the matched repeated random-row splits.", flush=True)
        _run_child(
            _child_arguments(
                args,
                args.kd_target,
                kd_root,
                prediction_target_label="Kd (L/g)",
            ),
            batch_dir / "audit" / "kd_child_console.txt",
            args.console_verbosity,
        )

    logkd_batch = _locate_child_batch(logkd_root, args.model)
    kd_batch = _locate_child_batch(kd_root, args.model)
    logkd_summary = _completed_summary(logkd_batch, LOGKD_ENDPOINT)
    kd_summary = _completed_summary(kd_batch, KD_ENDPOINT)
    native_metrics = _paired_native_metrics(logkd_summary, kd_summary)
    common_metrics = _common_scale_metrics(logkd_batch, kd_batch, logkd_summary, kd_summary)
    differences = _paired_differences(common_metrics)
    prediction_detail_path = _write_testing_prediction_detail(
        batch_dir,
        logkd_batch,
        kd_batch,
        logkd_summary,
        kd_summary,
    )

    summary_dir = batch_dir / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)
    native_metrics.to_csv(summary_dir / "native_target_metrics.csv", index=False)
    _metric_summary(native_metrics).to_csv(summary_dir / "native_target_metric_summary.csv", index=False)
    common_metrics.to_csv(summary_dir / "common_scale_metrics.csv", index=False)
    _metric_summary(common_metrics).to_csv(summary_dir / "common_scale_metric_summary.csv", index=False)
    differences.to_csv(summary_dir / "common_scale_paired_differences.csv", index=False)
    _difference_summary(differences).to_csv(
        summary_dir / "common_scale_difference_summary.csv", index=False
    )
    figure_paths: list[str] = []
    figure_failures: list[str] = []
    for evaluation_scale, stem, title in (
        (
            "Kd [L/g]",
            "common_scale_kd_performance_comparison",
            "Kd-trained versus logKd-trained: performance on the Kd scale",
        ),
        (
            "log10(Kd [L/g])",
            "common_scale_logkd_performance_comparison",
            "Kd-trained versus logKd-trained: performance on the log10(Kd) scale",
        ),
    ):
        calculated = common_metrics.loc[
            common_metrics["evaluation_scale"].eq(evaluation_scale)
            & common_metrics["metric_status"].eq("calculated")
        ]
        available_endpoints = set(calculated["model_endpoint"].astype(str))
        if available_endpoints != {LOGKD_ENDPOINT, KD_ENDPOINT}:
            missing = sorted({LOGKD_ENDPOINT, KD_ENDPOINT}.difference(available_endpoints))
            message = (
                f"{evaluation_scale} comparison figure skipped because no calculable "
                f"repeat was available for: {', '.join(missing)}"
            )
            figure_failures.append(message)
            print(f"WARNING: {message}", file=sys.stderr)
            continue
        try:
            figure_paths.extend(generate_aggregate_figure(
                batch_dir,
                plot_type="comparison",
                input_relative_path=Path("summary") / "common_scale_metrics.csv",
                output_stem=stem,
                group_column="model_endpoint",
                groups=(LOGKD_ENDPOINT, KD_ENDPOINT),
                where={"evaluation_scale": evaluation_scale},
                title=title,
            ))
        except RuntimeError as error:
            message = f"{evaluation_scale} comparison figure generation failed: {error}"
            figure_failures.append(message)
            print(f"WARNING: {message}", file=sys.stderr)
    for endpoint, target_label, stem in (
        (LOGKD_ENDPOINT, "logKd", "logkd_predicted_vs_actual"),
        (KD_ENDPOINT, "Kd (L/g)", "kd_predicted_vs_actual"),
    ):
        try:
            figure_paths.extend(generate_aggregate_figure(
                batch_dir,
                plot_type="predicted-vs-actual",
                input_relative_path=prediction_detail_path.relative_to(batch_dir),
                output_stem=stem,
                group_column="model_endpoint",
                where={"model_endpoint": endpoint},
                title="Random-row outer split: predicted versus actual across repeats",
                target_label=target_label,
            ))
        except RuntimeError as error:
            message = f"{endpoint} prediction figure generation failed: {error}"
            figure_failures.append(message)
            print(f"WARNING: {message}", file=sys.stderr)
    (batch_dir / "audit" / "comparison_config.json").write_text(
        json.dumps(
            {
                "comparison": "Kd_vs_log10_Kd",
                "model": args.model,
                "kd_target": args.kd_target,
                "logkd_target": args.logkd_target,
                "data_mode": args.data_mode,
                "split_strategy": "random_row",
                "outer_allocation_method": "random_row",
                "inner_allocation_method": "random_row",
                "pairing": "identical frozen outer testing memberships required",
                "common_scale_metrics": {
                    "Kd": "logKd predictions inverse-transformed with 10 ** prediction",
                    "log10(Kd)": "Kd predictions log-transformed only when all are positive",
                },
                "logkd_child_batch": str(logkd_batch),
                "kd_child_batch": str(kd_batch),
                "testing_prediction_detail": "detail/testing_predictions.parquet",
                "figures": figure_paths,
                "figure_generation_failures": figure_failures,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    _write_readme(batch_dir)
    print(f"Completed paired Kd versus logKd comparison: {batch_dir}", flush=True)


if __name__ == "__main__":
    main()
