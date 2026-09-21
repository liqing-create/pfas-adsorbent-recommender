"""Model-agnostic regression metrics for PFAS logKd modeling."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from .logkd_support import SUPPORT_INDEX_SPECS


LOW_LOGKD_TAIL_THRESHOLD = 0.0
HIGH_LOGKD_TAIL_THRESHOLD = 4.0
FACTOR_2_LOGKD_TOLERANCE = math.log10(2.0)
# ``prediction_context`` writes this column, so every breakdown computed from a
# prediction frame can weight one pre-expansion source record equally.  An
# isotherm expanded into many rows would otherwise drive whichever group it
# lands in, while counting once in the headline metrics beside it.
PREDICTION_WEIGHT_COLUMN = "source_row_weight"


def prediction_weights(predictions: pd.DataFrame) -> pd.Series:
    """Return the per-row source weights a prediction frame must carry.

    Missing weights are an error rather than a silent fall back to equal rows:
    an unweighted breakdown would disagree with the weighted headline metrics
    without saying so anywhere in its output.
    """

    if PREDICTION_WEIGHT_COLUMN not in predictions.columns:
        raise ValueError(
            f"Prediction frames must carry {PREDICTION_WEIGHT_COLUMN!r} so grouped "
            "metrics weight each source record equally; build the frame with "
            "prediction_context, which writes it."
        )
    weights = pd.to_numeric(predictions[PREDICTION_WEIGHT_COLUMN], errors="coerce")
    if not np.isfinite(weights).all() or (weights < 0).any():
        raise ValueError(f"{PREDICTION_WEIGHT_COLUMN} must be finite and nonnegative.")
    return weights


def spearman_corr(y_true, y_pred, sample_weight=None) -> float:
    y_true = pd.Series(np.asarray(y_true, dtype=float)).rank(method="average")
    y_pred = pd.Series(np.asarray(y_pred, dtype=float)).rank(method="average")
    if len(y_true) < 2 or y_true.nunique(dropna=True) < 2 or y_pred.nunique(dropna=True) < 2:
        return float("nan")
    if sample_weight is None:
        weights = np.ones(len(y_true), dtype=float)
    else:
        weights = np.asarray(sample_weight, dtype=float)
        if weights.shape != y_true.to_numpy().shape:
            raise ValueError("sample_weight must have the same shape as y_true.")
        if not np.isfinite(weights).all() or (weights < 0).any() or float(weights.sum()) <= 0:
            raise ValueError("sample_weight must be finite, nonnegative, and have a positive sum.")
    x = y_true.to_numpy(dtype=float)
    y = y_pred.to_numpy(dtype=float)
    x = x - float(np.average(x, weights=weights))
    y = y - float(np.average(y, weights=weights))
    denom = float(np.sqrt(np.sum(weights * x**2) * np.sum(weights * y**2)))
    return float(np.sum(weights * x * y) / denom) if denom > 0 else float("nan")


def metrics_dict(y_true, y_pred, sample_weight=None) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if sample_weight is None:
        weights = np.ones(len(y_true), dtype=float)
    else:
        weights = np.asarray(sample_weight, dtype=float)
        if weights.shape != y_true.shape:
            raise ValueError("sample_weight must have the same shape as y_true.")
        if not np.isfinite(weights).all() or (weights < 0).any() or float(weights.sum()) <= 0:
            raise ValueError("sample_weight must be finite, nonnegative, and have a positive sum.")
    residual = y_pred - y_true
    average = float(np.average(y_true, weights=weights)) if len(y_true) else float("nan")
    ss_res = float(np.sum(weights * residual**2))
    ss_tot = float(np.sum(weights * (y_true - average) ** 2))
    return {
        "n": int(len(y_true)),
        "mae": float(np.average(np.abs(residual), weights=weights)) if len(y_true) else float("nan"),
        "rmse": float(math.sqrt(np.average(residual**2, weights=weights))) if len(y_true) else float("nan"),
        "r2": float(1.0 - ss_res / ss_tot) if len(y_true) > 1 and ss_tot > 0 else float("nan"),
        "spearman": spearman_corr(y_true, y_pred, weights),
    }


def logkd_tail_metrics_dict(
    y_true,
    y_pred,
    sample_weight=None,
    *,
    target_scale: str = "logkd",
    low_threshold: float = LOW_LOGKD_TAIL_THRESHOLD,
    high_threshold: float = HIGH_LOGKD_TAIL_THRESHOLD,
) -> dict[str, float | int | str]:
    """Return low/middle/high true-logKd diagnostic metrics for one prediction set.

    This is a post-prediction diagnostic only: it neither filters rows nor
    affects fitting, split construction, or hyperparameter tuning. Low and
    high tails are defined strictly as ``true logKd < low_threshold`` and
    ``true logKd > high_threshold``. The middle range is the inclusive
    complement, ``low_threshold <= true logKd <= high_threshold``. MAE,
    RMSE, R2, Spearman correlation, and factor-2 accuracy are evaluated on
    the log10(Kd) scale with the supplied source-row weights.

    ``target_scale`` accepts ``"logkd"`` for direct log10(Kd) predictions or
    ``"kd"`` for raw Kd predictions.  Raw-Kd predictions must be positive to
    calculate a log-scale diagnostic; if any prediction in a tail is nonpositive
    or non-finite, that tail's metrics are left missing rather than dropping it.
    """

    if target_scale not in {"logkd", "kd"}:
        raise ValueError("target_scale must be 'logkd' or 'kd'.")
    if low_threshold >= high_threshold:
        raise ValueError("low_threshold must be below high_threshold.")

    y_true_array = np.asarray(y_true, dtype=float)
    y_pred_array = np.asarray(y_pred, dtype=float)
    if y_true_array.shape != y_pred_array.shape:
        raise ValueError("y_true and y_pred must have the same shape.")
    if sample_weight is None:
        weights = np.ones(len(y_true_array), dtype=float)
    else:
        weights = np.asarray(sample_weight, dtype=float)
        if weights.shape != y_true_array.shape:
            raise ValueError("sample_weight must have the same shape as y_true.")
        if not np.isfinite(weights).all() or (weights < 0).any() or float(weights.sum()) <= 0:
            raise ValueError("sample_weight must be finite, nonnegative, and have a positive sum.")

    if target_scale == "logkd":
        y_true_logkd = y_true_array
        y_pred_logkd = y_pred_array
        invalid_prediction = ~np.isfinite(y_pred_logkd)
    else:
        if not np.isfinite(y_true_array).all() or (y_true_array <= 0).any():
            raise ValueError("Raw Kd targets must be finite and strictly positive.")
        y_true_logkd = np.log10(y_true_array)
        invalid_prediction = ~np.isfinite(y_pred_array) | (y_pred_array <= 0)
        y_pred_logkd = np.full(len(y_pred_array), np.nan)
        y_pred_logkd[~invalid_prediction] = np.log10(y_pred_array[~invalid_prediction])

    output: dict[str, float | int | str] = {
        "low_logkd_tail_threshold": float(low_threshold),
        "high_logkd_tail_threshold": float(high_threshold),
        "factor_2_logkd_tolerance": float(FACTOR_2_LOGKD_TOLERANCE),
    }
    stratum_specs = (
        ("low_logkd", y_true_logkd < low_threshold),
        (
            "middle_logkd",
            (y_true_logkd >= low_threshold) & (y_true_logkd <= high_threshold),
        ),
        ("high_logkd", y_true_logkd > high_threshold),
    )
    for prefix, mask in stratum_specs:
        n_rows = int(mask.sum())
        invalid_count = int(invalid_prediction[mask].sum())
        output[f"{prefix}_n"] = n_rows
        output[f"{prefix}_invalid_prediction_count"] = invalid_count
        if n_rows == 0:
            output[f"{prefix}_metric_status"] = (
                "no_testing_rows_in_middle_range"
                if prefix == "middle_logkd"
                else "no_testing_rows_in_tail"
            )
            output[f"{prefix}_mae"] = float("nan")
            output[f"{prefix}_rmse"] = float("nan")
            output[f"{prefix}_r2"] = float("nan")
            output[f"{prefix}_spearman"] = float("nan")
            output[f"{prefix}_factor_2_accuracy"] = float("nan")
            continue
        if invalid_count:
            output[f"{prefix}_metric_status"] = "nonpositive_or_nonfinite_prediction"
            output[f"{prefix}_mae"] = float("nan")
            output[f"{prefix}_rmse"] = float("nan")
            output[f"{prefix}_r2"] = float("nan")
            output[f"{prefix}_spearman"] = float("nan")
            output[f"{prefix}_factor_2_accuracy"] = float("nan")
            continue
        tail_metrics = metrics_dict(
            y_true_logkd[mask], y_pred_logkd[mask], weights[mask]
        )
        absolute_error = np.abs(y_pred_logkd[mask] - y_true_logkd[mask])
        output[f"{prefix}_metric_status"] = "calculated"
        output[f"{prefix}_mae"] = float(tail_metrics["mae"])
        output[f"{prefix}_rmse"] = float(tail_metrics["rmse"])
        output[f"{prefix}_r2"] = float(tail_metrics["r2"])
        output[f"{prefix}_spearman"] = float(tail_metrics["spearman"])
        output[f"{prefix}_factor_2_accuracy"] = float(
            np.average(
                absolute_error <= FACTOR_2_LOGKD_TOLERANCE,
                weights=weights[mask],
            )
        )
    return output


def grouped_metrics(predictions: pd.DataFrame, group_col: str) -> pd.DataFrame:
    rows: list[dict[str, float | str]] = []
    if group_col not in predictions.columns:
        return pd.DataFrame()
    weights = prediction_weights(predictions)
    for value, sub in predictions.groupby(group_col, dropna=False):
        if len(sub) < 3:
            continue
        row = metrics_dict(sub["y_true"], sub["y_pred"], weights.loc[sub.index])
        row[group_col] = value
        rows.append(row)
    return pd.DataFrame(rows)


def support_index_diagnostics(predictions: pd.DataFrame) -> pd.DataFrame:
    """Measure whether higher structural support corresponds to lower test error."""
    if predictions.empty or not {"y_true", "y_pred"}.issubset(predictions.columns):
        return pd.DataFrame()
    work = predictions.copy()
    work["absolute_error"] = (work["y_pred"] - work["y_true"]).abs()
    pfas_col = "pfas_key" if "pfas_key" in work.columns else "PFAS_name"
    rows: list[dict[str, float | int | str | bool]] = []
    strata = [("all_test_rows", work)]
    if "same_pfas_in_training" in work.columns:
        same = work["same_pfas_in_training"].fillna(False).astype(bool)
        strata.append(("unseen_pfas_only", work.loc[~same]))
    for stratum, stratum_data in strata:
        for spec in SUPPORT_INDEX_SPECS:
            score_col = str(spec["score_col"])
            if score_col not in stratum_data.columns:
                continue
            valid = stratum_data[
                [pfas_col, score_col, "absolute_error", PREDICTION_WEIGHT_COLUMN]
            ].dropna()
            if valid.empty:
                continue
            # A weighted mean per PFAS, so a PFAS studied through one expanded
            # isotherm is summarized by that record rather than by its row count.
            totals = valid.assign(
                _weighted_error=valid["absolute_error"] * valid[PREDICTION_WEIGHT_COLUMN]
            ).groupby(pfas_col, as_index=False).agg(
                ad_score=(score_col, "first"),
                _weighted_error_sum=("_weighted_error", "sum"),
                _weight_sum=(PREDICTION_WEIGHT_COLUMN, "sum"),
            )
            totals["mean_absolute_error"] = (
                totals["_weighted_error_sum"] / totals["_weight_sum"]
            )
            pfas_level = totals.drop(columns=["_weighted_error_sum", "_weight_sum"])
            rows.append(
                {
                    "analysis_stratum": stratum,
                    "support_index": str(spec["support_index"]),
                    "score_column": score_col,
                    "higher_score_more_supported": bool(spec["higher_score_more_supported"]),
                    "n_rows": int(len(valid)),
                    "n_pfas": int(len(pfas_level)),
                    "n_unique_scores": int(valid[score_col].nunique()),
                    "row_spearman_score_vs_absolute_error": spearman_corr(
                        valid[score_col],
                        valid["absolute_error"],
                        valid[PREDICTION_WEIGHT_COLUMN],
                    ),
                    # Deliberately unweighted: each PFAS is already one point
                    # here, and the weighting was applied inside its mean above.
                    "pfas_spearman_score_vs_mean_absolute_error": spearman_corr(
                        pfas_level["ad_score"], pfas_level["mean_absolute_error"]
                    ),
                }
            )
    return pd.DataFrame(rows)


def support_score_bin_metrics(predictions: pd.DataFrame, bins: int = 4) -> pd.DataFrame:
    """Descriptive held-out error by structural-score quantile; no threshold is inferred."""
    if predictions.empty or not {"y_true", "y_pred"}.issubset(predictions.columns):
        return pd.DataFrame()
    work = predictions.copy()
    work["absolute_error"] = (work["y_pred"] - work["y_true"]).abs()
    pfas_col = "pfas_key" if "pfas_key" in work.columns else "PFAS_name"
    rows: list[dict[str, float | int | str]] = []
    for spec in SUPPORT_INDEX_SPECS:
        score_col = str(spec["score_col"])
        if score_col not in work.columns:
            continue
        valid = work[
            [pfas_col, score_col, "y_true", "y_pred", "absolute_error", PREDICTION_WEIGHT_COLUMN]
        ].dropna()
        if valid[score_col].nunique() < 2:
            continue
        quantiles = min(bins, int(valid[score_col].nunique()))
        valid = valid.assign(ad_score_bin=pd.qcut(valid[score_col], q=quantiles, duplicates="drop"))
        for label, sub in valid.groupby("ad_score_bin", observed=True):
            rows.append({
                "support_index": str(spec["support_index"]), "score_column": score_col,
                "higher_score_more_supported": bool(spec["higher_score_more_supported"]),
                "score_bin": str(label), "score_min": float(sub[score_col].min()), "score_max": float(sub[score_col].max()),
                "n_rows": int(len(sub)), "n_pfas": int(sub[pfas_col].nunique()),
                **metrics_dict(sub["y_true"], sub["y_pred"], sub[PREDICTION_WEIGHT_COLUMN]),
            })
    return pd.DataFrame(rows)
