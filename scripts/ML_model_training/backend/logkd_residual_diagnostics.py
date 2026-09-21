"""Leakage-conscious residual bias and error-risk diagnostics for logKd models.

This module deliberately separates three questions that were previously mixed:

* candidate screening: can a diagnostic vary within the evaluated scenario?
* signed bias: can available diagnostics predict ``y_pred - y_true``?
* error risk: can those diagnostics predict ``abs(y_pred - y_true)``?

The predictive analyses learn only from inner out-of-fold rows and are evaluated
on untouched outer-test rows.  Group heterogeneity is a separate retrospective
description; it must not be interpreted as a deployable reliability score.
"""

from __future__ import annotations

from math import ceil
from typing import Any, Iterable

import numpy as np
import pandas as pd

from .logkd_metrics import spearman_corr
from .logkd_support import kde_score_column


RIDGE_ALPHAS = (0.01, 0.1, 1.0, 10.0, 100.0)
MIN_CANDIDATE_COVERAGE = 0.80
MIN_MINORITY_COUNT = 5
MIN_MINORITY_FRACTION = 0.01
NEAR_CONSTANT_DOMINANT_FRACTION = 0.99
GROUP_LEVELS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("study", ("study_no",)),
    ("pfas", ("PFAS_name",)),
    ("adsorbent", ("adsorbent_id",)),
    ("pfas_adsorbent", ("PFAS_name", "adsorbent_id")),
)


def support_candidate_manifest() -> pd.DataFrame:
    """Return candidate diagnostics and their scientific roles."""

    rows = [
        ("pfas_identity_novelty", "pfas", "identity", "PFAS absent from the training partition"),
        ("pfas_morgan_novelty", "pfas", "structure", "1 - maximum Morgan-Tanimoto similarity"),
        (kde_score_column("pfas_descriptor"), "pfas", "density", "PFAS-descriptor KDE dissimilarity"),
        ("pfas_missing_fraction", "pfas", "recorded_information", "Missing selected PFAS inputs"),
        ("pfas_numeric_outside_range_fraction", "pfas", "extrapolation", "PFAS numeric inputs outside training ranges"),
        ("pfas_unseen_category_fraction", "pfas", "extrapolation", "PFAS categorical levels unseen in training"),
        ("adsorbent_identity_novelty", "adsorbent", "identity", "Adsorbent absent from the training partition"),
        (kde_score_column("adsorbent"), "adsorbent", "density", "Adsorbent-property KDE dissimilarity"),
        ("adsorbent_missing_fraction", "adsorbent", "recorded_information", "Missing selected adsorbent inputs"),
        ("adsorbent_numeric_outside_range_fraction", "adsorbent", "extrapolation", "Adsorbent numeric inputs outside training ranges"),
        ("adsorbent_unseen_category_fraction", "adsorbent", "extrapolation", "Adsorbent categorical levels unseen in training"),
        (kde_score_column("experimental_conditions"), "experimental_conditions", "density", "Experimental-condition KDE dissimilarity"),
        ("experimental_conditions_missing_fraction", "experimental_conditions", "recorded_information", "Missing selected experimental-condition inputs"),
        ("experimental_conditions_numeric_outside_range_fraction", "experimental_conditions", "extrapolation", "Condition inputs outside training ranges"),
        ("experimental_conditions_unseen_category_fraction", "experimental_conditions", "extrapolation", "Condition categories unseen in training"),
        (kde_score_column("full_input"), "joint", "joint_density", "Full-input KDE dissimilarity"),
        (kde_score_column("full_input_pca5"), "joint", "joint_density", "PCA-reduced full-input KDE dissimilarity"),
    ]
    return pd.DataFrame(rows, columns=["candidate", "block", "construct", "definition"]).assign(
        higher_value="less_observed_support",
        target_used_to_construct=False,
    )


def attach_identity_novelty(
    rows: pd.DataFrame,
    df_training: pd.DataFrame,
    df_query: pd.DataFrame,
) -> pd.DataFrame:
    """Attach PFAS, adsorbent, and Morgan novelty using training information only."""

    from .logkd_data import clean_text, normalize_pfas_key

    out = rows.reset_index(drop=True).copy()
    training_pfas = set(df_training.get("PFAS_name", pd.Series(dtype=object)).map(normalize_pfas_key))
    query_pfas = df_query.get("PFAS_name", pd.Series("", index=df_query.index)).map(normalize_pfas_key)
    adsorbent_column = "adsorbent_identity_key" if "adsorbent_identity_key" in df_training.columns else "adsorbent_id"
    training_adsorbents = set(df_training.get(adsorbent_column, pd.Series(dtype=object)).map(clean_text))
    query_adsorbents = df_query.get(adsorbent_column, pd.Series("", index=df_query.index)).map(clean_text)
    out["pfas_identity_novelty"] = (~query_pfas.isin(training_pfas)).astype(float).to_numpy()
    out["adsorbent_identity_novelty"] = (~query_adsorbents.isin(training_adsorbents)).astype(float).to_numpy()
    similarity = pd.to_numeric(out.get("max_morgan_similarity_to_training"), errors="coerce")
    out["pfas_morgan_novelty"] = 1.0 - similarity
    return out


def public_support_columns() -> list[str]:
    return support_candidate_manifest()["candidate"].tolist()


def compact_support_rows(rows: pd.DataFrame) -> pd.DataFrame:
    """Keep identifiers, prediction evidence, and canonical diagnostic scores."""

    identifiers = [
        "data_split", "validation_fold", "standardized_row_id", "source_row_index",
        "extraction_dataset", "study_no", "PFAS_name", "adsorbent_id",
        "y_true", "y_pred", "residual", "absolute_error", "source_row_weight",
    ]
    keep = [column for column in [*identifiers, *public_support_columns()] if column in rows.columns]
    return rows.loc[:, keep].copy()


def _weights(frame: pd.DataFrame) -> np.ndarray:
    values = pd.to_numeric(
        frame.get("source_row_weight", pd.Series(1.0, index=frame.index)),
        errors="coerce",
    ).fillna(0.0).to_numpy(float)
    return values if np.isfinite(values).all() and values.sum() > 0 else np.ones(len(frame), dtype=float)


def _weighted_mean(values: np.ndarray, weights: np.ndarray) -> float:
    valid = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    return float(np.average(values[valid], weights=weights[valid])) if valid.any() else np.nan


def _candidate_profile(frame: pd.DataFrame, candidate: str, prefix: str) -> dict[str, Any]:
    if candidate not in frame:
        return {
            f"{prefix}_coverage_fraction": 0.0,
            f"{prefix}_observed_count": 0,
            f"{prefix}_distinct_count": 0,
            f"{prefix}_dominant_value_fraction": np.nan,
            f"{prefix}_q05": np.nan,
            f"{prefix}_median": np.nan,
            f"{prefix}_q95": np.nan,
            f"{prefix}_variation_status": "missing_column",
        }
    values = pd.to_numeric(frame[candidate], errors="coerce")
    observed = values.dropna()
    coverage = float(len(observed) / len(frame)) if len(frame) else 0.0
    distinct = int(observed.nunique())
    dominant = float(observed.value_counts(normalize=True).max()) if len(observed) else np.nan
    q05, median, q95 = (
        (float(observed.quantile(0.05)), float(observed.median()), float(observed.quantile(0.95)))
        if len(observed)
        else (np.nan, np.nan, np.nan)
    )
    minority_count = int(len(observed) - observed.value_counts().max()) if len(observed) else 0
    minority_fraction = float(1.0 - dominant) if np.isfinite(dominant) else 0.0
    if coverage < MIN_CANDIDATE_COVERAGE:
        status = "insufficient_coverage"
    elif distinct < 2:
        status = "constant"
    elif (
        np.isclose(q05, q95, equal_nan=False)
        or dominant >= NEAR_CONSTANT_DOMINANT_FRACTION
        or (distinct <= 10 and (minority_count < MIN_MINORITY_COUNT or minority_fraction < MIN_MINORITY_FRACTION))
    ):
        status = "near_constant"
    else:
        status = "row_varying"
    return {
        f"{prefix}_coverage_fraction": coverage,
        f"{prefix}_observed_count": int(len(observed)),
        f"{prefix}_distinct_count": distinct,
        f"{prefix}_dominant_value_fraction": dominant,
        f"{prefix}_q05": q05,
        f"{prefix}_median": median,
        f"{prefix}_q95": q95,
        f"{prefix}_variation_status": status,
    }


def screen_residual_candidates(
    inner_oof: pd.DataFrame,
    outer_test: pd.DataFrame,
    candidate_manifest: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Classify diagnostics before modeling, without consulting either outcome."""

    records: list[dict[str, Any]] = []
    manifest = support_candidate_manifest() if candidate_manifest is None else candidate_manifest.copy()
    required = {"candidate", "block", "construct", "definition"}
    missing = required.difference(manifest.columns)
    if missing:
        raise ValueError(f"Candidate manifest is missing required columns: {sorted(missing)}")
    for metadata in manifest.itertuples(index=False):
        metadata_values = metadata._asdict()
        candidate = str(metadata.candidate)
        inner = _candidate_profile(inner_oof, candidate, "inner")
        outer = _candidate_profile(outer_test, candidate, "outer")
        inner_status = str(inner["inner_variation_status"])
        outer_status = str(outer["outer_variation_status"])
        if inner_status == "row_varying" and outer_status == "row_varying":
            role = "row_level_predictor"
            eligible = True
            reason = "varies with adequate coverage in both inner OOF and outer-test rows"
        elif outer_status in {"constant", "near_constant"}:
            role = "scenario_descriptor"
            eligible = False
            reason = "does not vary enough within this outer-test scenario to explain row-to-row residual differences"
        else:
            role = "unusable"
            eligible = False
            reason = f"inner={inner_status}; outer={outer_status}"
        records.append({
            **metadata_values,
            "analysis_role": role,
            "eligible_for_row_model": eligible,
            "screen_reason": reason,
            **inner,
            **outer,
        })
    return pd.DataFrame(records)


def _splitter(frame: pd.DataFrame):
    from sklearn.model_selection import GroupKFold, KFold

    groups = frame.get("study_no", pd.Series("", index=frame.index)).fillna("").astype(str)
    if groups.nunique() >= 3:
        splitter = GroupKFold(n_splits=min(5, int(groups.nunique())))
        return splitter, groups
    n_splits = min(5, len(frame))
    if n_splits < 2:
        return None, None
    return KFold(n_splits=n_splits, shuffle=True, random_state=0), None


def _ridge_predictions(
    inner: pd.DataFrame,
    outer: pd.DataFrame,
    features: list[str],
    target: str,
) -> tuple[np.ndarray, dict[str, Any]]:
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    X = inner.reindex(columns=features).apply(pd.to_numeric, errors="coerce")
    X_outer = outer.reindex(columns=features).apply(pd.to_numeric, errors="coerce")
    y = pd.to_numeric(inner[target], errors="coerce").to_numpy(float)
    weights = _weights(inner)
    splitter, groups = _splitter(inner)
    best_alpha, best_loss = RIDGE_ALPHAS[0], np.inf
    if splitter is not None:
        split_iterator = splitter.split(X, y, groups) if groups is not None else splitter.split(X, y)
        cached_splits = list(split_iterator)
        for alpha in RIDGE_ALPHAS:
            losses: list[float] = []
            for train_index, validation_index in cached_splits:
                pipe = Pipeline([
                    ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
                    ("scale", StandardScaler()),
                    ("model", Ridge(alpha=alpha)),
                ])
                pipe.fit(X.iloc[train_index], y[train_index], model__sample_weight=weights[train_index])
                estimate = pipe.predict(X.iloc[validation_index])
                losses.append(float(np.average(np.abs(estimate - y[validation_index]), weights=weights[validation_index])))
            loss = float(np.mean(losses))
            if loss < best_loss:
                best_alpha, best_loss = alpha, loss
    final = Pipeline([
        ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
        ("scale", StandardScaler()),
        ("model", Ridge(alpha=best_alpha)),
    ])
    final.fit(X, y, model__sample_weight=weights)
    return np.asarray(final.predict(X_outer), dtype=float), {
        "ridge_alpha": float(best_alpha),
        "nonlinear_min_samples_leaf": np.nan,
    }


def _nonlinear_predictions(
    inner: pd.DataFrame,
    outer: pd.DataFrame,
    features: list[str],
    target: str,
) -> tuple[np.ndarray, dict[str, Any]]:
    from sklearn.ensemble import ExtraTreesRegressor
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline

    X = inner.reindex(columns=features).apply(pd.to_numeric, errors="coerce")
    X_outer = outer.reindex(columns=features).apply(pd.to_numeric, errors="coerce")
    y = pd.to_numeric(inner[target], errors="coerce").to_numpy(float)
    weights = _weights(inner)
    min_samples_leaf = max(5, min(25, int(ceil(0.02 * len(inner)))))
    model = Pipeline([
        ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
        ("model", ExtraTreesRegressor(
            n_estimators=160,
            max_depth=6,
            min_samples_leaf=min_samples_leaf,
            max_features=1.0,
            random_state=0,
            n_jobs=1,
        )),
    ])
    model.fit(X, y, model__sample_weight=weights)
    return np.asarray(model.predict(X_outer), dtype=float), {
        "ridge_alpha": np.nan,
        "nonlinear_min_samples_leaf": int(min_samples_leaf),
    }


def _elasticnet_predictions(
    inner: pd.DataFrame,
    outer: pd.DataFrame,
    features: list[str],
    target: str,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Fit a sparse, nested-CV linear challenger for feature-resolved support."""
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import ElasticNet
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    X = inner.reindex(columns=features).apply(pd.to_numeric, errors="coerce")
    X_outer = outer.reindex(columns=features).apply(pd.to_numeric, errors="coerce")
    y = pd.to_numeric(inner[target], errors="coerce").to_numpy(float)
    weights = _weights(inner)
    splitter, groups = _splitter(inner)
    # A small, fixed log-scale grid keeps the challenger reproducible without
    # turning every diagnostic run into a large model-search exercise.
    alpha_grid = (0.003, 0.03, 0.3)
    l1_ratio_grid = (0.2, 0.8)
    best_alpha, best_l1_ratio, best_loss = alpha_grid[-1], l1_ratio_grid[0], np.inf
    if splitter is not None:
        split_iter = (
            splitter.split(X, y, groups)
            if groups is not None
            else splitter.split(X, y)
        )
        folds = list(split_iter)
        for alpha in alpha_grid:
            for l1_ratio in l1_ratio_grid:
                losses: list[float] = []
                for train_index, validation_index in folds:
                    pipe = Pipeline([
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scale", StandardScaler()),
                        ("model", ElasticNet(alpha=alpha, l1_ratio=l1_ratio, max_iter=20_000, random_state=0)),
                    ])
                    pipe.fit(X.iloc[train_index], y[train_index], model__sample_weight=weights[train_index])
                    estimate = pipe.predict(X.iloc[validation_index])
                    losses.append(float(np.average(np.abs(estimate - y[validation_index]), weights=weights[validation_index])))
                loss = float(np.mean(losses))
                if loss < best_loss:
                    best_alpha, best_l1_ratio, best_loss = alpha, l1_ratio, loss
    final = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("model", ElasticNet(alpha=best_alpha, l1_ratio=best_l1_ratio, max_iter=20_000, random_state=0)),
    ])
    final.fit(X, y, model__sample_weight=weights)
    coefficients = np.asarray(final.named_steps["model"].coef_, dtype=float)
    selected = [feature for feature, coefficient in zip(features, coefficients) if abs(coefficient) > 1e-10]
    return np.asarray(final.predict(X_outer), dtype=float), {
        "ridge_alpha": np.nan,
        "nonlinear_min_samples_leaf": np.nan,
        "elasticnet_alpha": float(best_alpha),
        "elasticnet_l1_ratio": float(best_l1_ratio),
        "elasticnet_selected_candidate_columns": ";".join(selected),
    }


def _evaluation_record(
    *,
    outcome: str,
    target_column: str,
    model_family: str,
    interpretation: str,
    estimate: np.ndarray,
    inner: pd.DataFrame,
    outer: pd.DataFrame,
    features: list[str],
    baseline: np.ndarray,
    ridge_alpha: float = np.nan,
    nonlinear_min_samples_leaf: float = np.nan,
    elasticnet_alpha: float = np.nan,
    elasticnet_l1_ratio: float = np.nan,
    elasticnet_selected_candidate_columns: str = "",
) -> dict[str, Any]:
    y = pd.to_numeric(outer[target_column], errors="coerce").to_numpy(float)
    weights = _weights(outer)
    if outcome == "absolute_error_risk":
        estimate = np.clip(estimate, 0.0, None)
    model_errors = estimate - y
    baseline_errors = baseline - y
    model_ss = float(np.sum(weights * model_errors**2))
    baseline_ss = float(np.sum(weights * baseline_errors**2))
    model_mae = float(np.average(np.abs(model_errors), weights=weights))
    baseline_mae = float(np.average(np.abs(baseline_errors), weights=weights))
    model_rmse = float(np.sqrt(np.average(model_errors**2, weights=weights)))
    baseline_rmse = float(np.sqrt(np.average(baseline_errors**2, weights=weights)))
    return {
        "outcome": outcome,
        "target_definition": "y_pred - y_true" if outcome == "signed_bias" else "abs(y_pred - y_true)",
        "model_family": model_family,
        "model_interpretation": interpretation,
        "candidate_count": len(features),
        "candidate_columns": ";".join(features),
        "n_inner_rows": len(inner),
        "n_outer_rows": len(outer),
        "inner_target_weighted_mean": _weighted_mean(
            pd.to_numeric(inner[target_column], errors="coerce").to_numpy(float), _weights(inner)
        ),
        "outer_target_weighted_mean": _weighted_mean(y, weights),
        "outer_predicted_target_weighted_mean": _weighted_mean(estimate, weights),
        "prediction_mae": model_mae,
        "constant_mean_baseline_mae": baseline_mae,
        "mae_improvement_vs_constant_mean": baseline_mae - model_mae,
        "prediction_rmse": model_rmse,
        "constant_mean_baseline_rmse": baseline_rmse,
        "rmse_improvement_vs_constant_mean": baseline_rmse - model_rmse,
        "out_of_sample_r2_vs_inner_mean_constant": 1.0 - model_ss / baseline_ss if baseline_ss > 0 else np.nan,
        "predicted_vs_observed_weighted_spearman": spearman_corr(y, estimate, weights),
        "mean_unexplained_target_after_model": _weighted_mean(y - estimate, weights),
        "ridge_alpha": ridge_alpha,
        "nonlinear_min_samples_leaf": nonlinear_min_samples_leaf,
        "elasticnet_alpha": elasticnet_alpha,
        "elasticnet_l1_ratio": elasticnet_l1_ratio,
        "elasticnet_selected_candidate_columns": elasticnet_selected_candidate_columns,
    }


def evaluate_outer_residual_analysis(
    inner_oof: pd.DataFrame,
    outer_test: pd.DataFrame,
    candidate_manifest: pd.DataFrame | None = None,
    include_elasticnet: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Screen candidates, then evaluate separate signed-bias and risk models."""

    required = {"residual", "absolute_error"}
    missing_inner = required.difference(inner_oof.columns)
    missing_outer = required.difference(outer_test.columns)
    if missing_inner or missing_outer:
        raise ValueError(
            "Residual analysis requires signed residual and absolute error in both datasets; "
            f"missing inner={sorted(missing_inner)}, outer={sorted(missing_outer)}"
        )
    manifest = support_candidate_manifest() if candidate_manifest is None else candidate_manifest.copy()
    screening = screen_residual_candidates(inner_oof, outer_test, manifest)
    features = screening.loc[screening["eligible_for_row_model"].eq(True), "candidate"].astype(str).tolist()
    evaluations: list[dict[str, Any]] = []
    for outcome, target_column in (
        ("signed_bias", "residual"),
        ("absolute_error_risk", "absolute_error"),
    ):
        inner_target = pd.to_numeric(inner_oof[target_column], errors="coerce").to_numpy(float)
        baseline_value = _weighted_mean(inner_target, _weights(inner_oof))
        baseline = np.full(len(outer_test), baseline_value)
        evaluations.append(_evaluation_record(
            outcome=outcome,
            target_column=target_column,
            model_family="constant_mean",
            interpretation="reference prediction using the inner-OOF weighted mean only",
            estimate=baseline,
            inner=inner_oof,
            outer=outer_test,
            features=[],
            baseline=baseline,
        ))
        if not features:
            continue
        ridge_estimate, ridge_meta = _ridge_predictions(inner_oof, outer_test, features, target_column)
        evaluations.append(_evaluation_record(
            outcome=outcome,
            target_column=target_column,
            model_family="ridge_linear",
            interpretation="additive linear benchmark after candidate screening",
            estimate=ridge_estimate,
            inner=inner_oof,
            outer=outer_test,
            features=features,
            baseline=baseline,
            **ridge_meta,
        ))
        if include_elasticnet:
            elasticnet_estimate, elasticnet_meta = _elasticnet_predictions(
                inner_oof,
                outer_test,
                features,
                target_column,
            )
            evaluations.append(_evaluation_record(
                outcome=outcome,
                target_column=target_column,
                model_family="elasticnet_sparse_linear",
                interpretation="sparse linear challenger for feature-resolved diagnostic selection",
                estimate=elasticnet_estimate,
                inner=inner_oof,
                outer=outer_test,
                features=features,
                baseline=baseline,
                **elasticnet_meta,
            ))
        nonlinear_estimate, nonlinear_meta = _nonlinear_predictions(inner_oof, outer_test, features, target_column)
        evaluations.append(_evaluation_record(
            outcome=outcome,
            target_column=target_column,
            model_family="extra_trees_nonlinear",
            interpretation="prespecified nonlinear and interaction challenger after candidate screening",
            estimate=nonlinear_estimate,
            inner=inner_oof,
            outer=outer_test,
            features=features,
            baseline=baseline,
            **nonlinear_meta,
        ))

    associations: list[dict[str, Any]] = []
    outer_weights = _weights(outer_test)
    residual = pd.to_numeric(outer_test["residual"], errors="coerce")
    absolute_error = pd.to_numeric(outer_test["absolute_error"], errors="coerce")
    for screen in screening.itertuples(index=False):
        candidate = str(screen.candidate)
        values = pd.to_numeric(
            outer_test.get(candidate, pd.Series(np.nan, index=outer_test.index)),
            errors="coerce",
        )
        estimable = str(screen.outer_variation_status) == "row_varying"
        valid = values.notna() & residual.notna() & absolute_error.notna()
        associations.append({
            "candidate": candidate,
            "block": screen.block,
            "construct": screen.construct,
            "analysis_role": screen.analysis_role,
            "association_status": "estimated" if estimable and valid.sum() >= 3 else "not_estimable",
            "n_outer_rows_with_score": int(valid.sum()),
            "weighted_spearman_with_signed_residual": (
                spearman_corr(values.loc[valid], residual.loc[valid], outer_weights[valid.to_numpy()])
                if estimable and valid.sum() >= 3 else np.nan
            ),
            "weighted_spearman_with_absolute_error": (
                spearman_corr(values.loc[valid], absolute_error.loc[valid], outer_weights[valid.to_numpy()])
                if estimable and valid.sum() >= 3 else np.nan
            ),
        })
    return screening, pd.DataFrame(evaluations), pd.DataFrame(associations)


def _quantile(values: pd.Series, probability: float) -> float:
    numeric = pd.to_numeric(values, errors="coerce").dropna()
    return float(numeric.quantile(probability)) if len(numeric) else np.nan


def _scenario_keys(frame: pd.DataFrame) -> list[str]:
    keys = ["split_strategy"]
    if "outer_allocation_method" in frame.columns:
        keys.append("outer_allocation_method")
    if "diagnostic_representation" in frame.columns:
        keys.append("diagnostic_representation")
    return keys


def summarize_candidate_screening(by_run: pd.DataFrame) -> pd.DataFrame:
    """Summarize diagnostic availability and variation separately by scenario."""

    if by_run.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    keys = [*_scenario_keys(by_run), "candidate", "block", "construct"]
    for key, group in by_run.groupby(keys, dropna=False, sort=False):
        roles = group["analysis_role"].astype(str)
        rows.append({
            **dict(zip(keys, key)),
            "n_runs": len(group),
            "fraction_runs_row_level_predictor": float(roles.eq("row_level_predictor").mean()),
            "fraction_runs_scenario_descriptor": float(roles.eq("scenario_descriptor").mean()),
            "fraction_runs_unusable": float(roles.eq("unusable").mean()),
            "median_inner_coverage_fraction": _quantile(group["inner_coverage_fraction"], 0.5),
            "median_outer_coverage_fraction": _quantile(group["outer_coverage_fraction"], 0.5),
            "median_outer_distinct_count": _quantile(group["outer_distinct_count"], 0.5),
        })
    return pd.DataFrame(rows)


def summarize_residual_model_runs(by_run: pd.DataFrame) -> pd.DataFrame:
    """Summarize held-out model evidence without issuing retain/remove decisions."""

    if by_run.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    keys = [
        *_scenario_keys(by_run),
        "outcome", "target_definition", "model_family", "model_interpretation",
    ]
    for key, group in by_run.groupby(keys, dropna=False, sort=False):
        improvement = pd.to_numeric(group["mae_improvement_vs_constant_mean"], errors="coerce")
        r2 = pd.to_numeric(group["out_of_sample_r2_vs_inner_mean_constant"], errors="coerce")
        rows.append({
            **dict(zip(keys, key)),
            "n_runs": len(group),
            "median_candidate_count": _quantile(group["candidate_count"], 0.5),
            "median_prediction_mae": _quantile(group["prediction_mae"], 0.5),
            "median_mae_improvement_vs_constant_mean": _quantile(improvement, 0.5),
            "q25_mae_improvement_vs_constant_mean": _quantile(improvement, 0.25),
            "q75_mae_improvement_vs_constant_mean": _quantile(improvement, 0.75),
            "fraction_runs_mae_better_than_constant_mean": float(improvement.gt(0).mean()),
            "median_out_of_sample_r2_vs_inner_mean_constant": _quantile(r2, 0.5),
            "q25_out_of_sample_r2_vs_inner_mean_constant": _quantile(r2, 0.25),
            "q75_out_of_sample_r2_vs_inner_mean_constant": _quantile(r2, 0.75),
            "fraction_runs_positive_r2": float(r2.gt(0).mean()),
            "median_predicted_vs_observed_weighted_spearman": _quantile(
                group["predicted_vs_observed_weighted_spearman"], 0.5
            ),
            "median_mean_unexplained_target_after_model": _quantile(
                group["mean_unexplained_target_after_model"], 0.5
            ),
        })
    return pd.DataFrame(rows)


def summarize_candidate_associations(by_run: pd.DataFrame) -> pd.DataFrame:
    """Summarize direct candidate associations for interpretation, not causality."""

    if by_run.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    keys = [*_scenario_keys(by_run), "candidate", "block", "construct"]
    for key, group in by_run.groupby(keys, dropna=False, sort=False):
        signed = pd.to_numeric(group["weighted_spearman_with_signed_residual"], errors="coerce")
        risk = pd.to_numeric(group["weighted_spearman_with_absolute_error"], errors="coerce")
        rows.append({
            **dict(zip(keys, key)),
            "n_runs": len(group),
            "n_runs_signed_association_estimable": int(signed.notna().sum()),
            "median_weighted_spearman_with_signed_residual": _quantile(signed, 0.5),
            "fraction_estimable_signed_associations_positive": float(signed.dropna().gt(0).mean()) if signed.notna().any() else np.nan,
            "n_runs_risk_association_estimable": int(risk.notna().sum()),
            "median_weighted_spearman_with_absolute_error": _quantile(risk, 0.5),
            "fraction_estimable_risk_associations_positive": float(risk.dropna().gt(0).mean()) if risk.notna().any() else np.nan,
        })
    return pd.DataFrame(rows)


def _one_way_random_intercept_components(
    source_residuals_by_group: list[np.ndarray],
) -> tuple[float, float, float]:
    """Estimate a one-way random-intercept variance decomposition by moments."""

    groups = [values[np.isfinite(values)] for values in source_residuals_by_group]
    groups = [values for values in groups if len(values)]
    group_count = len(groups)
    source_count = int(sum(len(values) for values in groups))
    if group_count < 2 or source_count <= group_count:
        return np.nan, np.nan, np.nan
    combined = np.concatenate(groups)
    grand_mean = float(np.mean(combined))
    group_means = np.asarray([float(np.mean(values)) for values in groups])
    group_sizes = np.asarray([len(values) for values in groups], dtype=float)
    between_ss = float(np.sum(group_sizes * (group_means - grand_mean) ** 2))
    within_ss = float(sum(np.sum((values - mean) ** 2) for values, mean in zip(groups, group_means)))
    between_ms = between_ss / (group_count - 1)
    within_ms = within_ss / (source_count - group_count)
    effective_group_size = (
        source_count - float(np.sum(group_sizes**2)) / source_count
    ) / (group_count - 1)
    if effective_group_size <= 0:
        return np.nan, np.nan, np.nan
    between_variance = max((between_ms - within_ms) / effective_group_size, 0.0)
    within_variance = max(within_ms, 0.0)
    total = between_variance + within_variance
    icc = between_variance / total if total > 0 else np.nan
    return float(between_variance), float(within_variance), float(icc)


def residual_group_heterogeneity(observed_support: pd.DataFrame) -> pd.DataFrame:
    """Describe retrospective signed-residual clustering within each outer run."""

    required = {"data_split", "residual", "source_row_weight", "source_row_index"}
    if observed_support.empty or not required.issubset(observed_support.columns):
        return pd.DataFrame()
    testing = observed_support.loc[observed_support["data_split"].eq("testing")].copy()
    context = [
        column for column in (
            "split_strategy", "outer_allocation_method", "run_id", "outer_assignment_id",
            "outer_repeat_id", "candidate_id", "random_seed",
        ) if column in testing
    ]
    if not context:
        testing["run_id"] = "single_run"
        context = ["run_id"]
    records: list[dict[str, Any]] = []
    for run_key, run in testing.groupby(context, dropna=False, sort=False):
        run_values = run_key if isinstance(run_key, tuple) else (run_key,)
        run_context = dict(zip(context, run_values))
        residual = pd.to_numeric(run["residual"], errors="coerce").to_numpy(float)
        weights = _weights(run)
        valid_run = np.isfinite(residual) & np.isfinite(weights) & (weights > 0)
        if not valid_run.any():
            continue
        grand_mean = _weighted_mean(residual[valid_run], weights[valid_run])
        total_ss = float(np.sum(weights[valid_run] * (residual[valid_run] - grand_mean) ** 2))
        for level, columns in GROUP_LEVELS:
            if not set(columns).issubset(run.columns):
                continue
            work = run.loc[valid_run, [*columns, "residual", "source_row_weight", "source_row_index"]].copy()
            group_means: list[float] = []
            group_weights: list[float] = []
            eligible_sign_consistency: list[bool] = []
            eligible_group_biases: list[float] = []
            source_residuals_by_group: list[np.ndarray] = []
            for _, group in work.groupby(list(columns), dropna=False, sort=False):
                group_residual = pd.to_numeric(group["residual"], errors="coerce").to_numpy(float)
                group_weight = _weights(group)
                mean = _weighted_mean(group_residual, group_weight)
                group_means.append(mean)
                group_weights.append(float(group_weight.sum()))
                source_means = group.groupby("source_row_index", dropna=False)["residual"].mean().dropna()
                source_residuals_by_group.append(source_means.to_numpy(float))
                if len(source_means) >= 3:
                    positive = float(source_means.gt(0).mean())
                    negative = float(source_means.lt(0).mean())
                    eligible_sign_consistency.append(max(positive, negative) >= 0.80)
                    eligible_group_biases.append(abs(mean))
            means = np.asarray(group_means, dtype=float)
            group_weight_array = np.asarray(group_weights, dtype=float)
            between_ss = float(np.sum(group_weight_array * (means - grand_mean) ** 2))
            random_intercept_variance, within_variance, random_intercept_icc = (
                _one_way_random_intercept_components(source_residuals_by_group)
            )
            records.append({
                **run_context,
                "group_level": level,
                "group_columns": ";".join(columns),
                "n_groups": len(means),
                "n_groups_with_at_least_3_source_rows": len(eligible_sign_consistency),
                "descriptive_between_group_residual_variance_fraction": between_ss / total_ss if total_ss > 0 else np.nan,
                "random_intercept_between_group_variance": random_intercept_variance,
                "random_intercept_within_group_variance": within_variance,
                "random_intercept_icc": random_intercept_icc,
                "random_intercept_estimator": "one-way method-of-moments on source-row residual means",
                "median_absolute_group_mean_residual": float(np.median(eligible_group_biases)) if eligible_group_biases else np.nan,
                "fraction_eligible_groups_at_least_80pct_same_sign": float(np.mean(eligible_sign_consistency)) if eligible_sign_consistency else np.nan,
                "interpretation_scope": "retrospective heterogeneity only; not a row-level reliability predictor or causal attribution",
            })
    return pd.DataFrame(records)


def summarize_group_heterogeneity(by_run: pd.DataFrame) -> pd.DataFrame:
    if by_run.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    keys = [*_scenario_keys(by_run), "group_level", "group_columns"]
    for key, group in by_run.groupby(keys, dropna=False, sort=False):
        rows.append({
            **dict(zip(keys, key)),
            "n_runs": len(group),
            "median_n_groups": _quantile(group["n_groups"], 0.5),
            "median_descriptive_between_group_residual_variance_fraction": _quantile(
                group["descriptive_between_group_residual_variance_fraction"], 0.5
            ),
            "q25_descriptive_between_group_residual_variance_fraction": _quantile(
                group["descriptive_between_group_residual_variance_fraction"], 0.25
            ),
            "q75_descriptive_between_group_residual_variance_fraction": _quantile(
                group["descriptive_between_group_residual_variance_fraction"], 0.75
            ),
            "median_random_intercept_between_group_variance": _quantile(
                group["random_intercept_between_group_variance"], 0.5
            ),
            "median_random_intercept_within_group_variance": _quantile(
                group["random_intercept_within_group_variance"], 0.5
            ),
            "median_random_intercept_icc": _quantile(group["random_intercept_icc"], 0.5),
            "median_absolute_group_mean_residual": _quantile(group["median_absolute_group_mean_residual"], 0.5),
            "median_fraction_eligible_groups_at_least_80pct_same_sign": _quantile(
                group["fraction_eligible_groups_at_least_80pct_same_sign"], 0.5
            ),
            "interpretation_scope": "retrospective heterogeneity only; not a row-level reliability predictor or causal attribution",
        })
    return pd.DataFrame(rows)
