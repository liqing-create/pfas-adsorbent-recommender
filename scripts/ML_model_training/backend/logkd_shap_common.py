"""Small shared utilities for the two logKd SHAP workflows.

The outer-robustness workflow explains independently evaluated outer models.
The deployment workflow explains the all-data ensemble used by the recommender.
Keeping the mechanics here prevents the two workflows from diverging while
keeping their inputs and outputs deliberately separate.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ANALYSIS_FORMAT = "logkd_shap_v4"
PREPROCESS_STEP = "preprocess"
MODEL_STEP = "model"
# SHAP values reconstruct predictions to floating-point precision. This retains
# the additivity guard while allowing a negligible 0.0001-logKd numerical gap.
SHAP_ADDITIVITY_TOLERANCE = 1e-4

# Shared manuscript palette from Manuscript/Results/plot scripts/plot_sankey.py.
# Keep the SHAP figures visually aligned with the paper's Fig. 1 Sankey.
MANUSCRIPT_PALETTE = (
    "#12A8D8",  # PFCA cyan
    "#E64B5D",  # PFSA rose
    "#8CCB78",  # PFECA green
    "#F2D34F",  # PASF yellow
    "#B784CC",  # Lavender
    "#79BFE3",  # Sky blue
    "#E895B7",  # Soft rose
    "#77C8BC",  # Soft teal
    "#B4B9C1",  # Light slate
    "#A69ACF",  # Periwinkle
)
IMPORTANCE_BAR_COLOR = MANUSCRIPT_PALETTE[0]
# Match the near-square 1961 x 1754 px performance-comparison figure while
# retaining enough vertical space for 15 ranked features and a top legend.
MANUSCRIPT_FIGURE_ASPECT_RATIO = 1961 / 1754
SHAP_IMPORTANCE_FIGURE_WIDTH = 8.8
# The beeswarm keeps the bar figure's plotting height, and widens only enough to
# hold the feature-value colour bar without narrowing the ranked feature rows.
SHAP_BEESWARM_FIGURE_WIDTH = 9.8
# Low-to-high feature values follow the conventional SHAP cool-to-warm ramp,
# expressed with the manuscript palette's cyan, lavender, and rose.
BEESWARM_COLORMAP_STOPS = (MANUSCRIPT_PALETTE[0], MANUSCRIPT_PALETTE[4], MANUSCRIPT_PALETTE[1])
# Categorical features and missing values carry no low/high position on the ramp.
BEESWARM_UNSCALED_COLOR = MANUSCRIPT_PALETTE[8]
# Feature-value colours are scaled between these percentiles so that a single
# extreme value cannot flatten the colour contrast of the remaining rows.
BEESWARM_COLOR_PERCENTILES = (5.0, 95.0)
# Marker area, in points squared, for a row carrying the mean source-row weight.
# Weighted rows scale this linearly, so a source record expanded into several
# model rows spends the same total ink as an unexpanded one and the beeswarm
# reads on the same footing as the weighted importance bars beside it.
BEESWARM_POINT_AREA = 11.0
# A record expanded into very many rows would otherwise shrink out of sight, so
# its points stop here.  The equal-ink property is exact only above this floor.
BEESWARM_MIN_POINT_AREA = 2.0
# The manuscript's multi-panel Figure 4 uses a compact, panel-ready beeswarm.
# Enlarging every marker by the same factor preserves the evidence-balancing
# weighting while making the swarm legible after panel assembly.
PANEL_BEESWARM_POINT_AREA_SCALE = 1.45
# scikit-learn names the category it groups below ``min_frequency`` this way.
INFREQUENT_ONEHOT_LEVEL = "infrequent_sklearn"


@dataclass
class EvaluationRun:
    """One saved outer-evaluation model and the data it was fitted against."""

    run_dir: Path
    pipeline: Any
    frame: pd.DataFrame
    split: pd.DataFrame
    selected: list[str]
    numeric: list[str]
    categorical: list[str]


@dataclass
class DeploymentBundle:
    """The all-data ensemble and its exact feature/data contract."""

    directory: Path
    models: list[Any]
    frame: pd.DataFrame
    selected: list[str]
    numeric: list[str]
    categorical: list[str]


@dataclass
class PipelineExplanation:
    """Additive SHAP decomposition for one fitted pipeline."""

    X: pd.DataFrame
    # The transformed design matrix the model actually saw.  Retaining it lets a
    # beeswarm colour one-hot rows by their exact 0/1 indicator, including the
    # encoder's imputed and infrequent-grouped categories.
    X_processed: np.ndarray
    processed_values: np.ndarray
    feature_map: pd.DataFrame
    original_values: pd.DataFrame
    base_values: np.ndarray
    predictions: np.ndarray
    additivity_predictions: np.ndarray
    additivity_reference: str
    backend: str
    max_additivity_error: float


@dataclass
class BeeswarmDisplay:
    """One explained model expressed as the rows a beeswarm actually draws."""

    # Signed SHAP values, one column per plotted row.
    values: pd.DataFrame
    # Matching colour positions on [0, 1]; NaN marks a point with no value.
    color_positions: pd.DataFrame
    # display_feature, parent_feature, level, row_kind for every plotted row.
    layout: pd.DataFrame
    # One weight per explained row, sized so the plotted points of a single
    # pre-expansion source record carry the same total marker area as one
    # unexpanded record.  Required: a beeswarm that silently fell back to equal
    # point sizes would contradict the weighted importance bars beside it, so
    # there is deliberately no unweighted path to reach by omission.
    weights: pd.Series


@dataclass
class EnsembleExplanation:
    """Exact decomposition of the arithmetic mean of several pipelines."""

    X: pd.DataFrame
    feature_map: pd.DataFrame
    original_values: pd.DataFrame
    base_values: np.ndarray
    predictions: np.ndarray
    member_mean_abs_values: pd.DataFrame
    backend: str
    max_additivity_error: float


def load_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8-sig"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in {path}: {exc}") from exc


def require_mapping(mapping: dict[str, Any], key: str, *, label: str = "configuration") -> dict[str, Any]:
    value = mapping.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"{label} requires an object named {key!r}.")
    return value


def require_keys(mapping: dict[str, Any], keys: tuple[str, ...], *, label: str) -> None:
    missing = [key for key in keys if key not in mapping]
    if missing:
        raise ValueError(f"{label} is missing required keys: {missing}")


def reconstruction_from_config(config: dict[str, Any]) -> dict[str, Any]:
    reconstruction = require_mapping(config, "analysis_reconstruction", label="run_config.json")
    require_keys(
        reconstruction,
        (
            "format",
            "input_path",
            "sheet_name",
            "model",
            "target",
            "pfas_features_path",
            "pfas_features_sheet",
            "skip_pfas_features_join",
            "data_mode",
            "kd_final_sources",
        ),
        label="analysis_reconstruction",
    )
    if reconstruction["format"] != ANALYSIS_FORMAT:
        raise ValueError(
            f"Unsupported analysis reconstruction format {reconstruction['format']!r}; "
            f"expected {ANALYSIS_FORMAT!r}."
        )
    return reconstruction


def resolve_reconstruction_path(path_text: str | None) -> Path | None:
    """Resolve a stored data path, including a project moved between user profiles.

    Evaluation metadata intentionally records absolute paths so a run can be
    reconstructed later.  A Box-synchronised project may subsequently move to a
    different Windows profile, however.  When the original path is unavailable,
    retain the portion beneath the project root and look for it under this copy
    of the Adsorbent Recommender project.
    """
    if not path_text:
        return None

    recorded = Path(path_text)
    if recorded.exists():
        return recorded

    project_name = "2. Adsorbent Recommender"
    notes_root_name = "My Box Notes"
    parts = recorded.parts
    try:
        project_index = next(
            index for index, part in enumerate(parts) if part.casefold() == project_name.casefold()
        )
    except StopIteration:
        try:
            notes_index = next(
                index for index, part in enumerate(parts) if part.casefold() == notes_root_name.casefold()
            )
        except StopIteration:
            return recorded
        current_notes_root = Path(__file__).resolve().parents[5]
        relocated = current_notes_root.joinpath(*parts[notes_index + 1 :])
        return relocated if relocated.exists() else recorded

    current_project_root = Path(__file__).resolve().parents[4]
    relocated = current_project_root.joinpath(*parts[project_index + 1 :])
    return relocated if relocated.exists() else recorded


def rebuild_model_frame(reconstruction: dict[str, Any]) -> pd.DataFrame:
    from .logkd_data import load_model_rows

    pfas_path_text = reconstruction["pfas_features_path"]
    pfas_path = resolve_reconstruction_path(pfas_path_text)
    input_path = resolve_reconstruction_path(str(reconstruction["input_path"]))
    if input_path is None:
        raise ValueError("analysis_reconstruction.input_path must be a non-empty path.")
    frame, _, _ = load_model_rows(
        input_path=input_path,
        sheet_name=str(reconstruction["sheet_name"]),
        model=str(reconstruction["model"]),
        target=str(reconstruction["target"]),
        pfas_features_path=pfas_path,
        pfas_features_sheet=str(reconstruction["pfas_features_sheet"]),
        skip_pfas_features_join=bool(reconstruction["skip_pfas_features_join"]),
        data_mode=str(reconstruction["data_mode"]),
        kd_final_sources=reconstruction["kd_final_sources"],
    )
    return frame.reset_index(drop=True)


def load_split_assignments(run_dir: Path, row_count: int) -> pd.DataFrame:
    path = run_dir / "split_assignments.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing split assignments: {path}")
    split = pd.read_csv(path)
    required = {"row_position", "split"}
    missing = sorted(required.difference(split.columns))
    if missing:
        raise ValueError(f"{path.name} lacks required columns: {missing}")
    positions = pd.to_numeric(split["row_position"], errors="coerce")
    expected = np.arange(row_count, dtype=int)
    if positions.isna().any() or len(split) != row_count or not np.array_equal(
        np.sort(positions.to_numpy(dtype=int)), expected
    ):
        raise ValueError(
            "The reconstructed model frame no longer matches split_assignments.csv; "
            "retrain before running SHAP."
        )
    if split["row_position"].duplicated().any():
        raise ValueError("split_assignments.csv contains duplicate row_position values.")
    return split


def testing_positions(split: pd.DataFrame) -> np.ndarray:
    positions = split.loc[split["split"].eq("testing"), "row_position"].to_numpy(dtype=int)
    if not len(positions):
        raise ValueError("The outer run has no held-out testing rows to explain.")
    return np.sort(positions)


def validate_pipeline(pipeline: Any) -> None:
    if not hasattr(pipeline, "named_steps"):
        raise ValueError("Saved model is not a scikit-learn pipeline.")
    missing = [step for step in (PREPROCESS_STEP, MODEL_STEP) if step not in pipeline.named_steps]
    if missing:
        raise ValueError(f"Saved pipeline is missing required steps: {missing}")


def load_evaluation_run(run_dir: Path) -> EvaluationRun:
    import joblib

    run_dir = run_dir.resolve()
    config_path = run_dir / "run_config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Missing run configuration: {config_path}")
    config = load_json(config_path)
    reconstruction = reconstruction_from_config(config)
    features = require_mapping(config, "features", label="run_config.json")
    require_keys(features, ("selected", "numeric", "categorical"), label="features")
    artifacts = require_mapping(config, "model_artifacts", label="run_config.json")
    model_filename = artifacts.get("model_filename")
    if not isinstance(model_filename, str) or not model_filename:
        raise ValueError(
            f"{run_dir} has no saved outer model. Rerun the evaluation with --save-model."
        )
    model_path = run_dir / model_filename
    if Path(model_filename).is_absolute() or len(Path(model_filename).parts) != 1 or not model_path.exists():
        raise FileNotFoundError(f"Invalid or missing saved outer model: {model_path}")
    pipeline = joblib.load(model_path)
    validate_pipeline(pipeline)
    frame = rebuild_model_frame(reconstruction)
    split = load_split_assignments(run_dir, len(frame))
    return EvaluationRun(
        run_dir=run_dir,
        pipeline=pipeline,
        frame=frame,
        split=split,
        selected=list(features["selected"]),
        numeric=list(features["numeric"]),
        categorical=list(features["categorical"]),
    )


def load_deployment_bundle(bundle_dir: Path) -> DeploymentBundle:
    import joblib
    from adsorbent_recommender.bundle import load_bundle_manifest

    directory = bundle_dir.resolve()
    manifest = load_bundle_manifest(directory)
    features = require_mapping(manifest, "features", label="recommender_bundle.json")
    require_keys(features, ("selected", "numeric", "categorical"), label="bundle features")
    ensemble_path = directory / str(manifest["artifacts"]["ensemble"])
    packet = joblib.load(ensemble_path)
    if not isinstance(packet, dict) or not isinstance(packet.get("models"), list) or not packet["models"]:
        raise ValueError("model_ensemble.joblib has an unsupported or empty ensemble format.")
    selected = list(features["selected"])
    numeric = list(features["numeric"])
    categorical = list(features["categorical"])
    for field, expected in (
        ("selected_features", selected),
        ("numeric_features", numeric),
        ("categorical_features", categorical),
    ):
        if list(packet.get(field, [])) != expected:
            raise ValueError(f"The ensemble's {field} does not match recommender_bundle.json.")
    models = list(packet["models"])
    for model in models:
        validate_pipeline(model)
    run_config_path = directory / str(manifest["artifacts"]["run_config"])
    reconstruction = reconstruction_from_config(load_json(run_config_path))
    return DeploymentBundle(
        directory=directory,
        models=models,
        frame=rebuild_model_frame(reconstruction),
        selected=selected,
        numeric=numeric,
        categorical=categorical,
    )


def as_dense(matrix: Any) -> np.ndarray:
    if hasattr(matrix, "toarray"):
        matrix = matrix.toarray()
    return np.asarray(matrix, dtype=float)


def processed_feature_names(pipeline: Any, width: int) -> list[str]:
    preprocessor = pipeline.named_steps[PREPROCESS_STEP]
    try:
        names = [str(name) for name in preprocessor.get_feature_names_out()]
    except Exception as exc:
        raise RuntimeError("The fitted preprocessor cannot provide transformed feature names.") from exc
    if len(names) != width:
        raise ValueError(f"Feature-name count ({len(names)}) does not match transformed width ({width}).")
    return names


def original_feature_for_processed(
    processed_feature: str,
    selected: list[str],
    numeric: list[str],
    categorical: list[str],
) -> str:
    if processed_feature in selected:
        return processed_feature
    for feature in sorted(categorical, key=len, reverse=True):
        if processed_feature.startswith(f"{feature}_") or processed_feature.startswith(f"{feature}="):
            return feature
    for prefix in ("missingindicator_", "missingindicator-"):
        if processed_feature.startswith(prefix):
            candidate = processed_feature[len(prefix) :]
            if candidate in selected:
                return candidate
    for feature in sorted(numeric, key=len, reverse=True):
        if processed_feature == feature or processed_feature.endswith(feature):
            return feature
    return processed_feature


def make_feature_map(
    names: list[str], selected: list[str], numeric: list[str], categorical: list[str]
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for index, name in enumerate(names):
        original = original_feature_for_processed(name, selected, numeric, categorical)
        rows.append(
            {
                "processed_feature_index": index,
                "processed_feature": name,
                "original_feature": original,
            }
        )
    return pd.DataFrame(rows)


def compute_contributions(
    model: Any, X_processed: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, str]:
    """Return contributions plus an optional prediction from SHAP's exact model path."""
    if hasattr(model, "get_booster"):
        import xgboost as xgb

        matrix = xgb.DMatrix(X_processed)
        booster = model.get_booster()
        contributions = np.asarray(
            booster.predict(matrix, pred_contribs=True), dtype=float
        )
        booster_predictions = np.asarray(booster.predict(matrix), dtype=float).reshape(-1)
        if len(booster_predictions) != len(X_processed):
            raise ValueError("Unexpected XGBoost prediction shape for SHAP additivity validation.")
        return (
            contributions[:, :-1],
            contributions[:, -1],
            booster_predictions,
            "xgboost_pred_contribs",
        )
    if hasattr(model, "coef_") and hasattr(model, "intercept_"):
        coefficients = np.asarray(model.coef_, dtype=float).reshape(-1)
        if len(coefficients) != X_processed.shape[1]:
            raise ValueError("Linear-model coefficient count does not match transformed feature count.")
        intercept = float(np.asarray(model.intercept_, dtype=float).reshape(-1)[0])
        return (
            X_processed * coefficients,
            np.full(len(X_processed), intercept),
            None,
            "linear_exact_decomposition",
        )
    try:
        import shap
    except Exception as exc:
        raise RuntimeError(
            "This estimator requires the optional 'shap' package for TreeExplainer."
        ) from exc
    explainer = shap.TreeExplainer(model)
    values = np.asarray(explainer.shap_values(X_processed), dtype=float)
    if values.ndim == 3 and values.shape[-1] == 1:
        values = values[..., 0]
    if values.shape != X_processed.shape:
        raise ValueError(f"Unexpected TreeExplainer SHAP shape {values.shape}; expected {X_processed.shape}.")
    expected = np.asarray(explainer.expected_value, dtype=float).reshape(-1)
    if expected.size != 1:
        raise ValueError("Only single-target regression models are supported.")
    return values, np.full(len(X_processed), float(expected[0])), None, "shap_tree_explainer"


def aggregate_to_original(shap_values: np.ndarray, feature_map: pd.DataFrame) -> pd.DataFrame:
    result = pd.DataFrame(index=np.arange(shap_values.shape[0]))
    for feature, indices in feature_map.groupby("original_feature", sort=False)["processed_feature_index"]:
        result[str(feature)] = shap_values[:, indices.to_numpy(dtype=int)].sum(axis=1)
    return result


def explain_pipeline(
    pipeline: Any,
    frame: pd.DataFrame,
    selected: list[str],
    numeric: list[str],
    categorical: list[str],
) -> PipelineExplanation:
    from .logkd_features import build_X

    validate_pipeline(pipeline)
    X = build_X(frame, selected, numeric, categorical)
    processed = as_dense(pipeline.named_steps[PREPROCESS_STEP].transform(X))
    names = processed_feature_names(pipeline, processed.shape[1])
    feature_map = make_feature_map(names, selected, numeric, categorical)
    values, base_values, shap_path_predictions, backend = compute_contributions(
        pipeline.named_steps[MODEL_STEP], processed
    )
    predictions = np.asarray(pipeline.predict(X), dtype=float)
    additivity_predictions = predictions if shap_path_predictions is None else shap_path_predictions
    additivity_reference = (
        "pipeline.predict" if shap_path_predictions is None else "xgboost.Booster.predict"
    )
    additivity_error = (
        float(np.max(np.abs(values.sum(axis=1) + base_values - additivity_predictions)))
        if len(X)
        else 0.0
    )
    if not np.isfinite(additivity_error) or additivity_error > SHAP_ADDITIVITY_TOLERANCE:
        raise ValueError(
            "SHAP additivity check failed "
            f"(maximum error {additivity_error:.3g}; "
            f"tolerance {SHAP_ADDITIVITY_TOLERANCE:.3g}; "
            f"reference {additivity_reference})."
        )
    if shap_path_predictions is not None:
        prediction_path_error = (
            float(np.max(np.abs(shap_path_predictions - predictions))) if len(X) else 0.0
        )
        if not np.isfinite(prediction_path_error) or prediction_path_error > SHAP_ADDITIVITY_TOLERANCE:
            raise ValueError(
                "XGBoost SHAP and pipeline prediction paths disagree "
                f"(maximum error {prediction_path_error:.3g}; "
                f"tolerance {SHAP_ADDITIVITY_TOLERANCE:.3g})."
            )
    return PipelineExplanation(
        X=X,
        X_processed=processed,
        processed_values=values,
        feature_map=feature_map,
        original_values=aggregate_to_original(values, feature_map),
        base_values=base_values,
        predictions=predictions,
        additivity_predictions=additivity_predictions,
        additivity_reference=additivity_reference,
        backend=backend,
        max_additivity_error=additivity_error,
    )

def required_sample_weight(
    sample_weight: pd.Series | np.ndarray | None,
    row_count: int,
) -> np.ndarray:
    """Return a validated one-dimensional weight vector for SHAP aggregation.

    There is deliberately no unweighted path.  Every SHAP result in this project
    is read against weighted importance numbers, so omitting the weights would
    silently produce an answer that disagrees with them: a source record
    expanded into many model rows would count once in one place and many times
    in another.  None is therefore an error, not a default.
    """

    if sample_weight is None:
        raise ValueError(
            "SHAP sample weights are required. Pass the same per-row weights the "
            "importance summary aggregates with, normally equal_source_row_weights(frame)."
        )
    weights = np.asarray(sample_weight, dtype=float).reshape(-1)
    if len(weights) != row_count:
        raise ValueError(
            f"SHAP sample-weight length ({len(weights)}) does not match "
            f"the explained row count ({row_count})."
        )
    if not np.isfinite(weights).all():
        raise ValueError("SHAP sample weights must all be finite.")
    if (weights < 0).any():
        raise ValueError("SHAP sample weights cannot be negative.")
    if weights.sum() <= 0:
        raise ValueError("SHAP sample weights must have positive total weight.")
    return weights


def _weighted_column_mean(values: np.ndarray, sample_weight: np.ndarray) -> np.ndarray:
    """Average rows by the supplied row weights."""

    return np.average(values, axis=0, weights=sample_weight)

def explain_ensemble(
    models: list[Any],
    frame: pd.DataFrame,
    selected: list[str],
    numeric: list[str],
    categorical: list[str],
    sample_weight: pd.Series | np.ndarray,
) -> EnsembleExplanation:
    weights = required_sample_weight(sample_weight, len(frame))
    explanations = [explain_pipeline(model, frame, selected, numeric, categorical) for model in models]
    first = explanations[0]
    reference_names = first.feature_map["processed_feature"].tolist()
    for explanation in explanations[1:]:
        if explanation.feature_map["processed_feature"].tolist() != reference_names:
            raise ValueError("Ensemble members have incompatible processed feature layouts.")
    processed_values = np.stack([item.processed_values for item in explanations], axis=0)
    base_values = np.stack([item.base_values for item in explanations], axis=0)
    predictions = np.stack([item.predictions for item in explanations], axis=0)
    additivity_predictions = np.stack([item.additivity_predictions for item in explanations], axis=0)
    mean_processed = processed_values.mean(axis=0)
    mean_base = base_values.mean(axis=0)
    mean_prediction = predictions.mean(axis=0)
    mean_additivity_prediction = additivity_predictions.mean(axis=0)
    error = (
        float(np.max(np.abs(mean_processed.sum(axis=1) + mean_base - mean_additivity_prediction)))
        if len(frame)
        else 0.0
    )
    if not np.isfinite(error) or error > SHAP_ADDITIVITY_TOLERANCE:
        raise ValueError(
            "Ensemble SHAP additivity check failed "
            f"(maximum error {error:.3g}; "
            f"tolerance {SHAP_ADDITIVITY_TOLERANCE:.3g})."
        )
    member_importance = pd.DataFrame(
        [
            _weighted_column_mean(
                np.abs(item.original_values.to_numpy(dtype=float)),
                weights,
            )
            for item in explanations
        ],
        columns=first.original_values.columns,
    )
    backends = sorted({item.backend for item in explanations})
    return EnsembleExplanation(
        X=first.X,
        feature_map=first.feature_map,
        original_values=aggregate_to_original(mean_processed, first.feature_map),
        base_values=mean_base,
        predictions=mean_prediction,
        member_mean_abs_values=member_importance,
        backend=", ".join(backends),
        max_additivity_error=error,
    )

def original_importance_summary(
    original_values: pd.DataFrame,
    sample_weight: pd.Series | np.ndarray,
) -> pd.DataFrame:
    """Summarize global SHAP importance using the required row weights."""
    values = original_values.to_numpy(dtype=float)
    weights = required_sample_weight(sample_weight, len(original_values))
    summary = pd.DataFrame(
        {
            "feature": original_values.columns,
            "mean_abs_shap": _weighted_column_mean(
                np.abs(values),
                weights,
            ),
            "mean_shap": _weighted_column_mean(
                values,
                weights,
            ),
        }
    )
    summary["rank"] = summary["mean_abs_shap"].rank(ascending=False, method="min").astype(int)
    return summary.sort_values(["mean_abs_shap", "feature"], ascending=[False, True], kind="stable").reset_index(drop=True)


def add_member_stability(summary: pd.DataFrame, member_mean_abs_values: pd.DataFrame) -> pd.DataFrame:
    stability = pd.DataFrame(
        {
            "feature": member_mean_abs_values.columns,
            "member_mean_abs_shap_sd": member_mean_abs_values.std(axis=0, ddof=1).fillna(0.0).to_numpy(),
        }
    )
    return summary.merge(stability, on="feature", how="left")


def top_local_explanations(
    explanation: EnsembleExplanation,
    input_frame: pd.DataFrame,
    top_features: int,
) -> pd.DataFrame:
    if top_features < 1:
        raise ValueError("--top-features must be at least one.")
    rows: list[dict[str, Any]] = []
    values = explanation.original_values.to_numpy(dtype=float)
    features = explanation.original_values.columns.to_numpy()
    for row_index in range(len(explanation.X)):
        order = np.argsort(np.abs(values[row_index]))[::-1][:top_features]
        for rank, feature_index in enumerate(order, start=1):
            feature = str(features[feature_index])
            row = {
                "input_row": row_index,
                "ensemble_prediction": float(explanation.predictions[row_index]),
                "ensemble_base_value": float(explanation.base_values[row_index]),
                "rank": rank,
                "feature": feature,
                "feature_value": explanation.X.iloc[row_index][feature] if feature in explanation.X.columns else np.nan,
                "shap_value": float(values[row_index, feature_index]),
                "abs_shap_value": float(abs(values[row_index, feature_index])),
            }
            for identifier in ("PFAS_name", "adsorbent_id", "adsorbent_name"):
                if identifier in input_frame.columns:
                    row[identifier] = input_frame.iloc[row_index][identifier]
            rows.append(row)
    return pd.DataFrame(rows)


def save_importance_bar(
    summary: pd.DataFrame,
    output_path: Path,
    *,
    title: str,
    max_display: int,
    error_column: str | None = None,
    repeat_table: pd.DataFrame | None = None,
) -> None:
    """Save feature-importance bars, optionally showing every outer-repeat value."""

    import matplotlib.pyplot as plt

    if max_display < 1:
        raise ValueError("--max-display must be at least one.")
    work = summary.head(max_display).iloc[::-1]
    y_positions = np.arange(len(work))
    figure_height = SHAP_IMPORTANCE_FIGURE_WIDTH / MANUSCRIPT_FIGURE_ASPECT_RATIO
    fig, axis = plt.subplots(
        figsize=(SHAP_IMPORTANCE_FIGURE_WIDTH, figure_height)
    )
    fig.subplots_adjust(left=0.36, right=0.98, bottom=0.13, top=0.88)
    xerr = work[error_column] if error_column and error_column in work.columns else None
    axis.barh(
        y_positions,
        work["mean_abs_shap"],
        color=IMPORTANCE_BAR_COLOR,
        zorder=2,
    )

    if repeat_table is not None:
        required_columns = {"run_dir", "feature", "mean_abs_shap"}
        missing_columns = sorted(required_columns.difference(repeat_table.columns))
        if missing_columns:
            raise ValueError(f"repeat_table lacks required columns: {missing_columns}")
        run_dirs = sorted(repeat_table["run_dir"].drop_duplicates().astype(str))
        if not run_dirs:
            raise ValueError("repeat_table has no outer runs to plot.")
        for run_index, run_dir in enumerate(run_dirs):
            run_values = (
                repeat_table.loc[repeat_table["run_dir"].astype(str).eq(run_dir)]
                .set_index("feature")["mean_abs_shap"]
                .reindex(work["feature"])
            )
            axis.scatter(
                run_values,
                y_positions,
                color=MANUSCRIPT_PALETTE[run_index % len(MANUSCRIPT_PALETTE)],
                edgecolors="white",
                linewidths=0.45,
                s=32,
                label=f"Outer repeat {run_index + 1}",
                zorder=3,
            )

    if xerr is not None:
        axis.errorbar(
            work["mean_abs_shap"],
            y_positions,
            xerr=xerr,
            fmt="none",
            ecolor="#111827",
            elinewidth=1.4,
            capsize=3,
            zorder=4,
        )
    axis.set_yticks(y_positions, work["feature"])
    axis.set_title(title)
    axis.set_xlabel("Mean absolute SHAP value for predicted log10(Kd [L/g])")
    axis.grid(axis="x", color="#D1D5DB", linewidth=0.8)
    axis.set_axisbelow(True)
    if repeat_table is not None:
        handles, labels = axis.get_legend_handles_labels()
        fig.legend(
            handles,
            labels,
            title="Outer runs",
            loc="upper center",
            bbox_to_anchor=(0.5, 0.995),
            ncol=min(3, len(handles)),
            frameon=False,
            fontsize=7,
            columnspacing=1.2,
            handletextpad=0.45,
        )
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def beeswarm_row_offsets(
    values: np.ndarray,
    *,
    bins: int = 100,
    max_offset: float = 0.4,
) -> np.ndarray:
    """Stack points that share a SHAP-value bin symmetrically about their row.

    The stacking order is fully determined by the SHAP values and their row
    order, so one explanation always produces the identical figure.
    """

    values = np.asarray(values, dtype=float).reshape(-1)
    count = len(values)
    if not count:
        return np.zeros(0, dtype=float)
    finite = values[np.isfinite(values)]
    low = float(finite.min()) if finite.size else 0.0
    high = float(finite.max()) if finite.size else 0.0
    safe = np.where(np.isfinite(values), values, low)
    span = high - low
    quantized = (
        np.round(bins * (safe - low) / span).astype(int)
        if span > 0
        else np.zeros(count, dtype=int)
    )
    offsets = np.zeros(count, dtype=float)
    layer = 0
    previous_bin: int | None = None
    for index in np.lexsort((np.arange(count), quantized)):
        current_bin = int(quantized[index])
        if current_bin != previous_bin:
            layer = 0
            previous_bin = current_bin
        offsets[index] = np.ceil(layer / 2) * ((layer % 2) * 2 - 1)
        layer += 1
    # An even-sized stack ends one point above its row, so re-centre each bin.
    _, membership, sizes = np.unique(quantized, return_inverse=True, return_counts=True)
    offsets -= (np.bincount(membership, weights=offsets) / sizes)[membership]
    peak = float(np.max(np.abs(offsets)))
    if peak > 0:
        offsets *= max_offset / peak
    return offsets


def level_display_label(processed_feature: str, original_feature: str) -> str:
    """Name one one-hot column the way a reader of the figure would."""

    level = processed_feature
    for separator in ("_", "="):
        prefix = f"{original_feature}{separator}"
        if processed_feature.startswith(prefix):
            level = processed_feature[len(prefix) :]
            break
    if level == INFREQUENT_ONEHOT_LEVEL:
        level = "rare levels (grouped)"
    return f"{original_feature} = {level}"


def build_beeswarm_display(
    explanation: PipelineExplanation,
    categorical: list[str],
    *,
    sample_weight: pd.Series | np.ndarray,
    split_categorical_levels: bool = True,
) -> BeeswarmDisplay:
    """Choose what one beeswarm row means and how its points are coloured.

    With ``split_categorical_levels`` a categorical input contributes one row
    per one-hot level, carrying that level's own SHAP values and coloured by its
    exact 0/1 indicator, as adsorption-ML papers plot them.  Every row then has
    a colour, and the imputed ``<missing>`` category becomes its own row rather
    than an uncoloured point.  Without the split, a categorical input keeps one
    row holding the summed contribution of all its levels, which no single
    feature value can colour.

    Numeric inputs always keep one aggregated row, coloured by the raw model
    input, so a numeric row still shows missing values as uncoloured points.

    ``sample_weight`` carries the same per-row weights the importance summary
    aggregates with, and is required: the figure sizes each point by its weight
    so an isotherm expanded into many model rows cannot outweigh an unexpanded
    record simply by contributing more points.
    """

    weights = required_sample_weight(sample_weight, len(explanation.X))
    categorical_features = set(categorical)
    values: dict[str, np.ndarray] = {}
    colors: dict[str, np.ndarray] = {}
    layout: list[dict[str, Any]] = []
    row_count = len(explanation.X)
    for original, group in explanation.feature_map.groupby("original_feature", sort=False):
        original = str(original)
        indices = group["processed_feature_index"].to_numpy(dtype=int)
        is_categorical = original in categorical_features
        if not (is_categorical and split_categorical_levels):
            values[original] = explanation.processed_values[:, indices].sum(axis=1)
            colors[original] = (
                percentile_scaled_colors(explanation.X[original])
                if original in explanation.X.columns and not is_categorical
                else np.full(row_count, np.nan)
            )
            layout.append(
                {
                    "display_feature": original,
                    "parent_feature": original,
                    "level": "",
                    "row_kind": "categorical_aggregate" if is_categorical else "numeric",
                }
            )
            continue
        for index, processed in zip(indices, group["processed_feature"], strict=True):
            label = level_display_label(str(processed), original)
            values[label] = explanation.processed_values[:, index]
            # The transformed column is the indicator itself: 1 present, 0 absent.
            colors[label] = np.clip(
                np.asarray(explanation.X_processed[:, index], dtype=float), 0.0, 1.0
            )
            layout.append(
                {
                    "display_feature": label,
                    "parent_feature": original,
                    "level": label.split(" = ", 1)[1],
                    "row_kind": "categorical_level",
                }
            )
    index = pd.RangeIndex(row_count)
    return BeeswarmDisplay(
        values=pd.DataFrame(values, index=index),
        color_positions=pd.DataFrame(colors, index=index),
        layout=pd.DataFrame(layout),
        weights=pd.Series(weights, index=index),
    )


def percentile_scaled_colors(feature_values: pd.Series) -> np.ndarray:
    """Scale one feature's values onto [0, 1]; non-numeric entries stay missing."""

    numeric = pd.to_numeric(feature_values, errors="coerce").to_numpy(dtype=float)
    finite = numeric[np.isfinite(numeric)]
    if finite.size < 2:
        return np.full(len(numeric), np.nan)
    low, high = (float(value) for value in np.percentile(finite, BEESWARM_COLOR_PERCENTILES))
    if high <= low:
        low, high = float(finite.min()), float(finite.max())
    if high <= low:
        return np.full(len(numeric), np.nan)
    scaled = np.clip((numeric - low) / (high - low), 0.0, 1.0)
    return np.where(np.isfinite(numeric), scaled, np.nan)


def beeswarm_point_areas(
    weights: pd.Series | np.ndarray,
    row_count: int,
) -> np.ndarray:
    """Size every plotted point by the weight its row carries.

    Marker area is proportional to the row weight, so the points of one
    pre-expansion source record together spend the marker area of a single
    unexpanded record.  The weights are the aggregation weights themselves,
    already normalized to a mean of one, so ``BEESWARM_POINT_AREA`` remains the
    area an unexpanded record's single point receives.
    """

    validated = required_sample_weight(weights, row_count)
    return np.maximum(BEESWARM_POINT_AREA * validated, BEESWARM_MIN_POINT_AREA)


def manuscript_beeswarm_feature_label(feature: str) -> str:
    """Return compact reader-facing labels for the manuscript beeswarm only."""

    labels = {
        "Adsorbent_dosage_value_mg/L": "Adsorbent dose (mg/L)",
        "PFAS_C0_value_mg/L": "Initial PFAS (mg/L)",
        "Mw (g/mol)": "MW (g/mol)",
        "pore_volume_micro_value_avg_cm3/g": "Micropore volume (cm³/g)",
        "rdkit_F_count": "Fluorine atoms",
        "ssa_(m2/g)_avg": "Specific surface area (m²/g)",
        "average_pore_diameter_(angstrom)": "Average pore diameter (Å)",
        "pore_volume_total_value_cm3/g": "Total pore volume (cm³/g)",
        "Nominal_grain_size_value_mm": "Nominal grain size (mm)",
        "rdkit_fraction_csp3": "Csp³ fraction",
        "phpzc": "pH at PZC",
        "rdkit_tpsa": "Topological polar surface area",
        "element_O_value": "Oxygen content",
        "contains_Ca": "Calcium present",
    }
    return labels.get(feature, feature)


def save_beeswarm(
    display: BeeswarmDisplay,
    output_path: Path,
    *,
    title: str | None,
    max_display: int,
    feature_order: list[str],
    subtitle: str | None = None,
    panel_ready: bool = False,
) -> list[str]:
    """Save a signed per-row SHAP beeswarm for one explained model.

    ``display`` supplies the plotted rows, their signed SHAP values, the colour
    position of every point, and the row weights, as built by
    ``build_beeswarm_display``.  ``feature_order`` fixes the ranked row order.

    Marker area follows the weights, so a source record expanded into several
    model rows cannot dominate the swarm by contributing more points than an
    unexpanded record, and the figure shows the same weighting the importance
    bars aggregate with.
    """

    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap, Normalize
    from matplotlib.lines import Line2D

    if max_display < 1:
        raise ValueError("--max-display must be at least one.")
    original_values = display.values
    color_positions = display.color_positions
    missing = [feature for feature in feature_order if feature not in original_values.columns]
    if missing:
        raise ValueError(f"Beeswarm ordering names features without SHAP values: {missing}")
    if original_values.empty or not len(original_values.columns):
        raise ValueError("The beeswarm needs at least one explained row and feature.")
    if list(color_positions.columns) != list(original_values.columns):
        raise ValueError("Beeswarm colour positions do not match the plotted SHAP columns.")
    if len(color_positions) != len(original_values):
        raise ValueError(
            f"Beeswarm colour rows ({len(color_positions)}) do not match "
            f"explained SHAP rows ({len(original_values)})."
        )
    point_areas = beeswarm_point_areas(display.weights, len(original_values))
    if panel_ready:
        point_areas = point_areas * PANEL_BEESWARM_POINT_AREA_SCALE

    displayed = list(feature_order)[:max_display]
    # Draw the most important feature at the top, as the importance bars do.
    plotted = displayed[::-1]
    colormap = LinearSegmentedColormap.from_list("logkd_shap_feature_value", BEESWARM_COLORMAP_STOPS)
    normalizer = Normalize(vmin=0.0, vmax=1.0)
    figure_height = SHAP_IMPORTANCE_FIGURE_WIDTH / MANUSCRIPT_FIGURE_ASPECT_RATIO
    fig, axis = plt.subplots(figsize=(SHAP_BEESWARM_FIGURE_WIDTH, figure_height))
    if panel_ready:
        # The beeswarm occupies a smaller panel in Figure 4 than it does on its
        # own, so reserve enough left margin for manuscript-size feature names.
        fig.subplots_adjust(left=0.40, right=0.89, bottom=0.13, top=0.91)
    else:
        fig.subplots_adjust(left=0.36, right=0.86, bottom=0.17, top=0.88)

    unscaled_points = False
    for row_position, feature in enumerate(plotted):
        shap_values = original_values[feature].to_numpy(dtype=float)
        offsets = beeswarm_row_offsets(shap_values)
        colors = color_positions[feature].to_numpy(dtype=float)
        scaled = np.isfinite(colors)
        if not scaled.all():
            unscaled_points = True
            axis.scatter(
                shap_values[~scaled],
                row_position + offsets[~scaled],
                color=BEESWARM_UNSCALED_COLOR,
                s=point_areas[~scaled],
                linewidths=0,
                alpha=0.85,
                zorder=2,
            )
        axis.scatter(
            shap_values[scaled],
            row_position + offsets[scaled],
            c=colors[scaled],
            cmap=colormap,
            norm=normalizer,
            s=point_areas[scaled],
            linewidths=0,
            alpha=0.85,
            zorder=3,
        )

    axis.axvline(0.0, color="#111827", linewidth=0.9, zorder=1)
    tick_labels = (
        [manuscript_beeswarm_feature_label(feature) for feature in plotted]
        if panel_ready
        else plotted
    )
    axis.set_yticks(np.arange(len(plotted)), tick_labels)
    axis.set_ylim(-0.6, len(plotted) - 0.4)
    axis.set_xlabel(
        "SHAP value for predicted log10(Kd [L/g])",
        fontsize=14 if panel_ready else None,
    )
    if panel_ready:
        axis.tick_params(axis="y", labelsize=16)
        axis.tick_params(axis="x", labelsize=12)
    else:
        axis.tick_params(axis="both")
    if title:
        axis.set_title(title, pad=16, fontsize=16 if panel_ready else None)
    if subtitle and not panel_ready:
        axis.text(
            0.5,
            1.012,
            subtitle,
            transform=axis.transAxes,
            ha="center",
            va="bottom",
            fontsize=8,
            color="#374151",
        )
    axis.grid(axis="x", color="#D1D5DB", linewidth=0.8)
    axis.set_axisbelow(True)
    handles: list[Line2D] = []
    if unscaled_points:
        handles.append(
            Line2D(
                [],
                [],
                marker="o",
                linestyle="none",
                color=BEESWARM_UNSCALED_COLOR,
                markersize=4,
                label="Missing value",
            )
        )
    # Point size is unreadable without a key, but an explanation whose source
    # records were never expanded draws every point the same size and so has
    # nothing to key.  The note below still states the rule in that case.
    if float(np.ptp(point_areas)) > 0:
        low, high = float(point_areas.min()), float(point_areas.max())
        low_weight = float(np.asarray(display.weights, dtype=float).min())
        high_weight = float(np.asarray(display.weights, dtype=float).max())
        handles.extend(
            Line2D(
                [],
                [],
                marker="o",
                linestyle="none",
                color="#4B5563",
                # Line2D sizes markers by diameter, scatter by area.
                markersize=2.0 * np.sqrt(area / np.pi),
                label=f"weight {weight:.2f}",
            )
            for area, weight in ((low, low_weight), (high, high_weight))
        )
    if handles and not panel_ready:
        axis.legend(
            handles=handles,
            # The lowest-ranked feature's points taper toward zero, so this
            # corner stays clear in a ranked beeswarm.
            loc="lower right",
            frameon=False,
            fontsize=7,
            handletextpad=0.4,
            labelspacing=0.35,
        )

    color_axis = fig.add_axes(
        (0.915, 0.15, 0.018, 0.72) if panel_ready else (0.885, 0.17, 0.018, 0.71)
    )
    color_bar = fig.colorbar(
        plt.cm.ScalarMappable(norm=normalizer, cmap=colormap), cax=color_axis
    )
    color_bar.set_ticks([0.0, 1.0])
    if panel_ready:
        # Stack the scale endpoints with the bar rather than spending a separate
        # right-hand text column in a tightly assembled manuscript panel.
        color_bar.set_ticklabels(["", ""])
        color_bar.ax.tick_params(length=0)
        color_axis.text(
            0.5, 1.025, "High", transform=color_axis.transAxes,
            ha="center", va="bottom", fontsize=13,
        )
        color_axis.text(
            0.5, -0.025, "Low", transform=color_axis.transAxes,
            ha="center", va="top", fontsize=13,
        )
    else:
        color_bar.set_ticklabels(["Low", "High"])
        color_bar.set_label("Feature value", fontsize=8)
    color_bar.ax.tick_params(labelsize=13 if panel_ready else None)
    color_bar.outline.set_visible(False)

    # The two row kinds are coloured on different scales, so say which is which.
    if not panel_ready:
        notes = [
            "Numeric rows are coloured between their "
            f"{BEESWARM_COLOR_PERCENTILES[0]:g}th and {BEESWARM_COLOR_PERCENTILES[1]:g}th percentile values."
        ]
        if display.layout["row_kind"].eq("categorical_level").any():
            notes.append("One-hot level rows are coloured 1 (category present) or 0 (absent).")
        notes.append(
            "Point area is proportional to the row's aggregation weight, so the points of one "
            "pre-expansion source record together carry the area of a single unexpanded record."
        )
        # One note per line: joined end to end they overrun the figure width.
        fig.text(0.02, 0.02, "\n".join(notes), fontsize=6.5, color="#6B7280", ha="left", va="bottom")
    fig.savefig(output_path, dpi=450 if panel_ready else 300)
    plt.close(fig)
    return displayed
