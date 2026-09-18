"""Train or finalize a PFAS logKd regressor.

Exploratory training uses a frozen testing partition: feature selection and
tuning occur exclusively in training and testing is evaluated once. Final
deployment uses all eligible rows for inner-CV tuning and refitting, without
creating a new outer testing partition.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

SCRIPTS_DIR = Path(__file__).resolve().parent.parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from backend import logkd_config as cfg
from backend.logkd_coverage import reference_retention_by_feature, summarize_reference_retention
from backend.logkd_data import equal_source_row_weights, load_model_rows
from backend.logkd_features import (
    add_feature_selection_arguments,
    as_float_array,
    build_X,
    conditional_numeric_applicability_features,
    feature_audit,
    feature_selection_kwargs,
    feature_value_mask,
    group_selected_features_by_bucket,
    joint_support_coverage,
    input_support_diagnostics,
)
from backend.logkd_metrics import (
    grouped_metrics,
    logkd_tail_metrics_dict,
    metrics_dict,
)
from backend.logkd_splits import (
    TRAINING,
    TESTING,
    build_testing_split,
    validation_diagnostics,
    load_testing_assignments,
    make_validation_assignments,
    outer_allocation_protocol_metadata,
    split_diagnostics,
    split_indices,
    validate_testing_split,
)


# Default XGBoost tuning profile.  Its ranges were retained after the paired
# full-versus-compact comparison on 2026-08-06, which used the same frozen
# outer splits and a 100-trial budget for each profile.  Runners can still
# pass a purpose-specific override through --compact-xgb-search-space-json.
DEFAULT_COMPACT_XGB_SEARCH_SPACE: dict[str, Any] = {
    "learning_rate": (0.02, 0.05),
    "n_estimators": (500, 1100),
    "learning_rate_times_n_estimators": (15.0, 45.0),
    "max_depth": (4, 6),
    "min_child_weight": (2.0, 10.0),
    "subsample": (0.65, 0.95),
    "colsample_bytree": (0.60, 0.90),
    "fixed": {"gamma": 0.0, "reg_alpha": 0.05, "reg_lambda": 5.0},
}


def compact_xgb_search_space_from_json(search_space_json: str | None) -> dict[str, Any]:
    """Parse a runner-supplied compact search space, or return the default."""

    if search_space_json is None:
        return {
            **{
                key: tuple(value)
                for key, value in DEFAULT_COMPACT_XGB_SEARCH_SPACE.items()
                if key != "fixed"
            },
            "fixed": dict(DEFAULT_COMPACT_XGB_SEARCH_SPACE["fixed"]),
        }
    try:
        raw = json.loads(search_space_json)
    except json.JSONDecodeError as error:
        raise ValueError("--compact-xgb-search-space-json must be valid JSON.") from error
    if not isinstance(raw, dict):
        raise ValueError("--compact-xgb-search-space-json must describe a JSON object.")

    core_range_keys = (
        "learning_rate",
        "n_estimators",
        "learning_rate_times_n_estimators",
        "max_depth",
        "min_child_weight",
        "subsample",
        "colsample_bytree",
    )
    regularization_keys = ("gamma", "reg_alpha", "reg_lambda")
    missing = [key for key in (*core_range_keys, "fixed") if key not in raw]
    if missing:
        raise ValueError(
            "--compact-xgb-search-space-json is missing keys: " + ", ".join(missing)
        )
    try:
        if not isinstance(raw["fixed"], dict):
            raise TypeError("fixed must be a JSON object")
        fixed_regularizers = {
            name: float(value)
            for name, value in raw["fixed"].items()
        }
        unknown_fixed = set(fixed_regularizers).difference(regularization_keys)
        if unknown_fixed:
            raise KeyError(f"Unknown fixed XGBoost parameters: {sorted(unknown_fixed)}")
        overlapping = set(fixed_regularizers).intersection(raw).intersection(
            regularization_keys
        )
        if overlapping:
            raise ValueError(
                "Regularizers cannot be both fixed and tuned: "
                + ", ".join(sorted(overlapping))
            )
        missing_regularizers = [
            name
            for name in regularization_keys
            if name not in fixed_regularizers and name not in raw
        ]
        if missing_regularizers:
            raise KeyError(
                "Every regularizer must be fixed or have a tuning range: "
                + ", ".join(missing_regularizers)
            )
        parsed = {
            "learning_rate": tuple(float(value) for value in raw["learning_rate"]),
            "n_estimators": tuple(int(value) for value in raw["n_estimators"]),
            "learning_rate_times_n_estimators": tuple(
                float(value) for value in raw["learning_rate_times_n_estimators"]
            ),
            "max_depth": tuple(int(value) for value in raw["max_depth"]),
            "min_child_weight": tuple(float(value) for value in raw["min_child_weight"]),
            "subsample": tuple(float(value) for value in raw["subsample"]),
            "colsample_bytree": tuple(float(value) for value in raw["colsample_bytree"]),
            **{
                name: tuple(float(value) for value in raw[name])
                for name in regularization_keys
                if name in raw
            },
            "fixed": fixed_regularizers,
        }
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(
            "--compact-xgb-search-space-json has an invalid compact XGBoost range."
        ) from error
    range_keys = (*core_range_keys, *(name for name in regularization_keys if name in parsed))
    if any(len(parsed[key]) != 2 for key in range_keys):
        raise ValueError("Every compact XGBoost range must contain exactly two values.")
    if any(parsed[key][0] > parsed[key][1] for key in range_keys):
        raise ValueError("Every compact XGBoost range must be ordered low, high.")
    if (
        parsed["learning_rate"][0] <= 0
        or parsed["n_estimators"][0] < 1
        or parsed["min_child_weight"][0] <= 0
        or parsed["learning_rate_times_n_estimators"][0] <= 0
    ):
        raise ValueError("Compact XGBoost learning, tree, and child-weight lower bounds must be positive.")
    if any(not 0 < value <= 1 for key in ("subsample", "colsample_bytree") for value in parsed[key]):
        raise ValueError("Compact XGBoost subsample and colsample_bytree bounds must be in (0, 1].")
    if "gamma" in parsed and parsed["gamma"][0] < 0:
        raise ValueError("Compact XGBoost gamma bounds must be nonnegative.")
    if "reg_alpha" in parsed and parsed["reg_alpha"][0] <= 0:
        raise ValueError("Compact XGBoost reg_alpha bounds must be positive for log sampling.")
    if "reg_lambda" in parsed and parsed["reg_lambda"][0] <= 0:
        raise ValueError("Compact XGBoost reg_lambda bounds must be positive for log sampling.")
    return parsed


# All regressors here implement ``fit(..., sample_weight=...)``.  Keeping the
# registry in the trainer means every exploratory comparison shares the same
# split-first feature audit, preprocessing scope, and source-row weighting.
REGRESSORS = (
    "xgb",
    "extra_trees",
    "rbf_svr",
    "ebm",
    "hist_gradient_boosting",
    "catboost",
    "random_forest",
    "ridge",
)
XGB_SEARCH_SPACES = ("full", "compact")


def console_path(path: Path) -> str:
    """Render paths safely in Windows terminals with a legacy code page."""

    encoding = sys.stdout.encoding or "utf-8"
    return str(path).encode(encoding, errors="backslashreplace").decode(encoding)


def frozen_assignment_reporting_metadata(
    assignment_path: Path,
    assignments: pd.DataFrame,
    cli_test_fraction: float,
) -> dict[str, Any]:
    """Recover truthful target metadata for a supplied frozen assignment.

    A frozen assignment contains memberships, not the optimization target that
    created them.  Reconstructing a group-count target as a CLI fraction of all
    groups is therefore wrong for row-size-matched allocations.  When its
    preparation sidecar is available, retain its recorded row target; otherwise
    report the group target as unavailable rather than inventing one.
    """

    testing_mask = assignments["split"].eq(TESTING)
    observed_test_rows = int(testing_mask.sum())
    observed_test_groups = int(assignments.loc[testing_mask, "split_unit"].nunique())
    total_groups = int(assignments["split_unit"].nunique())
    requested_rows_from_cli = float(len(assignments) * cli_test_fraction)
    metadata: dict[str, Any] = {
        "requested_test_fraction": float(cli_test_fraction),
        "requested_test_group_count": None,
        "requested_test_group_count_basis": "not_recorded_for_frozen_assignment",
        "observed_test_group_count": observed_test_groups,
        "observed_test_group_fraction": (
            float(observed_test_groups / total_groups) if total_groups else None
        ),
        "test_row_target_selection_method": (
            "frozen_assignment; original_row_target_not_recorded"
        ),
        "test_row_target_requested_rows": requested_rows_from_cli,
        "test_row_target_absolute_deviation_rows": abs(
            observed_test_rows - requested_rows_from_cli
        ),
        "test_row_target_proved_nearest": None,
        "testing_size_policy": "frozen_assignment; original_policy_not_recorded",
        "frozen_assignment_target_metadata_status": "sidecar_not_available",
        "frozen_assignment_target_metadata_source": None,
    }

    sidecar_path = assignment_path.parent / "split_preparation.json"
    if not sidecar_path.exists():
        return metadata
    try:
        sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
        prepared = sidecar["split"]["validation"]
    except (OSError, ValueError, KeyError, TypeError):
        metadata["frozen_assignment_target_metadata_status"] = "sidecar_unreadable"
        return metadata

    # Do not apply a neighbouring sidecar to a different assignment by mistake.
    if int(prepared.get("testing_rows", -1)) != observed_test_rows:
        metadata["frozen_assignment_target_metadata_status"] = "sidecar_row_count_mismatch"
        return metadata

    metadata.update({
        "requested_test_fraction": float(
            prepared.get("requested_test_fraction", cli_test_fraction)
        ),
        "test_row_target_selection_method": prepared.get(
            "test_row_target_selection_method",
            "frozen_assignment; original_row_target_not_recorded",
        ),
        "test_row_target_requested_rows": prepared.get(
            "test_row_target_requested_rows", requested_rows_from_cli
        ),
        "test_row_target_absolute_deviation_rows": prepared.get(
            "test_row_target_absolute_deviation_rows",
            abs(observed_test_rows - requested_rows_from_cli),
        ),
        "test_row_target_proved_nearest": prepared.get("test_row_target_proved_nearest"),
        "frozen_assignment_target_metadata_status": "matched_preparation_sidecar",
        "frozen_assignment_target_metadata_source": str(sidecar_path),
    })

    policy = prepared.get("testing_size_policy")
    if isinstance(policy, str) and "fraction_of_groups" in policy:
        metadata.update({
            "requested_test_group_count": prepared.get("requested_test_group_count"),
            "requested_test_group_count_basis": "recorded_in_preparation_sidecar",
            "testing_size_policy": policy,
        })
    else:
        metadata.update({
            "requested_test_group_count": None,
            "requested_test_group_count_basis": "not_applicable_row_target",
            "testing_size_policy": policy or "nearest_feasible_fraction_of_rows",
        })
    return metadata


@dataclass
class PreparedValidationFold:
    validation_fold: int
    training_indices: np.ndarray
    validation_indices: np.ndarray
    df_train: pd.DataFrame
    df_validation: pd.DataFrame
    X_train: pd.DataFrame
    y_train: pd.Series
    sample_weight_train: pd.Series
    X_validation: pd.DataFrame
    y_validation: pd.Series
    sample_weight_validation: pd.Series
    selected_features: list[str]
    numeric_features: list[str]
    categorical_features: list[str]
    feature_manifest: pd.DataFrame = field(default_factory=pd.DataFrame)


def parse_args(
    argument_extender: Callable[[argparse.ArgumentParser], None] | None = None,
    *,
    require_outer_allocation_method: bool = True,
) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a PFAS logKd model with a frozen training/testing split.")
    parser.add_argument("--input-path", type=Path, default=cfg.DEFAULT_INPUT)
    parser.add_argument("--sheet-name", default=cfg.SHEET_NAME)
    parser.add_argument("--model", choices=sorted(cfg.MODEL_CATEGORIES), default="AC")
    parser.add_argument("--target", default=cfg.TARGET)
    parser.add_argument("--output-root", type=Path, default=cfg.DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--pfas-features-path", type=Path, default=cfg.DEFAULT_PFAS_FEATURES)
    parser.add_argument("--pfas-features-sheet", default=cfg.DEFAULT_PFAS_FEATURES_SHEET)
    parser.add_argument("--skip-pfas-features-join", action="store_true")
    parser.add_argument(
        "--data-mode",
        choices=["baseline", "drop_unreliable"],
        default=cfg.DEFAULT_DATA_MODE,
    )
    parser.add_argument(
        "--kd-final-sources",
        nargs="+",
        metavar="SOURCE",
        help=(
            "Restrict rows to one or more Kd_final_source labels. Omit to use the "
            "current mixed-endpoint population."
        ),
    )
    parser.add_argument("--pfas-structure-features", choices=["none", "descriptors", "fingerprints", "all"], default="all")
    add_feature_selection_arguments(parser)
    parser.add_argument("--min-category-frequency", type=int, default=10)
    parser.add_argument("--split-strategy", choices=cfg.SPLIT_STRATEGIES, default="study")
    parser.add_argument(
        "--outer-allocation-method",
        choices=cfg.OUTER_ALLOCATION_METHODS,
        required=require_outer_allocation_method,
        default=None,
        help=(
            "Construct the outer testing split by seeded random source-row or group allocation."
        ),
    )
    parser.add_argument(
        "--inner-allocation-method",
        choices=cfg.INNER_ALLOCATION_METHODS,
        default=None,
        help=(
            "Optional explicit inner-CV allocation; omit to match the split strategy."
        ),
    )
    parser.add_argument("--test-fraction", type=float, default=cfg.DEFAULT_TEST_FRACTION)
    parser.add_argument(
        "--prepare-split-only",
        action="store_true",
        help="Construct and write a target-free frozen assignment, then exit before model fitting.",
    )
    parser.add_argument(
        "--split-assignment-path",
        type=Path,
        default=None,
        help="Previously frozen training/testing assignment CSV. It is verified before use.",
    )
    parser.add_argument("--validation-folds", type=int, default=cfg.VALIDATION_FOLDS)
    parser.add_argument(
        "--random-seed",
        type=int,
        default=cfg.DEFAULT_EXPERIMENT_RANDOM_SEED,
        help=(
            "Seed for outer/inner split allocation and hyperparameter-search sampling; "
            "it does not control estimator random_state."
        ),
    )
    parser.add_argument(
        "--model-random-seed",
        type=int,
        default=cfg.DEFAULT_MODEL_RANDOM_SEED,
        help=(
            "Seed for XGBoost, random-forest, or extra-trees stochasticity. "
            "Keep this fixed across outer repeats to isolate split sensitivity."
        ),
    )
    parser.add_argument("--regressor", choices=REGRESSORS, default="xgb")
    parser.add_argument(
        "--xgb-search-space",
        choices=XGB_SEARCH_SPACES,
        default=cfg.DEFAULT_EXPERIMENT_XGB_SEARCH_SPACE,
        help=(
            "XGBoost hyperparameter space (default: compact). compact tunes the "
            "core controls around the fixed baseline and can optionally tune "
            "runner-supplied gamma/reg_alpha/reg_lambda ranges."
        ),
    )
    parser.add_argument(
        "--compact-xgb-search-space-json",
        help=(
            "Optional runner-supplied JSON override for the compact XGBoost "
            "search space. Intended for wrappers, not routine direct use."
        ),
    )
    parser.add_argument(
        "--n-trials",
        type=int,
        default=cfg.DEFAULT_EXPERIMENT_N_TRIALS,
        help=(
            "Total parameter candidates per regressor. For XGBoost, positive values "
            "include the fixed default configuration as the first Optuna candidate; "
            "use 0 to evaluate only that fixed default."
        ),
    )
    parser.add_argument(
        "--early-stop-warmup",
        type=int,
        default=cfg.DEFAULT_EXPERIMENT_EARLY_STOP_WARMUP,
        help=(
            "Completed Optuna trials required before validation-noise-aware "
            "plateau stopping can begin (default: 100; the standard 200-trial "
            "budget leaves room for plateau stopping afterward)."
        ),
    )
    parser.add_argument(
        "--early-stop-patience",
        type=int,
        default=cfg.DEFAULT_EXPERIMENT_EARLY_STOP_PATIENCE,
        help="Completed trials without meaningful validation-RMSE improvement before stopping; use 0 to disable.",
    )
    parser.add_argument(
        "--early-stop-se-multiplier",
        type=float,
        default=cfg.DEFAULT_EXPERIMENT_EARLY_STOP_SE_MULTIPLIER,
        help="Meaningful-improvement threshold multiplier for the best trial's validation-fold RMSE standard error.",
    )
    parser.add_argument(
        "--early-stop-min-delta-floor",
        type=float,
        default=cfg.DEFAULT_EXPERIMENT_EARLY_STOP_MIN_DELTA_FLOOR,
        help="Minimum absolute validation-RMSE improvement required to reset plateau patience.",
    )
    parser.add_argument("--n-jobs", type=int, default=cfg.DEFAULT_EXPERIMENT_N_JOBS)
    parser.add_argument(
        "--console-verbosity",
        choices=("quiet", "normal"),
        default=cfg.DEFAULT_EXPERIMENT_CONSOLE_VERBOSITY,
        help=(
            "quiet reports tuning progress periodically; normal restores the "
            "tuner's own per-trial logging. The comparison wrappers capture this "
            "output, so it matters mainly when training is run directly."
        ),
    )
    parser.add_argument(
        "--numeric-missing-strategy",
        choices=("xgb_native", "median_indicator"),
        default=cfg.DEFAULT_EXPERIMENT_NUMERIC_MISSING_STRATEGY,
    )
    parser.add_argument(
        "--save-model",
        action="store_true",
        help="Save the outer-training refit pipeline as model.joblib for held-out SHAP analysis.",
    )
    if argument_extender is not None:
        argument_extender(parser)
    args = parser.parse_args()
    try:
        args.compact_xgb_search_space = compact_xgb_search_space_from_json(
            args.compact_xgb_search_space_json
        )
    except ValueError as error:
        parser.error(str(error))
    return args


def resolve_inner_allocation_method(args: argparse.Namespace) -> str:
    """Use matching random folds unless target-free evidence balancing is requested."""

    expected = "random_row" if args.split_strategy == "random_row" else "random_group"
    requested = args.inner_allocation_method or expected
    if requested == "evidence_balanced":
        return requested
    if requested != expected:
        raise ValueError(
            f"{requested} inner allocation requires split_strategy="
            f"{'random_row' if requested == 'random_row' else 'a grouped strategy'}"
        )
    return requested


def make_preprocessor(
    numeric_features: list[str],
    categorical_features: list[str],
    min_category_frequency: int,
    numeric_missing_strategy: str,
):
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import FunctionTransformer, OneHotEncoder

    if numeric_missing_strategy == "xgb_native":
        numeric_transformer = FunctionTransformer(as_float_array, feature_names_out="one-to-one")
    elif numeric_missing_strategy == "median_indicator":
        numeric_transformer = Pipeline(steps=[("imputer", SimpleImputer(strategy="median", add_indicator=True))])
    else:
        raise ValueError(f"Unknown numeric missing strategy: {numeric_missing_strategy}")

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value="<missing>")),
            (
                "onehot",
                OneHotEncoder(
                    handle_unknown="infrequent_if_exist",
                    min_frequency=min_category_frequency,
                    sparse_output=False,
                ),
            ),
        ]
    )
    numeric_model_features = [
        *numeric_features,
        *conditional_numeric_applicability_features(numeric_features),
    ]
    transformers = []
    if numeric_model_features:
        transformers.append(("numeric", numeric_transformer, numeric_model_features))
    if categorical_features:
        transformers.append(("categorical", categorical_transformer, categorical_features))
    return ColumnTransformer(transformers=transformers, remainder="drop", sparse_threshold=0.0, verbose_feature_names_out=False)


def default_params(regressor: str) -> dict[str, Any]:
    if regressor == "xgb":
        return {
            "n_estimators": 700,
            "learning_rate": 0.035,
            "max_depth": 4,
            "min_child_weight": 3.0,
            "subsample": 0.85,
            "colsample_bytree": 0.85,
            "gamma": 0.0,
            "reg_alpha": 0.05,
            "reg_lambda": 5.0,
        }
    if regressor == "random_forest":
        return {"n_estimators": 700, "max_features": 0.75, "min_samples_leaf": 2, "max_depth": None}
    if regressor == "extra_trees":
        return {"n_estimators": 700, "max_features": 0.75, "min_samples_leaf": 2, "max_depth": None}
    if regressor == "rbf_svr":
        return {"C": 10.0, "gamma": "scale", "epsilon": 0.05}
    if regressor == "ebm":
        # One outer bag keeps repeated nested evaluation tractable. The
        # comparison evaluates point predictions; EBM uncertainty bands are
        # not an outcome of this screen. A small interaction budget retains
        # the model's interpretable generalized-additive character.
        return {
            "interactions": 5,
            "learning_rate": 0.04,
            "max_rounds": 5000,
            "min_samples_leaf": 4,
            "reg_alpha": 0.01,
            "reg_lambda": 1.0,
            "outer_bags": 1,
            "inner_bags": 0,
        }
    if regressor == "hist_gradient_boosting":
        return {
            "max_iter": 400,
            "learning_rate": 0.05,
            "max_leaf_nodes": 31,
            "min_samples_leaf": 20,
            "l2_regularization": 0.1,
        }
    if regressor == "catboost":
        return {
            "iterations": 700,
            "learning_rate": 0.04,
            "depth": 5,
            "l2_leaf_reg": 5.0,
            "random_strength": 0.5,
        }
    if regressor == "ridge":
        return {"alpha": 10.0}
    raise ValueError(f"Unknown regressor: {regressor}")


def sample_params(regressor: str, rng: np.random.Generator) -> dict[str, Any]:
    if regressor == "xgb":
        return {
            "n_estimators": int(rng.integers(300, 1201)),
            "learning_rate": float(math.exp(rng.uniform(math.log(0.01), math.log(0.12)))),
            "max_depth": int(rng.integers(2, 8)),
            "min_child_weight": float(math.exp(rng.uniform(math.log(0.5), math.log(20.0)))),
            "subsample": float(rng.uniform(0.55, 1.0)),
            "colsample_bytree": float(rng.uniform(0.55, 1.0)),
            "gamma": float(rng.uniform(0.0, 10.0)),
            "reg_alpha": float(math.exp(rng.uniform(math.log(1e-4), math.log(10.0)))),
            "reg_lambda": float(math.exp(rng.uniform(math.log(0.1), math.log(30.0)))),
        }
    if regressor in {"random_forest", "extra_trees"}:
        return {
            "n_estimators": int(rng.choice([400, 700, 1000])),
            "max_features": [0.5, 0.75, 1.0, "sqrt"][int(rng.integers(0, 4))],
            "min_samples_leaf": int(rng.choice([1, 2, 4, 8])),
            "max_depth": [None, 4, 8, 12][int(rng.integers(0, 4))],
        }
    if regressor == "rbf_svr":
        return {
            "C": float(math.exp(rng.uniform(math.log(0.1), math.log(1000.0)))),
            "gamma": float(math.exp(rng.uniform(math.log(1e-4), math.log(1.0)))),
            "epsilon": float(math.exp(rng.uniform(math.log(0.01), math.log(0.30)))),
        }
    if regressor == "ebm":
        return {
            "interactions": int(rng.choice([0, 3, 5, 8])),
            "learning_rate": float(math.exp(rng.uniform(math.log(0.01), math.log(0.08)))),
            "max_rounds": 5000,
            "min_samples_leaf": int(rng.choice([2, 4, 8, 16])),
            "reg_alpha": float(math.exp(rng.uniform(math.log(1e-4), math.log(10.0)))),
            "reg_lambda": float(math.exp(rng.uniform(math.log(1e-4), math.log(10.0)))),
            "outer_bags": 1,
            "inner_bags": 0,
        }
    if regressor == "hist_gradient_boosting":
        return {
            "max_iter": int(rng.integers(150, 701)),
            "learning_rate": float(math.exp(rng.uniform(math.log(0.015), math.log(0.15)))),
            "max_leaf_nodes": int(rng.choice([15, 31, 63])),
            "min_samples_leaf": int(rng.choice([10, 20, 40, 80])),
            "l2_regularization": float(math.exp(rng.uniform(math.log(1e-4), math.log(10.0)))),
        }
    if regressor == "catboost":
        return {
            "iterations": int(rng.integers(300, 1001)),
            "learning_rate": float(math.exp(rng.uniform(math.log(0.015), math.log(0.12)))),
            "depth": int(rng.integers(3, 9)),
            "l2_leaf_reg": float(math.exp(rng.uniform(math.log(1.0), math.log(20.0)))),
            "random_strength": float(rng.uniform(0.0, 2.0)),
        }
    if regressor == "ridge":
        return {"alpha": float(math.exp(rng.uniform(math.log(0.01), math.log(1000.0))))}
    return default_params(regressor)


def suggest_optuna_params(
    trial,
    search_space: str = "compact",
    compact_search_space: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Suggest one XGBoost configuration for the selected Optuna search space."""

    if search_space == "full":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 300, 1500),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.12, log=True),
            "max_depth": trial.suggest_int("max_depth", 2, 8),
            "min_child_weight": trial.suggest_float("min_child_weight", 0.5, 20.0, log=True),
            "subsample": trial.suggest_float("subsample", 0.55, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.55, 1.0),
            "gamma": trial.suggest_float("gamma", 0.0, 10.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.1, 30.0, log=True),
        }
    if search_space != "compact":
        raise ValueError(f"Unknown XGBoost search space: {search_space}")

    # Focus on tree complexity, stochastic regularization, and the
    # learning-rate/tree-count trade-off.  The joint boosting-budget constraint
    # is a guardrail, not an assumption that the product determines capacity.
    params = default_params("xgb").copy()
    compact = compact_search_space or compact_xgb_search_space_from_json(None)
    params.update(compact["fixed"])
    learning_rate = trial.suggest_float(
        "learning_rate", *compact["learning_rate"], log=True
    )
    min_estimators = max(
        compact["n_estimators"][0],
        math.ceil(compact["learning_rate_times_n_estimators"][0] / learning_rate),
    )
    max_estimators = min(
        compact["n_estimators"][1],
        math.floor(compact["learning_rate_times_n_estimators"][1] / learning_rate),
    )
    if min_estimators > max_estimators:
        raise RuntimeError("Compact XGBoost learning-rate/tree-count bounds are infeasible.")
    params.update({
        "learning_rate": learning_rate,
        "n_estimators": trial.suggest_int("n_estimators", min_estimators, max_estimators),
        "max_depth": trial.suggest_int("max_depth", *compact["max_depth"]),
        "min_child_weight": trial.suggest_float(
            "min_child_weight", *compact["min_child_weight"], log=True
        ),
        "subsample": trial.suggest_float("subsample", *compact["subsample"]),
        "colsample_bytree": trial.suggest_float(
            "colsample_bytree", *compact["colsample_bytree"]
        ),
    })
    if "gamma" in compact:
        params["gamma"] = trial.suggest_float("gamma", *compact["gamma"])
    for name in ("reg_alpha", "reg_lambda"):
        if name in compact:
            params[name] = trial.suggest_float(name, *compact[name], log=True)
    return params


def make_estimator(regressor: str, params: dict[str, Any], random_state: int, n_jobs: int):
    if regressor == "xgb":
        from xgboost import XGBRegressor

        defaults = {
            "objective": "reg:squarederror",
            "eval_metric": "rmse",
            "tree_method": "hist",
            "random_state": random_state,
            "n_jobs": n_jobs,
        }
        defaults.update(params)
        return XGBRegressor(**defaults)
    if regressor == "random_forest":
        from sklearn.ensemble import RandomForestRegressor

        return RandomForestRegressor(**params, random_state=random_state, n_jobs=n_jobs)
    if regressor == "extra_trees":
        from sklearn.ensemble import ExtraTreesRegressor

        return ExtraTreesRegressor(**params, random_state=random_state, n_jobs=n_jobs)
    if regressor == "rbf_svr":
        from sklearn.svm import SVR

        return SVR(kernel="rbf", **params)
    if regressor == "ebm":
        try:
            from interpret.glassbox import ExplainableBoostingRegressor
        except ImportError as exc:
            raise RuntimeError(
                "InterpretML is required for --regressor ebm. Install the project "
                "requirements in the Python environment running this script."
            ) from exc

        return ExplainableBoostingRegressor(
            **params,
            objective="rmse",
            max_leaves=2,
            n_jobs=n_jobs,
            random_state=random_state,
        )
    if regressor == "hist_gradient_boosting":
        from sklearn.ensemble import HistGradientBoostingRegressor

        return HistGradientBoostingRegressor(
            **params,
            loss="squared_error",
            random_state=random_state,
        )
    if regressor == "catboost":
        try:
            from catboost import CatBoostRegressor
        except ImportError as exc:
            raise RuntimeError(
                "CatBoost is required for --regressor catboost. Install the "
                "project requirements in the Python environment running this script."
            ) from exc

        return CatBoostRegressor(
            **params,
            loss_function="RMSE",
            random_seed=random_state,
            thread_count=n_jobs,
            verbose=False,
            # The same working directory is reused across repeated subprocesses.
            # Disable CatBoost's unused ``catboost_info`` artifacts so they cannot
            # collide with a prior run still held open by Windows or sync software.
            allow_writing_files=False,
        )
    if regressor == "ridge":
        from sklearn.linear_model import Ridge

        return Ridge(**params)
    raise ValueError(f"Unknown regressor: {regressor}")


def make_pipeline(
    regressor: str,
    numeric_features: list[str],
    categorical_features: list[str],
    min_category_frequency: int,
    numeric_missing_strategy: str,
    params: dict[str, Any],
    random_state: int,
    n_jobs: int,
):
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    if regressor != "xgb" and numeric_missing_strategy == "xgb_native":
        raise ValueError(
            "--numeric-missing-strategy xgb_native is only valid for --regressor xgb; "
            "choose an explicit missing-value strategy for other regressors."
        )
    steps: list[tuple[str, Any]] = [
        (
            "preprocess",
            make_preprocessor(numeric_features, categorical_features, min_category_frequency, numeric_missing_strategy),
        )
    ]
    if regressor in {"ridge", "rbf_svr"}:
        steps.append(("scale", StandardScaler()))
    steps.append(("model", make_estimator(regressor, params, random_state, n_jobs)))
    return Pipeline(steps=steps)


def model_random_state(args: argparse.Namespace, offset: int = 0) -> int:
    """Return an estimator-only seed, independent of split and tuning seeds."""

    return int(args.model_random_seed) + int(offset)


def prepare_validation_folds(
    df_training: pd.DataFrame,
    y_training: pd.Series,
    args: argparse.Namespace,
    reference_manifest: pd.DataFrame | None = None,
) -> tuple[list[PreparedValidationFold], pd.DataFrame, pd.DataFrame]:
    assignments = make_validation_assignments(
        df_training,
        split_strategy=args.split_strategy,
        validation_folds=args.validation_folds,
        random_seed=args.random_seed,
        allocation_method=args.inner_allocation_method,
        reference_manifest=reference_manifest,
    )
    all_weights = equal_source_row_weights(df_training)
    prepared: list[PreparedValidationFold] = []
    manifests: list[pd.DataFrame] = []
    for fold in sorted(value for value in assignments["validation_fold"].unique() if value >= 0):
        train_idx = assignments.index[assignments["validation_fold"].ne(fold)].to_numpy(dtype=int)
        validation_idx = assignments.index[assignments["validation_fold"].eq(fold)].to_numpy(dtype=int)
        df_train = df_training.iloc[train_idx].reset_index(drop=True)
        df_validation = df_training.iloc[validation_idx].reset_index(drop=True)
        selected, numeric, categorical, manifest = feature_audit(
            df_train,
            model_name=args.model,
            target=args.target,
            pfas_structure_features=args.pfas_structure_features,
            **feature_selection_kwargs(args),
        )
        if not selected:
            raise ValueError(f"Inner validation fold {fold} selected no features from its training subset.")
        manifest = manifest.copy()
        manifest.insert(0, "fit_scope", "inner_training")
        manifest.insert(1, "validation_fold", int(fold))
        manifests.append(manifest)
        prepared.append(
            PreparedValidationFold(
                validation_fold=int(fold),
                training_indices=train_idx,
                validation_indices=validation_idx,
                df_train=df_train,
                df_validation=df_validation,
                X_train=build_X(df_train, selected, numeric, categorical),
                y_train=y_training.iloc[train_idx].reset_index(drop=True),
                sample_weight_train=all_weights.iloc[train_idx].reset_index(drop=True),
                X_validation=build_X(df_validation, selected, numeric, categorical),
                y_validation=y_training.iloc[validation_idx].reset_index(drop=True),
                sample_weight_validation=all_weights.iloc[validation_idx].reset_index(drop=True),
                selected_features=selected,
                numeric_features=numeric,
                categorical_features=categorical,
                feature_manifest=manifest,
            )
        )
    return prepared, assignments, pd.concat(manifests, ignore_index=True) if manifests else pd.DataFrame()


def metrics_by_feature_availability(
    df_testing: pd.DataFrame,
    y_true: pd.Series,
    y_pred: np.ndarray,
    selected: list[str],
    numeric: list[str],
    categorical: list[str],
    sample_weight: pd.Series | None = None,
) -> pd.DataFrame:
    """Report testing metrics separately for observed and missing feature values."""
    kind_by_feature = {feature: "numeric" for feature in numeric}
    kind_by_feature.update({feature: "categorical" for feature in categorical})
    rows: list[dict[str, Any]] = []
    for feature in selected:
        kind = kind_by_feature[feature]
        present = feature_value_mask(df_testing, feature, kind).to_numpy(dtype=bool)
        for availability, mask in (("present", present), ("missing", ~present)):
            if not mask.any():
                continue
            rows.append(
                {
                    "feature": feature,
                    "kind": kind,
                    "availability": availability,
                    **metrics_dict(
                        y_true.to_numpy()[mask],
                        y_pred[mask],
                        None if sample_weight is None else sample_weight.to_numpy(dtype=float)[mask],
                    ),
                }
            )
    return pd.DataFrame(rows)


def validation_score(
    prepared_folds: list[PreparedValidationFold],
    params: dict[str, Any],
    args: argparse.Namespace,
) -> tuple[float, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    all_y_true: list[np.ndarray] = []
    all_y_pred: list[np.ndarray] = []
    all_weights: list[np.ndarray] = []
    for prepared in prepared_folds:
        pipe = make_pipeline(
            args.regressor,
            prepared.numeric_features,
            prepared.categorical_features,
            args.min_category_frequency,
            args.numeric_missing_strategy,
            params,
            model_random_state(args, prepared.validation_fold),
            args.n_jobs,
        )
        pipe.fit(
            prepared.X_train,
            prepared.y_train,
            model__sample_weight=prepared.sample_weight_train.to_numpy(dtype=float),
        )
        prediction = pipe.predict(prepared.X_validation)
        row = metrics_dict(
            prepared.y_validation,
            prediction,
            prepared.sample_weight_validation,
        )
        row.update(
            {
                "validation_fold": prepared.validation_fold,
                "selected_feature_count": len(prepared.selected_features),
                "numeric_feature_count": len(prepared.numeric_features),
                "categorical_feature_count": len(prepared.categorical_features),
            }
        )
        rows.append(row)
        all_y_true.append(prepared.y_validation.to_numpy(dtype=float))
        all_y_pred.append(np.asarray(prediction, dtype=float))
        all_weights.append(prepared.sample_weight_validation.to_numpy(dtype=float))
    fold_metrics = pd.DataFrame(rows)
    overall = metrics_dict(
        np.concatenate(all_y_true),
        np.concatenate(all_y_pred),
        np.concatenate(all_weights),
    )
    return float(overall["rmse"]), fold_metrics


def fold_rmse_noise_summary(fold_metrics: pd.DataFrame) -> dict[str, float | int]:
    """Summarize uncertainty of a candidate's validation-fold RMSE."""
    values = pd.to_numeric(fold_metrics.get("rmse", pd.Series(dtype=float)), errors="coerce").dropna().to_numpy(dtype=float)
    if not len(values):
        return {"fold_rmse_mean": float("nan"), "fold_rmse_sd": float("nan"), "fold_rmse_se": float("nan"), "n_validation_folds": 0}
    if len(values) == 1:
        return {"fold_rmse_mean": float(values[0]), "fold_rmse_sd": 0.0, "fold_rmse_se": 0.0, "n_validation_folds": 1}
    fold_sd = float(np.std(values, ddof=1))
    return {
        "fold_rmse_mean": float(np.mean(values)),
        "fold_rmse_sd": fold_sd,
        "fold_rmse_se": float(fold_sd / math.sqrt(len(values))),
        "n_validation_folds": int(len(values)),
    }


class ValidationNoiseAwarePlateauStopper:
    """Stop Optuna once best validation RMSE no longer beats its fold-level uncertainty."""

    def __init__(self, warmup_trials: int, patience: int, se_multiplier: float, min_delta_floor: float):
        self.warmup_trials = warmup_trials
        self.patience = patience
        self.se_multiplier = se_multiplier
        self.min_delta_floor = min_delta_floor
        self.reference_best_value: float | None = None
        self.reference_best_trial: int | None = None
        self.trials_since_meaningful_improvement = 0

    def __call__(self, study, trial) -> None:
        completed = [
            current
            for current in study.trials
            if current.state.name == "COMPLETE" and current.value is not None and np.isfinite(current.value)
        ]
        if len(completed) < self.warmup_trials:
            return
        best_trial = study.best_trial
        best_value = float(best_trial.value)
        best_se = best_trial.user_attrs.get("fold_rmse_se", np.nan)
        best_se = float(best_se) if pd.notna(best_se) and np.isfinite(best_se) else 0.0
        min_delta = max(self.min_delta_floor, self.se_multiplier * best_se)
        if self.reference_best_value is None:
            self.reference_best_value = best_value
            self.reference_best_trial = int(best_trial.number)
            self.trials_since_meaningful_improvement = 0
            return
        if best_value < self.reference_best_value - min_delta:
            self.reference_best_value = best_value
            self.reference_best_trial = int(best_trial.number)
            self.trials_since_meaningful_improvement = 0
        else:
            self.trials_since_meaningful_improvement += 1
        study.set_user_attr("early_stop_reference_best_value", self.reference_best_value)
        study.set_user_attr("early_stop_reference_best_trial", self.reference_best_trial)
        study.set_user_attr("early_stop_trials_since_meaningful_improvement", self.trials_since_meaningful_improvement)
        study.set_user_attr("early_stop_current_min_delta", min_delta)
        study.set_user_attr("early_stop_best_trial_fold_rmse_se", best_se)
        if self.trials_since_meaningful_improvement >= self.patience:
            study.set_user_attr(
                "early_stop_reason",
                (
                    f"No best-validation-RMSE improvement > {min_delta:.6g} for "
                    f"{self.patience} completed trials after {self.warmup_trials} warmup trials."
                ),
            )
            study.stop()


def _random_search_tune(
    prepared_folds: list[PreparedValidationFold], args: argparse.Namespace
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    rng = np.random.default_rng(args.random_seed)
    # ``n_trials`` is a total comparable tuning budget, not an additional
    # random-search budget on top of the fixed default candidate.
    candidates = [default_params(args.regressor)] + [
        sample_params(args.regressor, rng) for _ in range(max(args.n_trials - 1, 0))
    ]
    trial_rows: list[dict[str, Any]] = []
    fold_rows: list[pd.DataFrame] = []
    best_params = candidates[0]
    best_score = math.inf
    for trial, params in enumerate(candidates):
        score, fold_metrics = validation_score(prepared_folds, params, args)
        trial_rows.append({"trial": trial, "mean_validation_rmse": score, "params_json": json.dumps(params, sort_keys=True)})
        if not fold_metrics.empty:
            fold_metrics.insert(0, "trial", trial)
            fold_rows.append(fold_metrics)
        if score < best_score:
            best_score = score
            best_params = params
    results = pd.DataFrame(trial_rows)
    metadata = {
        "method": "seeded_random_search",
        "n_trials_requested": int(args.n_trials),
        "n_trials_completed": int(len(results)),
        "selection_metric": "equal_source_row_inner_validation_rmse",
        "early_stop_reason": "not_applicable",
    }
    return best_params, results, pd.concat(fold_rows, ignore_index=True) if fold_rows else pd.DataFrame(), metadata


def _progress_reporter(n_trials: int, every: int = 25):
    """Report tuning progress on a fixed cadence instead of per trial."""

    def report(study, trial) -> None:
        completed = len([t for t in study.trials if t.value is not None])
        if completed and (completed % every == 0 or completed == n_trials):
            best = study.best_value if study.best_trial is not None else float("nan")
            print(
                f"  tuned {completed}/{n_trials} candidates; "
                f"best inner-validation RMSE {best:.4f}",
                flush=True,
            )

    return report


def tune_params(
    prepared_folds: list[PreparedValidationFold], args: argparse.Namespace
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    if args.n_trials < 0:
        raise ValueError("--n-trials must be zero or a positive integer.")
    if args.early_stop_warmup < 0 or args.early_stop_patience < 0:
        raise ValueError("Optuna early-stop warmup and patience must be nonnegative.")
    if args.early_stop_se_multiplier < 0 or args.early_stop_min_delta_floor < 0:
        raise ValueError("Optuna early-stop thresholds must be nonnegative.")

    if args.n_trials == 0:
        params = default_params(args.regressor)
        score, fold_metrics = validation_score(prepared_folds, params, args)
        results = pd.DataFrame(
            [{"trial": 0, "mean_validation_rmse": score, "params_json": json.dumps(params, sort_keys=True), "search_method": "fixed_default"}]
        )
        metadata = {
            "method": "fixed_default",
            "n_trials_requested": 0,
            "n_trials_completed": 1,
            "selection_metric": "equal_source_row_inner_validation_rmse",
            "early_stop_reason": "not_applicable",
        }
        return params, results, fold_metrics.assign(trial=0), metadata

    if args.regressor != "xgb":
        return _random_search_tune(prepared_folds, args)

    try:
        import optuna
    except ImportError as exc:
        raise RuntimeError(
            "Optuna is required for XGBoost tuning when --n-trials is positive. "
            "Install it in the Python environment used by --python-executable."
        ) from exc

    trial_fold_rows: list[pd.DataFrame] = []

    def objective(trial) -> float:
        params = suggest_optuna_params(
            trial,
            args.xgb_search_space,
            compact_search_space=getattr(args, "compact_xgb_search_space", None),
        )
        trial.set_user_attr("resolved_params", params)
        score, fold_metrics = validation_score(prepared_folds, params, args)
        noise = fold_rmse_noise_summary(fold_metrics)
        min_delta = max(
            args.early_stop_min_delta_floor,
            args.early_stop_se_multiplier * noise["fold_rmse_se"]
            if pd.notna(noise["fold_rmse_se"])
            else args.early_stop_min_delta_floor,
        )
        for key, value in noise.items():
            trial.set_user_attr(key, value)
        trial.set_user_attr("noise_aware_min_delta", min_delta)
        trial_fold_rows.append(fold_metrics.assign(trial=trial.number))
        return score

    # Optuna logs one INFO line per finished trial.  Across a 200-trial budget
    # that buries the few lines describing what the run is actually doing, so
    # quiet mode replaces it with a periodic summary rather than silence.
    quiet = getattr(args, "console_verbosity", "quiet") == "quiet"
    optuna.logging.set_verbosity(optuna.logging.WARNING if quiet else optuna.logging.INFO)

    baseline_params = default_params(args.regressor)
    sampler = optuna.samplers.TPESampler(seed=args.random_seed)
    study = optuna.create_study(direction="minimize", sampler=sampler)
    # Keep the fixed configuration in the same selection pool as the adaptive
    # candidates.  This makes a positive trial budget directly comparable to
    # --n-trials 0 and ensures tuning cannot discard a stronger known baseline.
    study.enqueue_trial(
        baseline_params,
        user_attrs={"candidate_role": "fixed_default_baseline"},
    )
    callbacks = []
    if quiet:
        callbacks.append(_progress_reporter(args.n_trials))
    if args.early_stop_patience > 0:
        callbacks.append(
            ValidationNoiseAwarePlateauStopper(
                warmup_trials=args.early_stop_warmup,
                patience=args.early_stop_patience,
                se_multiplier=args.early_stop_se_multiplier,
                min_delta_floor=args.early_stop_min_delta_floor,
            )
        )
    study.optimize(objective, n_trials=args.n_trials, callbacks=callbacks, show_progress_bar=False)
    completed = [
        trial for trial in study.trials if trial.state.name == "COMPLETE" and trial.value is not None and np.isfinite(trial.value)
    ]
    if not completed:
        raise RuntimeError("Optuna completed no finite tuning trials.")
    results = study.trials_dataframe()
    if "number" in results.columns:
        results.insert(0, "trial", results["number"])
    if "value" in results.columns:
        results["mean_validation_rmse"] = results["value"]
        results["best_value_so_far"] = results["value"].cummin()
    resolved_params_by_trial = {
        int(trial.number): json.dumps(trial.user_attrs["resolved_params"], sort_keys=True)
        for trial in completed
        if isinstance(trial.user_attrs.get("resolved_params"), dict)
    }
    results["resolved_params_json"] = results["trial"].map(resolved_params_by_trial)
    results["search_method"] = "optuna_tpe"
    early_stop_reason = study.user_attrs.get("early_stop_reason")
    if not early_stop_reason:
        early_stop_reason = "Reached requested maximum trial count."
    if quiet:
        # Plateau stopping usually lands off the reporting cadence, so the run
        # states its own outcome rather than leaving the last periodic line as
        # the final word.
        print(
            f"  tuning finished: {len(completed)} candidates evaluated; "
            f"best inner-validation RMSE {study.best_value:.4f}. {early_stop_reason}",
            flush=True,
        )
    metadata = {
        "method": "optuna_tpe_validation_noise_aware_plateau",
        "n_trials_requested": int(args.n_trials),
        "n_trials_completed": int(len(completed)),
        "search_space": args.xgb_search_space,
        "baseline_candidate": {
            "included": True,
            "trial": 0,
            "params": baseline_params,
        },
        "selection_metric": "equal_source_row_inner_validation_rmse",
        "best_trial": int(study.best_trial.number),
        "best_validation_rmse": float(study.best_value),
        "early_stop_config": {
            "warmup_trials": int(args.early_stop_warmup),
            "patience": int(args.early_stop_patience),
            "se_multiplier": float(args.early_stop_se_multiplier),
            "min_delta_floor": float(args.early_stop_min_delta_floor),
        },
        "early_stop_reason": early_stop_reason,
        "study_user_attributes": dict(study.user_attrs),
    }
    best_params = study.best_trial.user_attrs.get("resolved_params")
    if not isinstance(best_params, dict):
        raise RuntimeError("Best Optuna trial did not retain its resolved XGBoost parameters.")
    return (
        dict(best_params),
        results,
        pd.concat(trial_fold_rows, ignore_index=True) if trial_fold_rows else pd.DataFrame(),
        metadata,
    )


def load_data(args: argparse.Namespace) -> tuple[pd.DataFrame, dict[str, Any], pd.DataFrame]:
    """Load model rows; feature diagnostics are calculated after fitting scope is known."""

    df, data_info, pfas_features = load_model_rows(
        input_path=args.input_path,
        sheet_name=args.sheet_name,
        model=args.model,
        target=args.target,
        pfas_features_path=args.pfas_features_path,
        pfas_features_sheet=args.pfas_features_sheet,
        skip_pfas_features_join=args.skip_pfas_features_join,
        data_mode=args.data_mode,
        kd_final_sources=args.kd_final_sources,
    )
    print(f"Loaded {len(df):,} model rows.", flush=True)
    return df, data_info, pfas_features


def selected_feature_support_diagnostics(
    feature_manifest: pd.DataFrame,
    reference_retention: pd.DataFrame,
    testing_input_support: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Consolidate selected-feature support evidence into one table.

    Feature eligibility and selection remain in ``feature_manifest.csv``. This
    compact companion table is intentionally limited to the fitted features and
    joins their reference-retention evidence with outer-test input support.
    """

    required_manifest_columns = {"feature", "bucket", "kind", "selected"}
    if missing := required_manifest_columns.difference(feature_manifest.columns):
        raise ValueError(
            "feature_manifest is missing required support columns: "
            + ", ".join(sorted(missing))
        )
    key_columns = ["feature", "bucket", "kind"]
    leading_columns = [column for column in ("fit_scope", *key_columns) if column in feature_manifest]
    selected = feature_manifest.loc[
        feature_manifest["selected"].fillna(False).astype(bool),
        leading_columns,
    ].copy()
    selected.insert(0, "diagnostic_scope", "selected_feature_support")

    reference = reference_retention.copy()
    if not reference.empty:
        missing = set(key_columns).difference(reference.columns)
        if missing:
            raise ValueError(
                "reference_retention is missing required support columns: "
                + ", ".join(sorted(missing))
            )
        reference = reference.loc[reference["feature"].isin(selected["feature"])].copy()
    merged = selected.merge(reference, on=key_columns, how="left", validate="one_to_one")

    if testing_input_support is not None and not testing_input_support.empty:
        missing = set(key_columns).difference(testing_input_support.columns)
        if missing:
            raise ValueError(
                "testing_input_support is missing required support columns: "
                + ", ".join(sorted(missing))
            )
        outer_test = testing_input_support.drop(columns=["diagnostic_type"], errors="ignore")
        merged = merged.merge(outer_test, on=key_columns, how="left", validate="one_to_one")
    return merged


def remove_legacy_support_outputs(output_dir: Path) -> None:
    """Prevent an in-place rerun from retaining superseded support tables."""

    for filename in (
        "coverage_by_feature.csv",
        "joint_input_support.csv",
        "testing_input_support_by_feature.csv",
    ):
        (output_dir / filename).unlink(missing_ok=True)


def run_full_data_final_refit(
    args: argparse.Namespace,
    output_dir: Path,
    deployment_exporter: Callable[[dict[str, Any]], dict[str, Any]],
) -> None:
    """Tune by inner CV on all rows, then fit the final deployment artifacts.

    This path intentionally performs no outer holdout. Its inner-CV scores guide
    the final all-data fit, while generalization claims must come from a separate
    repeated outer-resampling study run before deployment.
    """

    df, data_info, pfas_features = load_data(args)
    y = df[args.target].astype(float).reset_index(drop=True)
    all_weights = equal_source_row_weights(df)
    _, _, _, allocation_reference_manifest = feature_audit(
        df,
        model_name=args.model,
        target=args.target,
        pfas_structure_features=args.pfas_structure_features,
        **feature_selection_kwargs(args),
    )

    print("Preparing full-data inner-validation folds and fold-specific feature audits.", flush=True)
    prepared_folds, validation_assignments, validation_feature_manifests = prepare_validation_folds(
        df,
        y,
        args,
        allocation_reference_manifest,
    )
    print("Tuning model parameters by inner validation on all eligible rows.", flush=True)
    best_params, validation_results, validation_fold_metrics, tuning_metadata = tune_params(
        prepared_folds,
        args,
    )

    selected, numeric, categorical, feature_manifest = feature_audit(
        df,
        model_name=args.model,
        target=args.target,
        pfas_structure_features=args.pfas_structure_features,
        **feature_selection_kwargs(args),
    )
    if not selected:
        raise ValueError("No features selected from all eligible rows. Inspect final_feature_manifest.csv.")
    feature_manifest = feature_manifest.copy()
    feature_manifest.insert(0, "fit_scope", "all_eligible_rows_final_refit")
    X = build_X(df, selected, numeric, categorical)

    # This apparent metric is retained only as a diagnostic. It must never be
    # presented as a generalization estimate because every row was used to fit it.
    diagnostic_pipe = make_pipeline(
        args.regressor,
        numeric,
        categorical,
        args.min_category_frequency,
        args.numeric_missing_strategy,
        best_params,
        model_random_state(args),
        args.n_jobs,
    )
    diagnostic_pipe.fit(X, y, model__sample_weight=all_weights.to_numpy(dtype=float))
    apparent_prediction = np.asarray(diagnostic_pipe.predict(X), dtype=float)
    apparent_metrics = metrics_dict(y, apparent_prediction, all_weights)
    all_predictions = prediction_context(
        df,
        np.arange(len(df), dtype=int),
        y,
        apparent_prediction,
        args.target,
    )
    all_predictions["source_row_weight"] = all_weights.to_numpy(dtype=float)
    all_predictions.insert(0, "data_split", "all_data_refit_apparent")

    reference_retention = reference_retention_by_feature(
        df,
        df,
        feature_manifest,
        scope="all_data_training_equals_complete_reference",
    )
    retention_summary = summarize_reference_retention(reference_retention)
    selected_feature_support = selected_feature_support_diagnostics(
        feature_manifest,
        reference_retention,
    )
    feature_manifest.to_csv(output_dir / "final_feature_manifest.csv", index=False)
    selected_feature_support.to_csv(output_dir / "selected_feature_support.csv", index=False)
    remove_legacy_support_outputs(output_dir)
    validation_feature_manifests.to_csv(output_dir / "validation_feature_manifests.csv", index=False)
    validation_assignments.to_csv(output_dir / "validation_assignments.csv", index=False)
    validation_diagnostics(df, validation_assignments, args.target).to_csv(
        output_dir / "validation_diagnostics.csv",
        index=False,
    )
    validation_results.to_csv(output_dir / "validation_results.csv", index=False)
    validation_fold_metrics.to_csv(output_dir / "validation_fold_metrics.csv", index=False)
    (output_dir / "tuning_summary.json").write_text(
        json.dumps(tuning_metadata, indent=2),
        encoding="utf-8",
    )
    all_predictions.to_csv(output_dir / "all_predictions.csv", index=False)
    pd.DataFrame(
        [
            {
                "split": "all_data_refit_apparent",
                "metric_weighting": "equal_source_row",
                "metric_interpretation": "diagnostic_only_not_a_generalization_estimate",
                **apparent_metrics,
                **retention_summary,
            }
        ]
    ).to_csv(output_dir / "metrics_summary.csv", index=False)

    realized_validation_folds = int(
        validation_assignments.loc[
            validation_assignments["validation_fold"].ge(0), "validation_fold"
        ].nunique()
    )
    config: dict[str, Any] = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "input_path": str(args.input_path),
        "sheet_name": args.sheet_name,
        "pfas_features": {
            "joined": not args.skip_pfas_features_join,
            "path": str(args.pfas_features_path),
            "sheet_name": args.pfas_features_sheet,
        },
        "model_family": args.model,
        "target": args.target,
        "kd_final_sources": list(args.kd_final_sources) if args.kd_final_sources else None,
        "output_dir": str(output_dir),
        "data": data_info,
        "analysis_reconstruction": {
            "format": "logkd_shap_v4",
            "input_path": str(args.input_path),
            "sheet_name": args.sheet_name,
            "model": args.model,
            "target": args.target,
            "pfas_features_path": str(args.pfas_features_path) if args.pfas_features_path else None,
            "pfas_features_sheet": args.pfas_features_sheet,
            "skip_pfas_features_join": bool(args.skip_pfas_features_join),
            "data_mode": args.data_mode,
            "kd_final_sources": list(args.kd_final_sources) if args.kd_final_sources else None,
        },
        "rows": {
            "total_model_rows": int(len(df)),
            "all_data_refit": int(len(df)),
            "testing": 0,
        },
        "source_row_weighting": {
            "method": "equal_total_weight_per_source_row",
            "source_group_column": "source_row_index",
            "all_data_source_rows": int(df["source_row_index"].nunique()),
        },
        "evaluation": {
            "mode": "deployment_finalization",
            "outer_test_performed": False,
            "generalization_metrics_source": "separate_repeated_outer_resampling_required",
            "apparent_refit_metrics_are_generalization_estimates": False,
        },
        "split": {
            "strategy": args.split_strategy,
            "allocation_method": None,
            "assignment_path": None,
            "protocol": "no_outer_holdout; inner validation on all eligible rows followed by all-data refit",
        },
        "inner_validation": {
            "requested_folds": args.validation_folds,
            "realized_folds": realized_validation_folds,
            "allocation_method": args.inner_allocation_method,
            "reference_retention_allocation": dict(
                validation_assignments.attrs.get("reference_retention_allocation", {})
            ),
        },
        "features": {
            "selected": selected,
            "selected_by_bucket": group_selected_features_by_bucket(selected),
            "numeric": numeric,
            "categorical": categorical,
            "pfas_structure_features": args.pfas_structure_features,
            "selection_policy": feature_selection_kwargs(args),
            "numeric_missing_strategy": args.numeric_missing_strategy,
            "reference_retention": retention_summary,
        },
        "regressor": args.regressor,
        "best_params": best_params,
        "hyperparameter_tuning": tuning_metadata,
        "training_settings": {
            "random_seed": int(args.random_seed),
            "model_random_seed": int(args.model_random_seed),
            "random_seed_role": "split allocation, validation allocation, and hyperparameter-search sampling",
            "model_random_seed_role": "estimator random_state; validation folds use deterministic offsets",
            "n_jobs": int(args.n_jobs),
            "min_category_frequency": int(args.min_category_frequency),
            "numeric_missing_strategy": args.numeric_missing_strategy,
        },
        "metrics": {"all_data_refit_apparent": apparent_metrics},
    }
    config["deployment"] = deployment_exporter(
        {
            "args": args,
            "output_dir": output_dir,
            "data": df,
            "data_info": data_info,
            "pfas_features": pfas_features,
            "best_params": best_params,
        }
    )
    (output_dir / "run_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    print(f"Wrote final deployment outputs to: {console_path(output_dir)}")


def prediction_context(
    df: pd.DataFrame,
    indices: np.ndarray,
    y_true: pd.Series,
    y_pred: np.ndarray,
    target: str,
) -> pd.DataFrame:
    context_columns = [
        "standardized_row_id",
        "source_row_index",
        "extraction_dataset",
        "study_no",
        "DOI",
        "PFAS_name",
        "adsorbent_id",
        "name_abbreviation",
        "name_commercial",
        "name_full",
        "adsorbent_category",
        "adsorbent_subcategory",
        "Kd_final_source",
        "apparent_removal_fraction",
        "apparent_removal_source",
        "logKd_conversion_spike_flag",
        "logKd_conversion_dip_flag",
        "low_Ce_detection_limit_flag",
        "concentration_difference_C0_minus_Ce_mg/L",
        "small_concentration_difference_flag",
        "low_removal_rate_flag",
        "high_apparent_removal_flag",
        "PFAS_C0_value_mg/L",
        "Adsorbent_dosage_value_mg/L",
        "pH",
        "Temperature_(°C)",
        "Contact_time_(h)",
        "Solution_volume_(mL)",
        "Water_type",
        target,
    ]
    context_columns = [column for column in context_columns if column in df.columns]
    out = df.iloc[indices][context_columns].copy().rename(columns={target: "y_true"})
    out["y_pred"] = y_pred
    out["residual"] = out["y_pred"] - out["y_true"]
    out["absolute_error"] = out["residual"].abs()
    out["squared_error"] = out["residual"] ** 2
    return out.reset_index(drop=True)


def main(
    *,
    deployment_exporter: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
    argument_extender: Callable[[argparse.ArgumentParser], None] | None = None,
    final_full_data_tuning: bool = False,
) -> None:
    args = parse_args(
        argument_extender,
        require_outer_allocation_method=not final_full_data_tuning,
    )
    args.inner_allocation_method = resolve_inner_allocation_method(args)
    if final_full_data_tuning:
        if deployment_exporter is None:
            raise ValueError("Final full-data tuning requires a deployment exporter.")
        if args.prepare_split_only or args.split_assignment_path is not None:
            raise ValueError(
                "Final full-data tuning cannot prepare or consume an outer split assignment. "
                "Use the exploratory trainer for outer-split evaluation."
            )
    if args.prepare_split_only and args.split_assignment_path is not None:
        raise ValueError("--prepare-split-only constructs a new assignment and cannot use --split-assignment-path.")
    output_dir = args.output_dir
    if output_dir is None:
        label = f"{args.model}_{args.regressor}_{args.split_strategy}_{args.random_seed}_{datetime.now():%Y%m%d_%H%M%S}"
        output_dir = args.output_root / label
    output_dir.mkdir(parents=True, exist_ok=True)

    if final_full_data_tuning:
        run_full_data_final_refit(args, output_dir, deployment_exporter)
        return

    df, data_info, pfas_features = load_data(args)
    if args.split_assignment_path is None:
        print(
            f"Constructing a {args.outer_allocation_method} {args.split_strategy} testing split "
            f"(target fraction {args.test_fraction:.3f}).",
            flush=True,
        )
        split_result = build_testing_split(
            df,
            split_strategy=args.split_strategy,
            test_fraction=args.test_fraction,
            random_seed=args.random_seed,
            target=args.target,
            allocation_method=args.outer_allocation_method,
        )
        print(
            f"Constructed split: {int(split_result.assignments['split'].eq(TESTING).sum()):,} testing rows; "+
            f"{int(split_result.assignments['split'].eq(TRAINING).sum()):,} training rows.",
            flush=True,
        )
        assignments = split_result.assignments
        split_diagnostic = split_result.diagnostics
        split_validation = split_result.validation
    else:
        assignments = load_testing_assignments(args.split_assignment_path, df)
        assignment_strategy = str(assignments["split_strategy"].iloc[0])
        if assignment_strategy != args.split_strategy:
            raise ValueError(
                f"--split-strategy {args.split_strategy!r} does not match the frozen assignment strategy {assignment_strategy!r}."
            )
        assignment_construction = str(assignments["split_construction"].iloc[0])
        protocol = outer_allocation_protocol_metadata(args.split_strategy, args.outer_allocation_method)
        if assignment_construction != protocol["split_construction"]:
            raise ValueError(
                f"Frozen assignment construction {assignment_construction!r} is incompatible with "
                f"--outer-allocation-method {args.outer_allocation_method!r}."
            )
        split_diagnostic = split_diagnostics(df, assignments, args.target)
        split_validation = validate_testing_split(assignments)
        observed_test_fraction = float(assignments["split"].eq(TESTING).mean())
        frozen_reporting = frozen_assignment_reporting_metadata(
            args.split_assignment_path,
            assignments,
            args.test_fraction,
        )
        split_validation.update({
            "assignment_source": str(args.split_assignment_path),
            "testing_rows": int(assignments["split"].eq(TESTING).sum()),
            "training_rows": int(assignments["split"].eq(TRAINING).sum()),
            "observed_test_fraction": observed_test_fraction,
            **protocol,
            **frozen_reporting,
            "split_construction": assignment_construction,
        })

    if args.split_strategy == "combination":
        split_validation["combination_component_representation_policy"] = (
            "enforced; each test PFAS and adsorbent remains represented in outer training"
        )

    if args.prepare_split_only:
        assignments.to_csv(output_dir / "split_assignments.csv", index=False)
        split_diagnostic.to_csv(output_dir / "split_diagnostics.csv", index=False)
        (output_dir / "split_preparation.json").write_text(
            json.dumps({"split": {"strategy": args.split_strategy, "validation": split_validation}}, indent=2),
            encoding="utf-8",
        )
        print(f"Wrote frozen split preparation to: {console_path(output_dir)}")
        return

    training_idx, testing_idx = split_indices(assignments)
    df_training = df.iloc[training_idx].reset_index(drop=True)
    df_testing = df.iloc[testing_idx].reset_index(drop=True)
    y_training = df_training[args.target].astype(float).reset_index(drop=True)
    y_testing = df_testing[args.target].astype(float).reset_index(drop=True)
    training_weights = equal_source_row_weights(df_training)
    testing_weights = equal_source_row_weights(df_testing)

    # The inner allocator must see only the frozen outer-training cohort.  This
    # manifest is fixed before any individual validation fold is held out.
    _, _, _, allocation_reference_manifest = feature_audit(
        df_training,
        model_name=args.model,
        target=args.target,
        pfas_structure_features=args.pfas_structure_features,
        **feature_selection_kwargs(args),
    )

    print("Preparing inner-validation folds and fold-specific feature audits.", flush=True)
    prepared_folds, validation_assignments, validation_feature_manifests = prepare_validation_folds(
        df_training,
        y_training,
        args,
        allocation_reference_manifest,
    )
    print("Tuning model parameters within the training partition.", flush=True)
    best_params, validation_results, validation_fold_metrics, tuning_metadata = tune_params(prepared_folds, args)
    print("Refitting the final model on training data and evaluating the frozen testing set.", flush=True)
    _, _, _, reference_feature_manifest = feature_audit(
        df,
        model_name=args.model,
        target=args.target,
        pfas_structure_features=args.pfas_structure_features,
        **feature_selection_kwargs(args),
    )
    selected, numeric, categorical, feature_manifest = feature_audit(
        df_training,
        model_name=args.model,
        target=args.target,
        pfas_structure_features=args.pfas_structure_features,
        **feature_selection_kwargs(args),
    )
    if not selected:
        raise ValueError("No features selected from the training partition. Inspect feature_manifest.csv.")
    feature_manifest = feature_manifest.copy()
    feature_manifest.insert(0, "fit_scope", "training_refit")
    reference_retention = reference_retention_by_feature(
        df,
        df_training,
        reference_feature_manifest,
        scope="outer_training_vs_complete_reference",
    )
    retention_summary = summarize_reference_retention(reference_retention)
    X_training = build_X(df_training, selected, numeric, categorical)
    X_testing = build_X(df_testing, selected, numeric, categorical)

    final_pipe = make_pipeline(
        args.regressor,
        numeric,
        categorical,
        args.min_category_frequency,
        args.numeric_missing_strategy,
        best_params,
        model_random_state(args),
        args.n_jobs,
    )
    final_pipe.fit(
        X_training,
        y_training,
        model__sample_weight=training_weights.to_numpy(dtype=float),
    )
    training_prediction = final_pipe.predict(X_training)
    testing_prediction = final_pipe.predict(X_testing)
    training_metrics = metrics_dict(y_training, training_prediction, training_weights)
    testing_metrics = metrics_dict(y_testing, testing_prediction, testing_weights)
    testing_logkd_tail_metrics = logkd_tail_metrics_dict(
        y_testing,
        testing_prediction,
        testing_weights,
        target_scale="kd" if args.target == "Kd_final_L/g" else "logkd",
    )
    testing_feature_availability_metrics = metrics_by_feature_availability(
        df_testing,
        y_testing,
        testing_prediction,
        selected,
        numeric,
        categorical,
        testing_weights,
    )
    joint_support = joint_support_coverage(df_training, selected, numeric, categorical)
    feature_input_support, _ = input_support_diagnostics(
        df_training,
        df_testing,
        selected,
        numeric,
        categorical,
        feature_manifest=feature_manifest,
    )
    training_predictions = prediction_context(df, training_idx, y_training, training_prediction, args.target)
    training_predictions["source_row_weight"] = training_weights.to_numpy(dtype=float)
    training_predictions.insert(0, "data_split", "training_refit")
    testing_predictions = prediction_context(df, testing_idx, y_testing, testing_prediction, args.target)
    testing_predictions["source_row_weight"] = testing_weights.to_numpy(dtype=float)
    testing_predictions.insert(0, "data_split", TESTING)
    all_predictions = pd.concat([training_predictions, testing_predictions], ignore_index=True, sort=False)

    split_assignments = assignments.reset_index(drop=True)
    selected_feature_support = selected_feature_support_diagnostics(
        feature_manifest,
        reference_retention,
        feature_input_support,
    )
    metrics_summary = pd.DataFrame(
        [
            {"split": "training_refit", "metric_weighting": "equal_source_row", **training_metrics, **retention_summary},
            {
                "split": TESTING,
                "metric_weighting": "equal_source_row",
                **testing_metrics,
                **testing_logkd_tail_metrics,
                **retention_summary,
            },
        ]
    )

    feature_manifest.to_csv(output_dir / "feature_manifest.csv", index=False)
    selected_feature_support.to_csv(output_dir / "selected_feature_support.csv", index=False)
    validation_feature_manifests.to_csv(output_dir / "validation_feature_manifests.csv", index=False)
    joint_support.to_csv(output_dir / "input_block_support.csv", index=False)
    remove_legacy_support_outputs(output_dir)
    split_assignments.to_csv(output_dir / "split_assignments.csv", index=False)
    split_diagnostic.to_csv(output_dir / "split_diagnostics.csv", index=False)
    validation_assignments.to_csv(output_dir / "validation_assignments.csv", index=False)
    validation_diagnostics(df_training, validation_assignments, args.target).to_csv(output_dir / "validation_diagnostics.csv", index=False)
    validation_results.to_csv(output_dir / "validation_results.csv", index=False)
    validation_fold_metrics.to_csv(output_dir / "validation_fold_metrics.csv", index=False)
    (output_dir / "tuning_summary.json").write_text(json.dumps(tuning_metadata, indent=2), encoding="utf-8")
    all_predictions.to_csv(output_dir / "all_predictions.csv", index=False)
    metrics_summary.to_csv(output_dir / "metrics_summary.csv", index=False)
    model_filename: str | None = None
    if args.save_model:
        import joblib

        model_filename = "model.joblib"
        joblib.dump(final_pipe, output_dir / model_filename)
    testing_feature_availability_metrics.to_csv(
        output_dir / "testing_metrics_by_feature_availability.csv", index=False
    )
    grouped_metrics(testing_predictions, "adsorbent_subcategory").to_csv(
        output_dir / "testing_metrics_by_adsorbent_subcategory.csv", index=False
    )
    realized_validation_folds = int(validation_assignments.loc[validation_assignments["validation_fold"].ge(0), "validation_fold"].nunique())
    config = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "input_path": str(args.input_path),
        "sheet_name": args.sheet_name,
        "pfas_features": {
            "joined": not args.skip_pfas_features_join,
            "path": str(args.pfas_features_path),
            "sheet_name": args.pfas_features_sheet,
        },
        "model_family": args.model,
        "target": args.target,
        "kd_final_sources": list(args.kd_final_sources) if args.kd_final_sources else None,
        "output_dir": str(output_dir),
        "data": data_info,
        "analysis_reconstruction": {
            "format": "logkd_shap_v4",
            "input_path": str(args.input_path),
            "sheet_name": args.sheet_name,
            "model": args.model,
            "target": args.target,
            "pfas_features_path": str(args.pfas_features_path) if args.pfas_features_path else None,
            "pfas_features_sheet": args.pfas_features_sheet,
            "skip_pfas_features_join": bool(args.skip_pfas_features_join),
            "data_mode": args.data_mode,
            "kd_final_sources": list(args.kd_final_sources) if args.kd_final_sources else None,
        },
        "rows": {
            "total_model_rows": int(len(df)),
            "training": int(len(training_idx)),
            "testing": int(len(testing_idx)),
        },
        "source_row_weighting": {
            "method": "equal_total_weight_per_source_row",
            "source_group_column": "source_row_index",
            "training_source_rows": int(df_training["source_row_index"].nunique()),
            "testing_source_rows": int(df_testing["source_row_index"].nunique()),
        },
        "split": {
            "strategy": args.split_strategy,
            "allocation_method": args.outer_allocation_method,
            "combination_component_representation_enforced": args.split_strategy == "combination",
            "requested_test_fraction": args.test_fraction,
            "assignment_path": str(args.split_assignment_path) if args.split_assignment_path else None,
            "validation": split_validation,
            "protocol": "one frozen testing; inner validation only within training",
        },
        "inner_validation": {
            "requested_folds": args.validation_folds,
            "realized_folds": realized_validation_folds,
            "allocation_method": args.inner_allocation_method,
            "reference_retention_allocation": dict(
                validation_assignments.attrs.get("reference_retention_allocation", {})
            ),
        },
        "features": {
            "selected": selected,
            "selected_by_bucket": group_selected_features_by_bucket(selected),
            "numeric": numeric,
            "categorical": categorical,
            "pfas_structure_features": args.pfas_structure_features,
            "selection_policy": feature_selection_kwargs(args),
            "support_diagnostics": [
                "feature_eligibility_and_data_sufficiency",
                "selected_feature_support",
                "input_block_support",
            ],
            "numeric_missing_strategy": args.numeric_missing_strategy,
            "reference_retention": retention_summary,
        },
        "regressor": args.regressor,
        "best_params": best_params,
        "hyperparameter_tuning": tuning_metadata,
        "training_settings": {
            "random_seed": int(args.random_seed),
            "model_random_seed": int(args.model_random_seed),
            "random_seed_role": "split allocation, validation allocation, and hyperparameter-search sampling",
            "model_random_seed_role": "estimator random_state; validation folds use deterministic offsets",
            "n_jobs": int(args.n_jobs),
            "min_category_frequency": int(args.min_category_frequency),
            "numeric_missing_strategy": args.numeric_missing_strategy,
        },
        "model_artifacts": {"model_filename": model_filename},
        "metrics": {
            "training_refit": training_metrics,
            TESTING: testing_metrics,
            "testing_logkd_tail_diagnostics": testing_logkd_tail_metrics,
        },
    }
    (output_dir / "run_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    if deployment_exporter is not None:
        config["deployment"] = deployment_exporter({
            "args": args,
            "output_dir": output_dir,
            "data": df,
            "data_info": data_info,
            "pfas_features": pfas_features,
            "best_params": best_params,
        })
        (output_dir / "run_config.json").write_text(
            json.dumps(config, indent=2), encoding="utf-8"
        )
    print(f"Wrote outputs to: {console_path(output_dir)}")
    print(metrics_summary.to_string(index=False))


if __name__ == "__main__":
    main()
