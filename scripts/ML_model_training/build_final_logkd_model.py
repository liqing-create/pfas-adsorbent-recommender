"""Build one deployable logKd model after an exploratory configuration is chosen.

This entry point tunes hyperparameters only through grouped inner validation on
all eligible rows, then refits the chosen configuration on those same rows.
Outer-split evaluation belongs in the separate exploratory resampling workflow;
only this script writes deployment artifacts.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

SCRIPTS_DIR = Path(__file__).resolve().parent.parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from adsorbent_recommender.bundle import (
    BUNDLE_FILENAME,
    BUNDLE_FORMAT,
    BUNDLE_VERSION,
    DEFAULT_FIXED_DOSE_MG_L,
    derive_feature_support_and_controls,
    export_recommender_bundle,
)
from adsorbent_recommender.catalog import DEFAULT_CATALOG_PATH, DEFAULT_CATALOG_SHEET, present
from adsorbent_recommender.evidence import DEFAULT_EVIDENCE_PATH, DEFAULT_EVIDENCE_SHEET
from backend.logkd_data import equal_source_row_weights, normalize_pfas_key
from backend.logkd_features import (
    build_X,
    feature_audit,
    feature_selection_kwargs,
    group_selected_features_by_bucket,
)
from train_logkd_model import (
    main as run_training,
    make_pipeline,
    model_random_state,
)


def add_deployment_arguments(parser: argparse.ArgumentParser) -> None:
    """Add arguments that are meaningful only for a deployable final model."""

    parser.description = (
        "Tune a chosen logKd recipe through grouped inner validation on all "
        "eligible rows, refit it on those rows, and export one deployable "
        "recommender bundle. Generalization metrics belong to a separate "
        "outer-resampling evaluation."
    )
    parser.add_argument("--catalog-path", type=Path, default=DEFAULT_CATALOG_PATH)
    parser.add_argument("--catalog-sheet", default=DEFAULT_CATALOG_SHEET)
    parser.add_argument(
        "--evidence-path",
        type=Path,
        default=DEFAULT_EVIDENCE_PATH,
        help="25 mg/L reference-dose workbook copied into the deployment bundle.",
    )
    parser.add_argument("--evidence-sheet", default=DEFAULT_EVIDENCE_SHEET)
    # The exploratory defaults suit the screening runners, not deployment.  The
    # recommender scores PFAS-adsorbent pairs across every adsorbent class, so
    # it is built on the pooled cohort, and its inner folds are grouped the way
    # the chosen combination-split evaluation grouped its outer testing rows.
    parser.set_defaults(model="Global", split_strategy="combination")
    parser.add_argument(
        "--fixed-dose-mg-l",
        type=float,
        default=DEFAULT_FIXED_DOSE_MG_L,
        help=(
            "Adsorbent dose the recommender scores every candidate at, so the "
            "ranking reflects affinity rather than a per-candidate dose search. "
            "Scoring only: training rows keep their own recorded doses."
        ),
    )
    parser.add_argument(
        "--ensemble-size",
        type=int,
        default=5,
        help="Number of deterministic all-data refits saved for prediction dispersion.",
    )


def _fit_ensemble(
    *,
    X: pd.DataFrame,
    y: pd.Series,
    weights: pd.Series,
    selected: list[str],
    numeric: list[str],
    categorical: list[str],
    best_params: dict[str, Any],
    args: argparse.Namespace,
) -> list[Any]:
    models: list[Any] = []
    for offset in range(args.ensemble_size):
        model = make_pipeline(
            args.regressor,
            numeric,
            categorical,
            args.min_category_frequency,
            args.numeric_missing_strategy,
            best_params,
            # Estimator stochasticity is seeded from the estimator seed, never
            # from the split and tuning seed, matching every other fit in the
            # project including this run's own diagnostic refit.
            model_random_state(args, offset),
            args.n_jobs,
        )
        model.fit(X, y, model__sample_weight=weights.to_numpy(dtype=float))
        models.append(model)
    return models


def export_final_deployment(context: dict[str, Any]) -> dict[str, Any]:
    """Refit on all eligible rows and write the sole deployment package."""

    args: argparse.Namespace = context["args"]
    if args.ensemble_size < 1:
        raise ValueError("--ensemble-size must be at least 1.")

    output_dir = Path(context["output_dir"])
    frame: pd.DataFrame = context["data"].reset_index(drop=True)
    target = args.target
    y = frame[target].astype(float).reset_index(drop=True)
    weights = equal_source_row_weights(frame)

    selected, numeric, categorical, feature_manifest = feature_audit(
        frame,
        model_name=args.model,
        target=target,
        pfas_structure_features=args.pfas_structure_features,
        **feature_selection_kwargs(args),
    )
    if not selected:
        raise ValueError("No features were selected for the all-data final refit.")

    X = build_X(frame, selected, numeric, categorical)
    models = _fit_ensemble(
        X=X,
        y=y,
        weights=weights,
        selected=selected,
        numeric=numeric,
        categorical=categorical,
        best_params=context["best_params"],
        args=args,
    )
    joblib.dump(
        {
            "format": "logkd_prediction_ensemble_v1",
            "models": models,
            "selected_features": selected,
            "numeric_features": numeric,
            "categorical_features": categorical,
        },
        output_dir / "model_ensemble.joblib",
    )

    if args.fixed_dose_mg_l <= 0:
        raise ValueError("--fixed-dose-mg-l must be a positive dose in mg/L.")
    bounds = derive_feature_support_and_controls(
        frame,
        selected,
        numeric,
        categorical,
        control_defaults={"Adsorbent_dosage_value_mg/L": args.fixed_dose_mg_l},
    )
    bounds["source_partition"] = "all_eligible_rows_final_refit"
    bounds["known_pfas_keys"] = sorted(
        {
            normalize_pfas_key(value)
            for value in frame["PFAS_name"]
            if normalize_pfas_key(value)
        }
    )
    feature_manifest = feature_manifest.copy()
    feature_manifest.insert(0, "fit_scope", "all_eligible_rows_final_refit")
    feature_manifest.to_csv(output_dir / "final_feature_manifest.csv", index=False)
    # A catalog product is scored only if it is characterized at least as well
    # as a typical row the model was fitted on.  The threshold is measured here
    # rather than fixed in code, so it follows the feature set this deployment
    # actually selected instead of a number recorded when some earlier one did.
    material_features = group_selected_features_by_bucket(selected).get("Adsorbent properties", [])
    # Presence is decided by the same predicate the catalog audit applies, not by
    # notna(): an unreported value reaches this frame as an empty string, which
    # is not null, so counting nulls would call every row completely
    # characterized and set the bar at every feature.
    material_present = pd.DataFrame(
        {
            feature: frame[feature].map(present) if feature in frame.columns else False
            for feature in material_features
        },
        index=frame.index,
    )
    material_coverage = (
        material_present.sum(axis=1) if material_features else pd.Series(dtype=float)
    )
    minimum_material_feature_count = (
        int(material_coverage.median()) if not material_coverage.empty else None
    )
    bounds["material_feature_coverage"] = {
        "required_material_feature_count": len(material_features),
        "training_median_present": minimum_material_feature_count,
        "training_maximum_present": int(material_coverage.max()) if not material_coverage.empty else None,
        "policy": "a catalog product must match the median training row's material-feature coverage",
    }

    (output_dir / "optimizer_bounds.json").write_text(
        json.dumps(bounds, indent=2), encoding="utf-8"
    )
    manifest = export_recommender_bundle(
        output_dir=output_dir,
        minimum_material_feature_count=minimum_material_feature_count,
        model_family=args.model,
        target=target,
        selected=selected,
        numeric_features=numeric,
        categorical_features=categorical,
        selected_by_bucket=group_selected_features_by_bucket(selected),
        bounds=bounds,
        catalog_path=args.catalog_path,
        catalog_sheet=args.catalog_sheet,
        model_adsorbent_ids=frame.get("adsorbent_id", pd.Series(dtype=object)),
        evidence_path=args.evidence_path,
        evidence_sheet=args.evidence_sheet,
        pfas_features_path=None if args.skip_pfas_features_join else args.pfas_features_path,
        pfas_features_sheet=args.pfas_features_sheet,
    )
    return {
        "protocol": (
            "hyperparameters tuned by inner validation on all eligible rows, "
            "then refit on all eligible rows for deployment; generalization "
            "metrics are supplied by separate outer-resampling evaluation"
        ),
        "training_rows": int(len(frame)),
        "source_rows": int(frame["source_row_index"].nunique()),
        "ensemble_size": int(args.ensemble_size),
        "features": {
            "selected": selected,
            "numeric": numeric,
            "categorical": categorical,
        },
        "bundle": {
            "format": BUNDLE_FORMAT,
            "version": BUNDLE_VERSION,
            "manifest_filename": BUNDLE_FILENAME,
            "status": manifest.get("status"),
        },
        "artifacts": {
            "ensemble": "model_ensemble.joblib",
            "feature_manifest": "final_feature_manifest.csv",
            "feature_support": "optimizer_bounds.json",
        },
    }


if __name__ == "__main__":
    run_training(
        deployment_exporter=export_final_deployment,
        argument_extender=add_deployment_arguments,
        final_full_data_tuning=True,
    )
