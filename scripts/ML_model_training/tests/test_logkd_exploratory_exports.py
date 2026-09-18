from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from backend.logkd_exploratory_exports import (
    model_run_directory,
    model_runs_root,
    quarantine_incomplete_model_runs,
    write_exploratory_outputs,
)


def write_run_artifacts(
    run_dir: Path,
    *,
    regressor: str,
    prediction: float,
) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([{
        "diagnostic_scope": "selected_feature_support",
        "feature": "feature_a",
        "bucket": "Experimental conditions",
        "kind": "numeric",
        "coverage_scope": "outer_training_vs_complete_reference",
        "entity_retention": 0.9,
        "contrast_retention": 0.8,
        "testing_nonmissing_rows": 1,
        "testing_outside_range_values": 0,
    }]).to_csv(
        run_dir / "selected_feature_support.csv",
        index=False,
    )
    pd.DataFrame([{
        "data_split": "testing",
        "row_id": "r2",
        "y_true": 1.0,
        "y_pred": prediction,
    }]).to_csv(
        run_dir / "all_predictions.csv",
        index=False,
    )
    pd.DataFrame([{
        "validation_fold": 0,
        "mae": abs(prediction - 1.0),
    }]).to_csv(
        run_dir / "validation_fold_metrics.csv",
        index=False,
    )
    pd.DataFrame([
        {
            "validation_fold": 0,
            "validation_reference_retention_score": 0.80,
        },
        {
            "validation_fold": 1,
            "validation_reference_retention_score": 0.60,
        },
    ]).to_csv(run_dir / "validation_diagnostics.csv", index=False)
    pd.DataFrame([
        {
            "data_split": "inner_oof_observed_support",
            "standardized_row_id": "r1",
            "source_row_index": "source-1",
            "study_no": "study-1",
            "PFAS_name": "PFAS-1",
            "adsorbent_id": "adsorbent-1",
            "residual": prediction - 1.0,
            "absolute_error": abs(prediction - 1.0),
            "source_row_weight": 1.0,
        },
        {
            "data_split": "testing",
            "standardized_row_id": "r2",
            "source_row_index": "source-2",
            "study_no": "study-2",
            "PFAS_name": "PFAS-2",
            "adsorbent_id": "adsorbent-2",
            "residual": prediction - 1.0,
            "absolute_error": abs(prediction - 1.0),
            "source_row_weight": 1.0,
        },
    ]).to_csv(run_dir / "observed_support_by_row.csv", index=False)
    pd.DataFrame([{
        "candidate": "pfas_morgan_novelty",
        "block": "pfas",
        "construct": "structure",
        "analysis_role": "row_level_predictor",
        "inner_coverage_fraction": 1.0,
        "outer_coverage_fraction": 1.0,
        "outer_distinct_count": 10,
    }]).to_csv(run_dir / "residual_candidate_screening.csv", index=False)
    pd.DataFrame([{
        "outcome": "signed_bias",
        "target_definition": "y_pred - y_true",
        "model_family": "ridge_linear",
        "model_interpretation": "additive linear benchmark after candidate screening",
        "candidate_count": 1,
        "prediction_mae": abs(prediction - 1.0),
        "mae_improvement_vs_constant_mean": 0.01,
        "out_of_sample_r2_vs_inner_mean_constant": 0.02,
        "predicted_vs_observed_weighted_spearman": 0.1,
        "mean_unexplained_target_after_model": 0.01,
    }]).to_csv(run_dir / "residual_model_evaluation.csv", index=False)
    pd.DataFrame([{
        "candidate": "pfas_morgan_novelty",
        "block": "pfas",
        "construct": "structure",
        "weighted_spearman_with_signed_residual": 0.1,
        "weighted_spearman_with_absolute_error": 0.2,
    }]).to_csv(run_dir / "residual_candidate_associations.csv", index=False)
    (run_dir / "run_config.json").write_text(
        json.dumps({
            "regressor": regressor,
            "best_params": {"alpha": 1.0},
            "features": {"selected": ["feature_a"]},
        }),
        encoding="utf-8",
    )


class ExploratoryExportTests(unittest.TestCase):
    def test_fitted_model_runs_are_retained_and_indexed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            batch_dir = Path(temporary_directory) / "experiment"
            run_dir = model_run_directory(batch_dir, "study", outer_repeat_id=1)
            write_run_artifacts(run_dir, regressor="ridge", prediction=1.1)
            (run_dir / "model.joblib").write_bytes(b"fitted-pipeline")
            (run_dir / "split_assignments.csv").write_text(
                "row_position,split\n0,testing\n", encoding="utf-8"
            )
            runs = pd.DataFrame([{
                "status": "completed",
                "run_directory": str(run_dir),
                "run_id": "run_1",
                "split_strategy": "study",
            }])

            manifest = write_exploratory_outputs(
                batch_dir,
                runs,
                run_context_columns=("split_strategy",),
                failures=("post-processing failed",),
            )

            self.assertTrue((run_dir / "model.joblib").exists())
            self.assertTrue((batch_dir / "model_runs" / "README.md").exists())
            self.assertEqual(manifest["audit"]["model_runs"], "audit/model_runs.csv")
            index = pd.read_csv(batch_dir / "audit" / "model_runs.csv")
            self.assertEqual(
                index.loc[0, "model_run_path"], "model_runs/study/outer_repeat_001"
            )
            self.assertEqual(index.loc[0, "split_strategy"], "study")
            self.assertTrue(bool(index.loc[0, "has_saved_model"]))
            self.assertTrue(bool(index.loc[0, "has_split_assignments"]))

    def test_retained_model_runs_cannot_be_deleted_through_cleanup(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            batch_dir = Path(temporary_directory) / "experiment"
            run_dir = model_run_directory(batch_dir, "study", outer_repeat_id=1)
            write_run_artifacts(run_dir, regressor="ridge", prediction=1.1)
            runs = pd.DataFrame([{
                "status": "completed",
                "run_directory": str(run_dir),
                "run_id": "run_1",
                "split_strategy": "study",
            }])

            with self.assertRaises(ValueError):
                write_exploratory_outputs(
                    batch_dir,
                    runs,
                    run_context_columns=("split_strategy",),
                    extra_cleanup_paths=(model_runs_root(batch_dir),),
                )
            self.assertTrue(run_dir.exists())

    def test_discarded_candidate_runs_leave_the_retained_tree(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            batch_dir = Path(temporary_directory) / "experiment"
            run_dir = model_run_directory(batch_dir, "study", outer_repeat_id=2)
            write_run_artifacts(run_dir, regressor="ridge", prediction=1.1)
            rows = [{"run_directory": str(run_dir), "status": "training_failed"}]

            quarantine_incomplete_model_runs(
                batch_dir, rows, candidate_id=7, seed=1234
            )

            self.assertFalse(run_dir.exists())
            relocated = Path(rows[0]["run_directory"])
            self.assertEqual(
                relocated.relative_to(batch_dir).as_posix(),
                "debug/incomplete_model_runs/candidate_007_seed_1234/study/"
                "outer_repeat_002",
            )
            self.assertTrue((relocated / "run_config.json").exists())

    def test_shared_backend_preserves_experiment_dimensions(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            batch_dir = Path(temporary_directory) / "experiment"
            work_root = model_runs_root(batch_dir)
            assignment_path = batch_dir / "assignment.csv"
            batch_dir.mkdir(parents=True)
            summary_dir = batch_dir / "summary"
            summary_dir.mkdir()
            (summary_dir / "residual_model_evaluation_summary.csv").write_text(
                "legacy residual output\n",
                encoding="utf-8",
            )
            pd.DataFrame({
                "row_id": ["r1", "r2"],
                "split": ["training", "testing"],
            }).to_csv(assignment_path, index=False)

            rows = []
            for index, (algorithm, threshold, prediction) in enumerate(
                [
                    ("ridge", 0.10, 1.1),
                    ("random_forest", 0.20, 0.9),
                ],
                start=1,
            ):
                run_dir = work_root / f"run_{index}"
                write_run_artifacts(
                    run_dir,
                    regressor=algorithm,
                    prediction=prediction,
                )
                rows.append({
                    "status": "completed",
                    "run_directory": str(run_dir),
                    "run_id": f"run_{index}",
                    "split_assignment_id": "frozen_split_1",
                    "split_assignment_path": str(assignment_path),
                    "algorithm": algorithm,
                    "feature_selection_threshold": threshold,
                    "split_strategy": "study",
                    "mae": abs(prediction - 1.0),
                    "rmse": abs(prediction - 1.0),
                    "r2": 0.5,
                    "spearman": 0.4,
                    "low_logkd_mae": 0.2,
                    "low_logkd_rmse": 0.3,
                    "low_logkd_r2": 0.1,
                    "low_logkd_spearman": 0.2,
                    "high_logkd_mae": 0.4,
                    "high_logkd_rmse": 0.5,
                    "high_logkd_r2": 0.2,
                    "high_logkd_spearman": 0.3,
                    "database_contrast_experimental_conditions": 0.6,
                    "contrast_retention_experimental_conditions": 0.8,
                    "database_contrast_adsorbent_properties": 0.5,
                    "contrast_retention_adsorbent_properties": 0.7,
                    "database_contrast_pfas_characteristics": 0.4,
                    "contrast_retention_pfas_characteristics": 0.9,
                })

            manifest = write_exploratory_outputs(
                batch_dir,
                pd.DataFrame(rows),
                run_context_columns=(
                    "algorithm",
                    "feature_selection_threshold",
                    "split_strategy",
                ),
                assignment_context_columns=("split_strategy",),
                assignment_id_column="split_assignment_id",
                assignment_path_column="split_assignment_path",
                performance_group_columns=("algorithm",),
                coverage_group_columns=("split_strategy",),
            )

            summary = pd.read_csv(
                batch_dir / "audit" / "run_summary.csv"
            )
            self.assertEqual(len(summary), 2)
            self.assertIn("algorithm", summary.columns)
            self.assertIn("feature_selection_threshold", summary.columns)
            self.assertIn("selected_feature_count", summary.columns)
            self.assertIn("inner_min_reference_retention_score", summary.columns)
            self.assertIn("inner_mean_reference_retention_score", summary.columns)
            self.assertIn("inner_reference_retention_score_sd", summary.columns)
            self.assertTrue(
                (summary["inner_min_reference_retention_score"].round(12) == 0.60).all()
            )
            self.assertTrue(
                (summary["inner_mean_reference_retention_score"].round(12) == 0.70).all()
            )
            self.assertTrue(
                (summary["inner_reference_retention_score_sd"].round(12) == 0.10).all()
            )
            self.assertFalse(
                any("path" in column.casefold() for column in summary.columns)
            )
            performance = pd.read_csv(
                batch_dir / "summary" / "performance_summary.csv"
            )
            self.assertEqual(set(performance["algorithm"]), {"ridge", "random_forest"})
            self.assertIn("high_logkd_spearman_mean", performance.columns)
            self.assertFalse(any(column.endswith(("_median", "_min", "_max", "_sem")) for column in performance.columns))
            coverage_summary = pd.read_csv(
                batch_dir / "summary" / "coverage_summary.csv"
            )
            self.assertEqual(set(coverage_summary["feature_block"]), {
                "Experimental conditions", "Adsorbent properties", "PFAS characteristics",
            })
            self.assertNotIn("training_contrast", " ".join(coverage_summary.columns))

            selected_feature_support = pd.read_csv(
                batch_dir / "detail" / "selected_feature_support_detail.csv"
            )
            self.assertEqual(len(selected_feature_support), 2)
            self.assertEqual(
                set(selected_feature_support["algorithm"]), {"ridge", "random_forest"}
            )
            self.assertEqual(
                set(selected_feature_support["split_strategy"]), {"study"}
            )
            self.assertEqual(
                len(
                    pd.read_parquet(
                        batch_dir
                        / "detail"
                        / "testing_predictions.parquet"
                    )
                ),
                2,
            )
            self.assertFalse(
                (batch_dir / "detail" / "observed_support_by_row.parquet").exists()
            )
            self.assertNotIn("residual_analysis", manifest)
            self.assertFalse((batch_dir / "residual_analysis").exists())
            self.assertFalse(
                (batch_dir / "summary" / "residual_model_evaluation_summary.csv").exists()
            )
            self.assertFalse(
                any(
                    path.name.startswith("residual_")
                    or "residual_analysis" in path.name
                    for path in (batch_dir / "summary").iterdir()
                )
            )
            self.assertFalse((batch_dir / "summary" / "run_summary.csv").exists())
            self.assertFalse((batch_dir / "summary" / "scenario_summary.csv").exists())
            self.assertFalse(
                (batch_dir / "summary" / "aggregate_performance_and_coverage.csv").exists()
            )
            self.assertEqual(
                len(
                    pd.read_parquet(
                        batch_dir / "audit" / "split_assignments.parquet"
                    )
                ),
                2,
            )
            self.assertEqual(
                len(
                    pd.read_parquet(
                        batch_dir
                        / "audit"
                        / "run_configurations.parquet"
                    )
                ),
                2,
            )
            self.assertTrue(work_root.exists())
            self.assertFalse((batch_dir / "debug").exists())
            self.assertIn("run_configurations", manifest["audit"])


if __name__ == "__main__":
    unittest.main()
