from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.logkd_support import (  # noqa: E402
    KDE_CANDIDATES,
    kde_candidate_support,
    kde_score_column,
    pfas_structural_diagnostics,
)
from backend.logkd_residual_diagnostics import (
    attach_identity_novelty,
    evaluate_outer_residual_analysis,
    residual_group_heterogeneity,
    screen_residual_candidates,
    summarize_residual_model_runs,
    support_candidate_manifest,
)
from backend.logkd_features import feature_resolved_support_diagnostics, input_support_diagnostics
from backend.logkd_metrics import logkd_tail_metrics_dict
from train_logkd_model import selected_feature_support_diagnostics  # noqa: E402
class ObservedSupportTests(unittest.TestCase):
    def test_tail_diagnostics_include_all_four_performance_metrics(self) -> None:
        metrics = logkd_tail_metrics_dict(
            y_true=[-1.0, -0.5, 2.0, 2.5],
            y_pred=[-0.8, -0.4, 2.1, 2.2],
            low_threshold=0.0,
            high_threshold=1.0,
        )

        for tail in ("low_logkd", "high_logkd"):
            for metric in ("mae", "rmse", "r2", "spearman"):
                self.assertIn(f"{tail}_{metric}", metrics)

    def test_simple_support_is_reported_separately_by_block(self) -> None:
        training = pd.DataFrame({
            "rdkit_x": [1.0, 2.0], "ssa_(m2/g)_avg": [100.0, 200.0],
            "pH": [6.0, 8.0], "Water_type": ["DI", "GW"],
        })
        query = pd.DataFrame({
            "rdkit_x": [3.0], "ssa_(m2/g)_avg": [150.0],
            "pH": [np.nan], "Water_type": ["surface"],
        })
        _, rows = input_support_diagnostics(
            training, query,
            ["rdkit_x", "ssa_(m2/g)_avg", "pH", "Water_type"],
            ["rdkit_x", "ssa_(m2/g)_avg", "pH"], ["Water_type"],
        )

        self.assertEqual(rows.loc[0, "pfas_numeric_values_outside_training_range"], 1)
        self.assertEqual(rows.loc[0, "adsorbent_numeric_values_outside_training_range"], 0)
        self.assertEqual(rows.loc[0, "experimental_conditions_values_missing"], 1)
        self.assertEqual(rows.loc[0, "experimental_conditions_categorical_values_unseen_in_training"], 1)

    def test_selected_feature_support_consolidates_retention_and_test_support(self) -> None:
        manifest = pd.DataFrame(
            {
                "feature": ["kept", "dropped"],
                "bucket": ["PFAS characteristics", "PFAS characteristics"],
                "kind": ["numeric", "numeric"],
                "selected": [True, False],
            }
        )
        retention = pd.DataFrame(
            {
                "feature": ["kept", "dropped"],
                "bucket": ["PFAS characteristics", "PFAS characteristics"],
                "kind": ["numeric", "numeric"],
                "contrast_retention": [0.8, 0.6],
            }
        )
        testing = pd.DataFrame(
            {
                "diagnostic_type": ["input_support"],
                "feature": ["kept"],
                "bucket": ["PFAS characteristics"],
                "kind": ["numeric"],
                "testing_outside_range_values": [2],
            }
        )

        result = selected_feature_support_diagnostics(manifest, retention, testing)

        self.assertEqual(result["feature"].tolist(), ["kept"])
        self.assertEqual(result.loc[0, "contrast_retention"], 0.8)
        self.assertEqual(result.loc[0, "testing_outside_range_values"], 2)
        self.assertNotIn("diagnostic_type", result.columns)

    def test_feature_resolved_support_keeps_feature_identity_and_extrapolation_direction(self) -> None:
        training = pd.DataFrame(
            {
                "pH": [5.0, 6.0, 7.0, 8.0],
                "Water_type": ["DI", "DI", "GW", "GW"],
            }
        )
        query = pd.DataFrame(
            {
                "pH": [4.0, 9.0, np.nan],
                "Water_type": ["DI", "surface", np.nan],
            }
        )
        manifest, matrix, events = feature_resolved_support_diagnostics(
            training,
            query,
            ["pH", "Water_type"],
            ["pH"],
            ["Water_type"],
        )

        pH_below = "feature_support::pH::below_training_distance"
        pH_above = "feature_support::pH::above_training_distance"
        water_unseen = "feature_support::Water_type::unseen_level"
        water_rarity = "feature_support::Water_type::level_rarity"
        self.assertGreater(matrix.loc[0, pH_below], 0.0)
        self.assertEqual(matrix.loc[0, pH_above], 0.0)
        self.assertEqual(matrix.loc[1, pH_below], 0.0)
        self.assertGreater(matrix.loc[1, pH_above], 0.0)
        self.assertEqual(matrix.loc[1, water_unseen], 1.0)
        self.assertGreater(matrix.loc[1, water_rarity], matrix.loc[0, water_rarity])
        self.assertEqual(matrix.loc[2, "feature_support::pH::missing"], 1.0)
        self.assertEqual(matrix.loc[2, "feature_support::Water_type::missing"], 1.0)
        self.assertEqual(len(events), 6)
        self.assertIn(pH_below, set(manifest["candidate"]))

    def test_scores_report_training_only_support(self) -> None:
        training = pd.DataFrame(
            {
                "pfas_numeric": np.linspace(0.0, 23.0, 24),
                "context_numeric": np.linspace(10.0, 33.0, 24),
                "context_category": ["A", "B"] * 12,
            }
        )
        query = pd.DataFrame(
            {
                "pfas_numeric": [2.0, 100.0, np.nan],
                "context_numeric": [12.0, 200.0, np.nan],
                "context_category": ["A", "C", np.nan],
            }
        )
        manifest = pd.DataFrame(
            {
                "feature": ["pfas_numeric", "context_numeric", "context_category"],
                "bucket": ["PFAS characteristics", "Experimental conditions", "Adsorbent properties"],
            }
        )

        result = kde_candidate_support(
            training,
            query,
            ["pfas_numeric", "context_numeric", "context_category"],
            ["pfas_numeric", "context_numeric"],
            ["context_category"],
            manifest,
        )

        for spec in KDE_CANDIDATES:
            candidate = spec["candidate"]
            self.assertTrue(result[f"support_{candidate}_kde_status"].eq("available").all())
            self.assertTrue(result[kde_score_column(candidate)].between(0.0, 1.0).all())
        self.assertEqual(result.loc[0, "support_full_input_pca5_kde_pca_components_requested"], 5)
        self.assertEqual(result.loc[0, "support_full_input_pca5_kde_pre_pca_input_dimensions"], 4)
        self.assertEqual(result.loc[0, "support_full_input_pca5_kde_pca_components_used"], 4)
        self.assertAlmostEqual(
            result.loc[0, "support_full_input_pca5_kde_pca_cumulative_explained_variance_ratio"],
            1.0,
        )
        first_query_only = kde_candidate_support(
            training,
            query.iloc[[0]].reset_index(drop=True),
            ["pfas_numeric", "context_numeric", "context_category"],
            ["pfas_numeric", "context_numeric"],
            ["context_category"],
            manifest,
        )
        self.assertAlmostEqual(
            result.loc[0, kde_score_column("full_input_pca5")],
            first_query_only.loc[0, kde_score_column("full_input_pca5")],
        )
        self.assertEqual(result.loc[1, "support_experimental_conditions_kde_numeric_features_outside_training_range_count"], 1)
        self.assertEqual(result.loc[1, "support_adsorbent_kde_unseen_categorical_feature_count"], 1)
        self.assertEqual(result.loc[2, "support_full_input_kde_missing_feature_count"], 3)

    def test_residual_redesign_separates_signed_bias_and_error_risk(self) -> None:
        candidate = "pfas_morgan_novelty"
        inner_score = np.linspace(0.0, 1.0, 60)
        outer_score = np.linspace(0.02, 0.98, 30)
        inner_residual = 0.8 * inner_score - 0.3
        outer_residual = 0.8 * outer_score - 0.3
        inner = pd.DataFrame({
            candidate: inner_score,
            "residual": inner_residual,
            "absolute_error": np.abs(inner_residual),
            "source_row_weight": np.ones(60),
            "study_no": [f"study-{index % 6}" for index in range(60)],
        })
        outer = pd.DataFrame({
            candidate: outer_score,
            "residual": outer_residual,
            "absolute_error": np.abs(outer_residual),
            "source_row_weight": np.ones(30),
            "study_no": [f"outer-{index % 3}" for index in range(30)],
        })

        screening, evaluation, associations = evaluate_outer_residual_analysis(inner, outer)

        self.assertEqual(
            screening.loc[screening["candidate"].eq(candidate), "analysis_role"].item(),
            "row_level_predictor",
        )
        self.assertEqual(set(evaluation["outcome"]), {"signed_bias", "absolute_error_risk"})
        self.assertEqual(
            set(evaluation["model_family"]),
            {"constant_mean", "ridge_linear", "extra_trees_nonlinear"},
        )
        self.assertGreater(
            evaluation.loc[
                evaluation["outcome"].eq("signed_bias")
                & evaluation["model_family"].eq("ridge_linear"),
                "out_of_sample_r2_vs_inner_mean_constant",
            ].item(),
            0,
        )
        self.assertGreater(
            associations.loc[associations["candidate"].eq(candidate), "weighted_spearman_with_signed_residual"].item(),
            0,
        )
        self.assertNotIn("domain_label", " ".join(support_candidate_manifest()["candidate"]))

    def test_residual_model_accepts_feature_resolved_manifest_and_sparse_challenger(self) -> None:
        candidate = "feature_support::pH::above_training_distance"
        inner_score = np.linspace(0.0, 1.0, 60)
        outer_score = np.linspace(0.02, 0.98, 30)
        manifest = pd.DataFrame(
            [{
                "candidate": candidate,
                "feature": "pH",
                "block": "Experimental conditions",
                "kind": "numeric",
                "construct": "feature_high_extrapolation",
                "definition": "Log-scaled distance above training range for pH",
            }]
        )
        inner = pd.DataFrame(
            {
                candidate: inner_score,
                "residual": 0.7 * inner_score - 0.2,
                "absolute_error": np.abs(0.7 * inner_score - 0.2),
                "source_row_weight": np.ones(60),
                "study_no": [f"study-{index % 6}" for index in range(60)],
            }
        )
        outer = pd.DataFrame(
            {
                candidate: outer_score,
                "residual": 0.7 * outer_score - 0.2,
                "absolute_error": np.abs(0.7 * outer_score - 0.2),
                "source_row_weight": np.ones(30),
                "study_no": [f"outer-{index % 3}" for index in range(30)],
            }
        )

        screening, evaluation, _ = evaluate_outer_residual_analysis(
            inner,
            outer,
            candidate_manifest=manifest,
            include_elasticnet=True,
        )

        self.assertEqual(screening.loc[0, "candidate"], candidate)
        self.assertIn("elasticnet_sparse_linear", set(evaluation["model_family"]))
        sparse = evaluation.loc[evaluation["model_family"].eq("elasticnet_sparse_linear")]
        self.assertTrue(sparse["elasticnet_selected_candidate_columns"].str.contains(candidate, regex=False).all())

    def test_constant_outer_candidate_becomes_scenario_descriptor(self) -> None:
        inner = pd.DataFrame({
            "pfas_identity_novelty": [0.0, 1.0] * 10,
            "pfas_missing_fraction": np.zeros(20),
        })
        outer = pd.DataFrame({
            "pfas_identity_novelty": np.ones(12),
            "pfas_missing_fraction": np.zeros(12),
        })

        screening = screen_residual_candidates(inner, outer).set_index("candidate")

        self.assertEqual(
            screening.loc["pfas_identity_novelty", "analysis_role"],
            "scenario_descriptor",
        )
        self.assertFalse(screening.loc["pfas_identity_novelty", "eligible_for_row_model"])
        self.assertEqual(
            screening.loc["pfas_missing_fraction", "analysis_role"],
            "scenario_descriptor",
        )

    def test_model_summary_reports_evidence_without_selection_decisions(self) -> None:
        by_run = pd.DataFrame({
            "split_strategy": ["study", "study"],
            "outcome": ["signed_bias", "signed_bias"],
            "target_definition": ["y_pred - y_true"] * 2,
            "model_family": ["ridge_linear"] * 2,
            "model_interpretation": ["additive linear benchmark"] * 2,
            "candidate_count": [3, 4],
            "prediction_mae": [0.4, 0.5],
            "mae_improvement_vs_constant_mean": [0.1, -0.1],
            "out_of_sample_r2_vs_inner_mean_constant": [0.2, -0.2],
            "predicted_vs_observed_weighted_spearman": [0.3, -0.1],
            "mean_unexplained_target_after_model": [0.02, -0.03],
        })

        summary = summarize_residual_model_runs(by_run)

        self.assertEqual(len(summary), 1)
        self.assertEqual(summary.loc[0, "fraction_runs_positive_r2"], 0.5)
        self.assertNotIn("decision", summary.columns)

    def test_model_summary_keeps_outer_allocation_methods_separate(self) -> None:
        by_run = pd.DataFrame({
            "split_strategy": ["study", "study"],
            "outer_allocation_method": ["random_group", "size_matched_random_group"],
            "outcome": ["signed_bias", "signed_bias"],
            "target_definition": ["y_pred - y_true"] * 2,
            "model_family": ["ridge_linear"] * 2,
            "model_interpretation": ["additive linear benchmark"] * 2,
            "candidate_count": [3, 3],
            "prediction_mae": [0.4, 0.5],
            "mae_improvement_vs_constant_mean": [0.1, -0.1],
            "out_of_sample_r2_vs_inner_mean_constant": [0.2, -0.2],
            "predicted_vs_observed_weighted_spearman": [0.3, -0.1],
            "mean_unexplained_target_after_model": [0.02, -0.03],
        })

        summary = summarize_residual_model_runs(by_run)

        self.assertEqual(len(summary), 2)
        self.assertEqual(
            set(summary["outer_allocation_method"]),
            {"random_group", "size_matched_random_group"},
        )

    def test_group_heterogeneity_is_explicitly_retrospective(self) -> None:
        rows = pd.DataFrame({
            "data_split": ["testing"] * 12,
            "split_strategy": ["study"] * 12,
            "run_id": ["run-1"] * 12,
            "study_no": ["A"] * 6 + ["B"] * 6,
            "PFAS_name": ["P1", "P2"] * 6,
            "adsorbent_id": ["X", "Y"] * 6,
            "source_row_index": list(range(12)),
            "source_row_weight": np.ones(12),
            "residual": [0.8] * 6 + [-0.8] * 6,
        })

        result = residual_group_heterogeneity(rows)

        study = result.loc[result["group_level"].eq("study")].iloc[0]
        self.assertAlmostEqual(study["descriptive_between_group_residual_variance_fraction"], 1.0)
        self.assertAlmostEqual(study["random_intercept_icc"], 1.0)
        self.assertIn("retrospective", study["interpretation_scope"])

class PfasStructuralSupportTests(unittest.TestCase):
    def test_morgan_similarity_is_calculated_from_smiles(self) -> None:
        pfas_features = pd.DataFrame(
            {
                "Abbreviation": ["PFBA", "PFOA", "PFHpA"],
                "SMILES": [
                    "C(=O)(C(C(C(F)(F)F)(F)F)(F)F)O",
                    "C(=O)(C(C(C(C(C(C(C(F)(F)F)(F)F)(F)F)(F)F)(F)F)(F)F)(F)F)O",
                    "C(=O)(C(C(C(C(C(C(F)(F)F)(F)F)(F)F)(F)F)(F)F)(F)F)O",
                ],
            }
        )
        training = pd.DataFrame({"PFAS_name": ["PFBA", "PFOA"]})
        testing = pd.DataFrame({"PFAS_name": ["PFHpA"]})

        diagnostics = pfas_structural_diagnostics(training, testing, pfas_features)

        self.assertEqual(diagnostics.loc[0, "morgan_similarity_status"], "available")
        self.assertTrue(0.0 <= diagnostics.loc[0, "max_morgan_similarity_to_training"] <= 1.0)
        self.assertEqual(diagnostics.loc[0, "nearest_morgan_pfas"], "PFOA")
        self.assertLess(diagnostics.loc[0, "max_morgan_similarity_to_training"], 1.0)


    def test_morgan_status_explains_missing_smiles(self) -> None:
        pfas_features = pd.DataFrame(
            {
                "Abbreviation": ["P1", "P2"],
                "SMILES": ["CCO", ""],
            }
        )

        diagnostics = pfas_structural_diagnostics(
            pd.DataFrame({"PFAS_name": ["P1"]}),
            pd.DataFrame({"PFAS_name": ["P2"]}),
            pfas_features,
        )

        self.assertEqual(diagnostics.loc[0, "morgan_similarity_status"], "missing_or_invalid_smiles")
        self.assertTrue(pd.isna(diagnostics.loc[0, "max_morgan_similarity_to_training"]))


if __name__ == "__main__":
    unittest.main()
