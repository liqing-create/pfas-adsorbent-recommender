from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.logkd_coverage import (  # noqa: E402
    availability_unit_keys,
    evidence_coverage_by_feature,
    reference_retention_by_feature,
    summarize_reference_retention,
)
from backend.logkd_features import adsorbent_unit_keys  # noqa: E402
from backend.logkd_splits import (  # noqa: E402
    TESTING,
    TRAINING,
    _objective_improves,
    build_testing_split,
    make_validation_assignments,
    validation_diagnostics,
    validate_testing_split,
)


def toy_rows() -> pd.DataFrame:
    return pd.DataFrame({
        "standardized_row_id": ["r1", "r2", "r3", "r4", "r5"],
        "source_row_index": [1, 2, 3, 4, 5],
        "study_no": ["s1", "s1", "s2", "s3", "s4"],
        "PFAS_name": ["PFOA", "PFOA", "PFBS", "GenX", "GenX"],
        "adsorbent_id": ["a1", "a1", "a2", "a3", "a3"],
        "adsorbent_identity_key": ["a1", "a1", "a2", "a3", "a3"],
        "pH": [3.0, 4.0, 3.0, 5.0, np.nan],
        "Water_type": ["fresh", "saline", "saline", "fresh", ""],
        "rdkit_tpsa": [17.0, 17.0, 18.0, 35.0, 35.0],
        "Kd_final_log10(L/g)": [1.0, 1.1, 1.2, 1.3, 1.4],
    })


class EvidenceCoverageTests(unittest.TestCase):
    def test_pH_availability_and_within_study_metrics_match_toy_example(self) -> None:
        df = toy_rows()
        manifest = pd.DataFrame({
            "feature": ["pH"],
            "bucket": ["Experimental conditions"],
        })
        result = evidence_coverage_by_feature(
            df, ["pH"], ["pH"], [], manifest, scope="toy"
        ).iloc[0]

        self.assertEqual(result["availability_unit"], "study")
        self.assertEqual(int(result["observed_rows"]), 4)
        self.assertAlmostEqual(float(result["row_availability"]), 4 / 5)
        self.assertEqual(int(result["total_availability_units"]), 4)
        self.assertEqual(int(result["reporting_availability_units"]), 3)
        self.assertAlmostEqual(float(result["unit_availability"]), 3 / 4)
        self.assertEqual(int(result["varying_reporting_studies"]), 1)
        self.assertAlmostEqual(float(result["studies_with_multiple_values"]), 1 / 3)
        self.assertAlmostEqual(float(result["share_of_variation_within_studies"]), 0.2284, places=3)
        self.assertAlmostEqual(float(result["between_study_variation_share"]), 0.7716, places=3)

    def test_pfas_features_use_unique_pfas_as_availability_units(self) -> None:
        df = toy_rows()
        manifest = pd.DataFrame({
            "feature": ["rdkit_tpsa"],
            "bucket": ["PFAS characteristics"],
        })
        result = evidence_coverage_by_feature(
            df, ["rdkit_tpsa"], ["rdkit_tpsa"], [], manifest
        ).iloc[0]

        self.assertEqual(result["availability_unit"], "unique PFAS")
        self.assertEqual(int(result["total_availability_units"]), 3)
        self.assertAlmostEqual(float(result["unit_availability"]), 1.0)

class AdsorbentUnitDefinitionTests(unittest.TestCase):
    """One study reporting a material under two labels is still one unit."""

    @staticmethod
    def two_label_rows() -> pd.DataFrame:
        return pd.DataFrame({
            "study_no": ["s1", "s1", "s2", "s3"],
            # The same resin, spelled differently inside a single study and
            # again across studies.
            "adsorbent_id": ["AER", "PFA694E", "PFA694E", "A520E"],
            "adsorbent_identity_key": [
                "specific::PUROFINEPFA694E",
                "specific::PUROFINEPFA694E",
                "specific::PUROFINEPFA694E",
                "specific::PUROLITEA520E",
            ],
            "ssa_(m2/g)_avg": [500.0, 500.0, 480.0, 900.0],
        })

    def test_resolved_identity_collapses_two_labels_within_one_study(self) -> None:
        keys = adsorbent_unit_keys(self.two_label_rows())

        self.assertEqual(keys.nunique(), 3)
        self.assertEqual(keys.iloc[0], keys.iloc[1])

    def test_raw_adsorbent_id_is_only_a_fallback(self) -> None:
        df = self.two_label_rows().drop(columns=["adsorbent_identity_key"])

        self.assertEqual(adsorbent_unit_keys(df).nunique(), 4)

    def test_blank_study_or_adsorbent_yields_no_unit(self) -> None:
        df = pd.DataFrame({
            "study_no": ["s1", "", "s2"],
            "adsorbent_identity_key": ["", "k1", "k2"],
        })

        self.assertEqual(list(adsorbent_unit_keys(df)), ["", "", "s2|k2"])

    def test_coverage_and_support_share_one_unit_definition(self) -> None:
        df = self.two_label_rows()
        label, keys = availability_unit_keys(df, "Adsorbent properties")

        self.assertEqual(label, "study-specific adsorbent")
        pd.testing.assert_series_equal(keys, adsorbent_unit_keys(df))


class ReferenceRetentionTests(unittest.TestCase):
    @staticmethod
    def reference_rows() -> pd.DataFrame:
        return pd.DataFrame({
            "study_no": ["A", "A", "B", "B"],
            "PFAS_name": ["P1", "P2", "P3", "P4"],
            "adsorbent_identity_key": ["AC1", "AC2", "AC3", "AC4"],
            "pH": [3.0, 5.0, 7.0, 7.0],
            "Specific surface area": [100.0, 200.0, 300.0, 400.0],
            "rdkit_tpsa": [10.0, 20.0, 30.0, 40.0],
        })

    @staticmethod
    def manifest() -> pd.DataFrame:
        return pd.DataFrame({
            "feature": ["pH", "Specific surface area", "rdkit_tpsa"],
            "bucket": [
                "Experimental conditions",
                "Adsorbent properties",
                "PFAS characteristics",
            ],
            "kind": ["numeric", "numeric", "numeric"],
            "selected": [True, False, False],
        })

    def test_retention_uses_the_complete_reference_as_denominator(self) -> None:
        reference = self.reference_rows()
        training = reference.iloc[:2].copy()
        detail = reference_retention_by_feature(
            reference,
            training,
            self.manifest(),
            numeric_bins=10,
        ).set_index("feature")

        for feature in detail.index:
            self.assertAlmostEqual(float(detail.loc[feature, "entity_retention"]), 0.5)
            self.assertAlmostEqual(
                float(detail.loc[feature, "reference_value_support_mass_retention"]),
                0.5,
            )
        self.assertAlmostEqual(
            float(detail.loc["pH", "contrast_retention"]),
            1.0,
        )
        self.assertAlmostEqual(
            float(detail.loc["Specific surface area", "contrast_retention"]),
            0.5,
        )
        self.assertAlmostEqual(
            float(detail.loc["rdkit_tpsa", "contrast_retention"]),
            0.5,
        )
        self.assertEqual(int(detail.loc["pH", "reference_total_studies"]), 2)
        self.assertAlmostEqual(float(detail.loc["pH", "database_contrast"]), 0.5)
        self.assertAlmostEqual(float(detail.loc["pH", "training_contrast"]), 0.5)
        self.assertAlmostEqual(float(detail.loc["pH", "contrast_retention"]), 1.0)

    def test_summary_separates_database_training_and_retained_contrast(self) -> None:
        reference = self.reference_rows()
        detail = reference_retention_by_feature(
            reference,
            reference.iloc[:2].copy(),
            self.manifest(),
            numeric_bins=10,
        )
        summary = summarize_reference_retention(detail)

        self.assertEqual(len(summary), 9)
        self.assertAlmostEqual(
            summary["database_contrast_experimental_conditions"],
            0.5,
        )
        self.assertAlmostEqual(
            summary["training_contrast_adsorbent_properties"],
            0.5,
        )
        self.assertAlmostEqual(
            summary["contrast_retention_experimental_conditions"],
            1.0,
        )
        self.assertFalse(any("evidence_coverage" in key for key in summary))

    def test_all_data_refit_keeps_database_contrast_and_has_complete_retention(self) -> None:
        reference = self.reference_rows()
        detail = reference_retention_by_feature(
            reference,
            reference,
            self.manifest(),
            numeric_bins=10,
        )
        summary = summarize_reference_retention(detail)

        self.assertAlmostEqual(summary["database_contrast_experimental_conditions"], 0.5)
        self.assertAlmostEqual(summary["training_contrast_experimental_conditions"], 0.5)
        self.assertTrue(
            all(
                np.isclose(value, 1.0)
                for key, value in summary.items()
                if key.startswith("contrast_retention_")
            )
        )


class RandomSplitTests(unittest.TestCase):
    def test_random_group_split_is_scenario_valid_and_not_coverage_optimized(self) -> None:
        df = pd.concat([toy_rows(), toy_rows().assign(
            standardized_row_id=lambda frame: "x" + frame["standardized_row_id"],
            source_row_index=lambda frame: frame["source_row_index"] + 10,
            study_no=lambda frame: "x" + frame["study_no"],
        )], ignore_index=True)
        result = build_testing_split(
            df, "study", 0.25, 123, "Kd_final_log10(L/g)", "random_group"
        )

        self.assertEqual(result.validation["evaluation_protocol"], "seeded_random_group_split")
        self.assertFalse(result.validation["coverage_objective_used_for_allocation"])
        self.assertTrue(validate_testing_split(result.assignments)["valid_split_labels"])
        self.assertTrue(result.assignments.groupby("source_row_index")["split"].nunique().eq(1).all())

    def test_combination_split_keeps_test_components_in_training(self) -> None:
        df = pd.DataFrame({
            "standardized_row_id": ["p1-a1", "p1-a2", "p2-a1", "p2-a2"],
            "source_row_index": [1, 2, 3, 4],
            "study_no": ["s1", "s2", "s3", "s4"],
            "PFAS_name": ["p1", "p1", "p2", "p2"],
            "adsorbent_id": ["a1", "a2", "a1", "a2"],
            "adsorbent_identity_key": ["a1", "a2", "a1", "a2"],
            "Kd_final_log10(L/g)": [1.0, 1.1, 1.2, 1.3],
        })
        result = build_testing_split(
            df, "combination", 0.5, 42, "Kd_final_log10(L/g)", "random_group"
        )
        training = result.assignments.loc[result.assignments["split"].eq(TRAINING)]
        testing = result.assignments.loc[result.assignments["split"].eq(TESTING)]

        self.assertTrue(set(testing["pfas_key"]).issubset(set(training["pfas_key"])))
        self.assertTrue(
            set(testing["adsorbent_identity_key"]).issubset(
                set(training["adsorbent_identity_key"])
            )
        )

    def test_size_matched_random_group_targets_nearest_feasible_row_count(self) -> None:
        rows: list[dict[str, object]] = []
        row_id = 0
        # A 20% test fraction requests 2.8 rows. The available whole-study
        # sizes are 5, 4, 3, and 2, so three rows is the nearest feasible
        # nonempty grouped holdout.
        for study_index, size in enumerate((5, 4, 3, 2), start=1):
            for _ in range(size):
                row_id += 1
                rows.append({
                    "standardized_row_id": f"matched-{row_id}",
                    "source_row_index": row_id,
                    "study_no": f"study-{study_index}",
                    "PFAS_name": f"pfas-{study_index}",
                    "adsorbent_id": f"adsorbent-{study_index}",
                    "adsorbent_identity_key": f"adsorbent-{study_index}",
                    "Kd_final_log10(L/g)": float(study_index),
                })
        df = pd.DataFrame(rows)

        first = build_testing_split(
            df,
            "study",
            0.20,
            123,
            "Kd_final_log10(L/g)",
            "size_matched_random_group",
        )
        second = build_testing_split(
            df,
            "study",
            0.20,
            123,
            "Kd_final_log10(L/g)",
            "size_matched_random_group",
        )

        self.assertTrue(first.assignments.equals(second.assignments))
        self.assertEqual(
            first.validation["evaluation_protocol"],
            "seeded_size_matched_random_group_split",
        )
        self.assertEqual(
            first.validation["testing_size_policy"],
            "nearest_feasible_fraction_of_rows",
        )
        self.assertEqual(first.validation["testing_rows"], 3)
        self.assertAlmostEqual(
            first.validation["test_row_target_requested_rows"], 2.8,
        )
        self.assertAlmostEqual(
            first.validation["test_row_target_absolute_deviation_rows"],
            0.2,
        )
        self.assertTrue(
            first.assignments.groupby("source_row_index")["split"].nunique().eq(1).all()
        )

    def test_validation_folds_remain_group_coherent(self) -> None:
        df = pd.concat([toy_rows(), toy_rows().assign(
            standardized_row_id=lambda frame: "x" + frame["standardized_row_id"],
            source_row_index=lambda frame: frame["source_row_index"] + 10,
            study_no=lambda frame: "x" + frame["study_no"],
        )], ignore_index=True)
        assignments = make_validation_assignments(df, "study", 3, 456)

        self.assertGreaterEqual(assignments["validation_fold"].ge(0).sum(), 1)
        self.assertTrue(assignments.groupby("split_unit")["validation_fold"].nunique().eq(1).all())


class EvidenceBalancedInnerAllocationTests(unittest.TestCase):
    @staticmethod
    def allocation_rows() -> pd.DataFrame:
        rows: list[dict[str, object]] = []
        for study_index in range(6):
            for row_index in range(3):
                rows.append({
                    "standardized_row_id": f"r{study_index}_{row_index}",
                    "source_row_index": study_index * 10 + row_index,
                    "study_no": f"s{study_index}",
                    "PFAS_name": f"p{row_index % 3}",
                    "adsorbent_id": f"a{row_index % 2}",
                    "adsorbent_identity_key": f"a{row_index % 2}",
                    "pH": float(study_index + row_index % 2),
                    "Water_type": ("fresh", "saline")[row_index % 2],
                    "rdkit_tpsa": float(10 + row_index),
                    "target": float(study_index),
                })
        return pd.DataFrame(rows)

    @staticmethod
    def manifest() -> pd.DataFrame:
        return pd.DataFrame({
            "feature": ["pH", "Water_type", "rdkit_tpsa"],
            "bucket": [
                "Experimental conditions",
                "Experimental conditions",
                "PFAS characteristics",
            ],
            "kind": ["numeric", "categorical", "numeric"],
        })

    def test_evidence_balanced_is_deterministic_and_reports_paired_diagnostics(self) -> None:
        df = self.allocation_rows()
        first = make_validation_assignments(
            df, "study", 3, 456,
            allocation_method="evidence_balanced", reference_manifest=self.manifest(),
        )
        second = make_validation_assignments(
            df.assign(target=-999.0), "study", 3, 456,
            allocation_method="evidence_balanced", reference_manifest=self.manifest(),
        )

        self.assertTrue(first.equals(second))
        metadata = first.attrs["reference_retention_allocation"]
        self.assertEqual(metadata["optimization_status"], "improved")
        self.assertLess(
            metadata["optimized_objective_loss"]["mean_fold_loss"],
            metadata["baseline_objective_loss"]["mean_fold_loss"],
        )
        diagnostics = validation_diagnostics(df, first, "target")
        self.assertIn("validation_reference_retention_score", diagnostics.columns)
        self.assertIn("validation_reference_contrast_retention_experimental_conditions", diagnostics.columns)

    def test_evidence_balanced_objective_prioritizes_mean_before_weakest_fold(self) -> None:
        # The candidate loses on its weakest fold (maximum loss 0.40 vs 0.30),
        # but its mean loss is better (0.20 vs 0.25).  Mean retention is now
        # the first lexicographic priority, so this is an improvement.
        self.assertTrue(_objective_improves((0.20, 0.40, 0.05), (0.25, 0.30, 0.05)))


if __name__ == "__main__":
    unittest.main()
