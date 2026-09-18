from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent
if not (ROOT / "logkd_residual_review.py").exists():
    ROOT = ROOT.parent
sys.path.insert(0, str(ROOT))

from logkd_residual_review import (  # noqa: E402
    build_residual_group_audit,
    write_residual_group_audit,
)


def _prediction_rows() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for study_no, source_start, residual in (
        ("study_suspicious", 10, 3.0),
        ("study_baseline", 20, 0.2),
    ):
        for source_offset in range(3):
            for outer_assignment in ("outer_1", "outer_2"):
                rows.append({
                    "data_split": "testing",
                    "split_strategy": "random_row",
                    "outer_allocation_method": "random_row",
                    "outer_assignment_id": outer_assignment,
                    "inner_allocation_method": "standard_random",
                    "source_row_index": source_start + source_offset,
                    "standardized_row_id": f"std_{source_start + source_offset}",
                    "study_no": study_no,
                    "Kd_final_source": "Freundlich",
                    "PFAS_name": "PFOA",
                    "adsorbent_id": f"AC_{source_offset}",
                    "y_true": 1.0,
                    "y_pred": 1.0 + residual,
                    "residual": residual,
                    "absolute_error": abs(residual),
                    "squared_error": residual**2,
                    "source_row_weight": 1.0,
                })
    rows.append({
        "data_split": "testing",
        "split_strategy": "random_row",
        "outer_allocation_method": "random_row",
        "outer_assignment_id": "outer_1",
        "inner_allocation_method": "coverage_aware",
        "source_row_index": 10,
        "standardized_row_id": "std_10",
        "study_no": "study_suspicious",
        "Kd_final_source": "Freundlich",
        "y_true": 1.0,
        "y_pred": 100.0,
        "residual": 99.0,
        "absolute_error": 99.0,
        "squared_error": 9801.0,
        "source_row_weight": 1.0,
    })
    return pd.DataFrame(rows)


class ResidualGroupAuditTests(unittest.TestCase):
    def test_systematic_study_source_signal_uses_one_inner_method_per_outer_split(self) -> None:
        groups, members, selection = build_residual_group_audit(
            _prediction_rows(),
            min_excess_median_absolute_error=0.5,
        )
        suspicious = groups.loc[groups["study_no"].eq("study_suspicious")].iloc[0]
        self.assertEqual(suspicious["audit_class"], "review_systematic_bias")
        self.assertEqual(suspicious["source_record_count"], 3)
        self.assertEqual(suspicious["heldout_source_assignment_count"], 6)
        self.assertAlmostEqual(suspicious["median_source_signed_residual"], 3.0)
        self.assertLess(suspicious["median_source_abs_error"], 10.0)
        self.assertEqual(selection["selected_inner_methods_by_outer_assignment_count"], {"standard_random": 2})
        self.assertEqual(len(groups), 1)
        self.assertNotIn("study_baseline", set(groups["study_no"]))
        self.assertEqual(len(members.loc[members["study_no"].eq("study_suspicious")]), 3)
        self.assertEqual(set(members["study_no"]), {"study_suspicious"})

    def test_non_actionable_screening_results_are_suppressed(self) -> None:
        predictions = _prediction_rows()
        suspicious = predictions["study_no"].eq("study_suspicious")
        source_offsets = predictions.loc[suspicious, "source_row_index"].astype(int) - 10
        mixed_residuals = source_offsets.map({0: 3.0, 1: -3.0, 2: 3.0})
        predictions.loc[suspicious, "residual"] = mixed_residuals
        predictions.loc[suspicious, "y_pred"] = (
            predictions.loc[suspicious, "y_true"] + mixed_residuals
        )
        predictions.loc[suspicious, "absolute_error"] = mixed_residuals.abs()
        predictions.loc[suspicious, "squared_error"] = mixed_residuals.pow(2)

        groups, members, selection = build_residual_group_audit(
            predictions,
            min_excess_median_absolute_error=0.5,
            min_same_sign_source_fraction=0.75,
        )

        self.assertTrue(groups.empty)
        self.assertTrue(members.empty)
        self.assertEqual(selection["study_reporting_source_groups_screened"], 2)
        self.assertEqual(selection["non_actionable_groups_suppressed"], 2)
        self.assertEqual(
            selection["screening_class_counts"]["high_error_mixed_direction"],
            1,
        )

    def test_grouped_holdout_pattern_is_retained_for_context_review(self) -> None:
        predictions = _prediction_rows().copy()
        predictions["split_strategy"] = "adsorbent"
        predictions["outer_allocation_method"] = "random_group"

        groups, members, selection = build_residual_group_audit(
            predictions,
            min_excess_median_absolute_error=0.5,
        )

        suspicious = groups.loc[groups["study_no"].eq("study_suspicious")].iloc[0]
        self.assertEqual(suspicious["audit_class"], "review_transfer_pattern")
        self.assertIn("not evidence to alter data", suspicious["manual_review_recommendation"])
        self.assertEqual(len(members.loc[members["study_no"].eq("study_suspicious")]), 3)
        self.assertEqual(selection["human_review_study_reporting_source_groups"], 1)
        self.assertEqual(selection["human_review_studies"], 1)

    def test_writer_creates_only_model_output_audit_files(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            prediction_path = root / "testing_predictions.parquet"
            audit_dir = root / "audit"
            _prediction_rows().to_parquet(prediction_path, index=False)
            manifest = write_residual_group_audit(
                prediction_path,
                audit_dir,
                min_excess_median_absolute_error=0.5,
            )
            self.assertTrue((audit_dir / "residual_group_audit.csv").exists())
            self.assertTrue((audit_dir / "residual_group_members.csv").exists())
            manifest_path = audit_dir / "residual_group_audit_manifest.json"
            self.assertTrue(manifest_path.exists())
            self.assertFalse((root / "record_review_decisions.csv").exists())
            persisted = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(persisted["counts"], manifest["counts"])
            self.assertEqual(persisted["schema_version"], 3)
            self.assertEqual(
                persisted["counts"]["human_review_study_reporting_source_groups"],
                1,
            )
            self.assertEqual(
                persisted["counts"]["non_actionable_groups_suppressed"],
                1,
            )


if __name__ == "__main__":
    unittest.main()
