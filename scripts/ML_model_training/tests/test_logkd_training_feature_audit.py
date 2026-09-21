from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend import logkd_config as cfg  # noqa: E402
from backend.logkd_features import build_X, candidate_features, feature_audit  # noqa: E402


class TrainingScopePFASAuditTests(unittest.TestCase):
    def _frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "PFAS_name": [f"PFAS-{index}" for index in range(1, 7)],
                "study_no": [f"study-{index}" for index in range(1, 7)],
                "adsorbent_id": [f"adsorbent-{index}" for index in range(1, 7)],
                "rdkit_mol_logp": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
                "rdkit_tpsa": [2.0, 4.0, 6.0, 8.0, 10.0, 12.0],
            }
        )

    def _audit(self, frame: pd.DataFrame, **overrides):
        policy = {
            "pfas_feature_family_policy": "all",
            "pfas_missing_rate_threshold": 0.20,
            "pfas_dominant_fraction_threshold": 0.95,
            "correlation_threshold": 0.90,
            "min_pairwise_observations": 3,
            "correlated_feature_handling": "select_one_per_group",
            "min_nonmissing_rows": 1,
            "min_reporting_study_fraction": 0.01,
            "min_reporting_adsorbent_fraction": 0.01,
            "min_categorical_level_rows": 1,
            "min_categorical_level_entities": 1,
        }
        policy.update(overrides)
        return feature_audit(frame, model_name="AC", target="Kd_final_log10(L/g)", **policy)

    def test_correlations_are_calculated_from_the_current_training_scope(self) -> None:
        selected, _numeric, _categorical, manifest = self._audit(self._frame())

        self.assertNotIn("rdkit_mol_logp", selected)
        self.assertIn("rdkit_tpsa", selected)
        dropped = manifest.set_index("feature").loc["rdkit_mol_logp"]
        self.assertEqual(dropped["reason"], "correlated_feature_group_pruned")
        self.assertEqual(int(dropped["pfas_training_unique_pfas"]), 6)
        self.assertEqual(dropped["correlation_selected_feature"], "rdkit_tpsa")

    def test_missingness_is_calculated_from_the_current_training_scope(self) -> None:
        frame = self._frame()
        frame.loc[[0, 1], "rdkit_mol_logp"] = None
        selected, _numeric, _categorical, manifest = self._audit(frame)

        self.assertNotIn("rdkit_mol_logp", selected)
        row = manifest.set_index("feature").loc["rdkit_mol_logp"]
        self.assertEqual(row["reason"], "pfas_high_missingness")
        self.assertAlmostEqual(float(row["pfas_training_missing_rate"]), 2 / 6)

    def test_no_screening_retains_sparse_and_constant_features_without_claiming_a_pass(self) -> None:
        frame = self._frame()
        frame["pH"] = [7.0, None, None, None, None, None]
        frame["rdkit_mol_logp"] = [1.0] * len(frame)

        selected, _numeric, _categorical, manifest = self._audit(
            frame,
            feature_screening_mode="none",
            correlated_feature_handling="use_all",
        )

        self.assertIn("pH", selected)
        self.assertIn("rdkit_mol_logp", selected)
        rows = manifest.set_index("feature")
        self.assertEqual(rows.loc["pH", "reason"], "retained_without_screening")
        self.assertEqual(rows.loc["rdkit_mol_logp", "reason"], "retained_without_screening")
        self.assertEqual(rows.loc["pH", "availability_independent_support_reason"], "not_applied")
        self.assertEqual(rows.loc["pH", "variation_support_reason"], "not_applied")
        self.assertTrue(pd.isna(rows.loc["pH", "availability_independent_support_pass"]))
        self.assertTrue(pd.isna(rows.loc["pH", "variation_support_pass"]))

    def test_bucket_ablation_excludes_candidates_before_screening(self) -> None:
        frame = self._frame()
        frame["pH"] = [4.0, 5.0, 6.0, 7.0, 8.0, 9.0]

        selected, _numeric, _categorical, manifest = self._audit(
            frame,
            excluded_feature_buckets=("PFAS characteristics",),
            correlated_feature_handling="use_all",
        )

        self.assertIn("pH", selected)
        self.assertNotIn("rdkit_mol_logp", selected)
        row = manifest.set_index("feature").loc["rdkit_mol_logp"]
        self.assertEqual(row["reason"], "feature_bucket_excluded_by_ablation")
        self.assertEqual(row["availability_independent_support_reason"], "not_assessed")

    def test_correlation_group_selection_prioritizes_lower_missingness(self) -> None:
        frame = self._frame()
        frame.loc[0, "rdkit_mol_logp"] = None
        selected, _numeric, _categorical, manifest = self._audit(frame)

        self.assertNotIn("rdkit_mol_logp", selected)
        self.assertIn("rdkit_tpsa", selected)
        dropped = manifest.set_index("feature").loc["rdkit_mol_logp"]
        self.assertEqual(dropped["reason"], "correlated_feature_group_pruned")
        self.assertEqual(dropped["correlation_selected_feature"], "rdkit_tpsa")

    def test_use_all_does_not_prune_a_correlated_group(self) -> None:
        selected, _numeric, _categorical, manifest = self._audit(
            self._frame(),
            correlated_feature_handling="use_all",
        )

        self.assertIn("rdkit_mol_logp", selected)
        self.assertIn("rdkit_tpsa", selected)
        row = manifest.set_index("feature").loc["rdkit_mol_logp"]
        self.assertFalse(bool(row["correlation_feature_dropped"]))
        self.assertEqual(row["correlation_selected_feature"], "rdkit_tpsa")

    def test_non_pfas_redundancy_is_also_calculated_from_the_training_scope(self) -> None:
        frame = self._frame()
        frame["pH"] = [4.0, 5.0, 6.0, 7.0, 8.0, 9.0]
        frame["Temperature_(°C)"] = [8.0, 10.0, 12.0, 14.0, 16.0, 18.0]
        frame.iloc[:, -1] = [0.0, 0.000001, 0.000002, 0.000003, 0.000004, 0.000005]
        selected, _numeric, _categorical, manifest = self._audit(
            frame,
            correlated_feature_handling="select_one_per_group",
            correlation_threshold=0.90,
            min_pairwise_observations=3,
        )

        self.assertIn("pH", selected)
        self.assertNotIn("Temperature_(°C)", selected)
        dropped = manifest.set_index("feature").loc["Temperature_(°C)"]
        self.assertEqual(dropped["reason"], "correlated_feature_group_pruned")
        self.assertEqual(dropped["correlation_selected_feature"], "pH")
        self.assertEqual(dropped["correlation_scope"], "training_rows")

    def test_adsorbent_correlation_uses_distinct_study_adsorbent_units(self) -> None:
        frame = self._frame()
        frame["ssa_(m2/g)_avg"] = [100.0, 110.0, 120.0, 130.0, 140.0, 150.0]
        frame["pore_volume_total_value_cm3/g"] = [0.10, 0.11, 0.12, 0.13, 0.14, 0.15]
        selected, _numeric, _categorical, manifest = self._audit(frame)

        self.assertIn("ssa_(m2/g)_avg", selected)
        self.assertNotIn("pore_volume_total_value_cm3/g", selected)
        dropped = manifest.set_index("feature").loc["pore_volume_total_value_cm3/g"]
        self.assertEqual(dropped["reason"], "correlated_feature_group_pruned")
        self.assertEqual(dropped["correlation_selected_feature"], "ssa_(m2/g)_avg")
        self.assertEqual(dropped["correlation_scope"], "unique_study_adsorbent")

    def test_adsorbent_specific_candidates_enter_global_but_not_other_specialists(self) -> None:
        fields = [
            "ion_exchange_capacity_value_meq/g",
            "ion_exchange_capacity_value_meq/L",
            "polymer_matrix",
        ]
        frame = pd.DataFrame(columns=fields)

        resin = candidate_features(frame, model_name="Resin", structure_mode="all")
        ac = candidate_features(frame, model_name="AC", structure_mode="all")
        cdp = candidate_features(frame, model_name="CDP", structure_mode="all")
        global_candidates = candidate_features(
            frame, model_name="Global", structure_mode="all"
        )

        self.assertTrue(set(fields).issubset(resin))
        self.assertTrue(set(fields).isdisjoint(ac))
        self.assertTrue(set(fields).isdisjoint(cdp))
        self.assertTrue(set(fields).issubset(global_candidates))

    def test_ac_feature_names_match_the_database_contract(self) -> None:
        ac = candidate_features(
            pd.DataFrame(), model_name="AC", structure_mode="all"
        )
        resin = candidate_features(
            pd.DataFrame(), model_name="Resin", structure_mode="all"
        )
        cdp = candidate_features(
            pd.DataFrame(), model_name="CDP", structure_mode="all"
        )
        global_candidates = candidate_features(
            pd.DataFrame(), model_name="Global", structure_mode="all"
        )

        self.assertIn("AC_raw_material", ac)
        self.assertIn("AC_activation_method", ac)
        self.assertNotIn("raw_material", ac)
        self.assertNotIn("activation_method", ac)
        self.assertEqual(
            set(cfg.AC_CANDIDATES),
            {"AC_raw_material", "AC_activation_method"},
        )
        for candidates in (resin, cdp):
            self.assertTrue(set(cfg.AC_CANDIDATES).isdisjoint(candidates))
        self.assertTrue(set(cfg.AC_CANDIDATES).issubset(global_candidates))


class ConditionalAdsorbentFeatureEncodingTests(unittest.TestCase):
    def _frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "adsorbent_category": [
                    "Activated carbon",
                    "Ion exchange resin",
                    "Nonionic resin",
                    "Cyclodextrin polymers",
                    "Unknown material",
                    None,
                ],
                "AC_raw_material": ["coal", None, None, None, None, None],
                "AC_activation_method": ["Physical_Steam", None, None, None, None, None],
                "ion_exchange_capacity_value_meq/g": [None, 2.1, None, None, None, None],
                "polymer_matrix": [None, "Polystyrene (PS)", "Polyacrylic", None, None, None],
                # Functional groups are not family-confined in the standardized
                # data, so this stays an ordinary sparse indicator.
                "contains_amine": [0, 1, None, 1, None, None],
            }
        )

    def test_global_candidates_include_all_explicit_specialist_properties(self) -> None:
        global_candidates = candidate_features(
            self._frame(), model_name="Global", structure_mode="all"
        )

        self.assertTrue(set(cfg.AC_CANDIDATES).issubset(global_candidates))
        self.assertTrue(set(cfg.RESIN_CANDIDATES).issubset(global_candidates))
        self.assertNotIn(
            "contains_amine", cfg.ADSORBENT_SPECIFIC_FEATURE_APPLICABILITY
        )

    def test_build_x_distinguishes_not_applicable_from_missing_and_unknown(self) -> None:
        frame = self._frame()
        X = build_X(
            frame,
            [
                "AC_raw_material",
                "AC_activation_method",
                "ion_exchange_capacity_value_meq/g",
                "polymer_matrix",
                "contains_amine",
            ],
            ["ion_exchange_capacity_value_meq/g", "contains_amine"],
            ["AC_raw_material", "AC_activation_method", "polymer_matrix"],
        )

        self.assertEqual(X.loc[0, "AC_raw_material"], "coal")
        self.assertEqual(X.loc[1, "AC_raw_material"], cfg.ADSORBENT_NOT_APPLICABLE_CATEGORY)
        self.assertEqual(X.loc[3, "AC_raw_material"], cfg.ADSORBENT_NOT_APPLICABLE_CATEGORY)
        self.assertEqual(X.loc[4, "AC_raw_material"], "<missing>")
        self.assertEqual(X.loc[5, "AC_raw_material"], "<missing>")
        self.assertEqual(X.loc[0, "polymer_matrix"], cfg.ADSORBENT_NOT_APPLICABLE_CATEGORY)
        self.assertEqual(X.loc[1, "polymer_matrix"], "Polystyrene (PS)")
        self.assertEqual(X.loc[2, "polymer_matrix"], "Polyacrylic")
        self.assertEqual(X.loc[3, "polymer_matrix"], cfg.ADSORBENT_NOT_APPLICABLE_CATEGORY)
        self.assertTrue(pd.isna(X.loc[0, "ion_exchange_capacity_value_meq/g"]))
        self.assertEqual(X.loc[1, "ion_exchange_capacity_value_meq/g"], 2.1)
        self.assertTrue(pd.isna(X.loc[2, "ion_exchange_capacity_value_meq/g"]))
        np.testing.assert_allclose(
            X["ion_exchange_capacity_value_meq/g__applicability"].to_numpy(),
            np.array([0.0, 1.0, 0.0, 0.0, np.nan, np.nan]),
            equal_nan=True,
        )
        self.assertEqual(X.loc[3, "contains_amine"], 1.0)

    def test_reported_value_for_a_configured_inapplicable_family_is_rejected(self) -> None:
        frame = self._frame().iloc[[0]].copy()
        frame.loc[frame.index[0], "ion_exchange_capacity_value_meq/g"] = 2.1

        with self.assertRaisesRegex(ValueError, "not applicable"):
            build_X(
                frame,
                ["ion_exchange_capacity_value_meq/g"],
                ["ion_exchange_capacity_value_meq/g"],
                [],
            )


class BinaryIndicatorHandlingTests(unittest.TestCase):
    """Database-side one-hot flags must not be one-hot encoded a second time.

    ``contains_<ion>`` columns are derived from one ``Inorganic_matter`` field
    and are already indicators, with a blank meaning the field was never
    reported.  Modeling them as numeric keeps one column per flag and leaves
    "never reported" missing, instead of turning it into a category that is
    duplicated across every flag derived from the same source field.
    """

    def _frame(self) -> pd.DataFrame:
        rows = 12
        return pd.DataFrame(
            {
                "PFAS_name": [f"PFAS-{index % 6 + 1}" for index in range(rows)],
                "study_no": [f"study-{index % 4 + 1}" for index in range(rows)],
                "adsorbent_id": [f"adsorbent-{index % 3 + 1}" for index in range(rows)],
                "pH": [3.0, 4.5, 5.0, 6.5, 7.0, 7.5, 8.0, 8.5, 9.0, 9.5, 10.0, 11.0],
                # Reported for eight rows, never reported for the last four.
                "contains_Ca": [True, False, True, False, True, False, True, False, None, None, None, None],
                "contains_Na": [False, True, False, True, False, True, False, True, None, None, None, None],
            }
        )

    def _audit(self, frame: pd.DataFrame, handling: str, **overrides):
        policy = {
            "pfas_feature_family_policy": "all",
            "feature_screening_mode": "none",
            "min_nonmissing_rows": 1,
            "min_reporting_study_fraction": 0.01,
            "min_reporting_adsorbent_fraction": 0.01,
            "min_categorical_level_rows": 1,
            "min_categorical_level_entities": 1,
            "binary_indicator_handling": handling,
        }
        policy.update(overrides)
        return feature_audit(
            frame, model_name="AC", target="Kd_final_log10(L/g)", **policy
        )

    def test_binary_indicators_are_numeric_by_default(self) -> None:
        self.assertEqual(cfg.DEFAULT_BINARY_INDICATOR_HANDLING, "numeric")
        _selected, numeric, categorical, manifest = self._audit(self._frame(), "numeric")

        for feature in ("contains_Ca", "contains_Na"):
            self.assertIn(feature, numeric)
            self.assertNotIn(feature, categorical)
            self.assertEqual(manifest.set_index("feature").loc[feature, "kind"], "numeric")

    def test_categorical_handling_restores_the_second_encoding(self) -> None:
        _selected, numeric, categorical, manifest = self._audit(self._frame(), "categorical")

        for feature in ("contains_Ca", "contains_Na"):
            self.assertIn(feature, categorical)
            self.assertNotIn(feature, numeric)
            self.assertEqual(manifest.set_index("feature").loc[feature, "kind"], "categorical")

    def test_a_genuinely_multiclass_feature_is_unaffected(self) -> None:
        frame = self._frame()
        frame["AC_raw_material"] = ["coconut", "coal", "wood"] * 4

        for handling in cfg.BINARY_INDICATOR_HANDLING_CHOICES:
            with self.subTest(handling=handling):
                _selected, numeric, categorical, _manifest = self._audit(frame, handling)
                self.assertIn("AC_raw_material", categorical)
                self.assertNotIn("AC_raw_material", numeric)

    def test_a_continuous_feature_is_unaffected(self) -> None:
        for handling in cfg.BINARY_INDICATOR_HANDLING_CHOICES:
            with self.subTest(handling=handling):
                _selected, numeric, _categorical, _manifest = self._audit(self._frame(), handling)
                self.assertIn("pH", numeric)

    def test_an_unknown_handling_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            self._audit(self._frame(), "one_hot_twice_please")

    def test_standard_screening_keeps_a_numeric_indicator(self) -> None:
        """Numeric value-diversity thresholds must not reject a 0/1 flag.

        A binary indicator has exactly two distinct values, which is below the
        default numeric distinct-value threshold.  It is screened on level
        support instead, so a well-represented flag survives screening.
        """

        selected, numeric, _categorical, manifest = self._audit(
            self._frame(), "numeric", feature_screening_mode="standard"
        )

        for feature in ("contains_Ca", "contains_Na"):
            row = manifest.set_index("feature").loc[feature]
            self.assertEqual(row["kind"], "numeric")
            self.assertTrue(bool(row["is_binary_indicator"]))
            self.assertEqual(int(row["supported_category_levels"]), 2)
            self.assertIn(feature, selected)
            self.assertIn(feature, numeric)

    def test_standard_screening_still_drops_a_constant_indicator(self) -> None:
        frame = self._frame()
        frame["contains_Na"] = [False] * 8 + [None] * 4

        selected, _numeric, _categorical, manifest = self._audit(
            frame, "numeric", feature_screening_mode="standard"
        )

        self.assertNotIn("contains_Na", selected)
        self.assertEqual(
            manifest.set_index("feature").loc["contains_Na", "reason"],
            "low_supported_category_levels",
        )

    def test_a_level_confined_to_one_study_is_unsupported(self) -> None:
        """Rows are not independent observations of a level.

        ``contains_Ca`` is true on eight rows, which clears any row floor, but
        every one of them belongs to one study.  The level describes a single
        unit, so it must not count as supported.
        """
        frame = self._frame()
        frame["study_no"] = ["study-1"] * 8 + [f"study-{index}" for index in range(2, 6)]
        frame["contains_Ca"] = [True] * 8 + [None] * 4

        selected, _numeric, _categorical, manifest = self._audit(
            frame,
            "numeric",
            feature_screening_mode="standard",
            min_categorical_level_entities=3,
        )
        row = manifest.set_index("feature").loc["contains_Ca"]

        self.assertNotIn("contains_Ca", selected)
        self.assertEqual(row["reason"], "low_supported_category_levels")
        self.assertEqual(int(row["rarest_level_entities"]), 1)

    def test_a_level_spread_across_studies_stays_supported(self) -> None:
        frame = self._frame()
        frame["study_no"] = [f"study-{index % 4 + 1}" for index in range(12)]
        frame["contains_Ca"] = [True, False] * 4 + [None] * 4

        selected, _numeric, _categorical, manifest = self._audit(
            frame,
            "numeric",
            feature_screening_mode="standard",
            min_categorical_level_entities=2,
        )
        row = manifest.set_index("feature").loc["contains_Ca"]

        self.assertIn("contains_Ca", selected)
        self.assertEqual(int(row["supported_category_levels"]), 2)
        self.assertEqual(int(row["rarest_level_entities"]), 2)

    def test_a_material_property_is_counted_over_adsorbent_identities(self) -> None:
        """A material feature counts materials, not study--adsorbent reports.

        One resin described by four studies is one unit here, even though
        reporting coverage counts those four reports separately.
        """
        frame = self._frame()
        frame["study_no"] = [f"study-{index % 4 + 1}" for index in range(12)]
        frame["adsorbent_identity_key"] = ["specific::ONE"] * 8 + ["specific::TWO"] * 4
        frame["contains_macroporous"] = [True] * 8 + [False] * 4

        selected, _numeric, _categorical, manifest = self._audit(
            frame,
            "numeric",
            feature_screening_mode="standard",
            min_categorical_level_entities=2,
        )
        row = manifest.set_index("feature").loc["contains_macroporous"]

        self.assertNotIn("contains_macroporous", selected)
        self.assertEqual(row["reason"], "low_supported_category_levels")
        self.assertEqual(int(row["rarest_level_entities"]), 1)


if __name__ == "__main__":
    unittest.main()
