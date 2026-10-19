"""Caveats must describe the score a candidate is ranked by, most severe first."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

from adsorbent_recommender.bundle import (
    TRAINING_SUPPORT_FILENAME,
    _snapshot_training_support,
    training_support_table,
)
from adsorbent_recommender.service import (
    RecommenderBundle,
    _candidate_caveats,
    _mark_rank_resolution,
    _material_characterization,
    _mixture_caveats,
    _scenario_caveats,
    _align_categorical_value,
    _training_support,
    canonicalize_catalog_categories,
)


TRAINING_ROWS = pd.DataFrame([
    {"study_no": "s1", "PFAS_name": "PFOA", "adsorbent_id": "F400-a", "name_commercial": "Filtrasorb 400", "name_abbreviation": "F400", "name_full": ""},
    {"study_no": "s2", "PFAS_name": "PFOA", "adsorbent_id": "GAC-1", "name_commercial": "Filtrasorb 400", "name_abbreviation": "", "name_full": None},
    {"study_no": "s2", "PFAS_name": "PFBS", "adsorbent_id": "GAC-1", "name_commercial": "Filtrasorb 400", "name_abbreviation": "", "name_full": None},
])


def _bundle(support: pd.DataFrame, fields: list[dict] | None = None) -> RecommenderBundle:
    manifest = {"input_contract": {"fields": fields or []}}
    return RecommenderBundle(
        Path("."), manifest, {}, [], {}, pd.DataFrame(),
        training_support=support, training_support_info={},
    )


class TrainingSupportTests(unittest.TestCase):
    def test_counts_records_and_studies_per_alias(self):
        table = training_support_table(TRAINING_ROWS).set_index(["kind", "key"])
        self.assertEqual(table.loc[("pfas", "pfoa"), "records"], 2)
        self.assertEqual(table.loc[("pfas", "pfoa"), "studies"], 2)
        # One record naming the same product twice still counts once.
        self.assertEqual(table.loc[("adsorbent", "filtrasorb400"), "records"], 3)
        self.assertEqual(table.loc[("adsorbent", "f400"), "studies"], 1)
        self.assertNotIn(("adsorbent", ""), table.index)

    def test_snapshot_records_the_tuned_validation_rmse(self):
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder)
            (output / "tuning_summary.json").write_text(json.dumps({"best_validation_rmse": 0.435}), encoding="utf-8")
            details = _snapshot_training_support(output, TRAINING_ROWS)
            self.assertTrue((output / TRAINING_SUPPORT_FILENAME).is_file())
        self.assertEqual(details["pfas_count"], 2)
        self.assertAlmostEqual(details["validation_rmse"], 0.435)

    def test_product_seen_under_its_commercial_name_counts_as_seen(self):
        bundle = _bundle(training_support_table(TRAINING_ROWS))
        material = pd.Series({
            "Product_key": "FILTRASORB400", "Name_Commercial": "Filtrasorb 400",
            "present_material_feature_count": 20, "required_material_feature_count": 27,
            "minimum_present_material_features": 9, "mapping_status": "requires_curated_mapping",
        })
        support = _training_support(bundle, "adsorbent", {"filtrasorb400"})
        report = _material_characterization(material, support)
        # The identity map missed it; the training rows did not.
        self.assertTrue(report["seen_in_training"])
        self.assertEqual(report["training_studies"], 2)
        self.assertEqual(report["warnings"], [])

        unseen = _material_characterization(
            material.copy().replace({"mapping_status": "exact_normalized_key"}),
            {"records": 0, "studies": 0},
        )
        self.assertFalse(unseen["seen_in_training"])


class CandidateCaveatTests(unittest.TestCase):
    seen = {"seen_in_training": True, "at_eligibility_floor": False}

    def test_database_score_carries_only_evidence_caveats(self):
        performance = {
            "study_count": 1, "match_count": 3, "exact_reference_dose_match_count": 0,
            "matched_logKd_min": 1.0, "matched_logKd_max": 2.5,
        }
        unseen = {"seen_in_training": False, "at_eligibility_floor": True}
        caveats = _candidate_caveats(
            "database_match", performance, {"mean_logKd": 5.0}, unseen,
            [{"feature": "SSA", "kind": "range", "text": "SSA is outside the development range [1, 2]."}], [],
        )
        codes = [caveat["code"] for caveat in caveats]
        self.assertEqual(codes, ["database_spread", "database_one_study", "database_near_dose"])

    def test_model_caveats_put_the_unseen_material_first(self):
        unseen_thin = {
            "seen_in_training": False, "at_eligibility_floor": True,
            "material_features_present": 9, "material_features_required": 27,
        }
        caveats = _candidate_caveats(
            "model_prediction", {}, {"mean_logKd": 4.6}, unseen_thin, [], ["value dropped"],
        )
        self.assertEqual(
            [caveat["code"] for caveat in caveats],
            ["unseen_material", "extreme_logkd", "thin_characterization", "inapplicable_value"],
        )

    def test_well_supported_prediction_has_no_caveats(self):
        self.assertEqual(_candidate_caveats("model_prediction", {}, {"mean_logKd": 2.0}, self.seen, [], []), [])

    def test_mixture_names_the_pfas_only_for_score_caveats(self):
        product = {"code": "unseen_material", "severity": "critical", "label": "Unseen material", "text": "unseen"}
        score = {"code": "extreme_logkd", "severity": "critical", "label": "Extreme logKd", "text": "low"}
        merged = _mixture_caveats([
            {"pfas": "PFOA", "caveats": [product]},
            {"pfas": "PFBA", "caveats": [product, score]},
        ])
        self.assertEqual([(caveat["code"], caveat["pfas"]) for caveat in merged], [
            ("unseen_material", []), ("extreme_logkd", ["PFBA"]),
        ])


class ScenarioCaveatTests(unittest.TestCase):
    def test_unseen_pfas_and_blank_inputs_are_said_once(self):
        bundle = _bundle(
            training_support_table(TRAINING_ROWS),
            [{"model_feature": "pH", "source": "water", "label": "pH"}],
        )
        lookup = SimpleNamespace(feature_row={"PFAS_name": "GenX"}, cache_key="GenX")
        caveats = _scenario_caveats(bundle, lookup, "GenX", [], ["pH"], [])
        self.assertEqual([caveat["code"] for caveat in caveats], ["unseen_pfas", "missing_inputs"])
        self.assertIn("pH", caveats[1]["text"])

    def test_without_a_support_table_no_pfas_caveat_is_guessed(self):
        lookup = SimpleNamespace(feature_row={"PFAS_name": "GenX"}, cache_key="GenX")
        self.assertEqual(_scenario_caveats(_bundle(pd.DataFrame()), lookup, "GenX", [], [], []), [])


class RankResolutionTests(unittest.TestCase):
    def test_candidates_within_the_error_of_the_top_are_flagged(self):
        items = [{"ranking_score": score} for score in (3.0, 2.7, 2.5)]
        summary = _mark_rank_resolution(items, 0.4)
        self.assertEqual(summary["within_error_count"], 2)
        self.assertEqual([item["rank_resolution"]["within_error_of_top"] for item in items], [True, True, False])
        self.assertEqual(_mark_rank_resolution(items, None), {"available": False})


class CatalogVocabularyTests(unittest.TestCase):
    """Catalog categories must reach the model in the training vocabulary."""

    def test_catalog_categories_use_the_database_canonical_names(self):
        catalog = pd.DataFrame([
            {"adsorbent_category": "Ion exchange resin", "polymer_matrix": "styrene-divinylbenzene",
             "adsorbent_subcategory": "Strong base anion (Type II)", "AC_raw_material": None},
            {"adsorbent_category": "Activated carbon", "polymer_matrix": None,
             "adsorbent_subcategory": "granular activated carbon (GAC)", "AC_raw_material": "Coconut shell"},
        ])
        result = canonicalize_catalog_categories(catalog)
        self.assertEqual(result.loc[0, "polymer_matrix"], "Polystyrene-divinylbenzene (PS-DVB)")
        self.assertEqual(result.loc[1, "adsorbent_subcategory"], "granular activated carbon")
        self.assertEqual(result.loc[1, "AC_raw_material"], "coconut")
        # Blank cells stay missing rather than becoming empty strings.
        self.assertTrue(pd.isna(result.loc[0, "AC_raw_material"]))
        self.assertTrue(pd.isna(result.loc[1, "polymer_matrix"]))
        # The input frame is not modified.
        self.assertEqual(catalog.loc[0, "polymer_matrix"], "styrene-divinylbenzene")

    def test_subcategory_qualifiers_do_not_make_a_new_category(self):
        support = {"kind": "categorical", "levels": ["strong base anion", "granular activated carbon"]}
        self.assertEqual(
            _align_categorical_value("adsorbent_subcategory", "Strong base anion (Type II)", support),
            "strong base anion",
        )
        self.assertEqual(
            _align_categorical_value("adsorbent_subcategory", "Weak base anion (Type II)", support),
            "Weak base anion (Type II)",
        )


if __name__ == "__main__":
    unittest.main()
