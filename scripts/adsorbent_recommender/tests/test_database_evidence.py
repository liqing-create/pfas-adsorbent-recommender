from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd


SCRIPTS_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(SCRIPTS_DIR))

from adsorbent_recommender.evidence import match_candidate_evidence, requested_conditions
from adsorbent_recommender.service import RecommenderBundle, recommend, recommend_mixture


def _row(**overrides):
    row = {
        "standardized_row_id": "std_1",
        "study_no": "study_1",
        "DOI": "10.1000/example",
        "authors": "A. Author",
        "article_title": "PFAS adsorption",
        "source_title": "Water Research",
        "publication_year": 2026,
        "performance_id": "PFOA|F400|isotherm|pH7",
        "adsorbent_id": "F400",
        "name_abbreviation": "F400",
        "name_commercial": "Filtrasorb 400",
        "name_full": "Filtrasorb 400 granular activated carbon",
        "PFAS_name": "PFOA",
        "Water_type": "ultrapure water",
        "PFAS_C0_value_mg/L": 1.0,
        "pH": 7.0,
        "Temperature_(°C)": 25.0,
        "Contact_time_(h)": 24.0,
        "organic_carbon_mg/L": 0.0,
        "TDS_(mg/L)": 0.0,
        "Ionic_strength_(mol/L)": 0.0,
        "search_logKd_log10(L/g)": 3.2,
        "search_Kd_L/g": 10**3.2,
        "search_dosage_mg/L": 25.0,
        "reference_dosage_target_mg/L": 25.0,
        "reference_dose_basis": "isotherm_recalculated_at_reference_dose",
        "reference_dose_method": "Langmuir",
        "reference_dose_exact_flag": True,
        "reference_isotherm_R2": 0.99,
        "contains_Ca": False,
    }
    row.update(overrides)
    return row


class DatabaseEvidenceTests(unittest.TestCase):
    def setUp(self):
        self.material = {
            "Product_key": "FILTRASORB400",
            "Name_Commercial": "Filtrasorb 400",
            "model_adsorbent_id": "F400",
        }
        self.pfas = {"PFAS_name": "Perfluorooctanoic acid", "Abbreviation": "PFOA"}
        self.scenario = {
            # These routing fixtures lack fitted support snapshots. Eligibility
            # and the default gate are exercised in test_material_support.py.
            "adsorbent": {"include_exploratory": True},
            "initial_concentration_mg_l": 1.1,
            "water": {"water_type": "Ultrapure water", "pH": 7.2, "ions": {}},
        }
        self.water = {"Water_type": "ultrapure water", "pH": 7.2}
        self.screening = {"fixed_conditions": {"Temperature_(°C)": 25.0}}

    def match(self, rows):
        return match_candidate_evidence(
            pd.DataFrame(rows),
            material=self.material,
            pfas_feature_row=self.pfas,
            pfas_cache_key="PFOA",
            scenario=self.scenario,
            normalized_water=self.water,
            screening_conditions=self.screening,
        )

    def test_exact_identity_gates_and_numeric_tolerances(self):
        matched = self.match([_row()])
        self.assertEqual(matched["status"], "matched")
        self.assertEqual(matched["performance"]["source"], "database_match")
        self.assertEqual(matched["citations"][0]["url"], "https://doi.org/10.1000/example")

        for changed in (
            {"PFAS_name": "PFOS"},
            {"name_commercial": "Other carbon", "adsorbent_id": "other", "name_abbreviation": "other", "name_full": "other"},
            {"Water_type": "groundwater"},
        ):
            self.assertEqual(self.match([_row(**changed)])["status"], "no_match")

        # pH 7.8 lies outside the explicit ±0.5-unit allowance.
        self.assertEqual(self.match([_row(pH=7.8)])["status"], "no_match")

    def test_evidence_selection_does_not_cherry_pick_high_logkd(self):
        closer_lower = _row(
            standardized_row_id="std_close",
            PFAS_C0_value_mg_L=1.1,
            pH=7.2,
            search_logKd_log10_L_g=2.8,
        )
        # Use the real column spellings after convenient keyword construction.
        closer_lower["PFAS_C0_value_mg/L"] = closer_lower.pop("PFAS_C0_value_mg_L")
        closer_lower["search_logKd_log10(L/g)"] = closer_lower.pop("search_logKd_log10_L_g")
        farther_higher = _row(
            standardized_row_id="std_far",
            PFAS_C0_value_mg_L=1.8,
            pH=7.45,
            search_logKd_log10_L_g=5.0,
        )
        farther_higher["PFAS_C0_value_mg/L"] = farther_higher.pop("PFAS_C0_value_mg_L")
        farther_higher["search_logKd_log10(L/g)"] = farther_higher.pop("search_logKd_log10_L_g")
        middle = _row(
            standardized_row_id="std_middle",
            PFAS_C0_value_mg_L=1.2,
            pH=7.25,
            search_logKd_log10_L_g=3.0,
        )
        middle["PFAS_C0_value_mg/L"] = middle.pop("PFAS_C0_value_mg_L")
        middle["search_logKd_log10(L/g)"] = middle.pop("search_logKd_log10_L_g")
        result = self.match([farther_higher, closer_lower, middle])
        self.assertEqual(result["status"], "matched")
        self.assertEqual(result["primary_record"]["standardized_row_id"], "std_close")
        self.assertEqual(result["performance"]["mean_logKd"], 3.0)
        self.assertEqual(result["performance"]["method"], "study_balanced_median")

    def test_missing_water_type_forces_model_fallback(self):
        result = match_candidate_evidence(
            pd.DataFrame([_row()]),
            material=self.material,
            pfas_feature_row=self.pfas,
            pfas_cache_key="PFOA",
            scenario={"initial_concentration_mg_l": 1.0, "water": {}},
            normalized_water={"Water_type": ""},
            screening_conditions=self.screening,
        )
        self.assertEqual(result["status"], "no_match")
        self.assertIn("water type", result["missing_exact_gates"])

    def test_contact_time_is_not_an_equilibrium_match_condition(self):
        conditions = requested_conditions(
            {"initial_concentration_mg_l": 1.0, "contact_time_h": 48.0, "water": {}},
            {},
            {},
        )
        self.assertNotIn("contact_time_h", conditions)

    def test_recommender_uses_database_per_candidate_and_model_for_the_rest(self):
        catalog = pd.DataFrame(
            [
                {"Product_key": "FILTRASORB400", "Name_Commercial": "Filtrasorb 400", "model_adsorbent_id": "F400", "adsorbent_subcategory": "GAC"},
                {"Product_key": "OTHER", "Name_Commercial": "Other carbon", "model_adsorbent_id": "OTHER", "adsorbent_subcategory": "GAC"},
            ]
        )
        manifest = {
            "model_family": "Global",
            "target": "Kd",
            "features": {"selected": [], "numeric": [], "categorical": []},
            "input_contract": {"fields": []},
            "capabilities": {"catalog_screening": True},
        }
        evidence_row = _row()
        evidence_row["search_logKd_log10(L/g)"] = 3.0
        bundle = RecommenderBundle(
            Path("."), manifest, {}, [], {"controls": {}, "feature_support": {}}, catalog,
            pd.DataFrame([evidence_row]),
            {"available": True, "rows": 1},
        )
        lookup = SimpleNamespace(
            status="cached", canonical_smiles="", cache_key="PFOA", warnings=(),
            feature_row=self.pfas, offline_request=None,
        )
        with (
            patch("adsorbent_recommender.service.resolve_pfas", return_value=(lookup, {"rows": 1})),
            patch("adsorbent_recommender.service.normalize_water_scenario", return_value=(self.water, [])),
            patch("adsorbent_recommender.service._model_row", return_value=({}, [], [])),
            patch("adsorbent_recommender.service.predict_logkd", return_value={"mean_logKd": 2.0, "ensemble_sd": 0.1}),
        ):
            result = recommend(bundle, self.scenario, top_k=2)
        by_product = {item["product"]["Product_key"]: item for item in result["recommendations"]}
        self.assertEqual(result["result_schema_version"], 5)
        self.assertEqual(by_product["FILTRASORB400"]["performance_source"], "database_match")
        self.assertEqual(by_product["OTHER"]["performance_source"], "model_prediction")
        self.assertEqual(result["evidence_routing"]["database_matched_candidates"], 1)
        self.assertEqual(result["evidence_routing"]["model_fallback_candidates"], 1)
        self.assertNotIn("laboratory_validation_plan", result)
        self.assertIn("adsorbent_properties", by_product["FILTRASORB400"])

    def test_mixture_ranks_by_each_adsorbents_worst_pfas_and_keeps_breakdown(self):
        catalog = pd.DataFrame(
            [
                {"Product_key": "FILTRASORB400", "Name_Commercial": "Filtrasorb 400", "model_adsorbent_id": "F400", "adsorbent_subcategory": "GAC"},
                {"Product_key": "OTHER", "Name_Commercial": "Other carbon", "model_adsorbent_id": "OTHER", "adsorbent_subcategory": "GAC"},
            ]
        )
        manifest = {
            "model_family": "Global",
            "target": "Kd",
            "features": {"selected": [], "numeric": [], "categorical": []},
            "input_contract": {"fields": []},
            "capabilities": {"catalog_screening": True},
        }
        bundle = RecommenderBundle(
            Path("."), manifest, {}, [], {"controls": {}, "feature_support": {}}, catalog,
            pd.DataFrame([_row()]), {"available": True, "rows": 1},
        )
        scenarios = [
            {**self.scenario, "pfas": {"identity": "PFOA", "identity_type": "abbreviation"}},
            {**self.scenario, "pfas": {"identity": "PFOS", "identity_type": "abbreviation"}, "initial_concentration_mg_l": 0.2},
        ]

        def resolve(_bundle, scenario, _python):
            name = scenario["pfas"]["identity"]
            lookup = SimpleNamespace(
                status="cached", canonical_smiles="", cache_key=name, warnings=(),
                feature_row={"PFAS_name": name, "Abbreviation": name}, offline_request=None,
            )
            return lookup, {"rows": 2}

        def model_row(_bundle, pfas, _scenario, _water, material):
            return ({"pfas": pfas["Abbreviation"], "product": material["Product_key"]}, [], [])

        model_scores = {
            ("PFOA", "FILTRASORB400"): 2.0,
            ("PFOA", "OTHER"): 3.2,
            ("PFOS", "FILTRASORB400"): 2.5,
            ("PFOS", "OTHER"): 1.5,
        }

        def predict(_bundle, row):
            return {"mean_logKd": model_scores[(row["pfas"], row["product"])], "ensemble_sd": 0.1}

        with (
            patch("adsorbent_recommender.service.resolve_pfas", side_effect=resolve),
            patch("adsorbent_recommender.service.normalize_water_scenario", return_value=(self.water, [])),
            patch("adsorbent_recommender.service._model_row", side_effect=model_row),
            patch("adsorbent_recommender.service.predict_logkd", side_effect=predict),
        ):
            result = recommend_mixture(bundle, scenarios, top_k=2)

        first = result["recommendations"][0]
        self.assertEqual(first["product"]["Product_key"], "FILTRASORB400")
        self.assertEqual(first["mixture_performance"]["worst_case_logKd"], 2.5)
        self.assertEqual(first["mixture_performance"]["limiting_pfas"], "PFOS")
        self.assertEqual(first["performance_source"], "mixed_evidence")
        self.assertEqual(len(first["pfas_results"]), 2)
        self.assertEqual(result["evidence_routing"]["database_matched_pairs"], 1)
        self.assertEqual(result["evidence_routing"]["total_candidate_pfas_pairs"], 4)


if __name__ == "__main__":
    unittest.main()
