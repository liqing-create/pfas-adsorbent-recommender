from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd


SCRIPTS_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(SCRIPTS_DIR))

from ML_model_training.backend import logkd_config as cfg
from adsorbent_recommender.catalog import catalog_eligibility, load_authoritative_catalog, present
from adsorbent_recommender.service import (
    ADSORBENT_CATEGORY_GROUPS,
    DEFAULT_RECOMMENDATION_COUNT,
    RecommenderBundle,
    _adsorbent_properties,
    _material_characterization,
    _requested_category_keys,
    _select_diverse,
    catalog_category_groups,
    filter_catalog_by_categories,
    fixed_test_conditions,
    pfas_features_source,
)
from adsorbent_recommender.pfas_cache import OPERA_CRITICAL_COLUMNS, PfasFeatureStore
from adsorbent_recommender.bundle import (
    BUNDLE_FILENAME,
    BUNDLE_VERSION,
    DEFAULT_FIXED_DOSE_MG_L,
    derive_feature_support_and_controls,
    export_recommender_bundle,
    load_bundle_manifest,
    make_bundle_self_contained,
    _snapshot_pfas_features,
)


class ConstantModel:
    def predict(self, frame):
        return np.full(len(frame), 4.0)


class RecommenderContractTests(unittest.TestCase):
    def test_default_recommendation_count_is_twenty(self):
        self.assertEqual(DEFAULT_RECOMMENDATION_COUNT, 20)

    def test_complete_result_request_preserves_global_rank_order(self):
        candidates = [
            {"product": {"adsorbent_subcategory": family}, "ranking_score": 4.0 - index}
            for index, family in enumerate(("A", "A", "A", "B"))
        ]
        self.assertEqual(_select_diverse(candidates, len(candidates)), candidates)
        # Short recommendation lists still retain the existing diversity rule.
        self.assertEqual(
            _select_diverse(candidates, 3),
            [candidates[0], candidates[1], candidates[3]],
        )

    def test_catalog_eligibility_uses_actual_selected_features(self):
        catalog = pd.DataFrame(
            [
                {"Product_key": "complete", "ssa_(m2/g)_avg": 1000, "AC_raw_material": "coal"},
                {"Product_key": "missing", "ssa_(m2/g)_avg": np.nan, "AC_raw_material": "coal"},
            ]
        )
        audit = catalog_eligibility(catalog, ["ssa_(m2/g)_avg", "AC_raw_material"])
        self.assertEqual(audit.loc[audit["Product_key"].eq("complete"), "eligible"].item(), True)
        self.assertEqual(audit.loc[audit["Product_key"].eq("missing"), "eligible"].item(), False)
        self.assertIn("ssa_(m2/g)_avg", audit.loc[audit["Product_key"].eq("missing"), "missing_required_material_features"].item())

    def test_support_contract_uses_only_equilibrium_controls(self):
        frame = pd.DataFrame(
            {
                "Adsorbent_dosage_value_mg/L": [1.0, 2.0, 4.0, 8.0],
                "Contact_time_(h)": [1.0, 2.0, 8.0, 24.0],
                "Temperature_(Â°C)": [20.0, 22.0, 25.0, 30.0],
                "Water_type": ["DI", "DI", "groundwater", "groundwater"],
            }
        )
        contract = derive_feature_support_and_controls(
            frame, list(frame.columns), list(frame.columns[:3]), ["Water_type"]
        )
        # Dose is held common across candidates so the ranking compares affinity
        # rather than each material's most favourable dose.
        dose = contract["controls"]["Adsorbent_dosage_value_mg/L"]
        self.assertEqual(dose["role"], "fixed")
        self.assertEqual(dose["default"], DEFAULT_FIXED_DOSE_MG_L)
        self.assertNotIn("Contact_time_(h)", contract["controls"])
        self.assertEqual(contract["feature_support"]["Contact_time_(h)"]["kind"], "numeric")
        self.assertEqual(contract["controls"]["Temperature_(Â°C)"]["role"], "fixed")
        self.assertEqual(contract["feature_support"]["Water_type"]["kind"], "categorical")

    def test_presence_predicate_treats_blank_placeholders_as_absent(self):
        # The training frame carries unreported values as empty strings rather
        # than nulls, so a null test would call every row fully characterized.
        # Coverage thresholds and catalog audits must both use this predicate.
        for blank in ("", "  ", "nan", "NaN", "N/A", "none", "not reported"):
            self.assertFalse(present(blank), blank)
        for value in ("0", "0.0", "False", "coal", 0, 0.0, False):
            self.assertTrue(present(value), repr(value))

        frame = pd.DataFrame({"a": ["1.0", ""], "b": ["", ""], "c": ["2.0", "3.0"]})
        self.assertEqual(frame.notna().sum(axis=1).tolist(), [3, 3])
        coverage = frame.apply(lambda column: column.map(present)).sum(axis=1)
        self.assertEqual(coverage.tolist(), [2, 1])

    def test_fixed_dose_is_overridable_and_flags_unsupported_choices(self):
        frame = pd.DataFrame({"Adsorbent_dosage_value_mg/L": [1.0, 10.0, 25.0, 50.0, 100.0]})
        selected = list(frame.columns)

        default = derive_feature_support_and_controls(frame, selected, selected, [])
        dose = default["controls"]["Adsorbent_dosage_value_mg/L"]
        self.assertEqual(dose["default"], DEFAULT_FIXED_DOSE_MG_L)
        self.assertTrue(dose["default_within_observed_support"])

        # A build may choose another dose without editing the policy table.
        overridden = derive_feature_support_and_controls(
            frame, selected, selected, [],
            control_defaults={"Adsorbent_dosage_value_mg/L": 50.0},
        )
        self.assertEqual(overridden["controls"]["Adsorbent_dosage_value_mg/L"]["default"], 50.0)

        # A dose outside the development range is recorded as such rather than
        # silently producing extrapolated scores.
        outside = derive_feature_support_and_controls(
            frame, selected, selected, [],
            control_defaults={"Adsorbent_dosage_value_mg/L": 5000.0},
        )
        self.assertFalse(outside["controls"]["Adsorbent_dosage_value_mg/L"]["default_within_observed_support"])

    def test_eligibility_threshold_reflects_partial_characterization(self):
        catalog = pd.DataFrame(
            [
                {"Product_key": "rich", "a": 1, "b": 2, "c": 3, "d": 4},
                {"Product_key": "adequate", "a": 1, "b": 2, "c": np.nan, "d": np.nan},
                {"Product_key": "thin", "a": 1, "b": np.nan, "c": np.nan, "d": np.nan},
            ]
        )
        required = ["a", "b", "c", "d"]

        # No threshold keeps the original all-present rule.
        strict = catalog_eligibility(catalog, required)
        self.assertEqual(strict["eligible"].tolist(), [True, False, False])
        self.assertEqual(strict["minimum_present_material_features"].tolist(), [4, 4, 4])

        # A threshold admits products characterized at least as well as the bar,
        # and still records how complete each one actually is.
        graded = catalog_eligibility(catalog, required, 2)
        self.assertEqual(graded["eligible"].tolist(), [True, True, False])
        self.assertEqual(graded["present_material_feature_count"].tolist(), [4, 2, 1])
        self.assertIn("b", graded.loc[graded["Product_key"].eq("thin"), "missing_required_material_features"].item())

        with self.assertRaises(ValueError):
            catalog_eligibility(catalog, required, 5)

    def test_catalog_labels_expand_into_model_indicator_columns(self):
        # The model consumes per-label indicators, so a catalog carrying only
        # the parent label must expand exactly as the training database did:
        # True/False where the parent is reported, blank where it is not.
        catalog = pd.DataFrame(
            [
                {"Product_key": "resin", "Functional_group": "quaternary ammonium", "Pore_class": "Gel"},
                {"Product_key": "carbon", "Functional_group": "", "Pore_class": "Macroporous"},
            ]
        )
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "catalog.xlsx"
            catalog.to_excel(path, sheet_name="Adsorbent_Properties", index=False)
            loaded = load_authoritative_catalog(path, "Adsorbent_Properties", category=None)

        resin = loaded[loaded["Product_key"].eq("resin")].iloc[0]
        carbon = loaded[loaded["Product_key"].eq("carbon")].iloc[0]
        # Case variation in the parent label must not lose the indicator.
        self.assertTrue(resin["contains_quaternary_ammonium"])
        self.assertTrue(resin["contains_gel"])
        self.assertFalse(resin["contains_macroporous"])
        # A reported pore class with an unreported functional group yields pore
        # indicators but leaves the functional-group indicators blank.
        self.assertTrue(carbon["contains_macroporous"])
        self.assertEqual(carbon["contains_quaternary_ammonium"], "")

    def test_every_model_family_screens_its_own_catalog_categories(self):
        catalog = pd.DataFrame(
            [
                {"Product_key": "ac-1", "Adsorbent_category": "Activated carbon", "ssa_(m2/g)_avg": 900},
                {"Product_key": "resin-1", "Adsorbent_category": "Ion exchange resin", "ssa_(m2/g)_avg": 40},
                {"Product_key": "clay-1", "Adsorbent_category": "Clay/mineral", "ssa_(m2/g)_avg": 20},
            ]
        )
        bounds = {"controls": {}, "feature_support": {"ssa_(m2/g)_avg": {"kind": "numeric", "count": 3}}}
        seen: dict[str, object] = {}

        def record(path, sheet, category=None):
            seen["category"] = category
            wanted = {str(label).casefold() for label in category}
            return catalog[catalog["Adsorbent_category"].str.casefold().isin(wanted)].copy()

        for family in ("AC", "Resin", "CDP", "Global"):
            with tempfile.TemporaryDirectory() as temp:
                output = Path(temp)
                for filename in ("run_config.json", "model_ensemble.joblib", "optimizer_bounds.json"):
                    (output / filename).write_text("{}", encoding="utf-8")
                with patch("adsorbent_recommender.bundle.load_authoritative_catalog", side_effect=record):
                    manifest = export_recommender_bundle(
                        output, family, "Kd", ["ssa_(m2/g)_avg"], ["ssa_(m2/g)_avg"], [],
                        {"Adsorbent properties": ["ssa_(m2/g)_avg"]}, bounds,
                    )
                # A Global bundle must screen the catalog rather than skip it,
                # and no family may be offered a category it was never fitted on.
                self.assertTrue(manifest["catalog"]["available"], family)
                self.assertEqual(seen["category"], cfg.MODEL_CATEGORIES[family], family)
                self.assertNotIn("Clay/mineral", seen["category"], family)

        self.assertEqual(manifest["catalog"]["screened_product_count"], 2)
        self.assertTrue(manifest["capabilities"]["catalog_screening"])

    def test_exported_bundle_validates_without_model_selection(self):
        catalog = pd.DataFrame(
            [{"Product_key": "ac-1", "Adsorbent_category": "Activated carbon", "SSA_(m2/g)": 900, "ssa_(m2/g)_avg": 900}]
        )
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp)
            for filename in ("run_config.json", "model_ensemble.joblib", "optimizer_bounds.json"):
                (output / filename).write_text("{}", encoding="utf-8")
            bounds = {"controls": {}, "feature_support": {"ssa_(m2/g)_avg": {"kind": "numeric", "count": 1}}}
            with patch("adsorbent_recommender.bundle.load_authoritative_catalog", return_value=catalog):
                manifest = export_recommender_bundle(
                    output, "AC", "Kd", ["ssa_(m2/g)_avg"], ["ssa_(m2/g)_avg"], [],
                    {"Adsorbent properties": ["ssa_(m2/g)_avg"]}, bounds,
                )
            self.assertTrue((output / BUNDLE_FILENAME).exists())
            self.assertTrue(manifest["capabilities"]["catalog_screening"])
            self.assertNotIn("model", manifest["artifacts"])
            self.assertNotIn("calibration", manifest["artifacts"])
            self.assertNotIn("applicability_domain_policy", manifest["artifacts"])
            self.assertNotIn("pfas_ad_reference", manifest["artifacts"])
            self.assertFalse((output / "applicability_domain_policy.json").exists())
            self.assertFalse((output / "pfas_ad_reference.csv").exists())
            self.assertNotIn("uncertainty_interval", manifest["capabilities"])
            self.assertNotIn("recipe_optimization", manifest["capabilities"])
            self.assertEqual(
                load_bundle_manifest(output)["bundle_version"],
                BUNDLE_VERSION,
            )

    def _export_minimal_bundle(self, output: Path, **kwargs):
        catalog = pd.DataFrame([{"Product_key": "ac-1", "Adsorbent_category": "Activated carbon", "ssa_(m2/g)_avg": 900}])
        bounds = {"controls": {}, "feature_support": {"ssa_(m2/g)_avg": {"kind": "numeric", "count": 1}}}
        output.mkdir(parents=True, exist_ok=True)
        for filename in ("run_config.json", "model_ensemble.joblib", "optimizer_bounds.json"):
            (output / filename).write_text("{}", encoding="utf-8")
        with patch("adsorbent_recommender.bundle.load_authoritative_catalog", return_value=catalog):
            return export_recommender_bundle(
                output, "AC", "Kd", ["ssa_(m2/g)_avg"], ["ssa_(m2/g)_avg"], [],
                {"Adsorbent properties": ["ssa_(m2/g)_avg"]}, bounds, **kwargs,
            )

    def test_exported_bundle_carries_its_pfas_feature_cache(self):
        with tempfile.TemporaryDirectory() as temp:
            source = Path(temp) / "training_machine" / "pfas_features.xlsx"
            source.parent.mkdir()
            pd.DataFrame([{"Canonical SMILES": "C", "Abbreviation": "EPF"}]).to_excel(source, index=False)
            output = Path(temp) / "bundle"
            manifest = self._export_minimal_bundle(output, pfas_features_path=source, pfas_features_sheet="Sheet1")
            self.assertTrue(manifest["pfas_features"]["available"])
            copied = output / manifest["pfas_features"]["snapshot_filename"]
            self.assertEqual(copied.read_bytes(), source.read_bytes())

            # Away from the training machine the recorded path is gone and the
            # bundle's own copy is what recommendation reads.
            source.unlink()
            config = {"pfas_features": {"path": str(source), "sheet_name": "Sheet1"}}
            bundle = RecommenderBundle(output, load_bundle_manifest(output), config, [], {}, pd.DataFrame())
            with patch.dict("os.environ", {"ADSORBENT_PFAS_FEATURES_PATH": ""}):
                self.assertEqual(pfas_features_source(bundle), (copied, "Sheet1"))

            copied.unlink()
            with self.assertRaisesRegex(FileNotFoundError, "pfas_features"):
                load_bundle_manifest(output)

    def test_bundled_pfas_cache_keeps_only_the_training_pfas(self):
        with tempfile.TemporaryDirectory() as temp:
            source = Path(temp) / "pfas_features.xlsx"
            with pd.ExcelWriter(source) as writer:
                pd.DataFrame([
                    {"Canonical SMILES": "OC(=O)C(F)(F)F", "Abbreviation": "TFA", "Compound Full Name": "Trifluoroacetic acid"},
                    {"Canonical SMILES": "FC(F)=C(F)F", "Abbreviation": None, "Compound Full Name": "Tetrafluoroethylene"},
                ]).to_excel(writer, sheet_name="Sheet1", index=False)
                pd.DataFrame([{"field": "created_at", "value": "2026-08-20"}]).to_excel(writer, sheet_name="generation_audit", index=False)
            records = pd.DataFrame({"PFAS_name": ["Trifluoroacetic acid", "TFA"]})
            output = Path(temp) / "bundle"
            output.mkdir()
            details = _snapshot_pfas_features(output, source, "Sheet1", records)
            sheets = pd.read_excel(output / "pfas_features.xlsx", sheet_name=None)
            self.assertEqual(list(sheets), ["Sheet1", "generation_audit"])
            self.assertEqual(sheets["Sheet1"]["Abbreviation"].tolist(), ["TFA"])
            self.assertEqual((details["source_rows"], details["rows"]), (2, 1))
            self.assertEqual(len(pd.read_excel(source)), 2)  # the shared workbook is untouched

    def test_bundle_exported_without_a_copy_can_be_made_self_contained(self):
        with tempfile.TemporaryDirectory() as temp:
            source = Path(temp) / "pfas_features.xlsx"
            pd.DataFrame([{"Canonical SMILES": "C", "Abbreviation": "EPF"}]).to_excel(source, index=False)
            output = Path(temp) / "bundle"
            manifest = self._export_minimal_bundle(output)
            self.assertFalse(manifest["pfas_features"]["available"])
            (output / "run_config.json").write_text(
                json.dumps({"pfas_features": {"path": str(source), "sheet_name": "Sheet1"}}), encoding="utf-8",
            )

            manifest = make_bundle_self_contained(output)
            self.assertTrue((output / "pfas_features.xlsx").exists())
            self.assertEqual(manifest["pfas_features"]["source_path"], str(source))
            self.assertEqual(load_bundle_manifest(output)["pfas_features"]["sheet_name"], "Sheet1")
            self.assertEqual(manifest["retrofits"][-1]["sections"][0], "pfas_features")

            # A second pass finds nothing missing and records no new retrofit.
            again = make_bundle_self_contained(output)
            self.assertEqual(len(again["retrofits"]), len(manifest["retrofits"]))

    def test_legacy_bundle_requires_rebuild(self):
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp)
            (output / BUNDLE_FILENAME).write_text(
                json.dumps({"bundle_format": "adsorbent_recommender_bundle", "bundle_version": 3}),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "rebuild"):
                load_bundle_manifest(output)

    def test_exact_cached_name_and_cas_lookup(self):
        row = {"SMILES": "C", "Canonical SMILES": "C", "PFAS_name": "Example PFAS", "CAS Number": "123-45-6", "Abbreviation": "EPF"}
        row.update({column: 1.0 for column in OPERA_CRITICAL_COLUMNS})
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "features.xlsx"
            pd.DataFrame([row]).to_excel(path, index=False)
            store = PfasFeatureStore(path)
            self.assertEqual(store.lookup_identifier("example pfas", "name").status, "cached")
            self.assertEqual(store.lookup_identifier("123456", "cas").status, "cached")
            self.assertEqual(store.lookup_identifier("999-99-9", "cas").status, "identifier_not_found")

    def test_fixed_condition_bundle_reports_the_dose_it_ranked_at(self):
        # The equilibrium ranking is only interpretable if the common
        # conditions travel with it.
        bounds = {
            "controls": {
                "Adsorbent_dosage_value_mg/L": {
                    "role": "fixed", "default": 25.0, "default_within_observed_support": True,
                },
                "Temperature_(°C)": {"role": "fixed", "default": 25.0},
            },
            "feature_support": {},
        }
        bundle = RecommenderBundle(Path("."), {"features": {}}, {}, [ConstantModel()], bounds, pd.DataFrame())
        conditions = fixed_test_conditions(bundle)
        self.assertEqual(conditions["status"], "fixed_conditions")
        self.assertEqual(conditions["dose_mg_L"], 25.0)
        self.assertTrue(conditions["dose_within_development_support"])


    def test_candidate_properties_are_category_aware_and_mark_model_inputs(self):
        resin = pd.Series({
            "adsorbent_category": "Ion exchange resin",
            "adsorbent_subcategory": "Strong base anion (Type I)",
            "AC_raw_material": "coal",
            "polymer_matrix": "polystyrene-DVB",
            "Functional_group": "quaternary ammonium",
            "ion_exchange_capacity_value_meq/L": 800.0,
        })
        properties = _adsorbent_properties(
            resin,
            ["adsorbent_subcategory", "polymer_matrix", "contains_quaternary_ammonium", "ion_exchange_capacity_value_meq/L", "AC_raw_material"],
        )
        by_label_unit = {(row["property"], row["unit"]): row for row in properties}
        self.assertNotIn(("AC raw material", ""), by_label_unit)
        capacity = by_label_unit[("Ion-exchange capacity", "meq/L")]
        self.assertEqual(capacity["value"], 800.0)
        self.assertTrue(capacity["used_by_selected_model"])
        self.assertTrue(by_label_unit[("Functional group", "")]["used_by_selected_model"])

    def test_characterization_flags_thin_and_unseen_products(self):
        thin = pd.Series({
            "Product_key": "thin", "present_material_feature_count": 9,
            "required_material_feature_count": 27, "minimum_present_material_features": 9,
            "missing_required_material_features": "phpzc", "mapping_status": "requires_curated_mapping",
        })
        rich = pd.Series({
            "Product_key": "rich", "present_material_feature_count": 22,
            "required_material_feature_count": 27, "minimum_present_material_features": 9,
            "missing_required_material_features": "", "mapping_status": "exact_normalized_key",
        })
        thin_report = _material_characterization(thin)
        self.assertEqual(thin_report["material_features_present"], 9)
        self.assertFalse(thin_report["seen_in_training"])
        # At the floor and unmatched: both facts must reach the caller.
        self.assertEqual(len(thin_report["warnings"]), 2)

        rich_report = _material_characterization(rich)
        self.assertTrue(rich_report["seen_in_training"])
        self.assertEqual(rich_report["warnings"], [])

class AdsorbentCategorySelectionTests(unittest.TestCase):
    """Screening one adsorbent family must screen exactly that family."""

    catalog = pd.DataFrame(
        [
            {"Product_key": "ac-1", "adsorbent_category": "Activated carbon"},
            {"Product_key": "ac-2", "adsorbent_category": "Activated carbon"},
            {"Product_key": "iex-1", "adsorbent_category": "Ion exchange resin"},
            {"Product_key": "non-1", "adsorbent_category": "Nonionic resin"},
            {"Product_key": "clay-1", "adsorbent_category": "Clay/mineral"},
        ]
    )

    def _bundle(self, catalog: pd.DataFrame) -> RecommenderBundle:
        return RecommenderBundle(Path("."), {}, {}, [], {}, catalog)

    def test_offered_families_are_the_configured_ones_with_their_counts(self):
        groups = {group["key"]: group for group in catalog_category_groups(self._bundle(self.catalog))}
        self.assertEqual(list(groups), [group["key"] for group in ADSORBENT_CATEGORY_GROUPS])
        self.assertEqual(groups["AC"]["product_count"], 2)
        # Both resin categories are one selectable family.
        self.assertEqual(groups["Resin"]["product_count"], 2)
        # A family the loaded bundle cannot screen is still reported, at zero,
        # so the caller can say it is unavailable rather than pretend it is
        # not part of the project.
        self.assertEqual(groups["CDP"]["product_count"], 0)

    def test_a_family_key_or_any_catalog_label_selects_that_family(self):
        self.assertEqual(_requested_category_keys({"adsorbent": {"categories": ["AC"]}}), (["AC"], []))
        self.assertEqual(
            _requested_category_keys({"adsorbent": {"categories": ["Ion exchange resin", "Nonionic resin"]}}),
            (["Resin"], []),
        )
        self.assertEqual(
            _requested_category_keys({"adsorbent": {"categories": ["AC", "Clay/mineral"]}}),
            (["AC"], ["Clay/mineral"]),
        )

    def test_selected_families_are_the_only_products_screened(self):
        filtered, warnings = filter_catalog_by_categories(self.catalog, ["Resin"])
        self.assertEqual(sorted(filtered["Product_key"]), ["iex-1", "non-1"])
        self.assertEqual(warnings, [])

    def test_no_selection_screens_the_whole_catalog(self):
        filtered, warnings = filter_catalog_by_categories(self.catalog, [])
        self.assertEqual(len(filtered), len(self.catalog))
        self.assertEqual(warnings, [])

    def test_a_catalog_without_categories_reports_that_it_could_not_filter(self):
        catalog = self.catalog.drop(columns=["adsorbent_category"])
        filtered, warnings = filter_catalog_by_categories(catalog, ["AC"])
        self.assertEqual(len(filtered), len(catalog))
        self.assertEqual(len(warnings), 1)


if __name__ == "__main__":
    unittest.main()
