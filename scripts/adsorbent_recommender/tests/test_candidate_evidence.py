from __future__ import annotations
import copy
import io
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from adsorbent_recommender.evidence import DEFAULT_MATCH_POLICY, _numeric_distance, match_candidate_evidence, requested_conditions
from adsorbent_recommender.service import normalize_water_scenario
from adsorbent_recommender.scenarios import results_csv
from adsorbent_recommender.tests.test_database_evidence import _row

MATERIAL = {"Product_key": "FILTRASORB400", "Name_Commercial": "Filtrasorb 400", "model_adsorbent_id": "F400"}


def match(rows, *, c0=1, water=None):
    scenario = {"initial_concentration_mg_l": c0, "water": {"water_type": "ultrapure water", **(water or {})}}
    normalized, _ = normalize_water_scenario(scenario)
    return match_candidate_evidence(pd.DataFrame(rows), material=MATERIAL,
        pfas_feature_row={"PFAS_name": "PFOA"}, pfas_cache_key="PFOA", scenario=scenario,
        normalized_water=normalized, screening_conditions={})


@pytest.mark.parametrize("observed,requested,expected", [(0, 0, 0), (0, 1, None), (1, 0, None),
    (-1, -1, None), (float("nan"), 0, None), (1, 2, 1), (.5, 1, 1), (1, 1, 0)])
def test_logarithmic_zero_fix_keeps_positive_tolerances_and_missing_distinct(observed, requested, expected):
    actual = _numeric_distance(observed, requested, DEFAULT_MATCH_POLICY["numeric_tolerances"]["organic_carbon_mg_L"])
    assert actual is None if expected is None else actual == pytest.approx(expected)


@pytest.mark.parametrize("field", ["organic_carbon_mg_l", "ionic_strength_mol_l", "tds_mg_l"])
def test_explicit_zero_matches_source_zero(field):
    result = match([_row()], water={field: 0})
    assert result["status"] == "matched" and result["performance"]["match_count"] == 1


@pytest.mark.parametrize("field,column", [("organic_carbon_mg_l", "organic_carbon_mg/L"),
    ("ionic_strength_mol_l", "Ionic_strength_(mol/L)"), ("tds_mg_l", "TDS_(mg/L)")])
def test_zero_request_does_not_match_unreported_or_positive_source(field, column):
    for value in (None, 1):
        row = _row(**{column: value})
        result = match([row], water={field: 0})
        assert result["status"] == "no_match"
        assert result["related_evidence"]["records"][0]["condition_differences"]


def test_normalization_preserves_explicit_ph_and_tds_zero_and_default_water():
    scenario = {"water": {"water_type": "groundwater", "pH": 0, "tds_mg_l": 0}}
    water, _ = normalize_water_scenario(scenario)
    assert water["pH"] == water["TDS_(mg/L)"] == 0
    assert requested_conditions(scenario, water, {})["tds_mg_L"] == 0
    assert normalize_water_scenario({"water": {}})[0]["Water_type"] == "ultrapure water"


def test_related_evidence_is_same_pfas_and_product_and_cannot_supply_a_score():
    rows = [_row(), _row(standardized_row_id="wrong_pfas", PFAS_name="PFOS"),
            _row(standardized_row_id="wrong_material", adsorbent_id="OTHER", name_commercial="OTHER", name_abbreviation="OTHER", name_full="OTHER")]
    result = match(rows, c0=.0001)
    assert result["status"] == "no_match" and "performance" not in result
    related = result["related_evidence"]
    assert related["record_count"] == 1 and related["used_for_ranking"] is False
    assert related["records"][0]["condition_differences"][0]["field"] == "initial_concentration_mg_L"
    assert related["records"][0]["standardized_row_id"] == "std_1"


def test_matched_and_related_records_are_disjoint_and_large_related_score_does_not_affect_aggregate():
    result = match([_row(), _row(standardized_row_id="different", Water_type="groundwater", **{"search_logKd_log10(L/g)": 99})])
    assert result["status"] == "matched" and result["performance"]["mean_logKd"] == 3.2
    assert result["performance"]["match_count"] == 1
    assert result["related_evidence"]["record_count"] == 1
    assert result["related_evidence"]["records"][0]["standardized_row_id"] == "different"
    assert result["related_evidence"]["excluded_by_condition"] == {"water_type": 1}


def test_related_records_explain_multiple_differences_and_unreported_values():
    result = match([_row(Water_type="groundwater", pH=None)], c0=.001, water={"pH": 7.2, "ions": {"Ca": True}})
    fields = {item["field"]: item for item in result["related_evidence"]["records"][0]["condition_differences"]}
    assert set(fields) == {"water_type", "initial_concentration_mg_L", "pH", "contains_Ca"}
    assert fields["pH"]["reason"] == "unreported"


def test_representative_related_records_cover_studies_and_are_not_selected_by_score():
    rows = [_row(standardized_row_id=f"r{index}", study_no=f"s{index // 3}", DOI=f"10.1234/{index // 3}") for index in range(9)]
    first = match(rows, c0=.001)["related_evidence"]
    for index, row in enumerate(rows):
        row["search_logKd_log10(L/g)"] = 10 - index
    second = match(rows, c0=.001)["related_evidence"]
    assert first["record_count"] == 9 and first["study_count"] == 3 and first["records_shown"] == 3
    assert [row["standardized_row_id"] for row in first["records"]] == [row["standardized_row_id"] for row in second["records"]]
    assert len({row["study_no"] for row in first["records"]}) == 3


def test_unverified_label_cannot_qualify_a_generated_structure_for_related_evidence():
    result = match_candidate_evidence(pd.DataFrame([_row()]), material=MATERIAL,
        pfas_feature_row={"Canonical SMILES": "new structure"}, pfas_cache_key=None,
        scenario={"pfas": {"label": "PFOA"}, "initial_concentration_mg_l": 1},
        normalized_water={"Water_type": "ultrapure water"}, screening_conditions={})
    assert result["status"] == "no_match" and not result.get("related_evidence", {}).get("record_count")


def test_missing_and_unavailable_evidence_remain_actionable():
    assert match([])["status"] == "no_match"
    result = match([_row(PFAS_name="PFOS")])
    assert result["related_evidence"]["record_count"] == 0


def test_results_csv_exports_related_counts_without_replacing_predicted_score():
    from adsorbent_recommender.tests.test_scenarios import result
    run = result()
    evidence = match([_row()], c0=.0001)
    run["recommendations"][0]["database_evidence"] = evidence
    csv = pd.read_csv(io.BytesIO(results_csv(run)))
    assert csv.iloc[0]["related_record_count"] == 1
    assert csv.iloc[0]["comparable_record_count"] == 0
    assert csv.iloc[0]["limiting_logkd"] == 4.2
