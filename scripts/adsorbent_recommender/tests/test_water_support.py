from __future__ import annotations

import copy
import io
import json
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from adsorbent_recommender.water_support import diagnostics, export_support, load_support, validate_snapshot
from adsorbent_recommender.service import load_recommender_bundle, recommend, recommend_mixture
from adsorbent_recommender.scenarios import results_csv, make_snapshot, snapshot_bytes, read_snapshot, validate_inputs
from adsorbent_recommender.tests.test_material_support import inference, scenario

ROOT = Path(__file__).resolve().parents[3]
DIRECTORY = ROOT / "output/ML_logKd/Global_xgb_combination_42_20261015_073742"
if not DIRECTORY.is_dir():
    DIRECTORY = ROOT / "output/ML_logKd/Global_xgb_combination_42_20261014_184656"


@pytest.fixture(scope="module")
def deployed():
    if not (DIRECTORY / "recommender_bundle.json").exists():
        pytest.skip("Deployment bundle unavailable")
    return load_recommender_bundle(DIRECTORY)


@pytest.mark.parametrize("water,limited", [("groundwater", True), ("tap water", True), ("surface water", True),
    ("wastewater", True), ("AFFF solution", True), ("others", True), ("ultrapure water", False), ("synthetic water", False)])
def test_warning_is_conditional_on_selected_water(deployed, water, limited):
    assert diagnostics(deployed.water_support, water)["limited"] is limited


def test_counts_match_fitted_records_and_show_study_concentration(deployed):
    g = diagnostics(deployed.water_support, "groundwater")
    assert (g["model_rows"], g["source_records"], g["studies"], g["studies_with_other_water"]) == (552, 325, 5, 2)
    assert g["crosswater_study_ids"] == ["study_195", "study_208"]
    assert g["largest_study"] == "study_38" and g["largest_study_source_records"] == 284
    t = diagnostics(deployed.water_support, "tap water")
    assert (t["model_rows"], t["source_records"], t["studies"], t["studies_with_other_water"]) == (133, 25, 1, 0)


def test_legacy_bundles_and_unseen_categories_do_not_fabricate_support():
    assert not diagnostics({}, "groundwater")["available"]
    assert not diagnostics({}, "groundwater")["limited"]
    assert load_support(Path("."), {"bundle_version": 4}) == {}


def test_exports_exclude_held_out_rows_and_count_expanded_source_siblings_once(tmp_path):
    rows = [{"data_split": split, "source_row_index": source, "study_no": study, "Water_type": water}
        for split, source, study, water in [("training", 1, "s1", "groundwater"),
        ("training", 1, "s1", "groundwater"), ("training", 2, "s1", "ultrapure water"),
        ("testing", 3, "s2", "groundwater")]]
    metadata = export_support(tmp_path, pd.DataFrame(rows))
    support = load_support(tmp_path, {"water_support": metadata})
    g = diagnostics(support, "groundwater")
    assert (g["model_rows"], g["source_records"], g["studies"], g["studies_with_other_water"]) == (2, 1, 1, 1)
    assert diagnostics(support, "unseen matrix")["seen_in_training"] is False
    assert not diagnostics(support, "unseen matrix")["limited"]
    (tmp_path / metadata["snapshot_filename"]).write_text("{}")
    with pytest.raises(ValueError, match="fingerprint"):
        load_support(tmp_path, {"water_support": metadata})


def test_invalid_identity_coverage_is_unavailable_and_source_ids_are_scoped(tmp_path):
    assert not export_support(tmp_path, pd.DataFrame([{"Water_type": "groundwater"}]))["available"]
    rows = pd.DataFrame([{"Water_type": "groundwater", "study_no": f"s{i}", "source_row_index": 1, "extraction_dataset": f"d{i}"} for i in range(2)])
    metadata = export_support(tmp_path, rows)
    assert diagnostics(load_support(tmp_path, {"water_support": metadata}), "groundwater")["source_records"] == 2
    rows["extraction_dataset"] = "same"
    assert not export_support(tmp_path, rows)["available"]


def test_snapshot_validation_rejects_inconsistent_counts(deployed):
    data = copy.deepcopy(deployed.water_support)
    data["water_types"][0]["studies"] = 999
    with pytest.raises(ValueError):
        validate_snapshot(data)


def test_warning_follows_model_scores_into_mixture_csv_and_json(inference, deployed, monkeypatch):
    from adsorbent_recommender import service
    inference.water_support = deployed.water_support
    monkeypatch.setattr(service, "normalize_water_scenario", lambda *args: ({"Water_type": "groundwater"}, []))
    request = scenario()
    request["water"] = {"water_type": "groundwater"}
    request["temperature_c"] = None
    request["concentration_input"] = {"value": 1, "unit": "mg/L"}
    run = recommend_mixture(inference, [request])
    assert run["water_support"]["limited"]
    assert any(c["code"] == "limited_water_evidence" for c in run["recommendations"][0]["caveats"])
    frame = pd.read_csv(io.BytesIO(results_csv(run)))
    assert frame.water_model_support_warning.str.contains("groundwater").all()
    assert frame.water_training_studies.eq(5).all()
    # Complete snapshots retain the same optional support object.
    data = validate_inputs({"pfas": [{"identity": "PFOA", "identity_type": "abbreviation", "initial_concentration_mg_l": 1}],
        "water": {"water_type": "groundwater"}})
    from adsorbent_recommender.scenarios import scenario_requests
    run["scenarios"] = scenario_requests(data)
    saved = read_snapshot(snapshot_bytes(make_snapshot("Groundwater", data, {"id": "test"}, run)))
    assert saved["result"]["water_support"] == run["water_support"]


def test_database_scores_keep_only_evidence_caveats(inference, deployed, monkeypatch):
    from adsorbent_recommender import service
    inference.water_support = deployed.water_support
    monkeypatch.setattr(service, "normalize_water_scenario", lambda *args: ({"Water_type": "groundwater"}, []))
    monkeypatch.setattr(service, "match_candidate_evidence", lambda *args, **kwargs:
        {"status": "matched", "performance": {"mean_logKd": 3.2, "study_count": 1, "match_count": 1}})
    run = recommend(inference, scenario(), top_k=3)
    assert run["recommendations"]
    assert all(item["performance_source"] == "database_match" for item in run["recommendations"])
    assert all(not any(c["code"] == "limited_water_evidence" for c in item["caveats"]) for item in run["recommendations"])
