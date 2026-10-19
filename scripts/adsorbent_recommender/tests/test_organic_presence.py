from __future__ import annotations

import csv
import io
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from adsorbent_recommender.service import normalize_water_scenario
from adsorbent_recommender.scenarios import (
    ScenarioError, read_scenario_csv, read_snapshot, scenario_inputs_csv, snapshot_bytes, validate_inputs,
)
from adsorbent_recommender.tests.test_scenarios import inputs, snapshot
from adsorbent_recommender.tests.test_candidate_evidence import match
from adsorbent_recommender.tests.test_database_evidence import _row


@pytest.mark.parametrize("present", [False, True])
def test_explicit_flag_reaches_model_features_without_a_composition_or_carbon_estimate(present):
    row, warnings = normalize_water_scenario({"water": {"organic_matter_present": present}})
    assert row["organic_matter_present"] is present
    assert row["Organic_matter"] in {"", "none"}
    assert not warnings
    if present:
        assert row["Water_type"] == "synthetic water"
        assert row["organic_carbon_mg/L"] == ""
    else:
        assert row["Water_type"] == "ultrapure water"
        assert row["organic_carbon_mg/L"] == 0


@pytest.mark.parametrize("present,carbon", [(True, 3), (False, 3), (True, 0), (False, 0)])
def test_measured_carbon_and_additive_presence_are_independent(present, carbon):
    row, _ = normalize_water_scenario({"water": {"water_type": "groundwater",
        "organic_matter_present": present, "organic_carbon_mg_l": carbon}})
    assert row["organic_carbon_mg/L"] == carbon
    assert row["organic_matter_present"] is present
    assert row["Water_type"] == "groundwater"


@pytest.mark.parametrize("value", ["humic acid", "true", 1, None, []])
def test_presence_input_is_boolean(value):
    with pytest.raises(ValueError, match="true or false"):
        normalize_water_scenario({"water": {"organic_matter_present": value}})
    data = inputs()
    data["water"]["organic_matter_present"] = value
    with pytest.raises(ScenarioError, match="true or false"):
        validate_inputs(data)


def test_reporting_organic_presence_does_not_change_inorganic_ions_or_ionic_strength():
    row, _ = normalize_water_scenario({"water": {"organic_matter_present": True,
        "inorganic_additives": [{"species": "NaCl", "concentration": 1, "unit": "mM"}]}})
    assert float(row["Ionic_strength_(mol/L)"]) == pytest.approx(.001)
    assert row["contains_Na"] and row["contains_Cl"] and row["organic_matter_present"]


def test_database_matches_require_a_marked_presence_flag():
    source = _row(Water_type="synthetic water", organic_matter_present=True)
    assert match([source], water={"organic_matter_present": True})["status"] == "matched"
    for value in (False, None):
        source["organic_matter_present"] = value
        rejected = match([source], water={"organic_matter_present": True})
        assert rejected["status"] == "no_match"
        assert "organic_matter_present" in rejected["related_evidence"]["excluded_by_condition"]


@pytest.mark.parametrize("present", [False, True])
def test_csv_and_json_roundtrip_the_presence_flag_without_composition_columns(present):
    saved = snapshot()
    saved["inputs"]["water"]["organic_matter_present"] = present
    saved["result"] = None
    assert read_snapshot(snapshot_bytes(saved))["inputs"] == saved["inputs"]
    payload = scenario_inputs_csv(saved)
    _, loaded = read_scenario_csv(payload)
    assert loaded["water"]["organic_matter_present"] is present
    assert loaded["water"]["organic_carbon_mg_l"] == saved["inputs"]["water"]["organic_carbon_mg_l"]
    rows = list(csv.DictReader(io.StringIO(payload.decode("utf-8-sig"))))
    assert "organic_matter" not in rows[0]
    assert all(row["organic_matter_present"] == str(present).lower() for row in rows)


def test_composition_text_and_malformed_csv_presence_are_rejected():
    data = inputs()
    data["water"]["organic_matter"] = "5 mg/L humic acid"
    with pytest.raises(ScenarioError, match="composition text"):
        validate_inputs(data)
    with pytest.raises(ValueError, match="composition text"):
        normalize_water_scenario({"water": data["water"]})
    rows = list(csv.DictReader(io.StringIO(scenario_inputs_csv(snapshot()).decode("utf-8-sig"))))
    rows[0]["organic_matter_present"] = "maybe"
    stream = io.StringIO()
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    with pytest.raises(ScenarioError, match="organic_matter_present"):
        read_scenario_csv(stream.getvalue().encode())
