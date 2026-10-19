from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from adsorbent_recommender.scenarios import (
    MAX_SCENARIO_BYTES, ScenarioError, add_snapshot, bundle_reference,
    compare_snapshots, comparison_csv, make_snapshot, read_snapshot,
    read_scenario_csv, results_csv, scenario_inputs_csv, scenario_requests, snapshot_bytes, validate_inputs,
)

SMILES = "O=C(O)C(F)(F)F"
OTHER = "O=C(O)C(F)(F)C(F)(F)F"
REFERENCE = {"id": "bundle-fingerprint", "name": "example", "target": "logKd"}


def inputs():
    return validate_inputs({"pfas": [{"identity": "Research PFAS", "identity_type": "name",
        "confirmed_smiles": SMILES, "canonical_smiles": SMILES, "structure_confirmed": True,
        "resolution": {"source": "pubchem", "cid": 123}, "label": "Study compound",
        "initial_concentration_mg_l": .0001, "concentration_value": 100, "concentration_unit": "ng/L"}],
        "water": {"water_type": "groundwater", "pH": 7.2, "organic_carbon_mg_l": 3,
                  "organic_matter_present": True, "inorganic_additives": [{"species": "CaCl2", "concentration": 2, "unit": "mM"}], "ions": {"Ca": True}},
        "temperature_c": None})


def result(data=None):
    data = data or inputs()
    run = {"result_schema_version": 5, "status": "recommendations_ready", "scenarios": scenario_requests(data),
        "bundle": {"directory": "old/computer/example", "model_family": "Global", "target": "logKd"},
        "mixture": {"aggregation": "maximin_logKd", "pfas_count": len(data["pfas"])},
        "pfas_lookups": [{"canonical_smiles": SMILES, "support": {"seen_in_training": False}} for _ in data["pfas"]],
        "scenario_caveats": [{"text": "Some matrix chemistry is unknown."}],
        "recommendations": [
            {"rank": 1, "product": {"Product_key": "a", "Name_Commercial": "Same display name"},
             "ranking_score": 4.2, "performance_source": "model_prediction",
             "mixture_performance": {"limiting_pfas": "Study compound", "limiting_pfas_unseen": True},
             "warnings": ["Validate experimentally."], "caveats": [{"text": "Exploratory prediction."}]},
            {"rank": 2, "product": {"Product_key": "b", "Name_Commercial": "Same display name"},
             "ranking_score": 4.0, "performance_source": "database_match"},
        ]}
    if len(data["pfas"]) > 1:
        for candidate in run["recommendations"]:
            candidate["pfas_results"] = [
                {"input_identity": row["identity"], "identity_type": row["identity_type"],
                 "initial_concentration_mg_l": row["initial_concentration_mg_l"],
                 "performance": {"mean_logKd": candidate["ranking_score"]},
                 "performance_source": candidate["performance_source"]}
                for row in data["pfas"]
            ]
    return run


def snapshot(name="Baseline", data=None, run=None):
    data = data or inputs()
    return make_snapshot(name, data, REFERENCE, run or result(data))


def test_roundtrip_retains_units_structure_provenance_and_all_evidence():
    saved = snapshot()
    loaded = read_snapshot(snapshot_bytes(saved))
    assert loaded["inputs"] == saved["inputs"]
    assert loaded["result"] == saved["result"]
    assert loaded["bundle_reference"] == REFERENCE
    assert loaded["_imported"] is True
    assert loaded["inputs"]["pfas"][0]["concentration_unit"] == "ng/L"
    assert loaded["result"]["recommendations"][0]["caveats"]
    saved["result"]["recommendations"][0]["ranking_score"] = 0
    assert loaded["result"]["recommendations"][0]["ranking_score"] == 4.2


def test_inputs_only_can_be_saved_before_screening():
    saved = make_snapshot("Draft", inputs(), REFERENCE)
    assert read_snapshot(snapshot_bytes(saved))["result"] is None


def test_complete_mixture_and_cached_confirmation_flags_roundtrip_without_dropping_rows():
    data = inputs()
    second = {"identity": "PFOS", "identity_type": "abbreviation", "structure_confirmed": True,
              "initial_concentration_mg_l": .00025, "concentration_value": .25, "concentration_unit": "µg/L"}
    data["pfas"].append(second)
    data = validate_inputs(data)
    run = result(data)
    run["pfas_lookups"][1]["canonical_smiles"] = OTHER
    saved = read_snapshot(snapshot_bytes(make_snapshot("Mixture", data, REFERENCE, run)))
    assert len(saved["inputs"]["pfas"]) == len(saved["result"]["scenarios"]) == 2
    assert saved["inputs"]["pfas"][1]["structure_confirmed"] is True
    assert saved["inputs"]["pfas"][1]["concentration_unit"] == "µg/L"


@pytest.mark.parametrize("mutate", [
    lambda data: data["pfas"][0].update(initial_concentration_mg_l=.001, concentration_value=1000),
    lambda data: data["water"].update(pH=8),
    lambda data: data.update(temperature_c=30),
    lambda data: data["pfas"][0].update(identity="Another PFAS"),
])
def test_results_for_old_inputs_cannot_be_attached_to_a_changed_scenario(mutate):
    data = inputs()
    run = result(data)
    mutate(data)
    with pytest.raises(ScenarioError, match="disagree"):
        make_snapshot("Changed", data, REFERENCE, run)


@pytest.mark.parametrize("mutate", [
    lambda doc: doc.update(scenario_version=99),
    lambda doc: doc["inputs"]["pfas"][0].update(concentration_value=1),
    lambda doc: doc["inputs"]["pfas"][0].update(identity_type="auto"),
    lambda doc: doc["inputs"]["pfas"][0].update(confirmed_smiles="not a structure"),
    lambda doc: doc["inputs"]["pfas"][0].update(confirmed_smiles=OTHER),
    lambda doc: doc["inputs"]["water"].update(ions={"Ca": "yes"}),
    lambda doc: doc["result"]["recommendations"][0].update(ranking_score=None),
    lambda doc: doc["result"]["recommendations"][1]["product"].update(Product_key="a"),
])
def test_corrupt_or_inconsistent_files_are_rejected(mutate):
    doc = snapshot()
    mutate(doc)
    with pytest.raises(ScenarioError):
        read_snapshot(json.dumps(doc).encode())


@pytest.mark.parametrize("payload", [b"not json", b"[]", b'{"format": NaN}', b"\xff", b"x" * (MAX_SCENARIO_BYTES + 1)],
                         ids=["malformed", "non-object", "nonfinite", "encoding", "oversize"])
def test_upload_validation_is_bounded_and_json_only(payload):
    with pytest.raises(ScenarioError):
        read_snapshot(payload)


def test_future_result_schema_keeps_inputs_and_requires_a_new_screen():
    doc = snapshot()
    doc["result"]["result_schema_version"] = 99
    loaded = read_snapshot(json.dumps(doc).encode())
    assert loaded["inputs"] == doc["inputs"] and loaded["result"] is None
    assert "unsupported" in loaded["_notices"][0]


def test_existing_complete_recommendation_exports_can_reload_inputs():
    loaded = read_snapshot(json.dumps(result()).encode(), "Existing export")
    assert loaded["name"] == "Existing export" and loaded["result"]
    assert loaded["inputs"]["pfas"][0]["identity"] == "Research PFAS"
    assert loaded["inputs"]["pfas"][0]["concentration_value"] == 100
    assert loaded["inputs"]["water"] == inputs()["water"]
    assert loaded["bundle_reference"]["id"] is None and loaded["_notices"]


def test_duplicate_names_and_ids_create_distinct_snapshots_without_overwriting():
    first = snapshot()
    library, added = add_snapshot([first], first)
    assert len(library) == 2 and added["name"] == "Baseline (2)"
    assert added["scenario_id"] != first["scenario_id"]
    assert first["name"] == "Baseline"


def test_candidate_comparison_joins_product_keys_and_preserves_missing_scores():
    first = snapshot()
    second = snapshot("Modified")
    second["result"]["recommendations"][0]["ranking_score"] = 4.5
    second["result"]["recommendations"] = second["result"]["recommendations"][:1]
    rows, warnings = compare_snapshots([first, second], first["scenario_id"])
    assert not warnings and len(rows) == 2
    assert rows[0]["Modified · ΔlogKd"] == pytest.approx(.3)
    assert rows[1]["Modified · logKd"] is None
    assert rows[1]["Modified · basis"] == "Not in saved ranking"
    assert "unseen PFAS" in rows[0]["Modified · caveats"]
    assert "Exploratory prediction" in rows[0]["Modified · caveats"]
    assert rows[0]["Baseline · basis"] == "Model" and rows[1]["Baseline · basis"] == "Database"


def test_database_scores_keep_only_caveats_for_the_evidence_actually_used():
    first, second = snapshot(), snapshot("Modified")
    item = second["result"]["recommendations"][1]
    item.update({"warnings": ["The unused model has weak characterization."],
                 "caveats": [{"text": "Evidence comes from one study.", "pfas": ["PFAS A"]}]})
    rows, _ = compare_snapshots([first, second], first["scenario_id"])
    assert rows[1]["Modified · caveats"] == "PFAS A: Evidence comes from one study."


@pytest.mark.parametrize("change", ["bundle", "mixture", "unknown_provenance"])
def test_score_deltas_are_withheld_for_different_contexts(change):
    first, second = snapshot(), snapshot("Modified")
    second["bundle_reference"] = copy.deepcopy(REFERENCE)
    if change == "mixture":
        second["result"]["pfas_lookups"][0]["canonical_smiles"] = OTHER
    else:
        second["bundle_reference"]["id"] = "different" if change == "bundle" else None
    rows, warnings = compare_snapshots([first, second], first["scenario_id"])
    assert warnings and rows[0]["Modified · ΔlogKd"] is None
    assert rows[0]["Modified · logKd"] == 4.2


def test_equivalent_protonation_and_aliases_share_structural_comparison_context():
    first, second = snapshot(), snapshot("Alias")
    second["result"]["pfas_lookups"][0]["canonical_smiles"] = "[O-]C(=O)C(F)(F)F"
    second["inputs"]["pfas"][0]["identity"] = "Another label"
    rows, warnings = compare_snapshots([first, second], first["scenario_id"])
    assert not warnings and rows[0]["Alias · ΔlogKd"] == 0


def test_comparison_csv_preserves_caveats_and_escapes_formula_text():
    payload = comparison_csv([{"Product": "=1+1", "logKd": -1.2, "Caveats": "Validate, then shortlist"}]).decode("utf-8-sig")
    assert "'=1+1" in payload and "-1.2" in payload and '"Validate, then shortlist"' in payload


def test_bundle_reference_changes_when_model_or_data_changes(tmp_path):
    (tmp_path / "recommender_bundle.json").write_text(json.dumps({"artifacts": {"ensemble": "weights.dat"}, "catalog": {"snapshot_filename": "catalog.csv"}}))
    (tmp_path / "weights.dat").write_bytes(b"unchanged model")
    (tmp_path / "catalog.csv").write_text("a,b\n")
    first = bundle_reference(tmp_path)
    (tmp_path / "catalog.csv").write_text("a,c\n")
    assert bundle_reference(tmp_path)["id"] != first["id"]
    assert "weights.dat" in first["files"]


def test_scenario_inputs_csv_roundtrips_full_water_units_and_structure():
    saved = snapshot()
    payload = scenario_inputs_csv(saved)
    name, loaded = read_scenario_csv(payload)
    assert name == "Baseline"
    assert loaded["water"] == saved["inputs"]["water"]
    assert loaded["temperature_c"] is None
    row = loaded["pfas"][0]
    assert row["confirmed_smiles"] == SMILES and row["structure_confirmed"]
    assert row["concentration_value"] == 100 and row["concentration_unit"] == "ng/L"
    assert payload.decode("utf-8-sig").splitlines()[0].startswith("scenario_name,row_type,pfas,identity_type,inorganic_species,")


def test_scenario_csv_disallows_different_water_between_pfas_rows():
    payload = b"row_type,pfas,identity_type,inorganic_species,concentration,unit,ph\npfas,PFOA,abbreviation,,100,ng/L,7\npfas,PFOS,abbreviation,,100,ng/L,8\n"
    with pytest.raises(ScenarioError, match="identical"):
        read_scenario_csv(payload)


def test_misspelled_water_headers_cannot_silently_drop_settings():
    with pytest.raises(ScenarioError, match="Unrecognized.*ph_value"):
        read_scenario_csv(b"row_type,pfas,inorganic_species,concentration,unit,ph_value\npfas,PFOA,,100,ng/L,7\n")


@pytest.mark.parametrize("extra", ["ion_ca\nmaybe", "ph\nND", "temperature_c\ninf"])
def test_invalid_scenario_water_values_are_actionable(extra):
    column, value = extra.split("\n")
    payload = f"row_type,pfas,identity_type,inorganic_species,concentration,unit,{column}\npfas,PFOA,abbreviation,,100,ng/L,{value}\n".encode()
    with pytest.raises(ScenarioError):
        read_scenario_csv(payload)


def test_results_csv_is_readable_but_cannot_be_mistaken_for_scenario_inputs():
    payload = results_csv(result(), "Study")
    text = payload.decode("utf-8-sig")
    assert "limiting_logkd" in text and "candidate_caveats" in text
    assert "Exploratory prediction" in text and "Study" in text
    with pytest.raises(ScenarioError, match="results CSV"):
        read_scenario_csv(payload)


def test_formula_like_scenario_labels_are_escaped_for_excel_and_roundtrip():
    saved = snapshot("=Study")
    saved["inputs"]["pfas"][0]["label"] = "'@PFAS"
    name, loaded = read_scenario_csv(scenario_inputs_csv(saved))
    assert name == "=Study" and loaded["pfas"][0]["label"] == "'@PFAS"
