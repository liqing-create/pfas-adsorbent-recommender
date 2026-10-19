from __future__ import annotations

import io
from pathlib import Path
from unittest.mock import patch

import pytest
from streamlit.testing.v1 import AppTest

from adsorbent_recommender.scenarios import SCENARIO_EXAMPLE_PATH, read_scenario_csv, snapshot_bytes
from adsorbent_recommender.mixture_inputs import EXAMPLE_CSV_PATH
from adsorbent_recommender.tests.test_scenarios import snapshot

ROOT = Path(__file__).resolve().parents[3]
APP = ROOT / "scripts/adsorbent_recommender/app.py"


@pytest.fixture
def app():
    if not (ROOT / "output/ML_logKd/Global_xgb_combination_42_20261014_184656/model_ensemble.joblib").exists():
        pytest.skip("The deployment bundle is unavailable")
    app = AppTest.from_file(str(APP), default_timeout=90).run()
    assert not app.exception
    return app


def add_pfoa(app):
    search = app.selectbox(key="ar_pfas_search_v2")
    search.select_index(next(index for index, label in enumerate(search.options) if label.startswith("PFOA ·"))).run()
    app.number_input(key="ar_pfas_add_c0_v2").set_value(.1).run()
    next(button for button in app.button if button.label == "Add PFAS").click().run()
    assert not app.exception
    return app.session_state["ar_pfas_entries_v2"][0]["id"]


def screen(app):
    app.button(key="ar_screen").click().run()
    assert not app.exception
    assert app.session_state["ar_result"]["status"] == "recommendations_ready"


def save(app, name):
    app.text_input(key="ar_scenario_name").set_value(name).run()
    app.button(key="ar_save_scenario").click().run()
    assert not app.exception
    assert "ar_scenario_error" not in app.session_state
    return app.session_state["ar_saved_scenarios"][-1]


def test_save_reload_preserves_inputs_and_excludes_stale_results(app):
    entry_id = add_pfoa(app)
    app.selectbox(key=f"ar_pfas_unit_{entry_id}").select("ng/L").run()
    app.number_input(key=f"ar_pfas_c0_{entry_id}").set_value(100)
    app.text_input(key="ph").set_value("7.2")
    app.text_input(key="organic_carbon").set_value("3")
    app.checkbox(key="organic_present").check()
    app.selectbox(key="ar_salt_species").select("CaCl2")
    app.number_input(key="ar_salt_concentration").set_value(2)
    app.checkbox(key="ion_Ca").set_value(True)
    app.selectbox(key="water_type").select("groundwater").run()
    app.button(key="ar_add_inorganic").click().run()
    screen(app)
    first = save(app, "Baseline")
    assert first["result"] and first["inputs"]["pfas"][0]["concentration_unit"] == "ng/L"
    assert first["inputs"]["pfas"][0]["canonical_smiles"]
    app.number_input(key=f"ar_pfas_c0_{entry_id}").set_value(500).run()
    app.text_input(key="ph").set_value("8").run()
    second = save(app, "Edited draft")
    assert second["result"] is None
    app.selectbox(key="ar_scenario_selected").select(first["scenario_id"]).run()
    app.button(key="ar_load_scenario").click().run()
    assert not app.exception
    assert "ar_result" not in app.session_state
    restored = app.session_state["ar_pfas_entries_v2"][0]
    assert restored["label"] == "PFOA" and restored["concentration_value"] == 100
    assert restored["concentration_unit"] == "ng/L" and restored["confirmed_smiles"]
    assert app.number_input(key="ar_pfas_c0_1").value == 100
    assert float(app.text_input(key="ph").value) == 7.2
    assert app.checkbox(key="organic_present").value
    assert app.session_state["ar_inorganic_entries"] == [{"species": "CaCl2", "concentration": 2.0, "unit": "mM"}]
    assert app.checkbox(key="ion_Ca").value
    assert app.text_input(key="temperature").value == ""
    screen(app)
    replay = save(app, "Replay")
    old = {row["product"]["Product_key"]: row["ranking_score"] for row in first["result"]["recommendations"]}
    new = {row["product"]["Product_key"]: row["ranking_score"] for row in replay["result"]["recommendations"]}
    assert new == old


def test_comparison_shows_inputs_scores_evidence_and_unseen_warning(app):
    first, second = snapshot(), snapshot("Higher pH")
    second["result"]["recommendations"][0]["ranking_score"] = 4.5
    app.session_state["ar_saved_scenarios"] = [first, second]
    app.run()
    app.multiselect(key="ar_scenario_compare").set_value([first["scenario_id"], second["scenario_id"]]).run()
    assert not app.exception
    frame = next(table.value for table in app.dataframe if "Product key" in table.value)
    assert frame.iloc[0]["Higher pH · ΔlogKd"] == pytest.approx(.3)
    assert frame.iloc[1]["Baseline · basis"] == "Database"
    assert any("not represented in model training" in element.value for element in app.warning)
    assert any("Input / result" in table.value for table in app.dataframe)
    assert any(element.key == "ar_comparison_download" for element in app.get("download_button"))


@pytest.mark.parametrize("valid", [False, True])
def test_json_import_is_atomic_and_reloads_confirmed_structure(app, valid):
    add_pfoa(app)
    before = app.session_state["ar_pfas_entries_v2"]
    payload = snapshot_bytes(snapshot("Imported study")) if valid else b'{"format": "wrong"}'
    def upload(label, **kwargs):
        return io.BytesIO(payload) if label == "Reload a scenario JSON" else None
    with patch("streamlit.file_uploader", side_effect=upload):
        app.run()
        app.button(key="ar_import_scenario").click().run()
        assert not app.exception
        if valid:
            row = app.session_state["ar_pfas_entries_v2"][0]
            assert row["identity"] == "Research PFAS" and row["confirmed_smiles"]
            assert row["concentration_value"] == 100 and row["concentration_unit"] == "ng/L"
            assert "ar_result" not in app.session_state
            assert len(app.session_state["ar_saved_scenarios"]) == 1
            assert app.session_state["ar_saved_scenarios"][0]["_imported"]
        else:
            assert app.session_state["ar_pfas_entries_v2"] == before
            assert not app.session_state["ar_saved_scenarios"]
            assert app.session_state["ar_scenario_error"]


def test_removing_saved_scenario_does_not_remove_editor_inputs(app):
    add_pfoa(app)
    saved = save(app, "Draft")
    app.button(key="ar_remove_scenario").click().run()
    assert not app.exception
    assert app.session_state["ar_saved_scenarios"] == []
    assert app.session_state["ar_pfas_entries_v2"][0]["label"] == "PFOA"


def test_scenario_library_is_private_to_each_browser_session(app):
    app.session_state["ar_saved_scenarios"] = [snapshot()]
    another = AppTest.from_file(str(APP), default_timeout=90).run()
    assert not another.exception and another.session_state["ar_saved_scenarios"] == []


def test_csv_mixture_can_be_saved_and_reloaded_with_every_original_unit(app):
    def upload(label, **kwargs):
        return io.BytesIO(EXAMPLE_CSV_PATH.read_bytes()) if label == "PFAS mixture CSV" else None
    with patch("streamlit.file_uploader", side_effect=upload):
        app.run()
        app.button(key="ar_import_csv").click().run()
    saved = save(app, "Four PFAS")
    assert len(saved["inputs"]["pfas"]) == 4
    app.number_input(key="ar_pfas_c0_1").set_value(900).run()
    app.button(key="ar_load_scenario").click().run()
    assert not app.exception
    entries = app.session_state["ar_pfas_entries_v2"]
    assert len(entries) == 4 and entries[0]["concentration_value"] == 100
    assert [row["concentration_unit"] for row in entries] == ["ng/L", "µg/L", "mg/L", "ng/L"]


def test_scenario_csv_previews_then_restores_full_inputs_after_confirmation(app):
    add_pfoa(app)
    def upload(label, **kwargs):
        return io.BytesIO(SCENARIO_EXAMPLE_PATH.read_bytes()) if label == "Reload scenario inputs CSV" else None
    with patch("streamlit.file_uploader", side_effect=upload):
        app.run()
        assert not app.exception
        assert app.button(key="ar_load_scenario_csv").disabled
        assert len(app.session_state["ar_pfas_entries_v2"]) == 1
        checkbox = next(box for box in app.checkbox if box.label == "I have checked the PFAS identities and supplied structures")
        checkbox.check().run()
        app.button(key="ar_load_scenario_csv").click().run()
        assert not app.exception
        assert len(app.session_state["ar_pfas_entries_v2"]) == 2
        assert [row["label"] for row in app.session_state["ar_pfas_entries_v2"]] == ["PFOA", "PFOS"]
        assert float(app.text_input(key="ph").value) == 7.2
        assert float(app.text_input(key="tds").value) == 500
        assert [row["species"] for row in app.session_state["ar_inorganic_entries"]] == ["NaCl", "CaCl2"]
        assert app.selectbox(key="water_type").value == "groundwater"
        assert "ar_result" not in app.session_state
