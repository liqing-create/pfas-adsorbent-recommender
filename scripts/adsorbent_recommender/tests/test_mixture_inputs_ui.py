from __future__ import annotations
import io
from pathlib import Path
from unittest.mock import patch

import pytest
from streamlit.testing.v1 import AppTest
from adsorbent_recommender.mixture_inputs import EXAMPLE_CSV_PATH

ROOT = Path(__file__).resolve().parents[3]
APP = ROOT / "scripts/adsorbent_recommender/app.py"

@pytest.fixture
def app():
    if not (ROOT / "output/ML_logKd/Global_xgb_combination_42_20261014_184656/model_ensemble.joblib").exists():
        pytest.skip("Deployed model bundle is unavailable")
    return AppTest.from_file(str(APP), default_timeout=60)

def test_add_and_edit_unit_changes_do_not_change_physical_concentration(app):
    app.run()
    assert not app.exception
    app.number_input(key="ar_pfas_add_c0_v2").set_value(.0001).run()
    app.selectbox(key="ar_pfas_add_unit_v1").select("ng/L").run()
    assert app.number_input(key="ar_pfas_add_c0_v2").value == 100
    choice = next(choice for choice in app.selectbox(key="ar_pfas_search_v2").options if choice.startswith("PFOA ·"))
    # One searchable option includes the abbreviation, full name, and CAS.
    app.selectbox(key="ar_pfas_search_v2").select_index(app.selectbox(key="ar_pfas_search_v2").options.index(choice)).run()
    next(button for button in app.button if button.label == "Add PFAS").click().run()
    assert not app.exception
    entry = app.session_state["ar_pfas_entries_v2"][0]
    assert entry["initial_concentration_mg_l"] == .0001
    app.selectbox(key=f"ar_pfas_unit_{entry['id']}").select("µg/L").run()
    assert not app.exception
    assert app.number_input(key=f"ar_pfas_c0_{entry['id']}").value == .1
    assert app.session_state["ar_pfas_entries_v2"][0]["initial_concentration_mg_l"] == .0001

def test_example_csv_has_preview_and_import_preserves_source_units(app):
    with patch("streamlit.file_uploader", return_value=io.BytesIO(EXAMPLE_CSV_PATH.read_bytes())):
        app.run()
        assert not app.exception
        preview = next(table.value for table in app.dataframe if "CSV line" in table.value)
        assert len(preview) == 4 and set(preview["Status"]) == {"Ready", "Review structure"}
        columns = list(preview.columns)
        assert columns.index("Identity type") == columns.index("Submitted PFAS") + 1
        assert preview.iloc[2]["Identity type"] == "smiles"
        assert preview.iloc[2]["Resolved PFAS"] == "PFHxA"
        assert app.button(key="ar_import_csv").label == "Confirm structures and import 4 PFAS"
        assert not app.button(key="ar_import_csv").disabled
        app.button(key="ar_import_csv").click().run()
        assert not app.exception
        entries = app.session_state["ar_pfas_entries_v2"]
        assert len(entries) == 4
        assert entries[0]["concentration_value"] == 100 and entries[0]["concentration_unit"] == "ng/L"
        assert entries[1]["concentration_value"] == .25 and entries[1]["concentration_unit"] == "µg/L"
        assert entries[0]["initial_concentration_mg_l"] == .0001
        assert entries[2]["identity_type"] == "smiles" and entries[2]["confirmed_smiles"] == preview.iloc[2]["Submitted PFAS"]
        assert any("Imported 4 PFAS" in element.value for element in app.success)

def test_invalid_csv_keeps_all_rows_in_preview_and_cannot_partially_import(app):
    payload = b"pfas,concentration,unit\nPFOA,100,ng/L\nPFOS,ND,ng/L\n"
    with patch("streamlit.file_uploader", return_value=io.BytesIO(payload)):
        app.run()
        assert not app.exception
        preview = next(table.value for table in app.dataframe if "CSV line" in table.value)
        assert len(preview) == 2 and "ND" in preview.iloc[1]["Status"]
        assert app.button(key="ar_import_csv").disabled
        assert "ar_pfas_entries_v2" not in app.session_state or not app.session_state["ar_pfas_entries_v2"]

def test_csv_smiles_are_previewed_before_explicit_confirmation(app):
    novel = "O=C(O)" + "C(F)(F)" * 24 + "F"
    payload = f"pfas,identity_type,concentration,unit,parent_smiles\nResearch PFAS,name,50,ng/L,{novel}\n".encode()
    with patch("streamlit.file_uploader", return_value=io.BytesIO(payload)):
        app.run()
        assert not app.exception
        assert app.button(key="ar_import_csv").label == "Confirm structures and import 1 PFAS"
        assert "ar_pfas_entries_v2" not in app.session_state or not app.session_state["ar_pfas_entries_v2"]
        app.button(key="ar_import_csv").click().run()
        assert not app.exception
        entry = app.session_state["ar_pfas_entries_v2"][0]
        assert entry["structure_confirmed"] and entry["confirmed_smiles"] == novel

def test_clearing_a_concentration_shows_validation_without_losing_the_mixture(app):
    with patch("streamlit.file_uploader", return_value=io.BytesIO(EXAMPLE_CSV_PATH.read_bytes())):
        app.run()
        app.button(key="ar_import_csv").click().run()
        first = app.session_state["ar_pfas_entries_v2"][0]
        app.number_input(key=f"ar_pfas_c0_{first['id']}").set_value(None).run()
        assert not app.exception
        assert app.button(key="ar_screen").disabled
        assert not app.exception
        assert len(app.session_state["ar_pfas_entries_v2"]) == 4
        assert any("concentration greater than zero" in error.value for error in app.error)
