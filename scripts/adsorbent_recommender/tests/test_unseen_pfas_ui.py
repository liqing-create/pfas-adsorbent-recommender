"""Exercise the deployed Streamlit widgets and session state without a browser server."""
from __future__ import annotations
from pathlib import Path
from unittest.mock import patch
import pytest
from streamlit.testing.v1 import AppTest
from adsorbent_recommender.pfas_identity import IdentityLookupError

ROOT = Path(__file__).resolve().parents[3]
APP = ROOT / "scripts/adsorbent_recommender/app.py"
NOVEL = "O=C(O)" + "C(F)(F)" * 24 + "F"

@pytest.fixture
def app():
    if not (ROOT / "output/ML_logKd/Global_xgb_combination_42_20261014_184656/model_ensemble.joblib").is_file():
        pytest.skip("The deployment bundle is unavailable")
    app = AppTest.from_file(str(APP), default_timeout=60).run()
    assert not app.exception
    return app

def custom(app, text, kind):
    app.number_input(key="ar_pfas_add_c0_v2").set_value(.1).run()
    # AppTest 1.59 cannot serialize accept_new_options values unless its local
    # options list includes the value that the real browser sends as string_value.
    search = app.selectbox(key="ar_pfas_search_v2")
    if text not in search.options:
        search.options.append(text)
    search.select(text)
    app.session_state["ar_pfas_search_v2"] = text
    app.session_state["ar_custom_identity_type"] = kind
    app.run()
    if text not in app.selectbox(key="ar_pfas_search_v2").options:
        app.selectbox(key="ar_pfas_search_v2").options.append(text)
    return app

def test_custom_smiles_confirmation_and_duplicates_preserve_the_mixture(app):
    custom(app, NOVEL, "smiles")
    assert not app.exception
    assert any("Structure that will be used" in c.value for c in app.caption)
    add = next(button for button in app.button if button.label == "Confirm structure and add PFAS")
    assert not add.disabled
    add.click().run()
    assert not app.exception
    entries = app.session_state["ar_pfas_entries_v2"]
    assert len(entries) == 1 and entries[0]["structure_confirmed"]
    custom(app, NOVEL, "smiles")
    next(button for button in app.button if button.label == "Confirm structure and add PFAS").click().run()
    assert len(app.session_state["ar_pfas_entries_v2"]) == 1
    assert any("already in the mixture" in warning.value for warning in app.warning)

def test_invalid_structure_cannot_be_added_and_existing_entries_are_kept(app):
    custom(app, NOVEL, "smiles")
    next(button for button in app.button if button.label == "Confirm structure and add PFAS").click().run()
    custom(app, "FC(F)(F)C(=O)[O-].[Na+]", "smiles")
    assert not app.exception
    assert next(button for button in app.button if button.label == "Confirm structure and add PFAS").disabled
    assert len(app.session_state["ar_pfas_entries_v2"]) == 1
    assert any("parent structure" in warning.value for warning in app.warning)

def test_external_identity_is_not_confirmed_automatically(app):
    custom(app, "unknown PFAS name", "name")
    candidates = [{"cid": 123, "title": "Example structure", "smiles": NOVEL,
        "url": "https://pubchem.ncbi.nlm.nih.gov/compound/123", "source": "pubchem", "query": "unknown PFAS name", "identity_type": "name"}]
    with patch("adsorbent_recommender.pfas_identity.resolve_public_identity", return_value=candidates):
        app.button(key="ar_resolve_identity").click().run()
    assert not app.exception
    assert next(button for button in app.button if button.label == "Confirm structure and add PFAS").disabled
    app.selectbox(key="ar_pfas_search_v2").options.append("unknown PFAS name")
    selection = next(widget for widget in app.selectbox if widget.label == "Select the intended compound")
    selection.select(0).run()
    assert not app.exception
    assert not next(button for button in app.button if button.label == "Confirm structure and add PFAS").disabled
    assert "ar_pfas_entries_v2" not in app.session_state or not app.session_state["ar_pfas_entries_v2"]

def test_generated_structure_can_be_screened_and_its_support_is_visible(app):
    custom(app, NOVEL, "smiles")
    next(button for button in app.button if button.label == "Confirm structure and add PFAS").click().run()
    app.button(key="ar_screen").click().run(timeout=90)
    assert not app.exception
    assert app.session_state["ar_result"]["status"] == "recommendations_ready"
    assert any("Exploratory prediction" in warning.value for warning in app.warning)
    tables = [element.value for element in app.dataframe]
    support = next(frame for frame in tables if "Model training" in frame)
    assert support.iloc[0]["Features"] == "Computed at runtime"
    assert support.iloc[0]["Model training"] == "Unseen · exploratory"

def test_public_lookup_failure_retains_the_mixture_and_offers_smiles(app):
    custom(app, NOVEL, "smiles")
    next(button for button in app.button if button.label == "Confirm structure and add PFAS").click().run()
    custom(app, "temporarily unavailable identity", "name")
    with patch("adsorbent_recommender.pfas_identity.resolve_public_identity",
               side_effect=IdentityLookupError("PubChem unavailable. Paste SMILES or retry.")):
        app.button(key="ar_resolve_identity").click().run()
    assert not app.exception
    assert len(app.session_state["ar_pfas_entries_v2"]) == 1
    assert any("Paste SMILES" in warning.value for warning in app.warning)
