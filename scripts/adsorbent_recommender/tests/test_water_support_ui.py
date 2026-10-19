from __future__ import annotations

import copy
import sys
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from adsorbent_recommender.service import load_recommender_bundle, recommend_mixture
from adsorbent_recommender.tests.test_water_support import DIRECTORY, ROOT


@pytest.fixture(scope="module")
def runs():
    if not DIRECTORY.is_dir():
        pytest.skip("Deployment bundle unavailable")
    bundle = copy.copy(load_recommender_bundle(DIRECTORY))
    bundle.catalog = bundle.catalog[bundle.catalog.Product_key.eq("FILTRASORB400")].copy()
    for model in bundle.models:
        model.set_params(model__n_jobs=1)
    results = {}
    for water in ("groundwater", "tap water", "ultrapure water", "synthetic water"):
        request = {"pfas": {"identity": "PFOA", "identity_type": "abbreviation"},
            "initial_concentration_mg_l": .0001, "water": {"water_type": water}}
        results[water] = recommend_mixture(bundle, [request])
    return results


@pytest.mark.parametrize("water,expected", [("groundwater", True), ("tap water", True),
    ("ultrapure water", False), ("synthetic water", False)])
def test_only_selected_weak_water_has_one_banner_and_optional_details(runs, water, expected):
    app = AppTest.from_file(str(ROOT / "scripts/adsorbent_recommender/app.py"), default_timeout=90)
    app.session_state["ar_result"] = runs[water]
    app.run()
    assert not app.exception
    messages = [item.value for item in app.warning if "Limited water-matrix evidence" in item.value]
    assert len(messages) == int(expected)
    assert any(expander.label == "Water-matrix support details" for expander in app.expander) is expected
    if expected:
        assert water in messages[0]


def test_database_only_and_filtered_database_views_do_not_show_model_banner(runs):
    result = copy.deepcopy(runs["groundwater"])
    for candidate in result["recommendations"]:
        candidate["performance_source"] = "database_match"
        for row in candidate.get("pfas_results", []):
            row["performance_source"] = "database_match"
    app = AppTest.from_file(str(ROOT / "scripts/adsorbent_recommender/app.py"), default_timeout=90)
    app.session_state["ar_result"] = result
    app.run()
    assert not app.exception
    assert not any("Limited water-matrix evidence" in item.value for item in app.warning)
