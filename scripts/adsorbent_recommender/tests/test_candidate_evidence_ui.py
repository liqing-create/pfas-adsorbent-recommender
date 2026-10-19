from __future__ import annotations

from copy import copy
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from adsorbent_recommender.service import load_recommender_bundle, recommend_mixture

ROOT = Path(__file__).resolve().parents[3]
APP = ROOT / "scripts/adsorbent_recommender/app.py"
DIRECTORY = ROOT / "output/ML_logKd/Global_xgb_combination_42_20261014_184656"


@pytest.fixture(scope="module")
def runs():
    if not (DIRECTORY / "model_ensemble.joblib").is_file():
        pytest.skip("The deployment bundle is unavailable")
    bundle = copy(load_recommender_bundle(DIRECTORY))
    bundle.catalog = bundle.catalog[bundle.catalog.Product_key.eq("FILTRASORB400")].copy()
    for model in bundle.models:
        model.set_params(model__n_jobs=1)
    results = {}
    for label, c0 in (("related", .0001), ("comparable", 50)):
        scenario = {"pfas": {"identity": "PFOA", "identity_type": "abbreviation"},
                    "initial_concentration_mg_l": c0,
                    # The zero-carbon comparison records were measured at 20–21 °C.
                    "temperature_c": 20 if label == "comparable" else None,
                    "water": {"water_type": "ultrapure water", "organic_carbon_mg_l": 0}}
        results[label] = recommend_mixture(bundle, [scenario], top_k=1)
    return results


def app_with_result(result):
    app = AppTest.from_file(str(APP), default_timeout=90)
    app.session_state["ar_result"] = result
    return app.run()


def test_related_studies_are_inside_candidate_details_with_no_new_ranking_columns(runs):
    app = app_with_result(runs["related"])
    assert not app.exception
    item = runs["related"]["recommendations"][0]
    assert item["performance_source"] == "model_prediction"
    related = item["database_evidence"]["related_evidence"]
    # Include same-PFAS/product records from other water matrices as context.
    assert related["record_count"] >= 13
    candidate = next(expander for expander in app.expander if expander.label.startswith("#1 · Filtrasorb"))
    assert any(f"{related['record_count']} related record(s)" in caption.value for caption in candidate.caption)
    assert any("No comparable experimental records" in caption.value for caption in candidate.caption)
    assert candidate.get("popover")
    ranking = next(table.value for table in app.dataframe if "Overall rank" in table.value)
    assert not any("related" in column.casefold() for column in ranking.columns)
    assert not any(expander.label == "Explore database evidence" for expander in app.expander)
    rows = next(table.value for table in candidate.dataframe if "Condition differences" in table.value)
    assert len(rows) <= 3 and all("Initial PFAS concentration" in value for value in rows["Condition differences"])
    assert app.get("link_button")


def test_zero_chemistry_retains_comparable_database_score_and_labels_it_separately(runs):
    result = runs["comparable"]
    item = result["recommendations"][0]
    assert item["performance_source"] == "database_match"
    assert item["database_evidence"]["performance"]["match_count"] > 0
    app = app_with_result(result)
    assert not app.exception
    assert any("Comparable evidence used for ranking" in caption.value for caption in app.caption)
    assert not any("No comparable experimental records" in caption.value for caption in app.caption)


def test_water_default_is_disclosed_in_existing_selector():
    app = AppTest.from_file(str(APP), default_timeout=90).run()
    assert not app.exception
    assert "Ultrapure water (default)" in app.selectbox(key="water_type").options
    assert app.selectbox(key="water_type").value == ""
