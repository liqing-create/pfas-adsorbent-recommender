from copy import deepcopy

from streamlit.testing.v1 import AppTest

from adsorbent_recommender.tests.test_candidate_evidence_ui import APP, runs
import adsorbent_recommender.app as app_module
from adsorbent_recommender.warning_labels import warning_labels, warning_values


def test_one_column_shows_both_severities_and_a_visible_meaning_legend(runs):
    result = deepcopy(runs["related"])
    item = result["recommendations"][0]
    item["caveats"] = [
        {"code": "database_spread", "severity": "critical", "label": "Records disagree", "text": "Recorded conflict explanation.", "pfas": ["PFOA"]},
        {"code": "study_dominated_adsorbent", "severity": "info", "label": "Training evidence concentrated", "text": "88% comes from one study.", "pfas": []},
        {"code": "limited_water_evidence", "severity": "info", "label": "Limited water-matrix evidence", "text": "Recorded matrix limitation.", "pfas": ["PFOA"]},
    ]
    original = deepcopy(result)
    app = AppTest.from_file(str(APP), default_timeout=90)
    app.session_state["ar_result"] = result
    app.run()
    assert not app.exception
    ranking_tables = [table.value for table in app.dataframe if "Warnings" in table.value]
    assert len(ranking_tables) == 1
    for table in ranking_tables:
        assert list(table.iloc[0]["Warnings"]) == warning_values(item)
        assert "Support notes" not in table and "Caveats" not in table
    sections, _, _ = app_module._shortlist_sections(result, result["recommendations"], [])
    first_row = sections[0]["rows"][0]
    assert first_row["warnings"] == [
        {"label": tag.label, "severity": tag.severity, "meaning": tag.meaning}
        for tag in warning_labels(item)
    ]
    assert any(element.value == "**Warning label legend**" for element in app.markdown)
    html = "\n".join(str(element.proto) for element in app.get("html"))
    for text in ("Conflicting data", "Study skew", "Limited matrix", "At least 80%", "Recorded conflict explanation.", "Recorded matrix limitation."):
        assert text in html
    assert "--warning-fill:rgba(" in html
    assert any("Rose ! = critical alert" in caption.value and "Teal i = supporting context" in caption.value for caption in app.caption)
    assert result == original


def test_no_recorded_caveats_leave_cells_blank_without_a_legend_or_reminder(runs):
    result = deepcopy(runs["related"])
    result["recommendations"][0]["caveats"] = []
    app = AppTest.from_file(str(APP), default_timeout=90)
    app.session_state["ar_result"] = result
    app.run()
    assert not app.exception
    sections, _, _ = app_module._shortlist_sections(result, result["recommendations"], [])
    assert sections[0]["rows"][0]["warnings"] == []
    html = "\n".join(str(element.proto) for element in app.get("html"))
    assert "class='ar-warning-label'" not in html and "None flagged" not in html
    assert not any(element.value == "**Warning label legend**" for element in app.markdown)
