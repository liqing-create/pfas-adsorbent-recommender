from copy import deepcopy
from unittest.mock import patch

from streamlit.testing.v1 import AppTest

from adsorbent_recommender.tests.test_candidate_evidence_ui import APP, ranking_table_data, runs
import adsorbent_recommender.app as app_module
from adsorbent_recommender.warning_labels import warning_labels


def test_one_column_shows_both_severities_and_each_label_explains_itself(runs):
    result = deepcopy(runs["related"])
    item = result["recommendations"][0]
    item["caveats"] = [
        {"code": "database_spread", "severity": "critical", "label": "Records disagree", "text": "Recorded conflict explanation.", "pfas": ["PFOA"]},
        {"code": "database_near_dose", "severity": "info", "label": "Near reference dose", "text": "Observed at 22 mg/L.", "pfas": []},
        {"code": "limited_water_evidence", "severity": "info", "label": "Limited water-matrix evidence", "text": "Recorded matrix limitation.", "pfas": ["PFOA"]},
    ]
    original = deepcopy(result)
    app = AppTest.from_file(str(APP), default_timeout=90)
    app.session_state["ar_result"] = result
    app.run()
    assert not app.exception
    expected_warnings = [
        {"label": tag.label, "severity": tag.severity, "meaning": tag.meaning}
        for tag in warning_labels(item)
    ]
    rendered_row = ranking_table_data(app)["sections"][0]["rows"][0]
    assert rendered_row["warnings"] == expected_warnings
    assert "supportNotes" not in rendered_row and "caveats" not in rendered_row
    sections, _, _ = app_module._shortlist_sections(result, result["recommendations"], [])
    assert sections[0]["rows"][0]["warnings"] == expected_warnings
    # Each label carries its meaning to the table's hover text, so no legend repeats it.
    assert any("20–30 mg/L" in warning["meaning"] for warning in rendered_row["warnings"])
    assert not any(element.value == "**Warning label legend**" for element in app.markdown)
    # Selecting the row opens the full caveat text beside each label.
    selected = AppTest.from_file(str(APP), default_timeout=90)
    selected.session_state["ar_result"] = result
    with patch("adsorbent_recommender.ranking_table.render_ranking_table", return_value=item["product"]["Product_key"]):
        selected.run()
    assert not selected.exception
    html = "\n".join(str(element.proto) for element in selected.get("html"))
    for text in ("Conflicting data", "Near dose", "Limited matrix", "Recorded conflict explanation.", "Recorded matrix limitation."):
        assert text in html
    assert "--warning-fill:rgba(" in html
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
