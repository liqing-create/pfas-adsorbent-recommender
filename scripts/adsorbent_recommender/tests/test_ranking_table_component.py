"""Structural checks for the shortlist's visual encodings."""
import inspect

import adsorbent_recommender.app as app_module
from adsorbent_recommender import ranking_table


def test_hero_uses_the_approved_literature_based_description():
    source = inspect.getsource(app_module.main)
    assert "Explore commercial adsorbent rankings derived from published PFAS adsorption data" in source
    assert "Shortlist adsorbents for experimental testing" not in source


def test_component_defines_family_fills_count_columns_and_selection_without_support_classes():
    section_css = ranking_table._CSS.split(".ar-ranking-section {", 1)[1].split("}", 1)[0]
    assert all(token in section_css for token in ("border:", "border-radius:", "padding:"))
    assert all(token in ranking_table._CSS for token in (
        "ar-family-ac", "ar-family-resin", "ar-family-cdp",
        "ar-top-tier",
    ))
    assert all(token in ranking_table._JS for token in (
        "setStateValue", 'row.family', 'row.studies', 'row.records', 'selected_key',
        '"Adsorbent studies"', '"Adsorbent records"', 'with any PFAS',
        'make("details"', 'make("summary"', 'open_sections', 'openSections',
    ))
    # The moderate/thin classification is not shown in any form.
    assert all(token not in ranking_table._CSS + ranking_table._JS for token in (
        "ar-support", "row.support", "supported", "repeating-linear-gradient", '"DOI studies"'))
    assert '"Rank"' not in ranking_table._JS
    assert 'row.rank' not in ranking_table._JS
    # Score first (it orders the rows), then basis and warnings, then evidence counts.
    js = ranking_table._JS
    headers = js.index('"Product", data.scoreLabel')
    assert headers < js.index('"Basis", "Warnings"', headers) < js.index('"Adsorbent studies", "Adsorbent records"', headers)
    cells = [js.index(f"addCell(tr, row.{name}") for name in ("product", "basis", "studies", "records")]
    assert cells == sorted(cells) and js.index("tr.append(warningCell)") < cells[2]


def test_shortlist_keeps_score_order_and_splits_tier_with_family_and_support_metadata():
    items = [
        {
            "rank": rank,
            "ranking_score": score,
            "product": {
                "Product_key": key,
                "Name_Commercial": name,
                "adsorbent_category": category,
            },
            "adsorbent_support": {
                "independent_studies": studies,
                "meets_default_ml_policy": moderate,
            },
            "performance": {"mean_logKd": score},
            "performance_source": "model_prediction",
            "caveats": [],
        }
        for rank, score, key, name, category, studies, moderate in [
            (1, 2.80, "ac", "Carbon", "Activated carbon", 3, True),
            (2, 2.55, "resin", "Resin", "Ion exchange resin", 1, False),
            (3, 2.10, "cdp", "CDP", "Cyclodextrin polymer", 2, True),
        ]
    ]
    result = {
        "mixture": {"pfas_count": 1},
        "rank_resolution": {"available": True, "threshold_log_units": .51},
    }
    groups = [
        {"key": "AC", "categories": ["Activated carbon"]},
        {"key": "Resin", "categories": ["Ion exchange resin", "Nonionic resin"]},
        {"key": "CDP", "categories": ["Cyclodextrin polymer"]},
    ]

    sections, score_label, is_mixture = app_module._shortlist_sections(result, items, groups)

    assert [section["kind"] for section in sections] == ["top", "remaining"]
    assert "caption" in sections[0] and "caption" not in sections[1]
    assert [[row["score"] for row in section["rows"]] for section in sections] == [[2.80, 2.55], [2.10]]
    assert [row["family"] for section in sections for row in section["rows"]] == ["AC", "Resin", "CDP"]
    assert all("support" not in row for section in sections for row in section["rows"])
    assert [row["studies"] for section in sections for row in section["rows"]] == [3, 1, 2]
    assert score_label == "log Kd" and not is_mixture


def test_result_controls_wrap_the_ranking_and_default_section_state_is_compact(monkeypatch):
    source = inspect.getsource(app_module._render_result)
    assert source.index('actions("top")') < source.index("_filter_result_items")
    instruction = "Select any adsorbent row to view its evidence and detailed information."
    assert source.index(instruction) < source.index("render_ranking_table") < source.index('actions("bottom")')
    assert 'st.caption("Select a row to see its evidence.")' not in source
    assert "adsorbents ranked" not in source and "ar-headline" not in source

    captured = {}

    def mount(**kwargs):
        captured.update(kwargs)
        return kwargs["default"]

    monkeypatch.setattr(ranking_table, "_ranking_table_component", lambda: mount)
    monkeypatch.setattr(ranking_table.st, "session_state", {})
    ranking_table.render_ranking_table(
        sections=[
            {"kind": "top", "rows": []},
            {"kind": "remaining", "rows": []},
        ],
        score_label="log Kd",
        is_mixture=False,
        selected_key=None,
        key="test_shortlist",
    )
    assert captured["data"]["openSections"] == {"top": True, "remaining": False}
