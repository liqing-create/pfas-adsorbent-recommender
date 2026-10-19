"""Structural checks for the shortlist's visual encodings."""
import adsorbent_recommender.app as app_module
from adsorbent_recommender import ranking_table


def test_component_defines_family_fills_support_hatching_and_selection():
    assert all(token in ranking_table._CSS for token in (
        "ar-family-ac", "ar-family-resin", "ar-family-cdp",
        "ar-support-moderate", "ar-support-thin", "repeating-linear-gradient",
        "ar-top-tier",
    ))
    assert all(token in ranking_table._JS for token in (
        '"Rank"', "setStateValue", 'row.rank', 'row.family', 'row.support',
        'selected_key',
    ))


def test_shortlist_keeps_ranks_and_splits_tier_with_family_and_support_metadata():
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
    assert [[row["rank"] for row in section["rows"]] for section in sections] == [[1, 2], [3]]
    assert [row["family"] for section in sections for row in section["rows"]] == ["AC", "Resin", "CDP"]
    assert [row["support"] for section in sections for row in section["rows"]] == ["moderate", "thin", "moderate"]
    assert score_label == "log Kd" and not is_mixture
