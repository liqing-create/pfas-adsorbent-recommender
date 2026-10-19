from copy import deepcopy

from adsorbent_recommender.service import _candidate_caveats
from adsorbent_recommender.warning_labels import (
    SEVERITIES, column_options, legend_labels, warning_labels, warning_values,
)


def caveat(code, severity="info", text="Explanation", label="Recorded label", **kwargs):
    return dict(code=code, severity=severity, text=text, label=label, **kwargs)


def test_database_caveats_show_all_recorded_limits_and_preserve_severity():
    item = {"caveats": _candidate_caveats("database_match", {
        "study_count": 1, "match_count": 3, "exact_reference_dose_match_count": 0,
        "matched_logKd_min": 1., "matched_logKd_max": 2.5,
    }, {"mean_logKd": 5.}, {"seen_in_training": False}, [], [])}
    assert warning_values(item) == ["Critical: Conflicting data", "Context: Near dose", "Context: One study"]
    assert [tag.severity for tag in warning_labels(item)] == ["critical", "info", "info"]
    assert not any("New adsorbent" in tag.value for tag in warning_labels(item))


def test_mixture_labels_deduplicate_without_losing_full_text_or_pfas_scope():
    item = {"caveats": [
        caveat("study_dominated_adsorbent", text="88% from study A", pfas=["PFOA"]),
        caveat("study_dominated_adsorbent", text="91% from study A", pfas=["PFOS"]),
        caveat("database_spread", "critical", pfas=["PFOS"]),
        caveat("limited_water_evidence", pfas=["PFOA"]),
    ]}
    before = deepcopy(item)
    assert warning_values(item) == ["Critical: Conflicting data", "Context: Limited matrix", "Context: Study skew"]
    assert item == before
    assert len(item["caveats"]) == 4


def test_pfas_level_alerts_are_not_hidden_from_candidate_warning_labels():
    item = {"caveats": [caveat("unseen_pfas", "critical"),
                         caveat("pfas_descriptors_out_of_range", "critical"),
                         caveat("limited_water_evidence")]}
    assert set(warning_values(item)) == {"Critical: New PFAS", "Critical: PFAS range", "Context: Limited matrix"}


def test_color_uses_recorded_severity_rather_than_guessing_from_code():
    item = {"caveats": [caveat("study_dominated_adsorbent", "critical")]}
    tag = warning_labels(item)[0]
    assert tag.value == "Critical: Study skew" and tag.label.startswith("!")
    options = column_options([item])
    assert options["color"] == [SEVERITIES["critical"]["fill"]]


def test_unknown_caveat_is_retained_and_unknown_severity_is_explicit():
    item = {"caveats": [caveat("future_code", "future_severity", text="New limitation", label="Future condition")]}
    tag = warning_labels(item)[0]
    assert tag.value == "Review: Future condition"
    assert tag.meaning == "New limitation" and tag.severity == "review"


def test_empty_warnings_leave_cells_and_legend_empty():
    assert warning_labels({}) == []
    assert warning_values({"caveats": []}) == []
    assert legend_labels([{}, {"caveats": []}]) == []
    assert column_options([{}])["options"] == []
    assert column_options([{}])["color"] == []


def test_filtering_and_candidate_order_do_not_change_color_or_legend_meaning():
    first = {"caveats": [caveat("study_dominated_adsorbent")]}
    second = {"caveats": [caveat("database_spread", "critical"), caveat("database_one_study")]}
    def colors(items):
        options = column_options(items)
        return dict(zip(options["options"], options["color"]))
    assert colors([first, second]) == colors([second, first])
    assert colors([first])["Context: Study skew"] == colors([first, second])["Context: Study skew"]
    assert len(legend_labels([first, first])) == 1


def test_many_warning_labels_reserve_space_instead_of_a_hidden_count():
    one = {"caveats": [caveat("study_dominated_adsorbent")]}
    many = {"caveats": [caveat(code, "critical") for code in (
        "unseen_material", "exploratory_adsorbent", "property_out_of_range", "extreme_logkd", "unseen_pfas")]}
    assert len(warning_values(many)) == 5
    assert column_options([many])["width"] > column_options([one])["width"]
    assert len(column_options([many])["options"]) == 5
