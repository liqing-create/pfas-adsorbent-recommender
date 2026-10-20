"""Regression checks for intentional inputs, readiness, and shortlist provenance."""
import json

import adsorbent_recommender.app as app_module

from adsorbent_recommender.tests.test_scenarios_ui import app, add_pfoa, screen
from adsorbent_recommender.warning_labels import warning_values


def choose(app, abbreviation):
    search = app.selectbox(key="ar_pfas_search_v2")
    search.select_index(next(index for index, label in enumerate(search.options)
                             if label.startswith(abbreviation + " ·"))).run()


def test_empty_mixture_requires_intentional_concentration(app):
    assert app.number_input(key="ar_pfas_add_c0_v2").value is None
    assert app.button(key="ar_screen").disabled
    assert not any(button.key == "ar_save_scenario" for button in app.button)
    choose(app, "PFOA")
    assert next(button for button in app.button if button.label == "Add PFAS").disabled
    app.selectbox(key="ar_pfas_add_unit_v1").select("ng/L").run()
    assert app.number_input(key="ar_pfas_add_c0_v2").value is None
    assert app.button(key="ar_screen").disabled


def test_search_exposes_full_name_and_cas_once_per_compound(app):
    options = app.selectbox(key="ar_pfas_search_v2").options
    matches = [label for label in options if label.startswith("PFOA ·")]
    assert len(matches) == 1
    assert "Perfluorooctanoic acid" in matches[0] and "335-67-1" in matches[0]


def test_dropdown_lists_training_pfas_and_other_entries_take_structure_review(app):
    # 3-6-OPFHpA is in the shared feature workbook but not in the training data,
    # so the bundled cache no longer carries it.
    options = app.selectbox(key="ar_pfas_search_v2").options
    assert len(options) == 128
    assert not any(label.startswith("3-6-OPFHpA ·") for label in options)
    search = app.selectbox(key="ar_pfas_search_v2")
    search.options.append("151772-58-6")
    search.select("151772-58-6").run()
    assert not app.exception
    assert app.session_state["ar_pfas_search_v2"] == "151772-58-6"
    assert any("Resolve the identifier with PubChem" in caption.value for caption in app.caption)


def test_adding_a_second_pfas_keeps_unit_but_requires_a_new_value(app):
    choose(app, "PFOA")
    app.selectbox(key="ar_pfas_add_unit_v1").select("ng/L").run()
    app.number_input(key="ar_pfas_add_c0_v2").set_value(75).run()
    next(button for button in app.button if button.label == "Add PFAS").click().run()
    assert app.session_state["ar_pfas_entries_v2"][0]["initial_concentration_mg_l"] == 75e-6
    assert app.number_input(key="ar_pfas_add_c0_v2").value is None
    assert app.selectbox(key="ar_pfas_add_unit_v1").value == "ng/L"
    choose(app, "PFOS")
    assert next(button for button in app.button if button.label == "Add PFAS").disabled
    app.number_input(key="ar_pfas_add_c0_v2").set_value(125).run()
    next(button for button in app.button if button.label == "Add PFAS").click().run()
    assert not app.exception
    rows = app.session_state["ar_pfas_entries_v2"]
    assert [row["concentration_value"] for row in rows] == [75, 125]
    assert [row["concentration_unit"] for row in rows] == ["ng/L", "ng/L"]
    assert not app.button(key="ar_screen").disabled


def test_unresolved_entry_blocks_screening_and_saving_until_removed(app):
    add_pfoa(app)
    rows = app.session_state["ar_pfas_entries_v2"]
    rows[0]["identity_ready"] = False
    app.session_state["ar_pfas_entries_v2"] = rows
    app.run()
    assert app.button(key="ar_screen").disabled
    assert not any(button.key == "ar_save_scenario" for button in app.button)
    assert any("Resolve the identity" in error.value for error in app.error)
    app.button(key="ar_remove_pfas_1").click().run()
    assert not app.exception and not app.session_state["ar_pfas_entries_v2"]


def test_corrected_mixture_clears_legacy_missing_input_error(app):
    app.session_state["ar_error"] = "Add at least one PFAS identity and initial concentration before screening."
    add_pfoa(app)
    assert "ar_error" not in app.session_state
    assert not app.button(key="ar_screen").disabled


def test_backend_error_remains_for_same_inputs_and_clears_after_edit(app):
    add_pfoa(app)
    screen(app)
    result = app.session_state["ar_result"]
    app.session_state["ar_error"] = "Example screening failure"
    app.session_state["ar_error_input_fingerprint"] = app.session_state["ar_last_run_fingerprint"]
    app.run()
    assert any(error.value == "Example screening failure" for error in app.error)
    app.text_input(key="ph").set_value("7").run()
    assert "ar_error" not in app.session_state
    assert app.session_state["ar_result"] is result
    assert any("Inputs have changed" in notice.value for notice in app.info)


def test_invalid_concentration_preserves_result_and_recovers_without_model_run(app):
    entry_id = add_pfoa(app)
    screen(app)
    result = app.session_state["ar_result"]
    app.number_input(key=f"ar_pfas_c0_{entry_id}").set_value(None).run()
    assert app.button(key="ar_screen").disabled and app.button(key="ar_save_scenario").disabled
    assert any("concentration greater than zero" in error.value for error in app.error)
    assert app.session_state["ar_result"] is result
    app.number_input(key=f"ar_pfas_c0_{entry_id}").set_value(.1).run()
    assert not app.exception and not app.error
    assert not app.button(key="ar_screen").disabled
    assert app.session_state["ar_result"] is result


def test_compact_shortlist_preserves_scores_basis_and_critical_flags(app):
    add_pfoa(app)
    screen(app)
    result = app.session_state["ar_result"]
    items = result["recommendations"]
    compact = app_module._shortlist_table(result, items)
    assert len(compact.columns) == 6 and "Studies" in compact
    assert "Overall rank" in compact and "Shortlist group" not in compact and "Warnings" in compact
    assert "Support notes" not in compact and "Caveats" not in compact
    assert compact.iloc[0]["Basis"] == "ML prediction"
    top = items[0]["ranking_score"]
    threshold = result["rank_resolution"]["threshold_log_units"]
    assert len(compact) == len(items)
    for index, item in enumerate(items):
        assert compact.iloc[index]["Overall rank"] == item["rank"]
        assert compact.iloc[index]["Studies"] == item["adsorbent_support"]["independent_studies"]
        assert compact.iloc[index]["logKd at reference dose"] == item["performance"]["mean_logKd"]
        assert list(compact.iloc[index]["Warnings"]) == warning_values(item)
    sections, score_label, is_mixture = app_module._shortlist_sections(
        result, items, app_module.catalog_category_groups(app_module.load_recommender_bundle(app_module._resolve_bundle_dir()))
    )
    expected_top = sum(top - item["ranking_score"] <= threshold for item in items)
    assert sections[0]["kind"] == "top" and len(sections[0]["rows"]) == expected_top
    assert sum(len(section["rows"]) for section in sections) == len(items)
    assert [row["rank"] for section in sections for row in section["rows"]] == [item["rank"] for item in items]
    assert {row["family"] for section in sections for row in section["rows"]} <= {"AC", "Resin", "CDP", "Other"}
    # Counts describe the adsorbent with any PFAS; there is no support class.
    assert all("support" not in row for section in sections for row in section["rows"])
    assert [(row["studies"], row["records"]) for section in sections for row in section["rows"]] == [
        (item["adsorbent_support"]["independent_studies"], item["adsorbent_support"]["source_records"]) for item in items]
    assert score_label == "log Kd" and not is_mixture
    assert not any(expander.label == "Complete ranking table" for expander in app.expander)
    # PFOA is in the training data: no support panel and no unseen-PFAS warning.
    assert not any(expander.label == "Model support and scoring details" for expander in app.expander)
    assert not any("not in the model's training data" in warning.value for warning in app.warning)
    assert any(block.proto.popover.label == "Keep for comparison" for block in app.get("popover"))
    assert "Download run (.xlsx)" in {button.label for button in app.get("download_button")}
    assert not any("JSON" in button.label for button in app.get("download_button"))
    assert not any("View results" in str(element.value) for element in app.markdown)
    assert any(
        "Select any adsorbent row" in str(element.value)
        for element in app.markdown
    )
    ranking = next(
        json.loads(element.proto.json)
        for element in app.get("bidi_component")
        if element.proto.component_name == "adsorbent_recommender_ranking_table"
    )
    assert ranking["openSections"] == {"top": True, "remaining": False}


def test_edited_inputs_label_historical_export_and_stay_saveable(app):
    add_pfoa(app)
    screen(app)
    app.text_input(key="ph").set_value("8").run()
    app.text_input(key="ar_scenario_name").set_value("Displayed run").run()
    assert not app.button(key="ar_save_scenario").disabled
    assert any(button.label == "Download run (.xlsx)" for button in app.get("download_button"))
