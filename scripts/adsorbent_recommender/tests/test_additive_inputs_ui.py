from __future__ import annotations

from adsorbent_recommender.tests.test_scenarios_ui import app, add_pfoa, save
from adsorbent_recommender.scenarios import read_scenario_csv, scenario_inputs_csv


def add_salt(app, species, concentration=1, unit="mM"):
    app.selectbox(key="ar_salt_species").select(species)
    app.number_input(key="ar_salt_concentration").set_value(concentration)
    app.selectbox(key="ar_salt_unit").select(unit).run()
    app.button(key="ar_add_inorganic").click().run()
    assert not app.exception


def test_multiple_salts_add_remove_and_duplicate_errors_preserve_inputs(app):
    assert not any(element.key == "inorganic" for element in app.text_input)
    assert app.button(key="ar_add_inorganic").disabled
    add_salt(app, "NaCl")
    add_salt(app, "CaCl2", 2)
    expected = [{"species": "NaCl", "concentration": 1.0, "unit": "mM"},
                {"species": "CaCl2", "concentration": 2.0, "unit": "mM"}]
    assert app.session_state["ar_inorganic_entries"] == expected
    assert any("0.007" in str(element.value) for element in app.caption)
    add_salt(app, "NaCl", 3)
    assert "duplicates" in app.session_state["ar_salt_error"]
    assert app.session_state["ar_inorganic_entries"] == expected
    app.button(key="ar_remove_inorganic_0").click().run()
    assert not app.exception
    assert app.session_state["ar_inorganic_entries"] == expected[1:]
    assert "ar_salt_error" not in app.session_state
    app.text_input(key="ph").set_value("7.5").run()
    assert app.session_state["ar_inorganic_entries"] == expected[1:]


def test_salt_units_and_entries_survive_scenario_reload_and_csv_export(app):
    add_pfoa(app)
    add_salt(app, "NaCl", 58.44, "mg/L")
    first = save(app, "With salt")
    _, exported = read_scenario_csv(scenario_inputs_csv(first))
    assert exported["water"]["inorganic_additives"] == first["inputs"]["water"]["inorganic_additives"]
    app.button(key="ar_remove_inorganic_0").click().run()
    assert app.session_state["ar_inorganic_entries"] == []
    app.button(key="ar_load_scenario").click().run()
    assert not app.exception
    assert app.session_state["ar_inorganic_entries"] == first["inputs"]["water"]["inorganic_additives"]
