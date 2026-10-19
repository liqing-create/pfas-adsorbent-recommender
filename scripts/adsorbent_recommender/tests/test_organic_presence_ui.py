from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from adsorbent_recommender.tests.test_scenarios_ui import app, add_pfoa, save
from adsorbent_recommender.scenarios import read_scenario_csv, scenario_inputs_csv


def test_organic_section_has_carbon_and_presence_only_and_preserves_them_on_reload(app):
    assert not any(element.key == "organic" or element.label == "Organic additives" for element in app.text_input)
    assert app.checkbox(key="organic_present").label == "Organic additives present"
    assert not app.checkbox(key="organic_present").value
    assert not any("compound list" in str(element.value).casefold() for element in app.caption)
    add_pfoa(app)
    app.text_input(key="organic_carbon").set_value("3")
    app.checkbox(key="organic_present").check().run()
    saved = save(app, "Organic addition")
    assert saved["inputs"]["water"]["organic_matter_present"] is True
    _, exported = read_scenario_csv(scenario_inputs_csv(saved))
    assert exported["water"]["organic_matter_present"] is True
    app.checkbox(key="organic_present").uncheck().run()
    app.text_input(key="organic_carbon").set_value("8").run()
    app.button(key="ar_load_scenario").click().run()
    assert not app.exception
    assert app.checkbox(key="organic_present").value
    assert float(app.text_input(key="organic_carbon").value) == 3
    assert app.selectbox(key="ar_salt_species")
