from __future__ import annotations

import copy
import io
import json
from pathlib import Path
from unittest.mock import patch

import pytest
from streamlit.testing.v1 import AppTest

from adsorbent_recommender.tests.test_scenarios import snapshot

ROOT = Path(__file__).resolve().parents[3]


@pytest.fixture
def app():
    if not (ROOT/'output/ML_logKd/Global_xgb_combination_42_20261014_184656/model_ensemble.joblib').is_file():
        pytest.skip('The deployment bundle is unavailable')
    app = AppTest.from_file(str(ROOT/'scripts/adsorbent_recommender/app.py'), default_timeout=90).run()
    assert not app.exception
    search = app.selectbox(key='ar_pfas_search_v2')
    search.select_index(next(index for index, label in enumerate(search.options) if label.startswith('PFOA ·'))).run()
    app.number_input(key='ar_pfas_add_c0_v2').set_value(.1).run()
    next(button for button in app.button if button.label == 'Add PFAS').click().run()
    return app


def screen_button(app):
    return app.button(key='ar_screen')


def test_blank_and_explicit_default_temperature_produce_identical_rankings(app):
    app.number_input(key='ar_pfas_c0_1').set_value(15)
    app.selectbox(key='water_type').select('synthetic water').run()
    screen_button(app).click().run()
    assert not app.exception
    first = app.session_state['ar_result']
    app.text_input(key='temperature').set_value('25').run()
    screen_button(app).click().run()
    assert not app.exception
    second = app.session_state['ar_result']
    simplify = lambda run: [(row['product']['Product_key'], row['rank'], row['ranking_score'], row['performance_source'])
                             for row in run['recommendations']]
    assert simplify(first) == simplify(second)
    assert first['scenarios'][0]['temperature_c'] is None
    assert first['screening_conditions']['fixed_conditions']['Temperature_(°C)'] == 25
    assert not any('temperature' in caveat['text'] for caveat in first['scenario_caveats']
                   if caveat['code'] == 'database_conditions_unchecked')


def test_invalid_fields_block_actions_preserve_results_and_skip_chemistry_preview(app):
    app.selectbox(key='water_type').select('groundwater')
    app.selectbox(key='ar_salt_species').select('NaCl').run()
    app.button(key='ar_add_inorganic').click().run()
    screen_button(app).click().run()
    assert not app.exception
    original = app.session_state['ar_result']
    old_fingerprint = app.session_state['ar_last_run_fingerprint']
    for key, raw in [('ph', 'nan'), ('temperature', 'inf'), ('temperature', '1e309'),
                     ('ionic_strength', '-.01'), ('organic_carbon', '-1'), ('tds', '-1'), ('ph', '7,2')]:
        app.text_input(key=key).set_value(raw).run()
        assert not app.exception
        assert app.error
        assert screen_button(app).disabled and app.button(key='ar_save_scenario').disabled
        assert app.session_state['ar_result'] is original
        assert app.session_state['ar_last_run_fingerprint'] == old_fingerprint
        assert any('Inputs have changed' in notice.value for notice in app.info)
        assert app.text_input(key=key).value == raw
        app.text_input(key=key).set_value('').run()
        assert not screen_button(app).disabled and not app.button(key='ar_save_scenario').disabled, (key, [error.value for error in app.error])
    # A hot-reloaded app must also label results from the previous behavior stale.
    previous = json.loads(old_fingerprint)
    previous.pop('screening_behavior_version')
    app.session_state['ar_last_run_fingerprint'] = json.dumps(previous, sort_keys=True)
    app.run()
    assert any('Inputs have changed' in notice.value for notice in app.info)
    corrupt = copy.deepcopy(original)
    corrupt['scenarios'][0]['water']['pH'] = float('nan')
    app.session_state['ar_result'] = corrupt
    app.run()
    assert not app.exception
    assert any('previous results cannot be displayed' in error.value for error in app.error)


def test_malformed_import_is_atomic_then_a_valid_import_still_works(app):
    screen_button(app).click().run()
    assert not app.exception
    historical = snapshot('Historical')
    app.session_state['ar_saved_scenarios'] = [historical]
    app.run()
    app.selectbox(key='ar_scenario_selected').select(historical['scenario_id']).run()
    original_entries = copy.deepcopy(app.session_state['ar_pfas_entries_v2'])
    original_result = app.session_state['ar_result']
    for field, value in [('pfas_results', {'bad': 1}), ('performance', 'bad'), ('adsorbent_support', 'bad')]:
        bad = snapshot('Invalid')
        bad['result']['recommendations'][0][field] = value
        payload = json.dumps(bad).encode()
        with patch('streamlit.file_uploader', side_effect=lambda label, **kwargs: io.BytesIO(payload) if label == 'Reload a scenario JSON' else None):
            app.run()
            app.button(key='ar_import_scenario').click().run()
        assert not app.exception
        assert field in app.session_state['ar_scenario_error']
        assert app.session_state['ar_pfas_entries_v2'] == original_entries
        assert app.session_state['ar_saved_scenarios'] == [historical]
        assert app.session_state['ar_scenario_selected'] == historical['scenario_id']
        assert app.session_state['ar_result'] is original_result
    # Also reject a failure discovered after file validation, during restore preparation.
    bad = snapshot('Unsupported water')
    bad['inputs']['water']['water_type'] = 'unsupported matrix'
    bad['result']['scenarios'][0]['water']['water_type'] = 'unsupported matrix'
    payload = json.dumps(bad).encode()
    with patch('streamlit.file_uploader', side_effect=lambda label, **kwargs: io.BytesIO(payload) if label == 'Reload a scenario JSON' else None):
        app.run()
        app.button(key='ar_import_scenario').click().run()
    assert not app.exception
    assert app.session_state['ar_pfas_entries_v2'] == original_entries
    assert app.session_state['ar_saved_scenarios'] == [historical]
    assert app.session_state['ar_result'] is original_result
    payload = json.dumps(snapshot('Valid import')).encode()
    with patch('streamlit.file_uploader', side_effect=lambda label, **kwargs: io.BytesIO(payload) if label == 'Reload a scenario JSON' else None):
        app.run()
        app.button(key='ar_import_scenario').click().run()
    assert not app.exception
    assert 'ar_scenario_error' not in app.session_state
    assert len(app.session_state['ar_saved_scenarios']) == 2
    assert 'ar_result' not in app.session_state


def test_corrupt_existing_library_entry_remains_removable(app):
    bad = snapshot('Corrupt old entry')
    bad['result']['recommendations'][0]['pfas_results'] = {'bad': 1}
    app.session_state['ar_saved_scenarios'] = [bad]
    app.run()
    app.selectbox(key='ar_scenario_selected').select(bad['scenario_id']).run()
    assert not app.exception
    assert any('cannot be exported' in error.value for error in app.error)
    app.button(key='ar_remove_scenario').click().run()
    assert not app.exception
    assert app.session_state['ar_saved_scenarios'] == []
