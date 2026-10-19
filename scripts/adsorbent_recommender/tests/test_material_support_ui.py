from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from adsorbent_recommender.tests.test_scenarios_ui import add_pfoa

ROOT = Path(__file__).resolve().parents[3]
APP = ROOT / 'scripts/adsorbent_recommender/app.py'


@pytest.fixture
def app():
    if not (ROOT / 'output/ML_logKd/Global_xgb_combination_42_20261014_184656/model_ensemble.joblib').is_file():
        pytest.skip('The deployment bundle is unavailable')
    result = AppTest.from_file(str(APP), default_timeout=90).run()
    assert not result.exception
    return result


def test_products_without_training_records_are_not_offered(app):
    assert not any(box.key == 'ar_include_exploratory_adsorbents' for box in app.checkbox)


def test_default_screening_ranks_only_products_in_the_training_data(app):
    add_pfoa(app)
    app.button(key='ar_screen').click().run(timeout=90)
    assert not app.exception and app.session_state['ar_result']['status'] == 'recommendations_ready'
    run = app.session_state['ar_result']
    assert run['adsorbent_screening']['include_exploratory'] is False
    assert all(item['performance_source'] == 'database_match' or item['adsorbent_support']['identity_verified']
               for item in run['recommendations'])
    assert any('Adsorbent support:' in item.value for item in app.caption)
