from __future__ import annotations

import copy
import json
from types import SimpleNamespace

import pandas as pd
import pytest

from adsorbent_recommender.evidence import match_candidate_evidence, requested_conditions
from adsorbent_recommender.input_validation import InputValidationError, finite_number
from adsorbent_recommender.scenarios import (
    ScenarioError, compare_snapshots, make_snapshot, read_scenario_csv, read_snapshot,
    recommendation_bytes, results_csv, scenario_inputs_csv, validate_inputs,
)
from adsorbent_recommender.service import (
    _model_row, fixed_test_conditions, normalize_water_scenario, recommend, recommend_mixture,
)
from adsorbent_recommender.tests.test_database_evidence import _row
from adsorbent_recommender.tests.test_scenarios import OTHER, REFERENCE, inputs, result, snapshot


@pytest.mark.parametrize('field', ['pH', 'organic_carbon_mg_l', 'tds_mg_l', 'ionic_strength_mol_l'])
@pytest.mark.parametrize('value', [float('nan'), float('inf'), -float('inf'), '1e309', True, 'ND'])
def test_service_rejects_invalid_water_before_scoring(field, value):
    request = {'initial_concentration_mg_l': .1, 'water': {field: value}}
    with pytest.raises(InputValidationError, match=field):
        normalize_water_scenario(request)
    with pytest.raises(InputValidationError, match=field):
        recommend_mixture(None, [{'initial_concentration_mg_l': .1}, request])


@pytest.mark.parametrize('value', [float('nan'), float('inf'), -float('inf'), '1e309', True, 'unknown'])
def test_service_rejects_temperature_before_identity_lookup(value):
    with pytest.raises(InputValidationError, match='Temperature'):
        recommend(None, {'initial_concentration_mg_l': .1, 'temperature_c': value})


@pytest.mark.parametrize('field', ['organic_carbon_mg_l', 'tds_mg_l', 'ionic_strength_mol_l'])
def test_negative_water_values_rejected_by_service_snapshots_and_csv(field):
    with pytest.raises(InputValidationError, match='zero or greater'):
        normalize_water_scenario({'water': {field: -.01}})
    data = inputs()
    data['water'][field] = -.01
    with pytest.raises(ScenarioError, match=field):
        validate_inputs(data)
    saved = snapshot()
    saved['inputs']['water'][field] = -.01
    with pytest.raises(ScenarioError, match=field):
        read_scenario_csv(scenario_inputs_csv(saved))


def test_finite_values_blanks_zeros_and_legacy_numeric_aliases_survive():
    assert finite_number('1e-6', 'value', allow_text=True) == .000001
    assert finite_number('', 'value', optional=True, allow_text=True) is None
    row, _ = normalize_water_scenario({'water': {'water_type': 'synthetic water', 'ph': 0,
        'DOC_mg_L': 0, 'TDS_mg_L': 0, 'ionic_strength_mol_L': 0}})
    assert row['pH'] == 0
    assert row['organic_carbon_mg/L'] == 0
    assert row['TDS_(mg/L)'] == 0
    assert row['Ionic_strength_(mol/L)'] == 0
    # Finite pH/temperature outside development ranges remain allowed inputs.
    assert finite_number(-1, 'pH') == -1
    assert finite_number(50, 'Temperature') == 50


def test_default_temperature_is_the_same_for_model_and_evidence():
    feature = 'Temperature_(°C)'
    bundle = SimpleNamespace(bounds={'controls': {feature: {'role': 'fixed', 'default': 25}}, 'feature_support': {}},
        manifest={'features': {'selected': [feature]}, 'input_contract': {'fields': []}})
    for supplied, expected in [(None, 25), (0, 0), (30, 30)]:
        request = {'initial_concentration_mg_l': 1, 'temperature_c': supplied}
        fixed = fixed_test_conditions(bundle, request)
        model_row, _, _ = _model_row(bundle, {}, request, {}, {})
        assert model_row[feature] == expected
        assert requested_conditions(request, {}, fixed)['temperature_C'] == expected
    bundle.bounds['controls'][feature]['default'] = 22
    assert fixed_test_conditions(bundle)['fixed_conditions'][feature] == 22


def test_default_temperature_excludes_unreported_and_outside_tolerance_records():
    evidence = pd.DataFrame([_row(standardized_row_id=str(temp), **{'Temperature_(°C)': temp})
                             for temp in (23, 25, 27, 30, None)])
    kwargs = dict(material={'Product_key': 'F400'}, pfas_feature_row={'Abbreviation': 'PFOA'}, pfas_cache_key='PFOA',
                  normalized_water={'Water_type': 'ultrapure water'},
                  screening_conditions={'fixed_conditions': {'Temperature_(°C)': 25}})
    blank = match_candidate_evidence(evidence, scenario={'initial_concentration_mg_l': 1, 'temperature_c': None}, **kwargs)
    explicit = match_candidate_evidence(evidence, scenario={'initial_concentration_mg_l': 1, 'temperature_c': 25}, **kwargs)
    assert blank == explicit
    assert {record['conditions']['temperature_C'] for record in blank['records']} == {23, 25, 27}
    assert 'temperature_C' in blank['related_evidence']['excluded_by_condition']


BAD_FIELDS = [
    ('pfas_results', {'bad': 1}), ('pfas_results', [1]),
    ('performance', 'bad'), ('prediction', []), ('adsorbent_support', 'bad'),
    ('pfas_support', []), ('pfas_lookup', {'support': None}),
    ('concentration_input', 'bad'), ('mixture_performance', None),
    ('database_evidence', {'related_evidence': 'bad'}),
    ('database_evidence', {'performance': None}), ('warnings', [1]), ('caveats', ['bad']),
    ('caveats', [{'text': 'Caution', 'pfas': 'wrong list type'}]),
]


@pytest.mark.parametrize('field,value', BAD_FIELDS)
def test_nested_malformed_results_raise_field_specific_errors(field, value):
    doc = snapshot()
    doc['result']['recommendations'][0][field] = value
    with pytest.raises(ScenarioError, match=field):
        read_snapshot(json.dumps(doc).encode())
    with pytest.raises(ScenarioError, match=field):
        results_csv(doc['result'])
    with pytest.raises(ScenarioError, match=field):
        compare_snapshots([doc, snapshot('Other')], doc['scenario_id'])


def test_mixture_archives_require_a_result_for_every_pfas():
    data = inputs()
    second = copy.deepcopy(data['pfas'][0])
    second.update(identity='Other PFAS', canonical_smiles=OTHER, confirmed_smiles=OTHER)
    data['pfas'].append(second)
    valid = make_snapshot('Mixture', data, REFERENCE, result(data))
    assert len(read_snapshot(json.dumps(valid).encode())['result']['recommendations'][0]['pfas_results']) == 2
    for pairs in (None, [], [{}]):
        bad = copy.deepcopy(valid)
        bad['result']['recommendations'][0]['pfas_results'] = pairs
        with pytest.raises(ScenarioError, match='every PFAS'):
            read_snapshot(json.dumps(bad).encode())


def test_single_pfas_legacy_fallback_and_strict_json_export():
    run = result()
    run['recommendations'][0]['product']['AC_raw_material'] = float('nan')
    payload = recommendation_bytes(run)
    assert b'NaN' not in payload and b'Infinity' not in payload
    assert json.loads(payload)['recommendations'][0]['product']['AC_raw_material'] is None
    assert read_snapshot(payload)['result']['recommendations']
    run['scenarios'][0]['water']['pH'] = float('nan')
    with pytest.raises(ScenarioError, match='pH'):
        recommendation_bytes(run)


def test_json_exponent_overflow_is_rejected_instead_of_becoming_null():
    payload = json.dumps(snapshot()).replace('"ranking_score": 4.2', '"ranking_score": 1e309').encode()
    with pytest.raises(ScenarioError, match='JSON number'):
        read_snapshot(payload)


@pytest.mark.parametrize('field,value', [('bundle', 'bad'), ('water_chemistry', []),
    ('screening_conditions', {'fixed_conditions': 'bad'}), ('mixture', None)])
def test_malformed_complete_recommendations_are_validated_before_conversion(field, value):
    run = result()
    run[field] = value
    with pytest.raises(ScenarioError, match=field):
        read_snapshot(json.dumps(run).encode())
