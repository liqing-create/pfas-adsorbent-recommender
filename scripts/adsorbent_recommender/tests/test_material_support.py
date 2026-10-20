from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from adsorbent_recommender.material_support import (
    DEFAULT_POLICY, describe, doi_key, enrich_studies, export_support, include_exploratory, support_table, validate_table,
)
from adsorbent_recommender.service import RecommenderBundle, recommend, recommend_mixture
from adsorbent_recommender.scenarios import (
    ScenarioError, make_snapshot, read_snapshot, read_scenario_csv, results_csv, scenario_inputs_csv, scenario_requests,
    snapshot_bytes, validate_inputs,
)
from adsorbent_recommender.tests.test_database_evidence import _row
from adsorbent_recommender.tests.test_scenarios import inputs, REFERENCE


def catalog():
    return pd.DataFrame([
        {"Product_key": key, "Name_Commercial": name, "adsorbent_category": "Activated carbon", "adsorbent_subcategory": "GAC",
         "present_material_feature_count": 13, "required_material_feature_count": 23, "minimum_present_material_features": 9}
        for key, name in [("FILTRASORB400", "Filtrasorb 400"), ("OTHER", "Other carbon"), ("UNSEEN", "New carbon")]
    ])


def records():
    full = [{"study_no": f"s{index % 2}", "DOI": f"10.1234/study{index % 2}", "source_row_index": index,
             "extraction_dataset": "A", "adsorbent_id": "F400", "name_commercial": "Filtrasorb 400", "PFAS_name": "PFOA"}
            for index in range(5)]
    thin = [{"study_no": "s3", "DOI": "10.1234/thin", "source_row_index": 20,
             "extraction_dataset": "A", "adsorbent_id": "OTHER", "name_commercial": "Other carbon", "PFAS_name": "PFOA"}
            for _ in range(8)]
    return pd.DataFrame(full + thin)


def bundle():
    products = catalog()
    return RecommenderBundle(Path('.'),
        {"model_family": "Global", "target": "logKd", "features": {"selected": []}, "input_contract": {"fields": []}},
        {}, [], {"controls": {}, "feature_support": {}}, products,
        material_support=support_table(products, records()), material_support_info={"available": True, "policy": dict(DEFAULT_POLICY)})


@pytest.fixture
def inference(monkeypatch):
    from adsorbent_recommender import service
    monkeypatch.setattr(service, "resolve_pfas", lambda *args: (SimpleNamespace(status="cached", ready=True, canonical_smiles="",
        cache_key="PFOA", warnings=(), feature_row={"PFAS_name": "PFOA"}, offline_request=None), {}))
    monkeypatch.setattr(service, "describe_support", lambda *args: {"seen_in_training": True})
    monkeypatch.setattr(service, "normalize_water_scenario", lambda *args: ({"Water_type": "ultrapure water"}, []))
    monkeypatch.setattr(service, "_model_row", lambda *args: ({"product": args[4]["Product_key"]}, [], []))
    monkeypatch.setattr(service, "predict_logkd", lambda _, row: {"mean_logKd": {"FILTRASORB400": 2, "OTHER": 4, "UNSEEN": 5}[row['product']], "ensemble_sd": .1})
    return bundle()


def scenario(exploratory=False, identity='PFOA'):
    return {"pfas": {"identity": identity, "identity_type": "abbreviation"}, "initial_concentration_mg_l": 1,
            "water": {}, "adsorbent": {"include_exploratory": exploratory}}


def test_property_coverage_does_not_imply_fitted_support_and_expanded_siblings_count_once():
    table = support_table(catalog(), records())
    thin = describe(table, {"available": True}, "OTHER")
    assert thin['training_rows'] == 8 and thin['source_records'] == thin['independent_studies'] == 1
    assert thin['status'] == 'limited' and not thin['meets_default_ml_policy']
    assert describe(table, {"available": True}, "FILTRASORB400")['meets_default_ml_policy']
    assert describe(table, {"available": True}, "UNSEEN")['status'] == 'unseen'


def test_duplicate_aliases_and_dois_do_not_inflate_support():
    rows = records().iloc[:5].copy()
    rows['name_abbreviation'] = 'Filtrasorb 400'
    rows['DOI'] = ['https://doi.org/10.1234/SAME', 'doi:10.1234/same', '10.1234/same', '10.1234/same', '10.1234/same']
    result = describe(support_table(catalog(), rows), {"available": True}, 'FILTRASORB400')
    assert result['training_rows'] == result['source_records'] == 5
    assert result['study_ids'] == 2 and result['independent_studies'] == 1
    assert not result['meets_default_ml_policy']


def test_bibliography_enrichment_resolves_missing_doi_but_conflicts_fail_closed():
    rows = records().iloc[:5].drop(columns='DOI')
    refs = records()[['study_no', 'DOI']].drop_duplicates()
    assert describe(support_table(catalog(), rows, refs), {"available": True}, 'FILTRASORB400')['meets_default_ml_policy']
    conflicting = pd.concat([refs, pd.DataFrame([{'study_no': 's0', 'DOI': '10.1234/conflict'}])])
    result = describe(support_table(catalog(), rows, conflicting), {"available": True}, 'FILTRASORB400')
    assert not result['meets_default_ml_policy'] and any('could not be verified' in text for text in result['reasons'])


def test_missing_provenance_and_legacy_snapshot_are_exploratory_only():
    table = support_table(catalog(), records().drop(columns=['source_row_index', 'DOI']))
    result = describe(table, {'available': True}, 'FILTRASORB400')
    assert result['source_records'] == result['independent_studies'] == 0
    assert not result['meets_default_ml_policy']
    assert describe(pd.DataFrame(), {}, 'FILTRASORB400')['status'] == 'unknown'


def test_source_ids_are_scoped_by_extraction_dataset():
    rows = records().iloc[:2].copy()
    rows['source_row_index'] = 1
    rows['extraction_dataset'] = ['A', 'B']
    assert support_table(catalog(), rows).iloc[0].source_records == 2


@pytest.mark.parametrize('value', ['true', 1, None])
def test_exploratory_setting_requires_boolean(value):
    with pytest.raises(ValueError, match='true or false'):
        include_exploratory({'adsorbent': {'include_exploratory': value}})


def test_default_ranking_keeps_thin_support_and_excludes_products_without_training_records(inference):
    result = recommend(inference, scenario(), top_k=3)
    assert [row['product']['Product_key'] for row in result['recommendations']] == ['OTHER', 'FILTRASORB400']
    assert [(row['Product_key'], row['reason']) for row in result['rejected_candidates']] == [('UNSEEN', 'adsorbent_not_in_training_data')]
    assert result['adsorbent_screening']['excluded_products'] == 1
    thin = result['recommendations'][0]
    assert thin['characterization']['seen_in_training'] is True
    # Few studies or records is context, not a caveat (tier placement did not depend on it).
    assert not any(row['code'] in {'thin_adsorbent_support', 'study_dominated_adsorbent'} for row in thin['caveats'])
    assert not any(row['code'] == 'exploratory_adsorbent' for row in thin['caveats'])


def test_opt_in_retains_products_without_training_records_with_critical_warning(inference):
    result = recommend(inference, scenario(True), top_k=3)
    assert len(result['recommendations']) == 3 and result['adsorbent_screening']['exploratory_products'] == 1
    unseen = next(row for row in result['recommendations'] if row['product']['Product_key'] == 'UNSEEN')
    assert any(row['code'] == 'exploratory_adsorbent' and row['severity'] == 'critical' for row in unseen['caveats'])
    thin = next(row for row in result['recommendations'] if row['product']['Product_key'] == 'OTHER')
    assert not any(row['code'] == 'exploratory_adsorbent' for row in thin['caveats']) and thin['ranking_score'] == 4


def test_comparable_database_evidence_survives_sparse_ml_support(inference):
    inference.evidence = pd.DataFrame([_row(adsorbent_id='OTHER', name_commercial='Other carbon', name_abbreviation='', name_full='')])
    result = recommend(inference, scenario(), top_k=3)
    thin = next(row for row in result['recommendations'] if row['product']['Product_key'] == 'OTHER')
    assert thin['performance_source'] == 'database_match' and thin['ranking_score'] == 3.2
    assert not any(row['code'] == 'exploratory_adsorbent' for row in thin['caveats'])


def test_mixture_excludes_product_if_any_pfas_requires_unsupported_ml_and_reports_which(inference, monkeypatch):
    from adsorbent_recommender import service
    original = service.match_candidate_evidence
    monkeypatch.setattr(service, 'match_candidate_evidence', lambda evidence, **kwargs:
        {'status': 'matched', 'performance': {'mean_logKd': 3.2}} if kwargs['material']['Product_key'] == 'UNSEEN' and kwargs['scenario']['pfas']['identity'] == 'PFOA' else original(evidence, **kwargs))
    result = recommend_mixture(inference, [scenario(), scenario(identity='PFOS')], top_k=3)
    assert [row['product']['Product_key'] for row in result['recommendations']] == ['OTHER', 'FILTRASORB400']
    rejection = next(row for row in result['rejected_candidates'] if row['Product_key'] == 'UNSEEN')
    assert len(rejection['pfas_rejections']) == 1


def test_mixture_marks_exploratory_limiting_score_and_retains_every_pfas(inference):
    result = recommend_mixture(inference, [scenario(True), scenario(True, 'PFOS')], top_k=3)
    assert len(result['recommendations']) == 3
    assert result['recommendations'][0]['mixture_performance']['limiting_score_exploratory_adsorbent']
    assert all(len(row['pfas_results']) == 2 for row in result['recommendations'])


def test_empty_default_set_is_actionable_and_mixed_modes_are_rejected(inference):
    inference.material_support_info = {}
    result = recommend_mixture(inference, [scenario(), scenario(identity='PFOS')])
    assert result['status'] == 'no_supported_candidates' and len(result['rejected_candidates']) == 3
    with pytest.raises(ValueError, match='shared'):
        recommend_mixture(inference, [scenario(), scenario(True, 'PFOS')])


def test_support_snapshot_export_and_validation(tmp_path):
    metadata = export_support(tmp_path, catalog(), records())
    table = pd.read_csv(tmp_path / metadata['snapshot_filename'])
    assert metadata['default_ml_supported_products'] == 1
    validate_table(table)
    table.loc[0, 'source_records'] = 99
    with pytest.raises(ValueError, match='contradict'):
        validate_table(table)


def test_csv_json_and_service_requests_preserve_exploratory_choice():
    data = validate_inputs({**inputs(), 'include_exploratory_adsorbents': True})
    saved = make_snapshot('Exploratory', data, REFERENCE)
    assert read_snapshot(snapshot_bytes(saved))['inputs']['include_exploratory_adsorbents']
    _, restored = read_scenario_csv(scenario_inputs_csv(saved))
    assert restored['include_exploratory_adsorbents']
    assert scenario_requests(restored)[0]['adsorbent']['include_exploratory']
    assert not validate_inputs(inputs())['include_exploratory_adsorbents']


def test_result_csv_carries_source_counts_and_warning(inference):
    result = recommend_mixture(inference, [scenario(True)], top_k=3)
    text = results_csv(result).decode('utf-8-sig')
    assert 'adsorbent_source_records' in text and 'Exploratory ML prediction' in text


def test_packaging_reads_study_dois_from_the_project_database_when_the_recorded_path_is_missing(tmp_path, monkeypatch):
    # A bundle trained on another machine records that machine's database path; packaging must not
    # fall back to the partial reference-dose bibliography while the project's own database exists.
    from adsorbent_recommender import bundle as packaging
    catalog().to_csv(tmp_path / "catalog_snapshot.csv", index=False)
    (tmp_path / "run_config.json").write_text('{"input_path": "C:/elsewhere/missing.xlsx"}', encoding="utf-8")
    database = tmp_path / "database.xlsx"
    records()[["study_no", "DOI"]].drop_duplicates().to_excel(database, sheet_name="Standardized_Full", index=False)
    monkeypatch.setattr(packaging.cfg, "DEFAULT_INPUT", database)
    metadata = packaging._snapshot_material_support(tmp_path, records().drop(columns="DOI"))
    assert metadata["bibliography_source"] == "Standardized_Full of database.xlsx"
    table = pd.read_csv(tmp_path / metadata["snapshot_filename"])
    assert describe(table, metadata, "FILTRASORB400")["independent_studies"] == 2
