"""Acceptance tests for structure generation, provenance, and unchanged ranking behavior."""
from __future__ import annotations
import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from unittest.mock import patch
import numpy as np
import pandas as pd
import pytest
from rdkit import Chem, rdBase

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "scripts"))
from adsorbent_recommender.pfas_cache import PfasFeatureStore
from adsorbent_recommender.pfas_descriptors import descriptor_row, prepare_structure
from adsorbent_recommender.pfas_identity import IdentityLookupError, _resolve, resolve_public_identity
from adsorbent_recommender.pfas_support import UNSEEN_WARNING
from adsorbent_recommender.service import load_recommender_bundle, recommend, recommend_mixture, resolve_pfas
from adsorbent_recommender.bundle import make_bundle_self_contained

BUNDLE_DIR = ROOT / "output/ML_logKd/Global_xgb_combination_42_20261014_184656"
NOVEL = "O=C(O)" + "C(F)(F)" * 24 + "F"

@pytest.fixture(scope="module")
def bundle():
    if not (BUNDLE_DIR / "model_ensemble.joblib").is_file():
        pytest.skip("The deployment bundle is unavailable")
    return load_recommender_bundle(BUNDLE_DIR)

def scenario(identity="PFOA", kind="abbreviation", **pfas):
    return {"pfas": {"identity": identity, "identity_type": kind, **pfas},
            "initial_concentration_mg_l": .1, "water": {"water_type": "Ultrapure water", "pH": 7}, "temperature_c": 25}

def test_all_deployed_descriptors_match_the_reviewed_snapshot(bundle):
    cache = pd.read_excel(BUNDLE_DIR / "pfas_features.xlsx")
    required = bundle.manifest["pfas_runtime"]["required_features"]
    assert len(required) == 16
    assert rdBase.rdkitVersion == bundle.manifest["pfas_runtime"]["rdkit_version"]
    for _, row in cache.iterrows():
        computed = descriptor_row(Chem.MolFromSmiles(row["SMILES"]))
        np.testing.assert_allclose([computed[col] for col in required], [row[col] for col in required], rtol=1e-9, atol=1e-9)

def test_generation_is_separate_from_cache_and_ignores_user_labels(bundle):
    path = BUNDLE_DIR / "pfas_features.xlsx"
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    lookup, _ = resolve_pfas(bundle, scenario("PFOA", "name", confirmed_smiles=NOVEL, structure_confirmed=True,
                                           label="PFOA", resolution={"source": "submitted_smiles"}))
    assert lookup.ready and lookup.status == "generated"
    assert lookup.cache_key is None
    assert "Abbreviation" not in lookup.feature_row
    assert lookup.feature_source == "computed_at_runtime"
    assert hashlib.sha256(path.read_bytes()).hexdigest() == digest

def test_equivalent_smiles_and_acid_base_aliases_reuse_curated_values(bundle):
    neutral, _ = resolve_pfas(bundle, scenario("PFBA"))
    mol = Chem.MolFromSmiles(neutral.canonical_smiles)
    reversed_smiles = Chem.MolToSmiles(Chem.RenumberAtoms(mol, list(reversed(range(mol.GetNumAtoms())))), canonical=False)
    equivalent, _ = resolve_pfas(bundle, scenario(reversed_smiles, "smiles"))
    anion, _ = resolve_pfas(bundle, scenario(neutral.canonical_smiles.replace("O=C(O)", "O=C([O-])"), "smiles"))
    assert equivalent.status == anion.status == "cached"
    assert equivalent.cache_key == anion.cache_key == neutral.cache_key
    assert anion.feature_row["Mw (g/mol)"] == neutral.feature_row["Mw (g/mol)"]

@pytest.mark.parametrize("smiles", ["not a SMILES", "FC(F)(F)C(=O)[O-].[Na+]", "*C(F)(F)F", "CCO"])
def test_invalid_disconnected_and_nonfluorinated_structures_are_actionable(bundle, smiles):
    lookup, _ = resolve_pfas(bundle, scenario(smiles, "smiles"))
    assert not lookup.ready and lookup.status == "invalid_structure"
    assert lookup.warnings

def test_only_selected_features_are_required(tmp_path):
    path = tmp_path / "cache.xlsx"
    pd.DataFrame([{"Canonical SMILES": "FC(F)(F)C(=O)O", "Abbreviation": "X", "rdkit_F_count": 3, "LogP_pred": np.nan}]).to_excel(path, index=False)
    store = PfasFeatureStore(path, "Sheet1", required_features=("rdkit_F_count",))
    assert store.lookup_identifier("X", "abbreviation").ready
    missing = PfasFeatureStore(path, "Sheet1", required_features=("Mw (g/mol)",)).lookup_identifier("X", "abbreviation")
    assert not missing.ready and "Mw (g/mol)" in missing.warnings[0]
    unsupported = PfasFeatureStore(path, "Sheet1", required_features=("LogP_pred",), allow_generation=True).lookup(NOVEL)
    assert not unsupported.ready and unsupported.status == "missing_required_features"

def test_ambiguous_local_identifier_requires_structure(tmp_path):
    path = tmp_path / "cache.xlsx"
    pd.DataFrame([{"Canonical SMILES": s, "Abbreviation": "X"} for s in ["FC(F)(F)C(=O)O", "FC(F)(F)S(=O)(=O)O"]]).to_excel(path, index=False)
    lookup = PfasFeatureStore(path).lookup_identifier("X", "abbreviation")
    assert lookup.status == "ambiguous_identity" and not lookup.ready

def test_public_resolution_returns_all_candidates_and_never_selects_one():
    _resolve.cache_clear()
    with patch("adsorbent_recommender.pfas_identity._get_json", side_effect=[
        {"IdentifierList": {"CID": [2, 1]}}, {"PropertyTable": {"Properties": [
            {"CID": 1, "Title": "One", "SMILES": "FC(F)(F)C(=O)O"},
            {"CID": 2, "Title": "Two", "SMILES": "FC(F)(F)S(=O)(=O)O"}]}}]) as get:
        candidates = resolve_public_identity("ambiguous test compound", "name")
        assert [r["cid"] for r in candidates] == [1, 2]
        assert all("confirmed" not in r for r in candidates)
        assert len(resolve_public_identity("ambiguous test compound", "name")) == 2
        assert get.call_count == 2

def test_public_resolution_errors_and_cas_checksum():
    _resolve.cache_clear()
    with patch("adsorbent_recommender.pfas_identity._get_json", side_effect=IdentityLookupError("Paste SMILES")):
        with pytest.raises(IdentityLookupError, match="Paste SMILES"):
            resolve_public_identity("unavailable compound", "name")
    with pytest.raises(IdentityLookupError, match="checksum"):
        resolve_public_identity("335-67-2", "cas")

def test_confirmed_structure_is_required_by_service(bundle):
    with pytest.raises(ValueError, match="Confirm"):
        resolve_pfas(bundle, scenario("X", "name", confirmed_smiles=NOVEL))

def test_cached_unseen_and_generated_pfas_have_training_diagnostics(bundle):
    # Keep this assertion fast; the deployed full-catalog mixture is tested separately.
    small = load_recommender_bundle(BUNDLE_DIR)
    small.catalog = small.catalog[small.catalog.Product_key.isin(["FILTRASORB400", "PUROFINEPFA694E"])]
    for entry in [scenario("PFDoA"), scenario(NOVEL, "smiles")]:
        result = recommend(small, entry, top_k=2)
        assert result["status"] == "recommendations_ready"
        assert result["pfas_support"]["seen_in_training"] is False
        assert len(result["pfas_support"]["nearest_training_pfas"]) == 3
        for item in result["recommendations"]:
            assert UNSEEN_WARNING in item["warnings"]
            assert any(c["code"] == "unseen_pfas" for c in item["caveats"])
    assert result["pfas_support"]["out_of_range_descriptors"]

def test_mixture_keeps_unresolved_entries_and_flags_limiting_unseen(bundle):
    small = load_recommender_bundle(BUNDLE_DIR)
    small.catalog = small.catalog[small.catalog.Product_key.isin(["FILTRASORB400", "PUROFINEPFA694E"])]
    unresolved = recommend_mixture(small, [scenario(), scenario("nonexistent identifier", "name")])
    assert unresolved["status"] == "pfas_not_ready"
    assert len(unresolved["unresolved_pfas"]) == 1
    with patch("adsorbent_recommender.service.predict_logkd", side_effect=lambda b, row: {
        "mean_logKd": -2.0 if row["rdkit_F_count"] > 40 else 4.0, "ensemble_sd": .1}):
        result = recommend_mixture(small, [scenario(), scenario(NOVEL, "smiles")])
    assert result["status"] == "recommendations_ready"
    assert all(item["mixture_performance"]["limiting_pfas_unseen"] for item in result["recommendations"])
    assert all(len(item["pfas_results"]) == 2 for item in result["recommendations"])

def test_version_four_bundles_keep_cached_screening_without_runtime_extension(bundle):
    old = load_recommender_bundle(BUNDLE_DIR)
    old.manifest.pop("pfas_runtime", None)
    old.manifest["capabilities"].pop("unseen_pfas_prediction", None)
    assert resolve_pfas(old, scenario())[0].ready
    new = resolve_pfas(old, scenario(NOVEL, "smiles"))[0]
    assert not new.ready and new.status == "uncached"

def test_repackaging_is_idempotent_and_keeps_model_bytes(bundle):
    digest = hashlib.sha256((BUNDLE_DIR / "model_ensemble.joblib").read_bytes()).hexdigest()
    before = (BUNDLE_DIR / "recommender_bundle.json").read_bytes()
    make_bundle_self_contained(BUNDLE_DIR)
    assert before == (BUNDLE_DIR / "recommender_bundle.json").read_bytes()
    assert digest == hashlib.sha256((BUNDLE_DIR / "model_ensemble.joblib").read_bytes()).hexdigest()

def test_all_cached_rows_resolve_without_changes_to_selected_features(bundle):
    cache = pd.read_excel(BUNDLE_DIR / "pfas_features.xlsx")
    for _, row in cache.iterrows():
        lookup, _ = resolve_pfas(bundle, scenario(row["Canonical SMILES"], "smiles"))
        assert lookup.ready and lookup.feature_source == "reviewed_cache"
        for feature in bundle.manifest["pfas_runtime"]["required_features"]:
            assert lookup.feature_row[feature] == row[feature]

def test_known_compound_rankings_equal_the_prechange_service(bundle):
    original = ROOT / ".current/scripts/adsorbent_recommender/service.py"
    if not original.is_file():
        original = ROOT / ".baseline/scripts/adsorbent_recommender/service.py"
    if not original.is_file():
        pytest.skip("Prechange snapshot is retained only in the implementation workspace")
    spec = importlib.util.spec_from_file_location("baseline_recommender_service", original)
    baseline = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = baseline
    spec.loader.exec_module(baseline)
    current = load_recommender_bundle(BUNDLE_DIR)
    before = baseline.load_recommender_bundle(BUNDLE_DIR)
    for model in current.models + before.models:
        model.set_params(model__n_jobs=1)
    entries = [scenario("PFOA"), scenario("PFOS")]
    # Compare unchanged scores over the full pre-filter catalog. The default
    # adsorbent support restriction has separate membership/routing tests.
    for entry in entries:
        entry["adsorbent"] = {"include_exploratory": True}
    expected = baseline.recommend_mixture(before, entries, top_k=len(before.catalog))
    actual = recommend_mixture(current, entries, top_k=len(current.catalog))
    assert actual["candidates_screened"] == expected["candidates_screened"]
    for a, b in zip(actual["recommendations"], expected["recommendations"], strict=True):
        assert a["product"]["Product_key"] == b["product"]["Product_key"]
        assert a["ranking_score"] == b["ranking_score"]
        assert a["performance_source"] == b["performance_source"]
        assert [row["performance"] for row in a["pfas_results"]] == [row["performance"] for row in b["pfas_results"]]
