from __future__ import annotations
import io
import math
from pathlib import Path
from types import SimpleNamespace

import pytest
from adsorbent_recommender.mixture_inputs import (
    EXAMPLE_CSV_PATH, MixtureInputError, append_prepared_import, from_mg_l,
    normalize_unit, prepare_csv_import, read_mixture_csv, to_mg_l,
)

SMILES = "O=C(O)C(F)(F)F"

def resolver(pfas):
    if pfas["identity"] == "unknown":
        return SimpleNamespace(ready=False, warnings=("Supply parent SMILES.",))
    return SimpleNamespace(ready=True, canonical_smiles=SMILES,
        status="generated" if pfas["identity_type"] == "smiles" else "cached",
        cache_key=None if pfas["identity_type"] == "smiles" else "Example PFAS",
        feature_row={}, feature_source="computed_at_runtime" if pfas["identity_type"] == "smiles" else "reviewed_cache",
        warnings=())

@pytest.mark.parametrize("unit,value", [("ng/L",100), ("µg/L",.1), ("μg/L",.1), ("ug/L",.1), ("mg/L",.0001)])
def test_equivalent_concentrations_have_the_same_model_input(unit, value):
    assert to_mg_l(value, unit) == .0001
    assert from_mg_l(.0001, unit) == value

def test_switching_units_preserves_physical_concentration_exactly():
    for concentration in [.1, .0001, 1e-9, 3.75]:
        for unit in ["ng/L", "µg/L", "mg/L"]:
            assert to_mg_l(from_mg_l(concentration, unit), unit) == concentration

@pytest.mark.parametrize("value", ["ND", "<5", ">100", "", "nan", "inf", "-1", "0", "1e-999", None])
def test_unquantified_or_invalid_concentrations_are_not_substituted(value):
    with pytest.raises(MixtureInputError):
        to_mg_l(value, "ng/L")

@pytest.mark.parametrize("unit", ["", "ppm", "ppb", "ng", None])
def test_unknown_or_missing_units_are_rejected(unit):
    with pytest.raises(MixtureInputError):
        normalize_unit(unit)

def test_example_is_parseable_and_shows_all_three_unit_scales():
    rows = read_mixture_csv(EXAMPLE_CSV_PATH.read_bytes())
    assert len(rows) == 4
    assert {normalize_unit(row["unit"]) for row in rows} == {"ng/L", "µg/L", "mg/L"}
    assert any(row["identity_type"] == "cas" for row in rows)
    assert any(row["identity_type"] == "smiles" and row["pfas"].startswith("O=C") for row in rows)
    assert EXAMPLE_CSV_PATH.read_text().splitlines()[0] == "pfas,identity_type,concentration,unit,parent_smiles"

def test_utf8_bom_headers_case_and_scientific_notation_are_supported():
    rows = read_mixture_csv("\ufeffPFAS,Concentration,Unit\nPFOA,1e2,µg/L\n".encode("utf-8"))
    prepared = prepare_csv_import(rows, resolver, [])
    assert prepared[0]["entry"]["initial_concentration_mg_l"] == .1

@pytest.mark.parametrize("payload", [b"", b"pfas,concentration\nPFOA,1\n", b"pfas,concentration,unit\n",
    b"pfas,concentration,unit\nPFOA,1,ng/L,extra\n", b"pfas,concentration,unit\nPFOA,1\n",
    b"pfas,concentration,unit,PFAS\nPFOA,1,ng/L,PFOA\n", b"pfas,concentration,unit\nPFOA,1,\xb5g/L\n"])
def test_bad_csv_structure_is_reported(payload):
    with pytest.raises(MixtureInputError):
        read_mixture_csv(payload)

def test_upload_size_and_row_limits_are_bounded():
    with pytest.raises(MixtureInputError, match="1 MB"):
        read_mixture_csv(b"x" * 1_000_001)
    with pytest.raises(MixtureInputError, match="100"):
        read_mixture_csv(b"pfas,concentration,unit\n" + b"PFOA,1,ng/L\n" * 101)

def test_nd_error_is_actionable_and_import_is_atomic():
    rows = read_mixture_csv(b"pfas,concentration,unit\nPFOA,100,ng/L\nPFOS,ND,ng/L\n")
    prepared = prepare_csv_import(rows, resolver, [])
    assert "ND" in prepared[1]["error"]
    existing = [{"id": 1, "structure_key": "other"}]
    with pytest.raises(MixtureInputError):
        append_prepared_import(existing, prepared, 2)
    assert existing == [{"id": 1, "structure_key": "other"}]

def test_aliased_structures_cannot_be_imported_twice_or_summed():
    rows = read_mixture_csv(b"pfas,concentration,unit\nPFOA,100,ng/L\nPFOA alias,0.1,ug/L\n")
    prepared = prepare_csv_import(rows, resolver, [])
    assert "duplicates" in prepared[1]["error"]
    prepared = prepare_csv_import(rows[:1], resolver, [{"structure_key": SMILES}])
    assert "duplicates" in prepared[0]["error"]

@pytest.mark.parametrize("structure_column", ["parent_smiles", "smiles"])
def test_unknown_names_require_smiles_and_submitted_structures_need_confirmation(structure_column):
    rows = read_mixture_csv(f"pfas,concentration,unit,{structure_column}\nunknown,10,ng/L,\n".encode())
    assert "SMILES" in prepare_csv_import(rows, resolver, [])[0]["error"]
    rows[0][structure_column] = SMILES
    prepared = prepare_csv_import(rows, resolver, [])
    assert not prepared[0]["error"] and prepared[0]["requires_confirmation"]
    existing, next_id = append_prepared_import([], prepared, 5)
    assert next_id == 6 and existing[0]["id"] == 5
    assert existing[0]["structure_confirmed"]
    assert existing[0]["confirmed_smiles"] == SMILES
    assert existing[0]["concentration_unit"] == "ng/L"
    assert existing[0]["initial_concentration_mg_l"] == .00001

def test_smiles_identity_is_read_directly_from_pfas_without_a_structure_column():
    rows = read_mixture_csv(f"pfas,identity_type,concentration,unit\n{SMILES},smiles,10,ng/L\n".encode())
    requests = []
    def capture(pfas):
        requests.append(pfas)
        return resolver(pfas)
    prepared = prepare_csv_import(rows, capture, [])
    assert requests == [{"identity": SMILES, "identity_type": "smiles"}]
    assert prepared[0]["identity_type"] == "smiles"
    assert not prepared[0]["error"] and prepared[0]["requires_confirmation"]
    entry = append_prepared_import([], prepared, 1)[0][0]
    assert entry["identity"] == entry["confirmed_smiles"] == SMILES

@pytest.mark.parametrize("identity,kind,parent,legacy", [
    ("unknown", "name", SMILES, "O=C(O)C(F)(F)C(F)(F)F"),
    (SMILES, "smiles", "O=C(O)C(F)(F)C(F)(F)F", ""),
])
def test_conflicting_structure_columns_cannot_silently_change_identity(identity, kind, parent, legacy):
    rows = [{"csv_row": 2, "pfas": identity, "identity_type": kind,
             "concentration": "10", "unit": "ng/L", "parent_smiles": parent, "smiles": legacy}]
    prepared = prepare_csv_import(rows, resolver, [])
    assert prepared[0]["error"] and not prepared[0]["entry"]
    with pytest.raises(MixtureInputError):
        append_prepared_import([], prepared, 1)

def test_legacy_and_parent_structure_columns_can_describe_equivalent_smiles():
    rows = [{"csv_row": 2, "pfas": "unknown", "identity_type": "name",
             "concentration": "10", "unit": "ng/L", "parent_smiles": SMILES, "smiles": "FC(F)(F)C(O)=O"}]
    prepared = prepare_csv_import(rows, resolver, [])
    assert not prepared[0]["error"]

def test_commit_rechecks_duplicates_if_the_mixture_changes():
    rows = read_mixture_csv(b"pfas,concentration,unit\nPFOA,1,mg/L\n")
    prepared = prepare_csv_import(rows, resolver, [])
    with pytest.raises(MixtureInputError, match="changed"):
        append_prepared_import([{"id": 1, "structure_key": SMILES}], prepared, 2)
