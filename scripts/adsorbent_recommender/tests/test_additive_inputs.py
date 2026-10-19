from __future__ import annotations

import copy
import csv
import io
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from adsorbent_recommender.additive_inputs import AdditiveInputError, database_additive_text, validate_additives
from adsorbent_recommender.service import normalize_water_scenario
from adsorbent_recommender.evidence import requested_conditions
from adsorbent_recommender.scenarios import (
    SCENARIO_VERSION, SCENARIO_EXAMPLE_PATH, ScenarioError, read_scenario_csv,
    scenario_inputs_csv, snapshot_bytes, read_snapshot,
)
from adsorbent_recommender.tests.test_scenarios import snapshot
from adsorbent_recommender.tests.test_candidate_evidence import match
from adsorbent_recommender.tests.test_database_evidence import _row


def salt(species="NaCl", concentration=1, unit="mM"):
    return {"species": species, "concentration": concentration, "unit": unit}


def water(additives, **kwargs):
    return normalize_water_scenario({"water": {"inorganic_additives": additives, **kwargs}})


@pytest.mark.parametrize("entries,ionic,ions", [
    ([salt()], .001, ("Na", "Cl")),
    ([salt("CaCl2", 2)], .006, ("Ca", "Cl")),
    ([salt(), salt("CaCl2", 2)], .007, ("Na", "Ca", "Cl")),
    ([salt("MgCl2", 1)], .003, ("Mg", "Cl")),
    ([salt("Na2SO4", 1)], .003, ("Na", "SO4")),
])
def test_structured_salts_compute_ionic_strength_and_cannot_be_erased_by_unchecked_ions(entries, ionic, ions):
    row, warnings = water(entries, ions={key: False for key in ("Na", "K", "Ca", "Mg", "Cl", "SO4")})
    assert float(row["Ionic_strength_(mol/L)"]) == pytest.approx(ionic)
    assert row["Water_type"] == "synthetic water"
    assert all(row[f"contains_{ion}"] for ion in ions)
    assert not warnings


@pytest.mark.parametrize("amount,unit", [(1, "mM"), (1000, "uM"), (1000, "µM"), (.001, "mol/L"),
    (.001, "M"), (58.44, "mg/L"), (58440, "ug/L"), (.05844, "g/L")])
def test_concentration_units_produce_equivalent_chemistry(amount, unit):
    row, _ = water([salt(concentration=amount, unit=unit)])
    assert float(row["Ionic_strength_(mol/L)"]) == pytest.approx(.001, rel=1e-4)


def test_measured_ionic_strength_takes_precedence_and_conflicts_are_visible():
    row, warnings = water([salt()], ionic_strength_mol_l=.1)
    assert row["Ionic_strength_(mol/L)"] == .1
    assert warnings and "differs" in " ".join(warnings).casefold() and "kept extracted" in " ".join(warnings).casefold()
    assert row["contains_Na"] and row["contains_Cl"]


def test_background_matrix_and_manual_other_ions_are_preserved():
    row, _ = water([salt()], water_type="groundwater", ions={"Ca": True, "Na": False})
    assert row["Water_type"] == "groundwater" and row["contains_Ca"] and row["contains_Na"]
    empty, _ = water([], ions={"Na": False, "Ca": False})
    assert empty["Water_type"] == "ultrapure water" and float(empty["Ionic_strength_(mol/L)"]) == 0
    assert empty["contains_Na"] is False and empty["contains_Ca"] is False


def test_ph_adjuster_convention_and_charge_imbalance_are_inspectable():
    row, _ = water([salt("HCl")])
    assert row["Water_type"] == "ultrapure water" and not row["inorganic_matter_present"]
    assert "pH-adjusters ignored" in row["normalization_trace"]
    conditions = requested_conditions({"water": {"inorganic_additives": [salt("HCl")]}}, row, {})
    assert "ionic_strength_mol_L" not in conditions
    assert match([_row()], water={"inorganic_additives": [salt("HCl")]})["status"] == "matched"
    row, warnings = water([salt("Na^+")])
    assert not row["Ionic_strength_(mol/L)"]
    assert "charge-imbalanced" in " ".join(warnings)


@pytest.mark.parametrize("entry", [salt(concentration=0), salt(concentration=-1), salt(concentration=True),
    salt(concentration=float("nan")), salt(concentration=float("inf")), salt(concentration="1"),
    salt(species="not a chemical"), salt(species="NaCl_1_mM"), salt(species="NaCl;CaCl2"),
    salt(unit="unknown"), salt(species="")])
def test_invalid_entries_are_rejected_without_guessing(entry):
    with pytest.raises(AdditiveInputError):
        validate_additives([entry])


def test_alias_duplicates_are_rejected_and_small_values_survive_the_shared_parser():
    with pytest.raises(AdditiveInputError, match="duplicates"):
        validate_additives([salt(), salt("sodium chloride", 2)])
    entries = [salt(concentration=1e-7)]
    assert "e-" not in database_additive_text(entries)
    assert float(water(entries)[0]["Ionic_strength_(mol/L)"]) == pytest.approx(1e-10)


def test_additive_text_is_no_longer_an_input_format():
    with pytest.raises(ValueError, match="structured"):
        normalize_water_scenario({"water": {"inorganic_matter": "1 mM NaCl"}})


def test_database_comparison_respects_derived_ionic_strength_and_inferred_ions():
    additives = [salt()]
    normalized, _ = water(additives)
    assert requested_conditions({"water": {"inorganic_additives": additives}}, normalized, {})["ionic_strength_mol_L"] == pytest.approx(.001)
    source = _row(Water_type="synthetic water", inorganic_matter_present=True,
                  contains_Na=True, contains_Cl=True, **{"Ionic_strength_(mol/L)": .001})
    assert match([source], water={"inorganic_additives": additives})["status"] == "matched"
    wrong = copy.deepcopy(source)
    wrong["Ionic_strength_(mol/L)"] = .1
    result = match([wrong], water={"inorganic_additives": additives})
    assert result["status"] == "no_match"
    assert "ionic_strength_mol_L" in result["related_evidence"]["excluded_by_condition"]
    wrong = copy.deepcopy(source)
    wrong["contains_Na"] = False
    assert match([wrong], water={"inorganic_additives": additives})["status"] == "no_match"


def test_current_json_and_csv_roundtrip_the_same_structured_entries():
    saved = snapshot()
    saved["inputs"]["water"]["inorganic_additives"] = [salt(), salt("CaCl2", 2)]
    saved["result"] = None
    assert SCENARIO_VERSION == saved["scenario_version"] == 3
    assert read_snapshot(snapshot_bytes(saved))["inputs"] == saved["inputs"]
    _, loaded = read_scenario_csv(scenario_inputs_csv(saved))
    assert loaded["water"] == saved["inputs"]["water"]
    records = list(csv.DictReader(io.StringIO(scenario_inputs_csv(saved).decode("utf-8-sig"))))
    assert [row["row_type"] for row in records] == ["pfas", "inorganic_additive", "inorganic_additive"]
    assert all(not row["pfas"] and not row["identity_type"] for row in records[1:])


def test_example_csv_includes_a_complete_mixture_and_multiple_additives():
    _, data = read_scenario_csv(SCENARIO_EXAMPLE_PATH.read_bytes())
    assert len(data["pfas"]) == 2
    assert data["water"]["inorganic_additives"] == [salt(), salt("CaCl2", 2)]


@pytest.mark.parametrize("mutation", ["unknown_species", "duplicate_species", "mixed_identity", "missing_amount", "wrong_type", "water_mismatch"])
def test_invalid_additive_csv_does_not_partially_load(mutation):
    saved = snapshot()
    records = list(csv.DictReader(io.StringIO(scenario_inputs_csv(saved).decode("utf-8-sig"))))
    if mutation == "unknown_species":
        records[-1]["inorganic_species"] = "unknown"
    elif mutation == "duplicate_species":
        records.append(dict(records[-1]))
    elif mutation == "mixed_identity":
        records[-1]["pfas"] = "PFOA"
    elif mutation == "missing_amount":
        records[-1]["concentration"] = ""
    elif mutation == "wrong_type":
        records[-1]["row_type"] = "salt"
    else:
        records[-1]["ph"] = "8"
    stream = io.StringIO()
    writer = csv.DictWriter(stream, fieldnames=list(records[0]))
    writer.writeheader()
    writer.writerows(records)
    with pytest.raises(ScenarioError):
        read_scenario_csv(stream.getvalue().encode())
