import math

import pandas as pd

from normalization_endpoints import build_reference_dose_search_data


def _equilibrium_projection(df: pd.DataFrame) -> pd.DataFrame:
    return df[
        [
            "standardized_row_id",
            "Kd_final_L/g",
            "Kd_final_log10(L/g)",
            "Kd_final_source",
            "PFAS_C0_value_mg/L",
            "Adsorbent_dosage_value_mg/L",
        ]
    ].copy()


def test_langmuir_is_recalculated_exactly_at_reference_dose():
    full = pd.DataFrame(
        [
            {
                "standardized_row_id": "std_1",
                "PFAS_C0_value_mg/L": 1.0,
                "Adsorbent_dosage_value_mg/L": 100.0,
                "Kd_final_L/g": 50.0,
                "Kd_final_log10(L/g)": math.log10(50.0),
                "Kd_final_source": "Langmuir",
                "Ce_final_mg/L": 0.5,
                "apparent_removal_fraction": 0.5,
                "Langmuir_Qm_value_mg/g": 100.0,
                "Langmuir_KL_value_L/mg": 2.0,
                "Langmuir_R2": 0.99,
                "DOI": "https://doi.org/10.1000/example",
                "authors": "Example Author",
                "article_title": "Example adsorption study",
                "source_title": "Example Journal",
                "publication_year": 2026,
            }
        ]
    )

    result = build_reference_dose_search_data(full, _equilibrium_projection(full))

    assert len(result) == 1
    row = result.iloc[0]
    assert row["reference_dose_basis"] == "isotherm_recalculated_at_reference_dose"
    assert row["reference_dose_method"] == "Langmuir"
    assert bool(row["reference_dose_exact_flag"])
    assert row["search_dosage_mg/L"] == 25.0
    ce = float(row["search_Ce_mg/L"])
    qe = 100.0 * 2.0 * ce / (1.0 + 2.0 * ce)
    assert math.isclose(ce + 0.025 * qe, 1.0, rel_tol=0, abs_tol=1e-10)
    assert math.isclose(row["search_Kd_L/g"], qe / ce, rel_tol=1e-10)
    assert row["DOI"] == "https://doi.org/10.1000/example"
    assert row["article_title"] == "Example adsorption study"


def test_freundlich_is_recalculated_exactly_at_reference_dose():
    full = pd.DataFrame(
        [
            {
                "standardized_row_id": "std_2",
                "PFAS_C0_value_mg/L": 0.8,
                "Adsorbent_dosage_value_mg/L": 10.0,
                "Kd_final_L/g": 20.0,
                "Kd_final_log10(L/g)": math.log10(20.0),
                "Kd_final_source": "Freundlich",
                "Ce_final_mg/L": 0.6,
                "apparent_removal_fraction": 0.25,
                "Freundlich_KF_value_(mg/g)/(mg/L)^(1/n)": 12.0,
                "Freundlich_exponent_value": 0.7,
                "Freundlich_R2": 0.97,
            }
        ]
    )

    result = build_reference_dose_search_data(full, _equilibrium_projection(full))

    assert len(result) == 1
    row = result.iloc[0]
    assert row["reference_dose_method"] == "Freundlich"
    ce = float(row["search_Ce_mg/L"])
    qe = 12.0 * ce**0.7
    assert math.isclose(ce + 0.025 * qe, 0.8, rel_tol=0, abs_tol=1e-10)
    assert math.isclose(row["search_Kd_L/g"], qe / ce, rel_tol=1e-10)


def test_near_dose_observation_is_retained_but_labelled_approximate():
    full = pd.DataFrame(
        [
            {
                "standardized_row_id": "std_3",
                "PFAS_C0_value_mg/L": 1.0,
                "Adsorbent_dosage_value_mg/L": 23.0,
                "Kd_final_L/g": 10.0,
                "Kd_final_log10(L/g)": 1.0,
                "Kd_final_source": "Removal",
                "Ce_final_mg/L": 0.8,
                "apparent_removal_fraction": 0.2,
            }
        ]
    )

    result = build_reference_dose_search_data(full, _equilibrium_projection(full))

    assert len(result) == 1
    row = result.iloc[0]
    assert row["reference_dose_basis"] == "observed_dose_within_tolerance"
    assert row["reference_dose_method"] == "Observed Removal"
    assert not bool(row["reference_dose_exact_flag"])
    assert row["search_dosage_mg/L"] == 23.0
    assert row["search_logKd_log10(L/g)"] == 1.0


def test_outside_tolerance_and_low_quality_isotherm_are_excluded():
    full = pd.DataFrame(
        [
            {
                "standardized_row_id": "std_4",
                "PFAS_C0_value_mg/L": 1.0,
                "Adsorbent_dosage_value_mg/L": 100.0,
                "Kd_final_L/g": 10.0,
                "Kd_final_log10(L/g)": 1.0,
                "Kd_final_source": "Kd",
                "Langmuir_Qm_value_mg/g": 100.0,
                "Langmuir_KL_value_L/mg": 2.0,
                "Langmuir_R2": 0.75,
            }
        ]
    )

    result = build_reference_dose_search_data(full, _equilibrium_projection(full))

    assert result.empty
