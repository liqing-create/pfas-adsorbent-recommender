# schemas.py
from typing import Dict, Type, List, Any

# ──────────────────────────────────────────────────────────────────────────────
# Experiment extraction fields
# ──────────────────────────────────────────────────────────────────────────────
EXPERIMENT_FIELDS: Dict[str, Type] = {
    # Water_Matrix
    "Water_type_full_name": str,
    "Water_type_abbreviation": str,
    "Water_type_class": str,
    "pH": str,

    # Bulk experiment params
    "Solution_volume_(mL)": str,
    "Temperature_(°C)": str,
    "Contact_time_(h)": str,
    "Mixing_speed_(rpm)": str,
    "PFAS_initial_concentration": str,
    "Adsorbent_dosage": str,
    "Inorganic_matter": str,
    "Ionic_strength": str,
    "TDS_(mg/L)": str,
    "Organic_matter": str,  
    "TOC_(mg/L)": str,
    "DOC_(mg/L)": str,
}

# ──────────────────────────────────────────────────────────────────────────────
# Performance extraction fields
# ──────────────────────────────────────────────────────────────────────────────
PERFORMANCE_FIELDS: Dict[str, Type] = {
    "Performance_id":       str,
    "PFAS_name":       str,
    "Adsorbent_id":  str,
    "Test_mode": str,
    "Differentiating_Condition":    str, 
    "Equilibrium_reached":    str, 

    # Freundlich
    "Freundlich_equation_form":   str,
    "Freundlich_KF_value":   str,
    "Freundlich_KF_unit":    str,
    "Freundlich_n_or_1/n_verbatim":    str,
    "Freundlich_R2":         str,

    # Langmuir
    "Langmuir_Qm_value":     str,
    "Langmuir_Qm_unit":      str,
    "Langmuir_KL_value":     str,
    "Langmuir_KL_unit":      str,
    "Langmuir_R2":           str,

    # Kd
    "Kd_value":            str,
    "Kd_unit":             str,

    # Removal
    "Removal_rate":    str,

    # Equilibrium uptake
    "Qe_value":            str,
    "Qe_unit":             str,

    # kinetic constants
    "PFO_k1_value":            str,
    "PFO_k1_unit":            str,
    "PFO_Qe_value":            str,
    "PFO_Qe_unit":            str,
    "PFO_R2":                 str,
    "PSO_k2_value":             str,
    "PSO_k2_unit":             str,
    "PSO_v0_value":             str,
    "PSO_v0_unit":             str,
    "PSO_Qe_value":             str,
    "PSO_Qe_unit":             str,
    "PSO_R2":                  str,


    # Bulk experiment params
    "Water_type": str,
    "pH": str,
    "Solution_volume_(mL)": str,
    "Temperature_(°C)": str,
    "Contact_time_(h)": str,
    "Mixing_speed_(rpm)": str,
    "PFAS_C0": str,
    "Adsorbent_dosage": str,
    "Inorganic_matter": str,
    "Ionic_strength": str,
    "TDS_(mg/L)": str,
    "Organic_matter": str,  
    "TOC_(mg/L)": str,
    "DOC_(mg/L)": str,
}


# ──────────────────────────────────────────────────────────────────────────────
# Adsorbent extraction fields
# ──────────────────────────────────────────────────────────────────────────────
ADSORBENT_FIELDS: Dict[str, Type] = {
    # Name
    "Vendor":               str,
    "Nominal_grain_size":  str,
    "AC_raw_material":   str,
    "AC_activation_method":   str,
    "Polymer_matrix":   str,
    "Functional_group":  str,
    "Ion_exchange_capacity": str,
    "C%":           str,
    "N%":           str,
    "O%":           str,
    "Elemental_composition_method":           str,
    "Pore_class":           str,
    "Total_pore_percentage":           str,
    "Micro_pore_percentage":           str,
    "Meso_pore_percentage":           str,
    "Macro_pore_percentage":           str,
    "Total_pore_volume_(cm3/g)":              str,
    "Micro_pore_volume_(cm3/g)":              str,
    "Meso_pore_volume_(cm3/g)":              str,
    "Macro_pore_volume_(cm3/g)":              str,
    "Average_pore_diameter_(Angstrom)":              str,
    "SSA_(m2/g)":   str,
    "pHPZC":              str,
    "Zeta_potential_(mV)": str,
}

REVIEW_FIELDS = {
    "Data_provenance": str,
    "Require_review": str,   
    "Review_reason": str,    
}

