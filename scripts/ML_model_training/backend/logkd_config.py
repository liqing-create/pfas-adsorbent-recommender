"""Shared configuration for the split-first PFAS logKd modeling pipeline."""

from __future__ import annotations

from pathlib import Path


BACKEND_DIR = Path(__file__).resolve().parent
ML_MODEL_TRAINING_DIR = BACKEND_DIR.parent
AD_ROOT = ML_MODEL_TRAINING_DIR.parents[1]
BOX_NOTES_ROOT = AD_ROOT.parents[1]

DEFAULT_INPUT = AD_ROOT / "output" / "Merge" / "pfas_adsorption_performance_database.xlsx"
DEFAULT_OUTPUT_ROOT = AD_ROOT / "output" / "ML_logKd"
DEFAULT_PFAS_FEATURES = BOX_NOTES_ROOT / "PFAS" / "pfas_features.xlsx"
DEFAULT_PFAS_FEATURES_SHEET = "Sheet1"

# Shared baseline for every repeated exploratory wrapper.  Keep these values in
# one place so a comparison changes only the modelling choice it names.
DEFAULT_EXPERIMENT_RANDOM_SEED = 42
DEFAULT_EXPERIMENT_OUTER_REPEATS = 5
DEFAULT_EXPERIMENT_N_TRIALS = 200
DEFAULT_EXPERIMENT_EARLY_STOP_WARMUP = 100
DEFAULT_EXPERIMENT_EARLY_STOP_PATIENCE = 25
DEFAULT_EXPERIMENT_EARLY_STOP_SE_MULTIPLIER = 0.25
DEFAULT_EXPERIMENT_EARLY_STOP_MIN_DELTA_FLOOR = 0.003
DEFAULT_EXPERIMENT_N_JOBS = 8
DEFAULT_EXPERIMENT_CONSOLE_VERBOSITY = "quiet"
DEFAULT_EXPERIMENT_XGB_SEARCH_SPACE = "compact"
DEFAULT_EXPERIMENT_NUMERIC_MISSING_STRATEGY = "xgb_native"
# The algorithm screen includes Ridge, which cannot fit numeric NaN values.
# It therefore evaluates every candidate under one shared imputed-input policy.
ALGORITHM_COMPARISON_NUMERIC_MISSING_STRATEGY = "median_indicator"
DEFAULT_GROUPED_OUTER_ALLOCATION_METHODS = ("random_group",)
SPLIT_STRATEGY_GROUPED_OUTER_ALLOCATION_METHODS = (
    "random_group",
    "size_matched_random_group",
)
DEFAULT_DATA_MODE = "drop_unreliable"

SHEET_NAME = "Equilibrium_Data"
TARGET = "Kd_final_log10(L/g)"
AUTOMATED_UNRELIABLE_FLAG_COLUMNS = (
    "logKd_conversion_spike_flag",
    "logKd_conversion_dip_flag",
    "low_Ce_detection_limit_flag",
    "small_concentration_difference_flag",
    "low_removal_rate_flag",
    "high_apparent_removal_flag",
)
HUMAN_REVIEW_UNRELIABLE_FLAG = "human_review_unreliable_flag"
UNRELIABLE_FLAG_COLUMNS = AUTOMATED_UNRELIABLE_FLAG_COLUMNS + (
    HUMAN_REVIEW_UNRELIABLE_FLAG,
)
# Older database builds legitimately lack the manual-review field.  Treat its
# absence as no manual exclusions so the audit command can create the review
# queue before the first build that materializes the field.
OPTIONAL_UNRELIABLE_FLAG_COLUMNS = (HUMAN_REVIEW_UNRELIABLE_FLAG,)
UNRELIABLE_FLAG_TRUTHY_VALUES = ("1", "true", "t", "yes", "y")

VALIDATION_FOLDS = 5
DEFAULT_TEST_FRACTION = 0.20
# This controls estimator stochasticity only.  It intentionally differs from
# the split/tuning seed so repeated outer splits assess data-partition effects
# without also changing the XGBoost, random-forest, or extra-trees draw.
DEFAULT_MODEL_RANDOM_SEED = 2025

# Evidence coverage is a diagnostic of what a fitted training partition can
# teach the model: availability plus value diversity and within-study contrast.
# It never selects or optimizes an allocation and never uses target values.
DEFAULT_EVIDENCE_COVERAGE_NUMERIC_BINS = 5

# The observed-support KDE candidates use the Schultz-style bounded Epanechnikov kernel on
# training-only, standardized selected inputs.  Bandwidth is estimated from a
# reproducible training-only sample for computational tractability.
DEFAULT_SUPPORT_KDE_BANDWIDTH_QUANTILE = 0.30
DEFAULT_SUPPORT_KDE_BANDWIDTH_SAMPLE_SIZE = 500
DEFAULT_SUPPORT_KDE_MIN_TRAINING_ROWS = 20
DEFAULT_SUPPORT_KDE_MIN_BANDWIDTH = 0.15
# This is a separately evaluated dimensionality-reduction candidate.  PCA is
# fitted only on the reference rows of each inner/outer training partition.
DEFAULT_SUPPORT_KDE_PCA_COMPONENTS = 5

MODEL_CATEGORIES = {
    "AC": ["Activated carbon"],
    "Resin": ["Ion exchange resin", "Nonionic resin"],
    "CDP": ["Cyclodextrin polymer", "Cyclodextrin polymers"],
    "Global": [
        "Activated carbon",
        "Ion exchange resin",
        "Nonionic resin",
        "Cyclodextrin polymer",
        "Cyclodextrin polymers",
    ],
}

SPLIT_STRATEGIES = (
    "random_row",
    "combination",
    "study",
    "pfas",
    "adsorbent",
)

# Final outer testing assignments remain strictly random.  The optional
# evidence-balanced method operates only on inner validation folds, against a
# fixed outer-training reference manifest, and never uses targets or models.
INNER_ALLOCATION_METHODS = ("random_row", "random_group", "evidence_balanced")
OUTER_ALLOCATION_METHODS = (
    "random_row",
    "random_group",
    "size_matched_random_group",
)

SPLIT_ROW_ID_CANDIDATES = ("standardized_row_id", "source_row_index")
DEFAULT_MIN_NONMISSING_ROWS = 50
DEFAULT_MIN_REPORTING_STUDY_FRACTION = 0.20
DEFAULT_MIN_REPORTING_ADSORBENT_FRACTION = 0.15
DEFAULT_MIN_CATEGORICAL_LEVELS = 2
DEFAULT_MIN_CATEGORICAL_LEVEL_ROWS = 20
# Rows are not independent observations of a level.  One material measured
# against many PFAS under many conditions carries hundreds of rows, so a level
# held by a single adsorbent, study, or PFAS still clears a row floor while
# describing one thing.  A level must therefore also appear in several
# independent units of the feature's own bucket -- adsorbent identity for a
# material property, study for an experimental condition, PFAS for a molecular
# descriptor -- before it counts as supported.
DEFAULT_MIN_CATEGORICAL_LEVEL_ENTITIES = 3
DEFAULT_MIN_NUMERIC_DISTINCT_VALUES = 3
DEFAULT_MIN_NUMERIC_IQR_BINS = 1.0

# How a feature whose reported values are only 0/1 is modeled.  Several inputs
# are already one-hot in the database because more than one can hold at once
# (the contains_<ion> flags derived from one Inorganic_matter field, for
# example).  "numeric" keeps each such column as one 0/1 input and leaves an
# unreported value missing, so it is encoded exactly once.  "categorical"
# one-hot encodes it a second time into present/absent/<missing> columns, which
# turns "never reported" into a modeled category and makes one unreported
# source field occupy one duplicated column per derived flag.
BINARY_INDICATOR_HANDLING_CHOICES = ("numeric", "categorical")
DEFAULT_BINARY_INDICATOR_HANDLING = "numeric"

# PFAS characteristics are screened within each training scope on the distinct
# PFAS represented there.  These criteria intentionally differ from the
# literature-extracted adsorbent and experimental-feature thresholds above.
DEFAULT_PFAS_MISSING_RATE_THRESHOLD = 0.20
DEFAULT_PFAS_DOMINANT_FRACTION_THRESHOLD = 0.95
# Every feature family uses this same high-correlation definition.  The
# screening unit differs by family only to avoid pseudo-replication.
DEFAULT_CORRELATION_THRESHOLD = 0.90
DEFAULT_MIN_PAIRWISE_OBSERVATIONS = 20
# ``use_all`` leaves every screened input available.  The default selects one
# input from each high-correlation group, using the documented quality order.
CORRELATED_FEATURE_HANDLING_CHOICES = ("use_all", "select_one_per_group")
DEFAULT_CORRELATED_FEATURE_HANDLING = "select_one_per_group"
# Optional researcher-defined tie-breaks for correlation groups.  Lower values
# are preferred after missingness, stratum coverage, and usable variation have
# been compared.  It is intentionally empty until a scientific preference is
# explicitly documented; configured candidate order is then the final stable
# tie-break rather than an unrecorded domain assumption.
CORRELATION_FEATURE_SCIENTIFIC_PRIORITY: dict[str, int] = {}

# Feature-family ablation policies.  The *_only policies use the named
# PFAS family or families while excluding the other PFAS characteristic
# families; non-PFAS experimental and adsorbent features remain available.
PFAS_FEATURE_FAMILY_POLICIES = (
    "all",
    "no_rdkit",
    "no_opera",
    "no_abraham",
    "rdkit_atlas_only",
    "rdkit_only",
)
DEFAULT_PFAS_FEATURE_FAMILY_POLICY = "rdkit_only"
PFAS_OPERA_SCALAR_CANDIDATES = (
    "LogP_pred",
    "LogWS_pred",
    "LogKOA_pred",
    "pKa_a_pred",
    "pKa_b_pred",
    "LogD55_pred",
    "LogD74_pred",
)
PFAS_ATLAS_CANDIDATES = ("First_Class", "Second_Class")

# pH is commonly reported to one decimal place; the default preserves the
# normalized precision of every other numeric feature while removing float noise.
DEFAULT_NUMERIC_ROUND_DECIMALS = 6
NUMERIC_FEATURE_ROUND_DECIMALS = {"pH": 1}

MISSING_TOKENS = {"", "na", "n/a", "nan", "none", "null", "not applicable"}
MODEL_INCLUDE_COL = "model_include"
MODEL_EXCLUDE_TOKENS = {"0", "false", "f", "no", "n", "exclude", "excluded", "drop", "remove"}
PFAS_KEY_CANDIDATES = ("Abbreviation", "PFAS_Abbreviation", "PFAS_name", "Name")

EXPERIMENTAL_CANDIDATES = [
    "PFAS_C0_value_mg/L",
    "Adsorbent_dosage_value_mg/L",
    "pH",
    "Temperature_(\u00b0C)",
    "Contact_time_(h)",
    "Solution_volume_(mL)",
    "Mixing_speed_(rpm)",
    "Water_type",
    "organic_carbon_mg/L",
    "TDS_(mg/L)",
    "Inorganic_matter",
    "inorganic_matter_present",
    "contains_Na",
    "contains_K",
    "contains_Ca",
    "contains_Mg",
    "contains_Cl",
    "contains_HCO3",
    "contains_SO4",
    "contains_phosphate",
    "Organic_matter",
    "organic_matter_present",
    "organic_matter_class",
    "Ionic_strength_(mol/L)",
]

# Core named PFAS characteristics.  Additional calculated scalar columns are
# discovered from the model-frame schema using their stable naming conventions.
PFAS_CANDIDATES = [
    "First_Class",
    "Second_Class",
    "LogD55_pred",
    "LogD74_pred",
    "LogKOA_pred",
    "LogP_pred",
    "LogWS_pred",
    "pKa_a_pred",
]

PFAS_RDKIT_CANDIDATES = [
    "rdkit_mol_logp",
    "rdkit_tpsa",
    "rdkit_num_h_acceptors",
    "rdkit_num_h_donors",
    "rdkit_num_rotatable_bonds",
    "rdkit_ring_count",
    "rdkit_num_aromatic_rings",
    "rdkit_fraction_csp3",
    "rdkit_N_count",
    "rdkit_Cl_count",
    "rdkit_perfluoroether_O_count",
    "rdkit_nonperfluoroether_O_count",
    "rdkit_F_count",
    "rdkit_hydrogenated_C_count",
    "rdkit_fluorinated_C_count",
    "rdkit_nonfluorinated_C_count",
    "rdkit_partially_fluorinated_C_count",
    "rdkit_terminal_CF3_count",
    "rdkit_longest_fluorinated_C_chain",
    "rdkit_fluorinated_segment_count",
    "rdkit_fluorinated_branch_point_count",
    "rdkit_fluorinated_chain_compactness",
    "rdkit_carboxyl_group_count",
    "rdkit_sulfonic_acid_group_count",
    "rdkit_sulfonamide_group_count",
    "rdkit_phosphonic_acid_group_count",
    "rdkit_noncarboxyl_carbonyl_count",
    "rdkit_positive_charge_count",
    "rdkit_negative_charge_count",
]

# Core PFAS characteristics must be available for every retained PFAS. These
# values are generated from SMILES and therefore belong to input quality, not
# tunable feature support. pKa_a_pred is intentionally optional because it may
# be chemically inapplicable or unavailable from OPERA for some structures.
PFAS_OPTIONAL_CANDIDATE_MIN_COVERAGE = {"pKa_a_pred": 0.90}
# The feature-selection analysis decides which derived PFAS scalars are usable.
# The pre-selection data gate therefore requires only identity/classification
# fields, rather than rejecting a PFAS because one optional model descriptor is
# absent.
PFAS_CORE_REQUIRED_COLUMNS = ("SMILES", "First_Class", "Second_Class")

AD_MORGAN_RADIUS = 2
UNSUITABLE_MODEL_FEATURE_PREFIXES = ("AD_", "AD_index_", "Conf_index_")
UNSUITABLE_MODEL_FEATURE_SUBSTRINGS = ("predRange",)

ADSORBENT_COMMON_CANDIDATES = [
    "adsorbent_subcategory",
    "contains_microporous",
    "contains_mesoporous",
    "contains_macroporous",
    "contains_gel",
    "contains_nonporous",
    "Nominal_grain_size_value_mm",
    "ssa_(m2/g)_avg",
    "average_pore_diameter_(angstrom)",
    "phpzc",
    "porosity_total_value",
    "porosity_micro_value_avg",
    "porosity_meso_value_avg",
    "porosity_macro_value_avg",
    "pore_volume_total_value_cm3/g",
    "pore_volume_micro_value_avg_cm3/g",
    "pore_volume_meso_value_avg_cm3/g",
    "pore_volume_macro_value_avg_cm3/g",
    "element_C_value",
    "element_N_value",
    "element_O_value",
    "zeta_potential_near_pH_7_(mV)",
]

AC_CANDIDATES = [
    "AC_raw_material",
    "AC_activation_method",
]

FUNCTIONAL_GROUP_INDICATOR_CANDIDATES = [
    "contains_quaternary_ammonium",
    "contains_sulfonic_acid",
    "contains_carboxyl",
    "contains_carbonyl",
    "contains_hydroxyl",
    "contains_lactone",
    "contains_pyridinic_nitrogen",
    "contains_pyrrolic_nitrogen",
    "contains_amino_acid",
    "contains_amine",
    "contains_amide",
    "contains_nitrile",
    "contains_alkoxy",
    "contains_glycosidic_bond",
    "contains_ester",
    "contains_benzyl_chloride",
    "contains_siloxane",
    "contains_unspecified_functional_groups",
]

RESIN_CANDIDATES = [
    "ion_exchange_capacity_value_meq/g",
    "ion_exchange_capacity_value_meq/L",
    *FUNCTIONAL_GROUP_INDICATOR_CANDIDATES,
    "polymer_matrix",
]

# The global model pools every named adsorbent family, so it may use the
# family-specific fields available for AC and resin records as well as the
# shared material properties.  Screening remains fold-local and can still
# reject a field with insufficient observations or variation.
GLOBAL_CANDIDATES = [
    "adsorbent_category",
    *AC_CANDIDATES,
    *RESIN_CANDIDATES,
]

# A blank may mean that a source did not report a property, whereas a property
# can also be chemically inapplicable to an adsorbent family.  Keep those two
# cases separate.  These are the fields whose semantics are genuinely confined
# to a known family in the standardized database.  Functional-group indicators
# deliberately do not appear here: their reported values span AC, ion-exchange
# resin, cyclodextrin polymer, and clay/mineral records, so treating their
# blanks outside resin as "not applicable" would discard observed information.
ADSORBENT_SPECIFIC_FEATURE_APPLICABILITY: dict[str, tuple[str, ...]] = {
    "AC_raw_material": ("Activated carbon",),
    "AC_activation_method": ("Activated carbon",),
    "ion_exchange_capacity_value_meq/g": ("Ion exchange resin",),
    "ion_exchange_capacity_value_meq/L": ("Ion exchange resin",),
    "polymer_matrix": ("Ion exchange resin", "Nonionic resin"),
}
ADSORBENT_NOT_APPLICABLE_CATEGORY = "<not_applicable>"
ADSORBENT_APPLICABILITY_INDICATOR_SUFFIX = "__applicability"

FEATURE_BUCKETS = [
    "PFAS characteristics",
    "Experimental conditions",
    "Adsorbent properties",
]

LEAKAGE_PREFIXES = (
    "Kd_",
    "Ce_",
    "Qe_",
    "PFO_",
    "PSO_",
    "Langmuir_",
    "Freundlich_",
    "Removal_",
    "apparent_removal_",
    "audit_",
)

LEAKAGE_EXACT = {
    TARGET,
    "Kd_final_source",
    "Kd_final_L/g",
    "Ce_final_source",
    "Ce_final_mg/L",
    "Removal_rate",
    "Removal_rate_avg_fraction",
    "apparent_removal_fraction",
    "apparent_removal_rate",
    "apparent_removal_source",
    "logKd_conversion_spike_flag",
    "logKd_conversion_dip_flag",
    "low_Ce_detection_limit_flag",
    "concentration_difference_C0_minus_Ce_mg/L",
    "small_concentration_difference_flag",
    "low_removal_rate_flag",
    "high_apparent_removal_flag",
    "Qt_from_removal_mg/g",
}

IDENTIFIER_COLUMNS = {
    "extraction_dataset",
    "study_no",
    "PFAS_name",
    "adsorbent_id",
    "name_abbreviation",
    "name_commercial",
    "name_full",
    "Abbreviation",
    "Compound Full Name",
    "SMILES",
    "Canonical SMILES",
    "Molecular Formula",
    "CAS Number",
    "DSSTox Substance ID",
}
