"""Paired feature-family-only and leave-one-family-out models."""

from __future__ import annotations

from typing import Any

from run_logkd_feature_threshold_comparison import main as run_feature_policy_comparison


DEFAULT_FEATURE_FAMILY_CONFIGURATIONS: dict[str, dict[str, Any]] = {
    "full": {"feature_screening_mode": "standard"},
    "only_adsorbent_properties": {
        "feature_screening_mode": "standard",
        "excluded_feature_buckets": [
            "Experimental conditions",
            "PFAS characteristics",
        ],
    },
    "only_experimental_conditions": {
        "feature_screening_mode": "standard",
        "excluded_feature_buckets": [
            "Adsorbent properties",
            "PFAS characteristics",
        ],
    },
    "only_pfas_characteristics": {
        "feature_screening_mode": "standard",
        "excluded_feature_buckets": [
            "Adsorbent properties",
            "Experimental conditions",
        ],
    },
    "without_adsorbent_properties": {
        "feature_screening_mode": "standard",
        "excluded_feature_buckets": ["Adsorbent properties"],
    },
    "without_experimental_conditions": {
        "feature_screening_mode": "standard",
        "excluded_feature_buckets": ["Experimental conditions"],
    },
    "without_pfas_characteristics": {
        "feature_screening_mode": "standard",
        "excluded_feature_buckets": ["PFAS characteristics"],
    },
}

# Match the comparison order used in the performance-figure mock-up: the full
# model, then all single-feature-family models, then all leave-one-family-out
# ablations.
FEATURE_FAMILY_COMPARISON_PLOT_ORDER = (
    "full",
    "only_adsorbent_properties",
    "only_experimental_conditions",
    "only_pfas_characteristics",
    "without_adsorbent_properties",
    "without_experimental_conditions",
    "without_pfas_characteristics",
)


def main() -> None:
    run_feature_policy_comparison(
        default_configurations=DEFAULT_FEATURE_FAMILY_CONFIGURATIONS,
        default_reference_configuration="full",
        experiment_label="feature_family_comparison",
        comparison_name="paired_feature_family_only_and_ablation",
        configuration_definition_filename="configuration_definitions.csv",
        include_excluded_feature_buckets=True,
        output_description="feature-family-only and ablation comparison",
        automatic_plot_group_column="threshold_configuration",
        automatic_plot_groups=FEATURE_FAMILY_COMPARISON_PLOT_ORDER,
        parser_description=(
            "Compare the full feature set with models using only one feature "
            "family at a time and leave-one-feature-family-out ablations, on "
            "frozen random-row splits."
        ),
    )


if __name__ == "__main__":
    main()
