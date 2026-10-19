"""Presentation labels for recorded score caveats; no scoring or severity changes."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable


@dataclass(frozen=True)
class WarningLabel:
    value: str
    label: str
    severity: str
    meaning: str


SEVERITIES = {
    "critical": {"name": "Critical", "mark": "!", "color": "#B36259", "fill": "rgba(179,98,89,0.16)", "legend": "Rose ! = critical alert", "order": 0},
    "info": {"name": "Context", "mark": "i", "color": "#397784", "fill": "rgba(57,119,132,0.14)", "legend": "Teal i = supporting context", "order": 1},
    "review": {"name": "Review", "mark": "?", "color": "#627A91", "fill": "rgba(98,122,145,0.14)", "legend": "Slate ? = review details", "order": 2},
}

# These describe the service's existing caveat codes. The actual caveat's
# severity controls its color, including for historical or future results.
LABELS = {
    "database_spread": ("Conflicting data", "Comparable database records span widely different logKd values."),
    "database_one_study": ("One study", "The database score is supported by one study only."),
    "database_near_dose": ("Near dose", "The database score uses 20–30 mg/L observations rather than exactly 25 mg/L."),
    "unseen_material": ("New adsorbent", "The model was not trained on this adsorbent."),
    "exploratory_adsorbent": ("Not in training", "The model score uses an adsorbent with no record in the training data."),
    "thin_adsorbent_support": ("Thinly supported", "The adsorbent is in the training data but below two DOI-identified studies or five source records; products meeting both are moderately supported."),
    "study_dominated_adsorbent": ("Study skew", "At least 80% of this adsorbent's source records come from one study, despite having multiple studies."),
    "property_out_of_range": ("Property range", "An adsorbent property lies outside the model's training range."),
    "property_unseen_category": ("New property", "An adsorbent property category was not represented in model training."),
    "extreme_logkd": ("Extreme logKd", "The predicted logKd is outside the model's better-supported 0–4 range."),
    "thin_characterization": ("Few properties", "Only the minimum required number of adsorbent properties is available."),
    "inapplicable_value": ("Value omitted", "A supplied value was not applicable to the model calculation."),
    "unseen_pfas": ("New PFAS", "The model was not trained on this PFAS."),
    "pfas_descriptors_out_of_range": ("PFAS range", "A molecular descriptor lies outside the model's training range."),
    "limited_water_evidence": ("Limited matrix", "Model training coverage for the selected water type is limited or concentrated in a few studies."),
    "thin_pfas": ("PFAS: one study", "This PFAS appears in model training from one study only."),
    "scenario_out_of_range": ("Input range", "A screening input lies outside the model's training range."),
    "scenario_unseen_category": ("New input", "A screening input category was not represented in model training."),
    "no_water_chemistry": ("No chemistry", "No water-chemistry measurements were reported."),
    "water_chemistry": ("Chemistry rule", "A water-chemistry normalization or interpretation rule applies."),
    "missing_inputs": ("Inputs blank", "Some model inputs were left unreported."),
    "database_conditions_unchecked": ("Match limits", "Database matching could not check conditions that were left unreported."),
}

def warning_labels(item: dict[str, Any]) -> list[WarningLabel]:
    labels = {}
    for caveat in item.get("caveats", []):
        severity = caveat.get("severity", "review")
        if severity not in SEVERITIES:
            severity = "review"
        label, meaning = LABELS.get(caveat.get("code"), (
            str(caveat.get("label") or "Other caveat"),
            str(caveat.get("text") or "See this candidate's full explanation."),
        ))
        spec = SEVERITIES[severity]
        value = f"{spec['name']}: {label}"
        labels.setdefault(value, WarningLabel(value, f"{spec['mark']} {label}", severity, meaning))
    return sorted(labels.values(), key=lambda tag: (SEVERITIES[tag.severity]["order"], tag.label.casefold()))


def warning_values(item: dict[str, Any]) -> list[str]:
    return [tag.value for tag in warning_labels(item)]


def legend_labels(items: Iterable[dict[str, Any]]) -> list[WarningLabel]:
    labels = {tag.value: tag for item in items for tag in warning_labels(item)}
    return sorted(labels.values(), key=lambda tag: (SEVERITIES[tag.severity]["order"], tag.label.casefold()))


def column_options(items: Iterable[dict[str, Any]]) -> dict[str, Any]:
    items = list(items)
    tags = legend_labels(items)
    lookup = {tag.value: tag.label for tag in tags}
    # Native colored tags occupy one line. Reserve enough space for every
    # label, rather than truncating them to an unexplained '+N' summary.
    width = max([280, *[24 + sum(30 + 9 * len(tag.label) for tag in warning_labels(item)) for item in items]])
    return {"options": [tag.value for tag in tags],
            # Translucent fills blend with the active light/dark theme. The
            # native renderer chooses a contrasting text color for each fill.
            "color": [SEVERITIES[tag.severity]["fill"] for tag in tags],
            "format_func": lambda value: lookup.get(value, str(value)), "width": width}
