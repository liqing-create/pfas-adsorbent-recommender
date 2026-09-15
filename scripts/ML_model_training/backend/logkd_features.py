"""Feature selection and feature-frame construction for PFAS logKd modeling."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from . import logkd_config as cfg
from .logkd_data import clean_text, coerce_numeric, normalize_pfas_key, normalized_missing_mask


FEATURE_SCREENING_MODES = ("standard", "none")
SCREENING_NOT_APPLIED = "not_applied"
RETAINED_WITHOUT_SCREENING = "retained_without_screening"

# These variables describe batch/kinetic protocol rather than equilibrium
# adsorption behavior. They are never model inputs, so they are reported by
# neither training nor the whole-database availability audit: the audit
# describes what the model input space looks like, and these are outside it.
# The standardized database still stores them for provenance and endpoint
# conversion.
EQUILIBRIUM_DOMAIN_EXCLUSIONS: dict[str, str] = {
    "Solution_volume_(mL)": "equilibrium_protocol_variable_excluded",
    "Contact_time_(h)": "equilibrium_kinetic_variable_excluded",
    "Mixing_speed_(rpm)": "equilibrium_kinetic_variable_excluded",
}


def _is_retained_feature_reason(reason: str) -> bool:
    """Return whether a feature remains eligible before correlation pruning."""
    return reason in {"selected", RETAINED_WITHOUT_SCREENING}


@dataclass(frozen=True)
class PFASFeatureAudit:
    """Target-free PFAS screening calculated inside one training scope."""

    unique_pfas_count: int
    candidates: tuple[str, ...]
    kinds: dict[str, str]
    missing_rate: dict[str, float]
    nonmissing_count: dict[str, int]
    dominant_fraction: dict[str, float]
    unique_count: dict[str, int]
    numeric_iqr_bins: dict[str, float]
    correlated_with: dict[str, tuple[str, ...]]
    max_absolute_correlation: dict[str, float]
    selection_reasons: dict[str, str]

    def metrics(self, feature: str) -> dict[str, Any]:
        correlated = self.correlated_with.get(feature, ())
        reason = self.selection_reasons.get(feature, "not_pfas_candidate")
        return {
            "pfas_training_candidate": feature in self.candidates,
            "pfas_training_unique_pfas": self.unique_pfas_count,
            "pfas_training_missing_rate": self.missing_rate.get(feature, np.nan),
            "pfas_training_nonmissing_unique_pfas": self.nonmissing_count.get(feature, 0),
            "pfas_training_kind": self.kinds.get(feature, ""),
            "pfas_training_dominant_fraction": self.dominant_fraction.get(feature, np.nan),
            "pfas_training_unique_values": self.unique_count.get(feature, 0),
            "pfas_training_numeric_iqr_bins": self.numeric_iqr_bins.get(feature, np.nan),
            "pfas_training_max_absolute_correlation": self.max_absolute_correlation.get(feature, np.nan),
            "pfas_training_correlated_with": "; ".join(correlated),
            "pfas_training_correlation_count": len(correlated),
            "pfas_training_selection_reason": reason,
            "pfas_training_selection_pass": reason == "selected",
        }

    def selection_reason(self, feature: str) -> str:
        return self.selection_reasons.get(feature, "not_pfas_candidate")

    def correlation_rank_key(
        self,
        feature: str,
        candidate_order: dict[str, int],
    ) -> tuple[float, float, float, int, int]:
        coverage = self.nonmissing_count.get(feature, 0) / max(self.unique_pfas_count, 1)
        spread = self.numeric_iqr_bins.get(feature, np.nan)
        return (
            self.missing_rate.get(feature, float("inf")),
            -coverage,
            -float(spread) if pd.notna(spread) else float("inf"),
            cfg.CORRELATION_FEATURE_SCIENTIFIC_PRIORITY.get(feature, 0),
            candidate_order[feature],
        )


def _is_pfas_scalar_candidate(column: str) -> bool:
    name = clean_text(column)
    return bool(
        name == "Mw (g/mol)"
        or name.startswith("abraham_")
        or name.startswith("rdkit_")
        or name.endswith("_pred")
    )


def _first_nonmissing(values: pd.Series) -> Any:
    observed = values.loc[~normalized_missing_mask(values)]
    return observed.iloc[0] if not observed.empty else np.nan


def _unique_pfas_scope(df: pd.DataFrame, candidates: list[str]) -> pd.DataFrame:
    """Return one deterministic descriptor record per PFAS in ``df``."""

    if "PFAS_name" not in df.columns:
        return pd.DataFrame(columns=candidates)
    keys = df["PFAS_name"].map(normalize_pfas_key)
    usable = keys.ne("")
    if not usable.any():
        return pd.DataFrame(columns=candidates)
    available = [column for column in candidates if column in df.columns]
    scoped = df.loc[usable, available].copy()
    scoped.insert(0, "__pfas_key", keys.loc[usable].to_numpy())
    # ``GroupBy.first`` has the same first-observed-value meaning as the
    # previous nested loop once project-specific missing-value markers have
    # been converted to actual nulls.  Doing this column-wise avoids one
    # tiny Series allocation for every PFAS/feature pair.
    for feature in available:
        scoped[feature] = scoped[feature].mask(normalized_missing_mask(scoped[feature]))
    scope = scoped.groupby("__pfas_key", sort=True)[available].first()
    return scope.reindex(columns=candidates) if not scope.empty else pd.DataFrame(columns=candidates)


def _pfas_value_summary(feature: str, values: pd.Series) -> tuple[str, int, int, float, float, pd.Series]:
    missing = normalized_missing_mask(values)
    observed = values.loc[~missing]
    nonmissing_count = int(len(observed))
    if not nonmissing_count:
        return "numeric", 0, 0, np.nan, np.nan, pd.Series(dtype=float)
    numeric = coerce_numeric(observed)
    numeric_ratio = float(numeric.notna().mean())
    if numeric_ratio >= 0.80:
        decimals = int(cfg.NUMERIC_FEATURE_ROUND_DECIMALS.get(feature, cfg.DEFAULT_NUMERIC_ROUND_DECIMALS))
        comparable = numeric.dropna().astype(float).round(decimals)
        kind = "numeric"
        iqr = float(comparable.quantile(0.75) - comparable.quantile(0.25)) if not comparable.empty else np.nan
        numeric_iqr_bins = iqr / (10.0 ** (-decimals)) if pd.notna(iqr) else np.nan
    else:
        comparable = observed.map(clean_text)
        comparable = comparable.loc[comparable.ne("")]
        kind = "categorical"
        numeric_iqr_bins = np.nan
    unique_count = int(comparable.nunique(dropna=True))
    dominant_fraction = float(comparable.value_counts(normalize=True).iloc[0]) if not comparable.empty else np.nan
    return kind, nonmissing_count, unique_count, dominant_fraction, numeric_iqr_bins, comparable


def _pfas_missing_rate_threshold(feature: str, default_threshold: float) -> float:
    required_coverage = cfg.PFAS_OPTIONAL_CANDIDATE_MIN_COVERAGE.get(feature)
    if required_coverage is None:
        return default_threshold
    return min(default_threshold, 1.0 - float(required_coverage))


def audit_pfas_training_features(
    df: pd.DataFrame,
    candidates: list[str],
    *,
    feature_screening_mode: str,
    missing_rate_threshold: float,
    dominant_fraction_threshold: float,
    correlation_threshold: float,
    min_pairwise_observations: int,
) -> PFASFeatureAudit:
    """Screen PFAS inputs from the distinct PFAS in one training dataframe."""

    scope = _unique_pfas_scope(df, candidates)
    unique_pfas_count = int(len(scope))
    kinds: dict[str, str] = {}
    missing_rate: dict[str, float] = {}
    nonmissing_count: dict[str, int] = {}
    dominant_fraction: dict[str, float] = {}
    unique_count: dict[str, int] = {}
    numeric_iqr_bins: dict[str, float] = {}
    selection_reasons: dict[str, str] = {}
    numeric_values: dict[str, pd.Series] = {}

    for feature in candidates:
        if feature not in df.columns:
            selection_reasons[feature] = "missing_from_input"
            continue
        if not unique_pfas_count:
            selection_reasons[feature] = "missing_pfas_identifier"
            continue
        values = scope[feature]
        kind, observed, distinct, dominant, iqr_bins, comparable = _pfas_value_summary(feature, values)
        kinds[feature] = kind
        missing_rate[feature] = float(normalized_missing_mask(values).mean())
        nonmissing_count[feature] = observed
        unique_count[feature] = distinct
        numeric_iqr_bins[feature] = iqr_bins
        dominant_fraction[feature] = dominant
        if feature_screening_mode == "none":
            selection_reasons[feature] = RETAINED_WITHOUT_SCREENING
        elif missing_rate[feature] > _pfas_missing_rate_threshold(feature, missing_rate_threshold):
            selection_reasons[feature] = "pfas_high_missingness"
        elif pd.notna(dominant) and dominant >= dominant_fraction_threshold:
            selection_reasons[feature] = "pfas_near_constant"
        else:
            selection_reasons[feature] = "selected"
        if kind == "numeric":
            numeric_values[feature] = pd.to_numeric(scope[feature], errors="coerce")

    correlated_with: dict[str, list[str]] = {}
    max_absolute_correlation: dict[str, float] = {}
    eligible_numeric = [
        feature
        for feature in candidates
        if _is_retained_feature_reason(selection_reasons.get(feature, ""))
        and feature in numeric_values
    ]
    for index, first in enumerate(eligible_numeric):
        for second in eligible_numeric[index + 1:]:
            pair = pd.concat([numeric_values[first], numeric_values[second]], axis=1).dropna()
            if len(pair) < min_pairwise_observations:
                continue
            if pair.iloc[:, 0].nunique(dropna=True) < 2 or pair.iloc[:, 1].nunique(dropna=True) < 2:
                continue
            strength = abs(float(pair.iloc[:, 0].corr(pair.iloc[:, 1])))
            if not np.isfinite(strength) or strength < correlation_threshold:
                continue
            correlated_with.setdefault(first, []).append(second)
            correlated_with.setdefault(second, []).append(first)
            max_absolute_correlation[first] = max(max_absolute_correlation.get(first, 0.0), strength)
            max_absolute_correlation[second] = max(max_absolute_correlation.get(second, 0.0), strength)

    return PFASFeatureAudit(
        unique_pfas_count=unique_pfas_count,
        candidates=tuple(candidates),
        kinds=kinds,
        missing_rate=missing_rate,
        nonmissing_count=nonmissing_count,
        dominant_fraction=dominant_fraction,
        unique_count=unique_count,
        numeric_iqr_bins=numeric_iqr_bins,
        correlated_with={feature: tuple(sorted(values)) for feature, values in correlated_with.items()},
        max_absolute_correlation=max_absolute_correlation,
        selection_reasons=selection_reasons,
    )


def _pfas_correlation_components(
    candidate_features: list[str],
    analysis: PFASFeatureAudit,
) -> tuple[dict[str, str], dict[str, str], set[str]]:
    """Return deterministic correlation-group selections for training-scope PFAS inputs."""

    candidate_order = list(dict.fromkeys(candidate_features))
    order = {feature: index for index, feature in enumerate(candidate_order)}
    eligible = {
        feature
        for feature in candidate_order
        if _is_retained_feature_reason(analysis.selection_reason(feature))
    }
    if not eligible:
        return {}, {}, set()

    adjacency = {feature: set() for feature in eligible}
    for feature in eligible:
        for correlated in analysis.correlated_with.get(feature, ()):
            if correlated in eligible:
                adjacency[feature].add(correlated)
                adjacency[correlated].add(feature)

    component_ids: dict[str, str] = {}
    selected_features: dict[str, str] = {}
    dropped: set[str] = set()
    visited: set[str] = set()
    component_number = 0
    for start in sorted(eligible, key=order.__getitem__):
        if start in visited or not adjacency[start]:
            continue
        stack = [start]
        component: list[str] = []
        visited.add(start)
        while stack:
            current = stack.pop()
            component.append(current)
            for neighbor in sorted(adjacency[current], key=order.__getitem__, reverse=True):
                if neighbor not in visited:
                    visited.add(neighbor)
                    stack.append(neighbor)
        component.sort(key=order.__getitem__)
        component_number += 1
        component_id = f"corr_{component_number:03d}"
        selected_feature = min(component, key=lambda feature: analysis.correlation_rank_key(feature, order))
        for feature in component:
            component_ids[feature] = component_id
            selected_features[feature] = selected_feature
        dropped.update(feature for feature in component if feature != selected_feature)

    return component_ids, selected_features, dropped


def _non_pfas_correlation_components(
    df: pd.DataFrame,
    candidate_features: list[str],
    *,
    threshold: float,
    min_pairwise_observations: int,
    stratum_coverage: dict[str, float],
) -> tuple[dict[str, str], dict[str, str], set[str], dict[str, tuple[str, ...]], dict[str, float]]:
    """Audit numeric redundancy within each non-PFAS feature family."""

    component_ids: dict[str, str] = {}
    selected_features: dict[str, str] = {}
    dropped: set[str] = set()
    correlated_with: dict[str, set[str]] = {}
    max_absolute_correlation: dict[str, float] = {}
    candidates_by_bucket: dict[str, list[str]] = {}
    for feature in candidate_features:
        candidates_by_bucket.setdefault(feature_bucket(feature), []).append(feature)

    for bucket, features in candidates_by_bucket.items():
        scope = _unique_adsorbent_scope(df, features) if bucket == "Adsorbent properties" else df[features]
        adjacency = {feature: set() for feature in features}
        missing_rate = {
            feature: float(normalized_missing_mask(scope[feature]).mean()) if len(scope) else 1.0
            for feature in features
        }
        iqr_bins: dict[str, float] = {}
        for feature in features:
            values = coerce_numeric(scope[feature]).dropna()
            decimals = int(cfg.NUMERIC_FEATURE_ROUND_DECIMALS.get(feature, cfg.DEFAULT_NUMERIC_ROUND_DECIMALS))
            iqr = float(values.quantile(0.75) - values.quantile(0.25)) if not values.empty else np.nan
            iqr_bins[feature] = iqr / (10.0 ** (-decimals)) if pd.notna(iqr) else np.nan
        for index, first in enumerate(features):
            first_values = coerce_numeric(scope[first])
            for second in features[index + 1:]:
                pair = pd.concat([first_values, coerce_numeric(scope[second])], axis=1).dropna()
                if len(pair) < min_pairwise_observations:
                    continue
                if pair.iloc[:, 0].nunique(dropna=True) < 2 or pair.iloc[:, 1].nunique(dropna=True) < 2:
                    continue
                strength = abs(float(pair.iloc[:, 0].corr(pair.iloc[:, 1])))
                if not np.isfinite(strength) or strength < threshold:
                    continue
                adjacency[first].add(second)
                adjacency[second].add(first)
                correlated_with.setdefault(first, set()).add(second)
                correlated_with.setdefault(second, set()).add(first)
                max_absolute_correlation[first] = max(max_absolute_correlation.get(first, 0.0), strength)
                max_absolute_correlation[second] = max(max_absolute_correlation.get(second, 0.0), strength)

        component_number = 0
        visited: set[str] = set()
        bucket_id = "experimental" if bucket == "Experimental conditions" else "adsorbent"
        order = {feature: index for index, feature in enumerate(features)}
        for start in features:
            if start in visited or not adjacency[start]:
                continue
            stack = [start]
            component: list[str] = []
            visited.add(start)
            while stack:
                current = stack.pop()
                component.append(current)
                for neighbor in sorted(adjacency[current], key=order.__getitem__, reverse=True):
                    if neighbor not in visited:
                        visited.add(neighbor)
                        stack.append(neighbor)
            component.sort(key=order.__getitem__)
            component_number += 1
            component_id = f"corr_{bucket_id}_{component_number:03d}"
            def rank_key(feature: str) -> tuple[float, float, float, int, int]:
                spread = iqr_bins.get(feature, np.nan)
                coverage = stratum_coverage.get(feature, 1.0 - missing_rate[feature])
                return (
                    missing_rate[feature],
                    -coverage,
                    -float(spread) if pd.notna(spread) else float("inf"),
                    cfg.CORRELATION_FEATURE_SCIENTIFIC_PRIORITY.get(feature, 0),
                    order[feature],
                )

            selected_feature = min(component, key=rank_key)
            for feature in component:
                component_ids[feature] = component_id
                selected_features[feature] = selected_feature
            dropped.update(feature for feature in component if feature != selected_feature)

    return (
        component_ids,
        selected_features,
        dropped,
        {feature: tuple(sorted(values)) for feature, values in correlated_with.items()},
        max_absolute_correlation,
    )


def as_float_array(values):
    if hasattr(values, "to_numpy"):
        return values.to_numpy(dtype=float)
    return np.asarray(values, dtype=float)


def is_pfas_structure_feature(column: str) -> bool:
    return str(column).startswith("rdkit_")


def is_pfas_characteristic_feature(column: str) -> bool:
    """Identify PFAS characteristics from the stable model-input schema."""

    name = clean_text(column)
    return bool(
        name in cfg.PFAS_CANDIDATES
        or name in cfg.PFAS_RDKIT_CANDIDATES
        or is_pfas_structure_feature(name)
        or name.startswith("abraham_")
        or name == "Mw (g/mol)"
        or name.endswith("_pred")
    )


def pfas_feature_family(feature: str) -> str:
    """Return the generation family for a PFAS characteristic feature."""

    name = clean_text(feature)
    if name in cfg.PFAS_ATLAS_CANDIDATES:
        return "pfas_atlas"
    if name == "Mw (g/mol)" or name.startswith("rdkit_"):
        return "rdkit"
    if name in cfg.PFAS_OPERA_SCALAR_CANDIDATES or name.endswith("_pred"):
        return "opera"
    if name.startswith("abraham_"):
        return "abraham"
    return ""


def is_excluded_pfas_feature_family(feature: str, policy: str) -> bool:
    """Return whether a PFAS feature is excluded by the requested family policy.

    In addition to the original leave-one-family-out policies, the explicit
    ``rdkit_only`` and ``rdkit_atlas_only`` policies support the final model
    choices established by the feature-family ablation. Non-PFAS features are
    never excluded here.
    """

    family = pfas_feature_family(feature)
    if not family or policy == "all":
        return False
    if policy == "rdkit_only":
        return family != "rdkit"
    if policy == "rdkit_atlas_only":
        return family not in {"rdkit", "pfas_atlas"}
    if policy.startswith("no_"):
        return family == policy.removeprefix("no_")
    return False


def is_unsuitable_model_feature(column: str) -> bool:
    name = str(column)
    return name.startswith(cfg.UNSUITABLE_MODEL_FEATURE_PREFIXES) or any(
        token in name for token in cfg.UNSUITABLE_MODEL_FEATURE_SUBSTRINGS
    )


def is_leakage_col(column: str, target: str) -> bool:
    return column == target or column in cfg.LEAKAGE_EXACT or any(column.startswith(p) for p in cfg.LEAKAGE_PREFIXES)


def feature_bucket(feature: str) -> str:
    if is_pfas_characteristic_feature(feature):
        return "PFAS characteristics"
    if feature in cfg.EXPERIMENTAL_CANDIDATES:
        return "Experimental conditions"
    if (
        feature in cfg.ADSORBENT_COMMON_CANDIDATES
        or feature in cfg.AC_CANDIDATES
        or feature in cfg.RESIN_CANDIDATES
        or feature in cfg.GLOBAL_CANDIDATES
    ):
        return "Adsorbent properties"
    return "Other"


def group_selected_features_by_bucket(selected: list[str]) -> dict[str, list[str]]:
    grouped = {bucket: [] for bucket in cfg.FEATURE_BUCKETS}
    for feature in selected:
        grouped.setdefault(feature_bucket(feature), []).append(feature)
    return {bucket: features for bucket, features in grouped.items() if features}


def candidate_features(
    df: pd.DataFrame,
    model_name: str,
    structure_mode: str,
) -> list[str]:
    schema_pfas = [
        str(column)
        for column in df.columns
        if _is_pfas_scalar_candidate(str(column))
    ]
    pfas_candidates = list(dict.fromkeys(cfg.PFAS_CANDIDATES + cfg.PFAS_RDKIT_CANDIDATES + schema_pfas))
    if structure_mode not in {"descriptors", "all"}:
        pfas_candidates = [
            feature for feature in pfas_candidates if not is_pfas_structure_feature(feature)
        ]
    candidates = cfg.EXPERIMENTAL_CANDIDATES + pfas_candidates + cfg.ADSORBENT_COMMON_CANDIDATES
    if model_name == "AC":
        candidates += cfg.AC_CANDIDATES
    if model_name == "Resin":
        candidates += cfg.RESIN_CANDIDATES
    if model_name == "Global":
        candidates += cfg.GLOBAL_CANDIDATES
    return list(dict.fromkeys(candidates))


def conditional_numeric_applicability_features(numeric_features: list[str]) -> list[str]:
    """Return generated state inputs for conditionally applicable numerics.

    The continuous field retains its observed value while the companion input
    separates a chemically inapplicable field (0) from an applicable one (1).
    An unknown adsorbent category remains missing rather than being guessed.
    """

    return [
        f"{feature}{cfg.ADSORBENT_APPLICABILITY_INDICATOR_SUFFIX}"
        for feature in numeric_features
        if feature in cfg.ADSORBENT_SPECIFIC_FEATURE_APPLICABILITY
    ]


def _conditional_applicability_state(df: pd.DataFrame, feature: str) -> pd.Series:
    """Return 1 for applicable, 0 for known-inapplicable, and NaN if unknown."""

    applicable_categories = cfg.ADSORBENT_SPECIFIC_FEATURE_APPLICABILITY.get(feature)
    state = pd.Series(np.nan, index=df.index, dtype=float)
    if not applicable_categories or "adsorbent_category" not in df.columns:
        return state
    categories = df["adsorbent_category"].map(clean_text)
    known_categories = set(cfg.MODEL_CATEGORIES["Global"])
    known = categories.isin(known_categories)
    state.loc[known] = categories.loc[known].isin(applicable_categories).astype(float)
    return state


def _reject_reported_inapplicable_values(
    df: pd.DataFrame,
    feature: str,
    applicability: pd.Series,
) -> None:
    """Fail loudly if the database contradicts an explicit applicability rule."""

    reported = ~normalized_missing_mask(df[feature])
    invalid = reported & applicability.eq(0.0)
    if invalid.any():
        raise ValueError(
            f"{feature} is reported for {int(invalid.sum())} adsorbent rows where "
            "the configured family rule says it is not applicable. Review "
            "ADSORBENT_SPECIFIC_FEATURE_APPLICABILITY rather than silently "
            "discarding those values."
        )


def all_candidate_features(
    df: pd.DataFrame,
    model_name: str,
    structure_mode: str,
) -> list[str]:
    return candidate_features(df, model_name, structure_mode)


def _distinct_nonempty(series: pd.Series | None, mask: pd.Series | None = None) -> int:
    if series is None:
        return 0
    scoped = series if mask is None else series[mask]
    values = scoped.map(clean_text)
    return int(values[values != ""].nunique(dropna=True))


# A material property is reported once per study per material, so the
# independent reporting unit for an adsorbent feature pairs the study with the
# resolved material identity.  ``adsorbent_identity_key`` carries that
# resolution: it is built from the canonical commercial name, so a resin
# labelled "AER" in one study and "PFA694E" in another is one material, while
# study-local labels stay study-local.  Raw ``adsorbent_id`` is the study's own
# spelling and is only a fallback for frames written before identity
# resolution; keying on it would split one material into several units as soon
# as a study spelled it two ways.
ADSORBENT_UNIT_IDENTITY_COLUMNS = ("adsorbent_identity_key", "adsorbent_id")


def adsorbent_unit_keys(df: pd.DataFrame) -> pd.Series:
    """Return the canonical study-specific adsorbent unit, or "" when unusable.

    This is the single definition of that unit.  Every availability, support,
    and coverage statistic reported for an adsorbent feature must use it, so
    that the audit workbook and the training feature audit cannot drift apart
    on what counts as one reporting unit.
    """
    study = df.get("study_no", pd.Series("", index=df.index, dtype="object")).map(clean_text)
    adsorbent = pd.Series("", index=df.index, dtype="object")
    for column in ADSORBENT_UNIT_IDENTITY_COLUMNS:
        if column in df.columns:
            adsorbent = df[column].map(clean_text)
            break
    valid = (study != "") & (adsorbent != "")
    return (study + "|" + adsorbent).where(valid, "")


def _unique_adsorbent_scope(df: pd.DataFrame, candidates: list[str]) -> pd.DataFrame:
    """Return one deterministic property record per study--adsorbent unit."""

    keys = adsorbent_unit_keys(df)
    usable = keys.ne("")
    if not usable.any():
        return pd.DataFrame(columns=candidates)
    scoped = df.loc[usable, candidates].copy()
    scoped.insert(0, "__study_adsorbent_key", keys.loc[usable].to_numpy())
    rows: list[dict[str, Any]] = []
    for key, group in scoped.groupby("__study_adsorbent_key", sort=True):
        row = {"__study_adsorbent_key": key}
        for feature in candidates:
            row[feature] = _first_nonmissing(group[feature])
        rows.append(row)
    return (
        pd.DataFrame(rows).set_index("__study_adsorbent_key")
        if rows
        else pd.DataFrame(columns=candidates)
    )


ADSORBENT_MATERIAL_IDENTITY_COLUMNS = ("adsorbent_identity_key", "adsorbent_id")


def screening_entity_keys(df: pd.DataFrame, bucket: str) -> pd.Series:
    """Return the independent unit that one feature level is counted over.

    Level support asks how many separate things show a value, so the unit has
    to match what the feature describes.  A material property is counted over
    adsorbent identities, an experimental condition over studies, and a PFAS
    descriptor over molecules.  Rows outside any usable unit return "" and are
    not counted, exactly as in :func:`_distinct_nonempty`.

    This is deliberately the material identity rather than the study-specific
    adsorbent unit used by :func:`adsorbent_unit_keys`.  Reporting coverage asks
    how often a field was populated, so it counts each study's report of a
    material separately.  Level support asks how many distinct materials carry
    a value, so one resin described by six papers is one unit, not six.
    ``adsorbent_identity_key`` already keeps a named material together across
    studies and holds a study-local label apart, so it is that unit.
    """
    if bucket == "PFAS characteristics":
        return df.get("PFAS_name", pd.Series("", index=df.index, dtype="object")).map(normalize_pfas_key)
    if bucket == "Adsorbent properties":
        for column in ADSORBENT_MATERIAL_IDENTITY_COLUMNS:
            if column in df.columns:
                keys = df[column].map(clean_text)
                if keys.ne("").any():
                    return keys
        return pd.Series("", index=df.index, dtype="object")
    return df.get("study_no", pd.Series("", index=df.index, dtype="object")).map(clean_text)


def _level_entity_counts(levels: pd.Series, entity_keys: pd.Series | None) -> pd.Series | None:
    """Count distinct nonblank entities behind each observed level.

    ``None`` means the caller could not supply a unit, so the entity floor is
    not applied and level support falls back to row counts alone.  A bucket
    whose identifier is missing altogether is rejected by
    :func:`_support_selection_reason` instead, where the reason names the
    missing identifier.
    """
    if entity_keys is None:
        return None
    aligned = entity_keys.reindex(levels.index)
    usable = aligned.notna() & aligned.astype("object").ne("")
    if not usable.any():
        return None
    return aligned[usable].groupby(levels[usable]).nunique()


def _supported_levels(
    counts: pd.Series,
    entity_counts: pd.Series | None,
    min_categorical_level_rows: int,
    min_categorical_level_entities: int,
) -> tuple[int, int | float]:
    """Return supported-level count and the entity support of the rarest level."""
    supported = counts >= min_categorical_level_rows
    if entity_counts is None:
        return int(supported.sum()), np.nan
    entities = entity_counts.reindex(counts.index).fillna(0).astype(int)
    supported = supported & (entities >= min_categorical_level_entities)
    return int(supported.sum()), (int(entities.min()) if len(entities) else np.nan)


def _support_summary(df: pd.DataFrame, value_mask: pd.Series) -> dict[str, int | float]:
    pfas_keys = df.get("PFAS_name", pd.Series("", index=df.index, dtype="object")).map(normalize_pfas_key)
    has_pfas = pfas_keys != ""
    study_no = df["study_no"] if "study_no" in df.columns else None
    study_adsorbent = adsorbent_unit_keys(df)
    total_pfas = int(pfas_keys[has_pfas].nunique(dropna=True))
    total_studies = _distinct_nonempty(study_no)
    total_adsorbents = _distinct_nonempty(study_adsorbent)
    reporting_pfas = int(pfas_keys[value_mask & has_pfas].nunique(dropna=True))
    reporting_studies = _distinct_nonempty(study_no, value_mask)
    reporting_adsorbents = _distinct_nonempty(study_adsorbent, value_mask)
    return {
        "nonmissing_rows": int(value_mask.sum()),
        "row_fraction": float(value_mask.mean()) if len(df) else np.nan,
        "total_pfas": total_pfas,
        "reporting_pfas": reporting_pfas,
        "pfas_coverage": reporting_pfas / total_pfas if total_pfas else np.nan,
        "total_studies": total_studies,
        "reporting_studies": reporting_studies,
        "study_coverage": reporting_studies / total_studies if total_studies else np.nan,
        "total_adsorbents": total_adsorbents,
        "reporting_adsorbents": reporting_adsorbents,
        "adsorbent_coverage": reporting_adsorbents / total_adsorbents if total_adsorbents else np.nan,
    }


def _numeric_diversity(
    feature: str,
    numeric: pd.Series,
    value_mask: pd.Series,
    min_categorical_level_rows: int = cfg.DEFAULT_MIN_CATEGORICAL_LEVEL_ROWS,
    entity_keys: pd.Series | None = None,
    min_categorical_level_entities: int = cfg.DEFAULT_MIN_CATEGORICAL_LEVEL_ENTITIES,
) -> dict[str, int | float]:
    decimals = int(cfg.NUMERIC_FEATURE_ROUND_DECIMALS.get(feature, cfg.DEFAULT_NUMERIC_ROUND_DECIMALS))
    rounded = numeric[value_mask].astype(float).round(decimals)
    iqr = float(rounded.quantile(0.75) - rounded.quantile(0.25)) if not rounded.empty else np.nan
    # A 0/1 indicator holds its entire signal in two values, so it is screened
    # on how well both values are represented, exactly as a two-level
    # categorical is.  Numeric value-diversity and spread thresholds exist to
    # drop near-constant continuous features and would always reject it.
    binary = _is_binary_indicator(rounded)
    counts = rounded.value_counts(dropna=True)
    supported_levels: int | float = np.nan
    rarest_level_entities: int | float = np.nan
    if binary:
        supported_levels, rarest_level_entities = _supported_levels(
            counts,
            _level_entity_counts(rounded, entity_keys),
            min_categorical_level_rows,
            min_categorical_level_entities,
        )
    return {
        "unique_values": int(rounded.nunique(dropna=True)),
        "is_binary_indicator": binary,
        "supported_category_levels": supported_levels,
        "rarest_level_entities": rarest_level_entities,
        "numeric_round_decimals": decimals,
        "numeric_iqr": iqr,
        "numeric_iqr_bins": iqr / (10.0 ** (-decimals)) if pd.notna(iqr) else np.nan,
    }


def _categorical_diversity(
    values: pd.Series,
    min_categorical_level_rows: int,
    entity_keys: pd.Series | None = None,
    min_categorical_level_entities: int = cfg.DEFAULT_MIN_CATEGORICAL_LEVEL_ENTITIES,
) -> dict[str, int | float]:
    levels = values.map(clean_text)
    levels = levels[levels != ""]
    counts = levels.value_counts(dropna=True)
    supported_levels, rarest_level_entities = _supported_levels(
        counts,
        _level_entity_counts(levels, entity_keys),
        min_categorical_level_rows,
        min_categorical_level_entities,
    )
    return {
        "unique_values": int(len(counts)),
        "is_binary_indicator": False,
        "supported_category_levels": supported_levels,
        "rarest_level_entities": rarest_level_entities,
        "numeric_round_decimals": np.nan,
        "numeric_iqr": np.nan,
        "numeric_iqr_bins": np.nan,
    }


def _is_binary_indicator(values: pd.Series) -> bool:
    unique = pd.Series(values.dropna().unique(), dtype=float)
    return bool(len(unique) <= 2 and unique.isin([0.0, 1.0]).all())


def _feature_kind(
    numeric: pd.Series,
    missing: pd.Series,
    numeric_ratio: float,
    binary_indicator_handling: str,
) -> str:
    """Decide whether one candidate input is modeled as numeric or categorical.

    A column that is not reliably numeric is categorical.  A 0/1 indicator is
    numeric unless ``binary_indicator_handling`` asks for the second one-hot
    encoding described in ``logkd_config.BINARY_INDICATOR_HANDLING_CHOICES``.
    """

    if binary_indicator_handling not in cfg.BINARY_INDICATOR_HANDLING_CHOICES:
        raise ValueError(
            "binary_indicator_handling must be one of "
            f"{cfg.BINARY_INDICATOR_HANDLING_CHOICES}; got {binary_indicator_handling!r}."
        )
    if numeric_ratio < 0.80:
        return "categorical"
    if binary_indicator_handling == "categorical" and _is_binary_indicator(numeric[~missing]):
        return "categorical"
    return "numeric"


def _diversity_selection_reason(
    kind: str,
    diversity: dict[str, int | float],
    min_categorical_levels: int,
    min_numeric_distinct_values: int,
    min_numeric_iqr_bins: float,
) -> str:
    if kind == "categorical" or diversity.get("is_binary_indicator"):
        return "selected" if diversity["supported_category_levels"] >= min_categorical_levels else "low_supported_category_levels"
    if diversity["unique_values"] < min_numeric_distinct_values:
        return "low_numeric_value_diversity"
    if diversity["numeric_iqr_bins"] < min_numeric_iqr_bins:
        return "low_numeric_spread"
    return "selected"


def _support_selection_reason(
    feature: str,
    bucket: str,
    support: dict[str, int | float],
    min_nonmissing_rows: int,
    min_reporting_study_fraction: float,
    min_reporting_adsorbent_fraction: float,
) -> str:
    if support["nonmissing_rows"] < min_nonmissing_rows:
        return "low_nonmissing_row_support"
    if support["total_studies"] == 0:
        return "missing_study_identifier"
    if support["study_coverage"] < min_reporting_study_fraction:
        return "low_reporting_study_coverage"
    if bucket == "Adsorbent properties":
        if support["total_adsorbents"] == 0:
            return "missing_adsorbent_identifier"
        if support["adsorbent_coverage"] < min_reporting_adsorbent_fraction:
            return "low_reporting_adsorbent_coverage"
    return "selected"


def _validate_selection_thresholds(
    min_nonmissing_rows: int,
    min_reporting_study_fraction: float,
    min_reporting_adsorbent_fraction: float,
    min_categorical_levels: int,
    min_categorical_level_rows: int,
    min_categorical_level_entities: int,
    min_numeric_distinct_values: int,
    min_numeric_iqr_bins: float,
) -> None:
    if min_nonmissing_rows < 1:
        raise ValueError("min_nonmissing_rows must be at least 1.")
    for name, value in (
        ("min_reporting_study_fraction", min_reporting_study_fraction),
        ("min_reporting_adsorbent_fraction", min_reporting_adsorbent_fraction),
    ):
        if not 0.0 < value <= 1.0:
            raise ValueError(f"{name} must be greater than 0 and at most 1.")
    if min_categorical_levels < 2 or min_categorical_level_rows < 1:
        raise ValueError("Categorical support thresholds must allow at least two levels and one row per level.")
    if min_categorical_level_entities < 1:
        raise ValueError("min_categorical_level_entities must be at least 1.")
    if min_numeric_distinct_values < 2 or min_numeric_iqr_bins < 0:
        raise ValueError("Numeric diversity thresholds must allow at least two values and a nonnegative IQR.")


def feature_availability_audit(
    df: pd.DataFrame,
    model_name: str,
    pfas_structure_features: str = "all",
    binary_indicator_handling: str = cfg.DEFAULT_BINARY_INDICATOR_HANDLING,
) -> tuple[list[str], list[str], list[str], pd.DataFrame]:
    """Describe candidate-input availability without making selection decisions.

    Unlike :func:`feature_audit`, this function applies no support, diversity,
    PFAS-family, or correlation policy.  It is for whole-database QA reports;
    model training must make its own feature-selection decisions inside each
    training partition.  Variables in :data:`EQUILIBRIUM_DOMAIN_EXCLUSIONS` are
    omitted entirely: they can never enter an equilibrium logKd model, so they
    are not part of the input space this audit describes.
    """
    candidates = [
        feature
        for feature in all_candidate_features(df, model_name, pfas_structure_features)
        if feature not in EQUILIBRIUM_DOMAIN_EXCLUSIONS
    ]
    available_features: list[str] = []
    numeric_features: list[str] = []
    categorical_features: list[str] = []
    rows: list[dict[str, Any]] = []

    for feature in candidates:
        bucket = feature_bucket(feature)
        if feature not in df.columns:
            rows.append(
                {
                    "feature": feature,
                    "bucket": bucket,
                    "kind": pd.NA,
                    "availability_status": "missing_from_input",
                    "numeric_ratio": np.nan,
                    "nonmissing_rows": 0,
                    "row_fraction": 0.0 if len(df) else np.nan,
                    "unique_values": 0,
                    "reported_category_levels": np.nan,
                    "numeric_round_decimals": np.nan,
                    "numeric_iqr": np.nan,
                    "numeric_iqr_bins": np.nan,
                    **_support_summary(df, pd.Series(False, index=df.index)),
                }
            )
            continue

        missing = normalized_missing_mask(df[feature])
        numeric = coerce_numeric(df[feature])
        numeric_ratio = float(numeric.notna().sum() / max(int((~missing).sum()), 1))
        kind = _feature_kind(numeric, missing, numeric_ratio, binary_indicator_handling)
        value_mask = numeric.notna() if kind == "numeric" else ~missing
        support = _support_summary(df, value_mask)
        entity_keys = screening_entity_keys(df, bucket)
        if kind == "numeric":
            diversity = _numeric_diversity(feature, numeric, value_mask, entity_keys=entity_keys)
            reported_category_levels: int | float = np.nan
        else:
            reported_levels = df.loc[value_mask, feature].map(clean_text)
            reported_category_levels = int(reported_levels[reported_levels.ne("")].nunique())
            diversity = {
                "unique_values": reported_category_levels,
                "numeric_round_decimals": np.nan,
                "numeric_iqr": np.nan,
                "numeric_iqr_bins": np.nan,
            }

        available_features.append(feature)
        if kind == "numeric":
            numeric_features.append(feature)
        else:
            categorical_features.append(feature)
        rows.append(
            {
                "feature": feature,
                "bucket": bucket,
                "kind": kind,
                "availability_status": "assessed",
                "numeric_ratio": numeric_ratio,
                **support,
                **diversity,
                "reported_category_levels": reported_category_levels,
            }
        )

    return available_features, numeric_features, categorical_features, pd.DataFrame(rows)


def add_feature_selection_arguments(parser: Any) -> None:
    parser.add_argument(
        "--feature-screening-mode",
        choices=FEATURE_SCREENING_MODES,
        default="standard",
        help=(
            "standard applies availability and variation screening; none retains all "
            "model-eligible features before the configured correlation policy."
        ),
    )
    parser.add_argument(
        "--pfas-missing-rate-threshold",
        type=float,
        default=cfg.DEFAULT_PFAS_MISSING_RATE_THRESHOLD,
    )
    parser.add_argument(
        "--pfas-dominant-fraction-threshold",
        type=float,
        default=cfg.DEFAULT_PFAS_DOMINANT_FRACTION_THRESHOLD,
    )
    parser.add_argument(
        "--correlation-threshold",
        type=float,
        default=cfg.DEFAULT_CORRELATION_THRESHOLD,
        help="absolute Pearson correlation that defines one high-correlation feature group in every family",
    )
    parser.add_argument(
        "--min-pairwise-observations",
        type=int,
        default=cfg.DEFAULT_MIN_PAIRWISE_OBSERVATIONS,
        help="minimum complete observations required to assess a feature pair in every family",
    )
    parser.add_argument(
        "--correlated-feature-handling",
        choices=cfg.CORRELATED_FEATURE_HANDLING_CHOICES,
        default=cfg.DEFAULT_CORRELATED_FEATURE_HANDLING,
        help="use all screened features or select one quality-ranked feature per high-correlation group",
    )
    parser.add_argument(
        "--pfas-feature-family-policy",
        choices=cfg.PFAS_FEATURE_FAMILY_POLICIES,
        default=cfg.DEFAULT_PFAS_FEATURE_FAMILY_POLICY,
        help=(
            "PFAS family policy: use all, exclude one family, or use only "
            "RDKit / RDKit-plus-Atlas PFAS characteristics"
        ),
    )
    parser.add_argument(
        "--exclude-feature-buckets",
        nargs="+",
        choices=cfg.FEATURE_BUCKETS,
        default=(),
        metavar="BUCKET",
        help=(
            "Exclude every candidate in one or more feature buckets before "
            "training-scope screening. Intended for paired bucket-ablation studies."
        ),
    )
    parser.add_argument(
        "--exclude-features",
        nargs="+",
        default=(),
        metavar="FEATURE",
        help=(
            "Explicitly exclude one or more exact candidate feature names. "
            "Intended for paired feature-ablation studies; unlike the central "
            "equilibrium-domain exclusions, these exclusions are run-specific."
        ),
    )
    parser.add_argument("--min-nonmissing-rows", type=int, default=cfg.DEFAULT_MIN_NONMISSING_ROWS)
    parser.add_argument("--min-reporting-study-fraction", type=float, default=cfg.DEFAULT_MIN_REPORTING_STUDY_FRACTION)
    parser.add_argument("--min-reporting-adsorbent-fraction", type=float, default=cfg.DEFAULT_MIN_REPORTING_ADSORBENT_FRACTION)
    parser.add_argument("--min-categorical-levels", type=int, default=cfg.DEFAULT_MIN_CATEGORICAL_LEVELS)
    parser.add_argument("--min-categorical-level-rows", type=int, default=cfg.DEFAULT_MIN_CATEGORICAL_LEVEL_ROWS)
    parser.add_argument(
        "--min-categorical-level-entities",
        type=int,
        default=cfg.DEFAULT_MIN_CATEGORICAL_LEVEL_ENTITIES,
        help=(
            "Distinct independent units a level must appear in before it counts as "
            "supported: adsorbent identities for a material property, studies for an "
            "experimental condition, PFAS for a molecular descriptor."
        ),
    )
    parser.add_argument("--min-numeric-distinct-values", type=int, default=cfg.DEFAULT_MIN_NUMERIC_DISTINCT_VALUES)
    parser.add_argument("--min-numeric-iqr-bins", type=float, default=cfg.DEFAULT_MIN_NUMERIC_IQR_BINS)
    parser.add_argument(
        "--binary-indicator-handling",
        choices=cfg.BINARY_INDICATOR_HANDLING_CHOICES,
        default=cfg.DEFAULT_BINARY_INDICATOR_HANDLING,
        help=(
            "numeric keeps a 0/1 input as one column and leaves an unreported value "
            "missing; categorical one-hot encodes it again into present/absent/<missing> "
            "columns, which models 'never reported' as a category."
        ),
    )


def feature_selection_kwargs(args: Any) -> dict[str, Any]:
    return {
        "feature_screening_mode": args.feature_screening_mode,
        "pfas_missing_rate_threshold": args.pfas_missing_rate_threshold,
        "pfas_dominant_fraction_threshold": args.pfas_dominant_fraction_threshold,
        "correlation_threshold": args.correlation_threshold,
        "min_pairwise_observations": args.min_pairwise_observations,
        "correlated_feature_handling": args.correlated_feature_handling,
        "pfas_feature_family_policy": getattr(
            args,
            "pfas_feature_family_policy",
            cfg.DEFAULT_PFAS_FEATURE_FAMILY_POLICY,
        ),
        "binary_indicator_handling": getattr(
            args,
            "binary_indicator_handling",
            cfg.DEFAULT_BINARY_INDICATOR_HANDLING,
        ),
        "excluded_feature_buckets": tuple(getattr(args, "exclude_feature_buckets", ())),
        "excluded_features": tuple(getattr(args, "exclude_features", ())),
        "min_nonmissing_rows": args.min_nonmissing_rows,
        "min_reporting_study_fraction": args.min_reporting_study_fraction,
        "min_reporting_adsorbent_fraction": args.min_reporting_adsorbent_fraction,
        "min_categorical_levels": args.min_categorical_levels,
        "min_categorical_level_rows": args.min_categorical_level_rows,
        "min_categorical_level_entities": args.min_categorical_level_entities,
        "min_numeric_distinct_values": args.min_numeric_distinct_values,
        "min_numeric_iqr_bins": args.min_numeric_iqr_bins,
    }


def feature_audit(
    df: pd.DataFrame,
    model_name: str,
    target: str,
    pfas_structure_features: str = "all",
    min_nonmissing_rows: int = cfg.DEFAULT_MIN_NONMISSING_ROWS,
    min_reporting_study_fraction: float = cfg.DEFAULT_MIN_REPORTING_STUDY_FRACTION,
    min_reporting_adsorbent_fraction: float = cfg.DEFAULT_MIN_REPORTING_ADSORBENT_FRACTION,
    min_categorical_levels: int = cfg.DEFAULT_MIN_CATEGORICAL_LEVELS,
    min_categorical_level_rows: int = cfg.DEFAULT_MIN_CATEGORICAL_LEVEL_ROWS,
    min_categorical_level_entities: int = cfg.DEFAULT_MIN_CATEGORICAL_LEVEL_ENTITIES,
    min_numeric_distinct_values: int = cfg.DEFAULT_MIN_NUMERIC_DISTINCT_VALUES,
    min_numeric_iqr_bins: float = cfg.DEFAULT_MIN_NUMERIC_IQR_BINS,
    feature_screening_mode: str = "standard",
    pfas_missing_rate_threshold: float = cfg.DEFAULT_PFAS_MISSING_RATE_THRESHOLD,
    pfas_dominant_fraction_threshold: float = cfg.DEFAULT_PFAS_DOMINANT_FRACTION_THRESHOLD,
    correlation_threshold: float = cfg.DEFAULT_CORRELATION_THRESHOLD,
    min_pairwise_observations: int = cfg.DEFAULT_MIN_PAIRWISE_OBSERVATIONS,
    correlated_feature_handling: str = cfg.DEFAULT_CORRELATED_FEATURE_HANDLING,
    pfas_feature_family_policy: str = cfg.DEFAULT_PFAS_FEATURE_FAMILY_POLICY,
    binary_indicator_handling: str = cfg.DEFAULT_BINARY_INDICATOR_HANDLING,
    excluded_feature_buckets: tuple[str, ...] = (),
    excluded_features: tuple[str, ...] = (),
) -> tuple[list[str], list[str], list[str], pd.DataFrame]:
    if feature_screening_mode not in FEATURE_SCREENING_MODES:
        raise ValueError(
            f"feature_screening_mode must be one of {FEATURE_SCREENING_MODES}; "
            f"got {feature_screening_mode!r}."
        )
    _validate_selection_thresholds(
        min_nonmissing_rows,
        min_reporting_study_fraction,
        min_reporting_adsorbent_fraction,
        min_categorical_levels,
        min_categorical_level_rows,
        min_categorical_level_entities,
        min_numeric_distinct_values,
        min_numeric_iqr_bins,
    )
    if not 0.0 <= pfas_missing_rate_threshold < 1.0:
        raise ValueError("pfas_missing_rate_threshold must be in [0, 1).")
    if not 0.0 < pfas_dominant_fraction_threshold <= 1.0:
        raise ValueError("pfas_dominant_fraction_threshold must be in (0, 1].")
    if not 0.0 < correlation_threshold <= 1.0:
        raise ValueError("correlation_threshold must be in (0, 1].")
    if min_pairwise_observations < 2:
        raise ValueError("min_pairwise_observations must be at least 2.")
    if correlated_feature_handling not in cfg.CORRELATED_FEATURE_HANDLING_CHOICES:
        raise ValueError(
            f"correlated_feature_handling must be one of {cfg.CORRELATED_FEATURE_HANDLING_CHOICES}; "
            f"got {correlated_feature_handling!r}."
        )
    if pfas_feature_family_policy not in cfg.PFAS_FEATURE_FAMILY_POLICIES:
        raise ValueError(
            f"pfas_feature_family_policy must be one of {cfg.PFAS_FEATURE_FAMILY_POLICIES}; "
            f"got {pfas_feature_family_policy!r}."
        )
    excluded_buckets = tuple(dict.fromkeys(excluded_feature_buckets))
    unknown_buckets = sorted(set(excluded_buckets).difference(cfg.FEATURE_BUCKETS))
    if unknown_buckets:
        raise ValueError(
            f"excluded_feature_buckets contains unknown buckets: {unknown_buckets}."
        )

    candidate_list = all_candidate_features(df, model_name, pfas_structure_features)
    explicit_exclusions = tuple(dict.fromkeys(excluded_features))
    unknown_exclusions = sorted(set(explicit_exclusions).difference(candidate_list))
    if unknown_exclusions:
        raise ValueError(
            "excluded_features contains unknown/noncandidate feature names: "
            f"{unknown_exclusions}."
        )
    explicit_exclusion_set = frozenset(explicit_exclusions)
    excluded_bucket_set = frozenset(excluded_buckets)
    screened_candidate_list = [
        feature
        for feature in candidate_list
        if feature_bucket(feature) not in excluded_bucket_set
        and feature not in explicit_exclusion_set
        and feature not in EQUILIBRIUM_DOMAIN_EXCLUSIONS
    ]
    candidate_order = {feature: index for index, feature in enumerate(candidate_list)}
    pfas_candidates = [
        feature for feature in screened_candidate_list
        if feature_bucket(feature) == "PFAS characteristics"
    ]
    pfas_audit = audit_pfas_training_features(
        df,
        pfas_candidates,
        feature_screening_mode=feature_screening_mode,
        missing_rate_threshold=pfas_missing_rate_threshold,
        dominant_fraction_threshold=pfas_dominant_fraction_threshold,
        correlation_threshold=correlation_threshold,
        min_pairwise_observations=min_pairwise_observations,
    )
    correlation_candidate_list = [
        feature for feature in pfas_candidates
        if not is_excluded_pfas_feature_family(feature, pfas_feature_family_policy)
    ]
    correlation_component_ids, correlation_selected_features, correlation_dropped = _pfas_correlation_components(
        correlation_candidate_list,
        pfas_audit,
    )
    rows: list[dict[str, Any]] = []
    selected: list[str] = []
    for feature in candidate_list:
        bucket = feature_bucket(feature)
        family = pfas_feature_family(feature)
        is_pfas_feature = bucket == "PFAS characteristics"
        pfas_metrics = pfas_audit.metrics(feature) if is_pfas_feature else {}
        base_row = {
            "feature": feature,
            "bucket": bucket,
            "feature_screening_mode": feature_screening_mode,
            "screening_thresholds_applied": feature_screening_mode == "standard",
            "excluded_feature_buckets": "; ".join(excluded_buckets),
            "explicitly_excluded_features": "; ".join(explicit_exclusions),
            "equilibrium_domain_exclusion_reason": EQUILIBRIUM_DOMAIN_EXCLUSIONS.get(feature, ""),
            "adsorbent_specific_applicability": feature in cfg.ADSORBENT_SPECIFIC_FEATURE_APPLICABILITY,
            "applicable_adsorbent_categories": "; ".join(
                cfg.ADSORBENT_SPECIFIC_FEATURE_APPLICABILITY.get(feature, ())
            ),
            "not_applicable_encoding": (
                cfg.ADSORBENT_NOT_APPLICABLE_CATEGORY
                if feature in cfg.ADSORBENT_SPECIFIC_FEATURE_APPLICABILITY
                else ""
            ),
            "pfas_feature_family": family,
            "pfas_feature_family_policy": pfas_feature_family_policy,
            **pfas_metrics,
            "correlated_feature_handling": correlated_feature_handling,
            "correlation_component_id": correlation_component_ids.get(feature, ""),
            "correlation_selected_feature": correlation_selected_features.get(feature, ""),
            "correlation_feature_dropped": bool(
                is_pfas_feature
                and correlated_feature_handling == "select_one_per_group"
                and feature in correlation_dropped
            ),
            "pfas_missing_rate_threshold": (
                _pfas_missing_rate_threshold(feature, pfas_missing_rate_threshold)
                if is_pfas_feature
                else np.nan
            ),
            "pfas_dominant_fraction_threshold": (
                pfas_dominant_fraction_threshold if is_pfas_feature else np.nan
            ),
            "correlation_threshold": correlation_threshold,
            "min_pairwise_observations": min_pairwise_observations,
            "correlation_scope": "unique_pfas" if is_pfas_feature else "",
            "correlation_selection_order": (
                "lowest_missingness; highest_stratum_coverage; greatest_usable_variation; "
                "scientific_priority; configured_candidate_order"
            ),
            "correlation_scientific_priority": cfg.CORRELATION_FEATURE_SCIENTIFIC_PRIORITY.get(feature, 0),
            "correlation_candidate_order": candidate_order[feature],
            "training_correlated_with": (
                pfas_metrics.get("pfas_training_correlated_with", "") if is_pfas_feature else ""
            ),
            "training_max_absolute_correlation": (
                pfas_metrics.get("pfas_training_max_absolute_correlation", np.nan)
                if is_pfas_feature
                else np.nan
            ),
        }
        if bucket in excluded_bucket_set:
            rows.append(
                {
                    **base_row,
                    "selected": False,
                    "reason": "feature_bucket_excluded_by_ablation",
                    "availability_independent_support_reason": "not_assessed",
                    "availability_independent_support_pass": pd.NA,
                    "variation_support_reason": "not_assessed",
                    "variation_support_pass": pd.NA,
                }
            )
            continue
        if feature not in df.columns:
            rows.append({**base_row, "selected": False, "reason": "missing_from_input"})
            continue
        if feature in EQUILIBRIUM_DOMAIN_EXCLUSIONS:
            rows.append(
                {
                    **base_row,
                    "selected": False,
                    "reason": EQUILIBRIUM_DOMAIN_EXCLUSIONS[feature],
                    "availability_independent_support_reason": "not_assessed",
                    "availability_independent_support_pass": pd.NA,
                    "variation_support_reason": "not_assessed",
                    "variation_support_pass": pd.NA,
                }
            )
            continue
        if feature in explicit_exclusion_set:
            rows.append(
                {
                    **base_row,
                    "selected": False,
                    "reason": "feature_excluded_by_ablation",
                    "availability_independent_support_reason": "not_assessed",
                    "availability_independent_support_pass": pd.NA,
                    "variation_support_reason": "not_assessed",
                    "variation_support_pass": pd.NA,
                }
            )
            continue
        if feature in cfg.IDENTIFIER_COLUMNS or is_leakage_col(feature, target):
            rows.append({**base_row, "selected": False, "reason": "identifier_or_leakage"})
            continue
        if is_unsuitable_model_feature(feature):
            rows.append({**base_row, "selected": False, "reason": "unsuitable_range_or_uncertainty"})
            continue
        if is_excluded_pfas_feature_family(feature, pfas_feature_family_policy):
            rows.append({
                **base_row,
                "selected": False,
                "reason": "pfas_feature_family_excluded",
                "availability_independent_support_reason": "pfas_feature_family_excluded",
                "availability_independent_support_pass": False,
                "variation_support_reason": "pfas_feature_family_excluded",
                "variation_support_pass": False,
            })
            continue

        missing = normalized_missing_mask(df[feature])
        numeric = coerce_numeric(df[feature])
        numeric_ratio = float(numeric.notna().sum() / max(int((~missing).sum()), 1))
        kind = _feature_kind(numeric, missing, numeric_ratio, binary_indicator_handling)
        value_mask = numeric.notna() if kind == "numeric" else ~missing
        entity_keys = screening_entity_keys(df, bucket)
        diversity = (
            _numeric_diversity(
                feature,
                numeric,
                value_mask,
                min_categorical_level_rows,
                entity_keys,
                min_categorical_level_entities,
            )
            if kind == "numeric"
            else _categorical_diversity(
                df.loc[value_mask, feature],
                min_categorical_level_rows,
                entity_keys,
                min_categorical_level_entities,
            )
        )
        # A 0/1 indicator is screened on level support even when it is modeled
        # as numeric, so the level thresholds apply to it and must be reported.
        screened_on_levels = kind == "categorical" or bool(diversity["is_binary_indicator"])
        support = _support_summary(df, value_mask)
        if is_pfas_feature:
            pfas_reason = pfas_audit.selection_reason(feature)
            if correlated_feature_handling == "select_one_per_group" and feature in correlation_dropped:
                pfas_reason = "correlated_feature_group_pruned"
            selected_flag = _is_retained_feature_reason(pfas_reason)
            screening_not_applied = pfas_reason == RETAINED_WITHOUT_SCREENING
            if selected_flag:
                selected.append(feature)
            rows.append(
                {
                    **base_row,
                    "selected": selected_flag,
                    "reason": pfas_reason,
                    "availability_independent_support_reason": (
                        SCREENING_NOT_APPLIED if screening_not_applied else pfas_reason
                    ),
                    "availability_independent_support_pass": (
                        pd.NA if screening_not_applied else selected_flag
                    ),
                    "variation_support_reason": (
                        SCREENING_NOT_APPLIED if screening_not_applied else pfas_reason
                    ),
                    "variation_support_pass": pd.NA if screening_not_applied else selected_flag,
                    "kind": kind,
                    "numeric_ratio": numeric_ratio,
                    "nonmissing_rows": support["nonmissing_rows"],
                    "row_fraction": support["row_fraction"],
                    "unique_values": diversity["unique_values"],
                    "is_binary_indicator": diversity["is_binary_indicator"],
                    "supported_category_levels": diversity["supported_category_levels"],
                    "rarest_level_entities": diversity["rarest_level_entities"],
                    "numeric_round_decimals": diversity["numeric_round_decimals"],
                    "numeric_iqr": diversity["numeric_iqr"],
                    "numeric_iqr_bins": diversity["numeric_iqr_bins"],
                    "total_pfas": support["total_pfas"],
                    "reporting_pfas": support["reporting_pfas"],
                    "pfas_coverage": support["pfas_coverage"],
                    "total_studies": support["total_studies"],
                    "reporting_studies": support["reporting_studies"],
                    "study_coverage": support["study_coverage"],
                    "total_adsorbents": support["total_adsorbents"],
                    "reporting_adsorbents": support["reporting_adsorbents"],
                    "adsorbent_coverage": support["adsorbent_coverage"],
                }
            )
            continue
        screening_applied = feature_screening_mode == "standard"
        if screening_applied:
            variation_support_reason = _diversity_selection_reason(
                kind,
                diversity,
                min_categorical_levels,
                min_numeric_distinct_values,
                min_numeric_iqr_bins,
            )
            availability_independent_support_reason = _support_selection_reason(
                feature,
                bucket,
                support,
                min_nonmissing_rows,
                min_reporting_study_fraction,
                min_reporting_adsorbent_fraction,
            )
            selected_flag = (
                variation_support_reason == "selected"
                and availability_independent_support_reason == "selected"
            )
            reason = (
                availability_independent_support_reason
                if availability_independent_support_reason != "selected"
                else variation_support_reason
            )
        else:
            variation_support_reason = SCREENING_NOT_APPLIED
            availability_independent_support_reason = SCREENING_NOT_APPLIED
            selected_flag = True
            reason = RETAINED_WITHOUT_SCREENING
        if selected_flag:
            selected.append(feature)

        rows.append(
            {
                **base_row,
                "selected": selected_flag,
                "reason": reason,
                "availability_independent_support_reason": availability_independent_support_reason,
                "availability_independent_support_pass": (
                    availability_independent_support_reason == "selected"
                    if screening_applied
                    else pd.NA
                ),
                "variation_support_reason": variation_support_reason,
                "variation_support_pass": (
                    variation_support_reason == "selected" if screening_applied else pd.NA
                ),
                "kind": kind,
                "numeric_ratio": numeric_ratio,
                "nonmissing_rows": support["nonmissing_rows"],
                "row_fraction": support["row_fraction"],
                "unique_values": diversity["unique_values"],
                "is_binary_indicator": diversity["is_binary_indicator"],
                "supported_category_levels": diversity["supported_category_levels"],
                "rarest_level_entities": diversity["rarest_level_entities"],
                "numeric_round_decimals": diversity["numeric_round_decimals"],
                "numeric_iqr": diversity["numeric_iqr"],
                "numeric_iqr_bins": diversity["numeric_iqr_bins"],
                "total_pfas": support["total_pfas"],
                "reporting_pfas": support["reporting_pfas"],
                "pfas_coverage": support["pfas_coverage"],
                "total_studies": support["total_studies"],
                "reporting_studies": support["reporting_studies"],
                "study_coverage": support["study_coverage"],
                "total_adsorbents": support["total_adsorbents"],
                "reporting_adsorbents": support["reporting_adsorbents"],
                "adsorbent_coverage": support["adsorbent_coverage"],
                "min_nonmissing_rows": min_nonmissing_rows,
                "min_reporting_study_fraction": min_reporting_study_fraction,
                "min_reporting_adsorbent_fraction": (
                    min_reporting_adsorbent_fraction if bucket == "Adsorbent properties" else np.nan
                ),
                "min_categorical_levels": (
                    min_categorical_levels if screened_on_levels else np.nan
                ),
                "min_categorical_level_rows": (
                    min_categorical_level_rows if screened_on_levels else np.nan
                ),
                "min_categorical_level_entities": (
                    min_categorical_level_entities if screened_on_levels else np.nan
                ),
                "min_numeric_distinct_values": min_numeric_distinct_values if kind == "numeric" else np.nan,
                "min_numeric_iqr_bins": min_numeric_iqr_bins if kind == "numeric" else np.nan,
            }
        )

    non_pfas_numeric_candidates = [
        str(row["feature"])
        for row in rows
        if row.get("selected")
        and row.get("bucket") in {"Experimental conditions", "Adsorbent properties"}
        and row.get("kind") == "numeric"
    ]
    non_pfas_stratum_coverage = {
        str(row["feature"]): float(
            row["adsorbent_coverage"]
            if row["bucket"] == "Adsorbent properties"
            else row["study_coverage"]
        )
        for row in rows
        if str(row.get("feature")) in non_pfas_numeric_candidates
    }
    (
        non_pfas_component_ids,
        non_pfas_selected_features,
        non_pfas_dropped,
        non_pfas_correlated_with,
        non_pfas_max_correlation,
    ) = _non_pfas_correlation_components(
        df,
        non_pfas_numeric_candidates,
        threshold=correlation_threshold,
        min_pairwise_observations=min_pairwise_observations,
        stratum_coverage=non_pfas_stratum_coverage,
    )
    for row in rows:
        feature = str(row["feature"])
        if feature not in non_pfas_numeric_candidates:
            continue
        row.update(
            {
                "correlation_component_id": non_pfas_component_ids.get(feature, ""),
                "correlation_selected_feature": non_pfas_selected_features.get(feature, ""),
                "correlation_feature_dropped": bool(
                    correlated_feature_handling == "select_one_per_group" and feature in non_pfas_dropped
                ),
                "correlation_scope": (
                    "unique_study_adsorbent"
                    if row["bucket"] == "Adsorbent properties"
                    else "training_rows"
                ),
                "training_correlated_with": "; ".join(
                    non_pfas_correlated_with.get(feature, ())
                ),
                "training_max_absolute_correlation": non_pfas_max_correlation.get(feature, np.nan),
            }
        )
        if correlated_feature_handling == "select_one_per_group" and feature in non_pfas_dropped:
            row.update(
                {
                    "selected": False,
                    "reason": "correlated_feature_group_pruned",
                }
            )
            selected.remove(feature)

    audit = pd.DataFrame(rows)
    if "selected" not in audit.columns:
        audit["selected"] = False
    if "kind" not in audit.columns:
        audit["kind"] = np.nan
    numeric_features = audit.loc[(audit["selected"]) & (audit["kind"] == "numeric"), "feature"].tolist()
    categorical_features = audit.loc[(audit["selected"]) & (audit["kind"] == "categorical"), "feature"].tolist()
    return selected, numeric_features, categorical_features, audit


def _feature_value_mask(df: pd.DataFrame, feature: str, kind: str) -> pd.Series:
    if feature not in df.columns:
        return pd.Series(False, index=df.index)
    if kind == "numeric":
        return coerce_numeric(df[feature]).notna()
    return ~normalized_missing_mask(df[feature])


def feature_value_mask(df: pd.DataFrame, feature: str, kind: str) -> pd.Series:
    """Return the target-free observed-value mask used by all support audits."""
    return _feature_value_mask(df, feature, kind)


def add_feature_availability_summary(audit: pd.DataFrame, total_model_rows: int) -> pd.DataFrame:
    """Add stable observed/missing counts for reporting feature-support audits.

    This is intentionally shared by the modeling and figure-audit workflows so
    their availability tables use identical missing-value semantics.
    """
    result = audit.copy()
    total = int(total_model_rows)
    nonmissing = (
        pd.to_numeric(result.get("nonmissing_rows", 0), errors="coerce")
        .fillna(0)
        .clip(lower=0, upper=total)
        .round()
        .astype(int)
    )
    result["total_model_rows"] = total
    result["missing_rows"] = total - nonmissing
    result["missing_fraction"] = result["missing_rows"] / total if total else np.nan
    result = result.sort_values(["nonmissing_rows", "feature"], ascending=[False, True], kind="stable").reset_index(drop=True)
    result.insert(0, "availability_plot_order", np.arange(1, len(result) + 1))
    return result


def _threshold_deficit(value: float | int, threshold: float | int) -> float:
    """Return a normalized, nonnegative shortfall from an existing support threshold."""
    if pd.isna(threshold):
        return 0.0
    threshold = float(threshold)
    if threshold <= 0:
        return 0.0
    if pd.isna(value):
        return 1.0
    return max(0.0, (threshold - float(value)) / threshold)


def feature_support_profile(
    df: pd.DataFrame,
    selected: list[str],
    numeric_features: list[str],
    categorical_features: list[str],
    *,
    min_nonmissing_rows: int = cfg.DEFAULT_MIN_NONMISSING_ROWS,
    min_reporting_study_fraction: float = cfg.DEFAULT_MIN_REPORTING_STUDY_FRACTION,
    min_reporting_adsorbent_fraction: float = cfg.DEFAULT_MIN_REPORTING_ADSORBENT_FRACTION,
    min_categorical_levels: int = cfg.DEFAULT_MIN_CATEGORICAL_LEVELS,
    min_categorical_level_rows: int = cfg.DEFAULT_MIN_CATEGORICAL_LEVEL_ROWS,
    min_categorical_level_entities: int = cfg.DEFAULT_MIN_CATEGORICAL_LEVEL_ENTITIES,
    min_numeric_distinct_values: int = cfg.DEFAULT_MIN_NUMERIC_DISTINCT_VALUES,
    min_numeric_iqr_bins: float = cfg.DEFAULT_MIN_NUMERIC_IQR_BINS,
) -> pd.DataFrame:
    """Profile selected feature support and normalized selection-threshold deficits.

    The profile is target-free and uses the same observed-value, diversity, and
    study/material support definitions as :func:`feature_audit`.  It is used by
    split selection as a score, never as a second feature-selection policy.
    """
    _validate_selection_thresholds(
        min_nonmissing_rows,
        min_reporting_study_fraction,
        min_reporting_adsorbent_fraction,
        min_categorical_levels,
        min_categorical_level_rows,
        min_categorical_level_entities,
        min_numeric_distinct_values,
        min_numeric_iqr_bins,
    )
    kind_by_feature = {feature: "numeric" for feature in numeric_features}
    kind_by_feature.update({feature: "categorical" for feature in categorical_features})
    rows: list[dict[str, Any]] = []
    for feature in selected:
        kind = kind_by_feature.get(feature)
        if kind not in {"numeric", "categorical"}:
            raise ValueError(f"Feature {feature!r} has no declared numeric/categorical kind.")
        bucket = feature_bucket(feature)
        value_mask = _feature_value_mask(df, feature, kind)
        support = _support_summary(df, value_mask)
        entity_keys = screening_entity_keys(df, bucket)
        if kind == "numeric":
            values = coerce_numeric(df.get(feature, pd.Series(np.nan, index=df.index, dtype=float)))
            diversity = _numeric_diversity(
                feature,
                values,
                value_mask,
                min_categorical_level_rows,
                entity_keys,
                min_categorical_level_entities,
            )
        else:
            values = df.get(feature, pd.Series("", index=df.index, dtype="object"))
            diversity = _categorical_diversity(
                values.loc[value_mask],
                min_categorical_level_rows,
                entity_keys,
                min_categorical_level_entities,
            )

        deficits = {
            "nonmissing_deficit": _threshold_deficit(support["nonmissing_rows"], min_nonmissing_rows),
            "study_coverage_deficit": _threshold_deficit(support["study_coverage"], min_reporting_study_fraction),
            "adsorbent_coverage_deficit": (
                _threshold_deficit(support["adsorbent_coverage"], min_reporting_adsorbent_fraction)
                if bucket == "Adsorbent properties"
                else 0.0
            ),
            "numeric_distinct_deficit": (
                _threshold_deficit(diversity["unique_values"], min_numeric_distinct_values)
                if kind == "numeric"
                else 0.0
            ),
            "numeric_iqr_deficit": (
                _threshold_deficit(diversity["numeric_iqr_bins"], min_numeric_iqr_bins)
                if kind == "numeric"
                else 0.0
            ),
            "categorical_level_deficit": (
                _threshold_deficit(diversity["supported_category_levels"], min_categorical_levels)
                if kind == "categorical"
                else 0.0
            ),
        }
        rows.append(
            {
                "feature": feature,
                "bucket": bucket,
                "kind": kind,
                "total_rows": int(len(df)),
                "nonmissing_rows": support["nonmissing_rows"],
                "missing_rows": int(len(df) - support["nonmissing_rows"]),
                "missing_fraction": 1.0 - support["row_fraction"] if len(df) else np.nan,
                **support,
                **diversity,
                **deficits,
                "support_deficit_score": float(sum(deficits.values())),
            }
        )
    return pd.DataFrame(rows)


def joint_support_coverage(
    df: pd.DataFrame,
    selected: list[str],
    numeric_features: list[str],
    categorical_features: list[str],
) -> pd.DataFrame:
    """Describe simultaneous availability of selected input-feature groups.

    This is diagnostic coverage only. Missing values remain available to XGBoost;
    the table records how much complete joint evidence exists for interpretation.
    """
    kind_by_feature = {feature: "numeric" for feature in numeric_features}
    kind_by_feature.update({feature: "categorical" for feature in categorical_features})
    groups: dict[str, list[str]] = {
        bucket: [feature for feature in selected if feature_bucket(feature) == bucket]
        for bucket in cfg.FEATURE_BUCKETS
    }
    groups["All selected features"] = list(selected)
    rows: list[dict[str, Any]] = []
    for feature_group, features in groups.items():
        if not features:
            rows.append(
                {
                    "coverage_type": "joint_support",
                    "feature_group": feature_group,
                    "selected_feature_count": 0,
                    "complete_rows": 0,
                    "complete_row_fraction": np.nan,
                }
            )
            continue
        complete = pd.Series(True, index=df.index)
        for feature in features:
            complete &= _feature_value_mask(df, feature, kind_by_feature[feature])
        support = _support_summary(df, complete)
        rows.append(
            {
                "coverage_type": "joint_support",
                "feature_group": feature_group,
                "selected_feature_count": len(features),
                "complete_rows": support["nonmissing_rows"],
                "complete_row_fraction": support["row_fraction"],
                "complete_pfas": support["reporting_pfas"],
                "pfas_coverage": support["pfas_coverage"],
                "complete_studies": support["reporting_studies"],
                "study_coverage": support["study_coverage"],
                "complete_adsorbents": support["reporting_adsorbents"],
                "adsorbent_coverage": support["adsorbent_coverage"],
            }
        )
    return pd.DataFrame(rows)


def input_support_diagnostics(
    df_training: pd.DataFrame,
    df_testing: pd.DataFrame,
    selected: list[str],
    numeric_features: list[str],
    categorical_features: list[str],
    numeric_range_tolerance_fraction: float = 0.0,
    feature_manifest: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Describe test inputs relative to training-only input support.

    No score threshold is imposed. These diagnostics show extrapolation,
    unseen categories, and missingness shifts without changing the split.
    ``numeric_range_tolerance_fraction`` expands each training numeric
    range by that fraction of its observed span; the default of zero preserves
    exact range support.
    """
    if numeric_range_tolerance_fraction < 0:
        raise ValueError("numeric_range_tolerance_fraction must be nonnegative.")
    kind_by_feature = {feature: "numeric" for feature in numeric_features}
    kind_by_feature.update({feature: "categorical" for feature in categorical_features})
    bucket_by_feature = {}
    if feature_manifest is not None and {"feature", "bucket"}.issubset(feature_manifest.columns):
        bucket_by_feature = (
            feature_manifest.dropna(subset=["feature"])
            .drop_duplicates("feature", keep="last")
            .set_index("feature")["bucket"]
            .to_dict()
        )
    bucket_for = lambda feature: str(bucket_by_feature.get(feature) or feature_bucket(feature))
    feature_rows: list[dict[str, Any]] = []
    block_slugs = {
        "PFAS characteristics": "pfas",
        "Adsorbent properties": "adsorbent",
        "Experimental conditions": "experimental_conditions",
    }
    row_stats = pd.DataFrame(
        {"test_row_position": np.arange(len(df_testing), dtype=int)}
    )
    for bucket, slug in block_slugs.items():
        block_count = sum(bucket_for(feature) == bucket for feature in selected)
        row_stats[f"{slug}_selected_feature_count"] = block_count
        row_stats[f"{slug}_values_present"] = 0
        row_stats[f"{slug}_values_supported_by_training"] = 0
        row_stats[f"{slug}_numeric_values_outside_training_range"] = 0
        row_stats[f"{slug}_categorical_values_unseen_in_training"] = 0
        row_stats[f"{slug}_values_missing"] = 0

    for feature in selected:
        kind = kind_by_feature[feature]
        dev_present = _feature_value_mask(df_training, feature, kind)
        test_present = _feature_value_mask(df_testing, feature, kind)
        supported = pd.Series(False, index=df_testing.index)
        numeric_outside = pd.Series(False, index=df_testing.index)
        categorical_unseen = pd.Series(False, index=df_testing.index)
        bucket = bucket_for(feature)
        row: dict[str, Any] = {
            "diagnostic_type": "input_support",
            "feature": feature,
            "bucket": bucket,
            "kind": kind,
            "training_nonmissing_rows": int(dev_present.sum()),
            "training_missing_fraction": float((~dev_present).mean()) if len(df_training) else np.nan,
            "testing_nonmissing_rows": int(test_present.sum()),
            "testing_missing_fraction": float((~test_present).mean()) if len(df_testing) else np.nan,
        }
        if kind == "numeric":
            dev_values = coerce_numeric(df_training[feature])[dev_present]
            test_values = coerce_numeric(df_testing[feature])
            if len(dev_values):
                lower, upper = float(dev_values.min()), float(dev_values.max())
                tolerance = (upper - lower) * float(numeric_range_tolerance_fraction)
                supported_lower = lower - tolerance
                supported_upper = upper + tolerance
                supported = test_present & test_values.between(supported_lower, supported_upper, inclusive="both")
                numeric_outside = test_present & ~supported
            else:
                lower, upper, tolerance = np.nan, np.nan, np.nan
                supported_lower, supported_upper = np.nan, np.nan
                numeric_outside = test_present
            row.update(
                {
                    "training_min": lower,
                    "training_max": upper,
                    "training_range_tolerance_fraction": float(numeric_range_tolerance_fraction),
                    "training_range_tolerance": tolerance,
                    "supported_lower": supported_lower,
                    "supported_upper": supported_upper,
                    "testing_below_training_range": int((test_present & (test_values < supported_lower)).sum()) if len(dev_values) else 0,
                    "testing_above_training_range": int((test_present & (test_values > supported_upper)).sum()) if len(dev_values) else 0,
                    "testing_supported_values": int(supported.sum()),
                    "testing_outside_range_values": int(numeric_outside.sum()),
                }
            )
        else:
            dev_levels = set(df_training.loc[dev_present, feature].map(clean_text))
            test_levels = df_testing[feature].map(clean_text)
            supported = test_present & test_levels.isin(dev_levels)
            categorical_unseen = test_present & ~supported
            row.update(
                {
                    "training_supported_levels": len(dev_levels),
                    "testing_supported_values": int(supported.sum()),
                    "testing_unseen_category_values": int(categorical_unseen.sum()),
                }
            )
        feature_rows.append(row)
        slug = block_slugs.get(bucket)
        if slug is not None:
            row_stats[f"{slug}_values_present"] += test_present.to_numpy(dtype=int)
            row_stats[f"{slug}_values_supported_by_training"] += supported.to_numpy(dtype=int)
            row_stats[f"{slug}_numeric_values_outside_training_range"] += numeric_outside.to_numpy(dtype=int)
            row_stats[f"{slug}_categorical_values_unseen_in_training"] += categorical_unseen.to_numpy(dtype=int)
            row_stats[f"{slug}_values_missing"] += (~test_present).to_numpy(dtype=int)

    for slug in block_slugs.values():
        count = row_stats[f"{slug}_selected_feature_count"].replace(0, np.nan)
        numeric_count = sum(
            bucket_for(feature) == next(bucket for bucket, value in block_slugs.items() if value == slug)
            for feature in numeric_features
        )
        categorical_count = sum(
            bucket_for(feature) == next(bucket for bucket, value in block_slugs.items() if value == slug)
            for feature in categorical_features
        )
        present = row_stats[f"{slug}_values_present"].replace(0, np.nan)
        row_stats[f"{slug}_missing_fraction"] = row_stats[f"{slug}_values_missing"] / count
        row_stats[f"{slug}_numeric_outside_range_fraction"] = (
            row_stats[f"{slug}_numeric_values_outside_training_range"] / numeric_count
            if numeric_count else np.nan
        )
        row_stats[f"{slug}_unseen_category_fraction"] = (
            row_stats[f"{slug}_categorical_values_unseen_in_training"] / categorical_count
            if categorical_count else np.nan
        )
        row_stats[f"{slug}_unsupported_present_fraction"] = (
            1.0 - row_stats[f"{slug}_values_supported_by_training"] / present
        )
    return pd.DataFrame(feature_rows), row_stats


def feature_resolved_support_diagnostics(
    df_training: pd.DataFrame,
    df_query: pd.DataFrame,
    selected: list[str],
    numeric_features: list[str],
    categorical_features: list[str],
    feature_manifest: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Create reference-relative, feature-level support diagnostics.

    The existing ``input_support_diagnostics`` intentionally compresses support
    events to block-level fractions.  This companion function preserves the
    individual feature event that produced a support loss while using exactly
    the same training-only missing-value and range semantics.  It returns a
    candidate manifest, a wide row matrix for modeling, and a compact long-form
    audit table.  It never inspects labels or predictions.

    Numeric support is represented by a missingness indicator plus separate
    low-side and high-side log distances beyond the training range.  Categorical
    support is represented by a missingness indicator, an unseen-level
    indicator, and a smoothed training-level rarity score.
    """
    kind_by_feature = {feature: "numeric" for feature in numeric_features}
    kind_by_feature.update({feature: "categorical" for feature in categorical_features})
    bucket_by_feature: dict[str, str] = {}
    if feature_manifest is not None and {"feature", "bucket"}.issubset(feature_manifest.columns):
        bucket_by_feature = (
            feature_manifest.dropna(subset=["feature"])
            .drop_duplicates("feature", keep="last")
            .set_index("feature")["bucket"]
            .astype(str)
            .to_dict()
        )

    def _bucket(feature: str) -> str:
        return str(bucket_by_feature.get(feature) or feature_bucket(feature))

    def _candidate(feature: str, construct: str) -> str:
        return f"feature_support::{feature}::{construct}"

    matrix_columns: dict[str, np.ndarray] = {}
    events: list[pd.DataFrame] = []
    manifest_rows: list[dict[str, Any]] = []
    for feature in selected:
        kind = kind_by_feature.get(feature)
        if kind is None or feature not in df_query.columns or feature not in df_training.columns:
            continue
        bucket = _bucket(feature)
        training_present = feature_value_mask(df_training, feature, kind)
        query_present = feature_value_mask(df_query, feature, kind)
        missing_candidate = _candidate(feature, "missing")
        missing = (~query_present).astype(float)
        matrix_columns[missing_candidate] = missing.to_numpy()
        common = pd.DataFrame(
            {
                "query_row_position": np.arange(len(df_query), dtype=int),
                "feature": feature,
                "bucket": bucket,
                "kind": kind,
                "missing": missing.to_numpy(float),
            },
            index=df_query.index,
        )
        manifest_rows.append(
            {
                "candidate": missing_candidate,
                "feature": feature,
                "block": bucket,
                "kind": kind,
                "construct": "feature_missingness",
                "definition": f"{feature} is missing in the query row",
            }
        )
        if kind == "numeric":
            training_values = coerce_numeric(df_training.loc[training_present, feature]).dropna()
            query_values = coerce_numeric(df_query[feature])
            if len(training_values):
                lower = float(training_values.min())
                upper = float(training_values.max())
                iqr = float(training_values.quantile(0.75) - training_values.quantile(0.25))
                observed_span = upper - lower
                scale = iqr if np.isfinite(iqr) and iqr > 0 else observed_span
                scale = float(scale) if np.isfinite(scale) and scale > 0 else 1.0
                below = np.log1p(np.maximum(0.0, lower - query_values.to_numpy(float)) / scale)
                above = np.log1p(np.maximum(0.0, query_values.to_numpy(float) - upper) / scale)
            else:
                lower = upper = scale = np.nan
                below = np.full(len(df_query), np.nan)
                above = np.full(len(df_query), np.nan)
            below = np.where(query_present.to_numpy(bool), below, 0.0)
            above = np.where(query_present.to_numpy(bool), above, 0.0)
            below_candidate = _candidate(feature, "below_training_distance")
            above_candidate = _candidate(feature, "above_training_distance")
            matrix_columns[below_candidate] = below
            matrix_columns[above_candidate] = above
            common["below_training_distance"] = below
            common["above_training_distance"] = above
            common["training_lower"] = lower
            common["training_upper"] = upper
            common["training_scale"] = scale
            manifest_rows.extend(
                [
                    {
                        "candidate": below_candidate,
                        "feature": feature,
                        "block": bucket,
                        "kind": kind,
                        "construct": "feature_low_extrapolation",
                        "definition": f"Log-scaled distance below the training range for {feature}",
                    },
                    {
                        "candidate": above_candidate,
                        "feature": feature,
                        "block": bucket,
                        "kind": kind,
                        "construct": "feature_high_extrapolation",
                        "definition": f"Log-scaled distance above the training range for {feature}",
                    },
                ]
            )
        else:
            training_levels = df_training.loc[training_present, feature].map(clean_text)
            counts = training_levels.value_counts(dropna=False)
            total = int(counts.sum())
            level_count = int(len(counts))
            query_levels = df_query[feature].map(clean_text)
            level_observations = query_levels.map(counts).fillna(0).to_numpy(float)
            unseen = (query_present.to_numpy(bool) & (level_observations <= 0)).astype(float)
            denominator = float(total + max(level_count, 1))
            rarity = -np.log((level_observations + 1.0) / denominator) if total else np.full(len(df_query), np.nan)
            rarity = np.where(query_present.to_numpy(bool), rarity, 0.0)
            unseen_candidate = _candidate(feature, "unseen_level")
            rarity_candidate = _candidate(feature, "level_rarity")
            matrix_columns[unseen_candidate] = unseen
            matrix_columns[rarity_candidate] = rarity
            common["unseen_level"] = unseen
            common["level_rarity"] = rarity
            common["training_observed_rows"] = total
            common["training_level_count"] = level_count
            manifest_rows.extend(
                [
                    {
                        "candidate": unseen_candidate,
                        "feature": feature,
                        "block": bucket,
                        "kind": kind,
                        "construct": "feature_unseen_category",
                        "definition": f"{feature} has a categorical level absent from training",
                    },
                    {
                        "candidate": rarity_candidate,
                        "feature": feature,
                        "block": bucket,
                        "kind": kind,
                        "construct": "feature_category_rarity",
                        "definition": f"Smoothed training-only rarity of the {feature} level",
                    },
                ]
            )
        events.append(common.reset_index(drop=True))
    candidate_manifest = pd.DataFrame(manifest_rows)
    event_table = pd.concat(events, ignore_index=True, sort=False) if events else pd.DataFrame()
    matrix = pd.DataFrame(matrix_columns, index=df_query.index)
    return candidate_manifest, matrix.reset_index(drop=True), event_table


def build_X(df: pd.DataFrame, selected: list[str], numeric: list[str], categorical: list[str]) -> pd.DataFrame:
    """Build a model frame while preserving applicability separately from missingness.

    Categorical adsorbent-specific fields use ``<not_applicable>`` for known
    out-of-family rows and retain ``<missing>`` for an applicable but unreported
    value.  Numeric fields remain numeric: their generated applicability column
    records 1 (applicable), 0 (known inapplicable), or NaN (unknown category).
    """

    missing = [col for col in selected if col not in df.columns]
    if missing:
        raise ValueError(f"Selected features missing from model frame: {missing}")
    X = df[selected].copy()
    applicability = {
        feature: _conditional_applicability_state(df, feature)
        for feature in selected
        if feature in cfg.ADSORBENT_SPECIFIC_FEATURE_APPLICABILITY
    }
    for col in numeric:
        if col not in X.columns:
            continue
        state = applicability.get(col)
        if state is not None:
            _reject_reported_inapplicable_values(df, col, state)
            X.loc[state.eq(0.0), col] = np.nan
            X[f"{col}{cfg.ADSORBENT_APPLICABILITY_INDICATOR_SUFFIX}"] = state
        X[col] = coerce_numeric(X[col])
    for col in categorical:
        if col not in X.columns:
            continue
        state = applicability.get(col)
        if state is not None:
            _reject_reported_inapplicable_values(df, col, state)
        values = X[col].map(clean_text).replace("", "<missing>")
        if state is not None:
            values = values.mask(state.eq(0.0), cfg.ADSORBENT_NOT_APPLICABLE_CATEGORY)
        X[col] = values
    return X
