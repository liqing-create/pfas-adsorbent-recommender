"""Target-free observed-support diagnostics for logKd predictions.

The functions in this module compare a PFAS with the PFAS represented in a
model's training partition. They report continuous similarity, density,
missingness, range, and category support without assigning domain labels.
"""

from __future__ import annotations

from typing import Any, Iterable

import numpy as np
import pandas as pd

from . import logkd_config as cfg
from .logkd_data import clean_text, normalize_pfas_key


SUPPORT_INDEX_SPECS = (
    {
        "support_index": "morgan_tanimoto",
        "score_col": "max_morgan_similarity_to_training",
        "higher_score_more_supported": True,
    },
)

KDE_CANDIDATES = (
    {
        "candidate": "pfas_descriptor",
        "description": "selected PFAS descriptor support",
        "buckets": frozenset({"PFAS characteristics"}),
        "block": "pfas",
    },
    {
        "candidate": "adsorbent",
        "description": "selected adsorbent-property support",
        "buckets": frozenset({"Adsorbent properties"}),
        "block": "adsorbent",
    },
    {
        "candidate": "experimental_conditions",
        "description": "selected experimental-condition support",
        "buckets": frozenset({"Experimental conditions"}),
        "block": "experimental_conditions",
    },
    {
        "candidate": "full_input",
        "description": "all selected model-input support",
        "buckets": None,
        "block": "joint",
    },
    {
        "candidate": "full_input_pca5",
        "description": "all selected model-input support after training-fitted five-component PCA",
        "buckets": None,
        "block": "joint",
        "pca_components": cfg.DEFAULT_SUPPORT_KDE_PCA_COMPONENTS,
    },
)


def kde_score_column(candidate: str) -> str:
    return f"support_{candidate}_kde_dissimilarity"


def _candidate_prefix(candidate: str) -> str:
    return f"support_{candidate}_kde"


def _category_token(value: Any) -> str:
    """Represent missing categorical values explicitly without treating them as novel."""

    text = clean_text(value)
    return text if text else "__missing__"


def _selected_features_for_candidate(
    candidate_spec: dict[str, Any],
    selected_features: Iterable[str],
    feature_manifest: pd.DataFrame | None,
) -> list[str]:
    selected = list(dict.fromkeys(selected_features))
    buckets = candidate_spec["buckets"]
    if buckets is None:
        return selected
    if (
        feature_manifest is None
        or feature_manifest.empty
        or not {"feature", "bucket"}.issubset(feature_manifest.columns)
    ):
        return []
    lookup = (
        feature_manifest.loc[:, [column for column in ("feature", "bucket") if column in feature_manifest.columns]]
        .dropna(subset=["feature"])
        .drop_duplicates("feature", keep="last")
        .set_index("feature")["bucket"]
        .to_dict()
    )
    return [feature for feature in selected if lookup.get(feature) in buckets]


def _kde_unavailable(
    query: pd.DataFrame,
    *,
    candidate: str,
    status: str,
    missing_count: np.ndarray,
    numeric_outside_count: np.ndarray,
    unseen_categorical_count: np.ndarray,
    selected_count: int,
    numeric_count: int,
    categorical_count: int,
    training_rows: int,
    pca_components_requested: int,
) -> pd.DataFrame:
    """Return a schema-stable unavailable Schultz-style KDE result."""

    prefix = _candidate_prefix(candidate)
    rows = len(query)
    return pd.DataFrame(
        {
            f"{prefix}_status": status,
            kde_score_column(candidate): np.full(rows, np.nan),
            f"{prefix}_density_ratio": np.full(rows, np.nan),
            f"{prefix}_log_density": np.full(rows, np.nan),
            f"{prefix}_training_rows": training_rows,
            f"{prefix}_selected_feature_count": selected_count,
            f"{prefix}_input_dimensions": 0,
            f"{prefix}_pre_pca_input_dimensions": 0,
            f"{prefix}_pca_components_requested": pca_components_requested,
            f"{prefix}_pca_components_used": 0,
            f"{prefix}_pca_cumulative_explained_variance_ratio": np.nan,
            f"{prefix}_bandwidth": np.nan,
            f"{prefix}_kernel": "epanechnikov",
            f"{prefix}_missing_feature_count": missing_count,
            f"{prefix}_missing_feature_fraction": missing_count / selected_count if selected_count else np.full(rows, np.nan),
            f"{prefix}_numeric_features_outside_training_range_count": numeric_outside_count,
            f"{prefix}_unseen_categorical_feature_count": unseen_categorical_count,
            f"{prefix}_numeric_feature_count": numeric_count,
            f"{prefix}_categorical_feature_count": categorical_count,
        }
    )


def _single_kde_support(
    df_training: pd.DataFrame,
    df_query: pd.DataFrame,
    *,
    candidate_spec: dict[str, Any],
    selected_features: Iterable[str],
    numeric_features: Iterable[str],
    categorical_features: Iterable[str],
    feature_manifest: pd.DataFrame | None,
) -> pd.DataFrame:
    """Calculate one Schultz-style dissimilarity using training rows only.

    The score is ``d = 1 - KDE(query) / max(KDE(training))``.  Numeric inputs
    are median-imputed and standardized using the reference rows; categorical
    inputs use reference-level one-hot encoding.  As in Schultz et al., the
    KDE uses an Epanechnikov kernel and a nearest-neighbour bandwidth estimate.
    No target, outer-test row, or supervised model information participates in
    constructing the score.  For a PCA candidate, the PCA projection is fitted
    only on the supplied training reference rows before query rows are
    transformed.
    """

    from sklearn.cluster import estimate_bandwidth
    from sklearn.decomposition import PCA
    from sklearn.neighbors import KernelDensity

    candidate = str(candidate_spec["candidate"])
    prefix = _candidate_prefix(candidate)
    pca_components_requested = int(candidate_spec.get("pca_components") or 0)
    if pca_components_requested < 0:
        raise ValueError(f"PCA components for candidate '{candidate}' must be nonnegative.")
    training = df_training.reset_index(drop=True)
    query = df_query.reset_index(drop=True)
    selected = [
        feature
        for feature in _selected_features_for_candidate(candidate_spec, selected_features, feature_manifest)
        if feature in training.columns
    ]
    numeric = [feature for feature in numeric_features if feature in selected]
    categorical = [feature for feature in categorical_features if feature in selected]
    rows = len(query)
    missing_counts = np.zeros(rows, dtype=int)
    numeric_outside_counts = np.zeros(rows, dtype=int)
    unseen_categorical_counts = np.zeros(rows, dtype=int)
    train_parts: list[np.ndarray] = []
    query_parts: list[np.ndarray] = []

    for feature in numeric:
        train_values = pd.to_numeric(training[feature], errors="coerce")
        query_values = pd.to_numeric(query.reindex(columns=[feature])[feature], errors="coerce")
        missing_counts += query_values.isna().to_numpy(dtype=int)
        observed = train_values.dropna()
        if observed.empty:
            continue
        lower, upper = float(observed.min()), float(observed.max())
        numeric_outside_counts += (query_values.notna() & ((query_values < lower) | (query_values > upper))).to_numpy(dtype=int)
        median = float(observed.median())
        train_filled = train_values.fillna(median).to_numpy(dtype=float)
        query_filled = query_values.fillna(median).to_numpy(dtype=float)
        sd = float(train_filled.std(ddof=0))
        if not np.isfinite(sd) or sd <= 0:
            continue
        train_parts.append(((train_filled - train_filled.mean()) / sd)[:, None])
        query_parts.append(((query_filled - train_filled.mean()) / sd)[:, None])

    for feature in categorical:
        train_values = training[feature].map(_category_token)
        query_values = query.reindex(columns=[feature])[feature].map(_category_token)
        missing_counts += query_values.eq("__missing__").to_numpy(dtype=int)
        levels = sorted(set(train_values))
        unseen_categorical_counts += (query_values.ne("__missing__") & ~query_values.isin(levels)).to_numpy(dtype=int)
        if len(levels) < 2:
            continue
        train_parts.append(np.column_stack([train_values.eq(level).to_numpy(dtype=float) for level in levels]))
        query_parts.append(np.column_stack([query_values.eq(level).to_numpy(dtype=float) for level in levels]))

    base = _kde_unavailable(
        query,
        candidate=candidate,
        status="insufficient_variable_selected_inputs",
        missing_count=missing_counts,
        numeric_outside_count=numeric_outside_counts,
        unseen_categorical_count=unseen_categorical_counts,
        selected_count=len(selected),
        numeric_count=len(numeric),
        categorical_count=len(categorical),
        training_rows=len(training),
        pca_components_requested=pca_components_requested,
    )
    if len(training) < cfg.DEFAULT_SUPPORT_KDE_MIN_TRAINING_ROWS:
        base[f"{prefix}_status"] = "insufficient_training_rows_for_kde"
        return base
    if not train_parts:
        return base
    train_matrix = np.hstack(train_parts)
    query_matrix = np.hstack(query_parts)
    variable = np.isfinite(train_matrix.std(axis=0, ddof=0)) & (train_matrix.std(axis=0, ddof=0) > 0)
    if not variable.any():
        return base
    train_matrix = train_matrix[:, variable]
    query_matrix = query_matrix[:, variable]
    base[f"{prefix}_pre_pca_input_dimensions"] = int(train_matrix.shape[1])
    if pca_components_requested:
        usable_components = min(
            pca_components_requested,
            int(train_matrix.shape[0]),
            int(train_matrix.shape[1]),
        )
        if usable_components < 1:
            return base
        pca = PCA(n_components=usable_components, svd_solver="full")
        train_matrix = pca.fit_transform(train_matrix)
        query_matrix = pca.transform(query_matrix)
        base[f"{prefix}_pca_components_used"] = usable_components
        base[f"{prefix}_pca_cumulative_explained_variance_ratio"] = float(
            np.sum(pca.explained_variance_ratio_)
        )
    bandwidth_sample_size = min(len(training), cfg.DEFAULT_SUPPORT_KDE_BANDWIDTH_SAMPLE_SIZE)
    bandwidth = float(
        estimate_bandwidth(
            train_matrix,
            quantile=cfg.DEFAULT_SUPPORT_KDE_BANDWIDTH_QUANTILE,
            n_samples=bandwidth_sample_size,
            random_state=0,
        )
    )
    if not np.isfinite(bandwidth) or bandwidth <= 0:
        bandwidth = cfg.DEFAULT_SUPPORT_KDE_MIN_BANDWIDTH
    bandwidth = max(float(cfg.DEFAULT_SUPPORT_KDE_MIN_BANDWIDTH), bandwidth)
    density = KernelDensity(kernel="epanechnikov", bandwidth=bandwidth).fit(train_matrix)
    training_log_density = density.score_samples(train_matrix)
    query_log_density = density.score_samples(query_matrix)
    maximum_training_log_density = float(np.nanmax(training_log_density))
    with np.errstate(over="ignore", invalid="ignore"):
        density_ratio = np.exp(query_log_density - maximum_training_log_density)
    density_ratio = np.clip(density_ratio, 0.0, 1.0)
    dissimilarity = 1.0 - density_ratio

    base[f"{prefix}_status"] = "available"
    base[kde_score_column(candidate)] = dissimilarity
    base[f"{prefix}_density_ratio"] = density_ratio
    base[f"{prefix}_log_density"] = query_log_density
    base[f"{prefix}_input_dimensions"] = int(train_matrix.shape[1])
    base[f"{prefix}_bandwidth"] = bandwidth
    return base


def kde_candidate_support(
    df_training: pd.DataFrame,
    df_query: pd.DataFrame,
    selected_features: Iterable[str],
    numeric_features: Iterable[str],
    categorical_features: Iterable[str],
    feature_manifest: pd.DataFrame | None,
) -> pd.DataFrame:
    """Score support with independent raw-space and PCA-reduced KDE candidates."""

    frames = [
        _single_kde_support(
            df_training,
            df_query,
            candidate_spec=spec,
            selected_features=selected_features,
            numeric_features=numeric_features,
            categorical_features=categorical_features,
            feature_manifest=feature_manifest,
        )
        for spec in KDE_CANDIDATES
    ]
    return pd.concat(frames, axis=1) if frames else pd.DataFrame(index=range(len(df_query)))


def pfas_feature_lookup(pfas_features: pd.DataFrame | None) -> pd.DataFrame:
    """Return one normalized feature row per PFAS identity."""
    if pfas_features is None or pfas_features.empty:
        return pd.DataFrame()
    props = pfas_features.copy()
    if "__pfas_key" not in props.columns:
        key_col = next((column for column in cfg.PFAS_KEY_CANDIDATES if column in props.columns), None)
        if key_col is None:
            return pd.DataFrame()
        props["__pfas_key"] = props[key_col].map(normalize_pfas_key)
    props = props.loc[props["__pfas_key"] != ""].drop_duplicates("__pfas_key", keep="first")
    return props.set_index("__pfas_key", drop=False)


def _key_from_row(row: pd.Series | dict[str, Any]) -> str:
    for column in ("__pfas_key", "PFAS_name", "Abbreviation", "Compound Full Name", "Name"):
        value = row.get(column) if isinstance(row, dict) else row.get(column)
        key = normalize_pfas_key(value)
        if key:
            return key
    return ""


def _display_lookup(df: pd.DataFrame) -> dict[str, str]:
    out: dict[str, str] = {}
    for _, row in df.iterrows():
        key = _key_from_row(row)
        if not key or key in out:
            continue
        out[key] = clean_text(row.get("PFAS_name") or row.get("Abbreviation") or row.get("Compound Full Name") or key)
    return out


def _morgan_fingerprints(props: pd.DataFrame) -> tuple[dict[str, Any], Any | None]:
    """Build Morgan fingerprints directly from cached PFAS SMILES.

    The PFAS feature table contains SMILES and scalar RDKit descriptors, but
    intentionally does not persist thousands of fingerprint-bit columns.
    Structural similarity is therefore calculated from SMILES rather than depending
    on an optional, precomputed feature family.  We use the same unfolded
    count-Morgan representation (radius 2) documented by
    ``generate_pfas_features.py``: unlike binary Morgan fingerprints, it
    retains repeated environments and therefore distinguishes PFAS homologues
    with otherwise identical local environments.
    """
    try:
        from rdkit import Chem, DataStructs
        from rdkit.Chem import rdFingerprintGenerator
    except ImportError:
        return {}, None

    generator = rdFingerprintGenerator.GetMorganGenerator(radius=cfg.AD_MORGAN_RADIUS)
    fingerprints: dict[str, Any] = {}
    smiles_columns = [column for column in ("Canonical SMILES", "SMILES") if column in props.columns]
    for key, row in props.iterrows():
        for column in smiles_columns:
            smiles = clean_text(row.get(column))
            if not smiles:
                continue
            molecule = Chem.MolFromSmiles(smiles)
            if molecule is not None:
                fingerprints[str(key)] = generator.GetSparseCountFingerprint(molecule)
                break
    return fingerprints, DataStructs


def _morgan_similarity(a: Any, b: Any, data_structs: Any) -> float:
    return float(data_structs.TanimotoSimilarity(a, b))


def _diagnostics_for_keys(
    training_keys: Iterable[str],
    test_keys: Iterable[str],
    props: pd.DataFrame,
    display: dict[str, str],
) -> pd.DataFrame:
    training_keys = {key for key in training_keys if key}
    test_keys = sorted({key for key in test_keys if key})
    if not test_keys:
        return pd.DataFrame()
    if props.empty:
        return pd.DataFrame(
            {
                "pfas_key": test_keys,
                "PFAS_name": [display.get(key, key) for key in test_keys],
                "same_pfas_in_training": [key in training_keys for key in test_keys],
                "pfas_identity_status": ["seen_in_training" if key in training_keys else "unseen_in_training" for key in test_keys],
                "pfas_feature_status": "missing_features",
                "morgan_similarity_status": "missing_or_unusable_features",
                "structural_support_status": "insufficient_structural_features",
            }
        )

    morgan_fingerprints, data_structs = _morgan_fingerprints(props)
    usable_morgan = [key for key in sorted(training_keys) if key in morgan_fingerprints]

    rows: list[dict[str, Any]] = []
    for key in test_keys:
        in_props = key in props.index
        same = key in training_keys
        morgan_usable = key in morgan_fingerprints and bool(usable_morgan) and data_structs is not None

        nearest_morgan, max_morgan = "", np.nan
        if morgan_usable:
            values = [
                (candidate, _morgan_similarity(morgan_fingerprints[key], morgan_fingerprints[candidate], data_structs))
                for candidate in usable_morgan
            ]
            values = [(candidate, score) for candidate, score in values if pd.notna(score)]
            if values:
                nearest_morgan, max_morgan = max(values, key=lambda item: item[1])

        status = (
            "seen_pfas_in_training"
            if same
            else "structural_score_available"
            if morgan_usable
            else "insufficient_structural_features"
        )
        rows.append(
            {
                "pfas_key": key,
                "PFAS_name": display.get(key, key),
                "same_pfas_in_training": bool(same),
                "pfas_identity_status": "seen_in_training" if same else "unseen_in_training",
                "pfas_feature_status": "feature_available" if in_props else "missing_features",
                "nearest_morgan_pfas": display.get(nearest_morgan, nearest_morgan),
                "max_morgan_similarity_to_training": float(max_morgan) if pd.notna(max_morgan) else np.nan,
                "morgan_similarity_status": (
                    "available"
                    if morgan_usable
                    else "rdkit_not_installed"
                    if data_structs is None
                    else "missing_or_invalid_smiles"
                ),
                "structural_support_status": status,
            }
        )
    return pd.DataFrame(rows)


def pfas_structural_diagnostics(
    df_training: pd.DataFrame,
    df_testing: pd.DataFrame,
    pfas_features: pd.DataFrame | None,
) -> pd.DataFrame:
    """Score testing PFAS identities against training PFAS identities."""
    training_keys = {_key_from_row(row) for _, row in df_training.iterrows()}
    test_keys = {_key_from_row(row) for _, row in df_testing.iterrows()}
    display = _display_lookup(pd.concat([df_training, df_testing], ignore_index=True, sort=False))
    return _diagnostics_for_keys(training_keys, test_keys, pfas_feature_lookup(pfas_features), display)


def observed_support_for_testing(df: pd.DataFrame, assignments: pd.DataFrame, pfas_features: pd.DataFrame | None) -> pd.DataFrame:
    """Return continuous structural diagnostics for the frozen testing partition."""
    if "split" not in assignments.columns:
        raise ValueError("Testing assignments must contain a split column.")
    training = df.loc[assignments["split"].eq("training")].reset_index(drop=True)
    testing = df.loc[assignments["split"].eq("testing")].reset_index(drop=True)
    return pfas_structural_diagnostics(training, testing, pfas_features)


def merge_support_to_rows(predictions: pd.DataFrame, diagnostics: pd.DataFrame) -> pd.DataFrame:
    """Attach one PFAS-level structural diagnostic record to each prediction row."""
    if predictions.empty or diagnostics.empty:
        return predictions.copy()
    out = predictions.copy()
    if "pfas_key" not in out.columns:
        if "PFAS_name" not in out.columns:
            return out
        out["pfas_key"] = out["PFAS_name"].map(normalize_pfas_key)
    return out.merge(diagnostics, on="pfas_key", how="left", suffixes=("", "_diagnostic"))
