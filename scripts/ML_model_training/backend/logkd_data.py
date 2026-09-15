"""Data loading and PFAS feature joining for PFAS logKd modeling."""

from __future__ import annotations

import re
import time
import unicodedata
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np
import pandas as pd

from . import logkd_config as cfg


def clean_text(value: Any) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip()


def normalize_pfas_key(value: Any) -> str:
    if pd.isna(value):
        return ""
    text = unicodedata.normalize("NFKC", str(value)).lower().strip()
    return re.sub(r"[^a-z0-9]+", "", text)


def normalized_missing_mask(series: pd.Series) -> pd.Series:
    text = series.map(lambda value: clean_text(value).casefold())
    return series.isna() | text.isin(cfg.MISSING_TOKENS)


def coerce_numeric(series: pd.Series) -> pd.Series:
    # pandas leaves an all-True/False column as bool dtype, which then rejects
    # the arithmetic every caller performs (quantiles, differences, rounding).
    # Callers want a numeric value, so always hand back a float.
    return pd.to_numeric(series.where(~normalized_missing_mask(series)), errors="coerce").astype(float)


# Box and OneDrive hydrate cloud-backed workbooks on first access, which
# surfaces as a transient OSError (EINVAL/EBUSY/EACCES) rather than a missing
# file. Callers treat any OSError as a permanent failure, so a single blip can
# silently drop a whole cohort from a multi-hour comparison. Retry first.
INPUT_READ_ATTEMPTS = 4
INPUT_READ_RETRY_SECONDS = 2.0


def _read_with_retry(path: Path, reader: Callable[[], pd.DataFrame]) -> pd.DataFrame:
    last_error: OSError | None = None
    for attempt in range(INPUT_READ_ATTEMPTS):
        if attempt:
            time.sleep(INPUT_READ_RETRY_SECONDS * attempt)
        try:
            return reader()
        except FileNotFoundError:
            raise
        except OSError as error:
            last_error = error
    raise OSError(
        f"Failed to read {path} after {INPUT_READ_ATTEMPTS} attempts: {last_error}"
    ) from last_error


def read_input_table(path: Path, sheet_name: str) -> pd.DataFrame:
    path = Path(path)
    if path.suffix.casefold() == ".csv":
        return _read_with_retry(path, lambda: pd.read_csv(path, low_memory=False))
    return _read_with_retry(
        path, lambda: pd.read_excel(path, sheet_name=sheet_name, keep_default_na=False)
    )


def model_include_mask(df: pd.DataFrame) -> pd.Series:
    if cfg.MODEL_INCLUDE_COL not in df.columns:
        return pd.Series(True, index=df.index)
    series = df[cfg.MODEL_INCLUDE_COL]
    if series.dtype == bool:
        return series.fillna(False)
    values = series.map(lambda value: clean_text(value).casefold())
    return ~values.isin(cfg.MODEL_EXCLUDE_TOKENS)


def model_category_mask(df: pd.DataFrame, model: str) -> pd.Series:
    if "adsorbent_category" not in df.columns:
        raise ValueError("Input table is missing required column 'adsorbent_category'.")
    return df["adsorbent_category"].map(clean_text).isin(cfg.MODEL_CATEGORIES[model])


def finite_target_mask(df: pd.DataFrame, target: str) -> pd.Series:
    if target not in df.columns:
        raise ValueError(f"Input table is missing required target column {target!r}.")
    y = coerce_numeric(df[target])
    return y.notna() & np.isfinite(y)


def normalized_kd_final_sources(sources: Sequence[str] | None) -> tuple[str, ...] | None:
    """Return requested final-Kd source labels in a comparison-safe form.

    ``None`` intentionally means no source restriction, preserving the current
    mixed-endpoint modelling population.  Source labels are matched
    case-insensitively because the normalized workbook is an external input
    whose capitalization may vary across database rebuilds.
    """

    if sources is None:
        return None
    normalized = tuple(
        dict.fromkeys(
            clean_text(source).casefold()
            for source in sources
            if clean_text(source)
        )
    )
    if not normalized:
        raise ValueError("kd_final_sources must contain at least one nonblank source label.")
    return normalized


def equal_source_row_weights(df: pd.DataFrame) -> pd.Series:
    """Give every pre-expansion source record equal total weight."""

    if "source_row_index" not in df.columns:
        raise ValueError("Modeling requires source_row_index for equal source-record weighting.")
    source_rows = df["source_row_index"].map(clean_text)
    if source_rows.eq("").any():
        raise ValueError("Modeling requires a nonblank source_row_index for every row.")
    sizes = source_rows.map(source_rows.value_counts())
    weights = 1.0 / sizes.astype(float)
    return weights * (len(weights) / float(weights.sum()))


def flag_mask(df: pd.DataFrame, flag_col: str) -> pd.Series:
    if flag_col not in df.columns:
        if flag_col in cfg.OPTIONAL_UNRELIABLE_FLAG_COLUMNS:
            return pd.Series(False, index=df.index)
        raise ValueError(f"Input table is missing flag column {flag_col!r}.")
    series = df[flag_col]
    if series.dtype == bool:
        return series.fillna(False)
    values = series.map(lambda value: clean_text(value).casefold())
    return values.isin(cfg.UNRELIABLE_FLAG_TRUTHY_VALUES)


def read_pfas_features(path: Path, sheet_name: str = cfg.DEFAULT_PFAS_FEATURES_SHEET) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"PFAS feature workbook not found: {path}")
    props = _read_with_retry(path, lambda: pd.read_excel(path, sheet_name=sheet_name))
    props.columns = [c.strip() if isinstance(c, str) else c for c in props.columns]
    key_col = next((c for c in cfg.PFAS_KEY_CANDIDATES if c in props.columns), None)
    if key_col is None:
        raise ValueError(f"PFAS feature workbook has no key column in {cfg.PFAS_KEY_CANDIDATES}: {path}")
    props = props.copy()
    props["__pfas_key"] = props[key_col].map(normalize_pfas_key)
    props = props[props["__pfas_key"] != ""]

    # Key normalization folds case, punctuation and Unicode, so two workbook rows
    # can collapse onto one PFAS.  Dropping the later one silently would make the
    # retained descriptors depend on row order, which is invisible downstream and
    # impossible to notice in a result.  Identical duplicates are harmless and are
    # still collapsed; conflicting ones are refused.
    duplicated = props["__pfas_key"].duplicated(keep=False)
    if duplicated.any() and "SMILES" in props.columns:
        conflicts = []
        for pfas_key, group in props[duplicated].groupby("__pfas_key"):
            distinct = group["SMILES"].map(clean_text).unique()
            if len(distinct) > 1:
                labels = "; ".join(str(v) for v in group[key_col])
                conflicts.append(f"{pfas_key} ({labels}): {len(distinct)} different SMILES")
        if conflicts:
            raise ValueError(
                f"{path} maps several rows onto one PFAS key with conflicting structures, "
                "so which structure is used would depend on row order: "
                + " | ".join(conflicts)
            )

    props = props.drop_duplicates("__pfas_key", keep="first")
    return props.reset_index(drop=True)


def attach_pfas_features(df: pd.DataFrame, pfas_features: pd.DataFrame | None) -> pd.DataFrame:
    if pfas_features is None or pfas_features.empty or "PFAS_name" not in df.columns:
        return df
    props = pfas_features.copy()
    if "__pfas_key" not in props.columns:
        key_col = next((c for c in cfg.PFAS_KEY_CANDIDATES if c in props.columns), None)
        if key_col is None:
            return df
        props["__pfas_key"] = props[key_col].map(normalize_pfas_key)
    props = props[props["__pfas_key"] != ""].drop_duplicates("__pfas_key", keep="first").set_index("__pfas_key")
    if props.empty:
        return df

    keys = df["PFAS_name"].map(normalize_pfas_key)
    aligned = props.reindex(keys).reset_index(drop=True)
    aligned.index = df.index
    prop_cols = [c for c in aligned.columns if c != "__pfas_key"]

    out = df.copy()
    missing_cols = [c for c in prop_cols if c not in out.columns]
    existing_cols = [c for c in prop_cols if c in out.columns]
    if missing_cols:
        out = pd.concat([out, aligned[missing_cols]], axis=1)
    for col in existing_cols:
        blank_mask = normalized_missing_mask(out[col])
        if blank_mask.any():
            combined = out[col].astype("object")
            combined.loc[blank_mask] = aligned.loc[blank_mask, col].to_numpy()
            out[col] = combined
    return out


def apply_data_mode(
    df: pd.DataFrame,
    data_mode: str,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    before = len(df)
    out = df
    counts = {
        "rows_removed_conversion_spike": 0,
        "rows_removed_conversion_dip": 0,
        "rows_removed_conversion_spike_and_dip": 0,
        "rows_removed_low_Ce_detection_limit": 0,
        "rows_removed_small_concentration_difference": 0,
        "rows_removed_low_removal_rate": 0,
        "rows_removed_high_apparent_removal": 0,
        "rows_removed_unreliable": 0,
    }
    if data_mode == "drop_unreliable":
        human_review_col = cfg.HUMAN_REVIEW_UNRELIABLE_FLAG
        automated_masks = {
            column: flag_mask(df, column)
            for column in cfg.AUTOMATED_UNRELIABLE_FLAG_COLUMNS
        }
        no_rows = pd.Series(False, index=df.index)
        spike = automated_masks.get("logKd_conversion_spike_flag", no_rows)
        dip = automated_masks.get("logKd_conversion_dip_flag", no_rows)
        low_ce = automated_masks.get("low_Ce_detection_limit_flag", no_rows)
        small_difference = automated_masks.get("small_concentration_difference_flag", no_rows)
        low_removal = automated_masks.get("low_removal_rate_flag", no_rows)
        high_apparent_removal = automated_masks.get("high_apparent_removal_flag", no_rows)
        human_review = flag_mask(df, human_review_col)
        automated_unreliable = no_rows.copy()
        for mask in automated_masks.values():
            automated_unreliable |= mask
        unreliable = automated_unreliable | human_review
        counts = {
            "rows_removed_conversion_spike": int(spike.sum()),
            "rows_removed_conversion_dip": int(dip.sum()),
            "rows_removed_conversion_spike_and_dip": int((spike & dip).sum()),
            "rows_removed_low_Ce_detection_limit": int(low_ce.sum()),
            "rows_removed_small_concentration_difference": int(small_difference.sum()),
            "rows_removed_low_removal_rate": int(low_removal.sum()),
            "rows_removed_high_apparent_removal": int(high_apparent_removal.sum()),
            "rows_removed_human_review_unreliable": int(human_review.sum()),
            "rows_removed_unreliable": int(unreliable.sum()),
        }
        out = df.loc[~unreliable].copy()
    elif data_mode != "baseline":
        raise ValueError(f"Unknown data_mode: {data_mode}")
    info = {
        "data_mode": data_mode,
        "rows_before_data_mode": int(before),
        "rows_after_data_mode": int(len(out)),
        "rows_dropped_by_data_mode": int(before - len(out)),
        "unreliable_flag_columns": list(cfg.UNRELIABLE_FLAG_COLUMNS),
        **counts,
    }
    return out.reset_index(drop=True), info


def apply_pfas_core_quality_gate(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    if "PFAS_name" not in df.columns:
        raise ValueError("PFAS core quality gate requires PFAS_name.")
    missing_columns = [column for column in cfg.PFAS_CORE_REQUIRED_COLUMNS if column not in df.columns]
    if missing_columns:
        raise ValueError(f"PFAS core quality gate is missing required columns: {missing_columns}")

    valid = df["PFAS_name"].map(normalize_pfas_key).ne("")
    failure_columns: dict[str, pd.Series] = {}
    for column in cfg.PFAS_CORE_REQUIRED_COLUMNS:
        missing = normalized_missing_mask(df[column])
        failure_columns[column] = missing
        valid &= ~missing

    failures = []
    for pfas_name, group in df.loc[~valid].groupby("PFAS_name", dropna=False):
        indices = group.index
        missing = [column for column, mask in failure_columns.items() if mask.loc[indices].any()]
        failures.append(f"{clean_text(pfas_name)} ({', '.join(missing)})")

    info = {
        "rows_before_pfas_core_quality_gate": int(len(df)),
        "rows_after_pfas_core_quality_gate": int(valid.sum()),
        "rows_dropped_by_pfas_core_quality_gate": int((~valid).sum()),
        "pfas_dropped_by_core_quality_gate": int(df.loc[~valid, "PFAS_name"].map(normalize_pfas_key).nunique()),
        "pfas_core_quality_failures": "; ".join(failures),
        "pfas_core_required_columns": list(cfg.PFAS_CORE_REQUIRED_COLUMNS),
    }
    out = df.loc[valid].copy().reset_index(drop=True)
    if out.empty:
        raise ValueError("No modeling rows remain after the PFAS core quality gate.")
    return out, info


def load_model_rows(
    input_path: Path,
    sheet_name: str,
    model: str,
    target: str,
    pfas_features_path: Path | None = None,
    pfas_features_sheet: str = cfg.DEFAULT_PFAS_FEATURES_SHEET,
    skip_pfas_features_join: bool = False,
    data_mode: str = cfg.DEFAULT_DATA_MODE,
    kd_final_sources: Sequence[str] | None = None,
) -> tuple[pd.DataFrame, dict[str, Any], pd.DataFrame | None]:
    df = read_input_table(input_path, sheet_name)
    df.columns = [c.strip() if isinstance(c, str) else c for c in df.columns]
    df[target] = coerce_numeric(df[target])
    category_and_target = model_category_mask(df, model) & finite_target_mask(df, target)
    requested_sources = normalized_kd_final_sources(kd_final_sources)
    if requested_sources is None:
        source_mask = pd.Series(True, index=df.index)
    else:
        source_column = "Kd_final_source"
        if source_column not in df.columns:
            raise ValueError(
                "Final-Kd source filtering requires input column 'Kd_final_source'."
            )
        source_mask = df[source_column].map(lambda value: clean_text(value).casefold()).isin(
            requested_sources
        )
    source_eligible = category_and_target & source_mask
    include_mask = model_include_mask(df)
    eligible = source_eligible & include_mask
    model_include_excluded = int((source_eligible & ~include_mask).sum())
    df = df.loc[eligible].copy()
    if df.empty:
        raise ValueError("No modeling rows remain after filtering.")

    pfas_features = None
    if not skip_pfas_features_join and pfas_features_path is not None:
        pfas_features = read_pfas_features(pfas_features_path, pfas_features_sheet)
        df = attach_pfas_features(df, pfas_features)

    if "source_row_index" not in df.columns:
        raise ValueError("Model input is missing source_row_index required for source-record cohesion.")
    df = df.reset_index(drop=True)

    df, data_info = apply_data_mode(df, data_mode)
    df, pfas_quality_info = apply_pfas_core_quality_gate(df)
    data_info.update(
        {
            "rows_excluded_by_model_include": model_include_excluded,
            "model_include_column": cfg.MODEL_INCLUDE_COL if cfg.MODEL_INCLUDE_COL in df.columns else None,
            "kd_final_source_filter": list(requested_sources) if requested_sources is not None else None,
            "rows_before_kd_final_source_filter": int(category_and_target.sum()),
            "rows_after_kd_final_source_filter": int(source_eligible.sum()),
            "rows_dropped_by_kd_final_source_filter": int(
                category_and_target.sum() - source_eligible.sum()
            ),
            **pfas_quality_info,
        }
    )
    return df, data_info, pfas_features
