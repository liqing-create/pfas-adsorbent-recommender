"""Create reproducible figures from logKd experiment summary exports.

Reader-facing performance figures use ``summary/performance_summary.csv``;
row-level audit views may use ``detail/testing_predictions.parquet``. The
script never changes the experiment output.
Every figure is saved as a 300-DPI PNG. By default, plots are compact,
panel-ready outputs without a figure title, legend, or explanatory footnote;
pass ``--standalone`` to restore a self-contained layout.

Examples
--------
Paired random-versus-evidence-balanced allocation results::

    python plot_logkd_results.py comparison ^
        --input "...\\summary\\performance_summary.csv" ^
        --group-column inner_allocation_method

Performance distributions across split strategies::

    python plot_logkd_results.py comparison ^
        --input "...\\summary\\performance_summary.csv"

Algorithm comparison::

    python plot_logkd_results.py comparison ^
        --input "...\\summary\\performance_summary.csv"

Coverage results are intentionally compact and tabular in
``summary/coverage_summary.csv``. They are split-allocation QA, rather than
row-level residual predictors.

Held-out predicted-versus-actual values across every outer repeat::

    python plot_logkd_results.py predicted-vs-actual ^
        --input "...\\detail\\testing_predictions.parquet" ^
        --scenario-column split_strategy

The prediction plot writes one standalone figure per scenario. Each figure
overlays all available outer-repeat predictions, using a distinct color for
each repeat. The default
``--data-split testing`` therefore avoids mixing refit-training predictions
with held-out predictions.
"""

from __future__ import annotations

import argparse
import math
import sys
from collections.abc import Iterable, Sequence
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # The script is intended for reproducible file output.
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.text import Text
import numpy as np
import pandas as pd


# -----------------------------------------------------------------------------
# Shared manuscript style
# -----------------------------------------------------------------------------
# These colors are copied from the manuscript Sankey figure.  The extensions
# retain its clean cyan/coral/green/yellow character while avoiding a run of
# visually similar yellow-brown categories in larger comparisons.
SANKY_PALETTE = {
    "blue": "#12A8D8",
    "coral": "#E64B5D",
    "green": "#8CCB78",
    "yellow": "#F2D34F",
    "purple": "#B784CC",
    "navy": "#79BFE3",
    "magenta": "#E895B7",
    "teal": "#77C8BC",
    "slate": "#B4B9C1",
    "indigo": "#A69ACF",
    "gray": "#5A5A5A",
    "light_gray": "#EDEDED",
    "grid": "#D9D9D9",
    "ink": "#111111",
}
# Default sequence for every categorical comparison. It begins with the four
# Sankey colors and then adds cool and purple accents that remain distinct.
MANUSCRIPT_CATEGORY_COLORS = (
    SANKY_PALETTE["blue"],
    SANKY_PALETTE["coral"],
    SANKY_PALETTE["green"],
    SANKY_PALETTE["yellow"],
    SANKY_PALETTE["purple"],
    SANKY_PALETTE["navy"],
    SANKY_PALETTE["magenta"],
    SANKY_PALETTE["teal"],
    SANKY_PALETTE["slate"],
    SANKY_PALETTE["indigo"],
)
FALLBACK_MARKERS = ("o", "s", "^", "D", "P", "X", "v", "<", ">")
# Start with the four manuscript Sankey colors, then use compatible companion
# colors that remain visually distinct when all ten outer repeats are overlaid.
REPEAT_COLORS = (
    SANKY_PALETTE["blue"],
    SANKY_PALETTE["coral"],
    SANKY_PALETTE["green"],
    SANKY_PALETTE["yellow"],
    SANKY_PALETTE["purple"],
    SANKY_PALETTE["navy"],
    SANKY_PALETTE["magenta"],
    SANKY_PALETTE["teal"],
    SANKY_PALETTE["slate"],
    SANKY_PALETTE["indigo"],
)
PANEL_FONT_SCALE = 1.4

PERFORMANCE_METRICS = ("mae", "rmse", "r2", "spearman")
LOGKD_RANGE_METRIC_PREFIXES = ("low_logkd", "middle_logkd", "high_logkd")
DEFAULT_LOW_LOGKD_TAIL_THRESHOLD = 0.0
DEFAULT_HIGH_LOGKD_TAIL_THRESHOLD = 4.0
PREDICTION_RANGE_FIGURE_SIZE = (7.8, 7.8)
# The range headings and values must both remain legible after this square
# panel is reduced in a multi-panel manuscript figure.
PREDICTION_RANGE_HEADING_FONT_SIZE = 12.0
PREDICTION_RANGE_METRIC_FONT_SIZE = 12.0
PREDICTION_RANGE_METRIC_LINE_SPACING = 1.18
PREDICTION_RANGE_HEADING_TOP_OFFSET = 0.18
PREDICTION_RANGE_METRIC_TOP_OFFSET = 0.43
PREDICTION_RANGE_HIGH_ANNOTATION_OFFSET = 0.04
PREDICTION_RANGE_ANNOTATION_RIGHT_PADDING_PIXELS = 8.0
DEFAULT_RETENTION_COLUMN = "inner_mean_reference_retention_score"
REFERENCE_RETENTION_PROFILE_PANELS = (
    (
        "Database contrast",
        (
            ("Experimental", "database_contrast_experimental_conditions"),
            ("Adsorbent", "database_contrast_adsorbent_properties"),
            ("PFAS", "database_contrast_pfas_characteristics"),
        ),
    ),
    (
        "Training contrast",
        (
            ("Experimental", "training_contrast_experimental_conditions"),
            ("Adsorbent", "training_contrast_adsorbent_properties"),
            ("PFAS", "training_contrast_pfas_characteristics"),
        ),
    ),
    (
        "Contrast retention",
        (
            ("Experimental", "contrast_retention_experimental_conditions"),
            ("Adsorbent", "contrast_retention_adsorbent_properties"),
            ("PFAS", "contrast_retention_pfas_characteristics"),
        ),
    ),
)
COMPARISON_GROUP_COLUMN = "comparison_group"
LEGACY_COMPARISON_GROUP_COLUMNS = (
    "split_strategy",
    "inner_allocation_method",
    "regressor",
    "numeric_missing_strategy",
    "threshold_configuration",
    "correlated_feature_handling",
    "tuning_profile",
)
RETENTION_METRICS = (
    "inner_min_reference_retention_score",
    "inner_mean_reference_retention_score",
    "inner_reference_retention_score_sd",
)
ERROR_BAR_CHOICES = ("ci95", "sd", "sem", "none")
T_CRITICAL_95 = {
    1: 12.706,
    2: 4.303,
    3: 3.182,
    4: 2.776,
    5: 2.571,
    6: 2.447,
    7: 2.365,
    8: 2.306,
    9: 2.262,
    10: 2.228,
    11: 2.201,
    12: 2.179,
    13: 2.160,
    14: 2.145,
    15: 2.131,
    16: 2.120,
    17: 2.110,
    18: 2.101,
    19: 2.093,
    20: 2.086,
    21: 2.080,
    22: 2.074,
    23: 2.069,
    24: 2.064,
    25: 2.060,
    26: 2.056,
    27: 2.052,
    28: 2.048,
    29: 2.045,
    30: 2.042,
}
METRIC_LABELS = {
    "mae": "MAE (lower is better)",
    "rmse": "RMSE (lower is better)",
    "r2": "R² (higher is better)",
    "spearman": "Spearman correlation (higher is better)",
    "inner_min_reference_retention_score": "Minimum inner retention",
    "inner_mean_reference_retention_score": "Mean inner retention",
    "inner_reference_retention_score_sd": "Inner-retention SD (lower is better)",
    "database_contrast_experimental_conditions": "Database contrast - Experimental",
    "database_contrast_adsorbent_properties": "Database contrast - Adsorbent",
    "database_contrast_pfas_characteristics": "Database contrast - PFAS",
    "training_contrast_experimental_conditions": "Training contrast - Experimental",
    "training_contrast_adsorbent_properties": "Training contrast - Adsorbent",
    "training_contrast_pfas_characteristics": "Training contrast - PFAS",
    "contrast_retention_experimental_conditions": "Contrast retention - Experimental",
    "contrast_retention_adsorbent_properties": "Contrast retention - Adsorbent",
    "contrast_retention_pfas_characteristics": "Contrast retention - PFAS",
}
METRIC_AXIS_LABELS = {
    "mae": "MAE",
    "rmse": "RMSE",
    "r2": "R²",
    "spearman": "Spearman",
}
DISPLAY_LABELS = {
    "random_row": "Random row",
    "random_group": "Random group",
    "evidence_balanced": "Evidence-balanced",
    "pfas": "PFAS-held-out",
    "adsorbent": "Adsorbent-held-out",
    "study": "Study-held-out",
    "combination": "Combination-held-out",
    "size_matched_random_group": "Size-matched random group",
    "catboost": "CatBoost",
    "extra_trees": "Extra Trees",
    "hist_gradient_boosting": "Hist. Gradient Boosting",
    "xgb": "XGBoost",
    # Model-family scopes are acronyms; the title-case fallback would render
    # them as "Ac" and "Cdp".
    "AC": "AC",
    "CDP": "CDP",
}
STRATEGY_COLORS = {
    "random_row": SANKY_PALETTE["blue"],
    "pfas": SANKY_PALETTE["coral"],
    "adsorbent": SANKY_PALETTE["green"],
    "study": SANKY_PALETTE["yellow"],
    "combination": SANKY_PALETTE["gray"],
}
STRATEGY_MARKERS = {
    "random_row": "o",
    "pfas": "s",
    "adsorbent": "^",
    "study": "D",
    "combination": "P",
}
PREDICTION_REPEAT_COLUMN_CANDIDATES = (
    "outer_repeat_id",
    "repeat_id",
    "run_id",
    "run_directory",
    "run_dir",
    "seed",
    "random_seed",
)


def safe_print(value: object) -> None:
    """Print paths safely when a Windows console uses a legacy code page."""

    text = str(value)
    stream = sys.stdout
    encoding = getattr(stream, "encoding", None) or "utf-8"
    try:
        stream.write(text + "\n")
    except UnicodeEncodeError:
        stream.write(text.encode(encoding, errors="replace").decode(encoding, errors="replace") + "\n")


def output_slug(value: object) -> str:
    """Return a concise, filename-safe suffix for an output category."""

    slug = "".join(
        character.casefold() if character.isalnum() else "_"
        for character in str(value)
    ).strip("_")
    return slug or "scenario"


def configure_style() -> None:
    """Apply the quiet white, black, and Sankey-palette manuscript style."""

    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "axes.edgecolor": SANKY_PALETTE["ink"],
        "axes.labelcolor": SANKY_PALETTE["ink"],
        "xtick.color": SANKY_PALETTE["ink"],
        "ytick.color": SANKY_PALETTE["ink"],
        "legend.frameon": False,
    })


def display_label(value: object) -> str:
    text = str(value)
    if text in DISPLAY_LABELS:
        return DISPLAY_LABELS[text]
    return " ".join(text.replace("__", " ").replace("_", " ").split()).title()


def metric_label(metric: str) -> str:
    return METRIC_LABELS.get(metric, metric.replace("_", " "))


def metric_axis_label(metric: str) -> str:
    return METRIC_AXIS_LABELS.get(metric, metric_label(metric))


def ordered_categories(
    values: Iterable[object],
    requested_order: Sequence[str] | None = None,
) -> list[str]:
    present = [str(value) for value in values if pd.notna(value)]
    unique = list(dict.fromkeys(present))
    if requested_order:
        requested = [str(value) for value in requested_order]
        return [value for value in requested if value in unique] + [
            value for value in unique if value not in requested
        ]
    preferred = [
        "random_row", "pfas", "adsorbent", "study", "combination",
    ]
    return [value for value in preferred if value in unique] + sorted(
        (value for value in unique if value not in preferred),
        key=str.casefold,
    )


def category_styles(categories: Sequence[str]) -> dict[str, tuple[str, str]]:
    """Assign stable color and marker pairs to categorical result groups."""

    styles: dict[str, tuple[str, str]] = {}
    assigned_colors: set[str] = set()
    for index, category in enumerate(categories):
        color = STRATEGY_COLORS.get(category)
        if color is None:
            color = next(
                (
                    candidate
                    for candidate in MANUSCRIPT_CATEGORY_COLORS
                    if candidate not in assigned_colors
                ),
                None,
            )
            if color is None:
                raise ValueError(
                    "The plot supports at most "
                    f"{len(MANUSCRIPT_CATEGORY_COLORS)} comparison groups with "
                    "unique colors; pass a defined grouping or reduce the groups."
                )
        marker = STRATEGY_MARKERS.get(category, FALLBACK_MARKERS[index % len(FALLBACK_MARKERS)])
        styles[category] = (color, marker)
        assigned_colors.add(color)
    return styles


def read_summary(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Run summary was not found: {path}")
    data = (
        pd.read_parquet(path)
        if path.suffix.casefold() in {".parquet", ".pq"}
        else pd.read_csv(path)
    )
    if data.empty:
        raise ValueError(f"Run summary contains no rows: {path}")
    if "status" in data.columns:
        data = data.loc[data["status"].eq("completed")].copy()
    if data.empty:
        raise ValueError("Run summary has no completed runs to plot.")
    return data


def filter_summary(data: pd.DataFrame, clauses: Sequence[str]) -> pd.DataFrame:
    """Restrict a summary to exact categorical values supplied as column=value."""

    filtered = data.copy()
    for clause in clauses:
        column, separator, value = clause.partition("=")
        column = column.strip()
        value = value.strip()
        if not separator or not column or not value:
            raise ValueError(
                "Each --where value must have the form column=value; "
                f"received {clause!r}."
            )
        require_columns(filtered, (column,), context="The requested filter")
        filtered = filtered.loc[filtered[column].astype(str).eq(value)].copy()
        if filtered.empty:
            raise ValueError(
                f"The requested filter {column}={value!r} matched no summary rows."
            )
    return filtered


def select_comparison_groups(
    data: pd.DataFrame,
    *,
    group_column: str,
    groups: Sequence[str] | None,
) -> pd.DataFrame:
    """Restrict a comparison figure to the requested non-empty groups."""

    if not groups:
        return data
    requested = list(dict.fromkeys(str(group) for group in groups))
    available = set(data[group_column].dropna().astype(str))
    missing = [group for group in requested if group not in available]
    if missing:
        raise ValueError(
            "The requested comparison groups are absent from the input: "
            + ", ".join(missing)
        )
    return data.loc[data[group_column].astype(str).isin(requested)].copy()


def require_columns(data: pd.DataFrame, columns: Iterable[str], *, context: str) -> None:
    missing = [column for column in columns if column not in data.columns]
    if missing:
        raise ValueError(
            f"{context} requires columns that are absent from the run summary: {missing}"
        )


def resolve_comparison_group_column(
    data: pd.DataFrame,
    requested_column: str | None,
    *,
    input_name: str | None = None,
) -> str:
    """Use the standardized aggregate field, with a safe legacy fallback."""

    if requested_column:
        require_columns(data, (requested_column,), context="The comparison plot")
        return requested_column
    if COMPARISON_GROUP_COLUMN in data.columns:
        usable = data[COMPARISON_GROUP_COLUMN].dropna().astype(str).str.strip()
        if not usable.empty and usable.ne("").any():
            return COMPARISON_GROUP_COLUMN
    if input_name == "performance_summary.csv" and "split_strategy" in data.columns:
        return "split_strategy"

    candidates = [
        column
        for column in LEGACY_COMPARISON_GROUP_COLUMNS
        if column in data.columns
        and data[column].dropna().astype(str).str.strip().ne("").any()
    ]
    varying = [
        column
        for column in candidates
        if data[column].dropna().astype(str).str.strip().nunique() > 1
    ]
    if len(varying) == 1:
        return varying[0]
    if len(candidates) == 1:
        return candidates[0]
    if not candidates:
        raise ValueError(
            "The comparison plot could not find a standardized 'comparison_group' "
            "column or a recognized legacy comparison column."
        )
    options = ", ".join(varying or candidates)
    raise ValueError(
        "The comparison dimension is ambiguous for this legacy summary (found: "
        f"{options}). Pass --group-column once to select it explicitly."
    )


def comparison_group_label(group_column: str) -> str:
    """Return a reader-friendly label for the chosen comparison dimension."""

    if group_column == COMPARISON_GROUP_COLUMN:
        return "Comparison group"
    return display_label(group_column)


def prediction_panel_label(scenario: str) -> str:
    """Return a compact testing-context label for a prediction panel."""

    parts = [part for part in str(scenario).split("__") if part]
    if not parts:
        return "Testing scenario"
    return f"Testing: {display_label(parts[0])}"


def comparison_legend_label(group: str) -> str:
    """Return a concise legend label without redundant allocation detail."""

    scenario = str(group).split("__", maxsplit=1)[0]
    compact_labels = {
        "only_pfas_characteristics": "Only PFAS",
        "without_pfas_characteristics": "Without PFAS",
        "only_adsorbent_properties": "Only adsorbent",
        "without_adsorbent_properties": "Without adsorbent",
        "only_experimental_conditions": "Only experimental",
        "without_experimental_conditions": "Without experimental",
    }
    return compact_labels.get(scenario, display_label(scenario))


def panel_context_label(data: pd.DataFrame, *, group_column: str | None) -> str:
    """Summarize the fixed testing context without adding a figure-level title."""

    lines: list[str] = []
    if group_column != "split_strategy" and "split_strategy" in data.columns:
        values = data["split_strategy"].dropna().astype(str).str.strip().unique()
        if len(values) == 1 and values[0]:
            lines.append(f"Testing: {display_label(values[0])}")
    if group_column != "outer_allocation_method" and "outer_allocation_method" in data.columns:
        values = data["outer_allocation_method"].dropna().astype(str).str.strip().unique()
        if len(values) == 1 and values[0]:
            lines.append(f"Outer allocation: {display_label(values[0])}")
    if lines:
        return "\n".join(lines)

    comparison_label = {
        COMPARISON_GROUP_COLUMN: "Testing scenarios: split-strategy comparison",
        "split_strategy": "Testing scenarios: split-strategy comparison",
        "regressor": "Testing scenarios: algorithm comparison",
        "numeric_missing_strategy": "Testing scenarios: missing-data strategy",
        "threshold_configuration": "Testing scenarios: feature-threshold configuration",
        "correlated_feature_handling": "Testing scenarios: correlation handling",
        "tuning_profile": "Testing scenarios: tuning profile",
        "inner_allocation_method": "Testing scenarios: inner allocation",
    }.get(group_column)
    return comparison_label or "Testing scenario"


def add_panel_context(
    axis: plt.Axes,
    label: str,
    *,
    x: float = 0.98,
    y: float = 0.98,
    vertical_alignment: str = "top",
) -> None:
    """Place a small in-panel testing label suitable for assembled figures."""

    axis.text(
        x,
        y,
        label,
        transform=axis.transAxes,
        ha="right",
        va=vertical_alignment,
        fontsize=8.0,
        color=SANKY_PALETTE["gray"],
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.78, "pad": 1.8},
        zorder=6,
    )


def enlarge_panel_text(figure: plt.Figure) -> None:
    """Increase text for compact panels that will be reduced in a multi-panel figure."""

    for artist in figure.findobj(match=lambda item: isinstance(item, Text)):
        if artist.get_visible() and artist.get_fontsize() > 0:
            artist.set_fontsize(artist.get_fontsize() * PANEL_FONT_SCALE)


def usable_metrics(data: pd.DataFrame, requested: Sequence[str]) -> list[str]:
    available: list[str] = []
    missing: list[str] = []
    for metric in requested:
        if metric not in data.columns:
            missing.append(metric)
            continue
        if pd.to_numeric(data[metric], errors="coerce").notna().any():
            available.append(metric)
    if missing:
        raise ValueError(
            "Requested metric columns are absent from the run summary: "
            + ", ".join(missing)
        )
    if not available:
        raise ValueError("No requested metric has finite values to plot.")
    return available


def usable_aggregate_metrics(data: pd.DataFrame, requested: Sequence[str]) -> list[str]:
    """Validate metric names against a wide aggregate summary."""

    available: list[str] = []
    missing: list[str] = []
    for metric in requested:
        mean_column = f"{metric}_mean"
        if mean_column not in data.columns:
            missing.append(mean_column)
            continue
        if pd.to_numeric(data[mean_column], errors="coerce").notna().any():
            available.append(metric)
    if missing:
        raise ValueError(
            "Requested aggregate metric columns are absent from the run summary: "
            + ", ".join(missing)
        )
    if not available:
        raise ValueError("No requested aggregate metric has finite values to plot.")
    return available


def axes_grid(metric_count: int) -> tuple[plt.Figure, np.ndarray]:
    columns = min(2, metric_count)
    rows = int(np.ceil(metric_count / columns))
    figure, axes = plt.subplots(
        rows,
        columns,
        figsize=(3.8 * columns, 2.9 * rows),
        squeeze=False,
    )
    figure.subplots_adjust(hspace=0.42, wspace=0.28)
    return figure, axes.ravel()


def error_bar_label(error_bars: str) -> str:
    return {
        "ci95": "95% t confidence interval",
        "sd": "sample standard deviation",
        "sem": "standard error of the mean",
        "none": "no uncertainty bars",
    }[error_bars]


def aggregate_values(values: pd.Series | np.ndarray, error_bars: str) -> tuple[float, float, int]:
    """Return a mean, symmetric error-bar half-width, and finite run count."""

    numeric = pd.to_numeric(pd.Series(values), errors="coerce").dropna()
    count = int(len(numeric))
    if not count:
        return float("nan"), float("nan"), 0
    mean = float(numeric.mean())
    if count < 2 or error_bars == "none":
        return mean, 0.0, count
    sd = float(numeric.std(ddof=1))
    if error_bars == "sd":
        return mean, sd, count
    sem = sd / np.sqrt(count)
    if error_bars == "sem":
        return mean, sem, count
    degrees_of_freedom = count - 1
    critical = T_CRITICAL_95.get(degrees_of_freedom, 1.96)
    return mean, critical * sem, count


def aggregate_error_from_row(
    row: pd.Series,
    metric: str,
    error_bars: str,
    *,
    column_suffix: str = "",
) -> float:
    """Read a requested error-bar size from a wrapper's aggregate row."""

    def value(field: str, fallback: object = np.nan) -> object:
        result = row.get(f"{metric}_{field}{column_suffix}")
        return result if pd.notna(result) else fallback

    def completed_run_count() -> object:
        """Read the run count used by common aggregate-summary schemas."""

        for field in (
            f"completed_repeats{column_suffix}",
            f"n_complete{column_suffix}",
            f"n_complete_runs{column_suffix}",
            "completed_repeats",
            "n_complete",
            "n_complete_runs",
        ):
            result = row.get(field)
            if pd.notna(result):
                return result
        return np.nan

    if error_bars == "none":
        return 0.0
    if error_bars == "ci95":
        half_width = value("normal_95ci_half_width")
        if pd.notna(half_width):
            return float(half_width)
        lower = value("normal_95ci_lower")
        upper = value("normal_95ci_upper")
        if pd.notna(lower) and pd.notna(upper):
            return abs(float(upper) - float(lower)) / 2.0
        sem = value("sem")
        if pd.notna(sem):
            return 1.96 * float(sem)
        standard_deviation = value(
            "sd",
            value("std", row.get(f"standard_deviation{column_suffix}")),
        )
        n = value("n", completed_run_count())
        if pd.notna(standard_deviation) and pd.notna(n) and float(n) > 1:
            critical = T_CRITICAL_95.get(int(float(n)) - 1, 1.96)
            return critical * float(standard_deviation) / np.sqrt(float(n))
    if error_bars == "sd":
        result = value("sd", value("std", row.get(f"standard_deviation{column_suffix}")))
    elif error_bars == "sem":
        result = value("sem")
        if pd.isna(result):
            standard_deviation = value(
                "sd",
                value("std", row.get(f"standard_deviation{column_suffix}")),
            )
            n = value("n", completed_run_count())
            result = (
                float(standard_deviation) / np.sqrt(float(n))
                if pd.notna(standard_deviation) and pd.notna(n) and float(n) > 0
                else np.nan
            )
    else:
        result = np.nan
    return float(result) if pd.notna(result) else 0.0


def comparison_aggregate_table(
    data: pd.DataFrame,
    *,
    group_column: str,
    metrics: Sequence[str],
    error_bars: str,
) -> tuple[pd.DataFrame, str, bool]:
    """Normalize raw or wrapper-aggregate performance tables for plotting.

    The boolean indicates whether the original input is repeat-level and can
    therefore support an optional raw-run overlay.
    """

    require_columns(data, [group_column], context="The comparison plot")
    raw_metric_columns = all(metric in data.columns for metric in metrics)
    wide_aggregate_columns = all(f"{metric}_mean" in data.columns for metric in metrics)
    long_aggregate_columns = {"metric", "mean"}.issubset(data.columns)
    records: list[dict[str, float | str | int]] = []

    if raw_metric_columns:
        categories = ordered_categories(data[group_column])
        for category in categories:
            subset = data.loc[data[group_column].astype(str).eq(category)]
            for metric in metrics:
                mean, error, count = aggregate_values(subset[metric], error_bars)
                if count:
                    records.append({
                        "group": category,
                        "metric": metric,
                        "mean": mean,
                        "error": error,
                        "n": count,
                    })
        return (
            pd.DataFrame(records),
            f"Bars show mean ± {error_bar_label(error_bars)} computed from completed outer repeats.",
            True,
        )

    if wide_aggregate_columns:
        for _, row in data.iterrows():
            group = str(row[group_column])
            for metric in metrics:
                mean = pd.to_numeric(pd.Series([row[f"{metric}_mean"]]), errors="coerce").iloc[0]
                if pd.isna(mean):
                    continue
                n_value = row.get(f"{metric}_n", row.get("n_complete", 0))
                records.append({
                    "group": group,
                    "metric": metric,
                    "mean": float(mean),
                    "error": aggregate_error_from_row(row, metric, error_bars),
                    "n": int(n_value) if pd.notna(n_value) else 0,
                })
        source_note = (
            f"Bars use the runner-reported means ± {error_bar_label(error_bars)}."
        )
    elif long_aggregate_columns:
        subset = data.loc[data["metric"].isin(metrics)].copy()
        for _, row in subset.iterrows():
            mean = pd.to_numeric(pd.Series([row["mean"]]), errors="coerce").iloc[0]
            if pd.isna(mean):
                continue
            metric = str(row["metric"])
            n_value = row.get("completed_repeats", 0)
            records.append({
                "group": str(row[group_column]),
                "metric": metric,
                "mean": float(mean),
                "error": aggregate_error_from_row(row, metric, error_bars),
                "n": int(n_value) if pd.notna(n_value) else 0,
            })
        source_note = (
            f"Bars use the runner-reported means ± {error_bar_label(error_bars)}."
        )
    else:
        raise ValueError(
            "The comparison input must be either a run_summary.csv with raw metric "
            "columns or a wrapper aggregate summary with <metric>_mean columns."
        )

    aggregate = pd.DataFrame(records)
    if aggregate.empty:
        raise ValueError("The aggregate summary contains no finite values for the requested metrics.")
    duplicate_counts = aggregate.groupby(["group", "metric"], dropna=False).size()
    if (duplicate_counts > 1).any():
        duplicates = duplicate_counts.loc[duplicate_counts > 1].index.tolist()
        raise ValueError(
            "The aggregate summary has more than one row for a plotted group/metric "
            f"combination: {duplicates}. Choose a more specific group column or use run_summary.csv."
        )
    return aggregate, source_note, False


def style_axis(axis: plt.Axes, *, zero_reference: bool = False) -> None:
    axis.grid(axis="y", color=SANKY_PALETTE["grid"], linewidth=0.8, zorder=0)
    axis.spines["top"].set_visible(True)
    axis.spines["right"].set_visible(True)
    axis.spines["left"].set_linewidth(0.9)
    axis.spines["bottom"].set_linewidth(0.9)
    axis.spines["top"].set_linewidth(0.9)
    axis.spines["right"].set_linewidth(0.9)
    if zero_reference:
        axis.axhline(0, color=SANKY_PALETTE["gray"], linewidth=0.9, zorder=1)


def resolve_prediction_scenario_column(
    data: pd.DataFrame,
    requested_column: str | None,
) -> str:
    """Choose the scenario field carried by a consolidated prediction export."""

    if requested_column:
        require_columns(data, (requested_column,), context="The predicted-versus-actual plot")
        return requested_column
    if COMPARISON_GROUP_COLUMN in data.columns:
        usable = data[COMPARISON_GROUP_COLUMN].dropna().astype(str).str.strip()
        if not usable.empty and usable.ne("").any():
            return COMPARISON_GROUP_COLUMN

    candidates = [
        column
        for column in LEGACY_COMPARISON_GROUP_COLUMNS
        if column in data.columns
        and data[column].dropna().astype(str).str.strip().ne("").any()
    ]
    varying = [
        column
        for column in candidates
        if data[column].dropna().astype(str).str.strip().nunique() > 1
    ]
    if len(varying) == 1:
        return varying[0]
    if len(candidates) == 1:
        return candidates[0]
    if not candidates:
        raise ValueError(
            "The predicted-versus-actual plot could not find a standardized "
            "'comparison_group' column or a recognized scenario column. Pass "
            "--scenario-column explicitly."
        )
    options = ", ".join(varying or candidates)
    raise ValueError(
        "The scenario dimension is ambiguous in this prediction export (found: "
        f"{options}). Pass --scenario-column once to select it explicitly."
    )


def resolve_prediction_repeat_column(
    data: pd.DataFrame,
    requested_column: str | None,
) -> str | None:
    """Return a run identifier used to color and count overlaid repeats."""

    if requested_column:
        require_columns(data, (requested_column,), context="The predicted-versus-actual plot")
        return requested_column
    return next(
        (
            column
            for column in PREDICTION_REPEAT_COLUMN_CANDIDATES
            if column in data.columns and data[column].notna().any()
        ),
        None,
    )


def prediction_plot_data(
    data: pd.DataFrame,
    *,
    scenario_column: str,
    data_split: str | None,
) -> pd.DataFrame:
    """Select finite predictions for the requested held-out data split."""

    require_columns(
        data,
        ("y_true", "y_pred", scenario_column),
        context="The predicted-versus-actual plot",
    )
    plotted = data.copy()
    if data_split and data_split.casefold() != "all":
        require_columns(data, ("data_split",), context="The predicted-versus-actual plot")
        plotted = plotted.loc[
            plotted["data_split"].astype(str).str.casefold().eq(data_split.casefold())
        ].copy()
        if plotted.empty:
            raise ValueError(
                f"The predicted-versus-actual plot found no rows with data_split={data_split!r}."
            )

    plotted["y_true"] = pd.to_numeric(plotted["y_true"], errors="coerce")
    plotted["y_pred"] = pd.to_numeric(plotted["y_pred"], errors="coerce")
    plotted = plotted.dropna(subset=["y_true", "y_pred", scenario_column]).copy()
    if plotted.empty:
        raise ValueError("The predicted-versus-actual plot needs finite y_true and y_pred values.")
    return plotted


def resolve_prediction_performance_summary_path(
    input_path: Path,
    requested_path: Path | None,
) -> Path | None:
    """Find the aggregate summary associated with a prediction-detail export."""

    if requested_path is not None:
        path = requested_path.expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"Prediction performance summary was not found: {path}")
        return path

    candidates = (
        input_path.parent.parent / "summary" / "performance_summary.csv",
        input_path.parent.parent / "summary" / "scenario_summary.csv",
    )
    return next((path for path in candidates if path.exists()), None)


def filter_prediction_performance_summary(
    data: pd.DataFrame,
    clauses: Sequence[str],
) -> pd.DataFrame:
    """Apply compatible prediction filters without rejecting a reduced summary."""

    filtered = data.copy()
    for clause in clauses:
        column, separator, value = clause.partition("=")
        column = column.strip()
        value = value.strip()
        if not separator or not column or not value:
            raise ValueError(
                "Each --where value must have the form column=value; "
                f"received {clause!r}."
            )
        if column not in filtered.columns:
            continue
        filtered = filtered.loc[filtered[column].astype(str).eq(value)].copy()
    return filtered


def prediction_range_metric_summary_row(
    summary_data: pd.DataFrame | None,
    *,
    scenario_column: str,
    scenario: str,
) -> pd.Series | None:
    """Return one matching aggregate summary row when it is unambiguous."""

    if summary_data is None or scenario_column not in summary_data.columns:
        return None
    metric_columns = (
        f"{prefix}_{metric}_{statistic}"
        for prefix in LOGKD_RANGE_METRIC_PREFIXES
        for metric in PERFORMANCE_METRICS
        for statistic in ("mean", "sd")
    )
    if any(column not in summary_data.columns for column in metric_columns):
        return None
    matched = summary_data.loc[
        summary_data[scenario_column].astype(str).eq(str(scenario))
    ]
    return matched.iloc[0] if len(matched) == 1 else None


def _format_prediction_range_metric(
    mean: float,
    sd: float,
    *,
    precision: int = 2,
) -> str:
    """Format an aggregate metric consistently, including unavailable estimates."""

    if not np.isfinite(mean):
        return "not estimable"
    if not np.isfinite(sd):
        sd = 0.0
    # Removing the spaces around ± saves horizontal room.  The high-logKd
    # range uses one display decimal because its narrow plot region otherwise
    # cannot accommodate the larger, reader-facing annotation text.
    return f"{mean:.{precision}f}±{sd:.{precision}f}"


def prediction_range_metric_blocks_from_summary(summary_row: pd.Series) -> tuple[str, str, str]:
    """Build the compact low/middle/high annotations from exported summary fields."""

    def block(prefix: str, heading: str, *, precision: int = 2) -> str:
        def value(metric: str) -> str:
            return _format_prediction_range_metric(
                float(summary_row[f"{prefix}_{metric}_mean"]),
                float(summary_row[f"{prefix}_{metric}_sd"]),
                precision=precision,
            )

        return "\n".join((
            heading,
            f"MAE {value('mae')}",
            f"RMSE {value('rmse')}",
            f"R² {value('r2')}",
            f"ρ {value('spearman')}",
        ))

    return (
        block("low_logkd", "Low logKd"),
        block("middle_logkd", "Middle logKd"),
        block("high_logkd", "High logKd", precision=1),
    )


def prediction_range_metric_blocks_from_predictions(
    data: pd.DataFrame,
    *,
    repeat_column: str | None,
    repeats: Sequence[str],
    low_threshold: float,
    high_threshold: float,
) -> tuple[str, str, str]:
    """Fallback range annotations for prediction exports without an aggregate summary."""

    strata = (
        ("Low logKd", data["y_true"].lt(low_threshold), 2),
        (
            "Middle logKd",
            data["y_true"].ge(low_threshold) & data["y_true"].le(high_threshold),
            2,
        ),
        ("High logKd", data["y_true"].gt(high_threshold), 1),
    )

    def block(heading: str, mask: pd.Series, precision: int) -> str:
        subset = data.loc[mask]
        groups = (
            [subset.loc[subset[repeat_column].astype(str).eq(repeat)] for repeat in repeats]
            if repeat_column and repeats
            else [subset]
        )
        metric_values: dict[str, list[float]] = {
            "MAE": [], "RMSE": [], "R²": [], "ρ": [],
        }
        for group in groups:
            y_true = group["y_true"].to_numpy(dtype=float)
            y_pred = group["y_pred"].to_numpy(dtype=float)
            if not len(group):
                continue
            residual = y_pred - y_true
            metric_values["MAE"].append(float(np.mean(np.abs(residual))))
            metric_values["RMSE"].append(float(np.sqrt(np.mean(residual ** 2))))
            total_sum_of_squares = float(np.sum((y_true - np.mean(y_true)) ** 2))
            metric_values["R²"].append(
                float(1.0 - np.sum(residual ** 2) / total_sum_of_squares)
                if total_sum_of_squares > 0
                else float("nan")
            )
            metric_values["ρ"].append(
                float(pd.Series(y_true).rank().corr(pd.Series(y_pred).rank()))
                if len(y_true) > 1
                else float("nan")
            )

        lines = [heading]
        for metric, values in metric_values.items():
            finite = np.asarray(values, dtype=float)
            finite = finite[np.isfinite(finite)]
            mean = float(np.mean(finite)) if len(finite) else float("nan")
            sd = float(np.std(finite, ddof=1)) if len(finite) > 1 else 0.0
            lines.append(
                f"{metric} {_format_prediction_range_metric(mean, sd, precision=precision)}"
            )
        return "\n".join(lines)

    return tuple(
        block(heading, mask, precision) for heading, mask, precision in strata
    )  # type: ignore[return-value]


def prediction_range_upper_limit_for_annotations(
    figure: plt.Figure,
    axis: plt.Axes,
    *,
    lower: float,
    upper: float,
    high_threshold: float,
    high_metrics: str,
) -> float:
    """Expand only the upper limit needed to keep high-range values readable."""

    # A high-logKd tail can have a much narrower numerical span than the other
    # two regions.  Measure the real annotation so the high block stays in its
    # own range instead of shrinking the reader-facing metric type.
    axis.set_xlim(lower, upper)
    axis.set_ylim(lower, upper)
    axis.set_aspect("equal", adjustable="box")
    probe = axis.text(
        0.0,
        0.0,
        high_metrics,
        fontsize=PREDICTION_RANGE_METRIC_FONT_SIZE,
        linespacing=PREDICTION_RANGE_METRIC_LINE_SPACING,
    )
    try:
        figure.canvas.draw()
        renderer = figure.canvas.get_renderer()
        text_width = probe.get_window_extent(renderer).width
        axes_width = axis.get_window_extent(renderer).width
    finally:
        probe.remove()
    required_fraction = (
        text_width + PREDICTION_RANGE_ANNOTATION_RIGHT_PADDING_PIXELS
    ) / axes_width
    if required_fraction >= 1.0:
        raise ValueError("The high-logKd metric annotation is wider than the plot panel.")
    high_anchor = high_threshold + PREDICTION_RANGE_HIGH_ANNOTATION_OFFSET
    annotation_upper = (high_anchor - required_fraction * lower) / (1.0 - required_fraction)
    return max(upper, annotation_upper)


def prediction_metric_annotation(
    data: pd.DataFrame,
    *,
    repeat_column: str | None,
    repeats: Sequence[str],
) -> str:
    """Summarize per-repeat prediction metrics as mean plus/minus sample SD."""

    groups = (
        [data.loc[data[repeat_column].astype(str).eq(repeat)] for repeat in repeats]
        if repeat_column and repeats
        else [data]
    )
    metric_values: dict[str, list[float]] = {
        "MAE": [],
        "RMSE": [],
        "R²": [],
        "Spearman": [],
    }
    for group in groups:
        y_true = group["y_true"].to_numpy(dtype=float)
        y_pred = group["y_pred"].to_numpy(dtype=float)
        residual = y_pred - y_true
        metric_values["MAE"].append(float(np.mean(np.abs(residual))))
        metric_values["RMSE"].append(float(np.sqrt(np.mean(residual ** 2))))
        total_sum_of_squares = float(np.sum((y_true - np.mean(y_true)) ** 2))
        metric_values["R²"].append(
            float(1.0 - np.sum(residual ** 2) / total_sum_of_squares)
            if total_sum_of_squares > 0
            else float("nan")
        )
        metric_values["Spearman"].append(
            float(pd.Series(y_true).rank().corr(pd.Series(y_pred).rank()))
            if len(y_true) > 1
            else float("nan")
        )

    lines: list[str] = []
    for label, values in metric_values.items():
        finite = np.asarray(values, dtype=float)
        finite = finite[np.isfinite(finite)]
        if not len(finite):
            continue
        mean = float(np.mean(finite))
        sd = float(np.std(finite, ddof=1)) if len(finite) > 1 else 0.0
        lines.append(f"{label} = {mean:.2f} ± {sd:.2f}")
    return "\n".join(lines)


def plot_predicted_vs_actual(
    data: pd.DataFrame,
    *,
    scenario_column: str,
    scenario: str,
    repeat_column: str | None,
    performance_summary_row: pd.Series | None,
    low_logkd_tail_threshold: float,
    high_logkd_tail_threshold: float,
    title: str | None,
    data_split_label: str,
    target_label: str,
    panel_label: str | None,
    standalone: bool,
) -> plt.Figure:
    """Plot every repeat for one scenario in a standalone predicted-vs-actual figure."""

    subset = data.loc[data[scenario_column].astype(str).eq(scenario)].copy()
    if subset.empty:
        raise ValueError(
            f"The predicted-versus-actual plot found no rows for scenario {scenario!r}."
        )

    repeats = ordered_categories(subset[repeat_column]) if repeat_column else []
    if len(repeats) > len(REPEAT_COLORS):
        raise ValueError(
            "The predicted-versus-actual plot supports at most "
            f"{len(REPEAT_COLORS)} uniquely colored outer repeats; found {len(repeats)}."
        )
    repeat_colors = {
        repeat: REPEAT_COLORS[index]
        for index, repeat in enumerate(repeats)
    }
    scenario_color, marker = category_styles((scenario,))[scenario]
    range_metric_blocks = (
        prediction_range_metric_blocks_from_summary(performance_summary_row)
        if performance_summary_row is not None
        else prediction_range_metric_blocks_from_predictions(
            subset,
            repeat_column=repeat_column,
            repeats=repeats,
            low_threshold=low_logkd_tail_threshold,
            high_threshold=high_logkd_tail_threshold,
        )
    )
    figure, axis = plt.subplots(figsize=PREDICTION_RANGE_FIGURE_SIZE)
    figure.subplots_adjust(
        left=0.13,
        right=0.97,
        bottom=0.11,
        top=0.72 if standalone and len(repeats) > 1 else 0.84 if standalone else 0.98,
    )
    values = subset[["y_true", "y_pred"]].to_numpy(dtype=float).ravel()
    lower, upper = float(np.min(values)), float(np.max(values))
    padding = max(0.08 * (upper - lower), 0.10)
    lower -= padding
    upper += padding
    high_heading, high_metrics = range_metric_blocks[-1].split("\n", maxsplit=1)
    del high_heading
    upper = prediction_range_upper_limit_for_annotations(
        figure,
        axis,
        lower=lower,
        upper=upper,
        high_threshold=high_logkd_tail_threshold,
        high_metrics=high_metrics,
    )

    if repeats:
        for repeat in repeats:
            repeat_subset = subset.loc[subset[repeat_column].astype(str).eq(repeat)]
            if repeat_subset.empty:
                continue
            axis.scatter(
                repeat_subset["y_true"],
                repeat_subset["y_pred"],
                color=repeat_colors[repeat],
                marker=marker,
                s=35,
                alpha=0.55,
                edgecolor="none",
                rasterized=True,
                zorder=3,
            )
    else:
        axis.scatter(
            subset["y_true"],
            subset["y_pred"],
            color=scenario_color,
            marker=marker,
            s=35,
            alpha=0.26,
            edgecolor="none",
            rasterized=True,
            zorder=3,
        )
    axis.plot(
        (lower, upper),
        (lower, upper),
        color=SANKY_PALETTE["ink"],
        linestyle="--",
        linewidth=1.0,
        zorder=2,
    )
    for threshold in (low_logkd_tail_threshold, high_logkd_tail_threshold):
        if lower < threshold < upper:
            axis.axvline(
                threshold,
                color=SANKY_PALETTE["gray"],
                linestyle=(0, (4, 3)),
                linewidth=1.15,
                zorder=2,
            )
    annotation_positions = (
        lower + 0.12,
        low_logkd_tail_threshold + 0.15,
        high_logkd_tail_threshold + PREDICTION_RANGE_HIGH_ANNOTATION_OFFSET,
    )
    for x, annotation in zip(annotation_positions, range_metric_blocks, strict=True):
        heading, metrics = annotation.split("\n", maxsplit=1)
        axis.text(
            x,
            upper - PREDICTION_RANGE_HEADING_TOP_OFFSET,
            heading,
            ha="left",
            va="top",
            fontsize=PREDICTION_RANGE_HEADING_FONT_SIZE,
            fontweight="bold",
            color=SANKY_PALETTE["ink"],
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.88, "pad": 1.5},
            zorder=6,
        )
        axis.text(
            x,
            upper - PREDICTION_RANGE_METRIC_TOP_OFFSET,
            metrics,
            ha="left",
            va="top",
            fontsize=PREDICTION_RANGE_METRIC_FONT_SIZE,
            color=SANKY_PALETTE["ink"],
            linespacing=PREDICTION_RANGE_METRIC_LINE_SPACING,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.93, "pad": 1.5},
            zorder=6,
        )
    axis.set_xlabel(f"Actual {target_label}", fontsize=18)
    axis.set_ylabel(f"Predicted {target_label}", fontsize=18)
    axis.set_xlim(lower, upper)
    axis.set_ylim(lower, upper)
    axis.set_aspect("equal", adjustable="box")
    style_axis(axis)
    axis.grid(False)
    axis.spines["top"].set_visible(True)
    axis.spines["right"].set_visible(True)
    axis.spines["top"].set_linewidth(0.9)
    axis.spines["right"].set_linewidth(0.9)
    axis.tick_params(width=1.3, length=7, labelsize=13)
    for spine in axis.spines.values():
        spine.set_linewidth(1.3)
    if standalone:
        figure.suptitle(
            (
                f"{title} - {display_label(scenario)}"
                if title
                else f"{display_label(scenario)}: predicted versus actual {target_label}"
            ),
            y=0.98,
            fontsize=13,
            fontweight="bold",
        )
    else:
        axis.text(
            0.97,
            0.04,
            panel_label or prediction_panel_label(scenario),
            transform=axis.transAxes,
            ha="right",
            va="bottom",
            fontsize=11,
            color=SANKY_PALETTE["gray"],
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.84, "pad": 1.7},
            zorder=6,
        )
    if standalone and len(repeats) > 1:
        repeat_handles = [
            Line2D(
                [],
                [],
                color=color,
                marker="o",
                linestyle="None",
                markersize=6,
                label=f"Repeat {display_label(repeat)}",
            )
            for repeat, color in repeat_colors.items()
        ]
        add_figure_legend(
            figure,
            repeat_handles,
            title="Outer repeat",
            y=0.945,
        )
    if standalone:
        repeat_note = "Color identifies outer repeat; " if len(repeats) > 1 else ""
        count_note = f"n = {len(subset):,} {data_split_label.casefold()} predictions"
        if repeats:
            count_note += f" across {len(repeats)} outer repeats"
        figure.text(
            0.5,
            0.045,
            f"{repeat_note}{count_note}; the dashed line is perfect agreement.",
            ha="center",
            va="bottom",
            fontsize=8.5,
            color=SANKY_PALETTE["gray"],
        )
    return figure


def save_figure(
    figure: plt.Figure,
    *,
    input_path: Path,
    output_dir: Path | None,
    output_stem: str | None,
    dpi: int,
) -> list[Path]:
    directory = output_dir if output_dir is not None else input_path.parent / "figures"
    directory.mkdir(parents=True, exist_ok=True)
    stem = output_stem or input_path.stem
    path = directory / f"{stem}.png"
    figure.savefig(path, dpi=dpi, bbox_inches="tight", pad_inches=0.05)
    plt.close(figure)
    return [path]


def add_figure_legend(
    figure: plt.Figure,
    handles: Sequence[object],
    *,
    title: str,
    y: float = 1.01,
) -> None:
    if handles:
        figure.legend(
            handles=handles,
            loc="upper center",
            bbox_to_anchor=(0.5, y),
            ncol=min(5, len(handles)),
            title=title,
            columnspacing=1.2,
            handletextpad=0.5,
        )


def row_major_legend_handles(
    handles: Sequence[object],
    *,
    ncol: int,
) -> list[object]:
    """Reorder legend handles because Matplotlib fills legend columns first."""

    if ncol < 1:
        raise ValueError("ncol must be at least 1.")
    rows = math.ceil(len(handles) / ncol)
    return [
        handles[row * ncol + column]
        for column in range(ncol)
        for row in range(rows)
        if row * ncol + column < len(handles)
    ]


def plot_group_comparison(
    data: pd.DataFrame,
    *,
    group_column: str,
    metrics: Sequence[str],
    group_order: Sequence[str] | None,
    error_bars: str,
    show_runs: bool,
    title: str | None,
    standalone: bool,
) -> plt.Figure:
    aggregate, source_note, input_is_raw = comparison_aggregate_table(
        data,
        group_column=group_column,
        metrics=metrics,
        error_bars=error_bars,
    )
    metrics = [metric for metric in metrics if metric in set(aggregate["metric"])]
    categories = ordered_categories(aggregate["group"], group_order)
    if len(categories) < 2:
        raise ValueError("The comparison plot needs at least two non-empty groups.")
    if show_runs and not input_is_raw:
        raise ValueError(
            "--show-runs needs run_summary.csv because aggregate summaries do not "
            "contain the individual outer-repeat values."
        )
    styles = category_styles(categories)
    figure, axis = plt.subplots(
        figsize=(10.8, 5.8) if standalone else (6.6, 6.6)
    )
    metric_positions = np.arange(len(metrics), dtype=float)
    bar_width = min(0.76 / len(categories), 0.18)
    category_offsets = (
        np.arange(len(categories), dtype=float) - (len(categories) - 1) / 2
    ) * bar_width
    lower_bounds: list[float] = []
    upper_bounds: list[float] = []

    for category_index, category in enumerate(categories):
        subset = aggregate.loc[aggregate["group"].astype(str).eq(category)].set_index("metric")
        means: list[float] = []
        errors: list[float] = []
        for metric in metrics:
            if metric in subset.index:
                row = subset.loc[metric]
                means.append(float(row["mean"]))
                errors.append(float(row["error"]))
            else:
                means.append(float("nan"))
                errors.append(0.0)
        positions = metric_positions + category_offsets[category_index]
        color, marker = styles[category]
        axis.bar(
            positions,
            means,
            width=bar_width,
            yerr=errors if error_bars != "none" else None,
            capsize=3.5,
            error_kw={
                "ecolor": SANKY_PALETTE["ink"],
                "elinewidth": 1.0,
                "capthick": 1.0,
            },
            color=color,
            edgecolor=SANKY_PALETTE["ink"],
            linewidth=0.85,
            zorder=2,
        )
        if show_runs:
            for metric_index, metric in enumerate(metrics):
                values = pd.to_numeric(
                    data.loc[data[group_column].astype(str).eq(category), metric],
                    errors="coerce",
                ).dropna().to_numpy()
                if not len(values):
                    continue
                offsets = (
                    np.linspace(-bar_width * 0.24, bar_width * 0.24, len(values))
                    if len(values) > 1
                    else np.array([0.0])
                )
                axis.scatter(
                    np.full(len(values), positions[metric_index]) + offsets,
                    values,
                    s=20,
                    marker=marker,
                    facecolor="white",
                    edgecolor=SANKY_PALETTE["ink"],
                    linewidth=0.55,
                    zorder=3,
                )
        lower_bounds.extend(np.asarray(means) - np.asarray(errors))
        upper_bounds.extend(np.asarray(means) + np.asarray(errors))

    finite_lower = np.asarray(lower_bounds, dtype=float)
    finite_upper = np.asarray(upper_bounds, dtype=float)
    finite_lower = finite_lower[np.isfinite(finite_lower)]
    finite_upper = finite_upper[np.isfinite(finite_upper)]
    if len(finite_lower) and finite_lower.min() >= 0:
        axis.set_ylim(bottom=0)
    style_axis(
        axis,
        zero_reference=bool(
            len(finite_lower)
            and finite_lower.min() < 0 < finite_upper.max()
        ),
    )
    axis.set_xticks(metric_positions, [metric_axis_label(metric) for metric in metrics])
    axis.set_ylabel("Metric value")
    legend_handles = [
        Patch(
            facecolor=styles[category][0],
            edgecolor=SANKY_PALETTE["ink"],
            label=comparison_legend_label(category),
        )
        for index, category in enumerate(categories)
    ]
    if standalone:
        figure.suptitle(
            title or f"Held-out performance by {comparison_group_label(group_column)}",
            y=0.985,
            fontsize=13,
            fontweight="bold",
        )
        figure.text(
            0.5,
            0.945,
            source_note,
            ha="center",
            va="center",
            fontsize=8.5,
            color=SANKY_PALETTE["gray"],
        )
        add_figure_legend(
            figure,
            legend_handles,
            title=comparison_group_label(group_column),
            y=0.915,
        )
        figure.subplots_adjust(top=0.76, bottom=0.14, left=0.09, right=0.98)
    else:
        # Short feature-family labels fit three columns without stretching the
        # legend across the panel width, keeping it aligned with other panels.
        legend_columns = min(3, len(legend_handles))
        legend_rows = math.ceil(len(legend_handles) / legend_columns)
        figure.subplots_adjust(
            top=max(0.69, 0.94 - 0.03 * legend_rows),
            bottom=0.16,
            left=0.14,
            right=0.98,
        )
        figure.legend(
            handles=row_major_legend_handles(legend_handles, ncol=legend_columns),
            loc="upper left",
            bbox_to_anchor=(axis.get_position().x0, 0.99),
            bbox_transform=figure.transFigure,
            ncol=legend_columns,
            frameon=False,
            fontsize=8.5,
            columnspacing=1.1,
            handletextpad=0.45,
            borderaxespad=0.0,
            borderpad=0.0,
        )
        enlarge_panel_text(figure)
    return figure


def plot_group_comparison_with_retention(
    data: pd.DataFrame,
    *,
    group_column: str,
    metrics: Sequence[str],
    retention_column: str,
    group_order: Sequence[str] | None,
    error_bars: str,
    show_runs: bool,
    title: str | None,
    standalone: bool,
) -> plt.Figure:
    """Compare performance bars and one compact retention diagnostic in one panel."""

    performance, performance_note, input_is_raw = comparison_aggregate_table(
        data,
        group_column=group_column,
        metrics=metrics,
        error_bars=error_bars,
    )
    try:
        retention, _retention_note, retention_is_raw = comparison_aggregate_table(
            data,
            group_column=group_column,
            metrics=(retention_column,),
            error_bars=error_bars,
        )
    except ValueError as error:
        raise ValueError(
            "--include-retention needs the compact retention field "
            f"{retention_column!r}. This summary does not contain it. Use a newly "
            "generated summary or select another explicit --retention-column; the "
            "script will not silently average component retention diagnostics."
        ) from error
    metrics = [metric for metric in metrics if metric in set(performance["metric"])]
    categories = ordered_categories(performance["group"], group_order)
    if len(categories) < 2:
        raise ValueError("The comparison plot needs at least two non-empty groups.")
    if show_runs and (not input_is_raw or not retention_is_raw):
        raise ValueError(
            "--show-runs with --include-retention needs run_summary.csv because "
            "aggregate summaries do not contain individual outer-repeat values."
        )

    retention_by_group = retention.set_index("group")
    missing_retention_groups = [
        category for category in categories if category not in retention_by_group.index
    ]
    if missing_retention_groups:
        raise ValueError(
            "The requested retention diagnostic is unavailable for comparison groups: "
            + ", ".join(missing_retention_groups)
        )

    metric_colors = (
        SANKY_PALETTE["blue"],
        SANKY_PALETTE["coral"],
        SANKY_PALETTE["green"],
        SANKY_PALETTE["yellow"],
    )
    metric_hatches = ("", "//", "..", "xx", "\\\\", "oo", "--", "++")
    figure, axis = plt.subplots(
        figsize=(11.2, 6.1) if standalone else (7.0, 4.8)
    )
    positions = np.arange(len(categories), dtype=float)
    bar_width = min(0.78 / len(metrics), 0.18)
    metric_offsets = (
        np.arange(len(metrics), dtype=float) - (len(metrics) - 1) / 2
    ) * bar_width
    lower_bounds: list[float] = []
    upper_bounds: list[float] = []

    for metric_index, metric in enumerate(metrics):
        metric_rows = performance.loc[performance["metric"].eq(metric)].set_index("group")
        means: list[float] = []
        errors: list[float] = []
        for category in categories:
            if category in metric_rows.index:
                row = metric_rows.loc[category]
                means.append(float(row["mean"]))
                errors.append(float(row["error"]))
            else:
                means.append(float("nan"))
                errors.append(0.0)
        bar_positions = positions + metric_offsets[metric_index]
        axis.bar(
            bar_positions,
            means,
            width=bar_width,
            yerr=errors if error_bars != "none" else None,
            capsize=3.5,
            error_kw={
                "ecolor": SANKY_PALETTE["ink"],
                "elinewidth": 1.0,
                "capthick": 1.0,
            },
            color=metric_colors[metric_index % len(metric_colors)],
            edgecolor=SANKY_PALETTE["ink"],
            linewidth=0.85,
            hatch=metric_hatches[metric_index % len(metric_hatches)],
            zorder=2,
        )
        if show_runs:
            for category_index, category in enumerate(categories):
                values = pd.to_numeric(
                    data.loc[data[group_column].astype(str).eq(category), metric],
                    errors="coerce",
                ).dropna().to_numpy()
                if not len(values):
                    continue
                offsets = (
                    np.linspace(-bar_width * 0.24, bar_width * 0.24, len(values))
                    if len(values) > 1
                    else np.array([0.0])
                )
                axis.scatter(
                    np.full(len(values), bar_positions[category_index]) + offsets,
                    values,
                    s=18,
                    marker="o",
                    facecolor="white",
                    edgecolor=SANKY_PALETTE["ink"],
                    linewidth=0.5,
                    zorder=3,
                )
        lower_bounds.extend(np.asarray(means) - np.asarray(errors))
        upper_bounds.extend(np.asarray(means) + np.asarray(errors))

    finite_lower = np.asarray(lower_bounds, dtype=float)
    finite_upper = np.asarray(upper_bounds, dtype=float)
    finite_lower = finite_lower[np.isfinite(finite_lower)]
    finite_upper = finite_upper[np.isfinite(finite_upper)]
    if len(finite_lower) and finite_lower.min() >= 0:
        axis.set_ylim(bottom=0)
    style_axis(
        axis,
        zero_reference=bool(
            len(finite_lower) and finite_lower.min() < 0 < finite_upper.max()
        ),
    )
    axis.set_xticks(positions, [display_label(category) for category in categories])
    axis.tick_params(axis="x", labelrotation=12)
    for label in axis.get_xticklabels():
        label.set_horizontalalignment("right")
    axis.set_ylabel("Held-out metric value")

    retention_means = [float(retention_by_group.loc[category, "mean"]) for category in categories]
    retention_errors = [float(retention_by_group.loc[category, "error"]) for category in categories]
    retention_label = metric_label(retention_column)
    retention_axis = axis.twinx()
    retention_axis.errorbar(
        positions,
        retention_means,
        yerr=retention_errors if error_bars != "none" else None,
        color=SANKY_PALETTE["gray"],
        linestyle=":",
        linewidth=2.0,
        marker="o",
        markersize=6,
        markerfacecolor="white",
        markeredgecolor=SANKY_PALETTE["ink"],
        markeredgewidth=0.9,
        capsize=3.5,
        elinewidth=1.0,
        zorder=4,
    )
    if show_runs:
        for category_index, category in enumerate(categories):
            values = pd.to_numeric(
                data.loc[data[group_column].astype(str).eq(category), retention_column],
                errors="coerce",
            ).dropna().to_numpy()
            if len(values):
                offsets = (
                    np.linspace(-bar_width * 0.30, bar_width * 0.30, len(values))
                    if len(values) > 1
                    else np.array([0.0])
                )
                retention_axis.scatter(
                    np.full(len(values), positions[category_index]) + offsets,
                    values,
                    s=16,
                    facecolor="white",
                    edgecolor=SANKY_PALETTE["gray"],
                    linewidth=0.5,
                    zorder=5,
                )
    # Scores can equal 1.00.  Keep a small headroom so their open-circle markers
    # and uncertainty caps are not clipped at the top of the plotting area.
    retention_axis.set_ylim(0, 1.04)
    retention_axis.set_ylabel(retention_label)
    retention_axis.spines["top"].set_visible(False)
    retention_axis.spines["right"].set_linewidth(0.9)
    retention_axis.tick_params(axis="y", colors=SANKY_PALETTE["gray"])

    legend_handles = [
        Patch(
            facecolor=metric_colors[index % len(metric_colors)],
            edgecolor=SANKY_PALETTE["ink"],
            hatch=metric_hatches[index % len(metric_hatches)],
            label=metric_axis_label(metric),
        )
        for index, metric in enumerate(metrics)
    ]
    legend_handles.append(
        Line2D(
            [0],
            [0],
            color=SANKY_PALETTE["gray"],
            linestyle=":",
            linewidth=2.0,
            marker="o",
            markerfacecolor="white",
            markeredgecolor=SANKY_PALETTE["ink"],
            label=retention_label,
        )
    )
    if standalone:
        figure.suptitle(
            title
            or f"Held-out performance and retention by {comparison_group_label(group_column)}",
            y=0.985,
            fontsize=13,
            fontweight="bold",
        )
        figure.text(
            0.5,
            0.945,
            performance_note + f" The dotted line shows {retention_label} on the right axis.",
            ha="center",
            va="center",
            fontsize=8.5,
            color=SANKY_PALETTE["gray"],
        )
        add_figure_legend(figure, legend_handles, title="Bars / diagnostic", y=0.915)
        figure.subplots_adjust(top=0.76, bottom=0.18, left=0.09, right=0.90)
    else:
        add_panel_context(axis, panel_context_label(data, group_column=group_column))
        figure.subplots_adjust(top=0.96, bottom=0.18, left=0.12, right=0.89)
    return figure


def plot_reference_retention_profile(
    data: pd.DataFrame,
    *,
    group_column: str,
    group_order: Sequence[str] | None,
    error_bars: str,
    title: str | None,
    standalone: bool,
) -> plt.Figure:
    """Show database/training contrast and conditional retention in one matrix."""

    metrics = tuple(
        metric
        for _panel_title, series in REFERENCE_RETENTION_PROFILE_PANELS
        for _block_label, metric in series
    )
    try:
        aggregate, _source_note, input_is_raw = comparison_aggregate_table(
            data,
            group_column=group_column,
            metrics=metrics,
            error_bars=error_bars,
        )
    except ValueError as error:
        raise ValueError(
            "The retention-profile plot needs all nine database_contrast, "
            "training_contrast, and contrast_retention diagnostics. This input "
            "does not contain the complete profile."
        ) from error
    categories = ordered_categories(aggregate["group"], group_order)
    if len(categories) < 2:
        raise ValueError("The retention-profile plot needs at least two non-empty groups.")

    rows: list[tuple[str, str]] = []
    row_labels: list[str] = []
    for panel_title, series in REFERENCE_RETENTION_PROFILE_PANELS:
        for block_label, metric in series:
            rows.append((panel_title, metric))
            row_labels.append(block_label)

    means = np.full((len(rows), len(categories)), np.nan, dtype=float)
    errors = np.zeros((len(rows), len(categories)), dtype=float)
    for row_index, (_panel_title, metric) in enumerate(rows):
        metric_rows = aggregate.loc[aggregate["metric"].eq(metric)].set_index("group")
        missing = [category for category in categories if category not in metric_rows.index]
        if missing:
            raise ValueError(
                f"Retention metric {metric!r} is missing comparison groups: "
                + ", ".join(missing)
            )
        means[row_index] = [float(metric_rows.loc[category, "mean"]) for category in categories]
        errors[row_index] = [float(metric_rows.loc[category, "error"]) for category in categories]

    figure, axis = plt.subplots(
        figsize=(11.8, 7.4) if standalone else (8.2, 5.2)
    )
    color_map = LinearSegmentedColormap.from_list(
        "reference_retention",
        ("#F7F7F7", "#BFE8F5", SANKY_PALETTE["blue"]),
    )
    image = axis.imshow(
        means,
        cmap=color_map,
        vmin=0,
        vmax=1,
        aspect="auto",
        interpolation="nearest",
    )
    axis.set_xticks(np.arange(len(categories)), [display_label(category) for category in categories])
    axis.set_yticks(np.arange(len(row_labels)), row_labels)
    axis.tick_params(axis="x", labelrotation=15, length=0)
    axis.tick_params(axis="y", length=0)
    for label in axis.get_xticklabels():
        label.set_horizontalalignment("right")
    axis.set_xticks(np.arange(-0.5, len(categories), 1), minor=True)
    axis.set_yticks(np.arange(-0.5, len(rows), 1), minor=True)
    axis.grid(which="minor", color="white", linewidth=1.6)
    axis.tick_params(which="minor", bottom=False, left=False)
    for boundary in (2.5, 5.5):
        axis.axhline(boundary, color=SANKY_PALETTE["ink"], linewidth=1.3)
    for panel_index, (panel_title, _series) in enumerate(REFERENCE_RETENTION_PROFILE_PANELS):
        axis.text(
            -0.40,
            panel_index * 3 + 1,
            panel_title,
            transform=axis.get_yaxis_transform(),
            ha="right",
            va="center",
            fontsize=9,
            fontweight="bold",
            color=SANKY_PALETTE["ink"],
            clip_on=False,
        )
    for row_index in range(len(rows)):
        for column_index in range(len(categories)):
            label = (
                f"{means[row_index, column_index]:.1%}"
                if error_bars == "none"
                else f"{means[row_index, column_index]:.1%}\n±{errors[row_index, column_index]:.1%}"
            )
            axis.text(
                column_index,
                row_index,
                label,
                ha="center",
                va="center",
                fontsize=7.5,
                color=SANKY_PALETTE["ink"],
            )
    axis.spines["top"].set_visible(True)
    axis.spines["right"].set_visible(True)
    axis.spines["left"].set_linewidth(0.9)
    axis.spines["bottom"].set_linewidth(0.9)
    axis.spines["top"].set_linewidth(0.9)
    axis.spines["right"].set_linewidth(0.9)
    colorbar = figure.colorbar(image, ax=axis, fraction=0.042, pad=0.025)
    colorbar.set_ticks(np.linspace(0, 1, 6))
    colorbar.set_ticklabels([f"{value:.0%}" for value in np.linspace(0, 1, 6)])
    colorbar.set_label("Mean percentage")
    if not standalone:
        add_panel_context(
            axis,
            panel_context_label(data, group_column=group_column),
            y=1.02,
            vertical_alignment="bottom",
        )
        figure.subplots_adjust(top=0.92, bottom=0.14, left=0.36, right=0.92)
        return figure
    figure.suptitle(
        title or f"Within-study contrast support and retention by {comparison_group_label(group_column)}",
        y=0.985,
        fontsize=13,
        fontweight="bold",
    )
    uncertainty_note = (
        "Each cell reports mean only."
        if error_bars == "none"
        else "Each cell reports runner-reported mean ± 95% normal confidence interval."
        if error_bars == "ci95" and not input_is_raw
        else f"Each cell reports mean ± {error_bar_label(error_bars)}."
    )
    figure.text(
        0.5,
        0.935,
        uncertainty_note,
        ha="center",
        va="center",
        fontsize=8.5,
        color=SANKY_PALETTE["gray"],
    )
    figure.text(
        0.5,
        0.045,
        "Database contrast: within-study contrasts in the complete pre-split cohort. "
        "Training contrast: retained contrasts on the same database scale. "
        "Contrast retention: retained contrasts divided by database contrasts.",
        ha="center",
        va="center",
        fontsize=8.2,
        color=SANKY_PALETTE["gray"],
        wrap=True,
    )
    figure.subplots_adjust(top=0.86, bottom=0.17, left=0.39, right=0.92)
    return figure


def allocation_reference_method(
    strategy: str,
    available_methods: Sequence[str],
    requested_reference: str | None,
) -> str:
    if requested_reference:
        if requested_reference not in available_methods:
            raise ValueError(
                f"Reference method {requested_reference!r} is not available for "
                f"the {strategy!r} allocation comparison."
            )
        return requested_reference
    preferred = "random_row" if strategy == "random_row" else "random_group"
    if preferred in available_methods:
        return preferred
    random_methods = [method for method in available_methods if method.startswith("random")]
    if len(random_methods) == 1:
        return random_methods[0]
    raise ValueError(
        f"Could not identify the seeded-random reference method for {strategy!r}. "
        "Pass --reference-method explicitly."
    )


def paired_allocation_rows(
    data: pd.DataFrame,
    *,
    reference_method: str | None,
    challenger_method: str,
) -> tuple[pd.DataFrame, dict[str, str]]:
    """Pair one aggregate row per allocation method and split strategy."""

    require_columns(
        data,
        ("split_strategy", "inner_allocation_method"),
        context="The aggregate allocation plot",
    )
    available_challengers = ordered_categories(data["inner_allocation_method"])
    if challenger_method not in available_challengers:
        available_text = ", ".join(available_challengers) or "none"
        raise ValueError(
            "The aggregate allocation plot requires performance_summary.csv "
            f"with challenger method {challenger_method!r}. This input contains: "
            f"{available_text}."
        )

    pairs: list[pd.DataFrame] = []
    references: dict[str, str] = {}
    for strategy in ordered_categories(data["split_strategy"]):
        subset = data.loc[data["split_strategy"].astype(str).eq(strategy)].copy()
        methods = ordered_categories(subset["inner_allocation_method"])
        reference = allocation_reference_method(strategy, methods, reference_method)
        if challenger_method not in methods:
            continue
        reference_rows = subset.loc[
            subset["inner_allocation_method"].astype(str).eq(reference)
        ].copy()
        challenger_rows = subset.loc[
            subset["inner_allocation_method"].astype(str).eq(challenger_method)
        ].copy()
        if len(reference_rows) != 1 or len(challenger_rows) != 1:
            raise ValueError(
                "The aggregate allocation summary must contain exactly one row per "
                f"strategy and allocation method; strategy {strategy!r} has "
                f"{len(reference_rows)} reference rows and {len(challenger_rows)} "
                "challenger rows."
            )
        merged = reference_rows.merge(
            challenger_rows,
            on=["split_strategy"],
            how="inner",
            suffixes=("_reference", "_challenger"),
            validate="one_to_one",
        )
        pairs.append(merged)
        references[strategy] = reference
    if not pairs:
        raise ValueError(
            f"No aggregate allocation pairs were found for challenger method {challenger_method!r}."
        )
    return pd.concat(pairs, ignore_index=True, sort=False), references


def plot_allocation_comparison(
    data: pd.DataFrame,
    *,
    metrics: Sequence[str],
    reference_method: str | None,
    challenger_method: str,
    error_bars: str,
    title: str | None,
    standalone: bool,
) -> plt.Figure:
    metrics = usable_aggregate_metrics(data, metrics)
    pairs, _references = paired_allocation_rows(
        data,
        reference_method=reference_method,
        challenger_method=challenger_method,
    )
    strategies = ordered_categories(pairs["split_strategy"])
    figure, axes = axes_grid(len(metrics))

    for axis, metric in zip(axes, metrics, strict=False):
        reference_column = f"{metric}_mean_reference"
        challenger_column = f"{metric}_mean_challenger"
        if reference_column not in pairs or challenger_column not in pairs:
            axis.set_visible(False)
            continue
        positions = np.arange(len(strategies), dtype=float)
        reference_means: list[float] = []
        reference_errors: list[float] = []
        challenger_means: list[float] = []
        challenger_errors: list[float] = []
        plotted = 0
        for strategy in strategies:
            subset = pairs.loc[pairs["split_strategy"].astype(str).eq(strategy)]
            reference_values = pd.to_numeric(subset[reference_column], errors="coerce")
            challenger_values = pd.to_numeric(subset[challenger_column], errors="coerce")
            valid = reference_values.notna() & challenger_values.notna()
            if not valid.any():
                reference_means.append(float("nan"))
                reference_errors.append(0.0)
                challenger_means.append(float("nan"))
                challenger_errors.append(0.0)
                continue
            reference_values = reference_values.loc[valid]
            challenger_values = challenger_values.loc[valid]
            aggregate_row = subset.loc[valid].iloc[0]
            reference_mean = float(reference_values.iloc[0])
            reference_error = aggregate_error_from_row(
                aggregate_row,
                metric,
                error_bars,
                column_suffix="_reference",
            )
            challenger_mean = float(challenger_values.iloc[0])
            challenger_error = aggregate_error_from_row(
                aggregate_row,
                metric,
                error_bars,
                column_suffix="_challenger",
            )
            reference_means.append(reference_mean)
            reference_errors.append(reference_error)
            challenger_means.append(challenger_mean)
            challenger_errors.append(challenger_error)
            plotted += int(valid.sum())
        if not plotted:
            axis.set_visible(False)
            continue
        width = 0.30
        axis.bar(
            positions - width / 2,
            reference_means,
            width=width,
            yerr=reference_errors if error_bars != "none" else None,
            capsize=4,
            error_kw={"ecolor": SANKY_PALETTE["ink"], "elinewidth": 1.0, "capthick": 1.0},
            color=SANKY_PALETTE["blue"],
            edgecolor=SANKY_PALETTE["ink"],
            linewidth=0.9,
            zorder=2,
            label="Seeded random",
        )
        axis.bar(
            positions + width / 2,
            challenger_means,
            width=width,
            yerr=challenger_errors if error_bars != "none" else None,
            capsize=4,
            error_kw={"ecolor": SANKY_PALETTE["ink"], "elinewidth": 1.0, "capthick": 1.0},
            color=SANKY_PALETTE["coral"],
            edgecolor=SANKY_PALETTE["ink"],
            linewidth=0.9,
            hatch="//",
            zorder=2,
            label="Evidence-balanced",
        )
        axis.set_title(metric_label(metric))
        axis.set_xticks(
            positions,
            [display_label(strategy) for strategy in strategies],
            rotation=20,
            ha="right",
        )
        lower_bounds = np.asarray(reference_means + challenger_means) - np.asarray(
            reference_errors + challenger_errors
        )
        upper_bounds = np.asarray(reference_means + challenger_means) + np.asarray(
            reference_errors + challenger_errors
        )
        if np.nanmin(lower_bounds) >= 0:
            axis.set_ylim(bottom=0)
        style_axis(
            axis,
            zero_reference=bool(np.nanmin(lower_bounds) < 0 < np.nanmax(upper_bounds)),
        )

    for axis in axes[len(metrics):]:
        axis.set_visible(False)
    if not standalone:
        for axis in axes:
            if axis.get_visible():
                add_panel_context(
                    axis,
                    panel_context_label(data, group_column="inner_allocation_method"),
                )
                break
        figure.subplots_adjust(top=0.95, bottom=0.14, left=0.09, right=0.98)
        return figure
    allocation_handles = [
        Patch(facecolor=SANKY_PALETTE["blue"], edgecolor=SANKY_PALETTE["ink"], label="Seeded random"),
        Patch(facecolor=SANKY_PALETTE["coral"], edgecolor=SANKY_PALETTE["ink"], label="Evidence-balanced"),
    ]
    figure.suptitle(
        title or "Paired inner-allocation comparison",
        y=0.985,
        fontsize=13,
        fontweight="bold",
    )
    figure.text(
        0.5,
        0.965,
        f"Bars show mean ± {error_bar_label(error_bars)} across paired outer repeats.",
        ha="center",
        va="center",
        fontsize=8.5,
        color=SANKY_PALETTE["gray"],
    )
    add_figure_legend(
        figure,
        allocation_handles,
        title="Inner allocation method",
        y=0.945,
    )
    figure.subplots_adjust(top=0.88)
    return figure


def plot_retention_relationship(
    data: pd.DataFrame,
    *,
    retention_column: str,
    performance_metric: str,
    group_column: str | None,
    group_order: Sequence[str] | None,
    label_column: str | None,
    title: str | None,
    standalone: bool,
) -> plt.Figure:
    require_columns(
        data,
        (retention_column, performance_metric),
        context="The retention diagnostic plot",
    )
    columns = [retention_column, performance_metric]
    if group_column:
        require_columns(data, (group_column,), context="The retention diagnostic plot")
        columns.append(group_column)
    if label_column:
        require_columns(data, (label_column,), context="The retention diagnostic plot")
        columns.append(label_column)
    plotted = data.loc[:, columns].copy()
    plotted[retention_column] = pd.to_numeric(plotted[retention_column], errors="coerce")
    plotted[performance_metric] = pd.to_numeric(plotted[performance_metric], errors="coerce")
    plotted = plotted.dropna(subset=[retention_column, performance_metric])
    if len(plotted) < 3:
        raise ValueError("The retention diagnostic plot needs at least three finite runs.")

    figure, axis = plt.subplots(
        figsize=(6.5, 5.0) if standalone else (5.6, 4.5),
        constrained_layout=True,
    )
    if group_column:
        categories = ordered_categories(plotted[group_column], group_order)
        styles = category_styles(categories)
        for category in categories:
            subset = plotted.loc[plotted[group_column].astype(str).eq(category)]
            color, marker = styles[category]
            axis.scatter(
                subset[retention_column],
                subset[performance_metric],
                color=color,
                marker=marker,
                edgecolor=SANKY_PALETTE["ink"],
                linewidth=0.5,
                s=52,
                label=display_label(category),
                zorder=3,
            )
    else:
        axis.scatter(
            plotted[retention_column],
            plotted[performance_metric],
            color=SANKY_PALETTE["blue"],
            edgecolor=SANKY_PALETTE["ink"],
            linewidth=0.5,
            s=52,
            zorder=3,
        )
    if label_column:
        for _, row in plotted.iterrows():
            axis.annotate(
                str(row[label_column]),
                (row[retention_column], row[performance_metric]),
                xytext=(4, 4),
                textcoords="offset points",
                fontsize=7.5,
            )
    rho = plotted[retention_column].rank().corr(plotted[performance_metric].rank())
    if standalone: axis.text(
        0.02,
        0.98,
        f"n = {len(plotted)}; Spearman ρ = {rho:.2f}",
        transform=axis.transAxes,
        ha="left",
        va="top",
        fontsize=8.5,
        color=SANKY_PALETTE["gray"],
    )
    axis.set_xlabel(metric_label(retention_column))
    axis.set_ylabel(metric_label(performance_metric))
    if standalone:
        axis.set_title(title or "Inner retention and held-out performance")
    else:
        add_panel_context(axis, panel_context_label(data, group_column=group_column))
    performance_values = plotted[performance_metric]
    style_axis(
        axis,
        zero_reference=bool(
            performance_values.min() < 0 < performance_values.max()
        ),
    )
    if standalone and group_column and len(categories) > 1:
        axis.legend(title=display_label(group_column), loc="best")
    return figure


def add_common_output_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Input CSV or Parquet table, such as run_summary.csv or detail/testing_predictions.parquet.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Figure directory (default: a figures subdirectory beside --input).",
    )
    parser.add_argument(
        "--output-stem",
        default=None,
        help="Base filename without extension (default: the input CSV filename).",
    )
    parser.add_argument("--dpi", type=int, default=300, help="PNG resolution (default: 300).")
    parser.add_argument(
        "--title",
        default=None,
        help="Optional figure title, shown only with --standalone.",
    )
    parser.add_argument(
        "--standalone",
        action="store_true",
        help=(
            "Include the figure title, legend, and explanatory footnotes. "
            "By default, figures are panel-ready for manuscript assembly."
        ),
    )
    parser.add_argument(
        "--where",
        action="append",
        default=[],
        metavar="COLUMN=VALUE",
        help="Repeatable exact-match filter applied before plotting.",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create consistent manuscript figures from logKd run summaries."
    )
    subparsers = parser.add_subparsers(dest="plot_type", required=True)

    allocation = subparsers.add_parser(
        "allocation",
        help="Paired random-versus-evidence-balanced panels from aggregate allocation results.",
    )
    add_common_output_arguments(allocation)
    allocation.add_argument("--metrics", nargs="+", default=PERFORMANCE_METRICS)
    allocation.add_argument(
        "--include-retention",
        action="store_true",
        help="Append the three compact inner-retention diagnostics to --metrics.",
    )
    allocation.add_argument(
        "--reference-method",
        default=None,
        help="Reference method for every strategy; otherwise infer random_row/random_group.",
    )
    allocation.add_argument(
        "--challenger-method",
        default="evidence_balanced",
        help="Allocation method to compare with the seeded-random reference.",
    )
    allocation.add_argument(
        "--error-bars",
        choices=ERROR_BAR_CHOICES,
        default="ci95",
        help="Aggregate uncertainty shown around each mean (default: ci95).",
    )
    comparison = subparsers.add_parser(
        "comparison",
        help="Single grouped-bar comparison across models, split strategies, or configurations.",
    )
    add_common_output_arguments(comparison)
    comparison.add_argument(
        "--group-column",
        default=None,
        help=(
            "Optional override for a legacy or ambiguous table. New aggregate summaries "
            "use comparison_group automatically."
        ),
    )
    comparison.add_argument("--group-order", nargs="+", default=None)
    comparison.add_argument(
        "--groups",
        nargs="+",
        default=None,
        help=(
            "Optional exact comparison-group values to include. The supplied order is "
            "also used in the legend."
        ),
    )
    comparison.add_argument("--metrics", nargs="+", default=PERFORMANCE_METRICS)
    comparison.add_argument(
        "--include-retention",
        action="store_true",
        help=(
            "Add mean inner reference-retention as a dotted line on a secondary axis. "
            "The chart then uses comparison groups on the x-axis."
        ),
    )
    comparison.add_argument(
        "--retention-column",
        default=DEFAULT_RETENTION_COLUMN,
        help=(
            "Compact retention field for --include-retention "
            f"(default: {DEFAULT_RETENTION_COLUMN})."
        ),
    )
    comparison.add_argument(
        "--error-bars",
        choices=ERROR_BAR_CHOICES,
        default="ci95",
        help="Aggregate uncertainty shown around each mean (default: ci95).",
    )
    comparison.add_argument(
        "--show-runs",
        action="store_true",
        help="Overlay individual outer-repeat values; off by default.",
    )

    retention_profile = subparsers.add_parser(
        "retention-profile",
        help="Three-panel heatmap of database contrast, training contrast, and contrast retention.",
    )
    add_common_output_arguments(retention_profile)
    retention_profile.add_argument(
        "--group-column",
        default=None,
        help=(
            "Optional override for a legacy or ambiguous table. New aggregate summaries "
            "use comparison_group automatically."
        ),
    )
    retention_profile.add_argument("--group-order", nargs="+", default=None)
    retention_profile.add_argument(
        "--groups",
        nargs="+",
        default=None,
        help=(
            "Optional exact comparison-group values to include. The supplied order "
            "is used for the heatmap columns."
        ),
    )
    retention_profile.add_argument(
        "--error-bars",
        choices=ERROR_BAR_CHOICES,
        default="ci95",
        help="Aggregate uncertainty shown around each mean (default: ci95).",
    )

    retention = subparsers.add_parser(
        "retention",
        help="Scatter plot of a retention diagnostic against one held-out metric.",
    )
    add_common_output_arguments(retention)
    retention.add_argument(
        "--retention-column",
        default="inner_mean_reference_retention_score",
    )
    retention.add_argument("--metric", default="rmse")
    retention.add_argument(
        "--group-column",
        default=None,
        help="Optional categorical column used for color and marker shape.",
    )
    retention.add_argument("--group-order", nargs="+", default=None)
    retention.add_argument(
        "--label-column",
        default=None,
        help="Optional small point label column; use only for a few important points.",
    )

    predicted_vs_actual = subparsers.add_parser(
        "predicted-vs-actual",
        help=(
            "One predicted-versus-actual figure per scenario, with all repeated values "
            "overlaid."
        ),
    )
    add_common_output_arguments(predicted_vs_actual)
    predicted_vs_actual.add_argument(
        "--scenario-column",
        "--group-column",
        dest="scenario_column",
        default=None,
        help=(
            "Scenario column defining the panels. Defaults to comparison_group when "
            "available; --group-column is accepted for wrapper compatibility."
        ),
    )
    predicted_vs_actual.add_argument("--scenario-order", nargs="+", default=None)
    predicted_vs_actual.add_argument(
        "--panel-label",
        default=None,
        help=(
            "Optional testing-context label shared by all generated panels, such as "
            "'Testing: Random row'."
        ),
    )
    predicted_vs_actual.add_argument(
        "--data-split",
        default="testing",
        help=(
            "Value from data_split to show (default: testing). Pass 'all' to include "
            "every split."
        ),
    )
    predicted_vs_actual.add_argument(
        "--repeat-column",
        default=None,
        help=(
            "Optional run identifier used to color and report the overlaid repeats; "
            "otherwise a standard outer-repeat identifier is detected automatically."
        ),
    )
    predicted_vs_actual.add_argument(
        "--target-label",
        default="logKd",
        help="Human-readable target label for the axes and default title (default: logKd).",
    )
    predicted_vs_actual.add_argument(
        "--performance-summary",
        type=Path,
        default=None,
        help=(
            "Optional aggregate performance-summary CSV used for the displayed "
            "low/middle/high logKd metrics. By default, the plot looks for "
            "summary/performance_summary.csv beside the prediction detail export."
        ),
    )
    predicted_vs_actual.add_argument(
        "--low-logkd-tail-threshold",
        type=float,
        default=DEFAULT_LOW_LOGKD_TAIL_THRESHOLD,
        help=(
            "Vertical dashed-line position and low-range upper boundary "
            f"(default: {DEFAULT_LOW_LOGKD_TAIL_THRESHOLD:g})."
        ),
    )
    predicted_vs_actual.add_argument(
        "--high-logkd-tail-threshold",
        type=float,
        default=DEFAULT_HIGH_LOGKD_TAIL_THRESHOLD,
        help=(
            "Vertical dashed-line position and high-range lower boundary "
            f"(default: {DEFAULT_HIGH_LOGKD_TAIL_THRESHOLD:g})."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if (
        args.plot_type == "predicted-vs-actual"
        and args.low_logkd_tail_threshold >= args.high_logkd_tail_threshold
    ):
        raise SystemExit(
            "Error: --low-logkd-tail-threshold must be below "
            "--high-logkd-tail-threshold."
        )
    configure_style()
    input_path = args.input.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve() if args.output_dir else None
    data = filter_summary(read_summary(input_path), args.where)

    if args.plot_type == "allocation":
        metrics = list(args.metrics)
        if args.include_retention:
            metrics.extend(metric for metric in RETENTION_METRICS if metric not in metrics)
        figure = plot_allocation_comparison(
            data,
            metrics=metrics,
            reference_method=args.reference_method,
            challenger_method=args.challenger_method,
            error_bars=args.error_bars,
            title=args.title,
            standalone=args.standalone,
        )
        default_stem = "paired_inner_allocation"
    elif args.plot_type == "comparison":
        group_column = resolve_comparison_group_column(
            data,
            args.group_column,
            input_name=input_path.name,
        )
        comparison_data = select_comparison_groups(
            data,
            group_column=group_column,
            groups=args.groups,
        )
        group_order = args.groups or args.group_order
        if args.include_retention:
            figure = plot_group_comparison_with_retention(
                comparison_data,
                group_column=group_column,
                metrics=args.metrics,
                retention_column=args.retention_column,
                group_order=group_order,
                error_bars=args.error_bars,
                show_runs=args.show_runs,
                title=args.title,
                standalone=args.standalone,
            )
        else:
            figure = plot_group_comparison(
                comparison_data,
                group_column=group_column,
                metrics=args.metrics,
                group_order=group_order,
                error_bars=args.error_bars,
                show_runs=args.show_runs,
                title=args.title,
                standalone=args.standalone,
            )
        default_stem = (
            "performance_and_retention_comparison"
            if args.include_retention and group_column == COMPARISON_GROUP_COLUMN
            else "performance_comparison"
            if group_column == COMPARISON_GROUP_COLUMN
            else (
                f"performance_and_retention_by_{group_column}"
                if args.include_retention
                else f"performance_by_{group_column}"
            )
        )
    elif args.plot_type == "retention-profile":
        group_column = resolve_comparison_group_column(
            data,
            args.group_column,
            input_name=input_path.name,
        )
        retention_data = select_comparison_groups(
            data,
            group_column=group_column,
            groups=args.groups,
        )
        figure = plot_reference_retention_profile(
            retention_data,
            group_column=group_column,
            group_order=args.groups or args.group_order,
            error_bars=args.error_bars,
            title=args.title,
            standalone=args.standalone,
        )
        default_stem = (
            "reference_retention_profile"
            if group_column == COMPARISON_GROUP_COLUMN
            else f"reference_retention_profile_by_{group_column}"
        )
    elif args.plot_type == "retention":
        figure = plot_retention_relationship(
            data,
            retention_column=args.retention_column,
            performance_metric=args.metric,
            group_column=args.group_column,
            group_order=args.group_order,
            label_column=args.label_column,
            title=args.title,
            standalone=args.standalone,
        )
        default_stem = f"{args.retention_column}_vs_{args.metric}"
    else:
        scenario_column = resolve_prediction_scenario_column(data, args.scenario_column)
        repeat_column = resolve_prediction_repeat_column(data, args.repeat_column)
        prediction_data = prediction_plot_data(
            data,
            scenario_column=scenario_column,
            data_split=args.data_split,
        )
        performance_summary_path = resolve_prediction_performance_summary_path(
            input_path,
            args.performance_summary,
        )
        performance_summary_data = (
            filter_prediction_performance_summary(
                read_summary(performance_summary_path),
                args.where,
            )
            if performance_summary_path is not None
            else None
        )
        scenarios = ordered_categories(prediction_data[scenario_column], args.scenario_order)
        if not scenarios:
            raise ValueError("The predicted-versus-actual plot found no scenarios to display.")
        default_stem = (
            "predicted_vs_actual"
            if scenario_column == COMPARISON_GROUP_COLUMN
            else f"predicted_vs_actual_by_{scenario_column}"
        )

    if args.plot_type == "predicted-vs-actual":
        output_stem = args.output_stem or default_stem
        data_split_label = (
            "All data-split"
            if not args.data_split or args.data_split.casefold() == "all"
            else display_label(args.data_split)
        )
        saved = []
        for scenario in scenarios:
            figure = plot_predicted_vs_actual(
                prediction_data,
                scenario_column=scenario_column,
                scenario=scenario,
                repeat_column=repeat_column,
                performance_summary_row=prediction_range_metric_summary_row(
                    performance_summary_data,
                    scenario_column=scenario_column,
                    scenario=scenario,
                ),
                low_logkd_tail_threshold=args.low_logkd_tail_threshold,
                high_logkd_tail_threshold=args.high_logkd_tail_threshold,
                title=args.title,
                data_split_label=data_split_label,
                target_label=args.target_label,
                panel_label=args.panel_label,
                standalone=args.standalone,
            )
            saved.extend(save_figure(
                figure,
                input_path=input_path,
                output_dir=output_dir,
                output_stem=f"{output_stem}_{output_slug(scenario)}",
                dpi=args.dpi,
            ))
    else:
        saved = save_figure(
            figure,
            input_path=input_path,
            output_dir=output_dir,
            output_stem=args.output_stem or default_stem,
            dpi=args.dpi,
        )
    input_kind = "completed runs" if "status" in data.columns else "summary rows"
    safe_print(f"Read {len(data):,} {input_kind} from: {input_path}")
    for path in saved:
        safe_print(f"Saved: {path}")


if __name__ == "__main__":
    try:
        main()
    except (FileNotFoundError, ValueError) as error:
        raise SystemExit(f"Error: {error}") from None
