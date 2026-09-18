"""Compact, run-level diagnostics for inner-fold evidence retention."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


INNER_REFERENCE_RETENTION_SCORE_COLUMNS = (
    "inner_min_reference_retention_score",
    "inner_mean_reference_retention_score",
    "inner_reference_retention_score_sd",
)


def summarize_inner_reference_retention(run_dir: Path) -> dict[str, float]:
    """Summarize the composite retention score across realized inner folds.

    The detailed block/component measurements remain in each run's
    ``validation_diagnostics.csv``.  This helper deliberately exposes only the
    minimum, mean, and population standard deviation needed in a run summary.
    Missing or inapplicable diagnostics are represented as ``NaN`` so every
    consolidated wrapper keeps the same three columns without inventing a score.
    """

    summary = {
        column: float("nan")
        for column in INNER_REFERENCE_RETENTION_SCORE_COLUMNS
    }
    path = run_dir / "validation_diagnostics.csv"
    if not path.exists() or path.stat().st_size == 0:
        return summary
    try:
        diagnostics = pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return summary
    score_column = "validation_reference_retention_score"
    if score_column not in diagnostics.columns:
        return summary
    scores = pd.to_numeric(diagnostics[score_column], errors="coerce").dropna()
    if scores.empty:
        return summary
    return {
        "inner_min_reference_retention_score": float(scores.min()),
        "inner_mean_reference_retention_score": float(scores.mean()),
        "inner_reference_retention_score_sd": float(scores.std(ddof=0)),
    }
