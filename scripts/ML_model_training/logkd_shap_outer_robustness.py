"""Summarize held-out SHAP importance across repeated outer evaluations.

This script answers one question only: which original input features remain
important when the outer training/test assignment changes?  It deliberately
does not create local explanations or deployment-model outputs.

Within each held-out repeat, every source_row_index receives equal total
weight when local SHAP values are aggregated into global feature importance.
The beeswarm applies that same weight to the area of each plotted point, so an
isotherm expanded into many model rows cannot dominate the swarm by sheer point
count while counting once in the importance bars beside it.

The aggregated bar figure necessarily discards the sign and the spread of the
individual held-out SHAP values.  A signed beeswarm needs one single model, so
this workflow additionally explains the one outer repeat whose held-out
performance sits closest to the mean performance of the repeats, and plots that
repeat's per-row SHAP values.  The beeswarm is therefore a representative
illustration of a typical repeat, never a summary across repeats.

By default the beeswarm gives each one-hot category its own row, coloured by
that category's 0/1 indicator.  Its rows are therefore ranked independently of
the aggregated importance bars, where a categorical input keeps a single row
holding the summed contribution of all its levels.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from backend.logkd_data import equal_source_row_weights
from backend.logkd_shap_common import (
    BeeswarmDisplay,
    build_beeswarm_display,
    explain_pipeline,
    load_evaluation_run,
    original_importance_summary,
    save_beeswarm,
    save_importance_bar,
    testing_positions,
)


# Held-out metrics that define a "typical" repeat.  Each is standardized across
# the repeats before they are combined, so no metric's units dominate.
DEFAULT_REPRESENTATIVE_METRICS = ("rmse", "mae", "r2", "spearman")
# A repeated metric that is constant still has a floating-point-noise standard
# deviation.  Standardizing by that noise would amplify meaningless differences,
# so a metric counts as varying only above this relative tolerance.
METRIC_VARIATION_RELATIVE_TOLERANCE = 1e-12
# "levels" plots one row per one-hot category, as the adsorption-ML literature
# does, so every plotted point carries a colour.  "aggregate" keeps one row per
# input, matching the importance bars but leaving categorical rows uncoloured.
CATEGORICAL_BEESWARM_DISPLAYS = ("levels", "aggregate")
DEFAULT_CATEGORICAL_BEESWARM_DISPLAY = "levels"
# The representative beeswarm is inserted as one panel in Fig. 4, so it uses
# fewer ranked rows than the standalone robustness bar chart.
DEFAULT_BEESWARM_MAX_DISPLAY = 12
# Every artifact this workflow writes starts with this prefix.  Overwriting an
# existing scenario folder replaces exactly these files and nothing else.
SHAP_ARTIFACT_PREFIX = "shap_"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate held-out SHAP importance across repeated outer model runs."
    )
    parser.add_argument(
        "--run-dirs",
        nargs="+",
        type=Path,
        required=True,
        help="Completed outer-run directories for one fixed modelling configuration.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--top-k", type=int, default=5, help="Rank threshold used for stability frequency.")
    parser.add_argument("--max-display", type=int, default=15, help="Features shown in the one summary plot.")
    parser.add_argument(
        "--beeswarm-max-display",
        type=int,
        default=DEFAULT_BEESWARM_MAX_DISPLAY,
        help="Features shown in the compact representative beeswarm (default: 12).",
    )
    parser.add_argument(
        "--no-beeswarm",
        dest="beeswarm",
        action="store_false",
        default=True,
        help=(
            "Disable the signed per-row beeswarm of the repeat whose held-out "
            "performance is closest to the mean across repeats."
        ),
    )
    parser.add_argument(
        "--representative-metrics",
        nargs="+",
        default=list(DEFAULT_REPRESENTATIVE_METRICS),
        help=(
            "Held-out metrics used to choose the representative repeat "
            "(default: rmse mae r2 spearman)."
        ),
    )
    parser.add_argument(
        "--no-overwrite",
        dest="overwrite",
        action="store_false",
        default=True,
        help=(
            "Refuse to write into an existing output directory. By default a "
            "rerun replaces that directory's shap_* artifacts, because rerunning "
            "an analysis means correcting it; other files there are kept."
        ),
    )
    parser.add_argument(
        "--categorical-beeswarm-display",
        choices=CATEGORICAL_BEESWARM_DISPLAYS,
        default=DEFAULT_CATEGORICAL_BEESWARM_DISPLAY,
        help=(
            "levels gives each one-hot category its own coloured beeswarm row; "
            "aggregate keeps one uncoloured row per categorical input."
        ),
    )
    return parser.parse_args()


def load_testing_metrics(run_dir: Path, metrics: Sequence[str]) -> dict[str, float]:
    """Read one outer run's held-out metrics as written by the trainer."""

    path = run_dir / "metrics_summary.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing held-out metrics: {path}")
    summary = pd.read_csv(path)
    if "split" not in summary.columns:
        raise ValueError(f"{path} lacks a 'split' column.")
    testing = summary.loc[summary["split"].eq("testing")]
    if len(testing) != 1:
        raise ValueError(f"{path} must contain exactly one held-out testing row.")
    row = testing.iloc[0]
    return {
        metric: float(pd.to_numeric(row[metric], errors="coerce"))
        if metric in testing.columns
        else float("nan")
        for metric in metrics
    }


def representative_run_table(
    run_dirs: Sequence[Path],
    metrics: Sequence[str] = DEFAULT_REPRESENTATIVE_METRICS,
) -> pd.DataFrame:
    """Rank outer repeats by how typical their held-out performance is.

    Every usable metric is standardized across the repeats, and a repeat's
    typicality distance is the mean absolute standard score over those metrics.
    A metric is usable only when it is present and finite for every repeat and
    varies between them.  Selection uses held-out performance alone; it never
    inspects SHAP values, so it cannot bias the explanation it illustrates.
    """

    if not len(run_dirs):
        raise ValueError("Representative-run selection requires at least one outer run.")
    if not len(metrics):
        raise ValueError("Representative-run selection requires at least one metric.")
    table = pd.DataFrame(
        [
            {"run_dir": str(run_dir), **load_testing_metrics(run_dir, metrics)}
            for run_dir in run_dirs
        ]
    )
    usable: list[str] = []
    for metric in metrics:
        column = pd.to_numeric(table[metric], errors="coerce")
        mean = float(column.mean())
        deviation = float(column.std(ddof=1)) if len(column) > 1 else 0.0
        table[f"{metric}_repeat_mean"] = mean
        table[f"{metric}_repeat_sd"] = deviation
        varies = np.isfinite(deviation) and deviation > METRIC_VARIATION_RELATIVE_TOLERANCE * max(
            1.0, abs(mean)
        )
        if bool(np.isfinite(column).all()) and varies:
            table[f"{metric}_abs_standard_score"] = (column - mean).abs() / deviation
            usable.append(metric)
        else:
            table[f"{metric}_abs_standard_score"] = np.nan
    table["metrics_used_for_selection"] = ", ".join(usable)
    table["typicality_distance"] = (
        table[[f"{metric}_abs_standard_score" for metric in usable]].mean(axis=1)
        if usable
        else np.nan
    )
    # Without a usable metric every repeat is equally typical, so keep the first
    # run deterministically rather than failing the whole summary.
    position = int(table["typicality_distance"].idxmin()) if usable else 0
    table["selected"] = np.arange(len(table)) == position
    return table


def prepare_output_dir(output_dir: Path, *, overwrite: bool) -> list[str]:
    """Create the scenario folder, optionally replacing a previous analysis.

    Overwriting removes only this workflow's own ``shap_*`` artifacts, so a
    stale file from an earlier output schema cannot survive alongside the new
    results while anything else in the folder is left untouched.
    """

    if not output_dir.exists():
        output_dir.mkdir(parents=True, exist_ok=False)
        return []
    if not output_dir.is_dir():
        raise NotADirectoryError(f"SHAP output path exists but is not a directory: {output_dir}")
    if not overwrite:
        raise FileExistsError(
            f"SHAP output directory already exists: {output_dir}. "
            "Drop --no-overwrite to replace its shap_* artifacts."
        )
    replaced = [
        path.name
        for path in sorted(output_dir.glob(f"{SHAP_ARTIFACT_PREFIX}*"))
        if path.is_file()
    ]
    for name in replaced:
        (output_dir / name).unlink()
    return replaced


def _selected_run_dir(selection: pd.DataFrame) -> Path:
    return Path(str(selection.loc[selection["selected"], "run_dir"].iloc[0]))


def representative_row_values(
    display: BeeswarmDisplay,
    feature_frame: pd.DataFrame,
    testing_frame: pd.DataFrame,
    weights: pd.Series,
    predictions: np.ndarray,
    base_values: np.ndarray,
) -> pd.DataFrame:
    """Export every held-out SHAP value behind the beeswarm in long form.

    One row is one plotted point.  ``feature_value`` is what colours that point
    - a 0/1 indicator for a one-hot level row - while ``model_input_value``
    keeps the original input the parent feature carried for that held-out row.
    """

    features = [str(feature) for feature in display.values.columns]
    row_count = len(display.values)
    repeats = len(features)
    layout = display.layout.set_index("display_feature").reindex(features)
    parents = layout["parent_feature"].to_numpy(dtype=object)
    exported = pd.DataFrame(
        {
            "testing_row_position": np.repeat(np.arange(row_count), repeats),
            "feature": np.tile(np.asarray(features, dtype=object), row_count),
            "parent_feature": np.tile(parents, row_count),
            "level": np.tile(layout["level"].to_numpy(dtype=object), row_count),
            "row_kind": np.tile(layout["row_kind"].to_numpy(dtype=object), row_count),
            "shap_value": display.values.to_numpy(dtype=float).reshape(-1),
            "feature_value": display.color_positions.to_numpy(dtype=float).reshape(-1),
            "model_input_value": feature_frame.reindex(columns=list(parents))
            .to_numpy(dtype=object)
            .reshape(-1),
            "source_row_weight": np.repeat(np.asarray(weights, dtype=float), repeats),
            "model_prediction": np.repeat(np.asarray(predictions, dtype=float), repeats),
            "model_base_value": np.repeat(np.asarray(base_values, dtype=float), repeats),
        }
    )
    for identifier in ("source_row_index", "PFAS_name", "adsorbent_id", "adsorbent_name"):
        if identifier in testing_frame.columns:
            exported[identifier] = np.repeat(testing_frame[identifier].to_numpy(), repeats)
    return exported


def complete_repeat_table(per_run: pd.DataFrame) -> pd.DataFrame:
    """Represent a feature omitted by an outer model as zero contribution."""

    run_dirs = per_run["run_dir"].drop_duplicates().tolist()
    features = sorted(per_run["feature"].drop_duplicates().tolist())
    full_index = pd.MultiIndex.from_product([run_dirs, features], names=["run_dir", "feature"])
    completed = per_run.set_index(["run_dir", "feature"]).reindex(full_index).reset_index()
    completed["selected_in_run"] = completed["mean_abs_shap"].notna()
    for column in ("mean_abs_shap", "mean_shap"):
        completed[column] = completed[column].fillna(0.0)
    return completed


def robustness_summary(repeat_table: pd.DataFrame, top_k: int) -> pd.DataFrame:
    if top_k < 1:
        raise ValueError("--top-k must be at least one.")
    records: list[dict[str, object]] = []
    n_runs = repeat_table["run_dir"].nunique()
    for feature, group in repeat_table.groupby("feature", sort=False):
        selected = group["selected_in_run"].to_numpy(dtype=bool)
        selected_rows = group.loc[selected]
        ranks = pd.to_numeric(selected_rows.get("rank"), errors="coerce").dropna()
        records.append(
            {
                "feature": feature,
                "mean_abs_shap": float(group["mean_abs_shap"].mean()),
                "sd_abs_shap": float(group["mean_abs_shap"].std(ddof=1)) if n_runs > 1 else 0.0,
                "mean_shap": float(group["mean_shap"].mean()),
                "selected_runs": int(selected.sum()),
                "selected_fraction": float(selected.mean()),
                "top_k_runs": int((ranks <= top_k).sum()),
                "top_k_fraction": float((ranks <= top_k).sum() / n_runs),
                "mean_rank_when_selected": float(ranks.mean()) if not ranks.empty else np.nan,
            }
        )
    summary = pd.DataFrame(records)
    summary["rank"] = summary["mean_abs_shap"].rank(ascending=False, method="min").astype(int)
    return summary.sort_values(["mean_abs_shap", "feature"], ascending=[False, True], kind="stable").reset_index(drop=True)


def _beeswarm_subtitle(selection: pd.DataFrame, testing_rows: int) -> str:
    """Describe the selected repeat with the metrics that made it representative."""

    selected = selection.loc[selection["selected"]].iloc[0]
    used = [
        metric.strip()
        for metric in str(selected["metrics_used_for_selection"]).split(",")
        if metric.strip()
    ]
    labels = {"rmse": "RMSE", "mae": "MAE", "r2": "R2", "spearman": "Spearman"}
    reported = ", ".join(
        f"{labels.get(metric, metric)} {float(selected[metric]):.3g}" for metric in used
    )
    if not reported:
        return f"Representative outer repeat; {testing_rows:,} held-out rows"
    return (
        f"Outer repeat closest to the mean held-out performance "
        f"({reported}); {testing_rows:,} held-out rows"
    )


def generate_robustness_summary(
    run_dirs: list[Path],
    output_dir: Path,
    *,
    top_k: int = 5,
    max_display: int = 15,
    beeswarm_max_display: int = DEFAULT_BEESWARM_MAX_DISPLAY,
    beeswarm: bool = True,
    representative_metrics: Sequence[str] = DEFAULT_REPRESENTATIVE_METRICS,
    categorical_display: str = DEFAULT_CATEGORICAL_BEESWARM_DISPLAY,
    overwrite: bool = True,
) -> dict[str, object]:
    """Generate held-out SHAP robustness artifacts for one fixed scenario."""

    run_dirs = [path.resolve() for path in run_dirs]
    if len(run_dirs) < 2:
        raise ValueError("Outer-robustness SHAP requires at least two outer runs.")
    if len(set(run_dirs)) != len(run_dirs):
        raise ValueError("--run-dirs contains duplicate directories.")
    output_dir = output_dir.resolve()
    replaced_files = prepare_output_dir(output_dir, overwrite=overwrite)

    files = {
        "repeat_importance": "shap_repeat_importance.csv",
        "robustness_summary": "shap_robustness_summary.csv",
        "importance_plot": "shap_robustness_importance.png",
    }
    representative: dict[str, object] = {
        "enabled": bool(beeswarm),
        "status": "pending" if beeswarm else "disabled",
    }
    selection: pd.DataFrame | None = None
    representative_dir: Path | None = None
    if beeswarm:
        try:
            selection = representative_run_table(run_dirs, tuple(representative_metrics))
            representative_dir = _selected_run_dir(selection)
        except (OSError, ValueError, KeyError, pd.errors.ParserError) as error:
            representative = {
                "enabled": True,
                "status": "skipped",
                "reason": f"{type(error).__name__}: {error}",
            }

    rows: list[pd.DataFrame] = []
    representative_explanation: tuple[Any, pd.DataFrame, pd.Series, list[str]] | None = None
    for run_dir in run_dirs:
        run = load_evaluation_run(run_dir)
        positions = testing_positions(run.split)
        testing_frame = run.frame.iloc[positions].reset_index(drop=True)
        testing_weights = equal_source_row_weights(testing_frame)
        explanation = explain_pipeline(
            run.pipeline,
            testing_frame,
            run.selected,
            run.numeric,
            run.categorical,
        )
        summary = original_importance_summary(
            explanation.original_values,
            sample_weight=testing_weights,
        )
        summary.insert(0, "run_dir", str(run_dir))
        summary.insert(1, "testing_rows", len(positions))
        rows.append(summary)
        if representative_dir is not None and run_dir == representative_dir:
            representative_explanation = (
                explanation,
                testing_frame,
                testing_weights,
                list(run.categorical),
            )

    repeat_table = complete_repeat_table(pd.concat(rows, ignore_index=True))
    summary = robustness_summary(repeat_table, top_k)
    repeat_table.to_csv(output_dir / "shap_repeat_importance.csv", index=False)
    summary.to_csv(output_dir / "shap_robustness_summary.csv", index=False)
    save_importance_bar(
        summary,
        output_dir / "shap_robustness_importance.png",
        title="Held-out SHAP importance across outer repeats",
        max_display=max_display,
        error_column="sd_abs_shap",
        repeat_table=repeat_table,
    )

    if selection is not None and representative_explanation is not None:
        # A failed beeswarm must not discard the completed robustness summary.
        explanation, testing_frame, testing_weights, categorical = representative_explanation
        try:
            representative = _write_representative_beeswarm(
                output_dir,
                selection,
                explanation=explanation,
                testing_frame=testing_frame,
                weights=testing_weights,
                categorical=categorical,
                max_display=beeswarm_max_display,
                categorical_display=categorical_display,
            )
            files.update(
                {
                    "representative_selection": "shap_representative_run_selection.csv",
                    "representative_row_values": "shap_representative_row_values.csv",
                    "beeswarm_plot": "shap_representative_beeswarm.png",
                }
            )
        except Exception as error:  # Reported below, never silently dropped.
            representative = {
                "enabled": True,
                "status": "failed",
                "reason": f"{type(error).__name__}: {error}",
            }
    if representative.get("status") in {"skipped", "failed"}:
        print(
            f"WARNING: the representative-repeat beeswarm was not written for "
            f"{output_dir}: {representative.get('reason')}",
            file=sys.stderr,
            flush=True,
        )

    return {
        "output_dir": str(output_dir),
        "outer_runs": len(run_dirs),
        "features": len(summary),
        "files": files,
        "replaced_files": replaced_files,
        "representative_run": representative,
    }


def _write_representative_beeswarm(
    output_dir: Path,
    selection: pd.DataFrame,
    *,
    explanation: Any,
    testing_frame: pd.DataFrame,
    weights: pd.Series,
    categorical: list[str],
    max_display: int,
    categorical_display: str,
) -> dict[str, object]:
    """Write the selection audit, per-row SHAP values, and the beeswarm figure."""

    if categorical_display not in CATEGORICAL_BEESWARM_DISPLAYS:
        raise ValueError(
            f"categorical_display must be one of {sorted(CATEGORICAL_BEESWARM_DISPLAYS)}."
        )
    display = build_beeswarm_display(
        explanation,
        categorical,
        split_categorical_levels=categorical_display == "levels",
        # The same weights the importance bars aggregate with, so one source
        # record carries equal standing in both figures.
        sample_weight=weights,
    )
    # Plotted rows are ranked on their own contributions, so a split categorical
    # level competes as its own row rather than inheriting its parent's rank.
    display_ranking = original_importance_summary(display.values, sample_weight=weights)
    selection.to_csv(output_dir / "shap_representative_run_selection.csv", index=False)
    representative_row_values(
        display,
        explanation.X,
        testing_frame,
        weights,
        explanation.predictions,
        explanation.base_values,
    ).to_csv(output_dir / "shap_representative_row_values.csv", index=False)
    displayed = save_beeswarm(
        display,
        output_dir / "shap_representative_beeswarm.png",
        title=None,
        max_display=max_display,
        feature_order=[str(feature) for feature in display_ranking["feature"]],
        panel_ready=True,
    )
    selected = selection.loc[selection["selected"]].iloc[0]
    distance = float(selected["typicality_distance"])
    level_rows = int(display.layout["row_kind"].eq("categorical_level").sum())
    return {
        "enabled": True,
        "status": "completed",
        "run_dir": str(selected["run_dir"]),
        "testing_rows": int(len(testing_frame)),
        "metrics_used_for_selection": str(selected["metrics_used_for_selection"]),
        "typicality_distance": distance if np.isfinite(distance) else None,
        "categorical_display": categorical_display,
        "point_area_weighting": "equal_source_row_weight",
        "plotted_rows_available": int(len(display.values.columns)),
        "categorical_level_rows": level_rows,
        "features_displayed": displayed,
        "selection_rule": (
            "the outer repeat with the smallest mean absolute standard score of its "
            "held-out metrics relative to the repeat means; SHAP values are never "
            "used to select the repeat"
        ),
        "row_interpretation": (
            "categorical inputs are split into one-hot level rows coloured by their "
            "0/1 indicator, so beeswarm rows are ranked independently of the "
            "aggregated cross-repeat importance bars"
            if categorical_display == "levels"
            else "every input keeps one aggregated row, as in the importance bars"
        ),
    }


def main() -> None:
    args = parse_args()
    result = generate_robustness_summary(
        args.run_dirs,
        args.output_dir,
        top_k=args.top_k,
        max_display=args.max_display,
        beeswarm_max_display=args.beeswarm_max_display,
        beeswarm=args.beeswarm,
        representative_metrics=tuple(args.representative_metrics),
        categorical_display=args.categorical_beeswarm_display,
        overwrite=args.overwrite,
    )
    print(f"Wrote SHAP robustness summary to: {result['output_dir']}")
    print(f"Outer runs: {result['outer_runs']}; features: {result['features']}")
    replaced = result["replaced_files"]
    if replaced:
        print(f"Replaced {len(replaced)} existing artifact(s): {', '.join(replaced)}")
    representative = result["representative_run"]
    if representative.get("status") == "completed":
        distance = representative["typicality_distance"]
        print(
            f"Representative repeat for the beeswarm: {representative['run_dir']} "
            f"({representative['testing_rows']} held-out rows; typicality distance "
            f"{distance:.3g})"
            if distance is not None
            else f"Representative repeat for the beeswarm: {representative['run_dir']}"
        )


if __name__ == "__main__":
    main()
