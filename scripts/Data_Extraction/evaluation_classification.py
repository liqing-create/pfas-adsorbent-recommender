"""Evaluate configured classification outputs against human annotations."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

try:
    from .workflow_config import active_studies_by_group, add_project_import_paths
except ImportError:
    from workflow_config import active_studies_by_group, add_project_import_paths

try:
    from .classification_chunk_matching import match_chunks
except ImportError:
    from classification_chunk_matching import match_chunks

add_project_import_paths()

from chains.llm_usage import add_total_usage_row, summarize_classification_usage


LABELS = ("performance", "adsorbent", "experiment")
EVALUATION_LABELS = (*LABELS, "relevant")
METRIC_COLUMNS = ("precision", "recall", "f1_score", "accuracy")


def normalize_labels(value: Any) -> set[str]:
    values = value if isinstance(value, list) else str(value or "").split(",")
    return {str(item).strip() for item in values if str(item).strip()}


def is_positive(label: str, labels: set[str]) -> bool:
    return bool(labels & set(LABELS)) if label == "relevant" else label in labels


def metrics(tp: int, fp: int, tn: int, fn: int) -> dict[str, float | int]:
    support = tp + fn
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / support if support else np.nan
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "TP": tp,
        "FP": fp,
        "TN": tn,
        "FN": fn,
        "support_gt": support,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "accuracy": (tp + tn) / (tp + fp + tn + fn) if tp + fp + tn + fn else np.nan,
    }


def _usage_summary(
    study_id: str,
    predictions: list[dict],
    ground_truth: dict[int, set[str]] | None,
) -> dict:
    has_gt = ground_truth is not None
    predicted_relevant = predicted_irrelevant = 0
    ground_truth_relevant = ground_truth_irrelevant = evaluated = missing = 0
    for source_index, chunk in enumerate(predictions):
        predicted = normalize_labels(chunk.get("predicted_label"))
        predicted_relevant += int(is_positive("relevant", predicted))
        predicted_irrelevant += int(not is_positive("relevant", predicted))
        if not has_gt:
            continue
        if source_index not in ground_truth:
            missing += 1
            continue
        evaluated += 1
        actual = ground_truth[source_index]
        ground_truth_relevant += int(is_positive("relevant", actual))
        ground_truth_irrelevant += int(not is_positive("relevant", actual))
    if not has_gt:
        missing = len(predictions)
    return summarize_classification_usage(
        study_id,
        predictions,
        {
            "has_ground_truth": has_gt,
            "evaluated_chunks": evaluated,
            "chunks_without_ground_truth": missing,
            "predicted_relevant_chunks": predicted_relevant,
            "predicted_irrelevant_chunks": predicted_irrelevant,
            "ground_truth_relevant_chunks": ground_truth_relevant,
            "ground_truth_irrelevant_chunks": ground_truth_irrelevant,
        },
    )


def evaluate_group(study_paths: list) -> None:
    """Write one report set to the prediction directory shared by this group."""
    results: list[dict] = []
    usage_rows: list[dict] = []
    mismatches: list[dict] = []
    output_dir: Path = study_paths[0].classification_dir

    for paths in study_paths:
        if not paths.classified_file.exists():
            print(f"[{paths.study_id}] skipping: prediction missing: {paths.classified_file}")
            continue
        with paths.classified_file.open(encoding="utf-8") as handle:
            predictions = json.load(handle)
        gt_by_prediction: dict[int, set[str]] | None = None
        match_by_prediction = {}
        if paths.classification_annotation_file.exists():
            with paths.classification_annotation_file.open(encoding="utf-8") as handle:
                annotations = json.load(handle)
            chunk_matches = match_chunks(predictions, annotations)
            match_by_prediction = {
                match.source_index: match for match in chunk_matches
            }
            gt_by_prediction = {
                match.source_index: normalize_labels(
                    annotations[match.target_index].get("annotation")
                )
                for match in chunk_matches
            }
            exact_count = sum(match.method == "exact" for match in chunk_matches)
            fuzzy_count = sum(match.method == "fuzzy" for match in chunk_matches)
            print(
                f"[{paths.study_id}] GT content matches: "
                f"{exact_count} exact, {fuzzy_count} fuzzy, "
                f"{len(predictions) - len(chunk_matches)} unmatched predictions."
            )
        else:
            print(f"[{paths.study_id}] no human annotation; saving usage only.")
        usage_rows.append(
            _usage_summary(paths.study_id, predictions, gt_by_prediction)
        )
        if gt_by_prediction is None:
            continue

        reported_missing: set[int] = set()
        for label in EVALUATION_LABELS:
            counts = defaultdict(int)
            for source_index, chunk in enumerate(predictions):
                predicted = normalize_labels(chunk.get("predicted_label"))
                predicted_positive = is_positive(label, predicted)
                actual = gt_by_prediction.get(source_index)
                if actual is None:
                    if predicted_positive:
                        counts["FP"] += 1
                    if source_index not in reported_missing:
                        mismatches.append({
                            "study_id": paths.study_id, "chunk_id": chunk.get("chunk_id"),
                            "file_type": chunk.get("file_type"), "label": "missing_ground_truth",
                            "error_type": "FP", "missing_ground_truth": True,
                            "match_method": "unmatched", "match_similarity": 0.0,
                            "text": chunk.get("enriched_text", ""),
                            "llm_prediction": sorted(predicted), "ground_truth": [],
                        })
                        reported_missing.add(source_index)
                    continue
                chunk_match = match_by_prediction[source_index]
                actual_positive = is_positive(label, actual)
                if predicted_positive and actual_positive:
                    counts["TP"] += 1
                elif predicted_positive:
                    counts["FP"] += 1
                elif actual_positive:
                    counts["FN"] += 1
                else:
                    counts["TN"] += 1
                if label != "relevant" and predicted_positive != actual_positive:
                    mismatches.append({
                        "study_id": paths.study_id, "chunk_id": chunk.get("chunk_id"),
                        "file_type": chunk.get("file_type"), "label": label,
                        "error_type": "FP" if predicted_positive else "FN",
                        "missing_ground_truth": False, "text": chunk.get("enriched_text", ""),
                        "match_method": chunk_match.method,
                        "match_similarity": round(chunk_match.similarity, 4),
                        "llm_prediction": sorted(predicted), "ground_truth": sorted(actual),
                    })
            results.append({
                "study_id": paths.study_id,
                "summary_type": "per_study",
                "label": label,
                **metrics(counts["TP"], counts["FP"], counts["TN"], counts["FN"]),
            })

    output_dir.mkdir(parents=True, exist_ok=True)
    detail = pd.DataFrame(results)
    if not detail.empty:
        macro = detail.groupby("label", as_index=False)[list(METRIC_COLUMNS)].mean()
        macro.insert(0, "study_id", "macro_average")
        macro.insert(1, "summary_type", "macro_average")
        totals = detail.groupby("label", as_index=False)[["TP", "FP", "TN", "FN"]].sum()
        micro = pd.DataFrame([
            {"study_id": "micro_average", "summary_type": "micro_average", "label": row.label,
             **metrics(int(row.TP), int(row.FP), int(row.TN), int(row.FN))}
            for row in totals.itertuples()
        ])
        detail = pd.concat([detail, macro, micro], ignore_index=True, sort=False)
    detail.to_csv(output_dir / "model_performance_evaluation.csv", index=False)
    pd.DataFrame(add_total_usage_row(usage_rows, "classification_time_s") if usage_rows else []).to_csv(
        output_dir / "classification_usage_summary.csv", index=False
    )
    pd.DataFrame(mismatches).to_csv(output_dir / "mislabeled_chunks.csv", index=False)
    print(f"[{study_paths[0].group}] classification evaluation written to {output_dir}")


def main() -> None:
    for _group, paths in active_studies_by_group().items():
        evaluate_group(paths)


if __name__ == "__main__":
    main()
