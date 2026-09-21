import os
import json
import re
import unicodedata
import pandas as pd
import numpy as np
from typing import Dict, Tuple

try:
    from .workflow_config import active_studies_by_group, add_project_import_paths
except ImportError:
    from workflow_config import active_studies_by_group, add_project_import_paths

add_project_import_paths()

from chains.llm_usage import add_total_usage_row, summarize_extraction_usage
from schemas import EXPERIMENT_FIELDS, PERFORMANCE_FIELDS, ADSORBENT_FIELDS

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION (edit these paths to match your environment)
# ─────────────────────────────────────────────────────────────────────────────
GROUND_TRUTH_DIR = ""
PREDICTIONS_DIR = ""
MODEL_NAME = "GPT-4.1"

SELECTED_STUDY_IDS: list[str] = []

# Unique‐ID field per task
TASKS = {
    "adsorbent":  "adsorbent_id",
    "experiment": "experiment_id",
    "performance": "performance_id"
}
EXTRACTION_CHAIN_NAMES = ("adsorbent_study", "experiment_study", "performance")


# ─────────────────────────────────────────────────────────────────────────────
# NORMALIZATION HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def normalize_id(s: str) -> str:
    if s is None:
        return ""
    s = str(s).strip().lower()
    s = re.sub(r'[\/;,]+', '|', s)
    s = re.sub(r'\s*\|\s*', '|', s)
    return s

def _canonicalize_units(s: str) -> str:
    # Normalize Unicode, then fix common 'micro' variants & mojibake.
    s = unicodedata.normalize("NFKC", s)
    # Map Greek mu, micro sign, and mojibake 'Âµ' to a single 'µ'
    s = re.sub(r"(?:\u00B5|\u03BC|Âµ)", "µ", s)
    # Tidy common unit punctuation
    s = re.sub(r"\s*/\s*", "/", s)
    s = re.sub(r"\s*\^\s*", "^", s)
    # Specific helpful aliases (safe no-ops if absent)
    s = s.replace("μg/g", "µg/g")
    s = s.replace("ug/g", "µg/g")
    return s

def normalize_val(v) -> str:
    if v is None:
        return ""
    s = str(v).strip()
    s = _canonicalize_units(s)
    s = s.lower()
    if s in {"na", "n/a", "none", "null"}:
        return ""
    return s

# ─────────────────────────────────────────────────────────────────────────────
# LOAD GROUND‐TRUTH EXTRACTION
# ─────────────────────────────────────────────────────────────────────────────
def load_ground_truth(study_id):
    path = os.path.join(GROUND_TRUTH_DIR, f"{study_id}_ground_truth.json")
    try:
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError as e:
        raise FileNotFoundError(f"Ground truth file missing for {study_id}: {path}") from e
    except json.JSONDecodeError as e:
        raise ValueError(
            f"Invalid JSON in ground truth for {study_id}: {path} "
            f"(line {e.lineno}, column {e.colno}, char {e.pos})"
        ) from e
    out = {}
    for entry in data:
        ft, cid = entry["file_type"], entry["chunk_id"]
        ex = entry.get("extracted_data", {})
        for task, id_field in TASKS.items():
            # accept list OR {"records": [...]}
            block = ex.get(task, [])
            recs = block.get("records") if isinstance(block, dict) else block
            if not recs:
                continue
            for rec in recs:
                raw_uid = rec.get(id_field)
                if raw_uid is None:
                    continue
                parts = [p.strip() for p in str(raw_uid).split('|')]
                if task == "experiment":
                    uid_part = parts[-1]
                elif task == "performance":
                    uid_part = '|'.join(parts[:-1]) if len(parts) > 1 else parts[0]
                else:
                    uid_part = raw_uid
                uid = normalize_id(uid_part)
                orig_id = normalize_val(rec.get(id_field, ""))
                fields = {k.lower(): normalize_val(v)
                          for k, v in rec.items()
                          if k.lower() not in {id_field, "require_review", "review_reason", "_orig_id"}}
                fields['_orig_id'] = orig_id
                out[(ft, cid, task, uid)] = fields
    return out

# ─────────────────────────────────────────────────────────────────────────────
# LOAD PREDICTION EXTRACTION
# ─────────────────────────────────────────────────────────────────────────────
def load_prediction_entries(study_id: str, pred_dir: str) -> list[dict]:
    """Load all canonical extraction-chain outputs for a study."""
    artifact_dir = os.path.join(pred_dir, "_chain_artifacts", study_id)
    data: list[dict] = []
    found_paths: list[str] = []
    for chain_name in EXTRACTION_CHAIN_NAMES:
        path = os.path.join(artifact_dir, f"{study_id}_{chain_name}.json")
        if not os.path.exists(path):
            continue
        found_paths.append(path)
        try:
            with open(path, "r", encoding="utf-8") as f:
                payload = json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(
                f"Invalid JSON in extraction artifact for {study_id}: {path} "
                f"(line {e.lineno}, column {e.colno}, char {e.pos})"
            ) from e
        outputs = payload.get("outputs") if isinstance(payload, dict) else None
        if not isinstance(outputs, list):
            raise ValueError(f"Extraction artifact outputs must be a list: {path}")
        data.extend(entry for entry in outputs if isinstance(entry, dict))

    if not found_paths:
        raise FileNotFoundError(
            f"Extraction artifacts missing for {study_id}: {artifact_dir}"
        )
    return data


def load_predicted(
    study_id: str,
    pred_dir: str,
    prediction_data: list[dict] | None = None,
) -> Tuple[Dict[Tuple[str, str, str, str], dict], Dict[Tuple[str, str], str]]:
    data = prediction_data if prediction_data is not None else load_prediction_entries(study_id, pred_dir)


    out = {}
    text_map: Dict[Tuple[str, str], str] = {}
    for entry in data:
        ft, cid = entry["file_type"], entry["chunk_id"]
        ex = entry.get("extracted_data", {})
        text_map[(ft, cid)] = entry.get("enriched_text", "")
        for task_key, block in ex.items():
            task = task_key.lower()
            if task not in TASKS:
                continue
            id_field = TASKS[task]

            # accept list OR {"records": [...]}
            recs = block.get("records") if isinstance(block, dict) else block
            if not recs:
                continue
            for rec in recs:
                raw_uid = rec.get(id_field)
                if raw_uid is None:
                    continue
                parts = [p.strip() for p in str(raw_uid).split('|')]
                if task == "experiment":
                    uid_part = parts[-1]
                elif task == "performance":
                    uid_part = '|'.join(parts[:-1]) if len(parts) > 1 else parts[0]
                else:
                    uid_part = raw_uid
                uid = normalize_id(uid_part)
                orig_id = normalize_val(rec.get(id_field, ""))
                fields = {k.lower(): normalize_val(v)
                          for k, v in rec.items()
                          if k.lower() not in {id_field, "require_review", "review_reason", "_orig_id"}}
                fields['_orig_id'] = orig_id
                out[(ft, cid, task, uid)] = fields
    return out, text_map

# ─────────────────────────────────────────────────────────────────────────────
# EVALUATE FIELD‐LEVEL WITH TN, ACCURACY, AND CORRECTED F1
# ─────────────────────────────────────────────────────────────────────────────
def evaluate_fields(gt, pred, study, model, text_map):
    rows = []
    mis_rows = []
    for key in set(gt) | set(pred):
        ft, cid, task, uid = key
        gt_fields = gt.get(key, {})
        pr_fields = pred.get(key, {})
        all_fields = (set(gt_fields) | set(pr_fields)) - {"_orig_id"}
        for field in all_fields:
            gt_val = gt_fields.get(field, "")
            pr_val = pr_fields.get(field, "")
            # compute confusion counts
            tn = 1 if gt_val == "" and pr_val == "" else 0
            tp = 1 if gt_val and pr_val and gt_val == pr_val else 0
            fp = 1 if pr_val and (gt_val == "" or pr_val != gt_val) else 0
            fn = 1 if gt_val and (pr_val == "" or pr_val != gt_val) else 0
            total = tp + tn + fp + fn

            # accuracy
            accuracy = (tp + tn) / total if total > 0 else np.nan
            # precision: defined only if any predicted positives
            precision = tp / (tp + fp) if (tp + fp) > 0 else np.nan
            # recall: defined only if any actual positives
            recall    = tp / (tp + fn) if (tp + fn) > 0 else np.nan
            # F1: set 0 for mismatch (precision+recall=0), nan for TN-only
            if np.isnan(precision) or np.isnan(recall):
                f1 = np.nan
            elif precision == 0 and recall == 0:
                f1 = 0.0
            else:
                f1 = 2 * precision * recall / (precision + recall)

            rows.append({
                "model":        model,
                "study":        study,
                "file_type":    ft,
                "chunk_id":     cid,
                "task":         task,
                "uid_key":      uid,
                "gt_orig_id":   gt_fields.get('_orig_id', ''),
                "pred_orig_id": pr_fields.get('_orig_id', ''),
                "field_name":   field,
                "gt_value":     gt_val,
                "pred_value":   pr_val,
                "tn":           tn,
                "tp":           tp,
                "fp":           fp,
                "fn":           fn,
                "accuracy":     accuracy,
                "precision":    precision,
                "recall":       recall,
                "f1":           f1
            })


            # collect misclassified/mismatch rows for debugging CSV
            error_type = None
            if gt_val == "" and pr_val != "":
                error_type = "FP"
            elif gt_val != "" and pr_val == "":
                error_type = "FN"
            elif gt_val != "" and pr_val != "" and gt_val != pr_val:
                error_type = "MISMATCH"

            if error_type:
                mis_rows.append({
                    "model":        model,
                    "study":        study,
                    "file_type":    ft,
                    "chunk_id":     cid,
                    "task":         task,
                    "uid_key":      uid,
                    "field_name":   field,
                    "error_type":   error_type,
                    "gt_value":     gt_val,
                    "pred_value":   pr_val,
                    "gt_orig_id":   gt_fields.get('_orig_id', ''),
                    "pred_orig_id": pr_fields.get('_orig_id', ''),
                    "text":         text_map.get((ft, cid), "")
                })
    return pd.DataFrame(rows), pd.DataFrame(mis_rows)

# ─────────────────────────────────────────────────────────────────────────────
# SUMMARIZE BY FIELD (INCLUDING ACCURACY)
# ─────────────────────────────────────────────────────────────────────────────
def summarize_by_field(field_df):
    study_summary = (
        field_df
        .groupby(["model", "study", "field_name"])  
        .agg(
            occurrences=("f1", "size"),
            accuracy=("accuracy", "mean"),
            precision=("precision", "mean"),
            recall=("recall", "mean"),
            f1=("f1", "mean")
        )
        .reset_index()
    )
    overall_summary = (
        field_df
        .groupby(["model", "field_name"])  
        .agg(
            occurrences=("f1", "size"),
            accuracy=("accuracy", "mean"),
            precision=("precision", "mean"),
            recall=("recall", "mean"),
            f1=("f1", "mean")
        )
        .reset_index()
    )
    return study_summary, overall_summary

# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
def _main_for_current_group():
    models = [(MODEL_NAME, PREDICTIONS_DIR)]
    all_fields = []
    all_mis = []
    all_usage = []
    for model_name, pred_dir in models:
        for study in SELECTED_STUDY_IDS:
            prediction_data = load_prediction_entries(study, pred_dir)
            gt   = load_ground_truth(study)
            pred, text_map = load_predicted(study, pred_dir, prediction_data)
            df, mis_df     = evaluate_fields(gt, pred, study, model_name, text_map)
            all_fields.append(df)
            all_mis.append(mis_df)
            all_usage.append(summarize_extraction_usage(study, prediction_data, model_name, TASKS))
 

        field_df = pd.concat(all_fields, ignore_index=True)
        usage_df = pd.DataFrame(add_total_usage_row(all_usage, "extraction_time_s"))
        study_df, overall_df = summarize_by_field(field_df)

        # Sort overall fields by domain group, then alphabetically within each group
        group_map = {
            **{k.lower(): "experiment"  for k in EXPERIMENT_FIELDS},
            **{k.lower(): "performance" for k in PERFORMANCE_FIELDS},
            **{k.lower(): "adsorbent"   for k in ADSORBENT_FIELDS},
        }
        grp_order = pd.Categorical(
            overall_df["field_name"].map(group_map).fillna("zz_other"),
            categories=["experiment", "performance", "adsorbent", "zz_other"],
            ordered=True,
        )
        overall_df_sorted = (
            overall_df.assign(_grp=grp_order)
                      .sort_values(by=["_grp", "field_name"])
                      .drop(columns=["_grp"])
        )

        out_fn = os.path.join(pred_dir, f"extraction_evaluation_{model_name}.xlsx")
        with pd.ExcelWriter(out_fn, engine="xlsxwriter") as writer:
            field_df   .to_excel(writer, sheet_name="field_details",           index=False)
            study_df   .to_excel(writer, sheet_name="field_summary_by_study",  index=False)
            overall_df_sorted.to_excel(writer, sheet_name="field_summary_overall", index=False)
            usage_df.to_excel(writer, sheet_name="usage_summary", index=False)
 

        print(f"[{model_name}] → saved {out_fn}")

        usage_out = os.path.join(pred_dir, "extraction_usage_summary.csv")
        usage_df.to_csv(usage_out, index=False)
        print(f"[{model_name}] usage summary → {usage_out}")

        # Write misclassified fields CSV for convenient debugging (similar to other scripts)
        if all_mis:
            miscat_df = pd.concat(all_mis, ignore_index=True)
            mis_out = os.path.join(pred_dir, "misclassified_fields.csv")
            miscat_df.to_csv(mis_out, index=False)
            print(f"[{model_name}] misclassified fields → {mis_out}")

def main():
    """Evaluate each selected study group without crossing prediction roots."""
    global GROUND_TRUTH_DIR, PREDICTIONS_DIR, SELECTED_STUDY_IDS
    for group, study_paths in active_studies_by_group().items():
        PREDICTIONS_DIR = str(study_paths[0].extraction_dir)
        GROUND_TRUTH_DIR = str(study_paths[0].extraction_ground_truth_dir)
        SELECTED_STUDY_IDS = [
            paths.study_id
            for paths in study_paths
            if (
                any(
                    paths.extraction_chain_artifact(chain_name).exists()
                    for chain_name in EXTRACTION_CHAIN_NAMES
                )
                and paths.extraction_ground_truth_file.exists()
            )
        ]
        if not SELECTED_STUDY_IDS:
            print(f"[{group}] no selected studies have both extraction predictions and ground truth; skipping.")
            continue
        print(f"\n=== Evaluating extraction {group}: {', '.join(SELECTED_STUDY_IDS)} ===")
        _main_for_current_group()


if __name__ == "__main__":
    main()
