from __future__ import annotations

import copy
import html
import hashlib
import json
import os
import re
import time
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

try:
    from .workflow_config import ACTIVE_STUDIES, get_study_paths
except ImportError:  # Streamlit executes this file directly.
    from workflow_config import ACTIVE_STUDIES, get_study_paths

try:
    from text_highlighter import text_highlighter
except ImportError:  # pragma: no cover - UI fallback for environments without the component
    text_highlighter = None


st.set_page_config(
    page_title="Enumeration Review",
    layout="wide",
    initial_sidebar_state="expanded",
)

SELECTED_STUDY_IDS: list[str] = list(ACTIVE_STUDIES)

TASKS = ("adsorbent", "water_type", "performance")
PERFORMANCE_TEST_MODES = ("kinetic", "isotherm", "snapshot")
PERFORMANCE_ID_FORMAT = "PFAS|Adsorbent|Test_mode|Differentiating_Condition"

ADSORBENT_COLUMNS = [
    "Adsorbent_id",
    "Name_Abbreviation",
    "Name_Full",
    "Name_Commercial",
    "Adsorbent_category",
    "Adsorbent_subcategory",
    "Data_Source",
]
ADSORBENT_SCOPE_KEYS = ("in_scope", "out_of_scope")
ADSORBENT_SCOPE_LABELS = {
    "in_scope": "In-scope adsorbents",
    "out_of_scope": "Out-of-scope adsorbents",
}

ADSORBENT_CATEGORIES = [
    "Activated carbon",
    "Ion exchange resin",
    "Nonionic resin",
    "Cyclodextrin polymers",
    "Clay/mineral",
]

WATER_TYPE_COLUMNS = ["Full_name", "Abbreviation", "Class", "Data_Source"]
WATER_TYPE_CLASSES = [
    "ultrapure water",
    "synthetic water",
    "groundwater",
    "AFFF solution",
    "surface water",
    "wastewater",
    "landfill leachate",
    "tap water",
    "others",
]

HIGHLIGHT_TAG = "highlight"
HIGHLIGHT_COLOR = "yellow"
REVIEW_PANEL_HEIGHT = 760
PERFORMANCE_COLUMNS = ["Performance_id", "Data_Source"]


def coerce_path(raw: str | Path) -> Path:
    text = str(raw or "").strip().strip('"')
    return Path(os.path.expandvars(text)).expanduser()


def performance_records_to_text(records: list[dict]) -> str:
    lines = []
    for record in records or []:
        if not isinstance(record, dict):
            continue
        value = canonical_performance_id(record.get("Performance_id"))
        if value:
            lines.append(value)
    return "\n".join(lines)


def performance_text_to_records(text: str) -> list[dict]:
    records = []
    for line in str(text or "").splitlines():
        value = canonical_performance_id(line)
        if value:
            records.append({"Performance_id": value})
    return records

def load_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def save_json(data: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2, ensure_ascii=False)


def artifact_study_dir(root: Path, study_id: str) -> Path:
    return root / "_chain_artifacts" / study_id


def task_artifact_path(root: Path, study_id: str, task: str) -> Path:
    return artifact_study_dir(root, study_id) / f"{study_id}_{task}.json"


def manifest_path(root: Path, study_id: str) -> Path:
    return artifact_study_dir(root, study_id) / "manifest.json"


def chunk_key(file_type: Any, chunk_id: Any) -> str:
    file_type_text = str(file_type if file_type is not None else "").strip()
    chunk_id_text = str(chunk_id if chunk_id is not None else "").strip()
    return f"{file_type_text}::{chunk_id_text}"


def chunk_label_from_key(key: str) -> str:
    file_type, _, chunk_id = key.partition("::")
    return f"{file_type} {chunk_id}".strip()


def source_id_for(entry: dict) -> str:
    explicit = str(entry.get("source_id") or "").strip()
    if explicit:
        return explicit
    file_type = str(entry.get("file_type") or "unknown").strip()
    chunk_id = entry.get("chunk_id")
    return f"{file_type}_{chunk_id}" if chunk_id is not None else file_type


def source_id_for_values(file_type: Any, chunk_id: Any) -> str:
    file_type_text = str(file_type or "unknown").strip()
    return f"{file_type_text}_{chunk_id}" if chunk_id is not None else file_type_text


def highlights_path(root: Path, study_id: str) -> Path:
    return artifact_study_dir(root, study_id) / f"{study_id}_highlights.json"


def normalize_source_id(value: Any) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    return re.sub(r"\s+", "_", text)


def normalize_highlight_annotations(raw: object) -> list[dict]:
    if not isinstance(raw, list):
        return []
    out: list[dict] = []
    for annotation in raw:
        if not isinstance(annotation, dict):
            continue
        try:
            start = int(annotation.get("start"))
            end = int(annotation.get("end"))
        except Exception:
            continue
        if start < 0 or end <= start:
            continue
        item = dict(annotation)
        item.pop("color", None)
        item["start"] = start
        item["end"] = end
        item["tag"] = HIGHLIGHT_TAG
        out.append(item)
    return out


def load_highlights(root: Path, study_id: str) -> dict[str, list[dict]]:
    path = highlights_path(root, study_id)
    if not path.exists():
        return {}
    try:
        data = load_json(path)
    except Exception:
        return {}
    if not isinstance(data, dict):
        return {}
    highlights: dict[str, list[dict]] = {}
    for key, value in data.items():
        source_id = normalize_source_id(key)
        if source_id:
            highlights[source_id] = normalize_highlight_annotations(value)
    return highlights


def save_highlights(root: Path, study_id: str, highlights: dict[str, list[dict]]) -> Path:
    path = highlights_path(root, study_id)
    payload: dict[str, list[dict]] = {}
    for key, value in (highlights or {}).items():
        source_id = normalize_source_id(key)
        if source_id:
            payload[source_id] = normalize_highlight_annotations(value)
    save_json(payload, path)
    return path


def sort_chunk_key(item: tuple[str, dict]) -> tuple[str, int, str]:
    chunk = item[1]
    file_type = str(chunk.get("file_type") or "")
    raw_id = chunk.get("chunk_id")
    try:
        numeric_id = int(raw_id)
    except (TypeError, ValueError):
        numeric_id = 10**9
    return file_type, numeric_id, str(raw_id)


def empty_task_payload(study_id: str, task: str) -> dict:
    payload = {
        "study_folder": study_id,
        "chain_name": task,
        "task": task,
        "status": "missing",
        "mode": "",
        "message": "artifact not found",
        "created_at_unix": None,
        "provider": "",
        "model_name": "",
        "timings": {},
        "selected_chunk_metadata": [],
        "record_count": 0,
        "output_count": 0,
    }
    if task == "adsorbent":
        payload["adsorbents"] = empty_adsorbents()
    elif task == "water_type":
        payload["water_types"] = []
    elif task == "performance":
        payload["outputs"] = []
    return payload

def empty_adsorbents() -> dict[str, list[dict]]:
    return {scope: [] for scope in ADSORBENT_SCOPE_KEYS}


def normalize_adsorbents(raw: Any) -> dict[str, list[dict]]:
    if raw is None:
        return empty_adsorbents()
    if isinstance(raw, list):
        return {"in_scope": copy.deepcopy(raw), "out_of_scope": []}
    if isinstance(raw, dict):
        groups = empty_adsorbents()
        for scope in ADSORBENT_SCOPE_KEYS:
            value = raw.get(scope)
            if isinstance(value, list):
                groups[scope] = copy.deepcopy(value)
        return groups
    return empty_adsorbents()


def count_adsorbents(adsorbents: Any) -> int:
    groups = normalize_adsorbents(adsorbents)
    return sum(len(groups[scope]) for scope in ADSORBENT_SCOPE_KEYS)


def adsorbent_ids_for_scope(adsorbents: Any, scope: str) -> set[str]:
    groups = normalize_adsorbents(adsorbents)
    return {
        str(row.get("Adsorbent_id") or "").strip()
        for row in groups.get(scope, [])
        if isinstance(row, dict) and str(row.get("Adsorbent_id") or "").strip()
    }

def load_task_payload(root: Path, study_id: str, task: str) -> tuple[dict, Path | None]:
    path = task_artifact_path(root, study_id, task)
    if path.exists():
        return load_json(path), path

    return empty_task_payload(study_id, task), None


def load_manifest(root: Path, study_id: str) -> tuple[dict, Path | None]:
    path = manifest_path(root, study_id)
    if path.exists():
        return load_json(path), path
    return {}, None

def format_review_time(value: Any) -> str:
    try:
        timestamp = float(value)
    except (TypeError, ValueError):
        return ""
    return time.strftime("%Y-%m-%d %H:%M", time.localtime(timestamp))


def ground_truth_review_summary(root: Path, study_id: str) -> tuple[bool, str]:
    manifest, _ = load_manifest(root, study_id)
    if not manifest:
        return False, "not reviewed"

    status = str(manifest.get("review_status") or "saved").strip()
    reviewed_at = format_review_time(manifest.get("reviewed_at_unix"))
    label = f"{status} {reviewed_at}" if reviewed_at else status
    return status.casefold() == "reviewed", label

def discover_studies(*roots: Path) -> list[str]:
    studies: set[str] = set()
    for root in roots:
        if not root or not root.exists():
            continue
        artifact_root = root / "_chain_artifacts"
        if artifact_root.exists():
            for manifest in artifact_root.glob("*/manifest.json"):
                studies.add(manifest.parent.name)
            for child in artifact_root.iterdir():
                if not child.is_dir():
                    continue
                for task in TASKS:
                    if (child / f"{child.name}_{task}.json").exists():
                        studies.add(child.name)
                        break
    return sorted(studies)


def load_classified_entries(
    classified_root: Path, study_id: str
) -> tuple[dict[str, dict], Path | None]:
    path = classified_root / f"{study_id}_classified.json"
    if not path.exists():
        return {}, None
    data = load_json(path)
    entries: dict[str, dict] = {}
    for item in data if isinstance(data, list) else []:
        if not isinstance(item, dict):
            continue
        entries[chunk_key(item.get("file_type"), item.get("chunk_id"))] = item
    return entries, path


def abstract_chunk_metadata(classified_entries: dict[str, dict]) -> list[dict]:
    rows: list[dict] = []
    seen: set[str] = set()
    for _, entry in sorted(classified_entries.items(), key=sort_chunk_key):
        if not isinstance(entry, dict):
            continue
        if str(entry.get("file_type") or "").strip().lower() != "main_paper":
            continue
        if str(entry.get("section_label") or "").strip().lower() != "abstract":
            continue
        text = str(entry.get("enriched_text") or entry.get("text") or "").strip()
        if not text:
            continue
        key = chunk_key(entry.get("file_type"), entry.get("chunk_id"))
        if key in seen:
            continue
        seen.add(key)
        rows.append(
            {
                "study_folder": entry.get("study_folder"),
                "file_type": entry.get("file_type"),
                "chunk_id": entry.get("chunk_id"),
                "source_id": source_id_for(entry),
                "section_label": entry.get("section_label"),
                "predicted_label": entry.get("predicted_label"),
            }
        )
    return rows


def build_chunk_metadata(outputs: list[dict]) -> list[dict]:
    rows = []
    for entry in outputs or []:
        if not isinstance(entry, dict):
            continue
        rows.append(
            {
                "study_folder": entry.get("study_folder"),
                "file_type": entry.get("file_type"),
                "chunk_id": entry.get("chunk_id"),
                "source_id": source_id_for(entry),
                "predicted_label": entry.get("predicted_label"),
            }
        )
    return rows


def metadata_key(meta: dict) -> str:
    return chunk_key(meta.get("file_type"), meta.get("chunk_id"))


def dedupe_metadata(metadata: list[dict]) -> list[dict]:
    rows: list[dict] = []
    seen: set[str] = set()
    for meta in metadata or []:
        if not isinstance(meta, dict):
            continue
        key = metadata_key(meta)
        if not key.strip(":") or key in seen:
            continue
        seen.add(key)
        rows.append(copy.deepcopy(meta))
    return rows


def output_source_metadata(output: dict) -> list[dict]:
    raw = output.get("enriched_text_metadata")
    if isinstance(raw, list):
        return dedupe_metadata([item for item in raw if isinstance(item, dict)])
    return dedupe_metadata(build_chunk_metadata([output]))


def selected_performance_metadata(*payloads: dict) -> list[dict]:
    rows: list[dict] = []
    for payload in payloads:
        for meta in (payload or {}).get("selected_chunk_metadata") or []:
            if isinstance(meta, dict):
                rows.append(meta)
    return dedupe_metadata(rows)


def source_text_from_metadata(
    metadata: list[dict], classified_entries: dict[str, dict]
) -> str:
    lines: list[str] = []
    for idx, meta in enumerate(dedupe_metadata(metadata), start=1):
        key = metadata_key(meta)
        classified = classified_entries.get(key, {})
        text = str(classified.get("enriched_text") or classified.get("text") or "").strip()
        if not text:
            continue
        file_type = str(meta.get("file_type") or classified.get("file_type") or "unknown").strip()
        chunk_id = meta.get("chunk_id")
        source_id = meta.get("source_id") or source_id_for_values(file_type, chunk_id)
        labels = meta.get("predicted_label") or classified.get("predicted_label") or ""
        prompt_chunk_number = str(meta.get("prompt_chunk_number") or idx).strip()
        lines.append(
            f"Chunk {prompt_chunk_number} (Data_Source={source_id}; labels={labels}):\n{text}"
        )
    return "\n\n".join(lines).strip()


def normalize_id(value: Any) -> str:
    text = str(value or "").strip()
    text = re.sub(r"\s+", " ", text)
    return "|".join(part.strip() for part in text.split("|"))

def parse_performance_id(value: Any) -> tuple[str, str, str, str] | None:
    text = normalize_id(value)
    if not text:
        return None
    parts = [part.strip() for part in text.split("|")]
    if len(parts) != 4 or any(not part for part in parts):
        return None
    pfas, adsorbent_id, test_mode, condition = parts
    test_mode = test_mode.casefold()
    if test_mode not in PERFORMANCE_TEST_MODES:
        return None
    return pfas, adsorbent_id, test_mode, condition


def canonical_performance_id(value: Any) -> str:
    parsed = parse_performance_id(value)
    if parsed is None:
        return normalize_id(value)
    return "|".join(parsed)

def normalize_for_compare(value: Any) -> str:
    return canonical_performance_id(value).casefold()


def first_performance_enum(chunk: dict) -> dict:
    for enum in chunk.get("enumerations") or []:
        if isinstance(enum, dict) and enum.get("task") == "performance":
            return enum
    return {}


def performance_records_from_chunk(chunk: dict) -> list[dict]:
    enum = first_performance_enum(chunk)
    records = enum.get("records") if isinstance(enum, dict) else []
    return copy.deepcopy(records if isinstance(records, list) else [])


def count_performance_records(outputs: list[dict]) -> int:
    total = 0
    for chunk in outputs or []:
        total += sum(
            1
            for record in performance_records_from_chunk(chunk)
            if parse_performance_id(record.get("Performance_id")) is not None
        )
    return total


def records_to_df(records: list[dict], columns: list[str]) -> pd.DataFrame:
    rows = []
    for record in records or []:
        if isinstance(record, dict):
            rows.append({column: str(record.get(column) or "") for column in columns})
    return pd.DataFrame(rows, columns=columns)


def record_column_name(index: int) -> str:
    return f"Record {index + 1}"


def fit_text_width(
    values: list[Any],
    *,
    min_width: int,
    max_width: int,
) -> int:
    longest = 0
    for value in values:
        text = str(value or "")
        longest = max(longest, max((len(part) for part in text.splitlines()), default=0))
    return max(min_width, min(max_width, 28 + longest * 7))


def field_column_width(fields: list[str]) -> int:
    return fit_text_width(["Field", *fields], min_width=110, max_width=165)


def transposed_record_column_width(
    df: pd.DataFrame,
    column: str,
    *,
    record_count: int,
    field_width: int,
) -> int:
    values = df[column].tolist() if df is not None and column in df.columns else []
    target_table_width = 560
    count_cap = max(
        78,
        min(360, int((target_table_width - field_width) / max(1, record_count))),
    )
    return fit_text_width([column, *values], min_width=78, max_width=count_cap)


def records_to_transposed_df(
    records: list[dict],
    columns: list[str],
    record_count: int | None = None,
) -> pd.DataFrame:
    count = max(1, int(record_count or len(records or []) or 1))
    out_columns = ["Field"] + [record_column_name(i) for i in range(count)]
    rows = []
    for field in columns:
        row: dict[str, Any] = {"Field": field}
        for i in range(count):
            record = records[i] if i < len(records or []) and isinstance(records[i], dict) else {}
            row[record_column_name(i)] = str(record.get(field) or "")
        rows.append(row)
    return pd.DataFrame(rows, columns=out_columns)


def table_df_to_records(df: pd.DataFrame, columns: list[str]) -> list[dict]:
    if df is None or df.empty:
        return []
    cleaned = []
    for raw in df.to_dict("records"):
        row = {column: str(raw.get(column) or "").strip() for column in columns}
        if any(row.values()):
            cleaned.append(row)
    return cleaned


def resize_transposed_df(df: pd.DataFrame, columns: list[str], record_count: int) -> pd.DataFrame:
    if df is not None and not df.empty and "Field" not in df.columns:
        return records_to_transposed_df(table_df_to_records(df, columns), columns, record_count)

    out_columns = ["Field"] + [record_column_name(i) for i in range(max(1, record_count))]
    existing: dict[str, dict[str, Any]] = {}
    if df is not None and not df.empty:
        for _, row in df.iterrows():
            field = str(row.get("Field") or "").strip()
            if field:
                existing[field] = {column: row.get(column, "") for column in df.columns}

    rows = []
    for field in columns:
        old = existing.get(field, {})
        row = {"Field": field}
        for column in out_columns[1:]:
            row[column] = old.get(column, "")
        rows.append(row)
    return pd.DataFrame(rows, columns=out_columns)


def transposed_df_to_records(df: pd.DataFrame, columns: list[str]) -> list[dict]:
    if df is None or df.empty or "Field" not in df.columns:
        return []

    field_rows = []
    known_fields = set(columns)
    for _, row in df.iterrows():
        field = str(row.get("Field") or "").strip()
        if field and field in known_fields:
            field_rows.append((field, row))

    cleaned = []
    for record_column in [column for column in df.columns if str(column).startswith("Record ")]:
        row = {}
        for field, field_row in field_rows:
            value = field_row.get(record_column, "")
            if pd.isna(value):
                value = ""
            row[field] = str(value).strip()
        if any(row.values()):
            cleaned.append(row)
    return cleaned


def performance_df_to_records(df: pd.DataFrame) -> list[dict]:
    if df is None or df.empty:
        return []
    records: list[dict] = []
    for raw in df.to_dict("records"):
        normalized = canonical_performance_id(raw.get("Performance_id"))
        if normalized:
            records.append(
                {
                    "Performance_id": normalized,
                    "Data_Source": str(raw.get("Data_Source") or "").strip(),
                }
            )
    return records


def records_to_performance_df(records: list[dict]) -> pd.DataFrame:
    rows = []
    for record in records or []:
        if isinstance(record, dict):
            rows.append(
                {
                    "Performance_id": canonical_performance_id(record.get("Performance_id")),
                    "Data_Source": str(record.get("Data_Source") or "").strip(),
                }
            )
    return pd.DataFrame(rows, columns=PERFORMANCE_COLUMNS).astype("string")


def merge_performance_chunks(
    active_payload: dict,
    source_payload: dict,
    classified_entries: dict[str, dict],
) -> tuple[dict[str, dict], dict[str, list[dict]]]:
    chunks: dict[str, dict] = {}

    for payload in (source_payload, active_payload):
        for output in payload.get("outputs") or []:
            if not isinstance(output, dict):
                continue
            key = chunk_key(output.get("file_type"), output.get("chunk_id"))
            if not key.strip(":"):
                continue
            merged = copy.deepcopy(chunks.get(key, {}))
            output_copy = copy.deepcopy(output)
            metadata = output_source_metadata(output_copy)
            output_copy["enriched_text_metadata"] = metadata
            merged.update(output_copy)
            if not merged.get("enriched_text"):
                classified = classified_entries.get(key, {})
                merged["enriched_text"] = classified.get("enriched_text") or classified.get("text") or ""
            if not merged.get("enriched_text") and metadata:
                merged["enriched_text"] = source_text_from_metadata(metadata, classified_entries)
            chunks[key] = merged

    represented_source_keys: set[str] = set()
    for chunk in chunks.values():
        metadata = output_source_metadata(chunk)
        represented_source_keys.update(metadata_key(meta) for meta in metadata)

    orphan_metadata = [
        meta
        for meta in selected_performance_metadata(source_payload, active_payload)
        if metadata_key(meta) not in represented_source_keys
    ]
    if orphan_metadata:
        study_id = (
            active_payload.get("study_folder")
            or source_payload.get("study_folder")
            or ""
        )
        key = "study_level::performance_unassigned_sources"
        chunks.setdefault(
            key,
            {
                "study_folder": study_id,
                "file_type": "study_level",
                "chunk_id": "performance_unassigned_sources",
                "enriched_text": source_text_from_metadata(orphan_metadata, classified_entries),
                "enriched_text_metadata": orphan_metadata,
                "predicted_label": "performance",
                "annotation": [],
                "enumerations": [],
                "_synthetic_empty_source_group": True,
            },
        )

    active_records = {
        key: performance_records_from_chunk(chunk) for key, chunk in chunks.items()
    }
    return chunks, active_records


def load_review_state(
    study_id: str,
    prediction_root: Path,
    ground_truth_root: Path,
    classified_root: Path,
    prefer_ground_truth: bool = True,
) -> dict:
    source_payloads: dict[str, dict] = {}
    source_paths: dict[str, Path | None] = {}
    active_payloads: dict[str, dict] = {}
    active_paths: dict[str, Path | None] = {}
    loaded_from_ground_truth: dict[str, bool] = {}

    for task in TASKS:
        source_payloads[task], source_paths[task] = load_task_payload(
            prediction_root, study_id, task
        )
        gt_payload, gt_path = load_task_payload(ground_truth_root, study_id, task)
        use_gt = bool(prefer_ground_truth and gt_path)
        active_payloads[task] = gt_payload if use_gt else source_payloads[task]
        active_paths[task] = gt_path if use_gt else source_paths[task]
        loaded_from_ground_truth[task] = use_gt

    source_manifest, source_manifest_path = load_manifest(prediction_root, study_id)
    active_manifest, active_manifest_path = load_manifest(ground_truth_root, study_id)
    classified_entries, classified_path = load_classified_entries(classified_root, study_id)

    performance_chunks, performance_records = merge_performance_chunks(
        active_payloads["performance"],
        source_payloads["performance"],
        classified_entries,
    )

    return {
        "study_id": study_id,
        "prediction_root": prediction_root,
        "ground_truth_root": ground_truth_root,
        "classified_root": classified_root,
        "source_payloads": source_payloads,
        "source_paths": source_paths,
        "active_payloads": active_payloads,
        "active_paths": active_paths,
        "source_manifest": source_manifest,
        "source_manifest_path": source_manifest_path,
        "active_manifest": active_manifest,
        "active_manifest_path": active_manifest_path,
        "classified_path": classified_path,
        "loaded_from_ground_truth": loaded_from_ground_truth,
        "highlights": load_highlights(ground_truth_root, study_id),
        "adsorbents": normalize_adsorbents(active_payloads["adsorbent"].get("adsorbents")),
        "water_types": copy.deepcopy(active_payloads["water_type"].get("water_types") or []),
        "performance_chunks": performance_chunks,
        "performance_records": performance_records,
    }


def build_review_payload_base(
    study_id: str,
    task: str,
    source_payload: dict,
    source_path: Path | None,
    review_status: str,
    reviewed_at: float,
) -> dict:
    payload = copy.deepcopy(source_payload or empty_task_payload(study_id, task))
    payload["study_folder"] = payload.get("study_folder") or study_id
    payload["chain_name"] = task
    payload.pop("artifact_version", None)
    payload["task"] = task
    payload["source_status"] = payload.get("status", "")
    payload["status"] = "reviewed"
    payload["review_status"] = review_status
    payload["reviewed_at_unix"] = reviewed_at
    payload["source_artifact_path"] = str(source_path) if source_path else ""
    return payload


def build_adsorbent_payload(
    state: dict, records: dict[str, list[dict]], review_status: str, reviewed_at: float
) -> dict:
    study_id = state["study_id"]
    payload = build_review_payload_base(
        study_id,
        "adsorbent",
        state["source_payloads"]["adsorbent"],
        state["source_paths"]["adsorbent"],
        review_status,
        reviewed_at,
    )
    payload.pop("water_types", None)
    payload.pop("outputs", None)
    payload["adsorbents"] = normalize_adsorbents(records)
    payload["record_count"] = count_adsorbents(records)
    payload["output_count"] = 0
    return payload


def build_water_type_payload(
    state: dict, records: list[dict], review_status: str, reviewed_at: float
) -> dict:
    study_id = state["study_id"]
    payload = build_review_payload_base(
        study_id,
        "water_type",
        state["source_payloads"]["water_type"],
        state["source_paths"]["water_type"],
        review_status,
        reviewed_at,
    )
    payload.pop("adsorbents", None)
    payload.pop("outputs", None)
    payload["water_types"] = records
    payload["record_count"] = len(records)
    payload["output_count"] = 0
    return payload


def build_performance_outputs(
    chunks: dict[str, dict], records_by_key: dict[str, list[dict]]
) -> list[dict]:
    outputs = []
    for key, chunk in sorted(chunks.items(), key=sort_chunk_key):
        records = copy.deepcopy(records_by_key.get(key, []))
        if chunk.get("_synthetic_empty_source_group") and not records:
            continue
        output = {
            k: copy.deepcopy(v)
            for k, v in chunk.items()
            if not str(k).startswith("_")
        }
        enum_meta = {
            k: copy.deepcopy(v)
            for k, v in first_performance_enum(chunk).items()
            if k not in {"records", "extracted_count", "total_count"}
        }
        if records:
            enum_meta["task"] = "performance"
            enum_meta["total_count"] = len(records)
            enum_meta["extracted_count"] = len(records)
            enum_meta["records"] = records
            output["enumerations"] = [enum_meta]
        else:
            output["enumerations"] = []
        outputs.append(output)
    return outputs


def build_performance_payload(
    state: dict,
    records_by_key: dict[str, list[dict]],
    review_status: str,
    reviewed_at: float,
) -> dict:
    study_id = state["study_id"]
    payload = build_review_payload_base(
        study_id,
        "performance",
        state["source_payloads"]["performance"],
        state["source_paths"]["performance"],
        review_status,
        reviewed_at,
    )
    payload.pop("adsorbents", None)
    payload.pop("water_types", None)
    outputs = build_performance_outputs(state["performance_chunks"], records_by_key)
    payload["outputs"] = outputs
    payload["selected_chunk_metadata"] = (
        payload.get("selected_chunk_metadata")
        or selected_performance_metadata(state["source_payloads"]["performance"], payload)
        or build_chunk_metadata(outputs)
    )
    payload["record_count"] = count_performance_records(outputs)
    payload["output_count"] = len(outputs)
    return payload


def save_review_state(state: dict, output_root: Path, review_status: str) -> dict[str, Path]:
    study_id = state["study_id"]
    reviewed_at = time.time()

    adsorbent_payload = build_adsorbent_payload(
        state, state["adsorbents"], review_status, reviewed_at
    )
    water_type_payload = build_water_type_payload(
        state, state["water_types"], review_status, reviewed_at
    )
    performance_payload = build_performance_payload(
        state, state["performance_records"], review_status, reviewed_at
    )

    paths = {
        "adsorbent": task_artifact_path(output_root, study_id, "adsorbent"),
        "water_type": task_artifact_path(output_root, study_id, "water_type"),
        "performance": task_artifact_path(output_root, study_id, "performance"),
        "manifest": manifest_path(output_root, study_id),
    }

    save_json(adsorbent_payload, paths["adsorbent"])
    save_json(water_type_payload, paths["water_type"])
    save_json(performance_payload, paths["performance"])

    source_manifest = state.get("source_manifest") or {}
    manifest = {
        "study_id": study_id,
        "created_at_unix": source_manifest.get("created_at_unix"),
        "reviewed_at_unix": reviewed_at,
        "review_status": review_status,
        "provider": source_manifest.get("provider", ""),
        "model_name": source_manifest.get("model_name", ""),
        "source_manifest_path": str(state.get("source_manifest_path") or ""),
        "artifacts": {
            "adsorbent": str(paths["adsorbent"]),
            "water_type": str(paths["water_type"]),
            "performance": str(paths["performance"]),
        },
    }
    save_json(manifest, paths["manifest"])
    return paths


def validate_adsorbents(adsorbents: Any) -> tuple[list[str], list[str]]:
    errors: list[str] = []
    warnings: list[str] = []
    seen: dict[str, int] = {}
    row_number = 0
    for scope, records in normalize_adsorbents(adsorbents).items():
        label = ADSORBENT_SCOPE_LABELS.get(scope, scope)
        for idx, row in enumerate(records, start=1):
            if not isinstance(row, dict):
                continue
            row_number += 1
            row_label = f"{label} row {idx}"
            adsorbent_id = str(row.get("Adsorbent_id") or "").strip()
            category = str(row.get("Adsorbent_category") or "").strip()
            if not adsorbent_id:
                errors.append(f"{row_label}: Adsorbent_id is required.")
            else:
                norm = adsorbent_id.casefold()
                if norm in seen:
                    errors.append(
                        f"{row_label}: duplicate Adsorbent_id also appears on adsorbent row {seen[norm]}."
                    )
                seen[norm] = row_number
            if scope == "in_scope" and category and category not in ADSORBENT_CATEGORIES:
                errors.append(
                    f"{row_label}: Adsorbent_category must use the approved in-scope category list."
                )
            if not category:
                warnings.append(f"{row_label}: Adsorbent_category is blank.")
    return errors, warnings


def validate_water_types(records: list[dict]) -> tuple[list[str], list[str]]:
    errors: list[str] = []
    warnings: list[str] = []
    seen: set[tuple[str, str]] = set()
    for idx, row in enumerate(records, start=1):
        full_name = str(row.get("Full_name") or "").strip()
        abbreviation = str(row.get("Abbreviation") or "").strip()
        water_class = str(row.get("Class") or "").strip()
        if not (full_name or abbreviation):
            errors.append(f"Water types row {idx}: Full_name or Abbreviation is required.")
        if water_class and water_class not in WATER_TYPE_CLASSES:
            errors.append(f"Water types row {idx}: Class must use the approved class list.")
        if not water_class:
            warnings.append(f"Water types row {idx}: Class is blank.")
        key = (full_name.casefold(), abbreviation.casefold())
        if key != ("", "") and key in seen:
            warnings.append(f"Water types row {idx}: duplicate water matrix label.")
        seen.add(key)
    return errors, warnings


def validate_performance_id(value: str) -> str | None:
    text = normalize_id(value)
    if not text:
        return None
    parts = text.split("|")
    if len(parts) != 4 or any(not part.strip() for part in parts):
        return f"Performance_id must be {PERFORMANCE_ID_FORMAT}."
    test_mode = parts[2].strip().casefold()
    if test_mode not in PERFORMANCE_TEST_MODES:
        return "Test_mode must be one of: " + ", ".join(PERFORMANCE_TEST_MODES) + "."
    return None


def validate_performance(
    records_by_key: dict[str, list[dict]], adsorbent_ids: set[str]
) -> tuple[list[str], list[str]]:
    errors: list[str] = []
    warnings: list[str] = []
    ads_norm = {item.casefold() for item in adsorbent_ids if item}
    seen_global: dict[str, str] = {}
    for key, records in sorted(records_by_key.items()):
        seen: set[str] = set()
        for idx, row in enumerate(records, start=1):
            perf_id = normalize_id(row.get("Performance_id"))
            issue = validate_performance_id(perf_id)
            label = chunk_label_from_key(key)
            if issue:
                errors.append(f"{label} row {idx}: {issue}")
                continue
            norm = normalize_for_compare(perf_id)
            if norm in seen:
                errors.append(f"{label} row {idx}: duplicate Performance_id in this chunk.")
            seen.add(norm)
            if norm in seen_global and seen_global[norm] != label:
                warnings.append(
                    f"{label} row {idx}: duplicate Performance_id also appears in {seen_global[norm]}."
                )
            else:
                seen_global[norm] = label
            parsed = parse_performance_id(perf_id)
            adsorbent_id = parsed[1] if parsed else ""
            if adsorbent_id and adsorbent_id.casefold() != "na" and adsorbent_id.casefold() not in ads_norm:
                warnings.append(
                    f"{label} row {idx}: Adsorbent_id '{adsorbent_id}' is not in the reviewed adsorbent table."
                )
    return errors, warnings


def dataframe_with_counts(payloads: dict[str, dict], loaded_from_gt: dict[str, bool]) -> pd.DataFrame:
    rows = []
    for task in TASKS:
        payload = payloads.get(task) or {}
        rows.append(
            {
                "task": task,
                "status": payload.get("status", ""),
                "mode": payload.get("mode", ""),
                "records": payload.get("record_count", 0),
                "outputs": payload.get("output_count", 0),
                "loaded_from_ground_truth": loaded_from_gt.get(task, False),
                "message": payload.get("message", ""),
            }
        )
    return pd.DataFrame(rows)


def inject_source_text_css() -> None:
    st.markdown(
        """
        <style>
          .source-text-full {
            border: 1px solid rgba(49, 51, 63, 0.20);
            border-radius: 6px;
            padding: 0.75rem 0.9rem;
            margin: 0.35rem 0 0.8rem 0;
            background: rgba(250, 250, 250, 0.90);
            color: rgb(31, 35, 40);
            font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
            font-size: 0.92rem;
            line-height: 1.45;
            white-space: pre-wrap;
            overflow: auto;
            overflow-wrap: anywhere;
            max-height: 34rem;
          }
          .source-chunk-heading {
            margin-top: 0.55rem;
            margin-bottom: 0.10rem;
          }
        </style>
        """,
        unsafe_allow_html=True,
    )


def safe_component_key(*parts: Any) -> str:
    raw = "__".join(str(part) for part in parts if part is not None)
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", raw)[:512]

def compact_table_whitespace(text: str) -> str:
    """
    Remove excessive blank lines between a table caption/header and the markdown
    table body. This follows the same idea used in extraction_UI.py.
    """
    if not text:
        return ""
    cleaned = str(text).replace("\r\n", "\n").replace("\r", "\n")
    cleaned = re.sub(r"\n{2,}(?=\|)", "\n", cleaned)
    cleaned = re.sub(r"\n{4,}", "\n\n", cleaned)
    return cleaned.strip()


def is_table_chunk(source_id: str | None, meta: dict | None, text: str) -> bool:
    meta = meta or {}
    file_type = str(meta.get("file_type") or "").strip().lower()
    if file_type == "table":
        return True
    if str(source_id or "").startswith("table_"):
        return True
    return "\n|" in str(text or "")


def source_text_for_display(
    text: str,
    *,
    source_id: str | None = None,
    meta: dict | None = None,
) -> str:
    if is_table_chunk(source_id, meta, text):
        return compact_table_whitespace(text)
    return str(text or "")

def render_plain_source_text(
    text: str,
    annotations: list[dict] | None = None,
) -> None:
    raw_text = str(text or "")
    highlights = []
    last_end = 0
    for annotation in normalize_highlight_annotations(annotations or []):
        start = max(0, min(int(annotation["start"]), len(raw_text)))
        end = max(0, min(int(annotation["end"]), len(raw_text)))
        if start < end and start >= last_end:
            highlights.append((start, end))
            last_end = end

    if highlights:
        parts: list[str] = []
        cursor = 0
        for start, end in highlights:
            parts.append(html.escape(raw_text[cursor:start]))
            parts.append(f"<mark>{html.escape(raw_text[start:end])}</mark>")
            cursor = end
        parts.append(html.escape(raw_text[cursor:]))
        escaped = "".join(parts)
    else:
        escaped = html.escape(raw_text)

    st.markdown(
        f"<div class='source-text-full'>{escaped}</div>",
        unsafe_allow_html=True,
    )


def render_highlighted_source_text(
    state: dict,
    *,
    source_id: str,
    text: str,
    key_scope: str,
    read_only: bool = False,
) -> None:
    if not text:
        st.caption("No source text available.")
        return

    source_id = normalize_source_id(source_id)
    if not source_id:
        render_plain_source_text(text)
        return

    state.setdefault("highlights", {})
    existing = normalize_highlight_annotations(state["highlights"].get(source_id, []))
    if read_only:
        render_plain_source_text(text, existing)
        return

    if text_highlighter is None:
        st.caption("Highlight component is not installed; showing plain source text.")
        render_plain_source_text(text)
        return

    result = text_highlighter(
        text=text,
        labels=[(HIGHLIGHT_TAG, HIGHLIGHT_COLOR)],
        selected_label=HIGHLIGHT_TAG,
        annotations=existing,
        show_label_selector=False,
        key=safe_component_key("enum_th", state["study_id"], source_id, key_scope),
    )
    if result is not None:
        state["highlights"][source_id] = normalize_highlight_annotations(result)


def metadata_with_prompt_numbers(metadata: list[dict]) -> list[dict]:
    rows: list[dict] = []
    for idx, meta in enumerate(dedupe_metadata(metadata), start=1):
        row = copy.deepcopy(meta)
        row["prompt_chunk_number"] = str(row.get("prompt_chunk_number") or idx).strip()
        rows.append(row)
    return rows


def split_data_source_values(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, dict):
        tokens: list[str] = []
        for item in value.values():
            tokens.extend(split_data_source_values(item))
        return tokens
    if isinstance(value, (list, tuple, set)):
        tokens: list[str] = []
        for item in value:
            tokens.extend(split_data_source_values(item))
        return tokens

    text = str(value or "").strip()
    if not text:
        return []
    tokens = re.findall(
        r"(?:Data_Source|source_id|chunk_id)\s*[:=]\s*([A-Za-z0-9_.-]+)",
        text,
        flags=re.IGNORECASE,
    )
    cleaned = re.sub(
        r"^(?:Data_Source|source_id|chunk_id)\s*[:=]\s*",
        "",
        text,
        flags=re.IGNORECASE,
    )
    tokens.extend(part.strip() for part in re.split(r"\s*(?:;|,)\s*", cleaned) if part.strip())
    return tokens


def data_source_aliases(metadata: list[dict]) -> dict[str, str]:
    aliases: dict[str, str] = {}
    for meta in metadata_with_prompt_numbers(metadata):
        key = metadata_key(meta)
        if not key.strip(":"):
            continue
        source_id = str(
            meta.get("source_id") or source_id_for_values(meta.get("file_type"), meta.get("chunk_id"))
        ).strip()
        prompt_chunk_number = str(meta.get("prompt_chunk_number") or "").strip()
        for alias in (
            prompt_chunk_number,
            f"chunk {prompt_chunk_number}",
            f"chunk_{prompt_chunk_number}",
            source_id,
            f"Data_Source={source_id}",
            f"source_id={source_id}",
            f"chunk_id={meta.get('chunk_id')}",
        ):
            alias_text = str(alias or "").strip()
            if alias_text:
                aliases[normalize_source_id(alias_text).casefold()] = key
                aliases[alias_text.casefold()] = key
    return aliases


def source_keys_from_record(record: dict, metadata: list[dict]) -> list[str]:
    lower = {str(k).strip().lower(): v for k, v in (record or {}).items()}
    raw_source = lower.get("data_source") or lower.get("data_provenance")
    aliases = data_source_aliases(metadata)
    keys: list[str] = []
    seen: set[str] = set()
    for token in split_data_source_values(raw_source):
        cleaned = str(token or "").strip().strip("'\"`[](){}")
        cleaned = re.sub(r"^chunk\s*#?\s*", "", cleaned, flags=re.IGNORECASE).strip()
        alias_key = aliases.get(normalize_source_id(cleaned).casefold()) or aliases.get(cleaned.casefold())
        key = alias_key or cleaned
        if key and key not in seen:
            seen.add(key)
            keys.append(key)
    return keys


def records_have_data_sources(records: list[dict]) -> bool:
    for record in records or []:
        if not isinstance(record, dict):
            continue
        lower = {str(k).strip().lower(): v for k, v in record.items()}
        if split_data_source_values(lower.get("data_source") or lower.get("data_provenance")):
            return True
    return False


def records_for_adsorbent_source_review(adsorbents: Any) -> list[dict]:
    records: list[dict] = []
    for scope_records in normalize_adsorbents(adsorbents).values():
        records.extend(row for row in scope_records if isinstance(row, dict))
    return records


def render_source_chunk_items(
    metadata: list[dict],
    state: dict,
    classified_entries: dict[str, dict] | None = None,
    key_scope: str | None = None,
    read_only: bool = False,
) -> None:
    classified_entries = classified_entries or {}
    metadata = metadata_with_prompt_numbers(metadata)
    if not metadata:
        st.caption("No selected chunks recorded.")
        return
    for meta in metadata:
        key = chunk_key(meta.get("file_type"), meta.get("chunk_id"))
        classified = classified_entries.get(key, {})
        source_id = normalize_source_id(
            meta.get("source_id")
            or source_id_for_values(meta.get("file_type"), meta.get("chunk_id"))
        )
        text = source_text_for_display(
            classified.get("enriched_text") or classified.get("text") or "",
            source_id=source_id,
            meta=meta,
        )
        tags = []
        for tag in (
            meta.get("section_label") or classified.get("section_label"),
            meta.get("predicted_label") or classified.get("predicted_label"),
        ):
            tag_text = str(tag or "").strip()
            if tag_text:
                tags.append(f"<code>{html.escape(tag_text)}</code>")
        prompt_chunk_number = str(meta.get("prompt_chunk_number") or "").strip()
        chunk_label = f"Chunk {prompt_chunk_number}: " if prompt_chunk_number else ""
        source_label = f"{meta.get('file_type')} {meta.get('chunk_id')}".strip()
        st.markdown(
            "<div class='source-chunk-heading'>"
            f"<b>{html.escape(chunk_label + source_label)}</b> "
            f"{' '.join(tags)}"
            "</div>",
            unsafe_allow_html=True,
        )
        render_highlighted_source_text(
            state,
            source_id=source_id,
            text=text,
            key_scope=f"{key_scope or 'source_chunks'}_{key}",
            read_only=read_only,
        )


def render_source_chunks(
    title: str,
    metadata: list[dict],
    state: dict,
    classified_entries: dict[str, dict] | None = None,
    key_scope: str | None = None,
    read_only: bool = False,
) -> None:
    with st.expander(title, expanded=False):
        render_source_chunk_items(
            metadata,
            state,
            classified_entries,
            key_scope or title,
            read_only=read_only,
        )


def render_record_linked_source_chunk_items(
    metadata: list[dict],
    records: list[dict],
    state: dict,
    classified_entries: dict[str, dict] | None = None,
    key_scope: str | None = None,
    read_only: bool = False,
) -> None:
    metadata = metadata_with_prompt_numbers(metadata)
    if not metadata:
        st.caption("No selected chunks recorded.")
        return

    source_to_records: dict[str, list[int]] = {}
    records_without_sources = 0
    for idx, record in enumerate(records or [], start=1):
        if not isinstance(record, dict):
            continue
        source_keys = source_keys_from_record(record, metadata)
        if not source_keys:
            records_without_sources += 1
        for source_key in source_keys:
            source_to_records.setdefault(source_key, []).append(idx)

    meta_by_key = {metadata_key(meta): meta for meta in metadata}
    order_by_key = {metadata_key(meta): idx for idx, meta in enumerate(metadata)}
    if source_to_records:
        st.caption("Record-level Data_Source found. Cited source chunks are shown first.")
        used_keys = set(source_to_records)
        for source_key in sorted(source_to_records, key=lambda item: order_by_key.get(item, 10**9)):
            meta = meta_by_key.get(source_key)
            if meta:
                render_source_chunk_items(
                    [meta],
                    state,
                    classified_entries,
                    key_scope=f"{key_scope or 'source_chunks'}_{source_key}",
                    read_only=read_only,
                )
            else:
                st.caption(f"No matching source chunk found for `{source_key}`.")

        unused = [meta for meta in metadata if metadata_key(meta) not in used_keys]
        if unused:
            with st.expander("Source chunks not cited by Data_Source", expanded=False):
                render_source_chunk_items(
                    unused,
                    state,
                    classified_entries,
                    key_scope=f"{key_scope or 'source_chunks'}_unused",
                    read_only=read_only,
                )
        if records_without_sources:
            st.caption(f"{records_without_sources} record(s) have no Data_Source field.")
        return

    st.caption("No record-level Data_Source was found; showing all selected source chunks.")
    render_source_chunk_items(
        metadata,
        state,
        classified_entries,
        key_scope=key_scope or "source_chunks_all",
        read_only=read_only,
    )


def render_record_linked_source_chunks(
    title: str,
    metadata: list[dict],
    records: list[dict],
    state: dict,
    classified_entries: dict[str, dict] | None = None,
    key_scope: str | None = None,
    read_only: bool = False,
) -> None:
    with st.expander(title, expanded=records_have_data_sources(records)):
        render_record_linked_source_chunk_items(
            metadata,
            records,
            state,
            classified_entries,
            key_scope or title,
            read_only=read_only,
        )


def render_validation(errors: list[str], warnings: list[str]) -> None:
    if errors:
        with st.expander("Blocking validation issues", expanded=True):
            for item in errors:
                st.error(item)
    if warnings:
        with st.expander("Warnings", expanded=False):
            for item in warnings:
                st.warning(item)


def initialize_session_state(
    study_id: str,
    prediction_root: Path,
    ground_truth_root: Path,
    classified_root: Path,
    prefer_ground_truth: bool,
) -> None:
    reload_nonce = str(st.session_state.get("enum_review_reload_nonce", ""))
    signature = "|".join(
        [
            study_id,
            str(prediction_root),
            str(ground_truth_root),
            str(classified_root),
            str(prefer_ground_truth),
            reload_nonce,
        ]
    )
    if st.session_state.get("enum_review_signature") == signature:
        return

    state = load_review_state(
        study_id,
        prediction_root,
        ground_truth_root,
        classified_root,
        prefer_ground_truth=prefer_ground_truth,
    )
    state["session_key"] = hashlib.sha1(signature.encode("utf-8")).hexdigest()[:12]
    st.session_state.enum_review_signature = signature
    st.session_state.enum_review_state = state


def main() -> None:
    st.title("Enumeration Review")
    inject_source_text_css()

    with st.sidebar:
        prefer_ground_truth = st.checkbox("Resume saved ground truth", value=True)
        review_summaries = {
            sid: ground_truth_review_summary(
                get_study_paths(sid).enumeration_ground_truth_dir,
                sid,
            )
            for sid in SELECTED_STUDY_IDS
        }
        show_reviewed = st.checkbox("Show reviewed studies", value=False)
        study_options = (
            SELECTED_STUDY_IDS
            if show_reviewed
            else [sid for sid in SELECTED_STUDY_IDS if not review_summaries[sid][0]]
        )
        if not study_options:
            st.info("All configured studies are reviewed. Showing reviewed studies.")
            study_options = SELECTED_STUDY_IDS

        study_id = st.selectbox(
            "Study",
            study_options,
            format_func=lambda sid: f"{sid} [{review_summaries[sid][1]}]",
        )
        st.caption(f"{len(study_options)} of {len(SELECTED_STUDY_IDS)} configured studies shown")

        if st.button("Reload study"):
            st.session_state.pop("enum_review_signature", None)
            st.session_state.enum_review_reload_nonce = str(time.time())

    study_paths = get_study_paths(study_id)
    prediction_root = study_paths.enumeration_dir
    ground_truth_root = study_paths.enumeration_ground_truth_dir
    classified_root = study_paths.classification_dir

    initialize_session_state(
        study_id,
        prediction_root,
        ground_truth_root,
        classified_root,
        prefer_ground_truth,
    )
    state = st.session_state.enum_review_state
    classified_entries, _ = load_classified_entries(classified_root, study_id)

    active_manifest = state.get("active_manifest") or {}
    loaded_ground_truth_tasks = [
        task
        for task, loaded in (state.get("loaded_from_ground_truth") or {}).items()
        if loaded
    ]
    loaded_existing_ground_truth = bool(loaded_ground_truth_tasks)
    if loaded_existing_ground_truth and active_manifest:
        status = str(active_manifest.get("review_status") or "saved").strip()
        reviewed_at = format_review_time(active_manifest.get("reviewed_at_unix"))
        suffix = f" ({status}, {reviewed_at})" if reviewed_at else f" ({status})"
        st.success(f"Loaded saved ground truth for {study_id}{suffix}.")
    with st.sidebar:
        st.divider()
        st.subheader("Edit mode")
        if loaded_existing_ground_truth:
            edit_ground_truth = st.toggle(
                "Enable ground-truth editing",
                value=False,
                help=(
                    "Saved ground-truth artifacts are read-only by default. "
                    "Turn this on only when you want to revise and overwrite the GT files."
                ),
            )
            if edit_ground_truth:
                st.warning(
                    "Ground-truth editing is enabled. Saving will overwrite the GT artifacts "
                    "for this study."
                )
            else:
                st.info("Saved ground truth is open in read-only mode.")
        else:
            edit_ground_truth = True
            st.info("No saved ground truth was loaded; editing is enabled to create GT.")

    read_only_mode = not edit_ground_truth
    adsorbent_tab, water_tab, performance_tab = st.tabs(
        ["Adsorbents", "Water Types", "Performance"]
    )


    with adsorbent_tab:
        state["adsorbents"] = normalize_adsorbents(state["adsorbents"])
        metric_cols = st.columns(3)
        metric_cols[0].metric("Adsorbent IDs", count_adsorbents(state["adsorbents"]))
        metric_cols[1].metric("In scope", len(state["adsorbents"]["in_scope"]))
        metric_cols[2].metric("Out of scope", len(state["adsorbents"]["out_of_scope"]))
        ads_source_col, ads_record_col = st.columns([0.95, 1.15], gap="large")
        with ads_record_col:
            with st.container(height=REVIEW_PANEL_HEIGHT, border=False):
                for scope in ADSORBENT_SCOPE_KEYS:
                    st.subheader(ADSORBENT_SCOPE_LABELS[scope])
                    ads_editor_key = f"adsorbents_editor_{scope}_{state['session_key']}"
                    ads_df_key = f"{ads_editor_key}_df"
                    ads_count_key = f"{ads_editor_key}_record_count"
                    initial_ads_count = max(1, len(state["adsorbents"][scope]))
                    if ads_count_key not in st.session_state:
                        st.session_state[ads_count_key] = initial_ads_count
                    ads_record_count = int(
                        st.number_input(
                            "Record columns",
                            min_value=1,
                            max_value=max(50, initial_ads_count),
                            step=1,
                            disabled=read_only_mode,
                            key=ads_count_key,
                            width=160,
                        )
                    )
                    if ads_df_key not in st.session_state:
                        st.session_state[ads_df_key] = records_to_transposed_df(
                            state["adsorbents"][scope], ADSORBENT_COLUMNS, ads_record_count
                        )
                    st.session_state[ads_df_key] = resize_transposed_df(
                        st.session_state[ads_df_key],
                        ADSORBENT_COLUMNS,
                        ads_record_count,
                    )
                    ads_field_width = field_column_width(ADSORBENT_COLUMNS)
                    column_config = {"Field": st.column_config.TextColumn("Field", width=ads_field_width)}
                    for i in range(ads_record_count):
                        column = record_column_name(i)
                        column_config[column] = st.column_config.TextColumn(
                            column,
                            width=transposed_record_column_width(
                                st.session_state[ads_df_key],
                                column,
                                record_count=ads_record_count,
                                field_width=ads_field_width,
                            ),
                        )
                    edited_ads_df = st.data_editor(
                        st.session_state[ads_df_key],
                        num_rows="fixed",
                        hide_index=True,
                        width="stretch",
                        column_config=column_config,
                        disabled=True if read_only_mode else ["Field"],
                        key=ads_editor_key,
                    )
                    st.session_state[ads_df_key] = edited_ads_df
                    state["adsorbents"][scope] = transposed_df_to_records(edited_ads_df, ADSORBENT_COLUMNS)
        with ads_source_col:
            with st.container(height=REVIEW_PANEL_HEIGHT, border=False):
                render_source_chunks(
                    "Paper abstract",
                    abstract_chunk_metadata(classified_entries),
                    state,
                    classified_entries,
                    key_scope="adsorbent_abstract",
                    read_only=read_only_mode,
                )
                source_payload = state["source_payloads"]["adsorbent"]
                render_record_linked_source_chunks(
                    "Adsorbent source chunks",
                    source_payload.get("selected_chunk_metadata") or [],
                    records_for_adsorbent_source_review(state["adsorbents"]),
                    state,
                    classified_entries,
                    key_scope="adsorbent_sources",
                    read_only=read_only_mode,
                )
    with water_tab:
        st.metric("Water-type IDs", len(state["water_types"]))

        water_source_col, water_record_col = st.columns([0.95, 1.15], gap="large")
        with water_record_col:
            with st.container(height=REVIEW_PANEL_HEIGHT, border=False):
                water_editor_key = f"water_types_editor_{state['study_id']}"
                water_df_key = f"{water_editor_key}_df"
                water_count_key = f"{water_editor_key}_record_count"
                initial_water_count = max(1, len(state["water_types"]))
                if water_count_key not in st.session_state:
                    st.session_state[water_count_key] = initial_water_count
                water_record_count = int(
                    st.number_input(
                        "Record columns",
                        min_value=1,
                        max_value=max(50, initial_water_count),
                        step=1,
                        disabled=read_only_mode,
                        key=water_count_key,
                        width=160,
                    )
                )
                if water_df_key not in st.session_state:
                    st.session_state[water_df_key] = records_to_transposed_df(
                        state["water_types"], WATER_TYPE_COLUMNS, water_record_count
                    )
                st.session_state[water_df_key] = resize_transposed_df(
                    st.session_state[water_df_key],
                    WATER_TYPE_COLUMNS,
                    water_record_count,
                )
                water_field_width = field_column_width(WATER_TYPE_COLUMNS)
                water_column_config = {"Field": st.column_config.TextColumn("Field", width=water_field_width)}
                for i in range(water_record_count):
                    column = record_column_name(i)
                    water_column_config[column] = st.column_config.TextColumn(
                        column,
                        width=transposed_record_column_width(
                            st.session_state[water_df_key],
                            column,
                            record_count=water_record_count,
                            field_width=water_field_width,
                        ),
                    )
                edited_water_df = st.data_editor(
                    st.session_state[water_df_key],
                    num_rows="fixed",
                    hide_index=True,
                    width="stretch",
                    column_config=water_column_config,
                    disabled=True if read_only_mode else ["Field"],
                    key=water_editor_key,
                )
                st.session_state[water_df_key] = edited_water_df
                state["water_types"] = transposed_df_to_records(edited_water_df, WATER_TYPE_COLUMNS)
        with water_source_col:
            with st.container(height=REVIEW_PANEL_HEIGHT, border=False):
                render_source_chunks(
                    "Paper abstract",
                    abstract_chunk_metadata(classified_entries),
                    state,
                    classified_entries,
                    key_scope="water_type_abstract",
                    read_only=read_only_mode,
                )
                source_payload = state["source_payloads"]["water_type"]
                render_record_linked_source_chunks(
                    "Water-type source chunks",
                    source_payload.get("selected_chunk_metadata") or [],
                    state["water_types"],
                    state,
                    classified_entries,
                    key_scope="water_type_sources",
                    read_only=read_only_mode,
                )

    with performance_tab:
        shown_chunks = sorted(state["performance_chunks"].items(), key=sort_chunk_key)
        source_keys = {
            metadata_key(meta)
            for _, chunk in shown_chunks
            for meta in output_source_metadata(chunk)
            if metadata_key(meta).strip(":")
        }
        perf_metric_cols = st.columns(3)
        perf_metric_cols[0].metric("Review groups", len(shown_chunks))
        perf_metric_cols[1].metric("Source chunks", len(source_keys))
        perf_metric_cols[2].metric(
            "Performance IDs",
            sum(len(v) for v in state["performance_records"].values()),
        )

        st.caption(f"{len(shown_chunks)} performance review groups shown")
        for key, chunk in shown_chunks:
            records = state["performance_records"].get(key, [])
            enum_meta = first_performance_enum(chunk)
            source_metadata = output_source_metadata(chunk)
            source_count = len(source_metadata)
            mode = str(enum_meta.get("mode") or chunk.get("mode") or "").strip()
            file_type = str(chunk.get("file_type") or "").strip()
            chunk_id = chunk.get("chunk_id")
            source_label = f"{source_count} source chunk" + ("" if source_count == 1 else "s")
            title = (
                f"{file_type} {chunk_id} | {source_label} | "
                "Performance records"
            )
            with st.expander(title, expanded=False):
                st.caption(f"{len(records)} Performance_id")
                captions = [
                    str(chunk.get("predicted_label") or "").strip(),
                    mode,
                ]
                st.caption(" | ".join(item for item in captions if item))

                perf_source_col, perf_record_col = st.columns([0.95, 1.15], gap="large")
                with perf_record_col:
                    with st.container(height=REVIEW_PANEL_HEIGHT, border=False):
                        perf_editor_key = safe_component_key(
                            "performance_row_editor",
                            state["session_key"],
                            key,
                        )
                        perf_df_key = f"{perf_editor_key}_df"
                        if perf_df_key not in st.session_state:
                            st.session_state[perf_df_key] = records_to_performance_df(records)
                        edited_perf_df = st.data_editor(
                            st.session_state[perf_df_key],
                            num_rows="dynamic",
                            hide_index=True,
                            width="stretch",
                            column_config={
                                "Performance_id": st.column_config.TextColumn(
                                    "Performance_id",
                                    width="large",
                                ),
                                "Data_Source": st.column_config.TextColumn(
                                    "Data_Source",
                                    width="small",
                                    help="Chunk numbers cited by this record, e.g. 6 or 6, 7.",
                                ),
                            },
                            disabled=read_only_mode,
                            key=perf_editor_key,
                        )
                        # Keep the input dataframe stable for this keyed editor.
                        # Feeding the returned dataframe back into the same
                        # st.data_editor on every rerun can make Streamlit apply
                        # row edits one rerun late (requiring a second delete).
                        records = performance_df_to_records(edited_perf_df)
                        state["performance_records"][key] = records

                with perf_source_col:
                    with st.container(height=REVIEW_PANEL_HEIGHT, border=False):
                        is_grouped_source = file_type.casefold() == "study_level" or source_count > 1
                        if is_grouped_source and source_metadata:
                            st.markdown("**Underlying source chunks**")
                            render_record_linked_source_chunk_items(
                                source_metadata,
                                records,
                                state,
                                classified_entries,
                                key_scope=f"performance_sources_{key}",
                                read_only=read_only_mode,
                            )
                        else:
                            text = source_text_for_display(
                                chunk.get("enriched_text") or "",
                                source_id=source_id_for(chunk),
                                meta=chunk,
                            )
                            render_highlighted_source_text(
                                state,
                                source_id=source_id_for(chunk),
                                text=text,
                                key_scope=f"performance_{key}",
                                read_only=read_only_mode,
                            )

    ads_errors, ads_warnings = validate_adsorbents(state["adsorbents"])
    water_errors, water_warnings = validate_water_types(state["water_types"])
    adsorbent_ids = adsorbent_ids_for_scope(state["adsorbents"], "in_scope")
    perf_errors, perf_warnings = validate_performance(
        state["performance_records"], adsorbent_ids
    )
    all_errors = ads_errors + water_errors + perf_errors
    all_warnings = ads_warnings + water_warnings + perf_warnings

    with st.sidebar:
        st.divider()
        st.metric("Blocking issues", len(all_errors))
        st.metric("Warnings", len(all_warnings))
        st.metric(
            "Highlights",
            sum(len(items) for items in state.get("highlights", {}).values()),
        )
        review_status = st.selectbox(
            "Review status",
            ["reviewed", "draft"],
            index=0,
            disabled=read_only_mode,
        )
        save_clicked = st.button(
            "Save ground truth",
            type="primary",
            disabled=read_only_mode or bool(all_errors),
            width="stretch",
        )
        clear_highlights_clicked = st.button(
            "Clear highlights in session",
            disabled=read_only_mode,
            width="stretch",
        )
        if save_clicked:
            try:
                paths = save_review_state(state, ground_truth_root, review_status)
                save_highlights(
                    ground_truth_root,
                    study_id,
                    state.get("highlights", {}),
                )
            except Exception as exc:  # pragma: no cover - Streamlit surface
                st.error(f"Save failed: {exc}")
            else:
                st.session_state.pop("enum_review_signature", None)
                st.session_state.enum_review_last_saved = (
                    f"Saved ground truth and highlights for {study_id}: {paths['manifest']}"
                )
                st.rerun()
        if clear_highlights_clicked:
            state["highlights"] = {}
            st.success("Session highlights cleared. Save ground truth to persist this.")

    render_validation(all_errors, all_warnings)


if __name__ == "__main__":
    main()
