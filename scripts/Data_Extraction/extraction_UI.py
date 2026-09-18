import copy
import html
import json
import os
import re
import sys
from collections import defaultdict
from typing import Any

import pandas as pd
import streamlit as st
from streamlit import column_config as stcol

try:
    from .workflow_config import ACTIVE_STUDIES, add_project_import_paths, get_study_paths
except ImportError:  # Streamlit executes this file directly.
    from workflow_config import ACTIVE_STUDIES, add_project_import_paths, get_study_paths

add_project_import_paths()

try:
    from text_highlighter import text_highlighter
except ImportError:  # pragma: no cover - UI fallback for environments without the component
    text_highlighter = None


from schemas import ADSORBENT_FIELDS, PERFORMANCE_FIELDS, REVIEW_FIELDS


SELECTED_STUDY_IDS = list(ACTIVE_STUDIES)

TASK_ORDER = ["adsorbent", "performance"]
TASK_LABELS = {
    "adsorbent": "Adsorbent",
    "performance": "Performance",
}
ARTIFACT_KEYS = ["adsorbent_study", "performance"]
HIGHLIGHT_TAG = "highlight"
HIGHLIGHT_COLOR = "yellow"
REVIEW_PANEL_HEIGHT = 760


def _lower_field_names(*field_dicts: dict[str, Any]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for field_dict in field_dicts:
        for key in field_dict:
            norm = str(key).strip().lower()
            if norm and norm not in seen:
                seen.add(norm)
                out.append(norm)
    return out


def _with_front(base: list[str], front: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for key in front + base:
        norm = str(key).strip().lower()
        if norm and norm not in seen:
            seen.add(norm)
            out.append(norm)
    return out


BASE_FIELD_OPTIONS = {
    "adsorbent": _with_front(
        _lower_field_names(ADSORBENT_FIELDS, REVIEW_FIELDS),
        ["adsorbent_id", "data_provenance"],
    ),
    "performance": _with_front(
        _lower_field_names(PERFORMANCE_FIELDS, REVIEW_FIELDS),
        [
            "performance_id",
            "pfas_name",
            "adsorbent_id",
            "test_mode",
            "differentiating_condition",
            "data_provenance",
        ],
    ),
}


def _as_entry_list(payload: Any) -> list[dict]:
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    if isinstance(payload, dict):
        return [payload]
    return []


def _load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def _save_json(data: Any, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2, ensure_ascii=False)


def _find_task_key_case_insensitive(extracted_data: dict, task_lc: str) -> str | None:
    if not isinstance(extracted_data, dict):
        return None
    for key in extracted_data:
        if str(key).strip().lower() == task_lc:
            return key
    return None


def _records_from_task_payload(recs_or_obj: Any) -> list[dict]:
    records = recs_or_obj.get("records") if isinstance(recs_or_obj, dict) else recs_or_obj
    if isinstance(records, dict) and isinstance(records.get("Observations"), list):
        records = records.get("Observations")
    return [rec for rec in (records or []) if isinstance(rec, dict)] if isinstance(records, list) else []


def _task_from_entry(entry: dict) -> str:
    raw_task = str(entry.get("task") or "").strip().lower()
    if raw_task in TASK_LABELS:
        return raw_task

    predicted = str(entry.get("predicted_label") or "").strip().lower()
    if predicted in TASK_LABELS:
        return predicted

    extracted_data = entry.get("extracted_data") or {}
    if isinstance(extracted_data, dict):
        for key in extracted_data:
            task = str(key).strip().lower()
            if task in TASK_LABELS:
                return task

    return raw_task or predicted or "unknown"


def _canonical_task(task: str | None) -> str:
    task_lc = str(task or "").strip().lower()
    if task_lc in {"adsorbent_study", "adsorbent extraction"}:
        return "adsorbent"
    if task_lc in {"experiment_study", "experiment extraction"}:
        return "experiment"
    if task_lc in {"performance_value", "performance_unit"}:
        return "performance"
    return task_lc


def _is_supported_task_entry(entry: dict) -> bool:
    return _canonical_task(_task_from_entry(entry)) in TASK_LABELS


def _entry_mode(entry: dict) -> str:
    mode = entry.get("mode")
    if mode:
        return str(mode)
    task = _task_from_entry(entry)
    data = (entry.get("extracted_data") or {}).get(task) if isinstance(entry.get("extracted_data"), dict) else {}
    if isinstance(data, dict) and data.get("mode"):
        return str(data.get("mode"))
    if entry.get("_artifact_mode"):
        return str(entry.get("_artifact_mode"))
    return ""


def _entry_key(entry: dict) -> tuple[str, str, str, str]:
    task = _canonical_task(_task_from_entry(entry))
    return (
        str(entry.get("study_folder") or "").strip(),
        task,
        str(entry.get("chunk_id") or entry.get("entry_id") or "").strip(),
        _entry_mode(entry),
    )


def _normalize_entry(entry: dict, *, study_id: str, artifact_key: str | None = None) -> dict:
    out = copy.deepcopy(entry)
    out.setdefault("study_folder", study_id)
    task = _canonical_task(_task_from_entry(out))
    out["task"] = task
    out.setdefault("predicted_label", task)
    if artifact_key:
        out.setdefault("_artifact_key", artifact_key)
    out.setdefault("entry_id", str(out.get("chunk_id") or f"{task}_entry"))
    return out


def _artifact_path(study_id: str, artifact_key: str) -> str:
    paths = get_study_paths(study_id)
    return str(paths.extraction_artifact_dir / f"{study_id}_{artifact_key}.json")


def _path_fingerprint(path: str) -> tuple[str, int | None, int | None]:
    if not os.path.exists(path):
        return (path, None, None)
    stat = os.stat(path)
    return (path, int(stat.st_mtime_ns), int(stat.st_size))


def _prediction_cache_token(study_id: str) -> tuple:
    paths = [_artifact_path(study_id, artifact_key) for artifact_key in ARTIFACT_KEYS]
    return tuple(_path_fingerprint(path) for path in paths)


def _ground_truth_cache_token(study_id: str) -> tuple:
    return (_path_fingerprint(_ground_truth_path(study_id)),)


@st.cache_data(show_spinner=False)
def _load_artifact_entries(study_id: str, cache_token: tuple | None = None) -> list[dict]:
    entries: list[dict] = []
    for artifact_key in ARTIFACT_KEYS:
        path = _artifact_path(study_id, artifact_key)
        if not os.path.exists(path):
            continue
        try:
            payload = _load_json(path)
        except Exception:
            continue

        outputs = payload.get("outputs") if isinstance(payload, dict) else None
        if isinstance(outputs, list):
            for item in outputs:
                if isinstance(item, dict):
                    item2 = copy.deepcopy(item)
                    item2.setdefault("_artifact_mode", payload.get("mode"))
                    item2.setdefault("_artifact_status", payload.get("status"))
                    entries.append(_normalize_entry(item2, study_id=study_id, artifact_key=artifact_key))
        elif isinstance(payload, dict) and payload.get("extracted_data"):
            entries.append(_normalize_entry(payload, study_id=study_id, artifact_key=artifact_key))
    return entries


@st.cache_data(show_spinner=False)
def _load_predictions(study_id: str, cache_token: tuple | None = None) -> list[dict]:
    entries = _load_artifact_entries(study_id, cache_token)
    return [
        entry
        for entry in entries
        if _is_supported_task_entry(entry)
        and "irrelevant" not in str(entry.get("predicted_label") or "").lower()
    ]


def _ground_truth_path(study_id: str) -> str:
    return str(get_study_paths(study_id).extraction_ground_truth_file)


def _highlights_path(study_id: str) -> str:
    return str(get_study_paths(study_id).extraction_highlights_file)


def _normalize_highlight_annotations(raw: object) -> list[dict]:
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


def _load_highlights(study_id: str) -> dict[str, list[dict]]:
    path = _highlights_path(study_id)
    if not os.path.exists(path):
        return {}
    try:
        data = _load_json(path)
    except Exception:
        return {}
    if not isinstance(data, dict):
        return {}
    highlights: dict[str, list[dict]] = {}
    for key, value in data.items():
        source_id = _normalize_source_id(key)
        if source_id:
            highlights[source_id] = _normalize_highlight_annotations(value)
    return highlights


def _save_highlights(study_id: str, highlights: dict[str, list[dict]]) -> str:
    path = _highlights_path(study_id)
    payload: dict[str, list[dict]] = {}
    for key, value in (highlights or {}).items():
        source_id = _normalize_source_id(key)
        if source_id:
            payload[source_id] = _normalize_highlight_annotations(value)
    _save_json(payload, path)
    return path


def _load_gt(study_id: str) -> list[dict]:
    path = _ground_truth_path(study_id)
    if not os.path.exists(path):
        return []
    try:
        return [
            entry
            for entry in (
                _normalize_entry(item, study_id=study_id)
                for item in _as_entry_list(_load_json(path))
            )
            if _is_supported_task_entry(entry)
        ]
    except Exception as exc:
        st.warning(f"Could not read ground truth file: {exc}")
        return []


def _filter_new_entries(preds: list[dict], gt: list[dict]) -> list[dict]:
    gt_keys = {_entry_key(entry) for entry in gt}
    return [entry for entry in preds if _entry_key(entry) not in gt_keys]


def _records_for_entry(entry: dict, task_lc: str) -> tuple[str | None, list[dict]]:
    extracted_data = entry.get("extracted_data") or {}
    if not isinstance(extracted_data, dict):
        return None, []
    src_key = _find_task_key_case_insensitive(extracted_data, task_lc)
    if src_key is None:
        return None, []
    return src_key, _records_from_task_payload(extracted_data.get(src_key))


def _task_payload_for_entry(entry: dict, task_lc: str) -> dict:
    extracted_data = entry.get("extracted_data") or {}
    if not isinstance(extracted_data, dict):
        return {}
    src_key = _find_task_key_case_insensitive(extracted_data, task_lc)
    payload = extracted_data.get(src_key) if src_key is not None else {}
    return payload if isinstance(payload, dict) else {}


def _jsonish_to_str(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return str(value)


def _flatten_record(rec: dict) -> dict[str, str]:
    out: dict[str, str] = {}
    for key, value in (rec or {}).items():
        norm = str(key).strip().lower()
        if not norm:
            continue
        out[norm] = _jsonish_to_str(value)
    return out


def _field_order_for_records(task_lc: str, records: list[dict]) -> list[str]:
    fields: list[str] = []
    seen: set[str] = set()
    flattened_records = [_flatten_record(rec) for rec in (records or [])]
    for key in BASE_FIELD_OPTIONS.get(task_lc, []):
        if key not in seen:
            seen.add(key)
            fields.append(key)
    for key in sorted(st.session_state.custom_field_options.get(task_lc, set())):
        if key not in seen:
            seen.add(key)
            fields.append(key)
    for flat in flattened_records:
        for key in flat:
            if key not in seen and key != "__extras__":
                seen.add(key)
                fields.append(key)
    if any(isinstance(rec.get("__extras__"), dict) or "__extras__" in rec for rec in records or []):
        if "__extras__" not in seen:
            fields.append("__extras__")

    def has_value(field: str) -> bool:
        for flat in flattened_records:
            value = flat.get(field, "")
            if str(value or "").strip():
                return True
        return False

    valued_fields = [field for field in fields if has_value(field)]
    valued_field_set = set(valued_fields)
    blank_fields = [field for field in fields if field not in valued_field_set]
    return valued_fields + blank_fields


def _record_column_name(index: int) -> str:
    return f"Record {index + 1}"


def _fit_text_width(
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


def _field_column_width(fields: list[str]) -> int:
    return _fit_text_width(["Field", *fields], min_width=110, max_width=165)


def _identifier_value_width(values: list[Any]) -> int:
    longest = 0
    for value in values:
        text = str(value or "")
        longest = max(longest, max((len(part) for part in text.splitlines()), default=0))
    return max(130, min(460, 44 + longest * 8))


def _record_column_width(
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
    compact_width = _fit_text_width([column, *values], min_width=78, max_width=count_cap)
    if df is not None and "Field" in df.columns and column in df.columns:
        field_names = df["Field"].astype(str).str.strip().str.lower()
        performance_ids = [
            value
            for value in df.loc[field_names == "performance_id", column].tolist()
            if str(value or "").strip()
        ]
        if performance_ids:
            return max(compact_width, _identifier_value_width(performance_ids))
    return compact_width


def _editor_grid_height(field_count: int) -> int:
    row_height = 36
    header_height = 39
    content_height = header_height + row_height * max(1, field_count)
    return max(240, min(640, content_height))


def _records_to_editor_df(
    records: list[dict],
    task_lc: str,
    record_count: int | None = None,
) -> tuple[pd.DataFrame, list[str]]:
    fields = _field_order_for_records(task_lc, records)
    rows: list[dict[str, Any]] = []
    count = max(1, int(record_count or len(records or []) or 1))
    flattened = [_flatten_record(rec) for rec in (records or [])]
    columns = ["Field"] + [_record_column_name(i) for i in range(count)]
    for field in fields:
        row: dict[str, Any] = {"Field": field}
        for i in range(count):
            flat = flattened[i] if i < len(flattened) else {}
            row[_record_column_name(i)] = flat.get(field, "")
        rows.append(row)
    return pd.DataFrame(rows, columns=columns), fields


def _legacy_editor_df_to_records(df: pd.DataFrame, fields: list[str]) -> list[dict]:
    out: list[dict] = []
    if df is None or df.empty:
        return out
    for _, row in df.iterrows():
        rec: dict[str, str] = {}
        for field in fields:
            value = row.get(field, "")
            if pd.isna(value):
                value = ""
            value_s = str(value).strip()
            if value_s:
                rec[field] = value_s
        if rec:
            out.append(rec)
    return out


def _resize_editor_df(df: pd.DataFrame, fields: list[str], record_count: int) -> pd.DataFrame:
    columns = ["Field"] + [_record_column_name(i) for i in range(max(1, record_count))]
    if df is not None and not df.empty and "Field" not in df.columns:
        flattened = [_flatten_record(rec) for rec in _legacy_editor_df_to_records(df, fields)]
        rows = []
        for field in fields:
            row: dict[str, Any] = {"Field": field}
            for i, column in enumerate(columns[1:]):
                flat = flattened[i] if i < len(flattened) else {}
                row[column] = flat.get(field, "")
            rows.append(row)
        return pd.DataFrame(rows, columns=columns)

    existing: dict[str, dict[str, Any]] = {}
    if df is not None and not df.empty and "Field" in df.columns:
        for _, row in df.iterrows():
            field = str(row.get("Field") or "").strip()
            if field:
                existing[field] = {column: row.get(column, "") for column in df.columns}

    rows = []
    for field in fields:
        old = existing.get(field, {})
        row = {"Field": field}
        for column in columns[1:]:
            row[column] = old.get(column, "")
        rows.append(row)
    return pd.DataFrame(rows, columns=columns)


def _editor_df_to_records(df: pd.DataFrame, fields: list[str]) -> list[dict]:
    out: list[dict] = []
    if df is None or df.empty or "Field" not in df.columns:
        return out
    field_rows = []
    known_fields = set(fields)
    for _, row in df.iterrows():
        field = str(row.get("Field") or "").strip()
        if field and field in known_fields:
            field_rows.append((field, row))

    record_columns = [column for column in df.columns if str(column).startswith("Record ")]
    for column in record_columns:
        rec: dict[str, str] = {}
        for field, row in field_rows:
            value = row.get(column, "")
            if pd.isna(value):
                value = ""
            value_s = str(value).strip()
            if value_s:
                rec[field] = value_s
        if rec:
            out.append(rec)
    return out


def _write_records_to_entry(entry: dict, task_lc: str, src_key: str | None, records: list[dict]) -> None:
    extracted_data = copy.deepcopy(entry.get("extracted_data") or {})
    out_key = src_key or task_lc
    current = extracted_data.get(out_key)
    if isinstance(current, dict):
        current = copy.deepcopy(current)
        current["records"] = records
        current["extracted_count"] = len(records)
        extracted_data[out_key] = current
    else:
        extracted_data[out_key] = {
            "records": records,
            "extracted_count": len(records),
        }

    counts = copy.deepcopy(entry.get("extraction_counts") or {})
    if isinstance(counts, dict):
        counts["total_extracted"] = len(records)
        by_task = counts.setdefault("by_task", {})
        if isinstance(by_task, dict):
            task_counts = by_task.setdefault(task_lc, {})
            if isinstance(task_counts, dict):
                task_counts["extracted"] = len(records)
    entry["extracted_data"] = extracted_data
    entry["extraction_counts"] = counts


def _source_id_for_metadata(md: dict) -> str:
    sid = str(md.get("source_id") or "").strip()
    if sid:
        return sid
    file_type = str(md.get("file_type") or "unknown").strip()
    chunk_id = md.get("chunk_id")
    return f"{file_type}_{chunk_id}" if chunk_id is not None else file_type


def _parse_header_metadata(header: str) -> dict[str, str]:
    md: dict[str, str] = {}
    body = header.strip().strip("[]")
    for part in body.split(";"):
        if "=" not in part:
            continue
        key, value = part.split("=", 1)
        md[key.strip()] = value.strip()
    return md


def _header_metadata_value(md: dict[str, str], *keys: str) -> str:
    wanted = {key.strip().lower() for key in keys}
    for key, value in md.items():
        if str(key).strip().lower() in wanted:
            return str(value or "").strip()
    return ""


def _dedupe_source_blocks(blocks: list[dict]) -> list[dict]:
    out: list[dict] = []
    seen: set[tuple[str, str]] = set()
    for block in blocks:
        key = (
            _normalize_source_id(block.get("source_id")),
            str(block.get("text") or "").strip(),
        )
        if key in seen:
            continue
        seen.add(key)
        out.append(block)
    return out


def _parse_enriched_text_blocks(entry: dict) -> list[dict]:
    text = str(entry.get("enriched_text") or "")
    metadata = [md for md in (entry.get("enriched_text_metadata") or []) if isinstance(md, dict)]
    meta_by_sid = {_source_id_for_metadata(md): md for md in metadata}

    pattern = re.compile(r"(?m)^\[(?P<header>source_id=[^\]]+)\]\s*$")
    matches = list(pattern.finditer(text))
    blocks: list[dict] = []
    for i, match in enumerate(matches):
        header = match.group("header")
        header_md = _parse_header_metadata(header)
        sid = header_md.get("source_id") or ""
        start = match.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        block_text = text[start:end].strip()
        md = {**(meta_by_sid.get(sid, {}) or {}), **header_md}
        blocks.append({"source_id": sid, "metadata": md, "text": block_text})

    if blocks:
        return _dedupe_source_blocks(blocks)

    chunk_pattern = re.compile(
        r"^Chunk\s+\d+\s+\((?P<header>[^)]*\bdata_source\s*=[^)]*)\):\s*$",
        re.IGNORECASE | re.MULTILINE,
    )
    matches = list(chunk_pattern.finditer(text))
    for i, match in enumerate(matches):
        header_md = _parse_header_metadata(match.group("header"))
        sid = _header_metadata_value(header_md, "source_id", "data_source")
        labels = _header_metadata_value(header_md, "predicted_label", "labels")
        normalized_header_md = dict(header_md)
        if sid:
            normalized_header_md["source_id"] = sid
        if labels:
            normalized_header_md["predicted_label"] = labels
        start = match.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        block_text = text[start:end].strip()
        md = {**(meta_by_sid.get(sid, {}) or {}), **normalized_header_md}
        blocks.append({"source_id": sid, "metadata": md, "text": block_text})

    if blocks:
        return _dedupe_source_blocks(blocks)

    if metadata:
        source_text = text.strip()
        if len(metadata) == 1:
            md = metadata[0]
            return [{"source_id": _source_id_for_metadata(md), "metadata": md, "text": source_text}]
        source_id = str(entry.get("chunk_id") or "combined_source")
        return [
            {
                "source_id": source_id,
                "metadata": {
                    "source_id": source_id,
                    "file_type": "combined",
                    "chunk_id": entry.get("chunk_id"),
                },
                "text": source_text,
            }
        ] if source_text else []

    return [{"source_id": str(entry.get("chunk_id") or "source"), "metadata": {}, "text": text.strip()}] if text.strip() else []


def _normalize_source_id(source_id: str | None) -> str:
    sid = str(source_id or "").strip()
    return sid


def _inject_source_text_css() -> None:
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


def _safe_component_key(*parts: Any) -> str:
    raw = "__".join(str(part) for part in parts if part is not None)
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", raw)[:512]


def _compact_table_whitespace(text: str) -> str:
    if not text:
        return ""
    cleaned = str(text).replace("\r\n", "\n").replace("\r", "\n")
    cleaned = re.sub(r"\n{2,}(?=\|)", "\n", cleaned)
    cleaned = re.sub(r"\n{4,}", "\n\n", cleaned)
    return cleaned.strip()


def _is_table_chunk(source_id: str | None, meta: dict | None, text: str) -> bool:
    meta = meta or {}
    file_type = str(meta.get("file_type") or "").strip().lower()
    if file_type == "table":
        return True
    if str(source_id or "").startswith("table_"):
        return True
    return "\n|" in str(text or "")


def _source_text_for_display(
    text: str,
    *,
    source_id: str | None = None,
    meta: dict | None = None,
) -> str:
    if _is_table_chunk(source_id, meta, text):
        return _compact_table_whitespace(text)
    return str(text or "")


def _render_plain_source_text(
    text: str,
    annotations: list[dict] | None = None,
) -> None:
    raw_text = str(text or "")
    highlights = []
    last_end = 0
    for annotation in _normalize_highlight_annotations(annotations or []):
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


def _render_highlighted_source_text(
    highlights: dict[str, list[dict]],
    *,
    source_id: str,
    text: str,
    key_scope: str,
    read_only: bool = False,
) -> None:
    if not text:
        st.caption("No source text available.")
        return

    source_id = _normalize_source_id(source_id)
    if not source_id:
        _render_plain_source_text(text)
        return

    existing = _normalize_highlight_annotations(highlights.get(source_id, []))
    if read_only:
        _render_plain_source_text(text, existing)
        return

    if text_highlighter is None:
        st.caption("Highlight component is not installed; showing plain source text.")
        _render_plain_source_text(text, existing)
        return

    result = text_highlighter(
        text=text,
        labels=[(HIGHLIGHT_TAG, HIGHLIGHT_COLOR)],
        selected_label=HIGHLIGHT_TAG,
        annotations=existing,
        show_label_selector=False,
        key=_safe_component_key("extract_th", source_id, key_scope),
    )
    if result is not None:
        highlights[source_id] = _normalize_highlight_annotations(result)


def _source_ids_from_data_provenance(dp: Any) -> list[str]:
    out: list[str] = []

    def add(value: Any) -> None:
        sid = _normalize_source_id(str(value or ""))
        if sid and sid not in out:
            out.append(sid)

    if isinstance(dp, dict):
        for value in dp.values():
            for sid in _source_ids_from_data_provenance(value):
                add(sid)
        return out
    if isinstance(dp, list):
        for item in dp:
            for sid in _source_ids_from_data_provenance(item):
                add(sid)
        return out
    if dp is None:
        return out

    text = str(dp).strip()
    if not text:
        return out
    for match in re.finditer(r"source_id\s*=\s*([A-Za-z0-9_.-]+)", text):
        add(match.group(1))
    for chunk in re.split(r"[;,]", text):
        token = chunk.strip()
        if not token:
            continue
        if ":" in token:
            token = token.split(":", 1)[1].strip()
        if re.search(r"^(main_paper|supplementary_material|table|figure|study_level|unknown)_", token):
            add(token)
    return out


def _record_sources(record: dict) -> list[str]:
    lower = {str(k).strip().lower(): v for k, v in (record or {}).items()}
    return _source_ids_from_data_provenance(lower.get("data_provenance"))


def _source_matches(block_sid: str, wanted_sid: str) -> bool:
    block = _normalize_source_id(block_sid)
    wanted = _normalize_source_id(wanted_sid)
    return bool(block and wanted and block == wanted)


def _render_block(
    block: dict,
    *,
    key_prefix: str,
    highlights: dict[str, list[dict]],
    read_only: bool,
    interactive: bool = True,
) -> None:
    md = block.get("metadata") or {}
    sid = block.get("source_id") or "(missing source_id)"
    text = _source_text_for_display(block.get("text") or "", source_id=sid, meta=md)
    file_type = str(md.get("file_type") or "").strip()
    chunk_id = str(md.get("chunk_id") or "").strip()
    label = f"{file_type} {chunk_id}".strip() or str(sid)
    tags = []
    for tag in (
        md.get("section_label"),
        md.get("predicted_label"),
    ):
        tag_text = str(tag or "").strip()
        if tag_text:
            tags.append(f"<code>{html.escape(tag_text)}</code>")
    st.markdown(
        "<div class='source-chunk-heading'>"
        f"<b>{html.escape(label)}</b> "
        f"{' '.join(tags)}"
        "</div>",
        unsafe_allow_html=True,
    )
    _render_highlighted_source_text(
        highlights,
        source_id=str(sid),
        text=text,
        key_scope=f"{key_prefix}_{abs(hash(text))}",
        read_only=read_only or not interactive,
    )


def _render_grouped_blocks(
    blocks: list[dict],
    *,
    key_prefix: str,
    highlights: dict[str, list[dict]],
    read_only: bool,
    interactive: bool = True,
) -> None:
    if not blocks:
        st.caption("No source text available for this entry.")
        return
    for i, block in enumerate(blocks):
        md = block.get("metadata") or {}
        label = str(md.get("predicted_label") or md.get("file_type") or "unknown").strip() or "unknown"
        _render_block(
            block,
            key_prefix=f"{key_prefix}_{label}_{i}",
            highlights=highlights,
            read_only=read_only,
            interactive=interactive,
        )


def _render_source_review(
    entry: dict,
    records: list[dict],
    *,
    entry_key: str,
    highlights: dict[str, list[dict]],
    read_only: bool,
) -> None:
    st.markdown("### Source Evidence")
    blocks = _parse_enriched_text_blocks(entry)
    if not blocks:
        st.caption("No enriched source text available.")
        return

    source_to_records: dict[str, list[int]] = defaultdict(list)
    records_without_sources = 0
    for i, rec in enumerate(records or [], start=1):
        sources = _record_sources(rec)
        if not sources:
            records_without_sources += 1
        for sid in sources:
            source_to_records[_normalize_source_id(sid)].append(i)

    if source_to_records:
        st.caption("Record-level data_provenance found. Sources are grouped by the records that cite them.")
        used_sids = set(source_to_records)
        for sid in sorted(source_to_records, key=lambda x: (-len(source_to_records[x]), x)):
            matched = [block for block in blocks if _source_matches(block.get("source_id", ""), sid)]
            if matched:
                _render_grouped_blocks(
                    matched,
                    key_prefix=f"{entry_key}_{sid}",
                    highlights=highlights,
                    read_only=read_only,
                )
            else:
                st.caption(f"No matching source block found for `{sid}`.")
        unused = [block for block in blocks if _normalize_source_id(block.get("source_id")) not in used_sids]
        if unused:
            with st.expander("Source chunks not cited by record-level provenance", expanded=False):
                _render_grouped_blocks(
                    unused,
                    key_prefix=f"{entry_key}_unused",
                    highlights=highlights,
                    read_only=read_only,
                    interactive=False,
                )
        if records_without_sources:
            st.caption(f"{records_without_sources} record(s) have no data_provenance field.")
        return

    st.caption("No record-level provenance was found for this entry; showing the entry-level source context.")
    _render_grouped_blocks(
        blocks,
        key_prefix=f"{entry_key}_all",
        highlights=highlights,
        read_only=read_only,
    )


def _render_side_by_side_review(
    entry: dict,
    task_lc: str,
    global_idx: int,
    *,
    study_id: str,
    highlights: dict[str, list[dict]],
    read_only: bool,
) -> list[dict]:
    source_col, record_col = st.columns([0.95, 1.15], gap="large")
    entry_key = f"{study_id}_{task_lc}_{global_idx}"

    with record_col:
        with st.container(height=REVIEW_PANEL_HEIGHT, border=False):
            st.markdown("### Extracted Record")
            records_now = _render_editor(entry, task_lc, global_idx, read_only=read_only)

    with source_col:
        with st.container(height=REVIEW_PANEL_HEIGHT, border=False):
            _render_source_review(
                entry,
                records_now,
                entry_key=entry_key,
                highlights=highlights,
                read_only=read_only,
            )

    return records_now


def _entry_label(item: tuple[int, dict]) -> str:
    _, entry = item
    task = _canonical_task(_task_from_entry(entry))
    _, records = _records_for_entry(entry, task)
    mode = _entry_mode(entry)
    chunk_id = str(entry.get("chunk_id") or entry.get("entry_id") or "(no chunk_id)")
    if mode:
        return f"{chunk_id} | {mode} | {len(records)} record(s)"
    return f"{chunk_id} | {len(records)} record(s)"


def _render_entry_header(entry: dict, task_lc: str, local_idx: int, total: int) -> None:
    st.subheader(f"{TASK_LABELS.get(task_lc, task_lc.title())} Entry {local_idx + 1}/{total}")
    mode = _entry_mode(entry)
    task_payload = _task_payload_for_entry(entry, task_lc)
    cols = st.columns(4)
    cols[0].metric("Records", len(_records_for_entry(entry, task_lc)[1]))
    counts = entry.get("extraction_counts") or {}
    requested = counts.get("total_requested") if isinstance(counts, dict) else None
    cols[1].metric("Requested", requested if requested is not None else "-")
    cols[2].metric("Mode", mode or "-")
    cols[3].metric("Chunk", str(entry.get("chunk_id") or "-"))

    with st.expander("Entry metadata", expanded=False):
        st.json(
            {
                "entry_key": _entry_key(entry),
                "file_type": entry.get("file_type"),
                "predicted_label": entry.get("predicted_label"),
                "mode": mode,
                "artifact_key": entry.get("_artifact_key"),
                "artifact_mode": entry.get("_artifact_mode"),
                "extraction_counts": entry.get("extraction_counts"),
                "requested_records": task_payload.get("requested_records"),
            }
        )


def _ensure_custom_field_state() -> None:
    if "custom_field_options" not in st.session_state:
        st.session_state.custom_field_options = {task: set() for task in TASK_ORDER}
    for task in TASK_ORDER:
        st.session_state.custom_field_options.setdefault(task, set())


def _render_custom_field_controls(task_lc: str, entry_key: str) -> None:
    with st.expander("Add a custom editor field", expanded=False):
        new_field = st.text_input(
            "Field name",
            key=f"new_field_{entry_key}_{task_lc}",
            placeholder="e.g. reviewer_note",
        )
        if st.button("Add field", key=f"add_field_{entry_key}_{task_lc}"):
            field = str(new_field or "").strip().lower()
            if field:
                st.session_state.custom_field_options.setdefault(task_lc, set()).add(field)
                st.rerun()
            st.warning("Please provide a non-empty field name.")


def _render_editor(entry: dict, task_lc: str, global_idx: int, *, read_only: bool) -> list[dict]:
    src_key, records = _records_for_entry(entry, task_lc)
    entry_key = "__".join(str(x) for x in _entry_key(entry))
    _render_custom_field_controls(task_lc, entry_key)

    initial_record_count = max(1, len(records or []))
    draft_key = f"draft_df_{entry_key}_{global_idx}_{task_lc}"
    fields_key = f"draft_fields_{entry_key}_{global_idx}_{task_lc}"
    record_count_key = f"draft_record_count_{entry_key}_{global_idx}_{task_lc}"
    editor_version_key = f"draft_editor_version_{entry_key}_{global_idx}_{task_lc}"
    if editor_version_key not in st.session_state:
        st.session_state[editor_version_key] = 0
    if record_count_key not in st.session_state:
        st.session_state[record_count_key] = initial_record_count
    record_count = int(
        st.number_input(
            "Record columns",
            min_value=1,
            max_value=max(50, initial_record_count),
            step=1,
            disabled=read_only,
            key=record_count_key,
            width=160,
        )
    )

    df_init, fields = _records_to_editor_df(records, task_lc, record_count)
    if draft_key not in st.session_state or st.session_state.get(fields_key) != fields:
        st.session_state[draft_key] = df_init
        st.session_state[fields_key] = fields
        st.session_state[editor_version_key] += 1

    current_df = _resize_editor_df(st.session_state[draft_key], fields, record_count)
    st.session_state[draft_key] = current_df
    field_width = _field_column_width(fields)
    column_config = {"Field": stcol.TextColumn("Field", width=field_width, pinned=True)}
    for i in range(record_count):
        column = _record_column_name(i)
        column_config[column] = stcol.TextColumn(
            column,
            width=_record_column_width(
                current_df,
                column,
                record_count=record_count,
                field_width=field_width,
            ),
        )

    editor_instance = st.session_state[editor_version_key]
    editor_key = _safe_component_key(
        "editor",
        entry_key,
        global_idx,
        task_lc,
        record_count,
        editor_instance,
    )
    form_key = _safe_component_key(
        "edit_form",
        entry_key,
        global_idx,
        task_lc,
        record_count,
        editor_instance,
    )
    with st.form(form_key, clear_on_submit=False):
        edited_df = st.data_editor(
            current_df,
            num_rows="fixed",
            hide_index=True,
            height=_editor_grid_height(len(fields)),
            row_height=36,
            use_container_width=True,
            column_config=column_config,
            disabled=True if read_only else ["Field"],
            key=editor_key,
        )
        submitted = st.form_submit_button(
            "Commit edits for this entry",
            type="primary",
            disabled=read_only,
        )

    baseline_records = _editor_df_to_records(current_df, fields)
    new_records = _editor_df_to_records(edited_df, fields)
    if submitted and not read_only:
        _write_records_to_entry(entry, task_lc, src_key, new_records)
        st.session_state[draft_key] = _records_to_editor_df(
            new_records, task_lc, record_count
        )[0]
        st.session_state[editor_version_key] += 1
        st.success("Edits committed in session.")
        return new_records

    if new_records != baseline_records:
        # Preserve an unsaved draft outside the widget, then mount a fresh
        # editor instance. Feeding the changed dataframe back into the same
        # keyed st.data_editor can make Streamlit apply edits one rerun late.
        st.session_state[draft_key] = edited_df.copy()
        st.session_state[editor_version_key] += 1
        st.rerun()

    return records


def _entries_by_task(chunks: list[dict]) -> dict[str, list[tuple[int, dict]]]:
    grouped: dict[str, list[tuple[int, dict]]] = defaultdict(list)
    for i, entry in enumerate(chunks or []):
        task = _canonical_task(_task_from_entry(entry))
        if task in TASK_LABELS:
            grouped[task].append((i, entry))
    return grouped


def _save_all(
    study_id: str,
    chunks: list[dict],
    *,
    review_edit: bool,
    highlights: dict[str, list[dict]],
) -> None:
    get_study_paths(study_id).extraction_ground_truth_dir.mkdir(parents=True, exist_ok=True)
    path = _ground_truth_path(study_id)
    existing = [entry for entry in _load_gt(study_id) if _is_supported_task_entry(entry)]
    index = {_entry_key(entry): i for i, entry in enumerate(existing)}

    appended = 0
    updated = 0
    for entry in chunks or []:
        if not _is_supported_task_entry(entry):
            continue
        key = _entry_key(entry)
        if key in index:
            if review_edit:
                existing[index[key]] = entry
                updated += 1
        else:
            existing.append(entry)
            index[key] = len(existing) - 1
            appended += 1

    _save_json(existing, path)
    highlight_path = _save_highlights(study_id, highlights)

    if review_edit:
        st.success(
            f"Saved ground truth to `{path}` and highlights to `{highlight_path}`. "
            f"Updated {updated}, appended {appended}."
        )
    else:
        st.success(
            f"Saved ground truth to `{path}` and highlights to `{highlight_path}`. "
            f"Appended {appended}; existing entries were preserved."
        )


st.set_page_config(page_title="Extraction Annotation", layout="wide")
st.title("Extraction Annotation Tool")
_inject_source_text_css()
_ensure_custom_field_state()

study_id = st.selectbox("Select study", SELECTED_STUDY_IDS)
prediction_cache_token = _prediction_cache_token(study_id)
ground_truth_cache_token = _ground_truth_cache_token(study_id)

if st.button("Reload extraction files"):
    _load_predictions.clear()
    _load_artifact_entries.clear()
    st.session_state.pop(f"entries_{study_id}", None)
    st.session_state.pop(f"entries_mode_{study_id}", None)
    st.rerun()

highlight_session_key = f"highlights_{study_id}"
if highlight_session_key not in st.session_state:
    st.session_state[highlight_session_key] = _load_highlights(study_id)
highlights = st.session_state[highlight_session_key]
preds = _load_predictions(study_id, prediction_cache_token)
gt = _load_gt(study_id)
filtered = _filter_new_entries(preds, gt)
gt_path = _ground_truth_path(study_id)

review_mode = False
edit_unlocked = True
if gt:
    st.info(f"Ground truth exists for this study: `{gt_path}`")
    review_mode = st.checkbox(
        "Open existing ground truth for review",
        value=False,
        help="Review mode loads existing annotations. Editing is locked unless you unlock it.",
    )
    if review_mode:
        edit_unlocked = st.checkbox(
            "Unlock editing in review mode",
            value=False,
            help="Leave locked if you only want to inspect existing annotations.",
        )

if review_mode:
    chunks_to_use = gt
    mode_label = "review_unlocked" if edit_unlocked else "review_locked"
else:
    chunks_to_use = filtered
    mode_label = "annotate"

source_token = ground_truth_cache_token if review_mode else prediction_cache_token
state_source_key = (mode_label, source_token)
session_key = f"entries_{study_id}"
mode_key = f"entries_mode_{study_id}"
if session_key not in st.session_state or st.session_state.get(mode_key) != state_source_key:
    st.session_state[session_key] = copy.deepcopy(chunks_to_use)
    st.session_state[mode_key] = state_source_key

chunks = st.session_state[session_key]
if not chunks:
    if preds and gt and not review_mode:
        st.warning("No new task-level entries to annotate. Enable review mode to inspect existing ground truth.")
    else:
        st.warning("No extraction entries found for this study.")
    st.stop()

if not review_mode:
    st.caption(f"Loaded {len(preds)} predicted entry/entries; {len(preds) - len(filtered)} already in GT; {len(filtered)} ready to annotate.")
elif edit_unlocked:
    st.warning("Review mode editing is unlocked. Saving will overwrite matching GT entries.")
else:
    st.info("Review mode is read-only. Unlock editing before making changes.")

entries_by_task = _entries_by_task(chunks)
st.caption(
    "Entries by task: "
    + ", ".join(f"{TASK_LABELS[task]}={len(entries_by_task.get(task, []))}" for task in TASK_ORDER)
)
active_tasks = [task for task in TASK_ORDER if entries_by_task.get(task)]
tab_labels = [f"{TASK_LABELS[task]} ({len(entries_by_task[task])})" for task in active_tasks] + ["Save / JSON"]
tabs = st.tabs(tab_labels)

read_only = review_mode and not edit_unlocked

for tab, task_lc in zip(tabs[: len(active_tasks)], active_tasks):
    with tab:
        entries = entries_by_task[task_lc]
        if len(entries) == 1:
            selected = entries[0]
            st.caption(_entry_label(selected))
        else:
            selected = st.selectbox(
                "Select extraction entry",
                options=entries,
                format_func=_entry_label,
                key=f"entry_select_{study_id}_{task_lc}_{mode_label}",
            )
        global_idx, entry = selected
        local_idx = entries.index(selected)
        _render_entry_header(entry, task_lc, local_idx, len(entries))
        _render_side_by_side_review(
            entry,
            task_lc,
            global_idx,
            study_id=study_id,
            highlights=highlights,
            read_only=read_only,
        )

with tabs[-1]:
    st.markdown("### Save")
    if read_only:
        st.info("Review mode is read-only. Unlock editing to save changes.")
    if st.button("Save all annotations to ground truth", disabled=read_only):
        _save_all(
            study_id,
            chunks,
            review_edit=bool(review_mode and edit_unlocked),
            highlights=highlights,
        )

    with st.expander("All annotations JSON", expanded=False):
        st.json(chunks)
