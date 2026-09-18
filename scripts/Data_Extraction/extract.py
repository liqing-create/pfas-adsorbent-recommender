# scripts/extract.py
import glob
import json
import logging
import os
import re
import time
import traceback
from pathlib import Path
from typing import Any, Callable
from collections import defaultdict

try:
    from .workflow_config import (
        FIGURE_TABLE_INFO_CSV as CONFIG_FIGURE_TABLE_INFO_CSV,
        OVERWRITE_EXISTING_OUTPUTS,
        PROMPTS_DIR as CONFIG_PROMPTS_DIR,
        StudyPaths,
        add_project_import_paths,
        check_stage_input,
        get_study_paths,
        iter_active_study_paths,
    )
except ImportError:  # Supports ``python extract.py`` from this folder.
    from workflow_config import (
        FIGURE_TABLE_INFO_CSV as CONFIG_FIGURE_TABLE_INFO_CSV,
        OVERWRITE_EXISTING_OUTPUTS,
        PROMPTS_DIR as CONFIG_PROMPTS_DIR,
        StudyPaths,
        add_project_import_paths,
        check_stage_input,
        get_study_paths,
        iter_active_study_paths,
    )

add_project_import_paths()

from chains.extract_chain import create_extraction_chains
from chains.llm_config import get_llm
from chains.llm_usage import predict_with_usage
from schemas import ADSORBENT_FIELDS, EXPERIMENT_FIELDS, PERFORMANCE_FIELDS, REVIEW_FIELDS
try:
    from .helpers import extract_by_chunk as chunk_extract
    from .helpers.figure_caption_context import (
        load_caption_index_for_study,
        select_figure_captions_for_text,
    )
    from .helpers.post_process_extraction import (
        has_target_performance_metric,
        process_chain_artifact_payload,
    )
except ImportError:
    from helpers import extract_by_chunk as chunk_extract
    from helpers.figure_caption_context import load_caption_index_for_study, select_figure_captions_for_text
    from helpers.post_process_extraction import (
        has_target_performance_metric,
        process_chain_artifact_payload,
    )

logging.basicConfig(level=logging.WARNING)
for lib in ("httpx", "openai", "httpcore", "urllib3"):
    logging.getLogger(lib).setLevel(logging.WARNING)


def env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "y", "on"}


def env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return int(raw.strip())
    except ValueError:
        return default


CLASSIFIED_DIR = ""
ENUMERATION_DIR = ""
EXTRACTION_OUTPUT_DIR = ""
ENUMERATION_ARTIFACT_ROOT = ""
PROMPTS_DIR = str(CONFIG_PROMPTS_DIR)
ERROR_LOG_ROOT = ""
ARTIFACT_ROOT = ""
FIGURE_TABLE_INFO_CSV = str(CONFIG_FIGURE_TABLE_INFO_CSV)


def _configure_study_paths(paths: StudyPaths, *, output_dir: Path | None = None) -> None:
    """Point legacy helper functions at one config-resolved study group."""
    global CLASSIFIED_DIR, ENUMERATION_DIR, EXTRACTION_OUTPUT_DIR
    global ENUMERATION_ARTIFACT_ROOT, ERROR_LOG_ROOT, ARTIFACT_ROOT
    CLASSIFIED_DIR = str(paths.classification_dir)
    ENUMERATION_DIR = str(paths.enumeration_dir)
    run_output_dir = Path(output_dir) if output_dir is not None else paths.extraction_dir
    EXTRACTION_OUTPUT_DIR = str(run_output_dir)
    ENUMERATION_ARTIFACT_ROOT = str(paths.enumeration_dir / "_chain_artifacts")
    ERROR_LOG_ROOT = str(run_output_dir / "_error_logs")
    ARTIFACT_ROOT = str(run_output_dir / "_chain_artifacts")

LLM_PROVIDER = "openai"  # "openai" or "together"
OPENAI_MODEL = "gpt-4.1-2025-04-14"
TOGETHER_MODEL = "meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8"
LLM_MODELS = {
    "openai": OPENAI_MODEL,
    "together": TOGETHER_MODEL,
}

if LLM_PROVIDER not in LLM_MODELS:
    raise ValueError(
        f"Unsupported LLM provider: {LLM_PROVIDER}. "
        f"Configured providers: {sorted(LLM_MODELS)}"
    )

LLM_MODEL_NAME = LLM_MODELS[LLM_PROVIDER]

# Task toggles for the new workflow.
ENABLE_ADSORBENT = env_bool("ENABLE_ADSORBENT", True)      # study-level adsorbent extraction
ENABLE_EXPERIMENT = env_bool("ENABLE_EXPERIMENT", False)     # study-level experiment extraction
ENABLE_PERFORMANCE = env_bool("ENABLE_PERFORMANCE", True)  # hybrid performance extraction
FORCE_LLM_RERUN = env_bool("FORCE_LLM_RERUN", OVERWRITE_EXISTING_OUTPUTS)

BATCH_SIZE = 30
PERFORMANCE_HEAVY_TABLE_RECORD_THRESHOLD = env_int(
    "PERFORMANCE_HEAVY_TABLE_RECORD_THRESHOLD",
    BATCH_SIZE,
)
PERFORMANCE_HEAVY_TABLE_CHAR_THRESHOLD = env_int(
    "PERFORMANCE_HEAVY_TABLE_CHAR_THRESHOLD",
    0,
)
# Route performance extraction to sources cited by enumeration records.  Keep
# uncited non-table chunks as lightweight context, but drop uncited tables.
# Any missing or invalid Data_Source mapping falls back to the full group.
PERFORMANCE_SOURCE_AWARE_ROUTING = True
PERFORMANCE_KEEP_UNCITED_NON_TABLE_CONTEXT = True
ADSORBENT_SCOPE_KEYS = ("in_scope", "out_of_scope")
ADSORBENT_LEXICON_FIELDS = [
    "Adsorbent_id",
    "Name_Abbreviation",
    "Name_Full",
    "Name_Commercial",
    "Adsorbent_category",
    "Adsorbent_subcategory",
]

ADSORBENT_IDENTIFIER_OUTPUT_FIELDS = {
    "Adsorbent_id": "adsorbent_id",
    "Name_Abbreviation": "name_abbreviation",
    "Name_Full": "name_full",
    "Name_Commercial": "name_commercial",
    "Adsorbent_category": "adsorbent_category",
    "Adsorbent_subcategory": "adsorbent_subcategory",
}
EMPTY_LEXICON_VALUES = {"", "na", "n/a", "not found", "none", "null", "unknown"}

def _split_labels(val: str) -> list[str]:
    return [p.strip().lower() for p in re.split(r"[,|/;]+", val or "") if p.strip()]


def _is_irrelevant(ch: dict) -> bool:
    if str(ch.get("vector_DB_label", "")).strip().lower() == "irrelevant":
        return True
    if "irrelevant" in _split_labels(str(ch.get("predicted_label", ""))):
        return True
    ann = ch.get("annotation")
    return isinstance(ann, list) and any(str(a).strip().lower() == "irrelevant" for a in ann)


def _is_labeled_chunk(ch: dict, label: str) -> bool:
    labels = _split_labels(str(ch.get("predicted_label", "")))
    ann = ch.get("annotation")
    if isinstance(ann, list):
        labels.extend(str(a).strip().lower() for a in ann)
    return label in labels


def _source_id_for(ent: dict) -> str:
    file_type = str(ent.get("file_type") or "unknown").strip()
    chunk_id = ent.get("chunk_id")
    return f"{file_type}_{chunk_id}" if chunk_id is not None else file_type


def _format_study_chunks(entries: list[dict]) -> str:
    lines: list[str] = []
    n = 0
    for ent in entries:
        text = (ent.get("enriched_text") or ent.get("text") or "").strip()
        if not text:
            continue
        if re.match(r"^Chunk\s+\d+\s+\(Data_Source=", text):
            lines.append(text)
            n += len(re.findall(r"(?m)^Chunk\s+\d+\s+\(Data_Source=", text))
            continue
        n += 1
        labels = ent.get("predicted_label") or ent.get("annotation") or ""
        if isinstance(labels, list):
            labels = ", ".join(str(item) for item in labels)
        lines.append(f"Chunk {n} (Data_Source={_source_id_for(ent)}; labels={labels}):\n{text}")
    return "\n\n".join(lines).rstrip()


def _build_enriched_text_metadata(entries: list[dict]) -> list[dict]:
    metadata: list[dict] = []
    for idx, ent in enumerate(entries or [], start=1):
        existing = ent.get("enriched_text_metadata")
        if isinstance(existing, list) and existing:
            metadata.extend(item for item in existing if isinstance(item, dict))
            continue
        metadata.append(
            {
                "prompt_chunk_number": idx,
                "study_folder": ent.get("study_folder"),
                "file_type": ent.get("file_type"),
                "chunk_id": ent.get("chunk_id"),
                "source_id": _source_id_for(ent),
                "predicted_label": ent.get("predicted_label"),
            }
        )
    return metadata


def _load_json(path: str):
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def _write_missing_log(study_id: str, missing_path: str) -> None:
    os.makedirs(EXTRACTION_OUTPUT_DIR, exist_ok=True)
    log_path = os.path.join(EXTRACTION_OUTPUT_DIR, "missing_files.log")
    with open(log_path, "a", encoding="utf-8") as logf:
        logf.write(f"{study_id}: missing {missing_path}\n")


def _load_classified_entries(study_id: str) -> list[dict] | None:
    in_file = os.path.join(CLASSIFIED_DIR, f"{study_id}_classified.json")
    if not os.path.exists(in_file):
        _write_missing_log(study_id, in_file)
        print(f"[{study_id}] skipped study-level tasks: classified input not found")
        return None
    return _load_json(in_file)


def _enumeration_artifact_path(study_id: str, task: str) -> str:
    return os.path.join(
        ENUMERATION_ARTIFACT_ROOT,
        study_id,
        f"{study_id}_{task}.json",
    )


def _load_enumeration_task_artifact(study_id: str, task: str) -> dict:
    path = _enumeration_artifact_path(study_id, task)
    if os.path.exists(path):
        return _load_json(path)
    print(f"[{study_id}] warning: no {task} enumeration artifact found at {path}")
    return {}


def _load_enumeration_vocab_payload(study_id: str) -> dict:
    adsorbent_payload = _load_enumeration_task_artifact(study_id, "adsorbent")
    water_type_payload = _load_enumeration_task_artifact(study_id, "water_type")
    return {
        "adsorbents": _normalize_adsorbents(adsorbent_payload.get("adsorbents")),
        "water_types": water_type_payload.get("water_types") or [],
        "artifact_paths": {
            "adsorbent": _enumeration_artifact_path(study_id, "adsorbent"),
            "water_type": _enumeration_artifact_path(study_id, "water_type"),
        },
    }

def _empty_adsorbents() -> dict[str, list[dict]]:
    return {key: [] for key in ADSORBENT_SCOPE_KEYS}


def _validate_adsorbent_records(records: Any, path: str) -> list[dict]:
    if records is None:
        return []
    if not isinstance(records, list):
        raise ValueError(f"{path} must be a list")
    for idx, item in enumerate(records):
        if not isinstance(item, dict):
            raise ValueError(f"{path}[{idx}] must be a JSON object")
    return records


def _normalize_adsorbents(raw: Any) -> dict[str, list[dict]]:
    if raw is None:
        return _empty_adsorbents()
    if isinstance(raw, list):
        return {"in_scope": _validate_adsorbent_records(raw, "adsorbents"), "out_of_scope": []}
    if isinstance(raw, dict):
        return {
            key: _validate_adsorbent_records(raw.get(key), f"adsorbents.{key}")
            for key in ADSORBENT_SCOPE_KEYS
        }
    raise ValueError("adsorbents must be an object with in_scope and out_of_scope lists")

def _render_adsorbent_lexicon(
    payload: dict,
    allowed_adsorbent_ids: set[str] | None = None,
) -> str:
    allowed_norm = {
        str(value).strip().casefold()
        for value in (allowed_adsorbent_ids or set())
        if str(value).strip()
    }
    filter_adsorbents = allowed_adsorbent_ids is not None
    groups = _normalize_adsorbents((payload or {}).get("adsorbents"))
    sections: list[str] = []
    for key, title in (
        ("in_scope", "In-scope adsorbents"),
        ("out_of_scope", "Out-of-scope adsorbents (do not extract records for these)"),
    ):
        lines: list[str] = []
        for item in groups[key]:
            if not isinstance(item, dict):
                continue
            adsorbent_id = str(item.get("Adsorbent_id") or "").strip()
            if key == "in_scope" and filter_adsorbents and adsorbent_id.casefold() not in allowed_norm:
                continue
            parts = []
            for field in ADSORBENT_LEXICON_FIELDS:
                value = str(item.get(field) or "").strip()
                if value:
                    parts.append(f"{field}: {value}")
            if parts:
                lines.append("- " + "; ".join(parts))
        body = "\n".join(lines) if lines else "(no entries)"
        sections.append(f"{title}:\n{body}")
    return "\n\n".join(sections)


def _clean_lexicon_value(value: Any) -> str:
    text = str(value or "").strip()
    return "" if text.casefold() in EMPTY_LEXICON_VALUES else text


def _adsorbent_match_keys(value: str) -> list[str]:
    keys: list[str] = []
    for candidate in (value, chunk_extract.normalize_id(value)):
        key = str(candidate or "").strip().casefold()
        if key and key not in keys:
            keys.append(key)
    return keys


def _build_adsorbent_identifier_index(payload: dict) -> dict[str, dict[str, str]]:
    index: dict[str, dict[str, str]] = {}
    groups = _normalize_adsorbents((payload or {}).get("adsorbents"))
    for item in groups["in_scope"]:
        row = {
            dst: _clean_lexicon_value(item.get(src))
            for src, dst in ADSORBENT_IDENTIFIER_OUTPUT_FIELDS.items()
        }
        row = {key: value for key, value in row.items() if value}
        adsorbent_id = row.get("adsorbent_id")
        if not adsorbent_id:
            continue
        for key in _adsorbent_match_keys(adsorbent_id):
            index.setdefault(key, row)
    return index


def _adsorbent_id_from_record(record: dict) -> str:
    for key in ("adsorbent_id", "Adsorbent_id"):
        value = _clean_lexicon_value(record.get(key))
        if value:
            return value
    extras = record.get("__extras__")
    if isinstance(extras, dict):
        for key in ("Adsorbent_id", "adsorbent_id"):
            value = _clean_lexicon_value(extras.get(key))
            if value:
                return value
    return ""


def _drop_identifier_extras(record: dict) -> None:
    extras = record.get("__extras__")
    if not isinstance(extras, dict):
        return
    for src, dst in ADSORBENT_IDENTIFIER_OUTPUT_FIELDS.items():
        extras.pop(src, None)
        extras.pop(dst, None)
    if not extras:
        record.pop("__extras__", None)


def _inject_adsorbent_identifier_fields(records: list[dict], payload: dict) -> list[dict]:
    index = _build_adsorbent_identifier_index(payload)
    for rec in records or []:
        if not isinstance(rec, dict):
            continue
        raw_id = _adsorbent_id_from_record(rec)
        if raw_id and not rec.get("adsorbent_id"):
            rec["adsorbent_id"] = raw_id

        match = next((index.get(key) for key in _adsorbent_match_keys(raw_id) if index.get(key)), None)
        if match:
            rec["adsorbent_id"] = match["adsorbent_id"]
            for key, value in match.items():
                if key != "adsorbent_id" and value and not _clean_lexicon_value(rec.get(key)):
                    rec[key] = value
        _drop_identifier_extras(rec)
    return records

def _performance_parse_fields() -> dict:
    return dict(PERFORMANCE_FIELDS)


def _experiment_parse_fields() -> dict:
    fields = dict(EXPERIMENT_FIELDS)
    fields["Experiment_id"] = str
    fields["Test_mode"] = str
    fields["Parametric_impact"] = str
    return fields


def _adsorbent_ids_from_performance_ids(performance_ids: list[str]) -> set[str]:
    adsorbent_ids: set[str] = set()
    for performance_id in performance_ids:
        parts = str(performance_id or "").split("|")
        if len(parts) != 4:
            continue
        adsorbent_id = parts[1].strip()
        if adsorbent_id and adsorbent_id.upper() != "NA":
            adsorbent_ids.add(adsorbent_id)
    return adsorbent_ids


def _is_heavy_performance_table(file_type: Any, text: str | None, value_ids: list[str]) -> bool:
    if str(file_type or "").strip().lower() != "table":
        return False
    if len(value_ids or []) > PERFORMANCE_HEAVY_TABLE_RECORD_THRESHOLD:
        return True
    return (
        PERFORMANCE_HEAVY_TABLE_CHAR_THRESHOLD > 0
        and len(text or "") > PERFORMANCE_HEAVY_TABLE_CHAR_THRESHOLD
    )


def _render_water_type_lexicon(payload: dict) -> str:
    lines: list[str] = []
    for item in payload.get("water_types") or []:
        if isinstance(item, dict):
            full_name = str(item.get("Full_name") or item.get("full_name") or "").strip()
            abbr = str(item.get("Abbreviation") or item.get("abbreviation") or "").strip()
            klass = str(item.get("Class") or item.get("class") or "").strip()
            parts = []
            if full_name:
                parts.append(f"Full_name={full_name}")
            if abbr:
                parts.append(f"Abbreviation={abbr}")
            if klass:
                parts.append(f"Class={klass}")
            if parts:
                lines.append("- " + "; ".join(parts))
        elif isinstance(item, str) and item.strip():
            lines.append("- " + item.strip())
    return "\n".join(lines) if lines else "(no entries)"


def _usage_metadata(usage) -> dict:
    return {
        "provider": getattr(usage, "provider", LLM_PROVIDER),
        "model_name": getattr(usage, "model_name", LLM_MODEL_NAME),
        "duration_s": round(float(getattr(usage, "elapsed_s", 0.0) or 0.0), 3),
        "prompt_tokens": getattr(usage, "prompt_tokens", None),
        "completion_tokens": getattr(usage, "completion_tokens", None),
        "total_tokens": getattr(usage, "total_tokens", None),
        "total_cost_usd": getattr(usage, "total_cost_usd", None),
    }


def dump_error_context(
    study_id: str,
    chunk_id: str,
    prompt_text: str,
    raw_text: str,
    warnings: list[str],
    exc: Exception,
    timings: list[float] | None = None,
) -> None:
    os.makedirs(ERROR_LOG_ROOT, exist_ok=True)
    path = os.path.join(ERROR_LOG_ROOT, f"{study_id}_{chunk_id}_error.txt")
    with open(path, "a", encoding="utf-8") as fh:
        fh.write("\n" + "=" * 80 + "\n")
        fh.write(f"{study_id} | {chunk_id}\n")
        fh.write(f"{type(exc).__name__}: {exc}\n\n")
        if warnings:
            fh.write("--- warnings ---\n")
            fh.write("\n".join(warnings) + "\n\n")
        fh.write("--- prompt ---\n")
        fh.write((prompt_text or "(prompt missing)").rstrip() + "\n\n")
        fh.write("--- raw LLM output ---\n")
        fh.write((raw_text or "(raw output missing)").rstrip() + "\n\n")
        if timings:
            fh.write("--- extraction timings (s) ---\n")
            for item in timings:
                fh.write(f"{item:.3f}\n")
            fh.write("\n")
        fh.write("--- traceback ---\n")
        fh.write(traceback.format_exc())
        fh.write("\n")


def _artifact_study_dir(study_id: str) -> str:
    return os.path.join(ARTIFACT_ROOT, study_id)


def _artifact_path(study_id: str, chain_name: str) -> str:
    return os.path.join(_artifact_study_dir(study_id), f"{study_id}_{chain_name}.json")


def save_chain_artifact(study_id: str, chain_name: str, payload: dict) -> str:
    os.makedirs(_artifact_study_dir(study_id), exist_ok=True)
    path = _artifact_path(study_id, chain_name)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, ensure_ascii=False)
    return path


def load_chain_artifact(study_id: str, chain_name: str) -> dict:
    return _load_json(_artifact_path(study_id, chain_name))


def _load_reusable_artifact(study_id: str, chain_name: str) -> dict | None:
    path = _artifact_path(study_id, chain_name)
    if not os.path.exists(path):
        return None
    payload = load_chain_artifact(study_id, chain_name)
    status = str(payload.get("status", "")).strip().lower()
    if status in {"ran", "skipped", "ran_with_errors"}:
        print(f"[{study_id}] Reusing {chain_name} artifact; skipping LLM: {path}")
        return payload
    print(f"[{study_id}] Ignoring non-terminal {chain_name} artifact with status={status!r}: {path}")
    return None


def _artifact_record_count(outputs: list[dict], task: str) -> int:
    total = 0
    for item in outputs or []:
        data = ((item or {}).get("extracted_data") or {}).get(task) or {}
        records = data.get("records") or []
        if isinstance(records, list):
            total += len(records)
    return total


def _build_chain_artifact_payload(
    *,
    study_id: str,
    chain_name: str,
    task: str,
    status: str,
    mode: str,
    outputs: list[dict],
    message: str = "",
) -> dict:
    return {
        "study_folder": study_id,
        "chain_name": chain_name,
        "task": task,
        "status": status,
        "mode": mode,
        "message": message,
        "created_at_unix": time.time(),
        "provider": LLM_PROVIDER,
        "model_name": LLM_MODEL_NAME,
        "output_count": len(outputs or []),
        "record_count": _artifact_record_count(outputs or [], task),
        "outputs": outputs or [],
    }


def _artifact_outputs(payload: dict) -> list[dict]:
    outputs = payload.get("outputs")
    return outputs if isinstance(outputs, list) else []


def _post_process_chain_payload(
    study_id: str,
    chain_name: str,
    payload: dict,
) -> tuple[dict, dict]:
    processed, summary = process_chain_artifact_payload(payload)
    if summary.get("changed"):
        print(
            f"[{study_id}] Post-processed {chain_name}: "
            f"dropped adsorbent={summary.get('adsorbent_records_dropped', 0)}, "
            f"performance={summary.get('performance_records_dropped', 0)}"
        )
    return processed, summary


def save_artifact_manifest(study_id: str, artifacts: dict[str, str]) -> str:
    os.makedirs(_artifact_study_dir(study_id), exist_ok=True)
    path = os.path.join(_artifact_study_dir(study_id), "manifest.json")
    payload = {
        "study_id": study_id,
        "created_at_unix": time.time(),
        "provider": LLM_PROVIDER,
        "model_name": LLM_MODEL_NAME,
        "artifacts": artifacts,
    }
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, ensure_ascii=False)
    return path


def _run_study_level_extraction(
    *,
    study_id: str,
    task: str,
    chain_name: str,
    entries: list[dict],
    chain,
    prompt_inputs: dict,
    fmap: dict,
    postprocess_records: Callable[[list[dict]], list[dict]] | None = None,
) -> dict:
    mode = f"study_level_all_{task}_chunks"
    selected_entries = [
        ent for ent in entries
        if isinstance(ent, dict) and not _is_irrelevant(ent) and _is_labeled_chunk(ent, task)
    ]
    if not selected_entries:
        message = f"no {task}-labeled chunks found"
        print(f"[{study_id}] {message}")
        return _build_chain_artifact_payload(
            study_id=study_id,
            chain_name=chain_name,
            task=task,
            status="skipped",
            mode=mode,
            outputs=[],
            message=message,
        )

    enriched_text = _format_study_chunks(selected_entries)
    if not enriched_text:
        message = f"no text found in {task}-labeled chunks"
        print(f"[{study_id}] {message}")
        return _build_chain_artifact_payload(
            study_id=study_id,
            chain_name=chain_name,
            task=task,
            status="skipped",
            mode=mode,
            outputs=[],
            message=message,
        )

    caption_index = load_caption_index_for_study(study_id, csv_path=FIGURE_TABLE_INFO_CSV)
    figure_captions = select_figure_captions_for_text(enriched_text, caption_index)
    prompt_kwargs_all = {
        "enriched_text": enriched_text,
        "experiment_chunks": enriched_text,
        "adsorbent_chunks": enriched_text,
        "figure_captions": figure_captions or "",
        **prompt_inputs,
    }
    input_variables = set(chain.prompt.input_variables)
    prompt_kwargs = {
        key: value
        for key, value in prompt_kwargs_all.items()
        if key in input_variables
    }

    ctx_prompts: list[str] = []
    ctx_raws: list[str] = []
    ctx_warnings: list[str] = []
    timings: list[float] = []
    chunk_id = f"{task}_all"

    try:
        prompt = chain.prompt.format_prompt(**prompt_kwargs).to_string()
        ctx_prompts.append(prompt)
        print(f"[{study_id}][{task}_study] PROMPT:\n{prompt}\n")

        start = time.perf_counter()
        call = predict_with_usage(
            chain,
            provider=LLM_PROVIDER,
            model_name=LLM_MODEL_NAME,
            **prompt_kwargs,
        )
        raw = call.text
        usage = call.usage
        elapsed_s = float(getattr(usage, "elapsed_s", 0.0) or (time.perf_counter() - start))
        timings.append(elapsed_s)
        ctx_raws.append(raw)

        print(f"[{study_id}][{task}_study] RAW OUTPUT:\n{raw}\n")
        print(f"[{study_id}][{task}_study]  -> prompt tokens:     {usage.prompt_tokens}")
        print(f"[{study_id}][{task}_study]  -> completion tokens: {usage.completion_tokens}")
        print(f"[{study_id}][{task}_study]  -> total tokens:      {usage.total_tokens}")
        print(f"[{study_id}][{task}_study]  -> elapsed seconds:   {elapsed_s:.3f}\n")

        records = chunk_extract.parse_records(raw, fmap)
        if not records:
            message = f"no valid {task} records returned"
            print(f"[{study_id}] {message}")
            return _build_chain_artifact_payload(
                study_id=study_id,
                chain_name=chain_name,
                task=task,
                status="skipped",
                mode=mode,
                outputs=[],
                message=message,
            )
        if postprocess_records is not None:
            records = postprocess_records(records)
        usage_record = _usage_metadata(usage)
        usage_record["duration_s"] = round(elapsed_s, 3)
        outputs = [
            {
                "study_folder": study_id,
                "file_type": "study_level",
                "chunk_id": chunk_id,
                "enriched_text": enriched_text,
                "enriched_text_metadata": _build_enriched_text_metadata(selected_entries),
                "predicted_label": task,
                "extracted_data": {
                    task: {
                        "records": records,
                        "extracted_count": len(records),
                        "mode": mode,
                        "timings": {
                            "batches_s": timings,
                            "total_s": elapsed_s,
                            "token_counts": [usage_record],
                        },
                    }
                },
                "extraction_counts": {
                    "by_task": {
                        task: {
                            "mode": mode,
                            "extracted": len(records),
                        }
                    },
                    "total_extracted": len(records),
                    "total_requested": len(records),
                },
            }
        ]
        return _build_chain_artifact_payload(
            study_id=study_id,
            chain_name=chain_name,
            task=task,
            status="ran",
            mode=mode,
            outputs=outputs,
        )
    except Exception as exc:
        logging.error("[%s] study-level %s extraction failed", study_id, task, exc_info=True)
        dump_error_context(
            study_id=study_id,
            chunk_id=chunk_id,
            prompt_text="\n\n".join(ctx_prompts),
            raw_text="\n\n".join(ctx_raws),
            warnings=ctx_warnings,
            exc=exc,
            timings=timings,
        )
        return _build_chain_artifact_payload(
            study_id=study_id,
            chain_name=chain_name,
            task=task,
            status="failed",
            mode=mode,
            outputs=[],
            message=str(exc),
        )


def _select_experiment_context_entries(entries: list[dict] | None) -> list[dict]:
    """Return experiment-labeled classified chunks for performance extraction context."""
    return [
        ent for ent in entries or []
        if isinstance(ent, dict) and not _is_irrelevant(ent) and _is_labeled_chunk(ent, "experiment")
    ]


def _entry_source_keys(ent: dict) -> set[str]:
    keys: set[str] = set()
    metadata = ent.get("enriched_text_metadata")
    if isinstance(metadata, list):
        for item in metadata:
            if not isinstance(item, dict):
                continue
            source_id = item.get("source_id")
            if source_id:
                keys.add(str(source_id))
                continue
            file_type = item.get("file_type")
            chunk_id = item.get("chunk_id")
            if file_type is not None and chunk_id is not None:
                keys.add(f"{file_type}_{chunk_id}")
    if not keys:
        keys.add(_source_id_for(ent))
    return keys


def _dedupe_entries_by_source(entries: list[dict]) -> list[dict]:
    deduped: list[dict] = []
    seen: set[str] = set()
    for ent in entries:
        if not isinstance(ent, dict):
            continue
        keys = _entry_source_keys(ent)
        if keys and keys.issubset(seen):
            continue
        deduped.append(ent)
        seen.update(keys)
    return deduped


def _dedupe_enriched_text_metadata(metadata: list[dict]) -> list[dict]:
    deduped: list[dict] = []
    seen: set[str] = set()
    for item in metadata or []:
        if not isinstance(item, dict):
            continue
        source_id = item.get("source_id")
        if not source_id:
            file_type = item.get("file_type")
            chunk_id = item.get("chunk_id")
            if file_type is not None and chunk_id is not None:
                source_id = f"{file_type}_{chunk_id}"
        key = str(source_id or "")
        if key and key in seen:
            continue
        if key:
            seen.add(key)
        deduped.append(item)
    return deduped


def _dedupe_formatted_chunks(text: str) -> str:
    """Remove repeated pre-formatted Chunk blocks by Data_Source and renumber them."""
    text = str(text or "").strip()
    if not text:
        return ""

    header_re = re.compile(r"(?m)^Chunk\s+\d+\s+\(Data_Source=([^;)\n]+)[^)]*\):")
    matches = list(header_re.finditer(text))
    if not matches:
        return text

    blocks: list[str] = []
    seen: set[str] = set()
    prefix = text[:matches[0].start()].strip()
    if prefix:
        blocks.append(prefix)

    chunk_no = 0
    for idx, match in enumerate(matches):
        source_id = match.group(1).strip()
        if source_id in seen:
            continue
        seen.add(source_id)

        start = match.start()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(text)
        block = text[start:end].strip()
        chunk_no += 1
        block = re.sub(
            r"^Chunk\s+\d+",
            f"Chunk {chunk_no}",
            block,
            count=1,
        )
        blocks.append(block)

    return "\n\n".join(blocks).rstrip()


def _metadata_source_id(item: dict) -> str:
    source_id = str(item.get("source_id") or "").strip()
    if source_id:
        return source_id
    file_type = item.get("file_type")
    chunk_id = item.get("chunk_id")
    if file_type is None or chunk_id is None:
        return ""
    return f"{file_type}_{chunk_id}"


def _normalize_prompt_chunk_number(value: Any) -> str:
    text = str(value if value is not None else "").strip()
    if text.endswith(".0") and text[:-2].isdigit():
        text = text[:-2]
    return str(int(text)) if text.isdigit() else text


def _filter_formatted_chunks_by_source(
    text: str,
    allowed_source_ids: set[str],
) -> tuple[str | None, set[str]]:
    """Keep selected pre-formatted Chunk blocks, preserving any text prefix.

    ``None`` means the expected source blocks could not be mapped safely and
    the caller should use the unfiltered text.
    """

    text = str(text or "").strip()
    if not text:
        return None, set()
    allowed = {str(source_id).strip().casefold() for source_id in allowed_source_ids}
    header_re = re.compile(r"(?m)^Chunk\s+\d+\s+\(Data_Source=([^;)\n]+)[^)]*\):")
    matches = list(header_re.finditer(text))
    if not matches:
        return None, set()

    blocks: list[str] = []
    found: set[str] = set()
    prefix = text[:matches[0].start()].strip()
    if prefix:
        blocks.append(prefix)

    chunk_no = 0
    for index, match in enumerate(matches):
        source_id = match.group(1).strip()
        normalized = source_id.casefold()
        if normalized not in allowed:
            continue
        start = match.start()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        block = text[start:end].strip()
        chunk_no += 1
        block = re.sub(r"^Chunk\s+\d+", f"Chunk {chunk_no}", block, count=1)
        blocks.append(block)
        found.add(normalized)

    if not allowed.issubset(found):
        return None, found
    return "\n\n".join(blocks).rstrip(), found


def _route_performance_entry_sources(
    entry: dict,
    records: list[dict],
) -> tuple[dict, dict]:
    """Apply source-aware table pruning with a full-group safety fallback."""

    routed_entry = dict(entry)
    metadata = [
        item
        for item in entry.get("enriched_text_metadata") or []
        if isinstance(item, dict)
    ]
    available_sources = [
        _metadata_source_id(item)
        for item in metadata
        if _metadata_source_id(item)
    ]

    def fallback(reason: str) -> tuple[dict, dict]:
        routing = {
            "mode": "fallback_all_sources",
            "reason": reason,
            "available_sources": available_sources,
            "referenced_sources": [],
            "retained_non_table_context": [],
            "excluded_table_sources": [],
        }
        routed_entry["_performance_source_routing"] = routing
        return routed_entry, routing

    if not PERFORMANCE_SOURCE_AWARE_ROUTING:
        return fallback("source-aware routing disabled")
    if not metadata:
        return fallback("enriched_text_metadata is missing")
    if not records:
        return fallback("enumeration records are missing")

    number_to_source: dict[str, str] = {}
    for item in metadata:
        source_id = _metadata_source_id(item)
        number = _normalize_prompt_chunk_number(item.get("prompt_chunk_number"))
        if not source_id or not number:
            return fallback("source metadata lacks prompt_chunk_number or source_id")
        if number in number_to_source and number_to_source[number] != source_id:
            return fallback(f"prompt chunk number {number} maps to multiple sources")
        number_to_source[number] = source_id

    referenced_sources: set[str] = set()
    for record in records:
        if not isinstance(record, dict):
            return fallback("enumeration contains a non-object record")
        raw_source = str(record.get("Data_Source") or "").strip()
        source_numbers = [_normalize_prompt_chunk_number(value) for value in re.findall(r"\d+", raw_source)]
        if not raw_source or not source_numbers:
            return fallback("an enumeration record has no parseable Data_Source")
        unresolved = [number for number in source_numbers if number not in number_to_source]
        if unresolved:
            return fallback(
                "Data_Source references unknown prompt chunk number(s): "
                + ", ".join(sorted(set(unresolved)))
            )
        referenced_sources.update(number_to_source[number] for number in source_numbers)

    retained_non_table_context = {
        source_id
        for source_id, item in (
            (_metadata_source_id(item), item) for item in metadata
        )
        if source_id
        and str(item.get("file_type") or "").strip().casefold() != "table"
        and source_id not in referenced_sources
    }
    allowed_sources = set(referenced_sources)
    if PERFORMANCE_KEEP_UNCITED_NON_TABLE_CONTEXT:
        allowed_sources.update(retained_non_table_context)
    else:
        retained_non_table_context.clear()

    excluded_table_sources = {
        source_id
        for source_id, item in (
            (_metadata_source_id(item), item) for item in metadata
        )
        if source_id
        and str(item.get("file_type") or "").strip().casefold() == "table"
        and source_id not in referenced_sources
    }
    filtered_text, _found = _filter_formatted_chunks_by_source(
        str(entry.get("enriched_text") or ""),
        allowed_sources,
    )
    if filtered_text is None:
        return fallback("formatted source blocks could not be mapped safely")

    allowed_normalized = {source_id.casefold() for source_id in allowed_sources}
    routed_entry["enriched_text"] = filtered_text
    routed_entry["enriched_text_metadata"] = [
        item
        for item in metadata
        if _metadata_source_id(item).casefold() in allowed_normalized
    ]
    routing = {
        "mode": "referenced_sources_with_non_table_context",
        "reason": "",
        "available_sources": available_sources,
        "referenced_sources": sorted(referenced_sources),
        "retained_non_table_context": sorted(retained_non_table_context),
        "excluded_table_sources": sorted(excluded_table_sources),
    }
    routed_entry["_performance_source_routing"] = routing
    return routed_entry, routing


def _build_performance_source_entries(
    *,
    context_entries: list[dict],
    performance_entries: list[dict],
    source_text: str | None = None,
) -> list[dict]:
    call_entries: list[dict] = []
    if source_text is not None and len(performance_entries) == 1:
        ent = dict(performance_entries[0])
        ent["enriched_text"] = source_text
        call_entries.append(ent)
    else:
        call_entries.extend(performance_entries)
    return _dedupe_entries_by_source([*context_entries, *call_entries])


def _performance_ids_from_records(records: list[dict]) -> list[str]:
    ids: list[str] = []
    for rec in records or []:
        if not isinstance(rec, dict):
            continue
        value = rec.get("Performance_id") or rec.get("performance_id")
        if not value:
            continue
        value_s = str(value).strip()
        if value_s:
            ids.append(value_s)
    return ids


def _configure_chunk_extractor() -> None:
    chunk_extract.configure_runtime(
        provider=LLM_PROVIDER,
        model_name=LLM_MODEL_NAME,
        error_log_root=ERROR_LOG_ROOT,
    )


def _load_performance_enumeration_entries(study_id: str) -> tuple[list[dict], str | None]:
    artifact_path = _enumeration_artifact_path(study_id, "performance")
    if os.path.exists(artifact_path):
        payload = _load_json(artifact_path)
        outputs = payload.get("outputs")
        if isinstance(outputs, list):
            print(f"[{study_id}] Using performance enumeration artifact: {artifact_path}")
            return outputs, artifact_path

    _write_missing_log(study_id, artifact_path)
    return [], None


def _flatten_performance_enumerations(entries: list[dict]) -> list[dict]:
    flattened: list[dict] = []
    for ent in entries or []:
        if not isinstance(ent, dict):
            continue

        if ent.get("task") == "performance":
            flattened.append(ent)
            continue

        common = {
            "study_folder": ent.get("study_folder"),
            "file_type": ent.get("file_type"),
            "chunk_id": ent.get("chunk_id"),
            "enriched_text": ent.get("enriched_text"),
            "enriched_text_metadata": ent.get("enriched_text_metadata") or [],
            "predicted_label": ent.get("predicted_label"),
        }
        for enum in ent.get("enumerations") or []:
            if not isinstance(enum, dict) or enum.get("task") != "performance":
                continue
            flattened.append(
                {
                    **common,
                    "task": "performance",
                    "records": enum.get("records") or [],
                    "extracted_count": enum.get("extracted_count"),
                    "timings": enum.get("timings") or {},
                }
            )
    return flattened

PERFORMANCE_VALUE_UNIT_PAIRS = [
    ("Freundlich_KF_value", "Freundlich_KF_unit"),
    ("Langmuir_Qm_value", "Langmuir_Qm_unit"),
    ("Langmuir_KL_value", "Langmuir_KL_unit"),
    ("Kd_value", "Kd_unit"),
    ("Qe_value", "Qe_unit"),
    ("PFO_k1_value", "PFO_k1_unit"),
    ("PFO_Qe_value", "PFO_Qe_unit"),
    ("PSO_k2_value", "PSO_k2_unit"),
    ("PSO_v0_value", "PSO_v0_unit"),
    ("PSO_Qe_value", "PSO_Qe_unit"),
]
PERFORMANCE_UNIT_FIELDS = {unit for _, unit in PERFORMANCE_VALUE_UNIT_PAIRS}
PERFORMANCE_UNIT_FIELD_BY_LOWER = {
    field.lower(): field
    for field in PERFORMANCE_UNIT_FIELDS | {"Freundlich_equation_form"}
}
EMPTY_FIELD_VALUES = {"", "na", "n/a", "null", "none", "not found", "unknown"}
UNIT_FIELD_KEYWORDS = {
    "Freundlich_KF_unit": ["freundlich", "kf", "k f"],
    "Langmuir_Qm_unit": ["langmuir", "qm", "q m"],
    "Langmuir_KL_unit": ["langmuir", "kl", "k l"],
    "Kd_unit": ["kd", "k d", "log kd", "distribution coefficient", "adsorption affinity"],
    "Qe_unit": ["qe", "q e", "equilibrium adsorption capacity", "equilibrium adsorption density", "equilibrium uptake", "adsorbed amount"],
    "PFO_k1_unit": ["pseudo first order", "pfo", "k1", "k 1"],
    "PFO_Qe_unit": ["pseudo first order", "pfo", "qe", "q e"],
    "PSO_k2_unit": ["pseudo second order", "pso", "k2", "k 2"],
    "PSO_v0_unit": ["initial adsorption rate", "v0", "v 0", "pso"],
    "PSO_Qe_unit": ["pseudo second order", "pso", "qe", "q e"],
}


def _field_is_valid(value: Any) -> bool:
    if value is None:
        return False
    text = str(value).strip()
    return text.casefold() not in EMPTY_FIELD_VALUES


def _find_key_ci(record: dict, canonical_key: str) -> str | None:
    target = canonical_key.casefold()
    for key in record.keys():
        if str(key).casefold() == target:
            return key
    return None


def _get_ci_value(record: dict, canonical_key: str) -> Any:
    key = _find_key_ci(record, canonical_key)
    return record.get(key) if key else None


def _set_ci_value(record: dict, canonical_key: str, value: Any) -> None:
    key = _find_key_ci(record, canonical_key) or canonical_key.lower()
    record[key] = value


def _append_record_warning(record: dict, message: str) -> None:
    warnings = record.get("_warnings")
    if not isinstance(warnings, list):
        warnings = []
        record["_warnings"] = warnings
    if message not in warnings:
        warnings.append(message)


def _unit_norm(value: Any) -> str:
    text = str(value or "").strip()
    text = text.replace("μ", "µ").replace("Âµ", "µ").replace("Î¼", "µ")
    text = re.sub(r"\s+", "", text)
    return text.casefold()


def _single_supported_unit(value: Any) -> str:
    if not _field_is_valid(value):
        return ""
    parts = [part.strip() for part in str(value).split(";") if part.strip()]
    if not parts:
        return ""
    by_norm: dict[str, str] = {}
    for part in parts:
        if not _field_is_valid(part):
            continue
        by_norm.setdefault(_unit_norm(part), part)
    if len(by_norm) != 1:
        return ""
    return next(iter(by_norm.values()))


def _unique_supported_unit(values: list[Any]) -> str:
    by_norm: dict[str, str] = {}
    for value in values:
        single = _single_supported_unit(value)
        if single:
            by_norm.setdefault(_unit_norm(single), single)
    if len(by_norm) != 1:
        return ""
    return next(iter(by_norm.values()))


def _performance_record_infos(outputs: list[dict]) -> list[dict]:
    infos: list[dict] = []
    for output in outputs or []:
        section = ((output.get("extracted_data") or {}).get("performance") or {})
        if not isinstance(section, dict):
            continue
        records = section.get("records")
        if not isinstance(records, list):
            continue
        for rec in records:
            if isinstance(rec, dict):
                infos.append(
                    {
                        "output": output,
                        "section": section,
                        "record": rec,
                        "scope": (
                            output.get("study_folder"),
                            output.get("file_type"),
                            output.get("chunk_id"),
                        ),
                    }
                )
    return infos


def _record_needs_unit(record: dict, value_field: str, unit_field: str) -> bool:
    return _field_is_valid(_get_ci_value(record, value_field)) and not _field_is_valid(
        _get_ci_value(record, unit_field)
    )


def _missing_performance_unit_fields(outputs: list[dict]) -> tuple[set[str], list[dict]]:
    missing_fields: set[str] = set()
    missing_records: list[dict] = []
    for info in _performance_record_infos(outputs):
        rec = info["record"]
        for value_field, unit_field in PERFORMANCE_VALUE_UNIT_PAIRS:
            if _record_needs_unit(rec, value_field, unit_field):
                missing_fields.add(unit_field)
                missing_records.append(
                    {
                        "unit_field": unit_field,
                        "value_field": value_field,
                        "file_type": info["output"].get("file_type"),
                        "chunk_id": info["output"].get("chunk_id"),
                        "performance_id": _get_ci_value(rec, "Performance_id"),
                    }
                )
    return missing_fields, missing_records


def _borrow_units_from_extracted_records(outputs: list[dict]) -> dict:
    infos = _performance_record_infos(outputs)
    units_by_scope: dict[tuple, dict[str, list[Any]]] = defaultdict(lambda: defaultdict(list))
    units_by_study: dict[str, list[Any]] = defaultdict(list)

    for info in infos:
        rec = info["record"]
        for unit_field in PERFORMANCE_UNIT_FIELDS:
            unit_value = _single_supported_unit(_get_ci_value(rec, unit_field))
            if not unit_value:
                continue
            units_by_scope[info["scope"]][unit_field].append(unit_value)
            units_by_study[unit_field].append(unit_value)

    fills: list[dict] = []
    for info in infos:
        rec = info["record"]
        for value_field, unit_field in PERFORMANCE_VALUE_UNIT_PAIRS:
            if not _record_needs_unit(rec, value_field, unit_field):
                continue
            unit_value = _unique_supported_unit(units_by_scope[info["scope"]][unit_field])
            source = "same extraction output"
            if not unit_value:
                unit_value = _unique_supported_unit(units_by_study[unit_field])
                source = "study-level extracted records"
            if not unit_value:
                continue
            _set_ci_value(rec, unit_field, unit_value)
            _append_record_warning(rec, f"{unit_field} filled from {source}")
            fills.append(
                {
                    "unit_field": unit_field,
                    "value": unit_value,
                    "source": source,
                    "file_type": info["output"].get("file_type"),
                    "chunk_id": info["output"].get("chunk_id"),
                    "performance_id": _get_ci_value(rec, "Performance_id"),
                }
            )
    return {"fill_count": len(fills), "fills": fills}


def _entry_text(entry: dict) -> str:
    return str(entry.get("enriched_text") or entry.get("text") or "").strip()


def _entry_has_non_table_text_source(entry: dict) -> bool:
    metadata = entry.get("enriched_text_metadata")
    if isinstance(metadata, list) and metadata:
        file_types = [
            str(item.get("file_type") or "").strip().lower()
            for item in metadata
            if isinstance(item, dict)
        ]
        return any(file_type != "table" for file_type in file_types if file_type)
    return str(entry.get("file_type") or "").strip().lower() != "table"


def _keyword_text(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(text or "").casefold())


def _entry_matches_missing_unit_keywords(entry: dict, missing_fields: set[str]) -> bool:
    text = _keyword_text(_entry_text(entry))
    if not text:
        return False
    for field in missing_fields:
        for keyword in UNIT_FIELD_KEYWORDS.get(field, []):
            if _keyword_text(keyword).strip() in text:
                return True
    return False


def _select_unit_rescue_entries(
    candidates: list[dict],
    missing_fields: set[str],
) -> list[dict]:
    text_candidates = [
        entry
        for entry in candidates
        if _entry_text(entry) and _entry_has_non_table_text_source(entry)
    ]
    if not text_candidates:
        return []
    matched = [
        entry
        for entry in text_candidates
        if _entry_matches_missing_unit_keywords(entry, missing_fields)
    ]
    return matched or text_candidates


def _provenance_source_ids(outputs: list[dict]) -> set[str]:
    """Return source IDs cited by extracted records' ``Data_provenance``."""
    source_ids: set[str] = set()
    for output in outputs or []:
        number_to_source = {
            _normalize_prompt_chunk_number(match.group(1)): match.group(2).strip()
            for match in re.finditer(
                r"(?m)^Chunk\s+(\d+)\s+\(Data_Source=([^;)\n]+)",
                str(output.get("enriched_text") or ""),
            )
        }
        metadata = [
            item
            for item in output.get("enriched_text_metadata") or []
            if isinstance(item, dict)
        ]
        if not number_to_source:
            for item in metadata:
                number = _normalize_prompt_chunk_number(item.get("prompt_chunk_number"))
                source_id = _metadata_source_id(item)
                if number and source_id:
                    number_to_source.setdefault(number, source_id)
        if not number_to_source:
            continue
        section = ((output.get("extracted_data") or {}).get("performance") or {})
        records = section.get("records") if isinstance(section, dict) else []
        for record in records or []:
            if not isinstance(record, dict):
                continue
            raw_provenance = str(_get_ci_value(record, "Data_provenance") or "")
            for number in re.findall(r"\d+", raw_provenance):
                source_id = number_to_source.get(_normalize_prompt_chunk_number(number))
                if source_id:
                    source_ids.add(source_id.casefold())
    return source_ids


def _build_unit_rescue_candidates(
    entries: list[dict],
    outputs: list[dict],
) -> list[dict]:
    """Return individual non-table performance sources absent from record provenance.

    Enumeration batches can contain both a value table and explanatory text.  A
    batch-level ``records`` check hides the explanatory source whenever the
    table yields IDs, even if no extracted record cites that source.  Split the
    batches here so unit rescue can inspect each uncited performance chunk.
    """
    cited_source_ids = _provenance_source_ids(outputs)
    candidates: list[dict] = []
    seen_source_ids: set[str] = set()

    for entry in entries or []:
        if not isinstance(entry, dict):
            continue
        metadata = [
            item
            for item in entry.get("enriched_text_metadata") or []
            if isinstance(item, dict)
        ]
        for item in metadata:
            source_id = _metadata_source_id(item)
            source_key = source_id.casefold()
            if (
                not source_id
                or source_key in cited_source_ids
                or source_key in seen_source_ids
                or str(item.get("file_type") or "").strip().casefold() == "table"
                or not _is_labeled_chunk(item, "performance")
            ):
                continue
            source_text, _found = _filter_formatted_chunks_by_source(
                str(entry.get("enriched_text") or ""),
                {source_id},
            )
            if source_text is None:
                logging.warning(
                    "Skipping unit-rescue source %s: formatted chunk could not be isolated",
                    source_id,
                )
                continue
            candidates.append(
                {
                    "study_folder": item.get("study_folder") or entry.get("study_folder"),
                    "file_type": item.get("file_type"),
                    "chunk_id": item.get("chunk_id"),
                    "enriched_text": source_text,
                    "enriched_text_metadata": [item],
                    "predicted_label": item.get("predicted_label"),
                }
            )
            seen_source_ids.add(source_key)
    return candidates


def _canonical_unit_field(raw_key: Any) -> str:
    return PERFORMANCE_UNIT_FIELD_BY_LOWER.get(str(raw_key or "").strip().lower(), "")


def _merge_unit_map(
    records: list[dict],
    needed_fields: set[str],
) -> tuple[dict[str, str], dict[str, list[str]]]:
    values_by_field: dict[str, list[Any]] = defaultdict(list)
    conflicts: dict[str, list[str]] = {}
    for rec in records or []:
        if not isinstance(rec, dict):
            continue
        for key, value in rec.items():
            field = _canonical_unit_field(key)
            if field not in needed_fields:
                continue
            if not _field_is_valid(value):
                continue
            single = _single_supported_unit(value)
            if single:
                values_by_field[field].append(single)
            else:
                conflicts.setdefault(field, []).append(str(value).strip())

    unit_map: dict[str, str] = {}
    for field, values in values_by_field.items():
        unit_value = _unique_supported_unit(values)
        if unit_value:
            unit_map[field] = unit_value
        elif values:
            conflicts.setdefault(field, []).extend(str(value) for value in values)

    conflicts = {
        field: sorted({_unit for _unit in values if _unit})
        for field, values in conflicts.items()
        if values
    }
    return unit_map, conflicts


def _apply_unit_map_to_missing_records(
    outputs: list[dict],
    unit_map: dict[str, str],
) -> dict:
    fills: list[dict] = []
    if not unit_map:
        return {"fill_count": 0, "fills": fills}
    for info in _performance_record_infos(outputs):
        rec = info["record"]
        for value_field, unit_field in PERFORMANCE_VALUE_UNIT_PAIRS:
            if unit_field not in unit_map:
                continue
            if not _record_needs_unit(rec, value_field, unit_field):
                continue
            unit_value = unit_map[unit_field]
            _set_ci_value(rec, unit_field, unit_value)
            _append_record_warning(rec, f"{unit_field} filled from unit rescue")
            fills.append(
                {
                    "unit_field": unit_field,
                    "value": unit_value,
                    "source": "unit_rescue",
                    "file_type": info["output"].get("file_type"),
                    "chunk_id": info["output"].get("chunk_id"),
                    "performance_id": _get_ci_value(rec, "Performance_id"),
                }
            )
    return {"fill_count": len(fills), "fills": fills}


def _attach_unit_repair_metadata(outputs: list[dict], metadata: dict) -> None:
    if not outputs or not metadata:
        return
    for output in outputs:
        section = ((output.get("extracted_data") or {}).get("performance") or {})
        if isinstance(section, dict):
            section["unit_repair"] = metadata
            return


def _extract_performance_unit_map(
    *,
    study_id: str,
    unit_entries: list[dict],
    needed_fields: set[str],
    chain,
    fmap: dict,
    caption_index,
) -> tuple[dict[str, str], dict | None, str | None]:
    if not unit_entries or not needed_fields:
        return {}, None, None

    mode = "study_level_performance_unit_rescue"
    enriched_text = _format_study_chunks(unit_entries)
    if not enriched_text:
        return {}, None, "performance_unit_rescue: no text found"

    figure_captions = select_figure_captions_for_text(enriched_text, caption_index)
    prompt_kwargs_all = {
        "enriched_text": enriched_text,
        "figure_captions": figure_captions or "",
        "requested_unit_fields": [
            unit_field
            for _, unit_field in PERFORMANCE_VALUE_UNIT_PAIRS
            if unit_field in needed_fields
        ],
    }
    input_variables = set(chain.prompt.input_variables)
    prompt_kwargs = {
        key: value
        for key, value in prompt_kwargs_all.items()
        if key in input_variables
    }

    ctx_prompts: list[str] = []
    ctx_raws: list[str] = []
    ctx_warnings: list[str] = []
    timings: list[float] = []
    chunk_id = "performance_unit_rescue"

    try:
        prompt = chain.prompt.format_prompt(**prompt_kwargs).to_string()
        ctx_prompts.append(prompt)
        print(f"[{study_id}][performance_unit_rescue] PROMPT:\n{prompt}\n")

        start = time.perf_counter()
        call = predict_with_usage(
            chain,
            provider=LLM_PROVIDER,
            model_name=LLM_MODEL_NAME,
            **prompt_kwargs,
        )
        raw = call.text
        usage = call.usage
        elapsed_s = float(getattr(usage, "elapsed_s", 0.0) or (time.perf_counter() - start))
        timings.append(elapsed_s)
        ctx_raws.append(raw)

        records = chunk_extract.parse_records(raw, fmap)
        unit_map, conflicts = _merge_unit_map(records, needed_fields)
        usage_record = _usage_metadata(usage)
        usage_record["duration_s"] = round(elapsed_s, 3)
        metadata = {
            "mode": mode,
            "needed_unit_fields": sorted(needed_fields),
            "extracted_unit_map": unit_map,
            "conflicts": conflicts,
            "source_metadata": _build_enriched_text_metadata(unit_entries),
            "timings": {
                "batches_s": timings,
                "total_s": elapsed_s,
                "token_counts": [usage_record],
            },
        }
        return unit_map, metadata, None
    except Exception as exc:
        logging.error("[%s] failed processing performance unit rescue chunks", study_id, exc_info=True)
        dump_error_context(
            study_id=study_id,
            chunk_id=chunk_id,
            prompt_text="\n\n".join(ctx_prompts),
            raw_text="\n\n".join(ctx_raws),
            warnings=ctx_warnings,
            exc=exc,
            timings=timings,
        )
        return {}, None, f"{chunk_id}: {exc}"


def _repair_performance_units(
    *,
    study_id: str,
    outputs: list[dict],
    rescue_candidates: list[dict],
    unit_chain,
    fmap: dict,
    caption_index,
) -> str | None:
    before_missing_fields, before_missing_records = _missing_performance_unit_fields(outputs)
    rescue_metadata = None
    rescue_apply = {"fill_count": 0, "fills": []}
    selected_rescue_entries: list[dict] = []
    rescue_error = None

    if before_missing_fields:
        selected_rescue_entries = _select_unit_rescue_entries(
            rescue_candidates,
            before_missing_fields,
        )
        if selected_rescue_entries:
            print(
                f"[{study_id}] Performance unit rescue: "
                f"{len(before_missing_fields)} missing unit field(s), "
                f"{len(selected_rescue_entries)} uncited non-table performance source(s)."
            )
            unit_map, rescue_metadata, rescue_error = _extract_performance_unit_map(
                study_id=study_id,
                unit_entries=selected_rescue_entries,
                needed_fields=before_missing_fields,
                chain=unit_chain,
                fmap=fmap,
                caption_index=caption_index,
            )
            rescue_apply = _apply_unit_map_to_missing_records(outputs, unit_map)
        else:
            print(
                f"[{study_id}] Performance unit rescue skipped: "
                "no uncited non-table performance chunks available."
            )

    after_rescue_fields, after_rescue_records = _missing_performance_unit_fields(outputs)
    deterministic = _borrow_units_from_extracted_records(outputs)
    after_borrow_fields, after_borrow_records = _missing_performance_unit_fields(outputs)
    final_missing_fields, final_missing_records = _missing_performance_unit_fields(outputs)
    repair_metadata = {
        "missing_unit_fields_before": sorted(before_missing_fields),
        "missing_record_count_before": len(before_missing_records),
        "rescue_candidate_count": len(rescue_candidates),
        "selected_rescue_candidate_count": len(selected_rescue_entries),
        "unit_rescue": rescue_metadata,
        "unit_rescue_apply": rescue_apply,
        "missing_unit_fields_after_unit_rescue": sorted(after_rescue_fields),
        "missing_record_count_after_unit_rescue": len(after_rescue_records),
        "deterministic_borrow": deterministic,
        "missing_unit_fields_after_borrow": sorted(after_borrow_fields),
        "missing_record_count_after_borrow": len(after_borrow_records),
        "missing_unit_fields_after": sorted(final_missing_fields),
        "missing_record_count_after": len(final_missing_records),
    }
    if (
        before_missing_fields
        or deterministic.get("fill_count")
        or rescue_metadata
        or final_missing_fields
    ):
        _attach_unit_repair_metadata(outputs, repair_metadata)
    return rescue_error


def _extract_performance_value_records(
    *,
    study_id: str,
    llm,
    chain,
    entries: list[dict],
    value_ids: list[str],
    prompt_inputs: dict,
    lexicon_payload: dict,
    caption_index,
    fmap: dict,
    mode: str,
    output_file_type: str,
    output_chunk_id: Any,
    output_predicted_label: str = "performance",
    source_text: str | None = None,
    log_chunk_id: str | None = None,
) -> tuple[dict | None, str | None]:
    if not value_ids:
        return None, None

    source_routing = next(
        (
            entry.get("_performance_source_routing")
            for entry in entries
            if isinstance(entry, dict) and entry.get("_performance_source_routing")
        ),
        None,
    )
    context_entries = prompt_inputs.get("experiment_context_entries") or []
    source_entries = _build_performance_source_entries(
        context_entries=context_entries if isinstance(context_entries, list) else [],
        performance_entries=entries,
        source_text=source_text,
    )
    text = _dedupe_formatted_chunks(_format_study_chunks(source_entries))
    if not text and source_text is not None:
        text = _dedupe_formatted_chunks(str(source_text or "").strip())
    if not text:
        return None, f"{output_chunk_id}: no text found"

    ctx_prompts: list[str] = []
    ctx_raws: list[str] = []
    ctx_warnings: list[str] = []
    perf_timings: list[float] = []
    perf_tokens: list[dict] = []
    error_chunk_id = str(log_chunk_id or output_chunk_id)

    try:
        figure_captions = select_figure_captions_for_text(text, caption_index)
        adsorbent_list = _render_adsorbent_lexicon(
            lexicon_payload,
            _adsorbent_ids_from_performance_ids(value_ids),
        )
        value_prompt_inputs = {
            **prompt_inputs,
            "adsorbent_lexicon": adsorbent_list,
            "adsorbent_list": adsorbent_list,
        }
        recs, batch_times, total_s = chunk_extract.extract_with_batch(
            text,
            figure_captions,
            chain,
            llm,
            fmap,
            "Performance_id",
            value_ids,
            BATCH_SIZE,
            ctx_prompts,
            ctx_raws,
            ctx_warnings,
            perf_timings,
            perf_tokens,
            prompt_inputs=value_prompt_inputs,
        )
        # A condition-only response is not a performance record.  Discard it
        # before unit rescue so an invalid ID cannot yield a partial output.
        recs = [record for record in recs if has_target_performance_metric(record)]
        if not recs:
            return None, None
        returned_ids = [
            rec.get("performance_id")
            for rec in recs
            if rec.get("performance_id")
        ]
        missing = sorted(
            {chunk_extract.normalize_id(i) for i in value_ids}
            - {chunk_extract.normalize_id(i) for i in returned_ids}
        )
        return (
            {
                "study_folder": study_id,
                "file_type": output_file_type,
                "chunk_id": output_chunk_id,
                "enriched_text": text,
                "enriched_text_metadata": _dedupe_enriched_text_metadata(
                    _build_enriched_text_metadata(source_entries)
                ),
                "predicted_label": output_predicted_label,
                "source_routing": source_routing or {
                    "mode": "not_available",
                    "reason": "source routing metadata was not provided",
                },
                "extracted_data": {
                    "performance": {
                        "records": recs,
                        "extracted_count": len(recs),
                        "mode": mode,
                        "timings": {
                            "batches_s": batch_times,
                            "total_s": total_s,
                            "token_counts": perf_tokens,
                        },
                    }
                },
                "extraction_counts": {
                    "by_task": {
                        "performance": {
                            "mode": mode,
                            "requested": len(value_ids),
                            "extracted": len(recs),
                            "missing_ids": missing,
                        }
                    },
                    "total_extracted": len(recs),
                    "total_requested": len(value_ids),
                },
            },
            None,
        )
    except Exception as exc:
        logging.error("[%s] failed processing performance value source %r", study_id, error_chunk_id, exc_info=True)
        dump_error_context(
            study_id=study_id,
            chunk_id=error_chunk_id,
            prompt_text="\n\n".join(ctx_prompts),
            raw_text="\n\n".join(ctx_raws),
            warnings=ctx_warnings,
            exc=exc,
            timings=perf_timings,
        )
        return None, f"{error_chunk_id}: {exc}"


def _process_performance_by_chunk(
    study_id: str,
    llm,
    value_chain,
    unit_chain,
    chain_name: str,
    prompt_inputs: dict,
    lexicon_payload: dict,
    performance_entries_override: list[dict] | None = None,
) -> dict:
    mode = "hybrid_performance_normal_grouped_heavy_table_batched_unit"
    if performance_entries_override is None:
        raw_entries, source_path = _load_performance_enumeration_entries(study_id)
    else:
        raw_entries = performance_entries_override
        source_path = "provided performance enumeration entries"
    if not source_path:
        message = "skipped performance: enumeration input not found"
        print(f"[{study_id}] {message}")
        return _build_chain_artifact_payload(
            study_id=study_id,
            chain_name=chain_name,
            task="performance",
            status="skipped",
            mode=mode,
            outputs=[],
            message=message,
        )

    entries = _flatten_performance_enumerations(raw_entries)
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for ent in entries:
        if not isinstance(ent, dict) or ent.get("task") != "performance":
            continue
        records = ent.get("records") or []
        key = (
            ent.get("study_folder"),
            ent.get("file_type"),
            ent.get("chunk_id"),
            ent.get("enriched_text"),
            tuple(
                json.dumps(item, sort_keys=True)
                for item in (ent.get("enriched_text_metadata") or [])
                if isinstance(item, dict)
            ),
            ent.get("predicted_label"),
        )
        if not records:
            continue
        groups[key].extend(records)

    if not groups:
        message = "no performance enumeration records found"
        print(f"[{study_id}] {message}")
        return _build_chain_artifact_payload(
            study_id=study_id,
            chain_name=chain_name,
            task="performance",
            status="skipped",
            mode=mode,
            outputs=[],
            message=message,
        )

    _configure_chunk_extractor()
    caption_index = load_caption_index_for_study(study_id, csv_path=FIGURE_TABLE_INFO_CSV)
    fmap = {**_performance_parse_fields(), **REVIEW_FIELDS}
    outputs: list[dict] = []
    failed_chunks: list[str] = []
    normal_value_jobs: list[tuple[dict, list[str]]] = []
    heavy_value_jobs: list[tuple[dict, list[str]]] = []

    for (sf, ft, cid, text, metadata_key, label), records in groups.items():
        value_ids = _performance_ids_from_records(records)
        if not value_ids:
            continue

        entry = {
            "study_folder": sf,
            "file_type": ft,
            "chunk_id": cid,
            "enriched_text": text,
            "enriched_text_metadata": [
                json.loads(item)
                for item in metadata_key
            ],
            "predicted_label": label,
        }
        entry, source_routing = _route_performance_entry_sources(entry, records)
        if source_routing.get("mode") == "fallback_all_sources":
            print(
                f"[{study_id}] Performance source routing fallback for {cid}: "
                f"{source_routing.get('reason')}"
            )
        elif source_routing.get("excluded_table_sources"):
            print(
                f"[{study_id}] Performance source routing for {cid}: "
                f"referenced={source_routing.get('referenced_sources')}, "
                f"excluded uncited tables={source_routing.get('excluded_table_sources')}"
            )
        routed_text = str(entry.get("enriched_text") or "")
        if _is_heavy_performance_table(ft, routed_text, value_ids):
            heavy_value_jobs.append((entry, value_ids))
        else:
            normal_value_jobs.append((entry, value_ids))

    char_threshold_note = (
        f", table chars > {PERFORMANCE_HEAVY_TABLE_CHAR_THRESHOLD}"
        if PERFORMANCE_HEAVY_TABLE_CHAR_THRESHOLD > 0
        else ""
    )
    normal_value_id_count = sum(len(value_ids) for _, value_ids in normal_value_jobs)
    heavy_value_id_count = sum(len(value_ids) for _, value_ids in heavy_value_jobs)
    print(
        f"[{study_id}] Performance routing: "
        f"{len(normal_value_jobs)} normal enumeration group(s) "
        f"({normal_value_id_count} requested value IDs), "
        f"{len(heavy_value_jobs)} heavy table group(s) "
        f"({heavy_value_id_count} requested value IDs), "
        f"Heavy table threshold: table value IDs > {PERFORMANCE_HEAVY_TABLE_RECORD_THRESHOLD}"
        f"{char_threshold_note}."
    )

    for entry, value_ids in normal_value_jobs:
        cid = entry.get("chunk_id")
        normal_output, normal_error = _extract_performance_value_records(
            study_id=study_id,
            llm=llm,
            chain=value_chain,
            entries=[entry],
            value_ids=value_ids,
            prompt_inputs=prompt_inputs,
            lexicon_payload=lexicon_payload,
            caption_index=caption_index,
            fmap=fmap,
            mode="same_group_performance_value_normal",
            output_file_type=str(entry.get("file_type") or "study_level"),
            output_chunk_id=cid,
            output_predicted_label=str(entry.get("predicted_label") or "performance"),
            source_text=str(entry.get("enriched_text") or ""),
            log_chunk_id=str(cid),
        )
        if normal_output is not None:
            outputs.append(normal_output)
        if normal_error:
            failed_chunks.append(normal_error)

    for entry, value_ids in heavy_value_jobs:
        cid = entry.get("chunk_id")
        heavy_output, heavy_error = _extract_performance_value_records(
            study_id=study_id,
            llm=llm,
            chain=value_chain,
            entries=[entry],
            value_ids=value_ids,
            prompt_inputs=prompt_inputs,
            lexicon_payload=lexicon_payload,
            caption_index=caption_index,
            fmap=fmap,
            mode="chunk_level_performance_value_heavy_table",
            output_file_type=str(entry.get("file_type") or "table"),
            output_chunk_id=cid,
            output_predicted_label=str(entry.get("predicted_label") or "performance"),
            source_text=str(entry.get("enriched_text") or ""),
            log_chunk_id=str(cid),
        )
        if heavy_output is not None:
            outputs.append(heavy_output)
        if heavy_error:
            failed_chunks.append(heavy_error)

    if outputs:
        unit_rescue_candidates = _build_unit_rescue_candidates(entries, outputs)
        print(
            f"[{study_id}] Performance unit rescue candidates: "
            f"{len(unit_rescue_candidates)} individual non-table performance chunk(s) "
            "not cited by extracted-record Data_provenance."
        )
        unit_error = _repair_performance_units(
            study_id=study_id,
            outputs=outputs,
            rescue_candidates=unit_rescue_candidates,
            unit_chain=unit_chain,
            fmap=fmap,
            caption_index=caption_index,
        )
        if unit_error:
            failed_chunks.append(unit_error)

    if not outputs and not failed_chunks:
        message = "no performance value or unit records found"
        print(f"[{study_id}] {message}")
        return _build_chain_artifact_payload(
            study_id=study_id,
            chain_name=chain_name,
            task="performance",
            status="skipped",
            mode=mode,
            outputs=[],
            message=message,
        )

    status = "ran_with_errors" if failed_chunks else "ran"
    message = ""
    if failed_chunks:
        message = "failed chunks: " + "; ".join(failed_chunks)
    return _build_chain_artifact_payload(
        study_id=study_id,
        chain_name=chain_name,
        task="performance",
        status=status,
        mode=mode,
        outputs=outputs,
        message=message,
    )


def _enabled_chain_keys() -> set[str]:
    keys: set[str] = set()
    if ENABLE_ADSORBENT:
        keys.add("adsorbent_study")
    if ENABLE_EXPERIMENT:
        keys.add("experiment_study")
    if ENABLE_PERFORMANCE:
        keys.add("performance")
    return keys


def process_study(
    study_id: str,
    llm,
    *,
    paths: StudyPaths | None = None,
    classified_entries_override: list[dict] | None = None,
    performance_entries_override: list[dict] | None = None,
    output_dir: Path | None = None,
    force_llm_rerun: bool | None = None,
) -> None:
    paths = paths or get_study_paths(study_id)
    _configure_study_paths(paths, output_dir=output_dir)
    enabled = _enabled_chain_keys()
    rerun_enabled_chains = FORCE_LLM_RERUN if force_llm_rerun is None else force_llm_rerun
    force_rerun_keys = enabled if rerun_enabled_chains else set()
    if not enabled:
        print(f"[{study_id}] all extraction tasks are disabled")
        return

    artifacts: dict[str, str] = {}
    run_keys: set[str] = set()

    for task_enabled, chain_name in [
        (ENABLE_ADSORBENT, "adsorbent_study"),
        (ENABLE_EXPERIMENT, "experiment_study"),
        (ENABLE_PERFORMANCE, "performance"),
    ]:
        if not task_enabled:
            continue
        if chain_name in force_rerun_keys:
            path = _artifact_path(study_id, chain_name)
            if os.path.exists(path):
                print(f"[{study_id}] Force rerun enabled for {chain_name}; ignoring artifact: {path}")
            else:
                print(f"[{study_id}] Force rerun enabled for {chain_name}; no reusable artifact found")
            run_keys.add(chain_name)
            continue
        payload = _load_reusable_artifact(study_id, chain_name)
        if payload is not None:
            payload, summary = _post_process_chain_payload(study_id, chain_name, payload)
            if summary.get("changed"):
                save_chain_artifact(study_id, chain_name, payload)
                print(f"[{study_id}] Updated reusable {chain_name} artifact after post-processing")
            artifacts[chain_name] = _artifact_path(study_id, chain_name)
        else:
            run_keys.add(chain_name)

    chains = (
        create_extraction_chains(PROMPTS_DIR, llm, BATCH_SIZE, enabled_tasks=run_keys)
        if run_keys
        else {}
    )
    classified_entries: list[dict] | None = classified_entries_override
    classified_entries_loaded = classified_entries_override is not None
    lexicon_payload: dict | None = None

    def get_classified_entries() -> list[dict] | None:
        nonlocal classified_entries, classified_entries_loaded
        if not classified_entries_loaded:
            classified_entries = _load_classified_entries(study_id)
            classified_entries_loaded = True
        return classified_entries

    def get_lexicon_payload() -> dict:
        nonlocal lexicon_payload
        if lexicon_payload is None:
            lexicon_payload = _load_enumeration_vocab_payload(study_id)
        return lexicon_payload

    if ENABLE_ADSORBENT and "adsorbent_study" in run_keys:
        entries = get_classified_entries()
        if entries:
            lexicon_payload = get_lexicon_payload()
            adsorbent_lexicon = _render_adsorbent_lexicon(lexicon_payload)
            payload = _run_study_level_extraction(
                study_id=study_id,
                task="adsorbent",
                chain_name="adsorbent_study",
                entries=entries,
                chain=chains["adsorbent_study"],
                prompt_inputs={
                    "adsorbent_lexicon": adsorbent_lexicon,
                    "adsorbent_list": adsorbent_lexicon,
                },
                fmap={"Adsorbent_id": str, **ADSORBENT_FIELDS, **REVIEW_FIELDS},
                postprocess_records=lambda records: _inject_adsorbent_identifier_fields(
                    records,
                    lexicon_payload,
                ),
            )
        else:
            payload = _build_chain_artifact_payload(
                study_id=study_id,
                chain_name="adsorbent_study",
                task="adsorbent",
                status="skipped",
                mode="study_level_all_adsorbent_chunks",
                outputs=[],
                message="classified input not found",
            )
        payload, _summary = _post_process_chain_payload(study_id, "adsorbent_study", payload)
        path = save_chain_artifact(study_id, "adsorbent_study", payload)
        artifacts["adsorbent_study"] = path
        print(f"[{study_id}] Saved adsorbent_study artifact: {path}")

    if ENABLE_EXPERIMENT and "experiment_study" in run_keys:
        entries = get_classified_entries()
        if entries:
            lexicon_payload = get_lexicon_payload()
            water_type_list = _render_water_type_lexicon(lexicon_payload)
            adsorbent_list = _render_adsorbent_lexicon(lexicon_payload)
            payload = _run_study_level_extraction(
                study_id=study_id,
                task="experiment",
                chain_name="experiment_study",
                entries=entries,
                chain=chains["experiment_study"],
                prompt_inputs={
                    "water_type_lexicon": water_type_list,
                    "water_type_list": water_type_list,
                    "adsorbent_lexicon": adsorbent_list,
                    "adsorbent_list": adsorbent_list,
                },
                fmap={**_experiment_parse_fields(), **REVIEW_FIELDS},
            )
        else:
            payload = _build_chain_artifact_payload(
                study_id=study_id,
                chain_name="experiment_study",
                task="experiment",
                status="skipped",
                mode="study_level_all_experiment_chunks",
                outputs=[],
                message="classified input not found",
            )
        payload, _summary = _post_process_chain_payload(study_id, "experiment_study", payload)
        path = save_chain_artifact(study_id, "experiment_study", payload)
        artifacts["experiment_study"] = path
        print(f"[{study_id}] Saved experiment_study artifact: {path}")

    if ENABLE_PERFORMANCE and "performance" in run_keys:
        experiment_context_entries: list[dict] = []
        entries = get_classified_entries()
        if entries:
            experiment_context_entries = _select_experiment_context_entries(entries)
            experiment_context = _format_study_chunks(experiment_context_entries)
            print(
                f"[{study_id}] Performance experiment context: "
                f"{len(experiment_context_entries)} classified chunk(s), "
                f"{len(experiment_context)} chars."
            )
        else:
            print(f"[{study_id}] Performance experiment context: no classified input found")
        lexicon_payload = get_lexicon_payload()
        water_type_list = _render_water_type_lexicon(lexicon_payload)
        adsorbent_list = _render_adsorbent_lexicon(lexicon_payload)
        payload = _process_performance_by_chunk(
            study_id,
            llm,
            chains["performance_value"],
            chains["performance_unit"],
            "performance",
            {
                "experiment_context_entries": experiment_context_entries,
                "water_type_lexicon": water_type_list,
                "water_type_list": water_type_list,
                "adsorbent_lexicon": adsorbent_list,
                "adsorbent_list": adsorbent_list,
            },
            lexicon_payload,
            performance_entries_override=performance_entries_override,
        )
        payload, _summary = _post_process_chain_payload(study_id, "performance", payload)
        path = save_chain_artifact(study_id, "performance", payload)
        artifacts["performance"] = path
        print(f"[{study_id}] Saved performance artifact: {path}")

    if artifacts:
        manifest_path = save_artifact_manifest(study_id, artifacts)
        print(f"[{study_id}] Saved artifact manifest: {manifest_path}")

def _study_ids_to_process() -> list[str]:
    return [paths.study_id for paths in iter_active_study_paths()]


def main() -> None:
    print(f"Using {LLM_PROVIDER} model: {LLM_MODEL_NAME}")
    print(
        "Task toggles: "
        f"adsorbent={ENABLE_ADSORBENT}, "
        f"experiment={ENABLE_EXPERIMENT}, "
        f"performance={ENABLE_PERFORMANCE}"
    )

    llm = get_llm(provider=LLM_PROVIDER, model_name=LLM_MODEL_NAME)
    for paths in iter_active_study_paths():
        print(f"\n=== Processing study: {paths.study_id} ({paths.group}) ===")
        try:
            check_stage_input(paths, "extraction")
        except FileNotFoundError as exc:
            print(f"[SKIP] {exc}")
            continue
        process_study(paths.study_id, llm, paths=paths)


if __name__ == "__main__":
    main()
