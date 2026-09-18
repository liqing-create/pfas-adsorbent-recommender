import glob
import json
import logging
import os
import re
import time
import traceback
from pathlib import Path
from typing import Any, List, Union

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
except ImportError:  # Supports ``python enumerate.py`` from this folder.
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

from chains.enumerate_chain import (
    build_study_abstract,
    build_study_chunks,
    create_adsorbent_lexicon_chain,
    create_performance_enumeration_chain,
    create_water_type_lexicon_chain,
)
from chains.llm_config import get_llm
from chains.llm_usage import predict_with_usage
try:
    from .helpers.figure_caption_context import (
        load_caption_index_for_study,
        select_figure_captions_for_text,
    )
    from .helpers.post_process_enumeration import (
        adsorbent_identity_keys as build_adsorbent_identity_keys,
        find_adsorbent_in_pfas_slot,
        find_out_of_scope_adsorbent_slot,
        process_performance_records,
        remove_performance_id_records,
    )
except ImportError:
    from helpers.figure_caption_context import load_caption_index_for_study, select_figure_captions_for_text
    from helpers.post_process_enumeration import (
        adsorbent_identity_keys as build_adsorbent_identity_keys,
        find_adsorbent_in_pfas_slot,
        find_out_of_scope_adsorbent_slot,
        process_performance_records,
        remove_performance_id_records,
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
        return int(str(raw).strip())
    except ValueError:
        logging.warning("Invalid integer for %s=%r; using %s", name, raw, default)
        return default


CLASSIFIED_DIR = ""
ENUM_OUTPUT_DIR = ""
PROMPTS_DIR = str(CONFIG_PROMPTS_DIR)
ERROR_LOG_ROOT = ""
ARTIFACT_ROOT = ""
LOG_PATH = ""
FIGURE_TABLE_INFO_CSV = str(CONFIG_FIGURE_TABLE_INFO_CSV)


def _configure_study_paths(paths: StudyPaths, *, output_dir: Path | None = None) -> None:
    """Point legacy helper functions at one config-resolved study group."""
    global CLASSIFIED_DIR, ENUM_OUTPUT_DIR, ERROR_LOG_ROOT, ARTIFACT_ROOT, LOG_PATH
    CLASSIFIED_DIR = str(paths.classification_dir)
    run_output_dir = Path(output_dir) if output_dir is not None else paths.enumeration_dir
    ENUM_OUTPUT_DIR = str(run_output_dir)
    ERROR_LOG_ROOT = str(run_output_dir / "_error_logs")
    ARTIFACT_ROOT = str(run_output_dir / "_chain_artifacts")
    LOG_PATH = str(run_output_dir / "log.txt")

LLM_PROVIDER = "openai"  # "openai" or "together"
OPENAI_MODEL = "gpt-4.1-2025-04-14"
TOGETHER_MODEL = "meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8"
LLM_MODELS = {
    "openai": OPENAI_MODEL,
    "together": TOGETHER_MODEL,
}
LLM_MODEL_NAME = LLM_MODELS.get(LLM_PROVIDER, OPENAI_MODEL)

ENABLE_ADSORBENT = env_bool("ENABLE_ADSORBENT", True)
ENABLE_WATER_TYPE = env_bool("ENABLE_WATER_TYPE", True)
ENABLE_PERFORMANCE = env_bool("ENABLE_PERFORMANCE", True)
FORCE_LLM_RERUN = env_bool("FORCE_LLM_RERUN", OVERWRITE_EXISTING_OUTPUTS)

ENFORCE_PERFORMANCE_FILE_GATE = False
PERFORMANCE_ENUM_HEAVY_TABLE_CHAR_THRESHOLD = env_int(
    "PERFORMANCE_ENUM_HEAVY_TABLE_CHAR_THRESHOLD",
    8000,
)
PERFORMANCE_ENUM_MAX_NORMAL_TABLE_CHARS_PER_CALL = env_int(
    "PERFORMANCE_ENUM_MAX_NORMAL_TABLE_CHARS_PER_CALL",
    8000,
)
PERFORMANCE_ENUM_MAX_TABLE_CELL_PROXY_PER_CALL = env_int(
    "PERFORMANCE_ENUM_MAX_TABLE_CELL_PROXY_PER_CALL",
    200,
)
PERFORMANCE_ENUM_SPLIT_ON_FAILURE = env_bool("PERFORMANCE_ENUM_SPLIT_ON_FAILURE", True)
PERFORMANCE_ENUM_VALIDATION_RETRIES = env_int("PERFORMANCE_ENUM_VALIDATION_RETRIES", 1)

NORMALIZER = re.compile(r"\s*[<>]+\s*")
ADSORBENT_SCOPE_KEYS = ("in_scope", "out_of_scope")
ADSORBENT_FIELDS = [
    "Adsorbent_id",
    "Name_Abbreviation",
    "Name_Full",
    "Name_Commercial",
    "Adsorbent_category",
    "Adsorbent_subcategory",
]

def _log_to_file(line: str) -> None:
    os.makedirs(ENUM_OUTPUT_DIR, exist_ok=True)
    with open(LOG_PATH, "a", encoding="utf-8") as lf:
        lf.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} | {line}\n")


def normalize_id(raw: str) -> str:
    cleaned = NORMALIZER.sub("/", str(raw or "").strip())
    return "|".join(p.strip() for p in cleaned.split("|"))


def _is_na_like_slot(value: str) -> bool:
    normalized = re.sub(r"[^a-z0-9]+", "", str(value or "").casefold())
    return normalized in {"", "na", "none", "null"}


def is_no_performance_id(value: Any) -> bool:
    parts = normalize_id(value).split("|")
    head = re.sub(r"[^a-z0-9]+", " ", parts[0].casefold()).strip()
    return head in {"no performance id", "no performance ids"} and all(
        _is_na_like_slot(part) for part in parts[1:]
    )


def is_unit_performance_id(value: Any) -> bool:
    return normalize_id(value).strip().casefold() == "unit"


def _format_exception_traceback(exc: Exception) -> str:
    return "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))


def _is_retryable_performance_split_error(exc: Exception | None) -> bool:
    if exc is None:
        return False
    text = f"{type(exc).__name__}: {exc}".casefold()
    retryable_markers = (
        "context_length_exceeded",
        "context length",
        "maximum context",
        "too many tokens",
        "token limit",
        "prompt is too long",
        "request too large",
    )
    return any(marker in text for marker in retryable_markers)


def _split_labels(val: str) -> List[str]:
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


def _build_chunk_metadata(entries: list[dict]) -> list[dict]:
    metadata: list[dict] = []
    for idx, ent in enumerate(entries or [], start=1):
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


def _data_source_aliases(entries: list[dict]) -> dict[str, str]:
    aliases: dict[str, str] = {}
    for idx, ent in enumerate(entries or [], start=1):
        if not isinstance(ent, dict):
            continue
        source_id = _source_id_for(ent)
        prompt_chunk_number = str(idx)
        for alias in (
            source_id,
            f"Data_Source={source_id}",
            f"source_id={source_id}",
            prompt_chunk_number,
            f"chunk {prompt_chunk_number}",
            f"chunk_{prompt_chunk_number}",
        ):
            aliases[str(alias).strip().casefold()] = prompt_chunk_number
        chunk_id = ent.get("chunk_id")
        if chunk_id is not None:
            chunk_key = str(chunk_id).strip()
            aliases[f"chunk_id={chunk_key}".casefold()] = prompt_chunk_number
    return aliases


def _iter_data_source_tokens(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        tokens: list[str] = []
        for item in value:
            tokens.extend(_iter_data_source_tokens(item))
        return tokens
    if isinstance(value, dict):
        tokens: list[str] = []
        for item in value.values():
            tokens.extend(_iter_data_source_tokens(item))
        return tokens

    text = str(value).strip()
    if not text:
        return []
    tokens = re.findall(
        r"(?:Data_Source|source_id|chunk_id)\s*[:=]\s*([A-Za-z0-9_.-]+)",
        text,
        flags=re.IGNORECASE,
    )
    cleaned = text.strip(" \t\r\n'\"`[](){}")
    tokens.extend(part.strip() for part in re.split(r"\s*(?:;|,)\s*", cleaned) if part.strip())
    return tokens


def _normalize_data_source_token(token: str, aliases: dict[str, str]) -> str:
    cleaned = str(token or "").strip(" \t\r\n'\"`[](){}")
    cleaned = re.sub(
        r"^(?:Data_Source|source_id|chunk_id)\s*[:=]\s*",
        "",
        cleaned,
        flags=re.IGNORECASE,
    ).strip()
    chunk_match = re.match(r"^chunk\s*#?\s*(\S+)$", cleaned, flags=re.IGNORECASE)
    if chunk_match:
        cleaned = chunk_match.group(1).strip()
    return aliases.get(cleaned.casefold(), cleaned)


def _merge_data_sources(*values: Any, entries: list[dict] | None = None) -> str:
    aliases = _data_source_aliases(entries or [])
    merged: list[str] = []
    seen: set[str] = set()
    for value in values:
        for token in _iter_data_source_tokens(value):
            source_id = _normalize_data_source_token(token, aliases)
            key = source_id.casefold()
            if source_id and key not in seen:
                seen.add(key)
                merged.append(source_id)
    return ", ".join(merged)


def _normalize_data_sources(value: Any, entries: list[dict]) -> str:
    data_source = _merge_data_sources(value, entries=entries)
    source_entries = [ent for ent in entries or [] if isinstance(ent, dict)]
    if not data_source and len(source_entries) == 1:
        return "1"
    return data_source


def _normalize_record_data_sources(records: list[dict], entries: list[dict]) -> list[dict]:
    for rec in records:
        if isinstance(rec, dict):
            rec["Data_Source"] = _normalize_data_sources(rec.get("Data_Source"), entries)
    return records


def _normalize_adsorbent_data_sources(
    groups: dict[str, list[dict]],
    entries: list[dict],
) -> dict[str, list[dict]]:
    for records in groups.values():
        _normalize_record_data_sources(records, entries)
    return groups


def _get_ci_value(mapping: dict, *keys: str) -> Any:
    lookup = {str(k).casefold(): v for k, v in mapping.items()}
    for key in keys:
        folded = key.casefold()
        if folded in lookup:
            return lookup[folded]
    return None


def _load_json(path: str):
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def _write_missing_log(study_id: str, missing_path: str) -> None:
    os.makedirs(ENUM_OUTPUT_DIR, exist_ok=True)
    log_path = os.path.join(ENUM_OUTPUT_DIR, "missing_files.log")
    with open(log_path, "a", encoding="utf-8") as logf:
        logf.write(f"{study_id}: missing {missing_path}\n")


def _load_classified_entries(study_id: str) -> list[dict] | None:
    in_file = os.path.join(CLASSIFIED_DIR, f"{study_id}_classified.json")
    if not os.path.exists(in_file):
        _write_missing_log(study_id, in_file)
        print(f"[{study_id}] skipped enumeration tasks: classified input not found")
        return None
    return _load_json(in_file)


def _parse_json_object(raw: str):
    raw = (raw or "").strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        start, end = raw.find("{"), raw.rfind("}")
        if start >= 0 and end > start:
            return json.loads(raw[start:end + 1])
        raise


def parse_enumeration_output(output_text: str, id_field: str) -> List[dict]:
    records: List[dict] = []
    for line in output_text.splitlines():
        match = re.match(r"^\s*\d+\.\s*(.+)$", line.strip())
        if match:
            value = normalize_id(match.group(1))
            if id_field == "Performance_id" and is_no_performance_id(value):
                continue
            if value:
                records.append({id_field: value})
    return records


def _usage_metadata(usage) -> dict:
    if hasattr(usage, "metadata_dict"):
        meta = usage.metadata_dict()
    else:
        meta = {
            "provider": getattr(usage, "provider", LLM_PROVIDER),
            "model_name": getattr(usage, "model_name", LLM_MODEL_NAME),
            "prompt_tokens": getattr(usage, "prompt_tokens", None),
            "completion_tokens": getattr(usage, "completion_tokens", None),
            "total_tokens": getattr(usage, "total_tokens", None),
            "total_cost_usd": getattr(usage, "total_cost_usd", None),
        }
    elapsed = meta.pop("elapsed_s", getattr(usage, "elapsed_s", None))
    meta["duration_s"] = round(float(elapsed or 0.0), 3)
    return meta


def dump_error_context(
    study_id: str,
    chunk_id: str,
    ctx_prompts: list[str],
    ctx_raws: list[Union[str, dict]],
    exc: Exception,
) -> None:
    os.makedirs(ERROR_LOG_ROOT, exist_ok=True)
    fname = f"{study_id}_{chunk_id}_error.txt"
    fpath = os.path.join(ERROR_LOG_ROOT, fname)

    with open(fpath, "a", encoding="utf-8") as fh:
        fh.write("\n" + "=" * 80 + "\n")
        fh.write(f"{study_id} | chunk {chunk_id}\n\n")
        fh.write("--- full traceback ---\n")
        fh.write(_format_exception_traceback(exc))
        fh.write("\n\n")
        fh.write("--- prompts sent ---\n")
        for prompt in ctx_prompts:
            fh.write((prompt or "").rstrip() + "\n" + "-" * 40 + "\n")
        fh.write("\n")
        fh.write("--- raw outputs ---\n")
        for entry in ctx_raws:
            if isinstance(entry, dict):
                raw = entry.get("raw", "")
                duration = entry.get("duration_s")
                fh.write((raw or "").rstrip() + "\n")
                if duration is not None:
                    fh.write(f"[duration_s: {duration:.3f}]\n")
            else:
                fh.write((entry or "").rstrip() + "\n")
            fh.write("-" * 40 + "\n")
        fh.write("\n")

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

def _artifact_study_dir(study_id: str) -> str:
    return os.path.join(ARTIFACT_ROOT, study_id)


def _artifact_path(study_id: str, task: str) -> str:
    return os.path.join(_artifact_study_dir(study_id), f"{study_id}_{task}.json")


def _artifact_record_count(payload: dict, task: str) -> int:
    if task == "adsorbent":
        groups = _normalize_adsorbents(payload.get("adsorbents"))
        return sum(len(groups[key]) for key in ADSORBENT_SCOPE_KEYS)
    if task == "water_type":
        records = payload.get("water_types") or []
        return len(records) if isinstance(records, list) else 0
    if task == "performance":
        total = 0
        for item in payload.get("outputs") or []:
            for enum in item.get("enumerations") or []:
                if enum.get("task") == "performance":
                    records = enum.get("records") or []
                    if isinstance(records, list):
                        total += sum(
                            1
                            for record in records
                            if isinstance(record, dict)
                            and not is_no_performance_id(record.get("Performance_id"))
                            and not is_unit_performance_id(record.get("Performance_id"))
                        )
        return total
    return 0


def _build_task_artifact_payload(
    *,
    study_id: str,
    task: str,
    status: str,
    mode: str,
    message: str = "",
    adsorbents: dict[str, list[dict]] | list[dict] | None = None,
    water_types: list[dict] | None = None,
    outputs: list[dict] | None = None,
    timings: dict | None = None,
    selected_chunk_metadata: list[dict] | None = None,
) -> dict:
    payload = {
        "study_folder": study_id,
        "chain_name": task,
        "task": task,
        "status": status,
        "mode": mode,
        "message": message,
        "created_at_unix": time.time(),
        "provider": LLM_PROVIDER,
        "model_name": LLM_MODEL_NAME,
        "timings": timings or {},
        "selected_chunk_metadata": selected_chunk_metadata or [],
    }
    if adsorbents is not None:
        payload["adsorbents"] = _normalize_adsorbents(adsorbents)
    if water_types is not None:
        payload["water_types"] = water_types
    if outputs is not None:
        payload["outputs"] = outputs
    payload["record_count"] = _artifact_record_count(payload, task)
    payload["output_count"] = len(outputs or [])
    return payload


def save_task_artifact(study_id: str, task: str, payload: dict) -> str:
    os.makedirs(_artifact_study_dir(study_id), exist_ok=True)
    path = _artifact_path(study_id, task)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, ensure_ascii=False)
    return path


def load_task_artifact(study_id: str, task: str) -> dict:
    return _load_json(_artifact_path(study_id, task))


def _load_reusable_artifact(study_id: str, task: str) -> dict | None:
    path = _artifact_path(study_id, task)
    if not os.path.exists(path):
        return None
    payload = load_task_artifact(study_id, task)
    status = str(payload.get("status", "")).strip().lower()
    if status in {"ran", "skipped", "ran_with_errors"}:
        print(f"[{study_id}] Reusing {task} artifact; skipping LLM: {path}")
        return payload
    print(f"[{study_id}] Ignoring non-terminal {task} artifact with status={status!r}: {path}")
    return None


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


def _render_adsorbent_list(payload: dict) -> str:
    groups = _normalize_adsorbents((payload or {}).get("adsorbents"))
    sections: list[str] = []
    for key, title in (
        ("in_scope", "In-scope adsorbents"),
        ("out_of_scope", "Out-of-scope adsorbents (do not enumerate Performance_id records for these)"),
    ):
        lines: list[str] = []
        for item in groups[key]:
            parts = []
            for field in ADSORBENT_FIELDS:
                value = str(item.get(field) or "").strip()
                if value:
                    parts.append(f"{field}={value}")
            if parts:
                lines.append("- " + "; ".join(parts))
        body = "\n".join(lines) if lines else "(no entries)"
        sections.append(f"{title}:\n{body}")
    return "\n\n".join(sections)


def _format_adsorbent_in_pfas_slot_feedback(violations: list[dict[str, str]]) -> str:
    examples = []
    for violation in violations[:10]:
        examples.append(
            f"{violation['Performance_id']} "
            f"(first slot {violation['PFAS_slot']!r} matches {violation['matched_adsorbent']})"
        )
    if len(violations) > len(examples):
        examples.append(f"... and {len(violations) - len(examples)} more")
    bad_ids = "; ".join(examples)
    return (
        "Your previous output failed Performance_id validation. The first slot of "
        "Performance_id must be PFAS_name, but these candidate IDs put a known "
        f"adsorbent from Adsorbent_list in that slot: {bad_ids}. Re-read the "
        "source rows and return the complete corrected JSON. If a row has no "
        "assignable PFAS because the PFAS cell is blank, corrupted, or ambiguous "
        "in a multi-PFAS table, omit that record entirely. Do not output "
        "<Adsorbent_id>|NA|<Test_mode>|<Differentiating_Condition>."
    )


def _adsorbent_list_with_validation_feedback(adsorbent_list: str, feedback: str) -> str:
    if not feedback:
        return adsorbent_list
    return (
        f"{adsorbent_list}\n\n"
        "Validation feedback from previous attempt (not source evidence):\n"
        f"{feedback}"
    )


def _render_water_type_list(payload: dict) -> str:
    lines: list[str] = []
    for item in payload.get("water_types") or []:
        if not isinstance(item, dict):
            continue
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
    return "\n".join(lines) if lines else "(no entries)"


def _run_study_json_task(
    *,
    study_id: str,
    task: str,
    label: str,
    output_key: str,
    entries: list[dict],
    chain,
) -> dict:
    mode = f"study_level_all_{task}_chunks"
    selected = [
        ch for ch in entries
        if isinstance(ch, dict) and not _is_irrelevant(ch) and _is_labeled_chunk(ch, label)
    ]
    abstract = build_study_abstract(entries)
    paper_chunks = build_study_chunks(selected)
    if not paper_chunks:
        # The abstract often names and categorizes the study adsorbent even when
        # no chunk is labeled "adsorbent" (for example, when the classifier
        # requires property information that the study does not report).
        if task == "adsorbent" and output_key == "adsorbents" and abstract:
            mode = "study_level_adsorbent_abstract_only"
            print(
                f"[{study_id}] no adsorbent-labeled chunks found; "
                "enumerating adsorbents from the abstract only"
            )
        else:
            message = f"no {label}-labeled chunks found"
            print(f"[{study_id}] skipped {task}: {message}")
            return _build_task_artifact_payload(
                study_id=study_id,
                task=task,
                status="skipped",
                mode=mode,
                message=message,
                adsorbents=_empty_adsorbents() if output_key == "adsorbents" else None,
                water_types=[] if output_key == "water_types" else None,
                selected_chunk_metadata=_build_chunk_metadata(selected),
            )

    prompt = ""
    raw = ""
    try:
        prompt = chain.prompt.format_prompt(
            abstract=abstract,
            paper_chunks=paper_chunks,
        ).to_string()
        print(f"[{study_id}][{task}] PROMPT:\n{prompt}\n")
        call = predict_with_usage(
            chain,
            provider=LLM_PROVIDER,
            model_name=LLM_MODEL_NAME,
            abstract=abstract,
            paper_chunks=paper_chunks,
        )
        raw = call.text
        usage = call.usage
        print(f"[{study_id}][{task}] RAW OUTPUT:\n{raw}\n")
        print(f"[{study_id}][{task}]  -> prompt tokens:     {usage.prompt_tokens}")
        print(f"[{study_id}][{task}]  -> completion tokens: {usage.completion_tokens}")
        print(f"[{study_id}][{task}]  -> total tokens:      {usage.total_tokens}\n")

        parsed = _parse_json_object(raw)
        if not isinstance(parsed, dict):
            raise ValueError(f"{task} output must be a JSON object")
        if output_key == "adsorbents":
            records = _normalize_adsorbents(parsed.get(output_key))
            records = _normalize_adsorbent_data_sources(records, selected)
        else:
            records = parsed.get(output_key) or []
            if not isinstance(records, list):
                raise ValueError(f"{task}.{output_key} must be a list")
            records = _normalize_record_data_sources(records, selected)
        timings = {
            task: {
                "token_counts": [_usage_metadata(usage)],
            }
        }
        return _build_task_artifact_payload(
            study_id=study_id,
            task=task,
            status="ran",
            mode=mode,
            adsorbents=records if output_key == "adsorbents" else None,
            water_types=records if output_key == "water_types" else None,
            timings=timings,
            selected_chunk_metadata=_build_chunk_metadata(selected),
        )
    except Exception as exc:
        logging.error("[%s] %s enumeration failed", study_id, task, exc_info=True)
        dump_error_context(study_id, task, [prompt], [{"raw": raw}], exc)
        return _build_task_artifact_payload(
            study_id=study_id,
            task=task,
            status="failed",
            mode=mode,
            message=str(exc),
            adsorbents=_empty_adsorbents() if output_key == "adsorbents" else None,
            water_types=[] if output_key == "water_types" else None,
            selected_chunk_metadata=_build_chunk_metadata(selected),
        )


def _has_performance(chunks: list[dict]) -> bool:
    return any(
        isinstance(ch, dict)
        and (
            (
                isinstance(ch.get("annotation"), list)
                and "performance" in [
                    str(lbl).strip().lower()
                    for lbl in ch.get("annotation", [])
                    if isinstance(lbl, str)
                ]
            )
            or ("performance" in _split_labels(str(ch.get("predicted_label", ""))))
        )
        for ch in (chunks or [])
    )


def _chunk_text(ch: dict) -> str:
    return str((ch or {}).get("enriched_text") or (ch or {}).get("text") or "").strip()


def _chunk_char_count(ch: dict) -> int:
    return len(_chunk_text(ch))


def _sum_chunk_chars(entries: list[dict]) -> int:
    return sum(_chunk_char_count(ent) for ent in entries or [])


def _chunk_table_char_count(ch: dict) -> int:
    if str((ch or {}).get("file_type") or "").strip().lower() != "table":
        return 0
    return _chunk_char_count(ch)


def _sum_table_chars(entries: list[dict]) -> int:
    return sum(_chunk_table_char_count(ent) for ent in entries or [])


def _split_markdown_table_row(line: str) -> list[str]:
    """Split a pipe-table row without treating escaped pipes as delimiters."""
    value = str(line or "").strip()
    if value.startswith("|"):
        value = value[1:]
    if value.endswith("|"):
        value = value[:-1]

    cells: list[str] = []
    buffer: list[str] = []
    escaped = False
    for character in value:
        if escaped:
            buffer.append(character)
            escaped = False
        elif character == "\\":
            buffer.append(character)
            escaped = True
        elif character == "|":
            cells.append("".join(buffer).strip())
            buffer = []
        else:
            buffer.append(character)
    cells.append("".join(buffer).strip())
    return cells


def _table_cell_metrics_from_text(text: str) -> dict[str, int]:
    pipe_rows = [
        line
        for line in (text or "").splitlines()
        if line.lstrip().startswith("|") and "|" in line.lstrip()[1:]
    ]
    columns = max(
        (len(_split_markdown_table_row(row)) for row in pipe_rows),
        default=0,
    )
    return {
        "pipe_rows": len(pipe_rows),
        "columns": columns,
        "cell_proxy": max(0, len(pipe_rows) - 2) * columns,
    }


def _chunk_table_cell_proxy(ch: dict) -> int:
    if str((ch or {}).get("file_type") or "").strip().lower() != "table":
        return 0
    return _table_cell_metrics_from_text(_chunk_text(ch)).get("cell_proxy", 0)


def _sum_table_cell_proxy(entries: list[dict]) -> int:
    return sum(_chunk_table_cell_proxy(ent) for ent in entries or [])


def _is_heavy_performance_table_for_enumeration(ch: dict) -> bool:
    if PERFORMANCE_ENUM_HEAVY_TABLE_CHAR_THRESHOLD <= 0:
        return False
    if str((ch or {}).get("file_type") or "").strip().lower() != "table":
        return False
    return _chunk_char_count(ch) > PERFORMANCE_ENUM_HEAVY_TABLE_CHAR_THRESHOLD


def _batch_normal_performance_entries(entries: list[dict]) -> list[list[dict]]:
    """Keep grouped normal prompts below table-cell and table-text budgets."""
    if not entries:
        return []
    max_table_chars = PERFORMANCE_ENUM_MAX_NORMAL_TABLE_CHARS_PER_CALL
    max_table_cell_proxy = PERFORMANCE_ENUM_MAX_TABLE_CELL_PROXY_PER_CALL
    if max_table_chars <= 0 and max_table_cell_proxy <= 0:
        return [entries]

    batches: list[list[dict]] = []
    current: list[dict] = []
    current_table_chars = 0
    current_table_cell_proxy = 0
    for ent in entries:
        ent_table_chars = _chunk_table_char_count(ent)
        ent_table_cell_proxy = _chunk_table_cell_proxy(ent)
        chars_over = (
            max_table_chars > 0
            and current_table_chars + ent_table_chars > max_table_chars
        )
        proxy_over = (
            max_table_cell_proxy > 0
            and current_table_cell_proxy + ent_table_cell_proxy > max_table_cell_proxy
        )
        if current and (chars_over or proxy_over):
            batches.append(current)
            current = []
            current_table_chars = 0
            current_table_cell_proxy = 0
        current.append(ent)
        current_table_chars += ent_table_chars
        current_table_cell_proxy += ent_table_cell_proxy

    if current:
        batches.append(current)
    return batches


def _format_performance_source(entries: list[dict]) -> str:
    with_text = [ent for ent in entries or [] if _chunk_text(ent)]
    return build_study_chunks(with_text)


def _performance_json_records(raw: str) -> list[Any]:
    parsed = _parse_json_object(raw)
    if isinstance(parsed, list):
        return parsed
    if not isinstance(parsed, dict):
        raise ValueError("performance output must be a JSON object")
    if _get_ci_value(parsed, "Performance_id") is not None:
        return [parsed]
    records = _get_ci_value(parsed, "Records")
    if records is None:
        raise ValueError("performance output JSON must contain a Records list")
    if not isinstance(records, list):
        raise ValueError("performance Records must be a list")
    return records


def _add_performance_record(
    records_by_id: dict[str, dict],
    order: list[str],
    performance_id: str,
    data_source: str,
) -> None:
    existing = records_by_id.get(performance_id)
    if existing is not None:
        existing["Data_Source"] = _merge_data_sources(existing.get("Data_Source"), data_source)
        return
    records_by_id[performance_id] = {
        "Performance_id": performance_id,
        "Data_Source": data_source,
    }
    order.append(performance_id)


def _parse_performance_records(raw: str, entries: list[dict]) -> list[dict]:
    records_by_id: dict[str, dict] = {}
    order: list[str] = []
    try:
        raw_records = _performance_json_records(raw)
    except json.JSONDecodeError:
        raw_records = parse_enumeration_output(raw, "Performance_id")

    for item in raw_records:
        if isinstance(item, dict):
            value = _get_ci_value(item, "Performance_id")
            raw_source = _get_ci_value(item, "Data_Source")
        else:
            value = item
            raw_source = ""
        norm = normalize_id(value)
        if is_no_performance_id(norm) or is_unit_performance_id(norm):
            continue
        if norm:
            data_source = _normalize_data_sources(raw_source, entries)
            _add_performance_record(records_by_id, order, norm, data_source)
    return [records_by_id[performance_id] for performance_id in order]


def _enumerate_performance_source(
    *,
    study_id: str,
    entries: list[dict],
    chain,
    caption_index,
    adsorbent_list: str,
    adsorbent_identity_keys: dict[str, str] | None,
    out_of_scope_adsorbent_identity_keys: dict[str, str] | None,
    water_type_list: str,
    mode: str,
    output_file_type: str,
    output_chunk_id: Any,
    output_predicted_label: str = "performance",
) -> tuple[dict | None, list[str], list[Union[str, dict]], Exception | None]:
    ctx_prompts: list[str] = []
    ctx_raws: list[Union[str, dict]] = []
    source_text = _format_performance_source(entries)
    if not source_text:
        return None, ctx_prompts, ctx_raws, ValueError(f"{output_chunk_id}: no text found")

    try:
        figure_captions = select_figure_captions_for_text(source_text, caption_index)
        max_attempts = max(1, PERFORMANCE_ENUM_VALIDATION_RETRIES + 1)
        uses_validation_feedback_var = "validation_feedback" in str(
            getattr(chain.prompt, "template", "")
        )
        prompt_input_variables = set(getattr(chain.prompt, "input_variables", []) or [])
        accepts_validation_feedback = (
            uses_validation_feedback_var
            or "validation_feedback" in prompt_input_variables
        )
        validation_feedback = ""
        validation_retry_count = 0
        validation_removed: list[str] = []
        out_of_scope_removed: list[str] = []
        timings: list[dict] = []
        records: list[dict] = []
        post_changed = False
        removed_conditions: list[str] = []
        invalid_removed: list[str] = []

        for attempt_idx in range(1, max_attempts + 1):
            adsorbent_list_for_prompt = (
                adsorbent_list
                if uses_validation_feedback_var
                else _adsorbent_list_with_validation_feedback(adsorbent_list, validation_feedback)
            )
            prompt_kwargs = {
                "enriched_text": source_text,
                "figure_captions": figure_captions or "",
                "adsorbent_list": adsorbent_list_for_prompt,
                "water_type_list": water_type_list,
            }
            if accepts_validation_feedback:
                prompt_kwargs["validation_feedback"] = validation_feedback
            attempt_label = (
                f"{output_chunk_id}"
                if attempt_idx == 1
                else f"{output_chunk_id} validation_retry_{attempt_idx - 1}"
            )
            list_prompt = chain.prompt.format_prompt(**prompt_kwargs).to_string()
            print(f"[{study_id}][performance] LIST PROMPT {attempt_label} mode={mode}:\n{list_prompt}\n")
            ctx_prompts.append(list_prompt)

            call = predict_with_usage(
                chain,
                provider=LLM_PROVIDER,
                model_name=LLM_MODEL_NAME,
                **prompt_kwargs,
            )
            raw = call.text
            usage = call.usage
            token_info = _usage_metadata(usage)
            token_info["attempt"] = attempt_idx
            timings.append(token_info)
            ctx_raws.append({"raw": raw, "duration_s": token_info["duration_s"]})
            print(f"[{study_id}][performance] LIST RAW OUTPUT {attempt_label}:\n{raw}\n")
            print(f"[{study_id}][performance]  -> prompt tokens:     {usage.prompt_tokens}")
            print(f"[{study_id}][performance]  -> completion tokens: {usage.completion_tokens}")
            print(f"[{study_id}][performance]  -> total tokens:      {usage.total_tokens}\n")

            records = _parse_performance_records(raw, entries)
            post_changed, removed_conditions, invalid_removed = process_performance_records(records)
            if post_changed:
                print(
                    f"[{study_id}][performance] post-processed {attempt_label}; "
                    f"removed zero-concentration conditions: {removed_conditions}; "
                    f"invalid removed: {invalid_removed}"
                )

            violations = find_adsorbent_in_pfas_slot(
                records,
                adsorbent_identity_keys or {},
            )
            if violations and attempt_idx < max_attempts:
                validation_retry_count += 1
                validation_feedback = _format_adsorbent_in_pfas_slot_feedback(violations)
                print(
                    f"[{study_id}][performance] validation failed for {attempt_label}; "
                    f"retrying with feedback: {validation_feedback}"
                )
                continue
            if violations:
                validation_feedback = _format_adsorbent_in_pfas_slot_feedback(violations)
                validation_removed = remove_performance_id_records(records, violations)
                post_changed = post_changed or bool(validation_removed)
                print(
                    f"[{study_id}][performance] validation failed after retries for "
                    f"{attempt_label}; removed invalid IDs: {validation_removed}"
                )
            break

        out_of_scope_violations = find_out_of_scope_adsorbent_slot(
            records,
            out_of_scope_adsorbent_identity_keys or {},
        )
        if out_of_scope_violations:
            out_of_scope_removed = remove_performance_id_records(records, out_of_scope_violations)
            post_changed = post_changed or bool(out_of_scope_removed)
            print(
                f"[{study_id}][performance] removed out-of-scope adsorbent IDs for "
                f"{output_chunk_id}: {out_of_scope_removed}"
            )

        post_processing = {}
        if post_changed:
            post_processing.update(
                {
                    "changed": True,
                    "removed_conditions": removed_conditions,
                    "invalid_removed": invalid_removed,
                }
            )
        if validation_retry_count or validation_removed:
            post_processing["validation_retry_count"] = validation_retry_count
            post_processing["adsorbent_in_pfas_slot_removed"] = validation_removed
            if validation_feedback:
                post_processing["validation_feedback"] = validation_feedback
        if out_of_scope_removed:
            post_processing["out_of_scope_adsorbent_removed"] = out_of_scope_removed

        combined = {
            "study_folder": study_id,
            "file_type": output_file_type,
            "chunk_id": output_chunk_id,
            "enriched_text": source_text,
            "enriched_text_metadata": _build_chunk_metadata(entries),
            "predicted_label": output_predicted_label,
            "annotation": [],
            "enumerations": [
                {
                    "task": "performance",
                    "mode": mode,
                    "extracted_count": len(records),
                    "records": records,
                    **({"post_processing": post_processing} if post_processing else {}),
                    "timings": {
                        "lists": timings,
                    },
                }
            ],
        }
        return combined, ctx_prompts, ctx_raws, None
    except Exception as exc:
        return None, ctx_prompts, ctx_raws, exc


def _run_performance_source_with_fallback(
    *,
    study_id: str,
    entries: list[dict],
    chain,
    caption_index,
    adsorbent_list: str,
    adsorbent_identity_keys: dict[str, str] | None,
    out_of_scope_adsorbent_identity_keys: dict[str, str] | None,
    water_type_list: str,
    mode: str,
    output_file_type: str,
    output_chunk_id: Any,
    output_predicted_label: str = "performance",
) -> tuple[list[dict], list[str]]:
    output, prompts, raws, exc = _enumerate_performance_source(
        study_id=study_id,
        entries=entries,
        chain=chain,
        caption_index=caption_index,
        adsorbent_list=adsorbent_list,
        adsorbent_identity_keys=adsorbent_identity_keys,
        out_of_scope_adsorbent_identity_keys=out_of_scope_adsorbent_identity_keys,
        water_type_list=water_type_list,
        mode=mode,
        output_file_type=output_file_type,
        output_chunk_id=output_chunk_id,
        output_predicted_label=output_predicted_label,
    )
    if exc is None:
        return ([output] if output is not None else []), []

    if (
        PERFORMANCE_ENUM_SPLIT_ON_FAILURE
        and len(entries) > 1
        and _is_retryable_performance_split_error(exc)
    ):
        mid = max(1, len(entries) // 2)
        left_entries = entries[:mid]
        right_entries = entries[mid:]
        print(
            f"[{study_id}][performance] {output_chunk_id} failed; "
            f"retrying as {len(left_entries)} + {len(right_entries)} chunks"
        )
        left_outputs, left_errors = _run_performance_source_with_fallback(
            study_id=study_id,
            entries=left_entries,
            chain=chain,
            caption_index=caption_index,
            adsorbent_list=adsorbent_list,
            adsorbent_identity_keys=adsorbent_identity_keys,
            out_of_scope_adsorbent_identity_keys=out_of_scope_adsorbent_identity_keys,
            water_type_list=water_type_list,
            mode=f"{mode}_fallback_split",
            output_file_type=output_file_type,
            output_chunk_id=f"{output_chunk_id}_part1",
            output_predicted_label=output_predicted_label,
        )
        right_outputs, right_errors = _run_performance_source_with_fallback(
            study_id=study_id,
            entries=right_entries,
            chain=chain,
            caption_index=caption_index,
            adsorbent_list=adsorbent_list,
            adsorbent_identity_keys=adsorbent_identity_keys,
            out_of_scope_adsorbent_identity_keys=out_of_scope_adsorbent_identity_keys,
            water_type_list=water_type_list,
            mode=f"{mode}_fallback_split",
            output_file_type=output_file_type,
            output_chunk_id=f"{output_chunk_id}_part2",
            output_predicted_label=output_predicted_label,
        )
        return left_outputs + right_outputs, left_errors + right_errors

    if PERFORMANCE_ENUM_SPLIT_ON_FAILURE and len(entries) > 1:
        print(
            f"[{study_id}][performance] {output_chunk_id} failed without a retryable "
            f"split signal; not splitting. Error: {type(exc).__name__}: {exc}"
        )

    logging.error(
        "[%s] performance source %r failed: %s: %s",
        study_id,
        output_chunk_id,
        type(exc).__name__,
        exc,
    )
    dump_error_context(study_id, str(output_chunk_id), prompts, raws, exc)
    return [], [str(output_chunk_id)]


def _run_performance_task(
    *,
    study_id: str,
    entries: list[dict],
    chain,
    adsorbent_payload: dict,
    water_type_payload: dict,
) -> dict:
    mode = "hybrid_performance_normal_grouped_heavy_table"
    has_performance = _has_performance(entries)
    if ENFORCE_PERFORMANCE_FILE_GATE and not has_performance:
        message = "no predicted_label=performance"
        _log_to_file(f"{study_id} skipped performance (gate on): {message}")
        print(f"[{study_id}] skipped performance (gate on): {message}")
        return _build_task_artifact_payload(
            study_id=study_id,
            task="performance",
            status="skipped",
            mode=mode,
            message=message,
            outputs=[],
        )
    if not ENFORCE_PERFORMANCE_FILE_GATE and not has_performance:
        _log_to_file(f"{study_id} proceeding with performance (gate off): no predicted_label=performance")
        print(f"[{study_id}] proceeding with performance (gate off): no predicted_label=performance")

    selected = [
        ch for ch in entries
        if isinstance(ch, dict) and not _is_irrelevant(ch) and _is_labeled_chunk(ch, "performance")
    ]
    if not selected:
        message = "no performance-labeled chunks found"
        print(f"[{study_id}] skipped performance: {message}")
        return _build_task_artifact_payload(
            study_id=study_id,
            task="performance",
            status="skipped",
            mode=mode,
            message=message,
            outputs=[],
        )

    caption_index = load_caption_index_for_study(study_id, csv_path=FIGURE_TABLE_INFO_CSV)
    adsorbent_list = _render_adsorbent_list(adsorbent_payload)
    adsorbent_identity_keys = build_adsorbent_identity_keys(
        adsorbent_payload
    )
    out_of_scope_adsorbent_identity_keys = build_adsorbent_identity_keys(
        adsorbent_payload,
        scope_keys=("out_of_scope",),
    )
    water_type_list = _render_water_type_list(water_type_payload)
    outputs: list[dict] = []
    failed_chunks: list[str] = []
    selected = [ch for ch in selected if _chunk_text(ch)]
    if not selected:
        message = "no text found in performance-labeled chunks"
        print(f"[{study_id}] skipped performance: {message}")
        return _build_task_artifact_payload(
            study_id=study_id,
            task="performance",
            status="skipped",
            mode=mode,
            message=message,
            outputs=[],
        )

    normal_entries: list[dict] = []
    heavy_table_entries: list[dict] = []
    blocked_table_entries: list[dict] = []
    for ch in selected:
        if (
            str((ch or {}).get("file_type") or "").strip().lower() == "table"
            and PERFORMANCE_ENUM_MAX_TABLE_CELL_PROXY_PER_CALL > 0
            and _chunk_table_cell_proxy(ch) > PERFORMANCE_ENUM_MAX_TABLE_CELL_PROXY_PER_CALL
        ):
            blocked_table_entries.append(ch)
        elif _is_heavy_performance_table_for_enumeration(ch):
            heavy_table_entries.append(ch)
        else:
            normal_entries.append(ch)

    for ch in blocked_table_entries:
        source_id = _source_id_for(ch)
        proxy = _chunk_table_cell_proxy(ch)
        msg = (
            f"{source_id} blocked - table cell proxy {proxy} exceeds "
            f"{PERFORMANCE_ENUM_MAX_TABLE_CELL_PROXY_PER_CALL}; rerun md_preprocessing "
            "so this table is split before enumeration"
        )
        print(f"[{study_id}] Performance enumeration guard: {msg}")
        failed_chunks.append(msg)

    print(
        f"[{study_id}] Performance enumeration routing: "
        f"{len(normal_entries)} normal chunks "
        f"({_sum_chunk_chars(normal_entries)} chars, {_sum_table_chars(normal_entries)} table chars, "
        f"{_sum_table_cell_proxy(normal_entries)} table proxy), "
        f"{len(heavy_table_entries)} heavy table chunks run separately, "
        f"{len(blocked_table_entries)} dense table chunks blocked. "
        f"Heavy table threshold: table chars > {PERFORMANCE_ENUM_HEAVY_TABLE_CHAR_THRESHOLD}. "
        f"Normal group thresholds: table proxy <= {PERFORMANCE_ENUM_MAX_TABLE_CELL_PROXY_PER_CALL}, "
        f"table chars <= {PERFORMANCE_ENUM_MAX_NORMAL_TABLE_CHARS_PER_CALL}."
    )

    normal_batches = _batch_normal_performance_entries(normal_entries)
    if len(normal_batches) > 1:
        print(
            f"[{study_id}] Performance normal chunks split into "
            f"{len(normal_batches)} grouped LLM calls by proxy/char thresholds."
        )

    for batch_idx, normal_batch in enumerate(normal_batches, start=1):
        batch_count = len(normal_batches)
        output_chunk_id = (
            "performance_normal_all"
            if batch_count == 1
            else f"performance_normal_batch_{batch_idx}"
        )
        batch_mode = (
            "study_level_performance_normal_grouped"
            if batch_count == 1
            else "study_level_performance_normal_batched"
        )
        print(
            f"[{study_id}] Performance normal batch {batch_idx}/{batch_count}: "
            f"{len(normal_batch)} chunks, {_sum_chunk_chars(normal_batch)} chars, "
            f"{_sum_table_chars(normal_batch)} table chars, "
            f"{_sum_table_cell_proxy(normal_batch)} table proxy."
        )
        normal_outputs, normal_errors = _run_performance_source_with_fallback(
            study_id=study_id,
            entries=normal_batch,
            chain=chain,
            caption_index=caption_index,
            adsorbent_list=adsorbent_list,
            adsorbent_identity_keys=adsorbent_identity_keys,
            out_of_scope_adsorbent_identity_keys=out_of_scope_adsorbent_identity_keys,
            water_type_list=water_type_list,
            mode=batch_mode,
            output_file_type="study_level",
            output_chunk_id=output_chunk_id,
            output_predicted_label="performance",
        )
        outputs.extend(normal_outputs)
        failed_chunks.extend(normal_errors)

    for ch in heavy_table_entries:
        chunk_id = ch.get("chunk_id")
        heavy_outputs, heavy_errors = _run_performance_source_with_fallback(
            study_id=study_id,
            entries=[ch],
            chain=chain,
            caption_index=caption_index,
            adsorbent_list=adsorbent_list,
            adsorbent_identity_keys=adsorbent_identity_keys,
            out_of_scope_adsorbent_identity_keys=out_of_scope_adsorbent_identity_keys,
            water_type_list=water_type_list,
            mode="chunk_level_performance_heavy_table",
            output_file_type=str(ch.get("file_type") or "table"),
            output_chunk_id=chunk_id,
            output_predicted_label=str(ch.get("predicted_label") or "performance"),
        )
        outputs.extend(heavy_outputs)
        failed_chunks.extend(heavy_errors)

    status = "ran_with_errors" if failed_chunks else "ran"
    message = "failed chunks: " + "; ".join(failed_chunks) if failed_chunks else ""
    return _build_task_artifact_payload(
        study_id=study_id,
        task="performance",
        status=status,
        mode=mode,
        message=message,
        outputs=outputs,
        selected_chunk_metadata=_build_chunk_metadata(selected),
    )


def _required_tasks() -> list[str]:
    required: list[str] = []
    if ENABLE_ADSORBENT or ENABLE_PERFORMANCE:
        required.append("adsorbent")
    if ENABLE_WATER_TYPE or ENABLE_PERFORMANCE:
        required.append("water_type")
    if ENABLE_PERFORMANCE:
        required.append("performance")
    return required

def _enabled_tasks() -> set[str]:
    enabled: set[str] = set()
    if ENABLE_ADSORBENT:
        enabled.add("adsorbent")
    if ENABLE_WATER_TYPE:
        enabled.add("water_type")
    if ENABLE_PERFORMANCE:
        enabled.add("performance")
    return enabled

def process_study(
    study_id: str,
    llm,
    *,
    paths: StudyPaths | None = None,
    classified_entries_override: list[dict] | None = None,
    output_dir: Path | None = None,
    force_llm_rerun: bool | None = None,
) -> None:
    paths = paths or get_study_paths(study_id)
    _configure_study_paths(paths, output_dir=output_dir)
    required_tasks = _required_tasks()
    rerun_enabled_tasks = FORCE_LLM_RERUN if force_llm_rerun is None else force_llm_rerun
    force_rerun_tasks = _enabled_tasks() if rerun_enabled_tasks else set()
    if not required_tasks:
        print(f"[{study_id}] all enumeration tasks are disabled")
        return

    artifacts: dict[str, str] = {}
    payloads: dict[str, dict] = {}
    run_tasks: set[str] = set()

    for task in required_tasks:
        if task in force_rerun_tasks:
            path = _artifact_path(study_id, task)
            if os.path.exists(path):
                print(f"[{study_id}] Force rerun enabled for {task}; ignoring artifact: {path}")
            else:
                print(f"[{study_id}] Force rerun enabled for {task}; no reusable artifact found")
            run_tasks.add(task)
            continue
        payload = _load_reusable_artifact(study_id, task)
        if payload is None:
            run_tasks.add(task)
        else:
            payloads[task] = payload
            artifacts[task] = _artifact_path(study_id, task)

    classified_entries: list[dict] | None = classified_entries_override
    classified_entries_loaded = classified_entries_override is not None

    def get_classified_entries() -> list[dict] | None:
        nonlocal classified_entries, classified_entries_loaded
        if not classified_entries_loaded:
            classified_entries = _load_classified_entries(study_id)
            classified_entries_loaded = True
        return classified_entries

    if "adsorbent" in run_tasks:
        entries = get_classified_entries()
        if entries:
            chain = create_adsorbent_lexicon_chain(PROMPTS_DIR, llm)
            payload = _run_study_json_task(
                study_id=study_id,
                task="adsorbent",
                label="adsorbent",
                output_key="adsorbents",
                entries=entries,
                chain=chain,
            )
        else:
            payload = _build_task_artifact_payload(
                study_id=study_id,
                task="adsorbent",
                status="skipped",
                mode="study_level_all_adsorbent_chunks",
                message="classified input not found",
                adsorbents=_empty_adsorbents(),
            )
        path = save_task_artifact(study_id, "adsorbent", payload)
        artifacts["adsorbent"] = path
        payloads["adsorbent"] = payload
        print(f"[{study_id}] Saved adsorbent artifact: {path}")

    if "water_type" in run_tasks:
        entries = get_classified_entries()
        if entries:
            chain = create_water_type_lexicon_chain(PROMPTS_DIR, llm)
            payload = _run_study_json_task(
                study_id=study_id,
                task="water_type",
                label="experiment",
                output_key="water_types",
                entries=entries,
                chain=chain,
            )
        else:
            payload = _build_task_artifact_payload(
                study_id=study_id,
                task="water_type",
                status="skipped",
                mode="study_level_all_water_type_chunks",
                message="classified input not found",
                water_types=[],
            )
        path = save_task_artifact(study_id, "water_type", payload)
        artifacts["water_type"] = path
        payloads["water_type"] = payload
        print(f"[{study_id}] Saved water_type artifact: {path}")

    if "performance" in run_tasks:
        entries = get_classified_entries()
        if entries:
            chain = create_performance_enumeration_chain(PROMPTS_DIR, llm)
            payload = _run_performance_task(
                study_id=study_id,
                entries=entries,
                chain=chain,
                adsorbent_payload=payloads.get("adsorbent", {}),
                water_type_payload=payloads.get("water_type", {}),
            )
        else:
            payload = _build_task_artifact_payload(
                study_id=study_id,
                task="performance",
                status="skipped",
                mode="hybrid_performance_normal_grouped_heavy_table",
                message="classified input not found",
                outputs=[],
            )
        path = save_task_artifact(study_id, "performance", payload)
        artifacts["performance"] = path
        payloads["performance"] = payload
        print(f"[{study_id}] Saved performance artifact: {path}")

    if artifacts:
        manifest_path = save_artifact_manifest(study_id, artifacts)
        print(f"[{study_id}] Saved artifact manifest: {manifest_path}")

def _study_ids_to_process() -> list[str]:
    return [paths.study_id for paths in iter_active_study_paths()]


def main() -> None:
    global LLM_MODEL_NAME
    if LLM_PROVIDER not in LLM_MODELS:
        raise ValueError(
            f"Unsupported LLM provider: {LLM_PROVIDER}. "
            f"Configured providers: {sorted(LLM_MODELS)}"
        )
    LLM_MODEL_NAME = LLM_MODELS[LLM_PROVIDER]
    print(f"Using {LLM_PROVIDER} model: {LLM_MODEL_NAME}")
    print(
        "Task toggles: "
        f"adsorbent={ENABLE_ADSORBENT}, "
        f"water_type={ENABLE_WATER_TYPE}, "
        f"performance={ENABLE_PERFORMANCE}"
    )
    if ENABLE_PERFORMANCE:
        print("Performance enumeration will reuse or create adsorbent and water_type artifacts.")

    llm = get_llm(provider=LLM_PROVIDER, model_name=LLM_MODEL_NAME)
    for paths in iter_active_study_paths():
        print(f"\n=== Processing study: {paths.study_id} ({paths.group}) ===")
        try:
            check_stage_input(paths, "enumeration")
        except FileNotFoundError as exc:
            print(f"[SKIP] {exc}")
            continue
        process_study(paths.study_id, llm, paths=paths)


if __name__ == "__main__":
    main()
