from __future__ import annotations

import hashlib
import json
import os
import re
import time
from pathlib import Path
from typing import Any

import pandas as pd
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

try:
    from .workflow_config import (
        PROMPTS_DIR as CONFIG_PROMPTS_DIR,
        active_studies_by_group,
        add_project_import_paths,
    )
except ImportError:
    from workflow_config import (
        PROMPTS_DIR as CONFIG_PROMPTS_DIR,
        active_studies_by_group,
        add_project_import_paths,
    )

add_project_import_paths()

from chains.llm_config import get_llm
from chains.llm_usage import predict_with_usage


def coerce_path(raw: str | Path) -> Path:
    text = str(raw or "").strip().strip('"')
    return Path(os.path.expandvars(text)).expanduser()


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
        print(f"[WARN] Invalid integer for {name}={raw!r}; using {default}.")
        return default


PREDICTIONS_DIR = Path()
GROUND_TRUTH_DIR = Path()
PROMPTS_DIR = CONFIG_PROMPTS_DIR

MODEL_NAME = os.getenv("ENUM_EVAL_MODEL_NAME", "")
ENABLE_LLM_JUDGE = env_bool("ENUM_EVAL_ENABLE_LLM_JUDGE", True)
FORCE_LLM_RERUN = env_bool("FORCE_LLM_RERUN", False)
INCREMENTAL_EVAL_UPDATE = env_bool("ENUM_EVAL_INCREMENTAL_UPDATE", True)

LLM_PROVIDER = os.getenv("ENUM_EVAL_LLM_PROVIDER", "openai").strip().lower()
OPENAI_MODEL = os.getenv("ENUM_EVAL_OPENAI_MODEL", "gpt-4.1-2025-04-14")
TOGETHER_MODEL = os.getenv(
    "ENUM_EVAL_TOGETHER_MODEL",
    "meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8",
)
LLM_MODELS = {
    "openai": OPENAI_MODEL,
    "together": TOGETHER_MODEL,
}
LLM_MODEL_NAME = os.getenv(
    "ENUM_EVAL_LLM_MODEL",
    LLM_MODELS.get(LLM_PROVIDER, OPENAI_MODEL),
)
JUDGE_PROMPT_FILE = os.getenv("ENUM_EVAL_JUDGE_PROMPT_FILE", "evaluation_enumeration.j2")
JUDGE_CACHE_KEY_FIELDS = (
    "task",
    "ground_truth_records",
    "predicted_records",
)
JUDGE_ARTIFACT_ROOT = Path()
JUDGE_CALL_STATS = {
    "called": 0,
    "reused": 0,
    "skipped_disabled": 0,
    "skipped_no_candidates": 0,
    "skipped_no_gt": 0,
    "skipped_no_pred": 0,
    "skipped_empty_both": 0,
}

SELECTED_STUDY_IDS: list[str] = []

ALL_TASKS = ("adsorbent", "water_type", "performance")
TASKS = ("adsorbent", "water_type", "performance")
TIME_FIELD = "enumeration_time_s"

ADSORBENT_COLUMNS = [
    "Adsorbent_scope",
    "Adsorbent_id",
    "Name_Abbreviation",
    "Name_Full",
    "Name_Commercial",
    "Adsorbent_category",
    "Adsorbent_subcategory",
]
ADSORBENT_COMPARE_FIELDS = [
    "Name_Abbreviation",
    "Name_Full",
    "Name_Commercial",
    "Adsorbent_category",
    "Adsorbent_subcategory",
]
ADSORBENT_SCOPE_KEYS = ("in_scope", "out_of_scope")
WATER_TYPE_COLUMNS = ["Full_name", "Abbreviation", "Class"]
WATER_TYPE_COMPARE_FIELDS = ["Class"]

NORMALIZER = re.compile(r"\s*[<>]+\s*")
CONDITION_SEPARATOR = re.compile(r"\s*(?:;|&&)\s*")
GENERIC_CONDITION_PREFIX = re.compile(
    r"^(?:matrix|water[\s_-]*matrix)[\s_:-]+",
    re.IGNORECASE,
)
DECIMAL_NUMBER = re.compile(r"(?<![A-Za-z0-9.])(\d+\.\d+)(?![A-Za-z0-9.])")
NUMBER_UNIT_SPACE = re.compile(r"(?<=\d)\s+(?=[A-Za-zμµu°/%])")
ZERO_CONCENTRATION_UNIT_SUFFIX = re.compile(
    r"(?:^|_)0(?:\.0+)?_"
    r"(?:m|mm|um|nm|pm|mol/l|mmol/l|umol/l|nmol/l|g/l|mg/l|ug/l|ng/l|"
    r"ppm|ppb|ppt)$",
    re.IGNORECASE,
)
NUMERIC_CONDITION_VALUE_UNIT_SUFFIX = re.compile(
    r"^(?P<subject>.+)_(?P<value>\d+(?:\.\d+)?)_(?P<unit>[^_]+)$"
)

STRICT_DIMENSION_CONDITION_CATEGORIES = {
    "ph",
    "time",
    "temperature",
    "pfas_initial_concentration",
    "adsorbent_dosage",
    "toc",
    "doc",
}
STRICT_VALUE_CONDITION_CATEGORIES = STRICT_DIMENSION_CONDITION_CATEGORIES | {
    "ionic_strength",
}

def load_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def read_text(path: Path) -> str:
    with open(path, "r", encoding="utf-8") as fh:
        return fh.read().strip()


def create_evaluation_judge_chain(prompts_dir: Path, llm) -> LLMChain:
    raw_tpl = read_text(prompts_dir / JUDGE_PROMPT_FILE)
    prompt = PromptTemplate(
        input_variables=["task", "ground_truth_records", "predicted_records"],
        template=raw_tpl,
        template_format="jinja2",
    )
    return LLMChain(llm=llm, prompt=prompt, verbose=False)


def parse_json_object(raw: str) -> dict[str, Any]:
    raw = (raw or "").strip()
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        start, end = raw.find("{"), raw.rfind("}")
        if start < 0 or end <= start:
            raise
        parsed = json.loads(raw[start : end + 1])
    if not isinstance(parsed, dict):
        raise ValueError("judge output must be a JSON object")
    return parsed


def usage_metadata(usage) -> dict[str, Any]:
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


def stable_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def safe_file_fragment(value: Any, fallback: str = "context") -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value or "").strip())
    cleaned = cleaned.strip("_")
    return (cleaned or fallback)[:80]


def judge_payload_hash(
    *,
    task: str,
    ground_truth_records: list[dict[str, Any]],
    predicted_records: list[dict[str, Any]],
) -> str:
    payload = {
        "task": task,
        "ground_truth_records": ground_truth_records,
        "predicted_records": predicted_records,
    }
    return hashlib.sha256(stable_json(payload).encode("utf-8")).hexdigest()


def accepted_judge_payload_hashes(
    *,
    task: str,
    ground_truth_records: list[dict[str, Any]],
    predicted_records: list[dict[str, Any]],
) -> set[str]:
    hashes = {
        judge_payload_hash(
            task=task,
            ground_truth_records=ground_truth_records,
            predicted_records=predicted_records,
        )
    }

    # Reuse artifacts written before the visible cache-version controls were removed.
    legacy_payload = {
        "artifact_version": 1,
        "cache_key_version": 3 if str(task).strip().lower() == "performance" else 2,
        "task": task,
        "ground_truth_records": ground_truth_records,
        "predicted_records": predicted_records,
    }
    hashes.add(hashlib.sha256(stable_json(legacy_payload).encode("utf-8")).hexdigest())
    return hashes


def judge_artifact_path(
    *,
    study_id: str,
    task: str,
    context_label: str,
) -> Path:
    filename = (
        f"{safe_file_fragment(task)}_"
        f"{safe_file_fragment(context_label)}.json"
    )
    return JUDGE_ARTIFACT_ROOT / safe_file_fragment(study_id, "study") / filename


def save_judge_artifact(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, ensure_ascii=False)


def normalize_id(value: Any) -> str:
    text = NORMALIZER.sub("/", str(value or "").strip())
    return "|".join(part.strip() for part in text.split("|"))


def normalize_for_compare(value: Any) -> str:
    return normalize_id(value).casefold()


def is_na_like_slot(value: Any) -> bool:
    normalized = re.sub(r"[^a-z0-9]+", "", str(value or "").casefold())
    return normalized in {"", "na", "none", "null"}


def is_no_performance_id(value: Any) -> bool:
    parts = normalize_id(value).split("|")
    head = re.sub(r"[^a-z0-9]+", " ", parts[0].casefold()).strip()
    return head in {"no performance id", "no performance ids"} and all(
        is_na_like_slot(part) for part in parts[1:]
    )


def as_number(value: Any) -> float:
    if value is None:
        return 0.0
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.0
    if pd.isna(number):
        return 0.0
    return number


def as_cost(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, str) and not value.strip():
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    try:
        cost = float(value)
    except (TypeError, ValueError):
        return None
    if pd.isna(cost):
        return None
    return cost


def source_id_for(entry: dict[str, Any]) -> str:
    explicit = str(entry.get("source_id") or "").strip()
    if explicit:
        return explicit
    file_type = str(entry.get("file_type") or "unknown").strip()
    chunk_id = normalize_chunk_id(entry.get("chunk_id"))
    return f"{file_type}_{chunk_id}" if chunk_id else file_type


def chunk_key(file_type: Any, chunk_id: Any) -> tuple[str, str]:
    return str(file_type or "unknown").strip(), normalize_chunk_id(chunk_id)


def normalize_chunk_id(value: Any) -> str:
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass

    text = str(value).strip()
    if not text:
        return ""

    match = re.fullmatch(r"([+-]?\d+)\.0+", text)
    if match:
        return str(int(match.group(1)))
    return text


def normalize_chunk_id_column(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "chunk_id" not in df.columns:
        return df
    df = df.copy()
    df["chunk_id"] = df["chunk_id"].map(normalize_chunk_id)
    return df


def chunk_key_from_entry(entry: dict[str, Any]) -> tuple[str, str]:
    return chunk_key(entry.get("file_type"), entry.get("chunk_id"))


def metadata_key(meta: dict[str, Any]) -> tuple[str, str]:
    return chunk_key(meta.get("file_type"), meta.get("chunk_id"))


def dedupe_metadata(metadata: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for meta in metadata or []:
        if not isinstance(meta, dict):
            continue
        key = metadata_key(meta)
        if key == ("unknown", "") or key in seen:
            continue
        seen.add(key)
        rows.append(dict(meta))
    return rows


def output_source_metadata(output: dict[str, Any]) -> list[dict[str, Any]]:
    raw = output.get("enriched_text_metadata")
    if isinstance(raw, list):
        return dedupe_metadata([item for item in raw if isinstance(item, dict)])
    return dedupe_metadata(build_chunk_metadata([output]))


def source_metadata_label(metadata: list[dict[str, Any]]) -> str:
    labels: list[str] = []
    for meta in dedupe_metadata(metadata):
        source_id = str(meta.get("source_id") or source_id_for(meta)).strip()
        if source_id:
            labels.append(source_id)
    return ";".join(labels)


def task_artifact_path(root: Path, study_id: str, task: str) -> Path:
    return root / "_chain_artifacts" / study_id / f"{study_id}_{task}.json"


def manifest_path(root: Path, study_id: str) -> Path:
    return root / "_chain_artifacts" / study_id / "manifest.json"


def empty_payload(study_id: str, task: str, message: str = "artifact not found") -> dict[str, Any]:
    payload: dict[str, Any] = {
        "study_folder": study_id,
        "chain_name": task,
        "artifact_version": None,
        "task": task,
        "status": "missing",
        "mode": "",
        "message": message,
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


def empty_adsorbents() -> dict[str, list[dict[str, Any]]]:
    return {scope: [] for scope in ADSORBENT_SCOPE_KEYS}


def normalize_adsorbents(raw: Any) -> dict[str, list[dict[str, Any]]]:
    if raw is None:
        return empty_adsorbents()
    if isinstance(raw, list):
        return {"in_scope": [record for record in raw if isinstance(record, dict)], "out_of_scope": []}
    if isinstance(raw, dict):
        groups = empty_adsorbents()
        for scope in ADSORBENT_SCOPE_KEYS:
            records = raw.get(scope)
            if isinstance(records, list):
                groups[scope] = [record for record in records if isinstance(record, dict)]
        return groups
    return empty_adsorbents()


def flatten_adsorbents(raw: Any) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for scope, items in normalize_adsorbents(raw).items():
        for item in items:
            record = dict(item)
            record["Adsorbent_scope"] = scope
            records.append(record)
    return records


def count_adsorbents(raw: Any) -> int:
    return len(flatten_adsorbents(raw))


def payload_record_count(payload: dict[str, Any], task: str) -> int:
    if task == "adsorbent":
        return count_adsorbents(payload.get("adsorbents"))
    if task == "performance":
        outputs = payload.get("outputs")
        if isinstance(outputs, list):
            return count_performance_records(outputs)
    return int(payload.get("record_count") or 0)


def count_performance_records(outputs: list[dict[str, Any]]) -> int:
    total = 0
    for output in outputs or []:
        if not isinstance(output, dict):
            continue
        for enum in output.get("enumerations") or []:
            if isinstance(enum, dict) and enum.get("task") == "performance":
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


def build_chunk_metadata(outputs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for output in outputs or []:
        if not isinstance(output, dict):
            continue
        rows.append(
            {
                "study_folder": output.get("study_folder"),
                "file_type": output.get("file_type"),
                "chunk_id": output.get("chunk_id"),
                "source_id": source_id_for(output),
                "predicted_label": output.get("predicted_label"),
            }
        )
    return rows


def load_task_payload(root: Path, study_id: str, task: str) -> tuple[dict[str, Any], Path | None]:
    path = task_artifact_path(root, study_id, task)
    if path.exists():
        payload = load_json(path)
        if isinstance(payload, dict) and task == "adsorbent":
            payload["adsorbents"] = normalize_adsorbents(payload.get("adsorbents"))
            payload["record_count"] = count_adsorbents(payload.get("adsorbents"))
        if isinstance(payload, dict) and task == "performance":
            payload["record_count"] = count_performance_records(payload.get("outputs") or [])
        return payload, path

    return empty_payload(study_id, task), None


def load_manifest(root: Path, study_id: str) -> tuple[dict[str, Any], Path | None]:
    path = manifest_path(root, study_id)
    if path.exists():
        return load_json(path), path
    return {}, None


def discover_studies(*roots: Path, tasks: tuple[str, ...] = ALL_TASKS) -> list[str]:
    studies: set[str] = set()
    for root in roots:
        if not root.exists():
            continue
        artifact_root = root / "_chain_artifacts"
        if artifact_root.exists():
            for child in artifact_root.iterdir():
                if child.is_dir():
                    if (child / "manifest.json").exists():
                        studies.add(child.name)
                        continue
                    if any((child / f"{child.name}_{task}.json").exists() for task in tasks):
                        studies.add(child.name)
    return sorted(studies)


def validate_tasks(tasks: tuple[str, ...] | str) -> tuple[str, ...]:
    raw_tasks = (tasks,) if isinstance(tasks, str) else tasks
    selected: list[str] = []
    for task in raw_tasks:
        normalized = str(task).strip()
        if not normalized:
            continue
        if normalized not in ALL_TASKS:
            raise ValueError(
                f"Unsupported task in TASKS: {normalized!r}. "
                f"Expected one or more of: {', '.join(ALL_TASKS)}"
            )
        if normalized not in selected:
            selected.append(normalized)
    if not selected:
        raise ValueError(f"TASKS must include at least one of: {', '.join(ALL_TASKS)}")
    return tuple(selected)


def selected_studies(tasks: tuple[str, ...]) -> list[str]:
    # The wrapper sets this list from workflow_config for one study group.
    # Do not discover or override unselected studies here.
    return list(SELECTED_STUDY_IDS)


def infer_model_name(payloads: dict[str, dict[str, Any]], manifest: dict[str, Any]) -> str:
    if MODEL_NAME:
        return MODEL_NAME
    if manifest.get("model_name"):
        return str(manifest.get("model_name"))
    for payload in payloads.values():
        if payload.get("model_name"):
            return str(payload.get("model_name"))
    return "unknown_model"


def json_cell(value: Any) -> str:
    if value in (None, ""):
        return ""
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def id_list(values: set[str]) -> str:
    return ";".join(sorted(values))

def metric_values(tp: int, fp: int, fn: int) -> tuple[float | None, float | None, float | None]:
    precision = tp / (tp + fp) if tp + fp else None
    recall = tp / (tp + fn) if tp + fn else None

    if precision is None and recall is None:
        f1 = None
    elif precision is None:
        f1 = 0.0 if fn else None
    elif recall is None:
        f1 = 0.0 if fp else None
    elif precision + recall:
        f1 = 2 * precision * recall / (precision + recall)
    else:
        f1 = 0.0
    return precision, recall, f1

def eval_status(ground_count: int, pred_count: int) -> str:
    if ground_count == 0 and pred_count == 0:
        return "empty_gt_empty_pred"
    if ground_count == 0:
        return "empty_gt_pred_nonempty"
    if pred_count == 0:
        return "gt_nonempty_pred_empty"
    return "gt_nonempty_pred_nonempty"

def indexed_judge_records(
    records_by_key: dict[str, dict[str, Any]],
    keys: list[str],
    record_key_fn=None,
) -> tuple[list[dict[str, Any]], dict[int, str]]:
    judge_records: list[dict[str, Any]] = []
    index_to_key: dict[int, str] = {}
    for index, key in enumerate(keys):
        record = dict(records_by_key[key])
        judge_record = {
            "_judge_index": index,
            **record,
        }
        if record_key_fn is not None:
            record_key = normalize_id(record_key_fn(record))
            if record_key:
                judge_record["_record_key"] = record_key
        judge_records.append(judge_record)
        index_to_key[index] = key
    return judge_records, index_to_key


def judge_result_template(study_id: str, task: str, context_label: str) -> dict[str, Any]:
    return {
        "call_id": f"{study_id}|{task}|{context_label}",
        "provider": "",
        "model_name": "",
        "artifact_path": "",
        "prompt": "",
        "raw": "",
        "parsed": {},
        "usage": {},
        "matches": [],
        "reported_unmatched_gt_indices": [],
        "reported_unmatched_pred_indices": [],
        "ignored_matches": [],
        "error": "",
    }


def int_or_none(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def int_list(values: Any) -> list[int]:
    if not isinstance(values, list):
        return []
    parsed: list[int] = []
    for value in values:
        index = int_or_none(value)
        if index is not None:
            parsed.append(index)
    return parsed


def validate_judge_matches(
    parsed: dict[str, Any],
    *,
    gt_count: int,
    pred_count: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    valid_matches: list[dict[str, Any]] = []
    ignored_matches: list[dict[str, Any]] = []
    used_gt: set[int] = set()
    used_pred: set[int] = set()

    raw_matches = parsed.get("matches") or []
    if not isinstance(raw_matches, list):
        return valid_matches, [{"reason": "matches is not a list", "match": raw_matches}]

    for raw_match in raw_matches:
        if not isinstance(raw_match, dict):
            ignored_matches.append({"reason": "match is not an object", "match": raw_match})
            continue
        gt_index = int_or_none(raw_match.get("gt_index"))
        pred_index = int_or_none(raw_match.get("pred_index"))
        reason = ""
        if gt_index is None or pred_index is None:
            reason = "missing or non-integer index"
        elif gt_index < 0 or gt_index >= gt_count:
            reason = "gt_index out of range"
        elif pred_index < 0 or pred_index >= pred_count:
            reason = "pred_index out of range"
        elif gt_index in used_gt:
            reason = "duplicate gt_index"
        elif pred_index in used_pred:
            reason = "duplicate pred_index"

        if reason:
            ignored_matches.append({"reason": reason, "match": raw_match})
            continue

        used_gt.add(gt_index)
        used_pred.add(pred_index)
        valid_matches.append(
            {
                "gt_index": gt_index,
                "pred_index": pred_index,
            }
        )
    return valid_matches, ignored_matches


def hydrate_judge_result_from_artifact(
    *,
    base_result: dict[str, Any],
    artifact_path: Path,
    artifact: dict[str, Any],
    gt_count: int,
    pred_count: int,
) -> dict[str, Any]:
    parsed = artifact.get("parsed_output") or {}
    if not isinstance(parsed, dict):
        parsed = {}
    matches, ignored = validate_judge_matches(
        parsed,
        gt_count=gt_count,
        pred_count=pred_count,
    )
    base_result.update(
        {
            "provider": artifact.get("provider", ""),
            "model_name": artifact.get("model_name", ""),
            "artifact_path": str(artifact_path),
            "prompt": artifact.get("prompt", ""),
            "raw": artifact.get("raw_output", ""),
            "parsed": parsed,
            "usage": artifact.get("usage") or {},
            "matches": matches,
            "reported_unmatched_gt_indices": int_list(parsed.get("unmatched_gt_indices")),
            "reported_unmatched_pred_indices": int_list(parsed.get("unmatched_pred_indices")),
            "ignored_matches": ignored or artifact.get("ignored_matches") or [],
            "error": artifact.get("error", ""),
        }
    )
    return base_result


def run_semantic_judge(
    *,
    study_id: str,
    task: str,
    context_label: str,
    judge_chain: LLMChain | None,
    ground_truth_records: list[dict[str, Any]],
    predicted_records: list[dict[str, Any]],
) -> dict[str, Any]:
    result = judge_result_template(study_id, task, context_label)
    if not ENABLE_LLM_JUDGE or judge_chain is None:
        JUDGE_CALL_STATS["skipped_disabled"] += 1
        return result
    if not ground_truth_records or not predicted_records:
        JUDGE_CALL_STATS["skipped_no_candidates"] += 1
        if ground_truth_records:
            JUDGE_CALL_STATS["skipped_no_pred"] += 1
        elif predicted_records:
            JUDGE_CALL_STATS["skipped_no_gt"] += 1
        else:
            JUDGE_CALL_STATS["skipped_empty_both"] += 1
        return result

    result["provider"] = LLM_PROVIDER
    result["model_name"] = LLM_MODEL_NAME
    prompt_kwargs = {
        "task": task,
        "ground_truth_records": ground_truth_records,
        "predicted_records": predicted_records,
    }
    prompt = judge_chain.prompt.format_prompt(**prompt_kwargs).to_string()
    payload_hash = judge_payload_hash(
        task=task,
        ground_truth_records=ground_truth_records,
        predicted_records=predicted_records,
    )
    accepted_payload_hashes = accepted_judge_payload_hashes(
        task=task,
        ground_truth_records=ground_truth_records,
        predicted_records=predicted_records,
    )
    artifact_path = judge_artifact_path(
        study_id=study_id,
        task=task,
        context_label=context_label,
    )
    result["artifact_path"] = str(artifact_path)
    result["prompt"] = prompt

    if artifact_path.exists() and not FORCE_LLM_RERUN:
        try:
            artifact = load_json(artifact_path)
            if isinstance(artifact, dict):
                if artifact.get("payload_hash") not in accepted_payload_hashes:
                    print(
                        f"[{study_id}][{task}][judge] Ignoring stale judge artifact "
                        f"(payload hash changed): {artifact_path}"
                    )
                else:
                    JUDGE_CALL_STATS["reused"] += 1
                    print(f"[{study_id}][{task}][judge] Reusing judge artifact: {artifact_path}")
                    return hydrate_judge_result_from_artifact(
                        base_result=result,
                        artifact_path=artifact_path,
                        artifact=artifact,
                        gt_count=len(ground_truth_records),
                        pred_count=len(predicted_records),
                    )
        except Exception as exc:
            print(
                f"[{study_id}][{task}][judge] Could not reuse judge artifact "
                f"{artifact_path}: {exc}; rerunning."
            )

    print(f"[{study_id}][{task}][judge] PROMPT {context_label}:\n{prompt}\n")
    JUDGE_CALL_STATS["called"] += 1
    try:
        call = predict_with_usage(
            judge_chain,
            provider=LLM_PROVIDER,
            model_name=LLM_MODEL_NAME,
            **prompt_kwargs,
        )
        raw = call.text
        usage = call.usage
        usage_info = usage_metadata(usage)
        result["raw"] = raw
        result["usage"] = usage_info
        print(f"[{study_id}][{task}][judge] RAW OUTPUT {context_label}:\n{raw}\n")
        print(f"[{study_id}][{task}][judge]  -> prompt tokens:     {usage.prompt_tokens}")
        print(f"[{study_id}][{task}][judge]  -> completion tokens: {usage.completion_tokens}")
        print(f"[{study_id}][{task}][judge]  -> total tokens:      {usage.total_tokens}\n")

        parsed = parse_json_object(raw)
        matches, ignored = validate_judge_matches(
            parsed,
            gt_count=len(ground_truth_records),
            pred_count=len(predicted_records),
        )
        result.update(
            {
                "raw": raw,
                "parsed": parsed,
                "matches": matches,
                "reported_unmatched_gt_indices": int_list(parsed.get("unmatched_gt_indices")),
                "reported_unmatched_pred_indices": int_list(parsed.get("unmatched_pred_indices")),
                "ignored_matches": ignored,
            }
        )
        if ignored:
            print(
                f"[{study_id}][{task}][judge] Ignored {len(ignored)} invalid match(es) "
                f"for {context_label}: {json_cell(ignored)}"
            )
    except Exception as exc:
        result["error"] = str(exc)
        print(f"[{study_id}][{task}][judge] ERROR {context_label}: {exc}")

    save_judge_artifact(
        artifact_path,
        {
            "created_at_unix": time.time(),
            "call_id": result.get("call_id", ""),
            "study_id": study_id,
            "task": task,
            "context_label": context_label,
            "provider": result.get("provider", ""),
            "model_name": result.get("model_name", ""),
            "prompt_file": str(PROMPTS_DIR / JUDGE_PROMPT_FILE),
            "payload_hash": payload_hash,
            "payload_hash_fields": list(JUDGE_CACHE_KEY_FIELDS),
            "prompt_in_payload_hash": False,
            "force_llm_rerun": FORCE_LLM_RERUN,
            "ground_truth_records": ground_truth_records,
            "predicted_records": predicted_records,
            "prompt": prompt,
            "raw_output": result.get("raw", ""),
            "parsed_output": result.get("parsed") or {},
            "usage": result.get("usage") or {},
            "matches": result.get("matches") or [],
            "reported_unmatched_gt_indices": result.get("reported_unmatched_gt_indices") or [],
            "reported_unmatched_pred_indices": result.get("reported_unmatched_pred_indices") or [],
            "ignored_matches": result.get("ignored_matches") or [],
            "error": result.get("error", ""),
        },
    )
    print(f"[{study_id}][{task}][judge] Saved judge artifact: {artifact_path}")
    return result


def judge_row_fields(
    judge_result: dict[str, Any],
    decision: str,
    *,
    gt_index: int | str = "",
    pred_index: int | str = "",
) -> dict[str, Any]:
    usage = judge_result.get("usage") or {}
    return {
        "judge_decision": decision,
        "judge_call_id": judge_result.get("call_id", ""),
        "judge_provider": judge_result.get("provider", ""),
        "judge_model": judge_result.get("model_name", ""),
        "judge_artifact_path": judge_result.get("artifact_path", ""),
        "judge_gt_index": gt_index,
        "judge_pred_index": pred_index,
        "judge_error": judge_result.get("error", ""),
        "judge_ignored_matches": json_cell(judge_result.get("ignored_matches") or []),
        "judge_reported_unmatched_gt_indices": id_list(
            {str(i) for i in judge_result.get("reported_unmatched_gt_indices") or []}
        ),
        "judge_reported_unmatched_pred_indices": id_list(
            {str(i) for i in judge_result.get("reported_unmatched_pred_indices") or []}
        ),
        "judge_prompt_tokens": usage.get("prompt_tokens"),
        "judge_completion_tokens": usage.get("completion_tokens"),
        "judge_total_tokens": usage.get("total_tokens"),
        "judge_duration_s": usage.get("duration_s"),
        "judge_raw_output": judge_result.get("raw", ""),
    }


def add_usage_call(row: dict[str, Any], call: dict[str, Any], task: str) -> None:
    prompt = as_number(call.get("prompt_tokens"))
    completion = as_number(call.get("completion_tokens"))
    total = as_number(call.get("total_tokens")) or prompt + completion
    elapsed = as_number(call.get("duration_s", call.get("elapsed_s")))
    cost = as_cost(call.get("total_cost_usd"))

    row["llm_call_count"] += 1
    row[f"{task}_llm_call_count"] += 1
    row["prompt_tokens"] += prompt
    row["completion_tokens"] += completion
    row["total_tokens"] += total
    row[TIME_FIELD] += elapsed
    row["chunks_missing_token_usage"] += int(total == 0)
    row["chunks_missing_time_usage"] += int(elapsed == 0)
    if cost is not None:
        row["total_cost_usd"] = as_number(row.get("total_cost_usd")) + cost
        row["_has_cost"] = True


def task_usage_calls(payload: dict[str, Any], task: str) -> list[dict[str, Any]]:
    calls: list[dict[str, Any]] = []
    task_timings = (payload.get("timings") or {}).get(task) or {}
    for call in task_timings.get("token_counts") or []:
        if isinstance(call, dict):
            calls.append(call)

    if task == "performance":
        for output in payload.get("outputs") or []:
            if not isinstance(output, dict):
                continue
            for enum in output.get("enumerations") or []:
                if not isinstance(enum, dict) or enum.get("task") != "performance":
                    continue
                for call in (enum.get("timings") or {}).get("lists") or []:
                    if isinstance(call, dict):
                        calls.append(call)
    return calls


def summarize_usage(
    study_id: str,
    pred_payloads: dict[str, dict[str, Any]],
    pred_paths: dict[str, Path | None],
    manifest: dict[str, Any],
    tasks: tuple[str, ...],
) -> dict[str, Any]:
    model_name = infer_model_name(pred_payloads, manifest)
    row: dict[str, Any] = {
        "model": model_name,
        "study_id": study_id,
        "summary_type": "per_study",
        "has_prediction": any(path is not None for path in pred_paths.values()),
        "llm_output_chunks": 0,
        "llm_call_count": 0,
        "chunks_missing_token_usage": 0,
        "chunks_missing_time_usage": 0,
        "prompt_tokens": 0.0,
        "completion_tokens": 0.0,
        "total_tokens": 0.0,
        TIME_FIELD: 0.0,
        "total_cost_usd": None,
        "_has_cost": False,
    }
    for task in tasks:
        payload = pred_payloads.get(task) or empty_payload(study_id, task)
        selected_count = len(payload.get("selected_chunk_metadata") or [])
        record_count = payload_record_count(payload, task)
        output_count = int(payload.get("output_count") or 0)
        row[f"{task}_llm_call_count"] = 0
        row[f"{task}_selected_chunks"] = selected_count
        row[f"{task}_records"] = record_count
        row[f"{task}_outputs"] = output_count
        row["llm_output_chunks"] += selected_count
        for call in task_usage_calls(payload, task):
            add_usage_call(row, call, task)

    finalize_usage_row(row)
    row.pop("_has_cost", None)
    return row


def finalize_usage_row(row: dict[str, Any]) -> None:
    for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
        row[key] = int(row.get(key) or 0)
    row[TIME_FIELD] = round(as_number(row.get(TIME_FIELD)), 3)
    if row.get("_has_cost"):
        row["total_cost_usd"] = round(as_number(row.get("total_cost_usd")), 8)
    else:
        row["total_cost_usd"] = None

    calls = as_number(row.get("llm_call_count"))
    chunks = as_number(row.get("llm_output_chunks"))
    row["avg_total_tokens_per_llm_chunk"] = round(row["total_tokens"] / chunks, 3) if chunks else 0
    row["avg_time_per_llm_chunk_s"] = round(row[TIME_FIELD] / chunks, 3) if chunks else 0
    row["avg_total_tokens_per_llm_call"] = round(row["total_tokens"] / calls, 3) if calls else 0
    row["avg_time_per_llm_call_s"] = round(row[TIME_FIELD] / calls, 3) if calls else 0


def add_total_usage_row(rows: list[dict[str, Any]], tasks: tuple[str, ...]) -> list[dict[str, Any]]:
    if not rows:
        return rows
    total: dict[str, Any] = {
        "model": rows[0].get("model", ""),
        "study_id": "all_studies",
        "summary_type": "total",
        "has_prediction": any(bool(row.get("has_prediction")) for row in rows),
        "_has_cost": any(row.get("total_cost_usd") not in (None, "") for row in rows),
    }
    fields_to_sum = [
        "llm_output_chunks",
        "llm_call_count",
        "chunks_missing_token_usage",
        "chunks_missing_time_usage",
        "prompt_tokens",
        "completion_tokens",
        "total_tokens",
        TIME_FIELD,
        "total_cost_usd",
    ]
    for task in tasks:
        fields_to_sum.extend(
            [
                f"{task}_llm_call_count",
                f"{task}_selected_chunks",
                f"{task}_records",
                f"{task}_outputs",
            ]
        )
    for field in fields_to_sum:
        if field == "total_cost_usd" and not total["_has_cost"]:
            total[field] = None
        else:
            total[field] = sum(as_number(row.get(field)) for row in rows)
    finalize_usage_row(total)
    total.pop("_has_cost", None)
    return rows + [total]


def record_key_adsorbent(record: dict[str, Any]) -> str:
    adsorbent_id = normalize_for_compare(record.get("Adsorbent_id"))
    scope = normalize_for_compare(record.get("Adsorbent_scope"))
    return f"{scope}|{adsorbent_id}" if adsorbent_id else ""


def display_key_adsorbent(record: dict[str, Any]) -> str:
    adsorbent_id = normalize_id(record.get("Adsorbent_id"))
    scope = normalize_id(record.get("Adsorbent_scope"))
    return f"{scope}|{adsorbent_id}" if scope and adsorbent_id else adsorbent_id


def judge_record_key_adsorbent(record: dict[str, Any]) -> str:
    return normalize_id(record.get("Adsorbent_scope"))


def record_key_water_type(record: dict[str, Any]) -> str:
    full_name = normalize_for_compare(record.get("Full_name"))
    abbreviation = normalize_for_compare(record.get("Abbreviation"))
    return f"{full_name}|{abbreviation}" if full_name or abbreviation else ""


def display_key_water_type(record: dict[str, Any]) -> str:
    full_name = normalize_id(record.get("Full_name"))
    abbreviation = normalize_id(record.get("Abbreviation"))
    return f"{full_name}|{abbreviation}" if full_name or abbreviation else ""


def normalize_equivalence_alias(value: Any) -> str:
    text = NORMALIZER.sub("/", str(value or "").strip())
    text = text.replace("\u00b5", "u").replace("\u03bc", "u")
    text = text.replace("\u00ae", "").replace("\u2122", "").replace("\u00a9", "")
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\s*/\s*", "/", text)
    text = re.sub(r"\s*\^\s*", "^", text)
    text = re.sub(r"\s*-\s*", "-", text)
    return text.strip().casefold()


def alias_key_variants(value: Any) -> set[str]:
    base = normalize_equivalence_alias(value)
    if not base:
        return set()
    variants = {base, base.replace(" ", "_")}
    condition_form = normalize_condition_token(value)
    if condition_form:
        variants.add(condition_form)
    return variants


def lexicon_aliases(record: dict[str, Any], task: str) -> set[str]:
    if task == "adsorbent":
        fields = [
            "Adsorbent_id",
            "Name_Abbreviation",
            "Name_Full",
            "Name_Commercial",
        ]
    elif task == "water_type":
        fields = ["Full_name", "Abbreviation", "Class"]
    else:
        return set()

    aliases: set[str] = set()
    for field in fields:
        raw_value = str(record.get(field) or "").strip()
        if not raw_value:
            continue
        for value in re.split(r"\s*(?:;|\|\|)\s*", raw_value):
            aliases.update(alias_key_variants(value))
    return aliases


def build_lexicon_equivalence_map(
    task: str,
    matched_pairs: list[tuple[str, str]],
    pred_records: dict[str, dict[str, Any]],
    gt_records: dict[str, dict[str, Any]],
) -> dict[str, str]:
    alias_classes: dict[str, set[str]] = {}
    for index, (pred_key, gt_key) in enumerate(matched_pairs):
        pred_record = pred_records.get(pred_key, {})
        gt_record = gt_records.get(gt_key, {})
        aliases = lexicon_aliases(pred_record, task) | lexicon_aliases(gt_record, task)
        if not aliases:
            continue
        class_key = f"__{task}_eq_{index}__"
        for alias in aliases:
            alias_classes.setdefault(alias, set()).add(class_key)

    return {
        alias: next(iter(class_keys))
        for alias, class_keys in alias_classes.items()
        if len(class_keys) == 1
    }


def records_for_task(payload: dict[str, Any], task: str) -> list[dict[str, Any]]:
    if task == "adsorbent":
        return flatten_adsorbents(payload.get("adsorbents"))
    key = "water_types"
    records = payload.get(key) or []
    return [record for record in records if isinstance(record, dict)]


def record_map(
    records: list[dict[str, Any]],
    task: str,
) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    key_fn = record_key_adsorbent if task == "adsorbent" else record_key_water_type
    display_fn = display_key_adsorbent if task == "adsorbent" else display_key_water_type
    mapped: dict[str, dict[str, Any]] = {}
    duplicates = []
    for index, record in enumerate(records, start=1):
        key = key_fn(record)
        if not key:
            continue
        if key in mapped:
            duplicates.append(
                {
                    "row_index": index,
                    "record_key": display_fn(record),
                    "record": json_cell(record),
                }
            )
            continue
        mapped[key] = record
    return mapped, duplicates


def field_agreement_rows(
    *,
    study_id: str,
    model_name: str,
    task: str,
    matched_pairs: list[tuple[str, str]],
    pred_records: dict[str, dict[str, Any]],
    gt_records: dict[str, dict[str, Any]],
    fields: list[str],
) -> list[dict[str, Any]]:
    rows = []
    for field in fields:
        exact = 0
        gt_nonblank = 0
        exact_when_gt_nonblank = 0
        pred_nonblank = 0
        for pred_key, gt_key in matched_pairs:
            pred_value = normalize_for_compare(pred_records[pred_key].get(field))
            gt_value = normalize_for_compare(gt_records[gt_key].get(field))
            exact += int(pred_value == gt_value)
            gt_nonblank += int(bool(gt_value))
            pred_nonblank += int(bool(pred_value))
            exact_when_gt_nonblank += int(bool(gt_value) and pred_value == gt_value)
        matched = len(matched_pairs)
        rows.append(
            {
                "model": model_name,
                "study": study_id,
                "task": task,
                "field": field,
                "matched_records": matched,
                "exact_matches": exact,
                "mismatches": matched - exact,
                "gt_nonblank": gt_nonblank,
                "pred_nonblank": pred_nonblank,
                "exact_match_rate": exact / matched if matched else 0.0,
                "exact_match_rate_when_gt_nonblank": (
                    exact_when_gt_nonblank / gt_nonblank if gt_nonblank else 0.0
                ),
            }
        )
    return rows


def evaluate_lexicon_task(
    *,
    study_id: str,
    task: str,
    model_name: str,
    pred_payload: dict[str, Any],
    gt_payload: dict[str, Any],
    judge_chain: LLMChain | None = None,
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    dict[str, str],
]:
    pred_records, pred_duplicates = record_map(records_for_task(pred_payload, task), task)
    gt_records, gt_duplicates = record_map(records_for_task(gt_payload, task), task)
    display_fn = display_key_adsorbent if task == "adsorbent" else display_key_water_type
    fields = ADSORBENT_COMPARE_FIELDS if task == "adsorbent" else WATER_TYPE_COMPARE_FIELDS

    pred_keys = set(pred_records)
    gt_keys = set(gt_records)
    ground_count = len(gt_keys)
    pred_count = len(pred_keys)
    exact_matched = pred_keys & gt_keys
    exact_fp = pred_keys - gt_keys
    exact_fn = gt_keys - pred_keys

    fn_keys = sorted(exact_fn)
    fp_keys = sorted(exact_fp)
    judge_record_key_fn = judge_record_key_adsorbent if task == "adsorbent" else None
    gt_judge_records, gt_index_to_key = indexed_judge_records(
        gt_records,
        fn_keys,
        judge_record_key_fn,
    )
    pred_judge_records, pred_index_to_key = indexed_judge_records(
        pred_records,
        fp_keys,
        judge_record_key_fn,
    )
    gt_key_to_index = {key: index for index, key in gt_index_to_key.items()}
    pred_key_to_index = {key: index for index, key in pred_index_to_key.items()}
    judge_result = run_semantic_judge(
        study_id=study_id,
        task=task,
        context_label="study",
        judge_chain=judge_chain,
        ground_truth_records=gt_judge_records,
        predicted_records=pred_judge_records,
    )

    semantic_pairs: list[tuple[str, str, dict[str, Any]]] = []
    semantic_pred_keys: set[str] = set()
    semantic_gt_keys: set[str] = set()
    for match in judge_result.get("matches") or []:
        gt_key = gt_index_to_key[match["gt_index"]]
        pred_key = pred_index_to_key[match["pred_index"]]
        semantic_pairs.append((pred_key, gt_key, match))
        semantic_pred_keys.add(pred_key)
        semantic_gt_keys.add(gt_key)

    fp = exact_fp - semantic_pred_keys
    fn = exact_fn - semantic_gt_keys
    matched_pairs = [(key, key) for key in sorted(exact_matched)] + [
        (pred_key, gt_key) for pred_key, gt_key, _ in semantic_pairs
    ]
    equivalence_map = build_lexicon_equivalence_map(
        task,
        matched_pairs,
        pred_records,
        gt_records,
    )

    tp_n = len(matched_pairs)
    fp_n = len(fp)
    fn_n = len(fn)
    precision, recall, f1 = metric_values(tp_n, fp_n, fn_n)
    metric_rows = [
        {
            "model": model_name,
            "study": study_id,
            "task": task,
            "granularity": "study",
            "file_type": "",
            "chunk_id": "",
            "source_id": "",
            "ground_count": ground_count,
            "pred_count": pred_count,
            "eval_status": eval_status(ground_count, pred_count),
            "tp": tp_n,
            "fp": fp_n,
            "fn": fn_n,
            "exact_tp": len(exact_matched),
            "semantic_tp": len(semantic_pairs),
            "tp_ids": id_list({display_fn(gt_records[gt_key]) for _, gt_key in matched_pairs}),
            "exact_tp_ids": id_list({display_fn(gt_records[key]) for key in exact_matched}),
            "semantic_tp_ids": id_list(
                {
                    f"{display_fn(pred_records[pred_key])}=>{display_fn(gt_records[gt_key])}"
                    for pred_key, gt_key, _ in semantic_pairs
                }
            ),
            "fp_ids": id_list({display_fn(pred_records[key]) for key in fp}),
            "fn_ids": id_list({display_fn(gt_records[key]) for key in fn}),
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }
    ]

    detail_rows: list[dict[str, Any]] = []
    mismatch_rows: list[dict[str, Any]] = []
    for key in sorted(exact_matched):
        pred_record = pred_records[key]
        gt_record = gt_records[key]
        differing_fields = [
            field
            for field in fields
            if normalize_for_compare(pred_record.get(field)) != normalize_for_compare(gt_record.get(field))
        ]
        detail_rows.append(
            {
                "model": model_name,
                "study_id": study_id,
                "task": task,
                "record_key": display_fn(gt_record),
                "match_type": "TP",
                "differing_fields": ";".join(differing_fields),
                "predicted_record": json_cell(pred_record),
                "ground_truth_record": json_cell(gt_record),
                "judge_decision": "exact_tp",
            }
        )
        if differing_fields:
            mismatch_rows.append(
                {
                    "study_id": study_id,
                    "file_type": "",
                    "chunk_id": "",
                    "source_id": "",
                    "task": task,
                    "error_type": "FIELD_MISMATCH",
                    "record_key": display_fn(gt_record),
                    "text": "",
                    "predicted_ids": display_fn(pred_record),
                    "ground_truth_ids": display_fn(gt_record),
                    "fp_ids": "",
                    "fn_ids": "",
                    "differing_fields": ";".join(differing_fields),
                    "predicted_record": json_cell(pred_record),
                    "ground_truth_record": json_cell(gt_record),
                }
            )

    for pred_key, gt_key, match in semantic_pairs:
        pred_record = pred_records[pred_key]
        gt_record = gt_records[gt_key]
        differing_fields = [
            field
            for field in fields
            if normalize_for_compare(pred_record.get(field)) != normalize_for_compare(gt_record.get(field))
        ]
        judge_fields = judge_row_fields(
            judge_result,
            "semantic_tp",
            gt_index=match["gt_index"],
            pred_index=match["pred_index"],
        )
        detail_rows.append(
            {
                "model": model_name,
                "study_id": study_id,
                "task": task,
                "record_key": display_fn(gt_record),
                "match_type": "SEMANTIC_TP",
                "differing_fields": ";".join(differing_fields),
                "predicted_record": json_cell(pred_record),
                "ground_truth_record": json_cell(gt_record),
                **judge_fields,
            }
        )
        mismatch_rows.append(
            {
                "study_id": study_id,
                "file_type": "",
                "chunk_id": "",
                "source_id": "",
                "task": task,
                "error_type": "SEMANTIC_TP",
                "record_key": display_fn(gt_record),
                "text": "",
                "predicted_ids": display_fn(pred_record),
                "ground_truth_ids": display_fn(gt_record),
                "fp_ids": "",
                "fn_ids": "",
                "differing_fields": ";".join(differing_fields),
                "predicted_record": json_cell(pred_record),
                "ground_truth_record": json_cell(gt_record),
                **judge_fields,
            }
        )

    had_judge_candidates = bool(gt_judge_records and pred_judge_records)
    fp_decision = "semantic_fp"
    fn_decision = "semantic_fn"
    if not ENABLE_LLM_JUDGE:
        fp_decision = "semantic_fp_judge_disabled"
        fn_decision = "semantic_fn_judge_disabled"
    elif not had_judge_candidates:
        fp_decision = "semantic_fp_no_candidate"
        fn_decision = "semantic_fn_no_candidate"
    elif judge_result.get("error"):
        fp_decision = "semantic_fp_judge_error"
        fn_decision = "semantic_fn_judge_error"

    for key in sorted(fp):
        pred_record = pred_records[key]
        mismatch_rows.append(
            {
                "study_id": study_id,
                "file_type": "",
                "chunk_id": "",
                "source_id": "",
                "task": task,
                "error_type": "FP",
                "record_key": display_fn(pred_record),
                "text": "",
                "predicted_ids": display_fn(pred_record),
                "ground_truth_ids": "",
                "fp_ids": display_fn(pred_record),
                "fn_ids": "",
                "differing_fields": "",
                "predicted_record": json_cell(pred_record),
                "ground_truth_record": "",
                **judge_row_fields(
                    judge_result,
                    fp_decision,
                    pred_index=pred_key_to_index.get(key, ""),
                ),
            }
        )
    for key in sorted(fn):
        gt_record = gt_records[key]
        mismatch_rows.append(
            {
                "study_id": study_id,
                "file_type": "",
                "chunk_id": "",
                "source_id": "",
                "task": task,
                "error_type": "FN",
                "record_key": display_fn(gt_record),
                "text": "",
                "predicted_ids": "",
                "ground_truth_ids": display_fn(gt_record),
                "fp_ids": "",
                "fn_ids": display_fn(gt_record),
                "differing_fields": "",
                "predicted_record": "",
                "ground_truth_record": json_cell(gt_record),
                **judge_row_fields(
                    judge_result,
                    fn_decision,
                    gt_index=gt_key_to_index.get(key, ""),
                ),
            }
        )

    for source, duplicates in (("prediction", pred_duplicates), ("ground_truth", gt_duplicates)):
        for duplicate in duplicates:
            mismatch_rows.append(
                {
                    "study_id": study_id,
                    "file_type": "",
                    "chunk_id": "",
                    "source_id": "",
                    "task": task,
                    "error_type": f"DUPLICATE_{source.upper()}",
                    "record_key": duplicate["record_key"],
                    "text": "",
                    "predicted_ids": duplicate["record_key"] if source == "prediction" else "",
                    "ground_truth_ids": duplicate["record_key"] if source == "ground_truth" else "",
                    "fp_ids": "",
                    "fn_ids": "",
                    "differing_fields": "",
                    "predicted_record": duplicate["record"] if source == "prediction" else "",
                    "ground_truth_record": duplicate["record"] if source == "ground_truth" else "",
                }
            )

    field_rows = field_agreement_rows(
        study_id=study_id,
        model_name=model_name,
        task=task,
        matched_pairs=matched_pairs,
        pred_records=pred_records,
        gt_records=gt_records,
        fields=fields,
    )
    return metric_rows, detail_rows, field_rows, mismatch_rows, equivalence_map


def first_performance_enum(chunk: dict[str, Any]) -> dict[str, Any]:
    for enum in chunk.get("enumerations") or []:
        if isinstance(enum, dict) and enum.get("task") == "performance":
            return enum
    return {}


def performance_record_ids(chunk: dict[str, Any]) -> set[str]:
    return set(performance_record_map(chunk))


def performance_record_map(chunk: dict[str, Any]) -> dict[str, dict[str, Any]]:
    enum = first_performance_enum(chunk)
    records_by_id: dict[str, dict[str, Any]] = {}
    for record in enum.get("records") or []:
        if not isinstance(record, dict):
            continue
        perf_id = normalize_id(record.get("Performance_id"))
        if is_no_performance_id(perf_id) or is_unit_performance_id(perf_id):
            continue
        if perf_id and perf_id not in records_by_id:
            normalized_record = dict(record)
            normalized_record["Performance_id"] = perf_id
            records_by_id[perf_id] = normalized_record
    return records_by_id


def display_key_performance(record: dict[str, Any]) -> str:
    return normalize_id(record.get("Performance_id"))


def is_unit_performance_id(value: Any) -> bool:
    return normalize_for_compare(value) == "unit"


def trim_decimal_match(match: re.Match[str]) -> str:
    return match.group(1).rstrip("0").rstrip(".")


def normalize_condition_token(value: Any) -> str:
    token = NORMALIZER.sub("/", str(value or "").strip())
    token = token.replace("\u00b5", "u").replace("\u03bc", "u")
    token = token.replace("\u00ae", "").replace("\u2122", "").replace("\u00a9", "")
    token = re.sub(r"\s+", " ", token)
    token = NUMBER_UNIT_SPACE.sub("_", token)
    token = re.sub(r"\s*_\s*", "_", token)
    token = re.sub(r"\s*/\s*", "/", token)
    token = re.sub(r"\s*\^\s*", "^", token)
    token = re.sub(r"\s*-\s*", "-", token)
    while True:
        stripped = GENERIC_CONDITION_PREFIX.sub("", token)
        if stripped == token:
            break
        token = stripped.strip()
    token = DECIMAL_NUMBER.sub(trim_decimal_match, token)
    token = re.sub(r"\s+", "_", token.strip())
    token = re.sub(r"_+", "_", token).strip("_")
    return token.casefold()


def normalize_with_equivalence(value: str, equivalences: dict[str, str]) -> str:
    return equivalences.get(value, value)


def normalize_condition_token_with_equivalences(
    value: Any,
    *,
    adsorbent_equivalences: dict[str, str] | None = None,
    water_type_equivalences: dict[str, str] | None = None,
) -> str:
    token = normalize_condition_token(value)
    if not token:
        return ""
    equivalence_maps = [
        adsorbent_equivalences or {},
        water_type_equivalences or {},
    ]
    for equivalences in equivalence_maps:
        if not equivalences:
            continue
        direct = normalize_with_equivalence(token, equivalences)
        if direct != token:
            token = direct
            continue
        for alias in sorted(equivalences, key=len, reverse=True):
            if token.startswith(f"{alias}_"):
                token = f"{equivalences[alias]}{token[len(alias):]}"
                break
    return token


def canonical_condition_tokens(
    value: Any,
    *,
    adsorbent_equivalences: dict[str, str] | None = None,
    water_type_equivalences: dict[str, str] | None = None,
) -> tuple[str, ...]:
    raw_tokens = CONDITION_SEPARATOR.split(str(value or "").strip())
    tokens = [
        normalize_condition_token_with_equivalences(
            token,
            adsorbent_equivalences=adsorbent_equivalences,
            water_type_equivalences=water_type_equivalences,
        )
        for token in raw_tokens
    ]
    return tuple(sorted(token for token in tokens if token))

def starts_with_condition_subject(token: str, subjects: set[str]) -> bool:
    return any(
        subject and (token == subject or token.startswith(f"{subject}_"))
        for subject in subjects
    )


def is_absent_zero_concentration_condition_token(
    token: str,
    *,
    protected_subjects: set[str] | None = None,
) -> bool:
    if not ZERO_CONCENTRATION_UNIT_SUFFIX.search(token):
        return False
    return not starts_with_condition_subject(token, protected_subjects or set())


def canonical_performance_condition_tokens(
    condition: Any,
    *,
    pfas_name: Any = "",
    adsorbent_id: Any = "",
    adsorbent_equivalences: dict[str, str] | None = None,
    water_type_equivalences: dict[str, str] | None = None,
) -> tuple[str, ...]:
    raw_tokens = CONDITION_SEPARATOR.split(str(condition or "").strip())
    normalized_tokens = [
        normalize_condition_token_with_equivalences(
            token,
            adsorbent_equivalences=adsorbent_equivalences,
            water_type_equivalences=water_type_equivalences,
        )
        for token in raw_tokens
    ]
    protected_subjects = {normalize_condition_token(pfas_name)}
    adsorbent_subject = normalize_condition_token_with_equivalences(
        adsorbent_id,
        adsorbent_equivalences=adsorbent_equivalences,
        water_type_equivalences=water_type_equivalences,
    )
    if adsorbent_subject:
        protected_subjects.add(adsorbent_subject)

    tokens = [
        token
        for token in normalized_tokens
        if token
        and not is_absent_zero_concentration_condition_token(
            token,
            protected_subjects=protected_subjects,
        )
    ]
    if tokens:
        return tuple(sorted(tokens))
    if normalized_tokens and all(
        is_na_like_slot(token)
        or is_absent_zero_concentration_condition_token(
            token,
            protected_subjects=protected_subjects,
        )
        for token in normalized_tokens
    ):
        return ("na",)
    return tuple()

def normalize_performance_id_component(value: Any) -> str:
    text = NORMALIZER.sub("/", str(value or "").strip())
    text = text.replace("\u00b5", "u").replace("\u03bc", "u")
    text = text.replace("\u00ae", "").replace("\u2122", "").replace("\u00a9", "")
    text = re.sub(r"\s+", " ", text)
    return text.casefold()


def canonical_performance_id(
    value: Any,
    *,
    adsorbent_equivalences: dict[str, str] | None = None,
    water_type_equivalences: dict[str, str] | None = None,
) -> tuple[str, str, tuple[str, ...]] | None:
    perf_id = normalize_id(value)
    if not perf_id or is_unit_performance_id(perf_id):
        return None
    parts = perf_id.split("|")
    if len(parts) != 3:
        return None
    pfas_name, adsorbent_id, condition = parts
    adsorbent_equivalences = adsorbent_equivalences or {}
    water_type_equivalences = water_type_equivalences or {}
    condition_tokens = canonical_performance_condition_tokens(
        condition,
        pfas_name=pfas_name,
        adsorbent_id=adsorbent_id,
        adsorbent_equivalences=adsorbent_equivalences,
        water_type_equivalences=water_type_equivalences,
    )
    if not condition_tokens:
        return None
    return (
        normalize_performance_id_component(pfas_name),
        normalize_with_equivalence(
            normalize_performance_id_component(adsorbent_id),
            adsorbent_equivalences,
        ),
        condition_tokens,
    )


def performance_candidate_slots(
    value: Any,
    *,
    adsorbent_equivalences: dict[str, str] | None = None,
    water_type_equivalences: dict[str, str] | None = None,
) -> tuple[str, str, bool] | None:
    perf_id = normalize_id(value)
    if not perf_id or is_unit_performance_id(perf_id):
        return None
    parts = perf_id.split("|")
    if len(parts) != 3:
        return None
    pfas_name, adsorbent_id, condition = parts
    adsorbent_equivalences = adsorbent_equivalences or {}
    condition_tokens = canonical_performance_condition_tokens(
        condition,
        pfas_name=pfas_name,
        adsorbent_id=adsorbent_id,
        adsorbent_equivalences=adsorbent_equivalences,
        water_type_equivalences=water_type_equivalences,
    )
    condition_is_na = not condition_tokens or (
        len(condition_tokens) == 1 and is_na_like_slot(condition_tokens[0])
    )
    return (
        normalize_performance_id_component(pfas_name),
        normalize_with_equivalence(
            normalize_performance_id_component(adsorbent_id),
            adsorbent_equivalences,
        ),
        condition_is_na,
    )

def performance_condition_token_set(
    value: Any,
    *,
    adsorbent_equivalences: dict[str, str] | None = None,
    water_type_equivalences: dict[str, str] | None = None,
) -> set[str] | None:
    perf_id = normalize_id(value)
    if not perf_id or is_unit_performance_id(perf_id):
        return None
    parts = perf_id.split("|")
    if len(parts) != 3:
        return None
    pfas_name, adsorbent_id, condition = parts
    tokens = canonical_performance_condition_tokens(
        condition,
        pfas_name=pfas_name,
        adsorbent_id=adsorbent_id,
        adsorbent_equivalences=adsorbent_equivalences,
        water_type_equivalences=water_type_equivalences,
    )
    return {token for token in tokens if not is_na_like_slot(token)}

def condition_token_measurement_signature(token: str) -> tuple[str, str, str] | None:
    match = NUMERIC_CONDITION_VALUE_UNIT_SUFFIX.match(token)
    if not match:
        return None
    subject = match.group("subject")
    value = match.group("value").rstrip("0").rstrip(".")
    unit = match.group("unit")
    if not subject or not value or not unit:
        return None
    return subject, value, unit


def condition_measurement_maps(
    tokens: set[str],
) -> tuple[dict[str, set[tuple[str, str]]], set[tuple[str, str]], bool]:
    by_subject: dict[str, set[tuple[str, str]]] = {}
    value_units: set[tuple[str, str]] = set()
    all_tokens_measured = True

    for token in tokens:
        signature = condition_token_measurement_signature(token)
        if signature is None:
            all_tokens_measured = False
            continue
        subject, value, unit = signature
        measurement = (value, unit)
        by_subject.setdefault(subject, set()).add(measurement)
        value_units.add(measurement)

    return by_subject, value_units, all_tokens_measured

def condition_token_category(
    token: str,
    *,
    pfas_name: Any = "",
    adsorbent_id: Any = "",
    adsorbent_equivalences: dict[str, str] | None = None,
    water_type_equivalences: dict[str, str] | None = None,
) -> str:
    if not token or is_na_like_slot(token):
        return ""

    if token == "ph" or token.startswith("ph_"):
        return "ph"
    if (
        token == "time"
        or token.startswith("time_")
        or token.startswith("contact_time_")
        or token.startswith("equilibrium_time_")
    ):
        return "time"
    if (
        token == "temperature"
        or token.startswith("temperature_")
        or token.startswith("temp_")
    ):
        return "temperature"
    if token == "toc" or token.startswith("toc_"):
        return "toc"
    if token == "doc" or token.startswith("doc_"):
        return "doc"
    if token == "ionic_strength" or token.startswith("ionic_strength_"):
        return "ionic_strength"

    pfas_subject = normalize_condition_token(pfas_name)
    if starts_with_condition_subject(token, {pfas_subject}):
        return "pfas_initial_concentration"

    adsorbent_subject = normalize_condition_token_with_equivalences(
        adsorbent_id,
        adsorbent_equivalences=adsorbent_equivalences,
        water_type_equivalences=water_type_equivalences,
    )
    if starts_with_condition_subject(token, {adsorbent_subject}):
        return "adsorbent_dosage"

    return ""


def performance_condition_category_map(
    value: Any,
    *,
    adsorbent_equivalences: dict[str, str] | None = None,
    water_type_equivalences: dict[str, str] | None = None,
) -> dict[str, set[str]] | None:
    perf_id = normalize_id(value)
    if not perf_id or is_unit_performance_id(perf_id):
        return None
    parts = perf_id.split("|")
    if len(parts) != 3:
        return None

    pfas_name, adsorbent_id, _ = parts
    tokens = performance_condition_token_set(
        perf_id,
        adsorbent_equivalences=adsorbent_equivalences,
        water_type_equivalences=water_type_equivalences,
    )
    if tokens is None:
        return None

    categories: dict[str, set[str]] = {}
    for token in tokens:
        category = condition_token_category(
            token,
            pfas_name=pfas_name,
            adsorbent_id=adsorbent_id,
            adsorbent_equivalences=adsorbent_equivalences,
            water_type_equivalences=water_type_equivalences,
        )
        if category:
            categories.setdefault(category, set()).add(token)
    return categories

def has_incompatible_differentiating_conditions(
    pred_id: str,
    gt_id: str,
    *,
    adsorbent_equivalences: dict[str, str] | None = None,
    water_type_equivalences: dict[str, str] | None = None,
) -> bool:
    pred_tokens = performance_condition_token_set(
        pred_id,
        adsorbent_equivalences=adsorbent_equivalences,
        water_type_equivalences=water_type_equivalences,
    )
    gt_tokens = performance_condition_token_set(
        gt_id,
        adsorbent_equivalences=adsorbent_equivalences,
        water_type_equivalences=water_type_equivalences,
    )
    if pred_tokens is None or gt_tokens is None:
        return False
    if pred_tokens == gt_tokens:
        return False

    # One side has all conditions from the other plus extra conditions.
    # Example:
    #   Time_0.5_h&&AC_5_mg/L  vs  Time_0.5_h
    if pred_tokens < gt_tokens or gt_tokens < pred_tokens:
        return True

    pred_categories = performance_condition_category_map(
        pred_id,
        adsorbent_equivalences=adsorbent_equivalences,
        water_type_equivalences=water_type_equivalences,
    )
    gt_categories = performance_condition_category_map(
        gt_id,
        adsorbent_equivalences=adsorbent_equivalences,
        water_type_equivalences=water_type_equivalences,
    )
    if pred_categories is None or gt_categories is None:
        return False

    pred_category_names = set(pred_categories)
    gt_category_names = set(gt_categories)

    # Same strict dimension, different values.
    # Examples:
    #   pH_5 vs pH_7
    #   Time_0.5_h vs Time_48_h
    #   AC_1_mg/L vs AC_5_mg/L
    for category in (
        pred_category_names
        & gt_category_names
        & STRICT_VALUE_CONDITION_CATEGORIES
    ):
        if pred_categories[category] != gt_categories[category]:
            return True

    # Different strict dimensions.
    # Examples:
    #   Time_0.5_h vs AC_5_mg/L
    #   pH_7 vs humic_acid_5_mg/L
    #   Temperature_25_°C vs DOC_4_mg/L
    changed_strict_dimensions = (
        pred_category_names ^ gt_category_names
    ) & STRICT_DIMENSION_CONDITION_CATEGORIES
    if changed_strict_dimensions:
        return True
    pred_measurements_by_subject, pred_value_units, pred_all_measured = (
        condition_measurement_maps(pred_tokens)
    )
    gt_measurements_by_subject, gt_value_units, gt_all_measured = (
        condition_measurement_maps(gt_tokens)
    )

    # Same measured condition subject but different value and/or unit.
    # Examples:
    #   NaCl_100_mM vs NaCl_50_mM
    #   humic_acid_5_mg/L vs humic_acid_10_mg/L
    #   DOC_5_mg/L vs DOC_5_ug/L
    for subject in set(pred_measurements_by_subject) & set(gt_measurements_by_subject):
        if pred_measurements_by_subject[subject] != gt_measurements_by_subject[subject]:
            return True

    # Different measured condition labels with different value/unit signatures.
    # This is still a clear mismatch because even if the labels were aliases,
    # the experimental value/unit would not match.
    # Examples:
    #   NaCl_100_mM vs humic_acid_5_mg/L
    #   Ca^2+_1_mg/L vs Mg^2+_5_mg/L
    if (
        pred_all_measured
        and gt_all_measured
        and pred_value_units
        and gt_value_units
        and pred_value_units != gt_value_units
    ):
        return True

    # Remaining chemical / matrix label differences are potentially alias cases.
    # Examples:
    #   HA_5_mg/L vs humic_acid_5_mg/L
    #   NOM vs natural_organic_matter
    return False

def plausible_performance_candidate(
    pred_id: str,
    gt_id: str,
    *,
    adsorbent_equivalences: dict[str, str] | None = None,
    water_type_equivalences: dict[str, str] | None = None,
) -> bool:
    pred_slots = performance_candidate_slots(
        pred_id,
        adsorbent_equivalences=adsorbent_equivalences,
        water_type_equivalences=water_type_equivalences,
    )
    gt_slots = performance_candidate_slots(
        gt_id,
        adsorbent_equivalences=adsorbent_equivalences,
        water_type_equivalences=water_type_equivalences,
    )
    if pred_slots is None or gt_slots is None:
        return True

    pred_pfas, pred_adsorbent, pred_condition_is_na = pred_slots
    gt_pfas, gt_adsorbent, gt_condition_is_na = gt_slots
    pred_pfas = "na" if is_na_like_slot(pred_pfas) else pred_pfas
    gt_pfas = "na" if is_na_like_slot(gt_pfas) else gt_pfas
    pred_adsorbent = "na" if is_na_like_slot(pred_adsorbent) else pred_adsorbent
    gt_adsorbent = "na" if is_na_like_slot(gt_adsorbent) else gt_adsorbent

    if (
        pred_pfas != gt_pfas
        or pred_adsorbent != gt_adsorbent
        or pred_condition_is_na != gt_condition_is_na
    ):
        return False

    if has_incompatible_differentiating_conditions(
        pred_id,
        gt_id,
        adsorbent_equivalences=adsorbent_equivalences,
        water_type_equivalences=water_type_equivalences,
    ):
        return False

    return True

def plausible_performance_candidate_ids(
    pred_ids: set[str],
    gt_ids: set[str],
    *,
    adsorbent_equivalences: dict[str, str] | None = None,
    water_type_equivalences: dict[str, str] | None = None,
) -> tuple[set[str], set[str]]:
    candidate_pred_ids: set[str] = set()
    candidate_gt_ids: set[str] = set()
    for pred_id in pred_ids:
        for gt_id in gt_ids:
            if plausible_performance_candidate(
                pred_id,
                gt_id,
                adsorbent_equivalences=adsorbent_equivalences,
                water_type_equivalences=water_type_equivalences,
            ):
                candidate_pred_ids.add(pred_id)
                candidate_gt_ids.add(gt_id)
    return candidate_pred_ids, candidate_gt_ids


def unique_canonical_performance_pairs(
    pred_ids: set[str],
    gt_ids: set[str],
    *,
    adsorbent_equivalences: dict[str, str] | None = None,
    water_type_equivalences: dict[str, str] | None = None,
) -> list[tuple[str, str]]:
    pred_by_key: dict[tuple[str, str, tuple[str, ...]], list[str]] = {}
    gt_by_key: dict[tuple[str, str, tuple[str, ...]], list[str]] = {}
    for perf_id in pred_ids:
        key = canonical_performance_id(
            perf_id,
            adsorbent_equivalences=adsorbent_equivalences,
            water_type_equivalences=water_type_equivalences,
        )
        if key is not None:
            pred_by_key.setdefault(key, []).append(perf_id)
    for perf_id in gt_ids:
        key = canonical_performance_id(
            perf_id,
            adsorbent_equivalences=adsorbent_equivalences,
            water_type_equivalences=water_type_equivalences,
        )
        if key is not None:
            gt_by_key.setdefault(key, []).append(perf_id)

    pairs: list[tuple[str, str]] = []
    for key in sorted(set(pred_by_key) & set(gt_by_key)):
        pred_matches = pred_by_key[key]
        gt_matches = gt_by_key[key]
        if len(pred_matches) == 1 and len(gt_matches) == 1:
            pairs.append((pred_matches[0], gt_matches[0]))
    return pairs


def performance_chunk_map(payload: dict[str, Any]) -> dict[tuple[str, str], dict[str, Any]]:
    chunks: dict[tuple[str, str], dict[str, Any]] = {}
    for output in payload.get("outputs") or []:
        if not isinstance(output, dict):
            continue
        key = chunk_key_from_entry(output)
        metadata = output_source_metadata(output)
        source_ids = source_metadata_label(metadata)
        enum = first_performance_enum(output)
        item = chunks.setdefault(
            key,
            {
                "file_type": key[0],
                "chunk_id": key[1],
                "source_id": source_id_for(output),
                "predicted_label": output.get("predicted_label"),
                "text": "",
                "mode": "",
                "source_metadata": [],
                "source_chunk_count": 0,
                "source_chunk_ids": "",
                "ids": set(),
                "records": {},
            },
        )
        item["source_id"] = source_id_for(output)
        item["predicted_label"] = output.get("predicted_label", item.get("predicted_label"))
        item["text"] = output.get("enriched_text") or output.get("text") or item.get("text") or ""
        item["mode"] = enum.get("mode") or output.get("mode") or item.get("mode", "")
        item["source_metadata"] = metadata
        item["source_chunk_count"] = len(metadata)
        item["source_chunk_ids"] = source_ids
        item["records"] = performance_record_map(output)
        item["ids"] = set(item["records"])
    return chunks


def evaluate_performance_task(
    *,
    study_id: str,
    model_name: str,
    pred_payload: dict[str, Any],
    gt_payload: dict[str, Any],
    judge_chain: LLMChain | None = None,
    adsorbent_equivalences: dict[str, str] | None = None,
    water_type_equivalences: dict[str, str] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    pred_chunks = performance_chunk_map(pred_payload)
    gt_chunks = performance_chunk_map(gt_payload)
    rows: list[dict[str, Any]] = []
    mismatch_rows: list[dict[str, Any]] = []

    for key in sorted(set(pred_chunks) | set(gt_chunks)):
        pred_chunk = pred_chunks.get(key, {})
        gt_chunk = gt_chunks.get(key, {})
        pred_ids = {normalize_id(value) for value in pred_chunk.get("ids", set()) if normalize_id(value)}
        gt_ids = {normalize_id(value) for value in gt_chunk.get("ids", set()) if normalize_id(value)}
        if not pred_ids and not gt_ids:
            continue
        ground_count = len(gt_ids)
        pred_count = len(pred_ids)

        pred_records_by_id = {
            normalize_id(key): dict(value)
            for key, value in (pred_chunk.get("records") or {}).items()
            if normalize_id(key) and isinstance(value, dict)
        }
        gt_records_by_id = {
            normalize_id(key): dict(value)
            for key, value in (gt_chunk.get("records") or {}).items()
            if normalize_id(key) and isinstance(value, dict)
        }
        for perf_id in pred_ids:
            pred_records_by_id.setdefault(perf_id, {"Performance_id": perf_id})
        for perf_id in gt_ids:
            gt_records_by_id.setdefault(perf_id, {"Performance_id": perf_id})

        exact_tp_ids = pred_ids & gt_ids
        exact_fp_ids = pred_ids - gt_ids
        exact_fn_ids = gt_ids - pred_ids
        file_type, chunk_id = key
        source_id = (
            pred_chunk.get("source_id")
            or gt_chunk.get("source_id")
            or f"{file_type}_{chunk_id}"
        )
        text = pred_chunk.get("text") or gt_chunk.get("text") or ""
        mode = pred_chunk.get("mode") or gt_chunk.get("mode") or ""
        source_chunk_count = max(
            int(pred_chunk.get("source_chunk_count") or 0),
            int(gt_chunk.get("source_chunk_count") or 0),
        )
        source_chunk_ids = pred_chunk.get("source_chunk_ids") or gt_chunk.get("source_chunk_ids") or ""

        unit_fp_ids = {perf_id for perf_id in exact_fp_ids if is_unit_performance_id(perf_id)}
        unit_fn_ids = {perf_id for perf_id in exact_fn_ids if is_unit_performance_id(perf_id)}
        judgeable_fp_ids = exact_fp_ids - unit_fp_ids
        judgeable_fn_ids = exact_fn_ids - unit_fn_ids

        python_pairs = unique_canonical_performance_pairs(
            judgeable_fp_ids,
            judgeable_fn_ids,
            adsorbent_equivalences=adsorbent_equivalences,
            water_type_equivalences=water_type_equivalences,
        )
        python_pred_ids = {pred_id for pred_id, _ in python_pairs}
        python_gt_ids = {gt_id for _, gt_id in python_pairs}
        judgeable_fp_ids -= python_pred_ids
        judgeable_fn_ids -= python_gt_ids

        candidate_pred_ids, candidate_gt_ids = plausible_performance_candidate_ids(
            judgeable_fp_ids,
            judgeable_fn_ids,
            adsorbent_equivalences=adsorbent_equivalences,
            water_type_equivalences=water_type_equivalences,
        )
        python_no_match_pred_ids = judgeable_fp_ids - candidate_pred_ids
        python_no_match_gt_ids = judgeable_fn_ids - candidate_gt_ids
        judgeable_fp_ids = candidate_pred_ids
        judgeable_fn_ids = candidate_gt_ids

        fn_id_list = sorted(judgeable_fn_ids)
        fp_id_list = sorted(judgeable_fp_ids)
        gt_judge_records, gt_index_to_key = indexed_judge_records(
            gt_records_by_id,
            fn_id_list,
        )
        pred_judge_records, pred_index_to_key = indexed_judge_records(
            pred_records_by_id,
            fp_id_list,
        )
        gt_key_to_index = {key: index for index, key in gt_index_to_key.items()}
        pred_key_to_index = {key: index for index, key in pred_index_to_key.items()}
        judge_result = run_semantic_judge(
            study_id=study_id,
            task="performance",
            context_label=str(source_id),
            judge_chain=judge_chain,
            ground_truth_records=gt_judge_records,
            predicted_records=pred_judge_records,
        )
        unit_judge_result = judge_result_template(
            study_id,
            "performance",
            f"{source_id}|unit_not_judged",
        )
        python_judge_result = judge_result_template(
            study_id,
            "performance",
            f"{source_id}|python",
        )
        python_no_match_judge_result = judge_result_template(
            study_id,
            "performance",
            f"{source_id}|python_no_match",
        )

        semantic_pairs: list[tuple[str, str, dict[str, Any]]] = []
        semantic_pred_ids: set[str] = set()
        semantic_gt_ids: set[str] = set()
        for match in judge_result.get("matches") or []:
            gt_id = gt_index_to_key[match["gt_index"]]
            pred_id = pred_index_to_key[match["pred_index"]]
            semantic_pairs.append((pred_id, gt_id, match))
            semantic_pred_ids.add(pred_id)
            semantic_gt_ids.add(gt_id)

        fp_ids = exact_fp_ids - python_pred_ids - semantic_pred_ids
        fn_ids = exact_fn_ids - python_gt_ids - semantic_gt_ids
        tp_ids = exact_tp_ids | python_gt_ids | semantic_gt_ids
        tp_n = len(exact_tp_ids) + len(python_pairs) + len(semantic_pairs)
        fp_n = len(fp_ids)
        fn_n = len(fn_ids)
        precision, recall, f1 = metric_values(tp_n, fp_n, fn_n)
        rows.append(
            {
                "model": model_name,
                "study": study_id,
                "task": "performance",
                "granularity": "performance_group",
                "file_type": file_type,
                "chunk_id": chunk_id,
                "source_id": source_id,
                "mode": mode,
                "source_chunk_count": source_chunk_count,
                "source_chunk_ids": source_chunk_ids,
                "ground_count": ground_count,
                "pred_count": pred_count,
                "eval_status": eval_status(ground_count, pred_count),
                "tp": tp_n,
                "fp": fp_n,
                "fn": fn_n,
                "exact_tp": len(exact_tp_ids),
                "python_tp": len(python_pairs),
                "semantic_tp": len(semantic_pairs),
                "tp_ids": id_list(tp_ids),
                "exact_tp_ids": id_list(exact_tp_ids),
                "python_tp_ids": id_list(
                    {f"{pred_id}=>{gt_id}" for pred_id, gt_id in python_pairs}
                ),
                "semantic_tp_ids": id_list(
                    {f"{pred_id}=>{gt_id}" for pred_id, gt_id, _ in semantic_pairs}
                ),
                "fp_ids": id_list(fp_ids),
                "fn_ids": id_list(fn_ids),
                "precision": precision,
                "recall": recall,
                "f1": f1,
            }
        )

        for pred_id, gt_id in python_pairs:
            pred_record = pred_records_by_id[pred_id]
            gt_record = gt_records_by_id[gt_id]
            mismatch_rows.append(
                {
                    "study_id": study_id,
                    "file_type": file_type,
                    "chunk_id": chunk_id,
                    "source_id": source_id,
                    "mode": mode,
                    "source_chunk_count": source_chunk_count,
                    "source_chunk_ids": source_chunk_ids,
                    "task": "performance",
                    "error_type": "PYTHON_TP",
                    "record_key": gt_id,
                    "text": text,
                    "predicted_ids": pred_id,
                    "ground_truth_ids": gt_id,
                    "fp_ids": "",
                    "fn_ids": "",
                    "differing_fields": "Performance_id",
                    "predicted_record": json_cell(pred_record),
                    "ground_truth_record": json_cell(gt_record),
                    **judge_row_fields(
                        python_judge_result,
                        "python_tp",
                    ),
                }
            )

        for pred_id, gt_id, match in semantic_pairs:
            pred_record = pred_records_by_id[pred_id]
            gt_record = gt_records_by_id[gt_id]
            mismatch_rows.append(
                {
                    "study_id": study_id,
                    "file_type": file_type,
                    "chunk_id": chunk_id,
                    "source_id": source_id,
                    "mode": mode,
                    "source_chunk_count": source_chunk_count,
                    "source_chunk_ids": source_chunk_ids,
                    "task": "performance",
                    "error_type": "SEMANTIC_TP",
                    "record_key": gt_id,
                    "text": text,
                    "predicted_ids": pred_id,
                    "ground_truth_ids": gt_id,
                    "fp_ids": "",
                    "fn_ids": "",
                    "differing_fields": "Performance_id",
                    "predicted_record": json_cell(pred_record),
                    "ground_truth_record": json_cell(gt_record),
                    **judge_row_fields(
                        judge_result,
                        "semantic_tp",
                        gt_index=match["gt_index"],
                        pred_index=match["pred_index"],
                    ),
                }
            )

        had_judge_candidates = bool(gt_judge_records and pred_judge_records)
        fp_decision = "semantic_fp"
        fn_decision = "semantic_fn"
        if not ENABLE_LLM_JUDGE:
            fp_decision = "semantic_fp_judge_disabled"
            fn_decision = "semantic_fn_judge_disabled"
        elif not had_judge_candidates:
            fp_decision = "semantic_fp_no_candidate"
            fn_decision = "semantic_fn_no_candidate"
        elif judge_result.get("error"):
            fp_decision = "semantic_fp_judge_error"
            fn_decision = "semantic_fn_judge_error"

        for perf_id in sorted(fp_ids):
            row_judge_result = unit_judge_result if is_unit_performance_id(perf_id) else judge_result
            row_decision = "performance_unit_not_judged" if is_unit_performance_id(perf_id) else fp_decision
            if perf_id in python_no_match_pred_ids:
                row_judge_result = python_no_match_judge_result
                row_decision = "python_no_match"
            mismatch_rows.append(
                {
                    "study_id": study_id,
                    "file_type": file_type,
                    "chunk_id": chunk_id,
                    "source_id": source_id,
                    "mode": mode,
                    "source_chunk_count": source_chunk_count,
                    "source_chunk_ids": source_chunk_ids,
                    "task": "performance",
                    "error_type": "FP",
                    "record_key": perf_id,
                    "text": text,
                    "predicted_ids": perf_id,
                    "ground_truth_ids": id_list(gt_ids),
                    "fp_ids": perf_id,
                    "fn_ids": "",
                    "differing_fields": "",
                    "predicted_record": json_cell(pred_records_by_id[perf_id]),
                    "ground_truth_record": "",
                    **judge_row_fields(
                        row_judge_result,
                        row_decision,
                        pred_index=pred_key_to_index.get(perf_id, ""),
                    ),
                }
            )
        for perf_id in sorted(fn_ids):
            row_judge_result = unit_judge_result if is_unit_performance_id(perf_id) else judge_result
            row_decision = "performance_unit_not_judged" if is_unit_performance_id(perf_id) else fn_decision
            if perf_id in python_no_match_gt_ids:
                row_judge_result = python_no_match_judge_result
                row_decision = "python_no_match"
            mismatch_rows.append(
                {
                    "study_id": study_id,
                    "file_type": file_type,
                    "chunk_id": chunk_id,
                    "source_id": source_id,
                    "mode": mode,
                    "source_chunk_count": source_chunk_count,
                    "source_chunk_ids": source_chunk_ids,
                    "task": "performance",
                    "error_type": "FN",
                    "record_key": perf_id,
                    "text": text,
                    "predicted_ids": id_list(pred_ids),
                    "ground_truth_ids": perf_id,
                    "fp_ids": "",
                    "fn_ids": perf_id,
                    "differing_fields": "",
                    "predicted_record": "",
                    "ground_truth_record": json_cell(gt_records_by_id[perf_id]),
                    **judge_row_fields(
                        row_judge_result,
                        row_decision,
                        gt_index=gt_key_to_index.get(perf_id, ""),
                    ),
                }
            )
    return rows, mismatch_rows


def usable_ground_truth(payload: dict[str, Any], path: Path | None) -> bool:
    return path is not None

def scoreable_metrics(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "eval_status" not in df.columns:
        return df
    return df[
        df["eval_status"].fillna("").astype(str) != "empty_gt_empty_pred"
    ].copy()

def macro_chunk_avg(df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    return (
        df.groupby(group_cols, dropna=False)
        .agg(
            rows_evaluated=("f1", "size"),
            precision=("precision", "mean"),
            recall=("recall", "mean"),
            f1=("f1", "mean"),
        )
        .reset_index()
    )


def id_weighted_macro(df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    weights = df["ground_count"].clip(lower=1)

    def weighted_mean(values: pd.Series) -> float:
        local_weights = weights.loc[values.index]
        denom = local_weights.sum()
        return float((values * local_weights).sum() / denom) if denom else 0.0

    return (
        df.groupby(group_cols, dropna=False)
        .agg(
            rows_evaluated=("f1", "size"),
            total_true_ids=("ground_count", "sum"),
            precision=("precision", weighted_mean),
            recall=("recall", weighted_mean),
            f1=("f1", weighted_mean),
        )
        .reset_index()
    )


def micro(df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    agg = (
        df.groupby(group_cols, dropna=False)
        .agg(
            tp=("tp", "sum"),
            fp=("fp", "sum"),
            fn=("fn", "sum"),
            rows_evaluated=("f1", "size"),
            total_true_ids=("ground_count", "sum"),
            total_pred_ids=("pred_count", "sum"),
        )
        .reset_index()
    )
    precision_denom = agg["tp"] + agg["fp"]
    recall_denom = agg["tp"] + agg["fn"]

    agg["precision"] = (agg["tp"] / precision_denom.where(precision_denom != 0)).fillna(0.0)
    agg["recall"] = (agg["tp"] / recall_denom.where(recall_denom != 0)).fillna(0.0)
    agg["f1"] = (2 * agg["precision"] * agg["recall"]) / (
        agg["precision"] + agg["recall"]
    )
    agg["f1"] = agg["f1"].fillna(0.0)
    return agg


def summarize_metrics(df: pd.DataFrame) -> tuple[pd.DataFrame, ...]:
    if df.empty:
        empty = pd.DataFrame()
        return empty, empty, empty, empty, empty, empty

    df = scoreable_metrics(df)
    if df.empty:
        empty = pd.DataFrame()
        return empty, empty, empty, empty, empty, empty

    study_cols = ["model", "study", "task", "granularity"]
    overall_cols = ["model", "task", "granularity"]
    return (
        macro_chunk_avg(df, study_cols),
        macro_chunk_avg(df, overall_cols),
        id_weighted_macro(df, study_cols),
        id_weighted_macro(df, overall_cols),
        micro(df, study_cols),
        micro(df, overall_cols),
    )


def safe_model_name(model_name: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", model_name.strip())
    return cleaned.strip("_") or "enumeration"


def join_unique(values: Any) -> str:
    seen: set[str] = set()
    out: list[str] = []
    for value in values:
        for part in str(value or "").split(";"):
            cleaned = part.strip()
            if cleaned and cleaned not in seen:
                seen.add(cleaned)
                out.append(cleaned)
    return ";".join(out)


def first_nonblank(values: Any) -> str:
    for value in values:
        text = str(value or "").strip()
        if text:
            return text
    return ""


def build_mishandled_chunk_rows(df_mismatches: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "study_id",
        "file_type",
        "chunk_id",
        "source_id",
        "mode",
        "source_chunk_count",
        "source_chunk_ids",
        "task",
        "problem_labels",
        "error_types",
        "judge_decisions",
        "mismatch_rows",
        "fp_ids",
        "fn_ids",
        "predicted_ids",
        "ground_truth_ids",
        "judge_artifact_paths",
        "text",
    ]
    if df_mismatches.empty:
        return pd.DataFrame(columns=columns)

    df = normalize_chunk_id_column(df_mismatches).fillna("")
    for column in columns + ["error_type", "judge_decision", "judge_artifact_path"]:
        if column not in df.columns:
            df[column] = ""

    chunk_mask = (
        df["study_id"].astype(str).str.strip().ne("")
        & df["file_type"].astype(str).str.strip().ne("")
        & df["chunk_id"].astype(str).str.strip().ne("")
    )
    problem_mask = ~df["error_type"].astype(str).str.strip().isin(
        {"PYTHON_TP", "SEMANTIC_TP"}
    )
    df = df[chunk_mask & problem_mask].copy()
    if df.empty:
        return pd.DataFrame(columns=columns)

    rows: list[dict[str, Any]] = []
    group_cols = [
        "study_id",
        "file_type",
        "chunk_id",
        "source_id",
        "mode",
        "source_chunk_count",
        "source_chunk_ids",
        "task",
    ]
    for keys, group in df.groupby(group_cols, dropna=False):
        study_id, file_type, chunk_id, source_id, mode, source_chunk_count, source_chunk_ids, task = keys
        error_types = join_unique(group["error_type"])
        judge_decisions = join_unique(group["judge_decision"])
        rows.append(
            {
                "study_id": study_id,
                "file_type": file_type,
                "chunk_id": chunk_id,
                "source_id": source_id,
                "mode": mode,
                "source_chunk_count": source_chunk_count,
                "source_chunk_ids": source_chunk_ids,
                "task": task,
                "problem_labels": join_unique([error_types, judge_decisions]),
                "error_types": error_types,
                "judge_decisions": judge_decisions,
                "mismatch_rows": len(group),
                "fp_ids": join_unique(group["fp_ids"]),
                "fn_ids": join_unique(group["fn_ids"]),
                "predicted_ids": join_unique(group["predicted_ids"]),
                "ground_truth_ids": join_unique(group["ground_truth_ids"]),
                "judge_artifact_paths": join_unique(group["judge_artifact_path"]),
                "text": first_nonblank(group["text"]),
            }
        )
    return pd.DataFrame(rows, columns=columns)

def read_excel_sheet(path: Path, sheet_name: str) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return normalize_chunk_id_column(pd.read_excel(path, sheet_name=sheet_name))
    except ValueError:
        return pd.DataFrame()


def merge_incremental_rows(
    *,
    old_df: pd.DataFrame,
    new_df: pd.DataFrame,
    selected_ids: set[str],
    study_col: str,
    selected_tasks: set[str] | None = None,
    task_col: str = "task",
    drop_extra_ids: set[str] | None = None,
) -> pd.DataFrame:
    drop_ids = set(selected_ids)
    if drop_extra_ids:
        drop_ids.update(drop_extra_ids)

    if old_df.empty:
        return new_df.copy()

    if study_col not in old_df.columns:
        return new_df.copy()

    drop_mask = old_df[study_col].astype(str).isin(drop_ids)
    if selected_tasks is not None and task_col in old_df.columns:
        drop_mask = drop_mask & old_df[task_col].astype(str).isin(selected_tasks)

    old_kept = old_df[~drop_mask].copy()

    if new_df.empty:
        return old_kept

    return pd.concat([old_kept, new_df], ignore_index=True, sort=False)

def _main_for_current_group() -> None:
    tasks = validate_tasks(TASKS)
    usage_tasks = ALL_TASKS
    studies = selected_studies(tasks)
    all_metric_rows: list[dict[str, Any]] = []
    all_record_detail_rows: list[dict[str, Any]] = []
    all_field_rows: list[dict[str, Any]] = []
    all_mismatch_rows: list[dict[str, Any]] = []
    usage_rows: list[dict[str, Any]] = []
    judge_chain: LLMChain | None = None

    if ENABLE_LLM_JUDGE:
        if LLM_PROVIDER not in LLM_MODELS:
            raise ValueError(
                f"Unsupported LLM provider: {LLM_PROVIDER}. "
                f"Configured providers: {sorted(LLM_MODELS)}"
        )
        print(f"Using {LLM_PROVIDER} judge model: {LLM_MODEL_NAME}")
        print(f"Using evaluation judge prompt: {PROMPTS_DIR / JUDGE_PROMPT_FILE}")
        print(f"Judge artifact cache: {JUDGE_ARTIFACT_ROOT}")
        print(f"Evaluation tasks: {', '.join(tasks)}")
        print(f"Force LLM rerun: {FORCE_LLM_RERUN}")
        judge_llm = get_llm(provider=LLM_PROVIDER, model_name=LLM_MODEL_NAME)
        judge_chain = create_evaluation_judge_chain(PROMPTS_DIR, judge_llm)
    else:
        print("LLM semantic judge disabled; exact-match evaluation only.")
        print(f"Evaluation tasks: {', '.join(tasks)}")

    for study_id in studies:
        pred_manifest, _ = load_manifest(PREDICTIONS_DIR, study_id)
        pred_payloads: dict[str, dict[str, Any]] = {}
        gt_payloads: dict[str, dict[str, Any]] = {}
        pred_paths: dict[str, Path | None] = {}
        gt_paths: dict[str, Path | None] = {}
        tasks_to_load = tuple(dict.fromkeys((*tasks, *usage_tasks)))
        for task in tasks_to_load:
            pred_payloads[task], pred_paths[task] = load_task_payload(
                PREDICTIONS_DIR, study_id, task
            )
            gt_payloads[task], gt_paths[task] = load_task_payload(
                GROUND_TRUTH_DIR, study_id, task
            )

        if not any(path is not None for path in pred_paths.values()):
            print(f"Warning: skipping usage for {study_id}; no prediction artifacts found.")
            continue

        model_name = infer_model_name(pred_payloads, pred_manifest)
        usage_rows.append(
            summarize_usage(
                study_id,
                pred_payloads,
                pred_paths,
                pred_manifest,
                tasks=usage_tasks,
            )
        )
        lexicon_equivalences: dict[str, dict[str, str]] = {
            "adsorbent": {},
            "water_type": {},
        }

        for task in ("adsorbent", "water_type"):
            if task not in tasks:
                continue
            if not usable_ground_truth(gt_payloads[task], gt_paths[task]):
                print(f"Warning: skipping {study_id} {task} metrics; no usable GT artifact.")
                continue
            (
                metric_rows,
                detail_rows,
                field_rows,
                mismatch_rows,
                equivalence_map,
            ) = evaluate_lexicon_task(
                study_id=study_id,
                task=task,
                model_name=model_name,
                pred_payload=pred_payloads[task],
                gt_payload=gt_payloads[task],
                judge_chain=judge_chain,
            )
            lexicon_equivalences[task] = equivalence_map
            all_metric_rows.extend(metric_rows)
            all_record_detail_rows.extend(detail_rows)
            all_field_rows.extend(field_rows)
            all_mismatch_rows.extend(mismatch_rows)

        if "performance" in tasks:
            if usable_ground_truth(gt_payloads["performance"], gt_paths["performance"]):
                perf_rows, perf_mismatches = evaluate_performance_task(
                    study_id=study_id,
                    model_name=model_name,
                    pred_payload=pred_payloads["performance"],
                    gt_payload=gt_payloads["performance"],
                    judge_chain=judge_chain,
                    adsorbent_equivalences=lexicon_equivalences["adsorbent"],
                    water_type_equivalences=lexicon_equivalences["water_type"],
                )
                all_metric_rows.extend(perf_rows)
                all_mismatch_rows.extend(perf_mismatches)
            else:
                print(f"Warning: skipping {study_id} performance metrics; no usable GT artifact.")

    output_dir = PREDICTIONS_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    df_metrics = pd.DataFrame(all_metric_rows)
    df_record_details = pd.DataFrame(all_record_detail_rows)
    df_fields = pd.DataFrame(all_field_rows)
    df_mismatches = pd.DataFrame(all_mismatch_rows)
    df_usage = pd.DataFrame(usage_rows) if usage_rows else pd.DataFrame()

    model_for_file = safe_model_name(
        MODEL_NAME or (str(df_usage["model"].iloc[0]) if not df_usage.empty else "enumeration")
    )
    excel_path = output_dir / f"enumeration_eval_{model_for_file}.xlsx"

    if INCREMENTAL_EVAL_UPDATE and excel_path.exists():
        selected_ids = set(studies)
        selected_tasks = set(tasks)

        old_metrics = read_excel_sheet(excel_path, "detail_metrics")
        old_record_details = read_excel_sheet(excel_path, "record_details")
        old_fields = read_excel_sheet(excel_path, "field_agreement")
        old_mismatches = read_excel_sheet(excel_path, "mismatches")
        old_usage = read_excel_sheet(excel_path, "usage_summary")

        df_metrics = merge_incremental_rows(
            old_df=old_metrics,
            new_df=df_metrics,
            selected_ids=selected_ids,
            study_col="study",
            selected_tasks=selected_tasks,
        )
        df_record_details = merge_incremental_rows(
            old_df=old_record_details,
            new_df=df_record_details,
            selected_ids=selected_ids,
            study_col="study_id",
            selected_tasks=selected_tasks,
        )
        df_fields = merge_incremental_rows(
            old_df=old_fields,
            new_df=df_fields,
            selected_ids=selected_ids,
            study_col="study",
            selected_tasks=selected_tasks,
        )
        df_mismatches = merge_incremental_rows(
            old_df=old_mismatches,
            new_df=df_mismatches,
            selected_ids=selected_ids,
            study_col="study_id",
            selected_tasks=selected_tasks,
        )
        df_usage = merge_incremental_rows(
            old_df=old_usage,
            new_df=df_usage,
            selected_ids=selected_ids,
            study_col="study_id",
            drop_extra_ids={"all_studies"},
        )

    df_metrics = normalize_chunk_id_column(df_metrics)
    df_mismatches = normalize_chunk_id_column(df_mismatches)
    df_mishandled_chunks = build_mishandled_chunk_rows(df_mismatches)

    usage_records = df_usage.to_dict("records") if not df_usage.empty else []
    df_usage = (
        pd.DataFrame(add_total_usage_row(usage_records, tasks=usage_tasks))
        if usage_records
        else pd.DataFrame()
    )

    (
        study_macro,
        overall_macro,
        study_weight,
        overall_weight,
        study_micro,
        overall_micro,
    ) = summarize_metrics(df_metrics)

    with pd.ExcelWriter(excel_path, engine="xlsxwriter") as writer:
        df_metrics.to_excel(writer, sheet_name="detail_metrics", index=False)
        df_record_details.to_excel(writer, sheet_name="record_details", index=False)
        study_macro.to_excel(writer, sheet_name="study_macro", index=False)
        overall_macro.to_excel(writer, sheet_name="overall_macro", index=False)
        study_weight.to_excel(writer, sheet_name="study_ID_weighted", index=False)
        overall_weight.to_excel(writer, sheet_name="overall_ID_weighted", index=False)
        study_micro.to_excel(writer, sheet_name="study_micro", index=False)
        overall_micro.to_excel(writer, sheet_name="overall_micro", index=False)
        df_fields.to_excel(writer, sheet_name="field_agreement", index=False)
        df_usage.to_excel(writer, sheet_name="usage_summary", index=False)
        df_mismatches.to_excel(writer, sheet_name="mismatches", index=False)
        df_mishandled_chunks.to_excel(writer, sheet_name="mishandled_chunks", index=False)

    usage_path = output_dir / "enumeration_usage_summary.csv"
    df_usage.to_csv(usage_path, index=False)

    mishandled_path = output_dir / "mishandled_chunks.csv"
    df_mishandled_chunks.to_csv(mishandled_path, index=False)

    print(f"Evaluation workbook saved to: {excel_path}")
    print(f"Usage summary saved to: {usage_path}")
    print(f"Mishandled chunks saved to: {mishandled_path}")
    print(
        "Judge calls: "
        f"called={JUDGE_CALL_STATS['called']}, "
        f"reused={JUDGE_CALL_STATS['reused']}, "
        f"skipped_no_candidates={JUDGE_CALL_STATS['skipped_no_candidates']}, "
        f"skipped_no_gt={JUDGE_CALL_STATS['skipped_no_gt']}, "
        f"skipped_no_pred={JUDGE_CALL_STATS['skipped_no_pred']}, "
        f"skipped_empty_both={JUDGE_CALL_STATS['skipped_empty_both']}, "
        f"skipped_disabled={JUDGE_CALL_STATS['skipped_disabled']}"
    )
    print(f"Judge artifacts directory: {JUDGE_ARTIFACT_ROOT}")


def main() -> None:
    """Evaluate selected studies one group at a time in their own run folders."""
    global PREDICTIONS_DIR, GROUND_TRUTH_DIR, SELECTED_STUDY_IDS, JUDGE_ARTIFACT_ROOT
    for group, study_paths in active_studies_by_group().items():
        PREDICTIONS_DIR = study_paths[0].enumeration_dir
        GROUND_TRUTH_DIR = study_paths[0].enumeration_ground_truth_dir
        SELECTED_STUDY_IDS = [paths.study_id for paths in study_paths]
        JUDGE_ARTIFACT_ROOT = PREDICTIONS_DIR / "_evaluation_judge_artifacts"
        print(f"\n=== Evaluating enumeration {group}: {', '.join(SELECTED_STUDY_IDS)} ===")
        _main_for_current_group()


if __name__ == "__main__":
    main()
