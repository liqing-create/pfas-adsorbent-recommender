"""Internal batching helpers used by :mod:`extract`.

This module intentionally has no standalone workflow, study selection, or
input/output directories.  ``extract.py`` configures the LLM/error-log
context for each selected study group before using these helpers.
"""

from __future__ import annotations

import re
import time
import traceback
from collections import defaultdict, deque
from pathlib import Path
from typing import Any

try:
    from ..workflow_config import add_project_import_paths
except ImportError:  # Supports direct import when extract.py is run as a file.
    import sys

    _DATA_EXTRACTION_DIR = Path(__file__).resolve().parents[1]
    if str(_DATA_EXTRACTION_DIR) not in sys.path:
        sys.path.insert(0, str(_DATA_EXTRACTION_DIR))
    from workflow_config import add_project_import_paths

add_project_import_paths()

from chains.llm_usage import predict_with_usage


LLM_PROVIDER = "openai"
LLM_MODEL_NAME = ""
ERROR_LOG_ROOT = ""
MULTI_KEYS: set[str] = set()

NORMALIZER = re.compile(r"\s*[/<>|]+\s*")
RECORD_SEPARATOR_RE = re.compile(r"(?m)^\s*-{3,}\s*(?:\([^)]*\))?\s*$")
NO_VALID_RECORDS = "NO_VALID_RECORDS"


def configure_runtime(*, provider: str, model_name: str, error_log_root: str | Path) -> None:
    """Set the per-study runtime values supplied by ``extract.py``."""
    global LLM_PROVIDER, LLM_MODEL_NAME, ERROR_LOG_ROOT
    LLM_PROVIDER = provider
    LLM_MODEL_NAME = model_name
    ERROR_LOG_ROOT = str(error_log_root)


def normalize_id(raw: str) -> str:
    cleaned = NORMALIZER.sub("|", str(raw or "").strip())
    return "|".join(part.strip() for part in cleaned.split("|") if part.strip())


def dump_error_context(
    study_id: str,
    chunk_id: str,
    prompt_text: str,
    raw_text: str,
    warnings: list[str],
    exc: Exception,
    timings: list[float] | None = None,
) -> None:
    """Append complete, study-scoped context for an extraction failure."""
    if not ERROR_LOG_ROOT:
        raise RuntimeError("extract_by_chunk runtime was not configured with an error-log directory")
    root = Path(ERROR_LOG_ROOT)
    root.mkdir(parents=True, exist_ok=True)
    path = root / f"{study_id}_{chunk_id}_error.txt"
    with path.open("a", encoding="utf-8") as handle:
        handle.write("\n" + "=" * 80 + "\n")
        handle.write(f"{study_id} | {chunk_id}\n{type(exc).__name__}: {exc}\n\n")
        if warnings:
            handle.write("--- warnings ---\n" + "\n".join(warnings) + "\n\n")
        handle.write("--- prompt ---\n" + prompt_text.rstrip() + "\n\n")
        handle.write("--- raw LLM output ---\n" + raw_text.rstrip() + "\n\n")
        if timings:
            handle.write("--- extraction timings (s) ---\n")
            handle.writelines(f"{value:.3f}\n" for value in timings)
            handle.write("\n")
        handle.write("--- traceback ---\n" + traceback.format_exc() + "\n")


def parse_flat(
    text: str,
    fmap: dict,
    *,
    keep_unknown: bool = True,
    extras_key: str = "__extras__",
    preserve_case_for_extras: bool = True,
) -> dict:
    """Parse one key/value record while preserving unmapped LLM fields."""
    output: dict[str, Any] = {}
    fmap_norm = {str(key).lower(): value for key, value in fmap.items()}
    multi_norm = {key.lower() for key in MULTI_KEYS}
    extras: dict[str, str] | None = {} if keep_unknown else None

    for line in text.splitlines():
        if ":" not in line:
            continue
        raw_key, raw_value = line.split(":", 1)
        key = raw_key.strip().lower()
        value = raw_value.strip()
        if key in fmap_norm:
            expected_type = fmap_norm[key]
            if key in multi_norm:
                output[key] = [part.strip() for part in value.split(";") if part.strip()]
            elif expected_type is float:
                try:
                    output[key] = float(value)
                except ValueError:
                    output[key] = None
            else:
                output[key] = value
        elif extras is not None:
            extras[raw_key.strip() if preserve_case_for_extras else key] = value

    if extras:
        output[extras_key] = extras
    return output


def parse_records(text: str, fmap: dict) -> list[dict]:
    normalized = RECORD_SEPARATOR_RE.sub("\n\n", str(text or "").strip())
    if normalized == NO_VALID_RECORDS:
        return []
    records: list[dict] = []
    for block in re.split(r"\n\s*\n", normalized):
        record = parse_flat(block, fmap, keep_unknown=True)
        if record:
            records.append(record)
    return records


def extract_with_batch(
    text: str,
    figure_captions: str,
    chain,
    llm,
    fmap: dict,
    id_key: str,
    record_ids: list[str],
    batch_size: int,
    ctx_prompts: list[str] | None = None,
    ctx_raws: list[str] | None = None,
    ctx_warnings: list[str] | None = None,
    ctx_timings: list[float] | None = None,
    ctx_token_info: list[dict] | None = None,
    *,
    prompt_inputs: dict | None = None,
    max_retries_per_id: int = 1,
) -> tuple[list[dict], list[float], float]:
    """Extract records in bounded batches and retry only missing IDs."""
    if not LLM_MODEL_NAME:
        raise RuntimeError("extract_by_chunk runtime was not configured with an LLM model")
    if batch_size < 1:
        raise ValueError("batch_size must be at least one")

    all_records: list[dict] = []
    if ctx_timings is not None:
        ctx_timings.clear()
    if ctx_token_info is not None:
        ctx_token_info.clear()

    started = time.perf_counter()
    pending: deque[str] = deque(record_ids[:batch_size])
    next_index = len(pending)
    attempts: dict[str, int] = defaultdict(int)
    batch_number = 0

    while pending:
        batch = [pending.popleft() for _ in range(min(batch_size, len(pending)))]
        # The first item is not included by the comprehension above when the
        # queue contains exactly one element.
        if not batch and pending:
            batch.append(pending.popleft())
        if not batch:
            # ``pending`` may have been empty after the list comprehension.
            break

        batch_number += 1
        all_prompt_values = {
            "enriched_text": text,
            "figure_captions": figure_captions or "",
            "record_to_extract": "; ".join(batch),
            **(prompt_inputs or {}),
        }
        input_variables = set(chain.prompt.input_variables)
        prompt_values = {
            key: value for key, value in all_prompt_values.items() if key in input_variables
        }
        prompt = chain.prompt.format_prompt(**prompt_values).to_string()
        print(f"[{id_key}] Prompt batch {batch_number}:\n{prompt}\n")
        if ctx_prompts is not None:
            ctx_prompts.append(prompt)

        call = predict_with_usage(
            chain,
            provider=LLM_PROVIDER,
            model_name=LLM_MODEL_NAME,
            **prompt_values,
        )
        raw = call.text
        usage = call.usage
        if ctx_raws is not None:
            ctx_raws.append(raw)
        if ctx_timings is not None:
            ctx_timings.append(usage.elapsed_s)
        if ctx_token_info is not None:
            ctx_token_info.append(
                {
                    "provider": usage.provider,
                    "model_name": usage.model_name,
                    "duration_s": round(usage.elapsed_s, 3),
                    "prompt_tokens": usage.prompt_tokens,
                    "completion_tokens": usage.completion_tokens,
                    "total_tokens": usage.total_tokens,
                    "total_cost_usd": usage.total_cost_usd,
                }
            )

        print(f"[{id_key}] Raw output batch {batch_number}:\n{raw}\n")
        expected = {normalize_id(value) for value in batch}
        no_valid_records = str(raw or "").strip() == NO_VALID_RECORDS
        records = parse_records(raw, fmap)
        # The sentinel means no ID in this batch produced a record.  Mark the
        # batch handled so it is not retried and cannot create a placeholder.
        returned: set[str] = set(expected) if no_valid_records else set()
        for record in records:
            record_id = record.get(id_key.lower())
            normalized_id = normalize_id(record_id) if record_id else ""
            if normalized_id not in expected:
                message = (
                    f"[{id_key}] ID mismatch: returned {record_id!r} "
                    f"(normalized {normalized_id!r}) not in {batch!r}"
                )
                print(message)
                if ctx_warnings is not None:
                    ctx_warnings.append(message)
            else:
                returned.add(normalized_id)
            all_records.append(record)

        retry = []
        for value in batch:
            normalized_id = normalize_id(value)
            attempts[normalized_id] += 1
            if normalized_id not in returned and attempts[normalized_id] <= max_retries_per_id:
                retry.append(value)
        if retry:
            print(f"[{id_key}] scheduling retry for missing IDs: {retry}")
            pending = deque(retry) + pending
        if not pending and next_index < len(record_ids):
            next_batch_end = min(next_index + batch_size, len(record_ids))
            pending.extend(record_ids[next_index:next_batch_end])
            next_index = next_batch_end

    return all_records, (ctx_timings or []), time.perf_counter() - started


__all__ = [
    "NO_VALID_RECORDS",
    "configure_runtime",
    "dump_error_context",
    "extract_with_batch",
    "normalize_id",
    "parse_flat",
    "parse_records",
]
