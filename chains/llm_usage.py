import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

try:
    from langchain_community.callbacks.manager import get_openai_callback
except Exception:
    get_openai_callback = None


@dataclass
class LLMUsage:
    provider: str
    model_name: str
    elapsed_s: float
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    total_tokens: Optional[int] = None
    total_cost_usd: Optional[float] = None

    def token_dict(self) -> Dict[str, Optional[int]]:
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
        }

    def flat_token_dict(self) -> Dict[str, Optional[int]]:
        return {
            "tokens_prompt": self.prompt_tokens,
            "tokens_completion": self.completion_tokens,
            "tokens_total": self.total_tokens,
        }

    def metadata_dict(self) -> Dict[str, Any]:
        return {
            "provider": self.provider,
            "model_name": self.model_name,
            "elapsed_s": round(self.elapsed_s, 3),
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "total_cost_usd": self.total_cost_usd,
        }


@dataclass
class LLMCallResult:
    text: str
    usage: LLMUsage


def _int_or_none(value) -> Optional[int]:
    try:
        return int(value)
    except Exception:
        return None


def _float_or_none(value) -> Optional[float]:
    try:
        value = float(value)
    except Exception:
        return None
    return value if value > 0 else None


def _usage_from_callback(cb, provider: str, model_name: str, elapsed_s: float) -> LLMUsage:
    if cb is None:
        return LLMUsage(provider=provider, model_name=model_name, elapsed_s=elapsed_s)

    prompt = _int_or_none(getattr(cb, "prompt_tokens", None))
    completion = _int_or_none(getattr(cb, "completion_tokens", None))
    total = _int_or_none(getattr(cb, "total_tokens", None))

    # For non-OpenAI providers, LangChain callbacks may exist but report all zeros.
    if prompt == 0 and completion == 0 and total == 0:
        prompt = completion = total = None

    return LLMUsage(
        provider=provider,
        model_name=model_name,
        elapsed_s=elapsed_s,
        prompt_tokens=prompt,
        completion_tokens=completion,
        total_tokens=total,
        total_cost_usd=_float_or_none(getattr(cb, "total_cost", None)),
    )


def predict_with_usage(chain, provider: str, model_name: str, **chain_inputs) -> LLMCallResult:
    cb = None

    if get_openai_callback is None:
        start = time.perf_counter()
        text = chain.predict(**chain_inputs)
        elapsed = time.perf_counter() - start
        return LLMCallResult(
            text=text,
            usage=_usage_from_callback(cb, provider, model_name, elapsed),
        )

    with get_openai_callback() as cb:
        start = time.perf_counter()
        text = chain.predict(callbacks=[cb], **chain_inputs)
        elapsed = time.perf_counter() - start

    return LLMCallResult(
        text=text,
        usage=_usage_from_callback(cb, provider, model_name, elapsed),
    )


def summarize_llm_usage(usages: List[Dict[str, Any]]) -> Dict[str, Any]:
    def _sum_optional(key: str):
        vals = [u.get(key) for u in usages if u.get(key) is not None]
        return sum(vals) if vals else None

    return {
        "call_count": len(usages),
        "elapsed_s": round(sum(float(u.get("elapsed_s") or 0.0) for u in usages), 3),
        "prompt_tokens": _sum_optional("prompt_tokens"),
        "completion_tokens": _sum_optional("completion_tokens"),
        "total_tokens": _sum_optional("total_tokens"),
        "total_cost_usd": _sum_optional("total_cost_usd"),
    }

def as_usage_number(value) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0

def summarize_classification_usage(study_id: str, llm_data: list, extra_counts: Dict[str, Any] | None = None) -> Dict[str, Any]:
    row = {
        "study_id": study_id, "summary_type": "per_study",
        "llm_output_chunks": len(llm_data), "llm_call_count": 0,
        "chunks_missing_token_usage": 0, "chunks_missing_time_usage": 0,
        "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0,
        "classification_time_s": 0.0,
    }
    for chunk in llm_data:
        tokens = chunk.get("tokens") or {}
        prompt = as_usage_number(tokens.get("prompt_tokens"))
        completion = as_usage_number(tokens.get("completion_tokens"))
        total = as_usage_number(tokens.get("total_tokens")) or prompt + completion
        elapsed = as_usage_number(chunk.get("classification_time_s"))
        called = elapsed > 0 or total > 0
        row["prompt_tokens"] += prompt
        row["completion_tokens"] += completion
        row["total_tokens"] += total
        row["classification_time_s"] += elapsed
        row["llm_call_count"] += int(called)
        row["chunks_missing_token_usage"] += int(called and total == 0)
        row["chunks_missing_time_usage"] += int(called and elapsed == 0)
    if extra_counts:
        row.update(extra_counts)
    _finalize_usage_row(row, "classification_time_s")
    return row

def summarize_enumeration_usage(study_id: str, prediction_data: list, model_name: str, tasks: Dict[str, str]) -> Dict[str, Any]:
    row = _new_usage_row(study_id, model_name, len(prediction_data), "enumeration_time_s", tasks)
    for entry in prediction_data:
        for enum in entry.get("enumerations", []):
            task = enum.get("task")
            for call in ((enum.get("timings") or {}).get("lists") or []):
                _add_usage_call(row, call, "enumeration_time_s", task, tasks)
    _finalize_usage_row(row, "enumeration_time_s")
    return row

def summarize_extraction_usage(study_id: str, prediction_data: list, model_name: str, tasks: Dict[str, str]) -> Dict[str, Any]:
    row = _new_usage_row(study_id, model_name, len(prediction_data), "extraction_time_s", tasks)
    for entry in prediction_data:
        for task, block in (entry.get("extracted_data") or {}).items():
            if task not in tasks or not isinstance(block, dict):
                continue
            timings = block.get("timings") or {}
            token_counts = timings.get("token_counts") or []
            batch_times = timings.get("batches_s") or []
            for i in range(max(len(token_counts), len(batch_times))):
                call = token_counts[i] if i < len(token_counts) and isinstance(token_counts[i], dict) else {}
                if "duration_s" not in call and "elapsed_s" not in call and i < len(batch_times):
                    call = {**call, "duration_s": batch_times[i]}
                _add_usage_call(row, call, "extraction_time_s", task, tasks)
    _finalize_usage_row(row, "extraction_time_s")
    return row

def add_total_usage_row(rows: list[dict], time_field: str, extra_total_fields: list[str] | None = None) -> list[dict]:
    if not rows:
        return rows
    total_fields = [
        "llm_output_chunks", "llm_call_count",
        "chunks_missing_token_usage", "chunks_missing_time_usage",
        "prompt_tokens", "completion_tokens", "total_tokens", time_field,
    ] + [k for k in rows[0] if k.endswith("_llm_call_count")]
    if extra_total_fields:
        total_fields.extend(extra_total_fields)
    total = {k: sum(as_usage_number(r.get(k)) for r in rows) for k in dict.fromkeys(total_fields)}
    total.update({"study_id": "all_studies", "summary_type": "total"})
    if "model" in rows[0]:
        total["model"] = rows[0].get("model")
    if "has_ground_truth" in rows[0]:
        total["has_ground_truth"] = all(bool(r.get("has_ground_truth")) for r in rows)
    _finalize_usage_row(total, time_field)
    return rows + [total]

def _new_usage_row(study_id: str, model_name: str, chunk_count: int, time_field: str, tasks: Dict[str, str]) -> Dict[str, Any]:
    row = {
        "model": model_name, "study_id": study_id, "summary_type": "per_study",
        "llm_output_chunks": chunk_count, "llm_call_count": 0,
        "chunks_missing_token_usage": 0, "chunks_missing_time_usage": 0,
        "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0,
        time_field: 0.0, "total_cost_usd": None,
    }
    for task in tasks:
        row[f"{task}_llm_call_count"] = 0
    return row

def _add_usage_call(row: Dict[str, Any], call: Dict[str, Any], time_field: str, task: str | None, tasks: Dict[str, str]) -> None:
    prompt = as_usage_number(call.get("prompt_tokens"))
    completion = as_usage_number(call.get("completion_tokens"))
    total = as_usage_number(call.get("total_tokens")) or prompt + completion
    elapsed = as_usage_number(call.get("duration_s", call.get("elapsed_s")))
    row["llm_call_count"] += 1
    if task in tasks:
        row[f"{task}_llm_call_count"] += 1
    row["prompt_tokens"] += prompt
    row["completion_tokens"] += completion
    row["total_tokens"] += total
    row[time_field] += elapsed
    row["chunks_missing_token_usage"] += int(total == 0)
    row["chunks_missing_time_usage"] += int(elapsed == 0)

def _finalize_usage_row(row: Dict[str, Any], time_field: str) -> None:
    for k in ("prompt_tokens", "completion_tokens", "total_tokens"):
        row[k] = int(row.get(k) or 0)
    row[time_field] = round(as_usage_number(row.get(time_field)), 3)
    calls = as_usage_number(row.get("llm_call_count"))
    chunks = as_usage_number(row.get("llm_output_chunks"))
    row["avg_total_tokens_per_llm_chunk"] = round(row["total_tokens"] / chunks, 3) if chunks else 0
    row["avg_time_per_llm_chunk_s"] = round(row[time_field] / chunks, 3) if chunks else 0
    row["avg_total_tokens_per_llm_call"] = round(row["total_tokens"] / calls, 3) if calls else 0
    row["avg_time_per_llm_call_s"] = round(row[time_field] / calls, 3) if calls else 0