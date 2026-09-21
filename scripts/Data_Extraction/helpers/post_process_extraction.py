from __future__ import annotations

import re
from copy import deepcopy
from pathlib import Path
from typing import Any, Callable

try:
    from ..workflow_config import add_project_import_paths
except ImportError:  # Supports direct import from a stage script.
    import sys

    _DATA_EXTRACTION_DIR = Path(__file__).resolve().parents[1]
    if str(_DATA_EXTRACTION_DIR) not in sys.path:
        sys.path.insert(0, str(_DATA_EXTRACTION_DIR))
    from workflow_config import add_project_import_paths

add_project_import_paths()

from schemas import ADSORBENT_FIELDS, PERFORMANCE_FIELDS, REVIEW_FIELDS


POST_PROCESS_VERSION = 11

VALID_TEST_MODES = {"kinetic", "isotherm", "snapshot"}

TARGET_PERFORMANCE_METRIC_FIELDS = [
    "Freundlich_KF_value",
    "Freundlich_n_or_1/n_verbatim",
    "Langmuir_Qm_value",
    "Langmuir_KL_value",
    "Kd_value",
    "Removal_rate",
    "Qe_value",
    "PFO_k1_value",
    "PFO_Qe_value",
    "PSO_k2_value",
    "PSO_v0_value",
    "PSO_Qe_value",
]

PROMPT_PLACEHOLDERS = {
    "",
    "--",
    "-",
    "n/a",
    "none",
    "null",
    "not found",
    "not extractable",
    "not reported",
    "unknown",
}

METRIC_EMPTY_TOKENS = PROMPT_PLACEHOLDERS | {"na"}

MASS_PER_VOLUME_PIPE_RE = re.compile(
    r"_(mg|ug|microg|ng|pg|g|kg|mmol|umol|mumol|mol)\|L\b",
    flags=re.IGNORECASE,
)
FREUNDLICH_CONCENTRATION_FACTOR_RE = re.compile(
    r"(?P<op>[/\*]?)\(?[a-z0-9]+/(?:ml|l)\)?\^\(?(?P<sign>[+-]?)(?P<form>1/n|n)\)?",
    flags=re.IGNORECASE,
)
FREUNDLICH_INVERSE_CONCENTRATION_FACTOR_RE = re.compile(
    r"(?P<op>[/\*]?)\(?(?:ml|l)/[a-z0-9]+\)?\^\(?(?P<sign>[+-]?)(?P<form>1/n|n)\)?",
    flags=re.IGNORECASE,
)
FREUNDLICH_VOLUME_EXPONENT_RE = re.compile(
    r"(?P<op>[/\*]?)(?<![a-z0-9])(?:ml|l)\^\(?(?P<sign>[+-]?)(?P<form>1/n|n)\)?",
    flags=re.IGNORECASE,
)
SCHEMA_FIELDS = (
    set(ADSORBENT_FIELDS)
    | set(PERFORMANCE_FIELDS)
    | set(REVIEW_FIELDS)
    | {
        "Adsorbent_id",
        "adsorbent_id",
        "name_full",
        "name_commercial",
        "name_abbreviation",
        "adsorbent_scope",
        "adsorbent_category",
        "adsorbent_subcategory",
    }
)
CANONICAL_BY_LOWER = {field.lower(): field for field in SCHEMA_FIELDS}

ADSORBENT_IDENTIFIER_FIELDS = {
    "Adsorbent_id",
    "adsorbent_id",
    "name_full",
    "name_commercial",
    "name_abbreviation",
    "adsorbent_scope",
    "adsorbent_category",
    "adsorbent_subcategory",
}


def fix_unit_pipes(value: Any) -> Any:
    """Repair common parsed condition tokens such as PFOS_5_ng|L."""
    if not isinstance(value, str):
        return value
    return MASS_PER_VOLUME_PIPE_RE.sub(lambda match: f"_{match.group(1)}/L", value)


def _canonical_field(field: str) -> str:
    return CANONICAL_BY_LOWER.get(str(field or "").strip().lower(), field)


def _is_blank(value: Any) -> bool:
    return value is None or (isinstance(value, str) and value.strip() == "")


def _token(value: Any) -> str:
    return str(value or "").strip().casefold()


def _is_prompt_placeholder(value: Any, *, include_na: bool = False) -> bool:
    token = _token(value)
    if token in PROMPT_PLACEHOLDERS:
        return True
    return include_na and token == "na"


def _get_ci(record: dict[str, Any], field: str) -> tuple[str, Any]:
    target = str(field).casefold()
    for key, value in record.items():
        if str(key).casefold() == target:
            return key, value
    return "", None


def _set_ci(record: dict[str, Any], field: str, value: Any) -> str:
    key, _existing = _get_ci(record, field)
    if not key:
        key = _canonical_field(field)
    record[key] = value
    return key


def _set_canonical_ci(record: dict[str, Any], field: str, value: Any) -> str:
    canonical = _canonical_field(field)
    key, _existing = _get_ci(record, field)
    if key and key != canonical:
        record.pop(key, None)
    record[canonical] = value
    return canonical


def _pop_ci(record: dict[str, Any], field: str) -> None:
    key, _value = _get_ci(record, field)
    if key:
        record.pop(key, None)


def _append_record_message(record: dict[str, Any], field: str, message: str) -> None:
    """Append a semicolon-delimited provenance message to an extracted record."""
    message = str(message or "").strip()
    if not message:
        return
    key, existing = _get_ci(record, field)
    if _is_blank(existing) or _is_prompt_placeholder(existing, include_na=True):
        _set_ci(record, field, message)
        return
    parts = [part.strip() for part in str(existing).split(";") if part.strip()]
    if message not in parts:
        parts.append(message)
    record[key] = "; ".join(parts)


def _append_review_reason(record: dict[str, Any], reason: str) -> None:
    reason = str(reason or "").strip()
    if not reason:
        return
    _set_ci(record, "Require_review", "Yes")
    _append_record_message(record, "Review_reason", reason)


def _append_normalization_trace(record: dict[str, Any], message: str) -> None:
    """Record non-blocking extraction provenance for downstream trace output."""
    _append_record_message(record, "normalization_trace", message)


def split_performance_id(value: Any) -> tuple[str, str, str, str] | None:
    text = str(fix_unit_pipes(value) or "").strip()
    parts = [part.strip() for part in text.split("|")]
    if len(parts) != 4 or any(not part for part in parts):
        return None
    pfas_name, adsorbent_id, test_mode, condition = parts
    test_mode = test_mode.casefold()
    if test_mode not in VALID_TEST_MODES:
        return None
    return pfas_name, adsorbent_id, test_mode, condition


def _field_values_match(field: str, existing: Any, expected: Any) -> bool:
    existing_text = str(fix_unit_pipes(existing) or "").strip()
    expected_text = str(expected or "").strip()
    if field == "Test_mode":
        return existing_text.casefold() == expected_text.casefold()
    return existing_text == expected_text


def _should_enforce_performance_id_field(field: str, expected: Any) -> bool:
    if field in {"PFAS_name", "Adsorbent_id"}:
        return not _is_prompt_placeholder(expected, include_na=True)
    return True


def _normalize_string_values(record: dict[str, Any]) -> None:
    for key, value in list(record.items()):
        if isinstance(value, str):
            record[key] = fix_unit_pipes(value)
        elif isinstance(value, dict):
            _normalize_string_values(value)


def _prune_prompt_placeholders(
    record: dict[str, Any],
    *,
    allowed_na_fields: set[str] | None = None,
) -> None:
    allowed = {field.casefold() for field in (allowed_na_fields or set())}
    for key, value in list(record.items()):
        if isinstance(value, dict):
            _prune_prompt_placeholders(value, allowed_na_fields=allowed_na_fields)
            if not value:
                record.pop(key, None)
            continue
        include_na = str(key).casefold() not in allowed
        if _is_prompt_placeholder(value, include_na=include_na):
            record.pop(key, None)


def _has_metric_value(record: dict[str, Any], field: str) -> bool:
    _key, value = _get_ci(record, field)
    if isinstance(value, list):
        return any(not _is_prompt_placeholder(item, include_na=True) for item in value)
    return not _is_prompt_placeholder(value, include_na=True)


def has_target_performance_metric(record: dict[str, Any]) -> bool:
    return any(_has_metric_value(record, field) for field in TARGET_PERFORMANCE_METRIC_FIELDS)


def _canonical_freundlich_form(value: Any) -> str:
    text = re.sub(r"\s+", "", str(value or "")).casefold()
    if text == "n":
        return "n"
    if text == "1/n":
        return "1/n"
    return ""


def _normalize_unit_expression(value: Any) -> str:
    text = str(value or "").casefold()
    text = (
        text.replace("\u00b5", "u")
        .replace("\u03bc", "u")
        .replace("\u2212", "-")
        .replace("\u2013", "-")
        .replace("\u2014", "-")
        .replace("\u00b7", "*")
        .replace("**", "^")
        .replace("{", "(")
        .replace("}", ")")
    )
    return re.sub(r"\s+", "", text)


def _unique_freundlich_form(candidates: list[str]) -> str:
    forms = {form for form in candidates if form in {"n", "1/n"}}
    return next(iter(forms)) if len(forms) == 1 else ""


def infer_freundlich_equation_form_from_kf_unit(value: Any) -> str:
    """Infer equation form from explicit Freundlich KF unit exponents."""
    if _is_prompt_placeholder(value, include_na=True):
        return ""
    unit = _normalize_unit_expression(value)
    candidates: list[str] = []

    for match in FREUNDLICH_CONCENTRATION_FACTOR_RE.finditer(unit):
        is_divided = match.group("op") == "/"
        is_negative = match.group("sign") == "-"
        if is_divided != is_negative:
            candidates.append(_canonical_freundlich_form(match.group("form")))

    for match in FREUNDLICH_INVERSE_CONCENTRATION_FACTOR_RE.finditer(unit):
        is_divided = match.group("op") == "/"
        is_negative = match.group("sign") == "-"
        if is_divided == is_negative:
            candidates.append(_canonical_freundlich_form(match.group("form")))

    for match in FREUNDLICH_VOLUME_EXPONENT_RE.finditer(unit):
        if match.group("op") != "/" and match.group("sign") != "-":
            candidates.append(_canonical_freundlich_form(match.group("form")))

    return _unique_freundlich_form(candidates)


def _infer_freundlich_equation_form_from_unit_record(
    record: dict[str, Any],
) -> tuple[str, str]:
    _key, value = _get_ci(record, "Freundlich_KF_unit")
    form = infer_freundlich_equation_form_from_kf_unit(value)
    if form:
        return form, "Freundlich_KF_unit"
    return "", ""


def post_process_performance_record(record: dict[str, Any]) -> dict[str, Any]:
    """Enforce current performance extraction contracts after LLM parsing."""
    for legacy_key in (
        "Adsorbent_name",
        "adsorbent_name",
        "Specific_Condition",
        "specific_condition",
        "Specific_Conditions",
        "specific_conditions",
    ):
        _pop_ci(record, legacy_key)

    _normalize_string_values(record)
    _prune_prompt_placeholders(
        record,
        allowed_na_fields={"Differentiating_Condition"},
    )

    _pid_key, performance_id = _get_ci(record, "Performance_id")
    parsed = split_performance_id(performance_id)
    if parsed is None:
        if not _is_blank(performance_id):
            _append_review_reason(
                record,
                "Invalid Performance_id; expected PFAS|Adsorbent_id|Test_mode|Differentiating_Condition",
            )
        return record

    pfas_name, adsorbent_id, test_mode, condition = parsed
    canonical_pid = f"{pfas_name}|{adsorbent_id}|{test_mode}|{condition}"
    _set_ci(record, "Performance_id", canonical_pid)

    expected_fields = {
        "PFAS_name": pfas_name,
        "Adsorbent_id": adsorbent_id,
        "Test_mode": test_mode,
        "Differentiating_Condition": condition,
    }
    for field, expected in expected_fields.items():
        if not _should_enforce_performance_id_field(field, expected):
            continue
        key, existing = _get_ci(record, field)
        if _is_blank(existing) or _is_prompt_placeholder(existing, include_na=True):
            _set_ci(record, field, expected)
            continue
        if not _field_values_match(field, existing, expected):
            record[key] = expected
            _append_review_reason(
                record,
                f"{field} corrected to match Performance_id",
            )

    return record


def _has_adsorbent_property(record: dict[str, Any]) -> bool:
    for field in ADSORBENT_FIELDS:
        _key, value = _get_ci(record, field)
        if not _is_prompt_placeholder(value, include_na=True):
            return True
    return False


def post_process_adsorbent_record(record: dict[str, Any]) -> dict[str, Any]:
    """Prune placeholders from adsorbent property records."""
    _normalize_string_values(record)
    _prune_prompt_placeholders(record)

    _key, adsorbent_id = _get_ci(record, "Adsorbent_id")
    _key2, adsorbent_id_lc = _get_ci(record, "adsorbent_id")
    if _is_blank(adsorbent_id) and not _is_blank(adsorbent_id_lc):
        _set_ci(record, "Adsorbent_id", adsorbent_id_lc)
    if _is_blank(adsorbent_id) and _is_blank(adsorbent_id_lc):
        _append_review_reason(record, "Missing Adsorbent_id")

    return record


def _section(data: dict[str, Any], task: str) -> tuple[str, dict[str, Any]]:
    for key, value in data.items():
        if str(key).casefold() != task.casefold():
            continue
        if isinstance(value, dict):
            return key, value
        if isinstance(value, list):
            data[key] = {"records": value}
            return key, data[key]
    return "", {}


def _records(section: dict[str, Any]) -> list[dict[str, Any]]:
    records = section.get("records")
    return records if isinstance(records, list) else []


def _sync_counts(output: dict[str, Any], dropped_by_task: dict[str, list[str]]) -> None:
    data = output.get("extracted_data")
    if not isinstance(data, dict):
        return

    counts = output.get("extraction_counts")
    by_task = counts.get("by_task") if isinstance(counts, dict) else {}
    total = 0

    for task in ("adsorbent", "performance"):
        _section_key, section = _section(data, task)
        records = _records(section)
        if not section and not records:
            continue
        extracted = len(records)
        total += extracted
        section["extracted_count"] = extracted

        task_counts = by_task.get(task) if isinstance(by_task, dict) else None
        if isinstance(task_counts, dict):
            task_counts["extracted"] = extracted
            dropped_ids = dropped_by_task.get(task) or []
            if dropped_ids:
                existing = task_counts.get("missing_ids")
                if not isinstance(existing, list):
                    existing = []
                task_counts["missing_ids"] = _dedupe([*existing, *dropped_ids])

    if isinstance(counts, dict):
        counts["total_extracted"] = total


def _dedupe(values: list[Any]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for value in values:
        text = str(value or "").strip()
        if not text or text in seen:
            continue
        seen.add(text)
        out.append(text)
    return out


def _merge_delimited_values(
    existing: Any,
    incoming: Any,
    delimiter: str,
    *,
    split_pattern: str = r"[,;]",
) -> str:
    pieces: list[str] = []
    for value in (existing, incoming):
        if _is_prompt_placeholder(value, include_na=True):
            continue
        pieces.extend(
            part.strip()
            for part in re.split(split_pattern, str(value))
            if part.strip()
        )
    return delimiter.join(_dedupe(pieces))


def _adsorbent_identity(record: dict[str, Any]) -> str:
    _key, value = _get_ci(record, "Adsorbent_id")
    if _is_prompt_placeholder(value, include_na=True):
        return ""
    return str(value).strip().casefold()


def _adsorbent_display_id(record: dict[str, Any]) -> str:
    _key, value = _get_ci(record, "Adsorbent_id")
    return str(value or "").strip()


def _values_match_for_merge(field: str, existing: Any, incoming: Any) -> bool:
    existing_text = str(fix_unit_pipes(existing) or "").strip()
    incoming_text = str(fix_unit_pipes(incoming) or "").strip()
    if str(field).casefold() in {"adsorbent_id", "adsorbent_scope"}:
        return existing_text.casefold() == incoming_text.casefold()
    return existing_text == incoming_text


def _merge_duplicate_adsorbent_record(
    base: dict[str, Any],
    incoming: dict[str, Any],
) -> list[str]:
    conflicts: list[str] = []
    for key, value in incoming.items():
        if _is_prompt_placeholder(value, include_na=True):
            continue
        base_key, existing = _get_ci(base, key)
        if not base_key:
            base[key] = value
            continue
        if _is_prompt_placeholder(existing, include_na=True):
            base[base_key] = value
            continue
        if _values_match_for_merge(key, existing, value):
            continue

        key_norm = str(key).casefold()
        if key_norm == "data_provenance":
            merged = _merge_delimited_values(existing, value, ",")
            if merged:
                base[base_key] = merged
            continue
        if key_norm == "review_reason":
            merged = _merge_delimited_values(
                existing,
                value,
                "; ",
                split_pattern=r";",
            )
            if merged:
                base[base_key] = merged
            continue
        if key_norm == "require_review":
            if str(value).strip().casefold() == "yes":
                base[base_key] = "Yes"
            continue

        conflicts.append(_canonical_field(str(key)))

    return conflicts


def _consolidate_adsorbent_records(
    records: list[dict[str, Any]],
    summary: dict[str, Any],
) -> list[dict[str, Any]]:
    consolidated: list[dict[str, Any]] = []
    by_id: dict[str, dict[str, Any]] = {}
    conflicts_by_id: dict[str, list[str]] = {}

    for record in records:
        identity = _adsorbent_identity(record)
        if not identity:
            consolidated.append(record)
            continue
        if identity not in by_id:
            by_id[identity] = record
            consolidated.append(record)
            continue

        base = by_id[identity]
        display_id = _adsorbent_display_id(base) or _adsorbent_display_id(record)
        conflicts = _merge_duplicate_adsorbent_record(base, record)
        summary["adsorbent_records_consolidated"] += 1
        summary["consolidated_adsorbent_ids"].append(display_id or identity)
        if conflicts:
            conflicts_by_id.setdefault(identity, []).extend(conflicts)

    for identity, conflicts in conflicts_by_id.items():
        record = by_id.get(identity)
        if not record:
            continue
        _append_review_reason(
            record,
            "Duplicate Adsorbent_id records consolidated; conflicting fields kept from first record: "
            + ", ".join(_dedupe(conflicts)),
        )

    return consolidated


def _process_adsorbent_section(
    output: dict[str, Any],
    summary: dict[str, Any],
) -> list[str]:
    data = output.get("extracted_data")
    if not isinstance(data, dict):
        return []
    _key, section = _section(data, "adsorbent")
    records = _records(section)
    if not records:
        return []

    kept: list[dict[str, Any]] = []
    dropped: list[str] = []
    for record in records:
        if not isinstance(record, dict):
            continue
        post_process_adsorbent_record(record)
        if not _has_adsorbent_property(record):
            _id_key, adsorbent_id = _get_ci(record, "Adsorbent_id")
            _id2_key, adsorbent_id_lc = _get_ci(record, "adsorbent_id")
            dropped_id = adsorbent_id or adsorbent_id_lc
            if dropped_id:
                dropped.append(str(dropped_id))
            summary["adsorbent_records_dropped"] += 1
            continue
        kept.append(record)

    consolidated = _consolidate_adsorbent_records(kept, summary)

    if len(consolidated) != len(records):
        section["records"] = consolidated
    return dropped


def _process_performance_section(
    output: dict[str, Any],
    summary: dict[str, Any],
) -> list[str]:
    data = output.get("extracted_data")
    if not isinstance(data, dict):
        return []
    _key, section = _section(data, "performance")
    records = _records(section)
    if not records:
        return []

    kept: list[dict[str, Any]] = []
    dropped: list[str] = []
    for record in records:
        if not isinstance(record, dict):
            continue
        post_process_performance_record(record)
        if not has_target_performance_metric(record):
            _pid_key, performance_id = _get_ci(record, "Performance_id")
            if performance_id:
                dropped.append(str(performance_id))
            summary["performance_records_dropped"] += 1
            continue
        kept.append(record)

    if len(kept) != len(records):
        section["records"] = kept
    return dropped


def _iter_performance_records(outputs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for output in outputs:
        if not isinstance(output, dict):
            continue
        data = output.get("extracted_data")
        if not isinstance(data, dict):
            continue
        _key, section = _section(data, "performance")
        records.extend(record for record in _records(section) if isinstance(record, dict))
    return records


def _performance_output_label(output: dict[str, Any], output_index: int) -> str:
    """Describe the source-table context represented by one extraction output."""
    parts = [f"output={output_index}"]
    chunk_id = str(output.get("chunk_id") or "").strip()
    if chunk_id:
        parts.append(f"chunk_id={chunk_id}")
    routing = output.get("source_routing")
    if isinstance(routing, dict):
        sources = routing.get("referenced_sources") or []
        if isinstance(sources, list) and sources:
            parts.append("sources=" + ",".join(str(source) for source in sources))
    return "; ".join(parts)


def _iter_performance_output_contexts(
    outputs: list[dict[str, Any]],
) -> list[tuple[str, list[dict[str, Any]]]]:
    """Keep safety-guard evidence within the extraction output that supplied it."""
    contexts: list[tuple[str, list[dict[str, Any]]]] = []
    for output_index, output in enumerate(outputs):
        if not isinstance(output, dict):
            continue
        data = output.get("extracted_data")
        if not isinstance(data, dict):
            continue
        _key, section = _section(data, "performance")
        records = [record for record in _records(section) if isinstance(record, dict)]
        if records:
            contexts.append(
                (
                    _performance_output_label(output, output_index),
                    records,
                )
            )
    return contexts


def _has_freundlich_context(record: dict[str, Any]) -> bool:
    return any(
        _has_metric_value(record, field)
        for field in (
            "Freundlich_KF_value",
            "Freundlich_KF_unit",
            "Freundlich_n_or_1/n_verbatim",
            "Freundlich_equation_form",
            "Freundlich_R2",
        )
    )


def _freundlich_context_label(record: dict[str, Any]) -> str:
    """Return the narrowest available provenance key for form inference."""
    _key, provenance = _get_ci(record, "data_provenance")
    if not _is_prompt_placeholder(provenance, include_na=True):
        text = re.sub(r"\s+", "", str(provenance or ""))
        if text:
            return f"data_provenance={text}"
    return "unscoped study context"


def _group_freundlich_contexts(
    records: list[dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    """Group records before applying a shared Freundlich-form decision."""
    groups: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        if not _has_freundlich_context(record):
            continue
        groups.setdefault(_freundlich_context_label(record), []).append(record)
    return groups


def _collect_freundlich_form_evidence(
    records: list[dict[str, Any]],
    infer_form: Callable[[dict[str, Any]], tuple[str, str]],
) -> tuple[str, dict[str, int]]:
    """Return a form only when the supplied evidence is internally consistent."""
    evidence_counts = {"n": 0, "1/n": 0}
    for record in records:
        form, _source = infer_form(record)
        if form in evidence_counts:
            evidence_counts[form] += 1
    return _unique_freundlich_form(
        [form for form, count in evidence_counts.items() for _ in range(count)]
    ), evidence_counts


def _format_freundlich_evidence(evidence_counts: dict[str, int]) -> str:
    return ", ".join(
        f"{form}={count}" for form, count in evidence_counts.items() if count
    ) or "none"


def _infer_missing_freundlich_equation_form_for_context(
    records: list[dict[str, Any]],
    context_label: str,
) -> tuple[str, dict[str, int], str, str, str]:
    """Fill only missing forms from unambiguous, non-equation evidence.

    Equation reading is intentionally left to the LLM: extracted text often
    contains LaTex or conversion artifacts that this guard cannot interpret
    reliably. The guard relies only on readable KF-unit evidence.
    """
    unit_form, unit_evidence = _collect_freundlich_form_evidence(
        records,
        _infer_freundlich_equation_form_from_unit_record,
    )
    if unit_form:
        source = (
            f"{context_label}: KF-unit evidence "
            f"({_format_freundlich_evidence(unit_evidence)})"
        )
        return unit_form, unit_evidence, source, "", "kf_unit"

    if any(unit_evidence.values()):
        return (
            "",
            unit_evidence,
            "",
            "conflicting KF-unit evidence "
            f"({_format_freundlich_evidence(unit_evidence)})",
            "unresolved",
        )

    return "", {}, "", "no readable KF-unit evidence", "unresolved"


def _apply_study_performance_context(
    outputs: list[dict[str, Any]],
    summary: dict[str, Any],
) -> None:
    records = _iter_performance_records(outputs)
    if not records:
        return

    all_evidence_counts = {"n": 0, "1/n": 0}
    safety_fill_counts = {
        "kf_unit": 0,
        "unresolved": 0,
    }
    for output_label, output_records in _iter_performance_output_contexts(outputs):
        for provenance_label, context_records in _group_freundlich_contexts(output_records).items():
            context_label = f"{output_label}; {provenance_label}"
            missing_records: list[dict[str, Any]] = []
            for record in context_records:
                _key, existing = _get_ci(record, "Freundlich_equation_form")
                if _canonical_freundlich_form(existing):
                    continue
                _pop_ci(record, "Freundlich_equation_form")
                missing_records.append(record)
            if not missing_records:
                continue

            form, evidence_counts, source, unresolved_reason, method = (
                _infer_missing_freundlich_equation_form_for_context(
                    context_records,
                    context_label,
                )
            )
            for key in all_evidence_counts:
                all_evidence_counts[key] += evidence_counts.get(key, 0)
            safety_fill_counts[method] += len(missing_records)
            if form:
                for record in missing_records:
                    _set_ci(record, "Freundlich_equation_form", form)
                    _append_normalization_trace(
                        record,
                        "Freundlich_equation_form filled by post-processing safety guard "
                        f"from {source}",
                    )
                continue

            for record in missing_records:
                _append_review_reason(
                    record,
                    f"Freundlich_equation_form left blank for {context_label}: "
                    f"{unresolved_reason}",
                )

    summary["freundlich_equation_form_evidence_counts"] = {
        key: value for key, value in all_evidence_counts.items() if value
    }
    summary["freundlich_equation_form_safety_fills"] = {
        key: value for key, value in safety_fill_counts.items() if value
    }
    resolved_forms = {
        _canonical_freundlich_form(_get_ci(record, "Freundlich_equation_form")[1])
        for record in records
        if _has_freundlich_context(record)
    }
    resolved_forms.discard("")
    all_forms_present = all(
        _canonical_freundlich_form(_get_ci(record, "Freundlich_equation_form")[1])
        for record in records
        if _has_freundlich_context(record)
    )
    if len(resolved_forms) == 1 and all_forms_present:
        summary["freundlich_equation_form_study_level"] = resolved_forms.pop()


def process_outputs(outputs: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Return post-processed extraction outputs and a compact audit summary."""
    original = deepcopy(outputs)
    processed = deepcopy(outputs)
    summary: dict[str, Any] = {
        "changed": False,
        "adsorbent_records_dropped": 0,
        "adsorbent_records_consolidated": 0,
        "performance_records_dropped": 0,
        "dropped_adsorbent_ids": [],
        "consolidated_adsorbent_ids": [],
        "dropped_performance_ids": [],
        "freundlich_equation_form_study_level": "",
        "freundlich_equation_form_evidence_counts": {},
        "freundlich_equation_form_safety_fills": {},
    }

    for output in processed:
        if not isinstance(output, dict):
            continue
        dropped_by_task = {
            "adsorbent": _process_adsorbent_section(output, summary),
            "performance": _process_performance_section(output, summary),
        }
        summary["dropped_adsorbent_ids"].extend(dropped_by_task["adsorbent"])
        summary["dropped_performance_ids"].extend(dropped_by_task["performance"])
        _sync_counts(output, dropped_by_task)

    _apply_study_performance_context(processed, summary)

    summary["dropped_adsorbent_ids"] = _dedupe(summary["dropped_adsorbent_ids"])
    summary["consolidated_adsorbent_ids"] = _dedupe(summary["consolidated_adsorbent_ids"])
    summary["dropped_performance_ids"] = _dedupe(summary["dropped_performance_ids"])
    summary["changed"] = processed != original
    return processed, summary


def _record_count(outputs: list[dict[str, Any]], task: str) -> int:
    total = 0
    for output in outputs or []:
        if not isinstance(output, dict):
            continue
        section = ((output.get("extracted_data") or {}).get(task) or {})
        if not isinstance(section, dict):
            continue
        records = section.get("records") or []
        if isinstance(records, list):
            total += len(records)
    return total


def process_chain_artifact_payload(payload: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """Post-process a chain artifact payload before saving or reuse."""
    if not isinstance(payload, dict):
        return payload, {"changed": False}
    outputs = payload.get("outputs")
    if not isinstance(outputs, list):
        return payload, {"changed": False}

    processed_outputs, summary = process_outputs(outputs)
    if not summary.get("changed"):
        return payload, summary

    processed_payload = deepcopy(payload)
    processed_payload["outputs"] = processed_outputs
    processed_payload["output_count"] = len(processed_outputs)
    task = str(processed_payload.get("task") or "")
    if task:
        processed_payload["record_count"] = _record_count(processed_outputs, task)
    processed_payload["post_process"] = {
        "version": POST_PROCESS_VERSION,
        "adsorbent_records_dropped": summary["adsorbent_records_dropped"],
        "adsorbent_records_consolidated": summary["adsorbent_records_consolidated"],
        "performance_records_dropped": summary["performance_records_dropped"],
        "dropped_adsorbent_ids": summary["dropped_adsorbent_ids"],
        "consolidated_adsorbent_ids": summary["consolidated_adsorbent_ids"],
        "dropped_performance_ids": summary["dropped_performance_ids"],
        "freundlich_equation_form_study_level": summary[
            "freundlich_equation_form_study_level"
        ],
        "freundlich_equation_form_evidence_counts": summary[
            "freundlich_equation_form_evidence_counts"
        ],
        "freundlich_equation_form_safety_fills": summary[
            "freundlich_equation_form_safety_fills"
        ],
    }
    return processed_payload, summary


__all__ = [
    "fix_unit_pipes",
    "has_target_performance_metric",
    "post_process_adsorbent_record",
    "post_process_performance_record",
    "process_chain_artifact_payload",
    "process_outputs",
    "split_performance_id",
]
