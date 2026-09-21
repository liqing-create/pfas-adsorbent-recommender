"""Synchronize approved Docling text repairs into current downstream artifacts.

This is deliberately a one-time, non-LLM migration.  It changes only the
currently configured classification and enumeration directories; it never
touches source JSON, chunked inputs, ground truth, archives, or extraction
outputs.  Classification labels are retained unchanged.

Run without ``--apply`` first.  A decision is changed only when its observed
match count in the scoped classified chunks equals the CSV ``expected_matches``
value.  The apply mode creates recoverable copies before overwriting files.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

from workflow_config import CHUNKED_DATA_DIR, OUTPUT_ROOT, get_study_paths


LEDGER_PATH = (
    Path(__file__).resolve().parents[1]
    / "Preprocessing"
    / "docling_text_repair_decisions.csv"
)
PLAN_PATH = OUTPUT_ROOT / "docling_text_repair_reextraction_plan.csv"
REPORT_PATH = OUTPUT_ROOT / "docling_text_repair_downstream_report.json"


@dataclass(frozen=True)
class RepairDecision:
    study: str
    source_file: str
    object_ref: str
    match_pattern: str
    replacement: str
    expected_matches: int
    row_number: int

    @property
    def table_index(self) -> int | None:
        matched = re.fullmatch(r"#/tables/(\d+)", self.object_ref)
        return int(matched.group(1)) if matched else None


def _load_approved_decisions(path: Path) -> list[RepairDecision]:
    decisions: list[RepairDecision] = []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        for row_number, row in enumerate(reader, start=2):
            if str(row.get("approval_status") or "").strip().casefold() != "approved":
                continue
            required = ("study", "source_file", "object_ref", "match_pattern", "replacement")
            missing = [name for name in required if not str(row.get(name) or "").strip()]
            if missing:
                raise ValueError(f"{path}:{row_number} is missing {missing}")
            try:
                expected_matches = int(str(row.get("expected_matches") or "").strip())
            except ValueError as exc:
                raise ValueError(f"{path}:{row_number} has invalid expected_matches") from exc
            if expected_matches < 1:
                raise ValueError(f"{path}:{row_number} expected_matches must be positive")
            decisions.append(
                RepairDecision(
                    study=str(row["study"]).strip(),
                    source_file=str(row["source_file"]).strip(),
                    object_ref=str(row["object_ref"]).strip(),
                    match_pattern=str(row["match_pattern"]),
                    replacement=str(row["replacement"]),
                    expected_matches=expected_matches,
                    row_number=row_number,
                )
            )
    return decisions


def _labels(entry: dict[str, Any]) -> set[str]:
    labels = [str(entry.get("predicted_label") or "")]
    annotation = entry.get("annotation")
    if isinstance(annotation, list):
        labels.extend(str(value) for value in annotation)
    return {
        token.strip().casefold()
        for value in labels
        for token in re.split(r"[,;]", value)
        if token.strip()
    }


def _source_id(entry: dict[str, Any]) -> str:
    return f"{entry.get('file_type')}_{entry.get('chunk_id')}"


def _matches_decision(entry: dict[str, Any], decision: RepairDecision) -> bool:
    file_type = str(entry.get("file_type") or "").strip()
    if decision.table_index is not None:
        try:
            chunk_id = int(entry.get("chunk_id"))
        except (TypeError, ValueError):
            return False
        return file_type == "table" and chunk_id == decision.table_index
    return file_type == decision.source_file


def _read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def _backup(path: Path, backup_root: Path) -> None:
    relative = path.relative_to(OUTPUT_ROOT)
    destination = backup_root / relative
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(path, destination)


def _replace_embedded_chunk_text(value: Any, replacements: Iterable[tuple[str, str]]) -> tuple[Any, int]:
    if isinstance(value, str):
        updated = value
        count = 0
        for old, new in replacements:
            occurrences = updated.count(old)
            if occurrences:
                updated = updated.replace(old, new)
                count += occurrences
        return updated, count
    if isinstance(value, list):
        total = 0
        updated_list = []
        for item in value:
            updated_item, count = _replace_embedded_chunk_text(item, replacements)
            updated_list.append(updated_item)
            total += count
        return updated_list, total
    if isinstance(value, dict):
        total = 0
        updated_dict = {}
        for key, item in value.items():
            updated_item, count = _replace_embedded_chunk_text(item, replacements)
            updated_dict[key] = updated_item
            total += count
        return updated_dict, total
    return value, 0


def _enumeration_files(study: str, enumeration_dir: Path) -> list[Path]:
    paths = [enumeration_dir / f"{study}_enumerated.json"]
    artifact_dir = enumeration_dir / "_chain_artifacts" / study
    if artifact_dir.is_dir():
        paths.extend(sorted(artifact_dir.rglob("*.json")))
    return [path for path in paths if path.is_file()]


def _task_names(labels: set[str]) -> list[str]:
    if "irrelevant" in labels:
        return []
    tasks: list[str] = []
    if "adsorbent" in labels:
        tasks.append("adsorbent")
    # Performance extraction uses performance-labeled sources plus
    # experiment-labeled context, so either label makes the task stale.
    if {"performance", "experiment"} & labels:
        tasks.append("performance")
    return tasks


def _entry_key(entry: dict[str, Any]) -> tuple[str, str]:
    return (str(entry.get("file_type") or ""), str(entry.get("chunk_id")))


def _is_repair_related_change(
    old_text: str,
    new_text: str,
    entry: dict[str, Any],
    decisions: list[RepairDecision],
) -> bool:
    """Require evidence that a chunk difference is linked to an approved repair."""
    for decision in decisions:
        if not _matches_decision(entry, decision):
            continue
        if re.search(decision.match_pattern, old_text) or decision.replacement in new_text:
            return True
    return False


def _synchronize_classified_entries(
    entries: list[dict[str, Any]],
    chunked_entries: list[dict[str, Any]],
    decisions: list[RepairDecision],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Copy repaired chunk text into classification while preserving labels.

    The regenerated chunk file is authoritative: it was built from the repaired
    Docling source and retains chunk identifiers.  This avoids applying a
    document-scoped regex to unrelated occurrences in other chunks.
    """
    chunked_by_key = {
        _entry_key(entry): entry
        for entry in chunked_entries
        if isinstance(entry, dict)
    }
    changed_chunks: list[dict[str, Any]] = []
    audit: list[dict[str, Any]] = []

    for entry in entries:
        key = _entry_key(entry)
        source_entry = chunked_by_key.get(key)
        if source_entry is None:
            continue
        old_text = str(entry.get("enriched_text") or "")
        new_text = str(source_entry.get("enriched_text") or "")
        if old_text == new_text:
            continue
        if not _is_repair_related_change(old_text, new_text, entry, decisions):
            audit.append(
                {
                    "status": "unexpected_chunk_difference",
                    "source_id": _source_id(entry),
                    "message": "Skipped because no approved repair matched the old or new text.",
                }
            )
            continue
        entry["enriched_text"] = new_text
        changed_chunks.append(
            {
                "source_id": _source_id(entry),
                "file_type": str(entry.get("file_type") or ""),
                "chunk_id": entry.get("chunk_id"),
                "labels": sorted(_labels(entry)),
                "old_text": old_text,
                "new_text": new_text,
            }
        )

    for decision in decisions:
        related = [chunk for chunk in changed_chunks if _matches_decision(chunk, decision)]
        audit.append(
            {
                "row_number": decision.row_number,
                "status": "synchronized" if related else "no_downstream_change",
                "source_ids": [chunk["source_id"] for chunk in related],
            }
        )
    return entries, changed_chunks, audit


def _write_plan(rows: list[dict[str, Any]], path: Path) -> None:
    fields = [
        "study",
        "group",
        "reextract_tasks",
        "affected_source_ids",
        "affected_labels",
        "affected_chunk_count",
        "reason",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--apply", action="store_true", help="write repaired artifacts and reports")
    parser.add_argument(
        "--backup-root",
        type=Path,
        default=None,
        help="directory for pre-change copies (apply mode only)",
    )
    parser.add_argument(
        "--audit-output",
        type=Path,
        default=None,
        help="optional JSON audit path; useful when reviewing a dry run",
    )
    parser.add_argument(
        "--plan-output",
        type=Path,
        default=None,
        help="optional CSV re-extraction-plan path; useful when reviewing a dry run",
    )
    args = parser.parse_args()

    decisions_by_study: dict[str, list[RepairDecision]] = defaultdict(list)
    for decision in _load_approved_decisions(LEDGER_PATH):
        decisions_by_study[decision.study].append(decision)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_root = args.backup_root or OUTPUT_ROOT / "_docling_text_repair_backups" / timestamp
    report: dict[str, Any] = {
        "mode": "apply" if args.apply else "dry_run",
        "ledger_path": str(LEDGER_PATH),
        "studies": {},
        "summary": {},
    }
    plan_rows: list[dict[str, Any]] = []
    summary = defaultdict(int)

    for study, decisions in sorted(decisions_by_study.items()):
        try:
            paths = get_study_paths(study)
        except ValueError as exc:
            report["studies"][study] = {"status": "unregistered", "message": str(exc)}
            summary["unregistered_studies"] += 1
            continue

        classified_path = paths.classified_file
        study_report: dict[str, Any] = {
            "group": paths.group,
            "classified_path": str(classified_path),
            "decision_count": len(decisions),
        }
        if not classified_path.is_file():
            study_report["status"] = "missing_classification"
            report["studies"][study] = study_report
            summary["missing_classification"] += 1
            continue

        chunked_path = CHUNKED_DATA_DIR / f"{study}_chunks.json"
        study_report["chunked_path"] = str(chunked_path)
        if not chunked_path.is_file():
            study_report["status"] = "missing_repaired_chunked_input"
            report["studies"][study] = study_report
            summary["missing_repaired_chunked_input"] += 1
            continue

        classified = _read_json(classified_path)
        if not isinstance(classified, list) or not all(isinstance(item, dict) for item in classified):
            study_report["status"] = "invalid_classification_json"
            report["studies"][study] = study_report
            summary["invalid_classification_json"] += 1
            continue

        chunked = _read_json(chunked_path)
        if not isinstance(chunked, list) or not all(isinstance(item, dict) for item in chunked):
            study_report["status"] = "invalid_chunked_json"
            report["studies"][study] = study_report
            summary["invalid_chunked_json"] += 1
            continue

        updated, changed_chunks, decision_audit = _synchronize_classified_entries(
            classified,
            chunked,
            decisions,
        )
        study_report["decision_audit"] = decision_audit
        study_report["changed_chunks"] = [
            {key: value for key, value in chunk.items() if key not in {"old_text", "new_text"}}
            for chunk in changed_chunks
        ]
        summary["approved_decisions"] += len(decisions)
        for result in decision_audit:
            summary[f"decisions_{result['status']}"] += 1

        if not changed_chunks:
            study_report["status"] = "no_classified_changes"
            report["studies"][study] = study_report
            continue

        replacements = [(chunk["old_text"], chunk["new_text"]) for chunk in changed_chunks]
        enum_file_counts: dict[str, int] = {}
        enum_updated: dict[Path, Any] = {}
        for enum_path in _enumeration_files(study, paths.enumeration_dir):
            original = _read_json(enum_path)
            repaired, replacement_count = _replace_embedded_chunk_text(original, replacements)
            if replacement_count:
                enum_updated[enum_path] = repaired
                enum_file_counts[str(enum_path)] = replacement_count

        by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for chunk in changed_chunks:
            for task in _task_names(set(chunk["labels"])):
                by_task[task].append(chunk)
        if by_task:
            plan_rows.append(
                {
                    "study": study,
                    "group": paths.group,
                    "reextract_tasks": ";".join(sorted(by_task)),
                    "affected_source_ids": ";".join(
                        sorted({chunk["source_id"] for chunks in by_task.values() for chunk in chunks})
                    ),
                    "affected_labels": ";".join(
                        sorted({label for chunks in by_task.values() for chunk in chunks for label in chunk["labels"]})
                    ),
                    "affected_chunk_count": len(
                        {chunk["source_id"] for chunks in by_task.values() for chunk in chunks}
                    ),
                    "reason": "Approved Docling unit repair changed text used by the listed extraction task(s).",
                }
            )

        study_report["status"] = "ready_to_apply"
        study_report["enumeration_replacement_counts"] = enum_file_counts
        study_report["reextraction_tasks"] = sorted(by_task)
        report["studies"][study] = study_report
        summary["studies_with_classified_changes"] += 1
        summary["classified_chunks_changed"] += len(changed_chunks)
        summary["enumeration_files_changed"] += len(enum_updated)
        summary["enumeration_embedded_chunk_replacements"] += sum(enum_file_counts.values())

        if args.apply:
            _backup(classified_path, backup_root)
            _write_json(classified_path, updated)
            for enum_path, enum_value in enum_updated.items():
                _backup(enum_path, backup_root)
                _write_json(enum_path, enum_value)

    report["summary"] = dict(sorted(summary.items()))
    plan_output = args.plan_output or (PLAN_PATH if args.apply else None)
    audit_output = args.audit_output or (REPORT_PATH if args.apply else None)
    if plan_output:
        _write_plan(plan_rows, plan_output)
        print(f"Wrote re-extraction plan: {plan_output}")
    if audit_output:
        _write_json(audit_output, report)
        print(f"Wrote audit report: {audit_output}")
    if args.apply:
        print(f"Backups: {backup_root}")
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))
    for row in plan_rows:
        print(
            f"REEXTRACT {row['study']}: {row['reextract_tasks']} "
            f"({row['affected_source_ids']})"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
