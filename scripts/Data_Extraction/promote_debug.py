"""Safely promote focused debug results into the normal workflow outputs.

The command is a dry run unless ``--apply`` is supplied.  It uses
``workflow_config.ACTIVE_STUDIES`` and the sibling ``*_debug`` directory
created by ``debug.py``.  Replacement matching is based on the underlying
source chunks, not synthetic batch names.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime
import json
import os
from pathlib import Path
import re
import shutil
from typing import Any

try:
    from .workflow_config import StudyPaths, iter_active_study_paths
    from .helpers.debug_filter import debug_output_dir, source_id_for_entry
except ImportError:  # Supports ``python promote_debug.py`` from this folder.
    from workflow_config import StudyPaths, iter_active_study_paths
    from helpers.debug_filter import debug_output_dir, source_id_for_entry


@dataclass(frozen=True)
class EntryIdentity:
    study_id: str
    task: str
    source_ids: frozenset[str]

    def describe(self) -> str:
        sources = ", ".join(sorted(self.source_ids))
        return f"{self.study_id} | {self.task} | {sources}"


@dataclass
class EntryChange:
    action: str
    identity: EntryIdentity
    old_entry: dict[str, Any] | None
    new_entry: dict[str, Any]


@dataclass
class FileUpdate:
    root: Path
    path: Path
    payload: Any
    changes: list[EntryChange]
    description: str


class UnsafeReplacementError(RuntimeError):
    """Raised when a focused replacement cannot be matched without data loss."""


def _load_json(path: Path) -> Any:
    if not path.is_file():
        raise FileNotFoundError(f"Required file not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _load_list(path: Path) -> list[dict[str, Any]]:
    payload = _load_json(path)
    if not isinstance(payload, list) or not all(isinstance(item, dict) for item in payload):
        raise ValueError(f"Expected a JSON list of objects: {path}")
    return payload


def _load_object(path: Path) -> dict[str, Any]:
    payload = _load_json(path)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return payload


def _entry_task(entry: dict[str, Any]) -> str:
    extracted_data = entry.get("extracted_data")
    if isinstance(extracted_data, dict) and extracted_data:
        return "+".join(sorted(str(key).strip().casefold() for key in extracted_data))

    enumeration_tasks = {
        str(item.get("task") or "").strip().casefold()
        for item in entry.get("enumerations") or []
        if isinstance(item, dict) and str(item.get("task") or "").strip()
    }
    if enumeration_tasks:
        return "+".join(sorted(enumeration_tasks))
    return "classification"


def _entry_source_ids(entry: dict[str, Any]) -> frozenset[str]:
    metadata = entry.get("enriched_text_metadata")
    if isinstance(metadata, list):
        candidates = [item for item in metadata if isinstance(item, dict)]
        entry_label = str(entry.get("predicted_label") or "").strip().casefold()
        same_task = [
            item
            for item in candidates
            if not str(item.get("predicted_label") or "").strip()
            or str(item.get("predicted_label") or "").strip().casefold() == entry_label
        ]
        if entry_label and same_task:
            candidates = same_task
        source_ids = {
            source_id_for_entry(item)
            for item in candidates
            if source_id_for_entry(item)
        }
        if source_ids:
            return frozenset(source_ids)

    direct = source_id_for_entry(entry)
    return frozenset({direct}) if direct else frozenset()


def _identity(entry: dict[str, Any], fallback_study_id: str) -> EntryIdentity:
    study_id = str(entry.get("study_folder") or fallback_study_id).strip().casefold()
    source_ids = _entry_source_ids(entry)
    if not source_ids:
        raise UnsafeReplacementError(
            f"Cannot determine source chunks for a debug entry in {fallback_study_id}."
        )
    return EntryIdentity(study_id=study_id, task=_entry_task(entry), source_ids=source_ids)


def _merge_entries(
    production: list[dict[str, Any]],
    debug_entries: list[dict[str, Any]],
    *,
    study_id: str,
) -> tuple[list[dict[str, Any]], list[EntryChange]]:
    """Replace exact source envelopes and reject ambiguous partial overlaps."""

    merged = list(production)
    identities = [_identity(entry, study_id) for entry in merged]
    debug_identities: set[EntryIdentity] = set()
    changes: list[EntryChange] = []

    for debug_entry in debug_entries:
        debug_identity = _identity(debug_entry, study_id)
        if debug_identity in debug_identities:
            raise UnsafeReplacementError(
                f"Duplicate debug source envelope: {debug_identity.describe()}"
            )
        debug_identities.add(debug_identity)

        exact = [index for index, identity in enumerate(identities) if identity == debug_identity]
        if len(exact) > 1:
            raise UnsafeReplacementError(
                f"Multiple production entries match {debug_identity.describe()}; refusing ambiguity."
            )
        if len(exact) == 1:
            index = exact[0]
            old_entry = merged[index]
            promoted_entry = dict(debug_entry)
            # Debug reruns may use a different synthetic batch name because
            # they process only a subset.  Preserve the production envelope
            # identity while replacing its newly generated contents.
            for key in ("study_folder", "file_type", "chunk_id"):
                if key in old_entry:
                    promoted_entry[key] = old_entry[key]
            if old_entry != promoted_entry:
                merged[index] = promoted_entry
                changes.append(
                    EntryChange("replace", debug_identity, old_entry, promoted_entry)
                )
            continue

        overlaps = [
            identity
            for identity in identities
            if identity.study_id == debug_identity.study_id
            and identity.task == debug_identity.task
            and identity.source_ids & debug_identity.source_ids
        ]
        if overlaps:
            overlap_text = "; ".join(identity.describe() for identity in overlaps)
            raise UnsafeReplacementError(
                "The debug result overlaps a larger or differently grouped production entry. "
                "Replacing it could discard unselected chunks, so promotion was refused. "
                f"Debug: {debug_identity.describe()} | Production: {overlap_text}"
            )

        merged.append(debug_entry)
        identities.append(debug_identity)
        changes.append(EntryChange("add", debug_identity, None, debug_entry))

    return merged, changes


def _record_count_enumeration(outputs: list[dict[str, Any]]) -> int:
    total = 0
    for output in outputs:
        for enumeration in output.get("enumerations") or []:
            if not isinstance(enumeration, dict) or enumeration.get("task") != "performance":
                continue
            for record in enumeration.get("records") or []:
                if not isinstance(record, dict):
                    continue
                value = str(record.get("Performance_id") or "").strip()
                normalized = "|".join(part.strip() for part in value.split("|"))
                parts = normalized.split("|")
                head = re.sub(r"[^a-z0-9]+", " ", parts[0].casefold()).strip()
                no_performance = head in {"no performance id", "no performance ids"} and all(
                    re.sub(r"[^a-z0-9]+", "", part.casefold())
                    in {"", "na", "none", "null"}
                    for part in parts[1:]
                )
                if not no_performance and normalized.casefold() != "unit":
                    total += 1
    return total


def _record_count_extraction(outputs: list[dict[str, Any]], task: str) -> int:
    total = 0
    for output in outputs:
        block = (output.get("extracted_data") or {}).get(task) or {}
        records = block.get("records") if isinstance(block, dict) else block
        if isinstance(records, list):
            total += len(records)
    return total


def _collect_selected_metadata(outputs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    seen: set[str] = set()
    for output in outputs:
        for item in output.get("enriched_text_metadata") or []:
            if not isinstance(item, dict):
                continue
            source_id = source_id_for_entry(item)
            if source_id and source_id not in seen:
                seen.add(source_id)
                selected.append(item)
    return selected


def _promotion_note(stage: str, changes: list[EntryChange]) -> dict[str, Any]:
    return {
        "promoted_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "stage": stage,
        "replaced": sum(change.action == "replace" for change in changes),
        "added": sum(change.action == "add" for change in changes),
        "source_envelopes": [change.identity.describe() for change in changes],
    }


def _append_promotion_note(payload: dict[str, Any], note: dict[str, Any]) -> None:
    history = payload.get("focused_replacements")
    if not isinstance(history, list):
        history = []
    payload["focused_replacements"] = [*history, note]


def _make_update(
    *,
    root: Path,
    path: Path,
    old_payload: Any,
    new_payload: Any,
    changes: list[EntryChange],
    description: str,
) -> FileUpdate | None:
    if old_payload == new_payload:
        return None
    return FileUpdate(root, path, new_payload, changes, description)


def _plan_classification(paths: StudyPaths, debug_suffix: str) -> list[FileUpdate]:
    production_path = paths.classified_file
    debug_root = debug_output_dir(paths.classification_dir, debug_suffix)
    debug_path = debug_root / production_path.name
    production = _load_list(production_path)
    debug_entries = _load_list(debug_path)
    merged, changes = _merge_entries(production, debug_entries, study_id=paths.study_id)

    updates: list[FileUpdate] = []
    update = _make_update(
        root=paths.classification_dir,
        path=production_path,
        old_payload=production,
        new_payload=merged,
        changes=changes,
        description=f"{paths.study_id} classification",
    )
    if update:
        updates.append(update)

    combined_path = paths.classification_dir / "all_studies_classified.json"
    if combined_path.is_file() and debug_entries:
        combined = _load_list(combined_path)
        merged_combined, combined_changes = _merge_entries(
            combined,
            debug_entries,
            study_id=paths.study_id,
        )
        update = _make_update(
            root=paths.classification_dir,
            path=combined_path,
            old_payload=combined,
            new_payload=merged_combined,
            changes=combined_changes,
            description=f"{paths.study_id} in combined classification",
        )
        if update:
            updates.append(update)
    return updates


def _plan_enumeration(paths: StudyPaths, debug_suffix: str) -> list[FileUpdate]:
    production_root = paths.enumeration_dir
    debug_root = debug_output_dir(production_root, debug_suffix)
    updates: list[FileUpdate] = []

    artifact_relative = (
        Path("_chain_artifacts")
        / paths.study_id
        / f"{paths.study_id}_performance.json"
    )
    production_artifact_path = production_root / artifact_relative
    debug_artifact_path = debug_root / artifact_relative
    production_artifact = _load_object(production_artifact_path)
    debug_artifact = _load_object(debug_artifact_path)
    artifact_outputs = production_artifact.get("outputs") or []
    debug_outputs = debug_artifact.get("outputs") or []
    if not isinstance(artifact_outputs, list) or not isinstance(debug_outputs, list):
        raise ValueError("Enumeration performance artifact outputs must be lists")
    merged_outputs, artifact_changes = _merge_entries(
        artifact_outputs,
        debug_outputs,
        study_id=paths.study_id,
    )
    merged_artifact = dict(production_artifact)
    merged_artifact["outputs"] = merged_outputs
    merged_artifact["output_count"] = len(merged_outputs)
    merged_artifact["record_count"] = _record_count_enumeration(merged_outputs)
    merged_artifact["selected_chunk_metadata"] = _collect_selected_metadata(merged_outputs)
    if artifact_changes:
        _append_promotion_note(
            merged_artifact,
            _promotion_note("enumeration", artifact_changes),
        )
    update = _make_update(
        root=production_root,
        path=production_artifact_path,
        old_payload=production_artifact,
        new_payload=merged_artifact,
        changes=artifact_changes,
        description=f"{paths.study_id} performance enumeration artifact",
    )
    if update:
        updates.append(update)
    return updates


def _extraction_tasks(entries: list[dict[str, Any]]) -> set[str]:
    tasks: set[str] = set()
    for entry in entries:
        data = entry.get("extracted_data")
        if isinstance(data, dict):
            tasks.update(str(task).strip().casefold() for task in data if str(task).strip())
    return tasks


def _plan_extraction(paths: StudyPaths, debug_suffix: str) -> list[FileUpdate]:
    production_root = paths.extraction_dir
    debug_root = debug_output_dir(production_root, debug_suffix)
    updates: list[FileUpdate] = []

    chain_names = {
        "adsorbent": "adsorbent_study",
        "experiment": "experiment_study",
        "performance": "performance",
    }
    for task, chain_name in chain_names.items():
        artifact_relative = (
            Path("_chain_artifacts")
            / paths.study_id
            / f"{paths.study_id}_{chain_name}.json"
        )
        production_artifact_path = production_root / artifact_relative
        debug_artifact_path = debug_root / artifact_relative
        if not debug_artifact_path.exists():
            continue
        production_artifact = _load_object(production_artifact_path)
        debug_artifact = _load_object(debug_artifact_path)
        artifact_outputs = production_artifact.get("outputs") or []
        debug_outputs = debug_artifact.get("outputs") or []
        if not isinstance(artifact_outputs, list) or not isinstance(debug_outputs, list):
            raise ValueError(f"Extraction {task} artifact outputs must be lists")
        merged_outputs, artifact_changes = _merge_entries(
            artifact_outputs,
            debug_outputs,
            study_id=paths.study_id,
        )
        merged_artifact = dict(production_artifact)
        merged_artifact["outputs"] = merged_outputs
        merged_artifact["output_count"] = len(merged_outputs)
        merged_artifact["record_count"] = _record_count_extraction(merged_outputs, task)
        if artifact_changes:
            _append_promotion_note(
                merged_artifact,
                _promotion_note("extraction", artifact_changes),
            )
        update = _make_update(
            root=production_root,
            path=production_artifact_path,
            old_payload=production_artifact,
            new_payload=merged_artifact,
            changes=artifact_changes,
            description=f"{paths.study_id} {task} extraction artifact",
        )
        if update:
            updates.append(update)
    return updates


def _atomic_write_json(path: Path, payload: Any) -> None:
    temporary = path.with_name(f".{path.name}.promote-{os.getpid()}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False)
            handle.write("\n")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _apply_updates(updates: list[FileUpdate], stage: str) -> list[Path]:
    timestamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")
    backup_directories: list[Path] = []
    grouped: dict[Path, list[FileUpdate]] = {}
    for update in updates:
        grouped.setdefault(update.root, []).append(update)

    for root, root_updates in grouped.items():
        root_resolved = root.resolve()
        backup_root = root / "_focused_replacement_backups" / timestamp
        backup_directories.append(backup_root)
        audit_updates: list[dict[str, Any]] = []

        for update in root_updates:
            relative = update.path.resolve().relative_to(root_resolved)
            backup_path = backup_root / relative
            backup_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(update.path, backup_path)
            _atomic_write_json(update.path, update.payload)
            audit_updates.append(
                {
                    "file": str(update.path),
                    "backup": str(backup_path),
                    "description": update.description,
                    "changes": [
                        {
                            "action": change.action,
                            "identity": change.identity.describe(),
                        }
                        for change in update.changes
                    ],
                }
            )

        audit = {
            "applied_at": datetime.now().astimezone().isoformat(timespec="seconds"),
            "stage": stage,
            "updates": audit_updates,
        }
        _atomic_write_json(backup_root / "promotion_audit.json", audit)
    return backup_directories


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    stage = parser.add_mutually_exclusive_group(required=True)
    stage.add_argument("--classify", dest="stage", action="store_const", const="classify")
    stage.add_argument("--enumerate", dest="stage", action="store_const", const="enumerate")
    stage.add_argument("--extract", dest="stage", action="store_const", const="extract")
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Apply the displayed replacements. Without this flag, nothing is written.",
    )
    parser.add_argument(
        "--debug-suffix",
        default="_debug",
        help="Suffix of the debug output directory created by debug.py (default: _debug).",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if not args.debug_suffix:
        raise SystemExit("The debug suffix cannot be empty.")

    planners = {
        "classify": _plan_classification,
        "enumerate": _plan_enumeration,
        "extract": _plan_extraction,
    }
    updates: list[FileUpdate] = []
    try:
        for paths in iter_active_study_paths():
            updates.extend(planners[args.stage](paths, args.debug_suffix))
    except (FileNotFoundError, ValueError, UnsafeReplacementError) as exc:
        raise SystemExit(f"Promotion refused: {exc}") from exc

    if not updates:
        print("No differences were found between the focused debug output and production.")
        return

    print(f"Focused {args.stage} promotion plan:")
    for update in updates:
        replaced = sum(change.action == "replace" for change in update.changes)
        added = sum(change.action == "add" for change in update.changes)
        print(f"  {update.description}: replace={replaced}, add={added}")
        print(f"    {update.path}")
        for change in update.changes:
            print(f"    - {change.action}: {change.identity.describe()}")

    if not args.apply:
        print("\nDRY RUN: no files were changed.")
        print(f"Run again with --{args.stage} --apply after reviewing this plan.")
        return

    backup_directories = _apply_updates(updates, args.stage)
    print(f"\nApplied {len(updates)} file update(s).")
    for directory in backup_directories:
        print(f"Backup and audit: {directory}")
    if args.stage == "classify":
        print("Reminder: existing enumeration and extraction outputs may now be stale.")
    elif args.stage == "enumerate":
        print("Reminder: existing extraction outputs may now be stale.")


if __name__ == "__main__":
    main()
