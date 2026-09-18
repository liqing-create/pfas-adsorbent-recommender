"""Refresh downstream artifacts after a small deterministic chunk edit.

Run this after ``docling_json_preprocess.py`` and ``chunk_json_input.py`` when
the new chunks differ only in a way that cannot change their LLM
classification or enumeration decisions (for example, a missing table unit).
It reuses the existing decisions, but replaces the source text and chunk
metadata that extraction consumes.  No LLM client or chain is created.

The study selection comes from this directory's ``workflow_config.py``. This
lets a preprocessing rerun and this refresh operate on exactly the same
``ACTIVE_STUDIES`` selection.

The default refreshes the classification and enumeration artifacts directly,
without creating backups. Use ``--dry-run`` when you only want to validate the
source lineage before writing.
"""

from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import os
import sys
import tempfile
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_EXTRACTION_DIR = SCRIPT_DIR.parent / "Data_Extraction"
DATA_EXTRACTION_CONFIG_PATH = DATA_EXTRACTION_DIR / "workflow_config.py"
CHUNK_MATCHING_PATH = DATA_EXTRACTION_DIR / "classification_chunk_matching.py"
PREPROCESSING_CONFIG_PATH = SCRIPT_DIR / "workflow_config.py"
ARTIFACT_TASKS = ("adsorbent", "water_type", "performance")
SOURCE_FIELDS = {
    "study_folder",
    "file_type",
    "chunk_id",
    "section_label",
    "enriched_text",
}
def _load_module(module_name: str, path: Path) -> Any:
    """Load a sibling module without colliding with preprocessing config names."""
    if not path.is_file():
        raise RuntimeError(f"Required module is missing: {path}")
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(module_name, None)
        raise
    return module


_data_extraction_config = _load_module(
    "_chunk_refresh_data_extraction_workflow_config",
    DATA_EXTRACTION_CONFIG_PATH,
)
_chunk_matching = _load_module(
    "_chunk_refresh_classification_chunk_matching",
    CHUNK_MATCHING_PATH,
)
StudyPaths = _data_extraction_config.StudyPaths
get_study_paths = _data_extraction_config.get_study_paths
match_chunks = _chunk_matching.match_chunks


class RefreshError(RuntimeError):
    """Raised before writing when artifact lineage cannot be proven safe."""


@dataclass(frozen=True)
class ChunkMapping:
    """One previous classified row paired with its replacement chunk."""

    old_index: int
    fresh_index: int
    method: str
    similarity: float


@dataclass
class StudyRefreshPlan:
    """All safe, in-memory changes for one selected study."""

    paths: StudyPaths
    fresh_chunks: list[dict[str, Any]]
    old_classified: list[dict[str, Any]]
    refreshed_classified: list[dict[str, Any]]
    mappings: list[ChunkMapping]
    updates: dict[Path, Any]
    warnings: list[str] = field(default_factory=list)


def _load_json(path: Path) -> Any:
    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except json.JSONDecodeError as exc:
        raise RefreshError(f"Invalid JSON in {path}: {exc}") from exc


def _require_list_payload(path: Path, label: str) -> list[dict[str, Any]]:
    if not path.is_file():
        raise RefreshError(f"Missing {label}: {path}")
    payload = _load_json(path)
    if not isinstance(payload, list) or not all(isinstance(item, dict) for item in payload):
        raise RefreshError(f"{label} must be a JSON list of objects: {path}")
    return payload


def _source_key(record: dict[str, Any]) -> tuple[str, str] | None:
    """Return the stable pipeline identifier when the record provides one."""
    file_type = str(record.get("file_type") or "").strip().casefold()
    chunk_id = record.get("chunk_id")
    if not file_type or chunk_id is None or str(chunk_id).strip() == "":
        return None
    return file_type, str(chunk_id).strip()


def _index_unique_source_keys(
    records: Sequence[dict[str, Any]],
    *,
    label: str,
) -> dict[tuple[str, str], int]:
    indexed: dict[tuple[str, str], int] = {}
    duplicate_keys: set[tuple[str, str]] = set()
    for index, record in enumerate(records):
        key = _source_key(record)
        if key is None:
            continue
        if key in indexed:
            duplicate_keys.add(key)
        else:
            indexed[key] = index
    if duplicate_keys:
        shown = ", ".join(f"{file_type}/{chunk_id}" for file_type, chunk_id in sorted(duplicate_keys))
        raise RefreshError(f"Duplicate file_type/chunk_id keys in {label}: {shown}")
    return indexed


def _build_chunk_mappings(
    old_records: Sequence[dict[str, Any]],
    fresh_records: Sequence[dict[str, Any]],
) -> list[ChunkMapping]:
    """Pair all fresh chunks with their old classified decision.

    Identical ``file_type``/``chunk_id`` pairs take priority because chunk IDs
    are the deterministic lineage emitted by preprocessing.  Any remaining
    rows are matched by the existing conservative content matcher; a fresh row
    that cannot be paired aborts the run rather than receiving an unsafe LLM
    decision.
    """
    old_by_key = _index_unique_source_keys(old_records, label="existing classification")
    fresh_by_key = _index_unique_source_keys(fresh_records, label="fresh chunks")
    mappings: list[ChunkMapping] = []
    used_old: set[int] = set()
    used_fresh: set[int] = set()

    for key, fresh_index in fresh_by_key.items():
        old_index = old_by_key.get(key)
        if old_index is None:
            continue
        mappings.append(ChunkMapping(old_index, fresh_index, "source_key", 1.0))
        used_old.add(old_index)
        used_fresh.add(fresh_index)

    remaining_old_indices = [index for index in range(len(old_records)) if index not in used_old]
    remaining_fresh_indices = [index for index in range(len(fresh_records)) if index not in used_fresh]
    if remaining_old_indices and remaining_fresh_indices:
        content_matches = match_chunks(
            [old_records[index] for index in remaining_old_indices],
            [fresh_records[index] for index in remaining_fresh_indices],
        )
        for match in content_matches:
            old_index = remaining_old_indices[match.source_index]
            fresh_index = remaining_fresh_indices[match.target_index]
            mappings.append(
                ChunkMapping(old_index, fresh_index, match.method, match.similarity)
            )
            used_old.add(old_index)
            used_fresh.add(fresh_index)

    unmatched_fresh = [index for index in range(len(fresh_records)) if index not in used_fresh]
    if unmatched_fresh:
        preview = ", ".join(
            _describe_record(fresh_records[index]) for index in unmatched_fresh[:8]
        )
        raise RefreshError(
            "Cannot safely reuse classifications for fresh chunk(s): "
            f"{preview}. Re-run classify.py for this study instead."
        )

    return sorted(mappings, key=lambda item: item.fresh_index)


def _describe_record(record: dict[str, Any]) -> str:
    key = _source_key(record)
    if key is not None:
        return f"{key[0]}/{key[1]}"
    return f"index with chunk_id={record.get('chunk_id')!r}"


def _refreshed_classification_record(
    old_record: dict[str, Any],
    fresh_chunk: dict[str, Any],
    study_id: str,
) -> dict[str, Any]:
    """Keep prior decision fields and substitute current source fields."""
    refreshed = {
        key: copy.deepcopy(value)
        for key, value in old_record.items()
        if key not in SOURCE_FIELDS
    }
    refreshed.update(
        {
            "study_folder": study_id,
            "file_type": fresh_chunk.get("file_type"),
            "chunk_id": fresh_chunk.get("chunk_id"),
            "section_label": fresh_chunk.get("section_label"),
            "enriched_text": str(fresh_chunk.get("enriched_text") or "").strip(),
        }
    )
    return refreshed


def _build_classified_records(
    *,
    study_id: str,
    old_records: Sequence[dict[str, Any]],
    fresh_chunks: Sequence[dict[str, Any]],
    mappings: Sequence[ChunkMapping],
) -> list[dict[str, Any]]:
    by_fresh_index = {mapping.fresh_index: mapping for mapping in mappings}
    return [
        _refreshed_classification_record(
            old_records[by_fresh_index[index].old_index],
            fresh_chunk,
            study_id,
        )
        for index, fresh_chunk in enumerate(fresh_chunks)
    ]


def _metadata_for_entries(entries: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    metadata: list[dict[str, Any]] = []
    for number, entry in enumerate(entries, start=1):
        file_type = entry.get("file_type")
        chunk_id = entry.get("chunk_id")
        source_id = f"{file_type}_{chunk_id}" if chunk_id is not None else str(file_type or "unknown")
        metadata.append(
            {
                "prompt_chunk_number": number,
                "study_folder": entry.get("study_folder"),
                "file_type": file_type,
                "chunk_id": chunk_id,
                "source_id": source_id,
                "predicted_label": entry.get("predicted_label"),
            }
        )
    return metadata


def _format_study_chunks(entries: Iterable[dict[str, Any]]) -> str:
    """Match ``chains.enumerate_chain.build_study_chunks`` without importing an LLM chain."""
    formatted: list[str] = []
    for entry in entries:
        text = str(entry.get("enriched_text") or entry.get("text") or "").strip()
        if not text:
            continue
        file_type = str(entry.get("file_type") or "unknown").strip()
        chunk_id = entry.get("chunk_id")
        source_id = f"{file_type}_{chunk_id}" if chunk_id is not None else file_type
        labels = entry.get("predicted_label") or entry.get("annotation") or ""
        if isinstance(labels, list):
            labels = ", ".join(str(value) for value in labels)
        formatted.append(
            f"Chunk {len(formatted) + 1} (Data_Source={source_id}; labels={labels}):\n{text}"
        )
    return "\n\n".join(formatted).strip()


def _old_index_from_metadata(
    metadata: dict[str, Any],
    old_by_key: dict[tuple[str, str], int],
    *,
    artifact_path: Path,
) -> int:
    key = _source_key(metadata)
    if key is None or key not in old_by_key:
        raise RefreshError(
            f"{artifact_path} references a classified source that cannot be resolved: "
            f"{_describe_record(metadata)}"
        )
    return old_by_key[key]


def _fresh_entries_for_metadata(
    metadata: Any,
    *,
    old_by_key: dict[tuple[str, str], int],
    new_by_old_index: dict[int, dict[str, Any]],
    artifact_path: Path,
) -> list[dict[str, Any]]:
    if not isinstance(metadata, list):
        raise RefreshError(f"{artifact_path} has non-list enriched_text_metadata")
    refreshed: list[dict[str, Any]] = []
    for item in metadata:
        if not isinstance(item, dict):
            raise RefreshError(f"{artifact_path} has non-object source metadata")
        old_index = _old_index_from_metadata(item, old_by_key, artifact_path=artifact_path)
        try:
            refreshed.append(new_by_old_index[old_index])
        except KeyError as exc:
            raise RefreshError(
                f"{artifact_path} references a source removed from fresh chunks: "
                f"{_describe_record(item)}"
            ) from exc
    return refreshed


def _selected_entries_from_task_payload(
    payload: dict[str, Any],
    *,
    old_by_key: dict[tuple[str, str], int],
    new_by_old_index: dict[int, dict[str, Any]],
    artifact_path: Path,
) -> list[dict[str, Any]]:
    metadata = payload.get("selected_chunk_metadata") or []
    return _fresh_entries_for_metadata(
        metadata,
        old_by_key=old_by_key,
        new_by_old_index=new_by_old_index,
        artifact_path=artifact_path,
    )


def _refresh_enumeration_payload(
    payload: dict[str, Any],
    *,
    task: str,
    old_by_key: dict[tuple[str, str], int],
    new_by_old_index: dict[int, dict[str, Any]],
    artifact_path: Path,
    refreshed_at: str,
) -> dict[str, Any]:
    """Refresh source-bearing fields while retaining existing enumeration IDs."""
    refreshed = copy.deepcopy(payload)
    selected_entries = _selected_entries_from_task_payload(
        payload,
        old_by_key=old_by_key,
        new_by_old_index=new_by_old_index,
        artifact_path=artifact_path,
    )
    refreshed["selected_chunk_metadata"] = _metadata_for_entries(selected_entries)
    refreshed["refreshed_from_chunks_at_utc"] = refreshed_at
    refreshed["refresh_method"] = "reuse_existing_llm_classification_and_enumeration"

    if task != "performance":
        return refreshed

    outputs = refreshed.get("outputs")
    if not isinstance(outputs, list):
        raise RefreshError(f"{artifact_path} performance artifact has non-list outputs")
    for output_index, output in enumerate(outputs):
        if not isinstance(output, dict):
            raise RefreshError(f"{artifact_path} outputs[{output_index}] is not an object")
        fresh_entries = _fresh_entries_for_metadata(
            output.get("enriched_text_metadata"),
            old_by_key=old_by_key,
            new_by_old_index=new_by_old_index,
            artifact_path=artifact_path,
        )
        output["enriched_text_metadata"] = _metadata_for_entries(fresh_entries)
        output["enriched_text"] = _format_study_chunks(fresh_entries)
    return refreshed


def _refresh_combined_classification(
    combined_path: Path,
    *,
    study_id: str,
    refreshed_classified: list[dict[str, Any]],
) -> list[dict[str, Any]] | None:
    if not combined_path.is_file():
        return None
    payload = _require_list_payload(combined_path, "combined classification artifact")
    remaining = [
        item for item in payload
        if str(item.get("study_folder") or "").strip() != study_id
    ]
    return [*remaining, *copy.deepcopy(refreshed_classified)]


def _refresh_manifest(
    manifest_path: Path, *, refreshed_at: str
) -> dict[str, Any] | None:
    if not manifest_path.is_file():
        return None
    payload = _load_json(manifest_path)
    if not isinstance(payload, dict):
        raise RefreshError(f"Enumeration manifest must be a JSON object: {manifest_path}")
    refreshed = copy.deepcopy(payload)
    refreshed["refreshed_from_chunks_at_utc"] = refreshed_at
    refreshed["refresh_method"] = "reuse_existing_llm_classification_and_enumeration"
    return refreshed


def build_study_refresh_plan(
    paths: StudyPaths,
    *,
    refreshed_at: str,
    refresh_enumeration: bool = True,
) -> StudyRefreshPlan:
    """Validate and construct all changes for one configured study without writing."""
    fresh_chunks = _require_list_payload(paths.chunked_input_file, "fresh chunk input")
    old_classified = _require_list_payload(paths.classified_file, "existing classification artifact")
    if not fresh_chunks:
        raise RefreshError(f"Fresh chunk input is empty: {paths.chunked_input_file}")
    if not old_classified:
        raise RefreshError(f"Existing classification artifact is empty: {paths.classified_file}")

    mappings = _build_chunk_mappings(old_classified, fresh_chunks)
    refreshed_classified = _build_classified_records(
        study_id=paths.study_id,
        old_records=old_classified,
        fresh_chunks=fresh_chunks,
        mappings=mappings,
    )
    old_by_key = _index_unique_source_keys(old_classified, label="existing classification")
    new_by_old_index = {
        mapping.old_index: refreshed_classified[mapping.fresh_index]
        for mapping in mappings
    }
    updates: dict[Path, Any] = {paths.classified_file: refreshed_classified}
    warnings: list[str] = []

    combined_path = paths.classification_dir / "all_studies_classified.json"
    combined = _refresh_combined_classification(
        combined_path,
        study_id=paths.study_id,
        refreshed_classified=refreshed_classified,
    )
    if combined is not None:
        updates[combined_path] = combined

    if refresh_enumeration:
        for task in ARTIFACT_TASKS:
            artifact_path = paths.enumeration_task_artifact(task)
            if not artifact_path.is_file():
                warnings.append(f"No {task} enumeration artifact found; left unchanged: {artifact_path}")
                continue
            payload = _load_json(artifact_path)
            if not isinstance(payload, dict):
                raise RefreshError(f"Enumeration artifact must be a JSON object: {artifact_path}")
            if str(payload.get("task") or "").strip() != task:
                raise RefreshError(
                    f"Enumeration artifact task mismatch ({payload.get('task')!r}, expected {task!r}): {artifact_path}"
                )
            updates[artifact_path] = _refresh_enumeration_payload(
                payload,
                task=task,
                old_by_key=old_by_key,
                new_by_old_index=new_by_old_index,
                artifact_path=artifact_path,
                refreshed_at=refreshed_at,
            )

        manifest_path = paths.enumeration_artifact_dir / "manifest.json"
        manifest = _refresh_manifest(manifest_path, refreshed_at=refreshed_at)
        if manifest is not None:
            updates[manifest_path] = manifest
    else:
        warnings.append("Enumeration artifacts were skipped by --classification-only.")

    return StudyRefreshPlan(
        paths=paths,
        fresh_chunks=fresh_chunks,
        old_classified=old_classified,
        refreshed_classified=refreshed_classified,
        mappings=mappings,
        updates=updates,
        warnings=warnings,
    )


def _load_preprocessing_active_study_ids() -> list[str]:
    if not PREPROCESSING_CONFIG_PATH.is_file():
        raise RefreshError(
            "Preprocessing workflow config is missing: "
            f"{PREPROCESSING_CONFIG_PATH}"
        )
    module_name = "_chunk_refresh_preprocessing_workflow_config"
    spec = importlib.util.spec_from_file_location(module_name, PREPROCESSING_CONFIG_PATH)
    if spec is None or spec.loader is None:
        raise RefreshError(f"Cannot import preprocessing workflow config: {PREPROCESSING_CONFIG_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
        selected = [paths.study_id for paths in module.iter_active_study_paths()]
    finally:
        sys.modules.pop(module_name, None)
    if not selected:
        raise RefreshError(f"No active studies selected in {PREPROCESSING_CONFIG_PATH}")
    return selected


def _write_artifact(path: Path, payload: Any) -> None:
    """Atomically replace one validated artifact without creating a backup."""
    if not path.is_file():
        raise RefreshError(f"Refusing to overwrite a missing artifact: {path}")
    _atomic_write_json(path, payload)


def _atomic_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    )
    temporary_path = Path(handle.name)
    try:
        with handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False)
            handle.write("\n")
        os.replace(temporary_path, path)
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise


def _mapping_counts(mappings: Sequence[ChunkMapping]) -> str:
    counts = Counter(mapping.method for mapping in mappings)
    return ", ".join(f"{method}={count}" for method, count in sorted(counts.items())) or "none"


def _print_plan(plan: StudyRefreshPlan) -> None:
    print(
        f"[{plan.paths.study_id}] {len(plan.fresh_chunks)} fresh chunks matched to "
        f"{len(plan.old_classified)} existing classifications ({_mapping_counts(plan.mappings)})."
    )
    for path in plan.updates:
        print(f"[{plan.paths.study_id}] refresh: {path}")
    for warning in plan.warnings:
        print(f"[{plan.paths.study_id}] WARNING: {warning}")


def run(*, dry_run: bool, classification_only: bool) -> int:
    study_ids = _load_preprocessing_active_study_ids()
    refreshed_at = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    plans: list[StudyRefreshPlan] = []
    failures: list[str] = []

    print(f"Active studies from preprocessing config: {', '.join(study_ids)}")
    for study_id in study_ids:
        try:
            plan = build_study_refresh_plan(
                get_study_paths(study_id),
                refreshed_at=refreshed_at,
                refresh_enumeration=not classification_only,
            )
            _print_plan(plan)
            plans.append(plan)
        except RefreshError as exc:
            failures.append(f"[{study_id}] {exc}")
            print(f"[ERROR] [{study_id}] {exc}")

    if failures:
        print("\nNo files were changed because at least one selected study failed validation.")
        return 1

    if dry_run:
        print("\nDry run succeeded. Re-run without --dry-run to refresh artifacts.")
        return 0

    written = 0
    for plan in plans:
        for path, payload in plan.updates.items():
            _write_artifact(path, payload)
            written += 1
            print(f"[{plan.paths.study_id}] wrote {path}")
    print(
        f"\nRefresh complete: {written} artifacts updated without backups. "
        "No LLM calls were made."
    )
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate the selected studies without writing artifacts.",
    )
    parser.add_argument(
        "--classification-only",
        action="store_true",
        help="Refresh classified JSON files only; leave enumeration artifacts unchanged.",
    )
    args = parser.parse_args()
    raise SystemExit(
        run(
            dry_run=args.dry_run,
            classification_only=args.classification_only,
        )
    )


if __name__ == "__main__":
    main()
