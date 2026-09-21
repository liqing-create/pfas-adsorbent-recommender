"""In-memory debug selection for the data-extraction stages.

Debug runs are intentionally opt-in.  They read the production stage inputs,
select rows named by an evaluator CSV, and write to a sibling ``*_debug``
directory.  No production input or output file is modified.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import os
from pathlib import Path
import re
from typing import Any, Iterable

import pandas as pd


def normalize_chunk_id(value: Any) -> str:
    text = str(value if value is not None else "").strip()
    if text.endswith(".0") and text[:-2].lstrip("+-").isdigit():
        return str(int(text[:-2]))
    return text


def normalize_study_id(value: Any) -> str:
    text = str(value if value is not None else "").strip()
    if not text:
        return ""
    if text.lower().startswith("study_"):
        suffix = text[6:]
    else:
        suffix = text
    suffix = normalize_chunk_id(suffix)
    if suffix.isdigit() and int(suffix) < 100:
        suffix = f"{int(suffix):02d}"
    return f"study_{suffix}" if suffix else ""


def normalize_source_id(value: Any) -> str:
    text = str(value if value is not None else "").strip().casefold()
    if not text:
        return ""
    # Source IDs are normally ``file_type_chunk_id``.  Only normalize a
    # numeric final component so file types containing underscores remain
    # unchanged.
    match = re.match(r"^(.*_)([+-]?\d+)(?:\.0)?$", text)
    if match:
        return f"{match.group(1)}{int(match.group(2))}"
    return text


def source_id_for_entry(entry: dict[str, Any]) -> str:
    explicit = normalize_source_id(entry.get("source_id"))
    if explicit:
        return explicit
    file_type = str(entry.get("file_type", "")).strip().casefold()
    chunk_id = normalize_chunk_id(entry.get("chunk_id", ""))
    return f"{file_type}_{chunk_id}" if chunk_id else file_type


def _split_source_ids(raw: Any) -> list[str]:
    if raw is None:
        return []
    if isinstance(raw, (list, tuple, set)):
        values: list[str] = []
        for item in raw:
            values.extend(_split_source_ids(item))
        return values
    return [
        normalize_source_id(str(part).strip().strip("[](){}'\" "))
        for part in re.split(r"[;,\n]+", str(raw))
        if str(part).strip()
    ]


def _first_value(row: dict[str, Any], names: Iterable[str]) -> str:
    for name in names:
        value = row.get(name)
        if value is not None and str(value).strip():
            return str(value).strip()
    return ""


@dataclass
class DebugTargets:
    """CSV targets for one study."""

    pairs: set[tuple[str, str]] = field(default_factory=set)
    source_ids: set[str] = field(default_factory=set)
    row_count: int = 0


@dataclass
class DebugSelection:
    """Parsed evaluator CSV grouped by normalized study ID."""

    targets_by_study: dict[str, DebugTargets]
    csv_path: Path
    row_count: int

    def targets_for(self, study_id: str) -> DebugTargets | None:
        return self.targets_by_study.get(normalize_study_id(study_id))


def load_debug_selection(
    csv_path: str | os.PathLike[str],
    *,
    task_filter: str | Iterable[str] | None = None,
    use_source_chunk_ids: bool = True,
) -> DebugSelection:
    """Read an evaluator CSV and return normalized per-study targets.

    Supported study columns are ``study_id``, ``study`` and
    ``study_folder``.  A row can identify a chunk using ``file_type`` plus
    ``chunk_id``, or a grouped source using ``source_chunk_ids``/``source_id``.
    """

    path = Path(csv_path).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"Debug CSV does not exist: {path}")
    df = pd.read_csv(path, dtype=str).fillna("")
    columns = set(df.columns)
    study_col = next(
        (name for name in ("study_id", "study", "study_folder") if name in columns),
        None,
    )
    if study_col is None:
        raise ValueError(
            f"Debug CSV must contain one of study_id, study, study_folder; found {sorted(columns)}"
        )

    filters: set[str] | None
    if task_filter is None or task_filter == "":
        filters = None
    elif isinstance(task_filter, str):
        filters = {item.strip().casefold() for item in task_filter.split(",") if item.strip()}
    else:
        filters = {str(item).strip().casefold() for item in task_filter if str(item).strip()}
    if filters:
        task_col = next(
            (name for name in ("task", "label", "task_name") if name in columns),
            None,
        )
        if task_col is None:
            raise ValueError(
                f"Task filtering was requested, but debug CSV has no task/label column: {sorted(columns)}"
            )
        df = df[df[task_col].astype(str).str.strip().str.casefold().isin(filters)]

    targets_by_study: dict[str, DebugTargets] = {}
    skipped = 0
    for row in df.to_dict("records"):
        study_id = normalize_study_id(row.get(study_col))
        if not study_id:
            skipped += 1
            continue
        targets = targets_by_study.setdefault(study_id, DebugTargets())
        targets.row_count += 1

        if use_source_chunk_ids:
            for value in _split_source_ids(
                _first_value(row, ("source_chunk_ids", "source_ids"))
            ):
                if value:
                    targets.source_ids.add(value)
            # ``source_id`` is the direct source key used by enumeration and
            # extraction evaluator CSVs.  It is safe to treat it as a source
            # target when present.
            source_id = _first_value(row, ("source_id",))
            if source_id:
                targets.source_ids.update(_split_source_ids(source_id))

        file_type = _first_value(row, ("file_type", "source_file_type")).casefold()
        chunk_id = normalize_chunk_id(
            _first_value(row, ("chunk_id", "source_chunk_id", "chunk"))
        )
        if file_type and chunk_id:
            targets.pairs.add((file_type, chunk_id))
        elif not targets.source_ids and (file_type or chunk_id):
            skipped += 1

    if skipped:
        print(f"[DEBUG] Skipped {skipped} CSV rows without study/chunk information.")
    if not targets_by_study:
        raise ValueError(f"Debug CSV contains no usable study/chunk rows after filtering: {path}")
    return DebugSelection(
        targets_by_study=targets_by_study,
        csv_path=path,
        row_count=sum(target.row_count for target in targets_by_study.values()),
    )


def _entry_source_ids(entry: dict[str, Any]) -> set[str]:
    ids = set()
    direct = source_id_for_entry(entry)
    if direct:
        ids.add(normalize_source_id(direct))

    for key in ("source_chunk_ids", "source_ids"):
        ids.update(_split_source_ids(entry.get(key)))

    metadata = entry.get("enriched_text_metadata")
    if isinstance(metadata, list):
        for item in metadata:
            if not isinstance(item, dict):
                continue
            source_id = source_id_for_entry(item)
            if source_id:
                ids.add(normalize_source_id(source_id))
    return {value for value in ids if value}


def _entry_pairs(entry: dict[str, Any]) -> set[tuple[str, str]]:
    pairs: set[tuple[str, str]] = set()
    direct = (
        str(entry.get("file_type", "")).strip().casefold(),
        normalize_chunk_id(entry.get("chunk_id", "")),
    )
    if direct[0] and direct[1]:
        pairs.add(direct)
    metadata = entry.get("enriched_text_metadata")
    if isinstance(metadata, list):
        for item in metadata:
            if not isinstance(item, dict):
                continue
            pair = (
                str(item.get("file_type", "")).strip().casefold(),
                normalize_chunk_id(item.get("chunk_id", "")),
            )
            if pair[0] and pair[1]:
                pairs.add(pair)
    return pairs


def filter_entries(entries: list[dict], targets: DebugTargets | None) -> list[dict]:
    """Keep entries matching direct pairs or grouped source IDs."""

    if targets is None:
        return []
    filtered: list[dict] = []
    for entry in entries or []:
        if not isinstance(entry, dict):
            continue
        if _entry_pairs(entry) & targets.pairs or _entry_source_ids(entry) & targets.source_ids:
            filtered.append(entry)
    return filtered


def debug_output_dir(production_dir: str | os.PathLike[str], suffix: str) -> Path:
    """Return a sibling output directory, refusing an unsafe empty suffix."""

    if not suffix:
        raise ValueError("Debug output suffix cannot be empty; refusing to write into production output")
    return Path(f"{Path(production_dir)}{suffix}")


def print_debug_summary(selection: DebugSelection, active_studies: Iterable[str]) -> None:
    active = [normalize_study_id(study_id) for study_id in active_studies]
    selected = [study_id for study_id in active if study_id in selection.targets_by_study]
    print(
        f"[DEBUG] CSV: {selection.csv_path} | rows={selection.row_count} | "
        f"active studies matched={len(selected)}/{len(active)}"
    )
    if selected:
        print(f"[DEBUG] Studies: {', '.join(selected)}")
