"""Run a CSV-selected debug subset without touching production artifacts.

The evaluator-produced CSV is discovered automatically from the selected
stage's configured output folder. Pass ``--csv`` only to use a different or
historical report. Choose exactly one stage with ``--classify``,
``--enumerate``, or ``--extract``. The selected study batch remains controlled
by ``workflow_config.ACTIVE_STUDIES``.
"""

from __future__ import annotations

import argparse
import json
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Any

try:
    from .workflow_config import (
        PROMPTS_DIR,
        StudyPaths,
        add_project_import_paths,
        check_stage_input,
        iter_active_study_paths,
    )
    from .helpers.debug_filter import (
        DebugSelection,
        debug_output_dir,
        filter_entries,
        load_debug_selection,
        print_debug_summary,
    )
except ImportError:  # Supports ``python debug.py`` from this folder.
    from workflow_config import (
        PROMPTS_DIR,
        StudyPaths,
        add_project_import_paths,
        check_stage_input,
        iter_active_study_paths,
    )
    from helpers.debug_filter import (
        DebugSelection,
        debug_output_dir,
        filter_entries,
        load_debug_selection,
        print_debug_summary,
    )

add_project_import_paths()

try:
    from . import classify as classification_stage
    from . import enumerate as enumeration_stage
    from . import extract as extraction_stage
except ImportError:  # Supports ``python debug.py`` from this folder.
    import classify as classification_stage
    import enumerate as enumeration_stage
    import extract as extraction_stage

from chains.llm_config import get_llm
from chains.classify_chain import create_classification_chain


# Optional defaults for repeated troubleshooting.  Command-line options take
# precedence, so these normally remain blank/default.
DEBUG_CSV_PATH = ""
DEBUG_TASK_FILTER = ""
DEBUG_OUTPUT_SUFFIX = "_debug"
USE_SOURCE_CHUNK_IDS = True

EVALUATOR_CSV_FILENAMES = {
    "classify": "mislabeled_chunks.csv",
    "enumerate": "mishandled_chunks.csv",
    "extract": "misclassified_fields.csv",
}


def _load_json(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, list):
        raise ValueError(f"Expected a JSON list: {path}")
    return payload


def _default_csv_path(stage: str, active_paths: list[StudyPaths]) -> Path:
    """Find the evaluator CSV for the selected stage and active study group."""

    directory_attribute = {
        "classify": "classification_dir",
        "enumerate": "enumeration_dir",
        "extract": "extraction_dir",
    }[stage]
    filename = EVALUATOR_CSV_FILENAMES[stage]
    candidates = list(
        dict.fromkeys(
            Path(getattr(paths, directory_attribute)) / filename
            for paths in active_paths
        )
    )
    existing = [path for path in candidates if path.is_file()]
    if len(existing) == 1:
        return existing[0]
    if not existing:
        locations = "\n  ".join(str(path) for path in candidates)
        raise SystemExit(
            f"No {filename} was found for the selected {stage} run. "
            "Run the corresponding evaluation script first, or pass --csv PATH."
            f"\nChecked:\n  {locations}"
        )

    locations = "\n  ".join(str(path) for path in existing)
    raise SystemExit(
        f"Multiple evaluator CSVs match the active studies for {stage}. "
        f"Pass --csv PATH to choose one explicitly.\nFound:\n  {locations}"
    )


def _select_entries(
    *,
    study_id: str,
    entries: list[dict],
    selection: DebugSelection,
    source_name: str,
) -> list[dict]:
    targets = selection.targets_for(study_id)
    if targets is None:
        print(f"[{study_id}] [DEBUG] no rows in CSV; skipping")
        return []
    filtered = filter_entries(entries, targets)
    print(
        f"[{study_id}] [DEBUG] selected {len(filtered)} of {len(entries)} "
        f"{source_name} entries from CSV"
    )
    return filtered


def _classification_jobs(selection: DebugSelection) -> list[tuple[StudyPaths, list[dict]]]:
    jobs: list[tuple[StudyPaths, list[dict]]] = []
    for paths in iter_active_study_paths():
        try:
            input_path = check_stage_input(paths, "classification")
            chunks = _load_json(input_path)
        except (FileNotFoundError, ValueError, json.JSONDecodeError) as exc:
            print(f"[SKIP] {paths.study_id}: {exc}")
            continue
        selected = _select_entries(
            study_id=paths.study_id,
            entries=chunks,
            selection=selection,
            source_name="chunk",
        )
        if selected:
            jobs.append((paths, selected))
    return jobs


def run_classification(selection: DebugSelection, output_suffix: str) -> None:
    jobs = _classification_jobs(selection)
    if not jobs:
        print("[DEBUG] No classified chunks matched the selected studies.")
        return

    model_name = (
        classification_stage.OPENAI_MODEL
        if classification_stage.LLM_PROVIDER == "openai"
        else classification_stage.TOGETHER_MODEL
    )
    print(f"Using {classification_stage.LLM_PROVIDER} model: {model_name}")
    llm = get_llm(provider=classification_stage.LLM_PROVIDER, model_name=model_name)
    chain = create_classification_chain(PROMPTS_DIR, llm)
    title_abstract_map = classification_stage.build_title_abstract_map(
        classification_stage.EXCEL_PATH
    )

    grouped_results: dict[Path, list[dict[str, Any]]] = defaultdict(list)
    for paths, chunks in jobs:
        output_dir = debug_output_dir(paths.classification_dir, output_suffix)
        output_file = output_dir / paths.classified_file.name
        print(f"\n=== Debug classification: {paths.study_id} ({paths.group}) ===")
        results = classification_stage.classify_study(
            paths.study_id,
            chain,
            llm,
            title_abstract_map,
            provider=classification_stage.LLM_PROVIDER,
            model_name=model_name,
            chunks_override=chunks,
            output_file_override=output_file,
            force_rerun=True,
        )
        grouped_results[output_dir].extend(results)

    for output_dir, results in grouped_results.items():
        output_dir.mkdir(parents=True, exist_ok=True)
        combined_file = output_dir / "all_studies_classified.json"
        with combined_file.open("w", encoding="utf-8") as handle:
            json.dump(results, handle, indent=2, ensure_ascii=False)
        print(f"[DEBUG] Combined classification output: {combined_file}")


def _enumeration_jobs(selection: DebugSelection) -> list[tuple[StudyPaths, list[dict]]]:
    jobs: list[tuple[StudyPaths, list[dict]]] = []
    for paths in iter_active_study_paths():
        try:
            input_path = check_stage_input(paths, "enumeration")
            entries = _load_json(input_path)
        except (FileNotFoundError, ValueError, json.JSONDecodeError) as exc:
            print(f"[SKIP] {paths.study_id}: {exc}")
            continue
        selected = _select_entries(
            study_id=paths.study_id,
            entries=entries,
            selection=selection,
            source_name="classified",
        )
        if selected:
            jobs.append((paths, selected))
    return jobs


def _seed_enumeration_prerequisite_artifacts(
    paths: StudyPaths,
    output_dir: Path,
) -> None:
    """Copy normal lexicon artifacts into the isolated debug run directory.

    Performance enumeration needs the study-wide adsorbent and water-type
    lexicons.  Copying their completed artifacts keeps the debug run
    self-contained while avoiding a second lexicon LLM call on a small,
    selected subset of chunks.
    """

    source_dir = paths.enumeration_dir / "_chain_artifacts" / paths.study_id
    destination_dir = output_dir / "_chain_artifacts" / paths.study_id
    for task in ("adsorbent", "water_type"):
        source = source_dir / f"{paths.study_id}_{task}.json"
        if not source.is_file():
            print(f"[{paths.study_id}] [DEBUG] prerequisite artifact not found: {source}")
            continue
        destination_dir.mkdir(parents=True, exist_ok=True)
        destination = destination_dir / source.name
        shutil.copy2(source, destination)
        print(f"[{paths.study_id}] [DEBUG] copied normal {task} artifact for reuse")


def run_enumeration(selection: DebugSelection, output_suffix: str) -> None:
    jobs = _enumeration_jobs(selection)
    if not jobs:
        print("[DEBUG] No classified chunks matched the selected studies.")
        return

    if enumeration_stage.LLM_PROVIDER not in enumeration_stage.LLM_MODELS:
        raise ValueError(
            f"Unsupported LLM provider: {enumeration_stage.LLM_PROVIDER}. "
            f"Configured providers: {sorted(enumeration_stage.LLM_MODELS)}"
        )
    enumeration_stage.LLM_MODEL_NAME = enumeration_stage.LLM_MODELS[
        enumeration_stage.LLM_PROVIDER
    ]
    print(f"Using {enumeration_stage.LLM_PROVIDER} model: {enumeration_stage.LLM_MODEL_NAME}")
    llm = get_llm(
        provider=enumeration_stage.LLM_PROVIDER,
        model_name=enumeration_stage.LLM_MODEL_NAME,
    )
    for paths, entries in jobs:
        output_dir = debug_output_dir(paths.enumeration_dir, output_suffix)
        print(f"\n=== Debug enumeration: {paths.study_id} ({paths.group}) ===")
        _seed_enumeration_prerequisite_artifacts(paths, output_dir)
        enumeration_stage.process_study(
            paths.study_id,
            llm,
            paths=paths,
            classified_entries_override=entries,
            output_dir=output_dir,
            force_llm_rerun=True,
        )


def _extraction_jobs(
    selection: DebugSelection,
) -> list[tuple[StudyPaths, list[dict] | None, list[dict]]]:
    jobs: list[tuple[StudyPaths, list[dict] | None, list[dict]]] = []
    for paths in iter_active_study_paths():
        try:
            enumeration_path = check_stage_input(paths, "extraction")
            enumeration_entries = _load_json(enumeration_path)
        except (FileNotFoundError, ValueError, json.JSONDecodeError) as exc:
            print(f"[SKIP] {paths.study_id}: {exc}")
            continue

        selected_enumeration = _select_entries(
            study_id=paths.study_id,
            entries=enumeration_entries,
            selection=selection,
            source_name="enumeration",
        )
        classified_entries: list[dict] | None = None
        if paths.classified_file.exists():
            classified_entries = _select_entries(
                study_id=paths.study_id,
                entries=_load_json(paths.classified_file),
                selection=selection,
                source_name="classified",
            )
        else:
            print(f"[{paths.study_id}] [DEBUG] classified input not found: {paths.classified_file}")

        if selected_enumeration or classified_entries:
            jobs.append((paths, classified_entries, selected_enumeration))
        else:
            print(f"[{paths.study_id}] [DEBUG] no matching extraction inputs; skipping")
    return jobs


def run_extraction(selection: DebugSelection, output_suffix: str) -> None:
    jobs = _extraction_jobs(selection)
    if not jobs:
        print("[DEBUG] No extraction inputs matched the selected studies.")
        return

    print(f"Using {extraction_stage.LLM_PROVIDER} model: {extraction_stage.LLM_MODEL_NAME}")
    llm = get_llm(
        provider=extraction_stage.LLM_PROVIDER,
        model_name=extraction_stage.LLM_MODEL_NAME,
    )
    for paths, classified_entries, enumeration_entries in jobs:
        output_dir = debug_output_dir(paths.extraction_dir, output_suffix)
        print(f"\n=== Debug extraction: {paths.study_id} ({paths.group}) ===")
        extraction_stage.process_study(
            paths.study_id,
            llm,
            paths=paths,
            classified_entries_override=classified_entries,
            performance_entries_override=enumeration_entries,
            output_dir=output_dir,
            force_llm_rerun=True,
        )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    stage = parser.add_mutually_exclusive_group(required=True)
    stage.add_argument("--classify", dest="stage", action="store_const", const="classify")
    stage.add_argument("--enumerate", dest="stage", action="store_const", const="enumerate")
    stage.add_argument("--extract", dest="stage", action="store_const", const="extract")
    parser.add_argument(
        "--csv",
        help="Optional evaluator CSV override (for an alternate or historical report).",
    )
    parser.add_argument(
        "--task-filter",
        help="Optional comma-separated task/label filter. Overrides DEBUG_TASK_FILTER.",
    )
    parser.add_argument(
        "--output-suffix",
        help="Suffix for sibling debug output directories. Overrides DEBUG_OUTPUT_SUFFIX.",
    )
    parser.add_argument(
        "--ignore-source-chunk-ids",
        action="store_true",
        help="Use only file_type/chunk_id rows, ignoring grouped source_chunk_ids/source_id fields.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    active_paths = list(iter_active_study_paths())
    csv_path = args.csv or DEBUG_CSV_PATH or _default_csv_path(args.stage, active_paths)
    task_filter = args.task_filter if args.task_filter is not None else DEBUG_TASK_FILTER
    output_suffix = args.output_suffix if args.output_suffix is not None else DEBUG_OUTPUT_SUFFIX
    use_source_chunk_ids = USE_SOURCE_CHUNK_IDS and not args.ignore_source_chunk_ids
    selection = load_debug_selection(
        csv_path,
        task_filter=task_filter,
        use_source_chunk_ids=use_source_chunk_ids,
    )
    print_debug_summary(selection, (paths.study_id for paths in active_paths))

    if args.stage == "classify":
        run_classification(selection, output_suffix)
    elif args.stage == "enumerate":
        run_enumeration(selection, output_suffix)
    else:
        run_extraction(selection, output_suffix)


if __name__ == "__main__":
    main()
