"""Convert selected source PDFs/DOCX files to ``*.docling.json`` safely."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

try:  # Package execution
    from . import workflow_config as config
except ImportError:  # Direct ``python convert_to_docling_json.py`` execution
    import workflow_config as config


def _write_docling_json(document: Any, destination: Path) -> None:
    """Persist a Docling document across supported Docling versions."""
    if hasattr(document, "save_as_json"):
        document.save_as_json(str(destination))
        return
    if hasattr(document, "export_to_dict"):
        destination.write_text(
            json.dumps(document.export_to_dict(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return
    raise RuntimeError("The installed Docling document has no JSON export method.")


def _write_markdown_if_supported(document: Any, destination: Path) -> None:
    """Keep the human-review Markdown companion when Docling supports it."""
    if hasattr(document, "save_as_markdown"):
        document.save_as_markdown(str(destination))
    elif hasattr(document, "export_to_markdown"):
        destination.write_text(document.export_to_markdown(), encoding="utf-8")


def convert_study(paths: config.StudyPaths, converter: Any) -> tuple[int, list[str]]:
    """Convert available source documents for one study and return a summary."""
    converted = 0
    messages: list[str] = []
    for source_stem in config.SOURCE_FILE_STEMS:
        destination = paths.docling_json_file(source_stem)
        if destination.is_file() and not config.OVERWRITE_EXISTING_OUTPUTS:
            messages.append(f"reuse existing {destination.name}")
            continue

        source_document = config.source_document_for(paths, source_stem)
        if source_document is None:
            if not destination.is_file():
                messages.append(f"skip {source_stem}: no PDF/DOCX source")
            continue

        try:
            result = converter.convert(source=str(source_document))
            document = getattr(result, "document", result)
            _write_docling_json(document, destination)
            _write_markdown_if_supported(document, paths.markdown_file(source_stem))
            converted += 1
            messages.append(f"converted {source_document.name} -> {destination.name}")
        except Exception as exc:
            messages.append(f"failed {source_document.name}: {exc}")
    return converted, messages


def main(*, check_only: bool = False) -> int:
    selected_paths = list(config.iter_active_study_paths())
    if not selected_paths:
        print("[ERROR] ACTIVE_STUDIES is empty; select at least one registered study.")
        return 2

    ready: list[config.StudyPaths] = []
    for paths in selected_paths:
        problems = config.conversion_preflight(paths)
        if problems:
            print(f"[ERROR] {paths.study_id}: " + " | ".join(problems))
            continue
        ready.append(paths)
        print(f"[READY] {paths.study_id} ({paths.group}) -> {paths.source_study_dir}")

    if check_only:
        return 0 if len(ready) == len(selected_paths) else 2
    if not ready:
        return 2

    # Import lazily so --check and import smoke tests never initialise Docling.
    try:
        from docling.document_converter import DocumentConverter
    except Exception as exc:
        print(f"[ERROR] Docling is required for conversion: {exc}", file=sys.stderr)
        return 2

    converter = DocumentConverter()
    converted = 0
    for paths in ready:
        count, messages = convert_study(paths, converter)
        converted += count
        for message in messages:
            print(f"[{paths.study_id}] {message}")
    print(f"[DONE] Converted {converted} source document(s) for {len(ready)} selected study/studies.")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="validate selected input paths without writing")
    raise SystemExit(main(check_only=parser.parse_args().check))
