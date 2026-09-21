from __future__ import annotations

import argparse
import csv
import os
from typing import Any, Dict, Iterable, List, Optional, Tuple

# IMPLEMENTATION IMPORTS ------------------------------------------------------
import json
import logging
import re
import sys
import tempfile
import warnings
from collections import defaultdict
from pathlib import Path

try:  # Package execution
    from . import workflow_config as config
except ImportError:  # Direct ``python chunk_json_input.py`` execution
    import workflow_config as config

config.add_project_import_paths()

try:
    from .helpers.section_heading_utils import SECTION_LABELS
except ImportError:
    from helpers.section_heading_utils import SECTION_LABELS

VERBOSE = False
ATTENTION_BY_STUDY: Dict[str, List[str]] = defaultdict(list)
CHUNK_ISSUE_DECISIONS_CSV = config.PROCESSED_DIR / "chunk_issues_decisions.csv"
CHUNK_ISSUE_FIELDNAMES = ["study", "source", "issue"]
CHUNK_ISSUE_DECISION_FIELDNAMES = [*CHUNK_ISSUE_FIELDNAMES, "Decision"]
CHUNK_ISSUE_SUPPRESSING_DECISIONS = {"ok", "ignore", "fixed"}
CHUNK_ISSUE_DECISION_ROWS: List[Dict[str, str]] = []
CHUNK_ISSUE_DECISIONS: Dict[Tuple[str, str, str], str] = {}
CURRENT_ATTENTION_ISSUE_ROWS: List[Dict[str, str]] = []
CURRENT_ATTENTION_KEYS: set[Tuple[str, str, str]] = set()
MAX_CHAR_THRESHOLD = 5000
EMBED_MODEL_ID = "nomic-ai/nomic-embed-text-v1.5"

def _info(msg: str) -> None:
    if VERBOSE:
        print(msg)


def _study_sort_key(name: str) -> int:
    m = re.search(r"(\d+)$", name or "")
    return int(m.group(1)) if m else 10**9


def _chunk_issue_key(row: Dict[str, Any]) -> Tuple[str, str, str]:
    return (
        str(row.get("study", "") or "").strip().casefold(),
        str(row.get("source", "") or "").strip().casefold(),
        str(row.get("issue", "") or "").strip().casefold(),
    )


def _normalize_chunk_issue_decision(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip()).casefold()


def _chunk_issue_is_suppressed(row: Dict[str, Any]) -> bool:
    return (
        _normalize_chunk_issue_decision(
            CHUNK_ISSUE_DECISIONS.get(_chunk_issue_key(row), "")
        )
        in CHUNK_ISSUE_SUPPRESSING_DECISIONS
    )


def _reset_attention_state() -> None:
    ATTENTION_BY_STUDY.clear()
    CHUNK_ISSUE_DECISION_ROWS.clear()
    CHUNK_ISSUE_DECISIONS.clear()
    CURRENT_ATTENTION_ISSUE_ROWS.clear()
    CURRENT_ATTENTION_KEYS.clear()


def _read_chunk_issue_decision_rows() -> List[Dict[str, str]]:
    if not CHUNK_ISSUE_DECISIONS_CSV.is_file():
        return []
    try:
        with CHUNK_ISSUE_DECISIONS_CSV.open(
            "r", encoding="utf-8-sig", newline=""
        ) as f:
            reader = csv.DictReader(f)
            columns = {
                str(column).strip().casefold(): column
                for column in (reader.fieldnames or [])
            }
            decision_column = columns.get("decision")
            if decision_column is None:
                return []
            rows: List[Dict[str, str]] = []
            for row in reader:
                if not row:
                    continue
                normalized = {
                    key: str(row.get(columns.get(key, key), "") or "").strip()
                    for key in CHUNK_ISSUE_FIELDNAMES
                }
                normalized["Decision"] = str(
                    row.get(decision_column, "") or ""
                ).strip()
                if all(_chunk_issue_key(normalized)):
                    rows.append(normalized)
            return rows
    except Exception as exc:
        print(
            f"[CHUNK DECISIONS] Could not read {CHUNK_ISSUE_DECISIONS_CSV}: {exc}",
            file=sys.stderr,
        )
        return []


def _attention_summary_detail(row: Dict[str, str]) -> str:
    source = str(row.get("source") or "").strip()
    issue = str(row.get("issue") or "").strip()
    return f"{source}: {issue}" if source else issue


def _load_chunk_issue_decisions() -> None:
    CHUNK_ISSUE_DECISION_ROWS.extend(_read_chunk_issue_decision_rows())
    for row in CHUNK_ISSUE_DECISION_ROWS:
        key = _chunk_issue_key(row)
        decision = _normalize_chunk_issue_decision(row.get("Decision"))
        if decision:
            CHUNK_ISSUE_DECISIONS[key] = decision
        if not _chunk_issue_is_suppressed(row):
            detail = _attention_summary_detail(row)
            if detail not in ATTENTION_BY_STUDY[row["study"]]:
                ATTENTION_BY_STUDY[row["study"]].append(detail)


def _record_attention(
    study_folder: str,
    detail: str,
    *,
    source: str = "chunking",
) -> bool:
    """Record a warning and return whether it should be shown this run."""
    if not study_folder:
        return False
    row = {
        "study": study_folder,
        "source": source or "chunking",
        "issue": detail,
        "Decision": "",
    }
    key = _chunk_issue_key(row)
    if not all(key):
        return False
    first_occurrence = key not in CURRENT_ATTENTION_KEYS
    if first_occurrence:
        CURRENT_ATTENTION_KEYS.add(key)
        CURRENT_ATTENTION_ISSUE_ROWS.append(row)
    if _chunk_issue_is_suppressed(row):
        return False
    summary_detail = _attention_summary_detail(row)
    if summary_detail not in ATTENTION_BY_STUDY[study_folder]:
        ATTENTION_BY_STUDY[study_folder].append(summary_detail)
    return first_occurrence


def _new_chunk_issue_decision_rows(
    existing_rows: Iterable[Dict[str, str]],
    issue_rows: Iterable[Dict[str, str]],
) -> List[Dict[str, str]]:
    existing_keys = {
        _chunk_issue_key(row)
        for row in existing_rows
        if all(_chunk_issue_key(row))
    }
    new_rows: List[Dict[str, str]] = []
    for row in issue_rows:
        normalized = {
            key: str(row.get(key, "") or "").strip()
            for key in CHUNK_ISSUE_FIELDNAMES
        }
        key = _chunk_issue_key(normalized)
        if not all(key) or key in existing_keys:
            continue
        new_rows.append({**normalized, "Decision": ""})
        existing_keys.add(key)
    return new_rows


def _append_new_chunk_issue_decisions() -> int:
    new_rows = _new_chunk_issue_decision_rows(
        CHUNK_ISSUE_DECISION_ROWS,
        CURRENT_ATTENTION_ISSUE_ROWS,
    )
    if not new_rows:
        return 0

    CHUNK_ISSUE_DECISIONS_CSV.parent.mkdir(parents=True, exist_ok=True)
    has_content = (
        CHUNK_ISSUE_DECISIONS_CSV.is_file()
        and CHUNK_ISSUE_DECISIONS_CSV.stat().st_size > 0
    )
    encoding = "utf-8" if has_content else "utf-8-sig"
    with CHUNK_ISSUE_DECISIONS_CSV.open("a", encoding=encoding, newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CHUNK_ISSUE_DECISION_FIELDNAMES)
        if not has_content:
            writer.writeheader()
        for row in new_rows:
            writer.writerow(
                {
                    key: str(row.get(key, "") or "")
                    for key in CHUNK_ISSUE_DECISION_FIELDNAMES
                }
            )
    print(
        f"[CHUNK DECISIONS] Appended {len(new_rows)} new warning row(s) to "
        f"{CHUNK_ISSUE_DECISIONS_CSV}"
    )
    return len(new_rows)


def _print_attention_summary() -> None:
    if not ATTENTION_BY_STUDY:
        print("\n[ATTENTION] No studies flagged for review.")
        return
    studies = sorted(ATTENTION_BY_STUDY, key=_study_sort_key)
    print(
        "\n[ATTENTION] Review only (not the active-study selection): "
        + ", ".join(studies)
    )
    print("[ATTENTION DETAILS]")
    for study in studies:
        print(f"- {study}:")
        for detail in ATTENTION_BY_STUDY[study][:8]:
            print(f"  - {detail}")
        if len(ATTENTION_BY_STUDY[study]) > 8:
            print(f"  - ... {len(ATTENTION_BY_STUDY[study]) - 8} more")


def _clear_resolved_section_attention(
    study: str,
    found_labels: Iterable[str],
) -> None:
    """Drop historical section-missing warnings resolved in current input.

    Section warnings are persisted for review, but a previous unresolved row
    should not keep appearing after preprocessing has regenerated the missing
    section label.  Current omissions are still recorded by
    :func:`_record_attention` below.
    """
    found = set(found_labels)
    required = {
        "ABSTRACT": SECTION_LABELS.get("abstract", "abstract"),
        "MATERIALS/METHODS": SECTION_LABELS.get(
            "materials_and_methods", "materials_and_methods"
        ),
        "RESULTS/DISCUSSION": SECTION_LABELS.get(
            "results_and_discussion", "results_and_discussion"
        ),
    }
    resolved_details = {
        f"chunk_input.jsonl: {display} section not present"
        for display, label in required.items()
        if label in found
    }
    if not resolved_details or study not in ATTENTION_BY_STUDY:
        return

    remaining = [
        detail
        for detail in ATTENTION_BY_STUDY[study]
        if detail not in resolved_details
    ]
    if remaining:
        ATTENTION_BY_STUDY[study] = remaining
    else:
        ATTENTION_BY_STUDY.pop(study, None)


logging.basicConfig(level=logging.WARNING)
logging.getLogger().setLevel(logging.WARNING)
for _name in ("docling", "transformers"):
    logging.getLogger(_name).setLevel(logging.WARNING)
warnings.filterwarnings("ignore", category=DeprecationWarning)
# Created only when a study actually needs chunking.  This keeps direct import
# smoke checks and ``--check`` fast and free of model-download side effects.
converter: Any = None
chunker: Any = None


def configure_chunking_runtime() -> None:
    """Initialise the Docling/embedding runtime on demand."""
    global converter, chunker
    if converter is not None and chunker is not None:
        return
    try:
        from docling.chunking import HybridChunker
        from docling.document_converter import DocumentConverter
        from transformers import AutoTokenizer
        from transformers.utils import logging as transformer_logging
    except Exception as exc:
        raise RuntimeError(f"Docling and transformers are required for chunking: {exc}") from exc
    transformer_logging.set_verbosity_error()
    converter = DocumentConverter()
    tokenizer = AutoTokenizer.from_pretrained(
        EMBED_MODEL_ID,
        trust_remote_code=True,
        model_max_length=8192,
    )
    chunker = HybridChunker(tokenizer=tokenizer, max_tokens=512, merge_peers=True)

_PIPE_HEADING_RE = re.compile(r"^(#{1,6}\s+)(.*\|.*)$", re.MULTILINE)


def normalize_docling_heading_pipes(md_text: str, study_folder: str = "", source_hint: str = "") -> str:
    """Normalize pipe-style markdown headings before Docling chunking."""
    def _replace(match: re.Match[str]) -> str:
        heading = re.sub(r"\s*\|\s*", " - ", match.group(2).strip())
        heading = re.sub(r"\s+", " ", heading).strip()
        return f"{match.group(1)}{heading}"

    return _PIPE_HEADING_RE.sub(_replace, md_text or "")


def _next_chunk_id(chunks_output: List[Dict[str, Any]], study_folder: str, file_type: str) -> int:
    mx = -1
    for c in chunks_output:
        if c.get("study_folder") == study_folder and c.get("file_type") == file_type:
            try:
                mx = max(mx, int(c.get("chunk_id", -1)))
            except Exception:
                pass
    return mx + 1


def _load_jsonl(path: Path, study_folder: str = "") -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line_no, line in enumerate(f, start=1):
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
                if isinstance(obj, dict):
                    rows.append(obj)
            except Exception as exc:
                detail = f"invalid JSONL row at line {line_no}: {exc}"
                if _record_attention(study_folder, detail, source=path.name):
                    print(f"⚠️ Skipping invalid JSONL row: {path}:{line_no} ({exc})")
    rows.sort(key=lambda r: (str(r.get("source_file") or ""), int(r.get("order_start") or 0)))
    return rows


def _record_markdown(record: Dict[str, Any]) -> str:
    return str(record.get("markdown") or record.get("text") or "").strip()


def _record_text(record: Dict[str, Any]) -> str:
    return str(record.get("text") or record.get("markdown") or "").strip()


def chunk_markdown_text(md_text: str, label: str, tmp_dir: Path, source_hint: str, study_folder: str = "") -> List[Dict[str, Any]]:
    configure_chunking_runtime()
    out: List[Dict[str, Any]] = []
    if not (md_text or "").strip():
        return out
    md_text = normalize_docling_heading_pipes(md_text, study_folder=study_folder, source_hint=source_hint or label)
    prefix = f"_tmp_{study_folder + '_' if study_folder else ''}{source_hint}_{label}_"
    fd, tmp_path = tempfile.mkstemp(prefix=prefix, suffix=".md", dir=str(tmp_dir), text=True)
    os.close(fd)
    tmp = Path(tmp_path)
    tmp.write_text(md_text, encoding="utf-8")
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            warnings.filterwarnings("ignore", category=DeprecationWarning)
            doc_container = converter.convert(source=str(tmp))
            dl_doc = doc_container.document
            for chunk in chunker.chunk(dl_doc=dl_doc):
                plain_text = chunk.text
                if not plain_text.strip():
                    continue
                enriched_text = "search_document: " + chunker.serialize(chunk=chunk)
                char_count = len(plain_text)
                out.append(
                    {
                        "text": plain_text,
                        "enriched_text": enriched_text,
                        "char_count": char_count,
                        "needs_split": char_count > MAX_CHAR_THRESHOLD,
                        "section_label": label,
                    }
                )
            for warning in caught:
                msg = str(warning.message)
                if "Headers and captions for this chunk are longer" in msg:
                    source = source_hint or label
                    detail = (
                        "Docling ignored overlong headers/captions during chunking; "
                        "verify output"
                    )
                    if _record_attention(study_folder, detail, source=source):
                        print(
                            f"⚠️ Docling warning → study={study_folder}, source={source} | "
                            "headers/captions exceeded chunk size and were ignored"
                        )
    finally:
        try:
            tmp.unlink()
        except OSError:
            pass
    return out


def make_single_chunk(md_text: str, label: str) -> Optional[Dict[str, Any]]:
    txt = (md_text or "").strip()
    if not txt:
        return None
    char_count = len(txt)
    return {
        "text": txt,
        "enriched_text": "search_document: " + txt,
        "char_count": char_count,
        "needs_split": char_count > MAX_CHAR_THRESHOLD,
        "section_label": label,
    }


def _emit_table_chunks(study: str, records: List[Dict[str, Any]], chunks_output: List[Dict[str, Any]]) -> None:
    table_count = _next_chunk_id(chunks_output, study, "table")
    for rec in records:
        table_text = _record_text(rec)
        if not table_text:
            continue
        char_count = len(table_text)
        chunks_output.append(
            {
                "study_folder": study,
                "file_type": "table",
                "chunk_id": table_count,
                "text": table_text,
                "enriched_text": "search_document: " + table_text,
                "char_count": char_count,
                "needs_split": char_count > MAX_CHAR_THRESHOLD,
            }
        )
        table_count += 1


def _emit_main_paper_chunks(study: str, folder_path: Path, records: List[Dict[str, Any]], chunks_output: List[Dict[str, Any]]) -> None:
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    order: List[str] = []
    for rec in records:
        label = str(rec.get("section_label") or "unknown")
        if label not in grouped:
            order.append(label)
        grouped[label].append(rec)

    main_chunk_id = _next_chunk_id(chunks_output, study, "main_paper")
    required = {
        "ABSTRACT": SECTION_LABELS.get("abstract", "abstract"),
        "MATERIALS/METHODS": SECTION_LABELS.get("materials_and_methods", "materials_and_methods"),
        "RESULTS/DISCUSSION": SECTION_LABELS.get("results_and_discussion", "results_and_discussion"),
    }
    found = set(grouped)
    _clear_resolved_section_attention(study, found)
    for display, label in required.items():
        if label not in found:
            detail = f"{display} section not present"
            if _record_attention(study, detail, source="chunk_input.jsonl"):
                print(f"⚠️ Section extraction warning → study={study}, source=chunk_input.jsonl | {display}: section not present")

    for label in order:
        recs = grouped[label]
        md_text = "\n\n".join(_record_markdown(r) for r in recs if _record_markdown(r)).strip()
        if not md_text:
            continue
        if label == SECTION_LABELS.get("abstract", "abstract"):
            ch = make_single_chunk(md_text, label=label)
            if ch:
                chunks_output.append(
                    {
                        "study_folder": study,
                        "file_type": "main_paper",
                        "chunk_id": main_chunk_id,
                        "text": ch["text"],
                        "enriched_text": ch["enriched_text"],
                        "char_count": ch["char_count"],
                        "needs_split": ch["needs_split"],
                        "section_label": ch["section_label"],
                    }
                )
                main_chunk_id += 1
        else:
            sec_chunks = chunk_markdown_text(md_text, label=label, tmp_dir=folder_path, source_hint="main_paper", study_folder=study)
            for ch in sec_chunks:
                chunks_output.append(
                    {
                        "study_folder": study,
                        "file_type": "main_paper",
                        "chunk_id": main_chunk_id,
                        "text": ch["text"],
                        "enriched_text": ch["enriched_text"],
                        "char_count": ch["char_count"],
                        "needs_split": ch["needs_split"],
                        "section_label": ch["section_label"],
                    }
                )
                main_chunk_id += 1


def _emit_supp_chunks(study: str, folder_path: Path, records: List[Dict[str, Any]], chunks_output: List[Dict[str, Any]]) -> None:
    # Keep true supplementary and routed main-paper body separated for auditability,
    # but they share the same output file_type and chunk_id sequence.
    grouped: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
    order: List[Tuple[str, str]] = []
    for rec in records:
        key = (str(rec.get("source_file") or ""), str(rec.get("original_file_type") or ""))
        if key not in grouped:
            order.append(key)
        grouped[key].append(rec)

    for source_file, original_file_type in order:
        recs = grouped[(source_file, original_file_type)]
        md_text = "\n\n".join(_record_markdown(r) for r in recs if _record_markdown(r)).strip()
        if not md_text:
            continue
        start_id = _next_chunk_id(chunks_output, study, "supplementary_material")
        source_hint = "main_paper_route_as_supp" if original_file_type == "main_paper" else (source_file or "supplementary_material")
        supp_chunks = chunk_markdown_text(md_text, label="supplementary_material", tmp_dir=folder_path, source_hint=source_hint, study_folder=study)
        if not supp_chunks:
            detail = "produced 0 supplementary chunks"
            if _record_attention(study, detail, source=source_hint):
                print(f"⚠️ Chunking warning → study={study}, source={source_hint} | produced 0 supplementary chunks")
        for i, ch in enumerate(supp_chunks):
            out = {
                "study_folder": study,
                "file_type": "supplementary_material",
                "chunk_id": start_id + i,
                "text": ch["text"],
                "enriched_text": ch["enriched_text"],
                "char_count": ch["char_count"],
                "needs_split": ch["needs_split"],
            }
            if original_file_type:
                out["original_file_type"] = original_file_type
            chunks_output.append(out)


def process_study_folder(paths: config.StudyPaths) -> List[Dict[str, Any]]:
    study_folder = paths.study_id
    folder_path = paths.processed_study_dir
    chunk_input_path = folder_path / "chunk_input.jsonl"
    if not chunk_input_path.is_file():
        detail = "chunk_input.jsonl not found; study skipped"
        if _record_attention(study_folder, detail, source="chunk_input.jsonl"):
            print(f"    [WARNING] Missing {chunk_input_path}. Run docling_json_preprocess.py first.")
        return []

    records = _load_jsonl(chunk_input_path, study_folder=study_folder)
    chunks_output: List[Dict[str, Any]] = []

    table_records = [r for r in records if r.get("file_type") == "table"]
    main_records = [r for r in records if r.get("file_type") == "main_paper"]
    supp_records = [r for r in records if r.get("file_type") == "supplementary_material"]

    _emit_table_chunks(study_folder, table_records, chunks_output)
    _emit_main_paper_chunks(study_folder, folder_path, main_records, chunks_output)
    _emit_supp_chunks(study_folder, folder_path, supp_records, chunks_output)

    return chunks_output


def merge_short_chunks(chunks: List[Dict[str, Any]], min_length: int = 50) -> List[Dict[str, Any]]:
    merged_chunks: List[Dict[str, Any]] = []
    current_chunk = ""
    current_metadata: Optional[Dict[str, Any]] = None
    for chunk in chunks:
        if chunk["file_type"] not in ["main_paper", "supplementary_material"]:
            merged_chunks.append(chunk)
            continue
        word_count = len(chunk["text"].split())
        if word_count < min_length:
            if current_metadata is None:
                current_metadata = chunk.copy()
                current_chunk = chunk["text"]
            else:
                if (
                    current_metadata.get("file_type") == "main_paper"
                    and chunk.get("file_type") == "main_paper"
                    and current_metadata.get("section_label") != chunk.get("section_label")
                ):
                    merged_text = current_chunk.strip()
                    current_metadata["text"] = merged_text
                    current_metadata["enriched_text"] = "search_document: " + merged_text
                    current_metadata["char_count"] = len(merged_text)
                    current_metadata["needs_split"] = len(merged_text) > MAX_CHAR_THRESHOLD
                    merged_chunks.append(current_metadata)
                    current_metadata = chunk.copy()
                    current_chunk = chunk["text"]
                    continue
                current_chunk += " " + chunk["text"]
        else:
            if current_metadata:
                merged_text = current_chunk.strip()
                current_metadata["text"] = merged_text
                current_metadata["enriched_text"] = "search_document: " + merged_text
                current_metadata["char_count"] = len(merged_text)
                current_metadata["needs_split"] = len(merged_text) > MAX_CHAR_THRESHOLD
                merged_chunks.append(current_metadata)
                current_metadata = None
                current_chunk = ""
            merged_chunks.append(chunk)
    if current_metadata:
        merged_text = current_chunk.strip()
        current_metadata["text"] = merged_text
        current_metadata["enriched_text"] = "search_document: " + merged_text
        current_metadata["char_count"] = len(merged_text)
        current_metadata["needs_split"] = len(merged_text) > MAX_CHAR_THRESHOLD
        merged_chunks.append(current_metadata)
    return merged_chunks


def main(*, check_only: bool = False) -> int:
    _reset_attention_state()
    _load_chunk_issue_decisions()
    selected_paths = list(config.iter_active_study_paths())
    if not selected_paths:
        print("[ERROR] ACTIVE_STUDIES is empty; select at least one registered study.")
        return 2

    ready_paths = []
    missing_inputs = False
    for paths in selected_paths:
        problems = config.chunk_preflight(paths)
        if problems:
            missing_inputs = True
            print(f"[ERROR] {paths.study_id}: " + " | ".join(problems))
            continue
        if not config.OVERWRITE_EXISTING_OUTPUTS and paths.chunked_output_file.is_file():
            print(f"[RETAIN] {paths.study_id}: existing {paths.chunked_output_file} will not be replaced")
            continue
        ready_paths.append(paths)
        if check_only:
            print(
                f"[CHECK OK] {paths.study_id} ({paths.group}): "
                f"would regenerate {paths.chunked_output_file}"
            )

    if check_only:
        return 2 if missing_inputs else 0
    if not ready_paths:
        if not missing_inputs:
            print("[DONE] No chunk files need regeneration.")
        if not check_only:
            _print_attention_summary()
        return 2 if missing_inputs else 0

    processed = 0
    for paths in ready_paths:
        study = paths.study_id
        _info(f"\n[INFO] Processing {study} ...")
        try:
            folder_chunks = process_study_folder(paths)
            if not folder_chunks:
                detail = "no parsable records found; study skipped"
                if _record_attention(study, detail, source="chunk_input.jsonl"):
                    print(f"    [WARNING] No parsable records found in {study}. Skipping write.")
                continue
            merged_chunks = merge_short_chunks(folder_chunks, min_length=50)
            paths.chunked_output_file.parent.mkdir(parents=True, exist_ok=True)
            with paths.chunked_output_file.open("w", encoding="utf-8") as f:
                json.dump(merged_chunks, f, indent=4, ensure_ascii=False)
            processed += 1
            _info(f"    [SUCCESS] Saved {len(merged_chunks)} chunks => {paths.chunked_output_file}")
        except Exception as exc:
            detail = f"failed processing study folder ({exc})"
            if _record_attention(study, detail, source="chunking"):
                print(f"    [ERROR] Failed processing {study}: {exc}")

    _append_new_chunk_issue_decisions()
    _info(f"\n[DONE] Completed processing {processed} of {len(ready_paths)} selected study folder(s).")
    _print_attention_summary()
    return 2 if missing_inputs else 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="validate selected input paths without writing")
    raise SystemExit(main(check_only=parser.parse_args().check))
