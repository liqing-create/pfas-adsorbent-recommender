from __future__ import annotations

import difflib
import json
import re
import shutil
import subprocess
import tempfile
import zipfile
from pathlib import Path
from typing import Any
from xml.etree import ElementTree as ET

from docling_core.types.doc import DoclingDocument, ImageRefMode


TABLE_ID_RE = re.compile(
    r"\btable\s+(?P<id>(?:supplementary\s+|supplemental\s+)?"
    r"(?:s|si|sm|st|a)?\s*[\.\-]?\s*\d+(?:\s*[a-z]|\.\d+)?)",
    re.IGNORECASE,
)
FIGURE_ID_RE = re.compile(
    r"\b(?:fig(?:ure)?\.?|scheme)\s*"
    r"(?P<id>(?:s|si|sm|sf|a)?\s*[\.\-]?\s*\d+(?:\s*[a-z]|\.\d+)?)",
    re.IGNORECASE,
)
REL_NS = "http://schemas.openxmlformats.org/package/2006/relationships"
OFFICE_REL_NS = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
VECTOR_EXTENSIONS = {".emf", ".wmf", ".svg"}


def normalize_space(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def normalize_match(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", normalize_space(text).lower()).strip()


def normalize_table_id(text: str) -> str:
    match = TABLE_ID_RE.search(text or "")
    raw = match.group("id") if match else (text or "")
    raw = raw.lower().replace("supplementary", "s").replace("supplemental", "s")
    return re.sub(r"[^a-z0-9]+", "", raw)


def normalize_figure_id(text: str) -> str:
    match = FIGURE_ID_RE.search(text or "")
    raw = match.group("id") if match else (text or "")
    return re.sub(r"[^a-z0-9]+", "", raw.lower())


def label_value(item: Any) -> str:
    label = getattr(item, "label", "")
    return getattr(label, "value", str(label))


def load_document(path: Path) -> DoclingDocument:
    return DoclingDocument.model_validate_json(path.read_text(encoding="utf-8"))


def table_markdown(table: Any, doc: DoclingDocument) -> str:
    for call in (
        lambda: table.export_to_markdown(doc=doc),
        lambda: table.export_to_markdown(doc),
        lambda: table.export_to_markdown(),
    ):
        try:
            return str(call())
        except Exception:
            pass
    return ""


def substantive_table(table: Any) -> bool:
    data = getattr(table, "data", None)
    return any(
        normalize_space(getattr(cell, "text", ""))
        for cell in list(getattr(data, "table_cells", []) or [])
    )


def _object_contexts(doc: DoclingDocument, wanted_label: str) -> list[dict[str, Any]]:
    stream = list(doc.iterate_items())
    contexts: list[dict[str, Any]] = []
    for position, (item, level) in enumerate(stream):
        if label_value(item) != wanted_label:
            continue

        before_parts: list[str] = []
        cursor = position - 1
        while cursor >= 0 and len(before_parts) < 30:
            candidate, _ = stream[cursor]
            if label_value(candidate) in {"table", "picture"}:
                break
            text = normalize_space(getattr(candidate, "text", ""))
            if text:
                before_parts.append(text)
            cursor -= 1
        before_parts.reverse()

        after_parts: list[str] = []
        cursor = position + 1
        exited_children = wanted_label != "table"
        while cursor < len(stream) and len(after_parts) < 20:
            candidate, candidate_level = stream[cursor]
            if label_value(candidate) in {"table", "picture"}:
                break
            if not exited_children:
                if candidate_level > level:
                    cursor += 1
                    continue
                exited_children = True
            text = normalize_space(getattr(candidate, "text", ""))
            if text:
                after_parts.append(text)
            cursor += 1

        contexts.append(
            {
                "index": len(contexts),
                "item": item,
                "self_ref": str(getattr(item, "self_ref", "")),
                "before": normalize_space(" ".join(before_parts)),
                "after": normalize_space(" ".join(after_parts)),
            }
        )
    return contexts


def table_contexts(doc: DoclingDocument) -> list[dict[str, Any]]:
    contexts = _object_contexts(doc, "table")
    for context in contexts:
        matches = list(TABLE_ID_RE.finditer(context["before"]))
        context["inferred_id"] = (
            f"Table {matches[-1].group('id')}" if matches else ""
        )
        context["markdown"] = table_markdown(context["item"], doc)
        context["substantive"] = substantive_table(context["item"])
    return contexts


def picture_contexts(doc: DoclingDocument) -> list[dict[str, Any]]:
    return _object_contexts(doc, "picture")


def _global_assignment(
    scores: list[tuple[float, int, int]], threshold: float = 20.0
) -> dict[int, tuple[int, float]]:
    assignments: dict[int, tuple[int, float]] = {}
    used_objects: set[int] = set()
    used_records: set[int] = set()
    for score, object_index, record_index in sorted(scores, reverse=True):
        if score < threshold:
            continue
        if object_index in used_objects or record_index in used_records:
            continue
        assignments[object_index] = (record_index, score)
        used_objects.add(object_index)
        used_records.add(record_index)
    return assignments


def match_tables(
    doc: DoclingDocument, llm_tables: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    contexts = table_contexts(doc)
    scores: list[tuple[float, int, int]] = []
    for context_index, context in enumerate(contexts):
        if not context["substantive"]:
            continue
        context_id = normalize_table_id(context["inferred_id"])
        context_text = normalize_match(context["before"])
        table_text = normalize_match(context["markdown"])
        for llm_index, record in enumerate(llm_tables):
            llm_id = normalize_table_id(
                str(record.get("table_id") or record.get("caption_text") or "")
            )
            caption = normalize_match(str(record.get("caption_text") or ""))
            score = (
                difflib.SequenceMatcher(None, caption, context_text).ratio() * 50.0
                if caption and context_text
                else 0.0
            )
            if context_id and llm_id and context_id == llm_id:
                score += 100.0
            anchor = normalize_match(str(record.get("table_start_anchor") or ""))
            if anchor and table_text:
                score += difflib.SequenceMatcher(None, anchor, table_text).ratio() * 30.0
            score -= abs(context["index"] - llm_index) * 2.0
            scores.append((score, context_index, llm_index))

    assignments = _global_assignment(scores)
    return [
        {
            **context,
            "llm_index": assignments.get(index, (None, None))[0],
            "score": assignments.get(index, (None, None))[1],
        }
        for index, context in enumerate(contexts)
    ]


def match_figures(
    doc: DoclingDocument, llm_figures: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    contexts = picture_contexts(doc)
    usable = [
        (index, record)
        for index, record in enumerate(llm_figures)
        if not re.match(
            r"^(?:#+\s*)?(?:\*{1,2})?\s*table\b",
            normalize_space(str(record.get("caption_text") or "")),
            re.IGNORECASE,
        )
    ]
    scores: list[tuple[float, int, int]] = []
    original_indices: dict[int, int] = {}
    for context_index, context in enumerate(contexts):
        surrounding = f"{context['before']} {context['after']}"
        surrounding_norm = normalize_match(surrounding)
        surrounding_ids = {
            normalize_figure_id(match.group(0))
            for match in FIGURE_ID_RE.finditer(surrounding)
        }
        for usable_index, (original_index, record) in enumerate(usable):
            original_indices[usable_index] = original_index
            figure_id = normalize_figure_id(str(record.get("figure_id") or ""))
            caption = normalize_match(str(record.get("caption_text") or ""))
            score = (
                difflib.SequenceMatcher(None, caption, surrounding_norm).ratio() * 70.0
                if caption and surrounding_norm
                else 0.0
            )
            if figure_id and figure_id in surrounding_ids:
                score += 100.0
            score -= abs(context["index"] - usable_index) * 2.0
            scores.append((score, context_index, usable_index))

    usable_assignments = _global_assignment(scores)
    output: list[dict[str, Any]] = []
    for index, context in enumerate(contexts):
        assignment = usable_assignments.get(index)
        output.append(
            {
                **context,
                "llm_index": original_indices[assignment[0]] if assignment else None,
                "score": assignment[1] if assignment else None,
            }
        )
    return output


def patch_caption_cache(
    doc: DoclingDocument, cached: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    patched = json.loads(json.dumps(cached))
    # A few historical caches classified a table image/caption as a figure.
    # Such records cannot be valid figure associations in the structured path.
    patched["figures"] = [
        record
        for record in list(patched.get("figures", []) or [])
        if not re.match(
            r"^(?:#+\s*)?(?:\*{1,2})?\s*table\b",
            normalize_space(str(record.get("caption_text") or "")),
            re.IGNORECASE,
        )
    ]
    table_matches = match_tables(doc, list(patched.get("tables", []) or []))
    figure_matches = match_figures(doc, list(patched.get("figures", []) or []))

    table_audit: list[dict[str, Any]] = []
    for match in table_matches:
        llm_index = match["llm_index"]
        if llm_index is None:
            continue
        rows = [
            line.strip()
            for line in match["markdown"].splitlines()
            if line.strip().startswith("|")
        ]
        data_rows = [
            row for row in rows if not re.match(r"^\|?\s*:?-{3,}", row)
        ]
        if data_rows:
            patched["tables"][llm_index]["table_start_anchor"] = data_rows[0]
            patched["tables"][llm_index]["table_end_anchor"] = data_rows[-1]
        table_audit.append(
            {
                "table_index": match["index"],
                "llm_index": llm_index,
                "score": match["score"],
                "inferred_id": match["inferred_id"],
            }
        )

    figure_audit: list[dict[str, Any]] = []
    for match in figure_matches:
        llm_index = match["llm_index"]
        if llm_index is None:
            continue
        key = f"image_{match['index']:06d}"
        patched["figures"][llm_index]["image_keys"] = [key]
        figure_audit.append(
            {
                "picture_index": match["index"],
                "llm_index": llm_index,
                "score": match["score"],
                "image_key": key,
            }
        )

    patched["_structured_adapter"] = {
        "table_matches": table_audit,
        "figure_matches": figure_audit,
        "note": "Captions/footnotes come from the existing LLM cache; table anchors and image keys were rebuilt from DoclingDocument objects.",
    }
    return patched, patched["_structured_adapter"]


def save_staging_markdown(
    doc: DoclingDocument,
    markdown_path: Path,
    recovered_manifest_path: Path | None = None,
) -> None:
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    doc.save_as_markdown(markdown_path, image_mode=ImageRefMode.REFERENCED)
    if not recovered_manifest_path or not recovered_manifest_path.is_file():
        return

    manifest = json.loads(recovered_manifest_path.read_text(encoding="utf-8"))
    recovery_by_index = {
        int(record["picture_index"]): record
        for record in manifest.get("pictures", [])
        if record.get("recovered_path")
    }
    if not recovery_by_index:
        return

    artifacts_dir = markdown_path.with_name(f"{markdown_path.stem}_artifacts")
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    replacement_links: list[str] = []
    for index, picture in enumerate(doc.pictures):
        if getattr(picture, "image", None) is not None:
            continue
        record = recovery_by_index.get(index)
        if not record:
            replacement_links.append("<!-- image -->")
            continue
        source = recovered_manifest_path.parent / record["recovered_path"]
        destination = artifacts_dir / f"image_{index:06d}_recovered.png"
        shutil.copy2(source, destination)
        replacement_links.append(
            f"![Image]({artifacts_dir.name}/{destination.name})"
        )

    text = markdown_path.read_text(encoding="utf-8")
    for replacement in replacement_links:
        text = text.replace("<!-- image -->", replacement, 1)
    markdown_path.write_text(text, encoding="utf-8")


def docx_image_occurrences(docx_path: Path) -> list[dict[str, str]]:
    occurrences: list[dict[str, str]] = []
    with zipfile.ZipFile(docx_path) as archive:
        rel_root = ET.fromstring(archive.read("word/_rels/document.xml.rels"))
        rel_map: dict[str, str] = {}
        for rel in rel_root.findall(f"{{{REL_NS}}}Relationship"):
            rel_id = rel.attrib.get("Id", "")
            target = rel.attrib.get("Target", "")
            if not rel_id or not target:
                continue
            member = target.lstrip("/") if target.startswith("/") else f"word/{target}"
            rel_map[rel_id] = str(Path(member).as_posix())

        root = ET.fromstring(archive.read("word/document.xml"))
        for element in root.iter():
            local = element.tag.rsplit("}", 1)[-1]
            rel_id = ""
            if local == "blip":
                rel_id = element.attrib.get(f"{{{OFFICE_REL_NS}}}embed", "")
            elif local == "imagedata":
                rel_id = element.attrib.get(f"{{{OFFICE_REL_NS}}}id", "")
            member = rel_map.get(rel_id, "")
            if member.startswith("word/media/"):
                occurrences.append({"relationship_id": rel_id, "member": member})
    return occurrences


def recover_missing_docx_images(
    docx_path: Path, doc: DoclingDocument, output_dir: Path
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_dir = output_dir / "raw"
    converted_dir = output_dir / "converted"
    raw_dir.mkdir(exist_ok=True)
    converted_dir.mkdir(exist_ok=True)
    occurrences = docx_image_occurrences(docx_path)
    missing = [
        index
        for index, picture in enumerate(doc.pictures)
        if getattr(picture, "image", None) is None
    ]
    if len(occurrences) == len(doc.pictures):
        mapping = {index: item for index, item in enumerate(occurrences)}
        mode = "all_picture_items_by_order"
    elif len(occurrences) == len(missing):
        mapping = {index: occurrences[pos] for pos, index in enumerate(missing)}
        mode = "missing_picture_items_by_order"
    else:
        mapping = {}
        mode = "unresolved_count_mismatch"

    records: list[dict[str, Any]] = []
    extracted: list[Path] = []
    with zipfile.ZipFile(docx_path) as archive:
        for index, picture in enumerate(doc.pictures):
            record = {
                "picture_index": index,
                "self_ref": str(getattr(picture, "self_ref", "")),
                "docling_has_image": getattr(picture, "image", None) is not None,
                "status": "already_decoded"
                if getattr(picture, "image", None) is not None
                else "unmapped",
            }
            occurrence = mapping.get(index)
            if occurrence and not record["docling_has_image"]:
                member = occurrence["member"]
                raw_path = raw_dir / f"{index:04d}_{Path(member).name}"
                raw_path.write_bytes(archive.read(member))
                record.update(
                    {
                        "source_member": member,
                        "raw_path": str(raw_path.relative_to(output_dir)),
                        "status": "extracted",
                    }
                )
                extracted.append(raw_path)
            records.append(record)

    vectors = [path for path in extracted if path.suffix.lower() in VECTOR_EXTENSIONS]
    if vectors:
        with tempfile.TemporaryDirectory(prefix="docling_media_profile_") as profile:
            subprocess.run(
                [
                    "soffice",
                    "--headless",
                    "--nologo",
                    "--nofirststartwizard",
                    f"-env:UserInstallation={Path(profile).resolve().as_uri()}",
                    "--convert-to",
                    "png",
                    "--outdir",
                    str(converted_dir),
                    *[str(path) for path in vectors],
                ],
                check=False,
                timeout=600,
            )

    for record in records:
        raw_relative = record.get("raw_path")
        if not raw_relative:
            continue
        raw_path = output_dir / raw_relative
        candidate = converted_dir / f"{raw_path.stem}.png"
        if candidate.is_file():
            record["status"] = "recovered"
            record["recovered_path"] = str(candidate.relative_to(output_dir))

    manifest = {
        "mapping_mode": mode,
        "picture_count": len(doc.pictures),
        "missing_count": len(missing),
        "recovered_count": sum(1 for item in records if item["status"] == "recovered"),
        "pictures": records,
    }
    (output_dir / "recovery_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return manifest
