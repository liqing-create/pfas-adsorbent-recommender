#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import re
import shutil
import tempfile
import traceback
import zipfile
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterator

import pypdfium2 as pdfium
import pypdfium2.raw as pdfium_c
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions, RapidOcrOptions
from docling.document_converter import (
    DocumentConverter,
    PdfFormatOption,
)

from structured_pipeline_common import recover_missing_docx_images


_NARY_PROPERTIES = re.compile(
    rb"(<m:naryPr(?:\s[^>]*)?>)(.*?)(</m:naryPr>)",
    flags=re.DOTALL,
)
_CARDINAL_ORIENTATIONS = (0, 90, 180, 270)
_ORIENTATION_TOLERANCE_DEGREES = 5.0
_MIXED_TABLE_MIN_CHARS = 350
_MIXED_TABLE_MIN_SHARE = 0.12
_MIXED_TABLE_MIN_HEIGHT_SHARE = 0.50
_MIXED_TABLE_MIN_WIDTH_SHARE = 0.10
_MIXED_TABLE_MIN_AREA_SHARE = 0.06
_MIXED_TABLE_MIN_ASPECT_RATIO = 2.0

@dataclass(frozen=True)
class PdfRotationOptions:
    enabled: bool = True
    min_chars: int = 500
    min_share: float = 0.70
    min_dominance_ratio: float = 2.5
    mixed_table_rescue: bool = True


def _angular_distance(left: float, right: float) -> float:
    difference = abs(left - right) % 360
    return min(difference, 360 - difference)


def _nearest_cardinal_orientation(angle: float) -> tuple[int, float]:
    orientation = min(
        _CARDINAL_ORIENTATIONS,
        key=lambda candidate: _angular_distance(angle, candidate),
    )
    return orientation, _angular_distance(angle, orientation)

def _empty_bbox() -> list[float]:
    return [float("inf"), float("inf"), float("-inf"), float("-inf")]


def _update_bbox(bbox: list[float], charbox: tuple[float, float, float, float]) -> None:
    left, bottom, right, top = charbox
    bbox[0] = min(bbox[0], left, right)
    bbox[1] = min(bbox[1], bottom, top)
    bbox[2] = max(bbox[2], left, right)
    bbox[3] = max(bbox[3], bottom, top)


def _bbox_metrics(bbox: list[float], page_width: float, page_height: float) -> dict:
    if not math.isfinite(bbox[0]) or bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
        return {
            "width_share": 0.0,
            "height_share": 0.0,
            "area_share": 0.0,
            "aspect_ratio": 0.0,
        }
    width = bbox[2] - bbox[0]
    height = bbox[3] - bbox[1]
    page_area = page_width * page_height if page_width and page_height else 0.0
    return {
        "width_share": width / page_width if page_width else 0.0,
        "height_share": height / page_height if page_height else 0.0,
        "area_share": (width * height) / page_area if page_area else 0.0,
        "aspect_ratio": height / width if width else float("inf"),
    }


def _mixed_orientation_table_rescue(
    *,
    page_number: int,
    existing_rotation: int,
    counts: dict[int, int],
    bboxes: dict[int, list[float]],
    classified_chars: int,
    dominant_orientation: int,
    page_width: float,
    page_height: float,
    enabled: bool,
) -> dict | None:
    if not enabled or dominant_orientation != 0 or classified_chars <= 0:
        return None

    candidates: list[dict[str, Any]] = []
    for orientation in (90, 180, 270):
        chars = counts[orientation]
        share = chars / classified_chars
        metrics = _bbox_metrics(bboxes[orientation], page_width, page_height)
        if (
            chars >= _MIXED_TABLE_MIN_CHARS
            and share >= _MIXED_TABLE_MIN_SHARE
            and metrics["height_share"] >= _MIXED_TABLE_MIN_HEIGHT_SHARE
            and metrics["width_share"] >= _MIXED_TABLE_MIN_WIDTH_SHARE
            and metrics["area_share"] >= _MIXED_TABLE_MIN_AREA_SHARE
            and metrics["aspect_ratio"] >= _MIXED_TABLE_MIN_ASPECT_RATIO
        ):
            candidates.append(
                {
                    "page": page_number,
                    "existing_rotation": existing_rotation,
                    "orientation": orientation,
                    "clockwise_correction": (-orientation) % 360,
                    "sideways_chars": chars,
                    "sideways_share": round(share, 6),
                }
            )
    return max(candidates, key=lambda item: item["sideways_chars"], default=None)

def _page_orientation_analysis(
    page: pdfium.PdfPage,
    *,
    page_number: int,
    options: PdfRotationOptions,
) -> dict:
    """
    Measure the visual orientation of embedded PDF text on one page.

    PDFium exposes each character's baseline angle separately from the page's
    existing rotation metadata. Combining both gives the orientation Docling's
    layout and table models will see when the page is rendered.
    """
    existing_rotation = int(page.get_rotation() or 0) % 360
    page_width, page_height = page.get_size()
    counts = {orientation: 0 for orientation in _CARDINAL_ORIENTATIONS}
    bboxes = {orientation: _empty_bbox() for orientation in _CARDINAL_ORIENTATIONS}
    unclassified_chars = 0
    text_page = page.get_textpage()
    try:
        for char_index in range(text_page.count_chars()):
            codepoint = int(
                pdfium_c.FPDFText_GetUnicode(text_page.raw, char_index)
            )
            if codepoint <= 0:
                continue
            try:
                character = chr(codepoint)
            except ValueError:
                continue
            if character.isspace():
                continue

            raw_angle = math.degrees(
                pdfium_c.FPDFText_GetCharAngle(text_page.raw, char_index)
            )
            visual_angle = (raw_angle + existing_rotation) % 360
            orientation, distance = _nearest_cardinal_orientation(visual_angle)
            if distance <= _ORIENTATION_TOLERANCE_DEGREES:
                counts[orientation] += 1
                try:
                    _update_bbox(bboxes[orientation], text_page.get_charbox(char_index))
                except Exception:
                    pass                
            else:
                unclassified_chars += 1
    finally:
        text_page.close()

    ranked = sorted(counts.items(), key=lambda item: item[1], reverse=True)
    dominant_orientation, dominant_chars = ranked[0]
    runner_up_chars = ranked[1][1]
    classified_chars = sum(counts.values())
    dominant_share = (
        dominant_chars / classified_chars if classified_chars else 0.0
    )
    dominance_ratio = (
        dominant_chars / runner_up_chars
        if runner_up_chars
        else (float("inf") if dominant_chars else 0.0)
    )

    table_rescue = _mixed_orientation_table_rescue(
        page_number=page_number,
        existing_rotation=existing_rotation,
        counts=counts,
        bboxes=bboxes,
        classified_chars=classified_chars,
        dominant_orientation=dominant_orientation,
        page_width=page_width,
        page_height=page_height,
        enabled=options.mixed_table_rescue,
    )

    correction = (-dominant_orientation) % 360
    reason = "rotate"
    should_rotate = correction != 0
    if classified_chars < options.min_chars:
        reason = "insufficient_text"
        should_rotate = False
    elif dominant_orientation == 0:
        reason = "already_upright"
        should_rotate = False
    elif dominant_share < options.min_share:
        reason = "orientation_share_below_threshold"
        should_rotate = False
    elif dominance_ratio < options.min_dominance_ratio:
        reason = "orientation_dominance_below_threshold"
        should_rotate = False

    return {
        "page": page_number,
        "existing_rotation": existing_rotation,
        "orientation_counts": {
            str(orientation): counts[orientation]
            for orientation in _CARDINAL_ORIENTATIONS
        },
        "unclassified_chars": unclassified_chars,
        "classified_chars": classified_chars,
        "dominant_orientation": dominant_orientation,
        "dominant_share": round(dominant_share, 6),
        "dominance_ratio": (
            None if math.isinf(dominance_ratio) else round(dominance_ratio, 6)
        ),
        "clockwise_correction": correction if should_rotate else 0,
        "suggest_rotation": should_rotate,
        "reason": reason,
        "table_rescue_rotation": table_rescue,
    }


def inspect_pdf_orientations(
    source: Path,
    options: PdfRotationOptions,
) -> dict:
    """Return conservative page-rotation recommendations for one PDF."""
    document = pdfium.PdfDocument(source)
    page_results: list[dict] = []
    try:
        for page_index in range(len(document)):
            page = document[page_index]
            try:
                page_results.append(
                    _page_orientation_analysis(
                        page,
                        page_number=page_index + 1,
                        options=options,
                    )
                )
            finally:
                page.close()
    finally:
        document.close()

    rotations = [
        result for result in page_results if result["suggest_rotation"]
    ]
    table_rescue_pages = [
        result["table_rescue_rotation"]
        for result in page_results
        if result.get("table_rescue_rotation")
    ]    
    review_pages = [
        result
        for result in page_results
        if result["dominant_orientation"] != 0
        and result["reason"]
        in {
            "orientation_share_below_threshold",
            "orientation_dominance_below_threshold",
        }
    ]
    return {
        "enabled": options.enabled,
        "status": "analyzed",
        "pages_scanned": len(page_results),
        "thresholds": {
            "min_chars": options.min_chars,
            "min_share": options.min_share,
            "min_dominance_ratio": options.min_dominance_ratio,
            "orientation_tolerance_degrees": _ORIENTATION_TOLERANCE_DEGREES,
        },
        "rotations": rotations,
        "table_rescue_pages": table_rescue_pages,
        "review_pages": review_pages,
    }


def _print_pdf_orientation_findings(source: Path, summary: dict) -> None:
    for item in summary.get("rotations", []) or []:
        print(
            f"[pdf-orientation] {source.name}: page {item['page']} -> "
            f"rotate {item['clockwise_correction']} degrees clockwise "
            f"(counts={item['orientation_counts']}, "
            f"share={item['dominant_share']:.3f}, "
            f"ratio={item['dominance_ratio']})",
            flush=True,
        )
    for item in summary.get("table_rescue_pages", []) or []:
        print(
            f"[pdf-table-rescue] {source.name}: page {item['page']} -> "
            f"auxiliary rotated conversion by "
            f"{item['clockwise_correction']} degrees clockwise "
            f"(orientation={item['orientation']}, "
           f"chars={item['sideways_chars']}, "
            f"share={item['sideways_share']:.3f})",
            flush=True,
        )        
    for item in summary.get("review_pages", []) or []:
        print(
            f"[pdf-orientation-review] {source.name}: page {item['page']} "
            f"left unchanged ({item['reason']}; "
            f"counts={item['orientation_counts']})",
            flush=True,
        )

def _write_pdf_with_extra_rotations(
    source: Path,
    destination: Path,
    rotations: list[dict],
) -> None:
    document = pdfium.PdfDocument(source)
    try:
        for item in rotations:
            page = document[int(item["page"]) - 1]
            try:
                final_rotation = (
                    int(page.get_rotation() or 0)
                    + int(item["clockwise_correction"])
                ) % 360
                page.set_rotation(final_rotation)
            finally:
                page.close()
        document.save(destination)
    finally:
        document.close()

@contextmanager
def auto_oriented_pdf(
    source: Path,
    options: PdfRotationOptions,
) -> Iterator[tuple[Path, dict]]:
    """
    Yield the original PDF or a temporary copy with strongly indicated page
    rotations written into PDF rotation metadata.

    Analysis or rewrite failures are non-fatal: Docling receives the original
    file, and the failure is preserved in the conversion report.
    """
    if source.suffix.lower() != ".pdf":
        yield source, {
            "enabled": options.enabled,
            "status": "not_pdf",
            "pages_scanned": 0,
            "rotations": [],
            "review_pages": [],
        }
        return
    if not options.enabled:
        yield source, {
            "enabled": False,
            "status": "disabled",
            "pages_scanned": 0,
            "rotations": [],
            "review_pages": [],
        }
        return

    try:
        summary = inspect_pdf_orientations(source, options)
    except Exception as error:
        summary = {
            "enabled": True,
            "status": "analysis_error",
            "pages_scanned": 0,
            "rotations": [],
            "review_pages": [],
            "error_type": type(error).__name__,
            "error": str(error),
        }
        print(
            f"[pdf-orientation-warning] {source.name}: analysis failed; "
            f"using original PDF ({type(error).__name__}: {error})",
            flush=True,
        )
        yield source, summary
        return

    _print_pdf_orientation_findings(source, summary)
    rotations = summary.get("rotations", []) or []
    if not rotations:
        yield source, summary
        return

    with tempfile.TemporaryDirectory(prefix="docling_oriented_pdf_") as temp_dir:
        oriented_source = Path(temp_dir) / source.name
        try:
            _write_pdf_with_extra_rotations(source, oriented_source, rotations)
        except Exception as error:
            summary["status"] = "rewrite_error"
            summary["error_type"] = type(error).__name__
            summary["error"] = str(error)
            print(
                f"[pdf-orientation-warning] {source.name}: rewrite failed; "
                f"using original PDF ({type(error).__name__}: {error})",
                flush=True,
            )
            yield source, summary
            return

        summary["status"] = "rotated_temporary_copy"
        yield oriented_source, summary


def _add_missing_nary_operators(document_xml: bytes) -> tuple[bytes, int]:
    """Make Word's implicit integral operator explicit for Docling's OMML parser."""
    repairs = 0

    def replace(match: re.Match[bytes]) -> bytes:
        nonlocal repairs
        body = match.group(2)
        if re.search(rb"<m:chr(?:\s|/|>)", body):
            return match.group(0)
        repairs += 1
        return (
            match.group(1)
            + b'<m:chr m:val="&#x222B;"/>'
            + body
            + match.group(3)
        )

    return _NARY_PROPERTIES.sub(replace, document_xml), repairs


def _write_docx_with_omml_repairs(source: Path, destination: Path) -> int:
    repairs = 0
    with zipfile.ZipFile(source, "r") as input_zip:
        document_xml = input_zip.read("word/document.xml")
        repaired_xml, repairs = _add_missing_nary_operators(document_xml)
        if repairs == 0:
            return 0

        with zipfile.ZipFile(destination, "w") as output_zip:
            for entry in input_zip.infolist():
                contents = input_zip.read(entry.filename)
                if entry.filename == "word/document.xml":
                    contents = repaired_xml
                output_zip.writestr(entry, contents)
    return repairs


@contextmanager
def docling_compatible_source(source: Path) -> Iterator[tuple[Path, int]]:
    """Yield the original source or a temporary, minimally repaired DOCX."""
    if source.suffix.lower() != ".docx":
        yield source, 0
        return

    with tempfile.TemporaryDirectory(prefix="docling_docx_compat_") as temp_dir:
        repaired_source = Path(temp_dir) / source.name
        repairs = _write_docx_with_omml_repairs(source, repaired_source)
        if repairs:
            yield repaired_source, repairs
        else:
            yield source, 0

def _table_page(table: dict) -> int | None:
    prov = table.get("prov") or []
    if not prov:
        return None
    try:
        return int(prov[0].get("page_no"))
    except Exception:
        return None


def _table_bbox_aspect(table: dict) -> float:
    prov = table.get("prov") or []
    bbox = (prov[0].get("bbox") if prov else {}) or {}
    try:
        width = abs(float(bbox.get("r", 0.0)) - float(bbox.get("l", 0.0)))
        height = abs(float(bbox.get("t", 0.0)) - float(bbox.get("b", 0.0)))
    except Exception:
        return 0.0
    return height / width if width else float("inf")


def _nonempty_table_cells(table: dict) -> int:
    cells = table.get("data", {}).get("table_cells", []) or []
    return sum(1 for cell in cells if str(cell.get("text") or "").strip())


def _table_quality(table: dict) -> tuple[int, int, int, float]:
    data = table.get("data", {}) or {}
    return (
        _nonempty_table_cells(table),
        int(data.get("num_rows") or 0),
        int(data.get("num_cols") or 0),
        -_table_bbox_aspect(table),
    )


def _replace_table_preserving_identity(original: dict, replacement: dict) -> dict:
    merged = json.loads(json.dumps(replacement))
    for key in ("self_ref", "parent", "children", "captions", "references", "footnotes"):
        if key in original:
            merged[key] = original[key]
    return merged


def _rescue_mixed_orientation_tables(
    *,
    source: Path,
    pdf_orientation: dict,
    json_path: Path,
) -> dict | None:
    rescue_rotations = pdf_orientation.get("table_rescue_pages") or []
    if not rescue_rotations:
        return None

    rescue_pages = {int(item["page"]) for item in rescue_rotations}
    original_json = json.loads(json_path.read_text(encoding="utf-8"))
    original_tables = list(original_json.get("tables") or [])
    targets_by_page: dict[int, list[int]] = {page: [] for page in rescue_pages}
    for index, table in enumerate(original_tables):
        page = _table_page(table)
        if page in rescue_pages and _table_bbox_aspect(table) >= _MIXED_TABLE_MIN_ASPECT_RATIO:
            targets_by_page[page].append(index)

    if not any(targets_by_page.values()):
        return {"pages": sorted(rescue_pages), "replaced_tables": 0}

    with tempfile.TemporaryDirectory(prefix="docling_table_rescue_") as temp_dir:
        temp_dir_path = Path(temp_dir)
        rotated_source = temp_dir_path / source.name
        auxiliary_json = temp_dir_path / "auxiliary.docling.json"
        _write_pdf_with_extra_rotations(source, rotated_source, rescue_rotations)
        auxiliary_doc = converter_for(rotated_source).convert(source=rotated_source).document
        auxiliary_doc.save_as_json(auxiliary_json)
        auxiliary_tables = json.loads(
            auxiliary_json.read_text(encoding="utf-8")
        ).get("tables", []) or []

    replaced = 0
    for page, target_indices in targets_by_page.items():
        candidates = [table for table in auxiliary_tables if _table_page(table) == page]
        candidates.sort(key=_table_quality, reverse=True)
        for target_index, candidate in zip(target_indices, candidates):
            if _nonempty_table_cells(candidate) < 10:
                continue
            original_tables[target_index] = _replace_table_preserving_identity(
                original_tables[target_index], candidate
            )
            replaced += 1

    if replaced:
        original_json["tables"] = original_tables
        json_path.write_text(
            json.dumps(original_json, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
    return {"pages": sorted(rescue_pages), "replaced_tables": replaced}

def converter_for(path: Path) -> DocumentConverter:
    if path.suffix.lower() != ".pdf":
        return DocumentConverter()
    options = PdfPipelineOptions()
    options.images_scale = 3.0
    options.generate_page_images = False
    options.generate_picture_images = True
    options.do_ocr = True
    options.ocr_options = RapidOcrOptions(lang=["eng"])
    # Formula enrichment can become pathologically slow on misdetected equations.
    # Equations are not a primary data source in this pipeline.
    options.do_formula_enrichment = False
    return DocumentConverter(
        format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=options)}
    )


def backup_existing(path: Path) -> None:
    if not path.exists():
        return
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup = path.with_name(f"{path.name}.bak_{timestamp}")
    if path.is_dir():
        shutil.copytree(path, backup)
    else:
        shutil.copy2(path, backup)


def convert_one(
    source: Path,
    force: bool,
    rotation_options: PdfRotationOptions,
) -> dict:
    destination = source.with_suffix(".docling.json")
    if destination.exists() and not force:
        return {"source": str(source), "output": str(destination), "status": "skipped"}
    if force:
        backup_existing(destination)
    table_rescue = None
    temporary = destination.with_suffix(".docling.json.tmp")
    with docling_compatible_source(source) as (compatible_source, omml_repairs):
        if omml_repairs:
            print(
                f"[docx-compat] {source.name}: made {omml_repairs} implicit "
                "integral operator(s) explicit in a temporary copy",
                flush=True,
            )
        with auto_oriented_pdf(
            compatible_source, rotation_options
        ) as (conversion_source, pdf_orientation):
            result = converter_for(conversion_source).convert(
                source=conversion_source
            )
            doc = result.document
            doc.save_as_json(temporary)
            if source.suffix.lower() == ".pdf":
                try:
                    table_rescue = _rescue_mixed_orientation_tables(
                        source=conversion_source,
                        pdf_orientation=pdf_orientation,
                        json_path=temporary,
                    )
                except Exception as error:
                    table_rescue = {
                        "pages": [
                            int(item["page"])
                            for item in pdf_orientation.get("table_rescue_pages", []) or []
                        ],
                        "replaced_tables": 0,
                        "error_type": type(error).__name__,
                        "error": str(error),
                    }
                    print(
                        f"[pdf-table-rescue-warning] {source.name}: "
                        f"auxiliary conversion failed ({type(error).__name__}: {error})",
                        flush=True,
                    )
    output_json = json.loads(temporary.read_text(encoding="utf-8")) 
    temporary.replace(destination)

    recovery = None
    if source.suffix.lower() == ".docx":
        recovery_dir = source.parent / f"{source.stem}_recovered_media"
        if force:
            backup_existing(recovery_dir)
        recovery = recover_missing_docx_images(source, doc, recovery_dir)

    return {
        "source": str(source),
        "output": str(destination),
        "status": "converted",
        "tables": len(output_json.get("tables", [])),
        "pictures": len(output_json.get("pictures", [])),        
        "omml_repairs": omml_repairs,
        "pdf_orientation": pdf_orientation,
        "table_rescue": table_rescue,
        "recovery": recovery,
    }


def rotation_report_record(
    source: Path,
    rotation_options: PdfRotationOptions,
) -> dict:
    if source.suffix.lower() != ".pdf":
        return {
            "source": str(source),
            "status": "rotation_report_skipped_non_pdf",
        }
    if not rotation_options.enabled:
        return {
            "source": str(source),
            "status": "rotation_report_disabled",
        }

    summary = inspect_pdf_orientations(source, rotation_options)
    _print_pdf_orientation_findings(source, summary)
    return {
        "source": str(source),
        "status": "rotation_report",
        "pdf_orientation": summary,
    }


def document_candidates(study_dir: Path) -> list[Path]:
    candidates: list[Path] = []
    main_pdf = study_dir / "main_paper.pdf"
    if main_pdf.is_file():
        candidates.append(main_pdf)

    supplementary_docx = sorted(study_dir.glob("supplementary*.docx"))
    if supplementary_docx:
        candidates.extend(supplementary_docx)
    else:
        supplementary_pdf = study_dir / "supplementary_material.pdf"
        if supplementary_pdf.is_file():
            candidates.append(supplementary_pdf)

    for name in ("supplementary_material.xlsx", "supplementary_table.xlsx"):
        path = study_dir / name
        if path.is_file():
            candidates.append(path)
    return candidates


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--studies", nargs="+", required=True)
    parser.add_argument(
        "--force",
        action="store_true",
        help="Back up and regenerate every existing output included in this run.",
    )
    parser.add_argument(
        "--force-study",
        action="append",
        default=[],
        help="Back up and regenerate one study. May be supplied more than once.",
    )
    parser.add_argument(
        "--auto-rotate-pdf-pages",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Before Docling conversion, use embedded PDF character angles to "
            "rotate pages whose text is strongly dominated by a non-upright "
            "orientation. Uses Docling's existing pypdfium2 dependency."
        ),
    )
    parser.add_argument(
        "--rotation-min-chars",
        type=int,
        default=500,
        help="Minimum classified non-whitespace characters required to rotate.",
    )
    parser.add_argument(
        "--rotation-min-share",
        type=float,
        default=0.70,
        help="Minimum share of classified characters in the dominant orientation.",
    )
    parser.add_argument(
        "--rotation-min-dominance-ratio",
        type=float,
        default=2.5,
        help="Minimum dominant-orientation count divided by the runner-up count.",
    )
    parser.add_argument(
        "--mixed-orientation-table-rescue",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "For mixed-orientation PDF pages, run an auxiliary rotated "
            "conversion and replace tall/narrow table objects on that page."
        ),
    )
    parser.add_argument(    
        "--rotation-report-only",
        action="store_true",
        help=(
            "Analyze PDF page orientations and write a report without running "
            "Docling or modifying existing outputs."
        ),
    )
    args = parser.parse_args()
    if args.rotation_min_chars < 1:
        parser.error("--rotation-min-chars must be at least 1")
    if not 0 < args.rotation_min_share <= 1:
        parser.error("--rotation-min-share must be in the interval (0, 1]")
    if args.rotation_min_dominance_ratio < 1:
        parser.error("--rotation-min-dominance-ratio must be at least 1")

    rotation_options = PdfRotationOptions(
        enabled=bool(args.auto_rotate_pdf_pages),
        min_chars=args.rotation_min_chars,
        min_share=args.rotation_min_share,
        min_dominance_ratio=args.rotation_min_dominance_ratio,
        mixed_table_rescue=bool(args.mixed_orientation_table_rescue),
    )

    force_studies = set(args.force_study)
    records = []
    error_count = 0
    for study in args.studies:
        study_dir = args.data_root / study
        if not study_dir.is_dir():
            records.append({"study": study, "status": "missing_study"})
            continue

        force_study = args.force or study in force_studies
        for source in document_candidates(study_dir):
            print(f"[check] {study}: {source.name}", flush=True)
            try:
                if args.rotation_report_only:
                    record = rotation_report_record(source, rotation_options)
                else:
                    record = convert_one(
                        source,
                        force_study,
                        rotation_options,
                    )
            except Exception as error:
                error_count += 1
                record = {
                    "source": str(source),
                    "output": str(source.with_suffix(".docling.json")),
                    "status": "error",
                    "error_type": type(error).__name__,
                    "error": str(error),
                }
                print(
                    f"[conversion-error] {study}: {source.name}: "
                    f"{type(error).__name__}: {error}",
                    flush=True,
                )
                traceback.print_exc()

            record["study"] = study
            records.append(record)

            if record["status"] == "skipped":
                print(f"[skip-existing] {record['output']}", flush=True)
            elif record["status"] == "converted":
                print(
                    f"[converted] {record['output']} "
                    f"(tables={record['tables']}, pictures={record['pictures']})",
                    flush=True,
                )

    report_name = (
        "pdf_rotation_report.json"
        if args.rotation_report_only
        else "docling_json_conversion_report.json"
    )
    report = args.data_root / "_logs" / report_name
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(json.dumps(records, indent=2, ensure_ascii=False), encoding="utf-8")
    if error_count:
        print(
            f"[completed-with-errors] {error_count} file(s) failed; "
            f"details are in {report}",
            flush=True,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
