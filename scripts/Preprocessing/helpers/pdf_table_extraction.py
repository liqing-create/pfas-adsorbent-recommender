from __future__ import annotations

import re
from pathlib import Path
from statistics import median
from typing import Any, Dict, List, Optional, Tuple

try:
    import pandas as pd
except Exception:  # pragma: no cover
    pd = None  # type: ignore

try:  # Package execution
    from .table_repair_core import (
        docling_header_from_markdown,
        escape_markdown_cell as _escape_md,
        grid_to_markdown,
        markdown_block_with_caption,
        normalize_markdown_cell,
    )
except ImportError:  # Direct helper execution/import
    from table_repair_core import (
        docling_header_from_markdown,
        escape_markdown_cell as _escape_md,
        grid_to_markdown,
        markdown_block_with_caption,
        normalize_markdown_cell,
    )


def sanitize_fragment(value: str) -> str:
    value = re.sub(r"[<>:\"/\\|?*]+", "_", value or "")
    value = re.sub(r"\s+", "_", value.strip())
    return value.strip("_") or "table"


def bbox_to_camelot_area(bbox: Dict[str, Any], pad: float = 3.0) -> str:
    """
    Convert a Docling table bbox to Camelot's table_areas format.

    Camelot expects x1,y1,x2,y2 where x1/y1 is top-left and x2/y2 is
    bottom-right, in PDF coordinates with origin at bottom-left. Docling bbox
    x coordinates are compatible; for the PDFs in this project Docling also
    stores BOTTOMLEFT y coordinates.
    """
    l = float(bbox["l"]) - pad
    t = float(bbox["t"]) + pad
    r = float(bbox["r"]) + pad
    b = float(bbox["b"]) - pad
    return f"{max(0.0, l):.2f},{t:.2f},{r:.2f},{max(0.0, b):.2f}"

def docling_bbox_to_pymupdf_clip(
    bbox: Optional[Dict[str, Any]],
    page_rect: Any,
    *,
    pad: float = 3.0,
    fitz_module: Any = None,
) -> Optional[Any]:
    """
    Convert a Docling table bbox into a PyMuPDF clip rectangle.

    Docling PDF bboxes in this pipeline normally use BOTTOMLEFT coordinates.
    PyMuPDF page coordinates use a top-left origin. The returned Rect limits
    Page.find_tables() to the known table region so text-strategy extraction
    does not absorb the rest of the page.
    """
    if not bbox or not all(k in bbox for k in ("l", "t", "r", "b")):
        return None
    if fitz_module is None:
        try:
            import fitz as fitz_module  # type: ignore
        except Exception:
            return None

    try:
        l = float(bbox["l"])
        t = float(bbox["t"])
        r = float(bbox["r"])
        b = float(bbox["b"])
        page_x0 = float(page_rect.x0)
        page_y0 = float(page_rect.y0)
        page_x1 = float(page_rect.x1)
        page_y1 = float(page_rect.y1)
    except Exception:
        return None

    origin = str(bbox.get("coord_origin") or "BOTTOMLEFT").upper()
    page_height = page_y1 - page_y0

    if origin == "BOTTOMLEFT":
        x0 = l - pad
        x1 = r + pad
        y0 = page_y0 + page_height - (t + pad)
        y1 = page_y0 + page_height - (b - pad)
    else:
        # Fallback for TOPLEFT-like bboxes.
        x0 = l - pad
        x1 = r + pad
        y0 = t - pad
        y1 = b + pad

    x0 = max(page_x0, min(page_x1, x0))
    x1 = max(page_x0, min(page_x1, x1))
    y0 = max(page_y0, min(page_y1, y0))
    y1 = max(page_y0, min(page_y1, y1))

    if x1 <= x0 or y1 <= y0:
        return None
    return fitz_module.Rect(x0, y0, x1, y1)

def derive_docling_column_separators(
    table_obj: Dict[str, Any],
    *,
    min_gap: float = 2.0,
) -> Optional[str]:
    """
    Derive Camelot stream column separators from Docling table cell bboxes.

    Camelot expects only internal vertical separators. For a 7-column table,
    this returns six coordinates. Use only single-column cells because spanning
    cells do not define reliable internal boundaries.
    """
    data = table_obj.get("data") or {}
    try:
        num_cols = int(data.get("num_cols") or 0)
    except Exception:
        num_cols = 0
    if num_cols <= 1:
        return None

    col_lefts: List[List[float]] = [[] for _ in range(num_cols)]
    col_rights: List[List[float]] = [[] for _ in range(num_cols)]

    for cell in data.get("table_cells") or []:
        try:
            start_col = int(cell.get("start_col_offset_idx"))
            end_col = int(cell.get("end_col_offset_idx"))
        except Exception:
            continue
        if end_col != start_col + 1:
            continue
        if start_col < 0 or start_col >= num_cols:
            continue
        bbox = cell.get("bbox") or {}
        l = bbox.get("l")
        r = bbox.get("r")
        if not isinstance(l, (int, float)) or not isinstance(r, (int, float)):
            continue
        l = float(l)
        r = float(r)
        if r <= l:
            continue
        col_lefts[start_col].append(l)
        col_rights[start_col].append(r)

    separators: List[float] = []
    for col_idx in range(num_cols - 1):
        if not col_rights[col_idx] or not col_lefts[col_idx + 1]:
            return None
        left_col_right_edge = max(col_rights[col_idx])
        right_col_left_edge = min(col_lefts[col_idx + 1])
        gap = right_col_left_edge - left_col_right_edge
        if gap < min_gap:
            return None
        separators.append((left_col_right_edge + right_col_left_edge) / 2.0)

    return ",".join(f"{x:.2f}" for x in separators)


def derive_robust_docling_column_separators(table_obj: Dict[str, Any]) -> Optional[str]:
    """
    Derive Camelot stream column separators from median Docling cell centers.

    This is less strict than derive_docling_column_separators(): it tolerates
    slight bbox overlaps and outlier cells, which are common in Docling output
    for dense PDF tables.
    """
    data = table_obj.get("data") or {}
    try:
        num_cols = int(data.get("num_cols") or 0)
    except Exception:
        num_cols = 0
    if num_cols <= 1:
        return None

    centers: List[List[float]] = [[] for _ in range(num_cols)]
    lefts: List[List[float]] = [[] for _ in range(num_cols)]
    rights: List[List[float]] = [[] for _ in range(num_cols)]

    for cell in data.get("table_cells") or []:
        try:
            start_col = int(cell.get("start_col_offset_idx"))
            end_col = int(cell.get("end_col_offset_idx"))
        except Exception:
            continue
        if end_col != start_col + 1:
            continue
        if start_col < 0 or start_col >= num_cols:
            continue
        bbox = cell.get("bbox") or {}
        l = bbox.get("l")
        r = bbox.get("r")
        if not isinstance(l, (int, float)) or not isinstance(r, (int, float)):
            continue
        l = float(l)
        r = float(r)
        if r <= l:
            continue
        lefts[start_col].append(l)
        rights[start_col].append(r)
        centers[start_col].append((l + r) / 2.0)

    separators: List[float] = []
    for col_idx in range(num_cols - 1):
        sep: Optional[float] = None
        if centers[col_idx] and centers[col_idx + 1]:
            sep = (median(centers[col_idx]) + median(centers[col_idx + 1])) / 2.0
        elif rights[col_idx] and lefts[col_idx + 1]:
            sep = (median(rights[col_idx]) + median(lefts[col_idx + 1])) / 2.0
        if sep is None:
            return None
        separators.append(sep)

    return ",".join(f"{x:.2f}" for x in separators)


def clean_dataframe(df: "pd.DataFrame") -> "pd.DataFrame":
    if pd is None:
        raise RuntimeError("pandas is required for Camelot table repair")
    if df is None:
        return pd.DataFrame()
    out = df.copy()
    out = out.map(
        lambda x: re.sub(
            r"\s+",
            " ",
            str(x or "").replace("\r", " ").replace("\n", " "),
        ).strip()
    )
    out = out.replace("", pd.NA)
    out = out.dropna(axis=0, how="all")
    out = out.dropna(axis=1, how="all")
    out = out.fillna("").reset_index(drop=True)
    out.columns = list(range(out.shape[1]))
    return out


def nonempty_cell_count(df: "pd.DataFrame") -> int:
    if df is None or df.empty:
        return 0
    return int(df.astype(str).map(lambda x: bool(str(x).strip())).sum().sum())


def _df_to_grid(df: "pd.DataFrame") -> List[List[str]]:
    df = clean_dataframe(df)
    if df.empty:
        return []
    return df.astype(str).values.tolist()


def dataframe_to_markdown(df: "pd.DataFrame") -> str:
    rows = _df_to_grid(df)
    if not rows:
        return ""
    n_cols = max(len(row) for row in rows)
    first_row = list(rows[0]) + [""] * (n_cols - len(rows[0]))
    lines = [
        "| " + " | ".join(_escape_md(x) for x in first_row[:n_cols]) + " |",
        "| " + " | ".join("---" for _ in range(n_cols)) + " |",
    ]
    for row in rows[1:]:
        row = list(row) + [""] * (n_cols - len(row))
        lines.append("| " + " | ".join(_escape_md(x) for x in row[:n_cols]) + " |")
    return "\n".join(lines).rstrip() + "\n"





_EMBEDDED_CAPTION_CONTEXT_PREFIX_PAT = (
    r"(?:(?:Supplementary|Supplemental|Supporting|Appendix)\s+)?"
)
_EMBEDDED_CAPTION_SUPP_NUM_PREFIX_PAT = r"(?:SM|SI|SF|ST|S|A)"
_EMBEDDED_CAPTION_ROMAN_NUM_PAT = r"[IVXLCDM]+\b"
_EMBEDDED_CAPTION_NUM_PAT = (
    rf"(?:{_EMBEDDED_CAPTION_SUPP_NUM_PREFIX_PAT}\s*[.\-]?\s*\d+|"
    rf"\d+S\b|\d+(?:[._\-]\d+)?|{_EMBEDDED_CAPTION_ROMAN_NUM_PAT})"
)
_EMBEDDED_CAPTION_LABEL_NUM_SEP_PAT = (
    r"(?:\s+|\s*[-\u2013\u2014]\s*|"
    rf"(?={_EMBEDDED_CAPTION_SUPP_NUM_PREFIX_PAT}\s*[.\-]?\s*\d+|\d+))"
)
_EMBEDDED_CAPTION_LINE_RE = re.compile(
    rf"""^\s*
        (?:\#{{1,6}}\s*)?
        (?:\d{{1,6}}\s+)?
        {_EMBEDDED_CAPTION_CONTEXT_PREFIX_PAT}
        (?P<kind>Table|Tab\.?|Tabel|Exhibit)
        \.?
        {_EMBEDDED_CAPTION_LABEL_NUM_SEP_PAT}
        (?P<num>{_EMBEDDED_CAPTION_NUM_PAT})
        \s*\.?\s*
        (?P<delim>[:|]|[-\u2013\u2014])?
        \s*
        (?P<desc>.*\S.*|)
        \s*$
    """,
    re.IGNORECASE | re.VERBOSE,
)
_EMBEDDED_CAPTION_LABEL_RE = re.compile(
    r"\b(?:Table|Tab\.?|Tabel|Exhibit)\s*\.?\s*"
    rf"(?:{_EMBEDDED_CAPTION_NUM_PAT})\b",
    re.IGNORECASE,
)
_EMBEDDED_CAPTION_PROSE_START_RE = re.compile(
    r"^(?:shows?|lists?|summari[sz]es?|presents?|reports?|contains?|provides?|"
    r"illustrates?|depicts?|compares?|demonstrates?|describes?|represents?)\b",
    re.IGNORECASE,
)
_EMBEDDED_CAPTION_MAX_CHARS = 2_000


def _normalize_embedded_caption_text(text: Any) -> str:
    return re.sub(
        r"\s+",
        " ",
        str(text or "").replace("\r", " ").replace("\n", " "),
    ).strip()


def _bounded_embedded_caption_candidate(text: Any) -> str:
    """Return one complete caption-like line, subject to caption-recognition bounds."""
    candidate = _normalize_embedded_caption_text(text)
    if not candidate or len(candidate) > _EMBEDDED_CAPTION_MAX_CHARS:
        return ""
    match = _EMBEDDED_CAPTION_LINE_RE.match(candidate)
    if not match:
        return ""

    # The candidate must contain exactly one table label. This prevents a
    # broad Camelot row or an overlapping PDF text layer from combining two
    # captions and then accepting whichever label appears first.
    if len(list(_EMBEDDED_CAPTION_LABEL_RE.finditer(candidate))) != 1:
        return ""

    desc = (match.group("desc") or "").strip()
    has_printed_delim = bool((match.group("delim") or "").strip())
    if not has_printed_delim:
        try:
            has_printed_delim = candidate[match.end("num") :].lstrip().startswith(".")
        except Exception:
            has_printed_delim = False
    if not has_printed_delim and _EMBEDDED_CAPTION_PROSE_START_RE.match(desc):
        return ""
    return candidate


def _unique_bounded_embedded_caption_candidates(texts: List[Any]) -> List[str]:
    candidates: List[str] = []
    seen: set[str] = set()
    for text in texts:
        candidate = _bounded_embedded_caption_candidate(text)
        if not candidate or candidate in seen:
            continue
        seen.add(candidate)
        candidates.append(candidate)
    return candidates


def embedded_caption_from_dataframe(df: "pd.DataFrame", *, max_rows: int = 5) -> str:
    """Return one bounded caption candidate from the table's early rows."""
    row_texts = [
        " ".join(cell for cell in row if cell)
        for row in _df_to_grid(df)[:max_rows]
    ]
    candidates = _unique_bounded_embedded_caption_candidates(row_texts)
    if len(candidates) != 1:
        return ""
    return candidates[0]


def embedded_caption_from_camelot_table(table: Any, *, max_lines: int = 20) -> str:
    """Return one bounded caption candidate from Camelot text lines."""
    textlines = list(getattr(table, "textlines", []) or [])
    candidates: List[str] = []
    try:
        textlines = sorted(textlines, key=lambda line: float(getattr(line, "y1", 0.0)), reverse=True)
    except Exception:
        pass
    for line in textlines[:max_lines]:
        try:
            text = line.get_text()
        except Exception:
            text = str(line or "")
        candidate = _bounded_embedded_caption_candidate(text)
        if candidate:
            candidates.append(candidate)
    unique_candidates = _unique_bounded_embedded_caption_candidates(candidates)
    if len(unique_candidates) == 1:
        return unique_candidates[0]
    return ""


def _best_camelot_dataframe(tables: Any) -> Tuple[Optional["pd.DataFrame"], int, Optional[int]]:
    best_df = None
    best_nonempty = -1
    best_idx = None
    for idx, table in enumerate(tables, start=1):
        df = clean_dataframe(table.df)
        n = nonempty_cell_count(df)
        if n > best_nonempty:
            best_df = df
            best_nonempty = n
            best_idx = idx
    return best_df, best_nonempty, best_idx


def extract_camelot_embedded_caption(
    *,
    pdf_path: str | Path,
    table_obj: Dict[str, Any],
    bbox_pad: float = 3.0,
    caption_top_pad: float = 85.0,
    min_nonempty_cells: int = 3,
) -> Dict[str, Any]:
    """
    Extract one bounded embedded-caption candidate around a Docling table region.

    Candidates must pass the same whole-line and size safeguards used by the
    structured caption recognizer. A caption found in the extracted table rows
    takes precedence over Camelot text lines because PDF text lines may include
    overlapping hidden or nearby captions.
    """
    try:
        import camelot
    except Exception as exc:
        return {"status": "failed", "error": f"Camelot import failed: {exc}"}

    prov = (table_obj.get("prov") or [{}])[0]
    try:
        page_no = int(prov.get("page_no") or 0)
    except Exception:
        page_no = 0
    bbox = prov.get("bbox") or {}
    if not page_no or not all(k in bbox for k in ("l", "t", "r", "b")):
        return {"status": "not_attempted", "attempted": False, "reason": "Docling page/bbox provenance missing"}

    area = bbox_to_camelot_area(bbox, pad=bbox_pad)
    try:
        parts = [float(x) for x in area.split(",")]
        parts[1] = parts[1] + float(caption_top_pad)
        area = ",".join(f"{max(0.0, x):.2f}" for x in parts)
    except Exception:
        pass

    attempts = []
    # Lattice is especially useful when a caption is printed inside the ruled
    # table. Stream remains the fallback for tables without reliable ruling.
    for flavor in ("lattice", "stream"):
        try:
            tables = camelot.read_pdf(
                str(pdf_path),
                pages=str(page_no),
                flavor=flavor,
                table_areas=[area],
                split_text=True,
                strip_text="\n",
                suppress_stdout=True,
            )
        except Exception as exc:
            attempts.append({"flavor": flavor, "status": "failed", "error": str(exc)})
            continue

        best_df = None
        best_nonempty = -1
        best_idx = None
        best_table = None
        for idx, table in enumerate(tables, start=1):
            df = clean_dataframe(table.df)
            n = nonempty_cell_count(df)
            if n > best_nonempty:
                best_df = df
                best_nonempty = n
                best_idx = idx
                best_table = table
        if best_df is None:
            attempts.append({
                "flavor": flavor,
                "status": "skipped",
                "tables_found": int(getattr(tables, "n", 0)),
                "best_nonempty_cells": best_nonempty,
            })
            continue

        # Prefer the candidate found in Camelot's extracted table rows. Text
        # lines may include overlapping/hidden PDF layers or nearby captions;
        # the table-row candidate is bounded to the extracted table itself.
        dataframe_caption = embedded_caption_from_dataframe(best_df)
        textline_caption = embedded_caption_from_camelot_table(best_table)
        caption = dataframe_caption or textline_caption
        if caption:
            return {
                "status": "extracted",
                "strategy": flavor,
                "caption_candidate": caption,
                "raw_block": dataframe_to_markdown(best_df),
                "pdf_path": str(pdf_path),
                "page_no": page_no,
                "camelot_area": area,
                "candidate_index": best_idx,
                "shape": [int(best_df.shape[0]), int(best_df.shape[1])],
                "nonempty_cells": best_nonempty,
                "attempts": attempts,
            }
        if best_nonempty < min_nonempty_cells:
            attempts.append({
                "flavor": flavor,
                "status": "skipped",
                "reason": "Camelot table did not meet the density threshold and contained no caption",
                "tables_found": int(getattr(tables, "n", 0)),
                "best_nonempty_cells": best_nonempty,
            })
            continue
        attempts.append({
            "flavor": flavor,
            "status": "not_found",
            "tables_found": int(getattr(tables, "n", 0)),
            "best_nonempty_cells": best_nonempty,
        })

    return {
        "status": "not_found" if attempts and any(a.get("status") == "not_found" for a in attempts) else "failed",
        "reason": "Camelot did not recover an embedded caption",
        "pdf_path": str(pdf_path),
        "page_no": page_no,
        "camelot_area": area,
        "attempts": attempts,
    }




def dataframe_to_markdown_with_header(
    df: "pd.DataFrame",
    header: List[str],
    *,
    caption_text: str = "",
    footnote_text: str = "",
) -> str:
    if pd is None:
        raise RuntimeError("pandas is required for Camelot table repair")
    df = clean_dataframe(df)
    header = [str(h or "").strip() for h in (header or [])]
    if df.empty or not header:
        return ""
    n_cols = len(header)
    if df.shape[1] != n_cols:
        return ""

    # Camelot generally includes the visual header as row 0. Replace it with
    # Docling's cleaner canonical header and keep Camelot's body rows.
    body_rows = df.iloc[1:].astype(str).values.tolist()
    if not body_rows:
        return ""

    lines: List[str] = []
    if caption_text.strip():
        lines.append(caption_text.strip())
        lines.append("")
    lines.append("| " + " | ".join(_escape_md(x) for x in header) + " |")
    lines.append("| " + " | ".join("---" for _ in header) + " |")
    for row in body_rows:
        row = list(row) + [""] * (n_cols - len(row))
        row = row[:n_cols]
        lines.append("| " + " | ".join(_escape_md(x) for x in row) + " |")
    if footnote_text.strip():
        lines.append("")
        lines.append(footnote_text.strip())
    return "\n".join(lines).rstrip() + "\n"


def _repair_docling_table_with_camelot_lattice(
    *,
    camelot_module: Any,
    pdf_path: str | Path,
    page_no: int,
    area: str,
    bbox: Dict[str, Any],
    caption_text: str,
    footnote_text: str,
    min_nonempty_cells: int,
) -> Dict[str, Any]:
    try:
        tables = camelot_module.read_pdf(
            str(pdf_path),
            pages=str(page_no),
            flavor="lattice",
            table_areas=[area],
            split_text=True,
            strip_text="\n",
            suppress_stdout=True,
        )
    except Exception as exc:
        return {
            "status": "failed",
            "attempted": True,
            "strategy": "lattice",
            "error": f"Camelot lattice extraction failed: {exc}",
            "pdf_path": str(pdf_path),
            "page_no": page_no,
            "camelot_area": area,
        }

    best_df, best_nonempty, best_idx = _best_camelot_dataframe(tables)
    if best_df is None or best_nonempty < min_nonempty_cells:
        return {
            "status": "attempted_no_output",
            "attempted": True,
            "strategy": "lattice",
            "reason": "Camelot lattice returned no usable table",
            "tables_found": int(getattr(tables, "n", 0)),
            "best_nonempty_cells": best_nonempty,
        }

    raw_block = dataframe_to_markdown(best_df)
    block = markdown_block_with_caption(
        raw_block,
        caption_text=caption_text,
        footnote_text=footnote_text,
    )
    return {
        "status": "repaired",
        "attempted": True,
        "strategy": "lattice",
        "block": block,
        "pdf_path": str(pdf_path),
        "page_no": page_no,
        "bbox": bbox,
        "camelot_area": area,
        "caption_candidate": embedded_caption_from_dataframe(best_df),
        "raw_block": raw_block,
        "candidate_index": best_idx,
        "shape": [int(best_df.shape[0]), int(best_df.shape[1])],
        "nonempty_cells": best_nonempty,
    }


def repair_docling_table_with_camelot(
    *,
    pdf_path: str | Path,
    table_obj: Dict[str, Any],
    docling_markdown: str,
    caption_text: str = "",
    footnote_text: str = "",
    bbox_pad: float = 3.0,
    column_min_gap: float = 2.0,
    min_nonempty_cells: int = 6,
    prefer_lattice: bool = False,
    use_docling_columns: bool = True,
) -> Dict[str, Any]:
    """
    Repair a Docling table using Camelot.

    Stream extraction is preferred when Docling provides reliable column
    separators. If those separators are unavailable, try an independent
    unguided stream extraction before falling back to Camelot lattice
    extraction. ``use_docling_columns`` suppresses only the guided attempt;
    it does not suppress the independent stream attempt.
    """
    try:
        import camelot
    except Exception as exc:
        return {
            "status": "failed",
            "attempted": False,
            "error": f"Camelot import failed: {exc}",
        }

    prov = (table_obj.get("prov") or [{}])[0]
    try:
        page_no = int(prov.get("page_no") or 0)
    except Exception:
        page_no = 0
    bbox = prov.get("bbox") or {}
    if not page_no or not all(k in bbox for k in ("l", "t", "r", "b")):
        return {
            "status": "not_attempted",
            "attempted": False,
            "reason": "Docling page/bbox provenance missing",
        }

    area = bbox_to_camelot_area(bbox, pad=bbox_pad)
    stream_attempts: List[Dict[str, Any]] = []

    if prefer_lattice:
        lattice_payload = _repair_docling_table_with_camelot_lattice(
            camelot_module=camelot,
            pdf_path=pdf_path,
            page_no=page_no,
            area=area,
            bbox=bbox,
            caption_text=caption_text,
            footnote_text=footnote_text,
            min_nonempty_cells=min_nonempty_cells,
        )
        lattice_payload["stream_attempts"] = [{
            "status": "not_attempted",
            "attempted": False,
            "strategy": "stream",
            "stream_mode": "guided",
            "reason": "prefer_lattice_for_suspect_docling_columns",
        }]
        lattice_payload["stream_attempt"] = lattice_payload["stream_attempts"][0]
        return lattice_payload

    columns = derive_docling_column_separators(table_obj, min_gap=column_min_gap)
    column_source = "strict_docling_gaps"
    if not columns:
        columns = derive_robust_docling_column_separators(table_obj)
        column_source = "robust_docling_medians"

    header = docling_header_from_markdown(docling_markdown)

    def run_stream_attempt(
        *,
        stream_mode: str,
        stream_columns: Optional[str],
        stream_header: List[str],
    ) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "status": "failed",
            "attempted": True,
            "strategy": "stream",
            "stream_mode": stream_mode,
            "pdf_path": str(pdf_path),
            "page_no": page_no,
            "bbox": bbox,
            "camelot_area": area,
        }
        if stream_columns:
            payload["docling_columns"] = stream_columns
            payload["docling_columns_source"] = column_source
        try:
            read_kwargs: Dict[str, Any] = {
                "pages": str(page_no),
                "flavor": "stream",
                "table_areas": [area],
                "split_text": True,
                "strip_text": "\n",
                "suppress_stdout": True,
            }
            if stream_columns:
                read_kwargs["columns"] = [stream_columns]
            tables = camelot.read_pdf(str(pdf_path), **read_kwargs)
            best_df, best_nonempty, best_idx = _best_camelot_dataframe(tables)
            payload.update({
                "tables_found": int(getattr(tables, "n", 0)),
                "best_nonempty_cells": best_nonempty,
            })
            if best_df is None or best_nonempty < min_nonempty_cells:
                payload.update({
                    "status": "attempted_no_output",
                    "reason": "Camelot stream returned no usable table",
                })
                return payload

            raw_block = dataframe_to_markdown(best_df)
            if stream_header:
                block = dataframe_to_markdown_with_header(
                    best_df,
                    stream_header,
                    caption_text=caption_text,
                    footnote_text=footnote_text,
                )
                if not block.strip():
                    payload.update({
                        "status": "attempted_no_output",
                        "reason": "Camelot stream table shape did not match Docling header",
                        "shape": [int(best_df.shape[0]), int(best_df.shape[1])],
                        "docling_header_columns": len(stream_header),
                    })
                    return payload
            else:
                block = markdown_block_with_caption(
                    raw_block,
                    caption_text=caption_text,
                    footnote_text=footnote_text,
                )

            payload.update({
                "status": "repaired" if block.strip() else "attempted_no_output",
                "block": block,
                "raw_block": raw_block,
                "candidate_index": best_idx,
                "shape": [int(best_df.shape[0]), int(best_df.shape[1])],
                "nonempty_cells": best_nonempty,
                "caption_candidate": embedded_caption_from_dataframe(best_df),
            })
            if not block.strip():
                payload["reason"] = "Camelot stream produced an empty Markdown block"
            return payload
        except Exception as exc:
            payload.update({
                "status": "failed",
                "error": f"Camelot stream extraction failed: {exc}",
            })
            return payload

    if use_docling_columns and columns and header:
        guided_attempt = run_stream_attempt(
            stream_mode="guided",
            stream_columns=columns,
            stream_header=header,
        )
    else:
        if not columns:
            reason = "Docling column separators unavailable"
        elif not header:
            reason = "Docling Markdown header unavailable"
        else:
            reason = "Docling-guided stream disabled because suspect columns require lattice preference"
        guided_attempt = {
            "status": "not_attempted",
            "attempted": False,
            "strategy": "stream",
            "stream_mode": "guided",
            "reason": f"Docling-guided stream not attempted: {reason}",
        }
    stream_attempts.append(guided_attempt)

    if guided_attempt.get("status") == "repaired":
        guided_attempt["stream_attempts"] = [dict(guided_attempt)]
        return guided_attempt

    unguided_attempt = run_stream_attempt(
        stream_mode="unguided",
        stream_columns=None,
        stream_header=[],
    )
    stream_attempts.append(unguided_attempt)
    if unguided_attempt.get("status") == "repaired":
        unguided_attempt["stream_attempts"] = [dict(item) for item in stream_attempts]
        return unguided_attempt

    lattice_payload = _repair_docling_table_with_camelot_lattice(
        camelot_module=camelot,
        pdf_path=pdf_path,
        page_no=page_no,
        area=area,
        bbox=bbox,
        caption_text=caption_text,
        footnote_text=footnote_text,
        min_nonempty_cells=min_nonempty_cells,
    )
    lattice_payload["stream_attempts"] = stream_attempts
    lattice_payload["stream_attempt"] = stream_attempts[-1] if stream_attempts else None
    return lattice_payload


def extract_pymupdf_table_candidate(
    *,
    pdf_path: str | Path,
    page_no: int,
    strategy: str,
    table_bbox: Optional[Dict[str, Any]] = None,
    bbox_pad: float = 3.0,
) -> Dict[str, Any]:
    try:
        import fitz  # PyMuPDF
    except Exception as exc:
        return {"status": "failed", "strategy": strategy, "error": f"PyMuPDF import failed: {exc}"}
    try:
        pdf = fitz.open(str(pdf_path))
        if page_no <= 0 or page_no > len(pdf):
            return {
                "status": "not_attempted",
                "attempted": False,
                "strategy": strategy,
                "reason": "page number out of range",
            }
        page = pdf[page_no - 1]
        clip_rect = docling_bbox_to_pymupdf_clip(
            table_bbox,
            page.rect,
            pad=bbox_pad,
            fitz_module=fitz,
        )
        clip_payload = (
            [float(clip_rect.x0), float(clip_rect.y0), float(clip_rect.x1), float(clip_rect.y1)]
            if clip_rect is not None
            else None
        )

        finder_kwargs: Dict[str, Any] = {"strategy": strategy}
        if clip_rect is not None:
            finder_kwargs["clip"] = clip_rect
        finder = page.find_tables(**finder_kwargs)
        tables = list(getattr(finder, "tables", []) or [])
        if not tables:
            payload = {
                "status": "attempted_no_output",
                "attempted": True,
                "strategy": strategy,
                "reason": "PyMuPDF find_tables returned no tables",
            }
            if clip_payload is not None:
                payload["pymupdf_clip"] = clip_payload
                payload["pymupdf_clip_source"] = "docling_table_bbox"
            return payload
        def table_score(table: Any) -> int:
            try:
                data = table.extract()
            except Exception:
                return 0
            cols = max((len(row) for row in data), default=0)
            nonempty = sum(1 for row in data for cell in row if normalize_markdown_cell(cell))
            return len(data) * cols + nonempty

        best = max(tables, key=table_score)
        grid = best.extract()
        block = grid_to_markdown(grid)
        return {
            "status": "extracted" if block.strip() else "attempted_no_output",
            "attempted": True,
            "strategy": strategy,
            "block": block,
            "tables_found": len(tables),
            "shape": [len(grid), max((len(row) for row in grid), default=0)],
            "bbox": [float(x) for x in getattr(best, "bbox", [])],
            **(
                {
                    "pymupdf_clip": clip_payload,
                    "pymupdf_clip_source": "docling_table_bbox",
                }
                if clip_payload is not None
                else {}
            ),
            **(
                {"reason": "PyMuPDF produced an empty Markdown block"}
                if not block.strip()
                else {}
            ),
        }
    except Exception as exc:
        return {
            "status": "failed",
            "attempted": True,
            "strategy": strategy,
            "error": str(exc),
        }
