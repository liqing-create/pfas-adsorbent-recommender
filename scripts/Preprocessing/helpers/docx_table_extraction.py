from __future__ import annotations

import re
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


@dataclass
class DocxTableEntry:
    table_index: int
    caption: str
    caption_id: str
    grid: List[List[str]]


@dataclass
class MarkdownStats:
    physical_rows: int
    columns: int
    nonempty_cells: int
    data_nonempty_cells: int
    numeric_cells: int
    max_cell_chars: int
    has_multiline_cells: bool


# -----------------------------------------------------------------------------
# DOCX parsing
# -----------------------------------------------------------------------------


def _iter_docx_table_entries(docx_path: Path) -> Iterable[DocxTableEntry]:
    try:
        from docx import Document  # type: ignore
        from docx.oxml.table import CT_Tbl  # type: ignore
        from docx.oxml.text.paragraph import CT_P  # type: ignore
        from docx.table import Table  # type: ignore
        from docx.text.paragraph import Paragraph  # type: ignore
    except Exception as exc:  # pragma: no cover - environment-dependent
        raise RuntimeError(f"python-docx is required for DOCX table repair: {exc}") from exc

    document = Document(str(docx_path))
    parent_elm = document.element.body
    pending_caption = ""
    table_index = 0

    for child in parent_elm.iterchildren():
        if isinstance(child, CT_P):
            paragraph = Paragraph(child, document)
            text = _clean_text_preserve_breaks(paragraph.text).replace("\n", " ").strip()
            if not text:
                continue
            caption_id = _canonical_table_id(_object_id_from_caption(text))
            if caption_id:
                pending_caption = text
            elif pending_caption:
                # Only keep an immediately preceding caption across blank lines.
                pending_caption = ""
            continue

        if isinstance(child, CT_Tbl):
            table_index += 1
            table = Table(child, document)
            caption = pending_caption
            pending_caption = ""
            grid = _extract_docx_table_grid(table)
            yield DocxTableEntry(
                table_index=table_index,
                caption=caption,
                caption_id=_canonical_table_id(_object_id_from_caption(caption)),
                grid=grid,
            )


def _extract_docx_table_grid(table: Any) -> List[List[str]]:
    rows: List[List[str]] = []
    max_width = 0
    for row in table.rows:
        cells = [_clean_text_preserve_breaks(cell.text) for cell in row.cells]
        max_width = max(max_width, len(cells))
        rows.append(cells)
    if max_width <= 0:
        return []
    return [row + [""] * (max_width - len(row)) for row in rows]


# -----------------------------------------------------------------------------
# Matching and rendering
# -----------------------------------------------------------------------------


def _select_docx_table_entry(
    *,
    entries: Sequence[DocxTableEntry],
    requested_id: str,
    requested_ordinal: Optional[int],
    expected_header_rows: Optional[List[List[str]]],
    docling_markdown: str,
) -> Tuple[Optional[DocxTableEntry], str]:
    if requested_id:
        exact = [entry for entry in entries if entry.caption_id == requested_id]
        if len(exact) == 1:
            return exact[0], "caption_id_exact"

    if requested_ordinal is not None:
        ordinal = [entry for entry in entries if entry.table_index == requested_ordinal]
        if len(ordinal) == 1:
            return ordinal[0], "table_ordinal"

    if expected_header_rows:
        ranked = sorted(
            entries,
            key=lambda entry: _header_overlap_score(entry.grid, expected_header_rows),
            reverse=True,
        )
        if ranked and _header_overlap_score(ranked[0].grid, expected_header_rows) >= 3:
            return ranked[0], "header_overlap"

    # Last-resort content-size match for the specific failure mode: Docling has
    # a tiny/header-only grid and exactly one DOCX table is much richer.  This is
    # intentionally conservative and rarely used.
    docling_stats = _markdown_stats(docling_markdown)
    rich = [
        entry
        for entry in entries
        if _grid_nonempty_cells(entry.grid) >= docling_stats.nonempty_cells + 8
    ]
    if len(rich) == 1:
        return rich[0], "single_richer_docx_table"

    return None, "not_found"


# -----------------------------------------------------------------------------
# Markdown helpers
# -----------------------------------------------------------------------------



def _cell_line_units(value: Any) -> List[str]:
    """Return clean line-position units from one DOCX cell."""
    raw = str(value or "").replace("\u00a0", " ")
    parts = [_clean_text(part) for part in re.split(r"\r?\n+", raw)]
    return [part for part in parts if part]


def _expand_position_aligned_multiline_rows(rows: List[List[str]]) -> List[List[str]]:
    """Expand DOCX rows that encode several aligned records inside cells.

    Word tables sometimes store one visible row with stacked values in several
    cells, for example PFAS=`PFOA\nPFOS...` and Kp1=`0.025\n0.042...`.
    Raw Markdown can only preserve those line breaks as `<br>`, which is hard
    to read.  This function expands only rows with strong positional evidence:
    at least two non-empty cells must share the same multi-line length, and all
    other non-empty cells must either be row-level singleton values or have that
    same length.
    """
    if not rows:
        return rows

    width = max((len(row) for row in rows), default=0)
    if width <= 0:
        return rows

    expanded: List[List[str]] = []
    for row in rows:
        padded = list(row) + [""] * (width - len(row))
        units_by_cell = [_cell_line_units(cell) for cell in padded]
        lengths = [len(units) for units in units_by_cell if units]
        if not lengths:
            expanded.append(padded)
            continue

        max_len = max(lengths)
        aligned_multiline_cells = sum(1 for length in lengths if length == max_len and length > 1)
        if max_len <= 1 or aligned_multiline_cells < 2:
            expanded.append(padded)
            continue

        # Do not guess when an intermediate-length cell exists. Example: lengths
        # [4, 3, 1] is ambiguous; [4, 4, 1, 0] is safe to expand.
        if any(length not in {1, max_len} for length in lengths):
            expanded.append(padded)
            continue

        for idx in range(max_len):
            out_row: List[str] = []
            for units in units_by_cell:
                if not units:
                    out_row.append("")
                elif len(units) == max_len:
                    out_row.append(units[idx])
                else:
                    out_row.append(units[0])
            expanded.append(out_row)

    return expanded

def _compose_table_block(table_body: str, caption: str = "", footnote: str = "") -> str:
    lines: List[str] = []
    if _clean_text(caption):
        lines.extend([_clean_text(caption), ""])
    if table_body.strip():
        lines.append(table_body.strip())
    if _clean_text(footnote):
        lines.extend(["", _clean_text_preserve_breaks(footnote)])
    return "\n".join(lines).rstrip() + "\n"


def _markdown_from_grid(rows: List[List[str]], *, preserve_line_breaks: bool) -> str:
    normalized = _trim_empty_edge_rows(rows)
    if not normalized:
        return ""
    width = max((len(row) for row in normalized), default=0)
    if width <= 0:
        return ""
    normalized = [row + [""] * (width - len(row)) for row in normalized]
    out = [
        "| " + " | ".join(_escape_md_cell(cell, preserve_line_breaks=preserve_line_breaks) for cell in normalized[0]) + " |",
        "| " + " | ".join("---" for _ in range(width)) + " |",
    ]
    for row in normalized[1:]:
        out.append("| " + " | ".join(_escape_md_cell(cell, preserve_line_breaks=preserve_line_breaks) for cell in row) + " |")
    return "\n".join(out).strip()


def _candidate_payload(
    *,
    name: str,
    block: str,
    status: str,
    accepted: bool,
    reasons: List[str],
    payload: Optional[Dict[str, Any]] = None,
    score: int = 0,
) -> Dict[str, Any]:
    stats = _markdown_stats(block)
    return {
        "name": name,
        "status": status,
        "accepted": accepted,
        "score": score,
        "shape": [max(0, stats.physical_rows - 1), stats.columns],
        "header_action": "docx_native" if name == "docx_native" else "none",
        "dropped_header_col_idx": None,
        "reasons": reasons,
        "payload": payload or {},
        "block": block.strip(),
    }


def _skipped_payload(route: str, reason: str) -> Dict[str, Any]:
    return {
        "status": "skipped",
        "route": route,
        "accepted": False,
        "reason": reason,
        "candidates": [],
    }


# -----------------------------------------------------------------------------
# Scoring helpers
# -----------------------------------------------------------------------------


def _markdown_stats(markdown: str) -> MarkdownStats:
    rows = _md_table_rows(markdown)
    physical_rows = max(0, len(rows) - 1) if len(rows) >= 2 else len(rows)
    columns = max((len(row) for row in rows), default=0)
    nonempty_cells = sum(1 for row in rows for cell in row if _clean_text(_strip_html(cell)))
    data_rows = rows[2:] if len(rows) >= 3 else []
    data_nonempty_cells = sum(1 for row in data_rows for cell in row if _clean_text(_strip_html(cell)))
    numeric_cells = sum(1 for row in rows for cell in row if _independent_numeric_count(cell) > 0)
    max_cell_chars = max((len(_clean_text(_strip_html(cell))) for row in rows for cell in row), default=0)
    has_multiline_cells = "<br>" in markdown or "<br/>" in markdown or "<br />" in markdown
    return MarkdownStats(
        physical_rows=physical_rows,
        columns=columns,
        nonempty_cells=nonempty_cells,
        data_nonempty_cells=data_nonempty_cells,
        numeric_cells=numeric_cells,
        max_cell_chars=max_cell_chars,
        has_multiline_cells=has_multiline_cells,
    )


def _grid_nonempty_cells(grid: List[List[str]]) -> int:
    return sum(1 for row in grid for cell in row if _clean_text(cell))


def _header_overlap_score(grid: List[List[str]], expected_header_rows: Optional[List[List[str]]]) -> int:
    if not grid or not expected_header_rows:
        return 0
    grid_tokens = set(_header_tokens(grid[: min(3, len(grid))]))
    expected_tokens = set(_header_tokens(expected_header_rows))
    return len(grid_tokens & expected_tokens)


def _header_tokens(rows: Sequence[Sequence[str]]) -> List[str]:
    tokens: List[str] = []
    for row in rows:
        for cell in row:
            for token in re.findall(r"[A-Za-z][A-Za-z0-9+-]{1,}", _clean_text(cell).casefold()):
                if len(token) >= 2:
                    tokens.append(token)
    return tokens


# -----------------------------------------------------------------------------
# Text normalization and caption parsing
# -----------------------------------------------------------------------------


def _clean_text(text: Any) -> str:
    s = str(text or "")
    s = s.replace("\u00a0", " ")
    s = re.sub(r"(?:GLYPH<0>|GLYPH&lt;0&gt;)", "-", s)
    s = re.sub(r"GLYPH<\d+>", "", s)
    s = re.sub(r"[ \t\r\n]+", " ", s)
    s = re.sub(r"\s+([,.;:%)\]])", r"\1", s)
    s = re.sub(r"([([])\s+", r"\1", s)
    return s.strip()


def _clean_text_preserve_breaks(text: Any) -> str:
    raw = str(text or "").replace("\u00a0", " ")
    lines = [_clean_text(part) for part in re.split(r"\r?\n+", raw)]
    return "\n".join(part for part in lines if part)


def _escape_md_cell(value: Any, *, preserve_line_breaks: bool) -> str:
    s = _clean_text_preserve_breaks(value)
    s = s.replace("|", r"\|")
    if preserve_line_breaks:
        s = re.sub(r"\s*\n\s*", "<br>", s)
    else:
        s = _clean_text(s)
    return s.strip()


def _trim_empty_edge_rows(rows: List[List[str]]) -> List[List[str]]:
    out = [list(row) for row in rows]
    while out and not any(_clean_text(cell) for cell in out[0]):
        out.pop(0)
    while out and not any(_clean_text(cell) for cell in out[-1]):
        out.pop()
    return out


def _strip_html(value: str) -> str:
    return re.sub(r"<[^>]+>", " ", value or "")


def _md_table_rows(block: str) -> List[List[str]]:
    return [_split_md_row(line) for line in (block or "").splitlines() if line.lstrip().startswith("|")]


def _split_md_row(line: str) -> List[str]:
    s = (line or "").strip()
    if s.startswith("|"):
        s = s[1:]
    if s.endswith("|"):
        s = s[:-1]
    cells: List[str] = []
    buf: List[str] = []
    escaped = False
    for ch in s:
        if escaped:
            buf.append(ch)
            escaped = False
            continue
        if ch == "\\":
            buf.append(ch)
            escaped = True
            continue
        if ch == "|":
            cells.append("".join(buf).strip())
            buf = []
        else:
            buf.append(ch)
    cells.append("".join(buf).strip())
    return cells


_CAPTION_RE = re.compile(
    r"^\s*(?:Supplementary\s+|Supplemental\s+|Supporting\s+)?"
    r"(?P<kind>Table|Tab\.?|Tabel|Exhibit)\s*\.?\s*"
    r"(?P<num>(?:S|SI|SM|SF|ST|A)?\s*[.\-]?\s*\d+(?:\s*[A-Za-z]|[._\-]\d+)?|\d+\s*S|[IVXLCDM]+)\b",
    re.I,
)


def _object_id_from_caption(caption: str) -> str:
    match = _CAPTION_RE.match(caption or "")
    if not match:
        return ""
    return _canonical_table_id(f"Table {match.group('num')}")


def _canonical_table_id(value: str) -> str:
    text = _clean_text(value)
    if not text:
        return ""
    match = re.search(
        r"\b(?:Table|Tab\.?|Tabel|Exhibit)\s*\.?\s*"
        r"((?:S|SI|SM|SF|ST|A)?\s*[.\-]?\s*\d+(?:\s*[A-Za-z]|[._\-]\d+)?|\d+\s*S|[IVXLCDM]+)\b",
        text,
        flags=re.I,
    )
    if not match:
        # Accept already-trimmed forms such as "S6" only if caller gave one.
        simple = re.fullmatch(r"((?:S|A)?\s*\d+[A-Za-z]?|[IVXLCDM]+)", text, flags=re.I)
        if not simple:
            return ""
        raw_num = simple.group(1)
    else:
        raw_num = match.group(1)
    num = re.sub(r"[\s._-]+", "", raw_num.upper())
    if re.fullmatch(r"\d+S", num):
        num = "S" + num[:-1]
    if re.fullmatch(r"(?:SI|SM|SF|ST)\d+", num):
        num = "S" + re.sub(r"^(?:SI|SM|SF|ST)", "", num)
    return f"Table {num}"


def _table_id_to_ordinal(table_id: str) -> Optional[int]:
    match = re.search(r"\bS?\s*(\d+)\b", table_id or "", flags=re.I)
    if match:
        try:
            return int(match.group(1))
        except Exception:
            return None
    roman = re.search(r"\b([IVXLCDM]+)\b", table_id or "", flags=re.I)
    if roman:
        return _roman_to_int(roman.group(1))
    return None


def _roman_to_int(value: str) -> Optional[int]:
    value = (value or "").upper()
    if not re.fullmatch(r"M{0,3}(?:CM|CD|D?C{0,3})(?:XC|XL|L?X{0,3})(?:IX|IV|V?I{0,3})", value):
        return None
    total = 0
    prev = 0
    vals = {"I": 1, "V": 5, "X": 10, "L": 50, "C": 100, "D": 500, "M": 1000}
    for ch in reversed(value):
        cur = vals[ch]
        if cur < prev:
            total -= cur
        else:
            total += cur
            prev = cur
    return total or None


_PLUS_MINUS_PAT = r"(?:\u00b1|\u00c2\u00b1|\+/-)"
_VALUE_NUM_PAT = r"[-+−–—]?(?:\d+(?:\.\d+)?|\.\d+)(?:[eE][-+]?\d+)?"
_VALUE_TOKEN_PAT = rf"{_VALUE_NUM_PAT}(?:\s*{_PLUS_MINUS_PAT}\s*{_VALUE_NUM_PAT})?"
_VALUE_ONLY_RE = re.compile(rf"^\s*{_VALUE_TOKEN_PAT}(?:\s*[,;]\s*{_VALUE_TOKEN_PAT})*\s*%?\s*$")
_VALUE_AT_SEGMENT_START_RE = re.compile(rf"(?:(?<=^)|(?<=[,;:\[(])\s*)({_VALUE_TOKEN_PAT})(?!\s*/\s*[A-Za-z])")


def _independent_numeric_count(cell: str) -> int:
    text = _clean_text(_strip_html(cell))
    if not re.search(r"\d", text):
        return 0
    if _VALUE_ONLY_RE.fullmatch(text):
        return len(re.findall(_VALUE_TOKEN_PAT, text))
    return len(list(_VALUE_AT_SEGMENT_START_RE.finditer(text)))




def iter_docx_table_entries(docx_path: Path | str) -> Iterable[DocxTableEntry]:
    return _iter_docx_table_entries(Path(docx_path))


def extract_docx_native_table_candidate(
    *,
    docx_path: Path | str,
    table_id: str,
    docling_markdown: str,
    caption_text: str = "",
    footnote_text: str = "",
    expected_header_rows: Optional[List[List[str]]] = None,
    table_ordinal_hint: Optional[int] = None,
    preserve_line_breaks: bool = False,
    expand_multiline_cells: bool = True,
) -> Dict[str, Any]:
    path = Path(docx_path)
    if not path.is_file():
        return {"status": "skipped", "reason": f"source DOCX not found: {path}", "docx_path": str(path)}

    try:
        entries = list(_iter_docx_table_entries(path))
    except Exception as exc:
        return {
            "status": "failed",
            "reason": "docx_table_extraction_failed",
            "error": str(exc),
            "docx_path": str(path),
        }

    requested_id = _canonical_table_id(table_id or caption_text)
    if not requested_id:
        requested_id = _canonical_table_id(_object_id_from_caption(caption_text))
    requested_ordinal = table_ordinal_hint or _table_id_to_ordinal(requested_id)

    match_entry, match_reason = _select_docx_table_entry(
        entries=entries,
        requested_id=requested_id,
        requested_ordinal=requested_ordinal,
        expected_header_rows=expected_header_rows,
        docling_markdown=docling_markdown,
    )
    if not match_entry:
        return {
            "status": "not_found",
            "reason": "matching_docx_table_not_found",
            "requested_table_id": requested_id or table_id,
            "requested_ordinal": requested_ordinal,
            "tables_scanned": len(entries),
            "docx_path": str(path),
        }

    render_grid = (
        _expand_position_aligned_multiline_rows(match_entry.grid)
        if expand_multiline_cells
        else match_entry.grid
    )
    body = _markdown_from_grid(render_grid, preserve_line_breaks=preserve_line_breaks)
    recovered_block = _compose_table_block(
        body,
        caption=caption_text or match_entry.caption,
        footnote=footnote_text,
    ).rstrip()
    recovered_stats = _markdown_stats(recovered_block)
    return {
        "status": "extracted" if recovered_block else "skipped",
        "strategy": match_reason,
        "block": recovered_block,
        "docx_path": str(path),
        "table_index": match_entry.table_index,
        "caption": match_entry.caption,
        "caption_id": match_entry.caption_id,
        "requested_table_id": requested_id or table_id,
        "requested_ordinal": requested_ordinal,
        "tables_scanned": len(entries),
        "shape": [len(match_entry.grid), max((len(row) for row in match_entry.grid), default=0)],
        "render_shape": [len(render_grid), max((len(row) for row in render_grid), default=0)],
        "stats": asdict(recovered_stats),
    }
