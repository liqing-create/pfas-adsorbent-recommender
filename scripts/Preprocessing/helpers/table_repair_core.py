from __future__ import annotations

import re
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple

_HTML_BREAK_RE = re.compile(
    r"(?:<\s*br\s*/?\s*>|&lt;\s*br\s*/?\s*&gt;)",
    re.I,
)


def normalize_markdown_cell(value: Any) -> str:
    """Normalize one extracted cell before serializing it into Markdown.

    PDF engines sometimes preserve visual line breaks inside one logical cell.
    For compact variable labels, join fragments without a space, e.g. dR\ni -> dRi,
    q\ne -> qe, and logK\nD -> logKD. For normal prose-like cells, collapse
    line breaks to spaces.
    """
    text = str(value or "")
    text = _HTML_BREAK_RE.sub(" ", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    parts = [part.strip() for part in text.split("\n") if part.strip()]
    if len(parts) >= 2:
        compact_parts = all(
            re.fullmatch(r"[A-Za-z0-9+\-*/().%]+", part or "")
            for part in parts
        )
        has_short_fragment = any(len(part) <= 2 for part in parts)
        text = "".join(parts) if compact_parts and has_short_fragment else " ".join(parts)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\s+([,.;:)\]])", r"\1", text)
    text = re.sub(r"([([])\s+", r"\1", text)
    return text.strip()


def _escape_md(value: Any) -> str:
    text = normalize_markdown_cell(value)
    # Avoid double-escaping already escaped literal pipes such as \|△Q\|,
    # which can otherwise become \\|△Q\\| and be parsed as table delimiters.
    text = re.sub(r"\\+\|", "|", text)
    return text.replace("|", r"\|").strip()


def markdown_block_with_caption(
    table_markdown: str,
    *,
    caption_text: str = "",
    footnote_text: str = "",
) -> str:
    lines: List[str] = []
    if caption_text.strip():
        lines.extend([caption_text.strip(), ""])
    if table_markdown.strip():
        lines.append(table_markdown.strip())
    if footnote_text.strip():
        lines.extend(["", footnote_text.strip()])
    return "\n".join(lines).rstrip() + "\n"


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


def docling_header_from_markdown(markdown_table: str) -> List[str]:
    for line in (markdown_table or "").splitlines():
        if line.lstrip().startswith("|"):
            return _split_md_row(line)
    return []


# Shared table repair candidate generation / validation -----------------------
# These helpers intentionally do not prefer a source format or extraction tool.
# Tool names are recorded only as provenance; selection is based on grid
# validity and table quality signals.

DEFAULT_OPTIONAL_BLANK_COLUMN_RE = re.compile(r"\b(?:crosslinker)\b", re.I)
_PLUS_MINUS_TOKEN_PAT = r"(?:\u00b1|\u00c2\u00b1|\u0105|\uff71|\+/-)"
_PLUS_MINUS_ONLY_RE = re.compile(rf"^{_PLUS_MINUS_TOKEN_PAT}$")
_VALUE_NUM_PAT = r"[-+\u2212\u2013\u2014]?(?:\d+(?:\.\d+)?|\.\d+)(?:[eE][-+]?\d+)?"
_NUMERIC_TOKEN_RE = re.compile(rf"^{_VALUE_NUM_PAT}(?:%)?$", re.I)
_VALUE_FRAGMENT_RE = re.compile(rf"^(?:{_VALUE_NUM_PAT}|n/?a|N/?A)$", re.I)
_PLUS_MINUS_VALUE_PAT = _PLUS_MINUS_TOKEN_PAT
_VALUE_TOKEN_PAT = rf"{_VALUE_NUM_PAT}(?:\s*{_PLUS_MINUS_VALUE_PAT}\s*{_VALUE_NUM_PAT})?"
_VALUE_ONLY_RE = re.compile(
    rf"^\s*{_VALUE_TOKEN_PAT}(?:\s*[,;]\s*{_VALUE_TOKEN_PAT})*\s*%?\s*$"
)
_VALUE_AT_SEGMENT_START_RE = re.compile(
    rf"(?:(?<=^)|(?<=[,;:\[(])\s*)({_VALUE_TOKEN_PAT})(?!\s*/\s*[A-Za-z])"
)
_TABLE_ID_OR_CAPTION_RE = re.compile(r"\b(?:table|tab\.?|figure|fig\.?)\s*(?:s|si|sm|sf|st|a)?\s*[.\-]?\s*\d+\b", re.I)
_TRAILING_PLUS_MINUS_RE = re.compile(rf"^\s*{_VALUE_NUM_PAT}\s*{_PLUS_MINUS_TOKEN_PAT}\s*$", re.I)
_LEADING_PLUS_MINUS_VALUE_RE = re.compile(rf"^\s*{_PLUS_MINUS_TOKEN_PAT}\s*{_VALUE_NUM_PAT}\s*$", re.I)
_PACKED_VALUE_NUM_PAT = r"[-+\u2212]?(?:\d+(?:\.\d+)?|\.\d+)(?:[eE][-+]?\d+)?"
_PACKED_VALUE_TOKEN_PAT = (
    rf"{_PACKED_VALUE_NUM_PAT}"
    rf"(?:(?:\s*[\u2013\u2014]\s*|-(?!\s)){_PACKED_VALUE_NUM_PAT})?"
    rf"(?:\s*{_PLUS_MINUS_TOKEN_PAT}\s*{_PACKED_VALUE_NUM_PAT})?"
    r"%?"
)
_PACKED_SAFE_NUM_PAT = (
    r"[-+\u2212]?"
    r"(?:(?:\d{1,3}(?:[,\s]\d{3})+)|(?:\d+(?:\.\d+)?)|(?:\.\d+))"
    r"(?:[eE][-+]?\d+)?"
)
_SINGLE_PACKED_NUMERIC_VALUE_RE = re.compile(
    rf"^\s*[\[(]?\s*{_PACKED_SAFE_NUM_PAT}\s*%?\s*[\])]?\s*$"
)
_SINGLE_NUMERIC_RANGE_RE = re.compile(
    rf"^\s*[\[(]?\s*{_PACKED_SAFE_NUM_PAT}\s*%?"
    rf"\s*[-\u2013\u2014]\s*"
    rf"{_PACKED_SAFE_NUM_PAT}\s*%?\s*[\])]?\s*$"
)
_SINGLE_PLUS_MINUS_VALUE_RE = re.compile(
    rf"^\s*[\[(]?\s*{_PACKED_SAFE_NUM_PAT}\s*%?"
    rf"\s*{_PLUS_MINUS_TOKEN_PAT}\s*"
    rf"{_PACKED_SAFE_NUM_PAT}\s*%?\s*[\])]?\s*$"
)
_NUMERIC_LIST_OR_ALTERNATIVE_RE = re.compile(
    rf"^\s*{_PACKED_SAFE_NUM_PAT}\s*%?"
    rf"(?:\s*(?:,|;|and|or)\s*{_PACKED_SAFE_NUM_PAT}\s*%?)+\s*$",
    re.I,
)

_PACKED_NUMERIC_VALUES_ONLY_RE = re.compile(
    rf"^\s*[\[(]?\s*{_PACKED_VALUE_TOKEN_PAT}"
    rf"(?:\s+{_PACKED_VALUE_TOKEN_PAT})*"
    rf"\s*[\])]?\s*$"
)

def _cells_to_md_row(cells: List[str]) -> str:
    return "| " + " | ".join(_escape_md(cell) for cell in (cells or [])) + " |"


def _body_pipe_lines(table_body: str) -> List[str]:
    return [line.rstrip() for line in (table_body or "").splitlines() if line.lstrip().startswith("|")]


def _is_separator_cells(cells: List[str]) -> bool:
    return bool(cells) and all(re.fullmatch(r":?-{3,}:?", (cell or "").replace(" ", "")) for cell in cells)


def _md_table_rows(block: str) -> List[List[str]]:
    rows = [_split_md_row(line) for line in _body_pipe_lines(block)]
    return [row for row in rows if row]


def _separator_row_index(rows: List[List[str]]) -> Optional[int]:
    for idx, row in enumerate(rows):
        if len(row) > 1 and _is_separator_cells(row):
            return idx
    return None


def _markdown_table_shape(block: str) -> Tuple[int, int]:
    rows = _md_table_rows(block)
    if not rows:
        return 0, 0
    sep_idx = _separator_row_index(rows)
    if sep_idx is None:
        return 0, max((len(row) for row in rows), default=0)
    data_rows = rows[sep_idx + 1 :]
    return len(data_rows), len(rows[sep_idx])


def _markdown_row_width_repair_reasons(block: str) -> List[str]:
    rows = _md_table_rows(block)
    if len(rows) < 2:
        return ["missing_markdown_table_grid"]
    sep_idx = _separator_row_index(rows)
    if sep_idx is None:
        return ["missing_separator_row"]
    if sep_idx == 0:
        return ["separator_without_header_row"]
    width = len(rows[sep_idx])
    reasons: List[str] = []
    for idx, row in enumerate(rows):
        if idx == sep_idx:
            continue
        if len(row) != width:
            if idx < sep_idx:
                reasons.append("header_width_mismatch")
            else:
                reasons.append("row_width_mismatch")
            break
    return reasons


def _independent_numeric_count(cell: str) -> int:
    """Count actual numeric table values, not numeric-looking unit tokens."""
    text = normalize_markdown_cell(re.sub(r"<[^>]+>", " ", cell or ""))
    # Match Docling-style normalized unit exponents from copied blocks too.
    text = re.sub(r"(?:GLYPH<0>|GLYPH&lt;0&gt;)", "-", text)
    text = re.sub(r"GLYPH<\d+>", "", text)
    text = re.sub(r"-\s+(?=\d)", "-", text)
    if not re.search(r"\d", text):
        return 0

    # Pure numeric cells may contain several real values separated by comma or
    # semicolon. Count those directly.
    if _VALUE_ONLY_RE.fullmatch(text):
        return len(re.findall(_VALUE_TOKEN_PAT, text))

    # Mixed text cells only count values that begin a value segment. This
    # counts "5 mg L−1" as one value, but does not count the exponent in
    # "μ mol g GLYPH<0> 1" or the parameter label "1/n".
    return len(list(_VALUE_AT_SEGMENT_START_RE.finditer(text)))

def _single_cell_dense_collapse_reasons(block: str) -> List[str]:
    rows = _md_table_rows(block)
    if len(rows) == 2:
        nonempty_header_cells = [cell for cell in rows[0] if normalize_markdown_cell(cell)]
        if len(nonempty_header_cells) == 1:
            cell = normalize_markdown_cell(nonempty_header_cells[0])
            if len(cell) >= 500 and _independent_numeric_count(cell) >= 10:
                return ["single_cell_dense_text_collapse"]
            if _independent_numeric_count(cell) >= 25:
                return ["single_cell_numeric_collapse"]
    return []


def _split_plus_minus_value_reasons(block: str) -> List[str]:
    """Detect over-fragmented text extraction such as 1 | ± | 1 instead of 1 ± 1."""
    rows = _md_table_rows(block)
    sep_idx = _separator_row_index(rows)
    if sep_idx is None:
        return []
    data_rows = rows[sep_idx + 1 :]
    if len(data_rows) < 2:
        return []
    affected = 0
    for row in data_rows:
        plusminus = sum(bool(_PLUS_MINUS_ONLY_RE.fullmatch(normalize_markdown_cell(cell))) for cell in row)
        numeric_tokens = sum(bool(_NUMERIC_TOKEN_RE.fullmatch(normalize_markdown_cell(cell))) for cell in row)
        split_pairs = 0
        normalized = [normalize_markdown_cell(cell) for cell in row]
        for idx, cell in enumerate(normalized):
            next_cell = normalized[idx + 1] if idx + 1 < len(normalized) else ""
            if _TRAILING_PLUS_MINUS_RE.fullmatch(cell) and _VALUE_FRAGMENT_RE.fullmatch(next_cell):
                split_pairs += 1
            if _VALUE_FRAGMENT_RE.fullmatch(cell) and _LEADING_PLUS_MINUS_VALUE_RE.fullmatch(next_cell):
                split_pairs += 1
        if (plusminus >= 2 and numeric_tokens >= 4) or split_pairs >= 2:
            affected += 1
    if affected >= max(2, len(data_rows) // 3):
        return ["split_plus_minus_values_across_cells"]
    return []



def _packed_numeric_value_count(cell: str) -> int:
    """Count whitespace-packed numeric values, excluding units and labels.

    Examples:
      "12.4 36.2" -> 2
      "(1.21–2.44)" -> 1
      "10 000" -> 1
      "20 -1800" -> 1
      "1, 3, 5, 7" -> 1
      "10 000 and 20 000" -> 1
      "2.69 × 103–1.92 × 105" -> 0 here, because it is not a simple packed
      numeric-only cell and should not be handled by this guard.
    """
    text = normalize_markdown_cell(re.sub(r"<[^>]+>", " ", cell or ""))
    text = re.sub(r"\s+", " ", text).strip()
    if not re.search(r"\d", text):
        return 0

    # Legitimate one-cell numeric expressions. These are not neighboring
    # value collapse, even though they contain multiple numeric-looking tokens.
    if (
        _SINGLE_PACKED_NUMERIC_VALUE_RE.fullmatch(text)
        or _SINGLE_NUMERIC_RANGE_RE.fullmatch(text)
        or _SINGLE_PLUS_MINUS_VALUE_RE.fullmatch(text)
        or _NUMERIC_LIST_OR_ALTERNATIVE_RE.fullmatch(text)
    ):
        return 1

    if not _PACKED_NUMERIC_VALUES_ONLY_RE.fullmatch(text):
        return 0
    return len(re.findall(_PACKED_VALUE_TOKEN_PAT, text))


def _neighboring_value_collapse_reasons(block: str) -> List[str]:
    """Detect values from neighboring columns collapsed into one cell.

    This is a candidate-validation signal, not a positive scoring signal. It
    catches cases such as "12.4 36.2" in one body cell, where the extractor
    preserved both numbers but not the cell boundary.
    """
    rows = _md_table_rows(block)
    sep_idx = _separator_row_index(rows)
    if sep_idx is None:
        return []

    data_rows = rows[sep_idx + 1 :]

    for row in data_rows:
        for cell in row:
            if _packed_numeric_value_count(cell) >= 2:
                return ["neighboring_value_collapse"]
    return []

def _caption_absorbed_reasons(block: str) -> List[str]:
    rows = _md_table_rows(block)
    if not rows:
        return []
    first_row_cells = [normalize_markdown_cell(cell) for cell in rows[0] if normalize_markdown_cell(cell)]
    first_row_text = " ".join(first_row_cells)
    compact = re.sub(r"[^a-z0-9]+", "", first_row_text.casefold())
    if len(rows[0]) >= 4 and (
        _TABLE_ID_OR_CAPTION_RE.search(first_row_text)
        or re.search(r"(?:table|tab|figure|fig)(?:s|si|sm|sf|st|a)?\d+", compact)
    ):
        return ["caption_text_absorbed_as_table_row"]
    return []

_LABEL_ASSIGNMENT_STATUS_EXPLICIT = "explicit"
_LABEL_ASSIGNMENT_STATUS_FILLDOWN = "fill_down_certified"
_LABEL_ASSIGNMENT_STATUS_UNCERTAIN = "uncertain"
_LABEL_ASSIGNMENT_STATUSES = {
    _LABEL_ASSIGNMENT_STATUS_EXPLICIT,
    _LABEL_ASSIGNMENT_STATUS_FILLDOWN,
    _LABEL_ASSIGNMENT_STATUS_UNCERTAIN,
}

def _labelish_cell(cell: str) -> bool:
    text = normalize_markdown_cell(cell)
    if not text or len(text) > 120:
        return False
    if not re.search(r"[A-Za-z]", text):
        return False
    # Keep labels such as "6:2 FTSA", but exclude purely numeric/value cells.
    if _VALUE_ONLY_RE.fullmatch(text):
        return False
    return True


def _record_like_row(row: List[str], label_col: int) -> bool:
    later = [
        normalize_markdown_cell(cell)
        for idx, cell in enumerate(row)
        if idx != label_col and normalize_markdown_cell(cell)
    ]
    if not later:
        return False
    numeric_later = sum(_independent_numeric_count(cell) > 0 for cell in later)
    return len(later) >= 2 or numeric_later >= 1

def _row_has_value_evidence_after(row: List[str], col_idx: int) -> bool:
    later = [
        normalize_markdown_cell(cell)
        for cell in row[col_idx + 1 :]
        if normalize_markdown_cell(cell)
    ]
    if not later:
        return False
    numeric_later = sum(_independent_numeric_count(cell) > 0 for cell in later)
    return numeric_later >= 1 or len(later) >= 2


def _row_has_any_content_after(row: List[str], col_idx: int) -> bool:
    return any(normalize_markdown_cell(cell) for cell in row[col_idx + 1 :])


def _looks_like_repeated_header_body_row(
    row: List[str],
    header_rows: List[List[str]],
    width: int,
) -> bool:
    padded = list(row) + [""] * max(0, width - len(row))
    row_norm = [normalize_markdown_cell(cell).casefold() for cell in padded[:width]]
    if not any(row_norm):
        return True

    for header in header_rows:
        header_padded = list(header) + [""] * max(0, width - len(header))
        header_norm = [
            normalize_markdown_cell(cell).casefold()
            for cell in header_padded[:width]
        ]
        comparable = [(a, b) for a, b in zip(row_norm, header_norm) if a or b]
        if comparable and all(a == b for a, b in comparable):
            return True

    nonempty = [normalize_markdown_cell(cell) for cell in padded if normalize_markdown_cell(cell)]
    if not nonempty:
        return True
    numeric_cells = sum(_independent_numeric_count(cell) > 0 for cell in nonempty)
    alpha_cells = sum(bool(re.search(r"[A-Za-z]", cell)) for cell in nonempty)
    return bool(
        len(nonempty) >= 3
        and numeric_cells == 0
        and alpha_cells >= max(2, len(nonempty) // 2)
    )


def _trim_leading_header_like_body_rows(
    body_rows: List[List[str]],
    header_rows: List[List[str]],
    width: int,
) -> List[List[str]]:
    rows = list(body_rows)
    while rows and _looks_like_repeated_header_body_row(rows[0], header_rows, width):
        rows.pop(0)
    return rows


def _candidate_sparse_row_label_columns(
    body_rows: List[List[str]],
    width: int,
) -> List[int]:
    """Infer sparse row-label columns from body geometry, not header names."""
    padded_rows = [list(row) + [""] * max(0, width - len(row)) for row in body_rows]
    candidates: List[int] = []

    for col_idx in range(width):
        rows_with_right_content = [
            row for row in padded_rows if _row_has_any_content_after(row, col_idx)
        ]
        if len(rows_with_right_content) < 2:
            continue

        labels_anywhere = [
            normalize_markdown_cell(row[col_idx])
            for row in padded_rows
            if _labelish_cell(row[col_idx])
        ]
        if not labels_anywhere:
            continue

        blank_record_rows = [
            row for row in rows_with_right_content
            if not normalize_markdown_cell(row[col_idx])
            and _row_has_value_evidence_after(row, col_idx)
        ]
        if not blank_record_rows:
            continue

        nonempty_in_col = [
            normalize_markdown_cell(row[col_idx])
            for row in padded_rows
            if normalize_markdown_cell(row[col_idx])
        ]
        if not nonempty_in_col:
            continue

        labelish_nonempty = sum(_labelish_cell(value) for value in nonempty_in_col)
        numeric_nonempty = sum(_independent_numeric_count(value) > 0 for value in nonempty_in_col)
        if labelish_nonempty >= max(1, numeric_nonempty + 1):
            candidates.append(col_idx)

    return candidates


def _row_label_assignment_status_for_column(
    body_rows: List[List[str]],
    col_idx: int,
    width: int,
) -> Dict[str, Any]:
    padded_rows = [list(row) + [""] * max(0, width - len(row)) for row in body_rows]
    value_rows = [
        row for row in padded_rows if _row_has_value_evidence_after(row, col_idx)
    ]
    label_only_rows = [
        row for row in padded_rows
        if _labelish_cell(row[col_idx])
        and not _row_has_value_evidence_after(row, col_idx)
    ]

    if label_only_rows:
        return {
            "column": col_idx,
            "status": _LABEL_ASSIGNMENT_STATUS_UNCERTAIN,
            "reason": "label_appears_without_record_values",
        }

    if not value_rows:
        return {
            "column": col_idx,
            "status": _LABEL_ASSIGNMENT_STATUS_UNCERTAIN,
            "reason": "no_value_bearing_rows_to_certify_filldown",
        }

    labels = [normalize_markdown_cell(row[col_idx]) for row in value_rows]
    if all(labels):
        return {
            "column": col_idx,
            "status": _LABEL_ASSIGNMENT_STATUS_EXPLICIT,
            "reason": "all_value_rows_have_labels",
        }

    if not labels[0]:
        return {
            "column": col_idx,
            "status": _LABEL_ASSIGNMENT_STATUS_UNCERTAIN,
            "reason": "value_rows_before_first_label",
        }

    seen_label = False
    blanks_after_label = 0
    for label in labels:
        if label:
            seen_label = True
            continue
        if not seen_label:
            return {
                "column": col_idx,
                "status": _LABEL_ASSIGNMENT_STATUS_UNCERTAIN,
                "reason": "blank_before_any_label",
            }
        blanks_after_label += 1

    return {
        "column": col_idx,
        "status": (
            _LABEL_ASSIGNMENT_STATUS_FILLDOWN
            if blanks_after_label
            else _LABEL_ASSIGNMENT_STATUS_EXPLICIT
        ),
        "reason": "first_value_row_has_label_and_later_blanks_fill_down",
    }


def _combine_label_assignment_status(statuses: List[str]) -> str:
    statuses = [status for status in statuses if status in _LABEL_ASSIGNMENT_STATUSES]
    if not statuses:
        return _LABEL_ASSIGNMENT_STATUS_EXPLICIT
    if _LABEL_ASSIGNMENT_STATUS_UNCERTAIN in statuses:
        return _LABEL_ASSIGNMENT_STATUS_UNCERTAIN
    if _LABEL_ASSIGNMENT_STATUS_FILLDOWN in statuses:
        return _LABEL_ASSIGNMENT_STATUS_FILLDOWN
    return _LABEL_ASSIGNMENT_STATUS_EXPLICIT


def _row_label_assignment_status(block: str) -> Dict[str, Any]:
    rows = _md_table_rows(block)
    sep_idx = _separator_row_index(rows)
    if sep_idx is None or sep_idx == 0:
        return {"status": _LABEL_ASSIGNMENT_STATUS_EXPLICIT, "columns": []}

    header_rows = rows[:sep_idx]
    body_rows = rows[sep_idx + 1 :]
    if len(body_rows) < 2:
        return {"status": _LABEL_ASSIGNMENT_STATUS_EXPLICIT, "columns": []}

    width = max((len(row) for row in rows), default=0)
    body_rows = _trim_leading_header_like_body_rows(body_rows, header_rows, width)
    sparse_label_cols = _candidate_sparse_row_label_columns(body_rows, width)
    if not sparse_label_cols:
        return {"status": _LABEL_ASSIGNMENT_STATUS_EXPLICIT, "columns": []}
    column_payloads = [
        _row_label_assignment_status_for_column(body_rows, col_idx, width)
        for col_idx in sparse_label_cols
    ]
    return {
        "status": _combine_label_assignment_status(
            [str(payload.get("status") or "") for payload in column_payloads]
        ),
        "columns": column_payloads,
    }
    return []


def _lower_header_cell_is_subheader(cell: str) -> bool:
    text = normalize_markdown_cell(cell)
    if not text:
        return False
    return bool(re.search(r"[A-Za-z0-9]", text))

def _column_label_assignment_status(block: str) -> Dict[str, Any]:
    """Infer unsafe column-label assignment from header geometry, not names."""
    rows = _md_table_rows(block)
    sep_idx = _separator_row_index(rows)
    if sep_idx is None or sep_idx < 2:
        return {"status": _LABEL_ASSIGNMENT_STATUS_EXPLICIT, "rows": []}

    header_rows = rows[:sep_idx]
    width = max((len(row) for row in header_rows), default=0)
    if width < 3:
        return {"status": _LABEL_ASSIGNMENT_STATUS_EXPLICIT, "rows": []}

    padded_headers = [list(row) + [""] * max(0, width - len(row)) for row in header_rows]
    row_payloads: List[Dict[str, Any]] = []

    for header_idx, header_row in enumerate(padded_headers[:-1]):
        lower_rows = padded_headers[header_idx + 1 :]
        subordinate_columns_before_first_label = 0

        for col_idx, cell in enumerate(header_row):
            text = normalize_markdown_cell(cell)
            lower_text = " ".join(
                normalize_markdown_cell(row[col_idx])
                for row in lower_rows
                if col_idx < len(row)
            )

            if _labelish_cell(text):
                if subordinate_columns_before_first_label > 0:
                    row_payloads.append({
                        "header_row": header_idx,
                        "status": _LABEL_ASSIGNMENT_STATUS_UNCERTAIN,
                        "reason": "upper_header_label_after_unlabeled_subheaders",
                    })
                break
            if not text and _lower_header_cell_is_subheader(lower_text):
                subordinate_columns_before_first_label += 1
    return {
        "status": _combine_label_assignment_status(
            [str(payload.get("status") or "") for payload in row_payloads]
        ),
        "rows": row_payloads,
    }

def _label_assignment_decision(block: str) -> Dict[str, Any]:
    row_status = _row_label_assignment_status(block)
    column_status = _column_label_assignment_status(block)
    combined = _combine_label_assignment_status([
        str(row_status.get("status") or ""),
        str(column_status.get("status") or ""),
    ])
    reasons: List[str] = []
    if row_status.get("status") == _LABEL_ASSIGNMENT_STATUS_UNCERTAIN:
        reasons.append("uncertain_row_label_assignment")
    if column_status.get("status") == _LABEL_ASSIGNMENT_STATUS_UNCERTAIN:
        reasons.append("uncertain_column_label_assignment")
    return {
        "label_assignment_status": combined,
        "row_label_assignment_status": row_status.get("status") or _LABEL_ASSIGNMENT_STATUS_EXPLICIT,
        "column_label_assignment_status": column_status.get("status") or _LABEL_ASSIGNMENT_STATUS_EXPLICIT,
        "row_label_assignment_columns": row_status.get("columns") or [],
        "column_label_assignment_rows": column_status.get("rows") or [],
        "reasons": reasons,
    }


def _label_assignment_repair_reasons(block: str) -> List[str]:
    return list(_label_assignment_decision(block).get("reasons") or [])

def _candidate_grid_repair_reasons(block: str) -> List[str]:
    reasons: List[str] = []
    for source in (
        _markdown_row_width_repair_reasons(block),
        _single_cell_dense_collapse_reasons(block),
        _split_plus_minus_value_reasons(block),
        _neighboring_value_collapse_reasons(block),
        _caption_absorbed_reasons(block),
        _label_assignment_repair_reasons(block),
    ):
        for reason in source:
            if reason not in reasons:
                reasons.append(reason)
    return reasons


def _bbox_area_and_dims(bbox: Any) -> Tuple[float, float, float]:
    """Return area, width, height for either Docling dict bbox or PyMuPDF list bbox."""
    try:
        if isinstance(bbox, dict):
            l = float(bbox["l"])
            r = float(bbox["r"])
            t = float(bbox["t"])
            b = float(bbox["b"])
        elif isinstance(bbox, (list, tuple)) and len(bbox) >= 4:
            l = float(bbox[0])
            b = float(bbox[1])
            r = float(bbox[2])
            t = float(bbox[3])
        else:
            return 0.0, 0.0, 0.0
    except Exception:
        return 0.0, 0.0, 0.0

    width = abs(r - l)
    height = abs(t - b)
    return width * height, width, height


def _candidate_bbox_repair_reasons(
    *,
    name: str,
    payload: Optional[Dict[str, Any]],
    docling_bbox: Optional[Dict[str, Any]],
) -> List[str]:
    """Reject external extraction candidates whose bbox disagrees with Docling.

    Keep this deliberately simple. The most reliable giveaway is often not the
    prose itself, but that the detected region is much larger or much smaller
    than the Docling table bbox.
    """
    if name == "docling_original":
        return []
    if not payload or not docling_bbox:
        return []

    candidate_area, candidate_width, candidate_height = _bbox_area_and_dims(payload.get("bbox"))
    docling_area, docling_width, docling_height = _bbox_area_and_dims(docling_bbox)
    if candidate_area <= 0 or docling_area <= 0:
        return []

    area_ratio = candidate_area / docling_area
    height_ratio = candidate_height / docling_height if docling_height > 0 else 0.0
    width_ratio = candidate_width / docling_width if docling_width > 0 else 0.0

    if area_ratio >= 2.25 or height_ratio >= 2.25 or width_ratio >= 2.25:
        return [f"{name}_bbox_much_larger_than_docling_table_bbox"]
    if area_ratio <= 0.45 or height_ratio <= 0.45 or width_ratio <= 0.45:
        return [f"{name}_bbox_much_smaller_than_docling_table_bbox"]

    return []


def _attempt_summary(name: str, payload: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    payload = payload or {}
    summary: Dict[str, Any] = {
        "name": name,
        "status": payload.get("status") or "unknown",
        "attempted": payload.get("attempted"),
        "strategy": payload.get("strategy"),
        "stream_mode": payload.get("stream_mode"),
        "reason": payload.get("reason"),
        "error": payload.get("error"),
        "produced_block": bool(str(payload.get("block") or "").strip()),
    }
    for key in (
        "tables_found",
        "shape",
        "bbox",
        "pymupdf_clip",
        "pymupdf_clip_source",
        "page_no",
        "camelot_area",
        "docling_columns",
        "docling_columns_source",
        "stream_attempt",
    ):
        if key in payload:
            summary[key] = payload.get(key)
    return summary

def _normalized_md_cells(cells: List[str]) -> List[str]:
    return [normalize_markdown_cell(cell).casefold() for cell in cells]


def _body_already_has_header(table_body: str, header_rows: List[List[str]]) -> bool:
    pipe_lines = _body_pipe_lines(table_body)
    if not pipe_lines or not header_rows:
        return False
    if len(pipe_lines) < len(header_rows):
        return False
    for idx, header in enumerate(header_rows):
        if _normalized_md_cells(_split_md_row(pipe_lines[idx])) != _normalized_md_cells(header):
            return False
    return True


def _header_width(header_rows: Optional[List[List[str]]]) -> int:
    if not header_rows:
        return 0
    return max((len(row) for row in header_rows), default=0)


def _drop_column_from_rows(rows: List[List[str]], drop_idx: int) -> List[List[str]]:
    out: List[List[str]] = []
    for row in rows:
        if drop_idx < len(row):
            out.append(row[:drop_idx] + row[drop_idx + 1 :])
        else:
            out.append(list(row))
    return out


def _project_header_rows_for_candidate_width(
    header_rows: Optional[List[List[str]]],
    observed_width: int,
    *,
    optional_blank_column_re: re.Pattern[str] = DEFAULT_OPTIONAL_BLANK_COLUMN_RE,
) -> Tuple[Optional[List[List[str]]], Optional[int], str]:
    expected_width = _header_width(header_rows)
    if not header_rows or expected_width <= 1 or observed_width <= 1:
        return None, None, "none"
    if observed_width == expected_width:
        return header_rows, None, "carried_header"
    if observed_width == expected_width - 1:
        first_header = header_rows[0]
        for idx, cell in enumerate(first_header):
            if optional_blank_column_re.search(normalize_markdown_cell(cell)):
                return _drop_column_from_rows(header_rows, idx), idx, "projected_header_drop_optional_blank_column"
    return None, None, "none"


def _inject_carried_table_header(table_body: str, header_rows: List[List[str]]) -> str:
    if not table_body.strip() or not header_rows:
        return table_body
    if _body_already_has_header(table_body, header_rows):
        return table_body
    pipe_lines = _body_pipe_lines(table_body)
    rows = [_split_md_row(line) for line in pipe_lines]
    sep_idx = _separator_row_index(rows)
    if sep_idx is None:
        data_lines = pipe_lines
    else:
        data_lines = pipe_lines[:sep_idx] + pipe_lines[sep_idx + 1 :]
    width = len(header_rows[0])
    normalized_header = [_cells_to_md_row(row + [""] * (width - len(row))) for row in header_rows]
    separator = _cells_to_md_row(["---"] * width)
    return "\n".join([normalized_header[0], separator, *normalized_header[1:], *data_lines]).strip()


def _prepare_repair_candidate_block(
    block: str,
    header_rows: Optional[List[List[str]]],
    *,
    caption_text: str = "",
    footnote_text: str = "",
) -> Tuple[str, Dict[str, Any]]:
    pipe_body = "\n".join(_body_pipe_lines(block)).strip()
    if not pipe_body:
        return block.strip(), {"header_action": "none", "dropped_header_col_idx": None}
    observed_rows = _md_table_rows(pipe_body)
    sep_idx = _separator_row_index(observed_rows)
    observed_width = len(observed_rows[sep_idx]) if sep_idx is not None else max((len(row) for row in observed_rows), default=0)

    if header_rows and _body_already_has_header(pipe_body, header_rows):
        prepared = markdown_block_with_caption(pipe_body, caption_text=caption_text, footnote_text=footnote_text).rstrip()
        return prepared, {"header_action": "already_had_carried_header", "dropped_header_col_idx": None}

    candidate_header_rows, dropped_idx, header_action = _project_header_rows_for_candidate_width(
        header_rows,
        observed_width,
    )
    if candidate_header_rows:
        injected_body = _inject_carried_table_header(pipe_body, candidate_header_rows)
        prepared = markdown_block_with_caption(injected_body, caption_text=caption_text, footnote_text=footnote_text).rstrip()
        return prepared, {"header_action": header_action, "dropped_header_col_idx": dropped_idx}

    prepared = markdown_block_with_caption(pipe_body, caption_text=caption_text, footnote_text=footnote_text).rstrip()
    return prepared, {"header_action": "none", "dropped_header_col_idx": None}


def grid_to_markdown(grid: List[List[Any]]) -> str:
    rows = [[normalize_markdown_cell(cell) for cell in row] for row in (grid or [])]
    while rows and not any(rows[0]):
        rows.pop(0)
    while rows and not any(rows[-1]):
        rows.pop()
    if not rows:
        return ""
    n_cols = max(len(row) for row in rows)
    first_row = list(rows[0]) + [""] * (n_cols - len(rows[0]))
    lines = [
        "| " + " | ".join(_escape_md(cell) for cell in first_row[:n_cols]) + " |",
        "| " + " | ".join("---" for _ in range(n_cols)) + " |",
    ]
    for row in rows[1:]:
        row = list(row) + [""] * (n_cols - len(row))
        lines.append("| " + " | ".join(_escape_md(cell) for cell in row[:n_cols]) + " |")
    return "\n".join(lines).rstrip() + "\n"




def _header_quality_score(block: str) -> int:
    rows = _md_table_rows(block)
    sep_idx = _separator_row_index(rows)
    if sep_idx is None or sep_idx == 0:
        return 0
    header_rows = rows[:sep_idx]
    score = 0
    for row in header_rows:
        nonempty = [normalize_markdown_cell(cell) for cell in row if normalize_markdown_cell(cell)]
        if not nonempty:
            continue
        alpha_cells = sum(bool(re.search(r"[A-Za-z]", cell)) for cell in nonempty)
        numeric_cells = sum(_independent_numeric_count(cell) > 0 for cell in nonempty)
        if alpha_cells >= max(1, len(nonempty) // 2) and numeric_cells <= max(1, len(nonempty) // 3):
            score += 1
    return score

def _candidate_header_rows(block: str) -> List[List[str]]:
    rows = _md_table_rows(block)
    sep_idx = _separator_row_index(rows)
    if sep_idx is None or sep_idx == 0:
        return []
    return rows[:sep_idx]


def _candidate_body_rows(block: str) -> List[List[str]]:
    rows = _md_table_rows(block)
    sep_idx = _separator_row_index(rows)
    if sep_idx is None:
        return []
    return rows[sep_idx + 1 :]


def _coverage_text(value: str) -> str:
    """Normalize table text for known-content coverage checks."""
    text = normalize_markdown_cell(value)
    text = text.casefold()
    text = re.sub(r"(?:GLYPH<0>|GLYPH&lt;0&gt;)", "-", text)
    text = re.sub(r"GLYPH<\d+>", "", text)
    text = text.replace("\u2212", "-").replace("\u2013", "-").replace("\u2014", "-")
    return text


def _coverage_tokens(value: str) -> List[str]:
    """Return comparable content tokens.

    These are intentionally simple content tokens.  They are used only to ask:
    did a repair candidate preserve known Docling table content somewhere?
    """
    text = _coverage_text(value)
    return re.findall(r"[a-z]+|\d+(?:\.\d+)?", text)

def _meaningful_coverage_tokens(value: str) -> List[str]:
    """Coverage tokens, excluding weak one-letter footnote markers."""
    return [
        token
        for token in _coverage_tokens(value)
        if len(token) > 1 or token.isdigit()
    ]


def _coverage_compact(value: str) -> str:
    """Compact one cell for spacing-insensitive matching.

    This is intentionally applied cell-by-cell. It treats "SAC a" and
    "SACa" as equivalent, but does not reconstruct a word split across
    neighboring cells such as "Carbon v" + "ariants".
    """
    return re.sub(r"[^a-z0-9]+", "", _coverage_text(value))


def _coverage_token_key(token: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", _coverage_text(token))


def _coverage_token_count_in_cell(token: str, cell: str) -> int:
    token_key = _coverage_token_key(token)
    if not token_key:
        return 0
    cell_key = _coverage_compact(cell)
    if not cell_key:
        return 0
    return len(re.findall(re.escape(token_key), cell_key))


def _missing_ref_cell_tokens(ref_cell: str, cand_cell: str) -> int:
    """Count missing reference tokens against one candidate cell.

    Matching is compact and therefore spacing-insensitive inside a cell, but it
    never joins text from separate candidate cells.
    """
    ref_counts = Counter(_meaningful_coverage_tokens(ref_cell))
    missing = 0
    for token, count in ref_counts.items():
        missing += max(0, count - _coverage_token_count_in_cell(token, cand_cell))
    return missing



_REFERENCE_UNIT_TOKEN_RE = re.compile(
    r"\b(?:[a-zµμ]+(?:\s*\d+)?)\s*(?:/\s*[a-zµμ]+(?:\s*\d+)?)+\b",
    re.I,
)


def _reference_lexical_token_keys(reference_block: str) -> set[str]:
    """Return reference words and scientific units whose splits are unsafe.

    Four-letter words include meaningful short labels such as ``coal`` while
    excluding most one-letter subscripts and common notation. Phrase-level
    stacking such as "Water" | "content" remains acceptable because each word
    is intact. Compact unit expressions (for example ``cm 3/g``) are included
    too: breaking them across cells changes the column schema.
    """
    tokens: set[str] = set()
    for row in _candidate_header_rows(reference_block) + _candidate_body_rows(reference_block):
        for cell in row:
            for token in _meaningful_coverage_tokens(cell):
                key = _coverage_token_key(token)
                if len(key) >= 4 and re.fullmatch(r"[a-z]+", key):
                    tokens.add(key)
            for unit in _REFERENCE_UNIT_TOKEN_RE.findall(_coverage_text(cell)):
                key = _coverage_compact(unit)
                if len(key) >= 3:
                    tokens.add(key)
    return tokens


def _is_complete_unit_cell(cell: str) -> bool:
    """Return whether a cell already contains a complete compound unit.

    A table can legitimately place the same unit in adjacent cells.  Their
    touching letter sequences may coincidentally spell another reference unit
    (for example ``ng/mg`` | ``ng/mg`` contains the boundary text ``mg/ng``).
    Such a boundary cannot be evidence of a broken unit.
    """
    # Units are commonly enclosed in parentheses, but the punctuation is not
    # part of the unit itself (``(ng/mg)`` -> ``ng/mg``).
    text = re.sub(r"^[^a-z0-9µμ]+|[^a-z0-9µμ]+$", "", _coverage_text(cell))
    return bool(_REFERENCE_UNIT_TOKEN_RE.fullmatch(text))


def _lexical_fragmentation_penalty(reference_block: str, candidate_block: str) -> int:
    """Penalize true word splits across adjacent cells.

    This does not penalize phrase/label splitting where each word remains intact.
    It only fires when a reference word is recoverable by concatenating adjacent
    candidate cells but is not present inside any one candidate cell. Each
    ``(row, column boundary, reference token)`` is counted once, even when
    overlapping two- and three-cell windows expose the same split.
    """
    ref_tokens = _reference_lexical_token_keys(reference_block)
    if not ref_tokens:
        return 0

    events: set[Tuple[str, int, int, str]] = set()
    row_groups = (
        ("header", _candidate_header_rows(candidate_block)),
        ("body", _candidate_body_rows(candidate_block)),
    )
    for section, rows in row_groups:
        for row_idx, row in enumerate(rows, start=1):
            for boundary_col in range(1, len(row)):
                left_cell = row[boundary_col - 1]
                if _is_complete_unit_cell(left_cell):
                    continue
                left_parts = re.findall(r"[a-z0-9]+", _coverage_text(left_cell))
                if not left_parts:
                    continue
                left_fragment = left_parts[-1]
                # A true broken word begins with the final lexical fragment in
                # the left cell. This anchor prevents accidental substrings
                # across two intact words, e.g. "adsorption" | "Soil" forming
                # the unrelated reference token "ions".
                right_text = "".join(
                    _coverage_compact(cell)
                    for cell in row[boundary_col : boundary_col + 2]
                )
                if not right_text:
                    continue
                for token in ref_tokens:
                    if (
                        len(left_fragment) >= len(token)
                        or not token.startswith(left_fragment)
                        or not right_text.startswith(token[len(left_fragment) :])
                    ):
                        continue
                    events.add((section, row_idx, boundary_col, token))
    return len(events)


# A single split may be a harmless PDF-layout artifact, but two distinct
# recovered token/boundary events show that the candidate's column boundaries
# are not trustworthy. Do not automatically select such a grid even if its
# headers look more complete than another candidate's.
_SYSTEMATIC_LEXICAL_FRAGMENTATION_THRESHOLD = 2


def _lexical_fragmentation_repair_reasons(penalty: int) -> List[str]:
    if int(penalty or 0) >= _SYSTEMATIC_LEXICAL_FRAGMENTATION_THRESHOLD:
        return ["systematic_word_fragmentation_across_cells"]
    return []

def _missing_ref_cell_tokens_against_best_cell(ref_cell: str, cand_cells: List[str]) -> int:
    if not _meaningful_coverage_tokens(ref_cell):
        return 0
    if not cand_cells:
        return len(_meaningful_coverage_tokens(ref_cell))
    return min(_missing_ref_cell_tokens(ref_cell, cand_cell) for cand_cell in cand_cells)


def _reference_header_spans(row: List[str]) -> List[Tuple[int, int]]:
    """Return consecutive same-text header spans in a rendered reference row.

    Docling often renders a merged header as repeated cells. For scoring, treat
    each consecutive repeated run as one logical header cell.
    """
    spans: List[Tuple[int, int]] = []
    idx = 0
    while idx < len(row):
        current_key = _coverage_compact(row[idx])
        end = idx + 1
        if current_key:
            while end < len(row) and _coverage_compact(row[end]) == current_key:
                end += 1
        spans.append((idx, end))
        idx = end
    return spans



def _header_missing_reference_tokens(reference_block: str, candidate_block: str) -> int:
    """Count known Docling header tokens missing from the candidate header.
    Matching is cell-aware. Compact differences inside one cell are ignored
    ("Matrix a" ~= "Matrixa"), but fragments in separate cells are not joined.
    Consecutive repeated reference headers are treated as one logical merged
    header span to avoid multiplying the same missing-token penalty.
    """
    ref_headers = _candidate_header_rows(reference_block)
    cand_headers = _candidate_header_rows(candidate_block)
    if not ref_headers or not cand_headers:
        return 0

    ref_width = max((len(row) for row in ref_headers), default=0)
    cand_width = max((len(row) for row in cand_headers), default=0)
    missing = 0

    if ref_width == cand_width and ref_width > 0:
        for row_idx, ref_row in enumerate(ref_headers):
            cand_row = cand_headers[row_idx] if row_idx < len(cand_headers) else []
            for start, end in _reference_header_spans(ref_row):
                ref_cell = ref_row[start] if start < len(ref_row) else ""
                if not normalize_markdown_cell(ref_cell):
                    continue
                # If Docling repeated a merged header across columns, allow the
                # candidate to put the complete header in any one cell in that
                # span. Do not join candidate cells, because that would hide
                # true word fragmentation: "Carbon v" + "ariants".
                cand_cells = [cand_row[col] if col < len(cand_row) else "" for col in range(start, end)]
                missing += _missing_ref_cell_tokens_against_best_cell(ref_cell, cand_cells)
        return missing

    cand_cells = [
        cell
        for row in cand_headers
        for cell in row
        if normalize_markdown_cell(cell)
    ]
    for ref_row in ref_headers:
        for start, _end in _reference_header_spans(ref_row):
            ref_cell = ref_row[start] if start < len(ref_row) else ""
            if not normalize_markdown_cell(ref_cell):
                continue
            missing += _missing_ref_cell_tokens_against_best_cell(ref_cell, cand_cells)
    return missing

def _coverage_cell_key(value: str) -> str:
    """Canonical key for detecting repeated merged-cell fill values."""
    return _coverage_compact(value)


def _coverage_cell_is_text_label(value: str) -> bool:
    """Return True for text-like labels that may be repeated from merged cells.

    Numeric values are deliberately not treated as merge-fill labels: repeated
    measurements in adjacent rows/columns may be real independent values and
    should still be counted with multiplicity.
    """
    text = normalize_markdown_cell(value)
    return bool(text and re.search(r"[A-Za-z]", text) and _coverage_cell_key(text))


def _same_coverage_cell(a: str, b: str) -> bool:
    a_key = _coverage_cell_key(a)
    return bool(a_key and a_key == _coverage_cell_key(b))


def _header_label_keys_by_column(header_rows: List[List[str]], width: int) -> List[set[str]]:
    keys_by_col: List[set[str]] = [set() for _ in range(max(0, width))]
    for row in header_rows:
        padded = list(row) + [""] * max(0, width - len(row))
        for col_idx, cell in enumerate(padded[:width]):
            if _coverage_cell_is_text_label(cell):
                keys_by_col[col_idx].add(_coverage_cell_key(cell))
    return keys_by_col


def _reference_body_cells_for_coverage(reference_block: str) -> List[str]:
    """Return reference body cells after removing merged-cell fill artifacts.

    Some extractors materialize merged cells by repeating the same label in all
    covered cells; others put the label only once and leave the rest blank. For
    coverage scoring these are equivalent. Therefore, only text-like label
    cells are de-duplicated across horizontal/vertical merge-fill patterns;
    numeric/value cells retain multiplicity.
    """
    header_rows = _candidate_header_rows(reference_block)
    body_rows = _candidate_body_rows(reference_block)
    width = max(
        max((len(row) for row in header_rows), default=0),
        max((len(row) for row in body_rows), default=0),
    )
    header_keys_by_col = _header_label_keys_by_column(header_rows, width)
    previous_label_key_by_col: Dict[int, str] = {}
    cells: List[str] = []

    for raw_row in body_rows:
        row = list(raw_row) + [""] * max(0, width - len(raw_row))
        for start, end in _reference_header_spans(row):
            ref_cell = row[start] if start < len(row) else ""
            if not normalize_markdown_cell(ref_cell):
                continue

            if not _coverage_cell_is_text_label(ref_cell):
                # Keep multiplicity for numeric/value cells, even when adjacent
                # values happen to be identical.
               for col_idx in range(start, min(end, width)):
                    cell = row[col_idx] if col_idx < len(row) else ""
                    if normalize_markdown_cell(cell):
                        cells.append(cell)
               continue

            key = _coverage_cell_key(ref_cell)
            cols = list(range(start, min(end, width))) or [start]

            # A body cell repeated from the header in the same column usually
            # means a vertical merge spanning header rows. Its contents are
            # already checked by _header_missing_reference_tokens().
            if any(col < len(header_keys_by_col) and key in header_keys_by_col[col] for col in cols):
                continue

            # A repeated text label in the same column is a vertical merged-cell
            # fill. Count the logical label once, not once per materialized row.
            if any(previous_label_key_by_col.get(col) == key for col in cols):
                continue

            cells.append(ref_cell)
            for col in cols:
                previous_label_key_by_col[col] = key

    return cells


def _body_missing_reference_tokens(reference_block: str, candidate_block: str) -> int:
    """Count known Docling body tokens missing from the candidate body.

    Counts are still corpus-level so row expansion/splitting is allowed, but
    candidate evidence is gathered cell-by-cell with compact matching. This
    preserves "SAC a" ~= "SACa" and "V micro" ~= "Vmicro", while preventing
    a token from being reconstructed across adjacent candidate cells. Repeated
    text labels caused by merged-cell fill are treated as one logical value.
    """
    ref_counts: Counter[str] = Counter()
    for cell in _reference_body_cells_for_coverage(reference_block):
        ref_counts.update(_meaningful_coverage_tokens(cell))

    cand_cells = [
        cell
        for row in _candidate_body_rows(candidate_block)
        for cell in row
        if normalize_markdown_cell(cell)
    ]

    missing = 0
    for token, count in ref_counts.items():
        supported = sum(_coverage_token_count_in_cell(token, cell) for cell in cand_cells)
        missing += max(0, count - supported)
    return missing

def _compact_text_for_match(value: str) -> str:
    """Normalize text for fuzzy containment checks against known outside text.

    This intentionally removes punctuation/spacing so a footnote split across
    PDF-tool columns can still match its clean Docling/nearby-text form.
    """
    return re.sub(r"[^0-9a-z]+", "", normalize_markdown_cell(value).casefold())


def _known_outside_text_sources(
    *,
    caption_text: str = "",
    footnote_text: str = "",
    known_outside_texts: Optional[List[str]] = None,
) -> List[Tuple[str, str]]:
    sources: List[Tuple[str, str]] = []
    for label, text in (("caption", caption_text), ("footnote", footnote_text)):
        compact = _compact_text_for_match(text)
        if len(compact) >= 24:
            sources.append((label, compact))
    for idx, text in enumerate(known_outside_texts or [], start=1):
        compact = _compact_text_for_match(text)
        if len(compact) >= 24:
            sources.append((f"known_outside_{idx}", compact))

    # Preserve order but avoid counting the same evidence twice.
    deduped: List[Tuple[str, str]] = []
    seen: set[str] = set()
    for label, compact in sources:
        if compact in seen:
            continue
        seen.add(compact)
        deduped.append((label, compact))
    return deduped


def _known_outside_grid_contamination(
    block: str,
    *,
    caption_text: str = "",
    footnote_text: str = "",
    known_outside_texts: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Detect candidate table-grid rows that are known to belong outside.

    Unknown extra candidate content is deliberately neutral. This only reports
    rows whose joined row text is a reasonably long substring of known caption,
    footnote, or caller-provided nearby outside text.
    """
    sources = _known_outside_text_sources(
        caption_text=caption_text,
        footnote_text=footnote_text,
        known_outside_texts=known_outside_texts,
    )
    if not sources:
        return {"count": 0, "rows": []}

    contaminated_rows: List[Dict[str, Any]] = []
    for row_idx, row in enumerate(_candidate_body_rows(block), start=1):
        row_text = " ".join(
            normalize_markdown_cell(cell)
            for cell in row
            if normalize_markdown_cell(cell)
        )
        row_compact = _compact_text_for_match(row_text)
        if len(row_compact) < 18:
            continue
        for label, outside_compact in sources:
            if row_compact in outside_compact or outside_compact in row_compact:
                contaminated_rows.append({
                    "row": row_idx,
                    "source": label,
                    "text": row_text[:240],
                })
                break

    return {"count": len(contaminated_rows), "rows": contaminated_rows}



TABLE_REPAIRABILITY_ALLOWED_STATUSES = {"clean", "table_repairable", "table_repair_needed", "uncertain"}
TABLE_REPAIRABILITY_REVIEW_STATUSES = {"table_repair_needed", "uncertain"}


def normalize_table_repairability_status(value: Any) -> str:
    status = re.sub(r"[\s\-]+", "_", str(value or "").strip().lower())
    legacy_map = {
        "position_aligned": "table_repairable",
        "position_misaligned": "table_repair_needed",
    }
    status = legacy_map.get(status, status)
    return status if status in TABLE_REPAIRABILITY_ALLOWED_STATUSES else "uncertain"


_REPAIR_REASON_WEIGHTS = {
    # Header/label assignment errors can make otherwise correct values
    # semantically unusable.  Treat them as at least as severe as value collapse.
    "uncertain_row_label_assignment": 800,
    "uncertain_column_label_assignment": 800,
    "row_label_value_glue": 800,
    "repeated_row_label_simple_value_glue": 800,
    "row_label_value_glue_requires_pdf_repair": 800,
    "table_grid_suspected_corrupt": 800,
    "row_label_leakage": 700,
    "adjacent_column_x_order_inversion": 700,
    "header_only_blank_body_rows": 700,
    "markdown_header_only_blank_body_rows": 700,
    "data_row_used_as_markdown_header": 650,
    "single_spanning_header_only_table": 650,
    "markdown_has_no_meaningful_data_rows": 650,
    "neighboring_value_collapse": 500,
    "docling_compacted_body_span": 500,
    "dense_value_cell_collapsed_records": 500,
    "single_cell_dense_text_collapse": 500,
    "single_cell_numeric_collapse": 500,
    "split_plus_minus_values_across_cells": 400,
    "systematic_word_fragmentation_across_cells": 800,
    "row_width_mismatch": 350,
    "header_width_mismatch": 300,
    "caption_text_absorbed_as_table_row": 250,
    "continuation_header_missing_due_to_column_mismatch": 250,
    "continuation_near_miss_column_count": 250,
    "continuation_starts_with_data_row_instead_of_header": 250,
    "header_not_compatible_with_expected_schema": 250,
    "missing_markdown_table_grid": 250,
}


def _unique_repair_reasons(*sources: Optional[List[str]]) -> List[str]:
    reasons: List[str] = []
    for source in sources:
        for reason in source or []:
            reason = str(reason or "")
            if reason and reason not in reasons:
                reasons.append(reason)
    return reasons


def _repair_reason_penalty_for_reasons(reasons: List[str]) -> int:
    seen: set[str] = set()
    penalty = 0
    for reason in reasons:
        reason = str(reason or "")
        if not reason or reason in seen:
            continue
        seen.add(reason)
        penalty += _REPAIR_REASON_WEIGHTS.get(reason, 100)
    return penalty


def _candidate_decision_reasons(candidate: Dict[str, Any]) -> List[str]:
    quality = candidate.get("quality") or {}
    return _unique_repair_reasons(
        list(candidate.get("reasons") or []),
        quality.get("decision_repair_reasons") or [],
        quality.get("candidate_grid_repair_reasons") or [],
        quality.get("bbox_repair_reasons") or [],
        quality.get("schema_repair_reasons") or [],
        quality.get("external_repair_reasons") or [],
    )


def _candidate_repair_reason_penalty(candidate: Dict[str, Any]) -> int:
    return _repair_reason_penalty_for_reasons(_candidate_decision_reasons(candidate))


def make_table_quality_decision(
    candidate: Optional[Dict[str, Any]],
    *,
    repairability_status: Optional[str] = None,
    phase: str = "",
    default_corrupt_action: str = "repair",
) -> Dict[str, Any]:
    """Return one shared judgement object for table quality and next action.

    ``repairability_status`` preserves the existing pipeline vocabulary:
    ``table_repairable`` means the table is corrupt but Markdown-expandable,
    not that it is clean enough to accept.
    """
    candidate = candidate or {}
    quality = candidate.get("quality") or {}
    reasons = _candidate_decision_reasons(candidate)
    accepted = bool(candidate.get("accepted"))

    if repairability_status is None:
        repairability_status = "clean" if accepted and not reasons else "table_repair_needed"
    repairability_status = normalize_table_repairability_status(repairability_status)

    if repairability_status == "clean" and not reasons:
        quality_status = "clean"
        recommended_action = "use"
    elif repairability_status == "table_repairable":
        quality_status = "corrupt"
        recommended_action = "expand"
    elif repairability_status == "uncertain":
        quality_status = "uncertain"
        recommended_action = "manual_review"
    else:
        quality_status = "corrupt"
        recommended_action = default_corrupt_action

    return {
        "phase": phase,
        "candidate": candidate.get("name"),
        "quality_status": quality_status,
        "repairability_status": repairability_status,
        "recommended_action": recommended_action,
        "accepted": quality_status == "clean",
        "shape": candidate.get("shape"),
        "header_action": candidate.get("header_action"),
        "repair_reasons": reasons,
        "repair_reason_penalty": int(
            quality.get("repair_reason_penalty")
            if quality.get("repair_reason_penalty") is not None
            else _repair_reason_penalty_for_reasons(reasons)
        ),
    }


def _make_table_repair_candidate(
    *,
    name: str,
    block: str,
    payload: Optional[Dict[str, Any]],
    expected_header_rows: Optional[List[List[str]]],
    caption_text: str,
    footnote_text: str,
    docling_bbox: Optional[Dict[str, Any]] = None,
    known_outside_texts: Optional[List[str]] = None,
    reference_block: str = "",
    extra_repair_reasons: Optional[List[str]] = None,
) -> Dict[str, Any]:
    prepared_block, header_meta = _prepare_repair_candidate_block(
        block,
        expected_header_rows,
        caption_text=caption_text,
        footnote_text=footnote_text,
    )
    data_rows, n_cols = _markdown_table_shape(prepared_block)
    grid_reasons = _candidate_grid_repair_reasons(prepared_block)
    label_assignment = _label_assignment_decision(prepared_block)
    label_assignment_reasons = list(label_assignment.get("reasons") or [])
    bbox_reasons: List[str] = []
    for reason in _candidate_bbox_repair_reasons(
        name=name,
        payload=payload,
        docling_bbox=docling_bbox,
    ):
        if reason not in bbox_reasons:
            bbox_reasons.append(reason)
    has_expected_header = bool(expected_header_rows)
    if has_expected_header:
        schema_compatible = header_meta.get("header_action") != "none"
    else:
        schema_compatible = data_rows >= 1 and n_cols >= 2
    schema_reasons = [] if schema_compatible else ["header_not_compatible_with_expected_schema"]
    header_quality = _header_quality_score(prepared_block)
    missing_reference_header_tokens = (
        _header_missing_reference_tokens(reference_block, prepared_block)
        if reference_block.strip()
        else 0
    )
    missing_reference_body_tokens = (
        _body_missing_reference_tokens(reference_block, prepared_block)
        if reference_block.strip()
        else 0
    )
    lexical_fragmentation_penalty = (
        _lexical_fragmentation_penalty(reference_block, prepared_block)
        if reference_block.strip()
        else 0
    )
    reference_reasons = _lexical_fragmentation_repair_reasons(
        lexical_fragmentation_penalty
    )
    external_reasons = _unique_repair_reasons(extra_repair_reasons or [])
    decision_reasons = _unique_repair_reasons(
        grid_reasons,
        bbox_reasons,
        schema_reasons,
        reference_reasons,
        external_reasons,
    )
    accepted = bool(prepared_block.strip()) and not decision_reasons
    known_outside_contamination = _known_outside_grid_contamination(
        prepared_block,
        caption_text=caption_text,
        footnote_text=footnote_text,
        known_outside_texts=known_outside_texts,
    )
    repair_reason_penalty = _repair_reason_penalty_for_reasons(decision_reasons)
    quality_status = "clean" if accepted else "corrupt"
    return {
        "name": name,
        "status": "accepted_candidate" if accepted else "rejected_candidate",
        "accepted": accepted,
        "shape": [data_rows, n_cols],
        "header_action": header_meta.get("header_action"),
        "dropped_header_col_idx": header_meta.get("dropped_header_col_idx"),
        "reasons": [] if accepted else decision_reasons,
        "quality": {
            "quality_status": quality_status,
            "repairability_status": "clean" if accepted else "table_repair_needed",
            "recommended_action": "use" if accepted else "repair",
            "requires_repair": not accepted,
            "header_quality": header_quality,
            "columns": n_cols,
            "schema_compatible": schema_compatible,
            "repair_reason_penalty": repair_reason_penalty,
            "label_assignment_status": label_assignment.get("label_assignment_status"),
            "row_label_assignment_status": label_assignment.get("row_label_assignment_status"),
            "column_label_assignment_status": label_assignment.get("column_label_assignment_status"),
            "row_label_assignment_columns": label_assignment.get("row_label_assignment_columns") or [],
            "column_label_assignment_rows": label_assignment.get("column_label_assignment_rows") or [],
            "label_assignment_penalty": sum(
                _REPAIR_REASON_WEIGHTS.get(reason, 100)
                for reason in label_assignment_reasons
            ),
            "label_assignment_reasons": label_assignment_reasons,
            "missing_reference_header_tokens": missing_reference_header_tokens,
            "missing_reference_body_tokens": missing_reference_body_tokens,
            "lexical_fragmentation_penalty": lexical_fragmentation_penalty,
            "reference_repair_reasons": list(reference_reasons),
            "candidate_grid_status": "clean" if not grid_reasons else "table_repair_needed",
            "candidate_grid_repair_reasons": list(grid_reasons),
            "bbox_repair_reasons": list(bbox_reasons),
            "schema_repair_reasons": list(schema_reasons),
            "external_repair_reasons": list(external_reasons),
            "decision_repair_reasons": list(decision_reasons),
            "known_outside_contamination": int(known_outside_contamination.get("count") or 0),
            "known_outside_contamination_rows": known_outside_contamination.get("rows") or [],
        },
        "payload": {k: v for k, v in (payload or {}).items() if k not in {"block", "raw_block"}},
        "block": prepared_block,
    }

def _modal_width(candidates: List[Dict[str, Any]]) -> Optional[int]:
    widths = [int((candidate.get("shape") or [0, 0])[1] or 0) for candidate in candidates]
    widths = [width for width in widths if width > 1]
    if not widths:
        return None
    counts: Dict[int, int] = {}
    for width in widths:
        counts[width] = counts.get(width, 0) + 1
    best_count = max(counts.values())
    if best_count <= 1:
        return None
    return sorted(
        (item for item in counts.items() if item[1] == best_count),
        key=lambda item: -item[0],
    )[0][0]


def _reference_width(candidates: List[Dict[str, Any]], consensus_width: Optional[int]) -> Optional[int]:
    for candidate in candidates:
        if candidate.get("name") != "docling_original":
            continue
        quality = candidate.get("quality") or {}
        width = int((candidate.get("shape") or [0, 0])[1] or 0)
        if width > 1 and int(quality.get("header_quality") or 0) > 0:
            return width
    return consensus_width


def _candidate_quality_key(
    candidate: Dict[str, Any],
    consensus_width: Optional[int],
    reference_width: Optional[int],
) -> Tuple[int, ...]:
    quality = candidate.get("quality") or {}
    shape = candidate.get("shape") or [0, 0]
    n_cols = int(shape[1] or 0)
    header_action = str(candidate.get("header_action") or "none")
    # Schema rank is about evidence quality, not the extraction tool.
    if header_action in {"carried_header", "already_had_carried_header"}:
        schema_rank = 3
    elif header_action == "projected_header_drop_optional_blank_column":
        schema_rank = 2
    else:
        schema_rank = 1
    width_rank = 1 if consensus_width and n_cols == consensus_width else 0
    width_distance = -abs(n_cols - consensus_width) if consensus_width else 0
    reference_width_rank = 1 if reference_width and n_cols == reference_width else 0
    reference_width_distance = -abs(n_cols - reference_width) if reference_width else 0
    return (
        schema_rank,
        reference_width_rank,
        width_rank,
        -min(20, int(quality.get("known_outside_contamination") or 0)),
        -min(2000, _candidate_repair_reason_penalty(candidate)),
        -min(2000, int(
            quality.get("label_assignment_penalty")
            if quality.get("label_assignment_penalty") is not None
            else quality.get("ambiguous_label_assignment_penalty") or 0
        )),
        -min(300, int(quality.get("missing_reference_body_tokens") or 0)),
        -min(100, int(quality.get("lexical_fragmentation_penalty") or 0)),
        -min(100, int(quality.get("missing_reference_header_tokens") or 0)),
        int(quality.get("header_quality") or 0),
        reference_width_distance,
        width_distance,
    )


def select_best_table_repair_candidate(candidates: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    accepted = [candidate for candidate in candidates if candidate.get("accepted")]
    reviewable = [
        candidate
        for candidate in candidates
        if str(candidate.get("block") or "").strip()
        and int((candidate.get("shape") or [0, 0])[0] or 0) >= 1
        and int((candidate.get("shape") or [0, 0])[1] or 0) >= 2
    ]
    # If no candidate is accepted, the fallback selection is provisional and
    # exists only to support comparison/manual review. It must not be treated
    # as evidence that the candidate is suitable for automatic use.
    pool = accepted if accepted else reviewable
    if not pool:
        return None

    consensus_width = _modal_width(pool)
    reference_width = _reference_width(pool, consensus_width)
    for candidate in pool:
        candidate.setdefault("quality", {})["consensus_columns"] = consensus_width
        candidate.setdefault("quality", {})["reference_columns"] = reference_width
        candidate.setdefault("quality", {})["selection_key"] = list(
            _candidate_quality_key(candidate, consensus_width, reference_width)
        )
    # The final name key is only for deterministic reproducibility when all
    # quality signals are equal; it is not a source-quality preference.
    return max(
        pool,
        key=lambda candidate: (
            _candidate_quality_key(candidate, consensus_width, reference_width),
            str(candidate.get("name") or ""),
        ),
    )



escape_markdown_cell = _escape_md
split_markdown_row = _split_md_row
markdown_table_rows = _md_table_rows
markdown_table_shape = _markdown_table_shape
candidate_grid_repair_reasons = _candidate_grid_repair_reasons
prepare_repair_candidate_block = _prepare_repair_candidate_block
make_table_repair_candidate = _make_table_repair_candidate
candidate_repair_reason_penalty = _candidate_repair_reason_penalty
attempt_summary = _attempt_summary
label_assignment_decision = _label_assignment_decision


def build_repair_selection_payload(
    *,
    route: str,
    candidates: List[Dict[str, Any]],
    attempts: Optional[List[Dict[str, Any]]] = None,
    source_path: Optional[str] = None,
    source_path_key: str = "source_path",
    page_no: Optional[int] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    selected = select_best_table_repair_candidate(candidates)
    selected_clean = bool(selected and selected.get("accepted"))
    selected_quality_decision = make_table_quality_decision(
        selected,
        repairability_status="clean" if selected_clean else "table_repair_needed",
        phase=f"{route}_selection",
        default_corrupt_action="manual_review",
    ) if selected else None
    payload: Dict[str, Any] = {
        "status": "accepted" if selected_clean else "table_repair_needed",
        "route": route,
        "selection_mode": (
            "accepted"
            if selected_clean
            else "provisional_review" if selected else "none"
        ),
        "attempts": attempts or [],
        "candidates": candidates,
        "selected_candidate_payload": selected,
        "selected_candidate": selected.get("name") if selected else None,
        "accepted": selected_clean,
        "selected_candidate_accepted": selected_clean,
        "selected_candidate_is_provisional": bool(selected and not selected_clean),
        "selected_candidate_quality_decision": selected_quality_decision,
    }
    if source_path is not None:
        payload[source_path_key] = str(source_path)
    if page_no is not None:
        payload["page_no"] = page_no
    if extra:
        payload.update(extra)
    if selected:
        payload.update({
            "block": str(selected.get("block") or "").rstrip(),
            "selected_candidate_shape": selected.get("shape"),
            "selected_candidate_header_action": selected.get("header_action"),
            "selected_candidate_dropped_header_col_idx": selected.get("dropped_header_col_idx"),
            "selected_candidate_quality": selected.get("quality"),
        })
    return payload
