"""Figure-caption context helpers for data-extraction prompts."""

from __future__ import annotations

import csv
import os
import re
from pathlib import Path

try:
    from ..workflow_config import FIGURE_TABLE_INFO_CSV
except ImportError:
    import sys

    _DATA_EXTRACTION_DIR = Path(__file__).resolve().parents[1]
    if str(_DATA_EXTRACTION_DIR) not in sys.path:
        sys.path.insert(0, str(_DATA_EXTRACTION_DIR))
    from workflow_config import FIGURE_TABLE_INFO_CSV

DEFAULT_FIGURE_TABLE_INFO_CSV = str(FIGURE_TABLE_INFO_CSV)

_CAPTION_LEAD_FIGID_PAT = re.compile(
    r"^\s*(?:\#{1,6}\s*)?(?:\*+\s*)?(?:fig(?:ure)?|scheme|exhibit)\.?\s*(?P<num>(?:S\s*\d+|\d+))\b",
    re.IGNORECASE,
)
_FIG_REF_LIST_PAT = (
    r"(?:S\s*\d+|\d+)(?:[A-Za-z])?"
    r"(?:\s*(?:(?:,\s*)?(?:and|&)\s*|,\s*|/\s*|\bto\b\s*|-|\u2013|\u2014)\s*"
    r"(?:S\s*\d+|\d+)(?:[A-Za-z])?)*"
)
_FIG_REF_BLOCK_PAT = re.compile(
    rf"\b(?:figures?|figs?|schemes?|exhibits?)\.?\s+({_FIG_REF_LIST_PAT})",
    re.IGNORECASE,
)
_FIG_TOKEN_PAT = re.compile(r"\b(S\s*\d+|\d+)(?:[A-Za-z])?\b", re.IGNORECASE)
_FIG_RANGE_PAT = re.compile(
    r"\b(?P<a>S\s*\d+|\d+)(?:[A-Za-z])?\s*(?:-|\u2013|\u2014|\bto\b)\s*"
    r"(?P<b>S\s*\d+|\d+)(?:[A-Za-z])?\b",
    re.IGNORECASE,
)

def _normalize_fig_id(tok: str) -> str:
    t = (tok or "").strip().upper()
    t = re.sub(r"^S\s+(\d+)$", r"S\1", t)
    if t.isdigit():
        return t
    if t.startswith("S") and t[1:].isdigit():
        return t
    return ""

def _expand_fig_range(a: str, b: str) -> list[str]:
    a0 = _normalize_fig_id(a)
    b0 = _normalize_fig_id(b)
    if not a0 or not b0:
        return []
    if a0.isdigit() and b0.isdigit():
        ia, ib = int(a0), int(b0)
        return [str(i) for i in range(ia, ib + 1)] if ia <= ib and (ib - ia) <= 50 else [a0, b0]
    if a0.startswith("S") and b0.startswith("S"):
        ia, ib = int(a0[1:]), int(b0[1:])
        return [f"S{i}" for i in range(ia, ib + 1)] if ia <= ib and (ib - ia) <= 50 else [a0, b0]
    return [a0, b0]

def _extract_referenced_fig_ids(text: str) -> set[str]:
    if not text:
        return set()
    refs: set[str] = set()
    for match in _FIG_REF_BLOCK_PAT.finditer(text):
        block = (match.group(1) or "").replace("\u2013", "-").replace("\u2014", "-")
        for rm in _FIG_RANGE_PAT.finditer(block):
            for tok in _expand_fig_range(rm.group("a") or "", rm.group("b") or ""):
                if nt := _normalize_fig_id(tok):
                    refs.add(nt)
        for tm in _FIG_TOKEN_PAT.finditer(block):
            if nt := _normalize_fig_id(tm.group(1) or ""):
                refs.add(nt)
    return refs

def _caption_sort_key(tok: str):
    t = _normalize_fig_id(tok)
    if t.isdigit():
        return (0, int(t))
    if t.startswith("S") and t[1:].isdigit():
        return (1, int(t[1:]))
    return (2, t)

def load_caption_index_for_study(
    study_id: str,
    *,
    csv_path: str = DEFAULT_FIGURE_TABLE_INFO_CSV,
    max_chars_per_caption: int = 2000,
) -> dict[str, str]:
    path = (csv_path or "").strip()
    if not path or not os.path.exists(path):
        return {}
    out: dict[str, str] = {}
    try:
        with open(path, "r", encoding="utf-8-sig", newline="") as f:
            for row in csv.DictReader(f):
                if not isinstance(row, dict):
                    continue
                if (row.get("study_folder") or "").strip() != study_id:
                    continue
                if (row.get("type") or "").strip().lower() != "figure":
                    continue
                caption = (row.get("caption") or "").strip()
                if not caption:
                    continue
                match = _CAPTION_LEAD_FIGID_PAT.match(caption)
                if not match:
                    continue
                fig_id = _normalize_fig_id(match.group("num") or "")
                if not fig_id or fig_id in out:
                    continue
                cap_trim = caption if len(caption) <= max_chars_per_caption else caption[:max_chars_per_caption].rstrip()
                out[fig_id] = f"- {cap_trim}"
    except Exception:
        return {}
    return out

def select_figure_captions_for_text(
    text: str,
    caption_index: dict[str, str],
    *,
    max_total_chars: int = 12000,
) -> str:
    lines = [
        caption_index[fig_id]
        for fig_id in sorted(_extract_referenced_fig_ids(text), key=_caption_sort_key)
        if fig_id in caption_index
    ]
    block = "\n".join(lines).strip()
    return block if len(block) <= max_total_chars else block[:max_total_chars].rstrip()

def select_figure_captions_for_entries(
    entries: list[dict],
    caption_index: dict[str, str],
    *,
    text_key: str = "enriched_text",
    max_total_chars: int = 12000,
) -> str:
    referenced: set[str] = set()
    for entry in entries or []:
        referenced |= _extract_referenced_fig_ids(entry.get(text_key) or "")
    lines = [
        caption_index[fig_id]
        for fig_id in sorted(referenced, key=_caption_sort_key)
        if fig_id in caption_index
    ]
    block = "\n".join(lines).strip()
    return block if len(block) <= max_total_chars else block[:max_total_chars].rstrip()
