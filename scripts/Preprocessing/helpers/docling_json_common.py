from __future__ import annotations

import re
from typing import Any, Iterable


def normalize_space(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def ref_value(ref: Any) -> str:
    if isinstance(ref, dict):
        return str(ref.get("$ref") or "")
    return str(ref or "")


def parse_ref(ref: Any) -> tuple[str, int]:
    value = ref_value(ref)
    match = re.fullmatch(r"#/(texts|tables|pictures|groups)/(\d+)", value)
    if not match:
        raise ValueError(f"Unsupported Docling ref: {value}")
    return match.group(1), int(match.group(2))


def resolve_ref(doc: dict[str, Any], ref: Any) -> dict[str, Any]:
    collection, index = parse_ref(ref)
    items = doc.get(collection) or []
    if index < 0 or index >= len(items):
        raise IndexError(f"Docling ref out of range: {ref_value(ref)}")
    obj = items[index]
    if not isinstance(obj, dict):
        raise TypeError(f"Docling ref does not resolve to an object: {ref_value(ref)}")
    return obj


def iter_body_refs(doc: dict[str, Any]) -> Iterable[str]:
    """Yield body refs in reading order, including nested text-node children."""
    emitted_refs: set[str] = set()
    visited_groups: set[str] = set()

    def visit(ref: Any) -> Iterable[str]:
        value = ref_value(ref)
        if not value:
            return
        try:
            collection, _ = parse_ref(value)
            item = resolve_ref(doc, value)
        except Exception:
            return

        if collection == "groups":
            if value in visited_groups:
                return
            visited_groups.add(value)
            for child in item.get("children") or []:
                yield from visit(child)
            return

        if value in emitted_refs:
            return
        emitted_refs.add(value)
        yield value

        # DOCX-derived JSON often nests paragraphs, tables, or pictures under
        # section_header/text nodes. Preserve that local order.
        if collection == "texts":
            for child in item.get("children") or []:
                yield from visit(child)

    for child in ((doc.get("body") or {}).get("children") or []):
        yield from visit(child)


def _clean_cell_text(text: str) -> str:
    text = normalize_space(text)
    text = text.replace("\\", "\\\\").replace("|", r"\|")
    return text.replace("\n", "<br>")


def _render_group_inline(
    doc: dict[str, Any],
    group: dict[str, Any],
) -> str:
    parts: list[str] = []
    for child_ref in group.get("children") or []:
        child_value = ref_value(child_ref)
        try:
            child = resolve_ref(doc, child_ref)
        except Exception:
            continue
        if child_value.startswith("#/groups/"):
            rendered = _render_group_inline(doc, child)
        elif child_value.startswith("#/pictures/"):
            rendered = ""
        else:
            rendered = normalize_space(str(child.get("text") or child.get("orig") or ""))
        if rendered:
            parts.append(rendered)
    return " ".join(parts)


def table_markdown(table: dict[str, Any], doc: dict[str, Any]) -> str:
    data = table.get("data") or {}
    grid = data.get("grid") or []
    if not grid:
        return ""

    ref_by_coordinate: dict[tuple[int, int], str] = {}
    for cell in data.get("table_cells") or []:
        ref = ref_value(cell.get("ref"))
        if not ref:
            continue
        try:
            row = int(cell.get("start_row_offset_idx") or 0)
            col = int(cell.get("start_col_offset_idx") or 0)
        except Exception:
            continue
        ref_by_coordinate[(row, col)] = ref

    rows: list[list[str]] = []
    width = max((len(row) for row in grid), default=0)
    for row_index, raw_row in enumerate(grid):
        row: list[str] = []
        for col_index in range(width):
            cell = raw_row[col_index] if col_index < len(raw_row) else {}
            text = normalize_space(str((cell or {}).get("text") or ""))
            rich_ref = ref_by_coordinate.get((row_index, col_index))
            if rich_ref:
                try:
                    rich_item = resolve_ref(doc, rich_ref)
                except Exception:
                    rich_item = {}
                if isinstance(rich_item, dict):
                    rich_text = _render_group_inline(doc, rich_item)
                    if rich_text and not text:
                        text = rich_text
            row.append(_clean_cell_text(text))
        rows.append(row)

    if not rows or not width:
        return ""
    output = [
        "| " + " | ".join(rows[0]) + " |",
        "| " + " | ".join("---" for _ in range(width)) + " |",
    ]
    output.extend("| " + " | ".join(row) + " |" for row in rows[1:])
    return "\n".join(output)


def substantive_table(table: dict[str, Any]) -> bool:
    return any(
        normalize_space(str(cell.get("text") or ""))
        for cell in ((table.get("data") or {}).get("table_cells") or [])
    )
