"""Accessible tiered ranking table for the Streamlit recommender UI."""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import streamlit as st


_HTML = """<div id="ar-ranking-table" class="ar-ranking-root"></div>"""

_CSS = r"""
.ar-ranking-root {
    color: var(--st-text-color, #262730);
    font-family: var(--st-font, "Source Sans Pro", sans-serif);
    line-height: 1.35;
}
.ar-ranking-legend {
    align-items: center;
    display: flex;
    flex-wrap: wrap;
    gap: .45rem 1rem;
    margin: .1rem 0 .75rem;
}
.ar-ranking-key {
    align-items: center;
    display: inline-flex;
    font-size: .82rem;
    gap: .38rem;
    white-space: nowrap;
}
.ar-ranking-swatch {
    border: 1px solid color-mix(in srgb, var(--st-text-color, #262730) 24%, transparent);
    border-radius: 3px;
    display: inline-block;
    height: .85rem;
    width: 1.45rem;
}
.ar-family-ac { --ar-family-rgb: 217, 135, 170; }
.ar-family-resin { --ar-family-rgb: 243, 181, 111; }
.ar-family-cdp { --ar-family-rgb: 85, 185, 155; }
.ar-family-other { --ar-family-rgb: 155, 165, 175; }
.ar-ranking-swatch.ar-solid {
    background: rgba(var(--ar-family-rgb), .42);
}
.ar-ranking-swatch.ar-hatched {
    background-color: rgba(var(--ar-family-rgb), .08);
    background-image: repeating-linear-gradient(
        135deg,
        rgba(var(--ar-family-rgb), .62) 0,
        rgba(var(--ar-family-rgb), .62) 2px,
        transparent 2px,
        transparent 7px
    );
}
.ar-ranking-support-key {
    border-left: 1px solid color-mix(in srgb, var(--st-text-color, #262730) 18%, transparent);
    margin-left: .15rem;
    padding-left: 1rem;
}
.ar-ranking-section {
    margin: 0 0 .85rem;
}
.ar-ranking-section.ar-top-tier {
    background: color-mix(in srgb, var(--st-primary-color, #0e7c8a) 4%, transparent);
    border: 2px solid color-mix(in srgb, var(--st-primary-color, #0e7c8a) 55%, transparent);
    border-radius: 11px;
    padding: .7rem;
}
.ar-ranking-section-header {
    align-items: baseline;
    display: flex;
    flex-wrap: wrap;
    gap: .25rem .55rem;
    justify-content: space-between;
    margin: 0 .1rem .5rem;
}
.ar-ranking-section-title {
    font-size: .98rem;
    font-weight: 700;
}
.ar-ranking-count {
    background: color-mix(in srgb, var(--st-text-color, #262730) 8%, transparent);
    border-radius: 999px;
    font-size: .75rem;
    font-weight: 600;
    padding: .12rem .48rem;
}
.ar-ranking-section-caption {
    color: color-mix(in srgb, var(--st-text-color, #262730) 67%, transparent);
    flex-basis: 100%;
    font-size: .78rem;
}
.ar-ranking-scroll {
    border: 1px solid color-mix(in srgb, var(--st-text-color, #262730) 14%, transparent);
    border-radius: 8px;
    overflow-x: auto;
}
.ar-ranking-section.ar-top-tier .ar-ranking-scroll {
    border-color: color-mix(in srgb, var(--st-primary-color, #0e7c8a) 28%, transparent);
}
.ar-ranking-table {
    border-collapse: separate;
    border-spacing: 0;
    font-size: .86rem;
    min-width: 760px;
    width: 100%;
}
.ar-ranking-table th {
    background: var(--st-secondary-background-color, #f0f2f6);
    border-bottom: 1px solid color-mix(in srgb, var(--st-text-color, #262730) 17%, transparent);
    color: color-mix(in srgb, var(--st-text-color, #262730) 75%, transparent);
    font-size: .74rem;
    font-weight: 650;
    letter-spacing: .015em;
    padding: .48rem .58rem;
    text-align: left;
    white-space: nowrap;
}
.ar-ranking-table td {
    border-bottom: 1px solid color-mix(in srgb, var(--st-text-color, #262730) 10%, transparent);
    padding: .52rem .58rem;
    vertical-align: middle;
}
.ar-ranking-table tbody tr:last-child td {
    border-bottom: 0;
}
.ar-ranking-row {
    cursor: pointer;
    outline: none;
}
.ar-ranking-row.ar-support-moderate td {
    background-color: rgba(var(--ar-family-rgb), .25);
}
.ar-ranking-row.ar-support-thin td {
    background-color: rgba(var(--ar-family-rgb), .07);
    background-image: repeating-linear-gradient(
        135deg,
        rgba(var(--ar-family-rgb), .42) 0,
        rgba(var(--ar-family-rgb), .42) 2px,
        transparent 2px,
        transparent 8px
    );
}
.ar-ranking-row:hover td,
.ar-ranking-row:focus-visible td {
    box-shadow: inset 0 0 0 999px color-mix(in srgb, var(--st-text-color, #262730) 5%, transparent);
}
.ar-ranking-row.ar-selected td {
    box-shadow:
        inset 0 2px var(--st-primary-color, #0e7c8a),
        inset 0 -2px var(--st-primary-color, #0e7c8a);
}
.ar-ranking-row.ar-selected td:first-child {
    box-shadow:
        inset 2px 0 var(--st-primary-color, #0e7c8a),
        inset 0 2px var(--st-primary-color, #0e7c8a),
        inset 0 -2px var(--st-primary-color, #0e7c8a);
}
.ar-ranking-row.ar-selected td:last-child {
    box-shadow:
        inset -2px 0 var(--st-primary-color, #0e7c8a),
        inset 0 2px var(--st-primary-color, #0e7c8a),
        inset 0 -2px var(--st-primary-color, #0e7c8a);
}
.ar-ranking-rank,
.ar-ranking-score,
.ar-ranking-studies {
    font-variant-numeric: tabular-nums;
    white-space: nowrap;
}
.ar-ranking-product {
    font-weight: 650;
    min-width: 10rem;
}
.ar-ranking-basis,
.ar-ranking-limit {
    white-space: nowrap;
}
.ar-ranking-warnings {
    display: flex;
    flex-wrap: wrap;
    gap: .25rem;
    min-width: 10rem;
}
.ar-warning-tag {
    border-radius: 999px;
    display: inline-flex;
    font-size: .72rem;
    font-weight: 650;
    line-height: 1.25;
    padding: .18rem .42rem;
    white-space: nowrap;
}
.ar-warning-critical {
    background: rgba(179, 98, 89, .16);
    color: #944c45;
}
.ar-warning-info {
    background: rgba(57, 119, 132, .14);
    color: #2f6975;
}
.ar-warning-review {
    background: rgba(98, 122, 145, .14);
    color: #536a80;
}
@media (prefers-color-scheme: dark) {
    .ar-warning-critical { color: #e1a39d; }
    .ar-warning-info { color: #84c4cf; }
    .ar-warning-review { color: #adc1d4; }
}
.ar-sr-only {
    clip: rect(0, 0, 0, 0);
    clip-path: inset(50%);
    height: 1px;
    overflow: hidden;
    position: absolute;
    white-space: nowrap;
    width: 1px;
}
"""

_JS = r"""
export default function(component) {
  const { data, parentElement, setStateValue } = component;
  const root = parentElement.querySelector("#ar-ranking-table");
  root.replaceChildren();

  const make = (tag, className, text) => {
    const element = document.createElement(tag);
    if (className) element.className = className;
    if (text !== undefined && text !== null) element.textContent = String(text);
    return element;
  };
  const familyClass = {
    AC: "ar-family-ac",
    Resin: "ar-family-resin",
    CDP: "ar-family-cdp",
    Other: "ar-family-other",
  };
  const familyName = {
    AC: "Activated carbon (AC)",
    Resin: "Resin",
    CDP: "Cyclodextrin polymer (CDP)",
    Other: "Other",
  };

  const legend = make("div", "ar-ranking-legend");
  ["AC", "Resin", "CDP"].forEach((family) => {
    const key = make("span", "ar-ranking-key");
    key.append(make("span", `ar-ranking-swatch ar-solid ${familyClass[family]}`));
    key.append(document.createTextNode(familyName[family]));
    legend.append(key);
  });
  if (data.showOther) {
    const key = make("span", "ar-ranking-key");
    key.append(make("span", "ar-ranking-swatch ar-solid ar-family-other"));
    key.append(document.createTextNode("Other class"));
    legend.append(key);
  }
  const solid = make("span", "ar-ranking-key ar-ranking-support-key");
  solid.append(make("span", "ar-ranking-swatch ar-solid ar-family-other"));
  solid.append(document.createTextNode("Moderately supported"));
  legend.append(solid);
  const hatched = make("span", "ar-ranking-key");
  hatched.append(make("span", "ar-ranking-swatch ar-hatched ar-family-other"));
  hatched.append(document.createTextNode("Thinly supported"));
  legend.append(hatched);
  root.append(legend);

  const selectRow = (rowElement, key) => {
    root.querySelectorAll(".ar-ranking-row").forEach((element) => {
      const selected = element.dataset.key === key;
      element.classList.toggle("ar-selected", selected);
      element.setAttribute("aria-selected", selected ? "true" : "false");
    });
    setStateValue("selected_key", key);
  };

  const addCell = (tr, value, className = "") => {
    const cell = make("td", className, value === null || value === undefined || value === "" ? "—" : value);
    tr.append(cell);
    return cell;
  };

  (data.sections || []).forEach((sectionData) => {
    const section = make("section", `ar-ranking-section ${sectionData.kind === "top" ? "ar-top-tier" : ""}`);
    const header = make("div", "ar-ranking-section-header");
    header.append(make("div", "ar-ranking-section-title", sectionData.title));
    header.append(make("span", "ar-ranking-count", `${sectionData.rows.length} candidate${sectionData.rows.length === 1 ? "" : "s"}`));
    if (sectionData.caption) header.append(make("div", "ar-ranking-section-caption", sectionData.caption));
    section.append(header);

    const scroll = make("div", "ar-ranking-scroll");
    const table = make("table", "ar-ranking-table");
    const thead = document.createElement("thead");
    const heading = document.createElement("tr");
    ["Rank", "Product", "DOI studies", data.scoreLabel, ...(data.isMixture ? ["Limiting PFAS"] : []), "Basis", "Warnings"].forEach((label) => {
      heading.append(make("th", "", label));
    });
    thead.append(heading);
    table.append(thead);

    const tbody = document.createElement("tbody");
    sectionData.rows.forEach((row) => {
      const family = familyClass[row.family] ? row.family : "Other";
      const support = row.support === "moderate" ? "moderate" : "thin";
      const tr = make("tr", `ar-ranking-row ${familyClass[family]} ar-support-${support}${row.key === data.selectedKey ? " ar-selected" : ""}`);
      tr.dataset.key = row.key;
      tr.tabIndex = 0;
      tr.setAttribute("role", "button");
      tr.setAttribute("aria-selected", row.key === data.selectedKey ? "true" : "false");
      tr.setAttribute("aria-label", `Rank ${row.rank}, ${row.product}, ${familyName[family]}, ${support === "moderate" ? "moderately" : "thinly"} supported. Select for evidence.`);
      tr.title = `${familyName[family]} · ${support === "moderate" ? "moderately" : "thinly"} supported`;
      tr.addEventListener("click", () => selectRow(tr, row.key));
      tr.addEventListener("keydown", (event) => {
        if (event.key === "Enter" || event.key === " ") {
          event.preventDefault();
          selectRow(tr, row.key);
        }
      });

      addCell(tr, row.rank, "ar-ranking-rank");
      addCell(tr, row.product, "ar-ranking-product");
      addCell(tr, row.studies, "ar-ranking-studies");
      addCell(tr, Number.isFinite(Number(row.score)) ? Number(row.score).toFixed(2) : null, "ar-ranking-score");
      if (data.isMixture) addCell(tr, row.limitingPfas, "ar-ranking-limit");
      addCell(tr, row.basis, "ar-ranking-basis");
      const warningCell = document.createElement("td");
      const warningList = make("div", "ar-ranking-warnings");
      (row.warnings || []).forEach((warning) => {
        const severity = ["critical", "info", "review"].includes(warning.severity) ? warning.severity : "review";
        const tag = make("span", `ar-warning-tag ar-warning-${severity}`, warning.label);
        if (warning.meaning) tag.title = warning.meaning;
        warningList.append(tag);
      });
      if (!(row.warnings || []).length) warningList.append(make("span", "ar-sr-only", "No recorded warnings"));
      warningCell.append(warningList);
      tr.append(warningCell);
      tbody.append(tr);
    });
    table.append(tbody);
    scroll.append(table);
    section.append(scroll);
    root.append(section);
  });
}
"""


_RANKING_TABLE = st.components.v2.component(
    "adsorbent_recommender_ranking_table",
    html=_HTML,
    css=_CSS,
    js=_JS,
)


def render_ranking_table(
    *,
    sections: list[dict[str, Any]],
    score_label: str,
    is_mixture: bool,
    selected_key: str | None,
    key: str,
) -> str | None:
    """Render the shortlist and return the selected product key, if any."""
    stored = st.session_state.get(key)
    if selected_key is None:
        if isinstance(stored, Mapping):
            selected_key = stored.get("selected_key")
        else:
            selected_key = getattr(stored, "selected_key", None)
    show_other = any(
        row.get("family") == "Other"
        for section in sections
        for row in section.get("rows", [])
    )
    result = _RANKING_TABLE(
        data={
            "sections": sections,
            "scoreLabel": score_label,
            "isMixture": is_mixture,
            "selectedKey": selected_key,
            "showOther": show_other,
        },
        default={"selected_key": selected_key},
        key=key,
        on_selected_key_change=lambda: None,
        width="stretch",
    )
    if isinstance(result, Mapping):
        value = result.get("selected_key")
    else:
        value = getattr(result, "selected_key", None)
    return str(value) if value else selected_key
