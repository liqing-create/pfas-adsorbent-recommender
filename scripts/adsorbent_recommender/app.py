"""Streamlit interface for one selected adsorbent recommender bundle.

Run with:
    cd scripts
    streamlit run adsorbent_recommender\\app.py

Colours, fonts and radii live in ``scripts/.streamlit/config.toml``.  Streamlit
resolves that file against the working directory, so launching from ``scripts``
is what applies the theme; from anywhere else the app still lays out and reads
correctly, it just falls back to Streamlit's stock palette.  The rules injected
below deliberately derive their neutrals from the app's own text colour instead
of naming greys, so they hold under either palette and in either colour mode.

The selected model-bundle directory is a deployment setting in the
``Configuration`` section below. It is intentionally not editable in the UI.
"""

from __future__ import annotations

import json
import os
import re
import sys
from html import escape
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import pandas as pd
import streamlit as st

# Streamlit executes this file as a script, so explicitly expose the scripts
# parent before importing the recommender package.
SCRIPTS_DIR = Path(__file__).resolve().parents[1]
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from adsorbent_recommender.bundle import BUNDLE_FILENAME
from adsorbent_recommender.service import (
    RESULT_SCHEMA_VERSION,
    BundleNotReadyError,
    catalog_category_groups,
    load_recommender_bundle,
    pfas_features_source,
    recommend_mixture,
)

# Importing the service puts the DB package on the path, which is what makes the
# normalizer's own class list reachable here.
from normalization_rules import WATER_TYPE_CLASSES  # noqa: E402


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
# The bundle directory may be set explicitly here, through the
# ADSORBENT_BUNDLE_DIR environment variable, or left to resolve to the most
# recently written bundle under the output root.  A hardcoded path silently
# goes stale whenever a model is rebuilt, and the resulting failure looks like
# a broken application rather than the missing setting it actually is.
MODEL_BUNDLE_DIR: Path | None = None
BUNDLE_SEARCH_ROOT = SCRIPTS_DIR.parent / "output" / "ML_logKd"


def _discover_bundles(root: Path) -> list[Path]:
    """Return bundle directories under ``root``, most recently written first.

    Only the run directories directly under the output root are examined.  A
    recursive search would walk every retained per-repeat model directory of
    every exploratory batch, which on synced storage takes long enough to look
    like the application has hung, and those directories never hold a bundle.
    """
    if not root.exists():
        return []
    manifests = [
        manifest
        for entry in root.iterdir()
        if entry.is_dir()
        for manifest in [entry / BUNDLE_FILENAME]
        if manifest.is_file()
    ]
    manifests.sort(key=lambda path: path.stat().st_mtime, reverse=True)
    return [manifest.parent for manifest in manifests]


def _resolve_bundle_dir() -> Path | None:
    """Resolve the configured bundle.

    Which bundle is loaded is a deployment setting, not a question the user
    answers per screening, so it is never offered in the UI.  The loaded bundle
    still travels with every result: ``recommend`` records its directory, model
    family and target under ``bundle`` in the downloaded JSON.
    """
    configured = MODEL_BUNDLE_DIR or os.environ.get("ADSORBENT_BUNDLE_DIR")
    if configured:
        return Path(configured)
    available = _discover_bundles(BUNDLE_SEARCH_ROOT)
    return available[0] if available else None


# ---------------------------------------------------------------------------
# Presentation
# ---------------------------------------------------------------------------
# Only structure lives here: type scale, widget colours and radii are set in
# .streamlit/config.toml so Streamlit's own components share them.  Every
# neutral below is mixed from ``currentColor`` rather than written as a grey,
# which is what lets one rule set serve the light and the dark palette; the one
# literal colour is the accent, lightened for dark mode so it keeps its
# contrast against a dark ground.
UI_STYLES = """
<style>
.stApp {
    --ar-accent: #0E7C8A;
    --ar-line: color-mix(in srgb, currentColor 16%, transparent);
    --ar-hairline: color-mix(in srgb, currentColor 9%, transparent);
    --ar-surface: color-mix(in srgb, currentColor 4%, transparent);
    --ar-muted: color-mix(in srgb, currentColor 62%, transparent);
    /* What sits on top of a filled accent chip.  The dark-mode accent is
       light enough that white numerals on it fall below readable contrast,
       so the two flip together. */
    --ar-on-accent: #FFFFFF;
}
@media (prefers-color-scheme: dark) {
    .stApp {
        --ar-accent: #3FC2CE;
        --ar-on-accent: #0F1519;
    }
}

/* The form is wide but not endless; an unbounded measure on a 4K monitor
   stretches label/field pairs so far apart they stop reading as pairs. */
[data-testid="stMainBlockContainer"] {
    max-width: 1440px;
    padding-top: 2.4rem;
    padding-bottom: 4rem;
}

/* ---- Masthead --------------------------------------------------------- */
.ar-hero {
    border: 1px solid var(--ar-line);
    border-radius: 16px;
    padding: 1.6rem 1.8rem;
    background:
        linear-gradient(135deg,
            color-mix(in srgb, var(--ar-accent) 13%, transparent) 0%,
            transparent 62%);
}
.ar-eyebrow {
    font-size: 0.85rem;
    font-weight: 700;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: var(--ar-accent);
}
.ar-hero h1 {
    margin: 0.4rem 0 0.45rem;
    font-size: 2.1rem;
    font-weight: 700;
    line-height: 1.14;
    letter-spacing: -0.02em;
}
.ar-hero p {
    margin: 0;
    max-width: 82ch;
    font-size: 1.05rem;
    line-height: 1.55;
    color: var(--ar-muted);
}

/* ---- Input cards ------------------------------------------------------ */
/* Every question the user answers belongs to one input group, and the groups
   are peers.  A numbered header states that rank once, so the sections no
   longer have to be forced to look equal by enlarging expander summaries. */
[class*="st-key-ar-card-"] {
    padding: 1.15rem 1.3rem 1.25rem !important;
    border-radius: 14px !important;
}
.ar-section {
    display: flex;
    align-items: flex-start;
    gap: 0.7rem;
    padding-bottom: 0.7rem;
    border-bottom: 1px solid var(--ar-hairline);
}
.ar-step {
    flex: 0 0 auto;
    width: 1.7rem;
    height: 1.7rem;
    border-radius: 9px;
    display: inline-flex;
    align-items: center;
    justify-content: center;
    font-size: 0.86rem;
    font-weight: 700;
    color: var(--ar-on-accent);
    background: var(--ar-accent);
}
.ar-section-title {
    font-size: 1.25rem;
    font-weight: 650;
    line-height: 1.3;
}
.ar-section-hint {
    margin-top: 0.1rem;
    font-size: 0.95rem;
    line-height: 1.4;
    color: var(--ar-muted);
}
/* The bold lines dividing water chemistry are subheadings of their group, so
   they sit between the group title and the field labels rather than reading as
   ordinary body text. */
.ar-subhead {
    margin: 1.1rem 0 0.15rem;
    font-size: 0.92rem;
    font-weight: 700;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: var(--ar-muted);
}
[class*="st-key-ar-ions"] {
    background: var(--ar-surface);
    border-color: var(--ar-hairline) !important;
    border-radius: 12px !important;
    padding: 0.75rem 1rem 0.25rem !important;
}

/* A selected PFAS is a compact, editable row rather than another full card. */
[class*="st-key-ar-pfas-entry-"] {
    padding: 0.65rem 0.8rem !important;
    margin-top: 0.45rem;
    border: 1px solid var(--ar-hairline);
    border-radius: 11px;
    background: var(--ar-surface);
}
[class*="st-key-ar-pfas-entry-"] [data-testid="stVerticalBlock"] {
    gap: 0.35rem;
}
.ar-identity-label {
    margin-bottom: 0.12rem;
    color: var(--ar-muted);
    font-size: 0.78rem;
    font-weight: 700;
    letter-spacing: 0.08em;
    text-transform: uppercase;
}
.ar-identity-value {
    min-height: 1.45rem;
    font-size: 0.96rem;
    line-height: 1.35;
}

/* ---- Action row ------------------------------------------------------- */
[class*="st-key-ar-run"] button {
    padding-top: 0.7rem;
    padding-bottom: 0.7rem;
    font-size: 1.05rem;
    font-weight: 650;
}
.ar-required {
    font-size: 0.95rem;
    color: var(--ar-muted);
}
.ar-required b { color: var(--ar-accent); }

/* ---- Results ---------------------------------------------------------- */
.ar-scenario {
    display: flex;
    flex-wrap: wrap;
    gap: 0.4rem;
    margin-bottom: 0.15rem;
}
.ar-chip {
    border: 1px solid var(--ar-hairline);
    border-radius: 999px;
    background: var(--ar-surface);
    padding: 0.25rem 0.75rem;
    font-size: 0.92rem;
    color: var(--ar-muted);
}
.ar-chip b {
    font-weight: 600;
    color: inherit;
    filter: none;
}
.ar-legend {
    display: flex;
    flex-wrap: wrap;
    gap: 1.1rem;
    margin: 0.9rem 0 0.35rem;
    font-size: 0.95rem;
    color: var(--ar-muted);
}
.ar-key { display: inline-flex; align-items: center; gap: 0.45rem; }
.ar-key-swatch {
    width: 1.05rem;
    height: 1.05rem;
    border-radius: 4px;
    border: 1px solid var(--ar-hairline);
}
.ar-headline {
    border: 1px solid var(--ar-line);
    border-left: 3px solid var(--ar-accent);
    border-radius: 12px;
    background: color-mix(in srgb, var(--ar-accent) 7%, transparent);
    padding: 0.95rem 1.15rem;
    font-size: 1.05rem;
    line-height: 1.55;
}
.ar-empty {
    border: 1px dashed var(--ar-line);
    border-radius: 14px;
    padding: 2.4rem 1.5rem;
    text-align: center;
    color: var(--ar-muted);
}
.ar-empty-title {
    font-size: 1.15rem;
    font-weight: 650;
    color: inherit;
}
.ar-empty p { margin: 0.35rem 0 0; font-size: 1rem; }

/* Streamlit truncates a metric label and value to one line each, which turns
   four summary cards into "Recomm...", "Databas..." the moment the row gets
   narrow.  A summary the reader cannot read is worse than a taller card. */
[data-testid="stMetricLabel"],
[data-testid="stMetricLabel"] *,
[data-testid="stMetricValue"],
[data-testid="stMetricValue"] * {
    white-space: normal !important;
    overflow: visible !important;
    text-overflow: clip !important;
    /* Streamlit breaks these mid-word to fit one line; wrapping is only an
       improvement if the break lands between words. */
    overflow-wrap: normal !important;
    word-break: normal !important;
}
[data-testid="stMetricLabel"] p {
    font-size: 0.95rem;
    font-weight: 600;
    line-height: 1.3;
}
[data-testid="stMetricValue"] {
    font-size: 1.9rem;
    line-height: 1.25;
}

/* Streamlit renders every widget label at 0.875rem regardless of the base
   size, so in a form that is almost entirely labels the theme's type scale
   never actually reaches the text the user reads.  The dropdown list is drawn
   in a portal outside the app container and needs its own rule, or the options
   stay smaller than the field that opened them. */
[data-testid="stWidgetLabel"] p,
[data-testid="stWidgetLabel"] label,
[data-testid="stCheckbox"] [data-testid="stMarkdownContainer"] p,
[role="listbox"] [role="option"],
[data-baseweb="popover"] [role="option"],
[data-baseweb="popover"] li {
    font-size: 1rem !important;
}
[data-testid="stCaptionContainer"],
[data-testid="stCaptionContainer"] p {
    font-size: 0.95rem;
    line-height: 1.5;
}

.stTabs button[role="tab"] {
    font-size: 1.05rem;
    font-weight: 600;
}
[data-testid="stDataFrame"] { border-radius: 10px; }
[data-testid="stExpander"] summary { font-weight: 600; }
</style>
"""


def _hero(title: str, eyebrow: str, subtitle: str) -> None:
    st.html(
        f"<div class='ar-hero'>"
        f"<div class='ar-eyebrow'>{eyebrow}</div>"
        f"<h1>{title}</h1>"
        f"<p>{subtitle}</p>"
        f"</div>"
    )


def _section(step: int, title: str, hint: str) -> None:
    st.html(
        f"<div class='ar-section'>"
        f"<span class='ar-step'>{step}</span>"
        f"<div><div class='ar-section-title'>{title}</div>"
        f"<div class='ar-section-hint'>{hint}</div></div>"
        f"</div>"
    )


def _subhead(text: str) -> None:
    st.html(f"<div class='ar-subhead'>{text}</div>")


def _chips(pairs: list[tuple[str, Any]]) -> None:
    chips = "".join(
        f"<span class='ar-chip'><b>{escape(str(label))}</b> {escape(str(value))}</span>"
        for label, value in pairs
        if value not in (None, "")
    )
    if chips:
        st.html(f"<div class='ar-scenario'>{chips}</div>")


def _card(key: str, height: str = "content"):
    """Return a bordered panel that renders as a top-level input group.

    Side-by-side groups pass ``height="stretch"`` so the pair reads as one row
    of peers; left to their content the shorter card ends in mid-air beside the
    taller one and the two stop looking like the same kind of thing.
    """
    return st.container(key=f"ar-card-{key}", border=True, height=height)


# A typed water type is standardized before it ever reaches the model, and any
# wording the normalizer does not recognise collapses to "others".  Offering the
# normalized classes directly keeps what the user chose and what the model sees
# the same thing.  The order below is the one that reads best in the form; a
# class added upstream is appended rather than silently dropped.
WATER_TYPE_ORDER = (
    "ultrapure water",
    "synthetic water",
    "tap water",
    "groundwater",
    "surface water",
    "wastewater",
    "landfill leachate",
    "AFFF solution",
    "others",
)
WATER_TYPE_OPTIONS = [
    "",
    *WATER_TYPE_ORDER,
    *sorted(WATER_TYPE_CLASSES - set(WATER_TYPE_ORDER)),
]


# The checkbox keys are the trained ``contains_<ion>`` feature names and cannot
# be renamed, so the ionic formulae live here as display labels only.  Phosphate
# speciates with pH; the feature covers phosphate as a whole, and PO4(3-) is the
# conventional way to write that in a water-quality summary.
ION_LABELS = {
    "Na": "Na⁺",
    "K": "K⁺",
    "Ca": "Ca²⁺",
    "Mg": "Mg²⁺",
    "Cl": "Cl⁻",
    "HCO3": "HCO₃⁻",
    "SO4": "SO₄²⁻",
    "phosphate": "PO₄³⁻",
}


# Abbreviation leads because that is how a PFAS is normally named in practice;
# a SMILES string is the precise form but rarely the one at hand.  The lookup
# matches the entered text against the selected identifier alone, so each type
# carries its own example rather than one prompt listing all four.
IDENTITY_TYPES = {
    "abbreviation": {"option": "Abbreviation", "field": "PFAS abbreviation", "example": "e.g., PFOA"},
    "name": {"option": "Name", "field": "PFAS name", "example": "e.g., perfluorooctanoic acid"},
    "cas": {"option": "CAS number", "field": "PFAS CAS number", "example": "e.g., 335-67-1"},
    "smiles": {"option": "SMILES", "field": "PFAS SMILES", "example": "e.g., OC(=O)C(F)(F)C(F)(F)C(F)(F)F"},
}


REQUIRED_MARK = " *"
REQUIRED_NOTE = (
    "Fields marked <b>*</b> are required. Every other field is optional, so leave it "
    "blank when the value is unknown."
)
RESULT_DISPLAY_LIMIT = 20
PFAS_ENTRIES_KEY = "ar_pfas_entries_v2"
PFAS_NEXT_ID_KEY = "ar_pfas_next_id_v2"
PFAS_SEARCH_KEY = "ar_pfas_search_v2"
PFAS_ADD_C0_KEY = "ar_pfas_add_c0_v2"
PfasChoice = tuple[str, str, str, str, str, str, str]


def _clean_display_value(value: Any) -> str:
    """Return a cache cell as display text without leaking pandas sentinels."""
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    return str(value).strip()


@st.cache_data(show_spinner=False)
def _load_pfas_choices(path: str, sheet_name: str, modified_ns: int) -> list[PfasChoice]:
    """Offer each identifier separately while retaining its compound metadata."""
    del modified_ns  # Part of the cache key so a replaced workbook is re-read.
    frame = pd.read_excel(path, sheet_name=sheet_name)
    choices: list[PfasChoice] = []
    seen: set[str] = set()
    for _, row in frame.iterrows():
        abbreviation = _clean_display_value(row.get("Abbreviation"))
        name = _clean_display_value(row.get("PFAS_name")) or _clean_display_value(row.get("Compound Full Name"))
        cas = _clean_display_value(row.get("CAS Number")) or _clean_display_value(row.get("CAS"))
        smiles = _clean_display_value(row.get("Canonical SMILES")) or _clean_display_value(row.get("SMILES"))
        cache_identity = smiles or abbreviation or cas or name
        if not cache_identity:
            continue
        # Separate options keep the dropdown visually clean while still making
        # the same PFAS discoverable by abbreviation, full name, or CAS number.
        for identity_type, identity in (
            ("abbreviation", abbreviation), ("name", name), ("cas", cas),
        ):
            if not identity:
                continue
            key = identity.casefold()
            if key in seen:
                continue
            seen.add(key)
            choices.append((
                identity_type, identity, identity,
                abbreviation, name, cas, cache_identity,
            ))
    return sorted(choices, key=lambda choice: choice[2].casefold())


def _pfas_choices(bundle: Any) -> list[PfasChoice]:
    path, sheet_name = pfas_features_source(bundle)
    if not path.is_file() or not sheet_name:
        return []
    return _load_pfas_choices(str(path), str(sheet_name), path.stat().st_mtime_ns)


def _infer_identity_type(identity: str) -> str:
    """Infer the service identifier route for a user-entered PFAS value."""
    text = identity.strip()
    if re.fullmatch(r"\d{2,7}-\d{2}-\d", text):
        return "cas"
    if any(character in text for character in "[]()=#@/\\"):
        return "smiles"
    # Name lookup also checks the abbreviation column, so it is the safest
    # fallback for both a common acronym such as PFOA and a full chemical name.
    return "name"


def _add_selected_pfas() -> None:
    selected = st.session_state.get(PFAS_SEARCH_KEY)
    concentration = st.session_state.get(PFAS_ADD_C0_KEY)
    if selected in (None, ""):
        st.session_state["ar_pfas_add_error"] = "Choose or enter a PFAS before adding it."
        return
    if concentration is None or float(concentration) <= 0:
        st.session_state["ar_pfas_add_error"] = "C₀ must be greater than zero."
        return
    if isinstance(selected, (tuple, list)) and len(selected) == 7:
        identity_type, identity, label, abbreviation, full_name, cas, cache_identity = map(str, selected)
        cached = True
    elif isinstance(selected, (tuple, list)) and len(selected) == 3:
        # Preserve a pending selection across a hot reload from the first UI
        # version; new choices always use the richer seven-field payload.
        identity_type, identity, label = map(str, selected)
        abbreviation = full_name = cas = ""
        cache_identity = identity
        cached = True
    else:
        identity = str(selected).strip()
        identity_type = _infer_identity_type(identity)
        label = identity
        abbreviation = identity if identity_type == "abbreviation" else ""
        full_name = identity if identity_type == "name" else ""
        cas = identity if identity_type == "cas" else ""
        cache_identity = identity
        cached = False
    entries = list(st.session_state.get(PFAS_ENTRIES_KEY, []))
    aliases = {
        "".join(character for character in value.casefold() if character.isalnum())
        for value in (identity, abbreviation, full_name, cas, cache_identity)
        if value
    }
    duplicate = any(
        aliases.intersection(
            set(entry.get("identity_aliases", []))
            | ({entry.get("normalized_identity")} if entry.get("normalized_identity") else set())
        )
        for entry in entries
    )
    if duplicate:
        st.session_state["ar_pfas_add_error"] = f"{label} is already in the mixture."
        return
    next_id = int(st.session_state.get(PFAS_NEXT_ID_KEY, 1))
    entries.append({
        "id": next_id,
        "identity": identity,
        "identity_type": identity_type,
        "label": label,
        "abbreviation": abbreviation,
        "full_name": full_name,
        "cas": cas,
        "cached_choice": cached,
        "normalized_identity": "".join(
            character for character in cache_identity.casefold() if character.isalnum()
        ),
        "identity_aliases": sorted(aliases),
        "initial_concentration_mg_l": float(concentration),
    })
    st.session_state[PFAS_ENTRIES_KEY] = entries
    st.session_state[PFAS_NEXT_ID_KEY] = next_id + 1
    st.session_state[PFAS_SEARCH_KEY] = None
    st.session_state.pop("ar_pfas_add_error", None)


def _remove_pfas(entry_id: int) -> None:
    st.session_state[PFAS_ENTRIES_KEY] = [
        entry for entry in st.session_state.get(PFAS_ENTRIES_KEY, [])
        if entry.get("id") != entry_id
    ]


def _identity_field(label: str, value: Any) -> None:
    """Render one PFAS identifier with a persistent field label."""
    text = _clean_display_value(value) or "—"
    st.html(
        f"<div class='ar-identity-label'>{escape(label)}</div>"
        f"<div class='ar-identity-value'>{escape(text)}</div>"
    )


def _number_input(
    label: str,
    key: str,
    example: str,
    unit: str = "",
    required: bool = False,
    help_text: str | None = None,
) -> float | None:
    """Read a number, naming its unit in the label and its format below it.

    The unit belongs to the quantity, not to the value the user is about to
    type, so it stays visible in the label instead of disappearing from the
    placeholder as soon as the field is filled in.  Every placeholder is an
    example for the same reason on the other side: whether a field is required
    is a property of the form, stated once by the marker, and a placeholder
    that says "optional" on some fields and shows a format on others reads as
    if the ones showing a format were mandatory.
    """
    display_label = f"{label} ({unit})" if unit else label
    if required:
        display_label += REQUIRED_MARK
    raw = st.text_input(display_label, key=key, placeholder=f"e.g., {example}", help=help_text)
    if not raw.strip():
        return None
    try:
        return float(raw)
    except ValueError:
        st.warning(f"{display_label} must be numeric; its value will be treated as missing.")
        return None


@st.cache_resource(show_spinner=False)
def _load_bundle(directory: str):
    return load_recommender_bundle(Path(directory))


def _water_form(height: str = "content") -> tuple[dict[str, Any], float | None]:
    with _card("water", height=height):
        _section(2, "Water chemistry", "Enter the common conditions first. Add detailed matrix chemistry only when it is known.")
        general_left, general_middle, general_right = st.columns(3)
        with general_left:
            water_type = st.selectbox(
                "Water type",
                WATER_TYPE_OPTIONS,
                key="water_type",
                format_func=lambda value: value or "Not specified",
            )
        with general_middle:
            ph = _number_input("pH", "ph", "7.2")
        with general_right:
            temperature = _number_input(
                "Temperature", "temperature", "25", "°C",
                help_text="Left blank, the screening uses 25 °C.",
            )
        organic_count = sum(
            bool(str(st.session_state.get(key, "")).strip())
            for key in ("organic_carbon", "organic")
        )
        organic_label = "Organic chemistry"
        if organic_count:
            organic_label += f" · {organic_count} value{'s' if organic_count != 1 else ''} entered"
        with st.expander(organic_label, expanded=False, icon=":material/eco:"):
            st.caption("Leave any unmeasured value blank. Closing this section preserves entered values.")
            # A measurement and its free-text additive field describe the same
            # part of the matrix, so each stays beside its counterpart.
            organic_left, organic_right = st.columns(2)
            with organic_left:
                organic_carbon = _number_input("Organic carbon", "organic_carbon", "5", "mg/L")
            with organic_right:
                organic = st.text_input("Organic additives", key="organic", placeholder="e.g., 5 mg/L humic acid")

        inorganic_count = sum(
            bool(str(st.session_state.get(key, "")).strip())
            for key in ("tds", "ionic_strength", "inorganic")
        )
        inorganic_count += sum(bool(st.session_state.get(f"ion_{ion}")) for ion in ION_LABELS)
        inorganic_label = "Inorganic chemistry"
        if inorganic_count:
            inorganic_label += f" · {inorganic_count} value{'s' if inorganic_count != 1 else ''} entered"
        with st.expander(inorganic_label, expanded=False, icon=":material/science:"):
            st.caption("Leave any unmeasured value blank. Closing this section preserves entered values.")
            inorganic_left, inorganic_middle, inorganic_right = st.columns(3)
            with inorganic_left:
                tds = _number_input("TDS", "tds", "500", "mg/L")
            with inorganic_middle:
                ionic_strength = _number_input("Ionic strength", "ionic_strength", "0.01", "mol/L")
            with inorganic_right:
                inorganic = st.text_input("Inorganic additives", key="inorganic", placeholder="e.g., 2 mM CaCl2")
            # The ions are themselves inorganic additives, so they belong with
            # the other inorganic fields.
            ion_values = {}
            with st.container(key="ar-ions", border=True):
                st.caption(
                    "Select ions known to be present. Their concentrations can be recorded "
                    "in the inorganic additive field above."
                )
                ions = st.columns(4)
                for index, (ion, label) in enumerate(ION_LABELS.items()):
                    with ions[index % 4]:
                        ion_values[ion] = st.checkbox(label, key=f"ion_{ion}")
    water = {
        "water_type": water_type, "pH": ph, "organic_carbon_mg_l": organic_carbon, "tds_mg_l": tds,
        "ionic_strength_mol_l": ionic_strength, "inorganic_matter": inorganic,
        "organic_matter": organic, "ions": ion_values,
    }
    return water, temperature


def _product_name(product: dict[str, Any]) -> str:
    return product.get("Name_Commercial") or product["Product_key"]


def _ranking_table(
    result: dict[str, Any],
    items: list[dict[str, Any]] | None = None,
    class_filtered: bool = False,
) -> pd.DataFrame:
    is_mixture = result.get("mixture", {}).get("pfas_count", 1) > 1
    rows = []
    displayed = result["recommendations"] if items is None else items
    for class_rank, item in enumerate(displayed, start=1):
        product = item["product"]
        prediction = item["prediction"]
        performance = item.get("performance", {"source": "model_prediction", **prediction})
        characterization = item.get("characterization", {})
        present = characterization.get("material_features_present")
        required = characterization.get("material_features_required")
        row = {
            "Overall rank": item["rank"], "Product": _product_name(product),
            "Class": product.get("adsorbent_category"),
            "Category": product.get("adsorbent_subcategory"),
            "Basis": _basis_label(item.get("performance_source")),
            # Two products can share a score while resting on very different
            # amounts of evidence, so the basis travels with the ranking.
            "Characterized on": None if present is None else f"{present}/{required} features",
            "Seen in training": characterization.get("seen_in_training"),
        }
        if class_filtered:
            row["Class rank"] = class_rank
        if is_mixture:
            mixture = item["mixture_performance"]
            row.update({
                "Worst-case logKd": mixture["worst_case_logKd"],
                "Mean logKd": mixture["mean_logKd"],
                "Limiting PFAS": mixture["limiting_pfas"],
                "Support": f"{mixture['database_supported_pfas']}/{mixture['pfas_count']} PFAS from database",
            })
        else:
            row.update({
                "logKd at reference dose": performance["mean_logKd"],
                "Support": (
                    f"{performance.get('match_count')} matched row(s)"
                    if item.get("performance_source") == "database_match"
                    else f"ensemble SD {prediction.get('ensemble_sd', float('nan')):.3f}"
                ),
            })
        rows.append(row)
    frame = pd.DataFrame(rows)
    rank_columns = ["Class rank", "Overall rank"] if class_filtered else ["Overall rank"]
    if is_mixture:
        column_order = rank_columns + [
            "Product", "Class", "Category", "Worst-case logKd", "Mean logKd",
            "Limiting PFAS", "Basis", "Support", "Characterized on", "Seen in training",
        ]
    else:
        column_order = rank_columns + [
            "Product", "Class", "Category", "logKd at reference dose",
            "Basis", "Support", "Characterized on", "Seen in training",
        ]
    return frame.reindex(columns=column_order)


# The two ways a number in the ranking can have been arrived at, and the tint
# each one gives its row.  The fills are translucent rather than solid because
# the data grid paints them over its own background, which is light or dark
# depending on the viewer's theme; an opaque pastel would strand dark text on a
# dark ground.  Teal is the app's accent and marks the stronger evidence; amber
# is the conventional "estimate" colour and is distinguishable from it under the
# common forms of colour blindness.  The written Basis column stays in the table
# regardless, so the distinction never rests on colour alone.
BASIS_DATABASE = "Database record"
BASIS_MODEL = "ML prediction"
BASIS_MIXED = "Database + ML"
BASIS_FILLS = {
    BASIS_DATABASE: "rgba(14, 124, 138, 0.20)",
    BASIS_MODEL: "rgba(198, 124, 8, 0.18)",
    BASIS_MIXED: "rgba(91, 78, 163, 0.18)",
}


def _basis_label(source: str | None) -> str:
    return {
        "database_match": BASIS_DATABASE,
        "model_prediction": BASIS_MODEL,
        "mixed_evidence": BASIS_MIXED,
    }.get(source, BASIS_MODEL)


def _ranking_styler(frame: pd.DataFrame):
    """Tint each row by whether its number was measured or predicted."""
    def tint(row: pd.Series) -> list[str]:
        fill = BASIS_FILLS.get(row.get("Basis"))
        return [f"background-color: {fill}" if fill else "" for _ in row]

    return frame.style.apply(tint, axis=1)


def _basis_legend() -> None:
    swatches = "".join(
        f"<span class='ar-key'><span class='ar-key-swatch' style='background:{fill}'></span>{label}</span>"
        for label, fill in BASIS_FILLS.items()
    )
    st.html(f"<div class='ar-legend'>{swatches}</div>")


def _ranking_column_config() -> dict[str, Any]:
    """Column presentation for a compact, quantitatively neutral ranking."""
    config = {
        "Overall rank": st.column_config.NumberColumn("Overall #", format="%d", width="small"),
        "Class rank": st.column_config.NumberColumn("Class #", format="%d", width="small"),
        "Product": st.column_config.TextColumn("Product", width="medium"),
        "logKd at reference dose": st.column_config.NumberColumn(
            "log Kd", format="%.3f", width="small",
            help="Higher is stronger affinity at the common 25 mg/L reference dose.",
        ),
        "Worst-case logKd": st.column_config.NumberColumn(
            "Worst-case log Kd", format="%.3f", width="small",
            help="The lowest logKd for this adsorbent among all PFAS in the mixture; ranking uses this value.",
        ),
        "Mean logKd": st.column_config.NumberColumn(
            "Mean log Kd", format="%.3f", width="small",
            help="Unweighted mean across the entered PFAS; concentrations are already included in each PFAS calculation.",
        ),
        "Basis": st.column_config.TextColumn("Basis", width="small"),
        "Seen in training": st.column_config.CheckboxColumn("In training set"),
    }


def _http_sources(value: Any) -> list[str]:
    """Return only complete external HTTP(S) URLs from a catalog source cell."""
    text = "" if value is None else str(value).strip()
    if text.casefold() in {"", "nan", "none", "<na>"}:
        return []
    sources = []
    for candidate in text.split(";"):
        candidate = candidate.strip()
        parsed = urlparse(candidate)
        if parsed.scheme in {"http", "https"} and parsed.netloc:
            sources.append(candidate)
    return sources


def _render_candidate_detail(result: dict[str, Any]) -> None:
    for item in result["recommendations"]:
        performance = item.get("performance", item["prediction"])
        basis = _basis_label(item.get("performance_source"))
        mixture = item.get("mixture_performance")
        if mixture and mixture.get("pfas_count", 1) > 1:
            score_label = f"worst-case log Kd {mixture['worst_case_logKd']:.3f}"
        else:
            score_label = f"log Kd {performance['mean_logKd']:.3f}"
        header = f"#{item['rank']} · {_product_name(item['product'])} · {score_label} · {basis}"
        with st.expander(header):
            if item.get("warnings"):
                st.warning("\n".join(item["warnings"]))

            pfas_results = item.get("pfas_results", [])
            if len(pfas_results) > 1:
                _subhead("Performance across the PFAS mixture")
                performance_rows = []
                for pfas_result in pfas_results:
                    value = pfas_result.get("performance") or pfas_result.get("prediction") or {}
                    performance_rows.append({
                        "PFAS": pfas_result.get("pfas"),
                        "C₀ (mg/L)": pfas_result.get("initial_concentration_mg_l"),
                        "logKd": value.get("mean_logKd"),
                        "Basis": _basis_label(pfas_result.get("performance_source")),
                        "Support": (
                            f"{value.get('match_count')} matched row(s)"
                            if pfas_result.get("performance_source") == "database_match"
                            else f"ensemble SD {(pfas_result.get('prediction') or {}).get('ensemble_sd', float('nan')):.3f}"
                        ),
                    })
                st.dataframe(pd.DataFrame(performance_rows), width="stretch", hide_index=True)
                st.caption("The ranking score is the lowest logKd in this table; the corresponding PFAS is limiting.")

            _subhead("Adsorbent properties")
            properties = pd.DataFrame(item.get("adsorbent_properties", []))
            if properties.empty:
                st.info("No detailed catalog properties are available for this product.")
            else:
                properties = properties.rename(columns={
                    "group": "Group", "property": "Property", "value": "Value", "unit": "Unit",
                    "used_by_selected_model": "Used by selected model",
                })
                # Property values intentionally mix categorical labels and measurements.
                # Normalize the display copy so Arrow does not infer conflicting types.
                properties["Value"] = properties["Value"].astype("string").fillna("")
                st.dataframe(
                    properties[["Group", "Property", "Value", "Unit", "Used by selected model"]],
                    width="stretch", hide_index=True,
                )
                st.caption(
                    "‘Used by selected model’ identifies model inputs, not candidate-specific feature importance. "
                    "Unreported properties are omitted."
                )

            product_sources = _http_sources(item["product"].get("catalog_data_sources"))
            if product_sources:
                st.markdown("**Product sources**")
                for index, source in enumerate(product_sources, start=1):
                    st.markdown(f"- [Open product datasheet {index}]({source})")

            evidence_sets = (
                [(row.get("pfas"), row.get("database_evidence", {})) for row in pfas_results]
                if len(pfas_results) > 1
                else [(None, item.get("database_evidence", {}))]
            )
            matched_evidence = [(label, evidence) for label, evidence in evidence_sets if evidence.get("status") == "matched"]
            if matched_evidence:
                _subhead("Database evidence")
                for pfas_label, evidence in matched_evidence:
                    if pfas_label:
                        st.markdown(f"**{pfas_label}**")
                    st.dataframe(pd.DataFrame(evidence.get("records", [])), width="stretch", hide_index=True)
                    for citation in evidence.get("citations", []):
                        source_urls = _http_sources(citation.get("url"))
                        citation_label = citation.get("label") or citation.get("doi") or "Performance source"
                        if source_urls:
                            st.markdown(f"- [{citation_label}]({source_urls[0]})")
                        else:
                            st.write(f"- {citation_label}")
            else:
                fallback = evidence_sets[0][1] if evidence_sets else {}
                st.caption(f"Database fallback reason: {fallback.get('reason', 'no comparable record')}")

            with st.expander("Technical prediction record (JSON)"):
                st.json({
                    "product": item["product"],
                    "characterization": item.get("characterization"),
                    "performance_used_for_ranking": item.get("performance"),
                    "prediction": item["prediction"],
                    "pfas_results": item.get("pfas_results"),
                    "screening_conditions": item["screening_conditions"],
                })


def _dose_label(dose: Any) -> str:
    """Format the common dose for a summary card.

    ``25.0 mg/L`` and ``25 mg/L`` say the same thing, and the trailing zero is
    what pushes the value onto a second line in a narrow card.
    """
    if dose is None:
        return "Per candidate"
    if isinstance(dose, (int, float)):
        return f"{dose:g} mg/L"
    return f"{dose} mg/L"


def _render_basis(result: dict[str, Any]) -> None:
    """State where the numbers came from, once, in one panel.

    Routing, dose support and the measurement-versus-estimate caveat were four
    separate coloured callouts stacked above the table.  They are one answer to
    one question, and a column of alternating info and warning banners gives
    the reader no way to tell which of them is the important one.
    """
    routing = result.get("evidence_routing", {})
    conditions = result.get("screening_conditions", {})
    is_mixture = result.get("mixture", {}).get("pfas_count", 1) > 1
    matched = (
        routing.get("database_matched_pairs")
        if is_mixture
        else routing.get("database_matched_candidates")
    ) or 0
    with st.container(border=True, key="ar-basis"):
        _subhead("Basis of these numbers")
        if is_mixture:
            total_pairs = routing.get("total_candidate_pfas_pairs", 0)
            st.markdown(
                f":green-badge[Database first] Comparable records supported **{matched} of {total_pairs}** "
                "screened candidate–PFAS pairs; the final model filled unmatched pairs."
            )
            st.markdown(
                ":violet-badge[Mixture rule] Adsorbents are ranked by their **lowest log Kd** across the "
                "entered PFAS, so strong performance for one compound cannot mask a weak one."
            )
            st.caption(
                "Each PFAS is evaluated independently at its own C₀ under the shared water chemistry. "
                "Competitive adsorption among PFAS is not represented by the current database/model workflow."
            )
        elif matched:
            st.markdown(
                f":green-badge[Database first] Comparable records were found for **{matched}** "
                f"candidate(s). The final model filled the remaining "
                f"**{routing.get('model_fallback_candidates', 0)}**."
            )
        else:
            st.markdown(
                ":orange-badge[Model only] No database record met the exact PFAS, adsorbent and "
                "standardized water-type gates plus the numeric tolerances, so every value below "
                "comes from the final model."
            )
        dose = conditions.get("dose_mg_L")
        if dose is not None:
            if matched:
                st.markdown(
                    f":blue-badge[Common dose] The comparison target is **{dose} mg/L**. Database values "
                    "use exact isotherm recalculations where available, otherwise explicitly labelled "
                    f"near-dose observations; model fallbacks are scored at exactly {dose} mg/L."
                )
            else:
                st.markdown(
                    f":blue-badge[Common dose] All **{result.get('candidates_screened')}** eligible products "
                    f"were scored at **{dose} mg/L**."
                )
            if conditions.get("dose_within_development_support") is False:
                st.markdown(
                    ":red-badge[Outside support] This dose lies outside the range the model was "
                    "developed on, so treat the ranking as indicative only."
                )
        st.caption(
            "Database values come from exact identity matches under the displayed tolerances. "
            "Near-dose observations remain labelled approximate. ML values are estimates, not measurements."
        )


def _filter_result_items(
    items: list[dict[str, Any]],
    category_groups: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], str | None]:
    """Apply a view-only broad-class filter to the complete global ranking."""
    categories_by_key = {
        group["key"]: {str(category).casefold() for category in group["categories"]}
        for group in category_groups
        if group.get("product_count")
    }
    counts = {
        key: sum(
            str(item.get("product", {}).get("adsorbent_category", "")).casefold() in categories
            for item in items
        )
        for key, categories in categories_by_key.items()
    }
    options = ["all", *[group["key"] for group in category_groups if counts.get(group["key"], 0)]]
    labels = {"all": f"All · {len(items)}"}
    labels.update({
        group["key"]: f"{group['label']} · {counts[group['key']]}"
        for group in category_groups
        if counts.get(group["key"], 0)
    })
    selected = st.pills(
        "Filter results by adsorbent class",
        options,
        default="all",
        format_func=lambda key: labels[key],
        key="ar_result_class_filter",
        width="stretch",
    ) or "all"
    if selected == "all":
        return items, None
    categories = categories_by_key[selected]
    return [
        item for item in items
        if str(item.get("product", {}).get("adsorbent_category", "")).casefold() in categories
    ], labels[selected].rsplit(" · ", 1)[0]


def _render_result(result: dict[str, Any], category_groups: list[dict[str, Any]]) -> None:
    status = result["status"]
    if status == "pfas_not_ready":
        st.error(
            "At least one PFAS is not ready for recommendation because its reviewed feature-cache row "
            "is unavailable. Resolve every PFAS in the mixture before screening.",
            icon=":material/error:",
        )
        unresolved = result.get("unresolved_pfas", [])
        if unresolved:
            st.dataframe(pd.DataFrame([
                {
                    "PFAS input": row.get("input", {}).get("identity"),
                    "Identity type": row.get("input", {}).get("identity_type"),
                    "Lookup status": row.get("lookup", {}).get("status"),
                }
                for row in unresolved
            ]), width="stretch", hide_index=True)
            for row in unresolved:
                request = row.get("offline_feature_request")
                if request:
                    with st.expander(f"Offline feature request · {row.get('input', {}).get('identity')}"):
                        st.json(request)
            return
        request = result.get("offline_feature_request")
        if request:
            st.json(request)
        else:
            st.info(
                "Use an exact cached abbreviation, name, CAS number, or SMILES, then update the "
                "reviewed PFAS feature cache if needed.",
                icon=":material/info:",
            )
        return
    if status != "recommendations_ready":
        scope = result.get("screening_scope", {})
        if scope.get("selected_categories") and not scope.get("screened_product_count"):
            st.warning(
                "No catalog product belongs to the selected adsorbent class(es): "
                f"{', '.join(scope['selected_categories'])}. Widen the selection and screen again.",
                icon=":material/filter_alt:",
            )
        else:
            st.warning("No catalog candidates could be recommended for this scenario.", icon=":material/filter_alt:")
        rejected = result.get("rejected_candidates", [])
        if rejected:
            st.dataframe(pd.DataFrame(rejected), width="stretch")
        return

    lookups = result.get("pfas_lookups") or [result.get("pfas_lookup", {})]
    conditions = result.get("screening_conditions", {})
    all_items = result["recommendations"]
    dose = conditions.get("dose_mg_L")

    pfas_labels = [lookup.get("cache_key") or lookup.get("canonical_smiles") for lookup in lookups]
    pfas_labels = [label for label in pfas_labels if label]
    result_label = " + ".join(pfas_labels) if len(pfas_labels) <= 3 else f"{len(pfas_labels)}-PFAS mixture"
    st.subheader(f"Results for {result_label}")
    _chips(st.session_state.get("ar_scenario_chips", []))
    filtered_items, selected_class = _filter_result_items(all_items, category_groups)
    items = filtered_items[:RESULT_DISPLAY_LIMIT]
    class_filtered = selected_class is not None
    if len(filtered_items) > len(items):
        st.caption(
            f"Showing the top {len(items)} of {len(filtered_items)} candidates"
            + (f" in {selected_class}" if selected_class else "")
            + ". The complete ranked result remains in the JSON download."
        )
    view_result = {**result, "recommendations": items}

    # These counts describe the current view; screening itself always covers
    # the complete eligible catalog and never reruns when this filter changes.
    scope = result.get("screening_scope", {})
    summary = st.columns(4)
    summary[0].metric("Candidates in view", len(filtered_items), border=True)
    summary[1].metric("Candidates screened", result.get("candidates_screened", "—"), border=True)
    summary[2].metric("Reference dose", _dose_label(dose), border=True)
    # ``matched`` counts every screened candidate with a comparable record; this
    # card sits above the ranking, so it has to count the rows the ranking shows
    # or it reads as "5 of these 20" when the five are not among them.
    if result.get("mixture", {}).get("pfas_count", 1) > 1:
        shown_matched = sum(
            row.get("performance_source") == "database_match"
            for item in items for row in item.get("pfas_results", [])
        )
        shown_pairs = sum(len(item.get("pfas_results", [])) for item in items)
        summary[3].metric("Database-backed pairs", f"{shown_matched} / {shown_pairs}", border=True)
    else:
        shown_matched = sum(item.get("performance_source") == "database_match" for item in items)
        summary[3].metric("Database-backed", f"{shown_matched} / {len(items)}", border=True)

    for warning in scope.get("warnings", []):
        st.warning(warning, icon=":material/filter_alt:")
    _render_basis(result)

    tab_labels = ["Ranking", "Candidate details"]
    if result.get("rejected_candidates"):
        tab_labels.append("Excluded candidates")
    tabs = st.tabs(tab_labels)

    with tabs[0]:
        if result.get("mixture", {}).get("pfas_count", 1) > 1 and items:
            top = items[0]
            mixture = top["mixture_performance"]
            st.html(
                "<div class='ar-headline'>"
                f"<b>{escape(_product_name(top['product']))}</b> has the strongest worst-case result"
                + (f" in {escape(selected_class)}" if selected_class else "") + ": "
                f"log K<sub>d</sub> {mixture['worst_case_logKd']:.3f}, limited by "
                f"<b>{escape(str(mixture['limiting_pfas']))}</b>. It is therefore the most balanced recommendation "
                "for the entered mixture.</div>"
            )
        else:
            top_database = [item for item in items if item.get("performance_source") == "database_match"][:2]
            if len(top_database) == 2:
                first, second = top_database
                st.html(
                    "<div class='ar-headline'>At the 25 mg/L reference dose under matched conditions, "
                    f"<b>{escape(_product_name(first['product']))}</b> has log K<sub>d</sub> "
                    f"{first['performance']['mean_logKd']:.3f}, compared with "
                    f"<b>{escape(_product_name(second['product']))}</b> at "
                    f"{second['performance']['mean_logKd']:.3f}. The database evidence therefore favours "
                    f"<b>{escape(_product_name(first['product']))}</b>.</div>"
                )
        frame = _ranking_table(result, items, class_filtered)
        _basis_legend()
        st.dataframe(
            _ranking_styler(frame), width="stretch", hide_index=True,
            column_config=_ranking_column_config(),
        )
        st.caption("Open the Candidate details tab for properties, warnings, evidence, and source links.")

    with tabs[1]:
        _render_candidate_detail(view_result)

    if result.get("rejected_candidates"):
        with tabs[2]:
            st.dataframe(pd.DataFrame(result["rejected_candidates"]), width="stretch")

    st.divider()
    st.download_button(
        "Download complete recommendation (JSON)",
        json.dumps(result, indent=2, default=str).encode("utf-8"),
        "recommendation.json",
        "application/json",
        icon=":material/download:",
    )


def _render_empty_state() -> None:
    st.html(
        "<div class='ar-empty'>"
        "<div class='ar-empty-title'>No screening run yet</div>"
        "<p>Add one or more PFAS with their concentrations, enter the known water chemistry, then run the screening. "
        "Every eligible adsorbent is evaluated automatically. "
        "The ranked adsorbents and the evidence behind each score will appear here.</p>"
        "</div>"
    )


def _scenario_chips(
    pfas_entries: list[dict[str, Any]], water: dict[str, Any],
    temperature: float | None,
) -> list[tuple[str, Any]]:
    """Summarize the inputs a result was produced from.

    Results persist across reruns, so the panel has to carry the scenario it
    belongs to; otherwise a table left on screen after the form was edited
    looks like it describes the values currently in the fields.
    """
    mixture_label = "; ".join(
        f"{entry['pfas']['identity']} ({entry['initial_concentration_mg_l']:g} mg/L)"
        for entry in pfas_entries
    )
    return [
        ("PFAS mixture", mixture_label),
        ("Water", water.get("water_type") or "not specified"),
        ("pH", water.get("pH")),
        ("T", None if temperature is None else f"{temperature} °C"),
        ("Adsorbents", "all eligible products"),
    ]


def _pfas_mixture_editor(bundle: Any) -> list[dict[str, Any]]:
    """Collect PFAS through a searchable cache-backed add-and-edit surface."""
    _section(
        1,
        "PFAS mixture",
        "Search the reviewed cache or enter a name, CAS number, abbreviation, or SMILES.",
    )
    choices = _pfas_choices(bundle)
    search_column, concentration_column, add_column = st.columns(
        (4.8, 1.6, 1.35), gap="small", vertical_alignment="bottom",
    )
    with search_column:
        selected = st.selectbox(
            "Search or enter PFAS *",
            choices,
            index=None,
            # Every cached identifier is its own option, so suggestions stay
            # concise instead of concatenating abbreviation, name, and CAS.
            format_func=lambda choice: choice[2] if isinstance(choice, (tuple, list)) else str(choice),
            placeholder="e.g., PFOA, perfluorooctanoic acid, or 335-67-1",
            accept_new_options=True,
            key=PFAS_SEARCH_KEY,
        )
    with concentration_column:
        st.number_input(
            "C₀ (mg/L) *", min_value=0.0, value=0.1, format="%.6g",
            key=PFAS_ADD_C0_KEY,
        )
    with add_column:
        st.button(
            "Add PFAS", type="secondary", icon=":material/add:", width="stretch",
            disabled=selected in (None, ""), on_click=_add_selected_pfas,
        )
    if st.session_state.get("ar_pfas_add_error"):
        st.warning(st.session_state["ar_pfas_add_error"], icon=":material/warning:")

    entries = list(st.session_state.get(PFAS_ENTRIES_KEY, []))
    if not entries:
        st.caption("No PFAS added yet. Search above, set its initial concentration, and select Add PFAS.")
        return []

    updated: list[dict[str, Any]] = []
    for entry in entries:
        entry_id = int(entry["id"])
        with st.container(key=f"ar-pfas-entry-{entry_id}"):
            if entry.get("cached_choice") and any(
                entry.get(key) for key in ("abbreviation", "full_name", "cas")
            ):
                abbreviation_column, name_column, cas_column = st.columns((1.15, 2.4, 1.35), gap="small")
                with abbreviation_column:
                    _identity_field("Abbreviation", entry.get("abbreviation"))
                with name_column:
                    _identity_field("Full name", entry.get("full_name"))
                with cas_column:
                    _identity_field("CAS number", entry.get("cas"))
            else:
                identity_label = IDENTITY_TYPES.get(entry.get("identity_type"), {}).get(
                    "option", "Entered identifier",
                )
                _identity_field(identity_label, entry.get("identity"))

            status_column, c0_column, remove_column = st.columns(
                (4.8, 1.6, 1.35), gap="small", vertical_alignment="center",
            )
            with status_column:
                st.caption(
                    "Reviewed cache entry"
                    if entry.get("cached_choice")
                    else f"Custom {entry['identity_type']} · validated when screening runs"
                )
            with c0_column:
                concentration = st.number_input(
                    "C₀ (mg/L)", min_value=0.0,
                    value=float(entry["initial_concentration_mg_l"]),
                    format="%.6g", key=f"ar_pfas_c0_{entry_id}",
                )
            with remove_column:
                st.button(
                    "Remove", key=f"ar_remove_pfas_{entry_id}",
                    icon=":material/delete:", width="stretch",
                    on_click=_remove_pfas, args=(entry_id,),
                )
        updated.append({**entry, "initial_concentration_mg_l": float(concentration)})
    st.session_state[PFAS_ENTRIES_KEY] = updated
    return updated


def _parse_pfas_entries(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], str | None]:
    """Validate PFAS rows and convert them to service scenarios."""
    entries: list[dict[str, Any]] = []
    for row_number, row in enumerate(rows, start=1):
        identity = str(row.get("identity", "")).strip()
        identity_type = str(row.get("identity_type", "")).strip()
        concentration = float(row.get("initial_concentration_mg_l", 0))
        if not identity or identity_type not in IDENTITY_TYPES:
            return [], f"PFAS mixture row {row_number} has an invalid identifier."
        if concentration <= 0:
            return [], f"C₀ must be greater than zero in mixture row {row_number}."
        entries.append({
            "pfas": {"identity": identity, "identity_type": identity_type},
            "initial_concentration_mg_l": concentration,
        })
    if not entries:
        return [], "Add at least one PFAS identity and initial concentration before screening."
    return entries, None


def _scenario_fingerprint(
    pfas_rows: list[dict[str, Any]], water: dict[str, Any], temperature: float | None,
) -> str:
    """Serialize the editable scenario so stale results can be labelled clearly."""
    payload = {
        "pfas": [
            {
                "identity": row.get("identity"),
                "identity_type": row.get("identity_type"),
                "initial_concentration_mg_l": row.get("initial_concentration_mg_l"),
            }
            for row in pfas_rows
        ],
        "water": water,
        "temperature_c": temperature,
    }
    return json.dumps(payload, sort_keys=True, default=str)


def main() -> None:
    st.set_page_config(
        page_title="Adsorbent Recommender",
        page_icon="💧",
        layout="wide",
    )
    st.html(UI_STYLES)
    # The masthead is the app's identity rather than a report on the bundle, so
    # it is drawn before anything is loaded and stays put whether that load
    # succeeds or fails.
    _hero(
        "PFAS adsorbent recommender",
        "Adsorbent screening · decision support",
        "Ranks catalog adsorbents for one or more PFAS under shared water conditions using database evidence first "
        "and a pre-trained logKd model where no comparable record exists.",
    )

    bundle_dir = _resolve_bundle_dir()
    if bundle_dir is None:
        st.error(
            "No model bundle was found. Build one with build_final_logkd_model.py, "
            f"or set ADSORBENT_BUNDLE_DIR. Searched: {BUNDLE_SEARCH_ROOT}",
            icon=":material/error:",
        )
        return

    try:
        bundle = _load_bundle(str(bundle_dir))
    # A bundle trained with a regressor that is not installed here raises
    # ImportError from deep inside joblib, which without this is an unhandled
    # traceback on a blank page rather than a statement of what is missing.
    except (BundleNotReadyError, FileNotFoundError, ValueError, KeyError, ImportError) as exc:
        st.error(f"Model bundle is not ready: {exc}", icon=":material/error:")
        return

    st.html("<div style='height:1.4rem'></div>")
    category_groups = catalog_category_groups(bundle)
    # PFAS identity and water chemistry are the two parameters users revisit.
    # Adsorbent classes are screened together and become a view-only result
    # filter, leaving both editable panels visible without a wizard or scrolling.
    identity_column, water_column = st.columns((4.6, 5.4), gap="medium")
    with identity_column, _card("pfas", height="stretch"):
        pfas_rows = _pfas_mixture_editor(bundle)
    with water_column:
        water, temperature = _water_form(height="stretch")

    st.html(f"<div class='ar-required' style='margin:1.1rem 0 .5rem'>{REQUIRED_NOTE}</div>")
    fingerprint = _scenario_fingerprint(pfas_rows, water, temperature)
    existing_result = st.session_state.get("ar_result")
    inputs_changed = (
        existing_result is not None
        and st.session_state.get("ar_last_run_fingerprint") != fingerprint
    )
    if inputs_changed:
        st.info(
            "Inputs have changed since the displayed results were generated. Update the screening to compare this scenario.",
            icon=":material/edit_note:",
        )
    with st.container(key="ar-run"):
        run = st.button(
            (
                f"Update screening · {len(bundle.catalog)} eligible adsorbents"
                if inputs_changed
                else f"Screen all {len(bundle.catalog)} eligible adsorbents"
            ),
            type="primary",
            width="stretch",
            icon=":material/query_stats:",
        )
    if run:
        pfas_entries, mixture_error = _parse_pfas_entries(pfas_rows)
        if mixture_error:
            st.session_state["ar_error"] = mixture_error
        else:
            shared_scenario = {
                "temperature_c": temperature,
                "water": {key: value for key, value in water.items() if value not in (None, "") or key == "ions"},
                "adsorbent": {"categories": []},
            }
            scenarios = [{**shared_scenario, **entry} for entry in pfas_entries]
            with st.spinner("Checking comparable database records, then modelling unmatched candidates…"):
                st.session_state["ar_result"] = recommend_mixture(
                    bundle, scenarios, top_k=max(1, len(bundle.catalog)),
                )
            st.session_state["ar_scenario_chips"] = _scenario_chips(pfas_entries, water, temperature)
            st.session_state["ar_last_run_fingerprint"] = fingerprint
            st.session_state.pop("ar_error", None)

    st.html("<div style='height:1rem'></div>")
    if st.session_state.get("ar_error"):
        st.error(st.session_state["ar_error"], icon=":material/error:")
    # The result is held in session state so that expanding a candidate, opening
    # a tab or switching bundles does not wipe the screening off the screen.
    result = st.session_state.get("ar_result")
    if result is not None and result.get("result_schema_version") != RESULT_SCHEMA_VERSION:
        st.session_state.pop("ar_result", None)
        result = None
        st.info("The result format was updated. Run the screening again to refresh the recommendation table.")
    if result is None:
        _render_empty_state()
    else:
        _render_result(result, category_groups)


if __name__ == "__main__":
    main()
