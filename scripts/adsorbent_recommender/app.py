"""Streamlit interface for one selected adsorbent recommender bundle.

Run from the repository root (the ``AD`` folder), as Streamlit Community Cloud
does:
    streamlit run scripts/adsorbent_recommender/app.py

Colours, fonts and radii live in ``.streamlit/config.toml`` at the repository
root.  Streamlit resolves that file against the working directory, so launching
from the root is what applies the theme; from anywhere else the app still lays
out and reads correctly, it just falls back to Streamlit's stock palette.  The rules injected
below deliberately derive their neutrals from the app's own text colour instead
of naming greys, so they hold under either palette and in either colour mode.

The selected model-bundle directory is a deployment setting in the
``Configuration`` section below. It is intentionally not editable in the UI.
"""

from __future__ import annotations

import json
import hashlib
import io
import math
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
    resolve_pfas,
    normalize_water_scenario,
    fixed_test_conditions,
)
from adsorbent_recommender.input_validation import (
    DEFAULT_TEMPERATURE_C, SCREENING_BEHAVIOR_VERSION, InputValidationError,
    effective_temperature, finite_number,
)
from adsorbent_recommender.additive_inputs import (
    ADDITIVE_INPUT_VERSION, ADDITIVE_UNITS, COMMON_SPECIES, AdditiveInputError, validate_additives,
)

from adsorbent_recommender.pfas_descriptors import DESCRIPTOR_VERSION, prepare_structure
from adsorbent_recommender.pfas_identity import IdentityLookupError, resolve_public_identity
from adsorbent_recommender.pfas_support import UNSEEN_WARNING
from adsorbent_recommender.water_support import VERSION as WATER_SUPPORT_VERSION
from adsorbent_recommender.material_support import SUPPORT_VERSION
from adsorbent_recommender.mixture_inputs import (
    CONCENTRATION_UNITS, EXAMPLE_CSV_PATH, MixtureInputError, append_prepared_import,
    from_mg_l, normalize_unit, prepare_csv_import, read_mixture_csv, to_mg_l,
)
from adsorbent_recommender.scenarios import (
    SCENARIO_EXAMPLE_PATH, ScenarioError, add_snapshot, bundle_reference, compare_snapshots, comparison_csv,
    make_snapshot, read_scenario_csv, read_snapshot, results_csv, scenario_inputs_csv,
    snapshot_bytes, validate_inputs, recommendation_bytes, validate_result_structure,
)
from adsorbent_recommender.warning_labels import (
    SEVERITIES, column_options as warning_column_options, legend_labels, warning_labels, warning_values,
)
from rdkit import Chem

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
    padding-top: 1.1rem;
    padding-bottom: 4rem;
}

/* ---- Masthead --------------------------------------------------------- */
.ar-hero {
    border: 1px solid var(--ar-line);
    border-radius: 16px;
    padding: 1rem 1.2rem;
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
    font-size: 1.8rem;
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
.ar-warning-label {
    display: inline-flex;
    align-items: center;
    white-space: nowrap;
    border: 1px solid color-mix(in srgb, var(--warning-color) 35%, transparent);
    border-radius: 999px;
    padding: 0.1rem 0.5rem;
    background: var(--warning-fill);
    color: color-mix(in srgb, var(--warning-color) 35%, currentColor);
    font-weight: 600;
    font-size: 0.82rem;
}
.ar-warning-legend {
    display: flex;
    flex-wrap: wrap;
    gap: 0.55rem 1.2rem;
    margin: 0.5rem 0 0.3rem;
}
.ar-warning-definition { flex: 1 1 360px; font-size: 0.88rem; line-height: 1.5; }
.ar-warning-detail { margin: 0.4rem 0; line-height: 1.5; }
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
    "Fields marked <b>*</b> are required. Input defaults and screening assumptions are documented in the README."
)
# Product names identify materials as reported in the source literature; the
# notice keeps that use from reading as endorsement or a performance claim.
PRODUCT_NAME_DISCLAIMER = (
    "Product names are trademarks of their respective owners and are used only to identify materials "
    "reported in the scientific literature. Their inclusion does not imply endorsement, recommendation, "
    "or affiliation by the authors or their institution. Scores are research estimates of adsorption "
    "affinity derived from published data and an XGBoost machine-learning model; they are not performance claims "
    "or guarantees for any product. Commercial formulations may change over time and vary between lots. "
    "This tool is provided \"as is\", without warranty of any kind."
)
PFAS_ENTRIES_KEY = "ar_pfas_entries_v2"
PFAS_NEXT_ID_KEY = "ar_pfas_next_id_v2"
PFAS_SEARCH_KEY = "ar_pfas_search_v2"
PFAS_ADD_C0_KEY = "ar_pfas_add_c0_v2"
PFAS_ADD_UNIT_KEY = "ar_pfas_add_unit_v1"
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


def _alias_key(value: str) -> str:
    """Normalize a name, abbreviation or CAS number for alias matching.

    Case, spacing and punctuation vary freely in how these are written
    ("PFOA", "pfoa", "335-67-1", "33567-1"), so only letters and digits count.
    SMILES are matched verbatim instead, since their punctuation is structure.
    """
    return "".join(character for character in value.casefold() if character.isalnum())


@st.cache_data(show_spinner=False)
def _load_pfas_choices(
    path: str, sheet_name: str, modified_ns: int,
) -> tuple[list[PfasChoice], dict[str, PfasChoice]]:
    """Return one dropdown option per compound plus an index of its aliases.

    The dropdown lists each compound once and formats its abbreviation, full
    name, and CAS together. The alias index also resolves a typed exact alias
    or SMILES to the same cached compound.
    """
    del modified_ns  # Part of the cache key so a replaced workbook is re-read.
    frame = pd.read_excel(path, sheet_name=sheet_name)
    choices: list[PfasChoice] = []
    aliases: dict[str, PfasChoice] = {}
    seen: set[str] = set()
    for _, row in frame.iterrows():
        abbreviation = _clean_display_value(row.get("Abbreviation"))
        name = _clean_display_value(row.get("PFAS_name")) or _clean_display_value(row.get("Compound Full Name"))
        cas = _clean_display_value(row.get("CAS Number")) or _clean_display_value(row.get("CAS"))
        smiles = _clean_display_value(row.get("Canonical SMILES")) or _clean_display_value(row.get("SMILES"))
        cache_identity = smiles or abbreviation or cas or name
        if not cache_identity or cache_identity.casefold() in seen:
            continue
        seen.add(cache_identity.casefold())
        # A compound without an abbreviation is listed under the next most
        # recognisable identifier rather than dropped from the dropdown.
        identity_type, identity = next(
            (kind, value)
            for kind, value in (
                ("abbreviation", abbreviation), ("name", name), ("cas", cas), ("smiles", smiles),
            )
            if value
        )
        choice: PfasChoice = (
            identity_type, identity, identity,
            abbreviation, name, cas, cache_identity,
        )
        choices.append(choice)
        for value in (abbreviation, name, cas):
            if value and _alias_key(value):
                aliases.setdefault(_alias_key(value), choice)
        if smiles:
            aliases.setdefault(smiles, choice)
    # Abbreviations first; compounds listed under a long systematic name follow,
    # rather than leading the list because their names start with locants.
    choices.sort(key=lambda choice: (choice[0] != "abbreviation", choice[2].casefold()))
    return choices, aliases


def _pfas_choices(bundle: Any) -> tuple[list[PfasChoice], dict[str, PfasChoice]]:
    path, sheet_name = pfas_features_source(bundle)
    if not path.is_file() or not sheet_name:
        return [], {}
    return _load_pfas_choices(str(path), str(sheet_name), path.stat().st_mtime_ns)


def _resolve_typed_pfas(aliases: dict[str, PfasChoice]) -> None:
    """Swap a typed full name, CAS number or SMILES for its cached compound.

    When free text names a cached compound, replace it with the reviewed
    choice. Unrecognized text remains on the explicit structure-review path.
    """
    st.session_state.pop("ar_pfas_add_error", None)
    selected = st.session_state.get(PFAS_SEARCH_KEY)
    if not isinstance(selected, str) or not selected.strip():
        return
    text = selected.strip()
    match = aliases.get(text) or aliases.get(_alias_key(text))
    if match is not None:
        st.session_state[PFAS_SEARCH_KEY] = match


def _pfas_choice_label(choice: Any) -> str:
    """Expose searchable names and CAS numbers without duplicating compounds."""
    if not isinstance(choice, (tuple, list)) or len(choice) != 7:
        return str(choice)
    return " · ".join(dict.fromkeys(value for value in (choice[2], choice[4], choice[5]) if value))


@st.cache_data(max_entries=128, show_spinner=False)
def _structure_png(canonical: str) -> bytes | None:
    # RDKit drawing needs X11 libraries (libXrender) that a bare Linux host may lack;
    # without them the SMILES shown next to each image still identifies the structure.
    try:
        from rdkit.Chem import Draw
    except ImportError:
        return None
    stream = io.BytesIO()
    Draw.MolToImage(Chem.MolFromSmiles(canonical), size=(480, 240)).save(stream, format="PNG")
    return stream.getvalue()


def _show_structure(canonical: str, width: Any = "stretch") -> None:
    png = _structure_png(canonical)
    if png is not None:
        st.image(png, width=width)


def _custom_structure_editor(bundle: Any, selected: str) -> None:
    """Resolve a public identifier explicitly and preview the exact structure before adding."""
    st.session_state.pop("ar_pending_structure", None)
    kind = st.segmented_control("Interpret entered identifier as", ["name", "cas", "smiles"],
        default="name", format_func=lambda value: {"name": "Name", "cas": "CAS number", "smiles": "SMILES"}[value],
        key="ar_custom_identity_type") or "name"
    query = selected.strip()
    stamp = hashlib.sha256(json.dumps([query, kind, DESCRIPTOR_VERSION]).encode()).hexdigest()[:16]
    resolution = {"source": "submitted_smiles", "query": query, "identity_type": kind}
    candidate_smiles = query if kind == "smiles" else None
    if kind != "smiles":
        st.caption("Resolve the identifier with PubChem, then confirm the intended molecular structure. You can always switch to SMILES.")
        if st.button("Find structures in PubChem", key="ar_resolve_identity", width="stretch"):
            try:
                with st.spinner("Looking up molecular structures…"):
                    candidates = resolve_public_identity(query, kind)
                st.session_state["ar_public_resolution"] = {"stamp": stamp, "candidates": candidates}
                st.session_state.pop("ar_public_error", None)
            except IdentityLookupError as exc:
                st.session_state["ar_public_error"] = {"stamp": stamp, "message": str(exc)}
                st.session_state.pop("ar_public_resolution", None)
        error = st.session_state.get("ar_public_error", {})
        if error.get("stamp") == stamp:
            st.warning(error["message"])
        response = st.session_state.get("ar_public_resolution", {})
        if response.get("stamp") == stamp:
            candidates = response["candidates"]
            if not candidates:
                st.info("No exact PubChem match was found. Check the identifier or switch to SMILES.")
            else:
                index = st.selectbox("Select the intended compound", list(range(len(candidates))), index=None,
                    format_func=lambda i: f"{candidates[i]['title']} · PubChem CID {candidates[i]['cid']}",
                    key=f"ar_public_choice_{stamp}")
                if index is not None:
                    resolution = candidates[index]
                    candidate_smiles = resolution["smiles"]
                    st.markdown(f"[Inspect this PubChem record]({resolution['url']})")
    if not candidate_smiles:
        return
    try:
        # Validation and normalization are shared with the service. The user's label does not enter evidence matching.
        lookup, _ = resolve_pfas(bundle, {"pfas": {"identity": candidate_smiles, "identity_type": "smiles"}})
        if not lookup.ready:
            for message in lookup.warnings:
                st.warning(message)
            return
        canonical = lookup.canonical_smiles
        _, structure_key = prepare_structure(canonical)
        _show_structure(canonical)
        st.caption("Structure that will be used for prediction")
        st.code(canonical, language=None)
        for message in lookup.warnings:
            st.caption(message)
        label = st.text_input("Display label (optional)", key=f"ar_custom_label_{stamp}",
                              placeholder=resolution.get("title", "e.g., investigation compound"))
        st.session_state["ar_pending_structure"] = {
            "identity": query, "identity_type": kind,
            "label": label.strip() or resolution.get("title") or query,
            "canonical_smiles": canonical, "structure_key": structure_key,
            "confirmed_smiles": candidate_smiles, "resolution": resolution,
        }
    except (ValueError, RuntimeError) as exc:
        st.warning(str(exc))


def _add_selected_pfas() -> None:
    selected = st.session_state.get(PFAS_SEARCH_KEY)
    concentration = st.session_state.get(PFAS_ADD_C0_KEY)
    if selected in (None, ""):
        st.session_state["ar_pfas_add_error"] = "Choose or enter a PFAS before adding it."
        return
    if concentration is None or not math.isfinite(float(concentration)) or float(concentration) <= 0:
        st.session_state["ar_pfas_add_error"] = "Initial concentration must be greater than zero."
        return
    unit = st.session_state.get(PFAS_ADD_UNIT_KEY, "mg/L")
    try:
        concentration_mg_l = to_mg_l(concentration, unit)
    except MixtureInputError as exc:
        st.session_state["ar_pfas_add_error"] = str(exc)
        return
    extra = {}
    if isinstance(selected, (tuple, list)) and len(selected) in {3, 7}:
        identity_type, identity, label = map(str, selected[:3])
        abbreviation, full_name, cas, cache_identity = map(str, selected[3:]) if len(selected) == 7 else ("", "", "", identity)
        cached = True
        try:
            _, structure_key = prepare_structure(cache_identity)
        except ValueError:
            structure_key = ""
    else:
        pending = st.session_state.get("ar_pending_structure", {})
        if pending.get("identity") != str(selected).strip():
            st.session_state["ar_pfas_add_error"] = "Resolve and review the intended structure before adding this PFAS."
            return
        identity, identity_type, label = pending["identity"], pending["identity_type"], pending["label"]
        abbreviation = full_name = cas = ""
        cache_identity = pending["canonical_smiles"]
        structure_key = pending["structure_key"]
        cached = False
        extra = {key: pending[key] for key in ("confirmed_smiles", "resolution", "canonical_smiles")}
        extra["structure_confirmed"] = True
    entries = list(st.session_state.get(PFAS_ENTRIES_KEY, []))
    duplicate = any(
        (structure_key and entry.get("structure_key") == structure_key)
        or (identity_type != "smiles" and entry.get("identity_type") == identity_type
            and _alias_key(entry.get("identity", "")) == _alias_key(identity))
        for entry in entries
    )
    if duplicate:
        st.session_state["ar_pfas_add_error"] = f"{label} is already in the mixture."
        return
    next_id = int(st.session_state.get(PFAS_NEXT_ID_KEY, 1))
    entries.append({"id": next_id, "identity": identity, "identity_type": identity_type, "label": label,
        "abbreviation": abbreviation, "full_name": full_name, "cas": cas, "cached_choice": cached,
        "structure_key": structure_key, "normalized_identity": structure_key or _alias_key(cache_identity),
        "initial_concentration_mg_l": concentration_mg_l,
        "concentration_value": float(concentration), "concentration_unit": unit, **extra})
    st.session_state[PFAS_ENTRIES_KEY] = entries
    st.session_state[PFAS_NEXT_ID_KEY] = next_id + 1
    st.session_state[PFAS_SEARCH_KEY] = None
    st.session_state[PFAS_ADD_C0_KEY] = None
    st.session_state.pop("ar_pending_structure", None)
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
    errors: dict[str, str] | None = None,
    nonnegative: bool = False,
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
    try:
        return finite_number(raw, display_label, optional=True, nonnegative=nonnegative, allow_text=True)
    except InputValidationError as exc:
        if errors is not None:
            errors[key] = str(exc)
        st.error(str(exc))
        return None


@st.cache_resource(show_spinner=False)
def _load_bundle(directory: str, manifest_mtime_ns: int = 0):
    return load_recommender_bundle(Path(directory))


def _add_inorganic_entry() -> None:
    try:
        entry = {"species": st.session_state.get("ar_salt_species"), "concentration": st.session_state.get("ar_salt_concentration"),
                 "unit": st.session_state.get("ar_salt_unit", "mM")}
        st.session_state["ar_inorganic_entries"] = validate_additives([*st.session_state.get("ar_inorganic_entries", []), entry])
        st.session_state.pop("ar_salt_error", None)
    except AdditiveInputError as exc:
        st.session_state["ar_salt_error"] = str(exc)


def _remove_inorganic_entry(index: int) -> None:
    st.session_state["ar_inorganic_entries"] = [row for position, row in enumerate(st.session_state.get("ar_inorganic_entries", [])) if position != index]
    st.session_state.pop("ar_salt_error", None)


def _inorganic_editor() -> list[dict[str, Any]]:
    st.session_state.setdefault("ar_inorganic_entries", [])
    st.caption("Add each inorganic species with its concentration and unit.")
    st.selectbox("Species", COMMON_SPECIES, index=None, accept_new_options=True, key="ar_salt_species", placeholder="e.g., NaCl")
    with st.container(horizontal=True, vertical_alignment="bottom"):
        st.number_input("Concentration", min_value=0.0, value=1.0, format="%g", key="ar_salt_concentration", width=200)
        st.selectbox("Unit", ADDITIVE_UNITS, key="ar_salt_unit", width=130)
    st.button("Add", key="ar_add_inorganic", on_click=_add_inorganic_entry, disabled=not st.session_state.get("ar_salt_species"))
    if st.session_state.get("ar_salt_error"):
        st.error(st.session_state["ar_salt_error"])
    entries = st.session_state["ar_inorganic_entries"]
    for index, entry in enumerate(entries):
        with st.container(horizontal=True, vertical_alignment="center"):
            st.write(f"**{entry['species']}** · {entry['concentration']:g} {entry['unit']}")
            st.button("Remove", key=f"ar_remove_inorganic_{index}", on_click=_remove_inorganic_entry, args=(index,))
    return entries


def _water_form(height: str = "content", default_temperature: float = DEFAULT_TEMPERATURE_C) -> tuple[dict[str, Any], float | None, dict[str, str]]:
    errors: dict[str, str] = {}
    with _card("water", height=height):
        _section(2, "Water chemistry", "Enter the common conditions first. Add detailed matrix chemistry only when it is known.")
        with st.container(horizontal=True, vertical_alignment="bottom"):
            water_type = st.selectbox(
                "Water type",
                WATER_TYPE_OPTIONS,
                key="water_type",
                format_func=lambda value: value or "Ultrapure water (default)",
                help="With no background additives, unspecified water defaults to ultrapure. Reported background additives are normalized as synthetic water, consistently with the database.",
                width=260,
            )
        general_middle, general_right = st.columns(2)
        with general_middle:
            ph = _number_input("pH", "ph", "7.2", errors=errors)
        with general_right:
            temperature = _number_input(
                "Temperature", "temperature", "25", "°C",
                help_text=f"Left blank, database matching and model predictions use {default_temperature:g} °C.",
                errors=errors,
            )
        organic_count = int(bool(str(st.session_state.get("organic_carbon", "")).strip()))
        organic_count += bool(st.session_state.get("organic_present", False))
        organic_label = "Organic chemistry"
        if organic_count:
            organic_label += f" · {organic_count} value{'s' if organic_count != 1 else ''} entered"
        with st.expander(organic_label, expanded=False, icon=":material/eco:"):
            st.caption("Leave any unmeasured value blank. Closing this section preserves entered values.")
            organic_left, organic_right = st.columns(2, vertical_alignment="bottom")
            with organic_left:
                organic_carbon = _number_input("Organic carbon", "organic_carbon", "5", "mg/L", errors=errors, nonnegative=True)
            with organic_right:
                organic_present = st.checkbox("Organic additives present", key="organic_present")

        inorganic_count = sum(
            bool(str(st.session_state.get(key, "")).strip())
            for key in ("tds", "ionic_strength")
        )
        inorganic_count += len(st.session_state.get("ar_inorganic_entries", []))
        inorganic_count += sum(bool(st.session_state.get(f"ion_{ion}")) for ion in ION_LABELS)
        inorganic_label = "Inorganic chemistry"
        if inorganic_count:
            inorganic_label += f" · {inorganic_count} value{'s' if inorganic_count != 1 else ''} entered"
        with st.expander(inorganic_label, expanded=False, icon=":material/science:"):
            st.caption("Leave any unmeasured value blank. Closing this section preserves entered values.")
            inorganic_left, inorganic_middle = st.columns(2)
            with inorganic_left:
                tds = _number_input("TDS", "tds", "500", "mg/L", errors=errors, nonnegative=True)
            with inorganic_middle:
                ionic_strength = _number_input("Ionic strength", "ionic_strength", "0.01", "mol/L", errors=errors, nonnegative=True)
            inorganic = _inorganic_editor()
            # The ions are themselves inorganic additives, so they belong with
            # the other inorganic fields.
            ion_values = {}
            with st.container(key="ar-ions", border=True):
                st.caption(
                    "Ions in listed additives are recognized automatically. Select other known ions only when their concentrations are unmeasured."
                )
                ions = st.columns(4)
                for index, (ion, label) in enumerate(ION_LABELS.items()):
                    with ions[index % 4]:
                        ion_values[ion] = st.checkbox(label, key=f"ion_{ion}")
            inorganic_preview = st.empty()
    water = {
        "water_type": water_type, "pH": ph, "organic_carbon_mg_l": organic_carbon, "tds_mg_l": tds,
        "ionic_strength_mol_l": ionic_strength, "inorganic_additives": inorganic,
        "organic_matter_present": organic_present, "ions": ion_values,
    }
    if inorganic and not errors:
        normalized, chemistry_warnings = normalize_water_scenario({"water": water})
        with inorganic_preview.container():
            ionic = _clean_display_value(normalized.get("Ionic_strength_(mol/L)"))
            st.caption(f"Ionic strength {'using the entered value' if ionic_strength is not None else 'from listed additives'}: {ionic or 'undetermined'} mol/L. Unlisted background salts are not included.")
            st.caption(f"Water type used for screening: {normalized.get('Water_type') or 'unreported'}.")
            if "pH-adjusters ignored" in str(normalized.get("normalization_trace", "")):
                st.caption("Strong acid/base pH adjusters are excluded from additive-derived ionic strength under the database convention.")
            for warning in chemistry_warnings:
                st.warning(warning)
    return water, temperature, errors


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
    threshold = _resolution_threshold(result)
    # Measured against the top of the current view, so a class filter compares
    # each product with the best one the user is actually looking at.
    top_score = float(displayed[0]["ranking_score"]) if displayed else None
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
            "Warnings": warning_values(item),
            "Within error of top": (
                None if threshold is None or top_score is None
                else top_score - float(item["ranking_score"]) <= threshold
            ),
            # Two products can share a score while resting on very different
            # amounts of evidence, so the basis travels with the ranking.
            "Characterized on": None if present is None else f"{present}/{required} features",
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
            "Limiting PFAS", "Within error of top", "Basis", "Warnings", "Support",
            "Characterized on",
        ]
    else:
        column_order = rank_columns + [
            "Product", "Class", "Category", "logKd at reference dose", "Within error of top",
            "Basis", "Warnings", "Support", "Characterized on",
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


def _ranking_column_config(items: list[dict[str, Any]] | None = None) -> dict[str, Any]:
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
        "Warnings": st.column_config.MultiselectColumn(
            "Warnings", disabled=True, **warning_column_options(items or []),
            help="Rose ! labels mark critical alerts; teal i labels provide supporting context. The legend explains each label. Select a row for the full text and affected PFAS.",
        ),
        "Within error of top": st.column_config.CheckboxColumn(
            "Within error of #1",
            help="Scores within the cross-validated error of a predicted difference between two adsorbents (about 0.5 log units) of the top candidate in this view. "
            "Treat small differences cautiously; this threshold is a screening heuristic, not a calibrated confidence interval.",
        ),
    }
    return config


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


def _render_usage_guidance() -> None:
    # The tool's purpose opens the reading guide rather than sitting above the
    # form as a warning: it sets expectations, it does not flag a problem.
    with st.expander("How to read these results"):
        st.markdown(
            "**What it is for:** shortlisting adsorbents worth testing. The ranking compares adsorption "
            "affinity (logKd) under the conditions you enter; it does not establish treatment performance "
            "at a site, so validate shortlisted materials with water-specific experiments before "
            "selecting an adsorbent."
        )
        st.markdown(
            "The app screens catalog activated carbons, ion-exchange and nonionic resins, and "
            "cyclodextrin polymers. A comparable curated database record is used when available; "
            "otherwise the Global XGBoost model estimates logKd at the common reference dose. "
            "The **Basis** column identifies the source. The **Top-performing tier** contains scores within "
            "the cross-validated error of a predicted difference between two adsorbents (about 0.5 log units) of the top score; "
            "this is a screening heuristic, not a confidence interval. "
            "**Warnings** shows concise labels for recorded score caveats: rose ! labels mark critical alerts, "
            "while teal i labels provide supporting context. The legend below the table explains each label."
        )
        st.markdown("Select a shortlist row to open the full warning text, affected PFAS, and evidence. Input defaults, evidence-matching rules, and interpretation assumptions are documented in the README.")
        readme = Path(__file__).with_name("README.md")
        if readme.is_file():
            st.download_button("Download README and assumptions", readme.read_bytes(), file_name="adsorbent_recommender_README.md",
                mime="text/markdown", key="ar_readme_download", icon=":material/description:")
        st.markdown(
            "**Most reliable:** model estimates for a PFAS and a product that both appear in the training "
            "data but were not tested together. This is the case the model was tuned for.\n\n"
            "**Less reliable:** products or PFAS absent from the training data, inputs outside the training "
            "range, products characterized on few properties, and logKd below 0 or above 4. Each of these "
            "is flagged as a caveat. Prediction error also grows for experiments unlike any in the database, "
            "because studies differ in protocol and reporting in ways the inputs do not capture."
        )
        st.markdown(
            "**Close scores are ties.** Two candidates are compared through the difference of two predictions, "
            "whose cross-validated error is about 0.5 log units, so candidates within that margin of each other "
            "should be treated as one group to test, "
            "not as a ranked order."
        )
        st.markdown(
            "**Water chemistry:** most source studies held pH, organic matter and ions at a single value, "
            "so the model learned their effects mainly from differences between studies. Rankings that "
            "change only because water chemistry was changed should be interpreted with caution."
        )
        st.markdown(
            "**Not modeled:** the ensemble's standard deviation is disagreement among fitted models, not a "
            "calibrated prediction interval. Mixture scores evaluate each PFAS separately and do not model "
            "competitive adsorption. logKd rankings do not predict breakthrough, removal targets, adsorbent "
            "lifetime, or operating cost."
        )
        st.markdown(f"**Product names:** {PRODUCT_NAME_DISCLAIMER}")
        paper_urls = _http_sources(os.environ.get("ADSORBENT_PAPER_URL"))
        if paper_urls:
            st.markdown(f"[Read the study]({paper_urls[0]})")
        else:
            st.caption("The study manuscript is in preparation. A paper link will appear here when available.")


def _candidate_caveats(item: dict[str, Any]) -> list[dict[str, Any]]:
    """Caveats for the score a candidate is ranked by, most severe first.

    The service decides which caveats belong to the score actually used, so a
    database-backed score never inherits the warnings of the model prediction
    computed beside it.  Here they only gain the PFAS they apply to.
    """
    caveats = []
    for caveat in item.get("caveats", []):
        names = [str(name) for name in caveat.get("pfas") or [] if name]
        prefix = f"{', '.join(names)}: " if names else ""
        caveats.append({**caveat, "display": prefix + caveat["text"]})
    return caveats


# Said once in the support panels, rather than on every candidate row.
PFAS_LEVEL_CAVEATS = {"unseen_pfas", "pfas_descriptors_out_of_range", "limited_water_evidence"}


def _row_flags(caveats: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """The caveats worth showing beside a score: critical and specific to the row."""
    return [
        caveat for caveat in caveats
        if caveat["severity"] == "critical" and caveat["code"] not in PFAS_LEVEL_CAVEATS
    ]


def _warning_chip(tag: Any) -> str:
    spec = SEVERITIES[tag.severity]
    return f"<span class='ar-warning-label' style='--warning-color:{spec['color']};--warning-fill:{spec['fill']}'>{escape(tag.label)}</span>"


def _render_warning_legend(items: list[dict[str, Any]]) -> None:
    tags = legend_labels(items)
    if not tags:
        return
    st.markdown("**Warning label legend**")
    present = {tag.severity for tag in tags}
    st.caption(" · ".join(spec["legend"] for severity, spec in SEVERITIES.items() if severity in present))
    definitions = "".join(
        f"<div class='ar-warning-definition'>{_warning_chip(tag)} {escape(tag.meaning)}</div>"
        for tag in tags
    )
    st.html(f"<div class='ar-warning-legend'>{definitions}</div>")


def _resolution_threshold(result: dict[str, Any]) -> float | None:
    resolution = result.get("rank_resolution") or {}
    return resolution.get("threshold_log_units") if resolution.get("available") else None


def _render_scenario_caveats(result: dict[str, Any]) -> None:
    """Caveats shared by every candidate, said once rather than on every row."""
    # The PFAS support panel already explains PFAS-level limits, and the reading
    # guide covers blank inputs and how database matches work.
    serious = [
        caveat["text"] for caveat in result.get("scenario_caveats", [])
        if caveat["severity"] == "critical" and caveat["code"] not in PFAS_LEVEL_CAVEATS
    ]
    if serious:
        st.warning(
            "**Applies to every candidate**\n\n" + "\n".join(f"- {text}" for text in serious),
            icon=":material/warning:",
        )


def _render_rank_resolution(result: dict[str, Any], items: list[dict[str, Any]]) -> None:
    threshold = _resolution_threshold(result)
    if threshold is None or len(items) < 2:
        return
    top = float(items[0]["ranking_score"])
    close = sum(top - float(item["ranking_score"]) <= threshold for item in items)
    # How to read the order, not a warning: plain text under the heading.
    if close > 1:
        group = f"All **{close}** candidates shown are" if close == len(items) else f"The top **{close}** candidates are"
        st.markdown(
            f"{group} within the model's error ({threshold:.2f} log units) of the top score; "
            "test them as a group rather than in rank order."
        )
    else:
        gap = top - float(items[1]["ranking_score"])
        st.markdown(
            f"The top candidate leads the next by {gap:.2f} log units, more than the model's error "
            f"({threshold:.2f} log units)."
        )


def _related_evidence_details(evidence: dict[str, Any], product_key: str, pfas_index: int, key_prefix: str = "") -> None:
    related = evidence.get("related_evidence", {})
    count = related.get("record_count", 0)
    if not count:
        return
    st.caption(f"{count} related record(s) from {related.get('study_count', 0)} identified studies have different or unreported conditions; their values are not used for ranking.")
    with st.popover("View studies and condition differences", key=f"ar_related_{key_prefix}{product_key}_{pfas_index}", icon=":material/article:"):
        st.caption(related.get("summary", "Same PFAS and catalog product, different conditions."))
        records = related.get("records", [])
        rows = []
        for record in records:
            differences = []
            for difference in record.get("condition_differences", []):
                observed = _clean_display_value(difference.get("observed")) or "unreported"
                requested = _clean_display_value(difference.get("requested")) or "unreported"
                differences.append(f"{difference['label']}: {observed} vs requested {requested} ({difference['reason']})")
            rows.append({"Record": record.get("standardized_row_id"), "Study": record.get("study_no"),
                "Water type": record.get("water_type"), "C₀ (mg/L)": record.get("conditions", {}).get("initial_concentration_mg_L"),
                "Reference-dose logKd": record.get("logKd_log10_L_g"), "Basis": record.get("reference_dose_basis"),
                "Condition differences": "; ".join(differences)})
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
        st.caption(f"Showing {len(records)} representative records of {count}, chosen by condition agreement and study coverage, without favoring higher logKd values. These are contextual evidence, not comparable experimental scores.")
        seen = set()
        for record in records:
            citation = record.get("citation") or {}
            label = citation.get("label") or citation.get("doi") or "Source study"
            links = _http_sources(citation.get("url"))
            key = links[0] if links else label
            if key in seen:
                continue
            seen.add(key)
            if links:
                st.link_button(label, links[0], key=f"ar_study_{key_prefix}{product_key}_{pfas_index}_{len(seen)}", icon=":material/open_in_new:")
            else:
                st.write(label)


def _render_candidate_detail(result: dict[str, Any], *, expanded: bool = False, key_prefix: str = "") -> None:
    for item in result["recommendations"]:
        performance = item.get("performance", item["prediction"])
        basis = _basis_label(item.get("performance_source"))
        mixture = item.get("mixture_performance")
        if mixture and mixture.get("pfas_count", 1) > 1:
            score_label = f"worst-case log Kd {mixture['worst_case_logKd']:.3f}"
        else:
            score_label = f"log Kd {performance['mean_logKd']:.3f}"
        header = f"#{item['rank']} · {_product_name(item['product'])} · {score_label} · {basis}"
        with st.expander(header, expanded=expanded):
            caveats = _candidate_caveats(item)
            for caveat in caveats:
                tag = warning_labels({"caveats": [caveat]})[0]
                st.html(f"<div class='ar-warning-detail'>{_warning_chip(tag)} {escape(caveat['display'])}</div>")
            characterization = item.get("characterization", {})
            support = item.get("adsorbent_support", {})
            if support.get("available"):
                st.caption(f"Product training support: {support['source_records']} distinct source records from {support['independent_studies']} DOI-identified studies ({support['training_rows']} fitted rows).")
                if support.get("reasons"):
                    st.caption("ML support limits: " + " ".join(support["reasons"]))
            elif characterization.get("training_records"):
                st.caption(
                    f"Training data for this product: {characterization['training_records']} records "
                    f"from {characterization.get('training_studies', 0)} studies."
                )

            pfas_results = item.get("pfas_results", [])
            if len(pfas_results) > 1:
                _subhead("Performance across the PFAS mixture")
                performance_rows = []
                for pfas_result in pfas_results:
                    value = pfas_result.get("performance") or pfas_result.get("prediction") or {}
                    performance_rows.append({
                        "PFAS": pfas_result.get("pfas"),
                        "C₀": (f"{pfas_result['concentration_input']['value']:g} {pfas_result['concentration_input']['unit']}"
                               if pfas_result.get("concentration_input") else f"{pfas_result.get('initial_concentration_mg_l'):g} mg/L"),
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
            _subhead("Experimental evidence")
            for pfas_index, (pfas_label, evidence) in enumerate(evidence_sets):
                if pfas_label:
                    st.markdown(f"**{pfas_label}**")
                if evidence.get("status") == "matched":
                    performance = evidence.get("performance", {})
                    st.caption(f"Comparable evidence used for ranking: {performance.get('match_count', 0)} record(s) across {performance.get('study_count', 0)} studies.")
                    st.dataframe(pd.DataFrame(evidence.get("records", [])), width="stretch", hide_index=True)
                    for citation in evidence.get("citations", []):
                        source_urls = _http_sources(citation.get("url"))
                        citation_label = citation.get("label") or citation.get("doi") or "Performance source"
                        if source_urls:
                            st.markdown(f"- [{citation_label}]({source_urls[0]})")
                        else:
                            st.write(f"- {citation_label}")
                else:
                    st.caption("Model prediction. No comparable experimental records for this request.")
                    if not evidence.get("related_evidence", {}).get("record_count"):
                        st.caption(f"Database fallback reason: {evidence.get('reason', 'no comparable record')}")
                _related_evidence_details(evidence, item["product"]["Product_key"], pfas_index, key_prefix)

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
    # The app screens one PFAS through the mixture path too, which counts
    # candidate–PFAS pairs; a single-PFAS pair count is the candidate count.
    matched = routing.get("database_matched_pairs", routing.get("database_matched_candidates")) or 0
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
                f"**{routing.get('model_fallback_pairs', routing.get('model_fallback_candidates', 0))}**."
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


def _render_pfas_support(result: dict[str, Any]) -> None:
    lookups = result.get("pfas_lookups") or [result.get("pfas_lookup", {})]
    with st.container(border=True):
        _subhead("PFAS support")
        rows = []
        for lookup in lookups:
            support = lookup.get("support", {})
            seen = support.get("seen_in_training")
            rows.append({"PFAS": lookup.get("label") or lookup.get("cache_key") or lookup.get("canonical_smiles"),
                "Features": "Computed at runtime" if lookup.get("feature_source") == "computed_at_runtime" else "Reviewed cache",
                "Model training": "Seen" if seen is True else "Unseen · exploratory" if seen is False else "Unknown",
                "Descriptors outside range": len(support.get("out_of_range_descriptors", []))})
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
        if any(lookup.get("support", {}).get("seen_in_training") is False for lookup in lookups):
            st.warning(UNSEEN_WARNING, icon=":material/warning:")
        with st.expander("Structures and nearest training PFAS"):
            st.caption("Similarity describes structural resemblance; it does not establish prediction reliability. Ensemble SD is not a calibrated prediction interval.")
            for lookup in lookups:
                st.markdown(f"**{escape(str(lookup.get('label') or lookup.get('cache_key') or 'PFAS'))}**")
                st.code(lookup.get("canonical_smiles", ""), language=None)
                support = lookup.get("support", {})
                neighbors = support.get("nearest_training_pfas", [])
                if neighbors:
                    st.dataframe(pd.DataFrame(neighbors).rename(columns={"label": "Training PFAS", "tanimoto_similarity": "Structural resemblance (Tanimoto)"}), hide_index=True, width="stretch")
                outside = support.get("out_of_range_descriptors", [])
                if outside:
                    st.warning("Molecular descriptors extend beyond training ranges. Treat the prediction with additional caution.")
                    st.dataframe(pd.DataFrame(outside), hide_index=True, width="stretch")
        for item in result.get("recommendations", [])[:1]:
            mixture = item.get("mixture_performance", {})
            if mixture.get("limiting_pfas_unseen"):
                st.warning(f"The top candidate’s limiting PFAS, {mixture.get('limiting_pfas')}, is unseen in training. Its exploratory estimate determines the mixture ranking.")


def _excluded_materials(result: dict[str, Any]) -> pd.DataFrame:
    rows = []
    for item in result.get("rejected_candidates", []):
        support = item.get("adsorbent_support", {})
        affected = item.get("pfas_rejections", [])
        rows.append({"Product": item.get("Product_key"), "PFAS requiring unsupported ML": ", ".join(str(row["pfas"]) for row in affected),
                     "Source records": support.get("source_records"), "DOI-identified studies": support.get("independent_studies"),
                     "Reason": " ".join(support.get("reasons", item.get("details", []))) or item.get("reason", "")})
    return pd.DataFrame(rows)


def _shortlist_table(result: dict[str, Any], items: list[dict[str, Any]]) -> pd.DataFrame:
    frame = _ranking_table(result, items)
    score = "Worst-case logKd" if result.get("mixture", {}).get("pfas_count", 1) > 1 else "logKd at reference dose"
    columns = ["Product", score]
    if "Limiting PFAS" in frame:
        columns.append("Limiting PFAS")
    columns += ["Basis", "Warnings"]
    compact = frame.reindex(columns=columns).copy()
    # Training studies behind each product, so thinly supported products are visible next to the score.
    compact.insert(2, "Studies", [(item.get("adsorbent_support") or {}).get("study_ids") for item in items])
    threshold = _resolution_threshold(result)
    top = float(items[0]["ranking_score"]) if items else None
    compact.insert(1, "Shortlist group", [
        "Top-performing tier" if threshold is not None and top - float(item["ranking_score"]) <= threshold
        else "Other" if threshold is not None else "Unassessed"
        for item in items
    ])
    return compact


def _render_water_support(result: dict[str, Any]) -> None:
    support = result.get("water_support", {})
    has_model_scores = any(item.get("performance_source") == "model_prediction" or any(
        row.get("performance_source") == "model_prediction" for row in item.get("pfas_results", []))
        for item in result.get("recommendations", []))
    if not support.get("limited") or not has_model_scores:
        return
    st.warning(support["warning"], icon=":material/warning:")
    with st.expander("Water-matrix support details"):
        st.caption(f"{support['model_rows']} model rows · {support['source_records']} distinct source records · {support['studies']} studies")
        st.caption(f"{support['studies_with_other_water']} of these studies also include another water type. This overlap does not establish otherwise matched experimental comparisons.")
        st.caption(f"Largest study: {support['largest_study']} supplies {support['largest_study_source_records']} of {support['source_records']} source records ({support['largest_study_source_share']:.0%}).")
        if "broad_water_category" in support.get("reasons", []):
            st.caption("The 'others' category groups different matrices and provides limited matrix specificity.")
        st.caption("This warning applies to model-derived scores. Comparable database values retain their own evidence caveats. Training coverage is not a calibrated confidence estimate.")


def _render_result(result: dict[str, Any], category_groups: list[dict[str, Any]], actions: Any = None) -> None:
    status = result["status"]
    if status == "pfas_not_ready":
        st.error(
            "At least one PFAS needs a confirmed structure or complete model features. "
            "Correct the listed entry; every PFAS must be resolved before mixture screening.",
            icon=":material/error:",
        )
        unresolved = result.get("unresolved_pfas", [])
        if unresolved:
            st.dataframe(pd.DataFrame([
                {
                    "PFAS input": row.get("input", {}).get("identity"),
                    "Identity type": row.get("input", {}).get("identity_type"),
                    "Lookup status": row.get("lookup", {}).get("status"),
                    "Action needed": " ".join(row.get("lookup", {}).get("warnings", [])),
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
                "Choose a reviewed PFAS or confirm a valid name/CAS/SMILES structure in the mixture editor.",
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
            st.warning("No candidate has an allowed score for every entered PFAS. Products without a training record are not ranked; review the exclusions below.", icon=":material/filter_alt:")
        rejected = result.get("rejected_candidates", [])
        if rejected:
            st.dataframe(_excluded_materials(result), width="stretch", hide_index=True)
        return

    lookups = result.get("pfas_lookups") or [result.get("pfas_lookup", {})]
    conditions = result.get("screening_conditions", {})
    all_items = result["recommendations"]
    dose = conditions.get("dose_mg_L")

    pfas_labels = [lookup.get("label") or lookup.get("cache_key") or lookup.get("canonical_smiles") for lookup in lookups]
    pfas_labels = [label for label in pfas_labels if label]
    result_label = " + ".join(pfas_labels) if len(pfas_labels) <= 3 else f"{len(pfas_labels)}-PFAS mixture"
    st.subheader(f"Results for {result_label}", anchor="screening-results")
    _chips(st.session_state.get("ar_scenario_chips", []))
    if actions is not None:
        actions()
    material_screening = result.get("adsorbent_screening", {})
    if material_screening:
        st.caption(f"Adsorbent support: {material_screening.get('ranked_products', 0)} products ranked; "
                   f"{material_screening.get('excluded_products', 0)} catalog products without a training record are not ranked.")
    filtered_items, selected_class = _filter_result_items(all_items, category_groups)
    # Every ranked product is shown, so the table matches the CSV and JSON downloads.
    items = filtered_items
    class_filtered = selected_class is not None
    view_result = {**result, "recommendations": items}
    _render_water_support(view_result)

    scope = result.get("screening_scope", {})
    # ``matched`` counts every screened candidate with a comparable record; this
    # card sits above the ranking, so it has to count the rows the ranking shows
    # or it reads as "5 of these 20" when the five are not among them.
    if result.get("mixture", {}).get("pfas_count", 1) > 1:
        shown_matched = sum(
            row.get("performance_source") == "database_match"
            for item in items for row in item.get("pfas_results", [])
        )
        shown_pairs = sum(len(item.get("pfas_results", [])) for item in items)
        database_summary = f"{shown_matched}/{shown_pairs} displayed PFAS–adsorbent pairs from database"
    else:
        shown_matched = sum(item.get("performance_source") == "database_match" for item in items)
        database_summary = f"{shown_matched}/{len(items)} displayed candidates from database"
    st.caption(f"{len(filtered_items)} candidates in view · {result.get('candidates_screened', '—')} screened · Reference dose {_dose_label(dose)} · {database_summary}")

    for warning in scope.get("warnings", []):
        st.warning(warning, icon=":material/filter_alt:")
    # Keep scientific limitations visible even when the detailed support tables
    # are collapsed below the shortlist.
    if any(lookup.get("support", {}).get("seen_in_training") is False for lookup in lookups):
        st.warning(UNSEEN_WARNING, icon=":material/warning:")
    if any(lookup.get("support", {}).get("out_of_range_descriptors") for lookup in lookups):
        st.warning("Molecular descriptors extend beyond training ranges. Treat the prediction with additional caution.")
    if conditions.get("dose_within_development_support") is False:
        st.warning("The reference dose is outside model development support; treat the ranking as indicative only.")

    tab_labels = ["Ranking", "Candidate details"]
    if result.get("rejected_candidates"):
        tab_labels.append("Excluded candidates")
    tabs = st.tabs(tab_labels)

    with tabs[0]:
        _render_scenario_caveats(result)
        if items:
            # Only caveats that change how far the top score can be trusted
            # earn a banner; notes on the evidence stay in Candidate details.
            top_caveats = _row_flags(_candidate_caveats(items[0]))
            if top_caveats:
                more = (
                    f" See Candidate details for {len(top_caveats) - 1} more."
                    if len(top_caveats) > 1 else ""
                )
                st.warning(
                    f"**Top-ranked candidate:** {top_caveats[0]['display']}{more}",
                    icon=":material/warning:",
                )
        if result.get("mixture", {}).get("pfas_count", 1) > 1 and items:
            top = items[0]
            mixture = top["mixture_performance"]
            if mixture.get("limiting_score_exploratory_adsorbent"):
                st.caption("The limiting mixture score is an exploratory adsorbent prediction; validate the shortlist experimentally.")
            st.html(
                "<div class='ar-headline'>"
                f"<b>{escape(_product_name(top['product']))}</b> has the highest worst-case screening score"
                + (f" in {escape(selected_class)}" if selected_class else "") + ": "
                f"log K<sub>d</sub> {mixture['worst_case_logKd']:.3f}, limited by "
                f"<b>{escape(str(mixture['limiting_pfas']))}</b>. Consider similarly scored products together.</div>"
            )
        else:
            top_database = [item for item in items if item.get("performance_source") == "database_match"][:2]
            if len(top_database) == 2:
                first, second = top_database
                threshold = _resolution_threshold(result)
                gap = first["performance"]["mean_logKd"] - second["performance"]["mean_logKd"]
                verdict = (
                    "The difference is smaller than the typical error, so the evidence does not clearly "
                    "separate them."
                    if threshold is not None and gap <= threshold
                    else "The database evidence therefore favours "
                    f"<b>{escape(_product_name(first['product']))}</b>."
                )
                st.html(
                    "<div class='ar-headline'>At the 25 mg/L reference dose under matched conditions, "
                    f"<b>{escape(_product_name(first['product']))}</b> has log K<sub>d</sub> "
                    f"{first['performance']['mean_logKd']:.3f}, compared with "
                    f"<b>{escape(_product_name(second['product']))}</b> at "
                    f"{second['performance']['mean_logKd']:.3f}. {verdict}</div>"
                )
        _render_rank_resolution(result, items)
        compact = _shortlist_table(result, items)
        config = _ranking_column_config(items)
        config["Shortlist group"] = st.column_config.TextColumn("Group", width="medium", help="Top-performing tier means within the cross-validated error of a predicted difference (about 0.5 log units) of the top score in this view. It is a screening heuristic, not a confidence interval or proof of equivalence.")
        config["Product"] = st.column_config.TextColumn("Product", width="medium", pinned=True)
        config["Basis"] = st.column_config.TextColumn("Basis", width=150)
        config["Studies"] = st.column_config.NumberColumn("Studies", width="small", help="Studies with training records for this product. Products with at least two DOI-identified studies and five source records are moderately supported; those below either threshold are thinly supported and carry a Thinly supported label. Candidate details give the reason.")
        config["logKd at reference dose"] = st.column_config.NumberColumn("log Kd", format="%.2f", width="small", help="Higher indicates stronger screening affinity. Display precision is not predictive accuracy.")
        config["Worst-case logKd"] = st.column_config.NumberColumn("Min log Kd", format="%.2f", width=110, help="The lowest score across the entered PFAS; this determines mixture ranking.")
        selection_id = hashlib.sha256(json.dumps([st.session_state.get("ar_last_run_fingerprint"), [(item['product']['Product_key'], item['ranking_score']) for item in items]], default=str).encode()).hexdigest()[:12]
        st.caption("Select a row to open its evidence below. Scores order candidates within each group; small differences do not establish superiority.")
        selected = st.dataframe(compact, width="stretch", hide_index=True,
            column_config=config, on_select="rerun", selection_mode="single-row", key=f"ar_shortlist_{selection_id}")
        selected_rows = selected.selection.rows
        _render_warning_legend(items)
        if selected_rows and 0 <= selected_rows[0] < len(items):
            _render_candidate_detail({**result, "recommendations": [items[selected_rows[0]]]}, expanded=True, key_prefix="selected_")
        with st.expander("Complete ranking table"):
            _basis_legend()
            st.dataframe(_ranking_styler(_ranking_table(result, items, class_filtered)), width="stretch", hide_index=True,
                column_config=_ranking_column_config(items))
            st.caption("This table includes technical columns. Export results for the complete adsorbent–PFAS record.")

    with tabs[1]:
        _render_candidate_detail(view_result)

    if result.get("rejected_candidates"):
        with tabs[2]:
            st.dataframe(_excluded_materials(result), width="stretch", hide_index=True)

    with st.expander("Model support and scoring details"):
        _render_pfas_support(result)
        _render_basis(result)
    st.caption("Table toolbar downloads export that table only. Export results includes every adsorbent–PFAS pair, evidence source, and caveat; download inputs to reload settings.")
    with st.expander("Advanced: complete recommendation JSON"):
        st.download_button("Download complete recommendation (JSON)",
            recommendation_bytes(result), "recommendation.json", "application/json",
            icon=":material/download:")


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
    temperature: float | None, water_type_used: str | None = None,
    default_temperature: float = DEFAULT_TEMPERATURE_C,
) -> list[tuple[str, Any]]:
    """Summarize the inputs a result was produced from.

    Results persist across reruns, so the panel has to carry the scenario it
    belongs to; otherwise a table left on screen after the form was edited
    looks like it describes the values currently in the fields.
    """
    mixture_label = "; ".join(
        f"{entry['pfas']['identity']} ({entry.get('concentration_input', {}).get('value', entry['initial_concentration_mg_l']):g} "
        f"{entry.get('concentration_input', {}).get('unit', 'mg/L')})"
        for entry in pfas_entries
    )
    return [
        ("PFAS mixture", mixture_label),
        ("Water", water_type_used or water.get("water_type") or "ultrapure water (default)"),
        ("pH", water.get("pH")),
        ("T", f"{default_temperature:g} °C (default)" if temperature is None else f"{temperature:g} °C"),
        ("Adsorbents", "all eligible products"),
    ]


def _change_concentration_unit(value_key: str, unit_key: str, previous_key: str) -> None:
    old_unit = st.session_state.get(previous_key, "mg/L")
    new_unit = st.session_state[unit_key]
    value = st.session_state.get(value_key)
    if value is not None:
        try:
            st.session_state[value_key] = from_mg_l(to_mg_l(value, old_unit, allow_zero=True), new_unit)
        except MixtureInputError:
            # Preserve invalid values for the existing validation message.
            st.session_state[unit_key] = old_unit
            return
    st.session_state[previous_key] = new_unit


def _unit_selector(unit_key: str, value_key: str, default: str = "mg/L", *, width: int | str = "stretch") -> str:
    previous_key = unit_key + "_previous"
    st.session_state.setdefault(unit_key, normalize_unit(default))
    st.session_state.setdefault(previous_key, st.session_state[unit_key])
    return st.selectbox("Unit", CONCENTRATION_UNITS, key=unit_key,
        on_change=_change_concentration_unit, args=(value_key, unit_key, previous_key), width=width)


def _commit_csv_import() -> None:
    try:
        prepared = st.session_state.get("ar_csv_prepared", [])
        entries, next_id = append_prepared_import(
            list(st.session_state.get(PFAS_ENTRIES_KEY, [])), prepared,
            int(st.session_state.get(PFAS_NEXT_ID_KEY, 1)))
        st.session_state[PFAS_ENTRIES_KEY] = entries
        st.session_state[PFAS_NEXT_ID_KEY] = next_id
        st.session_state["ar_csv_success"] = f"Imported {len(prepared)} PFAS. Concentrations were converted to mg/L for screening."
        st.session_state["ar_csv_imported_digest"] = st.session_state.get("ar_csv_digest")
        st.session_state.pop("ar_csv_error", None)
    except MixtureInputError as exc:
        st.session_state["ar_csv_error"] = str(exc)


def _csv_mixture_import(bundle: Any) -> None:
    with st.container():
        st.caption("Column order in the example: pfas, identity_type, concentration, unit, parent_smiles. Required: pfas, concentration, unit. Optional: identity_type, parent_smiles, label.")
        st.caption("identity_type describes pfas: name, abbreviation, cas, or smiles. For smiles, put the SMILES string directly in pfas. parent_smiles supplies an optional structure alongside a name, abbreviation, or CAS number. Existing files with a smiles column are also accepted.")
        st.download_button("Download example CSV", EXAMPLE_CSV_PATH.read_bytes(),
            file_name="pfas_mixture_example.csv", mime="text/csv", key="ar_csv_example_download",
            icon=":material/download:")
        st.caption("The example includes abbreviations, SMILES, and a CAS number, with illustrative ng/L, ug/L, and mg/L concentrations. Unknown names/CAS need parent_smiles or resolution in the manual editor.")
        uploaded = st.file_uploader("PFAS mixture CSV", type=["csv"], max_upload_size=1,
            key=f"ar_mixture_csv_{st.session_state.get('ar_csv_upload_epoch', 0)}",
            help="CSV UTF-8; at most 100 PFAS rows. ND, zero, and < or > values need explicit quantified concentrations.")
        if uploaded is None:
            st.session_state.pop("ar_csv_prepared", None)
            st.session_state.pop("ar_csv_imported_digest", None)
            st.session_state.pop("ar_csv_success", None)
            return
        payload = uploaded.getvalue()
        digest = hashlib.sha256(payload).hexdigest()
        if digest != st.session_state.get("ar_csv_digest"):
            st.session_state.pop("ar_csv_success", None)
            st.session_state.pop("ar_csv_error", None)
        st.session_state["ar_csv_digest"] = digest
        if st.session_state.get("ar_csv_imported_digest") == digest:
            st.success(st.session_state.get("ar_csv_success", "This file was already imported. Edit the mixture below or choose another file."))
            return
        try:
            rows = read_mixture_csv(payload)
            prepared = prepare_csv_import(rows,
                lambda pfas: resolve_pfas(bundle, {"pfas": pfas})[0],
                list(st.session_state.get(PFAS_ENTRIES_KEY, [])))
        except (MixtureInputError, ValueError) as exc:
            st.session_state.pop("ar_csv_prepared", None)
            st.error(str(exc))
            return
        st.session_state["ar_csv_prepared"] = prepared
        preview = []
        for row in prepared:
            entry = row.get("entry") or {}
            preview.append({"CSV line": row["csv_row"], "Submitted PFAS": row["pfas"],
                "Identity type": row["identity_type"],
                "Resolved PFAS": entry.get("label", ""), "C₀ (submitted)": row["concentration"],
                "Unit": entry.get("concentration_unit", row["unit"]), "C₀ (mg/L)": entry.get("initial_concentration_mg_l"),
                "Status": row["error"] or ("Review structure" if row.get("requires_confirmation") else "Ready"),
                "Structure used": row.get("structure", "")})
        st.dataframe(pd.DataFrame(preview), hide_index=True, width="stretch")
        errors = [row for row in prepared if row["error"]]
        if errors:
            st.warning(f"Correct {len(errors)} row(s) and upload the file again. The existing mixture is retained; no rows from this file have been added.")
        submitted_structures = [row for row in prepared if row.get("requires_confirmation") and not row["error"]]
        if submitted_structures:
            st.info("Check the structures shown in the preview before confirming this import. Supplied SMILES determine molecular identity.")
            with st.expander("Inspect supplied molecular structures"):
                for row in submitted_structures:
                    st.caption(f"CSV line {row['csv_row']} · {row['entry']['label']}")
                    _show_structure(row["structure"])
                    st.code(row["structure"], language=None)
        st.button((f"Confirm structures and import {len(prepared)} PFAS" if submitted_structures
                   else f"Import {len(prepared)} PFAS"), key="ar_import_csv", width="stretch",
            disabled=bool(errors), on_click=_commit_csv_import, icon=":material/upload:")
        if st.session_state.get("ar_csv_error"):
            st.error(st.session_state["ar_csv_error"])


def _pfas_mixture_editor(bundle: Any) -> list[dict[str, Any]]:
    """Collect PFAS through a searchable cache-backed add-and-edit surface."""
    _section(
        1,
        "PFAS mixture",
        "Search the reviewed cache or enter a name, CAS number, abbreviation, or SMILES.",
    )
    choices, aliases = _pfas_choices(bundle)
    # The search gets the full card width to itself: at the themed type size a
    # three-way split leaves the button column too narrow for its own label.
    selected = st.selectbox(
        "Search or enter PFAS *",
        choices,
        index=None,
        format_func=_pfas_choice_label,
        placeholder="e.g., PFOA, perfluorooctanoic acid, or 335-67-1",
        help=(
            "Search by abbreviation, full name, or CAS number. Each result identifies one reviewed compound. "
            "For an unlisted compound, choose the Add option and review its structure before adding it to the mixture."
        ),
        accept_new_options=True,
        key=PFAS_SEARCH_KEY,
        on_change=_resolve_typed_pfas,
        args=(aliases,),
    )
    if isinstance(selected, (tuple, list)) and len(selected) == 7:
        # Confirms which compound an abbreviation or a typed alias stands for.
        details = [value for value in (selected[4], selected[5]) if value and value != selected[2]]
        if details:
            st.caption(f"{selected[2]} · {' · '.join(details)}")
    custom = isinstance(selected, str) and bool(selected.strip())
    if custom:
        _custom_structure_editor(bundle, selected)
    with st.container(horizontal=True, vertical_alignment="bottom"):
        concentration = st.number_input(
            "Initial concentration (C₀) *", min_value=0.0, value=None, format="%.6g",
            key=PFAS_ADD_C0_KEY, width=240, placeholder="Enter a value",
        )
        _unit_selector(PFAS_ADD_UNIT_KEY, PFAS_ADD_C0_KEY, width=130)
    valid_concentration = concentration is not None and math.isfinite(concentration) and concentration > 0
    st.button(
        "Confirm structure and add PFAS" if custom else "Add PFAS", type="secondary",
        icon=":material/add:", width="content",
        disabled=not valid_concentration or selected in (None, "") or (custom and not st.session_state.get("ar_pending_structure")),
        on_click=_add_selected_pfas,
        help="Choose a PFAS and enter its positive initial concentration before adding it.",
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
        with st.container(key=f"ar-pfas-entry-{entry_id}", gap="xsmall"):
            title = entry.get("abbreviation") or entry.get("label") or entry.get("identity")
            st.markdown(f"**{escape(str(title))}**" + (f" · {escape(entry['cas'])}" if entry.get("cas") else ""))
            if entry.get("full_name"):
                st.caption(entry["full_name"])
            st.caption(
                "Needs identity review: remove this entry and resolve it above."
                if entry.get("identity_ready") is False
                else "Reviewed cache entry"
                if entry.get("cached_choice")
                else f"Confirmed structure · {entry['identity_type']}"
            )
            unit_key = f"ar_pfas_unit_{entry_id}"
            unit = st.session_state.get(unit_key, entry.get("concentration_unit", "mg/L"))
            value_key = f"ar_pfas_c0_{entry_id}"
            st.session_state.setdefault(value_key,
                from_mg_l(entry["initial_concentration_mg_l"], unit)
                if entry.get("initial_concentration_mg_l") is not None else None)
            with st.container(horizontal=True, vertical_alignment="bottom"):
                concentration = st.number_input(
                    "Initial concentration (C₀)", min_value=0.0,
                    value=None, format="%.6g", key=value_key, width=240,
                )
                unit = _unit_selector(unit_key, value_key, entry.get("concentration_unit", "mg/L"), width=130)
            with st.container(horizontal=True):
                st.button(
                    "Remove", key=f"ar_remove_pfas_{entry_id}",
                    icon=":material/delete:", width="content",
                    on_click=_remove_pfas, args=(entry_id,),
                )
                with st.popover("Identity details", icon=":material/science:"):
                    _identity_field("Entered identifier", entry.get("identity"))
                    _identity_field("Identity type", entry.get("identity_type"))
                    _identity_field("Full name", entry.get("full_name"))
                    _identity_field("CAS number", entry.get("cas"))
                    if entry.get("confirmed_smiles"):
                        st.code(entry["confirmed_smiles"], language=None)
        try:
            concentration_mg_l = to_mg_l(concentration, unit, allow_zero=True)
        except MixtureInputError:
            concentration_mg_l = None
        updated.append({**entry, "initial_concentration_mg_l": concentration_mg_l,
            "concentration_value": concentration, "concentration_unit": unit})
    st.session_state[PFAS_ENTRIES_KEY] = updated
    return updated


def _parse_pfas_entries(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], str | None]:
    """Validate PFAS rows and convert them to service scenarios."""
    entries: list[dict[str, Any]] = []
    for row_number, row in enumerate(rows, start=1):
        identity = str(row.get("identity", "")).strip()
        identity_type = str(row.get("identity_type", "")).strip()
        try:
            concentration = float(row.get("initial_concentration_mg_l", 0))
        except (ValueError, TypeError):
            return [], f"Enter a quantified concentration greater than zero in mixture row {row_number}."
        if not identity or identity_type not in IDENTITY_TYPES:
            return [], f"PFAS mixture row {row_number} has an invalid identifier."
        if row.get("identity_ready") is False:
            return [], f"Resolve the identity in mixture row {row_number} before screening."
        if not math.isfinite(concentration) or concentration <= 0:
            return [], f"Enter a quantified concentration greater than zero in mixture row {row_number}."
        entries.append({
            "pfas": {"identity": identity, "identity_type": identity_type, "label": row.get("label"),
                     **{key: row[key] for key in ("confirmed_smiles", "structure_confirmed", "resolution") if key in row}},
            "initial_concentration_mg_l": concentration,
            "concentration_input": {"value": row.get("concentration_value", concentration),
                                    "unit": row.get("concentration_unit", "mg/L")},
        })
    if not entries:
        return [], "Add at least one PFAS identity and initial concentration before screening."
    return entries, None


def _scenario_fingerprint(
    pfas_rows: list[dict[str, Any]], water: dict[str, Any], temperature: float | None,
    include_exploratory: bool = False,
) -> str:
    """Serialize the editable scenario so stale results can be labelled clearly."""
    payload = {
        "screening_behavior_version": SCREENING_BEHAVIOR_VERSION,
        "pfas": [
            {
                "identity": row.get("identity"),
                "identity_type": row.get("identity_type"),
                "initial_concentration_mg_l": row.get("initial_concentration_mg_l"),
                "confirmed_smiles": row.get("confirmed_smiles"),
                "label": row.get("label"),
                "descriptor_version": DESCRIPTOR_VERSION,
            }
            for row in pfas_rows
        ],
        "water": water,
        "temperature_c": temperature,
        "include_exploratory_adsorbents": include_exploratory,
        "adsorbent_support_version": SUPPORT_VERSION,
        "water_support_version": WATER_SUPPORT_VERSION,
        "additive_input_version": ADDITIVE_INPUT_VERSION,
    }
    return json.dumps(payload, sort_keys=True, default=str)


@st.cache_data(show_spinner=False, max_entries=8)
def _scenario_bundle_reference(directory: str, manifest_mtime_ns: int, behavior_version: int = SCREENING_BEHAVIOR_VERSION) -> dict:
    return bundle_reference(Path(directory))


def _prepare_scenario_restore(snapshot: dict, bundle: Any) -> dict[str, Any]:
    """Resolve and validate every value without changing editor/session state."""
    inputs = validate_inputs(snapshot["inputs"])
    water = inputs["water"]
    if water["water_type"] not in WATER_TYPE_OPTIONS:
        raise ScenarioError("The saved water type is unsupported by this editor. Use a supported water type in the scenario file.")
    entries, warnings, keys = [], list(snapshot.get("_notices", [])), set()
    current_reference = _scenario_bundle_reference(str(bundle.directory), (bundle.directory / BUNDLE_FILENAME).stat().st_mtime_ns)
    if not snapshot["bundle_reference"].get("id") or snapshot["bundle_reference"]["id"] != current_reference["id"]:
        warnings.append("This saved run came from a different or unverified model/data bundle. Loaded inputs will be screened with the current bundle.")
    for index, saved in enumerate(inputs["pfas"], 1):
        entry = {**saved, "id": index}
        pfas = {key: saved[key] for key in ("identity", "identity_type", "confirmed_smiles", "structure_confirmed", "resolution") if key in saved}
        if saved.get("canonical_smiles") and not pfas.get("confirmed_smiles"):
            pfas.update({"confirmed_smiles": saved["canonical_smiles"], "structure_confirmed": True})
            entry.update({"confirmed_smiles": saved["canonical_smiles"], "structure_confirmed": True})
        lookup, _ = resolve_pfas(bundle, {"pfas": pfas})
        entry.update({"identity_ready": lookup.ready, "cached_choice": lookup.ready and lookup.status == "cached"})
        for key in ("canonical_smiles", "structure_key", "abbreviation", "full_name", "cas"):
            entry.pop(key, None)
        if lookup.ready:
            _, key = prepare_structure(lookup.canonical_smiles)
            if key in keys:
                raise ScenarioError(f"Saved PFAS row {index} duplicates another compound. Keep one concentration per PFAS.")
            keys.add(key)
            feature = lookup.feature_row or {}
            entry.update({"canonical_smiles": lookup.canonical_smiles, "structure_key": key,
                          "label": lookup.cache_key or saved.get("label") or saved["identity"],
                          "abbreviation": _clean_display_value(feature.get("Abbreviation")),
                          "full_name": _clean_display_value(feature.get("PFAS_name") or feature.get("Compound Full Name")),
                          "cas": _clean_display_value(feature.get("CAS Number") or feature.get("CAS"))})
        else:
            warnings.append(f"{saved['identity']}: {' '.join(lookup.warnings)} Remove this entry and resolve it in the editor before screening.")
        entries.append(entry)
    updates = {PFAS_ENTRIES_KEY: entries, PFAS_NEXT_ID_KEY: len(entries) + 1,
               PFAS_SEARCH_KEY: None, "ar_custom_identity_type": "name"}
    for entry in entries:
        updates[f"ar_pfas_c0_{entry['id']}"] = entry["concentration_value"]
        updates[f"ar_pfas_unit_{entry['id']}"] = entry["concentration_unit"]
        updates[f"ar_pfas_unit_{entry['id']}_previous"] = entry["concentration_unit"]
    water_fields = {"ph": "pH", "organic_carbon": "organic_carbon_mg_l", "tds": "tds_mg_l",
                    "ionic_strength": "ionic_strength_mol_l"}
    updates["organic_present"] = water["organic_matter_present"]
    updates["ar_inorganic_entries"] = validate_additives(water.get("inorganic_additives", []))
    for widget, field in water_fields.items():
        value = water[field]
        updates[widget] = "" if value is None else str(value)
    updates["water_type"] = water["water_type"]
    updates["ar_include_exploratory_adsorbents"] = inputs.get("include_exploratory_adsorbents", False)
    updates["temperature"] = "" if inputs["temperature_c"] is None else str(inputs["temperature_c"])
    for ion in ION_LABELS:
        updates[f"ion_{ion}"] = water["ions"].get(ion, False)
    updates["ar_scenario_name"] = snapshot["name"]
    updates["ar_scenario_notices"] = warnings
    updates["ar_scenario_message"] = f"Loaded {snapshot['name']}. Run screening to refresh results with the current model and data."
    return updates


def _commit_scenario_restore(updates: dict[str, Any]) -> None:
    """Commit only prepared values; every fallible operation precedes this step."""
    for key in list(st.session_state):
        if key.startswith(("ar_pfas_c0_", "ar_pfas_unit_", "ar_csv_", "ar_public_", "ar_custom_label_")) and key != "ar_csv_upload_epoch":
            del st.session_state[key]
    st.session_state["ar_csv_upload_epoch"] = st.session_state.get("ar_csv_upload_epoch", 0) + 1
    for key in ("ar_result", "ar_last_run_fingerprint", "ar_scenario_chips", "ar_error", "ar_pending_structure", "ar_pfas_add_error", "ar_salt_error", "ar_scenario_error"):
        st.session_state.pop(key, None)
    st.session_state.update(updates)


def _restore_scenario(snapshot: dict, bundle: Any) -> None:
    _commit_scenario_restore(_prepare_scenario_restore(snapshot, bundle))


def _load_saved_scenario(bundle: Any) -> None:
    selected = st.session_state.get("ar_scenario_selected")
    snapshot = next((item for item in st.session_state.get("ar_saved_scenarios", []) if item["scenario_id"] == selected), None)
    if snapshot:
        try:
            _restore_scenario(snapshot, bundle)
        except (ScenarioError, ValueError, RuntimeError) as exc:
            st.session_state["ar_scenario_error"] = str(exc)


def _import_saved_scenario(bundle: Any) -> None:
    uploaded = st.session_state.get("ar_scenario_upload_payload")
    if uploaded is None:
        return
    try:
        snapshot = read_snapshot(uploaded["data"], Path(uploaded["name"]).stem or "Imported scenario")
        library, snapshot = add_snapshot(st.session_state.get("ar_saved_scenarios", []), snapshot)
        if snapshot.get("result"):
            results_csv(snapshot["result"], snapshot["name"])
            recommendation_bytes(snapshot["result"])
        updates = _prepare_scenario_restore(snapshot, bundle)
        _commit_scenario_restore(updates)
        st.session_state["ar_saved_scenarios"] = library
        st.session_state["ar_scenario_selected"] = snapshot["scenario_id"]
    except (ScenarioError, ValueError, RuntimeError) as exc:
        st.session_state["ar_scenario_error"] = str(exc)


def _load_scenario_csv(inputs: dict, name: str, bundle: Any) -> None:
    try:
        reference = _scenario_bundle_reference(str(bundle.directory), (bundle.directory / BUNDLE_FILENAME).stat().st_mtime_ns)
        snapshot = make_snapshot(name, inputs, reference)
        snapshot["_imported"] = True
        library, snapshot = add_snapshot(st.session_state.get("ar_saved_scenarios", []), snapshot)
        _restore_scenario(snapshot, bundle)
        st.session_state["ar_saved_scenarios"] = library
        st.session_state["ar_scenario_selected"] = snapshot["scenario_id"]
    except (ScenarioError, ValueError, RuntimeError) as exc:
        st.session_state["ar_scenario_error"] = str(exc)


def _scenario_csv_upload(bundle: Any) -> None:
    st.download_button("Download example inputs CSV", SCENARIO_EXAMPLE_PATH.read_bytes(),
        file_name="scenario_inputs_example.csv", mime="text/csv", key="ar_scenario_example_download", icon=":material/download:")
    st.caption("Use one row per PFAS or inorganic additive. row_type identifies the entry; species, concentration, and unit occupy separate columns. Repeat shared water settings on every row.")
    uploaded = st.file_uploader("Reload scenario inputs CSV", type=["csv"], max_upload_size=1, key="ar_scenario_inputs_upload",
        help="Use row_type pfas or inorganic_additive. Shared water settings repeat on every row. A results CSV cannot reload inputs.")
    if uploaded is None:
        return
    try:
        name, inputs = read_scenario_csv(uploaded.getvalue(), Path(getattr(uploaded, "name", "Imported scenario.csv")).stem[:80])
    except (ScenarioError, MixtureInputError, ValueError) as exc:
        st.error(str(exc))
        return
    st.caption(f"{name}: review the PFAS identities, structures, and shared water settings before loading.")
    st.dataframe(pd.DataFrame([{"PFAS": row["identity"], "Identity type": row["identity_type"],
        "Concentration": row["concentration_value"], "Unit": row["concentration_unit"],
        "Parent SMILES": row.get("confirmed_smiles", "")} for row in inputs["pfas"]]), hide_index=True, width="stretch")
    if inputs["water"]["inorganic_additives"]:
        st.dataframe(pd.DataFrame([{"Inorganic species": row["species"], "Concentration": row["concentration"], "Unit": row["unit"]}
            for row in inputs["water"]["inorganic_additives"]]), hide_index=True, width="stretch")
    st.caption("Shared water settings: " + "; ".join(f"{key}: {value}" for key, value in inputs["water"].items() if key not in {"ions", "inorganic_additives"} and value not in (None, ""))
               + f"; temperature: {str(inputs['temperature_c']) + ' °C' if inputs['temperature_c'] is not None else '25 °C (default)'}")
    st.caption("Ions marked present: " + (", ".join(ION_LABELS[key] for key, value in inputs["water"]["ions"].items() if value) or "None specified"))
    structures = [row for row in inputs["pfas"] if row.get("confirmed_smiles")]
    if structures:
        with st.expander("Inspect a structure from the scenario CSV"):
            index = st.selectbox("PFAS structure to inspect", range(len(structures)), index=None,
                format_func=lambda index: structures[index].get("label") or structures[index]["identity"], key="ar_scenario_inspect_structure")
            if index is not None:
                canonical, _ = prepare_structure(structures[index]["confirmed_smiles"])
                _show_structure(canonical, width=360)
                st.code(canonical, language=None)
    digest = hashlib.sha256(uploaded.getvalue()).hexdigest()[:16]
    reviewed = st.checkbox("I have checked the PFAS identities and supplied structures", key=f"ar_scenario_csv_reviewed_{digest}")
    st.button("Load scenario inputs", key="ar_load_scenario_csv", disabled=not reviewed,
              on_click=_load_scenario_csv, args=(inputs, name, bundle), icon=":material/restore:")


def _remove_saved_scenario() -> None:
    selected = st.session_state.get("ar_scenario_selected")
    st.session_state["ar_saved_scenarios"] = [item for item in st.session_state.get("ar_saved_scenarios", []) if item["scenario_id"] != selected]
    st.session_state["ar_scenario_selected"] = None
    st.session_state["ar_scenario_compare"] = [key for key in st.session_state.get("ar_scenario_compare", []) if key != selected]
    st.session_state.pop("ar_scenario_reference", None)


def _comparison_input_table(snapshots: list[dict]) -> pd.DataFrame:
    columns = {}
    for snapshot in snapshots:
        inputs = snapshot["inputs"]
        water = inputs["water"]
        result = snapshot["result"]
        best = result["recommendations"][0]
        columns[snapshot["name"]] = {
            "PFAS and initial concentrations": "; ".join(f"{row.get('label') or row['identity']}: {row['concentration_value']:g} {row['concentration_unit']}" for row in inputs["pfas"]),
            "Water type": water["water_type"] or "Ultrapure water (default)",
            "Exploratory adsorbents included": "Yes" if inputs.get("include_exploratory_adsorbents", False) else "No",
            "pH": water["pH"], "Temperature (°C)": inputs["temperature_c"] if inputs["temperature_c"] is not None else
                f"{effective_temperature({}, result.get('screening_conditions', {}).get('fixed_conditions', {})):g} (default)",
            "Organic carbon (mg/L)": water["organic_carbon_mg_l"], "Organic additives present": water["organic_matter_present"],
            "TDS (mg/L)": water["tds_mg_l"], "Ionic strength (mol/L)": water["ionic_strength_mol_l"],
            "Inorganic additives": "; ".join(f"{row['species']}: {row['concentration']:g} {row['unit']}" for row in water.get("inorganic_additives", [])),
            "Ions present": ", ".join(ION_LABELS[key] for key, value in water["ions"].items() if value) or "Not specified",
            "Highest-ranked product": _product_name(best["product"]), "Best limiting logKd": f"{best['ranking_score']:.3f}",
            "Model/data bundle": snapshot["bundle_reference"].get("name", "Unknown"),
            "Bundle fingerprint": (snapshot["bundle_reference"].get("id") or "Unavailable")[:12],
            "Saved at (UTC)": snapshot["saved_at"], "Result origin": "Imported snapshot" if snapshot.get("_imported") else "This session",
        }
    return pd.DataFrame(columns).fillna("Not specified").astype(str).rename_axis("Input / result").reset_index()


def _scenario_comparison(library: list[dict]) -> None:
    completed = {}
    for item in library:
        if item.get("result"):
            try:
                validate_result_structure(item["result"])
                completed[item["scenario_id"]] = item
            except ScenarioError as exc:
                st.error(f"Saved results for {item['name']} cannot be compared: {exc} Remove the scenario or load its inputs and screen again.")
    selected = st.multiselect("Scenarios to compare", list(completed), max_selections=4,
        format_func=lambda key: completed[key]["name"], key="ar_scenario_compare",
        help="Choose two to four saved screening runs. Scenarios saved without results become comparable after reloading, screening, and saving again.")
    if len(selected) < 2:
        st.caption("Save at least two completed runs, then select them here. Saved inputs without results are available under Save and reload.")
        return
    reference = st.selectbox("Reference scenario", selected, format_func=lambda key: completed[key]["name"], key="ar_scenario_reference")
    snapshots = [completed[key] for key in selected]
    rows, warnings = compare_snapshots(snapshots, reference)
    st.dataframe(_comparison_input_table(snapshots), hide_index=True, width="stretch")
    st.caption("Rankings align products by their catalog identity. logKd is the lowest score across that scenario's PFAS; larger values rank higher. ΔlogKd is relative to the selected reference, and is shown only for matching bundles and PFAS identities. Evidence sources may change between conditions. Small differences do not establish a reliable improvement.")
    if any(len(item["inputs"]["pfas"]) > 1 for item in snapshots):
        st.caption("Mixture scores screen each PFAS separately; competitive adsorption is not modeled.")
    for warning in warnings:
        st.warning(warning)
    frame = pd.DataFrame(rows)
    config = {key: st.column_config.NumberColumn(format="%.3f") for key in frame.columns if key.endswith((" · logKd", " · ΔlogKd"))}
    st.dataframe(frame, hide_index=True, width="stretch", column_config=config)
    st.download_button("Download comparison CSV", comparison_csv(rows), file_name="scenario_comparison.csv",
                       mime="text/csv", key="ar_comparison_download", icon=":material/download:")
    st.caption("Comparison CSV is a table for Excel. Scenario inputs CSV restores the settings for a fresh run. Advanced JSON snapshots preserve exact historical results and full provenance.")
    with st.expander("PFAS support and caveats in the saved runs"):
        for snapshot in snapshots:
            st.markdown(f"**{escape(snapshot['name'])}**")
            for notice in snapshot.get("_notices", []):
                st.info(notice)
            result = snapshot["result"]
            for lookup in result.get("pfas_lookups", []):
                support = lookup.get("support", {})
                label = lookup.get("label") or lookup.get("cache_key") or "PFAS"
                if support.get("seen_in_training") is False:
                    st.warning(f"{label}: {UNSEEN_WARNING}")
                if support.get("out_of_range_descriptors"):
                    st.warning(f"{label}: molecular descriptors exceeded training ranges in this saved run.")
            for caveat in result.get("scenario_caveats", []):
                st.caption(caveat["text"])
            st.caption("Candidate-specific caveats appear beside each scenario's score in the comparison table and CSV. JSON downloads retain the full evidence and support diagnostics.")


def _current_snapshot(bundle: Any, rows: list[dict], water: dict, temperature: float | None, fingerprint: str) -> dict:
    saved_rows = []
    for row in rows:
        saved = dict(row)
        pfas = {key: row[key] for key in ("identity", "identity_type", "confirmed_smiles", "structure_confirmed") if key in row}
        lookup, _ = resolve_pfas(bundle, {"pfas": pfas})
        if lookup.ready:
            saved["canonical_smiles"] = lookup.canonical_smiles
        saved_rows.append(saved)
    inputs = {"pfas": saved_rows, "water": water, "temperature_c": temperature,
              "include_exploratory_adsorbents": st.session_state.get("ar_include_exploratory_adsorbents", False)}
    result = st.session_state.get("ar_result")
    include_result = result is not None and result.get("status") == "recommendations_ready" and st.session_state.get("ar_last_run_fingerprint") == fingerprint
    reference = _scenario_bundle_reference(str(bundle.directory), (bundle.directory / BUNDLE_FILENAME).stat().st_mtime_ns)
    name = (st.session_state.get("ar_scenario_name") or "").strip() or "Screening run"
    return make_snapshot(name, inputs, reference, result if include_result else None)


def _save_current_scenario(bundle: Any, rows: list[dict], water: dict, temperature: float | None, fingerprint: str) -> None:
    try:
        snapshot = _current_snapshot(bundle, rows, water, temperature, fingerprint)
        library, snapshot = add_snapshot(st.session_state.get("ar_saved_scenarios", []), snapshot)
        st.session_state["ar_saved_scenarios"] = library
        st.session_state["ar_scenario_selected"] = snapshot["scenario_id"]
        st.session_state["ar_scenario_message"] = f"Saved {snapshot['name']} with {'completed screening results' if snapshot.get('result') else 'inputs only'} in this session. Download inputs for later reuse."
        st.session_state["ar_scenario_notices"] = []
        st.session_state.pop("ar_scenario_error", None)
        st.toast(f"Saved {snapshot['name']} in this session.", icon=":material/save:")
    except (ScenarioError, ValueError) as exc:
        st.session_state["ar_scenario_error"] = str(exc)


def _scenario_actions(bundle: Any, rows: list[dict], water: dict, temperature: float | None, fingerprint: str,
                      input_errors: dict[str, str], result: dict | None = None) -> None:
    _, mixture_error = _parse_pfas_entries(rows)
    valid = not input_errors and not mixture_error
    snapshot = None
    if valid:
        try:
            snapshot = _current_snapshot(bundle, rows, water, temperature, fingerprint)
            payload = scenario_inputs_csv(snapshot)
        except (ScenarioError, ValueError) as exc:
            st.error(f"Inputs cannot be saved or downloaded: {exc}")
            valid = False
    with st.container(horizontal=True, vertical_alignment="bottom"):
        with st.popover("Save in this session", icon=":material/save:"):
            st.caption("Saved runs last only for this browser session. Download inputs to keep the settings for later.")
            st.text_input("Scenario name", key="ar_scenario_name", max_chars=80, placeholder="e.g., Groundwater at pH 7")
            st.button("Save run", key="ar_save_scenario", icon=":material/save:", disabled=not valid,
                on_click=_save_current_scenario, args=(bundle, rows, water, temperature, fingerprint))
        st.download_button("Download inputs for later", payload if valid else b"", file_name="screening_inputs.csv",
            mime="text/csv", key="ar_current_inputs_download", disabled=not valid, icon=":material/download:")
        if result is not None and result.get("status") == "recommendations_ready":
            current = st.session_state.get("ar_last_run_fingerprint") == fingerprint and not input_errors
            st.download_button("Export results" if current else "Export displayed results", results_csv(result),
                file_name="recommendation_results.csv", mime="text/csv", key="ar_result_csv_download", icon=":material/download:")


def _input_importer(bundle: Any) -> None:
    with st.expander("Import inputs", icon=":material/upload:"):
        complete, mixture = st.tabs(["Complete scenario", "PFAS only"])
        with complete:
            st.caption("Restore the mixture and shared water settings. Loading replaces the editor after validation and identity review.")
            _scenario_csv_upload(bundle)
        with mixture:
            st.caption("Append PFAS and concentrations to the mixture. Current water settings are retained.")
            _csv_mixture_import(bundle)


def _scenario_workspace(bundle: Any, rows: list[dict], water: dict, temperature: float | None, fingerprint: str, input_errors: dict[str, str] | None = None) -> None:
    st.session_state.setdefault("ar_saved_scenarios", [])
    with st.expander("Saved runs and comparison", icon=":material/folder_open:"):
        manage, compare = st.tabs(["Saved runs", "Compare scenarios"])
        with manage:
            st.caption("This library lasts only for the current browser session. Load a saved run's inputs to screen again, or download its inputs for later reuse. Use Import inputs above for CSV files.")
            library = st.session_state["ar_saved_scenarios"]
            by_id = {item["scenario_id"]: item for item in library}
            selected = st.selectbox("Saved scenarios", list(by_id), index=None, key="ar_scenario_selected",
                format_func=lambda key: by_id[key]["name"], placeholder="Select a saved scenario")
            if selected:
                snapshot = by_id[selected]
                st.caption(f"{len(snapshot['inputs']['pfas'])} PFAS · {'Screening results saved' if snapshot.get('result') else 'Inputs only'} · {snapshot['saved_at']}")
                with st.container(horizontal=True):
                    st.button("Load inputs", key="ar_load_scenario", on_click=_load_saved_scenario, args=(bundle,), icon=":material/restore:")
                    filename = re.sub(r"[^A-Za-z0-9_-]+", "_", snapshot["name"]).strip("_") or "scenario"
                    try:
                        payload = scenario_inputs_csv(snapshot)
                        st.download_button("Download inputs CSV", payload, file_name=f"{filename}_inputs.csv",
                                           mime="text/csv", key="ar_scenario_inputs_download", icon=":material/download:")
                    except ScenarioError as exc:
                        st.info(str(exc))
                    if snapshot.get("result"):
                        try:
                            st.download_button("Download saved results CSV", results_csv(snapshot["result"], snapshot["name"]),
                                file_name=f"{filename}_results.csv", mime="text/csv", key="ar_saved_results_download", icon=":material/download:")
                        except ScenarioError as exc:
                            st.error(f"Saved results cannot be exported: {exc} Remove this scenario or load its inputs and screen again.")
                    st.button("Remove saved scenario", key="ar_remove_scenario", on_click=_remove_saved_scenario, icon=":material/delete:")
            with st.expander("Advanced: full JSON snapshots"):
                st.caption("Use JSON to preserve or restore exact historical results, full database evidence, and identity-resolution provenance. CSV is the normal input/output format; JSON is optional.")
                if selected:
                    try:
                        st.download_button("Download full scenario snapshot (JSON)", snapshot_bytes(by_id[selected]),
                            file_name=f"{filename}.json", mime="application/json", key="ar_scenario_download", icon=":material/download:")
                    except ScenarioError as exc:
                        st.error(str(exc))
                uploaded = st.file_uploader("Reload a scenario JSON", type=["json"], max_upload_size=20, key="ar_scenario_upload",
                    help="Accepts saved scenario snapshots and complete recommendation JSON exports, up to 20 MB.")
                st.session_state["ar_scenario_upload_payload"] = {"data": uploaded.getvalue(), "name": getattr(uploaded, "name", "scenario.json")} if uploaded is not None else None
                st.button("Import and load scenario", key="ar_import_scenario", on_click=_import_saved_scenario, args=(bundle,),
                          disabled=uploaded is None, icon=":material/upload:")
            if st.session_state.get("ar_scenario_error"):
                st.error(st.session_state["ar_scenario_error"])
            elif st.session_state.get("ar_scenario_message"):
                st.success(st.session_state["ar_scenario_message"])
            for warning in st.session_state.get("ar_scenario_notices", []):
                st.warning(warning)
        with compare:
            _scenario_comparison(st.session_state["ar_saved_scenarios"])


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
        "Shortlist adsorbents for experimental testing using comparable database evidence and model estimates.",
    )
    _render_usage_guidance()

    bundle_dir = _resolve_bundle_dir()
    if bundle_dir is None:
        st.error(
            "No model bundle was found. Build one with build_final_logkd_model.py, "
            f"or set ADSORBENT_BUNDLE_DIR. Searched: {BUNDLE_SEARCH_ROOT}",
            icon=":material/error:",
        )
        return

    try:
        bundle = _load_bundle(str(bundle_dir), (bundle_dir / BUNDLE_FILENAME).stat().st_mtime_ns)
    # A bundle trained with a regressor that is not installed here raises
    # ImportError from deep inside joblib, which without this is an unhandled
    # traceback on a blank page rather than a statement of what is missing.
    except (BundleNotReadyError, FileNotFoundError, ValueError, KeyError, ImportError) as exc:
        st.error(f"Model bundle is not ready: {exc}", icon=":material/error:")
        return

    _input_importer(bundle)
    category_groups = catalog_category_groups(bundle)
    # PFAS identity and water chemistry are the two parameters users revisit.
    # Adsorbent classes are screened together and become a view-only result
    # filter, leaving both editable panels visible without a wizard or scrolling.
    identity_column, water_column = st.columns((5.4, 4.6), gap="medium")
    with identity_column, _card("pfas"):
        pfas_rows = _pfas_mixture_editor(bundle)
    with water_column:
        default_temperature = effective_temperature({}, fixed_test_conditions(bundle)["fixed_conditions"])
        water, temperature, input_errors = _water_form(default_temperature=default_temperature)

    st.html(f"<div class='ar-required' style='margin:1.1rem 0 .5rem'>{REQUIRED_NOTE}</div>")
    # Products absent from the training data are not ranked: the model is used for untested pairings of
    # PFAS and adsorbents that are each in the training data. The service keeps an opt-in for analyses.
    exploratory = False
    fingerprint = _scenario_fingerprint(pfas_rows, water, temperature, exploratory)
    pfas_entries, mixture_error = _parse_pfas_entries(pfas_rows)
    error_fingerprint = st.session_state.get("ar_error_input_fingerprint")
    legacy_input_error = str(st.session_state.get("ar_error", "")).startswith(("Add at least one PFAS", "C₀ must", "Enter a quantified concentration"))
    if (error_fingerprint is not None and error_fingerprint != fingerprint) or (legacy_input_error and not mixture_error):
        st.session_state.pop("ar_error", None)
        st.session_state.pop("ar_error_input_fingerprint", None)
    existing_result = st.session_state.get("ar_result")
    inputs_changed = (
        existing_result is not None
        and (bool(input_errors) or bool(mixture_error) or st.session_state.get("ar_last_run_fingerprint") != fingerprint)
    )
    if inputs_changed:
        st.info(
            "Inputs have changed since the displayed results were generated. Update the screening to compare this scenario.",
            icon=":material/edit_note:",
        )
    with st.container(key="ar-run"):
        if mixture_error:
            if pfas_rows:
                st.error(mixture_error)
            else:
                st.caption(mixture_error)
        elif input_errors:
            st.caption("Correct the highlighted water fields before screening.")
        else:
            st.caption(f"{len(pfas_entries)} PFAS ready · {len(bundle.catalog)} catalog adsorbents will be screened.")
        run = st.button(
            (
                "Update screening"
                if inputs_changed
                else "Screen mixture"
            ),
            type="primary",
            width="stretch",
            icon=":material/query_stats:",
            disabled=bool(input_errors) or bool(mixture_error), key="ar_screen",
        )
    if run:
        if input_errors:
            st.session_state["ar_error"] = "Correct the invalid water fields before screening."
        elif mixture_error:
            st.session_state["ar_error"] = mixture_error
        else:
            shared_scenario = {
                "temperature_c": temperature,
                "water": {key: value for key, value in water.items() if value not in (None, "") or key == "ions"},
                "adsorbent": {"categories": [], "include_exploratory": exploratory},
            }
            scenarios = [{**shared_scenario, **entry} for entry in pfas_entries]
            try:
                with st.spinner("Checking comparable database records, then modelling unmatched candidates…"):
                    result = recommend_mixture(bundle, scenarios, top_k=max(1, len(bundle.catalog)))
                st.session_state["ar_result"] = result
                st.session_state["ar_scenario_chips"] = _scenario_chips(pfas_entries, water, temperature,
                    result.get("water_chemistry", {}).get("normalized_water_type"), default_temperature)
                st.session_state["ar_last_run_fingerprint"] = fingerprint
                st.session_state.pop("ar_error", None)
                st.session_state.pop("ar_error_input_fingerprint", None)
            except InputValidationError as exc:
                st.session_state["ar_error"] = str(exc)
                st.session_state["ar_error_input_fingerprint"] = fingerprint

    if st.session_state.get("ar_result") is not None:
        st.markdown("[View results ↓](#screening-results)")

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
        _scenario_actions(bundle, pfas_rows, water, temperature, fingerprint, input_errors)
        _render_empty_state()
    else:
        valid_result = True
        if result.get("status") == "recommendations_ready":
            try:
                validate_result_structure(result)
            except ScenarioError as exc:
                st.error(f"The previous results cannot be displayed: {exc} Correct the inputs and run screening again.")
                valid_result = False
        if valid_result:
            if result.get("status") != "recommendations_ready":
                _scenario_actions(bundle, pfas_rows, water, temperature, fingerprint, input_errors)
            _render_result(result, category_groups,
                actions=lambda: _scenario_actions(bundle, pfas_rows, water, temperature, fingerprint, input_errors, result))
        else:
            _scenario_actions(bundle, pfas_rows, water, temperature, fingerprint, input_errors)
    _scenario_workspace(bundle, pfas_rows, water, temperature, fingerprint, input_errors)
    # Kept outside the collapsed reading guide so it is visible wherever
    # product names are, not only to users who open the guide.
    st.divider()
    st.caption(PRODUCT_NAME_DISCLAIMER)


if __name__ == "__main__":
    main()
