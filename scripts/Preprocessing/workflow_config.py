"""Central configuration for the independent document preprocessing workflow.

Edit :data:`ACTIVE_STUDIES` in this file to choose the studies for conversion,
JSON-first preprocessing, and chunking.  This module intentionally does not
import ``Data_Extraction.workflow_config``: the two directories are separate
workflows that may run different selected batches.
"""

from __future__ import annotations

import sys
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Iterable, Literal


# Repository paths are derived from this file:
# <Box>/My Box Notes/2. Adsorbent Recommender/AD/scripts/Preprocessing
PREPROCESSING_DIR: Final[Path] = Path(__file__).resolve().parent
HELPERS_DIR: Final[Path] = PREPROCESSING_DIR / "helpers"
SCRIPTS_DIR: Final[Path] = PREPROCESSING_DIR.parent
AD_ROOT: Final[Path] = SCRIPTS_DIR.parent
RECOMMENDER_ROOT: Final[Path] = AD_ROOT.parent


def _configured_path(environment_variable: str, default: Path) -> Path:
    """Use an explicit mount path when this module runs outside the project tree."""
    return Path(os.environ.get(environment_variable, str(default))).expanduser()

# Keep preprocessing selection independent from Data_Extraction.
# Supported forms:
#   ACTIVE_STUDIES = ["all"]                        # every registered study
#   ACTIVE_STUDIES = ["training"]                  # one registered group
#   ACTIVE_STUDIES = ["study_111"]                 # one explicit study
#   ACTIVE_STUDIES = ["study_111", "study_122"]   # several explicit studies

ACTIVE_STUDIES: list[str] = ["study_221"]

# RERUN CONTROLS --------------------------------------------------------------
# Both stage scripts honor this setting. Set it back to False after the rerun.
OVERWRITE_EXISTING_OUTPUTS: bool = True

# Keep cached LLM table-repair/header decisions unless you deliberately changed
# an LLM prompt or response schema. Deterministic routing fixes do not require
# new LLM calls.
FORCE_RERUN_LLM: bool = False

StudyGroup = Literal["training", "validation", "application"]

# Copied from ``web of science search/studies.txt``.  The registry is kept
# locally so a preprocessing run never imports another workflow's config.
STUDY_GROUPS: Final[dict[StudyGroup, tuple[str, ...]]] = {
    "training": (
        "study_03", "study_08", "study_10", "study_100", "study_11", "study_24",
        "study_45", "study_51", "study_64", "study_67", "study_71", "study_72",
        "study_80", "study_94", "study_98", "study_01", "study_02", "study_04",
        "study_09", "study_13", "study_14", "study_17", "study_19", "study_23",
        "study_26", "study_34", "study_37", "study_40", "study_56", "study_58",
        "study_61", "study_62", "study_70", "study_73", "study_74", "study_89",
        "study_92", "study_95", "study_97", "study_101", "study_108",
    ),
    "validation": (
        "study_06", "study_07", "study_27", "study_35", "study_36", "study_41",
        "study_65", "study_76", "study_78", "study_93", "study_22", "study_28",
        "study_39", "study_81", "study_86", "study_87", "study_91", "study_99",
        "study_102", "study_106", "study_160", "study_164", "study_174",
        "study_195", "study_214", "study_229", "study_234",
    ),
    "application": (
        "study_05", "study_12", "study_15", "study_16", "study_18", "study_20",
        "study_21", "study_25",  "study_31", "study_32", "study_33",
        "study_38", "study_42", "study_43", "study_44", "study_46", "study_47",
        "study_48", "study_49", "study_50", "study_52", "study_53", "study_54",
        "study_55", "study_57", "study_59", "study_60", "study_63", "study_66",
        "study_68", "study_69", "study_75", "study_77", "study_79", "study_82",
        "study_83", "study_84", "study_85", "study_88", "study_90", 
        "study_103", "study_104", "study_105", "study_109", "study_110",
        "study_111", "study_112", "study_113", "study_114", "study_116", "study_117",
        "study_118", "study_119", "study_120", "study_121", "study_122", "study_124",
        "study_125", "study_126", "study_127", "study_128", "study_129", "study_131",
        "study_133", "study_134", "study_136", "study_137", "study_138", "study_144",
        "study_148", "study_161", "study_167", "study_172", "study_175", "study_179",
        "study_181", "study_184", "study_185", "study_186", "study_192", "study_199",
        "study_205", "study_206", "study_208", "study_209", "study_212", "study_213",
        "study_215", "study_218", "study_219", "study_220", "study_221", "study_224",
        "study_225", "study_226", "study_228", "study_230", "study_231", "study_232",
        "study_233", "study_236", "study_237", "study_240", "study_241", "study_245",
        "study_247", "study_249", "study_250", "study_254",
    ),
}


def _study_group_index() -> dict[str, StudyGroup]:
    index: dict[str, StudyGroup] = {}
    duplicates: set[str] = set()
    for group, study_ids in STUDY_GROUPS.items():
        for study_id in study_ids:
            normalized = str(study_id).strip()
            if not normalized:
                raise ValueError(f"Blank study ID in {group} registry")
            if normalized in index:
                duplicates.add(normalized)
            index[normalized] = group
    if duplicates:
        raise ValueError(
            "Study IDs must occur in exactly one group; duplicates: "
            + ", ".join(sorted(duplicates))
        )
    return index


STUDY_GROUP_BY_ID: Final[dict[str, StudyGroup]] = _study_group_index()

WRITE_DEBUG_ARTIFACTS: bool = True
DEBUG_PRINT_MAX_CHARS: int | None = None

# Raw source documents and generated preprocessing artifacts.  On Windows the
# project tree is complete, so these paths are derived above.  In WSL the
# scripts and data are commonly mounted separately; default to the established
# ``~/docling`` mount layout and permit explicit overrides for other layouts.
if os.name == "nt":
    _DEFAULT_SOURCE_STUDIES_DIR = RECOMMENDER_ROOT / "Adsorbent_Data"
    _DEFAULT_PROCESSED_DIR = RECOMMENDER_ROOT / "Processed"
    _DEFAULT_STUDY_RECORDS_XLSX = (
        RECOMMENDER_ROOT / "web of science search" / "within_scope_records.xlsx"
    )
else:
    _DEFAULT_SOURCE_STUDIES_DIR = SCRIPTS_DIR / "Adsorbent_Data"
    _DEFAULT_PROCESSED_DIR = SCRIPTS_DIR / "processed"
    _DEFAULT_STUDY_RECORDS_XLSX = SCRIPTS_DIR / "within_scope_records.xlsx"

SOURCE_STUDIES_DIR: Final[Path] = _configured_path(
    "ADSORBENT_SOURCE_STUDIES_DIR", _DEFAULT_SOURCE_STUDIES_DIR
)
PROCESSED_DIR: Final[Path] = _configured_path(
    "ADSORBENT_PROCESSED_DIR", _DEFAULT_PROCESSED_DIR
)
CHUNKED_DATA_DIR: Final[Path] = PROCESSED_DIR / "chunked_data"
STUDY_RECORDS_XLSX: Final[Path] = _configured_path(
    "ADSORBENT_STUDY_RECORDS_XLSX", _DEFAULT_STUDY_RECORDS_XLSX
)
FIGURE_TABLE_INFO_CSV: Final[Path] = PROCESSED_DIR / "figure_table_info.csv"
PROCESSING_REPORT_FILE: Final[Path] = PROCESSED_DIR / "processing_report.txt"

# A study can have either or both source files.  Docling accepts PDF and DOCX;
# the order is deterministic if a study happens to contain both formats.
SOURCE_FILE_STEMS: Final[tuple[str, ...]] = (
    "main_paper",
    "supplementary_material",
)
SUPPORTED_DOCUMENT_SUFFIXES: Final[tuple[str, ...]] = (".pdf", ".docx")


@dataclass(frozen=True)
class StudyPaths:
    """All upstream locations for one centrally selected study."""

    study_id: str
    group: str
    source_study_dir: Path
    processed_study_dir: Path

    @property
    def chunk_input_file(self) -> Path:
        return self.processed_study_dir / "chunk_input.jsonl"

    @property
    def document_objects_file(self) -> Path:
        return self.processed_study_dir / "document_objects.jsonl"

    @property
    def chunked_output_file(self) -> Path:
        return CHUNKED_DATA_DIR / f"{self.study_id}_chunks.json"

    def docling_json_file(self, source_stem: str) -> Path:
        return self.source_study_dir / f"{source_stem}.docling.json"

    def markdown_file(self, source_stem: str) -> Path:
        return self.source_study_dir / f"{source_stem}.md"

    def source_document_candidates(self, source_stem: str) -> tuple[Path, ...]:
        return tuple(
            self.source_study_dir / f"{source_stem}{suffix}"
            for suffix in SUPPORTED_DOCUMENT_SUFFIXES
        )


def get_study_paths(study_id: str) -> StudyPaths:
    """Resolve a registered study before any preprocessing stage writes."""
    normalized = str(study_id or "").strip()
    try:
        group = STUDY_GROUP_BY_ID[normalized]
    except KeyError as exc:
        raise ValueError(
            f"Unknown study ID {normalized!r}. Add it to exactly one STUDY_GROUPS entry."
        ) from exc
    return StudyPaths(
        study_id=normalized,
        group=group,
        source_study_dir=SOURCE_STUDIES_DIR / normalized,
        processed_study_dir=PROCESSED_DIR / normalized,
    )


def iter_active_study_paths() -> Iterable[StudyPaths]:
    """Yield validated active studies from one list-based selector."""
    selection = tuple(str(study_id).strip() for study_id in ACTIVE_STUDIES if str(study_id).strip())
    normalized_selection = tuple(study_id.casefold() for study_id in selection)
    selector_names = {"all", *STUDY_GROUPS}
    selector_entries = [study_id for study_id in normalized_selection if study_id in selector_names]

    if selector_entries:
        if len(selection) != 1:
            raise ValueError(
                "ACTIVE_STUDIES group selectors must be used alone; use ['all'], "
                "['training'], ['validation'], or ['application']."
            )
        normalized = normalized_selection[0]
        if normalized == "all":
            selected_study_ids = STUDY_GROUP_BY_ID.keys()
        else:
            selected_study_ids = STUDY_GROUPS[normalized]
    else:
        selected_study_ids = selection

    for study_id in selected_study_ids:
        yield get_study_paths(study_id)


def source_document_for(paths: StudyPaths, source_stem: str) -> Path | None:
    """Return the preferred available PDF/DOCX source, if any."""
    return next(
        (candidate for candidate in paths.source_document_candidates(source_stem) if candidate.is_file()),
        None,
    )


def existing_docling_sources(paths: StudyPaths) -> tuple[Path, ...]:
    """Return available structured source files in deterministic order."""
    return tuple(
        paths.docling_json_file(source_stem)
        for source_stem in SOURCE_FILE_STEMS
        if paths.docling_json_file(source_stem).is_file()
    )


def conversion_preflight(paths: StudyPaths) -> list[str]:
    """Require at least one raw/structured source without writing anything.

    Supplementary material is optional, so a study with only a main paper is
    valid.  Existing Docling JSON is also sufficient for a safe no-op rerun.
    """
    if not paths.source_study_dir.is_dir():
        return [f"source study directory is missing: {paths.source_study_dir}"]
    if existing_docling_sources(paths):
        return []
    if any(source_document_for(paths, source_stem) is not None for source_stem in SOURCE_FILE_STEMS):
        return []
    expected = ", ".join(
        candidate.name
        for source_stem in SOURCE_FILE_STEMS
        for candidate in paths.source_document_candidates(source_stem)
    )
    return [f"no PDF/DOCX or Docling JSON source found; expected one of {expected} in {paths.source_study_dir}"]


def json_preprocess_preflight(paths: StudyPaths) -> list[str]:
    """Describe missing structured inputs without writing a processed folder."""
    if not paths.source_study_dir.is_dir():
        return [f"source study directory is missing: {paths.source_study_dir}"]
    if existing_docling_sources(paths):
        return []
    expected = ", ".join(f"{stem}.docling.json" for stem in SOURCE_FILE_STEMS)
    return [
        f"no structured source found for {paths.study_id}; expected {expected} in "
        f"{paths.source_study_dir}. Convert the source document to Docling JSON first."
    ]


def chunk_preflight(paths: StudyPaths) -> list[str]:
    """Describe missing JSON-first records without writing a chunk output."""
    if paths.chunk_input_file.is_file():
        return []
    return [
        f"missing JSON-first records: {paths.chunk_input_file}. "
        "Run docling_json_preprocess.py first."
    ]


def add_project_import_paths() -> None:
    """Make root ``chains`` plus local helper modules importable directly."""
    for directory in (AD_ROOT, SCRIPTS_DIR, PREPROCESSING_DIR):
        directory_string = str(directory)
        if directory_string not in sys.path:
            sys.path.insert(0, directory_string)


__all__ = [
    "ACTIVE_STUDIES",
    "AD_ROOT",
    "CHUNKED_DATA_DIR",
    "DEBUG_PRINT_MAX_CHARS",
    "FIGURE_TABLE_INFO_CSV",
    "FORCE_RERUN_LLM",
    "HELPERS_DIR",
    "OVERWRITE_EXISTING_OUTPUTS",
    "PREPROCESSING_DIR",
    "PROCESSING_REPORT_FILE",
    "PROCESSED_DIR",
    "SOURCE_FILE_STEMS",
    "SOURCE_STUDIES_DIR",
    "STUDY_GROUPS",
    "STUDY_RECORDS_XLSX",
    "SUPPORTED_DOCUMENT_SUFFIXES",
    "StudyPaths",
    "WRITE_DEBUG_ARTIFACTS",
    "add_project_import_paths",
    "chunk_preflight",
    "conversion_preflight",
    "existing_docling_sources",
    "get_study_paths",
    "iter_active_study_paths",
    "json_preprocess_preflight",
    "source_document_for",
]
