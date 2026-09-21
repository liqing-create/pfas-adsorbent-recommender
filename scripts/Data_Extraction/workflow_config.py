"""Shared configuration for the adsorbent data-extraction workflow.

Edit :data:`ACTIVE_STUDIES` to choose the papers for the next run.  It accepts
``["all"]``, one group name such as ``["training"]``, or one or more explicit
study IDs.  The study-to-group registry and every input/output directory live
here so stage scripts never need a machine-specific path or a separate study
list.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Iterable, Literal


# Supported forms:
#   ACTIVE_STUDIES = ["all"]                       # every registered study
#   ACTIVE_STUDIES = ["training"]                 # one registered group
#   ACTIVE_STUDIES = ["study_111"]                # one explicit study
#   ACTIVE_STUDIES = ["study_111", "study_122"]  # several explicit studies
#
# Keep the current explicit batch as the default.  Stages make LLM calls, so
# switch to ["all"] or a group deliberately when running a larger batch.
ACTIVE_STUDIES: list[str] = ["study_221"]

# Safe default: existing terminal artifacts are reused.  Turn this on only
# when deliberately regenerating a selected study.
OVERWRITE_EXISTING_OUTPUTS: bool = True


# Repository paths are derived from this file:
# <Box>/My Box Notes/2. Adsorbent Recommender/AD/scripts/Data_Extraction
DATA_EXTRACTION_DIR: Final[Path] = Path(__file__).resolve().parent
SCRIPTS_DIR: Final[Path] = DATA_EXTRACTION_DIR.parent
AD_ROOT: Final[Path] = SCRIPTS_DIR.parent
RECOMMENDER_ROOT: Final[Path] = AD_ROOT.parent
BOX_NOTES_ROOT: Final[Path] = RECOMMENDER_ROOT.parent
BOX_ROOT: Final[Path] = BOX_NOTES_ROOT.parent

PROMPTS_DIR: Final[Path] = AD_ROOT / "prompts"
OUTPUT_ROOT: Final[Path] = AD_ROOT / "output"
CHUNKED_DATA_DIR: Final[Path] = RECOMMENDER_ROOT / "Processed" / "chunked_data"
STUDY_RECORDS_XLSX: Final[Path] = (
    RECOMMENDER_ROOT / "web of science search" / "within_scope_records.xlsx"
)
# Produced and maintained by ``scripts/Preprocessing/docling_json_preprocess.py``.
# Keeping this source-of-truth path here ensures caption context follows the
# same newly processed studies as the chunks consumed by classification.
FIGURE_TABLE_INFO_CSV: Final[Path] = RECOMMENDER_ROOT / "Processed" / "figure_table_info.csv"
STUDY_REGISTRY_REFERENCE: Final[Path] = (
    RECOMMENDER_ROOT / "web of science search" / "studies.txt"
)


StudyGroup = Literal["training", "validation", "application"]

# This is intentionally copied from ``web of science search/studies.txt``.
# Keep that text file as the human-readable reference, and edit this mapping
# when a study changes group.
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
        "study_21", "study_25", "study_31", "study_32", "study_33",
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


@dataclass(frozen=True)
class GroupDirectories:
    """Existing run and human-annotation directories for one study group."""

    classification: Path
    enumeration: Path
    extraction: Path
    classification_ground_truth: Path
    enumeration_ground_truth: Path
    extraction_ground_truth: Path


# Do not infer these paths from a shared naming convention.  Training and
# Application use different historical layouts, so the map is deliberately
# explicit and remains the only place to change a future run directory.
GROUP_DIRECTORIES: Final[dict[StudyGroup, GroupDirectories]] = {
    "training": GroupDirectories(
        classification=OUTPUT_ROOT / "Classification" / "Training" / "LLM_prediction" / "GPT4.1_20260701",
        enumeration=OUTPUT_ROOT / "Enumeration" / "Training" / "LLM_prediction" / "GPT4.1_20260603",
        extraction=OUTPUT_ROOT / "Extraction" / "Training" / "LLM_prediction" / "GPT4.1_20260603",
        classification_ground_truth=OUTPUT_ROOT / "Classification" / "Training" / "Ground_truth",
        enumeration_ground_truth=OUTPUT_ROOT / "Enumeration" / "Training" / "Ground_truth",
        extraction_ground_truth=OUTPUT_ROOT / "Extraction" / "Training" / "Ground_truth",
    ),
    "validation": GroupDirectories(
        classification=OUTPUT_ROOT / "Classification" / "Validation" / "LLM_prediction" / "GPT4.1_20260705",
        enumeration=OUTPUT_ROOT / "Enumeration" / "Validation" / "LLM_prediction" / "GPT4.1_20260705",
        extraction=OUTPUT_ROOT / "Extraction" / "Validation" / "LLM_prediction" / "GPT4.1_20260705",
        classification_ground_truth=OUTPUT_ROOT / "Classification" / "Validation" / "Ground_truth",
        enumeration_ground_truth=OUTPUT_ROOT / "Enumeration" / "Validation" / "Ground_truth",
        extraction_ground_truth=OUTPUT_ROOT / "Extraction" / "Validation" / "Ground_truth",
    ),
    "application": GroupDirectories(
        classification=OUTPUT_ROOT / "Classification" / "Application" / "GPT4.1_20260705",
        enumeration=OUTPUT_ROOT / "Enumeration" / "Application" / "GPT4.1_20260705",
        extraction=OUTPUT_ROOT / "Extraction" / "Application" / "GPT4.1_20260705",
        classification_ground_truth=OUTPUT_ROOT / "Classification" / "Application" / "Ground_truth",
        enumeration_ground_truth=OUTPUT_ROOT / "Enumeration" / "Application" / "Ground_truth",
        extraction_ground_truth=OUTPUT_ROOT / "Extraction" / "Application" / "Ground_truth",
    ),
}


@dataclass(frozen=True)
class StudyPaths:
    """All group-aware workflow locations for one study."""

    study_id: str
    group: StudyGroup
    classification_dir: Path
    enumeration_dir: Path
    extraction_dir: Path
    classification_ground_truth_dir: Path
    enumeration_ground_truth_dir: Path
    extraction_ground_truth_dir: Path

    @property
    def chunked_input_file(self) -> Path:
        return CHUNKED_DATA_DIR / f"{self.study_id}_chunks.json"

    @property
    def classified_file(self) -> Path:
        return self.classification_dir / f"{self.study_id}_classified.json"

    @property
    def enumeration_artifact_dir(self) -> Path:
        return self.enumeration_dir / "_chain_artifacts" / self.study_id

    @property
    def extraction_artifact_dir(self) -> Path:
        return self.extraction_dir / "_chain_artifacts" / self.study_id

    def enumeration_task_artifact(self, task: str) -> Path:
        """Return the canonical output for one enumeration task."""
        return self.enumeration_artifact_dir / f"{self.study_id}_{task}.json"

    def extraction_chain_artifact(self, chain_name: str) -> Path:
        """Return the canonical output for one extraction chain."""
        return self.extraction_artifact_dir / f"{self.study_id}_{chain_name}.json"

    @property
    def enumeration_error_dir(self) -> Path:
        return self.enumeration_dir / "_error_logs"

    @property
    def extraction_error_dir(self) -> Path:
        return self.extraction_dir / "_error_logs"

    @property
    def classification_annotation_file(self) -> Path:
        return self.classification_ground_truth_dir / f"annotations_{self.study_id}.json"

    @property
    def extraction_ground_truth_file(self) -> Path:
        return self.extraction_ground_truth_dir / f"{self.study_id}_ground_truth.json"

    @property
    def extraction_highlights_file(self) -> Path:
        return self.extraction_ground_truth_dir / f"{self.study_id}_highlights.json"


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


def get_study_paths(study_id: str) -> StudyPaths:
    """Resolve all locations for ``study_id`` or fail before a stage writes."""
    normalized = str(study_id or "").strip()
    try:
        group = STUDY_GROUP_BY_ID[normalized]
    except KeyError as exc:
        raise ValueError(
            f"Unknown study ID {normalized!r}. Add it to exactly one STUDY_GROUPS entry."
        ) from exc
    dirs = GROUP_DIRECTORIES[group]
    return StudyPaths(
        study_id=normalized,
        group=group,
        classification_dir=dirs.classification,
        enumeration_dir=dirs.enumeration,
        extraction_dir=dirs.extraction,
        classification_ground_truth_dir=dirs.classification_ground_truth,
        enumeration_ground_truth_dir=dirs.enumeration_ground_truth,
        extraction_ground_truth_dir=dirs.extraction_ground_truth,
    )


def iter_active_study_paths() -> Iterable[StudyPaths]:
    """Yield the selected studies in configured order, validating each one."""
    selection = tuple(
        str(study_id).strip()
        for study_id in ACTIVE_STUDIES
        if str(study_id).strip()
    )
    normalized_selection = tuple(study_id.casefold() for study_id in selection)
    selector_names = {"all", *STUDY_GROUPS}
    selector_entries = [
        study_id for study_id in normalized_selection if study_id in selector_names
    ]

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


def active_studies_by_group() -> dict[StudyGroup, list[StudyPaths]]:
    """Return the active selection partitioned by its output directories."""
    grouped: dict[StudyGroup, list[StudyPaths]] = {
        "training": [],
        "validation": [],
        "application": [],
    }
    for paths in iter_active_study_paths():
        grouped[paths.group].append(paths)
    return {group: entries for group, entries in grouped.items() if entries}


def add_project_import_paths() -> None:
    """Make ``chains`` and root-level ``schemas`` importable after relocation."""
    for directory in (AD_ROOT, SCRIPTS_DIR):
        directory_string = str(directory)
        if directory_string not in sys.path:
            sys.path.insert(0, directory_string)


def check_stage_input(paths: StudyPaths, stage: Literal["classification", "enumeration", "extraction"]) -> Path:
    """Return a required input file or raise a clear, non-writing error."""
    required = {
        "classification": paths.chunked_input_file,
        "enumeration": paths.classified_file,
        "extraction": paths.enumeration_task_artifact("performance"),
    }[stage]
    if not required.exists():
        raise FileNotFoundError(
            f"{paths.study_id} ({paths.group}) cannot start {stage}: required input is missing: {required}"
        )
    return required


__all__ = [
    "ACTIVE_STUDIES",
    "AD_ROOT",
    "CHUNKED_DATA_DIR",
    "DATA_EXTRACTION_DIR",
    "FIGURE_TABLE_INFO_CSV",
    "GROUP_DIRECTORIES",
    "OVERWRITE_EXISTING_OUTPUTS",
    "PROMPTS_DIR",
    "STUDY_GROUPS",
    "STUDY_GROUP_BY_ID",
    "STUDY_RECORDS_XLSX",
    "StudyPaths",
    "active_studies_by_group",
    "add_project_import_paths",
    "check_stage_input",
    "get_study_paths",
    "iter_active_study_paths",
]
