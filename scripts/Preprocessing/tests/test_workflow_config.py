from __future__ import annotations

import unittest
from dataclasses import replace
from pathlib import Path
import sys

PREPROCESSING_DIR = Path(__file__).resolve().parents[1]
if str(PREPROCESSING_DIR) not in sys.path:
    sys.path.insert(0, str(PREPROCESSING_DIR))

from workflow_config import (
    ACTIVE_STUDIES,
    CHUNKED_DATA_DIR,
    FIGURE_TABLE_INFO_CSV,
    PROCESSED_DIR,
    SOURCE_STUDIES_DIR,
    STUDY_GROUPS,
    chunk_preflight,
    conversion_preflight,
    get_study_paths,
    json_preprocess_preflight,
)


# The source documents and processed corpus are distributed separately from the
# code; tests that need them are skipped when the data folders are absent.
_DATA_PRESENT = SOURCE_STUDIES_DIR.exists() and PROCESSED_DIR.exists()
_needs_data = unittest.skipUnless(_DATA_PRESENT, "study data folders not present")


class WorkflowConfigTests(unittest.TestCase):
    def test_local_registry_is_unique_and_active_selection_is_valid(self) -> None:
        registered = [study_id for studies in STUDY_GROUPS.values() for study_id in studies]
        self.assertEqual(len(registered), len(set(registered)))
        for study_id in ACTIVE_STUDIES:
            self.assertIsNotNone(get_study_paths(study_id))

    def test_known_paths_resolve_for_each_group(self) -> None:
        expected_groups = {
            "study_03": "training",
            "study_174": "validation",
            "study_241": "application",
        }
        for study_id, expected_group in expected_groups.items():
            with self.subTest(study_id=study_id):
                paths = get_study_paths(study_id)
                self.assertEqual(paths.group, expected_group)
                self.assertEqual(paths.source_study_dir, SOURCE_STUDIES_DIR / study_id)
                self.assertEqual(paths.processed_study_dir, PROCESSED_DIR / study_id)
                self.assertEqual(paths.chunked_output_file, CHUNKED_DATA_DIR / f"{study_id}_chunks.json")

    @_needs_data
    def test_shared_roots_exist(self) -> None:
        self.assertTrue(SOURCE_STUDIES_DIR.exists())
        self.assertTrue(PROCESSED_DIR.exists())
        self.assertTrue(FIGURE_TABLE_INFO_CSV.exists())

    @_needs_data
    def test_existing_validation_study_passes_stage_preflights(self) -> None:
        paths = get_study_paths("study_174")
        self.assertEqual(json_preprocess_preflight(paths), [])
        self.assertEqual(chunk_preflight(paths), [])

    def test_unknown_study_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            get_study_paths("study_not_registered")

    def test_missing_inputs_report_without_creating_paths(self) -> None:
        normal_paths = get_study_paths("study_03")
        absent_root = Path("__preprocessing_preflight_missing__")
        missing_paths = replace(
            normal_paths,
            source_study_dir=absent_root / "source",
            processed_study_dir=absent_root / "processed",
        )
        self.assertFalse(missing_paths.source_study_dir.exists())
        self.assertIn("source study directory is missing", conversion_preflight(missing_paths)[0])
        self.assertIn("missing JSON-first records", chunk_preflight(missing_paths)[0])
        self.assertFalse(absent_root.exists())


if __name__ == "__main__":
    unittest.main()
