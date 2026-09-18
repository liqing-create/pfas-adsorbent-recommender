from __future__ import annotations

import sys
import unittest
from pathlib import Path


PREPROCESSING_DIR = Path(__file__).resolve().parents[1]
if str(PREPROCESSING_DIR) not in sys.path:
    sys.path.insert(0, str(PREPROCESSING_DIR))

import chunk_json_input as chunking


class ChunkIssueDecisionTests(unittest.TestCase):
    def setUp(self) -> None:
        chunking._reset_attention_state()

    def tearDown(self) -> None:
        chunking._reset_attention_state()

    def test_pipe_normalization_is_not_a_review_issue(self) -> None:
        normalized = chunking.normalize_docling_heading_pipes(
            "# Header | with pipes",
            study_folder="study_1",
            source_hint="main_paper",
        )

        self.assertEqual(normalized, "# Header - with pipes")
        self.assertEqual(chunking.ATTENTION_BY_STUDY, {})
        self.assertEqual(chunking.CURRENT_ATTENTION_ISSUE_ROWS, [])

    def test_completed_warning_is_not_reported_again(self) -> None:
        row = {
            "study": "study_1",
            "source": "main_paper",
            "issue": "Docling ignored overlong headers",
        }
        chunking.CHUNK_ISSUE_DECISIONS[chunking._chunk_issue_key(row)] = "fixed"

        should_report = chunking._record_attention(
            "study_1",
            row["issue"],
            source="main_paper",
        )

        self.assertFalse(should_report)
        self.assertEqual(chunking.ATTENTION_BY_STUDY, {})

    def test_only_unseen_warnings_are_added_with_blank_decisions(self) -> None:
        existing_rows = [
            {
                "study": "study_1",
                "source": "main_paper",
                "issue": "missing section",
                "Decision": "",
            }
        ]
        issue_rows = [
            *existing_rows,
            {
                "study": "study_2",
                "source": "supplementary_material",
                "issue": "produced 0 supplementary chunks",
            },
        ]

        result = chunking._new_chunk_issue_decision_rows(
            existing_rows,
            issue_rows,
        )

        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["study"], "study_2")
        self.assertEqual(result[0]["Decision"], "")

    def test_resolved_section_warning_is_removed_from_current_summary(self) -> None:
        chunking.ATTENTION_BY_STUDY["study_229"] = [
            "chunk_input.jsonl: MATERIALS/METHODS section not present",
            "chunk_input.jsonl: another issue",
        ]

        chunking._clear_resolved_section_attention(
            "study_229",
            {"abstract", "materials_and_methods"},
        )

        self.assertEqual(
            chunking.ATTENTION_BY_STUDY["study_229"],
            ["chunk_input.jsonl: another issue"],
        )


if __name__ == "__main__":
    unittest.main()
