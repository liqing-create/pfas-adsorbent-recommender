from __future__ import annotations

import sys
import unittest
from pathlib import Path


PREPROCESSING_DIR = Path(__file__).resolve().parents[1]
if str(PREPROCESSING_DIR) not in sys.path:
    sys.path.insert(0, str(PREPROCESSING_DIR))

import docling_json_preprocess as preprocessing


def _table_cell_proxy(block: str) -> int:
    pipe_lines = [line for line in block.splitlines() if line.lstrip().startswith("|")]
    first_row = pipe_lines[0].strip().strip("|").split("|") if pipe_lines else []
    return max(0, len(pipe_lines) - 2) * len(first_row)


class TableSplittingTests(unittest.TestCase):
    def test_split_uses_widest_row_not_first_header_row(self) -> None:
        # Mimics study_58: a narrow first header is expanded by wider data rows
        # during Markdown formatting.  The preprocessor must still honor the
        # same 200-cell proxy that enumeration uses.
        header = "| " + " | ".join(f"h{i}" for i in range(6)) + " |"
        separator = "| " + " | ".join("---" for _ in range(6)) + " |"
        data_rows = [
            "| " + " | ".join(f"r{row}c{col}" for col in range(8)) + " |"
            for row in range(27)
        ]
        block = "Table S6\n\n" + "\n".join([header, separator, *data_rows])

        original_resolver = preprocessing._resolve_table_header_rows_for_block
        preprocessing._resolve_table_header_rows_for_block = lambda **_: {"header_count": 1}
        try:
            parts = preprocessing._split_large_table_block(
                table_obj={},
                full_block=block,
                caption="Table S6",
                footnote="",
            )
        finally:
            preprocessing._resolve_table_header_rows_for_block = original_resolver

        self.assertGreater(len(parts), 1)
        self.assertTrue(all(_table_cell_proxy(part) <= 200 for part in parts))

    def test_table_issue_rows_are_unique_per_table(self) -> None:
        rows = [
            {
                "study": "study_114",
                "file_type": "main_paper",
                "table_name": "Uncaptioned Table 1",
                "issue": "caption_missing_or_invalid",
            },
            {
                "study": "study_114",
                "file_type": "main_paper",
                "table_name": "Uncaptioned Table 1",
                "issue": "table_repair_needed",
            },
        ]

        result = preprocessing._deduplicate_table_issue_rows(rows)

        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["issue"], "table_repair_needed")

    def test_blank_decisions_remain_pending(self) -> None:
        rows = [
            {
                "study": "study_114",
                "file_type": "main_paper",
                "table_name": "Uncaptioned Table 1",
                "issue": "table_repair_needed",
                "Decision": "",
                "fixed_file": "",
            }
        ]

        result = preprocessing._table_issue_decisions_from_rows(rows)

        self.assertEqual(result, {})

    def test_only_unseen_table_issues_are_added_to_decisions(self) -> None:
        existing_rows = [
            {
                "study": "study_1",
                "file_type": "main_paper",
                "table_name": "Table 1",
                "issue": "table_repair_needed",
                "Decision": "",
                "fixed_file": "",
            }
        ]
        issue_rows = [
            {
                "study": "study_1",
                "file_type": "main_paper",
                "table_name": "Table 1",
                "issue": "caption_missing_or_invalid",
            },
            {
                "study": "study_2",
                "file_type": "main_paper",
                "table_name": "Table 2",
                "issue": "table_repair_needed",
            },
        ]

        result = preprocessing._new_table_issue_decision_rows(
            existing_rows,
            issue_rows,
        )

        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["study"], "study_2")
        self.assertEqual(result[0]["Decision"], "")
        self.assertEqual(result[0]["fixed_file"], "")


if __name__ == "__main__":
    unittest.main()
