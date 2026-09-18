from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import patch


PREPROCESSING_DIR = Path(__file__).resolve().parents[1]
if str(PREPROCESSING_DIR) not in sys.path:
    sys.path.insert(0, str(PREPROCESSING_DIR))

import docling_json_preprocess as preprocess  # noqa: E402
from helpers.table_repair_core import (  # noqa: E402
    make_table_repair_candidate,
    select_best_table_repair_candidate,
)


class DoclingJsonPreprocessTests(unittest.TestCase):
    def test_systematically_fragmented_pdf_text_is_not_eligible_for_auto_selection(self):
        reference = """\
| Adsorbent | Manufacturer | Type | Reference One | Reference Two | Notes |
| --- | --- | --- | --- | --- | --- |
| Filtrasorb | Calgon | Bituminous coal | Crincoli | Dastgheib | primary |
"""
        camelot = make_table_repair_candidate(
            name="camelot_stream_unguided",
            block=reference,
            payload={},
            expected_header_rows=None,
            caption_text="",
            footnote_text="",
            reference_block=reference,
        )
        pymupdf = make_table_repair_candidate(
            name="pymupdf_text",
            block="""\
| Adsorbent | Manufacturer | Type | Reference One | Reference Two | Notes |
| --- | --- | --- | --- | --- | --- |
| Filtrasorb | Calgon | Bituminous coal | Crinc | oli Dastg | heib |
""",
            payload={},
            expected_header_rows=None,
            caption_text="",
            footnote_text="",
            reference_block=reference,
        )

        self.assertTrue(camelot["accepted"])
        self.assertFalse(pymupdf["accepted"])
        self.assertIn(
            "systematic_word_fragmentation_across_cells",
            pymupdf["reasons"],
        )
        self.assertEqual(
            select_best_table_repair_candidate([camelot, pymupdf])["name"],
            "camelot_stream_unguided",
        )

    def test_short_words_and_units_split_across_cells_require_repair(self):
        reference = """\
| Type | Micro-V, cm 3/g | Meso-V, cm 3/g |
| --- | --- | --- |
| Bituminous coal | 0.38 | 0.08 |
"""
        camelot = make_table_repair_candidate(
            name="camelot_stream_unguided",
            block="""\
| Type | Micro-V, c | m 3/g | Meso-V, c | m 3/g |
| --- | --- | --- | --- | --- |
| Bituminous coa | l 840 | 0.38 |  | 0.08 |
""",
            payload={},
            expected_header_rows=None,
            caption_text="",
            footnote_text="",
            reference_block=reference,
        )

        self.assertFalse(camelot["accepted"])
        self.assertGreaterEqual(
            camelot["quality"]["lexical_fragmentation_penalty"],
            3,
        )
        self.assertIn(
            "systematic_word_fragmentation_across_cells",
            camelot["reasons"],
        )

    def test_one_split_word_counts_once_across_overlapping_windows(self):
        reference = """\
| Adsorbate | Value |
| --- | --- |
| PFOA | 1 |
"""
        candidate = make_table_repair_candidate(
            name="candidate",
            block="""\
| Ad | sorbate | Value |
| --- | --- | --- |
| PFOA |  | 1 |
""",
            payload={},
            expected_header_rows=None,
            caption_text="",
            footnote_text="",
            reference_block=reference,
        )

        self.assertEqual(candidate["quality"]["lexical_fragmentation_penalty"], 1)
        self.assertTrue(candidate["accepted"])

    def test_intact_words_do_not_form_a_false_boundary_token(self):
        reference = """\
| Description | Received from |
| --- | --- |
| Calcium ions are removed. | Soil laboratory |
"""
        candidate = make_table_repair_candidate(
            name="docx_native",
            block="""\
| Description | Received from |
| --- | --- |
| Adsorption occurs. | Soil laboratory |
""",
            payload={},
            expected_header_rows=None,
            caption_text="",
            footnote_text="",
            reference_block=reference,
        )

        self.assertEqual(candidate["quality"]["lexical_fragmentation_penalty"], 0)
        self.assertTrue(candidate["accepted"])

    def test_adjacent_complete_units_do_not_form_a_false_fragment(self):
        reference = """\
| qe (ng/mg) | k2 (mg/ng hr) |
| --- | --- |
| 1.2 | 0.4 |
"""
        candidate = make_table_repair_candidate(
            name="docx_native",
            block="""\
| qe (ng/mg) | qe (ng/mg) | k2 (mg/ng hr) |
| --- | --- | --- |
| 1.2 | 1.2 | 0.4 |
""",
            payload={},
            expected_header_rows=None,
            caption_text="",
            footnote_text="",
            reference_block=reference,
        )

        self.assertEqual(candidate["quality"]["lexical_fragmentation_penalty"], 0)
        self.assertTrue(candidate["accepted"])

    @staticmethod
    def _study_218_molarity_decisions():
        return [
            preprocess.DoclingTextRepairDecision(
                study="study_218",
                source_file="main_paper",
                object_ref="#/texts/49",
                match_pattern=r"241\.5\s+m\s+M\b",
                replacement="241.5 \u03bcM",
                expected_matches=1,
                row_number=1,
            ),
            preprocess.DoclingTextRepairDecision(
                study="study_218",
                source_file="main_paper",
                object_ref="#/texts/67",
                match_pattern=r"120\.8\s+m\s+M\b",
                replacement="120.8 \u03bcM",
                expected_matches=1,
                row_number=2,
            ),
            preprocess.DoclingTextRepairDecision(
                study="study_218",
                source_file="main_paper",
                object_ref="#/texts/67",
                match_pattern=r"241\.5\s+m\s+Min\b",
                replacement="241.5 \u03bcM in",
                expected_matches=1,
                row_number=3,
            ),
            preprocess.DoclingTextRepairDecision(
                study="study_218",
                source_file="main_paper",
                object_ref="#/texts/67",
                match_pattern=r"241\.5\s+m\s+M\b",
                replacement="241.5 \u03bcM",
                expected_matches=1,
                row_number=4,
            ),
            preprocess.DoclingTextRepairDecision(
                study="study_218",
                source_file="main_paper",
                object_ref="#/texts/77",
                match_pattern=r"241\.5\s+m\s+M\b",
                replacement="241.5 \u03bcM",
                expected_matches=1,
                row_number=5,
            ),
        ]

    def test_approved_docling_text_repairs_are_loaded_from_csv_ledger(self):
        first = "Each PFAS solution was made as a master batch at 241.5 m M."
        batch = (
            "Initial tests used PFOA (241.5 m M); subsequent solutions were "
            "prepared to 241.5 m Min water. The blank sample was equimolar "
            "(120.8 m M)."
        )
        final = "All PFAS were dosed at 241.5 m M."
        document = {
            "texts": [
                {"self_ref": "#/texts/49", "text": first, "orig": first},
                {"self_ref": "#/texts/67", "text": batch, "orig": batch},
                {"self_ref": "#/texts/77", "text": final, "orig": final},
            ]
        }

        with patch.object(
            preprocess,
            "_load_docling_text_repair_decisions",
            return_value=self._study_218_molarity_decisions(),
        ):
            applied_count = preprocess._apply_approved_docling_text_repairs(
                document,
                study="study_218",
                source_file="main_paper",
            )

        self.assertEqual(applied_count, 5)
        self.assertEqual(
            document["texts"][0]["text"],
            "Each PFAS solution was made as a master batch at 241.5 μM.",
        )
        self.assertEqual(
            document["texts"][1]["text"],
            "Initial tests used PFOA (241.5 μM); subsequent solutions were "
            "prepared to 241.5 μM in water. The blank sample was equimolar "
            "(120.8 μM).",
        )
        self.assertEqual(document["texts"][2]["text"], "All PFAS were dosed at 241.5 μM.")
        self.assertEqual(document["texts"][1]["orig"], batch)

    def test_approved_docling_text_repairs_fail_if_the_source_changes(self):
        document = {
            "texts": [
                {"self_ref": "#/texts/49", "text": "241.5 mM"},
                {"self_ref": "#/texts/67", "text": "241.5 m M"},
                {"self_ref": "#/texts/77", "text": "241.5 m M"},
            ]
        }

        with (
            patch.object(
                preprocess,
                "_load_docling_text_repair_decisions",
                return_value=self._study_218_molarity_decisions(),
            ),
            self.assertRaisesRegex(ValueError, "match count changed"),
        ):
            preprocess._apply_approved_docling_text_repairs(
                document,
                study="study_218",
                source_file="main_paper",
            )

    def test_approved_docling_table_repairs_update_only_table_cell_text(self):
        first = "C s m g mg /C0 1"
        second = "v 0 m g mg /C0 1 h /C0 1"
        document = {
            "texts": [],
            "tables": [
                {
                    "self_ref": "#/tables/1",
                    "data": {
                        "grid": [
                            [
                                {"text": first, "orig": first},
                                {"text": second, "orig": second},
                            ]
                        ]
                    },
                }
            ],
        }
        decision = preprocess.DoclingTextRepairDecision(
            study="synthetic",
            source_file="main_paper",
            object_ref="#/tables/1",
            match_pattern=r"\b(C\s+s|v\s+0)\s+m\s+g\b",
            replacement=r"\1 μg",
            expected_matches=2,
            row_number=2,
        )

        with patch.object(
            preprocess,
            "_load_docling_text_repair_decisions",
            return_value=[decision],
        ):
            applied_count = preprocess._apply_approved_docling_text_repairs(
                document,
                study="synthetic",
                source_file="main_paper",
            )

        cells = document["tables"][0]["data"]["grid"][0]
        self.assertEqual(applied_count, 2)
        self.assertEqual(cells[0]["text"], "C s μg mg /C0 1")
        self.assertEqual(cells[1]["text"], "v 0 μg mg /C0 1 h /C0 1")
        self.assertEqual(cells[0]["orig"], first)
        self.assertEqual(cells[1]["orig"], second)

    def test_conclusions_does_not_hide_late_experimental_section(self):
        texts = [
            "■ RESULTS AND DISCUSSION",
            "result text",
            "■ CONCLUSIONS",
            "conclusion text that must stay out of cleaned records",
            "■ EXPERIMENTAL SECTION",
            "method text that must be retained",
            "■ REFERENCES",
            "reference text",
        ]
        document = {
            "body": {
                "children": [
                    {"$ref": f"#/texts/{index}"}
                    for index in range(len(texts))
                ]
            },
            "texts": [
                {
                    "self_ref": f"#/texts/{index}",
                    "label": "section_header" if text.startswith("■") else "text",
                    "text": text,
                    "children": [],
                    "content_layer": "body",
                }
                for index, text in enumerate(texts)
            ],
            "groups": [],
            "tables": [],
            "pictures": [],
        }
        records = []

        with (
            patch.object(preprocess, "_load_json", return_value=document),
            patch.object(
                preprocess,
                "_resolve_structured_captions_with_cache",
                return_value={},
            ),
            patch.object(
                preprocess,
                "_table_continuation_output_groups",
                return_value=[],
            ),
        ):
            preprocess._process_source(
                study="synthetic",
                study_dir=Path.cwd(),
                source_file="main_paper",
                json_path=Path("unused.json"),
                chunk_records=records,
                object_records=[],
                suspected_rows=[],
                figtab_rows=[],
            )

        main_records = [
            record for record in records if record.get("file_type") == "main_paper"
        ]
        retained_refs = {
            ref
            for record in main_records
            for ref in record.get("object_refs", [])
        }

        self.assertIn("#/texts/5", retained_refs)
        self.assertNotIn("#/texts/3", retained_refs)
        self.assertFalse(
            any(
                "conclusion text" in str(record.get("text"))
                for record in main_records
            )
        )


if __name__ == "__main__":
    unittest.main()
