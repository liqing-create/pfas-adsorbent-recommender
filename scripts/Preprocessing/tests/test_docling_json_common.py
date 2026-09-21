from __future__ import annotations

import unittest

from pathlib import Path
import sys

PREPROCESSING_DIR = Path(__file__).resolve().parents[1]
if str(PREPROCESSING_DIR) not in sys.path:
    sys.path.insert(0, str(PREPROCESSING_DIR))

from helpers.docling_json_common import iter_body_refs, table_markdown


def ref(value: str) -> dict[str, str]:
    return {"$ref": value}


class DoclingJsonCommonTests(unittest.TestCase):
    def test_iter_body_refs_descends_into_text_children(self):
        doc = {
            "body": {"children": [ref("#/texts/0")]},
            "groups": [],
            "texts": [
                {
                    "text": "Results",
                    "children": [ref("#/texts/1"), ref("#/pictures/0")],
                },
                {"text": "Figure 1. Nested caption.", "children": []},
            ],
            "tables": [],
            "pictures": [{"children": []}],
        }

        self.assertEqual(
            list(iter_body_refs(doc)),
            ["#/texts/0", "#/texts/1", "#/pictures/0"],
        )

    def test_table_markdown_uses_grid(self):
        doc = {}
        table = {
            "data": {
                "grid": [
                    [{"text": "Analyte"}, {"text": "Value"}],
                    [{"text": "PFOA"}, {"text": "10"}],
                ],
                "table_cells": [],
            }
        }

        self.assertEqual(
            table_markdown(table, doc),
            "| Analyte | Value |\n| --- | --- |\n| PFOA | 10 |",
        )


if __name__ == "__main__":
    unittest.main()
