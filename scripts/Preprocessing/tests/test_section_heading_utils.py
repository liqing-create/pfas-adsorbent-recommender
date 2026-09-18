from __future__ import annotations

import sys
import unittest
from pathlib import Path


PREPROCESSING_DIR = Path(__file__).resolve().parents[1]
if str(PREPROCESSING_DIR) not in sys.path:
    sys.path.insert(0, str(PREPROCESSING_DIR))

from helpers.section_heading_utils import (  # noqa: E402
    METHODS_LABEL,
    RESULTS_LABEL,
    classify_heading,
    normalize_heading,
)


class SectionHeadingUtilsTests(unittest.TestCase):
    def test_space_separated_major_section_number_is_removed(self):
        self.assertEqual(normalize_heading("2    Experimental"), "experimental")
        self.assertEqual(
            normalize_heading("3 Results and Discussion"),
            "results and discussion",
        )

    def test_space_separated_numbered_target_headings_are_classified(self):
        cases = {
            "2 Experimental": METHODS_LABEL,
            "2 Materials and Methods": METHODS_LABEL,
            "2 Materials and methods": METHODS_LABEL,
            "3 Results and Discussion": RESULTS_LABEL,
            "3 Results and discussion": RESULTS_LABEL,
        }
        for heading, expected in cases.items():
            with self.subTest(heading=heading):
                self.assertEqual(classify_heading(heading), expected)

    def test_existing_punctuation_numbering_remains_supported(self):
        self.assertEqual(classify_heading("2. Materials and Methods"), METHODS_LABEL)
        self.assertEqual(classify_heading("3) Results"), RESULTS_LABEL)
        self.assertEqual(classify_heading("2.1. Experimental methods"), METHODS_LABEL)


if __name__ == "__main__":
    unittest.main()
