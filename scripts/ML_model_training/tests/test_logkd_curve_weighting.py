from __future__ import annotations

import sys
import unittest
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.logkd_data import equal_source_row_weights  # noqa: E402
from backend.logkd_metrics import metrics_dict  # noqa: E402


class CurveWeightingTests(unittest.TestCase):
    def test_each_source_row_has_equal_total_weight(self) -> None:
        rows = pd.DataFrame({"source_row_index": [10, 10, 20]})
        weights = equal_source_row_weights(rows)

        self.assertAlmostEqual(float(weights.mean()), 1.0)
        totals = weights.groupby(rows["source_row_index"]).sum()
        self.assertAlmostEqual(float(totals.loc[10]), float(totals.loc[20]))

    def test_weighted_metrics_do_not_overrepresent_long_curve(self) -> None:
        rows = pd.DataFrame({"source_row_index": [10, 10, 20]})
        weights = equal_source_row_weights(rows)
        metrics = metrics_dict([0.0, 10.0, 0.0], [0.0, 0.0, 0.0], weights)

        self.assertAlmostEqual(metrics["mae"], 2.5)

if __name__ == "__main__":
    unittest.main()
