from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.logkd_data import apply_data_mode, coerce_numeric  # noqa: E402


class DataModeTests(unittest.TestCase):
    def test_drop_unreliable_includes_physical_apparent_removal_flags(self) -> None:
        frame = pd.DataFrame(
            {
                "logKd_conversion_spike_flag": [False, False, pd.NA, False, False],
                "logKd_conversion_dip_flag": [False, False, pd.NA, False, False],
                "low_Ce_detection_limit_flag": [False, False, False, False, False],
                "small_concentration_difference_flag": [False, False, False, True, False],
                "low_removal_rate_flag": [False, True, False, False, False],
                "high_apparent_removal_flag": [False, False, True, False, False],
                "human_review_unreliable_flag": [False, False, False, False, True],
            }
        )

        kept, info = apply_data_mode(frame, "drop_unreliable")

        self.assertEqual(kept.index.tolist(), [0])
        self.assertEqual(len(kept), 1)
        self.assertEqual(info["rows_removed_small_concentration_difference"], 1)
        self.assertEqual(info["rows_removed_low_removal_rate"], 1)
        self.assertEqual(info["rows_removed_high_apparent_removal"], 1)
        self.assertEqual(info["rows_removed_unreliable"], 4)


if __name__ == "__main__":
    unittest.main()


class CoerceNumericDtypeTests(unittest.TestCase):
    """A coerced value must support arithmetic regardless of the source dtype.

    pandas keeps an all-True/False column as bool, which then rejects the
    quantile and difference arithmetic that feature screening performs on
    numeric inputs.
    """

    def test_boolean_columns_coerce_to_float(self) -> None:
        for values in ([True, False, True], [True, False, None], [1, 0, 1], [1.5, 2.5, None]):
            with self.subTest(values=values):
                coerced = coerce_numeric(pd.Series(values))
                self.assertEqual(coerced.dtype, np.dtype("float64"))
                self.assertFalse(pd.isna(coerced.quantile(0.75) - coerced.quantile(0.25)))

    def test_unparseable_and_empty_inputs_stay_float(self) -> None:
        self.assertEqual(coerce_numeric(pd.Series(["a", "b"])).dtype, np.dtype("float64"))
        self.assertEqual(coerce_numeric(pd.Series([], dtype=object)).dtype, np.dtype("float64"))
