"""Tests for the live run-summary snapshots used by long wrapper runs."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parents[1]
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from backend.logkd_progress import finalize_run_summary, write_live_run_summary  # noqa: E402


class LiveRunSummaryTests(unittest.TestCase):
    def test_snapshot_is_readable_and_status_tracks_finalization(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            batch_dir = Path(temporary_directory)
            self.assertTrue(write_live_run_summary(batch_dir, []))
            summary_path = batch_dir / "summary" / "run_summary.csv"
            status_path = batch_dir / "summary" / "run_summary_status.json"
            self.assertFalse(summary_path.exists())
            self.assertEqual(json.loads(status_path.read_text(encoding="utf-8"))["rows_written"], 0)

            self.assertTrue(
                write_live_run_summary(
                    batch_dir,
                    [{"run_id": "first", "status": "completed", "mae": 0.12}],
                )
            )
            self.assertEqual(pd.read_csv(summary_path).to_dict("records"), [{"run_id": "first", "status": "completed", "mae": 0.12}])

            self.assertTrue(
                write_live_run_summary(
                    batch_dir,
                    [
                        {"run_id": "first", "status": "completed", "mae": 0.12},
                        {"run_id": "second", "status": "completed", "mae": 0.09},
                    ],
                )
            )
            self.assertEqual(len(pd.read_csv(summary_path)), 2)
            self.assertTrue(finalize_run_summary(batch_dir))
            self.assertEqual(json.loads(status_path.read_text(encoding="utf-8"))["state"], "finalized")


if __name__ == "__main__":
    unittest.main()
