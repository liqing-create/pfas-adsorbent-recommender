"""Atomic, live result snapshots for long-running training wrappers.

The comparison wrappers may take hours to finish a batch.  This module lets
them publish the rows that have already completed without exposing a partially
written CSV to a user who opens it mid-run.
"""

from __future__ import annotations

import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

import pandas as pd


DEFAULT_RUN_SUMMARY_PATH = Path("summary") / "run_summary.csv"


def _atomic_write_text(path: Path, contents: str) -> None:
    """Replace *path* only after its new contents have been fully written."""

    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent, text=True
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as stream:
            stream.write(contents)
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def _as_frame(rows: pd.DataFrame | Iterable[Mapping[str, Any]]) -> pd.DataFrame:
    return rows.copy() if isinstance(rows, pd.DataFrame) else pd.DataFrame(rows)


def _status_path(summary_path: Path) -> Path:
    return summary_path.with_name(f"{summary_path.stem}_status.json")


def write_live_run_summary(
    batch_dir: Path,
    rows: pd.DataFrame | Iterable[Mapping[str, Any]],
    *,
    relative_path: Path = DEFAULT_RUN_SUMMARY_PATH,
) -> bool:
    """Publish a current, atomically-replaced snapshot of completed run rows.

    The companion ``run_summary_status.json`` identifies the CSV as a running,
    provisional snapshot.  A reporting problem must not interrupt expensive
    model fitting, so an I/O failure is reported and the wrapper continues.
    """

    summary_path = batch_dir / relative_path
    frame = _as_frame(rows)
    try:
        # An entirely empty DataFrame has no CSV header and pandas cannot read
        # its output back. Publish the running status first; the CSV appears
        # with the first completed (or failed) training row.
        if len(frame.columns):
            _atomic_write_text(summary_path, frame.to_csv(index=False))
        _atomic_write_text(
            _status_path(summary_path),
            json.dumps(
                {
                    "state": "running",
                    "updated_at_utc": datetime.now(timezone.utc).isoformat(),
                    "rows_written": int(len(frame)),
                    "note": (
                        "This is a live snapshot. Completed rows are available now; "
                        "paired and aggregate summaries are finalized after the batch ends."
                    ),
                },
                indent=2,
            )
            + "\n",
        )
    except OSError as error:
        print(
            f"WARNING: could not update live run summary {summary_path}: {error}",
            flush=True,
        )
        return False
    return True


def finalize_run_summary(
    batch_dir: Path,
    *,
    relative_path: Path = DEFAULT_RUN_SUMMARY_PATH,
) -> bool:
    """Mark a wrapper's run summary as final after its normal final export."""

    summary_path = batch_dir / relative_path
    try:
        _atomic_write_text(
            _status_path(summary_path),
            json.dumps(
                {
                    "state": "finalized",
                    "updated_at_utc": datetime.now(timezone.utc).isoformat(),
                    "note": "The wrapper completed its final run-summary export.",
                },
                indent=2,
            )
            + "\n",
        )
    except OSError as error:
        print(
            f"WARNING: could not finalize live run-summary status {summary_path}: {error}",
            flush=True,
        )
        return False
    return True
