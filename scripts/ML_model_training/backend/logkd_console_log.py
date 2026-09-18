"""Durable console logging for child training processes.

Batch outputs live on Box/OneDrive, whose sync filter can invalidate an open
handle part way through a long write stream. That surfaces as ``OSError``
(EINVAL/EBUSY/EACCES) on ``write``, ``flush``, or ``close``. The console log is
a diagnostic artifact, not a result, so a failed write must degrade into a
recorded note rather than abort a multi-hour comparison that is otherwise fine.

Flushing after every line multiplies the exposure to that filter and buys no
durability the periodic flush does not already provide, so writes are flushed
on an interval instead.
"""

from __future__ import annotations

from pathlib import Path
import time
from types import TracebackType
from typing import TextIO


FLUSH_INTERVAL_SECONDS = 2.0


class ConsoleLogSink:
    """Append child-process output to a file, tolerating cloud-sync failures."""

    def __init__(
        self,
        path: Path,
        *,
        flush_interval_seconds: float = FLUSH_INTERVAL_SECONDS,
    ) -> None:
        self.path = Path(path)
        self._flush_interval = flush_interval_seconds
        self._handle: TextIO | None = None
        self._last_flush = 0.0
        self.error: str | None = None

    @property
    def degraded(self) -> bool:
        """True when output stopped reaching the file part way through."""

        return self.error is not None

    def __enter__(self) -> "ConsoleLogSink":
        self._last_flush = time.monotonic()
        try:
            self._handle = self.path.open("w", encoding="utf-8", errors="replace")
        except OSError as error:
            self._record(error)
        return self

    def write(self, line: str) -> None:
        handle = self._handle
        if handle is None:
            return
        try:
            handle.write(line)
            now = time.monotonic()
            if now - self._last_flush >= self._flush_interval:
                handle.flush()
                self._last_flush = now
        except OSError as error:
            self._handle = None
            self._record(error)
            self._close(handle)

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        handle, self._handle = self._handle, None
        if handle is None:
            return
        try:
            handle.flush()
        except OSError as error:
            self._record(error)
        self._close(handle)

    def _record(self, error: OSError) -> None:
        # Keep the first failure: later ones are consequences of the same blip.
        if self.error is None:
            self.error = f"console log {self.path.name} incomplete ({error})"

    @staticmethod
    def _close(handle: TextIO) -> None:
        try:
            handle.close()
        except OSError:
            pass
