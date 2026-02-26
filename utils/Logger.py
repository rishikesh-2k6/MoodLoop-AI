"""
utils/Logger.py
───────────────
Lightweight CSV run-logger for MoodLoop AI.

Appends one row per pipeline run to a persistent CSV file.
Creates the file (with a header) automatically on first write.
Thread-safe via a threading.Lock so multiple workers can safely
share a single Logger instance.
"""

from __future__ import annotations

import csv
import logging
import threading
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

log = logging.getLogger(__name__)


# ── Default CSV schema ────────────────────────────────────────────── #
DEFAULT_FIELDS: list[str] = [
    "run_id",
    "timestamp",
    "theme_name",
    "mood",
    "quote",
    "title",
    "caption",
    "background_file",
    "music_file",
    "video_output",
    "model",
    "trending_topics",
    "render_success",
    "error_message",
]


class RunLogger:
    """
    Append-only CSV logger for pipeline run metadata.

    Each call to :meth:`log_run` writes exactly one row to the CSV.
    On first write the file is created and the header row is inserted.

    Parameters
    ----------
    csv_path : Path
        Destination CSV file (created if it does not exist).
    fields : list[str] | None
        Column names.  Defaults to :data:`DEFAULT_FIELDS` when ``None``.
    encoding : str
        File encoding (default: ``"utf-8"``).

    Examples
    --------
    >>> logger = RunLogger(Path("metadata.csv"))
    >>> logger.log_run(run_id="20240224_120000", theme_name="dark_aesthetic", ...)
    """

    def __init__(
        self,
        csv_path: Path,
        fields: Optional[list[str]] = None,
        encoding: str = "utf-8",
    ) -> None:
        self.csv_path = Path(csv_path)
        self.fields: list[str] = fields if fields is not None else list(DEFAULT_FIELDS)
        self.encoding = encoding
        self._lock = threading.Lock()

        # Create parent directories if they don't exist
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)
        log.debug("RunLogger initialised → %s", self.csv_path)

    # ------------------------------------------------------------------ #
    #  Public API                                                          #
    # ------------------------------------------------------------------ #

    def log_run(self, **kwargs: Any) -> None:
        """
        Append a single row to the CSV log.

        Any keyword argument whose name appears in :attr:`fields` is
        written to the corresponding column.  Unrecognised keys are
        silently ignored.  Missing keys are written as empty strings.

        A ``timestamp`` column is automatically populated with the
        current ISO-8601 datetime if not supplied by the caller.

        Parameters
        ----------
        **kwargs
            Arbitrary run metadata.  Common keys:

            - ``run_id``         – e.g. ``"20240224_120000"``
            - ``theme_name``     – theme identifier
            - ``mood``           – theme mood string
            - ``quote``          – generated quote text
            - ``title``          – generated title
            - ``caption``        – generated caption
            - ``background_file``– path to the background image used
            - ``music_file``     – path to the music track used
            - ``video_output``   – path to the rendered MP4
            - ``model``          – Ollama model tag
            - ``trending_topics``– semicolon-separated topics
            - ``render_success`` – ``"true"`` / ``"false"``
            - ``error_message``  – failure details when applicable

        Raises
        ------
        OSError
            If the CSV file cannot be opened for writing.
        """
        # Ensure timestamp is always set
        if "timestamp" not in kwargs:
            kwargs["timestamp"] = datetime.now().isoformat(timespec="seconds")

        # Build row dict aligned to self.fields; unknown keys silently dropped
        row: dict[str, str] = {
            field: str(kwargs.get(field, "")) for field in self.fields
        }

        with self._lock:
            write_header = not self.csv_path.exists() or self.csv_path.stat().st_size == 0
            try:
                with self.csv_path.open("a", newline="", encoding=self.encoding) as fh:
                    writer = csv.DictWriter(
                        fh,
                        fieldnames=self.fields,
                        extrasaction="ignore",
                        quoting=csv.QUOTE_ALL,
                    )
                    if write_header:
                        writer.writeheader()
                        log.debug("RunLogger: created CSV with header → %s", self.csv_path)
                    writer.writerow(row)
                    log.info(
                        "RunLogger: row written (run_id=%s) → %s",
                        kwargs.get("run_id", "?"),
                        self.csv_path,
                    )
            except OSError as exc:
                log.error("RunLogger: failed to write CSV row: %s", exc)
                raise

    def log_run_from_dict(self, data: dict[str, Any]) -> None:
        """
        Convenience wrapper: accept a plain dict instead of keyword args.

        Parameters
        ----------
        data : dict[str, Any]
            Same key-value pairs as :meth:`log_run` kwargs.
        """
        self.log_run(**data)

    # ------------------------------------------------------------------ #
    #  Introspection helpers                                               #
    # ------------------------------------------------------------------ #

    def row_count(self) -> int:
        """
        Return the number of data rows currently in the CSV (excludes header).

        Returns
        -------
        int
            Number of logged runs; ``0`` if the file does not yet exist.
        """
        if not self.csv_path.exists():
            return 0
        with self._lock, self.csv_path.open(encoding=self.encoding) as fh:
            return max(0, sum(1 for _ in fh) - 1)   # subtract header line

    def last_run_id(self) -> Optional[str]:
        """
        Return the ``run_id`` of the most recently logged run.

        Returns
        -------
        str | None
            Last ``run_id`` value, or ``None`` if the log is empty.
        """
        if not self.csv_path.exists():
            return None
        with self._lock, self.csv_path.open(encoding=self.encoding) as fh:
            reader = csv.DictReader(fh)
            last: Optional[str] = None
            for row in reader:
                last = row.get("run_id")
            return last

    def __repr__(self) -> str:
        return f"RunLogger(csv_path={self.csv_path!r}, fields={self.fields!r})"
