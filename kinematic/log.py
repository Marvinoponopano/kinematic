"""Structured logging for the pipeline (Phase 9).

One JSON object per line, so a session's log can be filtered, diffed or plotted without parsing
prose::

    {"time": 1760272471.502, "level": "info", "logger": "kinematic.runtime",
     "event": "click", "kind": "left", "source": "recognizer", "emitted": 4}

Nothing is written unless you ask for it. The default level is ``warning`` and the pipeline speaks at
``info``/``debug``, so an ordinary run prints its banner and nothing else. The hot path is guarded by
``logger.isEnabledFor(logging.DEBUG)``, so a quiet configuration costs one attribute check per call
rather than a formatting pass per frame — this project's loop is camera-bound and logging must not
become the second thing it waits on.

The event names are the API: ``loop_start``, ``loop_stop``, ``pause``, ``resume``, ``tracking_lost``,
``tracking_restored``, ``click``, ``scroll``, ``sensitivity``, ``gesture_state``, ``frame``,
``diagnostics``, ``summary``.
"""

import json
import logging
import os
import sys
from typing import Any, Dict, Optional

#: Every logger in the package hangs off this name, so one ``configure()`` call covers the app.
LOGGER_NAME = "kinematic"

#: Acceptable level names for the CLI and the config field. An unknown name raises rather than
#: silently selecting the root default (which would look like the logging was broken).
LEVELS = ("debug", "info", "warning", "error", "critical")

# A library should not print anything it was not asked to print — including at error level, which is
# what Python falls back to `logging.lastResort` for. `configure()` replaces this handler with the real
# one, and the CLI always calls it, so an app run still reports its errors.
logging.getLogger(LOGGER_NAME).addHandler(logging.NullHandler())


class JsonLineFormatter(logging.Formatter):
    """Render one record as one JSON object: ``time``, ``level``, ``logger``, ``event``, fields.

    Anything passed as ``extra={"fields": {...}}`` is merged into the object, so a consumer can key
    off the named event and read its values without a regular expression.
    """

    def format(self, record: logging.LogRecord) -> str:
        payload: Dict[str, Any] = {
            "time": round(record.created, 3),
            "level": record.levelname.lower(),
            "logger": record.name,
            "event": getattr(record, "event", None) or record.getMessage(),
        }
        fields = getattr(record, "fields", None)
        if isinstance(fields, dict):
            payload.update(fields)
        if record.exc_info:
            payload["error"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


def level_number(level: str) -> int:
    """``"info"`` → ``logging.INFO``. Raises on an unknown name instead of logging nothing."""
    name = str(level).strip().lower()
    if name not in LEVELS:
        raise ValueError(f"unknown log level {level!r}; use one of {', '.join(LEVELS)}")
    return int(getattr(logging, name.upper()))


def get_logger(name: str = "") -> logging.Logger:
    """The logger for one part of the package (``get_logger("runtime")`` → ``kinematic.runtime``)."""
    return logging.getLogger(f"{LOGGER_NAME}.{name}" if name else LOGGER_NAME)


def log_event(logger: logging.Logger, event: str, level: str = "info", **fields: Any) -> None:
    """Write one named event with structured fields; a no-op when *level* is not enabled."""
    numeric = level_number(level)
    if not logger.isEnabledFor(numeric):
        return
    logger.log(numeric, event, extra={"event": event, "fields": fields})


def configure(level: str = "warning", path: Optional[str] = None, stream: Any = None) -> logging.Logger:
    """Point the package logger at a stream or a file at *level*, and return it.

    ``path`` appends JSON lines to that file (its directory is created if missing); otherwise lines go
    to ``stream``, or to stderr. Calling this again *replaces* the handlers rather than stacking them,
    and the package logger does not propagate, so a host application's logging configuration cannot
    duplicate or reformat these lines.
    """
    logger = get_logger()
    for existing in list(logger.handlers):
        logger.removeHandler(existing)
        existing.close()

    numeric = level_number(level)
    logger.setLevel(numeric)
    handler: logging.Handler
    if path:
        directory = os.path.dirname(os.path.abspath(path))
        if directory:
            os.makedirs(directory, exist_ok=True)
        handler = logging.FileHandler(path, encoding="utf-8")
    else:
        handler = logging.StreamHandler(sys.stderr if stream is None else stream)
    handler.setLevel(numeric)
    handler.setFormatter(JsonLineFormatter())
    logger.addHandler(handler)
    logger.propagate = False
    return logger
