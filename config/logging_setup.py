"""
config/logging_setup.py

Centralized structured (JSON) logging configuration for the sentiment
analysis pipeline. Import `get_logger(__name__)` from any module to get
a logger that emits single-line JSON records to stdout.
"""

import json
import logging
import sys
from datetime import datetime, timezone


class JsonFormatter(logging.Formatter):
    """Formats log records as single-line JSON objects."""

    def format(self, record: logging.LogRecord) -> str:
        log_record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
            "message": record.getMessage(),
        }

        if record.exc_info:
            log_record["exception"] = self.formatException(record.exc_info)

        # Callers can attach arbitrary structured context via:
        #   logger.info("msg", extra={"extra_fields": {"movie_id": "tt123"}})
        extra_fields = getattr(record, "extra_fields", None)
        if extra_fields:
            log_record.update(extra_fields)

        return json.dumps(log_record, default=str)


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """
    Returns a configured logger that emits structured JSON logs to stdout.

    Safe to call multiple times with the same name; handlers are only
    attached once per logger to avoid duplicate log lines.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(JsonFormatter())
        logger.addHandler(handler)
        logger.propagate = False

    return logger
