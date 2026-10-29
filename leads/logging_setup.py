"""Structured logging with run/agent context.

Every log line is JSON on disk (easy to grep / ship) and human-readable on the
console. Both carry ``run_id`` and ``agent`` so a single lead can be traced end
to end.
"""

from __future__ import annotations

import json
import logging
import sys
from contextvars import ContextVar
from pathlib import Path

run_id_var: ContextVar[str] = ContextVar("run_id", default="-")
agent_var: ContextVar[str] = ContextVar("agent", default="-")

_RESERVED = {
    "name", "msg", "args", "levelname", "levelno", "pathname", "filename",
    "module", "exc_info", "exc_text", "stack_info", "lineno", "funcName",
    "created", "msecs", "relativeCreated", "thread", "threadName",
    "processName", "process", "taskName", "message", "asctime",
}


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, object] = {
            "ts": self.formatTime(record, "%Y-%m-%dT%H:%M:%S%z"),
            "level": record.levelname,
            "logger": record.name,
            "run_id": getattr(record, "run_id", run_id_var.get()),
            "agent": getattr(record, "agent", agent_var.get()),
            "message": record.getMessage(),
        }
        for k, v in record.__dict__.items():
            if k not in _RESERVED and not k.startswith("_"):
                payload[k] = v
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str, ensure_ascii=False)


class ConsoleFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        agent = getattr(record, "agent", agent_var.get())
        prefix = getattr(record, "prefix", "") or agent
        stamp = self.formatTime(record, "%H:%M:%S")
        base = f"{stamp} {record.levelname:<7} [{prefix}] {record.getMessage()}"
        if record.exc_info:
            base += "\n" + self.formatException(record.exc_info)
        return base


def configure_logging(level: str = "INFO", log_dir: Path | None = None) -> None:
    """Idempotently configure root logging."""
    root = logging.getLogger()
    if getattr(root, "_leads_configured", False):
        root.setLevel(level.upper())
        return

    root.setLevel(level.upper())
    root.handlers.clear()

    console = logging.StreamHandler(sys.stderr)
    console.setFormatter(ConsoleFormatter())
    console.setLevel(level.upper())
    root.addHandler(console)

    if log_dir is not None:
        log_dir.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_dir / "leads.jsonl", encoding="utf-8")
        fh.setFormatter(JsonFormatter())
        fh.setLevel("DEBUG")
        root.addHandler(fh)

    # Third-party noise control.
    for noisy in ("urllib3", "requests", "charset_normalizer"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    root._leads_configured = True  # type: ignore[attr-defined]


class AgentLogger:
    """Thin adapter that stamps agent + run context on every record."""

    def __init__(self, name: str) -> None:
        self._log = logging.getLogger(name)
        self.name = name

    def _kw(self, **extra: object) -> dict[str, object]:
        base = {"run_id": run_id_var.get(), "agent": agent_var.get() or self.name}
        base.update(extra)
        return base

    def debug(self, msg: str, **extra: object) -> None:
        self._log.debug(msg, extra=self._kw(**extra))

    def info(self, msg: str, **extra: object) -> None:
        self._log.info(msg, extra=self._kw(**extra))

    def warning(self, msg: str, **extra: object) -> None:
        self._log.warning(msg, extra=self._kw(**extra))

    def error(self, msg: str, **extra: object) -> None:
        self._log.error(msg, extra=self._kw(**extra))

    def exception(self, msg: str, **extra: object) -> None:
        self._log.exception(msg, extra=self._kw(**extra))
