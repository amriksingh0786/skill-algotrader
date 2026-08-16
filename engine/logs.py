"""
Structured logging.

NUANCE #21/#22: three separate streams, because they answer different questions
and mixing them makes all three useless at 14:55 when something is wrong.

    logs/operational.jsonl  what the bot did      (entries, exits, orders)
    logs/debug.jsonl        why it did it         (rejected signals, gates)
    logs/errors.jsonl       what broke            (exceptions, rejections)
    logs/trades.jsonl       closed trades         (the analytics input)

JSON Lines throughout: greppable with `jq`, and loadable straight into Polars
for post-trade analysis without a parser.
"""

from __future__ import annotations

import json
import logging
import sys
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any

from .session import now_ist


class JsonFormatter(logging.Formatter):
    """One JSON object per line, with any extra fields merged in."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": datetime.fromtimestamp(record.created).isoformat(timespec="milliseconds"),
            "level": record.levelname,
            "event": record.getMessage(),
        }

        extra = getattr(record, "extra_fields", None)
        if extra:
            payload.update(extra)

        if record.exc_info:
            payload["exception"] = "".join(traceback.format_exception(*record.exc_info)).strip()

        return json.dumps(payload, default=str)


class ConsoleFormatter(logging.Formatter):
    """Human-readable console output. Terse — the JSON files hold the detail."""

    COLORS = {"DEBUG": "\033[90m", "INFO": "\033[0m", "WARNING": "\033[93m",
              "ERROR": "\033[91m", "CRITICAL": "\033[95m"}
    RESET = "\033[0m"

    def format(self, record: logging.LogRecord) -> str:
        color = self.COLORS.get(record.levelname, "")
        stamp = datetime.fromtimestamp(record.created).strftime("%H:%M:%S")
        message = record.getMessage()

        extra = getattr(record, "extra_fields", None) or {}
        detail = " ".join(
            f"{k}={v}" for k, v in extra.items()
            if k in ("symbol", "quantity", "price", "pnl", "reason", "order_id")
        )

        return f"{color}{stamp} {message}{(' | ' + detail) if detail else ''}{self.RESET}"


class TradingLogger:
    """
    Facade over the four streams.

    Usage:
        log = TradingLogger("logs")
        log.info("entry", symbol="RELIANCE", quantity=57, price=1847.35)
        log.debug("signal rejected", symbol="TCS", reason="cooldown")
        log.error("order failed", symbol="INFY", exc_info=True)
        log.trade({...})
    """

    def __init__(self, log_dir: str | Path = "logs", *, console: bool = True,
                 debug_to_console: bool = False) -> None:
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.trades_path = self.log_dir / "trades.jsonl"
        self.signals_path = self.log_dir / "signals.jsonl"

        self._logger = logging.getLogger("algotrader")
        self._logger.setLevel(logging.DEBUG)
        self._logger.handlers.clear()
        self._logger.propagate = False

        operational = logging.FileHandler(self.log_dir / "operational.jsonl")
        operational.setLevel(logging.INFO)
        operational.setFormatter(JsonFormatter())
        self._logger.addHandler(operational)

        debug = logging.FileHandler(self.log_dir / "debug.jsonl")
        debug.setLevel(logging.DEBUG)
        debug.setFormatter(JsonFormatter())
        self._logger.addHandler(debug)

        errors = logging.FileHandler(self.log_dir / "errors.jsonl")
        errors.setLevel(logging.ERROR)
        errors.setFormatter(JsonFormatter())
        self._logger.addHandler(errors)

        if console:
            stream = logging.StreamHandler(sys.stdout)
            stream.setLevel(logging.DEBUG if debug_to_console else logging.INFO)
            stream.setFormatter(ConsoleFormatter())
            self._logger.addHandler(stream)

    def _log(self, level: int, event: str, exc_info: bool = False, **fields: Any) -> None:
        self._logger.log(level, event, exc_info=exc_info, extra={"extra_fields": fields})

    def debug(self, event: str, **fields: Any) -> None:
        self._log(logging.DEBUG, event, **fields)

    def info(self, event: str, **fields: Any) -> None:
        self._log(logging.INFO, event, **fields)

    def warning(self, event: str, **fields: Any) -> None:
        self._log(logging.WARNING, event, **fields)

    def error(self, event: str, exc_info: bool = False, **fields: Any) -> None:
        self._log(logging.ERROR, event, exc_info=exc_info, **fields)

    def critical(self, event: str, exc_info: bool = False, **fields: Any) -> None:
        self._log(logging.CRITICAL, event, exc_info=exc_info, **fields)

    def trade(self, record: dict[str, Any]) -> None:
        """
        Append a closed trade. This file is the input to post-trade analytics, so
        it is written directly rather than through the logging module — the
        schema must stay stable even if log formatting changes.
        """
        payload = {"logged_at": now_ist().isoformat(timespec="seconds"), **record}
        with self.trades_path.open("a") as handle:
            handle.write(json.dumps(payload, default=str) + "\n")

    def signal(self, record: dict[str, Any], *, taken: bool, reason: str = "") -> None:
        """
        Append every signal, taken or not.

        Logging rejected signals is what makes attribution possible later: "the
        strategy stopped working" and "the risk limits blocked everything" look
        identical in a P&L curve and completely different here.
        """
        payload = {
            "logged_at": now_ist().isoformat(timespec="seconds"),
            "taken": taken,
            "rejection_reason": reason,
            **record,
        }
        with self.signals_path.open("a") as handle:
            handle.write(json.dumps(payload, default=str) + "\n")
