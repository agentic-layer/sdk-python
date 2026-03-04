from __future__ import annotations

import inspect
import json
import logging
import os
import sys
import traceback
from logging import Logger
from typing import Callable, NotRequired, TextIO, TypedDict, Union

import loguru
from loguru import logger


class JsonRecord(TypedDict):
    timestamp: str
    name: str | None
    level: str
    message: str
    function: str
    module: str
    line: int
    exception: NotRequired[dict[str, str]]


def _serialize(record: loguru.Record) -> str:
    log: JsonRecord = {
        "timestamp": record["time"].isoformat(),
        "name": record["name"],
        "level": record["level"].name,
        "message": record["message"],
        "function": record["function"],
        "module": record["module"],
        "line": record["line"],
    }

    if record["exception"] is not None:
        log["exception"] = {
            "stack": "".join(
                traceback.format_exception(
                    record["exception"].type,
                    record["exception"].value,
                    record["exception"].traceback,
                )
            ),
            "kind": getattr(record["exception"].type, "__name__", "None"),
            "message": str(record["exception"].value),
        }

    return json.dumps(log)


def _json_sink(message: loguru.Message) -> None:
    serialized = _serialize(message.record)
    sys.stderr.write(serialized + "\n")


def _configure_loguru() -> None:
    sink: Union[TextIO, Callable[[loguru.Message], None]]
    log_format = os.environ.get("LOG_FORMAT", "Text")
    if log_format == "JSON":
        sink = _json_sink
    else:
        sink = sys.stderr

    log_level = os.environ.get("LOGLEVEL", "INFO")

    logger.remove()
    logger.add(
        sink,
        # Log INFO by default
        filter={
            "": log_level,
            # Reduce verbosity of some noisy loggers
            "a2a.utils.telemetry": "INFO",
        },
    )


def setup_logging() -> None:
    """Initializes the application so that logging is handled by loguru"""

    _configure_loguru()

    # Some libraries we use log to standard logging and not to loguru. To also get the logs from these frameworks, we
    # add a handler to the root logger of standard logging, that converts the log entries to loguru. This way
    # loguru has the final say regarding logging, and we don't get a mixture of both logging frameworks
    logging.basicConfig(handlers=[InterceptHandler()], level=logging.NOTSET, force=True)

    # We have to replace the existing handlers from the loggers we want to intercept as well.
    for _log, _logger in logging.root.manager.loggerDict.items():
        # print("Checking logger: ", _log, " ", _logger)
        if not isinstance(_logger, Logger) or len(_logger.handlers) == 0:
            # logger not yet created or no custom handlers. Skipping
            continue
        for handler in _logger.handlers:
            if not isinstance(handler, logging.StreamHandler):
                # We only replace stream handlers, which write to the console
                # NullHandlers and other handlers are not replaced
                continue
            _logger.removeHandler(handler)
            _logger.addHandler(InterceptHandler())

            # Prevent duplicate logs
            if _logger.propagate:
                logger.debug("Disable propagate for logger {}", _log)
                _logger.propagate = False


class InterceptHandler(logging.Handler):
    """
    A Handler for the standard python logging that sends all incoming logs to loguru. Taken from the loguru documentation
    https://loguru.readthedocs.io/en/stable/overview.html
    """

    def emit(self, record: logging.LogRecord) -> None:
        # Get corresponding Loguru level if it exists.
        level: str | int
        try:
            level = logger.level(record.levelname).name
        except ValueError:
            level = record.levelno

        # Find caller from where originated the logged message.
        frame, depth = inspect.currentframe(), 0
        while frame and (depth == 0 or frame.f_code.co_filename == logging.__file__):
            frame = frame.f_back
            depth += 1

        logger.opt(depth=depth, exception=record.exc_info).log(level, record.getMessage())
