"""
Centralised Logging Module for RAID Identification Agent.
Provides console and file logging with:
- Standardized formatting (timestamp, level, module, message)
- Support for DEBUG, INFO, WARNING, ERROR, CRITICAL
- Duplicate handler prevention
- Sensitive data redaction (API keys, passwords, bearer tokens)
"""

import logging
import re
import sys
from pathlib import Path
from typing import Optional

from config import LOGS_DIR, LOG_FILE_PATH, LOG_LEVEL


class SensitiveDataFilter(logging.Filter):
    """
    Redacts sensitive credentials, tokens, and keys from log records.
    """
    PATTERNS = [
        (re.compile(r"(?i)(api[_-]?key\s*[:=]\s*['\"]?)([\w\-\.]{6,})(['\"]?)"), r"\1***REDACTED***\3"),
        (re.compile(r"(?i)(bearer\s+)([\w\-\.]{8,})"), r"\1***REDACTED***"),
        (re.compile(r"(?i)(password\s*[:=]\s*['\"]?)([^'\"\s]{3,})(['\"]?)"), r"\1***REDACTED***\3"),
        (re.compile(r"(?i)(secret\s*[:=]\s*['\"]?)([\w\-\.]{6,})(['\"]?)"), r"\1***REDACTED***\3"),
        (re.compile(r"(?i)(token\s*[:=]\s*['\"]?)([\w\-\.]{6,})(['\"]?)"), r"\1***REDACTED***\3"),
    ]

    def filter(self, record: logging.LogRecord) -> bool:
        if isinstance(record.msg, str):
            for pattern, replacement in self.PATTERNS:
                record.msg = pattern.sub(replacement, record.msg)
        return True


_logging_configured = False
_current_log_file: Optional[Path] = None


def setup_logging(
    level: Optional[str] = None,
    log_file: Optional[Path] = None,
    enable_console: bool = True,
    force: bool = False
) -> logging.Logger:
    """
    Sets up the central logger hierarchy for the application.
    Ensures handlers are configured without duplicate attachments.
    """
    global _logging_configured, _current_log_file
    log_level_str = (level or LOG_LEVEL).upper()
    numeric_level = getattr(logging, log_level_str, logging.INFO)

    root_logger = logging.getLogger("raid_agent")
    root_logger.setLevel(numeric_level)

    target_file = log_file or LOG_FILE_PATH

    # If already configured and parameters haven't changed, return existing logger
    if _logging_configured and not force and (log_file is None or target_file == _current_log_file):
        root_logger.setLevel(numeric_level)
        return root_logger

    # Safely close and clear pre-existing handlers to prevent leaks and duplicates
    for handler in list(root_logger.handlers):
        try:
            handler.close()
        except Exception:
            pass
    root_logger.handlers.clear()

    formatter = logging.Formatter(
        fmt="%(asctime)s [%(levelname)s] [%(name)s]: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    sensitive_filter = SensitiveDataFilter()

    # 1. Console Handler
    if enable_console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(numeric_level)
        console_handler.setFormatter(formatter)
        console_handler.addFilter(sensitive_filter)
        root_logger.addHandler(console_handler)

    # 2. File Handler
    try:
        target_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(str(target_file), encoding="utf-8")
        file_handler.setLevel(numeric_level)
        file_handler.setFormatter(formatter)
        file_handler.addFilter(sensitive_filter)
        root_logger.addHandler(file_handler)
        _current_log_file = target_file
    except Exception as exc:
        sys.stderr.write(f"Failed to initialize file logger at {target_file}: {exc}\n")

    # Do not propagate to Python root logger to prevent third-party pollution
    root_logger.propagate = False
    _logging_configured = True

    return root_logger


def get_logger(module_name: str) -> logging.Logger:
    """
    Returns a child logger under the 'raid_agent' namespace.
    If logging has not been initialized yet, setup_logging() is automatically invoked.
    """
    global _logging_configured
    if not _logging_configured:
        setup_logging()

    # Normalize name under raid_agent hierarchy
    if module_name.startswith("raid_agent."):
        logger_name = module_name
    elif module_name == "raid_agent":
        logger_name = "raid_agent"
    else:
        logger_name = f"raid_agent.{module_name}"

    return logging.getLogger(logger_name)
