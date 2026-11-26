"""
Unit tests for Centralised Logging module.
Verifies logger initialization, severity levels, file output,
prevention of duplicate handlers, and sensitive data redaction.
"""

import logging
from pathlib import Path
import tempfile
import pytest

from src.utils.logger import setup_logging, get_logger, SensitiveDataFilter


class TestCentralisedLogging:
    def test_logger_initialisation(self):
        """Tests that logger initializes properly with correct name and hierarchy."""
        logger = get_logger("test_module")
        assert logger.name == "raid_agent.test_module"
        assert isinstance(logger, logging.Logger)

    def test_severity_levels(self, tmp_path: Path):
        """Tests that messages at all severity levels can be logged."""
        test_log_file = tmp_path / "severity_test.log"
        logger = setup_logging(level="DEBUG", log_file=test_log_file, enable_console=False)

        child_logger = get_logger("severity_runner")
        child_logger.debug("Debug level verification")
        child_logger.info("Info level verification")
        child_logger.warning("Warning level verification")
        child_logger.error("Error level verification")
        child_logger.critical("Critical level verification")

        # Flush handlers
        for handler in logger.handlers:
            handler.flush()

        content = test_log_file.read_text(encoding="utf-8")
        assert "[DEBUG]" in content and "Debug level verification" in content
        assert "[INFO]" in content and "Info level verification" in content
        assert "[WARNING]" in content and "Warning level verification" in content
        assert "[ERROR]" in content and "Error level verification" in content
        assert "[CRITICAL]" in content and "Critical level verification" in content

    def test_file_logging(self, tmp_path: Path):
        """Tests that logs are written directly to the specified file with correct formatting."""
        log_file = tmp_path / "custom_run.log"
        setup_logging(level="INFO", log_file=log_file, enable_console=False)

        test_logger = get_logger("file_test")
        test_logger.info("Testing file persistence output")

        for handler in logging.getLogger("raid_agent").handlers:
            handler.flush()

        assert log_file.exists()
        log_text = log_file.read_text(encoding="utf-8")
        assert "Testing file persistence output" in log_text
        assert "[INFO]" in log_text
        assert "[raid_agent.file_test]" in log_text

    def test_no_duplicate_handlers(self, tmp_path: Path):
        """Tests that repeated calls to setup_logging do not multiply handlers."""
        log_file = tmp_path / "dup_test.log"
        logger1 = setup_logging(level="INFO", log_file=log_file, enable_console=True)
        handler_count_1 = len(logger1.handlers)

        logger2 = setup_logging(level="INFO", log_file=log_file, enable_console=True)
        handler_count_2 = len(logger2.handlers)

        assert handler_count_1 == handler_count_2

    def test_sensitive_data_filtering(self, tmp_path: Path):
        """Tests that sensitive keys, tokens, and passwords are automatically redacted in logs."""
        log_file = tmp_path / "sensitive_test.log"
        setup_logging(level="INFO", log_file=log_file, enable_console=False)

        logger = get_logger("security_test")
        logger.info("Connecting to API with api_key=AIzaSyA8394jklsdu92384")
        logger.info("Using auth header Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6")
        logger.info("DB connection with password='SuperSecretPassword123'")

        for handler in logging.getLogger("raid_agent").handlers:
            handler.flush()

        log_content = log_file.read_text(encoding="utf-8")
        assert "AIzaSyA8394jklsdu92384" not in log_content
        assert "Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6" not in log_content
        assert "SuperSecretPassword123" not in log_content
        assert "***REDACTED***" in log_content
