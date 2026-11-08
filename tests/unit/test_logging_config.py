"""Tests for console logging setup."""

import logging

import pytest

from app.core.logging_config import configure_logging


def test_logging_setup_is_idempotent_and_sets_level() -> None:
    root_logger = logging.getLogger()

    configure_logging("DEBUG")
    configure_logging("DEBUG")

    handlers = [
        handler
        for handler in root_logger.handlers
        if handler.get_name() == "llm-inference-performance-lab-console"
    ]
    assert root_logger.level == logging.DEBUG
    assert len(handlers) == 1
    assert handlers[0].formatter is not None
    assert "%(asctime)s" in handlers[0].formatter._fmt
    assert "%(name)s" in handlers[0].formatter._fmt


def test_logging_rejects_unknown_level() -> None:
    with pytest.raises(ValueError, match="Unsupported logging level"):
        configure_logging("TRACE")
