"""Tests for environment-backed application settings."""

from pathlib import Path

import pytest
from pydantic import ValidationError

from app.core.config import Settings


def test_settings_load_environment_variables(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LLM_LAB_ENVIRONMENT", "production")
    monkeypatch.setenv("LLM_LAB_LOG_LEVEL", "WARNING")
    monkeypatch.setenv("LLM_LAB_MODEL_CACHE_DIR", "~/models")

    settings = Settings(_env_file=None)

    assert settings.environment == "production"
    assert settings.log_level == "WARNING"
    assert settings.model_cache_dir == Path("~/models").expanduser()


def test_settings_reject_production_debug_mode() -> None:
    with pytest.raises(ValidationError, match="Debug mode must be disabled in production"):
        Settings(_env_file=None, environment="production", debug=True)


def test_settings_reject_invalid_log_level() -> None:
    with pytest.raises(ValidationError, match="log_level"):
        Settings(_env_file=None, log_level="TRACE")


def test_settings_reject_empty_directory() -> None:
    with pytest.raises(ValidationError, match="must not be empty"):
        Settings(_env_file=None, benchmark_output_dir=" ")
    with pytest.raises(ValidationError, match="must not be empty"):
        Settings(_env_file=None, benchmark_output_dir=None)


def test_default_model_cache_directory_expands_home() -> None:
    settings = Settings(_env_file=None)

    assert (
        settings.model_cache_dir
        == Path("~/.cache/llm-inference-performance-lab/models").expanduser()
    )


def test_dataset_sample_limit_must_be_positive() -> None:
    with pytest.raises(ValidationError, match="dataset_max_samples"):
        Settings(_env_file=None, dataset_max_samples=0)
