"""Validated environment-backed application settings."""

from pathlib import Path
from typing import Literal

from pydantic import field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings populated from environment variables and `.env`."""

    model_config = SettingsConfigDict(
        env_prefix="LLM_LAB_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
        validate_default=True,
    )

    environment: Literal["development", "production"] = "development"
    debug: bool = False
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"
    model_cache_dir: Path = Path("~/.cache/llm-inference-performance-lab/models")
    benchmark_output_dir: Path = Path("benchmarks/results")
    data_dir: Path = Path("data")
    dataset_max_samples: int = 5000

    @field_validator("model_cache_dir", "benchmark_output_dir", "data_dir", mode="before")
    @classmethod
    def validate_directory_path(cls, value: object) -> Path:
        if not isinstance(value, (str, Path)) or not str(value).strip():
            raise ValueError("Directory settings must not be empty.")
        return Path(value).expanduser()

    @field_validator("dataset_max_samples")
    @classmethod
    def validate_dataset_max_samples(cls, value: int) -> int:
        if value < 1:
            raise ValueError("dataset_max_samples must be at least 1.")
        return value

    @model_validator(mode="after")
    def validate_environment_options(self) -> "Settings":
        if self.environment == "production" and self.debug:
            raise ValueError("Debug mode must be disabled in production.")
        return self
