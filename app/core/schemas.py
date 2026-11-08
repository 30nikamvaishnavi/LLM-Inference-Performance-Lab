"""Validated data-transfer schemas shared by the API and backend adapters."""

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class StrictSchema(BaseModel):
    model_config = ConfigDict(extra="forbid")


class BackendCapabilities(StrictSchema):
    supported_devices: list[str] = Field(default_factory=list)
    features: list[str] = Field(default_factory=list)
    supports_time_to_first_token: bool = False
    supports_concurrent_requests: bool = False


class BackendInfo(StrictSchema):
    backend_name: str = Field(min_length=1, max_length=64)
    available: bool
    capabilities: BackendCapabilities | None = None
    unavailable_reason: str | None = None


class ModelInfo(StrictSchema):
    model_id: str = Field(min_length=1, max_length=256)
    backend_name: str = Field(min_length=1, max_length=64)
    loaded: bool = False
    model_format: str | None = None
    parameter_count: int | None = Field(default=None, ge=1)


class TokenizationRequest(StrictSchema):
    model_name: str = Field(min_length=1, max_length=256)
    text: str = Field(min_length=1)
    add_special_tokens: bool = True
    truncation: bool = False
    max_length: int | None = Field(default=None, ge=1)
    padding: Literal["none", "longest", "max_length"] = "none"

    @model_validator(mode="after")
    def validate_length_options(self) -> "TokenizationRequest":
        if self.truncation and self.max_length is None:
            raise ValueError("max_length is required when truncation is enabled.")
        if self.padding == "max_length" and self.max_length is None:
            raise ValueError("max_length is required for max_length padding.")
        return self


class TokenizationResponse(StrictSchema):
    model_name: str
    tokens: list[str]
    token_ids: list[int]
    token_count: int = Field(ge=0)
    elapsed_ms: float = Field(ge=0)
    special_tokens: dict[str, str | None] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_token_counts(self) -> "TokenizationResponse":
        if self.token_count != len(self.token_ids) or len(self.tokens) != len(self.token_ids):
            raise ValueError("token_count, tokens, and token_ids must have matching lengths.")
        return self


class InferenceRequest(StrictSchema):
    model_name: str = Field(min_length=1, max_length=256)
    backend_name: str = Field(min_length=1, max_length=64)
    prompt: str = Field(min_length=1)
    max_new_tokens: int = Field(default=128, ge=1, le=8192)
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    top_k: int | None = Field(default=None, ge=1)


class InferenceResponse(StrictSchema):
    model_name: str
    backend_name: str
    generated_text: str
    output_token_count: int = Field(ge=0)
    total_latency_ms: float = Field(ge=0)
    time_to_first_token_ms: float | None = Field(default=None, ge=0)
    time_per_output_token_ms: float | None = Field(default=None, ge=0)
    output_tokens_per_second: float | None = Field(default=None, ge=0)
    peak_memory_bytes: int | None = Field(default=None, ge=0)


class BenchmarkConfig(StrictSchema):
    model_name: str = Field(min_length=1, max_length=256)
    backend_name: str = Field(min_length=1, max_length=64)
    prompt_length: int = Field(ge=1)
    max_output_tokens: int = Field(ge=1, le=8192)
    repetitions: int = Field(default=5, ge=1, le=1000)
    concurrent_requests: int = Field(default=1, ge=1, le=256)
    warmup_iterations: int = Field(default=1, ge=0, le=100)


class BenchmarkResult(StrictSchema):
    config: BenchmarkConfig
    created_at: datetime
    successful_runs: int = Field(ge=1)
    total_latency_ms_median: float = Field(ge=0)
    total_latency_ms_p95: float = Field(ge=0)
    time_to_first_token_ms_median: float | None = Field(default=None, ge=0)
    time_per_output_token_ms_median: float | None = Field(default=None, ge=0)
    output_tokens_per_second: float | None = Field(default=None, ge=0)
    request_throughput_per_second: float | None = Field(default=None, ge=0)
    peak_memory_bytes: int | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def validate_successful_runs(self) -> "BenchmarkResult":
        if self.successful_runs > self.config.repetitions:
            raise ValueError("successful_runs cannot exceed configured repetitions.")
        if self.total_latency_ms_p95 < self.total_latency_ms_median:
            raise ValueError("The p95 latency cannot be lower than the median latency.")
        return self


class SystemInfo(StrictSchema):
    operating_system: str
    operating_system_version: str
    python_version: str
    cpu_count: int | None = Field(default=None, ge=1)
    memory_total_bytes: int | None = Field(default=None, ge=1)
    gpu_devices: list[str] | None = None
