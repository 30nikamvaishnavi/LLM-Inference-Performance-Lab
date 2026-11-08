"""Validated model lifecycle and inference benchmark schemas."""

from datetime import datetime

from pydantic import Field, model_validator

from app.core.schemas import InferenceResponse, StrictSchema
from app.datasets.schemas import LengthCategory


class ModelStatus(StrictSchema):
    backend_name: str
    configured_model_id: str
    downloaded: bool
    loaded: bool
    ready: bool
    model_id: str | None = None
    revision: str | None = None
    device: str = "cpu"
    loading_duration_seconds: float | None = Field(default=None, ge=0)
    max_sequence_length: int | None = Field(default=None, ge=1)
    tokenizer_name: str | None = None


class ModelLoadRequest(StrictSchema):
    model_id: str | None = Field(default=None, min_length=1, max_length=256)
    revision: str | None = Field(default=None, min_length=1, max_length=128)
    allow_download: bool = False


class ModelLoadResponse(StrictSchema):
    status: ModelStatus


class ModelTokenizeRequest(StrictSchema):
    text: str = Field(min_length=1, max_length=100_000)
    add_special_tokens: bool = True
    max_length: int | None = Field(default=None, ge=1, le=32768)
    truncation: bool = False

    @model_validator(mode="after")
    def validate_truncation(self) -> "ModelTokenizeRequest":
        if self.truncation and self.max_length is None:
            raise ValueError("max_length is required when truncation is enabled.")
        return self


class ModelTokenizeResponse(StrictSchema):
    model_id: str
    tokenizer_name: str
    tokens: list[str]
    token_ids: list[int]
    token_count: int = Field(ge=0)
    elapsed_ms: float = Field(ge=0)
    special_tokens: dict[str, str | None]
    max_sequence_length: int | None = Field(default=None, ge=1)


class GenerateRequest(StrictSchema):
    prompt: str = Field(min_length=1, max_length=100_000)
    max_new_tokens: int = Field(default=32, ge=1, le=256)
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    top_k: int | None = Field(default=None, ge=1, le=1000)
    seed: int | None = Field(default=None, ge=0)


class GenerateResponse(InferenceResponse):
    model_revision: str | None = None
    model_loading_duration_seconds: float | None = Field(default=None, ge=0)
    time_to_first_token_ms: float | None = None


class InferenceBenchmarkRequest(StrictSchema):
    dataset_name: str = Field(min_length=1, max_length=128)
    categories: list[str] | None = None
    length_categories: list[LengthCategory] | None = None
    max_samples: int = Field(default=5, ge=1, le=50)
    max_new_tokens: int = Field(default=16, ge=1, le=128)
    repetitions: int = Field(default=2, ge=1, le=20)
    warmup_iterations: int = Field(default=1, ge=0, le=5)
    seed: int = Field(default=42, ge=0)
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    top_k: int | None = Field(default=None, ge=1, le=1000)


class InferenceBenchmarkEntry(StrictSchema):
    sample_id: str
    repetition: int = Field(ge=1)
    input_token_count: int = Field(ge=0)
    generated_token_count: int = Field(ge=0)
    generation_duration_seconds: float = Field(ge=0)
    generated_tokens_per_second: float | None = Field(default=None, ge=0)
    peak_process_memory_bytes: int | None = Field(default=None, ge=0)
    cpu_utilization_percent: float | None = Field(default=None, ge=0)
    time_to_first_token_seconds: float | None = Field(default=None, ge=0)


class InferenceBenchmarkResult(StrictSchema):
    run_id: str
    created_at: datetime
    backend_name: str
    model_id: str
    model_revision: str | None = None
    python_version: str
    torch_version: str
    transformers_version: str
    operating_system: str
    device: str
    cpu_model: str | None = None
    cpu_count: int | None = Field(default=None, ge=1)
    model_loading_duration_seconds: float | None = Field(default=None, ge=0)
    measured_requests: int = Field(ge=0)
    median_generation_duration_seconds: float | None = Field(default=None, ge=0)
    p95_generation_duration_seconds: float | None = Field(default=None, ge=0)
    mean_generated_tokens_per_second: float | None = Field(default=None, ge=0)
    configuration: InferenceBenchmarkRequest
    input_sample_ids: list[str]
    input_token_counts: list[int]
    entries: list[InferenceBenchmarkEntry]
    warmup_iterations: int = Field(ge=0)
    warmup_runs_excluded: bool = True
    json_result_path: str
    csv_result_path: str


class InferenceBenchmarkSummary(StrictSchema):
    run_id: str
    created_at: datetime
    backend_name: str
    model_id: str
    measured_requests: int = Field(ge=0)
    result_file: str
