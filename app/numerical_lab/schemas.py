"""Validated schemas for numerical experiment results and requests."""

from datetime import datetime
from typing import Literal

from pydantic import Field, field_validator

from app.core.schemas import StrictSchema

ExperimentName = Literal[
    "precision",
    "matrix_multiplication",
    "numerical_stability",
    "reproducibility",
    "all",
]
MeasurementStatus = Literal["measured", "unsupported", "failed"]
ResultStatus = Literal["success", "partial", "failed"]


class ExperimentRunRequest(StrictSchema):
    experiment_name: ExperimentName = "all"
    seed: int = Field(default=42, ge=0, le=9_223_372_036_854_775_807)
    warmup_iterations: int = Field(default=2, ge=0, le=20)
    iterations: int = Field(default=5, ge=1, le=100)
    matrix_sizes: list[int] = Field(
        default_factory=lambda: [64, 256, 512], min_length=1, max_length=3
    )

    @field_validator("matrix_sizes")
    @classmethod
    def validate_matrix_sizes(cls, value: list[int]) -> list[int]:
        if len(set(value)) != len(value):
            raise ValueError("matrix_sizes must not contain duplicates.")
        if any(size < 16 or size > 512 for size in value):
            raise ValueError("matrix sizes must be between 16 and 512.")
        return value


class ExperimentMeasurement(StrictSchema):
    operation: str
    status: MeasurementStatus
    dtype: str | None = None
    dimensions: list[int] = Field(default_factory=list)
    warmup_iterations: int = Field(ge=0)
    measured_iterations: int = Field(ge=0)
    mean_latency_seconds: float | None = Field(default=None, ge=0)
    median_latency_seconds: float | None = Field(default=None, ge=0)
    p95_latency_seconds: float | None = Field(default=None, ge=0)
    max_absolute_error: float | None = Field(default=None, ge=0)
    mean_absolute_error: float | None = Field(default=None, ge=0)
    relative_error_l2: float | None = Field(default=None, ge=0)
    non_finite_value_count: int | None = Field(default=None, ge=0)
    reason: str | None = None
    metrics: dict[str, str | int | float | bool | None] = Field(default_factory=dict)


class ExperimentEnvironment(StrictSchema):
    python_version: str
    torch_version: str
    device: Literal["cpu"] = "cpu"
    operating_system: str
    cpu_model: str | None = None
    cpu_count: int | None = Field(default=None, ge=1)


class NumericalExperimentResult(StrictSchema):
    run_id: str
    created_at: datetime
    experiment_name: str
    status: ResultStatus
    random_seed: int = Field(ge=0)
    configuration: dict[str, str | int | float | bool | list[int] | None]
    environment: ExperimentEnvironment
    measurements: list[ExperimentMeasurement]
    json_result_path: str
    csv_result_path: str


class ExperimentInfo(StrictSchema):
    name: str
    description: str
    available: bool
    unavailable_reason: str | None = None


class ExperimentSummary(StrictSchema):
    run_id: str
    created_at: datetime
    experiment_name: str
    status: ResultStatus
    measurement_count: int = Field(ge=0)
    result_file: str
