"""Validated dataset and benchmark-input models."""

from datetime import datetime
from typing import Literal

from pydantic import Field, field_validator, model_validator

from app.core.schemas import StrictSchema

LengthCategory = Literal["short", "medium", "long"]


class DatasetSample(StrictSchema):
    sample_id: str = Field(min_length=1, max_length=128)
    category: str = Field(min_length=1, max_length=64)
    input_text: str = Field(min_length=1)
    length_category: LengthCategory
    source_info: str = Field(min_length=1, max_length=512)
    tags: list[str] = Field(default_factory=list)

    @field_validator("sample_id", "category", "source_info")
    @classmethod
    def reject_blank_strings(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("This value must not be blank.")
        return value


class DatasetMetadata(StrictSchema):
    dataset_name: str = Field(min_length=1, max_length=128)
    version: str = Field(min_length=1, max_length=64)
    configuration: str | None = Field(default=None, max_length=128)
    source: str = Field(min_length=1, max_length=512)
    retrieved_at: datetime | None = None
    license_info: str | None = Field(default=None, max_length=512)
    sample_count: int | None = Field(default=None, ge=0)


class DatasetStatistics(StrictSchema):
    dataset_name: str
    sample_count: int = Field(ge=0)
    category_counts: dict[str, int]
    length_category_counts: dict[str, int]
    total_characters: int = Field(ge=0)
    mean_characters_per_sample: float = Field(ge=0)


class DatasetInfo(StrictSchema):
    metadata: DatasetMetadata
    loaded: bool
    optional_dependency_available: bool | None = None
    unavailable_reason: str | None = None


class DatasetLoadResponse(StrictSchema):
    metadata: DatasetMetadata
    loaded_sample_count: int = Field(ge=0)


class DatasetCollection(StrictSchema):
    metadata: DatasetMetadata
    samples: list[DatasetSample]

    @model_validator(mode="after")
    def validate_unique_sample_ids(self) -> "DatasetCollection":
        ids = [sample.sample_id for sample in self.samples]
        if len(ids) != len(set(ids)):
            raise ValueError("Dataset sample IDs must be unique.")
        if self.metadata.sample_count is not None and (
            self.metadata.sample_count != len(self.samples)
        ):
            raise ValueError("Metadata sample_count must match the loaded sample count.")
        return self


class DatasetPage(StrictSchema):
    dataset_name: str
    total_samples: int = Field(ge=0)
    offset: int = Field(ge=0)
    limit: int = Field(ge=1)
    samples: list[DatasetSample]


class BenchmarkInput(StrictSchema):
    sample_id: str
    category: str
    length_category: LengthCategory
    input_text: str
    input_character_count: int = Field(ge=0)
    input_token_count: int | None = Field(default=None, ge=0)


class BenchmarkManifest(StrictSchema):
    dataset_name: str
    dataset_version: str
    seed: int
    created_at: datetime
    tokenizer_name: str | None = None
    selected_sample_ids: list[str]
    input_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    inputs: list[BenchmarkInput]

    @model_validator(mode="after")
    def validate_manifest_ids(self) -> "BenchmarkManifest":
        ids = [sample.sample_id for sample in self.inputs]
        if ids != self.selected_sample_ids:
            raise ValueError("selected_sample_ids must match the manifest input order.")
        if len(ids) != len(set(ids)):
            raise ValueError("Manifest sample IDs must be unique.")
        return self


class DatasetLoadRequest(StrictSchema):
    limit: int | None = Field(default=None, ge=1)


class TokenizeTextRequest(StrictSchema):
    text: str = Field(min_length=1)
    tokenizer_name: str = "simple"
    allow_download: bool = False

    @field_validator("tokenizer_name")
    @classmethod
    def validate_tokenizer_name(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("tokenizer_name must not be blank.")
        return value


class TokenizeTextResponse(StrictSchema):
    tokenizer_name: str
    tokens: list[str]
    token_count: int = Field(ge=0)
    character_count: int = Field(ge=0)
    characters_per_token: float | None = Field(default=None, ge=0)
    elapsed_seconds: float = Field(ge=0)
    tokens_per_second: float | None = Field(default=None, ge=0)


class TokenizationStatistics(StrictSchema):
    tokenizer_name: str
    dataset_name: str
    sample_count: int = Field(ge=0)
    token_count: int = Field(ge=0)
    tokens_per_second: float | None = Field(default=None, ge=0)
    mean_characters_per_token: float | None = Field(default=None, ge=0)
    input_length_distribution: dict[str, int]
    unavailable_reason: str | None = None


class PrepareBenchmarkRequest(StrictSchema):
    dataset_name: str = Field(min_length=1, max_length=128)
    categories: list[str] | None = None
    length_categories: list[LengthCategory] | None = None
    max_samples: int = Field(default=20, ge=1, le=5000)
    seed: int = 42
    tokenizer_name: str | None = None

    @model_validator(mode="after")
    def validate_filters(self) -> "PrepareBenchmarkRequest":
        if self.categories is not None and any(not item.strip() for item in self.categories):
            raise ValueError("Category filters must not contain blank values.")
        if self.tokenizer_name is not None and not self.tokenizer_name.strip():
            raise ValueError("tokenizer_name must not be blank.")
        return self
