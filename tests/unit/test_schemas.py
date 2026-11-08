"""Tests for shared request and result schemas."""

from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from app.core.schemas import (
    BackendCapabilities,
    BackendInfo,
    BenchmarkConfig,
    BenchmarkResult,
    InferenceRequest,
    TokenizationRequest,
    TokenizationResponse,
)


def test_backend_and_tokenization_schemas_validate() -> None:
    backend = BackendInfo(
        backend_name="test",
        available=True,
        capabilities=BackendCapabilities(supported_devices=["cpu"]),
    )
    request = TokenizationRequest(model_name="test-tokenizer", text="hello")
    response = TokenizationResponse(
        model_name="test-tokenizer",
        tokens=["hello"],
        token_ids=[1],
        token_count=1,
        elapsed_ms=0.5,
    )

    assert backend.capabilities is not None
    assert request.add_special_tokens is True
    assert response.token_count == len(response.token_ids)


def test_tokenization_requires_max_length_for_truncation_or_padding() -> None:
    with pytest.raises(ValidationError, match="max_length is required"):
        TokenizationRequest(
            model_name="test-tokenizer",
            text="hello",
            truncation=True,
        )
    with pytest.raises(ValidationError, match="max_length is required"):
        TokenizationRequest(
            model_name="test-tokenizer",
            text="hello",
            padding="max_length",
        )
    with pytest.raises(ValidationError, match="matching lengths"):
        TokenizationResponse(
            model_name="test-tokenizer",
            tokens=["hello", "world"],
            token_ids=[1],
            token_count=1,
            elapsed_ms=0.5,
        )


def test_inference_request_rejects_invalid_generation_options() -> None:
    with pytest.raises(ValidationError):
        InferenceRequest(
            model_name="test-model",
            backend_name="test",
            prompt="hello",
            max_new_tokens=0,
        )
    with pytest.raises(ValidationError):
        InferenceRequest(
            model_name="test-model",
            backend_name="test",
            prompt="hello",
            temperature=2.1,
        )


def test_benchmark_result_validates_sample_count_and_metrics() -> None:
    config = BenchmarkConfig(
        model_name="test-model",
        backend_name="test",
        prompt_length=16,
        max_output_tokens=8,
        repetitions=2,
    )
    result = BenchmarkResult(
        config=config,
        created_at=datetime.now(timezone.utc),
        successful_runs=2,
        total_latency_ms_median=10.0,
        total_latency_ms_p95=12.0,
    )

    assert result.time_to_first_token_ms_median is None
    with pytest.raises(ValidationError, match="cannot exceed configured repetitions"):
        BenchmarkResult(
            config=config,
            created_at=datetime.now(timezone.utc),
            successful_runs=3,
            total_latency_ms_median=10.0,
            total_latency_ms_p95=12.0,
        )
    with pytest.raises(ValidationError, match="p95 latency cannot be lower"):
        BenchmarkResult(
            config=config,
            created_at=datetime.now(timezone.utc),
            successful_runs=2,
            total_latency_ms_median=12.0,
            total_latency_ms_p95=10.0,
        )
