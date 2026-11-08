"""Tests for reproducible benchmark manifest generation."""

import json

import pytest

from app.benchmarking.dataset_preparation import prepare_benchmark_inputs
from app.datasets.base import DatasetLoadError
from app.datasets.registry import DatasetRegistry
from app.datasets.schemas import PrepareBenchmarkRequest
from app.tokenization.registry import TokenizerRegistry


def test_seeded_manifest_is_reproducible_and_records_token_counts(tmp_path) -> None:
    registry = DatasetRegistry(tmp_path / "data")
    registry.load("synthetic-benchmark")
    request = PrepareBenchmarkRequest(
        dataset_name="synthetic-benchmark",
        categories=["python_code", "technical_explanations"],
        max_samples=6,
        seed=123,
        tokenizer_name="simple",
    )
    tokenizers = TokenizerRegistry()

    first = prepare_benchmark_inputs(
        registry, request, tmp_path / "manifests", tokenizers.get("simple")
    )
    second = prepare_benchmark_inputs(
        registry, request, tmp_path / "manifests", tokenizers.get("simple")
    )
    persisted = json.loads(
        (tmp_path / "manifests" / "synthetic-benchmark-seed-123.json").read_text(encoding="utf-8")
    )

    assert first.selected_sample_ids == second.selected_sample_ids
    assert first.input_sha256 == second.input_sha256
    assert all(item.input_token_count is not None for item in first.inputs)
    assert all(item.input_character_count == len(item.input_text) for item in first.inputs)
    assert persisted["input_sha256"] == first.input_sha256


def test_manifest_without_tokenizer_preserves_unmeasured_counts(tmp_path) -> None:
    registry = DatasetRegistry(tmp_path / "data")
    registry.load("synthetic-benchmark")
    request = PrepareBenchmarkRequest(
        dataset_name="synthetic-benchmark",
        max_samples=2,
        seed=42,
    )

    manifest = prepare_benchmark_inputs(registry, request, tmp_path / "manifests")

    assert manifest.tokenizer_name is None
    assert all(item.input_token_count is None for item in manifest.inputs)


def test_manifest_requires_loaded_dataset_and_matching_samples(tmp_path) -> None:
    registry = DatasetRegistry(tmp_path / "data")
    with pytest.raises(DatasetLoadError, match="has not been loaded"):
        prepare_benchmark_inputs(
            registry,
            PrepareBenchmarkRequest(dataset_name="synthetic-benchmark"),
            tmp_path / "manifests",
        )

    registry.load("synthetic-benchmark")
    with pytest.raises(ValueError, match="No dataset samples match"):
        prepare_benchmark_inputs(
            registry,
            PrepareBenchmarkRequest(
                dataset_name="synthetic-benchmark",
                categories=["not-a-category"],
            ),
            tmp_path / "manifests",
        )
