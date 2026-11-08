"""Tests for repeated inference benchmarking and actual result persistence."""

import csv
import json
import sys
from types import SimpleNamespace

import pytest

from app.benchmarking.inference_runner import InferenceBenchmarkRunner
from app.core.config import Settings
from app.datasets.registry import DatasetRegistry
from app.inference.backends.transformers_backend import TransformersBackend
from app.inference.exceptions import InferenceError
from app.inference.schemas import InferenceBenchmarkRequest


def test_benchmark_excludes_warmups_and_saves_json_csv_without_overwrite(
    tmp_path, fake_inference_components, monkeypatch
):
    torch, _, model, tokenizer_loader, model_loader, _ = fake_inference_components
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(
        sys.modules,
        "transformers",
        SimpleNamespace(__version__="mock-transformers"),
    )
    settings = Settings(
        _env_file=None,
        data_dir=tmp_path / "data",
        benchmark_output_dir=tmp_path / "results",
        model_cache_dir=tmp_path / "models",
        inference_max_new_tokens=8,
    )
    backend = TransformersBackend(
        settings,
        torch_module=torch,
        tokenizer_loader=tokenizer_loader,
        model_loader=model_loader,
    )
    backend.load_model("sshleifer/tiny-gpt2")
    datasets = DatasetRegistry(settings.data_dir, max_samples=50)
    datasets.load("synthetic-benchmark", limit=2)
    runner = InferenceBenchmarkRunner(datasets, backend, settings)
    request = InferenceBenchmarkRequest(
        dataset_name="synthetic-benchmark",
        categories=["short_prompts"],
        max_samples=2,
        max_new_tokens=2,
        repetitions=2,
        warmup_iterations=1,
        seed=17,
    )

    result = runner.run(request)

    assert len(result.entries) == 4
    assert result.measured_requests == 4
    assert result.warmup_runs_excluded is True
    assert len(model.generation_calls) == 6
    assert result.input_sample_ids
    assert result.input_token_counts
    assert result.device == "cpu"
    assert result.model_revision == "model-revision"
    assert result.median_generation_duration_seconds is not None
    json_path = settings.benchmark_output_dir / "inference" / result.json_result_path
    csv_path = settings.benchmark_output_dir / "inference" / result.csv_result_path
    assert json_path.is_file()
    assert csv_path.is_file()
    stored_json = json.loads(json_path.read_text(encoding="utf-8"))
    assert stored_json["measured_requests"] == 4
    with csv_path.open(encoding="utf-8", newline="") as result_file:
        rows = list(csv.DictReader(result_file))
    assert len(rows) == 4
    assert rows[0]["run_id"] == result.run_id
    assert len(runner.list_results()) == 1


def test_benchmark_requires_explicitly_loaded_model_and_dataset(
    tmp_path, fake_inference_components
):
    torch, _, _, tokenizer_loader, model_loader, _ = fake_inference_components
    settings = Settings(_env_file=None, data_dir=tmp_path / "data")
    backend = TransformersBackend(
        settings,
        torch_module=torch,
        tokenizer_loader=tokenizer_loader,
        model_loader=model_loader,
    )
    datasets = DatasetRegistry(settings.data_dir)
    runner = InferenceBenchmarkRunner(datasets, backend, settings)
    request = InferenceBenchmarkRequest(dataset_name="synthetic-benchmark")

    with pytest.raises(InferenceError, match="Load a model"):
        runner.run(request)

    backend.load_model()
    with pytest.raises(InferenceError, match="Load the benchmark dataset"):
        runner.run(request)


def test_benchmark_rejects_empty_filter_selection(tmp_path, fake_inference_components):
    torch, _, _, tokenizer_loader, model_loader, _ = fake_inference_components
    settings = Settings(
        _env_file=None,
        data_dir=tmp_path / "data",
        inference_max_new_tokens=8,
    )
    backend = TransformersBackend(
        settings,
        torch_module=torch,
        tokenizer_loader=tokenizer_loader,
        model_loader=model_loader,
    )
    backend.load_model()
    datasets = DatasetRegistry(settings.data_dir)
    datasets.load("synthetic-benchmark", limit=2)
    runner = InferenceBenchmarkRunner(datasets, backend, settings)

    with pytest.raises(ValueError, match="No dataset samples match"):
        runner.run(
            InferenceBenchmarkRequest(
                dataset_name="synthetic-benchmark",
                categories=["not-a-category"],
            )
        )
