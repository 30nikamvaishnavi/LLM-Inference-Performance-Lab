"""Tests for offline dataset loading, validation, and statistics."""

import json
import sys
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from app.datasets.base import DatasetLoadError, DatasetNotFoundError
from app.datasets.jsonl import JsonlDatasetLoader, dump_jsonl
from app.datasets.registry import DatasetRegistry
from app.datasets.schemas import DatasetCollection, DatasetMetadata, DatasetSample
from app.datasets.synthetic import SyntheticDatasetLoader
from app.datasets.wikitext import WikiText2Loader


@pytest.fixture
def registry(tmp_path) -> DatasetRegistry:
    return DatasetRegistry(tmp_path / "data")


def test_synthetic_dataset_is_offline_deterministic_and_diverse() -> None:
    loader = SyntheticDatasetLoader()

    first = loader.load()
    second = loader.load()

    assert len(first.samples) == 35
    assert first == second
    assert len({sample.sample_id for sample in first.samples}) == 35
    assert {
        "short_prompts",
        "medium_paragraphs",
        "long_form",
        "technical_explanations",
        "python_code",
        "question_answer",
        "summarization_prompts",
    } == {sample.category for sample in first.samples}
    assert first.metadata.source.startswith("Original synthetic")
    assert first.metadata.license_info == "MIT"


def test_synthetic_sample_limit_and_invalid_limit() -> None:
    loader = SyntheticDatasetLoader()

    assert len(loader.load(limit=3).samples) == 3
    with pytest.raises(ValueError, match="at least 1"):
        loader.load(limit=0)


def test_dataset_metadata_and_collection_validation() -> None:
    sample = DatasetSample(
        sample_id="sample-1",
        category="short_prompts",
        input_text="Question?",
        length_category="short",
        source_info="original",
    )
    with pytest.raises(ValidationError):
        DatasetCollection(
            metadata=DatasetMetadata(
                dataset_name="test",
                version="1",
                source="test",
                sample_count=2,
            ),
            samples=[sample],
        )
    with pytest.raises(ValidationError):
        DatasetCollection(
            metadata=DatasetMetadata(
                dataset_name="test",
                version="1",
                source="test",
                sample_count=2,
            ),
            samples=[sample, sample],
        )


def test_registry_missing_dataset_is_explicit(registry: DatasetRegistry) -> None:
    with pytest.raises(DatasetNotFoundError):
        registry.get_metadata("../outside")
    with pytest.raises(DatasetNotFoundError):
        registry.load("not-registered")


def test_registry_lists_wikitext_without_downloading(registry: DatasetRegistry) -> None:
    listed = registry.list_datasets()

    wikitext = next(item for item in listed if item.metadata.dataset_name == "wikitext-2")
    assert wikitext.loaded is False
    assert wikitext.metadata.configuration == "wikitext-2-raw-v1"
    assert not registry.data_dir.exists()


def test_missing_datasets_dependency_fails_only_on_explicit_load(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import builtins

    original_import = builtins.__import__

    def reject_datasets(name, *args, **kwargs):
        if name == "datasets":
            raise ImportError("optional dependency is unavailable")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", reject_datasets)
    loader = WikiText2Loader(tmp_path / "datasets")

    with pytest.raises(DatasetLoadError, match="optional 'datasets' package"):
        loader.load(limit=2)
    assert not (tmp_path / "datasets").exists()


def test_wikitext_loader_uses_requested_config_and_honors_sample_limit(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = []

    def fake_load_dataset(*args, **kwargs):
        calls.append((args, kwargs))
        return [
            {"text": "first public paragraph"},
            {"text": "   "},
            {"text": "second public paragraph"},
            {"text": "third paragraph must not be loaded"},
        ]

    monkeypatch.setitem(
        sys.modules,
        "datasets",
        SimpleNamespace(load_dataset=fake_load_dataset),
    )
    loader = WikiText2Loader(tmp_path / "wikitext", max_samples=4)

    collection = loader.load(limit=2)

    assert len(collection.samples) == 2
    assert collection.samples[0].sample_id == "wikitext-train-0000001"
    assert collection.samples[1].sample_id == "wikitext-train-0000003"
    assert calls[0][0] == ("Salesforce/wikitext", "wikitext-2-raw-v1")
    assert calls[0][1]["split"] == "train"
    assert calls[0][1]["streaming"] is True
    saved_metadata = json.loads(
        (tmp_path / "wikitext" / "wikitext-2-raw-v1.metadata.json").read_text(encoding="utf-8")
    )
    assert saved_metadata["configuration"] == "wikitext-2-raw-v1"
    assert saved_metadata["retrieved_at"] is not None


def test_jsonl_loader_round_trip_and_limit(tmp_path) -> None:
    samples = SyntheticDatasetLoader().load(limit=3).samples
    jsonl_path = tmp_path / "custom.jsonl"
    dump_jsonl(jsonl_path, samples)
    metadata = DatasetMetadata(
        dataset_name="custom",
        version="1",
        source="local test fixture",
        sample_count=3,
    )
    JsonlDatasetLoader.write_metadata(jsonl_path, metadata)

    loaded = JsonlDatasetLoader(jsonl_path).load(limit=2)

    assert len(loaded.samples) == 2
    assert loaded.metadata.dataset_name == "custom"
    assert loaded.metadata.sample_count == 2


def test_jsonl_reports_missing_and_malformed_files(tmp_path) -> None:
    missing = JsonlDatasetLoader(tmp_path / "missing.jsonl")
    with pytest.raises(DatasetLoadError, match="does not exist"):
        missing.load()

    malformed_path = tmp_path / "malformed.jsonl"
    malformed_path.write_text('{"sample_id":', encoding="utf-8")
    with pytest.raises(DatasetLoadError, match="line 1"):
        JsonlDatasetLoader(malformed_path).load()


def test_jsonl_rejects_duplicate_ids(tmp_path) -> None:
    sample = SyntheticDatasetLoader().load(limit=1).samples[0]
    duplicate_path = tmp_path / "duplicate.jsonl"
    duplicate_path.write_text(
        f"{sample.model_dump_json()}\n{sample.model_dump_json()}\n",
        encoding="utf-8",
    )

    with pytest.raises(DatasetLoadError, match="Duplicate sample ID"):
        JsonlDatasetLoader(duplicate_path).load()


def test_local_jsonl_is_discovered_only_under_configured_data_directory(
    registry: DatasetRegistry,
) -> None:
    registry.data_dir.mkdir(parents=True)
    samples = SyntheticDatasetLoader().load(limit=1).samples
    dump_jsonl(registry.data_dir / "local-fixture.jsonl", samples)

    listed_names = {item.metadata.dataset_name for item in registry.list_datasets()}

    assert "local-fixture" in listed_names
    assert "outside-directory" not in listed_names


def test_filtering_statistics_and_deterministic_splits(
    registry: DatasetRegistry,
) -> None:
    collection = registry.load("synthetic-benchmark")
    short_python = registry.filter_samples(
        collection.samples,
        categories={"python_code"},
        length_categories={"short"},
    )
    stats = registry.statistics(collection)
    first_split = registry.split(collection.samples, seed=7)
    second_split = registry.split(collection.samples, seed=7)

    assert len(short_python) == 2
    assert sum(stats.category_counts.values()) == 35
    assert stats.sample_count == 35
    assert first_split == second_split
    all_split_ids = [
        sample.sample_id for split_samples in first_split.values() for sample in split_samples
    ]
    assert len(all_split_ids) == 35
    assert len(set(all_split_ids)) == 35
    with pytest.raises(ValueError, match="sum to less than 1"):
        registry.split(collection.samples, train_ratio=0.8, validation_ratio=0.2)


def test_invalid_local_metadata_is_not_silently_accepted(tmp_path) -> None:
    path = tmp_path / "bad-metadata.jsonl"
    path.write_text("", encoding="utf-8")
    path.with_name("bad-metadata.metadata.json").write_text(
        json.dumps({"dataset_name": "bad-metadata", "version": "1"}),
        encoding="utf-8",
    )

    with pytest.raises(DatasetLoadError, match="Invalid metadata sidecar"):
        JsonlDatasetLoader(path)
