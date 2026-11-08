"""Dataset loader registry, filtering, statistics, and local discovery."""

import logging
import re
from pathlib import Path

from pydantic import ValidationError

from app.datasets.base import (
    DatasetLoader,
    DatasetLoadError,
    DatasetNotFoundError,
)
from app.datasets.jsonl import JsonlDatasetLoader
from app.datasets.schemas import (
    DatasetCollection,
    DatasetInfo,
    DatasetMetadata,
    DatasetSample,
    DatasetStatistics,
)
from app.datasets.synthetic import SyntheticDatasetLoader
from app.datasets.wikitext import WikiText2Loader

logger = logging.getLogger(__name__)
_DATASET_NAME_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")


class DatasetRegistry:
    """Manage trusted dataset sources beneath an optional local data directory."""

    def __init__(self, data_dir: Path, max_samples: int = 5000) -> None:
        if max_samples < 1:
            raise ValueError("max_samples must be at least 1.")
        self.data_dir = data_dir.expanduser()
        self.max_samples = max_samples
        self._loaders: dict[str, DatasetLoader] = {
            "synthetic-benchmark": SyntheticDatasetLoader(),
            "wikitext-2": WikiText2Loader(self.data_dir, max_samples=max_samples),
        }
        self._loaded: dict[str, DatasetCollection] = {}

    def register(self, loader: DatasetLoader) -> None:
        """Register a loader, rejecting unsafe or duplicate public identifiers."""
        name = loader.metadata.dataset_name
        if not _DATASET_NAME_PATTERN.fullmatch(name):
            raise ValueError(f"Invalid dataset name: {name!r}")
        if name in self._loaders:
            raise ValueError(f"Dataset {name!r} is already registered.")
        self._loaders[name] = loader

    def _local_loaders(self) -> dict[str, JsonlDatasetLoader]:
        loaders: dict[str, JsonlDatasetLoader] = {}
        if not self.data_dir.is_dir():
            return loaders
        try:
            for path in self.data_dir.glob("*.jsonl"):
                name = path.stem
                if (
                    not path.is_symlink()
                    and _DATASET_NAME_PATTERN.fullmatch(name)
                    and name not in self._loaders
                ):
                    loaders[name] = JsonlDatasetLoader(path, dataset_name=name)
        except OSError as error:
            logger.warning("Unable to scan local dataset directory: %s", error)
        return loaders

    def list_datasets(self) -> list[DatasetInfo]:
        """Describe registered and local datasets without downloading anything."""
        infos: list[DatasetInfo] = []
        try:
            from importlib.util import find_spec

            datasets_available = find_spec("datasets") is not None
        except (ImportError, ValueError):
            datasets_available = False

        loaders = {**self._loaders, **self._local_loaders()}
        for name, loader in sorted(loaders.items()):
            optional_available = datasets_available if name == "wikitext-2" else None
            unavailable_reason = None
            if name == "wikitext-2" and not datasets_available:
                unavailable_reason = (
                    'Install optional support with: python -m pip install -e ".[datasets]"'
                )
            try:
                metadata = loader.metadata
            except DatasetLoadError as error:
                metadata = DatasetMetadata(
                    dataset_name=name,
                    version="unknown",
                    source="Local JSONL dataset with invalid metadata.",
                )
                unavailable_reason = str(error)
            infos.append(
                DatasetInfo(
                    metadata=metadata,
                    loaded=name in self._loaded,
                    optional_dependency_available=optional_available,
                    unavailable_reason=unavailable_reason,
                )
            )
        return infos

    def get_loader(self, dataset_name: str) -> DatasetLoader:
        """Return a registered loader or a constrained local JSONL loader."""
        if not _DATASET_NAME_PATTERN.fullmatch(dataset_name):
            raise DatasetNotFoundError(dataset_name)
        loader = self._loaders.get(dataset_name)
        if loader is not None:
            return loader
        local_loader = self._local_loaders().get(dataset_name)
        if local_loader is None:
            raise DatasetNotFoundError(dataset_name)
        return local_loader

    def get_metadata(self, dataset_name: str) -> DatasetMetadata:
        return self.get_loader(dataset_name).metadata

    def load(self, dataset_name: str, limit: int | None = None) -> DatasetCollection:
        """Explicitly load a dataset, bounded by the registry safety limit."""
        loader = self.get_loader(dataset_name)
        bounded_limit = self.max_samples if limit is None else min(limit, self.max_samples)
        collection = loader.load(bounded_limit)
        self._validate_collection(collection)
        self._loaded[dataset_name] = collection
        return collection

    def get_loaded(self, dataset_name: str) -> DatasetCollection:
        try:
            return self._loaded[dataset_name]
        except KeyError as error:
            self.get_loader(dataset_name)
            raise DatasetLoadError(
                f"Dataset {dataset_name!r} has not been loaded. "
                "Use the explicit dataset load endpoint first."
            ) from error

    @staticmethod
    def _validate_collection(collection: DatasetCollection) -> None:
        try:
            DatasetCollection.model_validate(collection.model_dump())
        except ValidationError as error:
            raise DatasetLoadError(f"Dataset validation failed: {error}") from error

    @staticmethod
    def filter_samples(
        samples: list[DatasetSample],
        categories: set[str] | None = None,
        length_categories: set[str] | None = None,
    ) -> list[DatasetSample]:
        """Filter samples by exact category and expected length labels."""
        return [
            sample
            for sample in samples
            if (categories is None or sample.category in categories)
            and (length_categories is None or sample.length_category in length_categories)
        ]

    @staticmethod
    def statistics(collection: DatasetCollection) -> DatasetStatistics:
        """Calculate text-only statistics without requiring a tokenizer."""
        category_counts: dict[str, int] = {}
        length_counts: dict[str, int] = {}
        total_characters = 0
        for sample in collection.samples:
            category_counts[sample.category] = category_counts.get(sample.category, 0) + 1
            length_counts[sample.length_category] = length_counts.get(sample.length_category, 0) + 1
            total_characters += len(sample.input_text)
        sample_count = len(collection.samples)
        return DatasetStatistics(
            dataset_name=collection.metadata.dataset_name,
            sample_count=sample_count,
            category_counts=category_counts,
            length_category_counts=length_counts,
            total_characters=total_characters,
            mean_characters_per_sample=(total_characters / sample_count if sample_count else 0.0),
        )

    @staticmethod
    def split(
        samples: list[DatasetSample],
        train_ratio: float = 0.8,
        validation_ratio: float = 0.1,
        seed: int = 42,
    ) -> dict[str, list[DatasetSample]]:
        """Deterministically split samples while keeping each sample in one set."""
        if train_ratio <= 0 or validation_ratio < 0:
            raise ValueError("Split ratios must be positive (validation may be zero).")
        if train_ratio + validation_ratio >= 1:
            raise ValueError("Train and validation ratios must sum to less than 1.")
        import random

        shuffled = list(samples)
        random.Random(seed).shuffle(shuffled)
        train_end = int(len(shuffled) * train_ratio)
        validation_end = train_end + int(len(shuffled) * validation_ratio)
        return {
            "train": shuffled[:train_end],
            "validation": shuffled[train_end:validation_end],
            "test": shuffled[validation_end:],
        }
