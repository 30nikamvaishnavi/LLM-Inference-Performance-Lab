"""Dataset loader contract and shared validation helpers."""

from abc import ABC, abstractmethod

from app.datasets.schemas import DatasetCollection, DatasetMetadata


class DatasetLoadError(RuntimeError):
    """Raised when an explicitly requested dataset cannot be loaded."""


class DatasetNotFoundError(KeyError):
    """Raised when a dataset name is not registered."""


class DatasetLoader(ABC):
    """Interface implemented by built-in, remote, and local dataset loaders."""

    @property
    @abstractmethod
    def metadata(self) -> DatasetMetadata:
        """Return source metadata without loading sample data."""

    @abstractmethod
    def load(self, limit: int | None = None) -> DatasetCollection:
        """Load and validate up to `limit` samples."""


def validate_sample_limit(limit: int | None, maximum: int) -> int:
    """Resolve and validate a requested row limit."""
    if limit is None:
        return maximum
    if limit < 1:
        raise ValueError("Sample limit must be at least 1.")
    return min(limit, maximum)
