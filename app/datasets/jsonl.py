"""Strict loader for local JSON Lines benchmark datasets."""

import json
from pathlib import Path

from pydantic import ValidationError

from app.datasets.base import DatasetLoader, DatasetLoadError
from app.datasets.schemas import DatasetCollection, DatasetMetadata, DatasetSample


class JsonlDatasetLoader(DatasetLoader):
    """Load one trusted local JSONL file and an optional metadata sidecar."""

    def __init__(
        self,
        path: Path,
        dataset_name: str | None = None,
        metadata_path: Path | None = None,
    ) -> None:
        self.path = path
        self._dataset_name = dataset_name or path.stem
        self.metadata_path = metadata_path or path.with_name(f"{path.stem}.metadata.json")
        self._metadata = self._read_metadata()

    @property
    def metadata(self) -> DatasetMetadata:
        return self._metadata

    def _read_metadata(self) -> DatasetMetadata:
        if not self.metadata_path.is_file():
            return DatasetMetadata(
                dataset_name=self._dataset_name,
                version="local",
                configuration="jsonl",
                source="Local JSONL file.",
                sample_count=None,
            )
        try:
            data = json.loads(self.metadata_path.read_text(encoding="utf-8"))
            return DatasetMetadata.model_validate(data)
        except (OSError, json.JSONDecodeError, ValidationError) as error:
            raise DatasetLoadError(
                f"Invalid metadata sidecar for dataset {self._dataset_name!r}: {error}"
            ) from error

    def load(self, limit: int | None = None) -> DatasetCollection:
        if limit is not None and limit < 1:
            raise ValueError("Sample limit must be at least 1.")
        if not self.path.is_file():
            raise DatasetLoadError(f"Dataset file does not exist: {self.path.name}")

        samples: list[DatasetSample] = []
        seen_ids: set[str] = set()
        try:
            with self.path.open(encoding="utf-8") as dataset_file:
                for line_number, line in enumerate(dataset_file, start=1):
                    if limit is not None and len(samples) >= limit:
                        break
                    if not line.strip():
                        continue
                    try:
                        sample = DatasetSample.model_validate_json(line)
                    except (ValidationError, ValueError) as error:
                        raise DatasetLoadError(
                            f"Invalid JSONL sample at line {line_number} "
                            f"in {self.path.name!r}: {error}"
                        ) from error
                    if sample.sample_id in seen_ids:
                        raise DatasetLoadError(
                            f"Duplicate sample ID {sample.sample_id!r} at line {line_number}."
                        )
                    seen_ids.add(sample.sample_id)
                    samples.append(sample)
        except OSError as error:
            raise DatasetLoadError(
                f"Unable to read dataset file {self.path.name!r}: {error}"
            ) from error

        metadata = self.metadata.model_copy(
            update={"dataset_name": self._dataset_name, "sample_count": len(samples)}
        )
        return DatasetCollection(metadata=metadata, samples=samples)

    @staticmethod
    def write_metadata(path: Path, metadata: DatasetMetadata) -> None:
        """Persist the standard JSON sidecar next to a JSONL corpus."""
        metadata_path = path.with_name(f"{path.stem}.metadata.json")
        metadata_path.write_text(
            json.dumps(metadata.model_dump(mode="json"), indent=2) + "\n",
            encoding="utf-8",
        )


def dump_jsonl(path: Path, samples: list[DatasetSample]) -> None:
    """Write validated samples to a JSONL file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as dataset_file:
        for sample in samples:
            dataset_file.write(sample.model_dump_json() + "\n")
