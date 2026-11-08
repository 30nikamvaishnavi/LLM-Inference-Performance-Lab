"""Explicit, bounded WikiText-2 loader using the optional datasets package."""

import json
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from app.datasets.base import DatasetLoader, DatasetLoadError, validate_sample_limit
from app.datasets.schemas import DatasetCollection, DatasetMetadata, DatasetSample

logger = logging.getLogger(__name__)

WIKITEXT_SOURCE = "https://huggingface.co/datasets/Salesforce/wikitext"
WIKITEXT_CONFIGURATION = "wikitext-2-raw-v1"


class WikiText2Loader(DatasetLoader):
    """Download/cache WikiText-2 only when `load` is explicitly invoked."""

    def __init__(self, data_dir: Path, max_samples: int = 5000) -> None:
        if max_samples < 1:
            raise ValueError("max_samples must be at least 1.")
        self.data_dir = data_dir
        self.max_samples = max_samples
        self.metadata_path = data_dir / "wikitext-2-raw-v1.metadata.json"
        self._metadata: DatasetMetadata | None = self._load_saved_metadata()

    @property
    def metadata(self) -> DatasetMetadata:
        return self._metadata or DatasetMetadata(
            dataset_name="wikitext-2",
            version=WIKITEXT_CONFIGURATION,
            configuration=WIKITEXT_CONFIGURATION,
            source=WIKITEXT_SOURCE,
            license_info=None,
            sample_count=None,
        )

    def _load_saved_metadata(self) -> DatasetMetadata | None:
        if not self.metadata_path.is_file():
            return None
        try:
            return DatasetMetadata.model_validate_json(
                self.metadata_path.read_text(encoding="utf-8")
            )
        except (OSError, ValueError) as error:
            logger.warning("Ignoring invalid WikiText metadata: %s", error)
            return None

    def load(self, limit: int | None = None) -> DatasetCollection:
        sample_limit = validate_sample_limit(limit, self.max_samples)
        try:
            from datasets import load_dataset
        except ImportError as error:
            raise DatasetLoadError(
                "WikiText-2 requires the optional 'datasets' package. "
                'Install it with: python -m pip install -e ".[datasets]"'
            ) from error

        try:
            self.data_dir.mkdir(parents=True, exist_ok=True)
            rows = load_dataset(
                "Salesforce/wikitext",
                WIKITEXT_CONFIGURATION,
                split="train",
                cache_dir=str(self.data_dir / "huggingface"),
                streaming=True,
            )
            samples: list[DatasetSample] = []
            for row_number, row in enumerate(rows, start=1):
                text = row.get("text")
                if not isinstance(text, str) or not text.strip():
                    continue
                samples.append(
                    DatasetSample(
                        sample_id=f"wikitext-train-{row_number:07d}",
                        category="wikitext",
                        input_text=text,
                        length_category=self._classify_length(text),
                        source_info=(f"WikiText-2 train split, {WIKITEXT_CONFIGURATION}."),
                        tags=["public-dataset", "wikitext"],
                    )
                )
                if len(samples) >= sample_limit:
                    break
            if not samples:
                raise DatasetLoadError(
                    "WikiText-2 loaded successfully but contained no non-empty text "
                    "within the requested sample limit."
                )
            info: Any = getattr(rows, "info", None)
            license_info = getattr(info, "license", None)
            metadata = DatasetMetadata(
                dataset_name="wikitext-2",
                version=WIKITEXT_CONFIGURATION,
                configuration=WIKITEXT_CONFIGURATION,
                source=WIKITEXT_SOURCE,
                retrieved_at=datetime.now(UTC),
                license_info=license_info if isinstance(license_info, str) else None,
                sample_count=len(samples),
            )
            self._write_metadata(metadata)
            self._metadata = metadata
            return DatasetCollection(metadata=metadata, samples=samples)
        except DatasetLoadError:
            raise
        except Exception as error:
            logger.warning(
                "WikiText-2 load failed (%s). No fallback dataset was substituted.",
                type(error).__name__,
            )
            raise DatasetLoadError(
                "Unable to load WikiText-2. Check internet access, the dataset "
                "source, and the optional datasets package; no substitute was used."
            ) from error

    @staticmethod
    def _classify_length(text: str) -> str:
        if len(text) < 280:
            return "short"
        if len(text) < 1200:
            return "medium"
        return "long"

    def _write_metadata(self, metadata: DatasetMetadata) -> None:
        temporary_path = self.metadata_path.with_suffix(".json.tmp")
        temporary_path.write_text(
            json.dumps(metadata.model_dump(mode="json"), indent=2) + "\n",
            encoding="utf-8",
        )
        temporary_path.replace(self.metadata_path)
