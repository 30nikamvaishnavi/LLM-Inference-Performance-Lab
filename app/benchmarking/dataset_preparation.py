"""Reproducible selection and persistence of inference benchmark inputs."""

import hashlib
import json
import logging
import random
from datetime import UTC, datetime
from pathlib import Path

from app.datasets.registry import DatasetRegistry
from app.datasets.schemas import (
    BenchmarkInput,
    BenchmarkManifest,
    PrepareBenchmarkRequest,
)
from app.tokenization.base import TokenizerBackend

logger = logging.getLogger(__name__)


def prepare_benchmark_inputs(
    registry: DatasetRegistry,
    request: PrepareBenchmarkRequest,
    output_dir: Path,
    tokenizer: TokenizerBackend | None = None,
) -> BenchmarkManifest:
    """Create a stable seeded manifest and atomically save it as JSON."""
    collection = registry.get_loaded(request.dataset_name)
    candidates = registry.filter_samples(
        collection.samples,
        categories=set(request.categories) if request.categories is not None else None,
        length_categories=(
            set(request.length_categories) if request.length_categories is not None else None
        ),
    )
    if not candidates:
        raise ValueError("No dataset samples match the requested filters.")
    selected = random.Random(request.seed).sample(
        candidates, min(request.max_samples, len(candidates))
    )
    inputs: list[BenchmarkInput] = []
    for sample in selected:
        tokens = tokenizer.encode(sample.input_text) if tokenizer is not None else None
        inputs.append(
            BenchmarkInput(
                sample_id=sample.sample_id,
                category=sample.category,
                length_category=sample.length_category,
                input_text=sample.input_text,
                input_character_count=len(sample.input_text),
                input_token_count=len(tokens) if tokens is not None else None,
            )
        )

    canonical_inputs = json.dumps(
        [{"sample_id": item.sample_id, "input_text": item.input_text} for item in inputs],
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    manifest = BenchmarkManifest(
        dataset_name=collection.metadata.dataset_name,
        dataset_version=collection.metadata.version,
        seed=request.seed,
        created_at=datetime.now(UTC),
        tokenizer_name=tokenizer.tokenizer_name if tokenizer is not None else None,
        selected_sample_ids=[item.sample_id for item in inputs],
        input_sha256=hashlib.sha256(canonical_inputs).hexdigest(),
        inputs=inputs,
    )

    safe_name = collection.metadata.dataset_name.replace("/", "_")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{safe_name}-seed-{request.seed}.json"
    temporary_path = output_path.with_suffix(".json.tmp")
    try:
        temporary_path.write_text(
            manifest.model_dump_json(indent=2) + "\n",
            encoding="utf-8",
        )
        temporary_path.replace(output_path)
    except OSError:
        logger.exception("Unable to persist benchmark input manifest")
        raise
    logger.info(
        "Prepared %d benchmark inputs for dataset %s",
        len(inputs),
        collection.metadata.dataset_name,
    )
    return manifest
