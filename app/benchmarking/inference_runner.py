"""Repeated CPU inference benchmarks with JSON and CSV result persistence."""

import csv
import logging
import os
import platform
import statistics
import sys
import uuid
from datetime import UTC, datetime
from pathlib import Path

from app.benchmarking.dataset_preparation import prepare_benchmark_inputs
from app.core.config import Settings
from app.datasets.base import DatasetLoadError, DatasetNotFoundError
from app.datasets.registry import DatasetRegistry
from app.datasets.schemas import PrepareBenchmarkRequest
from app.inference.backends.transformers_backend import TransformersBackend
from app.inference.exceptions import InferenceError
from app.inference.schemas import (
    GenerateRequest,
    InferenceBenchmarkEntry,
    InferenceBenchmarkRequest,
    InferenceBenchmarkResult,
)

logger = logging.getLogger(__name__)
_CSV_FIELDS = (
    "run_id",
    "created_at",
    "backend_name",
    "model_id",
    "model_revision",
    "sample_id",
    "repetition",
    "input_token_count",
    "generated_token_count",
    "generation_duration_seconds",
    "generated_tokens_per_second",
    "peak_process_memory_bytes",
    "cpu_utilization_percent",
    "time_to_first_token_seconds",
    "device",
)


class InferenceBenchmarkRunner:
    """Run warm-ups and timed repetitions over fixed Phase 3 corpus samples."""

    def __init__(
        self,
        dataset_registry: DatasetRegistry,
        backend: TransformersBackend,
        settings: Settings,
    ) -> None:
        self.dataset_registry = dataset_registry
        self.backend = backend
        self.settings = settings

    def run(self, request: InferenceBenchmarkRequest) -> InferenceBenchmarkResult:
        model_status = self.backend.model_status()
        if not model_status.ready or model_status.model_id is None:
            raise InferenceError("Load a model before starting an inference benchmark.")
        try:
            self.dataset_registry.get_loaded(request.dataset_name)
        except DatasetNotFoundError as error:
            raise InferenceError(
                f"Benchmark dataset {request.dataset_name!r} is not registered."
            ) from error
        except DatasetLoadError as error:
            raise InferenceError(
                "Load the benchmark dataset before starting the inference benchmark."
            ) from error
        manifest_request = PrepareBenchmarkRequest(
            dataset_name=request.dataset_name,
            categories=request.categories,
            length_categories=request.length_categories,
            max_samples=request.max_samples,
            seed=request.seed,
            tokenizer_name=self.backend.model_status().tokenizer_name,
        )
        manifest = prepare_benchmark_inputs(
            self.dataset_registry,
            manifest_request,
            self.settings.benchmark_output_dir / "manifests",
            self.backend.get_tokenizer_adapter(),
        )
        if not manifest.inputs:
            raise InferenceError("No benchmark inputs match the requested filters.")

        try:
            import torch
            import transformers
        except ImportError as error:
            raise InferenceError("PyTorch and Transformers must be installed.") from error

        for _ in range(request.warmup_iterations):
            for sample in manifest.inputs:
                self.backend.generate_request(self._generation_request(request, sample.input_text))

        entries: list[InferenceBenchmarkEntry] = []
        for repetition in range(1, request.repetitions + 1):
            for sample in manifest.inputs:
                response = self.backend.generate_request(
                    self._generation_request(request, sample.input_text)
                )
                entries.append(
                    InferenceBenchmarkEntry(
                        sample_id=sample.sample_id,
                        repetition=repetition,
                        input_token_count=response.input_token_count or 0,
                        generated_token_count=response.output_token_count,
                        generation_duration_seconds=response.total_latency_ms / 1000,
                        generated_tokens_per_second=response.output_tokens_per_second,
                        peak_process_memory_bytes=response.peak_memory_bytes,
                        cpu_utilization_percent=response.cpu_utilization_percent,
                        time_to_first_token_seconds=None,
                    )
                )

        run_id = uuid.uuid4().hex
        created_at = datetime.now(UTC)
        durations = sorted(entry.generation_duration_seconds for entry in entries)
        p95_index = max(0, (len(durations) * 95 + 99) // 100 - 1) if durations else 0
        measured_throughputs = [
            entry.generated_tokens_per_second
            for entry in entries
            if entry.generated_tokens_per_second is not None
        ]
        result = InferenceBenchmarkResult(
            run_id=run_id,
            created_at=created_at,
            backend_name=self.backend.backend_name,
            model_id=model_status.model_id,
            model_revision=model_status.revision,
            python_version=sys.version.split()[0],
            torch_version=torch.__version__,
            transformers_version=transformers.__version__,
            operating_system=f"{platform.system()} {platform.release()}",
            device="cpu",
            cpu_model=platform.processor() or None,
            cpu_count=os.cpu_count(),
            model_loading_duration_seconds=model_status.loading_duration_seconds,
            measured_requests=len(entries),
            median_generation_duration_seconds=(
                statistics.median(durations) if durations else None
            ),
            p95_generation_duration_seconds=durations[p95_index] if durations else None,
            mean_generated_tokens_per_second=(
                statistics.mean(measured_throughputs) if measured_throughputs else None
            ),
            configuration=request,
            input_sample_ids=manifest.selected_sample_ids,
            input_token_counts=[item.input_token_count or 0 for item in manifest.inputs],
            entries=entries,
            warmup_iterations=request.warmup_iterations,
            warmup_runs_excluded=True,
            json_result_path=f"{run_id}.json",
            csv_result_path=f"{run_id}.csv",
        )
        self._persist(result)
        return result

    @staticmethod
    def _generation_request(
        benchmark: InferenceBenchmarkRequest,
        prompt: str,
    ) -> GenerateRequest:
        return GenerateRequest(
            prompt=prompt,
            max_new_tokens=benchmark.max_new_tokens,
            temperature=benchmark.temperature,
            top_k=benchmark.top_k,
            seed=benchmark.seed,
        )

    def _persist(self, result: InferenceBenchmarkResult) -> None:
        result_dir = self.settings.benchmark_output_dir / "inference"
        result_dir.mkdir(parents=True, exist_ok=True)
        json_path = result_dir / result.json_result_path
        csv_path = result_dir / result.csv_result_path
        created_paths: list[Path] = []
        try:
            with json_path.open("x", encoding="utf-8", newline="\n") as result_file:
                created_paths.append(json_path)
                result_file.write(result.model_dump_json(indent=2) + "\n")
            with csv_path.open("x", encoding="utf-8", newline="") as result_file:
                created_paths.append(csv_path)
                writer = csv.DictWriter(result_file, fieldnames=_CSV_FIELDS)
                writer.writeheader()
                for entry in result.entries:
                    row = entry.model_dump()
                    row.update(
                        {
                            "run_id": result.run_id,
                            "created_at": result.created_at.isoformat(),
                            "backend_name": result.backend_name,
                            "model_id": result.model_id,
                            "model_revision": result.model_revision,
                            "device": result.device,
                        }
                    )
                    writer.writerow(row)
        except OSError:
            for created_path in created_paths:
                try:
                    created_path.unlink(missing_ok=True)
                except OSError:
                    logger.exception("Could not clean incomplete benchmark file")
            raise

    def list_results(self, limit: int = 50) -> list[InferenceBenchmarkResult]:
        if not 1 <= limit <= 500:
            raise ValueError("Result limit must be between 1 and 500.")
        result_dir = self.settings.benchmark_output_dir / "inference"
        if not result_dir.is_dir():
            return []
        results: list[InferenceBenchmarkResult] = []
        for path in sorted(
            result_dir.glob("*.json"),
            key=lambda item: item.stat().st_mtime,
            reverse=True,
        )[:limit]:
            try:
                results.append(
                    InferenceBenchmarkResult.model_validate_json(path.read_text(encoding="utf-8"))
                )
            except (OSError, ValueError):
                logger.warning("Skipping invalid benchmark result %s", path.name)
        return results
