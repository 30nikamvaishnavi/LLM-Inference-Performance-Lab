"""Experiment orchestration and non-overwriting JSON/CSV result persistence."""

import csv
import json
import logging
import re
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from app.core.config import Settings
from app.numerical_lab.common import NumericalExperimentError, environment_for, require_torch
from app.numerical_lab.matrix_multiplication import run_matrix_multiplication
from app.numerical_lab.numerical_stability import run_numerical_stability
from app.numerical_lab.precision import run_precision_comparison
from app.numerical_lab.reproducibility import run_reproducibility_check
from app.numerical_lab.schemas import (
    ExperimentRunRequest,
    ExperimentSummary,
    NumericalExperimentResult,
)

logger = logging.getLogger(__name__)
_EXPERIMENTS = (
    "precision",
    "matrix_multiplication",
    "numerical_stability",
    "reproducibility",
)
_RUN_ID_PATTERN = re.compile(r"^[0-9a-f]{32}\.json$")
_CSV_FIELDS = (
    "run_id",
    "created_at",
    "experiment_name",
    "status",
    "operation",
    "measurement_status",
    "dtype",
    "dimensions",
    "warmup_iterations",
    "measured_iterations",
    "mean_latency_seconds",
    "median_latency_seconds",
    "p95_latency_seconds",
    "max_absolute_error",
    "mean_absolute_error",
    "relative_error_l2",
    "non_finite_value_count",
    "reason",
    "metrics",
    "device",
    "python_version",
    "torch_version",
    "random_seed",
)


class NumericalExperimentRunner:
    """Run the selected experiment(s) and persist measured observations."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or Settings()

    def run(self, request: ExperimentRunRequest) -> NumericalExperimentResult:
        torch = require_torch()
        names = _EXPERIMENTS if request.experiment_name == "all" else (request.experiment_name,)
        measurements = []
        for name in names:
            if name == "precision":
                measurements.extend(
                    run_precision_comparison(
                        seed=request.seed,
                        warmup_iterations=request.warmup_iterations,
                        iterations=request.iterations,
                    )
                )
            elif name == "matrix_multiplication":
                measurements.extend(
                    run_matrix_multiplication(
                        seed=request.seed,
                        sizes=request.matrix_sizes,
                        warmup_iterations=request.warmup_iterations,
                        iterations=request.iterations,
                    )
                )
            elif name == "numerical_stability":
                measurements.extend(run_numerical_stability(seed=request.seed))
            else:
                measurements.extend(
                    run_reproducibility_check(
                        seed=request.seed,
                        warmup_iterations=request.warmup_iterations,
                        iterations=request.iterations,
                    )
                )
        statuses = {item.status for item in measurements}
        status = (
            "success"
            if statuses == {"measured"}
            else "failed"
            if statuses == {"failed"}
            else "partial"
        )
        run_id = uuid.uuid4().hex
        result = NumericalExperimentResult(
            run_id=run_id,
            created_at=datetime.now(UTC),
            experiment_name=request.experiment_name,
            status=status,
            random_seed=request.seed,
            configuration={
                "warmup_iterations": request.warmup_iterations,
                "iterations": request.iterations,
                "matrix_sizes": request.matrix_sizes,
                "p95_minimum_samples": 20,
            },
            environment=environment_for(torch),
            measurements=measurements,
            json_result_path=f"{run_id}.json",
            csv_result_path=f"{run_id}.csv",
        )
        self._persist(result)
        logger.info(
            "Persisted numerical experiment %s with status %s",
            result.run_id,
            result.status,
        )
        return result

    def _result_dir(self) -> Path:
        return self.settings.benchmark_output_dir / "numerical"

    def _persist(self, result: NumericalExperimentResult) -> None:
        result_dir = self._result_dir()
        result_dir.mkdir(parents=True, exist_ok=True)
        json_path = result_dir / result.json_result_path
        csv_path = result_dir / result.csv_result_path
        created: list[Path] = []
        try:
            with json_path.open("x", encoding="utf-8", newline="\n") as output:
                created.append(json_path)
                output.write(result.model_dump_json(indent=2) + "\n")
            with csv_path.open("x", encoding="utf-8", newline="") as output:
                created.append(csv_path)
                writer = csv.DictWriter(output, fieldnames=_CSV_FIELDS)
                writer.writeheader()
                for measurement in result.measurements:
                    row: dict[str, Any] = measurement.model_dump()
                    row["measurement_status"] = measurement.status
                    row.update(
                        {
                            "run_id": result.run_id,
                            "created_at": result.created_at.isoformat(),
                            "experiment_name": result.experiment_name,
                            "status": result.status,
                            "dimensions": json.dumps(measurement.dimensions),
                            "metrics": json.dumps(measurement.metrics, sort_keys=True),
                            "device": result.environment.device,
                            "python_version": result.environment.python_version,
                            "torch_version": result.environment.torch_version,
                            "random_seed": result.random_seed,
                        }
                    )
                    writer.writerow({field: row.get(field) for field in _CSV_FIELDS})
        except OSError as error:
            for path in created:
                try:
                    path.unlink(missing_ok=True)
                except OSError:
                    logger.exception("Could not clean incomplete numerical result file")
            raise NumericalExperimentError(
                f"Unable to persist numerical experiment results: {error}"
            ) from error

    def list_results(self, limit: int = 50) -> list[NumericalExperimentResult]:
        """Read only UUID-named experiment JSON files from the configured result directory."""
        if not 1 <= limit <= 500:
            raise ValueError("Result limit must be between 1 and 500.")
        result_dir = self._result_dir()
        if not result_dir.is_dir():
            return []
        results: list[NumericalExperimentResult] = []
        paths = sorted(
            (
                path
                for path in result_dir.glob("*.json")
                if _RUN_ID_PATTERN.fullmatch(path.name) and not path.is_symlink()
            ),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )
        for path in paths[:limit]:
            try:
                results.append(
                    NumericalExperimentResult.model_validate_json(path.read_text(encoding="utf-8"))
                )
            except (OSError, ValueError):
                logger.warning("Skipping invalid numerical experiment result %s", path.name)
        return results

    def get_result(self, run_id: str) -> NumericalExperimentResult:
        """Retrieve one result by a strict UUID hex identifier, never by caller-supplied path."""
        if not re.fullmatch(r"[0-9a-f]{32}", run_id):
            raise ValueError("run_id must be a 32-character lowercase hexadecimal identifier.")
        path = self._result_dir() / f"{run_id}.json"
        if path.is_symlink() or not path.is_file():
            raise FileNotFoundError("Numerical experiment result was not found.")
        try:
            return NumericalExperimentResult.model_validate_json(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as error:
            logger.warning("Unable to read numerical experiment result %s", path.name)
            raise NumericalExperimentError("Numerical experiment result is invalid.") from error

    def list_summaries(self, limit: int = 50) -> list[ExperimentSummary]:
        return [
            ExperimentSummary(
                run_id=result.run_id,
                created_at=result.created_at,
                experiment_name=result.experiment_name,
                status=result.status,
                measurement_count=len(result.measurements),
                result_file=result.json_result_path,
            )
            for result in self.list_results(limit)
        ]
