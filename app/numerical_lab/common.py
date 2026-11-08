"""Shared CPU-only timing, environment, and numerical comparison helpers."""

import importlib
import logging
import math
import os
import platform
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from app.numerical_lab.schemas import (
    ExperimentEnvironment,
    ExperimentMeasurement,
    MeasurementStatus,
)

logger = logging.getLogger(__name__)
_P95_MINIMUM_SAMPLES = 20


class NumericalExperimentError(RuntimeError):
    """Raised when an experiment cannot be initialized or persisted."""


@dataclass(frozen=True)
class TimedExecution:
    """Observed timing and any actual execution failure."""

    durations_seconds: list[float]
    warmups_completed: int
    result: Any | None = None
    error: Exception | None = None


def require_torch() -> Any:
    """Import PyTorch only when an experiment is explicitly run."""
    try:
        return importlib.import_module("torch")
    except (ImportError, OSError) as error:
        raise NumericalExperimentError(
            "PyTorch is unavailable. Install the optional CPU-compatible "
            "'pytorch' dependency in the project environment."
        ) from error


def environment_for(torch: Any) -> ExperimentEnvironment:
    """Capture runtime and host information for a CPU experiment."""
    return ExperimentEnvironment(
        python_version=sys.version.split()[0],
        torch_version=str(torch.__version__),
        device="cpu",
        operating_system=f"{platform.system()} {platform.release()}",
        cpu_model=platform.processor() or None,
        cpu_count=os.cpu_count(),
    )


def time_operation(
    operation: Callable[[], Any],
    warmup_iterations: int,
    iterations: int,
    *,
    label: str,
) -> TimedExecution:
    """Run warm-ups and repeated CPU timings, preserving any failure."""
    durations: list[float] = []
    warmups_completed = 0
    result: Any | None = None
    try:
        for _ in range(warmup_iterations):
            result = operation()
            warmups_completed += 1
        for _ in range(iterations):
            started = time.perf_counter()
            result = operation()
            durations.append(time.perf_counter() - started)
    except Exception as error:
        logger.warning(
            "Numerical operation %s failed (%s): %s",
            label,
            type(error).__name__,
            error,
        )
        return TimedExecution(durations, warmups_completed, result, error)
    return TimedExecution(durations, warmups_completed, result)


def classify_failure(error: Exception) -> MeasurementStatus:
    """Distinguish explicit unsupported-operation errors from other failures."""
    message = str(error).casefold()
    if isinstance(error, NotImplementedError) or any(
        phrase in message
        for phrase in ("not implemented", "not supported", "unsupported dtype", "not available")
    ):
        return "unsupported"
    return "failed"


def latency_summary(durations: list[float]) -> dict[str, float | None]:
    """Calculate measured mean/median and only report p95 with enough samples."""
    if not durations:
        return {
            "mean_latency_seconds": None,
            "median_latency_seconds": None,
            "p95_latency_seconds": None,
        }
    ordered = sorted(durations)
    p95 = (
        ordered[math.ceil(0.95 * len(ordered)) - 1]
        if len(ordered) >= _P95_MINIMUM_SAMPLES
        else None
    )
    return {
        "mean_latency_seconds": sum(ordered) / len(ordered),
        "median_latency_seconds": (
            ordered[len(ordered) // 2]
            if len(ordered) % 2
            else (ordered[len(ordered) // 2 - 1] + ordered[len(ordered) // 2]) / 2
        ),
        "p95_latency_seconds": p95,
    }


def compare_to_reference(torch: Any, actual: Any, reference: Any) -> dict[str, float | int | None]:
    """Measure absolute and relative L2 deviations after float64 conversion."""
    actual64 = actual.to(dtype=torch.float64)
    reference64 = reference.to(dtype=torch.float64)
    difference = (actual64 - reference64).abs()
    finite_mask = torch.isfinite(difference)
    finite_count = int(finite_mask.sum().item())
    non_finite_count = int(difference.numel()) - finite_count
    if non_finite_count:
        return {
            "max_absolute_error": None,
            "mean_absolute_error": None,
            "relative_error_l2": None,
            "non_finite_value_count": non_finite_count,
        }
    max_error = float(difference.max().item()) if difference.numel() else 0.0
    mean_error = float(difference.mean().item()) if difference.numel() else 0.0
    reference_norm = float(torch.linalg.vector_norm(reference64).item())
    difference_norm = float(torch.linalg.vector_norm(difference).item())
    relative = difference_norm / reference_norm if reference_norm else difference_norm
    return {
        "max_absolute_error": max_error,
        "mean_absolute_error": mean_error,
        "relative_error_l2": relative,
        "non_finite_value_count": 0,
    }


def failure_measurement(
    *,
    operation: str,
    dtype: str | None,
    dimensions: list[int],
    timed: TimedExecution,
) -> ExperimentMeasurement:
    """Build a transparent unsupported/failed record from a real exception."""
    if timed.error is None:
        raise ValueError("A failure measurement requires an execution exception.")
    status = classify_failure(timed.error)
    return ExperimentMeasurement(
        operation=operation,
        status=status,
        dtype=dtype,
        dimensions=dimensions,
        warmup_iterations=timed.warmups_completed,
        measured_iterations=len(timed.durations_seconds),
        **latency_summary(timed.durations_seconds),
        reason=f"{type(timed.error).__name__}: {timed.error}",
    )
