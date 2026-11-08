"""Deterministic floating-point precision comparison experiments."""

import logging
from typing import Any

from app.numerical_lab.common import (
    TimedExecution,
    compare_to_reference,
    failure_measurement,
    latency_summary,
    require_torch,
    time_operation,
)
from app.numerical_lab.schemas import ExperimentMeasurement

logger = logging.getLogger(__name__)
_DTYPES = ("float32", "float64", "float16", "bfloat16")


def run_precision_comparison(
    *,
    seed: int = 42,
    warmup_iterations: int = 2,
    iterations: int = 5,
) -> list[ExperimentMeasurement]:
    """Compare a deterministic elementwise workload against float64."""
    if seed < 0 or warmup_iterations < 0 or iterations < 1:
        raise ValueError("Seed and warm-ups must be non-negative; iterations must be positive.")
    torch = require_torch()
    generator = torch.Generator(device="cpu").manual_seed(seed)
    values64 = torch.randn(4096, generator=generator, dtype=torch.float64, device="cpu")
    reference = _workload(values64)
    measurements: list[ExperimentMeasurement] = []
    for dtype_name in _DTYPES:
        try:
            dtype = getattr(torch, dtype_name)
            input_values = values64.to(dtype=dtype)
        except Exception as error:
            logger.warning(
                "Could not prepare precision input for %s (%s): %s",
                dtype_name,
                type(error).__name__,
                error,
            )
            measurements.append(
                failure_measurement(
                    operation="elementwise_precision",
                    dtype=dtype_name,
                    dimensions=[4096],
                    timed=TimedExecution([], 0, error=error),
                )
            )
            continue

        def operation(values: Any = input_values) -> Any:
            return _workload(values)

        timed = time_operation(
            operation,
            warmup_iterations,
            iterations,
            label=f"precision/{dtype_name}",
        )
        if timed.error is not None:
            measurements.append(
                failure_measurement(
                    operation="elementwise_precision",
                    dtype=dtype_name,
                    dimensions=[4096],
                    timed=timed,
                )
            )
            continue
        output = timed.result.to(dtype=torch.float64)
        errors = compare_to_reference(torch, output, reference)
        measurements.append(
            ExperimentMeasurement(
                operation="elementwise_precision",
                status="measured",
                dtype=dtype_name,
                dimensions=[4096],
                warmup_iterations=timed.warmups_completed,
                measured_iterations=len(timed.durations_seconds),
                **latency_summary(timed.durations_seconds),
                max_absolute_error=errors["max_absolute_error"],
                mean_absolute_error=errors["mean_absolute_error"],
                relative_error_l2=errors["relative_error_l2"],
                non_finite_value_count=errors["non_finite_value_count"],
                metrics={"reference_dtype": "float64", "reference_operation": "sin(x)+x^2/7-x/4"},
            )
        )
    logger.info("Completed precision comparison on CPU with seed %d", seed)
    return measurements


def _workload(values: Any) -> Any:
    return values.sin() + values.square() / 7.0 - values / 4.0
