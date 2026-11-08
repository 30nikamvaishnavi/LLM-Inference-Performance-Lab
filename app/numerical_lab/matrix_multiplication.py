"""CPU matrix multiplication benchmarks across floating-point dtypes."""

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
_MAX_MATRIX_SIZE = 512


def run_matrix_multiplication(
    *,
    seed: int = 42,
    sizes: list[int] | None = None,
    warmup_iterations: int = 2,
    iterations: int = 5,
) -> list[ExperimentMeasurement]:
    """Benchmark square CPU matmul while comparing each output to float64."""
    selected_sizes = [64, 256, 512] if sizes is None else sizes
    if seed < 0 or warmup_iterations < 0 or iterations < 1:
        raise ValueError("Seed and warm-ups must be non-negative; iterations must be positive.")
    if (
        not selected_sizes
        or len(selected_sizes) > 3
        or len(set(selected_sizes)) != len(selected_sizes)
    ):
        raise ValueError("Provide one to three unique matrix sizes.")
    if any(size < 16 or size > _MAX_MATRIX_SIZE for size in selected_sizes):
        raise ValueError(f"Matrix sizes must be between 16 and {_MAX_MATRIX_SIZE}.")

    torch = require_torch()
    measurements: list[ExperimentMeasurement] = []
    for size in selected_sizes:
        generator = torch.Generator(device="cpu").manual_seed(seed + size)
        left64 = torch.randn((size, size), generator=generator, dtype=torch.float64, device="cpu")
        right64 = torch.randn((size, size), generator=generator, dtype=torch.float64, device="cpu")
        reference = left64 @ right64
        for dtype_name in _DTYPES:
            dtype = getattr(torch, dtype_name)
            try:
                left = left64.to(dtype=dtype)
                right = right64.to(dtype=dtype)
            except Exception as error:
                logger.warning(
                    "Could not prepare matmul inputs for size %d dtype %s (%s): %s",
                    size,
                    dtype_name,
                    type(error).__name__,
                    error,
                )
                measurements.append(
                    failure_measurement(
                        operation="matrix_multiplication",
                        dtype=dtype_name,
                        dimensions=[size, size, size],
                        timed=TimedExecution([], 0, error=error),
                    )
                )
                continue

            def operation(left_tensor: Any = left, right_tensor: Any = right) -> Any:
                return left_tensor @ right_tensor

            timed = time_operation(
                operation,
                warmup_iterations,
                iterations,
                label=f"matrix_multiplication/{size}/{dtype_name}",
            )
            if timed.error is not None:
                measurements.append(
                    failure_measurement(
                        operation="matrix_multiplication",
                        dtype=dtype_name,
                        dimensions=[size, size, size],
                        timed=timed,
                    )
                )
                continue
            errors = compare_to_reference(torch, timed.result, reference)
            measurements.append(
                ExperimentMeasurement(
                    operation="matrix_multiplication",
                    status="measured",
                    dtype=dtype_name,
                    dimensions=[size, size, size],
                    warmup_iterations=timed.warmups_completed,
                    measured_iterations=len(timed.durations_seconds),
                    **latency_summary(timed.durations_seconds),
                    max_absolute_error=errors["max_absolute_error"],
                    mean_absolute_error=errors["mean_absolute_error"],
                    relative_error_l2=errors["relative_error_l2"],
                    non_finite_value_count=errors["non_finite_value_count"],
                    metrics={
                        "left_shape": f"{size}x{size}",
                        "right_shape": f"{size}x{size}",
                        "reference_dtype": "float64",
                    },
                )
            )
    logger.info("Completed %d matrix multiplication measurements", len(measurements))
    return measurements
