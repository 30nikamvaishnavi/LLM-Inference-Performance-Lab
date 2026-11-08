"""Small, safe demonstrations of floating-point numerical stability."""

import logging

from app.numerical_lab.common import require_torch
from app.numerical_lab.schemas import ExperimentMeasurement

logger = logging.getLogger(__name__)


def run_numerical_stability(*, seed: int = 42) -> list[ExperimentMeasurement]:
    """Measure accumulation, cancellation, range, and stable-softmax behavior."""
    if seed < 0:
        raise ValueError("Seed must be non-negative.")
    torch = require_torch()

    repeated32 = torch.full((1000,), 0.1, dtype=torch.float32, device="cpu")
    repeated64 = torch.full((1000,), 0.1, dtype=torch.float64, device="cpu")
    sum32 = float(repeated32.sum().item())
    sum64 = float(repeated64.sum().item())

    large32 = torch.tensor(100_000_000.0, dtype=torch.float32)
    unit32 = torch.tensor(1.0, dtype=torch.float32)
    cancellation32 = float(((large32 + unit32) - large32).item())
    cancellation64 = float(
        (
            torch.tensor(100_000_000.0, dtype=torch.float64)
            + torch.tensor(1.0, dtype=torch.float64)
            - torch.tensor(100_000_000.0, dtype=torch.float64)
        ).item()
    )
    increment_lost = float((large32 + unit32).item()) == float(large32.item())

    measurements = [
        ExperimentMeasurement(
            operation="accumulation_and_cancellation",
            status="measured",
            dtype="float32/float64",
            dimensions=[1000],
            warmup_iterations=0,
            measured_iterations=1,
            metrics={
                "float32_sum_of_1000_tenths": sum32,
                "float64_sum_of_1000_tenths": sum64,
                "float32_sum_absolute_error_from_100": abs(sum32 - 100.0),
                "float32_cancellation_result": cancellation32,
                "float64_cancellation_reference": cancellation64,
                "float32_loses_unit_increment_at_1e8": increment_lost,
            },
        )
    ]
    try:
        half = torch.tensor(60_000.0, dtype=torch.float16, device="cpu")
        half_overflow = half * 2
        underflow_half = torch.tensor(1e-8, dtype=torch.float16, device="cpu")
        underflow_float = torch.tensor(1e-8, dtype=torch.float32, device="cpu")
        measurements.append(
            ExperimentMeasurement(
                operation="overflow_and_underflow",
                status="measured",
                dtype="float16/float32",
                dimensions=[],
                warmup_iterations=0,
                measured_iterations=1,
                metrics={
                    "float16_60000_times_2_is_infinite": bool(torch.isinf(half_overflow).item()),
                    "float16_1e-8_underflow_value": float(underflow_half.item()),
                    "float32_1e-8_value": float(underflow_float.item()),
                },
            )
        )
    except Exception as error:
        from app.numerical_lab.common import classify_failure

        measurements.append(
            ExperimentMeasurement(
                operation="overflow_and_underflow",
                status=classify_failure(error),
                dtype="float16/float32",
                dimensions=[],
                warmup_iterations=0,
                measured_iterations=0,
                reason=f"{type(error).__name__}: {error}",
            )
        )

    try:
        logits = torch.tensor([1000.0, 1001.0, 1002.0], dtype=torch.float32, device="cpu")
        exponentials = torch.exp(logits)
        naive_softmax = exponentials / exponentials.sum()
        stable_softmax = torch.softmax(logits, dim=0)
        measurements.append(
            ExperimentMeasurement(
                operation="softmax_numerical_stability",
                status="measured",
                dtype="float32",
                dimensions=[3],
                warmup_iterations=0,
                measured_iterations=1,
                metrics={
                    "naive_softmax_all_finite": bool(torch.isfinite(naive_softmax).all().item()),
                    "stable_softmax_all_finite": bool(torch.isfinite(stable_softmax).all().item()),
                    "stable_softmax_sum": float(stable_softmax.sum().item()),
                    "stable_softmax_max_probability": float(stable_softmax.max().item()),
                    "softmax_logits_max": float(logits.max().item()),
                },
            )
        )
    except Exception as error:
        from app.numerical_lab.common import classify_failure

        measurements.append(
            ExperimentMeasurement(
                operation="softmax_numerical_stability",
                status=classify_failure(error),
                dtype="float32",
                dimensions=[3],
                warmup_iterations=0,
                measured_iterations=0,
                reason=f"{type(error).__name__}: {error}",
            )
        )
    logger.info("Completed numerical stability demonstrations")
    return measurements
