"""Seeded tensor generation and repeated CPU operation reproducibility checks."""

import logging

from app.numerical_lab.common import (
    compare_to_reference,
    failure_measurement,
    latency_summary,
    require_torch,
    time_operation,
)
from app.numerical_lab.schemas import ExperimentMeasurement

logger = logging.getLogger(__name__)


def run_reproducibility_check(
    *,
    seed: int = 42,
    warmup_iterations: int = 2,
    iterations: int = 5,
) -> list[ExperimentMeasurement]:
    """Verify same-seed tensor equality and repeated CPU matmul equality."""
    if seed < 0 or warmup_iterations < 0 or iterations < 1:
        raise ValueError("Seed and warm-ups must be non-negative; iterations must be positive.")
    torch = require_torch()
    first_generator = torch.Generator(device="cpu").manual_seed(seed)
    repeated_generator = torch.Generator(device="cpu").manual_seed(seed)
    different_generator = torch.Generator(device="cpu").manual_seed(seed + 1)
    first = torch.randn((128, 128), generator=first_generator, dtype=torch.float32, device="cpu")
    repeated = torch.randn(
        (128, 128), generator=repeated_generator, dtype=torch.float32, device="cpu"
    )
    different_seed = torch.randn(
        (128, 128), generator=different_generator, dtype=torch.float32, device="cpu"
    )
    right = torch.randn((128, 128), generator=first_generator, dtype=torch.float32, device="cpu")
    reference = first @ right

    def operation() -> object:
        return first @ right

    timed = time_operation(
        operation,
        warmup_iterations,
        iterations,
        label="reproducibility/repeated_cpu_matmul",
    )
    if timed.error is not None:
        return [
            failure_measurement(
                operation="repeated_cpu_matmul",
                dtype="float32",
                dimensions=[128, 128, 128],
                timed=timed,
            )
        ]
    errors = compare_to_reference(torch, timed.result, reference)
    measurement = ExperimentMeasurement(
        operation="seeded_tensor_and_repeated_matmul_reproducibility",
        status="measured",
        dtype="float32",
        dimensions=[128, 128, 128],
        warmup_iterations=timed.warmups_completed,
        measured_iterations=len(timed.durations_seconds),
        **latency_summary(timed.durations_seconds),
        max_absolute_error=errors["max_absolute_error"],
        mean_absolute_error=errors["mean_absolute_error"],
        relative_error_l2=errors["relative_error_l2"],
        non_finite_value_count=errors["non_finite_value_count"],
        metrics={
            "seed": seed,
            "same_seed_tensors_equal": bool(torch.equal(first, repeated)),
            "different_seed_tensor_differs": not bool(torch.equal(first, different_seed)),
            "repeated_cpu_matmul_bitwise_equal": bool(torch.equal(timed.result, reference)),
            "deterministic_algorithms_enabled": bool(torch.are_deterministic_algorithms_enabled()),
            "scope": (
                "verified repeated operation on this CPU/runtime; not a cross-platform guarantee"
            ),
        },
    )
    logger.info("Completed reproducibility check with seed %d", seed)
    return [measurement]
