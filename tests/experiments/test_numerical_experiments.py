"""Numerical correctness and persistence tests using small real CPU tensors."""

import csv
import json

import pytest

from app.core.config import Settings
from app.numerical_lab import common, matrix_multiplication
from app.numerical_lab.common import TimedExecution
from app.numerical_lab.matrix_multiplication import run_matrix_multiplication
from app.numerical_lab.numerical_stability import run_numerical_stability
from app.numerical_lab.precision import run_precision_comparison
from app.numerical_lab.reproducibility import run_reproducibility_check
from app.numerical_lab.runner import NumericalExperimentRunner
from app.numerical_lab.schemas import ExperimentRunRequest


def test_precision_comparison_measures_supported_dtypes_and_reference_error() -> None:
    pytest.importorskip("torch")

    measurements = run_precision_comparison(seed=11, warmup_iterations=0, iterations=2)

    assert {item.dtype for item in measurements} == {
        "float32",
        "float64",
        "float16",
        "bfloat16",
    }
    assert all(item.status in {"measured", "unsupported", "failed"} for item in measurements)
    float64 = next(item for item in measurements if item.dtype == "float64")
    assert float64.status == "measured"
    assert float64.max_absolute_error == 0
    assert float64.relative_error_l2 == 0
    assert float64.measured_iterations == 2
    assert float64.p95_latency_seconds is None


def test_matmul_records_dimensions_latency_and_reference_deviation() -> None:
    pytest.importorskip("torch")

    measurements = run_matrix_multiplication(
        seed=7,
        sizes=[16],
        warmup_iterations=1,
        iterations=2,
    )

    assert len(measurements) == 4
    assert all(item.dimensions == [16, 16, 16] for item in measurements)
    for measurement in measurements:
        if measurement.status == "measured":
            assert measurement.measured_iterations == 2
            assert measurement.mean_latency_seconds is not None
            assert measurement.max_absolute_error is not None
    float64 = next(item for item in measurements if item.dtype == "float64")
    assert float64.max_absolute_error == 0


def test_unsupported_matmul_dtype_is_reported_without_substitution(monkeypatch) -> None:
    pytest.importorskip("torch")
    original = matrix_multiplication.time_operation
    calls = 0

    def unsupported_first_dtype(operation, warmups, iterations, *, label):
        nonlocal calls
        calls += 1
        if calls == 1:
            return TimedExecution([], 0, error=NotImplementedError("CPU dtype unsupported"))
        return original(operation, warmups, iterations, label=label)

    monkeypatch.setattr(matrix_multiplication, "time_operation", unsupported_first_dtype)
    measurements = run_matrix_multiplication(
        seed=3,
        sizes=[16],
        warmup_iterations=0,
        iterations=1,
    )

    assert measurements[0].status == "unsupported"
    assert "NotImplementedError" in measurements[0].reason
    assert measurements[0].measured_iterations == 0
    assert all(item.status == "measured" for item in measurements[1:])


def test_matrix_size_validation_rejects_excessive_or_empty_inputs() -> None:
    pytest.importorskip("torch")

    with pytest.raises(ValueError, match="between 16 and 512"):
        run_matrix_multiplication(sizes=[1024])
    with pytest.raises(ValueError, match="one to three unique"):
        run_matrix_multiplication(sizes=[])


def test_numerical_stability_demonstrations_capture_actual_values() -> None:
    pytest.importorskip("torch")

    measurements = run_numerical_stability(seed=19)
    by_operation = {item.operation: item for item in measurements}
    accumulation = by_operation["accumulation_and_cancellation"].metrics
    softmax = by_operation["softmax_numerical_stability"].metrics
    range_results = by_operation["overflow_and_underflow"]

    assert accumulation["float32_loses_unit_increment_at_1e8"] is True
    assert accumulation["float32_cancellation_result"] == 0
    assert accumulation["float64_cancellation_reference"] == 1
    assert softmax["naive_softmax_all_finite"] is False
    assert softmax["stable_softmax_all_finite"] is True
    assert softmax["stable_softmax_sum"] == pytest.approx(1.0)
    assert range_results.status in {"measured", "unsupported", "failed"}
    if range_results.status == "measured":
        assert range_results.metrics["float16_60000_times_2_is_infinite"] is True
        assert range_results.metrics["float16_1e-8_underflow_value"] == 0


def test_reproducibility_is_verified_for_seed_and_repeated_matmul() -> None:
    pytest.importorskip("torch")

    result = run_reproducibility_check(seed=2025, warmup_iterations=0, iterations=3)[0]

    assert result.status == "measured"
    assert result.metrics["seed"] == 2025
    assert result.metrics["same_seed_tensors_equal"] is True
    assert result.metrics["different_seed_tensor_differs"] is True
    assert result.metrics["repeated_cpu_matmul_bitwise_equal"] is True
    assert result.measured_iterations == 3


def test_latency_summary_only_reports_p95_with_twenty_samples() -> None:
    assert common.latency_summary([1.0, 3.0])["p95_latency_seconds"] is None
    p95 = common.latency_summary([float(value) for value in range(1, 21)])
    assert p95["p95_latency_seconds"] == 19


def test_missing_pytorch_dependency_has_explicit_error(monkeypatch) -> None:
    def missing_module(name: str):
        raise ImportError(f"{name} intentionally unavailable")

    monkeypatch.setattr(common.importlib, "import_module", missing_module)

    with pytest.raises(common.NumericalExperimentError, match="PyTorch is unavailable"):
        common.require_torch()


def test_runner_persists_unique_json_csv_and_loads_results(tmp_path) -> None:
    pytest.importorskip("torch")
    settings = Settings(
        _env_file=None,
        benchmark_output_dir=tmp_path / "benchmarks",
    )
    runner = NumericalExperimentRunner(settings)
    request = ExperimentRunRequest(
        experiment_name="precision",
        seed=29,
        warmup_iterations=0,
        iterations=1,
        matrix_sizes=[16],
    )

    first = runner.run(request)
    second = runner.run(request)

    assert first.run_id != second.run_id
    for result in (first, second):
        directory = settings.benchmark_output_dir / "numerical"
        json_path = directory / result.json_result_path
        csv_path = directory / result.csv_result_path
        stored = json.loads(json_path.read_text(encoding="utf-8"))
        assert stored["random_seed"] == 29
        assert stored["environment"]["device"] == "cpu"
        with csv_path.open(encoding="utf-8", newline="") as result_file:
            rows = list(csv.DictReader(result_file))
        assert len(rows) == len(result.measurements)
        assert rows[0]["run_id"] == result.run_id
        assert rows[0]["status"] == result.status
        assert rows[0]["measurement_status"] == result.measurements[0].status
    assert len(runner.list_results()) == 2
    assert len(runner.list_summaries()) == 2


def test_experiment_request_validates_limits_and_dimensions() -> None:
    with pytest.raises(ValueError):
        ExperimentRunRequest(matrix_sizes=[16, 16])
    with pytest.raises(ValueError):
        ExperimentRunRequest(matrix_sizes=[1024])
    with pytest.raises(ValueError):
        ExperimentRunRequest(iterations=0)
