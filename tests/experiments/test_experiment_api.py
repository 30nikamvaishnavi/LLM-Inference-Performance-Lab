"""FastAPI tests for numerical experiment execution and result discovery."""

import pytest
from fastapi.testclient import TestClient

from app.inference.registry import BackendRegistry
from app.main import create_app


def test_experiment_discovery_is_available_without_running_computations() -> None:
    client = TestClient(create_app(BackendRegistry()))

    response = client.get("/experiments")

    assert response.status_code == 200
    names = {item["name"] for item in response.json()}
    assert {
        "precision",
        "matrix_multiplication",
        "numerical_stability",
        "reproducibility",
        "all",
    } <= names


def test_experiment_api_runs_and_lists_cpu_result(tmp_path, monkeypatch) -> None:
    pytest.importorskip("torch")
    monkeypatch.setenv("LLM_LAB_BENCHMARK_OUTPUT_DIR", str(tmp_path / "benchmarks"))
    client = TestClient(create_app(BackendRegistry()))

    result = client.post(
        "/experiments/run",
        json={
            "experiment_name": "numerical_stability",
            "seed": 41,
            "warmup_iterations": 0,
            "iterations": 1,
            "matrix_sizes": [16],
        },
    )
    listed = client.get("/experiments/results")
    retrieved = client.get(f"/experiments/results/{result.json()['run_id']}")

    assert result.status_code == 200, result.text
    assert result.json()["status"] == "success"
    assert result.json()["environment"]["device"] == "cpu"
    assert listed.status_code == 200
    assert listed.json()[0]["run_id"] == result.json()["run_id"]
    assert retrieved.status_code == 200
    assert retrieved.json()["measurements"][0]["operation"] == "accumulation_and_cancellation"


def test_experiment_result_retrieval_rejects_paths_and_unknown_ids() -> None:
    client = TestClient(create_app(BackendRegistry()))

    invalid = client.get("/experiments/results/not-a-run-id")
    missing = client.get("/experiments/results/00000000000000000000000000000000")

    assert invalid.status_code == 422
    assert invalid.json()["error"]["code"] == "invalid_experiment_id"
    assert missing.status_code == 404
    assert missing.json()["error"]["code"] == "experiment_result_not_found"


def test_experiment_api_rejects_out_of_range_matrix_size() -> None:
    client = TestClient(create_app(BackendRegistry()))

    response = client.post(
        "/experiments/run",
        json={
            "experiment_name": "matrix_multiplication",
            "matrix_sizes": [1024],
        },
    )

    assert response.status_code == 422
    assert response.json()["error"]["code"] == "validation_error"


def test_experiment_results_limit_is_validated() -> None:
    client = TestClient(create_app(BackendRegistry()))

    response = client.get("/experiments/results?limit=0")

    assert response.status_code == 422
    assert response.json()["error"]["code"] == "invalid_result_limit"
