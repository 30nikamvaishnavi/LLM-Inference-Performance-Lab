"""Integration tests for the initial FastAPI surfaces."""

from fastapi.testclient import TestClient
from pydantic import BaseModel

from app.core.exceptions import ApplicationError
from app.core.schemas import (
    BackendCapabilities,
    InferenceRequest,
    InferenceResponse,
)
from app.inference.base import InferenceBackend
from app.inference.registry import BackendRegistry
from app.main import create_app


class AvailableBackend(InferenceBackend):
    @property
    def backend_name(self) -> str:
        return "test-backend"

    def is_available(self) -> bool:
        return True

    def get_capabilities(self) -> BackendCapabilities:
        return BackendCapabilities(supported_devices=["cpu"])

    def load_model(self, model_name: str, **kwargs: object) -> None:
        del model_name, kwargs

    def generate(self, request: InferenceRequest) -> InferenceResponse:
        del request
        return InferenceResponse(
            model_name="test-model",
            backend_name=self.backend_name,
            generated_text="test",
            output_token_count=1,
            total_latency_ms=1.0,
        )

    def unload_model(self) -> None:
        pass


def test_health_endpoint() -> None:
    client = TestClient(create_app(BackendRegistry()))

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok", "version": "0.1.0"}


def test_backend_discovery_reports_registered_backend() -> None:
    registry = BackendRegistry()
    registry.register(AvailableBackend())
    client = TestClient(create_app(registry))

    response = client.get("/backends")

    assert response.status_code == 200
    assert response.json() == [
        {
            "backend_name": "test-backend",
            "available": True,
            "capabilities": {
                "supported_devices": ["cpu"],
                "features": [],
                "supports_time_to_first_token": False,
                "supports_concurrent_requests": False,
            },
            "unavailable_reason": None,
        }
    ]


def test_system_information_is_explicit_about_unmeasured_hardware() -> None:
    client = TestClient(create_app(BackendRegistry()))

    response = client.get("/system/info")

    assert response.status_code == 200
    result = response.json()
    assert result["operating_system"]
    assert result["python_version"]
    assert result["cpu_count"] is None or result["cpu_count"] >= 1
    assert result["memory_total_bytes"] is None
    assert result["gpu_devices"] is None


def test_request_validation_error_is_structured_without_echoing_input() -> None:
    api = create_app(BackendRegistry())

    class ValidationProbe(BaseModel):
        value: int

    @api.post("/validation-probe")
    async def validation_probe(payload: ValidationProbe) -> dict[str, int]:
        return {"value": payload.value}

    client = TestClient(api)
    response = client.post(
        "/validation-probe",
        json={"value": "sensitive-input"},
    )

    assert response.status_code == 422
    result = response.json()
    assert result["error"]["code"] == "validation_error"
    assert result["error"]["details"][0]["location"] == ["body", "value"]
    assert "sensitive-input" not in response.text


def test_application_error_is_returned_as_structured_response() -> None:
    api = create_app(BackendRegistry())

    @api.get("/application-error-probe")
    async def application_error_probe() -> None:
        raise ApplicationError(
            code="backend_unavailable",
            message="The selected backend is unavailable.",
            status_code=503,
        )

    response = TestClient(api).get("/application-error-probe")

    assert response.status_code == 503
    assert response.json() == {
        "error": {
            "code": "backend_unavailable",
            "message": "The selected backend is unavailable.",
        }
    }


def test_dataset_discovery_metadata_loading_samples_and_statistics(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("LLM_LAB_DATA_DIR", str(tmp_path / "data"))
    client = TestClient(create_app(BackendRegistry()))

    listed = client.get("/datasets")
    metadata = client.get("/datasets/synthetic-benchmark")
    load = client.post(
        "/datasets/synthetic-benchmark/load",
        json={"limit": 10},
    )
    page = client.get("/datasets/synthetic-benchmark/samples?offset=2&limit=3")
    stats = client.get("/datasets/synthetic-benchmark/statistics")
    token_stats = client.get("/datasets/synthetic-benchmark/tokenization-statistics")

    assert listed.status_code == 200
    names = {item["metadata"]["dataset_name"] for item in listed.json()}
    assert {"synthetic-benchmark", "wikitext-2"} <= names
    assert metadata.json()["source"].startswith("Original synthetic")
    assert load.json()["loaded_sample_count"] == 10
    assert page.json()["total_samples"] == 10
    assert len(page.json()["samples"]) == 3
    assert stats.json()["sample_count"] == 10
    assert token_stats.json()["sample_count"] == 10


def test_tokenize_and_prepare_benchmark_inputs_api(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("LLM_LAB_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("LLM_LAB_BENCHMARK_OUTPUT_DIR", str(tmp_path / "benchmarks"))
    client = TestClient(create_app(BackendRegistry()))

    tokenized = client.post(
        "/tokenize",
        json={"text": "measure token speed", "tokenizer_name": "simple"},
    )
    client.post(
        "/datasets/synthetic-benchmark/load",
        json={"limit": 5},
    )
    prepared = client.post(
        "/benchmarks/prepare-inputs",
        json={
            "dataset_name": "synthetic-benchmark",
            "max_samples": 3,
            "seed": 77,
            "tokenizer_name": "simple",
        },
    )

    assert tokenized.status_code == 200
    assert tokenized.json()["token_count"] == 3
    assert tokenized.json()["tokens_per_second"] is not None
    assert prepared.status_code == 200
    assert len(prepared.json()["inputs"]) == 3
    assert all(item["input_token_count"] is not None for item in prepared.json()["inputs"])
    assert (tmp_path / "benchmarks" / "manifests" / "synthetic-benchmark-seed-77.json").is_file()


def test_dataset_api_errors_are_structured_and_do_not_accept_paths(tmp_path) -> None:
    client = TestClient(create_app(BackendRegistry()))

    unknown = client.post(
        "/datasets/../../outside/load",
        json={"limit": 1},
    )
    not_loaded = client.get("/datasets/synthetic-benchmark/statistics")

    assert unknown.status_code in {404, 405}
    assert not_loaded.status_code == 409
    assert not_loaded.json()["error"]["code"] == "dataset_not_loaded"
