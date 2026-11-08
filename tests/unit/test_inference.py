"""Tests for inference backend contracts and registry discovery."""

from unittest.mock import Mock

import pytest

from app.core.schemas import (
    BackendCapabilities,
    InferenceRequest,
    InferenceResponse,
)
from app.inference.base import InferenceBackend
from app.inference.registry import BackendRegistry


class SampleBackend(InferenceBackend):
    @property
    def backend_name(self) -> str:
        return "sample"

    def is_available(self) -> bool:
        return True

    def get_capabilities(self) -> BackendCapabilities:
        return BackendCapabilities(supported_devices=["cpu"])

    def load_model(self, model_name: str, **kwargs: object) -> None:
        del model_name, kwargs

    def generate(self, request: InferenceRequest) -> InferenceResponse:
        del request
        return InferenceResponse(
            model_name="sample-model",
            backend_name=self.backend_name,
            generated_text="sample",
            output_token_count=1,
            total_latency_ms=1.0,
        )

    def unload_model(self) -> None:
        pass


class MissingPackageBackend(SampleBackend):
    @property
    def required_packages(self) -> tuple[str, ...]:
        return ("sample_optional_backend_package",)


def test_abstract_backend_cannot_be_instantiated() -> None:
    with pytest.raises(TypeError, match="abstract"):
        InferenceBackend()


def test_registry_reports_missing_optional_package(monkeypatch: pytest.MonkeyPatch) -> None:
    backend = MissingPackageBackend()
    availability_check = Mock(return_value=True)
    monkeypatch.setattr(backend, "is_available", availability_check)
    monkeypatch.setattr("app.inference.registry.find_spec", lambda package: None)
    registry = BackendRegistry()
    registry.register(backend)

    result = registry.list_backends()

    assert len(result) == 1
    assert result[0].available is False
    assert "sample_optional_backend_package" in (result[0].unavailable_reason or "")
    availability_check.assert_not_called()


def test_registry_checks_available_backend_and_rejects_duplicate() -> None:
    registry = BackendRegistry()
    backend = SampleBackend()
    registry.register(backend)

    with pytest.raises(ValueError, match="already registered"):
        registry.register(backend)

    result = registry.list_backends()
    assert result[0].available is True
    assert result[0].capabilities is not None
