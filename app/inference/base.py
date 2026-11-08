"""Abstract contract implemented by inference backends."""

from abc import ABC, abstractmethod

from app.core.schemas import BackendCapabilities, InferenceRequest, InferenceResponse


class InferenceBackend(ABC):
    """Contract for optional model inference engines."""

    @property
    @abstractmethod
    def backend_name(self) -> str:
        """Return the stable identifier used to select this backend."""

    @property
    def required_packages(self) -> tuple[str, ...]:
        """Return importable top-level package names required by this adapter."""
        return ()

    @abstractmethod
    def is_available(self) -> bool:
        """Check runtime and hardware support after dependencies are present."""

    @abstractmethod
    def get_capabilities(self) -> BackendCapabilities:
        """Describe measured or implemented backend capabilities."""

    @abstractmethod
    def load_model(self, model_name: str, **kwargs: object) -> None:
        """Load a model into this backend."""

    @abstractmethod
    def generate(self, request: InferenceRequest) -> InferenceResponse:
        """Generate a response using a previously loaded model."""

    @abstractmethod
    def unload_model(self) -> None:
        """Release model resources held by this backend."""
