"""Backend registration and runtime availability discovery."""

from importlib.util import find_spec

from app.core.schemas import BackendCapabilities, BackendInfo
from app.inference.base import InferenceBackend


class BackendRegistry:
    """Store backend adapters and report their dependency/runtime availability."""

    def __init__(self) -> None:
        self._backends: dict[str, InferenceBackend] = {}

    def register(self, backend: InferenceBackend) -> None:
        """Register an adapter instance; duplicate names are rejected."""
        name = backend.backend_name.strip()
        if not name:
            raise ValueError("Backend names must not be empty.")
        if name in self._backends:
            raise ValueError(f"Backend {name!r} is already registered.")
        self._backends[name] = backend

    def get_backend(self, name: str) -> InferenceBackend | None:
        """Return a registered adapter by name, if present."""
        return self._backends.get(name)

    def list_backends(self) -> list[BackendInfo]:
        """Return current availability, checking optional imports before adapters."""
        results: list[BackendInfo] = []
        for name, backend in self._backends.items():
            missing_packages = [
                package for package in backend.required_packages if find_spec(package) is None
            ]
            if missing_packages:
                results.append(
                    BackendInfo(
                        backend_name=name,
                        available=False,
                        unavailable_reason=(
                            "Missing optional package(s): " + ", ".join(missing_packages)
                        ),
                    )
                )
                continue

            available = backend.is_available()
            capabilities: BackendCapabilities = backend.get_capabilities()
            reason = None
            if not available:
                reason = "The backend reports that its runtime or hardware is unavailable."
            results.append(
                BackendInfo(
                    backend_name=name,
                    available=available,
                    capabilities=capabilities,
                    unavailable_reason=reason,
                )
            )
        return results
