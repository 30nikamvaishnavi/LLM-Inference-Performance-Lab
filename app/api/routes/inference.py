"""Explicit model lifecycle and inference API operations."""

from fastapi import APIRouter, Request

from app.benchmarking.inference_runner import InferenceBenchmarkRunner
from app.core.exceptions import ApplicationError
from app.core.schemas import BackendInfo
from app.inference.backends.transformers_backend import TransformersBackend
from app.inference.exceptions import InferenceError
from app.inference.registry import BackendRegistry
from app.inference.schemas import (
    GenerateRequest,
    GenerateResponse,
    InferenceBenchmarkRequest,
    InferenceBenchmarkResult,
    InferenceBenchmarkSummary,
    ModelLoadRequest,
    ModelLoadResponse,
    ModelStatus,
    ModelTokenizeRequest,
    ModelTokenizeResponse,
)

router = APIRouter(tags=["inference"])


def _backend(request: Request) -> TransformersBackend:
    backend: TransformersBackend = request.app.state.transformers_backend
    return backend


def _translate_inference_error(error: InferenceError) -> ApplicationError:
    message = str(error)
    status_code = 400
    code = "inference_error"
    if "No model is loaded" in message or "Load a model" in message:
        status_code = 409
        code = "model_not_loaded"
    elif "has not been loaded" in message or "Load the benchmark dataset" in message:
        status_code = 409
        code = "dataset_not_loaded"
    elif "not registered" in message:
        status_code = 404
        code = "dataset_not_found"
    elif "Insufficient memory" in message:
        status_code = 507
        code = "insufficient_memory"
    elif "unavailable" in message.casefold() or "install" in message.casefold():
        status_code = 503
        code = "backend_unavailable"
    elif "Unable to load model" in message:
        status_code = 503
        code = "model_load_failed"
    return ApplicationError(code, message, status_code)


@router.get("/inference/backends", response_model=list[BackendInfo])
def list_inference_backends(request: Request) -> list[BackendInfo]:
    registry: BackendRegistry = request.app.state.backend_registry
    return registry.list_backends()


@router.get("/inference/status", response_model=ModelStatus)
def get_model_status(request: Request) -> ModelStatus:
    return _backend(request).model_status()


@router.post("/inference/load", response_model=ModelLoadResponse)
def load_model(
    body: ModelLoadRequest,
    request: Request,
) -> ModelLoadResponse:
    backend = _backend(request)
    settings = request.app.state.settings
    try:
        backend.load_model(
            body.model_id or settings.inference_model_id,
            revision=body.revision,
            allow_download=(body.allow_download and not settings.inference_offline_mode),
        )
    except InferenceError as error:
        raise _translate_inference_error(error) from error
    return ModelLoadResponse(status=backend.model_status())


@router.post("/inference/unload", response_model=ModelStatus)
def unload_model(request: Request) -> ModelStatus:
    backend = _backend(request)
    backend.unload_model()
    return backend.model_status()


@router.post("/inference/generate", response_model=GenerateResponse)
def generate_text(body: GenerateRequest, request: Request) -> GenerateResponse:
    try:
        return _backend(request).generate_request(body)
    except InferenceError as error:
        raise _translate_inference_error(error) from error


@router.post("/inference/tokenize", response_model=ModelTokenizeResponse)
def tokenize_with_model(
    body: ModelTokenizeRequest,
    request: Request,
) -> ModelTokenizeResponse:
    try:
        return _backend(request).tokenize(body)
    except InferenceError as error:
        raise _translate_inference_error(error) from error


@router.post(
    "/benchmarks/run",
    response_model=InferenceBenchmarkResult,
    tags=["benchmarks"],
)
def run_inference_benchmark(
    body: InferenceBenchmarkRequest,
    request: Request,
) -> InferenceBenchmarkResult:
    runner = InferenceBenchmarkRunner(
        request.app.state.dataset_registry,
        _backend(request),
        request.app.state.settings,
    )
    try:
        return runner.run(body)
    except InferenceError as error:
        raise _translate_inference_error(error) from error
    except ValueError as error:
        raise ApplicationError("invalid_benchmark_request", str(error), status_code=422) from error


@router.get(
    "/benchmarks/results",
    response_model=list[InferenceBenchmarkSummary],
    tags=["benchmarks"],
)
def get_inference_benchmark_results(
    request: Request,
    limit: int = 50,
) -> list[InferenceBenchmarkSummary]:
    if not 1 <= limit <= 500:
        raise ApplicationError(
            "invalid_result_limit",
            "limit must be between 1 and 500.",
            status_code=422,
        )
    runner = InferenceBenchmarkRunner(
        request.app.state.dataset_registry,
        _backend(request),
        request.app.state.settings,
    )
    return [
        InferenceBenchmarkSummary(
            run_id=result.run_id,
            created_at=result.created_at,
            backend_name=result.backend_name,
            model_id=result.model_id,
            measured_requests=len(result.entries),
            result_file=result.json_result_path,
        )
        for result in runner.list_results(limit)
    ]
