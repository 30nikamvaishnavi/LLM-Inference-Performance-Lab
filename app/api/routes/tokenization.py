"""Text tokenization and reproducible benchmark-input endpoints."""

from fastapi import APIRouter, Request

from app.benchmarking.dataset_preparation import prepare_benchmark_inputs
from app.core.exceptions import ApplicationError
from app.datasets.base import DatasetLoadError, DatasetNotFoundError
from app.datasets.registry import DatasetRegistry
from app.datasets.schemas import (
    BenchmarkManifest,
    PrepareBenchmarkRequest,
    TokenizeTextRequest,
    TokenizeTextResponse,
)
from app.tokenization.base import TokenizerUnavailableError
from app.tokenization.registry import TokenizerRegistry

router = APIRouter(tags=["tokenization", "benchmark inputs"])


@router.post("/tokenize", response_model=TokenizeTextResponse)
async def tokenize_text(body: TokenizeTextRequest, request: Request) -> TokenizeTextResponse:
    registry: TokenizerRegistry = request.app.state.tokenizer_registry
    try:
        tokenizer = registry.get(body.tokenizer_name, allow_download=body.allow_download)
    except TokenizerUnavailableError as error:
        raise ApplicationError("tokenizer_unavailable", str(error), status_code=503) from error
    except ValueError as error:
        raise ApplicationError("invalid_tokenizer", str(error), status_code=422) from error
    return TokenizeTextResponse.model_validate(registry.tokenize_text(tokenizer, body.text))


@router.post("/benchmarks/prepare-inputs", response_model=BenchmarkManifest)
async def prepare_inputs(body: PrepareBenchmarkRequest, request: Request) -> BenchmarkManifest:
    dataset_registry: DatasetRegistry = request.app.state.dataset_registry
    tokenizer_registry: TokenizerRegistry = request.app.state.tokenizer_registry
    try:
        dataset_registry.get_loaded(body.dataset_name)
        tokenizer = (
            tokenizer_registry.get(body.tokenizer_name) if body.tokenizer_name is not None else None
        )
        settings = request.app.state.settings
        return prepare_benchmark_inputs(
            dataset_registry,
            body,
            settings.benchmark_output_dir / "manifests",
            tokenizer,
        )
    except DatasetNotFoundError as error:
        raise ApplicationError(
            "dataset_not_found",
            f"Dataset {body.dataset_name!r} is not registered.",
            status_code=404,
        ) from error
    except DatasetLoadError as error:
        raise ApplicationError("dataset_not_loaded", str(error), status_code=409) from error
    except TokenizerUnavailableError as error:
        raise ApplicationError("tokenizer_unavailable", str(error), status_code=503) from error
    except ValueError as error:
        raise ApplicationError("invalid_benchmark_inputs", str(error), status_code=422) from error
