"""Dataset discovery, loading, statistics, and pagination endpoints."""

from typing import Annotated

from fastapi import APIRouter, Query, Request

from app.core.exceptions import ApplicationError
from app.datasets.base import DatasetLoadError, DatasetNotFoundError
from app.datasets.registry import DatasetRegistry
from app.datasets.schemas import (
    DatasetInfo,
    DatasetLoadRequest,
    DatasetLoadResponse,
    DatasetMetadata,
    DatasetPage,
    DatasetStatistics,
    LengthCategory,
    TokenizationStatistics,
)
from app.tokenization.base import TokenizerUnavailableError
from app.tokenization.registry import TokenizerRegistry

router = APIRouter(prefix="/datasets", tags=["datasets"])


def _registry(request: Request) -> DatasetRegistry:
    return request.app.state.dataset_registry


def _not_found(dataset_name: str) -> ApplicationError:
    return ApplicationError(
        code="dataset_not_found",
        message=f"Dataset {dataset_name!r} is not registered.",
        status_code=404,
    )


@router.get("", response_model=list[DatasetInfo])
async def list_datasets(request: Request) -> list[DatasetInfo]:
    return _registry(request).list_datasets()


@router.get("/{dataset_name}", response_model=DatasetMetadata)
async def get_dataset_metadata(dataset_name: str, request: Request) -> DatasetMetadata:
    try:
        return _registry(request).get_metadata(dataset_name)
    except DatasetNotFoundError as error:
        raise _not_found(dataset_name) from error
    except DatasetLoadError as error:
        raise ApplicationError("dataset_metadata_invalid", str(error), status_code=422) from error


@router.post("/{dataset_name}/load", response_model=DatasetLoadResponse)
async def load_dataset(
    dataset_name: str,
    request: Request,
    body: DatasetLoadRequest,
) -> DatasetLoadResponse:
    registry = _registry(request)
    try:
        collection = registry.load(dataset_name, body.limit)
    except DatasetNotFoundError as error:
        raise _not_found(dataset_name) from error
    except DatasetLoadError as error:
        raise ApplicationError("dataset_load_failed", str(error), status_code=503) from error
    except ValueError as error:
        raise ApplicationError("invalid_dataset_limit", str(error), status_code=422) from error
    return DatasetLoadResponse(
        metadata=collection.metadata,
        loaded_sample_count=len(collection.samples),
    )


@router.get("/{dataset_name}/samples", response_model=DatasetPage)
async def list_dataset_samples(
    dataset_name: str,
    request: Request,
    offset: Annotated[int, Query(ge=0)] = 0,
    limit: Annotated[int, Query(ge=1, le=500)] = 50,
    category: str | None = None,
    length_category: LengthCategory | None = None,
) -> DatasetPage:
    try:
        collection = _registry(request).get_loaded(dataset_name)
    except DatasetNotFoundError as error:
        raise _not_found(dataset_name) from error
    except DatasetLoadError as error:
        raise ApplicationError("dataset_not_loaded", str(error), status_code=409) from error
    samples = DatasetRegistry.filter_samples(
        collection.samples,
        categories={category} if category is not None else None,
        length_categories=({length_category} if length_category is not None else None),
    )
    return DatasetPage(
        dataset_name=dataset_name,
        total_samples=len(samples),
        offset=offset,
        limit=limit,
        samples=samples[offset : offset + limit],
    )


@router.get("/{dataset_name}/statistics", response_model=DatasetStatistics)
async def get_dataset_statistics(dataset_name: str, request: Request) -> DatasetStatistics:
    try:
        collection = _registry(request).get_loaded(dataset_name)
    except DatasetNotFoundError as error:
        raise _not_found(dataset_name) from error
    except DatasetLoadError as error:
        raise ApplicationError("dataset_not_loaded", str(error), status_code=409) from error
    return DatasetRegistry.statistics(collection)


@router.get(
    "/{dataset_name}/tokenization-statistics",
    response_model=TokenizationStatistics,
)
async def get_tokenization_statistics(
    dataset_name: str,
    request: Request,
    tokenizer_name: str = "simple",
) -> TokenizationStatistics:
    try:
        collection = _registry(request).get_loaded(dataset_name)
        tokenizer_registry: TokenizerRegistry = request.app.state.tokenizer_registry
        tokenizer = tokenizer_registry.get(tokenizer_name)
    except DatasetNotFoundError as error:
        raise _not_found(dataset_name) from error
    except DatasetLoadError as error:
        raise ApplicationError("dataset_not_loaded", str(error), status_code=409) from error
    except TokenizerUnavailableError as error:
        raise ApplicationError("tokenizer_unavailable", str(error), status_code=503) from error
    return tokenizer_registry.statistics(
        collection.metadata.dataset_name,
        collection.samples,
        tokenizer,
    )
