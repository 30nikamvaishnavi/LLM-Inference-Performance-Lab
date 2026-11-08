"""FastAPI application entry point."""

import logging
import os
import platform
import sys

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from app import __version__
from app.api.routes.datasets import router as datasets_router
from app.api.routes.tokenization import router as tokenization_router
from app.core.config import Settings
from app.core.exceptions import ApplicationError
from app.core.logging_config import configure_logging
from app.core.schemas import BackendInfo, SystemInfo
from app.datasets.registry import DatasetRegistry
from app.inference.registry import BackendRegistry
from app.tokenization.registry import TokenizerRegistry

logger = logging.getLogger(__name__)


def create_app(
    backend_registry: BackendRegistry | None = None,
    dataset_registry: DatasetRegistry | None = None,
    tokenizer_registry: TokenizerRegistry | None = None,
) -> FastAPI:
    """Create the API application, optionally with an injected backend registry."""
    settings = Settings()
    configure_logging(settings.log_level)

    api = FastAPI(
        title="LLM Inference & Performance Benchmarking Lab",
        description=(
            "A CPU-first engineering lab for exploring LLM inference and "
            "reproducible performance measurement."
        ),
        version=__version__,
        debug=settings.debug,
    )
    api.state.backend_registry = backend_registry or BackendRegistry()
    api.state.dataset_registry = dataset_registry or DatasetRegistry(
        settings.data_dir,
        max_samples=settings.dataset_max_samples,
    )
    api.state.tokenizer_registry = tokenizer_registry or TokenizerRegistry()
    api.state.settings = settings
    api.include_router(datasets_router)
    api.include_router(tokenization_router)

    @api.exception_handler(ApplicationError)
    async def application_error_handler(
        request: Request, exception: ApplicationError
    ) -> JSONResponse:
        del request
        return JSONResponse(
            status_code=exception.status_code,
            content={
                "error": {
                    "code": exception.code,
                    "message": exception.message,
                }
            },
        )

    @api.exception_handler(RequestValidationError)
    async def validation_error_handler(
        request: Request, exception: RequestValidationError
    ) -> JSONResponse:
        del request
        return JSONResponse(
            status_code=422,
            content={
                "error": {
                    "code": "validation_error",
                    "message": "Request validation failed.",
                    "details": [
                        {
                            "location": list(error["loc"]),
                            "type": error["type"],
                        }
                        for error in exception.errors()
                    ],
                }
            },
        )

    @api.exception_handler(StarletteHTTPException)
    async def http_error_handler(
        request: Request, exception: StarletteHTTPException
    ) -> JSONResponse:
        del request
        return JSONResponse(
            status_code=exception.status_code,
            content={
                "error": {
                    "code": "http_error",
                    "message": str(exception.detail),
                }
            },
            headers=exception.headers,
        )

    @api.exception_handler(Exception)
    async def unexpected_error_handler(request: Request, exception: Exception) -> JSONResponse:
        del request
        logger.error(
            "Unhandled API exception",
            exc_info=(type(exception), exception, exception.__traceback__),
        )
        return JSONResponse(
            status_code=500,
            content={
                "error": {
                    "code": "internal_server_error",
                    "message": "An unexpected error occurred.",
                }
            },
        )

    @api.get("/health", tags=["system"])
    async def health() -> dict[str, str]:
        return {"status": "ok", "version": __version__}

    @api.get("/backends", response_model=list[BackendInfo], tags=["inference"])
    async def list_backends(
        request: Request,
    ) -> list[BackendInfo]:
        registry: BackendRegistry = request.app.state.backend_registry
        return registry.list_backends()

    @api.get("/system/info", response_model=SystemInfo, tags=["system"])
    async def system_info() -> SystemInfo:
        return SystemInfo(
            operating_system=platform.system(),
            operating_system_version=platform.release(),
            python_version=sys.version.split()[0],
            cpu_count=os.cpu_count(),
        )

    return api


app = create_app()
