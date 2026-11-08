"""CPU-only numerical experiment discovery, execution, and results."""

from importlib.util import find_spec

from fastapi import APIRouter, Request

from app.core.exceptions import ApplicationError
from app.numerical_lab.common import NumericalExperimentError
from app.numerical_lab.runner import NumericalExperimentRunner
from app.numerical_lab.schemas import (
    ExperimentInfo,
    ExperimentRunRequest,
    ExperimentSummary,
    NumericalExperimentResult,
)

router = APIRouter(prefix="/experiments", tags=["experiments"])
_EXPERIMENT_INFO = (
    ("precision", "Compare float32, float64, float16, and bfloat16 elementwise accuracy."),
    ("matrix_multiplication", "Benchmark CPU matrix multiplication and compare to float64."),
    ("numerical_stability", "Demonstrate accumulation, cancellation, range, and stable softmax."),
    ("reproducibility", "Verify seeded tensors and repeated CPU operations."),
    ("all", "Run all numerical experiments."),
)


@router.get("", response_model=list[ExperimentInfo])
def list_experiments() -> list[ExperimentInfo]:
    """List experiments and report whether PyTorch can be imported."""
    try:
        torch_available = find_spec("torch") is not None
    except (ImportError, ValueError):
        torch_available = False
    reason = (
        None
        if torch_available
        else "Install the optional PyTorch dependency in the project environment."
    )
    return [
        ExperimentInfo(
            name=name,
            description=description,
            available=torch_available,
            unavailable_reason=reason,
        )
        for name, description in _EXPERIMENT_INFO
    ]


@router.post("/run", response_model=NumericalExperimentResult)
def run_experiments(
    body: ExperimentRunRequest,
    request: Request,
) -> NumericalExperimentResult:
    runner = NumericalExperimentRunner(request.app.state.settings)
    try:
        return runner.run(body)
    except NumericalExperimentError as error:
        raise ApplicationError(
            "numerical_experiment_unavailable",
            str(error),
            status_code=503,
        ) from error
    except ValueError as error:
        raise ApplicationError("invalid_experiment_request", str(error), status_code=422) from error


@router.get("/results", response_model=list[ExperimentSummary])
def list_experiment_results(
    request: Request,
    limit: int = 50,
) -> list[ExperimentSummary]:
    if not 1 <= limit <= 500:
        raise ApplicationError(
            "invalid_result_limit",
            "limit must be between 1 and 500.",
            status_code=422,
        )
    return NumericalExperimentRunner(request.app.state.settings).list_summaries(limit)


@router.get("/results/{run_id}", response_model=NumericalExperimentResult)
def get_experiment_result(run_id: str, request: Request) -> NumericalExperimentResult:
    try:
        return NumericalExperimentRunner(request.app.state.settings).get_result(run_id)
    except ValueError as error:
        raise ApplicationError("invalid_experiment_id", str(error), status_code=422) from error
    except FileNotFoundError as error:
        raise ApplicationError(
            "experiment_result_not_found",
            "Numerical experiment result was not found.",
            status_code=404,
        ) from error
    except NumericalExperimentError as error:
        raise ApplicationError(
            "experiment_result_invalid",
            str(error),
            status_code=500,
        ) from error
