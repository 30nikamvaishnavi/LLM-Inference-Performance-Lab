"""Run real CPU numerical experiments and persist JSON/CSV results."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.core.config import Settings
from app.numerical_lab.common import NumericalExperimentError
from app.numerical_lab.runner import NumericalExperimentRunner
from app.numerical_lab.schemas import ExperimentRunRequest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--experiment",
        choices=(
            "all",
            "precision",
            "matrix_multiplication",
            "numerical_stability",
            "reproducibility",
        ),
        default="all",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--matrix-sizes", type=int, nargs="+", default=[64, 256, 512])
    args = parser.parse_args()
    try:
        request = ExperimentRunRequest(
            experiment_name=args.experiment,
            seed=args.seed,
            warmup_iterations=args.warmups,
            iterations=args.iterations,
            matrix_sizes=args.matrix_sizes,
        )
        result = NumericalExperimentRunner(Settings()).run(request)
    except (NumericalExperimentError, ValueError) as error:
        print(f"Experiment run failed: {error}", file=sys.stderr)
        return 1
    output_dir = Settings().benchmark_output_dir / "numerical"
    print(result.model_dump_json(indent=2))
    print(f"JSON: {output_dir / result.json_result_path}")
    print(f"CSV: {output_dir / result.csv_result_path}")
    has_failed_operation = any(item.status == "failed" for item in result.measurements)
    return 1 if has_failed_operation else 0


if __name__ == "__main__":
    raise SystemExit(main())
