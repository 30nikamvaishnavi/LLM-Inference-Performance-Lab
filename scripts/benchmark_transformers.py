"""Run a real repeated benchmark over the loaded synthetic corpus."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.benchmarking.inference_runner import InferenceBenchmarkRunner
from app.core.config import Settings
from app.datasets.registry import DatasetRegistry
from app.inference.backends.transformers_backend import TransformersBackend
from app.inference.exceptions import InferenceError
from app.inference.schemas import InferenceBenchmarkRequest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="synthetic-benchmark")
    parser.add_argument("--samples", type=int, default=3)
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--repetitions", type=int, default=2)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--allow-download", action="store_true")
    args = parser.parse_args()

    settings = Settings()
    backend = TransformersBackend(settings)
    datasets = DatasetRegistry(settings.data_dir, settings.dataset_max_samples)
    try:
        backend.load_model(
            settings.inference_model_id,
            allow_download=args.allow_download,
        )
        datasets.load(args.dataset, limit=args.samples)
        result = InferenceBenchmarkRunner(datasets, backend, settings).run(
            InferenceBenchmarkRequest(
                dataset_name=args.dataset,
                max_samples=args.samples,
                max_new_tokens=args.max_new_tokens,
                repetitions=args.repetitions,
                warmup_iterations=args.warmups,
                seed=args.seed,
            )
        )
        print(result.model_dump_json(indent=2))
        print(f"JSON: {settings.benchmark_output_dir / 'inference' / result.json_result_path}")
        print(f"CSV: {settings.benchmark_output_dir / 'inference' / result.csv_result_path}")
    except (InferenceError, ValueError) as error:
        print(f"Benchmark failed: {error}", file=sys.stderr)
        return 1
    finally:
        backend.unload_model()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
