"""Run one explicit CPU inference request with the configured model."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.core.config import Settings
from app.inference.backends.transformers_backend import TransformersBackend
from app.inference.exceptions import InferenceError
from app.inference.schemas import GenerateRequest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--model-id", default=None)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--allow-download", action="store_true")
    args = parser.parse_args()
    backend = TransformersBackend(Settings())
    try:
        backend.load_model(
            args.model_id or backend.settings.inference_model_id,
            allow_download=args.allow_download,
        )
        response = backend.generate_request(
            GenerateRequest(
                prompt=args.prompt,
                max_new_tokens=args.max_new_tokens,
            )
        )
        print(response.model_dump_json(indent=2))
    except (InferenceError, ValueError) as error:
        print(f"Inference failed: {error}", file=sys.stderr)
        return 1
    finally:
        backend.unload_model()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
