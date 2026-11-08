"""Explicitly load/cache the educational tiny GPT-2 model, then release it."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.core.config import Settings
from app.inference.backends.transformers_backend import TransformersBackend
from app.inference.exceptions import InferenceError


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Explicitly download/load sshleifer/tiny-gpt2 into the configured "
            "Hugging Face cache, then unload it."
        )
    )
    parser.add_argument("--model-id", default=None)
    args = parser.parse_args()
    backend = TransformersBackend(Settings())
    model_id = args.model_id or backend.settings.inference_model_id
    try:
        print(f"Explicit model load requested: {model_id}")
        backend.load_model(model_id, allow_download=True)
        print(f"Model loaded on CPU: {backend.model_status().model_id}")
        print(f"Resolved revision: {backend.model_status().revision or 'unresolved'}")
        print("Tokenizer and model files are now in the Hugging Face cache.")
    except InferenceError as error:
        print(f"Model load failed: {error}", file=sys.stderr)
        return 1
    finally:
        backend.unload_model()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
