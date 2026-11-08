"""Report PyTorch/Transformers availability without loading a model."""

import importlib.util
import sys


def main() -> int:
    if importlib.util.find_spec("torch") is None:
        print("PyTorch is not installed in the selected Python environment.")
        return 1
    try:
        import torch
    except (ImportError, OSError) as error:
        print(f"PyTorch could not be imported: {error}")
        return 1

    print(f"Python: {sys.version.split()[0]}")
    print(f"PyTorch: {torch.__version__}")
    print("CUDA: not queried or initialized; selected inference device: cpu")
    try:
        import transformers
    except ImportError:
        print("Transformers: not installed")
        return 1
    print(f"Transformers: {transformers.__version__}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
