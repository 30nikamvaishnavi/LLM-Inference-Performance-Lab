# Dependency groups

`pyproject.toml` is the source of truth for dependency ranges and optional
extras. Install only the integration you intend to use:

- Core API and dashboard: `python -m pip install -e .`
- Development and tests: `python -m pip install -e ".[dev]"`
- Transformers tokenizers: `python -m pip install -e ".[transformers]"`
- PyTorch: `python -m pip install -e ".[pytorch]"`
- JAX: `python -m pip install -e ".[jax]"`
- llama.cpp: `python -m pip install -e ".[llama-cpp]"`

The vLLM and SGLang extras are marked for Linux and Python versions below 3.13.
Their installation is intentionally not part of Windows setup; consult each
project's current hardware-specific installation instructions before installing.
