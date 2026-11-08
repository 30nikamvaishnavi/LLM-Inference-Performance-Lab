# LLM Inference & Performance Benchmarking Lab

A CPU-first engineering project for exploring tokenization, transformer execution,
inference backends, numerical behavior, and reproducible LLM performance
measurements. It is designed as a benchmarking and learning platform, not as a
chatbot.

The project provides validated configuration and schemas, a deterministic
offline benchmark corpus, local and optional external dataset loaders, a
dependency-free tokenizer, reproducible benchmark-input manifests, backend
discovery contracts, and a FastAPI surface. Model inference and benchmark
execution are added incrementally; preparing inputs does not execute inference
or produce performance results.

## Architecture

```text
app/
├── api/             FastAPI surfaces
├── core/            Settings, logging, errors, and shared schemas
├── inference/       Backend contract, registry, and optional adapters
├── datasets/        Synthetic, JSONL, and opt-in WikiText-2 datasets
├── tokenization/    Offline and optional Hugging Face tokenization
├── transformer_lab/ Educational PyTorch implementations (future phase)
├── numerical_lab/   NumPy, PyTorch, and JAX experiments (future phase)
├── benchmarking/   Seeded input manifests; inference runner (future phase)
├── system/          Runtime and hardware information
└── dashboard/       Streamlit dashboard (future phase)
```

The API exposes `GET /health`, `GET /backends`, `GET /system/info`, dataset
discovery and loading, sample pagination, dataset/tokenization statistics,
`POST /tokenize`, and seeded benchmark-input preparation. See
[docs/datasets-and-tokenization.md](./docs/datasets-and-tokenization.md) for
routes and PowerShell examples. Dataset loading is explicit; listing WikiText-2
does not download it. Optional inference backends are not registered until
their adapters are implemented.

## Prerequisites

- Windows 11 64-bit (or Linux for optional deployment integrations)
- Python **3.11 recommended**; supported project range is Python 3.10–3.13
- Git
- No GPU, model weights, or API credentials are required for the scaffold

Python 3.11 was not detected in the development machine inspection. Install the
latest Python 3.11.x 64-bit release from the
[official Windows downloads page](https://www.python.org/downloads/windows/).
Choose a per-user installation and a user-writable destination; do not select
an all-users installation. Open a new terminal and confirm the launcher sees
it with `py -0p` and `py -3.11 --version`. The project does not modify global
Python installations.

## Windows setup

From the repository root in PowerShell, create an isolated environment and
install the project with its development tools:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
Copy-Item .env.example .env
```

The `.env` file is local and ignored by Git. Configuration can also be supplied
directly through `LLM_LAB_`-prefixed environment variables.

If Python 3.11 is not yet installed and you need to validate with an existing
interpreter, select it explicitly instead of using the default `python`:

```powershell
py -3.13 -m venv .venv
```

Python 3.13 is within the supported project range, but the recommended
development target remains Python 3.11. The repository's default interpreter
must not be assumed.

## Dependencies and optional integrations

The base install includes NumPy, Pydantic, FastAPI, Uvicorn, Streamlit, and
`.env` loading. Test and lint packages are in the `dev` extra. Install only
optional ML packages needed for a specific experiment:

```powershell
.\.venv\Scripts\python.exe -m pip install -e ".[transformers]"
.\.venv\Scripts\python.exe -m pip install -e ".[pytorch]"
.\.venv\Scripts\python.exe -m pip install -e ".[jax]"
```

`llama-cpp-python` is optional and may require a compatible wheel or a local
compiler toolchain on Windows. vLLM and SGLang are Linux deployment
integrations; their extras are marked for Linux and Python below 3.13 and are
not part of the Windows setup. Check their official hardware and installation
guidance before installing. None of these optional backends is required for
the current API or tests. JAX's CPU package can be used on Windows x86_64, but
its Windows wheel is documented as experimental; native AMD GPU support is
not available on Windows in JAX's current platform matrix.

## Configuration

Settings are validated by Pydantic Settings and loaded from the process
environment or `.env`:

| Variable | Default | Purpose |
| --- | --- | --- |
| `LLM_LAB_ENVIRONMENT` | `development` | `development` or `production` |
| `LLM_LAB_DEBUG` | `false` | Debug mode; rejected in production |
| `LLM_LAB_LOG_LEVEL` | `INFO` | Standard Python logging level |
| `LLM_LAB_MODEL_CACHE_DIR` | `~/.cache/llm-inference-performance-lab/models` | Model cache location |
| `LLM_LAB_BENCHMARK_OUTPUT_DIR` | `benchmarks/results` | Local benchmark output location |
| `LLM_LAB_DATA_DIR` | `data` | Local dataset and cache root |
| `LLM_LAB_DATASET_MAX_SAMPLES` | `5000` | Dataset load bound |
| `LLM_LAB_DATA_DIR` | `data` | Local JSONL and external dataset cache root |
| `LLM_LAB_DATASET_MAX_SAMPLES` | `5000` | Maximum samples loaded from an external/local dataset |

Logs include timestamps and module names. Application code should not log
complete prompts, credentials, or other sensitive values.

## Datasets and tokenization

The built-in `synthetic-benchmark` dataset contains 35 original samples across
short prompts, paragraphs, long-form text, technical explanations, code,
question-answer, and summarization categories. It requires no network access
and is intended for deterministic pipeline tests, not training or model-quality
evaluation.

WikiText-2 is an optional external text source for tokenization experiments.
Install `.\.venv\Scripts\python.exe -m pip install -e ".[datasets]"` only when
you intend to request it. Then explicitly call the dataset load endpoint. Its
files and retrieval metadata remain under `LLM_LAB_DATA_DIR` and are excluded
from Git. Review the upstream source and license before redistribution; see
[dataset and tokenization documentation](./docs/datasets-and-tokenization.md).

The default `simple` tokenizer is an offline baseline, not a model tokenizer.
The optional Hugging Face adapter is cached-only unless a download is
explicitly requested. Neither dataset loading nor benchmark manifest
preparation runs a model or fabricates performance measurements.

## Run the API

```powershell
.\.venv\Scripts\python.exe -m uvicorn app.main:app --reload
```

Open [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs) for the generated
OpenAPI documentation. Health, backend discovery, and system information are
available at `/health`, `/backends`, and `/system/info`.

## Tests and lint

```powershell
.\.venv\Scripts\python.exe -m pytest --cov=app --cov-report=term-missing
.\.venv\Scripts\ruff.exe check .
.\.venv\Scripts\ruff.exe format --check .
```

Tests use synthetic data and mocked optional integrations; they do not download
WikiText-2, tokenizers, or model weights.
GitHub Actions runs lint, formatting checks, and tests on Windows and Linux
across Python 3.10, 3.11, and 3.13.

## Hardware limitations

The core project is CPU-compatible and does not assume CUDA or NVIDIA hardware.
The inspected machine reports AMD Radeon integrated graphics and 15.3 GiB RAM;
the reported adapter memory is not a reliable estimate of total shared or
usable GPU memory. The initial system endpoint reports OS, Python, and CPU
count. It leaves memory and GPU inventory unset rather than presenting
unverified measurements. Future backend adapters must check actual runtime
support and report unavailable capabilities explicitly.

For Linux/WSL deployment, use a separate environment and follow the current
installation instructions for the chosen inference framework and driver
stack. WSL does not by itself guarantee AMD GPU support for a given framework.

## Development roadmap

1. Environment and project scaffolding
2. Dataset management, tokenization, and benchmark input preparation (this phase)
3. Educational transformer fundamentals
4. CPU-first Hugging Face inference
5. Optional llama.cpp integration
6. Optional vLLM and SGLang deployment adapters
7. NumPy, PyTorch, and JAX numerical experiments
8. Reproducible inference benchmarking engine
9. Streamlit dashboard and expanded API surfaces
10. Expanded CI and benchmark methodology documentation

This repository contains no benchmark results yet. Results will be recorded
only after actual executions with their settings and environment metadata.
