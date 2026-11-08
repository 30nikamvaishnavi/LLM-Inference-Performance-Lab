# LLM Inference Performance Lab

An engineering-focused platform for exploring, benchmarking, and understanding
Large Language Model (LLM) inference performance and numerical computing.

Built with Python, PyTorch, Hugging Face Transformers, and FastAPI, this project
investigates how model inference, floating-point precision, numerical
stability, and hardware constraints influence machine learning performance.

## Key features

- **LLM inference:** Run real Hugging Face Transformer models on CPU.
- **Performance benchmarking:** Measure inference latency, throughput, and
  execution performance.
- **Dataset and tokenization pipeline:** Prepare deterministic datasets and
  tokenized benchmark inputs.
- **Numerical precision analysis:** Compare float32, float64, float16, and
  bfloat16 operations where supported.
- **Matrix multiplication benchmarks:** Evaluate tensor operations across data
  types and dimensions.
- **Numerical stability:** Explore floating-point accumulation, cancellation,
  overflow, underflow, and stable softmax.
- **Reproducibility:** Run seeded experiments and persist structured results.
- **FastAPI integration:** Expose inference, benchmarking, and numerical
  experiment functionality through APIs.

## Technology stack

Python · PyTorch · Hugging Face Transformers · NumPy · FastAPI · Pydantic ·
Pytest · Ruff

## Hardware approach

Designed for CPU-first development. Optional acceleration and specialized
inference backends are treated separately rather than assumed to be available
on every machine.

## Current status

- **Phases 1–3:** Environment, architecture, dataset, and tokenization pipeline.
- **Phase 4:** CPU-based Transformer inference and benchmarking.
- **Phase 4.1:** Real tiny-GPT-2 inference and benchmark validation.
- **Phase 5:** PyTorch numerical precision, matrix multiplication, stability,
  and reproducibility experiments.

## Architecture

```text
app/
├── api/             FastAPI surfaces
├── core/            Settings, logging, errors, and shared schemas
├── inference/       Backend contract, registry, and CPU Transformers adapter
├── datasets/        Synthetic, JSONL, and opt-in WikiText-2 datasets
├── tokenization/    Offline and optional Hugging Face tokenization
├── transformer_lab/ Educational PyTorch implementations (future phase)
├── numerical_lab/   CPU PyTorch precision, matrix, stability, and reproducibility experiments
├── benchmarking/    Seeded manifests and repeated inference runner
├── system/          Runtime and hardware information
└── dashboard/       Streamlit dashboard (future phase)
```

The API exposes system, dataset, tokenization, model lifecycle, generation,
inference-benchmark, and CPU numerical-experiment routes. See
[docs/datasets-and-tokenization.md](./docs/datasets-and-tokenization.md) for
dataset workflows, [docs/numerical-experiments.md](./docs/numerical-experiments.md)
for Phase 5, and the generated OpenAPI page for all routes. Dataset and model
loading are explicit; listing an optional source or starting the API does not
download data or weights.

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
`.env` loading. Test and lint packages are in the `dev` extra. For CPU inference
on Windows, install a CPU-only PyTorch wheel in the project environment, then
install the Transformers and optional process-metrics extras:

```powershell
.\.venv\Scripts\python.exe -m pip install --index-url https://download.pytorch.org/whl/cpu "torch>=2.6,<3"
.\.venv\Scripts\python.exe -m pip install -e ".[dev,transformers,system-metrics]"
.\.venv\Scripts\python.exe scripts\check_torch_environment.py
```

The project also declares separate `pytorch` and `transformers` extras, which
can be installed together with
`.\.venv\Scripts\python.exe -m pip install -e ".[pytorch,transformers]"`.
For Windows CPU development, prefer the explicit CPU-index command above.
Install JAX separately only when running its numerical experiments:

```powershell
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
| `LLM_LAB_INFERENCE_MODEL_ID` | `sshleifer/tiny-gpt2` | Configured default model; not loaded automatically |
| `LLM_LAB_INFERENCE_OFFLINE_MODE` | `true` | Prevent API-triggered network downloads |
| `LLM_LAB_INFERENCE_MAX_NEW_TOKENS` | `256` | Maximum generation length enforced by the backend |

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

## PyTorch numerical experiments

Phase 5 provides CPU-only float32/float64/float16/bfloat16 precision comparisons,
matrix multiplication benchmarks, numerical-stability demonstrations, and
seeded reproducibility checks. Each execution records the actual operation
status and runtime metadata in unique JSON and CSV files under
`benchmarks/results/numerical/`. Unsupported dtypes are recorded with their
runtime error rather than replaced with another dtype. PyTorch remains an
optional dependency; the API does not import it or run experiments at startup.

Run all experiments on this CPU environment:

```powershell
.\.venv\Scripts\python.exe scripts\run_precision_experiments.py --experiment all --seed 42 --warmups 2 --iterations 5 --matrix-sizes 64 256 512
```

The API exposes `GET /experiments`, `POST /experiments/run`, and
`GET /experiments/results`. See
[docs/numerical-experiments.md](./docs/numerical-experiments.md) for
experiment definitions, measurement methodology, and the results from the
latest explicit CPU run.

## CPU model inference

The first model is `sshleifer/tiny-gpt2`, a small educational model for testing
the inference pipeline, not a production-quality assistant. PyTorch executes
the model on CPU; Transformers loads the model and its matching tokenizer.
Tokenization converts input text into model-specific token IDs, and
autoregressive generation repeatedly predicts the next token from the prompt
and tokens already generated.

Model loading is explicit and cache-only by default:

```powershell
.\.venv\Scripts\python.exe scripts\download_tiny_model.py
.\.venv\Scripts\python.exe scripts\run_inference.py --prompt "Explain CPU inference briefly." --max-new-tokens 24
```

Running the download command explicitly requests the weights. Alternatively,
pass `--allow-download` to `run_inference.py`. Model files use the configured
Hugging Face cache (`LLM_LAB_MODEL_CACHE_DIR`, default
`~/.cache/llm-inference-performance-lab/models`). No fallback model is loaded.
The API defaults to offline mode; to permit a requested download, set
`$env:LLM_LAB_INFERENCE_OFFLINE_MODE = "false"` and send
`{"allow_download": true}` to `POST /inference/load`.

Run a real repeated benchmark after the model is cached:

```powershell
.\.venv\Scripts\python.exe scripts\benchmark_transformers.py --samples 3 --max-new-tokens 16 --repetitions 3 --warmups 1
```

Results and the selected-input manifest are written beneath the configured
benchmark output directory. Each result records the model revision when
available, runtime versions, selected sample IDs, generation settings, and
measured entries. Use the same dataset, seed, filters, and generation settings
to reproduce input selection; timing will vary with machine load and software
environment.

`POST /inference/generate` accepts one prompt. `POST /inference/tokenize`
uses the tokenizer belonging to the currently loaded model. The benchmark API
uses samples from a dataset explicitly loaded through the dataset API. Model
loading, dataset loading, and benchmark execution are separate operations.

### Metric definitions and limitations

- Model loading duration covers tokenizer and model loading; it is separate
  from generation latency.
- Tokenization latency is measured separately for a generation request.
- Generation duration measures the `model.generate()` call, not tokenization
  or model loading. Generated tokens per second is generated token count
  divided by this duration; it is not prompt-plus-output throughput.
- Warm-up iterations run before recorded measurements and are excluded from
  benchmark statistics.
- This implementation cannot accurately observe first-token emission from
  `model.generate()`, so TTFT is returned as unavailable (`null`), not
  estimated.
- If optional `psutil` is installed, RSS and process CPU are sampled during
  generation. RSS is whole-process memory, not model-only or incremental
  memory. These values are unavailable when the dependency is missing.

Automated tests use deterministic fake models and tokenizers to check control
flow and persistence; their outputs are not inference measurements. Only
executing the scripts or API with loaded model weights creates real measured
results. The test suite does not download weights or run real inference.

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

The inference backend explicitly selects CPU and does not initialize CUDA or
assume NVIDIA hardware. It does not claim AMD GPU acceleration or ROCm support.
The inspected machine reports AMD Radeon integrated graphics and 15.3 GiB RAM;
the reported adapter memory is not a reliable estimate of total shared or
usable GPU memory. The system endpoint reports OS, Python, and CPU count; the
backend reports CPU as its execution device. The default generation cap is
conservative and configurable. Only one model can be loaded by a backend
instance; unload it to release references before loading another.

For Linux/WSL deployment, use a separate environment and follow the current
installation instructions for the chosen inference framework and driver
stack. WSL does not by itself guarantee AMD GPU support for a given framework.

## Development roadmap

1. Environment and project scaffolding
2. Dataset management, tokenization, and benchmark input preparation
3. CPU-first Hugging Face Transformers inference and repeated benchmarking
4. Educational transformer fundamentals
5. PyTorch numerical experiments and precision analysis
6. Optional llama.cpp integration
7. Optional vLLM and SGLang deployment adapters
8. NumPy/JAX numerical experiments and dashboard analysis

Benchmark and numerical experiment outputs are created only after actual
execution and remain excluded from Git.
