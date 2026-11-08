# Hardware and platform compatibility

## Development target

The application and inference backend are CPU-first. The backend explicitly
moves model parameters to CPU and does not initialize CUDA or claim ROCm/AMD GPU
support. No NVIDIA GPU is required.

The inspected development machine runs 64-bit Windows 11, has approximately
16 GiB RAM, and reports AMD Radeon integrated graphics with approximately
512 MiB of dedicated adapter memory. That adapter figure is not a reliable
estimate of shared or usable graphics memory. Model loading and generation
remain CPU-only.

## Transformers and PyTorch

The Phase 4 backend requires PyTorch and Transformers. On Windows, install a
CPU PyTorch wheel only inside the project virtual environment, then install
the Transformers extra:

```powershell
.\.venv\Scripts\python.exe -m pip install --index-url https://download.pytorch.org/whl/cpu "torch>=2.6,<3"
.\.venv\Scripts\python.exe -m pip install -e ".[transformers,system-metrics]"
.\.venv\Scripts\python.exe scripts\check_torch_environment.py
```

The `system-metrics` extra adds optional `psutil` sampling. Without it,
process-memory and CPU-utilization measurements are reported as unavailable.
Do not install CUDA or ROCm wheels for this Windows CPU path.

| Component | Role | Hardware behavior |
| --- | --- | --- |
| PyTorch | Tensor execution and model generation | Explicitly executes on CPU |
| Transformers | Model/tokenizer loading and autoregressive generation | Uses the selected model's tokenizer |
| `sshleifer/tiny-gpt2` | Small educational first model | Not intended for production-quality answers |

Model downloads are opt-in. The default configuration is offline/cache-only,
model weights are not loaded during app startup, and the backend allows one
loaded model at a time. The configured maximum generation length defaults to
256 tokens and can be lowered through `LLM_LAB_INFERENCE_MAX_NEW_TOKENS`.
Unload the model when finished to release references. Operating-system memory
pressure can still occur; generation/load out-of-memory errors are surfaced
without silently switching models or devices.

## Measurement limitations

Model/tokenizer loading time is reported separately from prompt tokenization
and generation. The generation duration covers the `model.generate()` call.
TTFT is currently unavailable because the backend does not instrument the
first token event. Optional process RSS sampling includes the Python process
and loaded model; it is not incremental model memory or GPU VRAM.

Automated tests use mocks and do not load model weights. A real benchmark
requires an explicit model download or an existing compatible local cache and
an explicit run. CPU timing depends on the processor, memory pressure, runtime
versions, and other running processes; results should be compared only under
matching model, prompt, generation, and environment settings.

## Other optional integrations

| Integration | Status and platform notes |
| --- | --- |
| JAX | Optional numerical experiments; the Windows x86_64 wheel is documented as experimental. Native AMD GPU support is not available on Windows in JAX's current platform matrix. |
| `llama-cpp-python` | Optional; a compatible Windows wheel or local compiler toolchain may be needed. Use a compatible GGUF model. |
| vLLM | Optional Linux deployment integration; verify current Python, driver, and accelerator requirements. |
| SGLang | Optional Linux deployment integration; verify current Python, driver, and accelerator requirements. |

WSL does not automatically make AMD GPU support available to every framework.
The application reports backend availability based on installed dependencies
and runtime checks rather than assuming an accelerator is usable.
