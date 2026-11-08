# Hardware and platform compatibility

## Development target

The core application and tests are CPU-first. They do not require CUDA, an
accelerator, API credentials, or model downloads. The project metadata accepts
Python 3.10 through 3.13; Python 3.11 is the recommended development version.

The inspected development machine runs 64-bit Windows 11, has 15.3 GiB RAM,
and reports AMD Radeon integrated graphics with 512 MiB of adapter memory.
That adapter figure is not a reliable estimate of shared or usable graphics
memory. No inference speed or GPU availability is inferred from it.

## Optional integrations

| Integration | Phase 2 status | Platform notes |
| --- | --- | --- |
| Transformers and tokenizers | Not installed | Later CPU experiments may use a small model; model downloads are opt-in. |
| PyTorch | Not installed | Install a wheel compatible with the selected Python and platform when needed. No CUDA variant is assumed. |
| JAX | Not installed | CPU is available on Windows x86_64, but its Windows wheel is documented as experimental. AMD GPU support is not available on native Windows in the current JAX platform matrix. |
| llama.cpp | Not installed | `llama-cpp-python` may require a compatible wheel or a local compiler toolchain on Windows; use a compatible GGUF model only. |
| vLLM | Not installed | Optional Linux deployment integration; hardware, driver, and Python support must be checked against its current installation guide. |
| SGLang | Not installed | Optional Linux deployment integration; hardware, driver, and Python support must be checked against its current installation guide. |

The vLLM and SGLang extras are guarded to Linux and Python below 3.13. Their
dependency declarations do not establish that a given accelerator or driver
combination is supported. WSL does not automatically make AMD GPU support
available to every framework.

The initial `/system/info` endpoint reports the OS, Python version, and CPU
count. It leaves memory and GPU inventory unset because Phase 2 does not add a
hardware-detection dependency or platform-specific probe.
