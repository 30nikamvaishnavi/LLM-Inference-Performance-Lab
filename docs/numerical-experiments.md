# Phase 5: PyTorch numerical experiments and precision analysis

Phase 5 is a small, CPU-only numerical lab implemented in `app/numerical_lab`.
It does not load a language model, download files, initialize CUDA, or assume
NVIDIA/AMD GPU support. PyTorch is an optional dependency; when absent, the
experiment API and application can still start, and an explicit run returns an
unavailable error.

## Experiments

### Floating-point precision

The precision experiment creates deterministic float64 synthetic inputs and
evaluates the same elementwise function in float32, float64, float16, and
bfloat16:

```text
sin(x) + x² / 7 - x / 4
```

Each successful output is compared with the float64 result. The result includes
maximum and mean absolute error, relative L2 error, and non-finite difference
count. Relative L2 error is `||output - reference||₂ / ||reference||₂` (or the
absolute difference norm if the reference norm is zero). The input tensor has
4,096 values. Unsupported runtime operations retain their actual exception
reason; there is no dtype fallback.

Float32 uses about 7 decimal digits of precision; float64 about 16. Float16 has
a narrower exponent range and fewer significand bits, so rounding, overflow,
and underflow can be more pronounced. Bfloat16 has fewer significand bits than
float16 but a float32-like exponent range. Availability is operator- and
runtime-specific: a dtype existing in PyTorch does not guarantee every CPU
kernel supports it.

### Matrix multiplication

The same seeded square matrices are evaluated as float32, float64, float16,
and bfloat16 and compared with a float64 product. Defaults are 64×64, 256×256,
and 512×512; caller-provided sizes are constrained to 16–512 and at most
three sizes. The JSON/CSV record includes dimensions, dtype, warm-up count,
measured iteration count, mean/median latency, and numerical deviations.

P95 latency is `null` unless at least 20 timings exist; smaller samples do not
support a useful tail estimate. These runs are single-process CPU timings,
subject to thread scheduling, CPU load, and library configuration. Do not
compare measurements across machines as if they were hardware-equivalent.

### Numerical stability

The stability experiment uses small tensors to record:

- Float32 and float64 sums of 1,000 tenths, plus float32 deviation from 100.
- Cancellation for `((1e8 + 1) - 1e8)` in float32 and float64.
- Whether a float32 value of `1e8` loses an increment of one.
- Float16 overflow for `60000 × 2` and underflow of `1e-8`, compared with
  float32.
- Naive softmax using direct exponentiation and stable `torch.softmax` on
  logits `[1000, 1001, 1002]`.

The naive softmax can produce infinities and `NaN`; subtracting a maximum (as
stable softmax implementations do) avoids that overflow while preserving the
normalized probabilities. Such stability is important in neural networks:
attention logits and intermediate activations can have large magnitudes, and
non-finite values can propagate through a generation step or training update.

### Reproducibility

Dedicated CPU `torch.Generator` instances verify that equal seeds produce
bitwise-equal random tensors, a different seed changes the tensor, and repeated
matrix multiplication produces an identical output on the current runtime.
The result records the seed and whether deterministic algorithms were enabled.
This is an observed property of the current CPU/PyTorch run, not a cross-version
or cross-platform guarantee.

## Benchmark methodology and output

Measured matrix and elementwise operations run the requested warm-up count
first, then collect each iteration using `time.perf_counter()`. The reported
latencies cover the operation itself; input construction, dtype conversion,
reference calculation, and error analysis are outside the timed interval.
The stability experiment reports observed numerical values rather than
pretending they are performance benchmarks. Reproducibility timing covers
repeated CPU matrix multiplication.

Each explicit experiment execution creates a UUID-named pair beneath
`LLM_LAB_BENCHMARK_OUTPUT_DIR\numerical` (default
`benchmarks\results\numerical`):

- JSON: environment, seed, configuration, per-operation statuses, metrics,
  error reasons, and timings.
- CSV: one row per measurement with metrics serialized in a column.

Files are opened exclusively; an existing result is never overwritten.
Generated numerical results are ignored by Git. Unsupported and failed
operations are represented distinctly and are not replaced with guessed
measurements.

## Reproduce on Windows PowerShell

Use the existing project environment and no model download:

```powershell
.\.venv\Scripts\python.exe scripts\run_precision_experiments.py --experiment all --seed 42 --warmups 2 --iterations 5 --matrix-sizes 64 256 512
```

To run one experiment:

```powershell
.\.venv\Scripts\python.exe scripts\run_precision_experiments.py --experiment precision --seed 42 --warmups 2 --iterations 5
.\.venv\Scripts\python.exe scripts\run_precision_experiments.py --experiment matrix_multiplication --seed 42 --warmups 2 --iterations 5 --matrix-sizes 64 256 512
.\.venv\Scripts\python.exe scripts\run_precision_experiments.py --experiment numerical_stability --seed 42
.\.venv\Scripts\python.exe scripts\run_precision_experiments.py --experiment reproducibility --seed 42 --warmups 2 --iterations 5
```

The API provides:

- `GET /experiments` — experiment names and whether the PyTorch package is
  installed.
- `POST /experiments/run` — run one experiment or `all`, with bounded seed,
  warm-up, iteration, and matrix-size settings.
- `GET /experiments/results?limit=50` — bounded summaries from the configured
  result directory.
- `GET /experiments/results/{run_id}` — retrieve one complete result using
  only its 32-character hexadecimal run ID; arbitrary paths are rejected.

Start the local API with:

```powershell
.\.venv\Scripts\python.exe -m uvicorn app.main:app --reload
```

## Actual local run

The following values come from a real script execution on Windows 11, Python
3.13.1, PyTorch 2.15.0+cpu, CPU device, AMD64 Family 23 Model 104 Stepping 1
(12 logical processors reported). Run ID:
`664449e64a704e2dace71777abed424e`. It used seed 42, two warm-ups, five measured
iterations per timed workload, and matrix dimensions 64, 256, and 512. The run
completed with status `success`: all 20 experiment records executed and none
were marked unsupported or failed. P95 values are absent because five timing
samples are below the 20-sample threshold.

Elementwise precision results (mean latency; maximum absolute error versus
float64 reference):

| Dtype | Mean latency | Maximum absolute error | Relative L2 error |
| --- | ---: | ---: | ---: |
| float32 | 68.04 µs | 2.3257e-7 | 6.8041e-8 |
| float64 | 65.56 µs | 0 | 0 |
| float16 | 110.26 µs | 0.0027680 | 0.00055818 |
| bfloat16 | 69.20 µs | 0.0206712 | 0.00439637 |

Mean matrix-multiplication latency (each cell is one dtype/size experiment;
these measurements are specific to this run):

| Dtype | 64×64 | 256×256 | 512×512 |
| --- | ---: | ---: | ---: |
| float32 | 55.82 µs | 206.60 µs | 1.851 ms |
| float64 | 54.24 µs | 893.70 µs | 4.827 ms |
| float16 | 1.042 ms | 50.857 ms | 412.872 ms |
| bfloat16 | 975.78 µs | 43.587 ms | 640.353 ms |

All four dtypes were supported for these specific CPU operations. Low-precision
matrix multiplication was substantially slower than float32 on this installed
CPU runtime; reduced precision must not be assumed to improve CPU performance.
For 512×512 products, relative L2 errors were about 2.44e-7 (float32),
3.60e-4 (float16), and 2.87e-3 (bfloat16); float64 was the reference.

Stability observations: summing 1,000 float32 tenths yielded
100.00000762939453 (absolute error 7.6294e-6); float32 cancellation of
`1e8 + 1 - 1e8` yielded 0 while float64 yielded 1; float16 `60000 × 2`
overflowed to infinity and float16 `1e-8` underflowed to 0; naive softmax on
`[1000, 1001, 1002]` was not finite while `torch.softmax` was finite and summed
to 1. The reproducibility check observed same-seed tensor equality,
different-seed inequality, and bitwise-equal repeated CPU matmul outputs.
PyTorch deterministic-algorithm mode was disabled; the observation is limited
to repeated operations on this CPU/runtime.

Actual result files:

- `benchmarks/results/numerical/664449e64a704e2dace71777abed424e.json`
- `benchmarks/results/numerical/664449e64a704e2dace71777abed424e.csv`

These are ignored generated experiment outputs, not checked-in benchmark
claims. This default-configuration run is the reported run; an earlier
shorter validation run has a separate UUID result and was not overwritten.
Timings are measured values from this one run, not estimates or universal
performance claims. Re-run the command to obtain new measurements.

## Hardware and software limitations

The development environment is Windows 11, Python 3.13.1, and CPU-only
PyTorch 2.15.0+cpu on an AMD CPU with integrated graphics. No CUDA or ROCm
operation is performed. The configured matrix ceiling is 512 per side to keep
the default experiments modest on a 16 GiB system. CPU measurements vary with
system load, PyTorch build, CPU thread count, and power state.

Tests use small tensors and do not download models. They verify experiment
behavior, while only the explicit script or API run produces the reported
measurements.
