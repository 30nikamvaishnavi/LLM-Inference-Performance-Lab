# Datasets, tokenization, and benchmark input preparation

## Why benchmark datasets rather than training data?

This project measures text processing and inference behavior; it does not train
foundation models. A compact, stable input corpus is more useful for repeatable
latency and tokenizer experiments than downloading a large training corpus.
The project-authored synthetic samples run offline and cover varied input
shapes. WikiText-2 is an optional public text source when more realistic
language samples are useful.

Neither corpus is a model-quality evaluation dataset in this phase. No
benchmark measurement is produced by merely loading or preparing samples.

## Built-in synthetic dataset

The `synthetic-benchmark` dataset contains 35 original project-authored
samples, five each in these categories:

- `short_prompts`
- `medium_paragraphs`
- `long_form`
- `technical_explanations`
- `python_code`
- `question_answer`
- `summarization_prompts`

It works offline and needs no optional framework or model. Sample metadata
includes a unique ID, category, approximate length label, source note, and
tags. Its source is marked as project-authored and its metadata identifies the
project MIT license.

## WikiText-2 (explicit opt-in)

The optional loader uses
[Salesforce/wikitext](https://huggingface.co/datasets/Salesforce/wikitext),
configuration `wikitext-2-raw-v1`, and the `train` split. The `datasets`
package is a separate optional extra; it is not installed with the base
environment. Installing that extra does not download the corpus.

The loader uses streaming and reads only up to the configured sample maximum.
A load request is required before any remote access happens. If the package,
network, or source is unavailable, the API reports an error and does not
substitute another dataset.

```powershell
.\.venv\Scripts\python.exe -m pip install -e ".[datasets]"
.\.venv\Scripts\python.exe -m uvicorn app.main:app --reload

# Explicitly load at most 100 non-empty training samples.
Invoke-RestMethod -Uri http://127.0.0.1:8000/datasets/wikitext-2/load `
  -Method Post -ContentType 'application/json' -Body '{"limit":100}'
```

Review the upstream dataset card and license before using or redistributing
downloaded text. The loader records the source, configuration, retrieval time,
and upstream license value when the installed dataset metadata exposes it.
Unavailable license metadata remains null and must not be interpreted as
permission to redistribute the corpus. Preserve required attribution and
follow the upstream terms.

## Local data and JSONL

`LLM_LAB_DATA_DIR` selects the local data root (default `data`). The Hugging
Face cache is stored below that root in `huggingface`; retrieval metadata is
stored as `wikitext-2-raw-v1.metadata.json`. Files under `data/` are excluded
from Git. `LLM_LAB_DATASET_MAX_SAMPLES` bounds a loader request, defaulting to
5000 samples.

Local benchmark datasets can be stored as `<name>.jsonl` under the configured
data root. Each non-blank line must be a `DatasetSample` JSON object with
`sample_id`, `category`, `input_text`, `length_category`, and `source_info`;
`tags` is optional. Duplicate IDs and malformed lines fail the load with a
line-specific error. An optional `<name>.metadata.json` sidecar uses the
`DatasetMetadata` schema. The API accepts registered dataset names only; it
does not accept arbitrary filesystem paths.

## Tokenization

The default `simple` tokenizer is deterministic, offline, and splits Unicode
word runs and punctuation. It is an educational baseline, **not** a substitute
for a model's tokenizer. Its decoder normalizes whitespace.

The optional Transformers adapter uses names of the form `hf:org/model`.
Cached tokenizers are used by default; no network request is made unless
`allow_download` is explicitly set on `POST /tokenize`. This loads a tokenizer,
not model weights. Tokenization throughput is a local timed observation and
should be repeated for performance claims.

```powershell
$body = @{ text = "Compare token counts for this sentence."; tokenizer_name = "simple" } |
  ConvertTo-Json
Invoke-RestMethod -Uri http://127.0.0.1:8000/tokenize `
  -Method Post -ContentType 'application/json' -Body $body

# Load the deterministic built-in samples before listing or measuring them.
Invoke-RestMethod -Uri http://127.0.0.1:8000/datasets/synthetic-benchmark/load `
  -Method Post -ContentType 'application/json' -Body '{"limit":35}'
Invoke-RestMethod -Uri http://127.0.0.1:8000/datasets/synthetic-benchmark/samples?offset=0`&limit=5
Invoke-RestMethod -Uri http://127.0.0.1:8000/datasets/synthetic-benchmark/tokenization-statistics
```

Available routes:

| Method and route | Behavior |
| --- | --- |
| `GET /datasets` | List built-in, optional, and configured local datasets without downloading |
| `GET /datasets/{name}` | Read source metadata |
| `POST /datasets/{name}/load` | Explicitly load a bounded number of samples |
| `GET /datasets/{name}/samples` | Paginate a loaded dataset; accepts `offset`, `limit`, `category`, and `length_category` |
| `GET /datasets/{name}/statistics` | Character and category statistics for a loaded dataset |
| `GET /datasets/{name}/tokenization-statistics` | Token count, throughput, character/token ratio, and input-length histogram |
| `POST /tokenize` | Encode supplied text and report tokenization metrics |
| `POST /benchmarks/prepare-inputs` | Filter and save a seeded manifest for future inference benchmarks |

Statistics, sample listing, and benchmark preparation require the dataset to
have been loaded first. The default sample page is capped at 500 rows.

## Reproducible benchmark inputs

Prepare inputs from an already loaded dataset. The manifest contains the seed,
dataset/version, ordered sample IDs, exact text, character counts, optional
token counts, tokenizer identity, creation time, and SHA-256 digest of the
selected IDs and text. It is written under
`LLM_LAB_BENCHMARK_OUTPUT_DIR/manifests` (default
`benchmarks/results/manifests`) using an atomic file replacement. A repeated
request with the same dataset contents and seed selects the same sample order
and yields the same input digest. The timestamp is expected to differ.

```powershell
$body = @{
  dataset_name = "synthetic-benchmark"
  categories = @("python_code", "technical_explanations")
  length_categories = @("short", "medium")
  max_samples = 8
  seed = 123
  tokenizer_name = "simple"
} | ConvertTo-Json

Invoke-RestMethod -Uri http://127.0.0.1:8000/benchmarks/prepare-inputs `
  -Method Post -ContentType 'application/json' -Body $body
```

Reuse that saved manifest for every backend in a comparison. It contains inputs,
not model outputs or performance results. Do not compare benchmark results
unless model, precision, prompts, generation settings, warm-up, concurrency,
and timing definitions are equivalent and recorded.
