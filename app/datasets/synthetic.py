"""Offline, deterministic synthetic benchmark corpus."""

from app.datasets.base import DatasetLoader
from app.datasets.schemas import DatasetCollection, DatasetMetadata, DatasetSample

_SAMPLES: tuple[tuple[str, str, str, str, tuple[str, ...]], ...] = (
    (
        "short-001",
        "short_prompts",
        "Explain tokenization in one sentence.",
        "short",
        ("definition",),
    ),
    (
        "short-002",
        "short_prompts",
        "What is time to first token?",
        "short",
        ("benchmarking",),
    ),
    (
        "short-003",
        "short_prompts",
        "Name two ways to reduce inference memory.",
        "short",
        ("inference",),
    ),
    (
        "short-004",
        "short_prompts",
        "Summarize why warm-up runs are useful.",
        "short",
        ("benchmarking",),
    ),
    (
        "short-005",
        "short_prompts",
        "Compare greedy decoding with sampling.",
        "short",
        ("generation",),
    ),
    (
        "medium-001",
        "medium_paragraphs",
        "A tokenizer converts text into model-readable units. Depending on its vocabulary, a unit may be a word, part of a word, punctuation, or a special marker. The resulting sequence length affects both the work and memory required by a model.",
        "medium",
        ("tokenization",),
    ),
    (
        "medium-002",
        "medium_paragraphs",
        "A useful benchmark separates setup from execution. Model loading is measured once, warm-up requests are reported separately, and repeated timed requests use identical generation settings. This makes later comparisons easier to interpret.",
        "medium",
        ("methodology",),
    ),
    (
        "medium-003",
        "medium_paragraphs",
        "On a CPU-only workstation, a small quantized model may be more practical than a large full-precision model. Actual feasibility still depends on the model architecture, context length, runtime, and available system memory.",
        "medium",
        ("hardware",),
    ),
    (
        "medium-004",
        "medium_paragraphs",
        "A dataset manifest records the exact input identifiers, selection seed, tokenizer, and text digest. Reusing the manifest across backends prevents accidental input drift during a performance comparison.",
        "medium",
        ("reproducibility",),
    ),
    (
        "medium-005",
        "medium_paragraphs",
        "Latency describes how long a request takes, while throughput describes completed work over time. Both depend on prompt size, generated output, batching, concurrency, and the particular hardware and software configuration.",
        "medium",
        ("metrics",),
    ),
    (
        "long-001",
        "long_form",
        "A careful inference experiment begins by writing down a question that can be answered with measurements. The experimenter selects a model and backend, records the software versions, and fixes the prompt set and generation parameters. Each timed run starts only after the model is loaded and the chosen warm-up policy is complete. Repeated observations reveal variability that a single request cannot show. Results should include the median and a high percentile, as well as output length and throughput. If a metric cannot be measured, it is recorded as unavailable rather than replaced with a guess. This discipline makes performance reports more useful than isolated speed claims.",
        "long",
        ("methodology", "reproducibility"),
    ),
    (
        "long-002",
        "long_form",
        "The memory footprint of text generation has several contributors. Model parameters occupy storage in a representation determined by their data type or quantization. During generation, temporary activations and attention state also require memory, and some runtimes reserve additional workspace. Longer contexts can increase the size of cached key and value tensors. A benchmark should distinguish process memory from accelerator memory and should name the measurement method. When a platform does not expose a reliable peak-memory counter, the report should say so. Reporting the model format, context, and output configuration helps readers understand why two runs may have different memory requirements.",
        "long",
        ("memory", "inference"),
    ),
    (
        "long-003",
        "long_form",
        "Input preparation is part of experimental control. A stable collection of samples can be filtered by topic and approximate length, then selected with a recorded random seed. The resulting manifest should preserve sample identifiers and the exact text sent to the backend. A cryptographic digest can detect changes to those inputs later. Token counts are meaningful only when the tokenizer and its options are identified. Reusing identical prompts does not make different models equivalent, but it removes one source of variation from a backend comparison. The researcher should still verify that precision, stopping rules, and requested output lengths are comparable before interpreting a result.",
        "long",
        ("datasets", "benchmarking"),
    ),
    (
        "long-004",
        "long_form",
        "A token throughput number needs a precise definition. Some reports divide generated tokens by total wall time, while others exclude queueing or prefill. The numerator may include or exclude special tokens, and the denominator may represent one request or a concurrent batch. Time to first token measures a different part of the user experience from total completion time. A result table should state its timing boundary, count convention, and concurrency. When streaming is unavailable, time to first token should remain empty rather than being inferred from aggregate latency. Clear definitions make measurements reproducible and prevent superficially similar numbers from being compared incorrectly.",
        "long",
        ("metrics", "definitions"),
    ),
    (
        "long-005",
        "long_form",
        "A local benchmark corpus should be small enough for routine testing and broad enough to exercise different input shapes. Short questions expose fixed request overhead, while longer passages exercise prefill behavior. Code and technical text may tokenize differently from ordinary prose. Synthetic samples support offline tests, but they do not represent every real-world distribution. A public corpus can add realism when its license and source are documented. Dataset retrieval should be explicit, bounded, and repeatable. Keeping external files outside version control avoids accidentally publishing copied material and prevents repository size from growing with cached data.",
        "long",
        ("datasets", "offline"),
    ),
    (
        "technical-001",
        "technical_explanations",
        "Autoregressive generation predicts the next token from the preceding context, appends the chosen token, and repeats until a stopping rule is met. Greedy decoding chooses the highest-scoring candidate at each step. Sampling first transforms model scores, often with a temperature, then draws a candidate from a probability distribution. These strategies trade determinism against output diversity.",
        "medium",
        ("generation", "decoding"),
    ),
    (
        "technical-002",
        "technical_explanations",
        "The attention mechanism projects input representations into queries, keys, and values. Similarity scores between queries and keys are scaled and normalized to produce weights, which combine value vectors. Multiple attention heads perform this operation in parallel with separate learned projections, then their outputs are joined and transformed.",
        "medium",
        ("transformers", "attention"),
    ),
    (
        "technical-003",
        "technical_explanations",
        "For a sample with P prompt tokens and G generated tokens, total token work depends on both prefill and iterative decode. Time per output token can be approximated only after defining which elapsed interval is divided by which counted tokens. Tiny output lengths magnify fixed overhead, so measurements should record requested and observed token counts.",
        "medium",
        ("metrics", "latency"),
    ),
    (
        "technical-004",
        "technical_explanations",
        "Quantization represents model weights with fewer bits than common floating-point formats. It can reduce weight storage and sometimes improve execution efficiency, but quality and speed depend on the quantization scheme and runtime kernels. The quantization label alone is insufficient to guarantee a particular memory footprint or performance.",
        "medium",
        ("quantization", "memory"),
    ),
    (
        "technical-005",
        "technical_explanations",
        "A tokenizer vocabulary maps text fragments to integer identifiers. Encoding is generally not a character-for-character conversion, so text length and token count have no fixed ratio. Special tokens can mark boundaries or tasks. Correct experiments record tokenizer identity and whether special tokens, truncation, or padding were enabled.",
        "medium",
        ("tokenization", "vocabulary"),
    ),
    (
        "code-001",
        "python_code",
        'def token_count(text: str, tokenizer) -> int:\n    """Return the number of encoded tokens."""\n    return len(tokenizer.encode(text))',
        "short",
        ("python", "tokenization"),
    ),
    (
        "code-002",
        "python_code",
        'def median_latency(samples_ms: list[float]) -> float:\n    if not samples_ms:\n        raise ValueError("At least one sample is required")\n    ordered = sorted(samples_ms)\n    middle = len(ordered) // 2\n    if len(ordered) % 2:\n        return ordered[middle]\n    return (ordered[middle - 1] + ordered[middle]) / 2',
        "medium",
        ("python", "statistics"),
    ),
    (
        "code-003",
        "python_code",
        'from pathlib import Path\nimport json\n\ndef write_manifest(path: Path, manifest: dict[str, object]) -> None:\n    path.parent.mkdir(parents=True, exist_ok=True)\n    temporary = path.with_suffix(path.suffix + ".tmp")\n    temporary.write_text(json.dumps(manifest, indent=2), encoding="utf-8")\n    temporary.replace(path)',
        "medium",
        ("python", "files"),
    ),
    (
        "code-004",
        "python_code",
        "def select_rows(rows, categories, maximum, seed):\n    import random\n\n    candidates = [row for row in rows if row.category in categories]\n    chooser = random.Random(seed)\n    chosen = chooser.sample(candidates, min(maximum, len(candidates)))\n    return chosen",
        "medium",
        ("python", "sampling"),
    ),
    (
        "code-005",
        "python_code",
        "def tokens_per_second(token_count: int, elapsed_seconds: float) -> float | None:\n    if elapsed_seconds <= 0:\n        return None\n    return token_count / elapsed_seconds",
        "short",
        ("python", "metrics"),
    ),
    (
        "qa-001",
        "question_answer",
        "Question: Why should a benchmark include warm-up requests?\nAnswer: Warm-up requests reduce the influence of one-time initialization and runtime setup on the timed measurements. Their count and treatment should be recorded.",
        "medium",
        ("question", "benchmarking"),
    ),
    (
        "qa-002",
        "question_answer",
        "Question: Does installing a framework prove that GPU inference works?\nAnswer: No. The framework, driver, device, model format, and selected runtime must all support the intended execution path. Availability should be checked directly.",
        "medium",
        ("question", "hardware"),
    ),
    (
        "qa-003",
        "question_answer",
        "Question: Why keep missing benchmark metrics as null?\nAnswer: A null value distinguishes an unmeasured or unsupported metric from a measured zero. This prevents reports from implying results that were never observed.",
        "medium",
        ("question", "metrics"),
    ),
    (
        "qa-004",
        "question_answer",
        "Question: What must stay constant when comparing backends?\nAnswer: Use the same model, model precision, input samples, prompt formatting, generation settings, warm-up procedure, and measurement definitions wherever the backends permit. Document any unavoidable differences.",
        "medium",
        ("question", "comparisons"),
    ),
    (
        "qa-005",
        "question_answer",
        "Question: Can a synthetic corpus represent production traffic?\nAnswer: Not completely. It is useful for deterministic offline tests and controlled experiments, while real workload conclusions require representative inputs with appropriate privacy and licensing review.",
        "medium",
        ("question", "datasets"),
    ),
    (
        "summary-001",
        "summarization_prompts",
        "Summarize the following note in one sentence: A benchmark should warm up the model, repeat the timed request, record token counts, and report a median latency instead of relying on one observation.",
        "medium",
        ("summarization",),
    ),
    (
        "summary-002",
        "summarization_prompts",
        "Summarize the following note in three bullets: The model is loaded before timing. Each backend receives the same manifest inputs. Missing memory measurements remain unavailable rather than being estimated.",
        "medium",
        ("summarization",),
    ),
    (
        "summary-003",
        "summarization_prompts",
        "Write a concise summary of this technical note: Tokenization converts text into model-specific IDs. Token count depends on the tokenizer and special-token settings. Character count alone cannot predict the exact sequence length.",
        "medium",
        ("summarization", "tokenization"),
    ),
    (
        "summary-004",
        "summarization_prompts",
        "Summarize this experiment log for a colleague: CPU execution was used. The same ten samples were processed twice with the same tokenizer. The second pass produced the same token counts, and no model inference was performed.",
        "medium",
        ("summarization", "reproducibility"),
    ),
    (
        "summary-005",
        "summarization_prompts",
        "Produce a short abstract from this note: A local text corpus supports offline pipeline tests. A separately requested public dataset can add realistic text. Dataset identity, configuration, source, retrieval time, and license details belong in its metadata.",
        "medium",
        ("summarization", "datasets"),
    ),
)


class SyntheticDatasetLoader(DatasetLoader):
    """Load the project-authored benchmark corpus without network access."""

    @property
    def metadata(self) -> DatasetMetadata:
        return DatasetMetadata(
            dataset_name="synthetic-benchmark",
            version="1.0.0",
            configuration="built-in",
            source="Original synthetic samples authored for this project.",
            license_info="MIT",
            sample_count=len(_SAMPLES),
        )

    def load(self, limit: int | None = None) -> DatasetCollection:
        if limit is not None and limit < 1:
            raise ValueError("Sample limit must be at least 1.")
        selected = _SAMPLES if limit is None else _SAMPLES[:limit]
        samples = [
            DatasetSample(
                sample_id=sample_id,
                category=category,
                input_text=input_text,
                length_category=length_category,
                source_info="Original synthetic benchmark sample; project-authored.",
                tags=list(tags),
            )
            for sample_id, category, input_text, length_category, tags in selected
        ]
        metadata = self.metadata.model_copy(update={"sample_count": len(samples)})
        return DatasetCollection(metadata=metadata, samples=samples)
