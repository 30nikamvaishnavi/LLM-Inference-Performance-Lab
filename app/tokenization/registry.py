"""Tokenizer construction and comparison helpers."""

import logging
import time

from app.datasets.schemas import DatasetSample, TokenizationStatistics
from app.tokenization.base import TokenizerBackend, TokenizerUnavailableError
from app.tokenization.simple import SimpleTokenizer

logger = logging.getLogger(__name__)
_LENGTH_BUCKETS = (
    ("0-32", 0, 32),
    ("33-128", 33, 128),
    ("129-512", 129, 512),
    ("513+", 513, None),
)


class TokenizerRegistry:
    """Provide built-in tokenization and lazily loaded optional adapters."""

    def __init__(self) -> None:
        self._tokenizers: dict[str, TokenizerBackend] = {
            "simple": SimpleTokenizer(),
        }

    def register(self, tokenizer: TokenizerBackend) -> None:
        name = tokenizer.tokenizer_name
        if not name.strip():
            raise ValueError("Tokenizer names must not be blank.")
        if name in self._tokenizers:
            raise ValueError(f"Tokenizer {name!r} is already registered.")
        self._tokenizers[name] = tokenizer

    def get(self, name: str = "simple", allow_download: bool = False) -> TokenizerBackend:
        if not name.strip():
            raise ValueError("Tokenizer name must not be blank.")
        existing = self._tokenizers.get(name)
        if existing is not None:
            return existing
        if not name.startswith("hf:"):
            raise TokenizerUnavailableError(
                f"Unknown tokenizer {name!r}. Use 'simple' or 'hf:<model-id>'."
            )
        from app.tokenization.huggingface import HuggingFaceTokenizer

        tokenizer = HuggingFaceTokenizer(name.removeprefix("hf:"), allow_download)
        self._tokenizers[name] = tokenizer
        return tokenizer

    @staticmethod
    def tokenize_text(tokenizer: TokenizerBackend, text: str) -> dict[str, object]:
        """Tokenize once and measure elapsed time and text/token ratio."""
        started = time.perf_counter()
        tokens = tokenizer.encode(text)
        elapsed = time.perf_counter() - started
        count = len(tokens)
        return {
            "tokenizer_name": tokenizer.tokenizer_name,
            "tokens": tokens,
            "token_count": count,
            "character_count": len(text),
            "characters_per_token": len(text) / count if count else None,
            "elapsed_seconds": elapsed,
            "tokens_per_second": count / elapsed if elapsed > 0 else None,
        }

    @staticmethod
    def batch_encode(tokenizer: TokenizerBackend, texts: list[str]) -> list[list[str]]:
        return [tokenizer.encode(text) for text in texts]

    @staticmethod
    def compare(text: str, tokenizers: list[TokenizerBackend]) -> dict[str, int]:
        """Return token counts for one input using each named tokenizer."""
        counts: dict[str, int] = {}
        for tokenizer in tokenizers:
            if tokenizer.tokenizer_name in counts:
                raise ValueError(f"Duplicate tokenizer name {tokenizer.tokenizer_name!r}.")
            counts[tokenizer.tokenizer_name] = len(tokenizer.encode(text))
        return counts

    @staticmethod
    def statistics(
        dataset_name: str,
        samples: list[DatasetSample],
        tokenizer: TokenizerBackend,
    ) -> TokenizationStatistics:
        started = time.perf_counter()
        counts: list[int] = []
        total_characters = 0
        distribution = {label: 0 for label, _, _ in _LENGTH_BUCKETS}
        for sample in samples:
            token_count = len(tokenizer.encode(sample.input_text))
            counts.append(token_count)
            total_characters += len(sample.input_text)
            for label, minimum, maximum in _LENGTH_BUCKETS:
                if token_count >= minimum and (maximum is None or token_count <= maximum):
                    distribution[label] += 1
                    break
        elapsed = time.perf_counter() - started
        total_tokens = sum(counts)
        return TokenizationStatistics(
            tokenizer_name=tokenizer.tokenizer_name,
            dataset_name=dataset_name,
            sample_count=len(samples),
            token_count=total_tokens,
            tokens_per_second=total_tokens / elapsed if elapsed > 0 else None,
            mean_characters_per_token=(total_characters / total_tokens if total_tokens else None),
            input_length_distribution=distribution,
        )
