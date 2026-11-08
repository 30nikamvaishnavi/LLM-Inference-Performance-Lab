"""Tests for dependency-free and optional tokenization functionality."""

import pytest

from app.datasets.registry import DatasetRegistry
from app.tokenization.base import TokenizerUnavailableError
from app.tokenization.registry import TokenizerRegistry
from app.tokenization.simple import SimpleTokenizer


def test_simple_tokenizer_counts_encodes_and_decodes() -> None:
    tokenizer = SimpleTokenizer()
    text = "Hello, tokenizer lab!"
    tokens = tokenizer.encode(text)

    assert tokens == ["Hello", ",", "tokenizer", "lab", "!"]
    assert len(tokens) == 5
    assert tokenizer.decode(tokens) == text


def test_batch_and_comparison_helpers() -> None:
    registry = TokenizerRegistry()
    tokenizer = registry.get("simple")

    assert registry.batch_encode(tokenizer, ["one", "two words"]) == [
        ["one"],
        ["two", "words"],
    ]
    assert registry.compare("two words", [tokenizer]) == {"simple": 2}


def test_tokenize_text_returns_measured_count_and_character_ratio() -> None:
    registry = TokenizerRegistry()

    result = registry.tokenize_text(registry.get(), "one two")

    assert result["token_count"] == 2
    assert result["character_count"] == 7
    assert result["characters_per_token"] == 3.5
    assert result["elapsed_seconds"] >= 0
    assert result["tokens_per_second"] is None or result["tokens_per_second"] >= 0


def test_tokenization_statistics_has_length_histogram(tmp_path) -> None:
    registry = DatasetRegistry(tmp_path / "data")
    collection = registry.load("synthetic-benchmark", limit=8)
    token_registry = TokenizerRegistry()

    stats = token_registry.statistics(
        collection.metadata.dataset_name,
        collection.samples,
        token_registry.get(),
    )

    assert stats.sample_count == 8
    assert stats.token_count > 0
    assert sum(stats.input_length_distribution.values()) == 8
    assert stats.mean_characters_per_token is not None


def test_unknown_tokenizer_and_huggingface_file_paths_fail_clearly() -> None:
    registry = TokenizerRegistry()

    with pytest.raises(TokenizerUnavailableError, match="Unknown tokenizer"):
        registry.get("not-a-tokenizer")
    with pytest.raises(ValueError, match="not a local filesystem path"):
        registry.get("hf:../../local-model")


def test_missing_transformers_dependency_is_reported_without_network(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import builtins

    original_import = builtins.__import__

    def reject_transformers(name, *args, **kwargs):
        if name == "transformers":
            raise ImportError("optional dependency is unavailable")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", reject_transformers)

    with pytest.raises(TokenizerUnavailableError, match="optional transformers extra"):
        TokenizerRegistry().get("hf:org/model")
