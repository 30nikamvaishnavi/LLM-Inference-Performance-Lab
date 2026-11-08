"""Dependency-free deterministic tokenizer for offline experiments."""

import re

from app.tokenization.base import TokenizerBackend

_TOKEN_PATTERN = re.compile(r"\w+|[^\w\s]", re.UNICODE)
_CLOSING_PUNCTUATION = re.compile(r"\s+([,.;:!?%)\]}])")
_OPENING_PUNCTUATION = re.compile(r"([\[(\{])\s+")


class SimpleTokenizer(TokenizerBackend):
    """Split Unicode word runs and punctuation without model-specific claims."""

    @property
    def tokenizer_name(self) -> str:
        return "simple"

    def encode(self, text: str) -> list[str]:
        return _TOKEN_PATTERN.findall(text)

    def decode(self, tokens: list[str]) -> str:
        text = " ".join(tokens)
        text = _CLOSING_PUNCTUATION.sub(r"\1", text)
        return _OPENING_PUNCTUATION.sub(r"\1", text)
