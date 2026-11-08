"""Tokenizer contract and explicit optional-dependency errors."""

from abc import ABC, abstractmethod


class TokenizerUnavailableError(RuntimeError):
    """Raised when a requested tokenizer cannot be loaded."""


class TokenizerBackend(ABC):
    """Small common API for local and optional tokenizer implementations."""

    @property
    @abstractmethod
    def tokenizer_name(self) -> str:
        """Return the stable tokenizer identifier."""

    @abstractmethod
    def encode(self, text: str) -> list[str]:
        """Convert text into a sequence of visible token representations."""

    @abstractmethod
    def decode(self, tokens: list[str]) -> str:
        """Convert token representations back to readable text."""
