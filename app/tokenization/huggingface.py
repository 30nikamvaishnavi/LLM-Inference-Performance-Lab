"""Opt-in Hugging Face tokenizer adapter; cached-only unless requested."""

import re
from typing import Any

from app.tokenization.base import TokenizerBackend, TokenizerUnavailableError

_MODEL_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*(/[A-Za-z0-9][A-Za-z0-9._-]*)?$")


class HuggingFaceTokenizer(TokenizerBackend):
    """Adapt a Transformers AutoTokenizer to the lab tokenizer interface."""

    def __init__(self, model_id: str, allow_download: bool = False) -> None:
        if not _MODEL_ID_PATTERN.fullmatch(model_id):
            raise ValueError("Use a Hugging Face model ID, not a local filesystem path.")
        try:
            from transformers import AutoTokenizer
        except ImportError as error:
            raise TokenizerUnavailableError(
                "Hugging Face tokenizers require the optional transformers extra. "
                'Install it with: python -m pip install -e ".[transformers]"'
            ) from error
        try:
            self._tokenizer: Any = AutoTokenizer.from_pretrained(
                model_id,
                local_files_only=not allow_download,
            )
        except OSError as error:
            if not allow_download:
                raise TokenizerUnavailableError(
                    f"Tokenizer {model_id!r} is not cached locally. "
                    "Explicitly allow a download or use a cached tokenizer."
                ) from error
            raise TokenizerUnavailableError(
                f"Unable to load Hugging Face tokenizer {model_id!r}."
            ) from error
        self._model_id = model_id

    @property
    def tokenizer_name(self) -> str:
        return f"hf:{self._model_id}"

    def encode(self, text: str) -> list[str]:
        return list(self._tokenizer.tokenize(text))

    def decode(self, tokens: list[str]) -> str:
        token_ids = self._tokenizer.convert_tokens_to_ids(tokens)
        if any(token_id is None for token_id in token_ids):
            raise ValueError("Tokenizer could not map every token back to an ID.")
        return str(self._tokenizer.decode(token_ids, skip_special_tokens=False))
