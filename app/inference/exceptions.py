"""Inference errors that can be translated to structured API responses."""


class InferenceError(RuntimeError):
    """Expected inference/backend failure safe to explain to the caller."""
