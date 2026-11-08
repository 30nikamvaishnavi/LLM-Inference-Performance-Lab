"""CPU-only, explicitly loaded Hugging Face Transformers inference backend."""

import gc
import logging
import re
import threading
import time
from collections.abc import Callable
from typing import Any

from app.core.config import Settings
from app.core.schemas import (
    BackendCapabilities,
    InferenceRequest,
    InferenceResponse,
)
from app.inference.base import InferenceBackend
from app.inference.exceptions import InferenceError
from app.inference.process_metrics import ProcessMetricsMonitor
from app.inference.schemas import (
    GenerateRequest,
    GenerateResponse,
    ModelStatus,
    ModelTokenizeRequest,
    ModelTokenizeResponse,
)

logger = logging.getLogger(__name__)
_MODEL_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*(/[A-Za-z0-9][A-Za-z0-9._-]*)?$")


class TransformersBackend(InferenceBackend):
    """Load one causal language model and its tokenizer on CPU only.

    Dependencies and weights are not loaded at application startup. A caller
    must explicitly invoke `load_model`; network access is separately gated
    by `allow_download`.
    """

    def __init__(
        self,
        settings: Settings | None = None,
        *,
        torch_module: Any | None = None,
        tokenizer_loader: Callable[..., Any] | None = None,
        model_loader: Callable[..., Any] | None = None,
    ) -> None:
        self.settings = settings or Settings()
        self._torch = torch_module
        self._tokenizer_loader = tokenizer_loader
        self._model_loader = model_loader
        self._model: Any | None = None
        self._tokenizer: Any | None = None
        self._model_id: str | None = None
        self._revision: str | None = None
        self._loading_duration_seconds: float | None = None
        self._max_sequence_length: int | None = None
        self._lock = threading.RLock()

    @property
    def backend_name(self) -> str:
        return "transformers"

    @property
    def required_packages(self) -> tuple[str, ...]:
        return ("torch", "transformers")

    def is_available(self) -> bool:
        """Report whether imports work and PyTorch can execute on CPU."""
        try:
            torch = self._torch or self._import_torch()
            return bool(torch.tensor([1], device="cpu").device.type == "cpu")
        except (ImportError, OSError, RuntimeError, AttributeError):
            return False

    def get_capabilities(self) -> BackendCapabilities:
        return BackendCapabilities(
            supported_devices=["cpu"],
            features=[
                "causal-generation",
                "model-tokenizer",
                "explicit-model-loading",
                "offline-cache-only-loading",
            ],
            supports_time_to_first_token=False,
            supports_concurrent_requests=False,
        )

    def load_model(
        self,
        model_name: str | None = None,
        *,
        revision: str | None = None,
        allow_download: bool = False,
        **kwargs: object,
    ) -> None:
        """Load tokenizer and model explicitly, with all tensors placed on CPU."""
        del kwargs
        model_id = model_name or self.settings.inference_model_id
        if not _MODEL_ID_PATTERN.fullmatch(model_id):
            raise InferenceError("Use a valid Hugging Face model ID, not a local path.")
        if revision is not None and not revision.strip():
            raise InferenceError("Model revision must not be blank.")
        with self._lock:
            if self._model is not None:
                if self._model_id == model_id and self._revision == revision:
                    return
                raise InferenceError(
                    "A model is already loaded. Unload it before loading another model."
                )
            torch = self._torch or self._import_torch()
            if not self._tokenizer_loader or not self._model_loader:
                try:
                    from transformers import AutoModelForCausalLM, AutoTokenizer
                except ImportError as error:
                    raise InferenceError(
                        "Transformers is unavailable. Install the optional "
                        "'transformers' dependency group."
                    ) from error
                tokenizer_loader = AutoTokenizer.from_pretrained
                model_loader = AutoModelForCausalLM.from_pretrained
            else:
                tokenizer_loader = self._tokenizer_loader
                model_loader = self._model_loader

            load_kwargs: dict[str, object] = {
                "cache_dir": str(self.settings.model_cache_dir),
                "local_files_only": not allow_download,
                "trust_remote_code": False,
            }
            if revision is not None:
                load_kwargs["revision"] = revision
            started = time.perf_counter()
            tokenizer: Any | None = None
            try:
                tokenizer = tokenizer_loader(model_id, **load_kwargs)
                model = model_loader(model_id, **load_kwargs)
                model.to(torch.device("cpu"))
                model.eval()
            except Exception as error:
                del tokenizer
                gc.collect()
                logger.warning(
                    "Model %s load failed (%s)",
                    model_id,
                    type(error).__name__,
                )
                if self._is_out_of_memory(error):
                    raise InferenceError(
                        "Insufficient memory while loading the model. "
                        "Unload other applications or use a smaller model."
                    ) from error
                raise InferenceError(
                    f"Unable to load model {model_id!r}. Check the model ID, "
                    "revision, cache, network permission, and installed packages."
                ) from error

            self._torch = torch
            self._model = model
            self._tokenizer = tokenizer
            self._model_id = model_id
            resolved_revision = getattr(getattr(model, "config", None), "_commit_hash", None)
            if not isinstance(resolved_revision, str):
                resolved_revision = getattr(tokenizer, "init_kwargs", {}).get("_commit_hash")
            self._revision = resolved_revision or revision
            self._loading_duration_seconds = time.perf_counter() - started
            self._max_sequence_length = self._infer_max_sequence_length(model, tokenizer)
            logger.info(
                "Loaded Transformers model %s on CPU (revision=%s)",
                model_id,
                self._revision or "unresolved",
            )

    def unload_model(self) -> None:
        """Release model/tokenizer references and collect Python objects."""
        with self._lock:
            self._model = None
            self._tokenizer = None
            self._model_id = None
            self._revision = None
            self._loading_duration_seconds = None
            self._max_sequence_length = None
        gc.collect()

    def model_status(self) -> ModelStatus:
        """Report configured, cached, loaded, and ready state separately."""
        with self._lock:
            loaded = self._model is not None
            status_model_id = self._model_id or self.settings.inference_model_id
            downloaded = loaded or self._is_model_cached(status_model_id)
            return ModelStatus(
                backend_name=self.backend_name,
                configured_model_id=self.settings.inference_model_id,
                downloaded=downloaded,
                loaded=loaded,
                ready=loaded and self._tokenizer is not None,
                model_id=self._model_id,
                revision=self._revision,
                device="cpu",
                loading_duration_seconds=self._loading_duration_seconds,
                max_sequence_length=self._max_sequence_length,
                tokenizer_name=f"transformers:{self._model_id}" if loaded else None,
            )

    def tokenize(self, request: ModelTokenizeRequest) -> ModelTokenizeResponse:
        """Tokenize using the actual tokenizer loaded with this model."""
        _, tokenizer, model_id = self._require_loaded()
        started = time.perf_counter()
        encoded = tokenizer(
            request.text,
            add_special_tokens=request.add_special_tokens,
            truncation=request.truncation,
            max_length=request.max_length if request.truncation else None,
        )
        token_ids = self._flatten_ids(encoded["input_ids"])
        tokens = list(tokenizer.convert_ids_to_tokens(token_ids))
        elapsed_ms = (time.perf_counter() - started) * 1000
        return ModelTokenizeResponse(
            model_id=model_id,
            tokenizer_name=f"transformers:{model_id}",
            tokens=tokens,
            token_ids=token_ids,
            token_count=len(token_ids),
            elapsed_ms=elapsed_ms,
            special_tokens=self._special_tokens(tokenizer),
            max_sequence_length=self._max_sequence_length,
        )

    def batch_tokenize(self, texts: list[str]) -> list[dict[str, Any]]:
        """Tokenize a bounded batch with the loaded model tokenizer."""
        if len(texts) > 256:
            raise InferenceError("A tokenization batch cannot exceed 256 texts.")
        if any(not text for text in texts):
            raise InferenceError("Every batch text must be non-empty.")
        _, tokenizer, _ = self._require_loaded()
        started = time.perf_counter()
        encoded = tokenizer(texts, padding=False, truncation=False)
        input_ids = encoded["input_ids"]
        return [
            {
                "token_ids": self._flatten_ids(ids),
                "token_count": len(ids),
                "tokenizer_name": f"transformers:{self._model_id}",
                "elapsed_ms": (time.perf_counter() - started) * 1000,
            }
            for ids in input_ids
        ]

    def get_tokenizer_adapter(self) -> "ModelTokenizerAdapter":
        """Return the loaded model tokenizer through the shared token interface."""
        _, tokenizer, model_id = self._require_loaded()
        return ModelTokenizerAdapter(tokenizer, model_id)

    def generate_request(self, request: GenerateRequest) -> GenerateResponse:
        """Generate on CPU and return only measurements taken in this call."""
        if not request.prompt.strip():
            raise InferenceError("Prompt must not be empty.")
        if request.max_new_tokens > self.settings.inference_max_new_tokens:
            raise InferenceError(
                "max_new_tokens exceeds the configured safety limit of "
                f"{self.settings.inference_max_new_tokens}."
            )
        if request.temperature == 0 and request.top_k is not None:
            raise InferenceError("top_k requires temperature greater than zero.")
        with self._lock:
            model, tokenizer, model_id = self._require_loaded()
            if request.seed is not None:
                self._torch.manual_seed(request.seed)

            tokenization_started = time.perf_counter()
            encoded = tokenizer(request.prompt, return_tensors="pt")
            input_ids = encoded["input_ids"].to(self._torch.device("cpu"))
            attention_mask = encoded.get("attention_mask")
            if attention_mask is not None:
                attention_mask = attention_mask.to(self._torch.device("cpu"))
            tokenization_ms = (time.perf_counter() - tokenization_started) * 1000
            input_token_count = int(input_ids.shape[-1])
            expected_total = input_token_count + request.max_new_tokens
            if self._max_sequence_length is not None and expected_total > self._max_sequence_length:
                raise InferenceError(
                    "Prompt and requested generation exceed the model's maximum "
                    f"sequence length ({self._max_sequence_length})."
                )

            generation_kwargs: dict[str, object] = {
                "input_ids": input_ids,
                "max_new_tokens": request.max_new_tokens,
                "do_sample": request.temperature > 0,
                "pad_token_id": (
                    tokenizer.pad_token_id
                    if tokenizer.pad_token_id is not None
                    else tokenizer.eos_token_id
                ),
            }
            if attention_mask is not None:
                generation_kwargs["attention_mask"] = attention_mask
            if request.temperature > 0:
                generation_kwargs["temperature"] = request.temperature
                if request.top_k is not None:
                    generation_kwargs["top_k"] = request.top_k

            monitor = ProcessMetricsMonitor()
            monitor.start()
            generation_started = time.perf_counter()
            try:
                with self._torch.inference_mode():
                    sequences = model.generate(**generation_kwargs)
            except Exception as error:
                if self._is_out_of_memory(error):
                    raise InferenceError(
                        "Insufficient memory during generation. Try fewer input "
                        "tokens or a shorter generation."
                    ) from error
                logger.warning("Model generation failed (%s)", type(error).__name__)
                raise InferenceError("Model generation failed.") from error
            finally:
                generation_duration = time.perf_counter() - generation_started
                memory_bytes, cpu_percent = monitor.stop()

            sequence_ids = self._flatten_ids(sequences[0])
            generated_ids = sequence_ids[input_token_count:]
            generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
            generated_count = len(generated_ids)
            tokens_per_second = (
                generated_count / generation_duration if generation_duration > 0 else None
            )
            return GenerateResponse(
                model_name=model_id,
                backend_name=self.backend_name,
                generated_text=generated_text,
                output_token_count=generated_count,
                total_latency_ms=generation_duration * 1000,
                time_to_first_token_ms=None,
                time_per_output_token_ms=(
                    generation_duration * 1000 / generated_count if generated_count else None
                ),
                output_tokens_per_second=tokens_per_second,
                peak_memory_bytes=memory_bytes,
                input_token_count=input_token_count,
                total_sequence_length=len(sequence_ids),
                tokenization_latency_ms=tokenization_ms,
                cpu_utilization_percent=cpu_percent,
                tokenizer_name=f"transformers:{model_id}",
                device="cpu",
                model_revision=self._revision,
                model_loading_duration_seconds=self._loading_duration_seconds,
            )

    def generate(self, request: InferenceRequest) -> InferenceResponse:
        """Satisfy the shared backend interface."""
        if request.backend_name != self.backend_name:
            raise InferenceError("Inference request backend_name does not match this backend.")
        if request.temperature == 0 and request.top_k is not None:
            raise InferenceError("top_k requires temperature greater than zero.")
        adapted = GenerateRequest(
            prompt=request.prompt,
            max_new_tokens=request.max_new_tokens,
            temperature=request.temperature,
            top_k=request.top_k,
        )
        return self.generate_request(adapted)

    def _require_loaded(self) -> tuple[Any, Any, str]:
        if self._model is None or self._tokenizer is None or self._model_id is None:
            raise InferenceError("No model is loaded. Load a model explicitly first.")
        return self._model, self._tokenizer, self._model_id

    def _is_model_cached(self, model_id: str) -> bool:
        try:
            from huggingface_hub import try_to_load_from_cache
        except ImportError:
            return False
        files = (
            "config.json",
            "model.safetensors",
            "pytorch_model.bin",
            "model.safetensors.index.json",
            "pytorch_model.bin.index.json",
        )
        cached = [
            try_to_load_from_cache(
                model_id,
                filename,
                cache_dir=str(self.settings.model_cache_dir),
            )
            for filename in files
        ]
        return isinstance(cached[0], str) and any(isinstance(item, str) for item in cached[1:])

    @staticmethod
    def _import_torch() -> Any:
        try:
            import torch
        except (ImportError, OSError) as error:
            raise InferenceError(
                "PyTorch is unavailable. Install the CPU-only PyTorch wheel "
                "inside this project's virtual environment."
            ) from error
        return torch

    @staticmethod
    def _infer_max_sequence_length(model: Any, tokenizer: Any) -> int | None:
        config = getattr(model, "config", None)
        candidates = [
            getattr(config, "max_position_embeddings", None),
            getattr(config, "n_positions", None),
            getattr(tokenizer, "model_max_length", None),
        ]
        for value in candidates:
            if isinstance(value, int) and 1 <= value < 1_000_000:
                return value
        return None

    @staticmethod
    def _flatten_ids(value: Any) -> list[int]:
        if hasattr(value, "tolist"):
            value = value.tolist()
        if value and isinstance(value[0], list):
            value = value[0]
        return [int(token_id) for token_id in value]

    @staticmethod
    def _special_tokens(tokenizer: Any) -> dict[str, str | None]:
        result: dict[str, str | None] = {}
        for name in ("bos", "eos", "unk", "pad", "sep", "cls"):
            token = getattr(tokenizer, f"{name}_token", None)
            result[name] = str(token) if token is not None else None
        return result

    @staticmethod
    def _is_out_of_memory(error: BaseException) -> bool:
        message = str(error).casefold()
        return "out of memory" in message or "not enough memory" in message


class ModelTokenizerAdapter:
    """Expose actual model tokenizer operations to the manifest pipeline."""

    def __init__(self, tokenizer: Any, model_id: str) -> None:
        self._tokenizer = tokenizer
        self._name = f"transformers:{model_id}"

    @property
    def tokenizer_name(self) -> str:
        return self._name

    def encode(self, text: str) -> list[str]:
        token_ids = self._tokenizer(
            text,
            add_special_tokens=True,
        )["input_ids"]
        return list(self._tokenizer.convert_ids_to_tokens(token_ids))

    def decode(self, tokens: list[str]) -> str:
        token_ids = self._tokenizer.convert_tokens_to_ids(tokens)
        if any(token_id is None for token_id in token_ids):
            raise ValueError("The model tokenizer could not map every token to an ID.")
        return str(self._tokenizer.decode(token_ids, skip_special_tokens=False))
