"""Mocked tests for the CPU Transformers backend; no weights are downloaded."""

import builtins

import pytest

from app.core.config import Settings
from app.core.schemas import InferenceRequest
from app.inference.backends.transformers_backend import TransformersBackend
from app.inference.exceptions import InferenceError
from app.inference.schemas import GenerateRequest, ModelTokenizeRequest


def create_backend(tmp_path, fake_inference_components, **settings_kwargs):
    torch, _, _, tokenizer_loader, model_loader, _ = fake_inference_components
    settings = Settings(
        _env_file=None,
        model_cache_dir=tmp_path / "model-cache",
        inference_max_new_tokens=8,
        **settings_kwargs,
    )
    return TransformersBackend(
        settings,
        torch_module=torch,
        tokenizer_loader=tokenizer_loader,
        model_loader=model_loader,
    )


def test_backend_is_cpu_only_and_reports_features(tmp_path, fake_inference_components):
    backend = create_backend(tmp_path, fake_inference_components)

    assert backend.is_available() is True
    capabilities = backend.get_capabilities()
    assert capabilities.supported_devices == ["cpu"]
    assert capabilities.supports_time_to_first_token is False
    assert capabilities.supports_concurrent_requests is False


def test_model_loading_is_explicit_cpu_only_and_records_revision(
    tmp_path, fake_inference_components
):
    backend = create_backend(tmp_path, fake_inference_components)
    _, _, model, _, _, calls = fake_inference_components

    backend.load_model("sshleifer/tiny-gpt2", revision="main")
    status = backend.model_status()

    assert calls["tokenizer"][0][1]["local_files_only"] is True
    assert calls["tokenizer"][0][1]["trust_remote_code"] is False
    assert calls["tokenizer"][0][1]["revision"] == "main"
    assert model.device.type == "cpu"
    assert model.evaluating is True
    assert status.loaded is True
    assert status.ready is True
    assert status.revision == "model-revision"
    assert status.tokenizer_name == "transformers:sshleifer/tiny-gpt2"


def test_explicit_download_permission_and_single_loaded_model(tmp_path, fake_inference_components):
    backend = create_backend(tmp_path, fake_inference_components)
    _, _, _, _, _, calls = fake_inference_components
    backend.load_model("sshleifer/tiny-gpt2", allow_download=True)

    assert calls["tokenizer"][0][1]["local_files_only"] is False
    with pytest.raises(InferenceError, match="Unload it"):
        backend.load_model("another/small-model")


def test_model_tokenizer_encodes_decodes_and_reports_special_tokens(
    tmp_path, fake_inference_components
):
    backend = create_backend(tmp_path, fake_inference_components)
    backend.load_model("sshleifer/tiny-gpt2")

    response = backend.tokenize(ModelTokenizeRequest(text="small model", add_special_tokens=False))
    batch = backend.batch_tokenize(["small model", "second"])

    assert response.tokenizer_name == "transformers:sshleifer/tiny-gpt2"
    assert response.token_count == len(response.token_ids) == 2
    assert response.tokens == ["token-35", "token-29"]
    assert response.special_tokens["eos"] == "<eos>"
    assert response.max_sequence_length == 32
    assert len(batch) == 2
    assert batch[0]["token_count"] == 2


def test_generation_uses_inference_mode_and_reports_actual_token_metrics(
    tmp_path, fake_inference_components
):
    backend = create_backend(tmp_path, fake_inference_components)
    torch, _, model, _, _, _ = fake_inference_components
    backend.load_model("sshleifer/tiny-gpt2")

    response = backend.generate_request(
        GenerateRequest(prompt="small prompt", max_new_tokens=2, seed=13)
    )

    assert response.input_token_count == 2
    assert response.output_token_count == 2
    assert response.total_sequence_length == 4
    assert response.output_tokens_per_second is not None
    assert response.time_to_first_token_ms is None
    assert response.tokenizer_name == "transformers:sshleifer/tiny-gpt2"
    assert response.device == "cpu"
    assert torch.seed == 13
    assert torch.inference_mode_calls == 1
    assert model.generation_calls[0]["do_sample"] is False
    assert "temperature" not in model.generation_calls[0]


def test_sampling_options_and_shared_interface_validation(tmp_path, fake_inference_components):
    backend = create_backend(tmp_path, fake_inference_components)
    _, _, model, _, _, _ = fake_inference_components
    backend.load_model()

    backend.generate(
        InferenceRequest(
            model_name="sshleifer/tiny-gpt2",
            backend_name="transformers",
            prompt="sample this",
            max_new_tokens=2,
            temperature=0.7,
            top_k=10,
        )
    )
    assert model.generation_calls[0]["do_sample"] is True
    assert model.generation_calls[0]["temperature"] == 0.7
    assert model.generation_calls[0]["top_k"] == 10

    with pytest.raises(InferenceError, match="does not match"):
        backend.generate(
            InferenceRequest(
                model_name="sshleifer/tiny-gpt2",
                backend_name="other",
                prompt="wrong backend",
            )
        )
    with pytest.raises(InferenceError, match="top_k requires"):
        backend.generate_request(
            GenerateRequest(prompt="invalid", max_new_tokens=2, top_k=5, temperature=0)
        )


def test_generation_rejects_empty_prompt_length_overflow_and_safety_limit(
    tmp_path, fake_inference_components
):
    backend = create_backend(tmp_path, fake_inference_components)
    backend.load_model()
    with pytest.raises(InferenceError, match="empty"):
        backend.generate_request(GenerateRequest(prompt="  "))
    with pytest.raises(InferenceError, match="safety limit"):
        backend.generate_request(GenerateRequest(prompt="one", max_new_tokens=9))
    with pytest.raises(InferenceError, match="maximum sequence length"):
        backend.generate_request(GenerateRequest(prompt="a " * 31, max_new_tokens=2))


def test_generate_and_tokenize_require_loaded_model(tmp_path, fake_inference_components):
    backend = create_backend(tmp_path, fake_inference_components)

    with pytest.raises(InferenceError, match="No model is loaded"):
        backend.generate_request(GenerateRequest(prompt="prompt", max_new_tokens=2))
    with pytest.raises(InferenceError, match="No model is loaded"):
        backend.tokenize(ModelTokenizeRequest(text="prompt"))


def test_load_failures_are_reported_and_do_not_leave_partial_state(
    tmp_path, fake_inference_components
):
    torch, _, _, tokenizer_loader, _, _ = fake_inference_components

    def failing_loader(model_id, **kwargs):
        del model_id, kwargs
        raise OSError("offline cache miss")

    backend = TransformersBackend(
        Settings(_env_file=None, model_cache_dir=tmp_path / "cache"),
        torch_module=torch,
        tokenizer_loader=tokenizer_loader,
        model_loader=failing_loader,
    )
    with pytest.raises(InferenceError, match="Unable to load model"):
        backend.load_model("sshleifer/tiny-gpt2")
    assert backend.model_status().loaded is False


def test_missing_pytorch_is_reported(monkeypatch, tmp_path, fake_inference_components):
    backend = create_backend(tmp_path, fake_inference_components)

    def missing_torch():
        raise InferenceError("PyTorch unavailable")

    monkeypatch.setattr(backend, "_import_torch", missing_torch)
    backend._torch = None
    assert backend.is_available() is False


def test_missing_transformers_is_reported_without_network(
    monkeypatch, tmp_path, fake_inference_components
):
    torch, _, _, _, _, _ = fake_inference_components
    backend = TransformersBackend(Settings(_env_file=None), torch_module=torch)
    original_import = builtins.__import__

    def reject_transformers(name, *args, **kwargs):
        if name == "transformers":
            raise ImportError("not installed")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", reject_transformers)
    with pytest.raises(InferenceError, match="Transformers is unavailable"):
        backend.load_model("sshleifer/tiny-gpt2")


def test_unload_releases_model_and_status_state(tmp_path, fake_inference_components):
    backend = create_backend(tmp_path, fake_inference_components)
    backend.load_model()

    backend.unload_model()
    status = backend.model_status()

    assert status.loaded is False
    assert status.ready is False
    assert status.model_id is None
    assert status.revision is None
