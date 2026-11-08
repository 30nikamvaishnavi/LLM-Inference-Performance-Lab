"""Deterministic doubles for optional inference integrations."""

from contextlib import nullcontext
from types import SimpleNamespace

import pytest


class FakeTensor:
    def __init__(self, value):
        self.value = value
        self.shape = (1, len(value[0]) if value and isinstance(value[0], list) else len(value))

    def to(self, device):
        del device
        return self

    def tolist(self):
        return self.value

    def __getitem__(self, index):
        value = self.value[index]
        if isinstance(value, list):
            return FakeTensor(value)
        return value


class FakeTokenizer:
    pad_token_id = None
    eos_token_id = 99
    model_max_length = 32
    bos_token = "<bos>"
    eos_token = "<eos>"
    unk_token = None
    pad_token = None
    sep_token = None
    cls_token = None
    init_kwargs = {"_commit_hash": "fake-revision"}

    def __call__(
        self,
        text,
        return_tensors=None,
        add_special_tokens=True,
        truncation=False,
        max_length=None,
        padding=False,
    ):
        del add_special_tokens, padding
        if isinstance(text, list):
            return {"input_ids": [self._encode_one(item) for item in text]}
        token_ids = self._encode_one(text)
        if truncation and max_length is not None:
            token_ids = token_ids[:max_length]
        if return_tensors == "pt":
            return {
                "input_ids": FakeTensor([token_ids]),
                "attention_mask": FakeTensor([[1] * len(token_ids)]),
            }
        return {"input_ids": token_ids}

    def _encode_one(self, text):
        return [ord(token[0]) % 80 for token in text.split()][:32]

    def tokenize(self, text):
        return text.split()

    def convert_ids_to_tokens(self, token_ids):
        return [f"token-{token_id}" for token_id in token_ids]

    def convert_tokens_to_ids(self, tokens):
        return [int(token.rsplit("-", 1)[-1]) for token in tokens]

    def decode(self, token_ids, skip_special_tokens=True):
        del skip_special_tokens
        return " ".join(f"generated-{token_id}" for token_id in token_ids)


class FakeModel:
    def __init__(self):
        self.config = SimpleNamespace(
            _commit_hash="model-revision",
            max_position_embeddings=32,
        )
        self.device = None
        self.evaluating = False
        self.generation_calls = []

    def to(self, device):
        self.device = device
        return self

    def eval(self):
        self.evaluating = True
        return self

    def generate(self, input_ids, **kwargs):
        self.generation_calls.append(kwargs)
        input_values = input_ids.tolist()[0]
        new_ids = [71] * min(kwargs["max_new_tokens"], 2)
        return FakeTensor([input_values + new_ids])


class FakeTorch:
    class Device:
        def __init__(self, device_type):
            self.type = device_type

        def __str__(self):
            return self.type

    def __init__(self):
        self.seed = None
        self.inference_mode_calls = 0
        self.__version__ = "mock-torch"

    def tensor(self, values, device):
        del values
        if isinstance(device, str):
            device = self.device(device)
        return SimpleNamespace(device=device)

    def device(self, device_type):
        return self.Device(device_type)

    def manual_seed(self, seed):
        self.seed = seed

    def inference_mode(self):
        self.inference_mode_calls += 1
        return nullcontext()


@pytest.fixture
def fake_inference_components():
    torch = FakeTorch()
    tokenizer = FakeTokenizer()
    model = FakeModel()
    calls = {"tokenizer": [], "model": []}

    def tokenizer_loader(model_id, **kwargs):
        calls["tokenizer"].append((model_id, kwargs))
        return tokenizer

    def model_loader(model_id, **kwargs):
        calls["model"].append((model_id, kwargs))
        return model

    return torch, tokenizer, model, tokenizer_loader, model_loader, calls
