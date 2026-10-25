import numpy as np
import re

from hallucination_detector.data.schema import RAGTruthRecord
from hallucination_detector.inference.model_registry import MODEL_CATALOG
from hallucination_detector.inference.model_registry import ModelRegistry
from hallucination_detector.modeling.deberta_classifier import (
    RAGTruthDebertaDataset,
    format_deberta_input,
)
from hallucination_detector.modeling.deberta_encoding import encode_deberta_input
import hallucination_detector.inference.model_registry as model_registry_module


class RecordingTokenizer:
    def __init__(self):
        self.seen = []
        self.vocab = {}

    def __call__(self, text, **kwargs):
        self.seen.append((text, kwargs))
        matches = list(re.finditer(r"\[CONTEXT\]|\[ANSWER\]|\w+|[^\w\s]", text))
        ids = [self.vocab.setdefault(match.group(), len(self.vocab) + 10) for match in matches]
        return {"input_ids": ids, "offset_mapping": [(m.start(), m.end()) for m in matches]}

    def num_special_tokens_to_add(self, pair=False):
        return 2

    def build_inputs_with_special_tokens(self, ids):
        return [101, *ids, 102]

    def create_token_type_ids_from_sequences(self, ids):
        return [0] * (len(ids) + 2)


def make_record():
    return RAGTruthRecord(
        id="1",
        source_id="s1",
        question="Where?",
        context="  Marie   was born in Warsaw. ",
        answer=" Marie was born in Paris. ",
        hallucinated=True,
    )


def test_deberta_training_formatter_matches_registry_ca_format():
    assert format_deberta_input("  Marie   was born in Warsaw. ", " Marie was born in Paris. ") == (
        "[CONTEXT] Marie was born in Warsaw. [ANSWER] Marie was born in Paris."
    )


def test_deberta_training_dataset_uses_same_single_string_and_no_longformer_mask():
    tokenizer = RecordingTokenizer()
    dataset = RAGTruthDebertaDataset([make_record()], tokenizer, max_length=512)

    item = dataset[0]

    text, options = tokenizer.seen[0]
    assert text == format_deberta_input(make_record().context, make_record().answer)
    assert options == {"add_special_tokens": False, "return_offsets_mapping": True, "truncation": False}
    assert set(item) == {"input_ids", "attention_mask", "token_type_ids", "labels"}
    assert item["labels"].item() == 1


def test_deberta_encoding_preserves_original_tokens_when_sequence_fits():
    tokenizer = RecordingTokenizer()
    context = "The short passage says seven boats arrived."
    answer = "Seven boats arrived."
    text = format_deberta_input(context, answer)
    raw = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True, truncation=False)
    expected = tokenizer.build_inputs_with_special_tokens(raw["input_ids"])
    actual = encode_deberta_input(context, answer, tokenizer, max_length=64)
    assert actual["input_ids"] == expected
    assert actual["truncation"] == {
        "context_truncated": False, "answer_truncated": False,
        "input_tokens": len(expected), "max_length": 64,
    }


def test_deberta_long_context_is_shortened_before_answer():
    tokenizer = RecordingTokenizer()
    answer = "The answer is definitely preserved."
    full = encode_deberta_input("early " * 60 + "late fact", answer, tokenizer, max_length=18)
    answer_ids = tokenizer(
        "[ANSWER] " + answer, add_special_tokens=False,
        return_offsets_mapping=True, truncation=False,
    )["input_ids"]
    # The helper retains the answer marker and all answer tokens when that
    # segment fits, while dropping context tokens first.
    assert full["truncation"]["context_truncated"] is True
    assert full["truncation"]["answer_truncated"] is False
    assert all(token_id in full["input_ids"] for token_id in answer_ids)
    assert len(full["input_ids"]) <= 18


def test_deberta_answer_over_budget_is_explicitly_reported():
    tokenizer = RecordingTokenizer()
    full = encode_deberta_input("small context", "claim " * 40, tokenizer, max_length=12)
    assert full["truncation"]["answer_truncated"] is True
    assert full["truncation"]["context_truncated"] is True
    assert len(full["input_ids"]) <= 12


def test_registry_input_modes_match_actual_training_inputs_and_flag_qca_selector_gap():
    assert MODEL_CATALOG["bilstm-attention"].input_mode == "Q+C+A"
    assert MODEL_CATALOG["deberta"].input_mode == "C+A"
    assert MODEL_CATALOG["longformer-head"].input_mode == "C+A"
    assert MODEL_CATALOG["longformer-qca"].provenance_status == "UNVERIFIED"
    assert "IDF statistics" in MODEL_CATALOG["longformer-qca"].limitations


def test_registry_deberta_tokenizer_matches_training_fast_tokenizer(monkeypatch, tmp_path):
    (tmp_path / "model.safetensors").write_bytes(b"checkpoint marker")
    calls = {}

    class DummyModel:
        def to(self, device):
            return self

        def eval(self):
            return self

    class DummyAutoTokenizer:
        @staticmethod
        def from_pretrained(path, **kwargs):
            calls.update(path=path, **kwargs)
            return object()

    monkeypatch.setattr(model_registry_module, "AutoTokenizer", DummyAutoTokenizer)
    monkeypatch.setattr(model_registry_module.AutoModelForSequenceClassification, "from_pretrained", lambda path: DummyModel())
    registry = ModelRegistry()
    monkeypatch.setattr(model_registry_module, "Path", lambda value: tmp_path if str(value).endswith("best_model") else __import__("pathlib").Path(value))
    registry._load_deberta("repo", "revision", "cpu")
    assert calls["use_fast"] is True


def test_registry_accepts_structured_context_for_inference(monkeypatch):
    structured_context = {"birthplace": "Warsaw", "country": "Poland"}

    class FakeTfidf:
        def predict_proba(self, records):
            assert records[0].context == structured_context
            return np.array([[0.8, 0.2]])

    registry = ModelRegistry()
    monkeypatch.setattr(registry, "load_model", lambda *args, **kwargs: FakeTfidf())

    result = registry.predict(
        "tfidf-lr", "Where was she born?", structured_context, "Warsaw, Poland."
    )

    assert result.prediction == "SUPPORTED"
    assert result.hallucination_probability == 0.2
import numpy as np

