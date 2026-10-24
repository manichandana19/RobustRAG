from __future__ import annotations

import numpy as np
import pytest
import torch

from hallucination_detector.data.schema import RAGTruthRecord
from hallucination_detector.data.bilstm_dataset import (
    Vocabulary,
    simple_tokenize,
    encode_qca_tokens,
    RAGTruthBiLSTMDataset,
    BiLSTMDataCollator,
)
from hallucination_detector.modeling.bilstm_classifier import (
    BahdanauAttention,
    BiLSTMClassifierNet,
    BiLSTMClassifier,
)
from hallucination_detector.training.bilstm_train import split_records, train_bilstm


@pytest.fixture
def sample_records() -> list[RAGTruthRecord]:
    return [
        RAGTruthRecord(
            id="rec_1",
            source_id="src_1",
            question="What is the capital of France?",
            context="France is a country in Western Europe. Paris is the capital city of France.",
            answer="Paris is the capital of France.",
            hallucinated=False,
            model="gpt-4",
            temperature=0.7,
            split="train",
            quality=None,
            annotations=[],
        ),
        RAGTruthRecord(
            id="rec_2",
            source_id="src_2",
            question="What is the capital of Germany?",
            context="Germany is a country in Central Europe with Berlin as its capital.",
            answer="Munich is the capital of Germany.",
            hallucinated=True,
            model="gpt-4",
            temperature=0.7,
            split="train",
            quality=None,
            annotations=[],
        ),
        RAGTruthRecord(
            id="rec_3",
            source_id="src_3",
            question="What is the capital of Italy?",
            context="Italy is in Southern Europe. Rome is the historic capital.",
            answer="Rome is the capital.",
            hallucinated=False,
            model="gpt-4",
            temperature=0.7,
            split="validation",
            quality=None,
            annotations=[],
        ),
        RAGTruthRecord(
            id="rec_4",
            source_id="src_4",
            question="What is the capital of Spain?",
            context="Spain is located in southwestern Europe. Madrid is its capital.",
            answer="Barcelona is the capital.",
            hallucinated=True,
            model="gpt-4",
            temperature=0.7,
            split="validation",
            quality=None,
            annotations=[],
        ),
        RAGTruthRecord(
            id="rec_5",
            source_id="src_5",
            question="What is the capital of Portugal?",
            context="Portugal is in southwestern Europe on the Iberian Peninsula. Lisbon is the capital.",
            answer="Lisbon is the capital of Portugal.",
            hallucinated=False,
            model="gpt-4",
            temperature=0.7,
            split="test",
            quality=None,
            annotations=[],
        ),
        RAGTruthRecord(
            id="rec_6",
            source_id="src_6",
            question="What is the capital of Austria?",
            context="Austria is a country in Central Europe. Vienna is the capital city.",
            answer="Salzburg is the capital.",
            hallucinated=True,
            model="gpt-4",
            temperature=0.7,
            split="test",
            quality=None,
            annotations=[],
        ),
    ]



def test_simple_tokenize():
    text = "What is the capital of France? It's Paris."
    tokens = simple_tokenize(text)
    assert "what" in tokens
    assert "france" in tokens
    assert "?" in tokens
    assert "paris" in tokens


def test_vocabulary_build_and_encode(sample_records):
    vocab = Vocabulary.build_from_records(sample_records, max_size=100, min_freq=1)
    assert len(vocab) > 2
    assert vocab.pad_id == 0
    assert vocab.unk_id == 1

    encoded = vocab.encode(["what", "unknownword123", "paris"])
    assert encoded[1] == vocab.unk_id
    assert encoded[0] != vocab.unk_id


def test_qca_truncation_preserves_qa():
    vocab = Vocabulary()
    vocab.add_token("question")
    vocab.add_token("context")
    vocab.add_token("answer")

    long_context = "word " * 200
    short_q = "what is this"
    short_a = "it is a test answer"

    encoded = encode_qca_tokens(
        question=short_q,
        context=long_context,
        answer=short_a,
        vocab=vocab,
        max_length=30,
        truncation_strategy="context_tail",
    )

    # Output length must not exceed max_length
    assert encoded["length"] <= 30
    assert len(encoded["input_ids"]) <= 30

    # Ensure answer tokens are preserved
    a_tokens = simple_tokenize(f"[answer] {short_a}")
    a_ids = vocab.encode(a_tokens)
    # The tail of input_ids should match answer ids
    assert encoded["input_ids"][-len(a_ids):] == a_ids


def test_bilstm_data_collator(sample_records):
    vocab = Vocabulary.build_from_records(sample_records, min_freq=1)
    dataset = RAGTruthBiLSTMDataset(sample_records, vocab, max_length=64)
    collator = BiLSTMDataCollator(pad_id=vocab.pad_id)

    batch = collator([dataset[0], dataset[1]])
    assert "input_ids" in batch
    assert "attention_mask" in batch
    assert "lengths" in batch
    assert "labels" in batch
    assert batch["input_ids"].shape[0] == 2
    assert batch["attention_mask"].shape == batch["input_ids"].shape
    assert batch["lengths"].shape == (2,)
    assert batch["labels"].shape == (2,)


def test_attention_mechanism():
    batch_size = 2
    seq_len = 5
    hidden_dim = 16
    rnn_outputs = torch.randn(batch_size, seq_len, hidden_dim)
    mask = torch.tensor([[1, 1, 1, 0, 0], [1, 1, 1, 1, 1]], dtype=torch.long)

    attn = BahdanauAttention(hidden_dim=hidden_dim, attention_dim=8)
    context_vector, weights = attn(rnn_outputs, mask)

    assert context_vector.shape == (batch_size, hidden_dim)
    assert weights.shape == (batch_size, seq_len)
    # Padded positions in sample 0 should have close to 0 weight
    assert weights[0, 3].item() < 1e-4
    assert weights[0, 4].item() < 1e-4
    # Weights should sum to 1
    assert torch.allclose(weights.sum(dim=-1), torch.ones(batch_size), atol=1e-5)


def test_bilstm_classifier_predict_and_proba(sample_records):
    vocab = Vocabulary.build_from_records(sample_records, min_freq=1)
    classifier = BiLSTMClassifier(
        vocab=vocab,
        embed_dim=16,
        hidden_dim=16,
        num_layers=1,
        dropout=0.1,
        use_attention=True,
        attention_dim=16,
        max_length=32,
    )

    probs = classifier.predict_proba(sample_records, batch_size=2, device="cpu")
    assert isinstance(probs, np.ndarray)
    assert probs.shape == (len(sample_records), 2)
    assert np.allclose(probs.sum(axis=1), 1.0, atol=1e-5)

    preds = classifier.predict(sample_records, batch_size=2, device="cpu")
    assert isinstance(preds, np.ndarray)
    assert preds.shape == (len(sample_records),)
    assert set(preds).issubset({0, 1})



def test_bilstm_save_and_load(sample_records, tmp_path):
    vocab = Vocabulary.build_from_records(sample_records, min_freq=1)
    classifier = BiLSTMClassifier(
        vocab=vocab,
        embed_dim=16,
        hidden_dim=16,
        num_layers=1,
        dropout=0.1,
        use_attention=True,
        attention_dim=16,
        max_length=32,
    )
    save_dir = tmp_path / "bilstm_ckpt"
    classifier.save(save_dir)

    assert (save_dir / "vocab.json").exists()
    assert (save_dir / "model_config.json").exists()
    assert (save_dir / "bilstm_weights.pt").exists()

    loaded = BiLSTMClassifier.load(save_dir, device="cpu")
    assert loaded.embed_dim == 16
    assert loaded.hidden_dim == 16
    assert loaded.use_attention is True
    assert len(loaded.vocab) == len(vocab)

    preds_orig = classifier.predict(sample_records, device="cpu")
    preds_loaded = loaded.predict(sample_records, device="cpu")
    assert np.array_equal(preds_orig, preds_loaded)


def test_training_smoke_test(sample_records, tmp_path, monkeypatch):
    """Smoke test to verify the complete training & evaluation loop end-to-end on synthetic sample data."""
    test_jsonl = tmp_path / "dummy_data.jsonl"
    import json
    with test_jsonl.open("w", encoding="utf-8") as f:
        for r in sample_records:
            f.write(json.dumps(r.__dict__) + "\n")

    cfg = {
        "project": {"name": "test-bilstm", "seed": 42},
        "data": {
            "processed_file": str(test_jsonl),
            "output_dir": str(tmp_path / "test_out"),
        },
        "bilstm_model": {
            "vocab_size": 100,
            "min_freq": 1,
            "embed_dim": 16,
            "hidden_dim": 16,
            "num_layers": 1,
            "dropout": 0.0,
            "use_attention": True,
            "attention_dim": 8,
            "max_length": 32,
            "truncation_strategy": "context_tail",
        },
        "bilstm_training": {
            "batch_size": 2,
            "learning_rate": 0.01,
            "weight_decay": 0.0,
            "epochs": 1,
            "grad_clip": 1.0,
            "use_class_weight": False,
        },
    }

    metrics = train_bilstm(cfg, device_override="cpu")
    assert "accuracy" in metrics
    assert "f1" in metrics
    assert (tmp_path / "test_out" / "metrics.json").exists()
    assert (tmp_path / "test_out" / "vocab.json").exists()


def test_vocab_section_markers(sample_records):
    """Regression test ensuring section delimiters are preserved in vocabulary."""
    vocab = Vocabulary.build_from_records(sample_records, min_freq=1)
    for marker in ["[question]", "[context]", "[answer]"]:
        assert marker in vocab.token2id
        assert vocab.token2id[marker] != vocab.unk_id

