import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pytest
import torch
import yaml

from hallucination_detector.data.bilstm_dataset import Vocabulary, encode_qca_tokens
from hallucination_detector.data.schema import RAGTruthRecord
from hallucination_detector.inference.model_registry import MODEL_CATALOG
from hallucination_detector.modeling.bilstm_classifier import BiLSTMClassifier
from hallucination_detector.modeling.optimized_bilstm import LABEL_MAPPING, OptimizedBiLSTMAdapter
from scripts.evaluate_validation_thresholds import read_validation_split


MODEL_CONFIG = {
    "embed_dim": 8, "hidden_dim": 8, "num_layers": 1, "dropout": 0.1,
    "use_attention": True, "attention_dim": 4, "max_length": 512,
    "truncation_strategy": "context_tail",
}


def write_bundle(path: Path) -> None:
    vocab = Vocabulary()
    for token in ("[question]", "[context]", "[answer]", "qword", "cword", "aword"):
        vocab.add_token(token)
    classifier = BiLSTMClassifier(vocab=vocab, **MODEL_CONFIG)
    classifier.save(path)
    training_config = {"bilstm_model": {**MODEL_CONFIG, "vocab_size": 50000, "spatial_dropout": 0.2}}
    (path / "config.yaml").write_text(yaml.safe_dump(training_config), encoding="utf-8")
    (path / "metrics.json").write_text(json.dumps({"classification_report": {
        "supported": {}, "hallucinated": {}}}), encoding="utf-8")


def test_optimized_bundle_strictly_loads_and_uses_its_own_model_config(tmp_path):
    write_bundle(tmp_path)
    adapter = OptimizedBiLSTMAdapter.from_artifacts(tmp_path, revision="rev", repo_id="repo")
    assert adapter.classifier.embed_dim == 8
    assert adapter.classifier.hidden_dim == 8
    assert adapter.artifact_metadata["input_order"] == ["question", "context", "answer"]
    assert adapter.artifact_metadata["label_mapping"] == {"0": "SUPPORTED", "1": "HALLUCINATED"}
    assert LABEL_MAPPING == {0: "SUPPORTED", 1: "HALLUCINATED"}
    assert MODEL_CATALOG["bilstm-optimized"].default_revision == "fe0c308b0ec98829ff2408d4b8073e664e9263a9"


def test_optimized_preprocessing_uses_q_context_answer_marker_order_and_context_tail():
    vocab = Vocabulary()
    # The existing tokenizer splits bracketed markers into punctuation and
    # words. This verifies its actual training-time behavior without changing it.
    for token in ("[", "]", "question", "context", "answer", "qword", "cstart", "cend", "aword"):
        vocab.add_token(token)
    encoded = encode_qca_tokens("qword", "cstart " + "filler " * 100 + "cend", "aword", vocab, max_length=12,
                                truncation_strategy="context_tail")
    tokens = [vocab.id2token[index] for index in encoded["input_ids"]]
    marker_positions = [tokens.index(token) for token in ("question", "context", "answer")]
    assert marker_positions == sorted(marker_positions)
    assert len(tokens) <= 12
    assert "qword" in tokens and "aword" in tokens
    assert "cstart" in tokens and "cend" not in tokens


def test_optimized_adapter_checks_config_preprocessing_and_safe_state_dict(tmp_path, monkeypatch):
    write_bundle(tmp_path)
    original_load = torch.load
    seen = {}

    def safe_load(*args, **kwargs):
        seen.update(kwargs)
        return original_load(*args, **kwargs)

    monkeypatch.setattr(torch, "load", safe_load)
    adapter = OptimizedBiLSTMAdapter.from_artifacts(tmp_path, revision="rev", repo_id="repo")
    assert seen.get("weights_only") is True
    records = [RAGTruthRecord("1", "s1", "qword", "cword", "aword", False),
               RAGTruthRecord("2", "s2", "qword", "cword", "aword", True)]
    batch = adapter.predict_proba(records, batch_size=2, device="cpu")
    singles = np.vstack([adapter.predict_proba([record], batch_size=1, device="cpu") for record in records])
    assert batch.shape == (2, 2)
    assert np.isfinite(batch).all()
    assert np.allclose(batch.sum(axis=1), 1.0, atol=1e-6)
    assert np.allclose(batch, singles, atol=1e-6)


def test_optimized_adapter_rejects_config_and_model_config_mismatch(tmp_path):
    write_bundle(tmp_path)
    config_path = tmp_path / "config.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config["bilstm_model"]["embed_dim"] = 99
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    with pytest.raises(ValueError, match="disagree"):
        OptimizedBiLSTMAdapter.from_artifacts(tmp_path, revision="rev", repo_id="repo")


def test_validation_reader_does_not_open_test_member(tmp_path):
    archive = tmp_path / "splits.zip"
    validation = {"id": "v", "source_id": "s", "question": "q", "context": "c", "answer": "a", "hallucinated": False}
    source = {"source_id": "s", "task_type": "QA"}
    with ZipFile(archive, "w") as output:
        output.writestr("data/processed/validation.jsonl", json.dumps(validation) + "\n")
        output.writestr("data/processed/test.jsonl", "intentionally invalid test data that must not be read")
        output.writestr("data/raw/source_info.jsonl", json.dumps(source) + "\n")
    records, tasks = read_validation_split(archive)
    assert [record.id for record in records] == ["v"]
    assert tasks == ["QA"]
