from __future__ import annotations

import json
from pathlib import Path
import zipfile

import numpy as np
import pytest
import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import WhitespaceSplit
from transformers import LongformerConfig, LongformerForSequenceClassification, PreTrainedTokenizerFast

from hallucination_detector.data.qca_artifacts import load_and_verify_qca_provenance, sha256_file
from hallucination_detector.data.schema import RAGTruthRecord
from hallucination_detector.inference.model_registry import ModelRegistry
from hallucination_detector.modeling.longformer_classifier import LongformerClassifier
from hallucination_detector.training.longformer_qca_pipeline import train_qca
from hallucination_detector.training.longformer_smart_train import predict_smart_proba


def record(record_id: str, split: str, hallucinated: bool) -> dict:
    return {
        "id": record_id,
        "source_id": f"source-{record_id}",
        "question": "Where was Marie Curie born?",
        "context": "Marie Curie was born in Warsaw. She later studied in Paris.",
        "answer": "Marie Curie was born in Paris." if hallucinated else "Marie Curie was born in Warsaw.",
        "hallucinated": hallucinated,
        "model": "synthetic-test",
        "temperature": 0.0,
        "split": split,
        "quality": None,
        "annotations": [],
    }


def write_archive(path: Path) -> None:
    train = [record("tr-0", "train", False), record("tr-1", "train", True)]
    validation = [record("va-0", "validation", False), record("va-1", "validation", True)]
    validation[0]["context"] += " ValidationOnlyUniqueTerm."
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("data/processed/train.jsonl", "".join(json.dumps(row) + "\n" for row in train))
        archive.writestr("data/processed/validation.jsonl", "".join(json.dumps(row) + "\n" for row in validation))
        # Invalid JSON proves the training/validation path never parses the held-out test member.
        archive.writestr("data/processed/test.jsonl", "THIS IS NOT JSON AND MUST NOT BE OPENED")


def make_tiny_local_longformer(directory: Path) -> None:
    directory.mkdir(parents=True)
    vocabulary = {
        "[PAD]": 0, "[UNK]": 1, "<s>": 2, "</s>": 3, "<mask>": 4,
        "where": 5, "was": 6, "marie": 7, "curie": 8, "born": 9,
        "warsaw": 10, "she": 11, "later": 12, "studied": 13,
        "in": 14, "paris": 15, ".": 16, "?": 17,
    }
    tokenizer_object = Tokenizer(WordLevel(vocabulary, unk_token="[UNK]"))
    tokenizer_object.pre_tokenizer = WhitespaceSplit()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer_object,
        unk_token="[UNK]", pad_token="[PAD]", cls_token="<s>",
        sep_token="</s>", mask_token="<mask>", bos_token="<s>", eos_token="</s>",
    )
    tokenizer.save_pretrained(directory)
    config = LongformerConfig(
        vocab_size=len(vocabulary), hidden_size=16, num_hidden_layers=1,
        num_attention_heads=2, intermediate_size=32, attention_window=[4],
        max_position_embeddings=514, pad_token_id=0, bos_token_id=2, eos_token_id=3,
        num_labels=2, id2label={0: "SUPPORTED", 1: "HALLUCINATED"},
        label2id={"SUPPORTED": 0, "HALLUCINATED": 1},
    )
    LongformerForSequenceClassification(config).save_pretrained(directory, safe_serialization=True)


def write_config(path: Path, base_model: Path) -> None:
    path.write_text(
        f"""project:
  name: qca-synthetic-smoke
  seed: 42
model:
  name: {base_model.as_posix()}
  num_labels: 2
  max_length: 512
  truncation_strategy: smart_selector
  global_attention_mode: cls
training:
  batch_size: 1
  gradient_accumulation_steps: 1
  learning_rate: 0.0001
  epochs: 1
  weight_decay: 0.0
  early_stopping_patience: 1
  bf16: false
  gradient_checkpointing: false
experiment:
  input_mode: qca
nlp_preprocessing:
  selector_type: idf_relevance_contradiction
  alpha: 0.4
  lead_bias: true
""",
        encoding="utf-8",
    )


def test_training_smoke_is_train_validation_only_and_roundtrips_bundle(tmp_path):
    archive = tmp_path / "tiny.zip"
    base_model = tmp_path / "tiny-base"
    output = tmp_path / "isolated-run"
    config = tmp_path / "tiny.yaml"
    write_archive(archive)
    make_tiny_local_longformer(base_model)
    write_config(config, base_model)

    smoke_device = "cuda:0" if torch.cuda.is_available() else "cpu"
    result = train_qca(archive, output, config, device_name=smoke_device)

    assert result["test_split_opened"] is False
    assert result["train_records"] == 2
    assert result["validation_records"] == 2
    bundle = Path(result["best_model_dir"])
    manifest = load_and_verify_qca_provenance(bundle)
    assert manifest["selector"]["fit_record_count"] == 2
    assert manifest["selector"]["train_split_sha256"] == manifest["training"]["dataset"]["train_sha256"]
    assert sha256_file(bundle / "sentence_selector.json") == manifest["artifact_hashes"]["sentence_selector.json"]
    assert "validationonlyuniqueterm" not in loaded_selector_tokens(bundle)

    with zipfile.ZipFile(archive) as zipped:
        validation = [
            RAGTruthRecord(**json.loads(line))
            for line in zipped.read("data/processed/validation.jsonl").decode().splitlines()
        ]
    registry = ModelRegistry(artifact_paths={"longformer-qca": bundle})
    loaded = registry.load_model("longformer-qca", device=smoke_device)
    assert loaded["provenance"]["run_signature"] == result["run_signature"]
    assert loaded["selector"].idf

    before_registry = predict_smart_proba(
        model=LongformerClassifier.from_pretrained(bundle, max_length=512),
        records=validation,
        batch_size=1,
        device=torch.device(smoke_device),
        bf16=False,
        input_mode="qca",
        truncation_strategy="smart_selector",
        global_attention_mode="cls",
        selector=loaded["selector"],
    )
    after_registry_rows = []
    for row in validation:
        prediction = registry.predict("longformer-qca", row.question, row.context, row.answer, device=smoke_device)
        after_registry_rows.append([prediction.supported_probability, prediction.hallucination_probability])
    after_registry = np.asarray(after_registry_rows)
    assert np.all(np.isfinite(before_registry))
    assert np.allclose(before_registry, after_registry, atol=1e-5)


def loaded_selector_tokens(bundle: Path) -> set[str]:
    from hallucination_detector.data.sentence_selector import SentenceSelector

    return set(SentenceSelector.load(bundle / "sentence_selector.json").idf)


def test_existing_output_is_never_overwritten_without_resume(tmp_path):
    output = tmp_path / "already-there"
    output.mkdir()
    (output / "user-file.txt").write_text("preserve", encoding="utf-8")
    with pytest.raises(FileExistsError, match="new path or --resume"):
        train_qca("unused.zip", output, "unused.yaml", device_name="cpu")
    assert (output / "user-file.txt").read_text(encoding="utf-8") == "preserve"


def test_split_mislabeled_train_member_is_rejected_without_test_access(tmp_path):
    archive = tmp_path / "leak.zip"
    with zipfile.ZipFile(archive, "w") as zipped:
        zipped.writestr("data/processed/train.jsonl", json.dumps(record("leak", "test", True)))
        zipped.writestr("data/processed/validation.jsonl", json.dumps(record("val", "validation", False)))
        zipped.writestr("data/processed/test.jsonl", "also invalid JSON")
    from hallucination_detector.training.longformer_smart_train import load_train_validation_archive

    with pytest.raises(ValueError, match="Split leakage"):
        load_train_validation_archive(archive)


def test_epoch_checkpoint_resumes_same_isolated_run(tmp_path, monkeypatch):
    from hallucination_detector.training import longformer_qca_pipeline as pipeline

    archive = tmp_path / "tiny.zip"
    base_model = tmp_path / "tiny-base"
    output = tmp_path / "resume-run"
    config = tmp_path / "tiny.yaml"
    write_archive(archive)
    make_tiny_local_longformer(base_model)
    write_config(config, base_model)
    config.write_text(config.read_text(encoding="utf-8").replace("  epochs: 1", "  epochs: 2"), encoding="utf-8")

    real_evaluate = pipeline.evaluate_smart_loss
    calls = {"count": 0}

    def interrupt_after_first_completed_epoch(*args, **kwargs):
        calls["count"] += 1
        if calls["count"] == 2:
            raise RuntimeError("synthetic interruption after epoch one")
        return real_evaluate(*args, **kwargs)

    monkeypatch.setattr(pipeline, "evaluate_smart_loss", interrupt_after_first_completed_epoch)
    with pytest.raises(RuntimeError, match="synthetic interruption"):
        train_qca(archive, output, config, device_name="cpu")
    assert (output / "resume_state.pt").is_file()

    monkeypatch.setattr(pipeline, "evaluate_smart_loss", real_evaluate)
    result = train_qca(archive, output, config, resume=True, device_name="cpu")
    assert result["test_split_opened"] is False
    assert result["best_epoch"] in (1, 2)
    assert not (output / "resume_state.pt").exists()
