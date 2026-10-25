import json
from types import SimpleNamespace

import pytest

import hallucination_detector.inference.model_registry as registry_module
from hallucination_detector.data.sentence_selector import SentenceSelector
from hallucination_detector.data.qca_artifacts import (
    PROVENANCE_FILENAME,
    QCA_ARTIFACT_TYPE,
    QCA_PROVENANCE_SCHEMA,
    artifact_hashes,
)


def write_manifest(checkpoint):
    hashes = artifact_hashes(checkpoint)
    (checkpoint / PROVENANCE_FILENAME).write_text(
        json.dumps({
            "schema_version": QCA_PROVENANCE_SCHEMA,
            "artifact_type": QCA_ARTIFACT_TYPE,
            "run_signature": "test-run",
            "training": {"seed": 42, "dataset": {"train_sha256": "a" * 64, "train_records": 2}},
            "selector": {
                "sha256": hashes["sentence_selector.json"],
                "train_split_sha256": "a" * 64,
                "fit_record_count": 2,
            },
            "artifact_hashes": hashes,
        }),
        encoding="utf-8",
    )


def make_bundle(checkpoint, selector_payload=None):
    checkpoint.mkdir(parents=True, exist_ok=True)
    if selector_payload is None:
        SentenceSelector(idf={"warsaw": 2.5}, default_idf=0.8).save(checkpoint / "sentence_selector.json")
    else:
        (checkpoint / "sentence_selector.json").write_text(selector_payload, encoding="utf-8")
    (checkpoint / "model.safetensors").write_bytes(b"test-weights")
    (checkpoint / "config.json").write_text("{}", encoding="utf-8")
    (checkpoint / "tokenizer.json").write_text("{}", encoding="utf-8")
    (checkpoint / "tokenizer_config.json").write_text("{}", encoding="utf-8")
    write_manifest(checkpoint)


class StubLongformer:
    def to(self, device):
        self.device = device
        return self

    def eval(self):
        return self


def stub_qca_loader(monkeypatch, checkpoint, model_loaded):
    monkeypatch.setattr(
        registry_module,
        "snapshot_download",
        lambda **kwargs: str(checkpoint),
    )
    monkeypatch.setattr(
        registry_module,
        "AutoTokenizer",
        SimpleNamespace(from_pretrained=lambda path: object()),
    )

    def from_pretrained(path):
        model_loaded.append(path)
        return StubLongformer()

    monkeypatch.setattr(
        registry_module,
        "LongformerForSequenceClassification",
        SimpleNamespace(from_pretrained=from_pretrained),
    )


def test_qca_loader_loads_the_saved_selector_artifact(tmp_path, monkeypatch):
    checkpoint = tmp_path / "checkpoint"
    make_bundle(checkpoint)
    expected_idf = {"warsaw": 2.5}
    # Rewrite the fixture using the exact expected fitted selector and refresh its manifest.
    SentenceSelector(idf=expected_idf, default_idf=0.8).save(checkpoint / "sentence_selector.json")
    write_manifest(checkpoint)
    model_loaded = []
    stub_qca_loader(monkeypatch, checkpoint, model_loaded)

    registry = registry_module.ModelRegistry(prefer_remote=True)
    loaded = registry.load_model("longformer-qca", device="cpu")

    assert loaded["selector_idf_loaded"] is True
    assert loaded["selector"].idf == expected_idf
    assert loaded["selector"].default_idf == 0.8
    assert len(model_loaded) == 1


def test_qca_loader_uses_training_split_config_selector_when_checkpoint_is_legacy(tmp_path, monkeypatch):
    checkpoint = tmp_path / "published_snapshot"
    checkpoint.mkdir()
    (checkpoint / "model.safetensors").write_bytes(b"published-qca-weights")
    selector_path = tmp_path / "qca_selector_idf.json"
    selector_path.write_text(json.dumps({
        "default_idf": 1.0,
        "idf": {"warsaw": 2.5},
        "meta": {"fitted_on": "split=train", "n_records": 13578,
                 "source": "data/processed/ragtruth.jsonl"},
    }), encoding="utf-8")
    model_loaded = []
    stub_qca_loader(monkeypatch, checkpoint, model_loaded)

    registry = registry_module.ModelRegistry(
        prefer_remote=True,
        artifact_paths={"longformer-qca": checkpoint, "longformer-qca-selector": selector_path},
    )
    revision = "4b145abf8cd9eb5eed52cabc344d17d524d622f9"
    loaded = registry.load_model("longformer-qca", device="cpu", revision=revision)

    assert loaded["selector"].idf == {"warsaw": 2.5}
    assert loaded["selector_idf_loaded"] is True
    assert loaded["selector_fitted"] is True
    assert loaded["selector_provenance_status"] == "UNVERIFIED_EXTERNAL_SELECTOR"
    assert loaded["selector_source"] == str(selector_path)
    assert loaded["checkpoint_revision"] == revision
    assert loaded["provenance"] is None
    assert len(model_loaded) == 1


def test_qca_loader_fails_when_external_config_selector_is_missing_or_not_train_fitted(tmp_path, monkeypatch):
    checkpoint = tmp_path / "published_snapshot"
    checkpoint.mkdir()
    (checkpoint / "model.safetensors").write_bytes(b"published-qca-weights")
    missing_path = tmp_path / "missing.json"
    model_loaded = []
    stub_qca_loader(monkeypatch, checkpoint, model_loaded)
    registry = registry_module.ModelRegistry(
        prefer_remote=True,
        artifact_paths={"longformer-qca": checkpoint, "longformer-qca-selector": missing_path},
    )
    with pytest.raises(FileNotFoundError, match="external fitted selector is missing"):
        registry.load_model("longformer-qca", device="cpu")
    assert model_loaded == []

    bad_path = tmp_path / "not_train_fitted.json"
    bad_path.write_text(json.dumps({"idf": {"warsaw": 2.5}, "default_idf": 1.0,
                                    "meta": {"fitted_on": "split=validation", "n_records": 2}}),
                        encoding="utf-8")
    registry = registry_module.ModelRegistry(
        prefer_remote=True,
        artifact_paths={"longformer-qca": checkpoint, "longformer-qca-selector": bad_path},
    )
    with pytest.raises(ValueError, match="fitted_on='split=train'"):
        registry.load_model("longformer-qca", device="cpu")
    assert model_loaded == []


def test_qca_loader_fails_actionably_when_selector_is_missing(tmp_path, monkeypatch):
    checkpoint = tmp_path / "checkpoint"
    make_bundle(checkpoint)
    (checkpoint / "sentence_selector.json").unlink()
    model_loaded = []
    stub_qca_loader(monkeypatch, checkpoint, model_loaded)

    registry = registry_module.ModelRegistry(prefer_remote=True)
    with pytest.raises(FileNotFoundError, match="sentence_selector.json") as error:
        registry.load_model("longformer-qca", device="cpu")

    assert "matching provenance" in str(error.value)
    assert model_loaded == []  # fail before loading the large checkpoint weights


def test_qca_loader_fails_actionably_when_selector_is_incompatible(tmp_path, monkeypatch):
    checkpoint = tmp_path / "checkpoint"
    make_bundle(checkpoint, json.dumps({"idf": {"warsaw": -1.0}, "default_idf": 1.0}))
    write_manifest(checkpoint)
    model_loaded = []
    stub_qca_loader(monkeypatch, checkpoint, model_loaded)

    registry = registry_module.ModelRegistry(prefer_remote=True)
    with pytest.raises(ValueError, match="malformed or mismatched") as error:
        registry.load_model("longformer-qca", device="cpu")

    assert "Invalid IDF score" in str(error.value)
    assert model_loaded == []


def test_qca_loader_rejects_weight_selector_mismatch_before_loading_weights(tmp_path, monkeypatch):
    checkpoint = tmp_path / "checkpoint"
    make_bundle(checkpoint)
    (checkpoint / "model.safetensors").write_bytes(b"weights-from-another-run")
    model_loaded = []
    stub_qca_loader(monkeypatch, checkpoint, model_loaded)

    registry = registry_module.ModelRegistry(prefer_remote=True)
    with pytest.raises(ValueError, match="hash mismatch.*different training runs"):
        registry.load_model("longformer-qca", device="cpu")
    assert model_loaded == []


def test_qca_loader_fails_closed_when_provenance_is_missing_or_malformed(tmp_path, monkeypatch):
    checkpoint = tmp_path / "checkpoint"
    make_bundle(checkpoint)
    (checkpoint / PROVENANCE_FILENAME).unlink()
    model_loaded = []
    stub_qca_loader(monkeypatch, checkpoint, model_loaded)
    registry = registry_module.ModelRegistry(prefer_remote=True)
    with pytest.raises(FileNotFoundError, match="provenance.json"):
        registry.load_model("longformer-qca", device="cpu")
    assert model_loaded == []

    (checkpoint / PROVENANCE_FILENAME).write_text("{bad json", encoding="utf-8")
    registry = registry_module.ModelRegistry(prefer_remote=True)
    with pytest.raises(ValueError, match="malformed or mismatched"):
        registry.load_model("longformer-qca", device="cpu")
    assert model_loaded == []


def test_qca_loader_rejects_selector_that_would_trigger_a_default_idf(tmp_path, monkeypatch):
    checkpoint = tmp_path / "checkpoint"
    make_bundle(checkpoint, json.dumps({"idf": {"warsaw": 2.0}}))
    write_manifest(checkpoint)
    model_loaded = []
    stub_qca_loader(monkeypatch, checkpoint, model_loaded)

    registry = registry_module.ModelRegistry(prefer_remote=True)
    with pytest.raises(ValueError, match="explicit 'default_idf'"):
        registry.load_model("longformer-qca", device="cpu")
    assert model_loaded == []
