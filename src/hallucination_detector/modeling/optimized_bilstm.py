"""Strict adapter for the separately published optimized BiLSTM artifact."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Sequence

import torch
import yaml

from hallucination_detector.data.bilstm_dataset import Vocabulary
from hallucination_detector.data.schema import RAGTruthRecord
from hallucination_detector.modeling.bilstm_classifier import BiLSTMClassifier


MODEL_CONFIG_FIELDS = (
    "embed_dim", "hidden_dim", "num_layers", "dropout", "use_attention",
    "attention_dim", "max_length", "truncation_strategy",
)
LABEL_MAPPING = {0: "SUPPORTED", 1: "HALLUCINATED"}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class OptimizedBiLSTMAdapter:
    """Own model instance with its Hub vocabulary, dimensions, and preprocessing."""

    def __init__(self, classifier: BiLSTMClassifier, artifact_metadata: dict, device: str | torch.device):
        self.classifier = classifier
        self.artifact_metadata = artifact_metadata
        self.device = torch.device(device)

    @classmethod
    def from_artifacts(cls, artifact_dir: str | Path, *, device: str | torch.device = "cpu",
                       revision: str, repo_id: str) -> "OptimizedBiLSTMAdapter":
        root = Path(artifact_dir)
        required = ("config.yaml", "model_config.json", "vocab.json", "bilstm_weights.pt", "metrics.json")
        missing = [name for name in required if not (root / name).is_file()]
        if missing:
            raise FileNotFoundError(
                f"Optimized BiLSTM bundle {root} is missing {missing}; download the complete pinned Hub artifact set."
            )

        training_config = yaml.safe_load((root / "config.yaml").read_text(encoding="utf-8"))
        model_config = json.loads((root / "model_config.json").read_text(encoding="utf-8"))
        metrics = json.loads((root / "metrics.json").read_text(encoding="utf-8"))
        configured = training_config.get("bilstm_model") if isinstance(training_config, dict) else None
        if not isinstance(configured, dict):
            raise ValueError("Optimized BiLSTM config.yaml lacks a bilstm_model object.")
        if not isinstance(model_config, dict) or not all(field in model_config for field in MODEL_CONFIG_FIELDS):
            raise ValueError(f"model_config.json must define {list(MODEL_CONFIG_FIELDS)}.")
        mismatches = {
            field: (configured.get(field), model_config.get(field))
            for field in MODEL_CONFIG_FIELDS
            if configured.get(field) != model_config.get(field)
        }
        if mismatches:
            raise ValueError(f"config.yaml and model_config.json disagree: {mismatches}.")
        if model_config["max_length"] != 512 or model_config["truncation_strategy"] != "context_tail":
            raise ValueError("Unsupported optimized BiLSTM preprocessing; expected 512-token context_tail encoding.")
        report = metrics.get("classification_report", {})
        if not {"supported", "hallucinated"}.issubset(report):
            raise ValueError("metrics.json does not document the supported/hallucinated class names.")

        vocab = Vocabulary.load(root / "vocab.json")
        markers = ["[question]", "[context]", "[answer]"]
        try:
            marker_ids = [vocab.token2id[token] for token in markers]
        except KeyError as exc:
            raise ValueError(f"vocab.json is missing the required Q+C+A marker {exc.args[0]!r}.") from exc
        if marker_ids != [2, 3, 4]:
            raise ValueError(f"Unexpected training marker IDs/order: {dict(zip(markers, marker_ids))}.")
        if (vocab.pad_id, vocab.unk_id) != (0, 1):
            raise ValueError("Optimized BiLSTM vocabulary must use pad_id=0 and unk_id=1.")

        classifier = BiLSTMClassifier(
            vocab=vocab,
            embed_dim=int(model_config["embed_dim"]),
            hidden_dim=int(model_config["hidden_dim"]),
            num_layers=int(model_config["num_layers"]),
            dropout=float(model_config["dropout"]),
            use_attention=bool(model_config["use_attention"]),
            attention_dim=int(model_config["attention_dim"]),
            max_length=int(model_config["max_length"]),
            truncation_strategy=str(model_config["truncation_strategy"]),
        )
        if classifier.net is None:
            raise RuntimeError("Optimized BiLSTM architecture did not initialize.")
        try:
            state_dict = torch.load(root / "bilstm_weights.pt", map_location="cpu", weights_only=True)
        except TypeError as exc:
            raise RuntimeError("This PyTorch version lacks safe weights_only checkpoint loading; upgrade PyTorch.") from exc
        if not isinstance(state_dict, dict) or not all(isinstance(value, torch.Tensor) for value in state_dict.values()):
            raise ValueError("bilstm_weights.pt is not a tensor-only state dictionary.")
        classifier.net.load_state_dict(state_dict, strict=True)
        target = torch.device(device)
        classifier.net.to(target).eval()

        hashes = {name: sha256_file(root / name) for name in required}
        artifact_metadata = {
            "repo_id": repo_id,
            "revision": revision,
            "artifact_hashes": hashes,
            "input_mode": "Q+C+A",
            "input_order": ["question", "context", "answer"],
            "tokenization": "lowercase Unicode word/punctuation regex from the project vocabulary pipeline",
            "section_marker_tokenization": "bracketed markers are split into punctuation and words by the existing tokenizer; dedicated [question]/[context]/[answer] vocabulary IDs are not emitted as single tokens",
            "max_length": int(model_config["max_length"]),
            "truncation_strategy": str(model_config["truncation_strategy"]),
            "truncation_behavior": "retains the context prefix and drops excess trailing context tokens under the existing context_tail implementation",
            "label_mapping": {str(key): value for key, value in LABEL_MAPPING.items()},
            "provenance_status": "PARTIAL: public artifact bundle; training code/split hashes are not published",
            "config_spatial_dropout": configured.get("spatial_dropout"),
            "note": "State dict strictly matches the project BiLSTM+Bahdanau implementation; inference uses the published model_config and vocabulary.",
        }
        return cls(classifier, artifact_metadata, target)

    def predict_proba(self, records: Sequence[RAGTruthRecord], batch_size: int = 64,
                      device: str | torch.device | None = None):
        return self.classifier.predict_proba(records, batch_size=batch_size, device=device or self.device)
