from __future__ import annotations

from dataclasses import dataclass, field
import json
import os
from pathlib import Path
import pickle
from typing import Any, Literal, Sequence

from huggingface_hub import hf_hub_download, snapshot_download
import numpy as np
import torch
import torch.nn.functional as F
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    LongformerForSequenceClassification,
)

from hallucination_detector.data.schema import RAGTruthRecord
from hallucination_detector.data.sentence_selector import SentenceSelector
from hallucination_detector.data.qca_artifacts import load_and_verify_qca_provenance
from hallucination_detector.data.torch_dataset import (
    RAGTruthTransformerDataset,
    LongformerDataCollator,
    encode_pair_with_answer_preservation,
)
from hallucination_detector.data.longformer_smart_dataset import (
    RAGTruthSmartLongformerDataset,
)
from hallucination_detector.modeling.bilstm_classifier import BiLSTMClassifier
from hallucination_detector.modeling.optimized_bilstm import OptimizedBiLSTMAdapter
from hallucination_detector.modeling.deberta_encoding import encode_deberta_input
from hallucination_detector.modeling.tfidf_logistic import (
    TfidfLogisticClassifier,
    build_input_text,
    normalize_text,
)


@dataclass
class PredictionResult:
    model_name: str
    model_id: str
    prediction: str  # "SUPPORTED" or "HALLUCINATED"
    is_hallucinated: bool
    hallucination_probability: float | None
    supported_probability: float | None
    raw_score: float | None = None
    is_calibrated: bool = False
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class ModelMetadata:
    key: str
    display_name: str
    repo_id: str
    architecture: str
    input_mode: str  # "C+A", "Q+C+A"
    description: str
    limitations: str
    default_revision: str = "main"
    status: Literal["AVAILABLE", "COMING_SOON", "ERROR"] = "AVAILABLE"
    provenance_status: Literal["PARTIAL", "UNVERIFIED"] = "PARTIAL"


MODEL_CATALOG: dict[str, ModelMetadata] = {
    "longformer-head": ModelMetadata(
        key="longformer-head",
        display_name="Longformer (Head Truncation)",
        repo_id="nishantwhat/robustrag-longformer-head",
        architecture="Longformer (1024 tokens) + Sliding Window (w=512)",
        input_mode="C+A",
        description="Preserves answer tokens when feasible and keeps the end of long contexts.",
        limitations="Evaluates Context + Answer. Does not directly condition on Question tokens.",
    ),
    "longformer-qca": ModelMetadata(
        key="longformer-qca",
        display_name="Longformer (Smart Q+C+A)",
        repo_id="nishantwhat/robustrag-longformer-qca",
        architecture="Longformer (1024 tokens) + Extractive SentenceSelector",
        input_mode="Q+C+A",
        description="Conditions on Question + Context + Answer with contradiction-aware sentence ranking.",
        limitations=(
            "Uses the training-split IDF statistics from configs/qca_selector_idf.json when the "
            "checkpoint has no verified bundle. The published checkpoint lacks a provenance "
            "manifest linking that selector to its weights, so this pairing is unverified."
        ),
        default_revision="4b145abf8cd9eb5eed52cabc344d17d524d622f9",
        provenance_status="UNVERIFIED",
    ),
    "deberta": ModelMetadata(
        key="deberta",
        display_name="DeBERTa-v3-base (512 tokens)",
        repo_id="nishantwhat/robustrag-deberta",
        architecture="DeBERTa-v3-base + Disentangled Attention",
        input_mode="C+A",
        description="512-token baseline cross-encoder that preserves the candidate answer when it fits.",
        limitations="Context is shortened first within the 512-token limit; an answer longer than the available budget is also shortened and flagged.",
    ),
    "bilstm-attention": ModelMetadata(
        key="bilstm-attention",
        display_name="BiLSTM + Bahdanau Attention",
        repo_id="nishantwhat/robustrag-bilstm-attention",
        architecture="Bidirectional LSTM (2-layer, 128-dim) + Additive Self-Attention",
        input_mode="Q+C+A",
        description="Neural recurrent baseline with additive attention over sequence tokens.",
        limitations="Uses a fixed 50k vocabulary and lacks pretrained contextual representations.",
    ),
    "bilstm-optimized": ModelMetadata(
        key="bilstm-optimized",
        display_name="Optimized BiLSTM + Attention (Q+C+A)",
        repo_id="nishantwhat/robustrag-optimized-bilstm",
        architecture="2-layer bidirectional LSTM (256 hidden/embedding) + Bahdanau attention (128)",
        input_mode="Q+C+A",
        description="Separate published BiLSTM checkpoint with its own vocabulary and preprocessing bundle.",
        limitations="Artifact provenance is partial; scores remain uncalibrated. Uses 512-token Q+C+A context_tail preprocessing.",
        default_revision="fe0c308b0ec98829ff2408d4b8073e664e9263a9",
        provenance_status="UNVERIFIED",
    ),
    "tfidf-lr": ModelMetadata(
        key="tfidf-lr",
        display_name="TF-IDF + Logistic Regression",
        repo_id="nishantwhat/robustrag-tfidf-lr",
        architecture="Classical Bag-of-Words (437k n-grams) + L2 Logistic Regression",
        input_mode="C+A",
        description="Fast classical lexical baseline. Sublinear TF unigrams and bigrams.",
        limitations="Lexical pattern matching only. Blind to semantic synonyms, negations, or complex reasoning.",
    ),
}

DEFAULT_HALLUCINATION_THRESHOLD = 0.5


def classify_hallucination_score(
    hallucination_probability: float,
    threshold: float = DEFAULT_HALLUCINATION_THRESHOLD,
) -> tuple[bool, str]:
    """Apply RobustRAG's binary decision rule (positive class: HALLUCINATED)."""
    if not np.isfinite(hallucination_probability) or not 0.0 <= hallucination_probability <= 1.0:
        raise ValueError("hallucination_probability must be finite and in [0, 1].")
    if not np.isfinite(threshold) or not 0.0 < threshold <= 1.0:
        raise ValueError("threshold must be finite and in (0, 1].")
    is_hallucinated = hallucination_probability >= threshold
    return is_hallucinated, "HALLUCINATED" if is_hallucinated else "SUPPORTED"


class ModelRegistry:
    """Lazy model loader and inference coordinator for RobustRAG models."""

    def __init__(
        self,
        cache_dir: str | Path | None = None,
        prefer_remote: bool = False,
        artifact_paths: dict[str, str | Path] | None = None,
    ) -> None:
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.prefer_remote = prefer_remote
        self.artifact_paths = {key: Path(value) for key, value in (artifact_paths or {}).items()}
        self._loaded_models: dict[str, Any] = {}

    def get_catalog(self) -> dict[str, ModelMetadata]:
        return MODEL_CATALOG

    def load_model(
        self,
        model_key: str,
        device: str | torch.device | None = None,
        revision: str | None = None,
    ) -> Any:
        """Lazily load a model from local cache or Hugging Face Hub."""
        if model_key not in MODEL_CATALOG:
            raise ValueError(f"Unknown model key '{model_key}'. Available: {list(MODEL_CATALOG.keys())}")

        if model_key in self._loaded_models:
            return self._loaded_models[model_key]

        meta = MODEL_CATALOG[model_key]
        rev = revision or meta.default_revision
        target_device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        if model_key == "tfidf-lr":
            model = self._load_tfidf(meta.repo_id, rev)
        elif model_key == "bilstm-attention":
            model = self._load_bilstm(meta.repo_id, rev, target_device)
        elif model_key == "bilstm-optimized":
            model = self._load_optimized_bilstm(meta.repo_id, rev, target_device)
        elif model_key == "deberta":
            model = self._load_deberta(meta.repo_id, rev, target_device)
        elif model_key == "longformer-head":
            model = self._load_longformer_head(meta.repo_id, rev, target_device)
        elif model_key == "longformer-qca":
            model = self._load_longformer_qca(meta.repo_id, rev, target_device)
        else:
            raise NotImplementedError(f"Loader for {model_key} not implemented.")

        self._loaded_models[model_key] = model
        return model

    def _load_tfidf(self, repo_id: str, revision: str) -> TfidfLogisticClassifier:
        local_path = Path("reports/experiments_backup/baseline_tfidf_logreg/model.pkl")
        if not self.prefer_remote and local_path.exists():
            with local_path.open("rb") as f:
                bundle = pickle.load(f)
            return bundle["model"] if isinstance(bundle, dict) else bundle

        model_file = hf_hub_download(repo_id=repo_id, filename="model.pkl", revision=revision, cache_dir=self.cache_dir)
        with open(model_file, "rb") as f:
            bundle = pickle.load(f)
        return bundle["model"] if isinstance(bundle, dict) else bundle

    def _load_bilstm(self, repo_id: str, revision: str, device: str | torch.device) -> BiLSTMClassifier:
        local_dir = Path("models/bilstm_attention_baseline")
        if not self.prefer_remote and (local_dir / "bilstm_weights.pt").exists() and (local_dir / "vocab.json").exists():
            return BiLSTMClassifier.load(local_dir, device=device)

        weights_file = hf_hub_download(repo_id=repo_id, filename="bilstm_weights.pt", revision=revision, cache_dir=self.cache_dir)
        model_dir = Path(weights_file).parent
        hf_hub_download(repo_id=repo_id, filename="vocab.json", revision=revision, cache_dir=self.cache_dir)
        hf_hub_download(repo_id=repo_id, filename="model_config.json", revision=revision, cache_dir=self.cache_dir)
        return BiLSTMClassifier.load(model_dir, device=device)

    def _load_optimized_bilstm(self, repo_id: str, revision: str, device: str | torch.device) -> OptimizedBiLSTMAdapter:
        local_dir = self.artifact_paths.get("bilstm-optimized")
        if local_dir is None:
            filenames = ("config.yaml", "model_config.json", "vocab.json", "bilstm_weights.pt", "metrics.json")
            paths = [hf_hub_download(repo_id=repo_id, filename=name, revision=revision, cache_dir=self.cache_dir)
                     for name in filenames]
            parent_dirs = {str(Path(path).parent) for path in paths}
            if len(parent_dirs) != 1:
                raise RuntimeError("Hub artifacts resolved to different revisions/directories; refusing a mixed BiLSTM bundle.")
            local_dir = Path(paths[0]).parent
        return OptimizedBiLSTMAdapter.from_artifacts(
            local_dir, device=device, revision=revision, repo_id=repo_id,
        )

    def _load_deberta(self, repo_id: str, revision: str, device: str | torch.device) -> dict[str, Any]:
        local_dir = Path("reports/experiments/baseline_deberta/best_model")
        if not self.prefer_remote and (local_dir / "model.safetensors").exists():
            model_path = str(local_dir)
        else:
            model_path = snapshot_download(repo_id=repo_id, revision=revision, cache_dir=self.cache_dir)

        tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
        model = AutoModelForSequenceClassification.from_pretrained(model_path)
        model.to(torch.device(device))
        model.eval()
        return {"model": model, "tokenizer": tokenizer, "device": device}

    def _load_longformer_head(self, repo_id: str, revision: str, device: str | torch.device) -> dict[str, Any]:
        local_dir = Path("reports/experiments/longformer_1024_head/best_model")
        if not self.prefer_remote and (local_dir / "model.safetensors").exists():
            model_path = str(local_dir)
        else:
            model_path = snapshot_download(repo_id=repo_id, revision=revision, cache_dir=self.cache_dir)

        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = LongformerForSequenceClassification.from_pretrained(model_path)
        model.to(torch.device(device))
        model.eval()
        return {"model": model, "tokenizer": tokenizer, "device": device}

    def _load_longformer_qca(self, repo_id: str, revision: str, device: str | torch.device) -> dict[str, Any]:
        local_dir = self.artifact_paths.get(
            "longformer-qca",
            Path("reports/experiments/longformer_1024_smart_qca/best_model"),
        )
        if "longformer-qca" in self.artifact_paths:
            model_path = str(local_dir)
        elif not self.prefer_remote and local_dir.exists():
            model_path = str(local_dir)
        else:
            model_path = snapshot_download(repo_id=repo_id, revision=revision, cache_dir=self.cache_dir)

        checkpoint_root = Path(model_path)
        selector_path = checkpoint_root / "sentence_selector.json"
        provenance_path = checkpoint_root / "provenance.json"
        provenance = None
        selector_training_metadata: dict[str, Any] = {}
        if provenance_path.exists() or selector_path.exists():
            # If a bundle has started to package either file, require its full hash-verified pair.
            # Never hide an incomplete/mismatched local candidate by silently switching selectors.
            try:
                provenance = load_and_verify_qca_provenance(checkpoint_root)
                selector = SentenceSelector.load(selector_path)
            except FileNotFoundError as exc:
                raise FileNotFoundError(
                    f"Longformer QCA checkpoint {checkpoint_root} is incomplete: {exc} "
                    "Restore a complete bundle with matching provenance or remove the incomplete candidate."
                ) from exc
            except (OSError, ValueError) as exc:
                raise ValueError(
                    f"Longformer QCA artifact bundle {checkpoint_root} is malformed or mismatched: {exc} "
                    "Restore the selector and weights from the same verified training run."
                ) from exc
            selector_status = "VERIFIED_BUNDLE"
            selector_source = str(selector_path)
            selector_training_metadata = provenance.get("training", {})
        else:
            # Match the author's deployed UI: the fitted IDF sidecar lives in configs/ while
            # the fine-tuned model/tokenizer live in the pinned Hugging Face snapshot.
            external_selector_path = self.artifact_paths.get(
                "longformer-qca-selector",
                Path(__file__).resolve().parents[3] / "configs" / "qca_selector_idf.json",
            )
            if not external_selector_path.is_file():
                raise FileNotFoundError(
                    f"Longformer QCA has no packaged selector/provenance and its external fitted selector "
                    f"is missing: {external_selector_path}. Restore configs/qca_selector_idf.json."
                )
            try:
                selector_document = json.loads(external_selector_path.read_text(encoding="utf-8"))
                if not isinstance(selector_document, dict):
                    raise ValueError("External QCA selector JSON must contain a top-level object.")
                selector_training_metadata = selector_document.get("meta", {})
                if (not isinstance(selector_training_metadata, dict)
                        or selector_training_metadata.get("fitted_on") != "split=train"
                        or not isinstance(selector_training_metadata.get("n_records"), int)
                        or selector_training_metadata["n_records"] <= 0):
                    raise ValueError(
                        "External QCA selector must declare meta.fitted_on='split=train' and a positive meta.n_records."
                    )
                selector = SentenceSelector.load(external_selector_path)
            except (OSError, json.JSONDecodeError, ValueError, TypeError) as exc:
                raise ValueError(
                    f"External Longformer QCA selector {external_selector_path} is missing, malformed, "
                    f"or lacks training-split metadata: {exc}"
                ) from exc
            selector_status = "UNVERIFIED_EXTERNAL_SELECTOR"
            selector_source = str(external_selector_path)

        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = LongformerForSequenceClassification.from_pretrained(model_path)
        model.to(torch.device(device))
        model.eval()
        return {
            "model": model,
            "tokenizer": tokenizer,
            "selector": selector,
            "selector_idf_loaded": True,
            "selector_fitted": True,
            "selector_provenance_status": selector_status,
            "selector_source": selector_source,
            "selector_training_metadata": selector_training_metadata,
            "checkpoint_revision": checkpoint_root.name if len(checkpoint_root.name) == 40 else revision,
            "provenance": provenance,
            "device": device,
        }

    def predict(
        self,
        model_key: str,
        question: str,
        context: str | dict[str, Any],
        answer: str,
        device: str | torch.device | None = None,
    ) -> PredictionResult:
        """Run single-instance inference with model-specific preprocessing."""
        meta = MODEL_CATALOG[model_key]
        model_obj = self.load_model(model_key, device=device)
        record = RAGTruthRecord(
            id="query_1",
            source_id="src_1",
            question=question.strip(),
            context=context.strip() if isinstance(context, str) else context,
            answer=answer.strip(),
            hallucinated=False,
        )

        if model_key == "tfidf-lr":
            probs = model_obj.predict_proba([record])
            p_sup, p_hal = float(probs[0, 0]), float(probs[0, 1])

        elif model_key == "bilstm-attention":
            target_dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
            probs = model_obj.predict_proba([record], device=target_dev)
            p_sup, p_hal = float(probs[0, 0]), float(probs[0, 1])

        elif model_key == "bilstm-optimized":
            target_dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
            probs = model_obj.predict_proba([record], batch_size=1, device=target_dev)
            p_sup, p_hal = float(probs[0, 0]), float(probs[0, 1])

        elif model_key == "deberta":
            tok = model_obj["tokenizer"]
            mod = model_obj["model"]
            dev = model_obj["device"]
            encoded = encode_deberta_input(context, answer, tok, max_length=512)
            truncation = encoded.pop("truncation")
            encoded = {k: torch.tensor([v], dtype=torch.long, device=dev) for k, v in encoded.items()}
            with torch.no_grad():
                out = mod(**encoded)
                probs = F.softmax(out.logits, dim=-1).cpu().numpy()[0]
            p_sup, p_hal = float(probs[0]), float(probs[1])

        elif model_key == "longformer-head":
            tok = model_obj["tokenizer"]
            mod = model_obj["model"]
            dev = model_obj["device"]
            encoded = encode_pair_with_answer_preservation(
                context=context,
                answer=answer,
                tokenizer=tok,
                max_length=1024,
                truncation_strategy="head",
                global_attention_mode="cls",
            )
            batch = {
                "input_ids": torch.tensor([encoded["input_ids"]], dtype=torch.long).to(dev),
                "attention_mask": torch.tensor([encoded["attention_mask"]], dtype=torch.long).to(dev),
                "global_attention_mask": torch.tensor([encoded["global_attention_mask"]], dtype=torch.long).to(dev),
            }
            with torch.no_grad():
                out = mod(**batch)
                probs = F.softmax(out.logits, dim=-1).cpu().numpy()[0]
            p_sup, p_hal = float(probs[0]), float(probs[1])

        elif model_key == "longformer-qca":
            tok = model_obj["tokenizer"]
            mod = model_obj["model"]
            selector = model_obj["selector"]
            dev = model_obj["device"]
            from hallucination_detector.data.longformer_smart_dataset import encode_qca_for_longformer

            encoded = encode_qca_for_longformer(
                question=question,
                context=context,
                answer=answer,
                tokenizer=tok,
                max_length=1024,
                truncation_strategy="smart_selector",
                global_attention_mode="cls",
                selector=selector,
            )
            batch = {
                "input_ids": torch.tensor([encoded["input_ids"]], dtype=torch.long).to(dev),
                "attention_mask": torch.tensor([encoded["attention_mask"]], dtype=torch.long).to(dev),
                "global_attention_mask": torch.tensor([encoded["global_attention_mask"]], dtype=torch.long).to(dev),
            }
            with torch.no_grad():
                out = mod(**batch)
                probs = F.softmax(out.logits, dim=-1).cpu().numpy()[0]
            p_sup, p_hal = float(probs[0]), float(probs[1])

        else:
            raise NotImplementedError(f"Prediction logic for {model_key} not implemented.")

        is_hal, pred_label = classify_hallucination_score(p_hal, DEFAULT_HALLUCINATION_THRESHOLD)

        details = {"input_mode": meta.input_mode, "architecture": meta.architecture}
        if model_key == "deberta":
            details["truncation"] = truncation
        if model_key == "longformer-qca":
            details["selector_idf_loaded"] = bool(model_obj["selector_idf_loaded"])
            details["selector_fitted"] = bool(model_obj["selector_fitted"])
            details["selector_provenance_status"] = model_obj["selector_provenance_status"]
            details["selector_source"] = model_obj["selector_source"]
            details["selector_training_metadata"] = model_obj["selector_training_metadata"]
            details["checkpoint_revision"] = model_obj["checkpoint_revision"]
        if model_key == "bilstm-optimized":
            details.update(model_obj.artifact_metadata)

        return PredictionResult(
            model_name=meta.display_name,
            model_id=meta.repo_id,
            prediction=pred_label,
            is_hallucinated=is_hal,
            hallucination_probability=p_hal,
            supported_probability=p_sup,
            raw_score=p_hal,
            is_calibrated=False,
            details=details,
        )
