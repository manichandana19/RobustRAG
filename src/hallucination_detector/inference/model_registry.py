from __future__ import annotations

from dataclasses import dataclass, field
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
from hallucination_detector.data.torch_dataset import (
    RAGTruthTransformerDataset,
    LongformerDataCollator,
    encode_pair_with_answer_preservation,
)
from hallucination_detector.data.longformer_smart_dataset import (
    RAGTruthSmartLongformerDataset,
)
from hallucination_detector.inference.deberta_encoding import encode_context_answer
from hallucination_detector.inference.selector_io import load_selector
from hallucination_detector.modeling.bilstm_classifier import BiLSTMClassifier
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


MODEL_CATALOG: dict[str, ModelMetadata] = {
    "longformer-head": ModelMetadata(
        key="longformer-head",
        display_name="Longformer (Head Truncation) - Project Best",
        repo_id="nishantwhat/robustrag-longformer-head",
        architecture="Longformer (1024 tokens) + Sliding Window (w=512)",
        input_mode="C+A",
        description="Top-performing model. Preserves 100% of answer tokens and uses head-truncation on context.",
        limitations="Evaluates Context + Answer. Does not directly condition on Question tokens.",
    ),
    "longformer-qca": ModelMetadata(
        key="longformer-qca",
        display_name="Longformer (Smart Q+C+A)",
        repo_id="nishantwhat/robustrag-longformer-qca",
        architecture="Longformer (1024 tokens) + Extractive SentenceSelector",
        input_mode="Q+C+A",
        description="Conditions on Question + Context + Answer with contradiction-aware sentence ranking.",
        limitations="Extractive selector depends on keyword & contradiction overlap cues.",
    ),
    "deberta": ModelMetadata(
        key="deberta",
        display_name="DeBERTa-v3-base (512 tokens)",
        repo_id="nishantwhat/robustrag-deberta",
        architecture="DeBERTa-v3-base + Disentangled Attention",
        input_mode="C+A",
        description="512-token baseline cross-encoder.",
        limitations="Subject to answer-truncation on contexts exceeding 512 tokens. Best on shorter QA passages.",
    ),
    "bilstm-attention": ModelMetadata(
        key="bilstm-attention",
        display_name="BiLSTM + Bahdanau Attention",
        repo_id="nishantwhat/robustrag-bilstm-attention",
        architecture="Bidirectional LSTM (2-layer, 128-dim) + Additive Self-Attention",
        input_mode="C+A",
        description="Neural recurrent baseline with additive attention over sequence tokens.",
        limitations="Fixed 50k vocabulary; lacks deep pre-trained contextual semantic transfer.",
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


class ModelRegistry:
    """Lazy model loader and inference coordinator for RobustRAG models."""

    def __init__(
        self,
        cache_dir: str | Path | None = None,
        prefer_remote: bool = False,
        deberta_preserve_answer: bool = True,
    ) -> None:
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.prefer_remote = prefer_remote
        # True: when input exceeds 512 tokens, shorten the context so the answer survives.
        # False: reproduce the legacy right-truncation the DeBERTa checkpoint was trained with.
        self.deberta_preserve_answer = deberta_preserve_answer
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

    def _load_deberta(self, repo_id: str, revision: str, device: str | torch.device) -> dict[str, Any]:
        local_dir = Path("reports/experiments/baseline_deberta/best_model")
        if not self.prefer_remote and (local_dir / "model.safetensors").exists():
            model_path = str(local_dir)
        else:
            model_path = snapshot_download(repo_id=repo_id, revision=revision, cache_dir=self.cache_dir)

        tokenizer = AutoTokenizer.from_pretrained(model_path)
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
        local_dir = Path("reports/experiments/longformer_1024_smart_qca/best_model")
        if not self.prefer_remote and (local_dir / "model.safetensors").exists():
            model_path = str(local_dir)
        else:
            model_path = snapshot_download(repo_id=repo_id, revision=revision, cache_dir=self.cache_dir)

        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = LongformerForSequenceClassification.from_pretrained(model_path)
        model.to(torch.device(device))
        model.eval()
        # Training fitted IDF weights on the train split; use the same table at inference.
        idf_path = Path(__file__).resolve().parents[3] / "configs" / "qca_selector_idf.json"
        if idf_path.exists():
            selector = load_selector(idf_path)
            selector_fitted = True
        else:
            # Unfitted fallback (all word weights 1.0): does NOT match training preprocessing.
            selector = SentenceSelector(default_idf=1.0)
            selector_fitted = False
        return {
            "model": model,
            "tokenizer": tokenizer,
            "selector": selector,
            "selector_fitted": selector_fitted,
            "device": device,
        }

    def predict(
        self,
        model_key: str,
        question: str,
        context: str,
        answer: str,
        device: str | torch.device | None = None,
    ) -> PredictionResult:
        """Run single-instance inference with model-specific preprocessing."""
        meta = MODEL_CATALOG[model_key]
        model_obj = self.load_model(model_key, device=device)
        extra_details: dict[str, Any] = {}
        record = RAGTruthRecord(
            id="query_1",
            source_id="src_1",
            question=question.strip(),
            context=context.strip(),
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

        elif model_key == "deberta":
            tok = model_obj["tokenizer"]
            mod = model_obj["model"]
            dev = model_obj["device"]
            # DeBERTa single-string C+A; context is shortened (not the answer) when too long.
            enc, trunc_info = encode_context_answer(
                tok,
                normalize_text(context),
                normalize_text(answer),
                max_length=512,
                preserve_answer=self.deberta_preserve_answer,
            )
            extra_details["truncation"] = trunc_info
            encoded = {
                "input_ids": torch.tensor([enc["input_ids"]], dtype=torch.long).to(dev),
                "attention_mask": torch.tensor([enc["attention_mask"]], dtype=torch.long).to(dev),
            }
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
            extra_details["selector_fitted"] = model_obj["selector_fitted"]

        else:
            raise NotImplementedError(f"Prediction logic for {model_key} not implemented.")

        is_hal = p_hal >= 0.5
        pred_label = "HALLUCINATED" if is_hal else "SUPPORTED"

        return PredictionResult(
            model_name=meta.display_name,
            model_id=meta.repo_id,
            prediction=pred_label,
            is_hallucinated=is_hal,
            hallucination_probability=p_hal,
            supported_probability=p_sup,
            raw_score=p_hal,
            is_calibrated=False,
            details={"input_mode": meta.input_mode, "architecture": meta.architecture, **extra_details},
        )
