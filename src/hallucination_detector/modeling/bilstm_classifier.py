from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any, Sequence, Literal

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from hallucination_detector.data.schema import RAGTruthRecord
from hallucination_detector.data.bilstm_dataset import (
    Vocabulary,
    RAGTruthBiLSTMDataset,
    BiLSTMDataCollator,
)


class BahdanauAttention(nn.Module):
    """
    Additive / Bahdanau-style self-attention layer.
    Computes a scalar alignment score for each time step over LSTM outputs.
    """

    def __init__(self, hidden_dim: int, attention_dim: int = 128) -> None:
        super().__init__()
        self.w_proj = nn.Linear(hidden_dim, attention_dim, bias=True)
        self.v_proj = nn.Linear(attention_dim, 1, bias=False)

    def forward(
        self,
        rnn_outputs: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            rnn_outputs: [batch_size, seq_len, hidden_dim]
            attention_mask: [batch_size, seq_len] with 1 for valid tokens, 0 for padding.

        Returns:
            context_vector: [batch_size, hidden_dim]
            attention_weights: [batch_size, seq_len]
        """
        # score = v^T tanh(W * h) -> [batch_size, seq_len, 1]
        scores = self.v_proj(torch.tanh(self.w_proj(rnn_outputs))).squeeze(-1)

        # Mask padding tokens with large negative value
        scores = scores.masked_fill(attention_mask == 0, -1e9)
        weights = F.softmax(scores, dim=-1)  # [batch_size, seq_len]

        # Context vector as weighted sum: [batch_size, hidden_dim]
        context_vector = torch.bmm(weights.unsqueeze(1), rnn_outputs).squeeze(1)
        return context_vector, weights


class BiLSTMClassifierNet(nn.Module):
    """
    PyTorch Neural Network module: Word Embeddings + Bidirectional LSTM + Attention.
    """

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 128,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.3,
        use_attention: bool = True,
        attention_dim: int = 128,
        num_classes: int = 2,
        pad_id: int = 0,
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.use_attention = use_attention
        self.pad_id = pad_id

        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embed_dim,
            padding_idx=pad_id,
        )
        self.dropout_layer = nn.Dropout(dropout)

        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            bidirectional=True,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        bilstm_out_dim = hidden_dim * 2

        if use_attention:
            self.attention = BahdanauAttention(
                hidden_dim=bilstm_out_dim,
                attention_dim=attention_dim,
            )
            classifier_in_dim = bilstm_out_dim
        else:
            self.attention = None
            classifier_in_dim = bilstm_out_dim

        self.fc = nn.Sequential(
            nn.Linear(classifier_in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        lengths: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """
        Forward pass.
        Returns:
            logits: [batch_size, num_classes]
            attn_weights: [batch_size, seq_len] or None
        """
        batch_size, seq_len = input_ids.shape
        embedded = self.dropout_layer(self.embedding(input_ids))

        # Enforce CPU lengths if packing
        if lengths is not None:
            lengths_cpu = lengths.cpu().clamp(min=1, max=seq_len)
            packed_embedded = nn.utils.rnn.pack_padded_sequence(
                embedded,
                lengths_cpu,
                batch_first=True,
                enforce_sorted=False,
            )
            packed_out, (hn, cn) = self.lstm(packed_embedded)
            lstm_out, _ = nn.utils.rnn.pad_packed_sequence(
                packed_out,
                batch_first=True,
                total_length=seq_len,
            )
        else:
            lstm_out, (hn, cn) = self.lstm(embedded)

        if self.use_attention and self.attention is not None:
            pooled, attn_weights = self.attention(lstm_out, attention_mask)
        else:
            # Mean pooling over non-padded tokens
            mask_expanded = attention_mask.unsqueeze(-1).float()
            sum_vectors = torch.sum(lstm_out * mask_expanded, dim=1)
            sum_mask = mask_expanded.sum(dim=1).clamp(min=1.0)
            pooled = sum_vectors / sum_mask
            attn_weights = None

        logits = self.fc(pooled)
        return logits, attn_weights


@dataclass
class BiLSTMClassifier:
    """
    High-level Scikit-Learn style interface matching LongformerClassifier & DeBERTaClassifier.

    Label convention:
        0 = supported
        1 = hallucinated
    """

    vocab: Vocabulary | None = None
    embed_dim: int = 128
    hidden_dim: int = 128
    num_layers: int = 2
    dropout: float = 0.3
    use_attention: bool = True
    attention_dim: int = 128
    max_length: int = 512
    truncation_strategy: str = "context_tail"
    net: BiLSTMClassifierNet | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.vocab is not None:
            self._init_network()

    def _init_network(self) -> None:
        assert self.vocab is not None
        self.net = BiLSTMClassifierNet(
            vocab_size=len(self.vocab),
            embed_dim=self.embed_dim,
            hidden_dim=self.hidden_dim,
            num_layers=self.num_layers,
            dropout=self.dropout,
            use_attention=self.use_attention,
            attention_dim=self.attention_dim,
            num_classes=2,
            pad_id=self.vocab.pad_id,
        )

    def fit_vocab(
        self,
        records: Sequence[RAGTruthRecord],
        max_size: int = 50000,
        min_freq: int = 2,
    ) -> None:
        self.vocab = Vocabulary.build_from_records(records, max_size=max_size, min_freq=min_freq)
        self._init_network()

    def build_dataset(
        self,
        records: Sequence[RAGTruthRecord],
    ) -> RAGTruthBiLSTMDataset:
        if self.vocab is None:
            raise RuntimeError("Vocabulary has not been initialized. Call fit_vocab() first.")
        return RAGTruthBiLSTMDataset(
            records=records,
            vocab=self.vocab,
            max_length=self.max_length,
            truncation_strategy=self.truncation_strategy,
        )

    def predict_proba(
        self,
        records: Sequence[RAGTruthRecord],
        batch_size: int = 32,
        device: str | torch.device = "cpu",
    ) -> np.ndarray:
        if self.net is None or self.vocab is None:
            raise RuntimeError("Model is not initialized.")

        dev = torch.device(device)
        self.net.to(dev)
        self.net.eval()

        dataset = self.build_dataset(records)
        collator = BiLSTMDataCollator(pad_id=self.vocab.pad_id)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collator)

        all_probs: list[np.ndarray] = []
        with torch.no_grad():
            for batch in loader:
                input_ids = batch["input_ids"].to(dev)
                attention_mask = batch["attention_mask"].to(dev)
                lengths = batch["lengths"].to(dev)

                logits, _ = self.net(input_ids, attention_mask, lengths=lengths)
                probs = F.softmax(logits, dim=-1).cpu().numpy()
                all_probs.append(probs)

        if not all_probs:
            return np.empty((0, 2), dtype=np.float32)
        return np.vstack(all_probs)

    def predict(
        self,
        records: Sequence[RAGTruthRecord],
        batch_size: int = 32,
        device: str | torch.device = "cpu",
    ) -> np.ndarray:
        probs = self.predict_proba(records, batch_size=batch_size, device=device)
        return np.argmax(probs, axis=1)

    def save(self, output_dir: str | Path) -> None:
        """Save vocabulary, hyperparameters, and PyTorch weights."""
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        if self.vocab is None or self.net is None:
            raise RuntimeError("Cannot save uninitialized model.")

        self.vocab.save(out_path / "vocab.json")

        meta = {
            "embed_dim": self.embed_dim,
            "hidden_dim": self.hidden_dim,
            "num_layers": self.num_layers,
            "dropout": self.dropout,
            "use_attention": self.use_attention,
            "attention_dim": self.attention_dim,
            "max_length": self.max_length,
            "truncation_strategy": self.truncation_strategy,
        }
        with (out_path / "model_config.json").open("w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

        torch.save(self.net.state_dict(), out_path / "bilstm_weights.pt")

    @classmethod
    def load(cls, model_dir: str | Path, device: str | torch.device = "cpu") -> BiLSTMClassifier:
        """Load vocabulary, config, and PyTorch weights."""
        in_path = Path(model_dir)
        vocab = Vocabulary.load(in_path / "vocab.json")

        with (in_path / "model_config.json").open("r", encoding="utf-8") as f:
            meta = json.load(f)

        classifier = cls(
            vocab=vocab,
            embed_dim=meta["embed_dim"],
            hidden_dim=meta["hidden_dim"],
            num_layers=meta["num_layers"],
            dropout=meta["dropout"],
            use_attention=meta["use_attention"],
            attention_dim=meta["attention_dim"],
            max_length=meta["max_length"],
            truncation_strategy=meta.get("truncation_strategy", "context_tail"),
        )
        classifier._init_network()
        weights = torch.load(in_path / "bilstm_weights.pt", map_location=device)
        classifier.net.load_state_dict(weights)
        classifier.net.to(torch.device(device))
        classifier.net.eval()
        return classifier
