from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
import re
from typing import Any, Sequence

from hallucination_detector.data.schema import RAGTruthRecord
from hallucination_detector.modeling.tfidf_logistic import normalize_text

_TOKEN_PATTERN = re.compile(r"\w+|[^\w\s]", re.UNICODE)


def simple_tokenize(text: str) -> list[str]:
    """Tokenize text into lowercase words and punctuation tokens."""
    return _TOKEN_PATTERN.findall(text.lower())


@dataclass
class Vocabulary:
    """Vocabulary mapping tokens to integer indices with <pad> and <unk>."""

    pad_token: str = "<pad>"
    unk_token: str = "<unk>"
    pad_id: int = 0
    unk_id: int = 1
    token2id: dict[str, int] = field(default_factory=dict)
    id2token: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.token2id:
            self.token2id = {self.pad_token: self.pad_id, self.unk_token: self.unk_id}
            self.id2token = [self.pad_token, self.unk_token]

    def __len__(self) -> int:
        return len(self.id2token)

    def add_token(self, token: str) -> int:
        if token not in self.token2id:
            idx = len(self.id2token)
            self.token2id[token] = idx
            self.id2token.append(token)
            return idx
        return self.token2id[token]

    def encode(self, tokens: list[str]) -> list[int]:
        return [self.token2id.get(token, self.unk_id) for token in tokens]

    @classmethod
    def build_from_records(
        cls,
        records: Sequence[RAGTruthRecord],
        max_size: int = 50000,
        min_freq: int = 2,
    ) -> Vocabulary:
        from collections import Counter

        counter: Counter[str] = Counter()
        for record in records:
            text = f"{normalize_text(record.question)} {normalize_text(record.context)} {normalize_text(record.answer)}"
            tokens = simple_tokenize(text)
            counter.update(tokens)

        vocab = cls()
        # Guarantee section markers are always in vocabulary
        for marker in ["[question]", "[context]", "[answer]"]:
            vocab.add_token(marker)

        for token, count in counter.most_common():

            if count < min_freq:
                break
            if len(vocab) >= max_size:
                break
            vocab.add_token(token)

        return vocab

    def save(self, filepath: str | Path) -> None:
        path = Path(filepath)
        data = {
            "pad_token": self.pad_token,
            "unk_token": self.unk_token,
            "pad_id": self.pad_id,
            "unk_id": self.unk_id,
            "id2token": self.id2token,
        }
        with path.open("w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False)

    @classmethod
    def load(cls, filepath: str | Path) -> Vocabulary:
        path = Path(filepath)
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        vocab = cls(
            pad_token=data["pad_token"],
            unk_token=data["unk_token"],
            pad_id=data["pad_id"],
            unk_id=data["unk_id"],
            token2id={tok: idx for idx, tok in enumerate(data["id2token"])},
            id2token=data["id2token"],
        )
        return vocab


def encode_qca_tokens(
    question: Any,
    context: Any,
    answer: Any,
    vocab: Vocabulary,
    max_length: int = 512,
    truncation_strategy: str = "context_tail",
    min_context_tokens: int = 32,
) -> dict[str, list[int]]:
    """
    Tokenize and encode (Question, Context, Answer) into a sequence.

    Budget allocation:
    - Special boundary tokens: <q>, <c>, <a> or standard separator markers can be represented as tokens.
      For general word embeddings, we prefix text sections with [question], [context], [answer] tokens.
    - Preserves question and answer up to available budget.
    - Context gets remaining budget and is truncated.
    """
    q_str = f"[question] {normalize_text(question)}"
    c_str = f"[context] {normalize_text(context)}"
    a_str = f"[answer] {normalize_text(answer)}"

    q_tokens = simple_tokenize(q_str)
    a_tokens = simple_tokenize(a_str)
    c_tokens = simple_tokenize(c_str)

    # Calculate budgets: Question + Answer are prioritized
    # Total sequence budget is max_length
    fixed_qa_len = len(q_tokens) + len(a_tokens)

    if fixed_qa_len >= max_length:
        # Edge case: Q + A exceed or equal max_length.
        # Preserve answer fully if possible, truncate Q from head/tail, allocate min context.
        # Keep answer first
        reserve_for_ctx = min(min_context_tokens, max(0, max_length // 8))
        avail_for_qa = max(1, max_length - reserve_for_ctx)

        if len(a_tokens) < avail_for_qa:
            q_budget = avail_for_qa - len(a_tokens)
            q_tokens = q_tokens[:q_budget]
        else:
            a_tokens = a_tokens[:avail_for_qa]
            q_tokens = []

        c_budget = max_length - (len(q_tokens) + len(a_tokens))
        c_tokens = c_tokens[:c_budget] if c_budget > 0 else []
    else:
        ctx_budget = max_length - fixed_qa_len
        if len(c_tokens) > ctx_budget:
            if truncation_strategy == "context_head":
                # Drop beginning of context, keep end
                c_tokens = c_tokens[-ctx_budget:] if ctx_budget > 0 else []
            else:
                # Default "context_tail": keep beginning of context, drop end
                c_tokens = c_tokens[:ctx_budget] if ctx_budget > 0 else []

    merged_tokens = q_tokens + c_tokens + a_tokens
    token_ids = vocab.encode(merged_tokens)
    seq_len = len(token_ids)
    mask = [1] * seq_len

    return {
        "input_ids": token_ids,
        "attention_mask": mask,
        "length": seq_len,
    }


class RAGTruthBiLSTMDataset:
    """PyTorch Dataset for BiLSTM hallucination classification."""

    def __init__(
        self,
        records: Sequence[RAGTruthRecord],
        vocab: Vocabulary,
        max_length: int = 512,
        truncation_strategy: str = "context_tail",
    ) -> None:
        import torch
        from torch.utils.data import Dataset

        self.records = list(records)
        self.vocab = vocab
        self.max_length = max_length
        self.truncation_strategy = truncation_strategy

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        import torch

        record = self.records[index]
        encoded = encode_qca_tokens(
            question=record.question,
            context=record.context,
            answer=record.answer,
            vocab=self.vocab,
            max_length=self.max_length,
            truncation_strategy=self.truncation_strategy,
        )
        return {
            "input_ids": torch.tensor(encoded["input_ids"], dtype=torch.long),
            "attention_mask": torch.tensor(encoded["attention_mask"], dtype=torch.long),
            "length": torch.tensor(encoded["length"], dtype=torch.long),
            "labels": torch.tensor(int(record.hallucinated), dtype=torch.long),
        }


@dataclass
class BiLSTMDataCollator:
    """Collator that dynamically pads batches to maximum batch length."""

    pad_id: int = 0

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, Any]:
        import torch

        max_len = max(len(f["input_ids"]) for f in features)
        batch_size = len(features)

        padded_input_ids = torch.full((batch_size, max_len), self.pad_id, dtype=torch.long)
        padded_attention_mask = torch.zeros((batch_size, max_len), dtype=torch.long)
        lengths = []
        labels = []

        for i, f in enumerate(features):
            ids = f["input_ids"]
            seq_len = len(ids)
            padded_input_ids[i, :seq_len] = ids
            padded_attention_mask[i, :seq_len] = f["attention_mask"]
            lengths.append(seq_len)
            labels.append(f["labels"])

        return {
            "input_ids": padded_input_ids,
            "attention_mask": padded_attention_mask,
            "lengths": torch.tensor(lengths, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }

