from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence, Literal
import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerBase

from hallucination_detector.data.schema import RAGTruthRecord
from hallucination_detector.modeling.tfidf_logistic import normalize_text


def encode_pair_with_answer_preservation(
    context: Any,
    answer: Any,
    tokenizer: PreTrainedTokenizerBase,
    max_length: int = 1024,
    truncation_strategy: Literal["head", "tail"] = "head",
    global_attention_mode: Literal["cls", "cls_and_answer", "none"] = "cls",
) -> dict[str, list[int]]:
    """
    Encode a (context, answer) pair ensuring the answer is preserved with 100% fidelity.

    Sequence structure (RoBERTa / Longformer style):
        <s> Context tokens </s></s> Answer tokens </s>

    Truncation budget logic:
        - Total budget = max_length tokens.
        - Delimiter overhead = 4 tokens (<s>, </s>, </s>, </s>).
        - Answer is tokenized first. If answer fits within max_length - 4, all answer tokens are kept.
        - Remaining budget is allocated to context:
            context_budget = max_length - 4 - len(answer_tokens)
        - Context is truncated to context_budget according to truncation_strategy:
            - "head": drops early context tokens (context_tokens[-context_budget:])
            - "tail": drops late context tokens (context_tokens[:context_budget])
        - If answer itself exceeds max_length - 4 (edge case), answer is truncated from the tail,
          leaving a minimal 16-token window for context.

    Global attention mask:
        - "cls": 1 at index 0 (<s>), 0 for all other tokens.
        - "cls_and_answer": 1 at index 0 and 1 for all answer tokens, 0 for context tokens.
        - "none": all 0s.
    """
    if truncation_strategy not in ("head", "tail"):
        raise ValueError(f"Unknown truncation_strategy: {truncation_strategy}. Must be 'head' or 'tail'.")

    norm_context = normalize_text(context)
    norm_answer = normalize_text(answer)

    cls_id = tokenizer.cls_token_id if tokenizer.cls_token_id is not None else 0
    sep_id = tokenizer.sep_token_id if tokenizer.sep_token_id is not None else 2

    ans_tokens = tokenizer.encode(norm_answer, add_special_tokens=False)
    ctx_tokens = tokenizer.encode(norm_context, add_special_tokens=False)

    num_special = 4  # <s> + </s></s> + </s>
    available_budget = max(1, max_length - num_special)

    if len(ans_tokens) > available_budget:
        # Edge case: answer exceeds total budget
        min_ctx = min(16, len(ctx_tokens))
        ans_budget = available_budget - min_ctx
        ans_tokens = ans_tokens[:ans_budget]
        ctx_budget = min_ctx
    else:
        ctx_budget = available_budget - len(ans_tokens)

    if len(ctx_tokens) > ctx_budget:
        if truncation_strategy == "head":
            ctx_tokens = ctx_tokens[-ctx_budget:] if ctx_budget > 0 else []
        else:  # tail
            ctx_tokens = ctx_tokens[:ctx_budget] if ctx_budget > 0 else []

    # Assemble sequence
    # <s> ctx </s></s> ans </s>
    input_ids = [cls_id] + ctx_tokens + [sep_id, sep_id] + ans_tokens + [sep_id]
    seq_len = len(input_ids)
    attention_mask = [1] * seq_len

    # Assemble global attention mask
    global_attention_mask = [0] * seq_len
    if global_attention_mode in ("cls", "cls_and_answer"):
        global_attention_mask[0] = 1

    if global_attention_mode == "cls_and_answer":
        # Answer tokens start after [cls_id] + ctx_tokens + [sep_id, sep_id]
        ans_start_idx = 1 + len(ctx_tokens) + 2
        ans_end_idx = ans_start_idx + len(ans_tokens)
        for idx in range(ans_start_idx, ans_end_idx):
            global_attention_mask[idx] = 1

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "global_attention_mask": global_attention_mask,
    }


class RAGTruthTransformerDataset(Dataset):
    """
    PyTorch Dataset for transformer hallucination classification with answer preservation.
    """

    def __init__(
        self,
        records: Sequence[RAGTruthRecord],
        tokenizer: PreTrainedTokenizerBase,
        input_mode: str = "ca",
        max_length: int = 1024,
        truncation_strategy: Literal["head", "tail"] = "head",
        global_attention_mode: Literal["cls", "cls_and_answer", "none"] = "cls",
    ) -> None:
        if input_mode != "ca":
            raise ValueError("Currently supports input_mode='ca'.")
        if max_length <= 0:
            raise ValueError("max_length must be positive.")

        self.records = list(records)
        self.tokenizer = tokenizer
        self.input_mode = input_mode
        self.max_length = max_length
        self.truncation_strategy = truncation_strategy
        self.global_attention_mode = global_attention_mode

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        encoded = encode_pair_with_answer_preservation(
            context=record.context,
            answer=record.answer,
            tokenizer=self.tokenizer,
            max_length=self.max_length,
            truncation_strategy=self.truncation_strategy,
            global_attention_mode=self.global_attention_mode,
        )

        item = {key: torch.tensor(val, dtype=torch.long) for key, val in encoded.items()}
        item["labels"] = torch.tensor(int(record.hallucinated), dtype=torch.long)
        return item


@dataclass
class LongformerDataCollator:
    """
    Dynamic batch padding collator for Longformer.

    Pads input_ids, attention_mask, and global_attention_mask up to the batch max length,
    rounded up to the nearest multiple of `pad_to_multiple_of` (default 512, which is
    required by Longformer's sliding-window CUDA kernels).
    """

    tokenizer: PreTrainedTokenizerBase
    pad_to_multiple_of: int = 512

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        max_len = max(len(f["input_ids"]) for f in features)
        if self.pad_to_multiple_of is not None and self.pad_to_multiple_of > 0:
            remainder = max_len % self.pad_to_multiple_of
            if remainder != 0:
                target_len = max_len + (self.pad_to_multiple_of - remainder)
            else:
                target_len = max_len
        else:
            target_len = max_len

        pad_token_id = (
            self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else 1
        )

        batch: dict[str, torch.Tensor] = {}

        # Pad sequence tensors
        for key in ["input_ids", "attention_mask", "global_attention_mask"]:
            if key not in features[0]:
                continue
            pad_val = pad_token_id if key == "input_ids" else 0
            padded_rows = []
            for f in features:
                row = f[key]
                if isinstance(row, torch.Tensor):
                    row = row.tolist()
                pad_count = target_len - len(row)
                padded_rows.append(row + [pad_val] * pad_count)
            batch[key] = torch.tensor(padded_rows, dtype=torch.long)

        if "labels" in features[0]:
            labels = [
                f["labels"].item() if isinstance(f["labels"], torch.Tensor) else f["labels"]
                for f in features
            ]
            batch["labels"] = torch.tensor(labels, dtype=torch.long)

        return batch
