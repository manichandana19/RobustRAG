"""DeBERTa input encoding for inference with optional answer preservation.

The published DeBERTa checkpoint was trained on single strings of the form
"[CONTEXT] <ctx> [ANSWER] <ans>" with plain right-truncation, which silently
removes the answer when the context is long (the repo's audit found this in
about half of test records). This module keeps that exact format when the input
fits, and when it does not fit it can either reproduce the legacy truncation or
shorten the context instead so the answer survives.
"""
from __future__ import annotations

from typing import Any

MIN_CONTEXT_TOKENS = 32


def encode_context_answer(
    tokenizer: Any,
    context: str,
    answer: str,
    max_length: int = 512,
    preserve_answer: bool = True,
) -> tuple[dict[str, list[int]], dict[str, Any]]:
    text = f"[CONTEXT] {context} [ANSWER] {answer}"
    full_ids = tokenizer(text, add_special_tokens=True, truncation=False)["input_ids"]
    info: dict[str, Any] = {
        "total_tokens": len(full_ids),
        "max_length": max_length,
        "truncated": False,
        "mode": "fits",
        "context_tokens_dropped": 0,
        "answer_tokens_dropped": 0,
    }

    if len(full_ids) <= max_length:
        return {"input_ids": full_ids, "attention_mask": [1] * len(full_ids)}, info

    info["truncated"] = True
    ans_ids = tokenizer.encode(f"[ANSWER] {answer}", add_special_tokens=False)
    ctx_ids = tokenizer.encode(f"[CONTEXT] {context}", add_special_tokens=False)

    if not preserve_answer:
        ids = full_ids[: max_length - 1] + [full_ids[-1]]  # right-truncate, keep final [SEP]
        overflow = len(full_ids) - max_length
        info["mode"] = "legacy_right_truncation"
        info["answer_tokens_dropped"] = min(len(ans_ids), overflow)
        info["context_tokens_dropped"] = max(0, overflow - len(ans_ids))
        return {"input_ids": ids, "attention_mask": [1] * len(ids)}, info

    body_budget = max_length - 2  # [CLS] ... [SEP]
    if len(ans_ids) > body_budget - MIN_CONTEXT_TOKENS:
        keep = max(1, body_budget - MIN_CONTEXT_TOKENS)
        info["answer_tokens_dropped"] = len(ans_ids) - keep
        ans_ids = ans_ids[:keep]
    ctx_budget = body_budget - len(ans_ids)
    info["context_tokens_dropped"] = max(0, len(ctx_ids) - ctx_budget)
    ctx_ids = ctx_ids[:ctx_budget]  # keep the start of the context, like training did
    ids = tokenizer.build_inputs_with_special_tokens(ctx_ids + ans_ids)
    info["mode"] = "answer_preserved"
    return {"input_ids": ids, "attention_mask": [1] * len(ids)}, info
