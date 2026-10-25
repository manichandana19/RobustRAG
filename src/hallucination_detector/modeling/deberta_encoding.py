"""Answer-preserving DeBERTa encoding shared by training and inference."""
from __future__ import annotations

from typing import Any

from hallucination_detector.modeling.tfidf_logistic import normalize_text


def format_deberta_input(context: object, answer: object) -> str:
    return f"[CONTEXT] {normalize_text(context)} [ANSWER] {normalize_text(answer)}"


def encode_deberta_input(context: object, answer: object, tokenizer: Any, max_length: int = 512) -> dict[str, Any]:
    """Keep the complete answer whenever it fits; trim context tokens first.

    Returns tokenizer fields plus `truncation` metadata. Token offsets come
    from the full formatted string so the normal, non-truncated representation
    remains byte-for-byte equivalent to standard tokenizer encoding.
    """
    if max_length <= 0:
        raise ValueError("max_length must be positive.")
    text = format_deberta_input(context, answer)
    answer_marker_start = text.index("[ANSWER]")
    answer_text_start = answer_marker_start + len("[ANSWER]")
    answer_text_start += len(text[answer_text_start:]) - len(text[answer_text_start:].lstrip())
    context_text_start = text.index("[CONTEXT]") + len("[CONTEXT]")

    encoded = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True, truncation=False)
    ids = list(encoded["input_ids"])
    offsets = list(encoded["offset_mapping"])
    prefix, context_ids, answer_prefix_ids, answer_ids = [], [], [], []
    for token_id, (start, end) in zip(ids, offsets):
        if end <= context_text_start:
            prefix.append(token_id)
        elif start < answer_marker_start:
            context_ids.append(token_id)
        elif start < answer_text_start:
            answer_prefix_ids.append(token_id)
        else:
            answer_ids.append(token_id)

    special_count = tokenizer.num_special_tokens_to_add(pair=False)
    available = max_length - special_count
    if available < 0:
        raise ValueError("max_length is smaller than the tokenizer's required special-token count.")
    mandatory = len(prefix) + len(answer_prefix_ids) + len(answer_ids)
    answer_truncated = mandatory > available
    if answer_truncated:
        # The complete answer cannot fit even after dropping context. Preserve
        # its opening tokens, bounded by the remaining budget.
        answer_budget = max(0, available - len(prefix) - len(answer_prefix_ids))
        answer_ids = answer_ids[:answer_budget]
        context_ids = []
    else:
        context_budget = available - mandatory
        context_ids = context_ids[:context_budget]

    content_ids = prefix + context_ids + answer_prefix_ids + answer_ids
    build_special = getattr(tokenizer, "build_inputs_with_special_tokens", None)
    if callable(build_special):
        input_ids = build_special(content_ids)
    else:
        # DebertaV2Tokenizer in some Transformers releases omits this helper
        # despite exposing the standard single-sequence CLS/SEP convention.
        cls_id, sep_id = tokenizer.cls_token_id, tokenizer.sep_token_id
        if cls_id is None or sep_id is None or special_count != 2:
            raise TypeError("DeBERTa answer-preserving encoding requires a tokenizer with CLS/SEP special tokens.")
        input_ids = [cls_id, *content_ids, sep_id]
    attention_mask = [1] * len(input_ids)
    result: dict[str, Any] = {"input_ids": input_ids, "attention_mask": attention_mask}
    make_token_types = getattr(tokenizer, "create_token_type_ids_from_sequences", None)
    if callable(make_token_types):
        result["token_type_ids"] = make_token_types(content_ids)
    result["truncation"] = {
        "context_truncated": len(context_ids) < sum(start >= context_text_start and start < answer_marker_start for start, _ in offsets),
        "answer_truncated": answer_truncated,
        "input_tokens": len(input_ids),
        "max_length": max_length,
    }
    return result
