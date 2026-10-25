from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence, Literal
import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerBase

from hallucination_detector.data.schema import RAGTruthRecord
from hallucination_detector.data.sentence_selector import SentenceSelector
from hallucination_detector.modeling.tfidf_logistic import normalize_text


def encode_qca_for_longformer(
    question: Any,
    context: Any,
    answer: Any,
    tokenizer: PreTrainedTokenizerBase,
    max_length: int = 1024,
    truncation_strategy: Literal["head", "tail", "smart_selector"] = "smart_selector",
    global_attention_mode: Literal["cls", "cls_and_qa", "cls_and_answer", "none"] = "cls",
    selector: SentenceSelector | None = None,
) -> dict[str, list[int]]:
    """
    Encode (Question, Context, Answer) for Longformer sequence classification.

    Sequence structure (RoBERTa / Longformer multi-segment convention):
        <s> Question tokens </s></s> Context tokens </s></s> Answer tokens </s>

    Delimiters overhead:
        6 tokens: <s> (cls) + </s></s> + </s></s> + </s>

    Token budget allocation logic:
        - Preserves Question and Answer with 100% fidelity whenever feasible.
        - Answer and Question are tokenized first.
        - If Q + A fits in (max_length - 6), remaining budget goes to Context.
        - If truncation_strategy == "smart_selector":
            Context sentences are selected via relevance and contradiction scoring,
            filling up to the exact Context token budget while preserving original sequential order.
        - If truncation_strategy in ("head", "tail"):
            Falls back to standard token slicing.
    """
    norm_q = normalize_text(question)
    norm_c = normalize_text(context)
    norm_a = normalize_text(answer)

    cls_id = tokenizer.cls_token_id if tokenizer.cls_token_id is not None else 0
    sep_id = tokenizer.sep_token_id if tokenizer.sep_token_id is not None else 2

    q_tokens = tokenizer.encode(norm_q, add_special_tokens=False)
    ans_tokens = tokenizer.encode(norm_a, add_special_tokens=False)

    num_special = 6  # <s> + </s></s> + </s></s> + </s>
    available_budget = max(1, max_length - num_special)

    # 1. Budget enforcement for Q and A
    qa_len = len(q_tokens) + len(ans_tokens)
    if qa_len >= available_budget:
        # Edge case: Question + Answer exceed total available budget
        # Guarantee minimal context window of 16 tokens
        min_ctx = 16
        avail_for_qa = max(1, available_budget - min_ctx)
        # Prioritize answer over question if needed
        if len(ans_tokens) <= avail_for_qa:
            q_budget = avail_for_qa - len(ans_tokens)
            q_tokens = q_tokens[:q_budget]
        else:
            ans_tokens = ans_tokens[:avail_for_qa]
            q_tokens = []
        ctx_budget = min_ctx
    else:
        ctx_budget = available_budget - qa_len

    # 2. Context processing
    if truncation_strategy == "smart_selector" and selector is not None and norm_c:
        # Sentence-level smart selection
        # Estimate ~1.3 tokens per word to guide sentence selection
        max_words_est = max(1, int(ctx_budget / 1.3))
        selected_pairs = selector.select_sentences(
            question=norm_q,
            context=norm_c,
            answer=norm_a,
            max_words=max_words_est,
        )
        selected_context_text = " ".join(s for _, s in selected_pairs)
        ctx_tokens = tokenizer.encode(selected_context_text, add_special_tokens=False)
        # Strict hard trim to exact token budget
        if len(ctx_tokens) > ctx_budget:
            ctx_tokens = ctx_tokens[:ctx_budget]
    else:
        # Standard tokenization
        ctx_tokens = tokenizer.encode(norm_c, add_special_tokens=False)
        if len(ctx_tokens) > ctx_budget:
            if truncation_strategy == "head":
                ctx_tokens = ctx_tokens[-ctx_budget:] if ctx_budget > 0 else []
            else:  # tail
                ctx_tokens = ctx_tokens[:ctx_budget] if ctx_budget > 0 else []

    # 3. Assemble sequence: <s> Q </s></s> C </s></s> A </s>
    input_ids = (
        [cls_id]
        + q_tokens
        + [sep_id, sep_id]
        + ctx_tokens
        + [sep_id, sep_id]
        + ans_tokens
        + [sep_id]
    )
    seq_len = len(input_ids)
    attention_mask = [1] * seq_len

    # 4. Global attention mask
    global_attention_mask = [0] * seq_len
    if global_attention_mode in ("cls", "cls_and_qa", "cls_and_answer"):
        global_attention_mask[0] = 1

    if global_attention_mode in ("cls_and_qa", "cls_and_answer"):
        # Answer tokens start after: [cls_id] + q_tokens + [sep_id, sep_id] + ctx_tokens + [sep_id, sep_id]
        ans_start_idx = 1 + len(q_tokens) + 2 + len(ctx_tokens) + 2
        ans_end_idx = ans_start_idx + len(ans_tokens)
        for idx in range(ans_start_idx, ans_end_idx):
            global_attention_mask[idx] = 1

    if global_attention_mode == "cls_and_qa":
        # Also give global attention to question tokens
        q_start_idx = 1
        q_end_idx = q_start_idx + len(q_tokens)
        for idx in range(q_start_idx, q_end_idx):
            global_attention_mask[idx] = 1

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "global_attention_mask": global_attention_mask,
    }


class RAGTruthSmartLongformerDataset(Dataset):
    """
    PyTorch Dataset for Longformer with Q+C+A support and smart evidence selection.
    Preserves backwards compatibility with standard C+A mode.
    """

    def __init__(
        self,
        records: Sequence[RAGTruthRecord],
        tokenizer: PreTrainedTokenizerBase,
        input_mode: Literal["ca", "qca"] = "qca",
        max_length: int = 1024,
        truncation_strategy: Literal["head", "tail", "smart_selector"] = "smart_selector",
        global_attention_mode: Literal["cls", "cls_and_qa", "cls_and_answer", "none"] = "cls",
        selector: SentenceSelector | None = None,
    ) -> None:
        if input_mode not in ("ca", "qca"):
            raise ValueError(f"Unsupported input_mode '{input_mode}'. Must be 'ca' or 'qca'.")
        if max_length <= 0:
            raise ValueError("max_length must be positive.")

        self.records = list(records)
        self.tokenizer = tokenizer
        self.input_mode = input_mode
        self.max_length = max_length
        self.truncation_strategy = truncation_strategy
        self.global_attention_mode = global_attention_mode
        self.selector = selector

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]

        if self.input_mode == "qca":
            encoded = encode_qca_for_longformer(
                question=record.question,
                context=record.context,
                answer=record.answer,
                tokenizer=self.tokenizer,
                max_length=self.max_length,
                truncation_strategy=self.truncation_strategy,
                global_attention_mode=self.global_attention_mode,
                selector=self.selector,
            )
        else:
            # Fallback to existing C+A encoder
            from hallucination_detector.data.torch_dataset import encode_pair_with_answer_preservation
            strat = "head" if self.truncation_strategy not in ("head", "tail") else self.truncation_strategy
            att_mode = "cls" if self.global_attention_mode not in ("cls", "cls_and_answer", "none") else self.global_attention_mode
            encoded = encode_pair_with_answer_preservation(
                context=record.context,
                answer=record.answer,
                tokenizer=self.tokenizer,
                max_length=self.max_length,
                truncation_strategy=strat,
                global_attention_mode=att_mode,
            )

        item = {key: torch.tensor(val, dtype=torch.long) for key, val in encoded.items()}
        item["labels"] = torch.tensor(int(record.hallucinated), dtype=torch.long)
        return item
