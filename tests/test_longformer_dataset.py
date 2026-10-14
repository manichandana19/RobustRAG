import pytest
import torch
from transformers import AutoTokenizer

from hallucination_detector.data.schema import RAGTruthRecord
from hallucination_detector.data.torch_dataset import (
    encode_pair_with_answer_preservation,
    RAGTruthTransformerDataset,
    LongformerDataCollator,
)


@pytest.fixture(scope="module")
def tokenizer():
    return AutoTokenizer.from_pretrained("allenai/longformer-base-4096")


def test_answer_preservation_on_long_context(tokenizer):
    # Context with ~1500 tokens
    long_context = "This is a long background context sentence about restaurants. " * 150
    answer = "The restaurant is open on Monday and serves vegan sushi."

    encoded = encode_pair_with_answer_preservation(
        context=long_context,
        answer=answer,
        tokenizer=tokenizer,
        max_length=1024,
        truncation_strategy="head",
    )

    input_ids = encoded["input_ids"]
    assert len(input_ids) <= 1024

    # Extract answer tokens directly
    ans_tokens = tokenizer.encode(answer, add_special_tokens=False)
    # The trailing tokens should be [sep_id, sep_id] + ans_tokens + [sep_id]
    expected_tail = [tokenizer.sep_token_id, tokenizer.sep_token_id] + ans_tokens + [tokenizer.sep_token_id]
    assert input_ids[-len(expected_tail):] == expected_tail


def test_head_vs_tail_truncation(tokenizer):
    context = "FIRST_MARKER " + ("middle text " * 300) + " LAST_MARKER"
    answer = "Simple answer."

    # Head truncation should keep LAST_MARKER and drop FIRST_MARKER
    encoded_head = encode_pair_with_answer_preservation(
        context=context,
        answer=answer,
        tokenizer=tokenizer,
        max_length=128,
        truncation_strategy="head",
    )
    decoded_head = tokenizer.decode(encoded_head["input_ids"])
    assert "LAST_MARKER" in decoded_head
    assert "FIRST_MARKER" not in decoded_head

    # Tail truncation should keep FIRST_MARKER and drop LAST_MARKER
    encoded_tail = encode_pair_with_answer_preservation(
        context=context,
        answer=answer,
        tokenizer=tokenizer,
        max_length=128,
        truncation_strategy="tail",
    )
    decoded_tail = tokenizer.decode(encoded_tail["input_ids"])
    assert "FIRST_MARKER" in decoded_tail
    assert "LAST_MARKER" not in decoded_tail


def test_global_attention_modes(tokenizer):
    context = "Short context."
    answer = "Short answer."

    # Mode: cls
    enc_cls = encode_pair_with_answer_preservation(
        context=context,
        answer=answer,
        tokenizer=tokenizer,
        max_length=512,
        global_attention_mode="cls",
    )
    g_mask_cls = enc_cls["global_attention_mask"]
    assert g_mask_cls[0] == 1
    assert sum(g_mask_cls) == 1

    # Mode: cls_and_answer
    enc_ans = encode_pair_with_answer_preservation(
        context=context,
        answer=answer,
        tokenizer=tokenizer,
        max_length=512,
        global_attention_mode="cls_and_answer",
    )
    g_mask_ans = enc_ans["global_attention_mask"]
    ans_len = len(tokenizer.encode(answer, add_special_tokens=False))
    assert g_mask_ans[0] == 1
    assert sum(g_mask_ans) == 1 + ans_len


def test_longformer_data_collator(tokenizer):
    collator = LongformerDataCollator(tokenizer=tokenizer, pad_to_multiple_of=512)

    features = [
        {
            "input_ids": [0, 10, 20, 30, 2],
            "attention_mask": [1, 1, 1, 1, 1],
            "global_attention_mask": [1, 0, 0, 0, 0],
            "labels": 1,
        },
        {
            "input_ids": [0, 40, 2],
            "attention_mask": [1, 1, 1],
            "global_attention_mask": [1, 0, 0],
            "labels": 0,
        },
    ]

    batch = collator(features)
    assert batch["input_ids"].shape == (2, 512)
    assert batch["attention_mask"].shape == (2, 512)
    assert batch["global_attention_mask"].shape == (2, 512)
    assert batch["labels"].tolist() == [1, 0]

    # Verify pad token id used in input_ids
    assert batch["input_ids"][1, 3].item() == tokenizer.pad_token_id
    # Verify 0 used for pad in masks
    assert batch["attention_mask"][1, 3].item() == 0
    assert batch["global_attention_mask"][1, 3].item() == 0


def test_dataset_item_types(tokenizer):
    record = RAGTruthRecord(
        id="test_1",
        source_id="src_1",
        question="What are the hours?",
        context={"name": "Cafe", "hours": "9am-5pm"},
        answer="The cafe is open 9am to 5pm.",
        hallucinated=False,
        model="gpt-4",
        temperature=0.7,
        split="test",
        quality=None,
        annotations=[],
    )
    dataset = RAGTruthTransformerDataset(
        records=[record],
        tokenizer=tokenizer,
        max_length=512,
        truncation_strategy="head",
    )
    item = dataset[0]

    assert isinstance(item["input_ids"], torch.Tensor)
    assert isinstance(item["attention_mask"], torch.Tensor)
    assert isinstance(item["global_attention_mask"], torch.Tensor)
    assert isinstance(item["labels"], torch.Tensor)
    assert item["labels"].item() == 0
