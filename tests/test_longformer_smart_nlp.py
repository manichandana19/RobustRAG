from __future__ import annotations

import pytest
import torch
from transformers import AutoTokenizer

from hallucination_detector.data.schema import RAGTruthRecord
from hallucination_detector.data.sentence_selector import (
    split_into_sentences,
    extract_keywords,
    extract_numbers_and_dates,
    extract_negation_markers,
    SentenceSelector,
)
from hallucination_detector.data.longformer_smart_dataset import (
    encode_qca_for_longformer,
    RAGTruthSmartLongformerDataset,
)
from hallucination_detector.data.torch_dataset import LongformerDataCollator


@pytest.fixture(scope="module")
def tokenizer():
    # Reuse locally cached tokenizer or standard checkpoint
    return AutoTokenizer.from_pretrained("reports/experiments/longformer_1024_head/best_model")


@pytest.fixture
def sample_records() -> list[RAGTruthRecord]:
    return [
        RAGTruthRecord(
            id="rec_1",
            source_id="src_1",
            question="What is the opening date of the museum?",
            context="The National Museum was established in 1895. It was opened to the public on June 15, 1900. Visitors can see historical art.",
            answer="The museum opened on June 15, 1900.",
            hallucinated=False,
            model="gpt-4",
            temperature=0.7,
            split="train",
            quality=None,
            annotations=[],
        ),
        RAGTruthRecord(
            id="rec_2",
            source_id="src_2",
            question="How many people survived the flight?",
            context="The flight carried 150 passengers and crew. Unfortunately, only 12 people survived the crash. 138 passengers perished.",
            answer="Over 50 passengers survived the crash.",
            hallucinated=True,
            model="gpt-4",
            temperature=0.7,
            split="train",
            quality=None,
            annotations=[],
        ),
        RAGTruthRecord(
            id="rec_3",
            source_id="src_3",
            question="Is the library open on weekends?",
            context="The Central Library is open from Monday through Friday. It is not open on Saturday or Sunday under any circumstances.",
            answer="The library is open on Sunday afternoon.",
            hallucinated=True,
            model="gpt-4",
            temperature=0.7,
            split="test",
            quality=None,
            annotations=[],
        ),
    ]


def test_sentence_splitting_and_fallbacks():
    text = "First sentence. Second sentence! Third sentence? Fourth sentence."
    sents = split_into_sentences(text)
    assert len(sents) == 4
    assert sents[0] == "First sentence."
    assert sents[-1] == "Fourth sentence."

    # Empty text
    assert split_into_sentences("") == []
    # No punctuation fallback
    assert split_into_sentences("Unpunctuated string of words") == ["Unpunctuated string of words"]


def test_linguistic_feature_extraction():
    answer = "The treaty was signed in 1945 by no fewer than 15 countries, but never ratified."
    keywords = extract_keywords(answer)
    numbers = extract_numbers_and_dates(answer)
    negations = extract_negation_markers(answer)

    assert "treaty" in keywords
    assert "1945" in numbers
    assert "15" in numbers
    assert "no" in negations
    assert "never" in negations


def test_contradiction_aware_sentence_selection(sample_records):
    selector = SentenceSelector.fit_from_records(sample_records)

    # Context with a clear negation contradiction
    context = (
        "The Central Library is open from Monday through Friday. "
        "It is not open on Saturday or Sunday under any circumstances. "
        "The building was designed by a famous architect."
    )
    question = "Is the library open on weekends?"
    answer = "The library is open on Sunday afternoon."

    selected = selector.select_sentences(
        question=question,
        context=context,
        answer=answer,
        max_sentences=1,
    )
    # The contradiction sentence should be prioritized due to shared topic and negation divergence
    assert len(selected) == 1
    assert "not open on Saturday or Sunday" in selected[0][1]


def test_selected_sentences_preserve_chronological_order(sample_records):
    selector = SentenceSelector.fit_from_records(sample_records)
    context = "Sentence 0 about cats. Sentence 1 about dogs. Sentence 2 about birds. Sentence 3 about fish."
    question = "Tell me about fish and cats"
    answer = "Fish swim and cats purr."

    selected = selector.select_sentences(
        question=question,
        context=context,
        answer=answer,
        max_sentences=2,
    )
    # Ensure original index order: i_0 < i_1
    indices = [idx for idx, _ in selected]
    assert indices == sorted(indices)


def test_encode_qca_for_longformer_budget_and_answer_preservation(tokenizer, sample_records):
    selector = SentenceSelector.fit_from_records(sample_records)

    long_context = "This is a detailed historical record of the event. " * 200
    question = "What happened at the historical event?"
    answer = "The historic declaration was signed in 1776."

    encoded = encode_qca_for_longformer(
        question=question,
        context=long_context,
        answer=answer,
        tokenizer=tokenizer,
        max_length=256,
        truncation_strategy="smart_selector",
        selector=selector,
    )

    input_ids = encoded["input_ids"]
    assert len(input_ids) <= 256

    # Verify delimiter structure: starts with cls, ends with sep
    assert input_ids[0] == tokenizer.cls_token_id
    assert input_ids[-1] == tokenizer.sep_token_id

    # Verify Answer tokens are fully preserved at the tail
    ans_tokens = tokenizer.encode(answer, add_special_tokens=False)
    assert input_ids[-len(ans_tokens) - 1 : -1] == ans_tokens


def test_smart_longformer_dataset_and_collator(tokenizer, sample_records):
    selector = SentenceSelector.fit_from_records(sample_records)
    dataset = RAGTruthSmartLongformerDataset(
        records=sample_records,
        tokenizer=tokenizer,
        input_mode="qca",
        max_length=512,
        truncation_strategy="smart_selector",
        selector=selector,
    )

    assert len(dataset) == 3
    item_0 = dataset[0]
    assert "input_ids" in item_0
    assert "attention_mask" in item_0
    assert "global_attention_mask" in item_0
    assert "labels" in item_0
    assert item_0["labels"].item() == 0

    # Batch dynamic padding with pad_to_multiple_of=512
    collator = LongformerDataCollator(tokenizer=tokenizer, pad_to_multiple_of=512)
    batch = collator([dataset[0], dataset[1]])

    assert batch["input_ids"].shape[0] == 2
    # Longformer sliding window kernel requires multiple of 512
    assert batch["input_ids"].shape[1] % 512 == 0
    assert batch["attention_mask"].shape == batch["input_ids"].shape
    assert batch["global_attention_mask"].shape == batch["input_ids"].shape
