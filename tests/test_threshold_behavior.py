import json
from zipfile import ZipFile

import numpy as np
import pytest

from hallucination_detector.data.schema import RAGTruthRecord
from hallucination_detector.inference.model_registry import (
    DEFAULT_HALLUCINATION_THRESHOLD,
    classify_hallucination_score,
)
from hallucination_detector.inference.task_verdicts import (
    DEFAULT_CONFIG,
    interpret_task_score,
    load_task_verdict_config,
)
from hallucination_detector.modeling.tfidf_logistic import build_labels
from scripts.evaluate_validation_thresholds import counts_at_threshold, read_validation_split, threshold_report


def _record(hallucinated: bool) -> RAGTruthRecord:
    return RAGTruthRecord(id="r", source_id="s", question="q", context="c", answer="a", hallucinated=hallucinated)


def test_default_threshold_boundary_and_positive_class_are_explicit():
    assert DEFAULT_HALLUCINATION_THRESHOLD == 0.5
    assert classify_hallucination_score(0.499999) == (False, "SUPPORTED")
    assert classify_hallucination_score(0.5) == (True, "HALLUCINATED")
    assert classify_hallucination_score(0.500001) == (True, "HALLUCINATED")


def test_invalid_probability_or_threshold_is_rejected():
    with pytest.raises(ValueError, match="probability"):
        classify_hallucination_score(float("nan"))
    with pytest.raises(ValueError, match="threshold"):
        classify_hallucination_score(0.5, 0)


def test_training_labels_are_supported_zero_hallucinated_one():
    assert build_labels([_record(False), _record(True)]) == [0, 1]


def test_threshold_confusion_counts_use_hallucinated_as_positive_and_ge_boundary():
    actual = np.asarray([False, False, True, True])
    scores = np.asarray([0.1, 0.5, 0.5, 0.8])
    result = counts_at_threshold(actual, scores, 0.5)
    assert result["confusion"] == {"tn": 1, "fp": 1, "fn": 0, "tp": 2}
    assert result["precision"] == pytest.approx(2 / 3)
    assert result["recall"] == 1.0
    grouped = threshold_report(actual, scores, ["QA", "QA", "Summary", "Summary"], [0.5])
    assert grouped["QA"]["0.50"]["confusion"] == {"tn": 1, "fp": 1, "fn": 0, "tp": 0}


def test_validation_reader_does_not_parse_held_out_test_member(tmp_path):
    archive = tmp_path / "splits.zip"
    validation_row = {"id": "v1", "source_id": "s1", "question": "q", "context": "c", "answer": "a", "hallucinated": True}
    with ZipFile(archive, "w") as zf:
        zf.writestr("data/processed/validation.jsonl", json.dumps(validation_row) + "\n")
        zf.writestr("data/processed/test.jsonl", "this is deliberately invalid json and must never be read")
        zf.writestr("data/raw/source_info.jsonl", json.dumps({"source_id": "s1", "task_type": "QA"}) + "\n")
    records, tasks = read_validation_split(archive)
    assert [row.id for row in records] == ["v1"]
    assert tasks == ["QA"]


def test_longformer_head_task_verdict_boundaries_and_uncalibrated_fallback():
    config = load_task_verdict_config(DEFAULT_CONFIG)
    cases = [
        ("QA", .299999, "LIKELY SUPPORTED"),
        ("QA", .30, "UNCERTAIN"),
        ("QA", .599999, "UNCERTAIN"),
        ("QA", .60, "LIKELY HALLUCINATED"),
        ("Data-to-text", .399999, "LIKELY SUPPORTED"),
        ("Data-to-text", .40, "UNCERTAIN"),
        ("Data-to-text", .699999, "UNCERTAIN"),
        ("Data-to-text", .70, "LIKELY HALLUCINATED"),
        ("Summarization", .99, "NO RELIABLE VERDICT"),
        ("Not sure", .499999, "SUPPORTED"),
        ("Not sure", .50, "HALLUCINATED"),
    ]
    for task, score, expected in cases:
        assert interpret_task_score("longformer-head", task, score, config).label == expected
    assert interpret_task_score("deberta", "QA", .5, config).label == "HALLUCINATED"
    assert "uncalibrated" in interpret_task_score("deberta", "QA", .5, config).threshold_description
