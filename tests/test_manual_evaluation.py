import csv
import json
from pathlib import Path

import pytest

from scripts.evaluate_manual_results import load_cases, summarize


ROOT = Path(__file__).resolve().parents[1]
CASE_PATH = ROOT / "reports/manual_evaluation/robustrag_manual_set_v1.json"
FIELDS = [
    "case_id", "model", "expected_label", "predicted_label",
    "hallucination_probability_uncalibrated", "pass_fail", "notes",
]


def test_manual_set_has_unique_ids_valid_labels_and_documented_ambiguity():
    cases = load_cases(CASE_PATH)
    assert len(cases) == 18
    assert sum(case["expected_label"] is None for case in cases.values()) == 2
    assert all(case["expected_label"] in {"SUPPORTED", "HALLUCINATED", None} for case in cases.values())


def test_manual_set_rejects_duplicate_ids_and_invalid_labels(tmp_path):
    data = json.loads(CASE_PATH.read_text(encoding="utf-8"))
    data["cases"][1]["case_id"] = data["cases"][0]["case_id"]
    duplicate_path = tmp_path / "duplicate.json"
    duplicate_path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate case_id"):
        load_cases(duplicate_path)

    data = json.loads(CASE_PATH.read_text(encoding="utf-8"))
    data["cases"][0]["expected_label"] = "MAYBE"
    invalid_path = tmp_path / "invalid.json"
    invalid_path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError, match="expected_label"):
        load_cases(invalid_path)


def test_manual_result_metrics_use_hallucinated_as_positive_and_skip_ambiguous(tmp_path):
    cases = load_cases(CASE_PATH)
    supported = next(case for case in cases.values() if case["expected_label"] == "SUPPORTED")
    hallucinated = next(case for case in cases.values() if case["expected_label"] == "HALLUCINATED")
    ambiguous = next(case for case in cases.values() if case["expected_label"] is None)
    result_path = tmp_path / "results.csv"
    with result_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerow({"case_id": supported["case_id"], "model": "tfidf-lr", "expected_label": "SUPPORTED", "predicted_label": "HALLUCINATED", "hallucination_probability_uncalibrated": "0.7", "pass_fail": "FAIL", "notes": ""})
        writer.writerow({"case_id": hallucinated["case_id"], "model": "tfidf-lr", "expected_label": "HALLUCINATED", "predicted_label": "HALLUCINATED", "hallucination_probability_uncalibrated": "0.8", "pass_fail": "PASS", "notes": ""})
        writer.writerow({"case_id": ambiguous["case_id"], "model": "tfidf-lr", "expected_label": "", "predicted_label": "SUPPORTED", "hallucination_probability_uncalibrated": "0.2", "pass_fail": "", "notes": "review"})
    report = summarize(cases, result_path)["tfidf-lr"]
    assert report["n"] == 2
    assert report["accuracy"] == 0.5
    assert report["precision"] == 0.5
    assert report["recall"] == 1.0
    assert report["f1"] == pytest.approx(2 / 3)
    assert report["confusion"] == {"tn": 0, "fp": 1, "fn": 0, "tp": 1}


def test_task_aware_labels_normalize_and_nonbinary_verdicts_are_review_only(tmp_path):
    cases = load_cases(CASE_PATH)
    supported = next(case for case in cases.values() if case["expected_label"] == "SUPPORTED")
    hallucinated = next(case for case in cases.values() if case["expected_label"] == "HALLUCINATED")
    result_path = tmp_path / "task-aware.csv"
    with result_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerow({"case_id": supported["case_id"], "model": "longformer-head", "expected_label": "SUPPORTED", "predicted_label": "UNCERTAIN", "hallucination_probability_uncalibrated": "0.45", "pass_fail": "REVIEW", "notes": ""})
        writer.writerow({"case_id": hallucinated["case_id"], "model": "longformer-head", "expected_label": "HALLUCINATED", "predicted_label": "LIKELY HALLUCINATED", "hallucination_probability_uncalibrated": "0.8", "pass_fail": "PASS", "notes": ""})
    report = summarize(cases, result_path)["longformer-head"]
    assert report["n"] == 1
    assert report["accuracy"] == 1.0
    assert report["confusion"] == {"tn": 0, "fp": 0, "fn": 0, "tp": 1}

