from hallucination_detector.inference.verdict import classify, load_thresholds

T = {
    "m": {
        "QA": {"reliable": True, "uncertain_from": 0.30, "likely_from": 0.60, "note": "n"},
        "Summarization": {"reliable": False, "note": "no skill"},
    }
}


def test_qa_bands():
    assert classify(0.067, "m", "QA", T).level == "ok"
    assert classify(0.342, "m", "QA", T).level == "warn"
    assert classify(0.60, "m", "QA", T).level == "bad"


def test_unreliable_task_gives_no_verdict():
    v = classify(0.9, "m", "Summarization", T)
    assert v.level == "info" and "NO RELIABLE" in v.label


def test_fallback_when_unknown():
    assert classify(0.6, "other", "QA", T).level == "bad"
    assert classify(0.4, "m", "Data-to-text", T).level == "ok"
    assert "uncalibrated" in classify(0.4, "m", "Data-to-text", T).label


def test_missing_file_returns_empty(tmp_path):
    assert load_thresholds(tmp_path / "nope.json") == {}
