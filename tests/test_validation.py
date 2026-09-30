from hallucination_detector.data.schema import RAGTruthRecord
from hallucination_detector.data.validation import (
    audit_records,
    validate_records,
)


def make_record(
    record_id: str,
    source_id: str,
    split: str = "train",
    hallucinated: bool = False,
    annotations: list[dict] | None = None,
) -> RAGTruthRecord:
    return RAGTruthRecord(
        id=record_id,
        source_id=source_id,
        question="Question",
        context="Context",
        answer="Answer",
        hallucinated=hallucinated,
        model="test-model",
        temperature=0.7,
        split=split,
        quality="good",
        annotations=annotations or [],
    )


def test_valid_records_pass_validation():
    records = [
        make_record("1", "source-1"),
        make_record(
            "2",
            "source-2",
            hallucinated=True,
            annotations=[{"label": "Unsupported"}],
        ),
    ]

    validate_records(records)


def test_duplicate_ids_fail_validation():
    records = [
        make_record("1", "source-1"),
        make_record("1", "source-2"),
    ]

    try:
        validate_records(records)
    except ValueError as exc:
        assert "Duplicate response IDs" in str(exc)
    else:
        raise AssertionError("Expected duplicate ID validation failure.")


def test_invalid_split_fails_validation():
    records = [
        make_record("1", "source-1", split="invalid"),
    ]

    try:
        validate_records(records)
    except ValueError as exc:
        assert "Unexpected split" in str(exc)
    else:
        raise AssertionError("Expected invalid split failure.")


def test_audit_detects_source_split_overlap():
    records = [
        make_record("1", "source-1", split="train"),
        make_record("2", "source-1", split="test"),
        make_record("3", "source-2", split="train"),
    ]

    audit = audit_records(records)

    assert audit.source_split_overlap == {
        "source-1": {"train", "test"}
    }