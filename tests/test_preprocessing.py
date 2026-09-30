from hallucination_detector.data.preprocessing import (
    build_ragtruth_records,
)


def test_build_supported_record():
    responses = [
        {
            "id": "1",
            "source_id": "100",
            "model": "test-model",
            "temperature": 0.7,
            "labels": [],
            "split": "train",
            "quality": "good",
            "response": "Supported answer.",
        }
    ]

    sources = [
        {
            "source_id": "100",
            "task_type": "QA",
            "source": "test",
            "source_info": "The actual context.",
            "prompt": "What happened?",
        }
    ]

    records = build_ragtruth_records(
        responses,
        sources,
    )

    record = records[0]

    assert record.id == "1"
    assert record.source_id == "100"
    assert record.question == "What happened?"
    assert record.context == "The actual context."
    assert record.answer == "Supported answer."
    assert record.hallucinated is False
    assert record.annotations == []


def test_build_hallucinated_record_preserves_annotations():
    annotations = [
        {
            "start": 10,
            "end": 20,
            "text": "incorrect claim",
            "meta": "test metadata",
            "label_type": "Evident Conflict",
            "implicit_true": False,
            "due_to_null": False,
        }
    ]

    responses = [
        {
            "id": "1",
            "source_id": "100",
            "model": "test-model",
            "temperature": 0.7,
            "labels": annotations,
            "split": "train",
            "quality": "good",
            "response": "Hallucinated answer.",
        }
    ]

    sources = [
        {
            "source_id": "100",
            "task_type": "QA",
            "source": "test",
            "source_info": "The actual context.",
            "prompt": "What happened?",
        }
    ]

    records = build_ragtruth_records(
        responses,
        sources,
    )

    record = records[0]

    assert record.hallucinated is True
    assert record.annotations == annotations

    annotation = record.annotations[0]

    assert annotation["start"] == 10
    assert annotation["end"] == 20
    assert annotation["text"] == "incorrect claim"
    assert annotation["meta"] == "test metadata"
    assert annotation["label_type"] == "Evident Conflict"
    assert annotation["implicit_true"] is False
    assert annotation["due_to_null"] is False


def test_missing_source_raises_error():
    responses = [
        {
            "id": "1",
            "source_id": "999",
            "labels": [],
            "split": "train",
            "response": "Answer.",
        }
    ]

    sources = []

    try:
        build_ragtruth_records(
            responses,
            sources,
        )
    except ValueError as exc:
        assert "999" in str(exc)
    else:
        raise AssertionError(
            "Expected ValueError"
        )