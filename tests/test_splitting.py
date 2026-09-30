from hallucination_detector.data.schema import RAGTruthRecord
from hallucination_detector.data.splitting import (
    source_grouped_train_validation_split,
)


def make_record(
    record_id: str,
    source_id: str,
) -> RAGTruthRecord:
    return RAGTruthRecord(
        id=record_id,
        source_id=source_id,
        question="Question",
        context="Context",
        answer="Answer",
        hallucinated=False,
        model="test-model",
        temperature=0.7,
        split="train",
        quality="good",
        annotations=[],
    )


def test_source_grouped_split_keeps_sources_together():
    records = [
        make_record("1", "source-1"),
        make_record("2", "source-1"),
        make_record("3", "source-2"),
        make_record("4", "source-2"),
        make_record("5", "source-3"),
        make_record("6", "source-3"),
        make_record("7", "source-4"),
        make_record("8", "source-4"),
        make_record("9", "source-5"),
        make_record("10", "source-5"),
    ]

    train, validation = source_grouped_train_validation_split(
        records,
        validation_ratio=0.2,
        seed=42,
    )

    train_sources = {record.source_id for record in train}
    validation_sources = {
        record.source_id
        for record in validation
    }

    assert train_sources.isdisjoint(validation_sources)

    assert all(
        record.split == "train"
        for record in train
    )

    assert all(
        record.split == "validation"
        for record in validation
    )


def test_split_is_reproducible():
    records = [
        make_record(str(i), f"source-{i // 2}")
        for i in range(20)
    ]

    train_a, validation_a = (
        source_grouped_train_validation_split(
            records,
            validation_ratio=0.2,
            seed=42,
        )
    )

    train_b, validation_b = (
        source_grouped_train_validation_split(
            records,
            validation_ratio=0.2,
            seed=42,
        )
    )

    assert [
        record.id for record in train_a
    ] == [
        record.id for record in train_b
    ]

    assert [
        record.id for record in validation_a
    ] == [
        record.id for record in validation_b
    ]