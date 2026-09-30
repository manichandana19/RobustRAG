from collections import defaultdict
from dataclasses import replace
from random import Random

from hallucination_detector.data.schema import RAGTruthRecord


def source_grouped_train_validation_split(
    records: list[RAGTruthRecord],
    validation_ratio: float = 0.1,
    seed: int = 42,
) -> tuple[list[RAGTruthRecord], list[RAGTruthRecord]]:
    """
    Split the existing training records by source_id.

    All responses belonging to the same source remain in the same split.
    """

    if not 0 < validation_ratio < 1:
        raise ValueError(
            "validation_ratio must be between 0 and 1."
        )

    train_records = [
        record
        for record in records
        if record.split == "train"
    ]

    if not train_records:
        raise ValueError("No training records found.")

    records_by_source: dict[
        str,
        list[RAGTruthRecord],
    ] = defaultdict(list)

    for record in train_records:
        records_by_source[record.source_id].append(record)

    source_ids = list(records_by_source)

    rng = Random(seed)
    rng.shuffle(source_ids)

    validation_source_count = max(
        1,
        round(len(source_ids) * validation_ratio),
    )

    validation_sources = set(
        source_ids[:validation_source_count]
    )

    new_train: list[RAGTruthRecord] = []
    validation: list[RAGTruthRecord] = []

    for source_id, source_records in records_by_source.items():
        if source_id in validation_sources:
            validation.extend(
                replace(record, split="validation")
                for record in source_records
            )
        else:
            new_train.extend(source_records)

    return new_train, validation