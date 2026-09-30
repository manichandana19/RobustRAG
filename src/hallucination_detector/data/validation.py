from numpy import record
from collections import Counter
from dataclasses import dataclass
from typing import Any

from hallucination_detector.data.schema import RAGTruthRecord


@dataclass
class DatasetAudit:
    total_records: int
    split_counts: dict[str, int]
    label_counts: dict[str, int]
    source_count: int
    duplicate_ids: list[str]
    duplicate_pairs: int
    source_split_overlap: dict[str, set[str]]
    empty_fields: dict[str, int]
    annotation_count: int
    annotation_label_counts: dict[str, int]
    model_counts: dict[str, int]
    quality_counts: dict[str, int]

def _get_field(record, field_name: str) -> str:
    """Safely extracts a field as a string from a dict or object."""
    if hasattr(record, field_name):
        val = getattr(record, field_name)
    elif isinstance(record, dict):
        val = record.get(field_name, "")
    else:
        val = ""
    return str(val) if val is not None else ""

def audit_records(records: list[RAGTruthRecord]) -> DatasetAudit:
    ids = [record.id for record in records]

    duplicate_counter = Counter(ids)
    duplicate_ids = sorted(
        record_id
        for record_id, count in duplicate_counter.items()
        if count > 1
    )

    duplicate_pairs = sum(
        count * (count - 1) // 2
        for count in duplicate_counter.values()
        if count > 1
    )

    split_counts = Counter(record.split for record in records)

    label_counts = Counter(
        "hallucinated" if record.hallucinated else "supported"
        for record in records
    )

    source_ids = {
        record.source_id
        for record in records
    }

    source_split_map: dict[str, set[str]] = {}

    for record in records:
        source_split_map.setdefault(record.source_id, set()).add(
            record.split
        )

    source_split_overlap = {
        source_id: splits
        for source_id, splits in source_split_map.items()
        if len(splits) > 1
    }

    empty_fields = {
        "question": sum(not _get_field(record, "question").strip() for record in records),
        "context": sum(not _get_field(record, "context").strip() for record in records),
        "answer": sum(not _get_field(record, "answer").strip() for record in records),
    }


    annotation_count = 0
    annotation_label_counts: Counter[str] = Counter()

    for record in records:
        annotation_count += len(record.annotations)

        for annotation in record.annotations:
            if not isinstance(annotation, dict):
                continue

            label = annotation.get("label")

            if label is not None:
                annotation_label_counts[str(label)] += 1

    model_counts = Counter(
        record.model or "<missing>"
        for record in records
    )

    quality_counts = Counter(
        record.quality or "<missing>"
        for record in records
    )

    return DatasetAudit(
        total_records=len(records),
        split_counts=dict(split_counts),
        label_counts=dict(label_counts),
        source_count=len(source_ids),
        duplicate_ids=duplicate_ids,
        duplicate_pairs=duplicate_pairs,
        source_split_overlap=source_split_overlap,
        empty_fields=empty_fields,
        annotation_count=annotation_count,
        annotation_label_counts=dict(annotation_label_counts),
        model_counts=dict(model_counts),
        quality_counts=dict(quality_counts),
    )


def validate_records(records: list[RAGTruthRecord]) -> None:
    if not records:
        raise ValueError("Dataset contains no records.")

    ids = [record.id for record in records]

    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate response IDs detected.")

    for record in records:
        if not record.id:
            raise ValueError("Record contains empty id.")

        if not record.source_id:
            raise ValueError(
                f"Record {record.id} contains empty source_id."
            )

        if not record.question.strip():
            raise ValueError(
                f"Record {record.id} contains empty question."
            )
        # Extract the context safely, whether 'record' is a Pydantic object/class or a dict
        if hasattr(record, 'context'):
            context_val = record.context
        elif isinstance(record, dict):
            context_val = record.get('context', '')
        else:
            context_val = ""

        # Change this:
        # if not record.context.strip():

        # To this:
        if not str(context_val).strip():
            raise ValueError(
                f"Record {record.id} contains empty context."
            )

        if not record.answer.strip():
            raise ValueError(
                f"Record {record.id} contains empty answer."
            )

        if record.split not in {"train", "validation", "test"}:
            raise ValueError(
                f"Unexpected split '{record.split}' "
                f"for record {record.id}."
            )

        if not isinstance(record.annotations, list):
            raise ValueError(
                f"Annotations must be a list for record {record.id}."
            )

        for annotation in record.annotations:
            if not isinstance(annotation, dict):
                raise ValueError(
                    f"Invalid annotation in record {record.id}."
                )