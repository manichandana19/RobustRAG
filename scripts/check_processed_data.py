import argparse
from collections import Counter
from pathlib import Path

from hallucination_detector.data.loader import load_jsonl


REQUIRED_FIELDS = {
    "id",
    "source_id",
    "question",
    "context",
    "answer",
    "hallucinated",
    "model",
    "temperature",
    "split",
    "quality",
    "annotations",
}


ANNOTATION_FIELDS = {
    "start",
    "end",
    "text",
    "meta",
    "label_type",
    "implicit_true",
    "due_to_null",
}


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/processed/ragtruth.jsonl"),
    )

    args = parser.parse_args()

    records = load_jsonl(args.input)

    assert records, "Processed dataset is empty."

    print(f"Records: {len(records)}")

    fields = set(records[0].keys())

    assert REQUIRED_FIELDS <= fields, (
        f"Missing fields: {REQUIRED_FIELDS - fields}"
    )

    splits = Counter(
        record["split"]
        for record in records
    )

    labels = Counter(
        "hallucinated"
        if record["hallucinated"]
        else "supported"
        for record in records
    )

    print("\nSplits:")
    for split, count in sorted(splits.items()):
        print(f"  {split}: {count}")

    print("\nLabels:")
    for label, count in sorted(labels.items()):
        print(f"  {label}: {count}")

    assert set(splits) == {
        "train",
        "validation",
        "test",
    }

    assert sum(splits.values()) == len(records)

    # Every annotation must retain the complete original schema.
    annotation_count = 0

    for record in records:
        assert isinstance(
            record["annotations"],
            list,
        )

        if record["hallucinated"]:
            assert record["annotations"], (
                f"Hallucinated record {record['id']} "
                "has no annotations."
            )

        for annotation in record["annotations"]:
            assert ANNOTATION_FIELDS <= set(
                annotation.keys()
            ), (
                f"Annotation in record {record['id']} "
                f"is missing fields: "
                f"{ANNOTATION_FIELDS - set(annotation.keys())}"
            )

            annotation_count += 1

    print(f"\nAnnotations preserved: {annotation_count}")

    # Source-level leakage check.
    source_splits: dict[str, set[str]] = {}

    for record in records:
        source_splits.setdefault(
            record["source_id"],
            set(),
        ).add(record["split"])

    leaked_sources = {
        source_id: splits
        for source_id, splits in source_splits.items()
        if len(splits) > 1
    }

    assert not leaked_sources, (
        f"Source leakage detected: {list(leaked_sources)[:10]}"
    )

    print("Source-level split leakage: none")

    print("\nAll processed-data checks passed.")


if __name__ == "__main__":
    main()