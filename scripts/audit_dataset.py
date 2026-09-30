import argparse
from pathlib import Path

from hallucination_detector.data.loader import load_jsonl
from hallucination_detector.data.preprocessing import build_ragtruth_records
from hallucination_detector.data.validation import (
    audit_records,
    validate_records,
)


def print_section(title: str) -> None:
    print(f"\n{'=' * 70}")
    print(title)
    print("=" * 70)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit the processed RAGTruth dataset."
    )

    parser.add_argument(
        "--responses",
        type=Path,
        default=Path("data/raw/response.jsonl"),
    )

    parser.add_argument(
        "--sources",
        type=Path,
        default=Path("data/raw/source_info.jsonl"),
    )

    args = parser.parse_args()

    responses = load_jsonl(args.responses)
    sources = load_jsonl(args.sources)

    records = build_ragtruth_records(
        responses=responses,
        sources=sources,
    )

    validate_records(records)

    audit = audit_records(records)

    print_section("Dataset")

    print(f"Total records: {audit.total_records}")
    print(f"Unique sources: {audit.source_count}")

    print_section("Splits")

    for split, count in sorted(audit.split_counts.items()):
        print(f"{split}: {count}")

    print_section("Labels")

    for label, count in sorted(audit.label_counts.items()):
        percentage = count / audit.total_records * 100
        print(f"{label}: {count} ({percentage:.2f}%)")

    print_section("Duplicates")

    print(f"Duplicate IDs: {len(audit.duplicate_ids)}")
    print(f"Duplicate pairs: {audit.duplicate_pairs}")

    print_section("Empty Fields")

    for field, count in audit.empty_fields.items():
        print(f"{field}: {count}")

    print_section("Annotations")

    print(f"Total annotations: {audit.annotation_count}")

    if audit.annotation_label_counts:
        print("Annotation labels:")

        for label, count in sorted(
            audit.annotation_label_counts.items()
        ):
            print(f"  {label}: {count}")
    else:
        print("No annotation labels found.")

    print_section("Models")

    for model, count in sorted(audit.model_counts.items()):
        print(f"{model}: {count}")

    print_section("Quality")

    for quality, count in sorted(audit.quality_counts.items()):
        print(f"{quality}: {count}")

    print_section("Source/Split Leakage")

    if audit.source_split_overlap:
        print(
            f"Sources appearing in multiple splits: "
            f"{len(audit.source_split_overlap)}"
        )

        for source_id, splits in list(
            audit.source_split_overlap.items()
        )[:20]:
            print(f"  source_id={source_id}: {sorted(splits)}")

        if len(audit.source_split_overlap) > 20:
            print("  ...")
    else:
        print("No source IDs appear across multiple splits.")

    print_section("Validation")

    print("All validation checks passed.")


if __name__ == "__main__":
    main()