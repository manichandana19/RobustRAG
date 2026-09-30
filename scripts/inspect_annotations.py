import argparse
import json
from collections import Counter
from pathlib import Path

from hallucination_detector.data.loader import load_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Inspect the actual structure of RAGTruth annotations."
    )

    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/raw/response.jsonl"),
    )

    parser.add_argument(
        "--samples",
        type=int,
        default=10,
    )

    args = parser.parse_args()

    records = load_jsonl(args.input)

    annotated_records = [
        record
        for record in records
        if record.get("labels")
    ]

    print(f"Total records: {len(records)}")
    print(f"Records with annotations: {len(annotated_records)}")

    annotation_count = sum(
        len(record.get("labels", []))
        for record in records
    )

    print(f"Total annotations: {annotation_count}")

    if not annotated_records:
        print("No annotated records found.")
        return

    print("\n" + "=" * 70)
    print("ANNOTATION STRUCTURE")
    print("=" * 70)

    key_counts = Counter()

    for record in annotated_records:
        for annotation in record["labels"]:
            if isinstance(annotation, dict):
                key_counts.update(annotation.keys())
            else:
                key_counts[f"<{type(annotation).__name__}>"] += 1

    print("\nAnnotation keys:")
    for key, count in key_counts.most_common():
        print(f"  {key}: {count}")

    print("\n" + "=" * 70)
    print("ANNOTATION SAMPLES")
    print("=" * 70)

    shown = 0

    for record in annotated_records:
        for annotation in record["labels"]:
            print(
                json.dumps(
                    annotation,
                    indent=2,
                    ensure_ascii=False,
                )
            )

            shown += 1

            if shown >= args.samples:
                return


if __name__ == "__main__":
    main()