import argparse
import json
from collections import Counter
from pathlib import Path

from hallucination_detector.data.loader import load_jsonl


def print_record_structure(record: dict, title: str) -> None:
    print(f"\n{'=' * 70}")
    print(title)
    print("=" * 70)

    print("Top-level fields:")
    for key, value in record.items():
        print(f"  {key}: {type(value).__name__}")

    print("\nFull record:")
    print(json.dumps(record, indent=2, ensure_ascii=False))


def inspect_file(path: Path, show_sample: bool) -> None:
    records = load_jsonl(path)

    print(f"\nFile: {path}")
    print(f"Records: {len(records)}")

    if not records:
        print("WARNING: file contains no records")
        return

    field_counter = Counter()

    for record in records:
        field_counter.update(record.keys())

    print("\nFields:")
    for field, count in field_counter.most_common():
        print(f"  {field}: {count}/{len(records)} records")

    if show_sample:
        print_record_structure(records[0], "First record")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Inspect a RAGTruth JSONL file."
    )

    parser.add_argument(
        "path",
        type=Path,
        help="Path to JSONL file",
    )

    parser.add_argument(
        "--sample",
        action="store_true",
        help="Print the first complete record",
    )

    args = parser.parse_args()

    inspect_file(args.path, args.sample)


if __name__ == "__main__":
    main()