import argparse
import json
from pathlib import Path

from hallucination_detector.data.loader import load_jsonl
from hallucination_detector.data.preprocessing import (
    build_ragtruth_records,
)
from hallucination_detector.data.splitting import (
    source_grouped_train_validation_split,
)
from hallucination_detector.data.validation import validate_records


def write_jsonl(path: Path, records) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(
                json.dumps(
                    record.__dict__,
                    ensure_ascii=False,
                )
                + "\n"
            )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare the final RAGTruth dataset."
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

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/processed"),
    )

    parser.add_argument(
        "--validation-ratio",
        type=float,
        default=0.1,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    args = parser.parse_args()

    responses = load_jsonl(args.responses)
    sources = load_jsonl(args.sources)

    records = build_ragtruth_records(
        responses=responses,
        sources=sources,
    )

    # Validate the canonical records before splitting.
    validate_records(records)

    original_test = [
        record
        for record in records
        if record.split == "test"
    ]

    original_train = [
        record
        for record in records
        if record.split == "train"
    ]

    train, validation = source_grouped_train_validation_split(
        original_train,
        validation_ratio=args.validation_ratio,
        seed=args.seed,
    )

    final_records = train + validation + original_test

    # Final validation after the split.
    validate_records(final_records)

    write_jsonl(
        args.output_dir / "train.jsonl",
        train,
    )

    write_jsonl(
        args.output_dir / "validation.jsonl",
        validation,
    )

    write_jsonl(
        args.output_dir / "test.jsonl",
        original_test,
    )

    # Also retain one combined processed file.
    write_jsonl(
        args.output_dir / "ragtruth.jsonl",
        final_records,
    )

    print("=" * 70)
    print("RAGTruth preprocessing complete")
    print("=" * 70)

    print(f"Raw responses: {len(responses)}")
    print(f"Raw sources: {len(sources)}")
    print()

    print(f"Train: {len(train)}")
    print(f"Validation: {len(validation)}")
    print(f"Test: {len(original_test)}")
    print(f"Total: {len(final_records)}")

    print()
    print(f"Output directory: {args.output_dir}")


if __name__ == "__main__":
    main()