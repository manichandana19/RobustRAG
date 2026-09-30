import argparse
from pathlib import Path

from hallucination_detector.evaluation.eda import run_eda


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run exploratory data analysis on RAGTruth."
    )

    parser.add_argument(
        "--processed",
        type=Path,
        default=Path(
            "data/processed/ragtruth.jsonl"
        ),
    )

    parser.add_argument(
        "--sources",
        type=Path,
        default=Path(
            "data/raw/source_info.jsonl"
        ),
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/eda"),
    )

    args = parser.parse_args()

    run_eda(
        processed_path=args.processed,
        source_path=args.sources,
        output_dir=args.output,
    )


if __name__ == "__main__":
    main()