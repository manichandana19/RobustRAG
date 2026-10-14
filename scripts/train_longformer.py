from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"

sys.path.insert(0, str(SRC_DIR))

from hallucination_detector.training.longformer_train import (  # noqa: E402
    train_longformer,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Longformer hallucination detector.")
    parser.add_argument(
        "--config",
        type=str,
        default=str(PROJECT_ROOT / "configs" / "longformer_1024_head.yaml"),
        help="Path to YAML configuration file.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(PROJECT_ROOT / "reports" / "experiments" / "longformer_1024_head"),
        help="Output directory to save model and evaluation artifacts.",
    )
    parser.add_argument(
        "--data",
        type=str,
        default=str(PROJECT_ROOT / "data" / "processed" / "ragtruth.jsonl"),
        help="Path to processed JSONL dataset.",
    )

    args = parser.parse_args()

    config_path = Path(args.config)
    output_dir = Path(args.output)
    processed_path = Path(args.data)

    metrics = train_longformer(
        processed_path=processed_path,
        output_dir=output_dir,
        config_path=config_path,
    )

    print("\nFinal Longformer test metrics:")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
