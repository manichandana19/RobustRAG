from __future__ import annotations

import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"

sys.path.insert(0, str(SRC_DIR))

from hallucination_detector.training.deberta_baseline import (  # noqa: E402
    train_deberta,
)


def main() -> None:
    config_path = PROJECT_ROOT / "configs" / "default.yaml"

    processed_path = (
        PROJECT_ROOT
        / "data"
        / "processed"
        / "ragtruth.jsonl"
    )

    output_dir = (
        PROJECT_ROOT
        / "reports"
        / "experiments"
        / "baseline_deberta"
    )

    metrics = train_deberta(
        processed_path=processed_path,
        output_dir=output_dir,
        config_path=config_path,
    )

    print("\nFinal DeBERTa test metrics:")
    print(
        json.dumps(
            metrics,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()