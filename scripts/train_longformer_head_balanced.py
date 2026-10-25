"""Train an isolated Longformer Head task-balanced candidate (train + validation only)."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hallucination_detector.training.longformer_head_balanced import (  # noqa: E402
    BASELINE_REVISION,
    train_candidate,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=ROOT / "data.zip")
    parser.add_argument("--init-model-dir", type=Path, required=True, help="Local cached Longformer Head snapshot; downloads are disabled.")
    parser.add_argument("--baseline-predictions", type=Path, default=ROOT / "reports/validation/longformer_head_20261025_175441/longformer_head_validation_predictions.csv")
    parser.add_argument("--baseline-analysis", type=Path, default=ROOT / "reports/validation/longformer_head_20261025_175441/longformer_head_validation_analysis.json")
    parser.add_argument("--output-dir", type=Path, required=True, help="New, unused candidate directory. Existing directories are refused.")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-train-steps", type=int, default=None, help="Optional optimizer-step cap for an isolated preflight; omit for the full configured run.")
    args = parser.parse_args()

    if args.init_model_dir.resolve().name != BASELINE_REVISION:
        parser.error(f"--init-model-dir must be the pinned cached baseline snapshot {BASELINE_REVISION}; network downloads and other checkpoints are not allowed.")
    analysis = json.loads(args.baseline_analysis.read_text(encoding="utf-8"))
    if analysis.get("split") != "validation" or analysis.get("artifact_revision") != BASELINE_REVISION:
        parser.error("Frozen baseline analysis must identify the validation split and the pinned Longformer Head revision.")
    summary = train_candidate(
        args.archive, args.init_model_dir, args.output_dir, args.baseline_predictions,
        epochs=args.epochs, batch_size=args.batch_size,
        accumulation=args.gradient_accumulation,
        learning_rate=args.learning_rate, seed=args.seed,
        max_train_steps=args.max_train_steps,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
