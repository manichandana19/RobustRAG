"""Backward-compatible entry point for the reproducible held-out evaluator."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent / "scripts"))

from evaluate_benchmark import main


if __name__ == "__main__":
    raise SystemExit(main())
