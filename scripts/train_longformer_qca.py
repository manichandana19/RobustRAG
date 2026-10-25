"""Train a new Longformer QCA candidate using train and validation data only."""

from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hallucination_detector.training.longformer_qca_pipeline import main  # noqa: E402


if __name__ == "__main__":
    main()
