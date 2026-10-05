from pathlib import Path
import sys

import yaml

# Allow execution as: python scripts/train_baseline.py
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hallucination_detector.training.baseline import train_baseline


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "configs" / "default.yaml"
DATASET_PATH = (
    PROJECT_ROOT / "data" / "processed" / "ragtruth.jsonl"
)
OUTPUT_DIR = (
    PROJECT_ROOT / "reports" / "experiments" / "baseline_tfidf_logreg"
)


def main() -> None:
    with CONFIG_PATH.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    metrics = train_baseline(
        dataset_path=DATASET_PATH,
        output_dir=OUTPUT_DIR,
        config=config,
    )

    print("Baseline training complete.")
    print(f"Output: {OUTPUT_DIR}")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"F1: {metrics['f1']:.4f}")
    print(f"ROC-AUC: {metrics['roc_auc']:.4f}")
    print(
        "Average Precision: "
        f"{metrics['average_precision']:.4f}"
    )
    print("Confusion matrix:")
    for row in metrics["confusion_matrix"]:
        print(row)


if __name__ == "__main__":
    main()
