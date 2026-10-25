"""Read-only consistency check for a saved held-out benchmark report and CSV."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import sys

from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def verify(run_dir: Path) -> list[dict[str, object]]:
    metrics_path = run_dir / "ragtruth_test_metrics.json"
    predictions_path = run_dir / "ragtruth_test_predictions.csv"
    report = json.loads(metrics_path.read_text(encoding="utf-8"))
    if report.get("split") != "test" or report.get("decision_threshold") != 0.5:
        raise ValueError("Expected the saved held-out test report with its original 0.5 threshold.")

    grouped: dict[str, list[dict[str, str]]] = {}
    with predictions_path.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            grouped.setdefault(row["model_key"], []).append(row)

    expected_models = list(report["requested_models"])
    if set(grouped) != set(expected_models):
        raise ValueError(f"Prediction model keys differ: CSV={sorted(grouped)}, report={sorted(expected_models)}")
    record_ids = {model: {row["record_id"] for row in rows} for model, rows in grouped.items()}
    if any(len(ids) != report["record_count"] for ids in record_ids.values()):
        raise ValueError("At least one model has an unexpected prediction count or duplicate record IDs.")
    if len({frozenset(ids) for ids in record_ids.values()}) != 1:
        raise ValueError("Models were not evaluated on identical record IDs.")

    results = []
    for model in expected_models:
        rows = grouped[model]
        if report["models"][model].get("status") != "ok":
            raise ValueError(f"Saved benchmark model {model} did not complete successfully.")
        labels = [int(row["true_label"] == "HALLUCINATED") for row in rows]
        predictions = [int(row["predicted_label"] == "HALLUCINATED") for row in rows]
        probabilities = [float(row["hallucination_probability"]) for row in rows]
        supported = [float(row["supported_probability"]) for row in rows]
        if any(not math.isfinite(value) or not 0 <= value <= 1 for value in probabilities + supported):
            raise ValueError(f"Invalid probability value in saved predictions for {model}.")
        if max(abs(a + b - 1.0) for a, b in zip(probabilities, supported)) > 0.002:
            raise ValueError(f"Saved class probabilities do not sum to one for {model}.")
        if predictions != [int(value >= 0.5) for value in probabilities]:
            raise ValueError(f"Saved labels do not match the original 0.5 threshold for {model}.")
        if [row["correct"].lower() == "true" for row in rows] != [a == b for a, b in zip(labels, predictions)]:
            raise ValueError(f"Saved correctness flags disagree with labels for {model}.")

        actual = {
            "accuracy": accuracy_score(labels, predictions),
            "precision": precision_score(labels, predictions, zero_division=0),
            "recall": recall_score(labels, predictions, zero_division=0),
            "f1": f1_score(labels, predictions, zero_division=0),
            "roc_auc": roc_auc_score(labels, probabilities),
            "average_precision": average_precision_score(labels, probabilities),
            "confusion_matrix": confusion_matrix(labels, predictions, labels=[0, 1]).tolist(),
        }
        saved = report["models"][model]["metrics"]
        for name, value in actual.items():
            if name == "confusion_matrix":
                equal = value == saved.get(name)
            else:
                equal = math.isclose(float(value), float(saved.get(name)), rel_tol=0, abs_tol=1e-10)
            if not equal:
                raise ValueError(f"Saved metric {name} for {model} does not match the prediction CSV.")
        results.append({
            "model": model,
            "records": len(rows),
            "confusion_matrix": actual["confusion_matrix"],
            "accuracy": actual["accuracy"],
        })
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args()
    for result in verify(args.run_dir):
        print(
            f"PASS {result['model']}: {result['records']} predictions; "
            f"confusion_matrix={result['confusion_matrix']}; accuracy={result['accuracy']:.6f}"
        )


if __name__ == "__main__":
    main()
