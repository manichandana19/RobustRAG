"""Evaluate the pinned optimized BiLSTM on RAGTruth validation only."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]

from hallucination_detector.evaluation.metrics import evaluate_binary_classifier  # noqa: E402
from hallucination_detector.inference.model_registry import MODEL_CATALOG, ModelRegistry  # noqa: E402
from scripts.evaluate_validation_thresholds import read_validation_split, sha256  # noqa: E402


def evaluate(archive: Path, output_dir: Path, device: str = "cpu", batch_size: int = 64) -> dict:
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite validation output directory: {output_dir}")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    records, tasks = read_validation_split(archive)
    registry = ModelRegistry()
    adapter = registry.load_model("bilstm-optimized", device=device)
    probabilities = adapter.predict_proba(records, batch_size=batch_size, device=device)
    if probabilities.shape != (len(records), 2):
        raise ValueError(f"Expected {(len(records), 2)} probabilities, received {probabilities.shape}.")
    if (not np.isfinite(probabilities).all() or (probabilities < 0).any()
            or (probabilities > 1).any() or not np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-5)):
        raise ValueError("Optimized BiLSTM returned non-finite or non-normalized probabilities.")

    parity_count = min(8, len(records))
    parity_deltas = []
    for index, record in enumerate(records[:parity_count]):
        result = registry.predict("bilstm-optimized", record.question, record.context, record.answer, device=device)
        single = np.asarray([result.supported_probability, result.hallucination_probability], dtype=float)
        if not np.allclose(probabilities[index], single, atol=1e-6, rtol=1e-6):
            raise RuntimeError(f"Batch/single parity failed at validation row {index} ({record.id}).")
        parity_deltas.append(float(np.max(np.abs(probabilities[index] - single))))

    positive_scores = probabilities[:, 1]
    predictions = (positive_scores >= 0.5).astype(int)
    metrics = evaluate_binary_classifier(records, predictions, probabilities, decision_threshold=0.5)
    task_metrics = {}
    for task in sorted(set(tasks)):
        indices = np.asarray([value == task for value in tasks])
        task_records = [row for row, include in zip(records, indices) if include]
        task_metrics[task] = evaluate_binary_classifier(
            task_records, predictions[indices], probabilities[indices], decision_threshold=0.5
        )

    metadata = adapter.artifact_metadata
    report = {
        "evaluation_type": "separate RAGTruth validation evaluation; not part of the historical benchmark reports",
        "split": "validation",
        "dataset": "data.zip:data/processed/validation.jsonl",
        "dataset_archive_sha256": sha256(archive),
        "record_count": len(records),
        "model_key": "bilstm-optimized",
        "model_name": MODEL_CATALOG["bilstm-optimized"].display_name,
        "repo_id": metadata["repo_id"],
        "artifact_revision": metadata["revision"],
        "artifact_hashes": metadata["artifact_hashes"],
        "artifact_provenance_status": metadata["provenance_status"],
        "input_mode": metadata["input_mode"],
        "input_order": metadata["input_order"],
        "preprocessing": {key: metadata[key] for key in ("tokenization", "section_marker_tokenization", "max_length", "truncation_strategy", "truncation_behavior")},
        "label_mapping": metadata["label_mapping"],
        "device": str(device),
        "batch_size": batch_size,
        "threshold": 0.5,
        "score_calibrated": False,
        "batch_single_parity": {"records_checked": parity_count, "max_abs_probability_delta": max(parity_deltas, default=0.0)},
        "metrics": metrics,
        "per_task_metrics": task_metrics,
        "limitations": [
            "The public repository contains no training source, training split hashes, or explicit label map; preprocessing and class order are corroborated by the bundled configs/vocabulary/metrics and matching repository training/evaluation implementation.",
            "The project's context_tail label is counterintuitive: the formatter retains the context prefix and drops excess trailing context tokens.",
            "Validation results are for this split only and do not establish real-world reliability or superiority over other models.",
            "The Hub metrics.json is a separate historical test report and was not used in this validation evaluation.",
        ],
    }

    output_dir.mkdir(parents=True)
    csv_path = output_dir / "optimized_bilstm_validation_predictions.csv"
    fields = ["record_id", "source_id", "task_type", "true_label", "predicted_label",
              "supported_probability", "hallucination_probability", "threshold", "correct"]
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for record, task, probs, pred in zip(records, tasks, probabilities, predictions):
            truth = int(record.hallucinated)
            writer.writerow({"record_id": record.id, "source_id": record.source_id, "task_type": task,
                "true_label": "HALLUCINATED" if truth else "SUPPORTED",
                "predicted_label": "HALLUCINATED" if pred else "SUPPORTED",
                "supported_probability": f"{probs[0]:.10f}", "hallucination_probability": f"{probs[1]:.10f}",
                "threshold": "0.5", "correct": int(pred == truth)})
    (output_dir / "optimized_bilstm_validation_metrics.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    metrics_dir = report["metrics"]
    confusion = metrics_dir["confusion_matrix"]
    md = ["# Optimized BiLSTM validation evaluation", "",
          "Separate validation-only report. This result is not merged into the historical benchmark and is not a general reliability claim.", "",
          f"- Rows: {report['record_count']}; split: `{report['split']}`; archive SHA-256: `{report['dataset_archive_sha256']}`",
          f"- Hub revision: `{report['artifact_revision']}`; input: `{report['input_mode']}`; threshold: `0.50` (uncalibrated)",
          f"- Batch/single parity: {parity_count} rows; max absolute probability delta `{report['batch_single_parity']['max_abs_probability_delta']:.3g}`.", "",
          "- Preprocessing note: bracketed section markers split into punctuation and words; the existing `context_tail` implementation retains the context prefix and drops excess trailing context.", "",
          "| Precision | Recall | F1 | ROC AUC | TN | FP | FN | TP |", "|---:|---:|---:|---:|---:|---:|---:|---:|",
          f"| {metrics_dir['precision']:.4f} | {metrics_dir['recall']:.4f} | {metrics_dir['f1']:.4f} | {metrics_dir['roc_auc']:.4f} | {confusion[0][0]} | {confusion[0][1]} | {confusion[1][0]} | {confusion[1][1]} |", "",
          "Positive class: HALLUCINATED. Confusion counts use rows=true and columns=predicted, ordered SUPPORTED then HALLUCINATED.", "",
          "## Per-task validation results", "",
          "| Task | N | Precision | Recall | F1 | ROC AUC | TN | FP | FN | TP |",
          "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for task, task_data in report["per_task_metrics"].items():
        task_confusion = task_data["confusion_matrix"]
        auc = "n/a" if task_data["roc_auc"] is None else f"{task_data['roc_auc']:.4f}"
        md.append(f"| {task} | {task_data['test_records']} | {task_data['precision']:.4f} | {task_data['recall']:.4f} | {task_data['f1']:.4f} | {auc} | {task_confusion[0][0]} | {task_confusion[0][1]} | {task_confusion[1][0]} | {task_confusion[1][1]} |")
    md.extend(["", "Scores are uncalibrated outputs. The Hub `metrics.json` describes a separate historical test report and was not used here.", ""])
    (output_dir / "optimized_bilstm_validation_report.md").write_text("\n".join(md), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=ROOT / "data.zip")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()
    report = evaluate(args.archive, args.output_dir, args.device, args.batch_size)
    metrics = report["metrics"]
    c = metrics["confusion_matrix"]
    print(json.dumps({"n": report["record_count"], "precision": metrics["precision"],
        "recall": metrics["recall"], "f1": metrics["f1"], "roc_auc": metrics["roc_auc"],
        "confusion_matrix_supported_hallucinated": c}, indent=2))
    print(f"PASS: wrote separate validation report to {args.output_dir}")


if __name__ == "__main__":
    main()
