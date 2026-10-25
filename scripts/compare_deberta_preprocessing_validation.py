"""Compare saved DeBERTa validation scores with answer-preserving preprocessing."""
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

from hallucination_detector.inference.model_registry import ModelRegistry  # noqa: E402
from hallucination_detector.modeling.deberta_encoding import encode_deberta_input  # noqa: E402
from scripts.evaluate_benchmark import batch_probabilities  # noqa: E402
from scripts.evaluate_validation_thresholds import read_validation_split, sha256  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=ROOT / "data.zip")
    parser.add_argument("--old-predictions", type=Path, default=ROOT / "reports/validation/threshold_sweep_20261025_205000/validation_predictions.csv")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=4)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {args.output_dir}")
    records, tasks = read_validation_split(args.archive)
    with args.old_predictions.open(newline="", encoding="utf-8-sig") as stream:
        old_rows = [row for row in csv.DictReader(stream) if row["model_key"] == "deberta"]
    if len(old_rows) != len(records):
        raise ValueError(f"Saved DeBERTa validation rows ({len(old_rows)}) do not match current validation split ({len(records)}).")
    for index, (record, row) in enumerate(zip(records, old_rows)):
        if int(row["record_id"]) != int(record.id) or str(row["source_id"]) != str(record.source_id):
            raise ValueError(f"Validation row alignment mismatch at position {index}; refusing comparison.")
    registry = ModelRegistry()
    print(f"Running new DeBERTa preprocessing on {len(records)} validation records using {args.device}.", flush=True)
    probs = batch_probabilities(registry, "deberta", records, args.batch_size, args.device)
    if probs.shape != (len(records), 2) or not np.isfinite(probs).all() or not np.allclose(probs.sum(axis=1), 1, atol=2e-5):
        raise ValueError("New validation inference returned invalid probabilities.")
    old_h = np.asarray([float(row["hallucination_probability"]) for row in old_rows])
    new_h = probs[:, 1]
    y = np.asarray([record.hallucinated for record in records], dtype=bool)
    old_pred, new_pred = old_h >= 0.5, new_h >= 0.5
    wrapper = registry.load_model("deberta", device=args.device)
    tokenization = [encode_deberta_input(record.context, record.answer, wrapper["tokenizer"], 512)["truncation"] for record in records]
    truncated_context = sum(item["context_truncated"] for item in tokenization)
    truncated_answer = sum(item["answer_truncated"] for item in tokenization)
    def metrics(pred, truth=y):
        tp = int(np.sum(pred & truth)); tn = int(np.sum(~pred & ~truth))
        fp = int(np.sum(pred & ~truth)); fn = int(np.sum(~pred & truth))
        return {"accuracy": float((tp + tn) / len(truth)), "precision": float(tp / (tp + fp)) if tp + fp else 0.0,
                "recall": float(tp / (tp + fn)) if tp + fn else 0.0,
            "f1": float(2 * tp / (2 * tp + fp + fn)) if 2 * tp + fp + fn else 0.0,
                "confusion": {"tn": tn, "fp": fp, "fn": fn, "tp": tp}}
    report = {
        "evaluation_type": "validation-only preprocessing comparison; not a held-out benchmark",
        "split": "validation", "records": len(records), "archive_sha256": sha256(args.archive),
        "old_prediction_csv_sha256": sha256(args.old_predictions), "old_artifact_revision": old_rows[0]["artifact_revision"],
        "device": args.device, "batch_size": args.batch_size, "threshold": 0.5,
        "old_metrics": metrics(old_pred), "new_metrics": metrics(new_pred),
        "prediction_changes": int(np.sum(old_pred != new_pred)),
        "mean_absolute_score_change": float(np.mean(np.abs(old_h - new_h))),
        "context_truncated_records": int(truncated_context), "answer_truncated_records": int(truncated_answer),
        "answer_truncation_policy": "context tokens are removed first; answer is shortened only when the answer segment itself exceeds the 512-token budget",
        "tasks": {task: {"n": int(sum(value == task for value in tasks)),
            "old": metrics(old_pred[np.asarray(tasks) == task], y[np.asarray(tasks) == task]),
            "new": metrics(new_pred[np.asarray(tasks) == task], y[np.asarray(tasks) == task])} for task in sorted(set(tasks))},
        "limitation": "The new model scores use answer-preserving preprocessing while the frozen checkpoint was trained with right truncation. This is a preprocessing sensitivity comparison, not a retrained-model performance claim.",
    }
    args.output_dir.mkdir(parents=True)
    (args.output_dir / "deberta_preprocessing_comparison.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: report[key] for key in ("records", "old_metrics", "new_metrics", "prediction_changes", "mean_absolute_score_change", "context_truncated_records", "answer_truncated_records")}, indent=2))
    print(f"PASS: saved comparison to {args.output_dir}")


if __name__ == "__main__":
    main()
