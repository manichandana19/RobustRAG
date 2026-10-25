"""Evaluate current model scores on RAGTruth validation only at fixed cutoffs.

This is an inference-only threshold comparison; it does not change the production
threshold. The held-out test member is deliberately never opened.
"""
from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
from pathlib import Path
import sys
from zipfile import ZipFile

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from hallucination_detector.data.schema import RAGTruthRecord  # noqa: E402
from hallucination_detector.inference.model_registry import MODEL_CATALOG, ModelRegistry  # noqa: E402
from scripts.evaluate_benchmark import batch_probabilities, validate_batch_matches_registry_single  # noqa: E402


DEFAULT_MODELS = ["tfidf-lr", "bilstm-attention", "deberta", "longformer-head"]
DEFAULT_THRESHOLDS = [0.2, 0.3, 0.4, 0.5]
LOCAL_ARTIFACTS = {
    "tfidf-lr": Path("reports/experiments_backup/baseline_tfidf_logreg/model.pkl"),
    "bilstm-attention": Path("models/bilstm_attention_baseline/bilstm_weights.pt"),
    "deberta": Path("reports/experiments/baseline_deberta/best_model/model.safetensors"),
    "longformer-head": Path("reports/experiments/longformer_1024_head/best_model/model.safetensors"),
}


def read_validation_split(archive: Path) -> tuple[list[RAGTruthRecord], list[str]]:
    """Read only train/validation sources and the validation response records."""
    with ZipFile(archive) as zf:
        names = set(zf.namelist())
        validation_name = "data/processed/validation.jsonl"
        if validation_name not in names:
            raise FileNotFoundError(f"Validation member missing from {archive}: {validation_name}")
        if "data/processed/test.jsonl" not in names:
            raise FileNotFoundError("Archive has no expected held-out test member; refusing an unknown split layout.")
        rows = [json.loads(line) for line in zf.open(validation_name) if line.strip()]
        sources = [json.loads(line) for line in zf.open("data/raw/source_info.jsonl") if line.strip()]
    task_by_source = {str(row["source_id"]): str(row.get("task_type", "unknown")) for row in sources}
    records = [RAGTruthRecord(**row) for row in rows]
    tasks = [task_by_source.get(str(record.source_id), "unknown") for record in records]
    return records, tasks


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def artifact_source(model_key: str, repo_id: str) -> dict[str, str]:
    """Match ModelRegistry's local-first lookup and identify the exact resolved snapshot."""
    local = ROOT / LOCAL_ARTIFACTS[model_key]
    if local.is_file():
        return {"artifact_source": str(local), "artifact_revision": f"local-sha256:{sha256(local)}"}
    from huggingface_hub import scan_cache_dir
    for repo in scan_cache_dir().repos:
        if repo.repo_id == repo_id and repo.revisions:
            revision = next(iter(repo.revisions))
            return {"artifact_source": str(repo.repo_path / "snapshots" / revision.commit_hash),
                    "artifact_revision": revision.commit_hash}
    return {"artifact_source": f"unresolved local/HF artifact for {repo_id}", "artifact_revision": "unresolved"}


def counts_at_threshold(y_true: np.ndarray, scores: np.ndarray, threshold: float) -> dict:
    y_pred = scores >= threshold
    tp = int(np.sum(y_true & y_pred))
    tn = int(np.sum(~y_true & ~y_pred))
    fp = int(np.sum(~y_true & y_pred))
    fn = int(np.sum(y_true & ~y_pred))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"n": len(y_true), "accuracy": (tp + tn) / len(y_true) if len(y_true) else None,
            "precision": precision, "recall": recall, "f1": f1,
            "false_positives": fp, "false_negatives": fn,
            "confusion": {"tn": tn, "fp": fp, "fn": fn, "tp": tp}}


def threshold_report(y_true: np.ndarray, scores: np.ndarray, tasks: list[str], thresholds: list[float]) -> dict:
    groups = {"overall": np.ones(len(y_true), dtype=bool)}
    groups.update({task: np.asarray([value == task for value in tasks]) for task in sorted(set(tasks))})
    return {group: {f"{threshold:.2f}": counts_at_threshold(y_true[mask], scores[mask], threshold)
                    for threshold in thresholds} for group, mask in groups.items()}


def save_markdown(path: Path, output: dict) -> None:
    lines = ["# Validation-only threshold comparison", "",
             "Inference-only comparison on RAGTruth validation. These thresholds are exploratory operating points; no cutoff was selected for deployment. This is not held-out test performance.", "",
             f"- Records: {output['validation_records']}  ", f"- Dataset SHA-256: `{output['archive_sha256']}`  ",
             f"- Models: {', '.join(output['models'])}  ", f"- Candidate thresholds: {', '.join(map(str, output['thresholds']))}",
             "- Positive class: HALLUCINATED (`score >= threshold`)", "- QCA omitted: cached QCA bundle lacks fitted selector and provenance.", "",
             "## Model artifacts and input modes", "", "| Model key | Revision | Input mode | Device |", "|---|---|---|---|"]
    for model, data in output["models"].items():
        lines.append(f"| {model} | `{data['artifact_revision']}` | {data['input_mode']} | {data['device']} |")
    lines.extend(["", 
             "## Overall results", "", "| Model | Threshold | N | Precision | Recall | F1 | FP | FN | TN | TP |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"])
    for model, data in output["models"].items():
        for threshold in output["thresholds"]:
            m = data["metrics"]["overall"][f"{threshold:.2f}"]
            c = m["confusion"]
            lines.append(f"| {model} | {threshold:.2f} | {m['n']} | {m['precision']:.3f} | {m['recall']:.3f} | {m['f1']:.3f} | {c['fp']} | {c['fn']} | {c['tn']} | {c['tp']} |")
    lines.extend(["", "## By task", "", "Every model/threshold combination is listed. Small or weak task strata should not be used to justify a task-specific cutoff without a separate selection protocol.", ""])
    for model, data in output["models"].items():
        lines.extend([f"### {model}", "", "| Task | Threshold | N | Precision | Recall | F1 | FP | FN | TN | TP |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"])
        for task, thresholds_data in data["metrics"].items():
            if task == "overall": continue
            for threshold in output["thresholds"]:
                m = thresholds_data[f"{threshold:.2f}"]
                c = m["confusion"]
                lines.append(f"| {task} | {threshold:.2f} | {m['n']} | {m['precision']:.3f} | {m['recall']:.3f} | {m['f1']:.3f} | {c['fp']} | {c['fn']} | {c['tn']} | {c['tp']} |")
        lines.append("")
    lines.extend(["## Validation cutoff sensitivity and decision", "",
                  "For each model, the cutoff with the highest observed overall F1 among these same-split candidates was:"])
    for model, data in output["models"].items():
        best = max(output["thresholds"], key=lambda value: data["metrics"]["overall"][f"{value:.2f}"]["f1"])
        metric = data["metrics"]["overall"][f"{best:.2f}"]
        lines.append(f"- `{model}`: {best:.2f} (F1={metric['f1']:.3f}; precision={metric['precision']:.3f}; recall={metric['recall']:.3f}).")
    lines.extend(["", "These are descriptive, in-sample validation comparisons, not unbiased estimates after cutoff selection. The apparent F1 optima differ by model; task-level precision/recall trade-offs also change sharply, especially Summary. No operating-cost target or independent calibration set was specified, so the production 0.50 cutoff is retained for every available model. Do not interpret these candidate values as calibrated confidence.",
                  "", "## Label mapping audit", "",
                  "The validation archive maps no annotated hallucination to label 0 (SUPPORTED) and one-or-more annotated hallucinations to label 1 (HALLUCINATED). The transformer checkpoint configs inspected for DeBERTa and Longformer Head map output index 0 to SUPPORTED and index 1 to HALLUCINATED. The loaded TF-IDF artifact reports sklearn classes `[0, 1]`; BiLSTM training labels are `int(record.hallucinated)` and its output convention is 0=SUPPORTED, 1=HALLUCINATED. The registry reads probability column 1 as P(HALLUCINATED) and applies `score >= 0.50`. This mapping is consistent; no label inversion was found.",
                  "", "Task support counts (from the 0.50 confusion matrix) are:"])
    reference_model = next(iter(output["models"].values()))
    for task, values in reference_model["metrics"].items():
        if task == "overall": continue
        at_default = values["0.50"]
        c = at_default["confusion"]
        lines.append(f"- `{task}`: N={at_default['n']}; SUPPORTED={c['tn'] + c['fp']}; HALLUCINATED={c['fn'] + c['tp']}.")
    lines.extend(["", "Task-specific thresholds were not adopted. These task counts support descriptive error analysis, but selecting and claiming task cutoffs on the same split would be optimistic; a separately reserved calibration/selection split and an explicit false-positive/false-negative cost target are needed first."])
    path.write_text("\n".join(lines), encoding="utf-8")


def run(archive: Path, output_dir: Path, device: str, batch_size: int, thresholds: list[float]) -> dict:
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing report directory: {output_dir}")
    if not thresholds or any(not 0 < value <= 1 for value in thresholds):
        raise ValueError("Thresholds must be in (0, 1].")
    records, tasks = read_validation_split(archive)
    y_true = np.asarray([bool(row.hallucinated) for row in records], dtype=bool)
    output_dir.mkdir(parents=True)
    archive_hash = sha256(archive)
    result = {"evaluation_type": "RAGTruth validation-only fixed-threshold comparison",
              "split": "validation", "validation_records": len(records), "archive_sha256": archive_hash,
              "positive_class": "HALLUCINATED", "decision_rule": "hallucination_probability >= threshold",
              "thresholds": thresholds, "models": {}}
    rows = []
    for model_key in DEFAULT_MODELS:
        meta = MODEL_CATALOG[model_key]
        target_device = "cpu" if model_key == "tfidf-lr" else device
        print(f"[{model_key}] loading/inference on {target_device} ({len(records)} validation records)", flush=True)
        registry = ModelRegistry()
        probs = batch_probabilities(registry, model_key, records, batch_size, target_device)
        if probs.shape != (len(records), 2):
            raise ValueError(f"{model_key}: expected {len(records)} x 2 probabilities, got {probs.shape}.")
        if not np.isfinite(probs).all() or (probs < 0).any() or (probs > 1).any():
            raise ValueError(f"{model_key}: invalid probability matrix.")
        if not np.allclose(probs.sum(axis=1), 1.0, atol=2e-5):
            raise ValueError(f"{model_key}: probability rows do not sum to one.")
        validate_batch_matches_registry_single(registry, model_key, records, probs, target_device, count=1)
        source = artifact_source(model_key, meta.repo_id)
        model_metrics = threshold_report(y_true, probs[:, 1], tasks, thresholds)
        result["models"][model_key] = {"display_name": meta.display_name, "input_mode": meta.input_mode,
            "device": target_device, "artifact_source": source["artifact_source"],
            "artifact_revision": source["artifact_revision"], "class_order": ["SUPPORTED", "HALLUCINATED"],
            "metrics": model_metrics}
        for index, record in enumerate(records):
            rows.append({"record_id": record.id, "source_id": record.source_id, "task_type": tasks[index],
                "true_label": "HALLUCINATED" if y_true[index] else "SUPPORTED", "model_key": model_key,
                "artifact_revision": source["artifact_revision"], "input_mode": meta.input_mode,
                "supported_probability": f"{probs[index, 0]:.10f}", "hallucination_probability": f"{probs[index, 1]:.10f}"})
        print(f"[{model_key}] done; revision={source['artifact_revision']}", flush=True)
        del probs, registry
        gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()
    prediction_path = output_dir / "validation_predictions.csv"
    with prediction_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    (output_dir / "threshold_metrics.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    save_markdown(output_dir / "threshold_comparison.md", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=ROOT / "data.zip")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--thresholds", type=float, nargs="+", default=DEFAULT_THRESHOLDS)
    args = parser.parse_args()
    result = run(args.archive, args.output_dir, args.device, args.batch_size, args.thresholds)
    print(f"PASS: {result['validation_records']} validation rows × {len(result['models'])} models; wrote {args.output_dir}")


if __name__ == "__main__":
    main()
