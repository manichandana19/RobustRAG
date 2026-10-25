"""Evaluate the registered RobustRAG models on the archived RAGTruth test split.

This script performs inference only. It never trains models or changes thresholds.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import gc
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace
from zipfile import ZipFile

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hallucination_detector.data.schema import RAGTruthRecord  # noqa: E402
from hallucination_detector.evaluation.metrics import evaluate_binary_classifier  # noqa: E402
from hallucination_detector.inference.model_registry import MODEL_CATALOG, ModelRegistry  # noqa: E402


LOCAL_ARTIFACTS = {
    "tfidf-lr": Path("reports/experiments_backup/baseline_tfidf_logreg/model.pkl"),
    "bilstm-attention": Path("models/bilstm_attention_baseline"),
    "deberta": Path("reports/experiments/baseline_deberta/best_model"),
    "longformer-head": Path("reports/experiments/longformer_1024_head/best_model"),
    "longformer-qca": Path("reports/experiments/longformer_1024_smart_qca/best_model"),
}

TRANSFORMER_MODEL_KEYS = {"deberta", "longformer-head", "longformer-qca"}
DEFAULT_MODEL_KEYS = ["tfidf-lr", "bilstm-attention", "deberta", "longformer-head", "longformer-qca"]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_test_split(archive: Path) -> tuple[list[RAGTruthRecord], dict[str, str]]:
    """Read the held-out test JSONL and source task labels directly from data.zip."""
    with ZipFile(archive) as zip_file:
        test_rows = [
            json.loads(line)
            for line in zip_file.open("data/processed/test.jsonl")
            if line.strip()
        ]
        sources = [
            json.loads(line)
            for line in zip_file.open("data/raw/source_info.jsonl")
            if line.strip()
        ]
    task_by_source = {str(row["source_id"]): str(row.get("task_type", "unknown")) for row in sources}
    records = [RAGTruthRecord(**row) for row in test_rows]
    return records, {record.id: task_by_source.get(record.source_id, "unknown") for record in records}


def artifact_provenance(
    model_key: str,
    repo_id: str,
    qca_artifact_dir: Path | None = None,
) -> dict[str, str]:
    """Report the registry's local-first source and pin the resolved Hub snapshot."""
    local_path = qca_artifact_dir if model_key == "longformer-qca" and qca_artifact_dir else ROOT / LOCAL_ARTIFACTS[model_key]
    if model_key == "longformer-qca" and qca_artifact_dir is not None:
        manifest_path = local_path / "provenance.json"
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if not isinstance(manifest, dict):
                raise ValueError("manifest root must be a JSON object")
            return {
                "artifact_source": str(local_path),
                "artifact_revision": f"local-run:{manifest.get('run_signature', 'unverified')}",
            }
        except (OSError, ValueError, json.JSONDecodeError):
            return {"artifact_source": str(local_path), "artifact_revision": "local:provenance-unreadable"}
    required_file = "model.pkl" if model_key == "tfidf-lr" else "bilstm_weights.pt" if model_key == "bilstm-attention" else "model.safetensors"
    if local_path.is_file() and (model_key == "tfidf-lr" or model_key == "bilstm-attention"):
        return {"artifact_source": str(local_path), "artifact_revision": f"local:{sha256_file(local_path)}"}
    if local_path.is_dir() and (local_path / required_file).is_file():
        return {"artifact_source": str(local_path), "artifact_revision": f"local:{sha256_file(local_path / required_file)}"}

    from huggingface_hub import scan_cache_dir

    for cached in scan_cache_dir().repos:
        if cached.repo_id == repo_id and cached.revisions:
            revision = next(iter(cached.revisions))
            snapshot = cached.repo_path / "snapshots" / revision.commit_hash
            return {"artifact_source": str(snapshot), "artifact_revision": revision.commit_hash}
    return {"artifact_source": f"Hugging Face Hub: {repo_id}", "artifact_revision": "unresolved"}


def batch_probabilities(
    registry: ModelRegistry,
    model_key: str,
    records: list[RAGTruthRecord],
    batch_size: int,
    device: str,
) -> np.ndarray:
    """Use registered artifacts and the model-specific project preprocessing in batches."""
    obj = registry.load_model(model_key, device=device)
    if model_key == "tfidf-lr":
        return np.asarray(obj.predict_proba(records), dtype=float)
    if model_key == "bilstm-attention":
        return np.asarray(obj.predict_proba(records, batch_size=batch_size, device=device), dtype=float)

    if model_key == "deberta":
        from hallucination_detector.modeling.deberta_classifier import DeBERTaClassifier

        wrapper = DeBERTaClassifier.__new__(DeBERTaClassifier)
        wrapper.tokenizer = obj["tokenizer"]
        wrapper.model = obj["model"]
        wrapper.max_length = 512
        wrapper.input_mode = "ca"
        wrapper.num_workers = 0
        return wrapper.predict_proba(records, batch_size=batch_size, device=device)

    if model_key == "longformer-head":
        from hallucination_detector.modeling.longformer_classifier import LongformerClassifier

        wrapper = LongformerClassifier.__new__(LongformerClassifier)
        wrapper.tokenizer = obj["tokenizer"]
        wrapper.model = obj["model"]
        wrapper.max_length = 1024
        wrapper.input_mode = "ca"
        wrapper.truncation_strategy = "head"
        wrapper.global_attention_mode = "cls"
        wrapper.num_workers = 0
        return wrapper.predict_proba(records, batch_size=batch_size, device=device)

    if model_key == "longformer-qca":
        from hallucination_detector.modeling.longformer_classifier import LongformerClassifier
        from hallucination_detector.training.longformer_smart_train import predict_smart_proba

        wrapper = LongformerClassifier.__new__(LongformerClassifier)
        wrapper.tokenizer = obj["tokenizer"]
        wrapper.model = obj["model"]
        wrapper.max_length = 1024
        return predict_smart_proba(
            model=wrapper,
            records=records,
            batch_size=batch_size,
            device=torch.device(device),
            bf16=False,
            input_mode="qca",
            truncation_strategy="smart_selector",
            global_attention_mode="cls",
            selector=obj["selector"],
        )
    raise ValueError(f"Unsupported registered model: {model_key}")


def validate_batch_matches_registry_single(
    registry: ModelRegistry,
    model_key: str,
    records: list[RAGTruthRecord],
    probabilities: np.ndarray,
    device: str,
    count: int = 2,
) -> None:
    """Check the batched evaluator against the app's single-record registry path."""
    for index, record in enumerate(records[:count]):
        result = registry.predict(
            model_key,
            question=record.question,
            context=str(record.context) if isinstance(record.context, str) else record.context,
            answer=record.answer,
            device=device,
        )
        expected = np.array([result.supported_probability, result.hallucination_probability])
        if not np.allclose(probabilities[index], expected, atol=2e-5, rtol=2e-5):
            raise RuntimeError(
                f"Batched evaluation does not match ModelRegistry.predict for "
                f"{model_key}, record {record.id}: {probabilities[index]} vs {expected}"
            )


def category_metrics(records, probs, predictions, categories) -> dict[str, dict]:
    output = {}
    category_values = sorted(set(categories))
    for category in category_values:
        indices = [i for i, value in enumerate(categories) if value == category]
        subset_records = [records[i] for i in indices]
        subset_probs = probs[indices]
        subset_predictions = predictions[indices]
        output[category] = evaluate_binary_classifier(
            subset_records, subset_predictions, subset_probs
        )
    return output


def run(
    archive: Path,
    output_dir: Path,
    device: str,
    batch_size: int,
    model_keys: list[str] | None = None,
    qca_artifact_dir: Path | None = None,
) -> int:
    records, task_by_id = read_test_split(archive)
    archive_hash = sha256_file(archive)
    categories = [task_by_id[record.id] for record in records]
    output_dir.mkdir(parents=True, exist_ok=True)
    prediction_path = output_dir / "ragtruth_test_predictions.csv"
    # Keep the established benchmark cohort fixed; optional catalog additions
    # must be evaluated into their own reports explicitly.
    selected_model_keys = list(DEFAULT_MODEL_KEYS) if model_keys is None else model_keys
    metrics_report = {
        "evaluation_type": "reproduced held-out benchmark evaluation",
        "dataset": "RAGTruth archive: data/processed/test.jsonl",
        "dataset_archive": str(archive),
        "dataset_archive_sha256": archive_hash,
        "split": "test",
        "record_count": len(records),
        "inference_device": {
            "transformers": device,
            "tfidf-lr": "cpu",
            "bilstm-attention": "cpu",
        },
        "batch_size": batch_size,
        "requested_models": selected_model_keys,
        "not_evaluated_models": [key for key in MODEL_CATALOG if key not in selected_model_keys],
        "decision_threshold": 0.5,
        "thresholds_reported_without_tuning": [0.25, 0.5, 0.75],
        "evaluated_at_utc": datetime.now(timezone.utc).isoformat(),
        "limitations": [
            "This evaluates the artifacts resolved by the current registry; their original training runs and dataset hashes are not fully verifiable from the available checkpoint metadata.",
            "Longformer QCA inference requires a fitted selector and a provenance manifest whose hashes match the bundled checkpoint.",
            "This evaluation measures this benchmark split only and does not establish deployment reliability or cross-benchmark generalization.",
        ],
        "models": {},
    }

    csv_fields = [
        "record_id", "source_id", "task_type", "true_label", "model_key", "model_name",
        "model_id", "artifact_revision", "artifact_source", "input_mode", "inference_device", "predicted_label",
        "supported_probability", "hallucination_probability", "correct", "error",
    ]
    rows = []
    for model_key in selected_model_keys:
        meta = MODEL_CATALOG[model_key]
        model_device = device if model_key in TRANSFORMER_MODEL_KEYS else "cpu"
        model_batch_size = batch_size if model_key in TRANSFORMER_MODEL_KEYS else max(batch_size, 64)
        registry = ModelRegistry(
            artifact_paths={"longformer-qca": qca_artifact_dir}
            if model_key == "longformer-qca" and qca_artifact_dir is not None
            else None
        )
        model_info = {
            "model_name": meta.display_name,
            "model_id": meta.repo_id,
            "input_mode": meta.input_mode,
            "inference_device": model_device,
            "batch_size": model_batch_size,
            "artifact_provenance_status": meta.provenance_status,
            **artifact_provenance(model_key, meta.repo_id, qca_artifact_dir),
        }
        model_rows = []
        try:
            if model_device.startswith("cuda") and torch.cuda.is_available():
                torch.cuda.set_device(torch.device(model_device))
                torch.cuda.reset_peak_memory_stats(torch.device(model_device))
            probs = batch_probabilities(registry, model_key, records, model_batch_size, model_device)
            if probs.shape != (len(records), 2):
                raise RuntimeError(f"Expected probability shape {(len(records), 2)}, got {probs.shape}")
            validate_batch_matches_registry_single(registry, model_key, records, probs, model_device)
            if model_key == "longformer-qca" and qca_artifact_dir is not None:
                model_info["artifact_provenance_status"] = "HASHES_VERIFIED"
            if model_device.startswith("cuda") and torch.cuda.is_available():
                torch.cuda.synchronize(torch.device(model_device))
                model_info["cuda_memory_peak_mib"] = {
                    "allocated": round(torch.cuda.max_memory_allocated(torch.device(model_device)) / 2**20, 2),
                    "reserved": round(torch.cuda.max_memory_reserved(torch.device(model_device)) / 2**20, 2),
                }
            predictions = (probs[:, 1] >= 0.5).astype(int)
            model_metrics = evaluate_binary_classifier(records, predictions, probs)
            fixed_threshold_views = {}
            for threshold in (0.25, 0.5, 0.75):
                fixed_threshold_views[str(threshold)] = evaluate_binary_classifier(
                    records, predictions, probs, decision_threshold=threshold
                )["threshold_metrics"]
            model_metrics["fixed_threshold_comparison"] = fixed_threshold_views
            model_metrics["per_task_type"] = category_metrics(records, probs, predictions, categories)
            model_metrics["correct_predictions"] = int((predictions == np.array([int(r.hallucinated) for r in records])).sum())
            model_info.update(status="ok", metrics=model_metrics)
            for record, category, probs_row, prediction in zip(records, categories, probs, predictions):
                y = int(record.hallucinated)
                model_rows.append({
                    "record_id": record.id,
                    "source_id": record.source_id,
                    "task_type": category,
                    "true_label": "HALLUCINATED" if y else "SUPPORTED",
                    "model_key": model_key,
                    "model_name": meta.display_name,
                    "model_id": meta.repo_id,
                    "artifact_revision": model_info["artifact_revision"],
                    "artifact_source": model_info["artifact_source"],
                    "input_mode": meta.input_mode,
                    "inference_device": model_device,
                    "predicted_label": "HALLUCINATED" if prediction else "SUPPORTED",
                    "supported_probability": float(probs_row[0]),
                    "hallucination_probability": float(probs_row[1]),
                    "correct": bool(int(prediction) == y),
                    "error": "",
                })
            print(f"{model_key}: evaluated {len(records)} records")
        except Exception as exc:
            message = f"{type(exc).__name__}: {exc}"
            model_info.update(status="error", error=message)
            if model_device.startswith("cuda") and torch.cuda.is_available():
                model_info["cuda_memory_peak_mib"] = {
                    "allocated": round(torch.cuda.max_memory_allocated(torch.device(model_device)) / 2**20, 2),
                    "reserved": round(torch.cuda.max_memory_reserved(torch.device(model_device)) / 2**20, 2),
                }
            print(f"{model_key}: {message}", file=sys.stderr)
            for record, category in zip(records, categories):
                model_rows.append({
                    "record_id": record.id,
                    "source_id": record.source_id,
                    "task_type": category,
                    "true_label": "HALLUCINATED" if record.hallucinated else "SUPPORTED",
                    "model_key": model_key,
                    "model_name": meta.display_name,
                    "model_id": meta.repo_id,
                    "artifact_revision": model_info.get("artifact_revision", "unresolved"),
                    "artifact_source": model_info.get("artifact_source", ""),
                    "input_mode": meta.input_mode,
                    "inference_device": model_device,
                    "predicted_label": "",
                    "supported_probability": "",
                    "hallucination_probability": "",
                    "correct": "",
                    "error": message,
                })
        metrics_report["models"][model_key] = model_info
        rows.extend(model_rows)
        del registry
        gc.collect()
        if model_device.startswith("cuda") and torch.cuda.is_available():
            torch.cuda.empty_cache()

    with prediction_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=csv_fields)
        writer.writeheader()
        writer.writerows(rows)
    json_path = output_dir / "ragtruth_test_metrics.json"
    json_path.write_text(json.dumps(metrics_report, indent=2, allow_nan=False), encoding="utf-8")
    write_markdown_report(output_dir / "ragtruth_test_comparison.md", metrics_report)
    successes = sum(info.get("status") == "ok" for info in metrics_report["models"].values())
    print(f"Predictions: {prediction_path} ({len(rows)} model-record rows)")
    print(f"Metrics: {json_path}")
    print(f"Comparison: {output_dir / 'ragtruth_test_comparison.md'}")
    return 0 if successes else 1


def write_markdown_report(path: Path, report: dict) -> None:
    lines = [
        "# RAGTruth held-out test evaluation",
        "",
        "This report was generated by inference on the archived 2,700-record test split. It is distinct from the six-example hand-authored diagnostic report and from historical README metric claims.",
        "",
        f"- Dataset archive SHA-256: `{report['dataset_archive_sha256']}`",
        f"- Test records: {report['record_count']}",
        f"- Inference devices: transformers `{report['inference_device']['transformers']}`, TF-IDF/BiLSTM `cpu`; transformer batch size: {report['batch_size']}; CPU BiLSTM batch size: 64",
        f"- Models requested: {', '.join(report['requested_models'])}",
        f"- Models not evaluated: {', '.join(report['not_evaluated_models']) or 'none'}",
        "- Threshold was fixed at 0.5; 0.25 and 0.75 are descriptive comparisons only and were not selected from these test results.",
        "",
        "| Model | Device | Status | Accuracy | Precision | Recall | F1 | ROC-AUC | Average Precision (AP) |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for key, info in report["models"].items():
        metrics = info.get("metrics", {})
        def show(value):
            return "N/A" if value is None else f"{value:.4f}"
        lines.append(
            f"| {info['model_name']} | {info['inference_device']} | {info['status']} | {show(metrics.get('accuracy'))} | "
            f"{show(metrics.get('precision'))} | {show(metrics.get('recall'))} | "
            f"{show(metrics.get('f1'))} | {show(metrics.get('roc_auc'))} | "
            f"{show(metrics.get('average_precision'))} |"
        )
        if info.get("status") == "error":
            lines.append(f"\n`{key}` error: {info['error']}\n")
    lines.extend(["", "## Interpretation limits", ""])
    lines.extend(f"- {item}" for item in report["limitations"])
    lines.extend([
        "- Per-class support, confusion matrices, fixed-threshold metrics, and per-task results are in `ragtruth_test_metrics.json`.",
        "- Per-record outputs and error rows are in `ragtruth_test_predictions.csv`.",
    ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=ROOT / "data.zip")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "reports/benchmarks")
    parser.add_argument("--device", default="cpu", help="Inference device, e.g. cpu or cuda")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument(
        "--qca-artifact-dir",
        type=Path,
        default=None,
        help="Optional frozen local QCA best_model bundle. Used only when --models includes longformer-qca.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=list(MODEL_CATALOG),
        help="Optional model keys to evaluate; defaults to all five registered models.",
    )
    args = parser.parse_args(argv)
    if args.batch_size <= 0:
        parser.error("--batch-size must be positive")
    return run(args.archive, args.output_dir, args.device, args.batch_size, args.models, args.qca_artifact_dir)


if __name__ == "__main__":
    raise SystemExit(main())
