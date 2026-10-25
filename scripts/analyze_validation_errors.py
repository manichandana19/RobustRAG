"""Analyze Longformer Head errors on RAGTruth validation data only.

This script performs inference with the registered checkpoint at the unchanged
0.5 threshold. It does not read or use the held-out test split.
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
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from hallucination_detector.data.schema import RAGTruthRecord  # noqa: E402
from hallucination_detector.inference.model_registry import ModelRegistry  # noqa: E402
from hallucination_detector.modeling.tfidf_logistic import normalize_text  # noqa: E402
from hallucination_detector.data.torch_dataset import encode_pair_with_answer_preservation  # noqa: E402
from scripts.evaluate_benchmark import (  # noqa: E402
    artifact_provenance,
    batch_probabilities,
    validate_batch_matches_registry_single,
)


CONTEXT_BINS = ("<512", "512-1023", ">=1024")
ANSWER_BINS = ("<128", "128-511", ">=512")


def metric_text(value: float | None) -> str:
    return "N/A" if value is None else f"{value:.4f}"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_validation_split(archive: Path) -> tuple[list[RAGTruthRecord], dict[str, str]]:
    with ZipFile(archive) as zip_file:
        records = [
            RAGTruthRecord(**json.loads(line))
            for line in zip_file.open("data/processed/validation.jsonl")
            if line.strip()
        ]
        sources = [
            json.loads(line)
            for line in zip_file.open("data/raw/source_info.jsonl")
            if line.strip()
        ]
    if not records or any(record.split != "validation" for record in records):
        raise ValueError("Expected a nonempty validation-only split.")
    task_by_source = {
        str(row["source_id"]): str(row.get("task_type", "unknown")) for row in sources
    }
    return records, {r.id: task_by_source.get(r.source_id, "unknown") for r in records}


def length_bin(value: int, boundaries: tuple[int, int], labels: tuple[str, ...]) -> str:
    if value < boundaries[0]:
        return labels[0]
    if value < boundaries[1]:
        return labels[1]
    return labels[2]


def input_characteristics(record: RAGTruthRecord, tokenizer, max_length: int = 1024) -> dict:
    context_text = normalize_text(record.context)
    answer_text = normalize_text(record.answer)
    context_token_count = len(tokenizer.encode(context_text, add_special_tokens=False))
    answer_token_count = len(tokenizer.encode(answer_text, add_special_tokens=False))
    encoded = encode_pair_with_answer_preservation(
        context=record.context,
        answer=record.answer,
        tokenizer=tokenizer,
        max_length=max_length,
        truncation_strategy="head",
        global_attention_mode="cls",
    )

    budget = max(1, max_length - 4)
    if answer_token_count > budget:
        context_budget = min(16, context_token_count)
        retained_answer = max(0, budget - context_budget)
    else:
        context_budget = budget - answer_token_count
        retained_answer = answer_token_count
    retained_context = min(context_token_count, context_budget)

    return {
        "question_chars": len(normalize_text(record.question)),
        "context_chars": len(context_text),
        "context_tokens": context_token_count,
        "context_token_bin": length_bin(context_token_count, (512, 1024), CONTEXT_BINS),
        "answer_chars": len(answer_text),
        "answer_tokens": answer_token_count,
        "answer_token_bin": length_bin(answer_token_count, (128, 512), ANSWER_BINS),
        "context_tokens_retained": retained_context,
        "context_truncated": retained_context < context_token_count,
        "answer_truncated": retained_answer < answer_token_count,
        "encoded_sequence_tokens": len(encoded["input_ids"]),
        "structured_context": isinstance(record.context, (dict, list, tuple)),
    }


def summarize(rows: list[dict]) -> dict:
    y_true = np.asarray([r["true_label"] == "HALLUCINATED" for r in rows], dtype=bool)
    y_pred = np.asarray([r["predicted_label"] == "HALLUCINATED" for r in rows], dtype=bool)
    tn = int((~y_true & ~y_pred).sum())
    fp = int((~y_true & y_pred).sum())
    fn = int((y_true & ~y_pred).sum())
    tp = int((y_true & y_pred).sum())
    supported_n = tn + fp
    hallucinated_n = fn + tp
    return {
        "records": len(rows),
        "supported": supported_n,
        "hallucinated": hallucinated_n,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
        "accuracy": (tn + tp) / len(rows) if rows else None,
        "false_positive_rate": fp / supported_n if supported_n else None,
        "false_negative_rate": fn / hallucinated_n if hallucinated_n else None,
        "mean_context_tokens": float(np.mean([r["context_tokens"] for r in rows])) if rows else None,
        "median_context_tokens": float(np.median([r["context_tokens"] for r in rows])) if rows else None,
        "mean_answer_tokens": float(np.mean([r["answer_tokens"] for r in rows])) if rows else None,
        "median_answer_tokens": float(np.median([r["answer_tokens"] for r in rows])) if rows else None,
        "context_truncated_count": sum(r["context_truncated"] for r in rows),
        "structured_context_count": sum(r["structured_context"] for r in rows),
    }


def grouped_summary(rows: list[dict], key_names: tuple[str, ...]) -> dict[str, dict]:
    groups: dict[str, list[dict]] = {}
    for row in rows:
        name = " / ".join(str(row[key]) for key in key_names)
        groups.setdefault(name, []).append(row)
    return {name: summarize(group) for name, group in sorted(groups.items())}


def markdown_report(path: Path, report: dict) -> None:
    lines = [
        "# Longformer Head validation error analysis",
        "",
        "Validation split only. Predictions are from the registered Longformer Head artifact, with the existing 0.5 threshold. No test records or test metrics were used.",
        "",
        f"- Dataset archive SHA-256: `{report['dataset_archive_sha256']}`",
        f"- Validation records: {report['validation_records']}",
        f"- Model ID: `{report['model_id']}`; artifact revision: `{report['artifact_revision']}`",
        f"- Device: `{report['device']}`; batch size: {report['batch_size']}; threshold: 0.5",
        "- Longformer Head uses Context + Answer; the question/instruction is not included in model tokens.",
        "",
        "## Overall validation results",
        "",
        "| Records | Supported | Hallucinated | TN | FP | FN | TP | Accuracy | FPR | FNR |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    m = report["overall"]
    lines.append(
        f"| {m['records']} | {m['supported']} | {m['hallucinated']} | {m['tn']} | {m['fp']} | {m['fn']} | {m['tp']} | "
        f"{m['accuracy']:.4f} | {m['false_positive_rate']:.4f} | {m['false_negative_rate']:.4f} |"
    )
    lines.extend([
        "",
        "## By task",
        "",
        "| Task | N | Supported | Hallucinated | FP | FN | FPR | FNR | Median context tokens | Context truncated | Structured context |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for task, values in report["by_task"].items():
        lines.append(
            f"| {task} | {values['records']} | {values['supported']} | {values['hallucinated']} | "
            f"{values['fp']} | {values['fn']} | {metric_text(values['false_positive_rate'])} | "
            f"{metric_text(values['false_negative_rate'])} | {values['median_context_tokens']:.1f} | "
            f"{values['context_truncated_count']} | {values['structured_context_count']} |"
        )
    lines.extend([
        "",
        "## By task and raw context-token length",
        "",
        "| Task / context bin | N | Supported | Hallucinated | FP | FN | FPR | FNR | Median context tokens | Context truncated |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for group, values in report["by_task_context_bin"].items():
        lines.append(
            f"| {group} | {values['records']} | {values['supported']} | {values['hallucinated']} | "
            f"{values['fp']} | {values['fn']} | {metric_text(values['false_positive_rate'])} | "
            f"{metric_text(values['false_negative_rate'])} | {values['median_context_tokens']:.1f} | "
            f"{values['context_truncated_count']} |"
        )
    lines.extend([
        "",
        "## By task and answer-token length",
        "",
        "| Task / answer bin | N | Supported | Hallucinated | FP | FN | FPR | FNR | Median answer tokens |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for group, values in report["by_task_answer_bin"].items():
        lines.append(
            f"| {group} | {values['records']} | {values['supported']} | {values['hallucinated']} | "
            f"{values['fp']} | {values['fn']} | {metric_text(values['false_positive_rate'])} | "
            f"{metric_text(values['false_negative_rate'])} | {values['median_answer_tokens']:.1f} |"
        )
    lines.extend([
        "",
        "## By task and whether the context was truncated",
        "",
        "| Task / context truncated | N | Supported | Hallucinated | FP | FN | FPR | FNR | Median context tokens |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for group, values in report["by_task_context_truncated"].items():
        lines.append(
            f"| {group} | {values['records']} | {values['supported']} | {values['hallucinated']} | "
            f"{values['fp']} | {values['fn']} | {metric_text(values['false_positive_rate'])} | "
            f"{metric_text(values['false_negative_rate'])} | {values['median_context_tokens']:.1f} |"
        )
    lines.extend([
        "",
        "## Interpretation limits",
        "",
        "- Error groups describe this validation split and this artifact; they do not establish generalization.",
        "- Context and answer lengths use the checkpoint tokenizer and Longformer Head's C+A tokenization. The question length is recorded for inspection, but the question is ignored by this architecture.",
        "- No answer was truncated in this validation split. In Summary, all 77 hallucinated cases with answers shorter than 128 tokens were missed; this pattern is not explained by answer truncation.",
        "- All Data2txt and QA contexts are structured objects, while Summary contexts are text. Their error patterns differ substantially, so structure alone does not explain the errors.",
        "- The per-record CSV supports joining errors back to `data.zip` for annotation review. No threshold or model choice was tuned from this analysis.",
        "",
        f"Per-record predictions and lengths: `{report['prediction_csv']}`.",
    ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(archive: Path, output_dir: Path, device: str, batch_size: int) -> int:
    records, task_by_id = read_validation_split(archive)
    archive_hash = sha256_file(archive)
    registry = ModelRegistry()
    model_obj = registry.load_model("longformer-head", device=device)
    tokenizer = model_obj["tokenizer"]

    try:
        probabilities = batch_probabilities(
            registry, "longformer-head", records, batch_size, device
        )
        validate_batch_matches_registry_single(
            registry, "longformer-head", records, probabilities, device
        )
        if probabilities.shape != (len(records), 2):
            raise RuntimeError(
                f"Expected probability shape {(len(records), 2)}, got {probabilities.shape}"
            )

        prediction_rows = []
        for record, probability in zip(records, probabilities):
            characteristics = input_characteristics(record, tokenizer)
            p_hall = float(probability[1])
            y = int(record.hallucinated)
            prediction_rows.append({
                "record_id": record.id,
                "source_id": record.source_id,
                "task_type": task_by_id[record.id],
                "true_label": "HALLUCINATED" if y else "SUPPORTED",
                "predicted_label": "HALLUCINATED" if p_hall >= 0.5 else "SUPPORTED",
                "hallucination_probability": p_hall,
                "supported_probability": float(probability[0]),
                "correct": (p_hall >= 0.5) == bool(y),
                "annotations": json.dumps(
                    sorted({a.get("label_type", "unknown") for a in record.annotations}),
                    ensure_ascii=False,
                ),
                **characteristics,
            })

        grouped = {
            "overall": summarize(prediction_rows),
            "by_task": grouped_summary(prediction_rows, ("task_type",)),
            "by_task_context_bin": grouped_summary(
                prediction_rows, ("task_type", "context_token_bin")
            ),
            "by_task_answer_bin": grouped_summary(
                prediction_rows, ("task_type", "answer_token_bin")
            ),
            "by_task_context_truncated": grouped_summary(
                prediction_rows, ("task_type", "context_truncated")
            ),
        }
        output_dir.mkdir(parents=True, exist_ok=True)
        csv_path = output_dir / "longformer_head_validation_predictions.csv"
        fields = list(prediction_rows[0])
        with csv_path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(prediction_rows)

        report = {
            "evaluation_type": "validation-only Longformer Head error analysis",
            "split": "validation",
            "validation_records": len(records),
            "dataset_archive_sha256": archive_hash,
            "threshold": 0.5,
            "device": device,
            "batch_size": batch_size,
            "model_id": "nishantwhat/robustrag-longformer-head",
            **artifact_provenance("longformer-head", "nishantwhat/robustrag-longformer-head"),
            **grouped,
            "prediction_csv": str(csv_path),
        }
        json_path = output_dir / "longformer_head_validation_analysis.json"
        json_path.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
        md_path = output_dir / "longformer_head_validation_analysis.md"
        markdown_report(md_path, report)
        print(f"Validation predictions: {csv_path} ({len(prediction_rows)} rows)")
        print(f"Validation analysis: {md_path}")
        print(f"Validation details: {json_path}")
        print(f"Overall: {json.dumps(report['overall'])}")
        return 0
    finally:
        del model_obj
        del registry
        gc.collect()
        if device.startswith("cuda") and torch.cuda.is_available():
            torch.cuda.empty_cache()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=ROOT / "data.zip")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=4)
    args = parser.parse_args(argv)
    if args.batch_size <= 0:
        parser.error("--batch-size must be positive")
    return run(args.archive, args.output_dir, args.device, args.batch_size)


if __name__ == "__main__":
    raise SystemExit(main())
