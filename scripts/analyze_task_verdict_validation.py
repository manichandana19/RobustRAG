"""Audit Longformer Head's proposed task-aware score bands on saved validation data."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hallucination_detector.inference.task_verdicts import DEFAULT_CONFIG, load_task_verdict_config  # noqa: E402


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def binary_metrics(rows: list[dict], threshold: float) -> dict:
    y = np.asarray([row["true_label"] == "HALLUCINATED" for row in rows], dtype=bool)
    scores = np.asarray([float(row["hallucination_probability"]) for row in rows], dtype=float)
    pred = scores >= threshold
    tp, tn = int(np.sum(y & pred)), int(np.sum(~y & ~pred))
    fp, fn = int(np.sum(~y & pred)), int(np.sum(y & ~pred))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"n": len(rows), "supported": int(np.sum(~y)), "hallucinated": int(np.sum(y)),
            "precision": precision, "recall": recall, "f1": f1,
            "confusion": {"tn": tn, "fp": fp, "fn": fn, "tp": tp}}


def auc_ci(rows: list[dict], *, seed: int = 42, repetitions: int = 2000) -> dict:
    y = np.asarray([row["true_label"] == "HALLUCINATED" for row in rows], dtype=bool)
    scores = np.asarray([float(row["hallucination_probability"]) for row in rows], dtype=float)
    observed = float(roc_auc_score(y, scores))
    rng = np.random.default_rng(seed)
    positive = np.flatnonzero(y)
    negative = np.flatnonzero(~y)
    boot = np.empty(repetitions, dtype=float)
    for i in range(repetitions):
        sample = np.concatenate((rng.choice(positive, len(positive), replace=True),
                                 rng.choice(negative, len(negative), replace=True)))
        boot[i] = roc_auc_score(y[sample], scores[sample])
    low, high = np.quantile(boot, [.025, .975])
    return {"auc": observed, "bootstrap_95pct_ci": [float(low), float(high)],
            "bootstrap_repetitions": repetitions, "bootstrap_seed": seed}


def summarize_bands(rows: list[dict], low: float, high: float) -> dict:
    score = lambda row: float(row["hallucination_probability"])
    bands = {
        "likely_supported": [row for row in rows if score(row) < low],
        "uncertain": [row for row in rows if low <= score(row) < high],
        "likely_hallucinated": [row for row in rows if score(row) >= high],
    }
    output = {"low_threshold": low, "high_threshold": high, "bands": {}}
    for name, subset in bands.items():
        n = len(subset)
        supported = sum(row["true_label"] == "SUPPORTED" for row in subset)
        hallucinated = n - supported
        output["bands"][name] = {"n": n, "coverage": n / len(rows) if rows else 0.0,
            "true_supported": supported, "true_hallucinated": hallucinated,
            "hallucinated_rate": hallucinated / n if n else None,
            "supported_rate": supported / n if n else None}
    return output


def analyze(predictions_path: Path, metrics_path: Path, config_path: Path) -> dict:
    metadata = json.loads(metrics_path.read_text(encoding="utf-8"))
    if metadata.get("split") != "validation" or metadata.get("evaluation_type", "").find("validation") < 0:
        raise ValueError("Threshold analysis only accepts saved validation metadata.")
    with predictions_path.open(newline="", encoding="utf-8-sig") as stream:
        rows = [row for row in csv.DictReader(stream) if row["model_key"] == "longformer-head"]
    if not rows or len(rows) != metadata.get("validation_records"):
        raise ValueError("Expected one Longformer Head prediction for each validation record.")
    if {row["task_type"] for row in rows} - {"QA", "Data2txt", "Summary"}:
        raise ValueError("Unexpected task value in validation predictions.")
    config = load_task_verdict_config(config_path)
    output = {"evaluation_type": "validation-only audit of predeclared Longformer Head verdict bands",
        "split": "validation", "validation_records": len(rows),
        "dataset_archive_sha256": metadata["archive_sha256"],
        "prediction_csv_sha256": sha256_file(predictions_path), "source_predictions": str(predictions_path),
        "task_bands": {}, "not_sure_binary_0_5": {},
        "summary_task": {}, "thresholds_selected_from_this_report": False}
    for task in ("QA", "Data2txt"):
        group = [row for row in rows if row["task_type"] == task]
        rule_name = "QA" if task == "QA" else "Data-to-text"
        rule = config["models"]["longformer-head"]["task_rules"][rule_name]
        output["task_bands"][rule_name] = {"n": len(group), "class_counts": binary_metrics(group, .5),
            "auc": auc_ci(group), "bands": summarize_bands(group, float(rule["supported_below"]), float(rule["hallucinated_at_or_above"]))}
    summary = [row for row in rows if row["task_type"] == "Summary"]
    output["summary_task"] = {"n": len(summary), "class_counts": binary_metrics(summary, .5), "auc": auc_ci(summary)}
    output["not_sure_binary_0_5"] = {"overall": binary_metrics(rows, .5), "by_task": {
        task: binary_metrics([row for row in rows if row["task_type"] == task], .5)
        for task in ("QA", "Data2txt", "Summary")}}
    return output


def to_markdown(report: dict) -> str:
    lines = ["# Longformer Head task-verdict validation evidence", "",
        "Validation-only analysis of the predeclared QA and Data-to-text bands. No threshold was fit from these results; no manual or held-out test examples were read.", "",
        f"- Validation records: {report['validation_records']}  ",
        f"- RAGTruth archive SHA-256: `{report['dataset_archive_sha256']}`  ",
        f"- Saved prediction SHA-256: `{report['prediction_csv_sha256']}`  ",
        "- Scores are uncalibrated model outputs, not probabilities of truth.", ""]
    for task, data in report["task_bands"].items():
        lines.extend([f"## {task}", "",
            f"N={data['n']}; labels SUPPORTED={data['class_counts']['supported']}, HALLUCINATED={data['class_counts']['hallucinated']}; ROC AUC={data['auc']['auc']:.3f} (stratified bootstrap 95% interval {data['auc']['bootstrap_95pct_ci'][0]:.3f}–{data['auc']['bootstrap_95pct_ci'][1]:.3f}).", "",
            f"Bands: score < {data['bands']['low_threshold']:.2f}; uncertain for {data['bands']['low_threshold']:.2f} ≤ score < {data['bands']['high_threshold']:.2f}; likely hallucinated for score ≥ {data['bands']['high_threshold']:.2f}.", "",
            "| Display band | N | Coverage | Actual SUPPORTED | Actual HALLUCINATED | Hallucinated share |", "|---|---:|---:|---:|---:|---:|"])
        for name, label in (("likely_supported", "Likely supported"), ("uncertain", "Uncertain"), ("likely_hallucinated", "Likely hallucinated")):
            band = data["bands"]["bands"][name]
            share = "—" if band["hallucinated_rate"] is None else f"{band['hallucinated_rate']:.1%}"
            lines.append(f"| {label} | {band['n']} | {band['coverage']:.1%} | {band['true_supported']} | {band['true_hallucinated']} | {share} |")
        lines.append("")
    summary = report["summary_task"]
    lines.extend(["## Summarization", "",
        f"N={summary['n']}; SUPPORTED={summary['class_counts']['supported']}; HALLUCINATED={summary['class_counts']['hallucinated']}; ROC AUC={summary['auc']['auc']:.3f} (95% interval {summary['auc']['bootstrap_95pct_ci'][0]:.3f}–{summary['auc']['bootstrap_95pct_ci'][1]:.3f}). This is near chance; show NO RELIABLE VERDICT for Longformer Head on this task.", "",
        "## Not sure / existing 0.50 cutoff", "",
        "The fallback remains the registry's `hallucination_score >= 0.50` binary decision. It is not calibrated. Confusion counts and scores by task:", "",
        "| Task | N | Precision | Recall | F1 | TN | FP | FN | TP |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|"])
    for task, data in report["not_sure_binary_0_5"]["by_task"].items():
        c = data["confusion"]
        lines.append(f"| {task} | {data['n']} | {data['precision']:.3f} | {data['recall']:.3f} | {data['f1']:.3f} | {c['tn']} | {c['fp']} | {c['fn']} | {c['tp']} |")
    lines.extend(["", "These are validation-set proportions, not calibration guarantees or real-world reliability estimates. The bands are fixed display/triage rules requested for the UI; validation reuse does not establish independent performance. The uncertain band is an abstention category and should not be counted as a correct/incorrect binary prediction."])
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, default=ROOT / "reports/validation/threshold_sweep_20261025_205000/validation_predictions.csv")
    parser.add_argument("--metrics", type=Path, default=ROOT / "reports/validation/threshold_sweep_20261025_205000/threshold_metrics.json")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite evidence report: {args.output_dir}")
    output = analyze(args.predictions, args.metrics, args.config)
    args.output_dir.mkdir(parents=True)
    (args.output_dir / "task_verdict_evidence.json").write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    (args.output_dir / "task_verdict_evidence.md").write_text(to_markdown(output), encoding="utf-8")
    print(f"PASS: wrote validation-only evidence for {output['validation_records']} Longformer Head rows to {args.output_dir}")


if __name__ == "__main__":
    main()
