"""Run hand-authored diagnostic examples through every registered model.

This is a smoke/behavior diagnostic, not a held-out benchmark. Results are
written as one row per model and example, including loading/inference errors.
"""
from __future__ import annotations

import argparse
import csv
from functools import lru_cache
from pathlib import Path
import sys
from collections import defaultdict

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

VALID_LABELS = {"SUPPORTED", "HALLUCINATED"}


def diagnostic_cases() -> list[dict[str, str]]:
    long_context = (
        "The following is a collection of unrelated historical notes. " * 170
        + "Marie Curie was born in Warsaw, Poland. She later conducted research in Paris."
    )
    return [
        {"case_id": "curie_supported", "category": "supported", "expected_label": "SUPPORTED", "question": "Where was Marie Curie born?", "context": "Marie Curie was born in Warsaw, Poland.", "answer": "Marie Curie was born in Warsaw, Poland."},
        {"case_id": "curie_contradiction", "category": "contradictory", "expected_label": "HALLUCINATED", "question": "Where was Marie Curie born?", "context": "Marie Curie was born in Warsaw, Poland.", "answer": "Marie Curie was born in Paris, France."},
        {"case_id": "unsupported_addition", "category": "unsupported-addition", "expected_label": "HALLUCINATED", "question": "Where was Marie Curie born?", "context": "Marie Curie was born in Warsaw, Poland.", "answer": "Marie Curie was born in Warsaw, Poland, in 1867 and won seven Olympic medals."},
        {"case_id": "irrelevant", "category": "irrelevant-answer", "expected_label": "HALLUCINATED", "question": "Where was Marie Curie born?", "context": "Marie Curie was born in Warsaw, Poland.", "answer": "The Pacific Ocean is the largest ocean on Earth."},
        {"case_id": "capital_supported", "category": "supported", "expected_label": "SUPPORTED", "question": "What is the capital of Japan?", "context": "Tokyo is the capital of Japan.", "answer": "Tokyo is the capital of Japan."},
        {"case_id": "long_context_contradiction", "category": "long-context-contradiction", "expected_label": "HALLUCINATED", "question": "Where was Marie Curie born?", "context": long_context, "answer": "Marie Curie was born in Paris, France."},
    ]


def validate_cases(cases: list[dict[str, str]]) -> None:
    """Fail early on missing labels, invalid labels, or duplicate case IDs."""
    seen: set[str] = set()
    for index, case in enumerate(cases):
        case_id = case.get("case_id", "")
        if not case_id:
            raise ValueError(f"Case at index {index} is missing a non-empty case_id")
        if case_id in seen:
            raise ValueError(f"Duplicate diagnostic case_id: {case_id}")
        seen.add(case_id)
        label = case.get("expected_label")
        if label not in VALID_LABELS:
            raise ValueError(
                f"Case '{case_id}' has invalid expected_label {label!r}; "
                f"expected one of {sorted(VALID_LABELS)}"
            )


def prediction_is_correct(expected_label: str, predicted_label: str) -> bool:
    if expected_label not in VALID_LABELS:
        raise ValueError(f"Invalid expected label: {expected_label!r}")
    return predicted_label in VALID_LABELS and predicted_label == expected_label


def aggregate_summary(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["model_key"])].append(row)
    summary = []
    for model_key, model_rows in grouped.items():
        count = len(model_rows)
        correct = sum(bool(row["correct"]) for row in model_rows)
        summary.append({
            "model_key": model_key,
            "model_name": model_rows[0]["model_name"],
            "cases": count,
            "correct_predictions": correct,
            "errors": count - correct,
            "diagnostic_accuracy": correct / count if count else 0.0,
        })
    return summary


def write_summary(path: Path, summary: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Tiny hand-authored diagnostic set - not benchmark performance",
        "",
        "Counts compare each prediction with the explicit expected labels in the diagnostic cases. "
        "This six-case set does not establish general model quality.",
        "",
        "| Model | Cases | Correct | Errors | Diagnostic accuracy |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in summary:
        lines.append(
            f"| {row['model_name']} | {row['cases']} | {row['correct_predictions']} | "
            f"{row['errors']} | {float(row['diagnostic_accuracy']):.1%} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


@lru_cache(maxsize=None)
def artifact_source(key: str, repo_id: str) -> str:
    """Describe the registry branch selected by this checkout at runtime."""
    from huggingface_hub import scan_cache_dir

    local_candidates = {
        "tfidf-lr": ROOT / "reports/experiments_backup/baseline_tfidf_logreg/model.pkl",
        "bilstm-attention": ROOT / "models/bilstm_attention_baseline/bilstm_weights.pt",
        "deberta": ROOT / "reports/experiments/baseline_deberta/best_model/model.safetensors",
        "longformer-head": ROOT / "reports/experiments/longformer_1024_head/best_model/model.safetensors",
        "longformer-qca": ROOT / "reports/experiments/longformer_1024_smart_qca/best_model/model.safetensors",
    }
    candidate = local_candidates[key]
    if candidate.exists():
        return f"local:{candidate.relative_to(ROOT)}"
    for cached in scan_cache_dir().repos:
        if cached.repo_id == repo_id:
            if cached.revisions:
                revision = next(iter(cached.revisions))
                revision_path = cached.repo_path / "snapshots" / revision.commit_hash
                return f"huggingface-cache:{revision_path}"
            return f"huggingface-cache:{cached.repo_path}"
    return f"huggingface-hub:{repo_id}@main (downloaded during inference)"


def run(output: Path, device: str | None = None) -> int:
    from hallucination_detector.inference.model_registry import MODEL_CATALOG, ModelRegistry

    cases = diagnostic_cases()
    validate_cases(cases)
    output.parent.mkdir(parents=True, exist_ok=True)
    registry = ModelRegistry()
    rows: list[dict[str, object]] = []
    for key, meta in MODEL_CATALOG.items():
        for case in cases:
            row: dict[str, object] = {
                **case, "model_key": key, "display_name": meta.display_name,
                "model_name": meta.display_name,
                "repo_id": meta.repo_id, "input_mode": meta.input_mode,
                "artifact_source": artifact_source(key, meta.repo_id),
                "revision": meta.default_revision,
                "label_convention": "0=SUPPORTED, 1=HALLUCINATED",
                "status": "error", "prediction": "", "predicted_label": "", "supported_probability": "",
                "hallucination_probability": "", "correct": False, "error": "",
            }
            try:
                result = registry.predict(
                    key, question=case["question"], context=case["context"],
                    answer=case["answer"], device=device,
                )
                row.update(status="ok", prediction=result.prediction,
                           predicted_label=result.prediction,
                           supported_probability=result.supported_probability,
                           hallucination_probability=result.hallucination_probability,
                           correct=prediction_is_correct(case["expected_label"], result.prediction))
            except Exception as exc:  # keep going so unavailable models are reported
                row["error"] = f"{type(exc).__name__}: {exc}"
                print(f"{key}/{case['case_id']}: {row['error']}", file=sys.stderr)
            rows.append(row)

    with output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = aggregate_summary(rows)
    summary_path = output.with_name("model_diagnostic_summary.md")
    write_summary(summary_path, summary)
    succeeded = sum(row["status"] == "ok" for row in rows)
    print(f"Wrote {len(rows)} model-case rows ({succeeded} successful) to {output}")
    print(f"Wrote per-model summary to {summary_path}")
    print("Tiny hand-authored diagnostic set only; not benchmark performance.")
    return 0 if succeeded else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "reports/diagnostics/model_diagnostic_predictions.csv")
    parser.add_argument("--device", default=None, help="Inference device, e.g. cpu or cuda")
    args = parser.parse_args()
    raise SystemExit(run(args.output, args.device))
