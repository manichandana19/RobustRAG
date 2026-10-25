"""Run deterministic functional inference checks for locally available models.

This is a code-path smoke suite, not an accuracy benchmark. Predictions are not
compared with hand-authored expected labels.
"""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from hallucination_detector.data.schema import RAGTruthRecord  # noqa: E402
from hallucination_detector.inference.model_registry import MODEL_CATALOG, ModelRegistry  # noqa: E402
from scripts.evaluate_benchmark import batch_probabilities  # noqa: E402


def _manual_contradiction() -> dict[str, str]:
    """Reuse the documented manual case without adding a second copy here."""
    path = ROOT / "reports" / "manual_evaluation" / "robustrag_manual_set_v1.json"
    cases = json.loads(path.read_text(encoding="utf-8"))["cases"]
    case = next(item for item in cases if item["case_id"] == "v1_qa_contradiction_location")
    return {field: case[field] for field in ("question", "context", "answer")}


CASES = {
    "supported": {
        "question": "What is the capital of Japan?",
        "context": "Tokyo is the capital of Japan.",
        "answer": "Tokyo is the capital of Japan.",
    },
    "direct_contradiction": _manual_contradiction(),
    "unsupported_addition": {
        "question": "What does the museum context say about visitor counts?",
        "context": "The museum opened in 1900 in Rome. It contains several Renaissance paintings.",
        "answer": "The museum receives 25,000 visitors every day.",
    },
    "long_context": {
        "question": "Where is the museum located?",
        "context": ("The museum opened in 1900 in Rome. It exhibits Renaissance art. " * 100).strip(),
        "answer": "The museum is located in Rome.",
    },
}


def check_result(result: Any) -> dict[str, Any]:
    p_supported = result.supported_probability
    p_hallucinated = result.hallucination_probability
    values = np.asarray([p_supported, p_hallucinated], dtype=float)
    if not np.all(np.isfinite(values)) or np.any(values < 0) or np.any(values > 1):
        raise AssertionError(f"Model returned invalid probabilities: {values.tolist()}")
    if not np.isclose(values.sum(), 1.0, atol=2e-3):
        raise AssertionError(f"Model probabilities do not sum to one: {values.tolist()}")
    expected_from_threshold = p_hallucinated >= 0.5
    if result.is_hallucinated != expected_from_threshold:
        raise AssertionError("Boolean verdict does not match the fixed 0.5 threshold.")
    expected_label = "HALLUCINATED" if expected_from_threshold else "SUPPORTED"
    if result.prediction != expected_label:
        raise AssertionError("Displayed prediction label does not match the fixed 0.5 threshold.")
    return {
        "prediction": result.prediction,
        "supported_probability": float(p_supported),
        "hallucination_probability": float(p_hallucinated),
        "probability_sum": float(values.sum()),
        "threshold_label_consistent": True,
    }


def run(model_keys: list[str], device: str) -> int:
    report: dict[str, Any] = {
        "suite": "functional inference smoke; not benchmark performance",
        "device": device,
        "cases": list(CASES),
        "models": {},
    }
    failures = 0
    for key in model_keys:
        model_result: dict[str, Any] = {"model_id": MODEL_CATALOG[key].repo_id, "cases": {}}
        try:
            registry = ModelRegistry()
            records = [
                RAGTruthRecord(
                    id=case_id,
                    source_id=case_id,
                    question=case["question"],
                    context=case["context"],
                    answer=case["answer"],
                    hallucinated=False,
                    split="validation",
                )
                for case_id, case in CASES.items()
            ]
            batch_size = 2
            batched = batch_probabilities(registry, key, records, batch_size, device)
            if batched.shape != (len(records), 2):
                raise AssertionError(f"Unexpected batch probability shape: {batched.shape}")
            for index, (case_id, case) in enumerate(CASES.items()):
                result = registry.predict(
                    key,
                    question=case["question"],
                    context=case["context"],
                    answer=case["answer"],
                    device=device,
                )
                checked = check_result(result)
                if not np.allclose(
                    batched[index],
                    [result.supported_probability, result.hallucination_probability],
                    atol=2e-5,
                    rtol=2e-5,
                ):
                    raise AssertionError(f"Batched/single inference mismatch for {key}/{case_id}.")
                checked["batched_single_parity"] = True
                model_result["cases"][case_id] = checked
            model_result["status"] = "passed"
        except Exception as exc:
            failures += 1
            model_result["status"] = "failed"
            model_result["error"] = f"{type(exc).__name__}: {exc}"
        finally:
            if "registry" in locals():
                del registry
            gc.collect()
            if device.startswith("cuda") and torch.cuda.is_available():
                torch.cuda.empty_cache()
        report["models"][key] = model_result
    print(json.dumps(report, indent=2))
    return 1 if failures else 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    available = [key for key, meta in MODEL_CATALOG.items() if meta.provenance_status != "UNVERIFIED"]
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--models", nargs="+", choices=available, default=available)
    args = parser.parse_args()
    raise SystemExit(run(args.models, args.device))


if __name__ == "__main__":
    main()
