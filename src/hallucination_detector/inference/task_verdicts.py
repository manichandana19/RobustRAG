"""Task-aware display verdicts for uncalibrated model scores."""
from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Any

import yaml


TASKS = ("QA", "Summarization", "Data-to-text", "Not sure")
DEFAULT_CONFIG = Path(__file__).resolve().parents[3] / "configs" / "task_verdicts.yaml"


@dataclass(frozen=True)
class TaskVerdict:
    label: str
    category: str
    threshold_description: str
    threshold: float | None
    abstain: bool = False


def load_task_verdict_config(path: str | Path = DEFAULT_CONFIG) -> dict[str, Any]:
    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("schema_version") != 1:
        raise ValueError("Task verdict configuration must use schema_version 1.")
    default = payload.get("default_hallucination_threshold")
    if not isinstance(default, (int, float)) or not math.isfinite(default) or not 0 < default <= 1:
        raise ValueError("default_hallucination_threshold must be in (0, 1].")
    head = payload.get("models", {}).get("longformer-head", {}).get("task_rules", {})
    for task in ("QA", "Data-to-text"):
        rule = head.get(task, {})
        low, high = rule.get("supported_below"), rule.get("hallucinated_at_or_above")
        if not all(isinstance(value, (int, float)) and math.isfinite(value) for value in (low, high)) or not 0 <= low < high <= 1:
            raise ValueError(f"Invalid Longformer Head threshold band for {task}.")
    if head.get("Summarization", {}).get("verdict") != "NO RELIABLE VERDICT":
        raise ValueError("Longformer Head Summarization must remain NO RELIABLE VERDICT.")
    if head.get("Not sure", {}).get("threshold") != default:
        raise ValueError("Longformer Head Not sure must use the configured default threshold.")
    return payload


def interpret_task_score(
    model_key: str,
    task: str,
    hallucination_score: float,
    config: dict[str, Any],
) -> TaskVerdict:
    """Turn a raw score into the requested UI verdict without changing model output."""
    if task not in TASKS:
        raise ValueError(f"task must be one of {TASKS}; got {task!r}.")
    if not isinstance(hallucination_score, (int, float)) or not math.isfinite(hallucination_score) or not 0 <= hallucination_score <= 1:
        raise ValueError("hallucination_score must be finite and in [0, 1].")

    default = float(config["default_hallucination_threshold"])
    if model_key == "longformer-head" and task == "Summarization":
        return TaskVerdict("NO RELIABLE VERDICT", "no_reliable", "Validation ROC AUC is near chance; no task verdict is issued.", None, True)

    if model_key == "longformer-head" and task in {"QA", "Data-to-text"}:
        rule = config["models"][model_key]["task_rules"][task]
        low = float(rule["supported_below"])
        high = float(rule["hallucinated_at_or_above"])
        if hallucination_score < low:
            return TaskVerdict("LIKELY SUPPORTED", "supported", f"score < {low:.2f}", low)
        if hallucination_score < high:
            return TaskVerdict("UNCERTAIN", "uncertain", f"{low:.2f} ≤ score < {high:.2f}", high)
        return TaskVerdict("LIKELY HALLUCINATED", "hallucinated", f"score ≥ {high:.2f}", high)

    threshold = default
    is_hallucinated = hallucination_score >= threshold
    label = "HALLUCINATED" if is_hallucinated else "SUPPORTED"
    return TaskVerdict(label, "hallucinated" if is_hallucinated else "supported",
                       f"fixed {threshold:.2f} cutoff; uncalibrated", threshold)
