"""Turn a raw hallucination probability into an honest, task-aware verdict.

Thresholds are measured, not assumed (see configs/thresholds.json). Models or
tasks without measured thresholds fall back to a plain 0.5 cutoff and say so.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

TASKS = ("QA", "Summarization", "Data-to-text")
NOT_SURE = "Not sure"


@dataclass(frozen=True)
class Verdict:
    label: str
    level: str  # "ok" | "warn" | "bad" | "info"
    explanation: str


def load_thresholds(path: str | Path) -> dict[str, Any]:
    try:
        with Path(path).open("r", encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def classify(
    p_hallucinated: float,
    model_key: str,
    task: str,
    thresholds: dict[str, Any],
) -> Verdict:
    entry = thresholds.get(model_key, {}).get(task)

    if entry is None:
        if p_hallucinated >= 0.5:
            return Verdict(
                "HALLUCINATED (uncalibrated)",
                "bad",
                "Plain 0.5 cutoff. No measured thresholds exist for this model/task, "
                "and this model's scores are known to be poorly calibrated.",
            )
        return Verdict(
            "SUPPORTED (uncalibrated)",
            "ok",
            "Plain 0.5 cutoff. No measured thresholds exist for this model/task, so this "
            "may miss hallucinations: scores below 0.5 are common even for hallucinated answers.",
        )

    if not entry.get("reliable", True):
        return Verdict("NO RELIABLE VERDICT", "info", entry.get("note", ""))

    if p_hallucinated >= entry["likely_from"]:
        return Verdict("LIKELY HALLUCINATED", "bad", entry.get("note", ""))
    if p_hallucinated >= entry["uncertain_from"]:
        return Verdict("UNCERTAIN - possible hallucination", "warn", entry.get("note", ""))
    return Verdict(
        "LIKELY SUPPORTED",
        "ok",
        "Low score, but not proof of faithfulness. " + entry.get("note", ""),
    )
