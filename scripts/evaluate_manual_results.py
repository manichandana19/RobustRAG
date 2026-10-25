"""Validate RobustRAG's hand-authored manual set and summarize recorded predictions."""
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

LABELS = {"SUPPORTED", "HALLUCINATED"}
PREDICTION_ALIASES = {"LIKELY SUPPORTED": "SUPPORTED", "LIKELY HALLUCINATED": "HALLUCINATED"}
REVIEW_PREDICTIONS = {"UNCERTAIN", "NO RELIABLE VERDICT"}
TASKS = {"QA", "Summary", "Data2txt"}
RESULT_FIELDS = {
    "case_id", "model", "expected_label", "predicted_label",
    "hallucination_probability_uncalibrated", "pass_fail", "notes",
}


def load_cases(path: Path) -> dict[str, dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not isinstance(payload.get("cases"), list):
        raise ValueError("Case file must be a JSON object containing a cases array.")
    cases: dict[str, dict] = {}
    required = {"case_id", "task_type", "context", "question", "answer", "expected_label", "failure_category", "rationale"}
    for i, case in enumerate(payload["cases"], start=1):
        if not isinstance(case, dict) or not required.issubset(case):
            raise ValueError(f"Case {i} is missing required fields: {sorted(required - set(case or {}))}.")
        case_id = case["case_id"]
        if not isinstance(case_id, str) or not case_id.strip() or case_id in cases:
            raise ValueError(f"Case {i} has an empty or duplicate case_id: {case_id!r}.")
        if case["task_type"] not in TASKS:
            raise ValueError(f"Case {case_id}: task_type must be QA or Summary.")
        warning_only = case.get("label_status") == "no_reliable_verdict" and case["expected_label"] == "NO RELIABLE VERDICT"
        if not warning_only and (not isinstance(case["context"], str) or not case["context"].strip() or not isinstance(case["answer"], str) or not case["answer"].strip()):
            raise ValueError(f"Case {case_id}: context and answer must be nonempty strings.")
        if warning_only and (case["context"] or case["answer"]):
            raise ValueError(f"Case {case_id}: warning-only case must not contain prediction inputs.")
        if case["question"] is not None and not isinstance(case["question"], str):
            raise ValueError(f"Case {case_id}: question must be a string or null.")
        expected = case["expected_label"]
        valid = expected in LABELS
        valid |= expected is None and case.get("label_status") == "ambiguous"
        valid |= expected == "NO RELIABLE VERDICT" and case.get("label_status") == "no_reliable_verdict"
        if not valid:
            raise ValueError(f"Case {case_id}: expected_label must be SUPPORTED, HALLUCINATED, null/ambiguous, or NO RELIABLE VERDICT/no_reliable_verdict.")
        if not isinstance(case["failure_category"], str) or not case["failure_category"].strip() or not isinstance(case["rationale"], str) or not case["rationale"].strip():
            raise ValueError(f"Case {case_id}: failure_category and rationale must be nonempty strings.")
        cases[case_id] = case
    if not cases:
        raise ValueError("Case set is empty.")
    return cases


def _metrics(y_true: list[int], y_pred: list[int]) -> dict:
    tp = sum(a == 1 and b == 1 for a, b in zip(y_true, y_pred))
    tn = sum(a == 0 and b == 0 for a, b in zip(y_true, y_pred))
    fp = sum(a == 0 and b == 1 for a, b in zip(y_true, y_pred))
    fn = sum(a == 1 and b == 0 for a, b in zip(y_true, y_pred))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "n": len(y_true), "accuracy": (tp + tn) / len(y_true) if y_true else None,
        "precision": precision, "recall": recall, "f1": f1,
        "confusion": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},
    }


def summarize(cases: dict[str, dict], results_path: Path) -> dict[str, dict]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    seen: set[tuple[str, str]] = set()
    with results_path.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames is None or not RESULT_FIELDS.issubset(reader.fieldnames):
            raise ValueError(f"Results CSV requires columns: {sorted(RESULT_FIELDS)}.")
        for line, row in enumerate(reader, start=2):
            case_id, model = row["case_id"], row["model"]
            if case_id not in cases:
                raise ValueError(f"Results row {line}: unknown case_id {case_id!r}.")
            if not model:
                raise ValueError(f"Results row {line}: model is required.")
            pair = (case_id, model)
            if pair in seen:
                raise ValueError(f"Results row {line}: duplicate case/model result {pair}.")
            seen.add(pair)
            case = cases[case_id]
            expected = row["expected_label"].strip().upper()
            case_expected = case["expected_label"] or ""
            if expected != case_expected:
                raise ValueError(f"Results row {line}: expected_label does not match case file for {case_id}.")
            predicted = row["predicted_label"].strip().upper()
            case_status = case.get("label_status")
            allowed_predictions = LABELS | set(PREDICTION_ALIASES) | REVIEW_PREDICTIONS
            if predicted not in allowed_predictions:
                raise ValueError(f"Results row {line}: predicted_label is invalid for case {case_id!r}.")
            probability = row["hallucination_probability_uncalibrated"].strip()
            if probability:
                value = float(probability)
                if not math.isfinite(value) or not 0 <= value <= 1:
                    raise ValueError(f"Results row {line}: probability must be finite and in [0, 1].")
            if expected in LABELS:
                if predicted in REVIEW_PREDICTIONS:
                    if row["pass_fail"].strip().upper() not in {"", "REVIEW"}:
                        raise ValueError(f"Results row {line}: non-binary verdicts must be marked REVIEW.")
                else:
                    binary_prediction = PREDICTION_ALIASES.get(predicted, predicted)
                    correct = expected == binary_prediction
                    if row["pass_fail"].strip().upper() not in {"", "PASS", "FAIL"}:
                        raise ValueError(f"Results row {line}: pass_fail must be PASS, FAIL, REVIEW, or blank.")
                    if row["pass_fail"].strip() and row["pass_fail"].strip().upper() != ("PASS" if correct else "FAIL"):
                        raise ValueError(f"Results row {line}: pass_fail disagrees with expected/predicted labels.")
                    grouped[model].append({"expected": expected, "predicted": binary_prediction})
            elif case_status == "no_reliable_verdict":
                if predicted != "NO RELIABLE VERDICT" or row["pass_fail"].strip().upper() not in {"", "NO RELIABLE VERDICT"}:
                    raise ValueError(f"Results row {line}: warning-only case must be recorded as NO RELIABLE VERDICT.")
            elif row["pass_fail"].strip().upper() not in {"", "REVIEW/AMBIGUOUS"}:
                raise ValueError(f"Results row {line}: ambiguous cases must be blank or marked REVIEW/AMBIGUOUS.")
    output = {}
    for model, rows in sorted(grouped.items()):
        y_true = [int(row["expected"] == "HALLUCINATED") for row in rows]
        y_pred = [int(row["predicted"] == "HALLUCINATED") for row in rows]
        output[model] = _metrics(y_true, y_pred)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, default=Path("reports/manual_evaluation/robustrag_manual_set_v1.json"))
    parser.add_argument("--additional-cases", type=Path, help="Optional second case file; IDs must be unique across both files.")
    parser.add_argument("--results", type=Path, help="Completed copy of results_template.csv.")
    parser.add_argument("--validate-only", action="store_true", help="Validate and count cases without reading result rows.")
    args = parser.parse_args()
    cases = load_cases(args.cases)
    if args.additional_cases:
        extra = load_cases(args.additional_cases)
        overlap = cases.keys() & extra.keys()
        if overlap:
            parser.error(f"Case IDs must be unique across files: {sorted(overlap)}")
        cases.update(extra)
    labels = sum(case["expected_label"] in LABELS for case in cases.values())
    ambiguous = sum(case["expected_label"] is None for case in cases.values())
    warning_only = sum(case["expected_label"] == "NO RELIABLE VERDICT" for case in cases.values())
    print(f"PASS: {len(cases)} unique cases; {labels} binary-labeled; {ambiguous} ambiguous/review-only; {warning_only} warning-only.")
    if args.results:
        print(json.dumps(summarize(cases, args.results), indent=2))
    elif not args.validate_only:
        parser.error("Provide --results or use --validate-only.")


if __name__ == "__main__":
    main()
