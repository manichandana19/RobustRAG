"""Check whether the Streamlit inference path (ModelRegistry.predict) reproduces
the benchmark behaviour on real RAGTruth test records.

Run from the repo root (where data/processed/test.jsonl lives):

    python verify_app_inference.py --model longformer-head --n 120
    python verify_app_inference.py --model longformer-head --n 120 \
        --ref-dir reports/experiments/longformer_1024_head/best_model

README benchmark for longformer-head on the full test set: ROC-AUC 0.8296,
recall 0.6426, accuracy 0.7978 (accuracy is NOT comparable here because this
script samples evenly per task/label cell instead of using the real 35% rate).
"""
from __future__ import annotations

import argparse
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

from hallucination_detector.data.loader import load_jsonl
from hallucination_detector.data.schema import RAGTruthRecord
from hallucination_detector.inference.model_registry import MODEL_CATALOG, ModelRegistry
from hallucination_detector.modeling.tfidf_logistic import normalize_text

LOCAL_DIRS = {
    "longformer-head": "reports/experiments/longformer_1024_head/best_model",
    "longformer-qca": "reports/experiments/longformer_1024_smart_qca/best_model",
    "deberta": "reports/experiments/baseline_deberta/best_model",
}

PASSAGES = (
    "passage 1:The word \u201cGemini\u201d is a Latin word for twins and it is one of the constellations "
    "that actually look like its name would suggest. The first known reference of the Gemini "
    "constellation was in Aristotle\u2019s Meteorologica, over 300 years BC. He mentions that he observed "
    "an occulting a star in Gemini and speaks of observing Jupiter in conjunction with it.\n\n"
    "passage 2:Gemini History. The History of Gemini. As other signs in the zodiac, Gemini is not in the "
    "same position as the constellation of Gemini. In the zodiac, it follows Taurus and takes the third "
    "30 degrees of the zodiacal circle. Gemini is a mutable sign that is preceding the summer, and as "
    "such, it announces change while ruling the time of year when Taurus spring has ended, and life on "
    "Earth is about to change.\n\n"
    "passage 3:Gemini Mythology The Story Behind the Gemini Zodiac Sign. The origins of some constellation "
    "myths are heavily debated. Gemini mythology, luckily, is pretty straightforward. By all accounts, "
    "the twins represented in Gemini mythology are Castor and Pollux of Greek mythology. They share the "
    "same mother, Leda (Greek mythology), but have different fathers. Castor's father is Tyndarus, the "
    "King of Sparta, and Leda's husband.\n\n"
)
PROMPT = (
    "Briefly answer the following question:\nhistory of the word gemini\n"
    "Bear in mind that your response should be strictly based on the following three passages:\n"
    + PASSAGES
    + 'In case the passages do not contain the necessary information to answer the question, '
    'please reply with: "Unable to answer based on given passages."\noutput:'
)
PROBES = {
    "A faithful (labeled SUPPORTED)": (
        'The word "Gemini" is Latin for twins and is used to name one of the constellations. The first '
        "known reference to Gemini was in Aristotle's Meteorologica, written over 300 years BC. In the "
        "zodiac, Gemini is the third sign, following Taurus and taking up 30 degrees of the zodiacal "
        "circle. It is a mutable sign that indicates the end of spring and the coming change of seasons. "
        "The twins represented in Gemini mythology are Castor and Pollux of Greek mythology, who share "
        "the same mother but have different fathers."
    ),
    "B subtle error (labeled HALLUCINATED)": (
        'Based on the given passages, we can gather that the word "Gemini" comes from the Latin language '
        "and refers to twins. It is also one of the constellations that resembles its name. The first "
        "known reference to the Gemini constellation was in Aristotle's Meteorologica, around 300 years "
        "ago. The constellation represents the twins Castor and Pollux in Greek mythology."
    ),
    "C blatant fabrication (should be HALLUCINATED)": (
        'The word "Gemini" comes from an ancient Greek word meaning "fire". It was first named by Galileo '
        "in 1610 AD, and the constellation represents the Roman gods Jupiter and Mars, who were brothers "
        "born in Egypt."
    ),
}


def task_of(question: str) -> str:
    if question.startswith("Instruction:"):
        return "Data2txt"
    if question.startswith("Briefly answer"):
        return "QA"
    return "Summary"


def stratified_sample(records: list[RAGTruthRecord], n: int, seed: int) -> list[RAGTruthRecord]:
    rng = random.Random(seed)
    cells: dict[tuple[str, bool], list[RAGTruthRecord]] = defaultdict(list)
    for r in records:
        cells[(task_of(r.question), bool(r.hallucinated))].append(r)
    per_cell = max(1, n // len(cells))
    sample: list[RAGTruthRecord] = []
    for key in sorted(cells):
        sample += rng.sample(cells[key], min(per_cell, len(cells[key])))
    return sample


def summarize(name: str, y: np.ndarray, p: np.ndarray) -> None:
    pred = p >= 0.5
    pos, neg = y == 1, y == 0
    auc = roc_auc_score(y, p) if pos.any() and neg.any() else float("nan")
    recall = pred[pos].mean() if pos.any() else float("nan")
    spec = (~pred[neg]).mean() if neg.any() else float("nan")
    print(
        f"{name:10s} n={len(y):4d}  AUC={auc:.3f}  recall@0.5={recall:.2f}  "
        f"specificity@0.5={spec:.2f}  mean P(hal) halluc={p[pos].mean():.2f} supported={p[neg].mean():.2f}"
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="longformer-head", choices=list(MODEL_CATALOG))
    ap.add_argument("--n", type=int, default=120, help="records to sample (CPU is slow: ~seconds each)")
    ap.add_argument("--data", default="data/processed/test.jsonl")
    ap.add_argument("--ref-dir", default=None, help="longformer-head only: checkpoint dir for official path")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--legacy-deberta",
        action="store_true",
        help="DeBERTa only: use the old right-truncation (answer can be erased) instead of keeping the answer",
    )
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    local = LOCAL_DIRS.get(args.model)
    if local is not None:
        has_local = (Path(local) / "model.safetensors").exists()
        print(f"[weights] {args.model}: {'LOCAL ' + local if has_local else 'NOT local -> Hugging Face download'}")
    print(f"[device] {device}\n")

    registry = ModelRegistry(deberta_preserve_answer=not args.legacy_deberta)

    print("=== Probe: Gemini example through the app path ===")
    context = normalize_text({"passages": PASSAGES, "question": "history of the word gemini"})
    for label, answer in PROBES.items():
        res = registry.predict(args.model, question=PROMPT, context=context, answer=answer)
        print(f"  P(hal)={res.hallucination_probability:.3f}  {label}")

    if not Path(args.data).exists():
        raise SystemExit(
            f"{args.data} not found. Extract data.zip into the repo root first "
            "(PowerShell: Expand-Archive data.zip -DestinationPath . -Force)."
        )
    records = [RAGTruthRecord(**r) for r in load_jsonl(args.data)]
    sample = stratified_sample(records, args.n, args.seed)
    print(f"\n=== Real test records through the app path (stratified, n={len(sample)}) ===")

    probs = []
    for i, r in enumerate(sample, 1):
        res = registry.predict(
            args.model, question=r.question, context=normalize_text(r.context), answer=r.answer
        )
        probs.append(res.hallucination_probability)
        if i % 20 == 0:
            print(f"  ...{i}/{len(sample)}")
    p = np.asarray(probs)
    y = np.asarray([int(r.hallucinated) for r in sample])

    tasks = np.asarray([task_of(r.question) for r in sample])

    def sweep(label: str, mask: np.ndarray) -> None:
        print(f"Threshold sweep [{label}] (recall = hallucinations flagged, specificity = supported left alone):")
        for t in (0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50, 0.60, 0.70):
            flagged = p[mask] >= t
            ym = y[mask]
            if (ym == 1).any() and (ym == 0).any():
                print(f"  t={t:.2f}  recall={flagged[ym == 1].mean():.2f}  specificity={(~flagged[ym == 0]).mean():.2f}")
        print()

    summarize("ALL", y, p)
    for t in ("QA", "Summary", "Data2txt"):
        summarize(t, y[tasks == t], p[tasks == t])
    print()
    for t in ("QA", "Summary", "Data2txt"):
        sweep(t, tasks == t)

    import csv

    tag = "_legacy" if args.legacy_deberta else ""
    out_csv = f"probs_{args.model}{tag}_{Path(args.data).stem}.csv"
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["id", "task", "label", "p_hallucinated"])
        for r, tk, pp in zip(sample, tasks, p):
            w.writerow([r.id, tk, int(r.hallucinated), f"{pp:.5f}"])
    print(f"Saved per-record probabilities to {out_csv}\n")

    if args.ref_dir:
        from hallucination_detector.modeling.longformer_classifier import LongformerClassifier

        print("\n=== Official evaluation path vs app path (same records) ===")
        ref = LongformerClassifier.from_pretrained(
            args.ref_dir,
            max_length=1024,
            input_mode="ca",  # eval_check.py passes "cls" here, which raises ValueError
            truncation_strategy="head",
            global_attention_mode="cls",
        )
        ref_p = ref.predict_proba(sample, batch_size=2, device=device)[:, 1]
        print(f"  max |app - reference| = {np.abs(ref_p - p).max():.4f}")
        print(f"  mean |app - reference| = {np.abs(ref_p - p).mean():.4f}")
        summarize("REFERENCE", y, ref_p)


if __name__ == "__main__":
    main()
