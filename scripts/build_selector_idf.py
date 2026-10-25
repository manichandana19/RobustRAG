"""Fit the QCA SentenceSelector IDF table on the training split and save it.

Run from the repo root:
    python scripts/build_selector_idf.py
Writes configs/qca_selector_idf.json, which the app loads for the Q+C+A model.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from hallucination_detector.data.loader import load_jsonl
from hallucination_detector.data.schema import RAGTruthRecord
from hallucination_detector.data.sentence_selector import SentenceSelector
from hallucination_detector.inference.selector_io import save_selector


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/processed/ragtruth.jsonl")
    ap.add_argument("--out", default="configs/qca_selector_idf.json")
    args = ap.parse_args()

    records = [RAGTruthRecord(**r) for r in load_jsonl(args.data)]
    train = [r for r in records if r.split == "train"]
    selector = SentenceSelector.fit_from_records(train)
    save_selector(
        selector,
        args.out,
        meta={"fitted_on": "split=train", "n_records": len(train), "source": args.data},
    )
    size_mb = Path(args.out).stat().st_size / 1e6
    print(f"Fitted on {len(train)} training records; {len(selector.idf)} terms -> {args.out} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
