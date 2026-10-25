"""Save/load a fitted SentenceSelector so inference matches training.

Training fits IDF statistics on the training split (SentenceSelector.fit_from_records).
The app used an unfitted selector (every word weight 1.0), which ranks context
sentences differently from what the Q+C+A Longformer saw during training.
"""
from __future__ import annotations

import json
from pathlib import Path

from hallucination_detector.data.sentence_selector import SentenceSelector


def save_selector(selector: SentenceSelector, path: str | Path, meta: dict | None = None) -> None:
    payload = {"default_idf": selector.default_idf, "meta": meta or {}, "idf": selector.idf}
    with Path(path).open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, separators=(",", ":"))


def load_selector(path: str | Path) -> SentenceSelector:
    with Path(path).open("r", encoding="utf-8") as f:
        payload = json.load(f)
    return SentenceSelector(idf=payload["idf"], default_idf=payload.get("default_idf", 1.0))
