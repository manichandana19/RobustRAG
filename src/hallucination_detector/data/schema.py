from dataclasses import dataclass
from typing import Any


@dataclass
class RAGTruthRecord:
    id: str
    source_id: str

    question: str
    context: str
    answer: str

    hallucinated: bool

    model: str | None
    temperature: float | None
    split: str
    quality: str | None

    annotations: list[dict[str, Any]]