from dataclasses import dataclass, field
from typing import Any


@dataclass
class RAGTruthRecord:
    id: str
    source_id: str

    question: str
    context: str | dict[str, Any]
    answer: str

    hallucinated: bool = False

    model: str | None = None
    temperature: float | None = None
    split: str = "test"
    quality: str | None = None

    annotations: list[dict[str, Any]] = field(default_factory=list)