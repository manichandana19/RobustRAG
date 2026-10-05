from dataclasses import dataclass
import json
import re
from typing import Any

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from hallucination_detector.data.schema import RAGTruthRecord


_WHITESPACE_RE = re.compile(r"\s+")


def normalize_text(value: Any) -> str:
    """Convert a field to deterministic text for TF-IDF."""
    if isinstance(value, str):
        text = value
    elif isinstance(value, dict):
        text = json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
        )
    elif isinstance(value, (list, tuple)):
        text = json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
        )
    elif value is None:
        text = ""
    else:
        text = str(value)

    return _WHITESPACE_RE.sub(" ", text).strip()


def build_input_text(
    record: RAGTruthRecord,
    input_mode: str = "ca",
) -> str:
    """Build deterministic text input from the configured fields."""
    if input_mode == "qa":
        fields = [
            ("QUESTION", record.question),
            ("ANSWER", record.answer),
        ]
    elif input_mode == "ca":
        fields = [
            ("CONTEXT", record.context),
            ("ANSWER", record.answer),
        ]
    elif input_mode == "qca":
        fields = [
            ("QUESTION", record.question),
            ("CONTEXT", record.context),
            ("ANSWER", record.answer),
        ]
    else:
        raise ValueError(
            f"Unsupported input_mode '{input_mode}'. "
            "Expected one of: qa, ca, qca."
        )

    return " ".join(
        f"[{name}] {normalize_text(value)}"
        for name, value in fields
    )


def build_texts(
    records: list[RAGTruthRecord],
    input_mode: str = "ca",
) -> list[str]:
    return [
        build_input_text(record, input_mode=input_mode)
        for record in records
    ]


def build_labels(records: list[RAGTruthRecord]) -> list[int]:
    return [int(record.hallucinated) for record in records]


@dataclass
class TfidfLogisticClassifier:
    """TF-IDF + Logistic Regression text classifier."""

    input_mode: str = "ca"
    ngram_range: tuple[int, int] = (1, 2)
    min_df: int = 2
    max_df: float = 0.95
    sublinear_tf: bool = True
    max_iter: int = 1000
    C: float = 1.0
    random_state: int = 42

    def __post_init__(self) -> None:
        self.vectorizer = TfidfVectorizer(
            ngram_range=self.ngram_range,
            min_df=self.min_df,
            max_df=self.max_df,
            sublinear_tf=self.sublinear_tf,
        )
        self.classifier = LogisticRegression(
            C=self.C,
            max_iter=self.max_iter,
            random_state=self.random_state,
        )
        self.is_fitted = False

    def fit(
        self,
        records: list[RAGTruthRecord],
    ) -> "TfidfLogisticClassifier":
        texts = build_texts(records, self.input_mode)
        labels = build_labels(records)

        matrix = self.vectorizer.fit_transform(texts)
        self.classifier.fit(matrix, labels)
        self.is_fitted = True

        return self

    def predict_proba(
        self,
        records: list[RAGTruthRecord],
    ):
        self._check_fitted()

        matrix = self.vectorizer.transform(
            build_texts(records, self.input_mode)
        )

        return self.classifier.predict_proba(matrix)

    def predict(
        self,
        records: list[RAGTruthRecord],
    ):
        self._check_fitted()

        matrix = self.vectorizer.transform(
            build_texts(records, self.input_mode)
        )

        return self.classifier.predict(matrix)

    def _check_fitted(self) -> None:
        if not self.is_fitted:
            raise RuntimeError("Classifier has not been fitted.")