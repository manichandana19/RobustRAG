import pickle
from pathlib import Path
from typing import Any

from hallucination_detector.data.schema import RAGTruthRecord
from hallucination_detector.modeling.tfidf_logistic import (
    TfidfLogisticClassifier,
)


def load_baseline(
    model_path: str | Path,
) -> TfidfLogisticClassifier:
    """Load a fitted baseline classifier."""
    with Path(model_path).open("rb") as file:
        bundle: dict[str, Any] = pickle.load(file)

    model = bundle.get("model")

    if not isinstance(model, TfidfLogisticClassifier):
        raise ValueError("Saved artifact does not contain the baseline model.")

    return model


def predict_record(
    model: TfidfLogisticClassifier,
    record: RAGTruthRecord,
) -> dict[str, Any]:
    """Return a binary prediction and hallucination probability."""
    prediction = int(model.predict([record])[0])
    probability = float(model.predict_proba([record])[0, 1])

    return {
        "prediction": prediction,
        "probability_hallucinated": probability,
    }
