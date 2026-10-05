from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from hallucination_detector.data.schema import RAGTruthRecord


def evaluate_binary_classifier(
    records: list[RAGTruthRecord],
    predictions: np.ndarray,
    probabilities: np.ndarray,
) -> dict[str, Any]:
    """Evaluate binary predictions using a fixed positive class of 1."""
    y_true = np.asarray(
        [int(record.hallucinated) for record in records]
    )
    y_pred = np.asarray(predictions)
    y_probability = np.asarray(probabilities)[:, 1]

    metrics: dict[str, Any] = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(
            precision_score(y_true, y_pred, zero_division=0)
        ),
        "recall": float(
            recall_score(y_true, y_pred, zero_division=0)
        ),
        "f1": float(
            f1_score(y_true, y_pred, zero_division=0)
        ),
        "roc_auc": float(roc_auc_score(y_true, y_probability)),
        "average_precision": float(
            average_precision_score(y_true, y_probability)
        ),
        "confusion_matrix": confusion_matrix(
            y_true,
            y_pred,
            labels=[0, 1],
        ).tolist(),
        "classification_report": classification_report(
            y_true,
            y_pred,
            labels=[0, 1],
            target_names=["supported", "hallucinated"],
            output_dict=True,
            zero_division=0,
        ),
        "test_records": int(len(records)),
        "positive_rate": float(y_true.mean()),
    }

    return metrics
