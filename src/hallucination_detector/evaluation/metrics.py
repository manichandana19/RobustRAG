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


def format_metric(value: float | None, precision: int = 4) -> str:
    """Format optional ranking metrics without turning undefined values into scores."""
    return "N/A" if value is None else f"{value:.{precision}f}"


def evaluate_binary_classifier(
    records: list[RAGTruthRecord],
    predictions: np.ndarray,
    probabilities: np.ndarray,
    decision_threshold: float = 0.5,
) -> dict[str, Any]:
    """Evaluate binary predictions (class 1 = hallucinated).

    ROC-AUC and average precision are returned as ``None`` for single-class
    subsets, where ranking performance is undefined. ``threshold_metrics``
    reports a transparent, fixed-threshold view without tuning that threshold.
    """
    if not records:
        raise ValueError("Cannot evaluate an empty record set.")
    if not 0.0 <= decision_threshold <= 1.0:
        raise ValueError("decision_threshold must be between 0 and 1.")

    y_true = np.asarray(
        [int(record.hallucinated) for record in records]
    )
    y_pred = np.asarray(predictions)
    y_probabilities = np.asarray(probabilities, dtype=float)
    if y_pred.shape != (len(records),):
        raise ValueError(
            f"predictions must have shape ({len(records)},), got {y_pred.shape}."
        )
    if y_probabilities.shape != (len(records), 2):
        raise ValueError(
            "probabilities must have shape "
            f"({len(records)}, 2), got {y_probabilities.shape}."
        )
    if not np.isin(y_pred, [0, 1]).all():
        raise ValueError("predictions must contain only class IDs 0 and 1.")
    if not np.isfinite(y_probabilities).all():
        raise ValueError("probabilities must be finite.")
    if ((y_probabilities < 0.0) | (y_probabilities > 1.0)).any():
        raise ValueError("probabilities must be between 0 and 1.")
    if not np.allclose(y_probabilities.sum(axis=1), 1.0, atol=1e-4):
        raise ValueError("Each probability row must sum to 1.")

    y_probability = y_probabilities[:, 1]
    threshold_predictions = (y_probability >= decision_threshold).astype(int)
    has_both_classes = np.unique(y_true).size == 2

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
        "roc_auc": (
            float(roc_auc_score(y_true, y_probability))
            if has_both_classes else None
        ),
        "average_precision": (
            float(average_precision_score(y_true, y_probability))
            if has_both_classes else None
        ),
        "decision_threshold": float(decision_threshold),
        "threshold_metrics": {
            "accuracy": float(accuracy_score(y_true, threshold_predictions)),
            "precision": float(precision_score(y_true, threshold_predictions, zero_division=0)),
            "recall": float(recall_score(y_true, threshold_predictions, zero_division=0)),
            "f1": float(f1_score(y_true, threshold_predictions, zero_division=0)),
            "confusion_matrix": confusion_matrix(
                y_true, threshold_predictions, labels=[0, 1]
            ).tolist(),
        },
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
