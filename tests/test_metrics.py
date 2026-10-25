import numpy as np
import pytest

from hallucination_detector.data.schema import RAGTruthRecord
from hallucination_detector.evaluation.metrics import evaluate_binary_classifier, format_metric


def record(label: bool, record_id: str) -> RAGTruthRecord:
    return RAGTruthRecord(
        id=record_id,
        source_id=f"source-{record_id}",
        question="Question",
        context="Evidence",
        answer="Answer",
        hallucinated=label,
    )


def test_evaluate_binary_classifier_returns_threshold_metrics():
    records = [record(False, "1"), record(True, "2")]
    probabilities = np.array([[0.8, 0.2], [0.3, 0.7]])

    metrics = evaluate_binary_classifier(records, np.array([0, 1]), probabilities)

    assert metrics["roc_auc"] == 1.0
    assert metrics["average_precision"] == 1.0
    assert metrics["decision_threshold"] == 0.5
    assert metrics["threshold_metrics"]["confusion_matrix"] == [[1, 0], [0, 1]]


def test_single_class_subset_has_undefined_ranking_metrics_not_an_exception():
    records = [record(False, "1"), record(False, "2")]
    probabilities = np.array([[0.8, 0.2], [0.4, 0.6]])

    metrics = evaluate_binary_classifier(records, np.array([0, 1]), probabilities)

    assert metrics["roc_auc"] is None
    assert metrics["average_precision"] is None
    assert metrics["classification_report"]["supported"]["support"] == 2
    assert format_metric(metrics["roc_auc"]) == "N/A"


@pytest.mark.parametrize(
    "predictions,probabilities",
    [
        (np.array([0]), np.array([[0.5, 0.5], [0.4, 0.6]])),
        (np.array([0, 1]), np.array([[0.5, 0.4], [0.4, 0.6]])),
    ],
)
def test_metrics_rejects_malformed_prediction_inputs(predictions, probabilities):
    records = [record(False, "1"), record(True, "2")]

    with pytest.raises(ValueError):
        evaluate_binary_classifier(records, predictions, probabilities)
