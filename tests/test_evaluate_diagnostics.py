import pytest

from scripts.evaluate_diagnostics import (
    aggregate_summary,
    diagnostic_cases,
    prediction_is_correct,
    validate_cases,
)


def test_hand_authored_cases_have_unique_ids_and_valid_expected_labels():
    cases = diagnostic_cases()

    validate_cases(cases)

    assert len({case["case_id"] for case in cases}) == len(cases)
    assert {case["expected_label"] for case in cases} == {"SUPPORTED", "HALLUCINATED"}


@pytest.mark.parametrize("bad_label", [None, "supported", "UNKNOWN"])
def test_validate_cases_rejects_invalid_expected_label(bad_label):
    case = {"case_id": "example", "expected_label": bad_label}

    with pytest.raises(ValueError, match="expected_label"):
        validate_cases([case])


def test_validate_cases_rejects_duplicate_case_ids():
    cases = [
        {"case_id": "duplicate", "expected_label": "SUPPORTED"},
        {"case_id": "duplicate", "expected_label": "HALLUCINATED"},
    ]

    with pytest.raises(ValueError, match="Duplicate"):
        validate_cases(cases)


def test_correctness_calculation_requires_matching_valid_labels():
    assert prediction_is_correct("SUPPORTED", "SUPPORTED") is True
    assert prediction_is_correct("SUPPORTED", "HALLUCINATED") is False
    assert prediction_is_correct("HALLUCINATED", "") is False


def test_summary_aggregation_counts_cases_correct_errors_and_accuracy():
    rows = [
        {"model_key": "m1", "model_name": "Model 1", "correct": True},
        {"model_key": "m1", "model_name": "Model 1", "correct": False},
        {"model_key": "m2", "model_name": "Model 2", "correct": True},
    ]

    summary = aggregate_summary(rows)

    assert summary == [
        {"model_key": "m1", "model_name": "Model 1", "cases": 2,
         "correct_predictions": 1, "errors": 1, "diagnostic_accuracy": 0.5},
        {"model_key": "m2", "model_name": "Model 2", "cases": 1,
         "correct_predictions": 1, "errors": 0, "diagnostic_accuracy": 1.0},
    ]
