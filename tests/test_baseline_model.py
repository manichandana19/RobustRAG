from hallucination_detector.data.schema import RAGTruthRecord
from hallucination_detector.modeling.tfidf_logistic import (
    TfidfLogisticClassifier,
    build_input_text,
)


def make_record(
    record_id: str,
    context: str,
    answer: str,
    hallucinated: bool,
) -> RAGTruthRecord:
    return RAGTruthRecord(
        id=record_id,
        source_id=f"source-{record_id}",
        question="What happened?",
        context=context,
        answer=answer,
        hallucinated=hallucinated,
        model="test-model",
        temperature=0.7,
        split="train",
        quality="good",
        annotations=[],
    )


def test_ca_input_uses_only_context_and_answer():
    record = make_record(
        "1",
        "Context text.",
        "Answer text.",
        False,
    )

    text = build_input_text(record, input_mode="ca")

    assert "[CONTEXT] Context text." in text
    assert "[ANSWER] Answer text." in text
    assert "What happened?" not in text


def test_classifier_produces_probabilities():
    records = [
        make_record(
            "1",
            "The sky is blue.",
            "The sky is blue.",
            False,
        ),
        make_record(
            "2",
            "The sky is blue.",
            "The sky is green.",
            True,
        ),
        make_record(
            "3",
            "Water freezes at zero degrees.",
            "Water freezes at zero degrees.",
            False,
        ),
        make_record(
            "4",
            "Water freezes at zero degrees.",
            "Water boils at zero degrees.",
            True,
        ),
    ]

    model = TfidfLogisticClassifier(
        input_mode="ca",
        min_df=1,
        max_df=1.0,
    )
    model.fit(records)

    probabilities = model.predict_proba(records)

    assert probabilities.shape == (4, 2)
    assert all(
        abs(float(row.sum()) - 1.0) < 1e-9
        for row in probabilities
    )

def test_classifier_handles_structured_context():
    records = [
        RAGTruthRecord(
            id="1",
            source_id="source-1",
            question="What happened?",
            context={
                "title": "Example",
                "text": "The sky is blue.",
            },
            answer="The sky is blue.",
            hallucinated=False,
            model="test-model",
            temperature=0.7,
            split="train",
            quality="good",
            annotations=[],
        ),
        RAGTruthRecord(
            id="2",
            source_id="source-2",
            question="What happened?",
            context={
                "title": "Example",
                "text": "The sky is blue.",
            },
            answer="The sky is green.",
            hallucinated=True,
            model="test-model",
            temperature=0.7,
            split="train",
            quality="good",
            annotations=[],
        ),
    ]

    model = TfidfLogisticClassifier(
        input_mode="ca",
        min_df=1,
        max_df=1.0,
    )

    model.fit(records)

    probabilities = model.predict_proba(records)

    assert probabilities.shape == (2, 2)