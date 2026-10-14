import pytest
import numpy as np
import torch

from hallucination_detector.data.schema import RAGTruthRecord
from hallucination_detector.modeling.longformer_classifier import LongformerClassifier


@pytest.fixture(scope="module")
def classifier():
    model = LongformerClassifier(
        model_name="allenai/longformer-base-4096",
        max_length=512,
        truncation_strategy="head",
    )
    return model


def test_classifier_fp32_initialization(classifier):
    for name, param in classifier.model.named_parameters():
        if param.is_floating_point():
            assert param.dtype == torch.float32, f"Param {name} is not FP32"


def test_enable_gradient_checkpointing(classifier):
    classifier.enable_gradient_checkpointing()
    # Check that model has gradient checkpointing enabled
    assert classifier.model.is_gradient_checkpointing


def test_predict_proba_on_sample_records(classifier):
    records = [
        RAGTruthRecord(
            id="rec_1",
            source_id="src_1",
            question="What is the capital of France?",
            context="France is a country in Europe. Paris is the capital city of France.",
            answer="Paris is the capital of France.",
            hallucinated=False,
            model="gpt-4",
            temperature=0.7,
            split="test",
            quality=None,
            annotations=[],
        ),
        RAGTruthRecord(
            id="rec_2",
            source_id="src_2",
            question="What is the capital of France?",
            context="France is a country in Europe. Paris is the capital city of France.",
            answer="Berlin is the capital of France.",
            hallucinated=True,
            model="gpt-4",
            temperature=0.7,
            split="test",
            quality=None,
            annotations=[],
        ),
    ]

    probs = classifier.predict_proba(records, batch_size=2, device="cpu", bf16=False)
    assert isinstance(probs, np.ndarray)
    assert probs.shape == (2, 2)
    assert np.allclose(probs.sum(axis=1), 1.0, atol=1e-5)

    preds = classifier.predict(records, batch_size=2, device="cpu", bf16=False)
    assert isinstance(preds, np.ndarray)
    assert preds.shape == (2,)
    assert set(preds).issubset({0, 1})
