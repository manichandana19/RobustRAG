import pytest
import torch

from hallucination_detector.data.schema import RAGTruthRecord
from hallucination_detector.modeling.longformer_classifier import LongformerClassifier
from hallucination_detector.training.longformer_train import (
    train_one_epoch,
    evaluate_loss,
)


@pytest.fixture(scope="module")
def sample_records():
    records = []
    for i in range(4):
        records.append(
            RAGTruthRecord(
                id=f"rec_{i}",
                source_id=f"src_{i}",
                question="What is the opening hour?",
                context="The coffee shop opens at 8am and closes at 6pm daily.",
                answer=f"The shop opens at {8 if i % 2 == 0 else 10}am.",
                hallucinated=(i % 2 != 0),
                model="gpt-4",
                temperature=0.7,
                split="train" if i < 2 else "validation",
                quality=None,
                annotations=[],
            )
        )
    return records


def test_train_and_eval_step_smoke(sample_records):
    model = LongformerClassifier(
        model_name="allenai/longformer-base-4096",
        max_length=128,
        truncation_strategy="head",
    )
    optimizer = torch.optim.AdamW(model.model.parameters(), lr=1e-5)

    class DummyScheduler:
        def step(self):
            pass

    scheduler = DummyScheduler()

    # Test training step
    train_loss = train_one_epoch(
        model=model,
        records=sample_records[:2],
        optimizer=optimizer,
        scheduler=scheduler,
        batch_size=2,
        gradient_accumulation_steps=1,
        device=torch.device("cpu"),
        bf16=False,
    )
    assert isinstance(train_loss, float)
    assert train_loss > 0.0

    # Test evaluation step
    eval_loss = evaluate_loss(
        model=model,
        records=sample_records[2:],
        batch_size=2,
        device=torch.device("cpu"),
        bf16=False,
    )
    assert isinstance(eval_loss, float)
    assert eval_loss > 0.0
