from __future__ import annotations

import sys
from pathlib import Path

# Dynamically resolve project root:
# longformer_train.py -> training (0) -> hallucination_detector (1) -> src (2) -> RobustRAG (3)
PROJECT_ROOT = Path(__file__).resolve().parents[3]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Import from audit_investigation located at project root
import audit_investigation
from audit_investigation import test_records

import json
import math
import random
import shutil
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup

from hallucination_detector.data.loader import load_jsonl
from hallucination_detector.data.schema import RAGTruthRecord
from hallucination_detector.data.torch_dataset import LongformerDataCollator
from hallucination_detector.evaluation.metrics import evaluate_binary_classifier
from hallucination_detector.modeling.longformer_classifier import LongformerClassifier


def set_seed(seed: int) -> None:
    """Set random seeds for reproducible experiments."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_processed_records(processed_path: str | Path) -> list[RAGTruthRecord]:
    """Load processed JSONL rows and validate them against the project schema."""
    raw_records = load_jsonl(processed_path)
    return [RAGTruthRecord(**record) for record in raw_records]


def split_records(
    records: list[RAGTruthRecord],
) -> tuple[list[RAGTruthRecord], list[RAGTruthRecord], list[RAGTruthRecord]]:
    """Use the pre-assigned train/validation/test splits; do not resplit randomly."""
    train_records = [record for record in records if record.split == "train"]
    validation_records = [
        record for record in records if record.split == "validation"
    ]
    test_records = [record for record in records if record.split == "test"]

    if not train_records:
        raise ValueError("No records found with split='train'.")
    if not validation_records:
        raise ValueError("No records found with split='validation'.")
    if not test_records:
        raise ValueError("No records found with split='test'.")

    return train_records, validation_records, test_records


def _autocast_context(device: torch.device, bf16: bool):
    """Enable BF16 autocast only when requested and supported."""
    if bf16 and device.type == "cuda" and torch.cuda.is_bf16_supported():
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)

    return torch.autocast(device_type=device.type, enabled=False)


def _make_loader(
    model: LongformerClassifier,
    records: list[RAGTruthRecord],
    batch_size: int,
    shuffle: bool,
    device: torch.device,
) -> DataLoader:
    dataset = model.build_dataset(records)
    collator = LongformerDataCollator(
        tokenizer=model.tokenizer,
        pad_to_multiple_of=512,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collator,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )


def evaluate_loss(
    model: LongformerClassifier,
    records: list[RAGTruthRecord],
    batch_size: int,
    device: torch.device,
    bf16: bool,
) -> float:
    """Calculate example-weighted mean validation cross-entropy loss."""
    if not records:
        raise ValueError("Cannot evaluate an empty validation set.")

    loader = _make_loader(model, records, batch_size, False, device)
    model.model.to(device)
    model.model.eval()

    total_loss = 0.0
    total_examples = 0

    with torch.no_grad():
        for batch in loader:
            batch = {key: value.to(device) for key, value in batch.items()}

            with _autocast_context(device, bf16):
                outputs = model.model(**batch)

            batch_size_actual = batch["labels"].size(0)
            total_loss += outputs.loss.item() * batch_size_actual
            total_examples += batch_size_actual

    if total_examples == 0:
        raise ValueError("Validation loader produced no examples.")

    return total_loss / total_examples


def train_one_epoch(
    model: LongformerClassifier,
    records: list[RAGTruthRecord],
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    batch_size: int,
    gradient_accumulation_steps: int,
    device: torch.device,
    bf16: bool,
) -> float:
    """Train one epoch, normalizing the accumulation groups correctly."""
    if not records:
        raise ValueError("Cannot train on an empty training set.")
    if gradient_accumulation_steps < 1:
        raise ValueError("gradient_accumulation_steps must be at least 1.")

    loader = _make_loader(model, records, batch_size, True, device)
    model.model.to(device)
    model.model.train()
    optimizer.zero_grad(set_to_none=True)

    total_loss = 0.0
    total_examples = 0
    num_batches = len(loader)

    for step, batch in enumerate(loader):
        batch = {key: value.to(device) for key, value in batch.items()}

        with _autocast_context(device, bf16):
            outputs = model.model(**batch)
            loss = outputs.loss

        batch_size_actual = batch["labels"].size(0)
        total_loss += loss.item() * batch_size_actual
        total_examples += batch_size_actual

        group_start = (step // gradient_accumulation_steps) * gradient_accumulation_steps
        group_size = min(gradient_accumulation_steps, num_batches - group_start)
        (loss / group_size).backward()

        should_update = (
            (step + 1) % gradient_accumulation_steps == 0
            or (step + 1) == num_batches
        )
        if should_update:
            torch.nn.utils.clip_grad_norm_(model.model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)

    if total_examples == 0:
        raise ValueError("Training loader produced no examples.")

    return total_loss / total_examples


def train_longformer(
    processed_path: str | Path,
    output_dir: str | Path,
    config_path: str | Path,
) -> dict[str, Any]:
    processed_path = Path(processed_path)
    output_dir = Path(output_dir)
    config_path = Path(config_path)

    if not processed_path.is_file():
        raise FileNotFoundError(f"Processed data file not found: {processed_path}")
    if not config_path.is_file():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    output_dir.mkdir(parents=True, exist_ok=True)

    with config_path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file) or {}

    project_config = config.get("project", {})
    model_config = config.get("model", {})
    training_config = config.get("training", {})
    experiment_config = config.get("experiment", {})

    seed = int(project_config.get("seed", 42))
    model_name = model_config.get("name", "allenai/longformer-base-4096")
    max_length = int(model_config.get("max_length", 1024))
    truncation_strategy = model_config.get("truncation_strategy", "head")
    global_attention_mode = model_config.get("global_attention_mode", "cls")

    batch_size = int(training_config.get("batch_size", 2))
    gradient_accumulation_steps = int(
        training_config.get("gradient_accumulation_steps", 8)
    )
    learning_rate = float(training_config.get("learning_rate", 2e-5))
    weight_decay = float(training_config.get("weight_decay", 0.01))
    epochs = int(training_config.get("epochs", 3))
    patience = int(training_config.get("early_stopping_patience", 1))
    bf16 = bool(training_config.get("bf16", True))
    gradient_checkpointing = bool(
        training_config.get("gradient_checkpointing", True)
    )
    input_mode = experiment_config.get("input_mode", "ca")

    if batch_size < 1:
        raise ValueError("training.batch_size must be at least 1.")
    if gradient_accumulation_steps < 1:
        raise ValueError("training.gradient_accumulation_steps must be at least 1.")
    if epochs < 1:
        raise ValueError("training.epochs must be at least 1.")
    if patience < 1:
        raise ValueError("training.early_stopping_patience must be at least 1.")

    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if bf16 and (device.type != "cuda" or not torch.cuda.is_bf16_supported()):
        raise RuntimeError(
            "BF16 is enabled, but a BF16-capable CUDA GPU is unavailable. "
            "Set training.bf16 to false in the YAML configuration."
        )

    print(f"Device: {device}")
    print(f"Model: {model_name}")
    print(f"Max length: {max_length}")
    print(f"Truncation strategy: {truncation_strategy}")
    print(f"Global attention mode: {global_attention_mode}")
    print(f"Micro-batch size: {batch_size}")
    print(f"Gradient accumulation steps: {gradient_accumulation_steps}")
    print(f"Effective batch size: {batch_size * gradient_accumulation_steps}")
    print(f"Learning rate: {learning_rate}")
    print(f"Epochs: {epochs}")
    print(f"BF16 autocast: {bf16}")
    print(f"Gradient checkpointing: {gradient_checkpointing}")

    records = load_processed_records(processed_path)
    train_records, validation_records, test_records = split_records(records)

    print(f"Train records: {len(train_records)}")
    print(f"Validation records: {len(validation_records)}")
    print(f"Test records: {len(test_records)}")

    model = LongformerClassifier(
        model_name=model_name,
        max_length=max_length,
        input_mode=input_mode,
        truncation_strategy=truncation_strategy,
        global_attention_mode=global_attention_mode,
    )
    if gradient_checkpointing:
        model.enable_gradient_checkpointing()

    optimizer = torch.optim.AdamW(
        model.model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )

    train_loader_length = math.ceil(len(train_records) / batch_size)
    steps_per_epoch = math.ceil(train_loader_length / gradient_accumulation_steps)
    total_training_steps = steps_per_epoch * epochs
    warmup_steps = int(total_training_steps * 0.1)

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_training_steps,
    )

    best_validation_loss = float("inf")
    best_epoch = 0
    epochs_without_improvement = 0
    history: list[dict[str, float]] = []
    best_model_dir = output_dir / "best_model"

    for epoch in range(1, epochs + 1):
        print(f"\n--- Epoch {epoch}/{epochs} ---")
        train_loss = train_one_epoch(
            model=model,
            records=train_records,
            optimizer=optimizer,
            scheduler=scheduler,
            batch_size=batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            device=device,
            bf16=bf16,
        )

        validation_loss = evaluate_loss(
            model=model,
            records=validation_records,
            batch_size=batch_size,
            device=device,
            bf16=bf16,
        )

        print(
            f"Epoch {epoch}: train_loss={train_loss:.4f}, "
            f"val_loss={validation_loss:.4f}"
        )

        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "validation_loss": validation_loss,
            }
        )

        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
            best_epoch = epoch
            epochs_without_improvement = 0

            print(f"New best validation loss: {best_validation_loss:.4f}. Saving best checkpoint...")
            if best_model_dir.exists():
                shutil.rmtree(best_model_dir)
            model.save_pretrained(best_model_dir)
        else:
            epochs_without_improvement += 1
            print(f"Validation loss did not improve. Patience count: {epochs_without_improvement}/{patience}")

        if epochs_without_improvement >= patience:
            print(f"Early stopping triggered after epoch {epoch}.")
            break

    print(f"\nLoading best checkpoint from epoch {best_epoch} for test set evaluation...")
    best_model = LongformerClassifier.from_pretrained(
        best_model_dir,
        max_length=max_length,
        input_mode=input_mode,
        truncation_strategy=truncation_strategy,
        global_attention_mode=global_attention_mode,
    )

    probabilities = best_model.predict_proba(
        test_records,
        batch_size=batch_size,
        device=device,
        bf16=bf16,
    )
    predictions = np.argmax(probabilities, axis=1)

    y_true = np.array([int(r.hallucinated) for r in test_records])
    y_pred = predictions
    y_prob = probabilities[:, 1]

    # CHANGE THIS OLD BLOCK:
    # metrics = evaluate_binary_classifier(
    #         y_true=y_true,
    #         y_pred=y_pred,
    #         y_prob=y_prob,
    #     )

    # TO THIS CORRECTED BLOCK:
    metrics = evaluate_binary_classifier(
        records=test_records,
        predictions=predictions,
        probabilities=probabilities,
    )



    metrics["best_epoch"] = best_epoch
    metrics["best_validation_loss"] = best_validation_loss
    metrics["bf16_autocast"] = bf16
    metrics["gradient_checkpointing"] = gradient_checkpointing
    metrics["micro_batch_size"] = batch_size
    metrics["gradient_accumulation_steps"] = gradient_accumulation_steps
    metrics["effective_batch_size"] = batch_size * gradient_accumulation_steps
    metrics["truncation_strategy"] = truncation_strategy
    metrics["global_attention_mode"] = global_attention_mode

    # Save metrics
    with (output_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    # Save training history
    with (output_dir / "training_history.json").open("w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    # Save test predictions
    predictions_path = output_dir / "test_predictions.jsonl"
    with predictions_path.open("w", encoding="utf-8") as f:
        for record, pred, prob in zip(test_records, predictions, probabilities[:, 1]):
            row = {
                "id": record.id,
                "source_id": record.source_id,
                "hallucinated": bool(record.hallucinated),
                "prediction": int(pred),
                "probability_hallucinated": float(prob),
            }
            f.write(json.dumps(row) + "\n")

    # Snapshot configuration
    with (output_dir / "config.yaml").open("w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, sort_keys=False)

    summary = {
        "model_name": model_name,
        "input_mode": input_mode,
        "max_length": max_length,
        "truncation_strategy": truncation_strategy,
        "global_attention_mode": global_attention_mode,
        "train_records": len(train_records),
        "validation_records": len(validation_records),
        "test_records": len(test_records),
        "best_epoch": best_epoch,
        "best_validation_loss": best_validation_loss,
        "metrics": metrics,
    }
    with (output_dir / "experiment_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\nExperiment completed successfully.")
    print(f"Artifacts saved to {output_dir}")
    print(f"Test Accuracy: {metrics.get('accuracy', 0):.4f}")
    print(f"Test F1:       {metrics.get('f1', 0):.4f}")
    print(f"Test ROC-AUC:  {metrics.get('roc_auc', 0):.4f}")

    return metrics
