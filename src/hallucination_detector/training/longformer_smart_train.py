from __future__ import annotations

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
from hallucination_detector.data.sentence_selector import SentenceSelector
from hallucination_detector.data.torch_dataset import LongformerDataCollator
from hallucination_detector.data.longformer_smart_dataset import (
    RAGTruthSmartLongformerDataset,
)
from hallucination_detector.evaluation.metrics import evaluate_binary_classifier
from hallucination_detector.modeling.longformer_classifier import LongformerClassifier


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_processed_records(processed_path: str | Path) -> list[RAGTruthRecord]:
    raw_records = load_jsonl(processed_path)
    return [RAGTruthRecord(**record) for record in raw_records]


def split_records(
    records: list[RAGTruthRecord],
) -> tuple[list[RAGTruthRecord], list[RAGTruthRecord], list[RAGTruthRecord]]:
    train_records = [record for record in records if record.split == "train"]
    validation_records = [record for record in records if record.split == "validation"]
    test_records = [record for record in records if record.split == "test"]

    if not train_records:
        raise ValueError("No records found with split='train'.")
    if not validation_records:
        raise ValueError("No records found with split='validation'.")
    if not test_records:
        raise ValueError("No records found with split='test'.")

    return train_records, validation_records, test_records


def _autocast_context(device: torch.device, bf16: bool):
    if bf16 and device.type == "cuda" and torch.cuda.is_bf16_supported():
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return torch.autocast(device_type=device.type, enabled=False)


def _make_smart_loader(
    model: LongformerClassifier,
    records: list[RAGTruthRecord],
    batch_size: int,
    shuffle: bool,
    device: torch.device,
    input_mode: Literal["ca", "qca"],
    truncation_strategy: Literal["head", "tail", "smart_selector"],
    global_attention_mode: Literal["cls", "cls_and_qa", "cls_and_answer", "none"],
    selector: SentenceSelector | None,
) -> DataLoader:
    dataset = RAGTruthSmartLongformerDataset(
        records=records,
        tokenizer=model.tokenizer,
        input_mode=input_mode,
        max_length=model.max_length,
        truncation_strategy=truncation_strategy,
        global_attention_mode=global_attention_mode,
        selector=selector,
    )
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


def evaluate_smart_loss(
    model: LongformerClassifier,
    records: list[RAGTruthRecord],
    batch_size: int,
    device: torch.device,
    bf16: bool,
    input_mode: Literal["ca", "qca"],
    truncation_strategy: Literal["head", "tail", "smart_selector"],
    global_attention_mode: Literal["cls", "cls_and_qa", "cls_and_answer", "none"],
    selector: SentenceSelector | None,
) -> float:
    loader = _make_smart_loader(
        model=model,
        records=records,
        batch_size=batch_size,
        shuffle=False,
        device=device,
        input_mode=input_mode,
        truncation_strategy=truncation_strategy,
        global_attention_mode=global_attention_mode,
        selector=selector,
    )
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

    return total_loss / max(1, total_examples)


def predict_smart_proba(
    model: LongformerClassifier,
    records: list[RAGTruthRecord],
    batch_size: int,
    device: torch.device,
    bf16: bool,
    input_mode: Literal["ca", "qca"],
    truncation_strategy: Literal["head", "tail", "smart_selector"],
    global_attention_mode: Literal["cls", "cls_and_qa", "cls_and_answer", "none"],
    selector: SentenceSelector | None,
) -> np.ndarray:
    loader = _make_smart_loader(
        model=model,
        records=records,
        batch_size=batch_size,
        shuffle=False,
        device=device,
        input_mode=input_mode,
        truncation_strategy=truncation_strategy,
        global_attention_mode=global_attention_mode,
        selector=selector,
    )
    model.model.to(device)
    model.model.eval()

    all_probs = []
    with torch.no_grad():
        for batch in loader:
            batch = {key: value.to(device) for key, value in batch.items()}
            with _autocast_context(device, bf16):
                outputs = model.model(**batch)
            probs = torch.softmax(outputs.logits, dim=-1).to(torch.float32).cpu().numpy()
            all_probs.append(probs)

    if not all_probs:
        return np.empty((0, 2), dtype=np.float32)
    return np.vstack(all_probs)


def train_smart_longformer(
    processed_path: str | Path,
    output_dir: str | Path,
    config_path: str | Path,
) -> dict[str, Any]:
    processed_path = Path(processed_path)
    output_dir = Path(output_dir)
    config_path = Path(config_path)

    output_dir.mkdir(parents=True, exist_ok=True)

    with config_path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file) or {}

    project_config = config.get("project", {})
    model_config = config.get("model", {})
    training_config = config.get("training", {})
    experiment_config = config.get("experiment", {})
    nlp_config = config.get("nlp_preprocessing", {})

    seed = int(project_config.get("seed", 42))
    model_name = model_config.get("name", "allenai/longformer-base-4096")
    max_length = int(model_config.get("max_length", 1024))
    truncation_strategy = model_config.get("truncation_strategy", "smart_selector")
    global_attention_mode = model_config.get("global_attention_mode", "cls")
    input_mode = experiment_config.get("input_mode", "qca")

    batch_size = int(training_config.get("batch_size", 2))
    gradient_accumulation_steps = int(training_config.get("gradient_accumulation_steps", 8))
    learning_rate = float(training_config.get("learning_rate", 2e-5))
    weight_decay = float(training_config.get("weight_decay", 0.01))
    epochs = int(training_config.get("epochs", 3))
    patience = int(training_config.get("early_stopping_patience", 1))
    bf16 = bool(training_config.get("bf16", True))
    gradient_checkpointing = bool(training_config.get("gradient_checkpointing", True))

    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Device: {device}")
    print(f"Model: {model_name}")
    print(f"Input Mode: {input_mode}")
    print(f"Max length: {max_length}")
    print(f"Truncation: {truncation_strategy}")
    print(f"Global Attention: {global_attention_mode}")

    records = load_processed_records(processed_path)
    train_records, val_records, test_records = split_records(records)

    # Fit SentenceSelector strictly on training split
    selector = None
    if truncation_strategy == "smart_selector":
        print("[SentenceSelector] Fitting IDF statistics on training records only...")
        selector = SentenceSelector.fit_from_records(train_records)

    model = LongformerClassifier(
        model_name=model_name,
        max_length=max_length,
        input_mode="ca",  # underlying class placeholder; dataset handles qca
        truncation_strategy="head",
        global_attention_mode="cls",
    )
    if gradient_checkpointing:
        model.enable_gradient_checkpointing()

    optimizer = torch.optim.AdamW(model.model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    train_loader = _make_smart_loader(
        model=model,
        records=train_records,
        batch_size=batch_size,
        shuffle=True,
        device=device,
        input_mode=input_mode,
        truncation_strategy=truncation_strategy,
        global_attention_mode=global_attention_mode,
        selector=selector,
    )

    num_batches = len(train_loader)
    steps_per_epoch = math.ceil(num_batches / gradient_accumulation_steps)
    total_steps = steps_per_epoch * epochs
    warmup_steps = int(total_steps * 0.1)

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    best_val_loss = float("inf")
    best_model_dir = output_dir / "best_model"
    epochs_no_improve = 0

    for epoch in range(1, epochs + 1):
        print(f"\n--- Epoch {epoch}/{epochs} ---")
        model.model.to(device)
        model.model.train()
        optimizer.zero_grad(set_to_none=True)

        total_loss = 0.0
        total_examples = 0

        for step, batch in enumerate(train_loader):
            batch = {k: v.to(device) for k, v in batch.items()}
            with _autocast_context(device, bf16):
                outputs = model.model(**batch)
                loss = outputs.loss

            b_actual = batch["labels"].size(0)
            total_loss += loss.item() * b_actual
            total_examples += b_actual

            group_start = (step // gradient_accumulation_steps) * gradient_accumulation_steps
            group_size = min(gradient_accumulation_steps, num_batches - group_start)
            (loss / group_size).backward()

            if (step + 1) % gradient_accumulation_steps == 0 or (step + 1) == num_batches:
                torch.nn.utils.clip_grad_norm_(model.model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

        train_loss = total_loss / max(1, total_examples)
        val_loss = evaluate_smart_loss(
            model=model,
            records=val_records,
            batch_size=batch_size,
            device=device,
            bf16=bf16,
            input_mode=input_mode,
            truncation_strategy=truncation_strategy,
            global_attention_mode=global_attention_mode,
            selector=selector,
        )

        print(f"Epoch {epoch}: Train Loss = {train_loss:.4f} | Val Loss = {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            print(f"Saving new best checkpoint (Val Loss: {best_val_loss:.4f})")
            if best_model_dir.exists():
                shutil.rmtree(best_model_dir)
            model.save_pretrained(best_model_dir)
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"Early stopping triggered at epoch {epoch}")
                break

    # Final evaluation using best model checkpoint
    print("\nLoading best model checkpoint for test set evaluation...")
    best_classifier = LongformerClassifier.from_pretrained(best_model_dir, max_length=max_length)

    test_probs = predict_smart_proba(
        model=best_classifier,
        records=test_records,
        batch_size=batch_size,
        device=device,
        bf16=bf16,
        input_mode=input_mode,
        truncation_strategy=truncation_strategy,
        global_attention_mode=global_attention_mode,
        selector=selector,
    )
    test_preds = np.argmax(test_probs, axis=1)
    metrics = evaluate_binary_classifier(test_records, test_preds, test_probs)

    with (output_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    with (output_dir / "config.yaml").open("w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, sort_keys=False)

    print("\n[Evaluation Complete]")
    print(f"Test Accuracy : {metrics['accuracy']:.4f}")
    print(f"Test F1       : {metrics['f1']:.4f}")
    print(f"Test ROC-AUC  : {metrics['roc_auc']:.4f}")
    print(f"Test Avg Prec : {metrics['average_precision']:.4f}")

    return metrics


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Train Smart NLP + Longformer detector.")
    parser.add_argument("--config", type=str, default="configs/experiments/longformer_qca_smart.yaml")
    parser.add_argument("--data", type=str, default="data/processed/ragtruth.jsonl")
    parser.add_argument("--output-dir", type=str, default="reports/experiments/longformer_1024_smart_qca")
    args = parser.parse_args()

    # Training is commented out to use the saved model checkpoint
    # train_smart_longformer(
    #     processed_path=args.data,
    #     output_dir=args.output_dir,
    #     config_path=args.config,
    # )

    # Evaluation using saved model checkpoint
    config_path = Path(args.config)
    output_dir = Path(args.output_dir)
    best_model_dir = output_dir / "best_model"

    print(f"\n--- Running evaluation on existing checkpoint ---")
    print(f"Loading config from: {config_path}")
    with config_path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file) or {}

    model_config = config.get("model", {})
    training_config = config.get("training", {})
    experiment_config = config.get("experiment", {})

    max_length = int(model_config.get("max_length", 1024))
    truncation_strategy = model_config.get("truncation_strategy", "smart_selector")
    global_attention_mode = model_config.get("global_attention_mode", "cls")
    input_mode = experiment_config.get("input_mode", "qca")
    batch_size = int(training_config.get("batch_size", 2))
    bf16 = bool(training_config.get("bf16", True))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Loading best checkpoint from: {best_model_dir}")

    best_classifier = LongformerClassifier.from_pretrained(best_model_dir, max_length=max_length)

    records = load_processed_records(args.data)
    train_records, val_records, test_records = split_records(records)

    selector = None
    if truncation_strategy == "smart_selector":
        print("[SentenceSelector] Fitting IDF statistics on training records...")
        selector = SentenceSelector.fit_from_records(train_records)

    print(f"Running evaluation on {len(test_records)} test records...")
    test_probs = predict_smart_proba(
        model=best_classifier,
        records=test_records,
        batch_size=batch_size,
        device=device,
        bf16=bf16,
        input_mode=input_mode,
        truncation_strategy=truncation_strategy,
        global_attention_mode=global_attention_mode,
        selector=selector,
    )

    test_preds = np.argmax(test_probs, axis=1)
    metrics = evaluate_binary_classifier(test_records, test_preds, test_probs)

    metrics_path = output_dir / "metrics.json"
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print(f"\n[Evaluation Complete - Saved to {metrics_path}]")
    print(f"Test Accuracy : {metrics['accuracy']:.4f}")
    print(f"Test F1       : {metrics['f1']:.4f}")
    print(f"Test ROC-AUC  : {metrics['roc_auc']:.4f}")
    print(f"Test Avg Prec : {metrics['average_precision']:.4f}")


if __name__ == "__main__":
    main()
