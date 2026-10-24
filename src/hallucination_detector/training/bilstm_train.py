from __future__ import annotations

import json
import math
import random
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import yaml

from hallucination_detector.data.loader import load_jsonl
from hallucination_detector.data.schema import RAGTruthRecord
from hallucination_detector.data.bilstm_dataset import (
    Vocabulary,
    RAGTruthBiLSTMDataset,
    BiLSTMDataCollator,
)
from hallucination_detector.evaluation.metrics import evaluate_binary_classifier
from hallucination_detector.modeling.bilstm_classifier import BiLSTMClassifier


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
    """Load processed JSONL rows and validate them against RAGTruthRecord."""
    raw_records = load_jsonl(processed_path)
    return [RAGTruthRecord(**record) for record in raw_records]


def split_records(
    records: list[RAGTruthRecord],
) -> tuple[list[RAGTruthRecord], list[RAGTruthRecord], list[RAGTruthRecord]]:
    """Use the pre-assigned train/validation/test splits."""
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


def train_bilstm(
    config: dict[str, Any],
    dataset_path: str | Path | None = None,
    output_dir: str | Path | None = None,
    device_override: str | None = None,
) -> dict[str, Any]:
    """Train BiLSTM (+ Attention) baseline model."""
    project_cfg = config.get("project", {})
    model_cfg = config.get("bilstm_model", {})
    train_cfg = config.get("bilstm_training", {})

    seed = project_cfg.get("seed", 42)
    set_seed(seed)

    data_path = Path(dataset_path or config.get("data", {}).get("processed_file", "data/processed/ragtruth.jsonl"))
    out_dir = Path(output_dir or config.get("data", {}).get("output_dir", "models/bilstm_baseline"))
    out_dir.mkdir(parents=True, exist_ok=True)

    records = load_processed_records(data_path)
    train_records, val_records, test_records = split_records(records)

    device_str = device_override or ("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_str)

    # 1. Build Vocabulary from training records
    vocab_size = model_cfg.get("vocab_size", 50000)
    min_freq = model_cfg.get("min_freq", 2)
    vocab = Vocabulary.build_from_records(train_records, max_size=vocab_size, min_freq=min_freq)

    # 2. Build Classifier wrapper
    classifier = BiLSTMClassifier(
        vocab=vocab,
        embed_dim=model_cfg.get("embed_dim", 128),
        hidden_dim=model_cfg.get("hidden_dim", 128),
        num_layers=model_cfg.get("num_layers", 2),
        dropout=model_cfg.get("dropout", 0.3),
        use_attention=model_cfg.get("use_attention", True),
        attention_dim=model_cfg.get("attention_dim", 128),
        max_length=model_cfg.get("max_length", 512),
        truncation_strategy=model_cfg.get("truncation_strategy", "context_tail"),
    )

    assert classifier.net is not None
    classifier.net.to(device)

    # 3. Datasets & Loaders
    collator = BiLSTMDataCollator(pad_id=vocab.pad_id)
    batch_size = train_cfg.get("batch_size", 32)

    train_dataset = classifier.build_dataset(train_records)
    val_dataset = classifier.build_dataset(val_records)
    test_dataset = classifier.build_dataset(test_records)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collator,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collator,
    )

    # 4. Optimization
    lr = float(train_cfg.get("learning_rate", 1e-3))
    weight_decay = float(train_cfg.get("weight_decay", 1e-4))
    epochs = int(train_cfg.get("epochs", 5))
    grad_clip = float(train_cfg.get("grad_clip", 1.0))

    # Compute positive class weight to handle slight class imbalance
    num_pos = sum(r.hallucinated for r in train_records)
    num_neg = len(train_records) - num_pos
    pos_weight = float(num_neg / max(1, num_pos))
    weight_tensor = torch.tensor([1.0, pos_weight if train_cfg.get("use_class_weight", False) else 1.0], device=device)
    criterion = nn.CrossEntropyLoss(weight=weight_tensor)

    optimizer = torch.optim.AdamW(classifier.net.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=1
    )

    best_val_f1 = -1.0
    best_weights_path = out_dir / "best_model.pt"

    for epoch in range(1, epochs + 1):
        classifier.net.train()
        total_loss = 0.0

        for batch in train_loader:
            optimizer.zero_grad()
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            lengths = batch["lengths"].to(device)
            labels = batch["labels"].to(device)

            logits, _ = classifier.net(input_ids, attention_mask, lengths=lengths)
            loss = criterion(logits, labels)
            loss.backward()

            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(classifier.net.parameters(), grad_clip)

            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / max(1, len(train_loader))

        # Evaluate on validation
        val_probs = classifier.predict_proba(val_records, batch_size=batch_size, device=device)
        val_preds = np.argmax(val_probs, axis=1)
        val_metrics = evaluate_binary_classifier(val_records, val_preds, val_probs)
        val_f1 = val_metrics["f1"]

        scheduler.step(val_f1)
        print(f"Epoch {epoch:02d}/{epochs:02d} | Train Loss: {avg_loss:.4f} | Val F1: {val_f1:.4f} | Val Acc: {val_metrics['accuracy']:.4f}")

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save(classifier.net.state_dict(), best_weights_path)


    # Load best checkpoint for final evaluation
    if best_weights_path.exists():
        classifier.net.load_state_dict(torch.load(best_weights_path, map_location=device))

    # Evaluate on test set
    test_probs = classifier.predict_proba(test_records, batch_size=batch_size, device=device)
    test_preds = np.argmax(test_probs, axis=1)
    test_metrics = evaluate_binary_classifier(test_records, test_preds, test_probs)

    # Save model artifacts
    classifier.save(out_dir)

    with (out_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(test_metrics, f, indent=2)

    with (out_dir / "config.yaml").open("w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, sort_keys=False)

    print(f"\n[Done] Training complete. Artifacts saved to: {out_dir}")
    print(f"Test Accuracy : {test_metrics['accuracy']:.4f}")
    print(f"Test F1 Score : {test_metrics['f1']:.4f}")
    print(f"Test ROC-AUC  : {test_metrics['roc_auc']:.4f}")
    print(f"Test Avg Prec : {test_metrics['average_precision']:.4f}")

    return test_metrics


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Train BiLSTM + Attention baseline on RAGTruth.")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/bilstm_attention.yaml",
        help="Path to YAML configuration file.",
    )
    parser.add_argument(
        "--data",
        type=str,
        default=None,
        help="Override path to processed RAGTruth JSONL file.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Override output directory for checkpoints and metrics.",
    )
    args = parser.parse_args()

    cfg_path = Path(args.config)
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config file not found: {cfg_path}")

    with cfg_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    print(f"[BiLSTM Baseline] Loading config from {cfg_path}")
    train_bilstm(
        config=cfg,
        dataset_path=args.data,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()

