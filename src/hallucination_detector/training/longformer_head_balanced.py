"""Isolated, validation-selected task-balanced Longformer Head candidate run."""
from __future__ import annotations

import csv
import hashlib
import json
import random
import platform
from collections import Counter
from pathlib import Path
from typing import Any
from zipfile import ZipFile

import numpy as np
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from torch.utils.data import DataLoader, WeightedRandomSampler
from transformers import get_linear_schedule_with_warmup

from hallucination_detector.data.schema import RAGTruthRecord
from hallucination_detector.data.torch_dataset import LongformerDataCollator
from hallucination_detector.modeling.longformer_classifier import LongformerClassifier

TASKS = ("Data2txt", "QA", "Summary")
SUBTYPES = ("supported", "conflict", "baseless", "mixed")
FOCUS_SUBTYPES = ("conflict", "baseless")
BASELINE_REVISION = "cf0c280635afbd1fd1baa7f5899042f1a4470d5b"


def _annotation_group(record: RAGTruthRecord) -> str:
    kinds = {str(a.get("label_type", "")).casefold() for a in record.annotations}
    conflict = any("conflict" in kind for kind in kinds)
    baseless = any("baseless info" in kind for kind in kinds)
    if conflict and baseless:
        return "mixed"
    if conflict:
        return "conflict"
    if baseless:
        return "baseless"
    return "supported"


def read_train_validation(archive: Path) -> tuple[list[RAGTruthRecord], list[RAGTruthRecord], dict[str, str], dict[str, str]]:
    """Read only train/validation records plus source task metadata; never open test.jsonl."""
    with ZipFile(archive) as zf:
        source_rows = [json.loads(line) for line in zf.open("data/raw/source_info.jsonl") if line.strip()]
        task_by_source = {str(row["source_id"]): str(row.get("task_type", "unknown")) for row in source_rows}
        raw: dict[str, list[dict[str, Any]]] = {}
        hashes: dict[str, str] = {}
        for split in ("train", "validation"):
            data = zf.read(f"data/processed/{split}.jsonl")
            hashes[f"data/processed/{split}.jsonl"] = hashlib.sha256(data).hexdigest()
            raw[split] = [json.loads(line) for line in data.splitlines() if line.strip()]
    train = [RAGTruthRecord(**row) for row in raw["train"]]
    validation = [RAGTruthRecord(**row) for row in raw["validation"]]
    if not train or not validation:
        raise ValueError("Both train and validation members must be nonempty.")
    if any(r.split != "train" for r in train) or any(r.split != "validation" for r in validation):
        raise ValueError("Archive member labels do not match their train/validation split names.")
    train_sources = {r.source_id for r in train}
    validation_sources = {r.source_id for r in validation}
    overlap = train_sources & validation_sources
    if overlap:
        raise ValueError(f"Source leakage between train and validation: {len(overlap)} source IDs overlap.")
    task_by_id = {r.id: task_by_source.get(r.source_id, "unknown") for r in train + validation}
    if any(task_by_id[r.id] not in TASKS for r in train + validation):
        raise ValueError("Missing or unknown source task_type in archive metadata.")
    return train, validation, task_by_id, hashes


def task_balanced_weights(records: list[RAGTruthRecord], task_by_id: dict[str, str]) -> tuple[torch.DoubleTensor, dict[str, int]]:
    """QA/Summary receive 80% of draws, balanced over supported/conflict/baseless/mixed strata.

    Data2txt receives the remaining 20%, preserving its training-only within-task mix.
    Each source remains in its original split; sampling is over training records only.
    """
    strata: dict[tuple[str, str], list[int]] = {}
    by_task: Counter[str] = Counter()
    for i, record in enumerate(records):
        task = task_by_id[record.id]
        strata.setdefault((task, _annotation_group(record)), []).append(i)
        by_task[task] += 1
    targets: dict[tuple[str, str], float] = {}
    for task in ("QA", "Summary"):
        for subtype in SUBTYPES:
            key = (task, subtype)
            if key not in strata:
                raise ValueError(f"Training split lacks required balanced stratum {key}.")
            targets[key] = 0.4 / len(SUBTYPES)
    if by_task["Data2txt"] == 0:
        raise ValueError("Training split has no Data2txt records.")
    weights = torch.zeros(len(records), dtype=torch.double)
    for key, indices in strata.items():
        task, subtype = key
        group_mass = targets[key] if task in ("QA", "Summary") else 0.2 * len(indices) / by_task["Data2txt"]
        per_record = group_mass / len(indices)
        weights[indices] = per_record
    counts = {f"{task}/{subtype}": len(indices) for (task, subtype), indices in sorted(strata.items())}
    return weights, counts


def binary_metrics(y_true: list[int], y_pred: list[int]) -> dict[str, Any]:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel().tolist()
    return {
        "n": len(y_true),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "confusion": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }


def validation_metrics(records: list[RAGTruthRecord], probabilities: np.ndarray, task_by_id: dict[str, str]) -> dict[str, Any]:
    if probabilities.shape != (len(records), 2) or not np.isfinite(probabilities).all():
        raise ValueError("Validation probabilities have an invalid shape or non-finite values.")
    if np.any(probabilities < 0) or np.any(probabilities > 1) or not np.allclose(probabilities.sum(axis=1), 1.0, atol=0.002):
        raise ValueError("Validation probabilities must be in [0, 1] and sum to one.")
    y = np.asarray([int(r.hallucinated) for r in records])
    pred = (probabilities[:, 1] >= 0.5).astype(int)
    report: dict[str, Any] = {"threshold": 0.5, "overall": binary_metrics(y.tolist(), pred.tolist()), "by_task": {}, "by_task_subtype": {}}
    groups = {"supported": [], "conflict": [], "baseless": [], "mixed": []}
    for i, record in enumerate(records):
        groups[_annotation_group(record)].append(i)
    for task in TASKS:
        indices = [i for i, r in enumerate(records) if task_by_id[r.id] == task]
        if not indices:
            continue
        report["by_task"][task] = binary_metrics(y[indices].tolist(), pred[indices].tolist())
        for subtype in ("conflict", "baseless", "mixed"):
            pos = [i for i in groups[subtype] if task_by_id[records[i].id] == task]
            if not pos:
                continue
            # Each subtype is evaluated against that task's supported records.
            subset = [i for i in groups["supported"] if task_by_id[records[i].id] == task] + pos
            report["by_task_subtype"][f"{task}/{subtype}"] = binary_metrics(y[subset].tolist(), pred[subset].tolist())
    focus_keys = [f"{task}/{subtype}" for task in ("QA", "Summary") for subtype in FOCUS_SUBTYPES]
    missing_focus = [key for key in focus_keys if key not in report["by_task_subtype"]]
    if missing_focus:
        raise ValueError(f"Validation data lacks required QA/Summary annotation strata: {missing_focus}")
    focus = [report["by_task_subtype"][key]["f1"] for key in focus_keys]
    report["focus_macro_f1_qa_summary_conflict_baseless"] = float(np.mean(focus))
    report["validation_loss"] = float(-np.mean(y * np.log(np.clip(probabilities[:, 1], 1e-7, 1)) + (1 - y) * np.log(np.clip(probabilities[:, 0], 1e-7, 1))))
    return report


def frozen_baseline_metrics(path: Path, records: list[RAGTruthRecord], task_by_id: dict[str, str]) -> dict[str, Any]:
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    by_id = {row["record_id"]: row for row in rows}
    if len(by_id) != len(rows) or set(by_id) != {r.id for r in records}:
        raise ValueError("Frozen baseline validation CSV IDs do not exactly match this validation split.")
    for record in records:
        row = by_id[record.id]
        expected = "HALLUCINATED" if record.hallucinated else "SUPPORTED"
        if row["true_label"] != expected or row["task_type"] != task_by_id[record.id]:
            raise ValueError(f"Frozen baseline labels/tasks disagree with validation record {record.id}.")
    probs = np.asarray([[float(by_id[r.id]["supported_probability"]), float(by_id[r.id]["hallucination_probability"])] for r in records])
    if not np.isfinite(probs).all() or np.any(probs < 0) or np.any(probs > 1):
        raise ValueError("Frozen baseline CSV contains invalid probabilities.")
    if not np.allclose(probs.sum(axis=1), 1.0, atol=0.002):
        raise ValueError("Frozen baseline class probabilities do not sum to one.")
    for record, probability in zip(records, probs):
        expected_prediction = "HALLUCINATED" if probability[1] >= 0.5 else "SUPPORTED"
        if by_id[record.id]["predicted_label"] != expected_prediction:
            raise ValueError(f"Frozen baseline prediction does not match the 0.5 threshold for {record.id}.")
    return validation_metrics(records, probs, task_by_id)


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def train_candidate(
    archive: Path,
    init_model_dir: Path,
    output_dir: Path,
    baseline_predictions: Path,
    *,
    epochs: int = 3,
    batch_size: int = 1,
    accumulation: int = 16,
    learning_rate: float = 1e-5,
    seed: int = 42,
    max_train_steps: int | None = None,
) -> dict[str, Any]:
    """Train from a local frozen Longformer Head bundle; only train/validation members are opened."""
    if output_dir.exists():
        raise FileExistsError(f"Candidate output already exists; refusing to overwrite: {output_dir}")
    if not archive.is_file() or not init_model_dir.is_dir():
        raise FileNotFoundError("Archive or local initialization checkpoint is missing; downloads are disabled.")
    if not (init_model_dir / "model.safetensors").is_file():
        raise FileNotFoundError(f"Initialization bundle lacks model.safetensors: {init_model_dir}")
    if epochs < 1 or batch_size < 1 or accumulation < 1:
        raise ValueError("epochs, batch_size and accumulation must be positive.")
    if max_train_steps is not None and max_train_steps < 1:
        raise ValueError("max_train_steps must be positive when provided.")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if not torch.cuda.is_available():
        raise RuntimeError("This experiment requires the requested CUDA GPU; CPU fallback is disabled.")
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("RTX 4050 BF16 support is required by this conservative run configuration.")
    device = torch.device("cuda:0")
    free, total = torch.cuda.mem_get_info(device)
    print(f"GPU preflight: {torch.cuda.get_device_name(device)}; {free / 2**30:.2f}/{total / 2**30:.2f} GiB free")
    train_records, validation_records, task_by_id, data_hashes = read_train_validation(archive)
    weights, sampling_counts = task_balanced_weights(train_records, task_by_id)
    baseline = frozen_baseline_metrics(baseline_predictions, validation_records, task_by_id)
    model = LongformerClassifier.from_pretrained(init_model_dir, max_length=1024, input_mode="ca", truncation_strategy="head", global_attention_mode="cls")
    model.enable_gradient_checkpointing()
    model.model.to(device)
    optimizer = torch.optim.AdamW(model.model.parameters(), lr=learning_rate, weight_decay=0.01)
    torch.cuda.reset_peak_memory_stats(device)
    dataset = model.build_dataset(train_records)
    collator = LongformerDataCollator(tokenizer=model.tokenizer, pad_to_multiple_of=512)
    steps_per_epoch = (len(train_records) + batch_size - 1) // batch_size
    updates_per_epoch = (steps_per_epoch + accumulation - 1) // accumulation
    planned_updates = updates_per_epoch * epochs
    total_updates = min(planned_updates, max_train_steps) if max_train_steps is not None else planned_updates
    scheduler = get_linear_schedule_with_warmup(optimizer, int(total_updates * 0.1), total_updates)
    generator = torch.Generator().manual_seed(seed)
    best_score = -1.0
    best_loss = float("inf")
    history: list[dict[str, Any]] = []
    optimizer_steps = 0
    peak_train_memory_bytes = 0
    output_dir.mkdir(parents=True)
    best_prediction_path = output_dir / "candidate_validation_predictions.csv"
    for epoch in range(1, epochs + 1):
        sampler = WeightedRandomSampler(weights, num_samples=len(train_records), replacement=True, generator=generator)
        loader = DataLoader(dataset, batch_size=batch_size, sampler=sampler, collate_fn=collator, num_workers=0, pin_memory=True)
        model.model.train()
        optimizer.zero_grad(set_to_none=True)
        train_loss_sum = 0.0
        batches_seen = 0
        for step, batch in enumerate(loader):
            batch = {key: value.to(device, non_blocking=True) for key, value in batch.items()}
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                loss = model.model(**batch).loss
            group_start = (step // accumulation) * accumulation
            group_count = min(accumulation, len(loader) - group_start)
            (loss / group_count).backward()
            train_loss_sum += float(loss.detach())
            batches_seen += 1
            if (step + 1) % accumulation == 0 or step + 1 == len(loader):
                torch.nn.utils.clip_grad_norm_(model.model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                optimizer_steps += 1
                if max_train_steps is not None and optimizer_steps >= max_train_steps:
                    break
        peak_train_memory_bytes = max(peak_train_memory_bytes, int(torch.cuda.max_memory_allocated(device)))
        probabilities = model.predict_proba(validation_records, batch_size=1, device=device, bf16=True)
        metrics = validation_metrics(validation_records, probabilities, task_by_id)
        entry = {"epoch": epoch, "train_loss": train_loss_sum / max(1, batches_seen), **metrics}
        history.append(entry)
        print(f"epoch={epoch} train_loss={entry['train_loss']:.5f} val_focus_macro_f1={metrics['focus_macro_f1_qa_summary_conflict_baseless']:.5f} val_loss={metrics['validation_loss']:.5f}")
        score, val_loss = metrics["focus_macro_f1_qa_summary_conflict_baseless"], metrics["validation_loss"]
        if score > best_score or (score == best_score and val_loss < best_loss):
            best_score, best_loss = score, val_loss
            model.save_pretrained(output_dir / "best_model")
            (output_dir / "best_validation_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
            with best_prediction_path.open("w", newline="", encoding="utf-8") as stream:
                fields = ["record_id", "source_id", "task_type", "annotation_group", "true_label", "predicted_label", "supported_probability", "hallucination_probability"]
                writer = csv.DictWriter(stream, fieldnames=fields)
                writer.writeheader()
                for record, probability, prediction in zip(validation_records, probabilities, (probabilities[:, 1] >= 0.5)):
                    writer.writerow({
                        "record_id": record.id, "source_id": record.source_id,
                        "task_type": task_by_id[record.id], "annotation_group": _annotation_group(record),
                        "true_label": "HALLUCINATED" if record.hallucinated else "SUPPORTED",
                        "predicted_label": "HALLUCINATED" if prediction else "SUPPORTED",
                        "supported_probability": float(probability[0]),
                        "hallucination_probability": float(probability[1]),
                    })
        if max_train_steps is not None and optimizer_steps >= max_train_steps:
            break
    (output_dir / "training_history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    summary = {
        "candidate": "longformer-head-task-balanced-v1", "split_access": ["train", "validation"],
        "baseline_revision": BASELINE_REVISION, "initial_model_dir": str(init_model_dir),
        "initial_weights_sha256": _sha256_file(init_model_dir / "model.safetensors"),
        "dataset_member_sha256": data_hashes, "train_records": len(train_records),
        "validation_records": len(validation_records), "train_sampling_strata": sampling_counts,
        "sampler_task_weights": {"QA": 0.4, "Summary": 0.4, "Data2txt": 0.2},
        "threshold": 0.5, "selection_metric": "QA/Summary conflict/baseless macro F1; ties by validation log loss",
        "baseline_validation_metrics": baseline, "best_validation_metrics": json.loads((output_dir / "best_validation_metrics.json").read_text(encoding="utf-8")),
        "epochs": epochs, "batch_size": batch_size, "gradient_accumulation": accumulation,
        "learning_rate": learning_rate, "seed": seed, "max_length": 1024, "bf16": True,
        "gradient_checkpointing": True,
        "max_train_steps": max_train_steps, "optimizer_steps_completed": optimizer_steps,
        "cuda_peak_memory_bytes": peak_train_memory_bytes,
    }
    (output_dir / "experiment_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    import sklearn
    import transformers
    package_versions = {"python": platform.python_version(), "torch": torch.__version__, "transformers": transformers.__version__, "numpy": np.__version__, "scikit_learn": sklearn.__version__}
    artifact_hashes = {
        str(path.relative_to(output_dir)): _sha256_file(path)
        for path in sorted(output_dir.rglob("*")) if path.is_file()
    }
    provenance = {
        "candidate": "longformer-head-task-balanced-v1", "dataset_members": data_hashes,
        "training_code_sha256": _sha256_file(Path(__file__)), "base_model_revision": BASELINE_REVISION,
        "base_weights_sha256": summary["initial_weights_sha256"], "output_artifacts_sha256": artifact_hashes,
        "training_config": {"epochs": epochs, "batch_size": batch_size, "gradient_accumulation": accumulation, "learning_rate": learning_rate, "seed": seed, "max_length": 1024, "bf16": True, "gradient_checkpointing": True, "max_train_steps": max_train_steps},
        "package_versions": package_versions, "split_access": ["train", "validation"],
    }
    (output_dir / "provenance.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    return summary
