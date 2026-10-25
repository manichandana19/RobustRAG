from __future__ import annotations

import json
import math
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from transformers import get_linear_schedule_with_warmup

from hallucination_detector.data.qca_artifacts import (
    PROVENANCE_FILENAME,
    QCA_ARTIFACT_TYPE,
    QCA_PROVENANCE_SCHEMA,
    canonical_json_sha256,
    load_and_verify_qca_provenance,
    sha256_file,
)
from hallucination_detector.data.sentence_selector import SentenceSelector
from hallucination_detector.modeling.longformer_classifier import LongformerClassifier
from hallucination_detector.training.longformer_smart_train import (
    TRAINING_STATE,
    _atomic_json,
    _atomic_torch_save,
    _package_versions,
    _publish_best_checkpoint,
    recover_checkpoint_publication,
    _resolve_base_model,
    _run_signature,
    _to_cpu,
    _make_smart_loader,
    evaluate_smart_loss,
    load_train_validation_archive,
    set_seed,
)


def _capture_rng_state() -> dict[str, Any]:
    return {
        "python_rng_state": random.getstate(),
        "numpy_rng_state": np.random.get_state(),
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_state": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def _restore_rng_state(state: dict[str, Any]) -> None:
    random.setstate(state["python_rng_state"])
    np.random.set_state(state["numpy_rng_state"])
    torch.set_rng_state(state["torch_rng_state"])
    if torch.cuda.is_available() and state.get("cuda_rng_state") is not None:
        torch.cuda.set_rng_state_all(state["cuda_rng_state"])


def train_qca(
    archive_path: str | Path,
    output_dir: str | Path,
    config_path: str | Path,
    *,
    resume: bool = False,
    device_name: str | None = None,
    batch_size_override: int | None = None,
    gradient_accumulation_override: int | None = None,
) -> dict[str, Any]:
    """Fit and package QCA using only the archive's train and validation members."""
    from hallucination_detector.training import longformer_smart_train as helpers

    archive_path = Path(archive_path)
    output_dir = Path(output_dir)
    config_path = Path(config_path)
    if output_dir.exists() and not resume:
        raise FileExistsError(f"Output directory already exists: {output_dir}; use a new path or --resume.")
    if resume and not output_dir.is_dir():
        raise FileNotFoundError(f"Cannot resume; output directory does not exist: {output_dir}")
    if batch_size_override is not None and batch_size_override <= 0:
        raise ValueError("batch size must be positive.")
    if gradient_accumulation_override is not None and gradient_accumulation_override <= 0:
        raise ValueError("gradient accumulation must be positive.")

    config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    project_config = config.get("project", {})
    model_config = config.get("model", {})
    training_config = config.setdefault("training", {})
    experiment_config = config.get("experiment", {})
    nlp_config = config.get("nlp_preprocessing", {})

    if experiment_config.get("input_mode", "qca") != "qca":
        raise ValueError("QCA pipeline requires experiment.input_mode='qca'.")
    if model_config.get("truncation_strategy", "smart_selector") != "smart_selector":
        raise ValueError("QCA pipeline requires model.truncation_strategy='smart_selector'.")
    if nlp_config.get("selector_type", "idf_relevance_contradiction") != "idf_relevance_contradiction":
        raise ValueError("Unsupported QCA selector_type; expected 'idf_relevance_contradiction'.")
    if float(nlp_config.get("alpha", 0.4)) != 0.4 or nlp_config.get("lead_bias", True) is not True:
        raise ValueError("This selector implementation supports alpha=0.4 and lead_bias=true only.")

    seed = int(project_config.get("seed", 42))
    model_name = str(model_config.get("name", "allenai/longformer-base-4096"))
    revision = str(model_config.get("revision", "main"))
    resume_manifest = None
    resolve_revision = revision
    if resume:
        resume_manifest_path = output_dir / PROVENANCE_FILENAME
        try:
            resume_manifest = json.loads(resume_manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError("Resume refused: existing run provenance is missing or malformed.") from exc
        prior_base = resume_manifest.get("base_model", {})
        if prior_base.get("id") != model_name or prior_base.get("requested_revision") != revision:
            raise ValueError("Resume refused: configured base model or requested revision changed.")
        resolve_revision = str(prior_base.get("resolved_revision", revision))
    max_length = int(model_config.get("max_length", 1024))
    max_length = int(training_config.get("max_length_override", max_length))
    if max_length <= 0 or max_length > 1024:
        raise ValueError("QCA max_length must be between 1 and 1024 for this RTX 4050 pipeline.")
    batch_size = int(batch_size_override or training_config.get("batch_size", 1))
    accumulation = int(
        gradient_accumulation_override
        or training_config.get("gradient_accumulation_steps", 16)
    )
    training_config["batch_size"] = batch_size
    training_config["gradient_accumulation_steps"] = accumulation
    model_config["max_length"] = max_length
    device = torch.device(device_name or ("cuda:0" if torch.cuda.is_available() else "cpu"))
    runtime_info = {
        "device": str(device),
        "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" and torch.cuda.is_available() else "cpu",
        "bf16_supported": bool(device.type == "cuda" and torch.cuda.is_bf16_supported()),
    }
    config["runtime"] = runtime_info
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(f"CUDA was requested ({device}), but this PyTorch build has no CUDA device.")
        free_bytes, total_bytes = torch.cuda.mem_get_info(device)
        print(
            f"GPU preflight: {torch.cuda.get_device_name(device)}; "
            f"{free_bytes / 1024**3:.2f} GiB free / {total_bytes / 1024**3:.2f} GiB total"
        )
        if total_bytes <= 8 * 1024**3 and batch_size != 1:
            raise ValueError("For GPUs with 8 GiB or less, use batch size 1 and gradient accumulation.")

    train_records, validation_records, dataset_info = load_train_validation_archive(archive_path)
    base_path, resolved_revision, base_local_hash = _resolve_base_model(model_name, resolve_revision)
    packages = _package_versions()
    source_path = Path(__file__).resolve()
    run_signature = _run_signature(
        dataset_info, config, config_path, source_path, model_name,
        resolved_revision, base_local_hash, packages,
    )
    device_config = {**runtime_info, "max_length": max_length}
    preprocessing = {
        "input_mode": "qca",
        "max_length": max_length,
        "truncation_strategy": "smart_selector",
        "global_attention_mode": model_config.get("global_attention_mode", "cls"),
        "selector_type": nlp_config.get("selector_type", "idf_relevance_contradiction"),
        "selector_alpha": float(nlp_config.get("alpha", 0.4)),
        "selector_lead_bias": bool(nlp_config.get("lead_bias", True)),
    }

    if output_dir.exists():
        manifest_path = output_dir / PROVENANCE_FILENAME
        state_path = output_dir / TRAINING_STATE
        if not manifest_path.is_file() or not state_path.is_file():
            raise ValueError("Existing run is not resume-safe: provenance or epoch checkpoint is missing.")
        run_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if run_manifest.get("run_signature") != run_signature:
            raise ValueError("Resume refused: run signature differs in data, config, code, base model, or packages.")
        if run_manifest.get("status") == "complete":
            raise ValueError("This training run is already complete; choose a new output directory.")
        if run_manifest.get("dataset") != dataset_info:
            raise ValueError("Resume refused: dataset or split hashes differ.")
        recover_checkpoint_publication(output_dir)
        selector_path = output_dir / "sentence_selector.json"
        expected_hash = run_manifest.get("selector", {}).get("sha256")
        if not selector_path.is_file() or sha256_file(selector_path) != expected_hash:
            raise ValueError("Resume refused: saved fitted selector is missing or has changed.")
        selector = SentenceSelector.load(selector_path)
    else:
        output_dir.parent.mkdir(parents=True, exist_ok=True)
        output_dir.mkdir()
        selector = SentenceSelector.fit_from_records(train_records)
        selector.save(output_dir / "sentence_selector.json")
        selector_hash = sha256_file(output_dir / "sentence_selector.json")
        run_manifest = {
            "schema_version": QCA_PROVENANCE_SCHEMA,
            "artifact_type": QCA_ARTIFACT_TYPE,
            "run_signature": run_signature,
            "status": "in_progress",
            "dataset": dataset_info,
            "seed": seed,
            "config": config,
            "config_sha256": sha256_file(config_path),
            "preprocessing": preprocessing,
            "base_model": {
                "id": model_name,
                "requested_revision": revision,
                "resolved_revision": resolved_revision,
                "local_artifact_sha256": base_local_hash or None,
            },
            "package_versions": packages,
            "runtime": device_config,
            "selector": {
                "sha256": selector_hash,
                "train_split_sha256": dataset_info["train_sha256"],
                "fit_record_count": len(train_records),
            },
            "training_source_sha256": sha256_file(source_path),
            "training_helpers_sha256": sha256_file(Path(helpers.__file__).resolve()),
        }
        _atomic_json(output_dir / PROVENANCE_FILENAME, run_manifest)

    print(
        f"Device={device}; model={model_name}@{resolved_revision}; train={len(train_records)}; "
        f"validation={len(validation_records)}; test split is not opened."
    )
    set_seed(seed)
    classifier = LongformerClassifier(
        model_name=base_path,
        max_length=max_length,
        input_mode="ca",  # Existing wrapper placeholder; the smart dataset constructs Q+C+A.
        truncation_strategy="head",
        global_attention_mode="cls",
    )
    gradient_checkpointing = bool(training_config.get("gradient_checkpointing", True))
    if gradient_checkpointing:
        classifier.enable_gradient_checkpointing()
    classifier.model.to(device)

    learning_rate = float(training_config.get("learning_rate", 2e-5))
    weight_decay = float(training_config.get("weight_decay", 0.01))
    epochs = int(training_config.get("epochs", 3))
    patience = int(training_config.get("early_stopping_patience", 1))
    bf16 = bool(training_config.get("bf16", True))
    optimizer = torch.optim.AdamW(classifier.model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    train_loader = _make_smart_loader(
        classifier, train_records, batch_size, True, device, "qca", "smart_selector",
        preprocessing["global_attention_mode"], selector,
    )
    batches_per_epoch = len(train_loader)
    steps_per_epoch = math.ceil(batches_per_epoch / accumulation)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(steps_per_epoch * epochs * 0.1),
        num_training_steps=steps_per_epoch * epochs,
    )

    state_path = output_dir / TRAINING_STATE
    start_epoch = 1
    best_epoch = 0
    best_validation_loss = float("inf")
    epochs_no_improve = 0
    if resume:
        state = torch.load(state_path, map_location="cpu", weights_only=False)
        if state.get("run_signature") != run_signature:
            raise ValueError("Resume checkpoint signature mismatch.")
        classifier.model.load_state_dict(state["model_state_dict"])
        optimizer.load_state_dict(state["optimizer_state_dict"])
        scheduler.load_state_dict(state["scheduler_state_dict"])
        start_epoch = int(state["epoch"]) + 1
        best_epoch = int(state["best_epoch"])
        best_validation_loss = float(state["best_validation_loss"])
        epochs_no_improve = int(state["epochs_no_improve"])
        _restore_rng_state(state["rng_state"])
        print(f"Resuming after completed epoch {start_epoch - 1}.")

    training_metadata = {
        "dataset": dataset_info,
        "seed": seed,
        "config": config,
        "config_sha256": sha256_file(config_path),
        "preprocessing": preprocessing,
        "base_model": run_manifest["base_model"],
        "package_versions": packages,
        "training_source_sha256": sha256_file(source_path),
        "training_helpers_sha256": sha256_file(Path(helpers.__file__).resolve()),
    }
    for epoch in range(start_epoch, epochs + 1):
        classifier.model.train()
        optimizer.zero_grad(set_to_none=True)
        running_loss = 0.0
        count = 0
        for step, batch in enumerate(train_loader):
            try:
                batch = {key: value.to(device) for key, value in batch.items()}
                if bf16 and device.type == "cuda" and torch.cuda.is_bf16_supported():
                    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                        outputs = classifier.model(**batch)
                else:
                    outputs = classifier.model(**batch)
                actual = int(batch["labels"].size(0))
                running_loss += float(outputs.loss.detach().item()) * actual
                count += actual
                group_start = (step // accumulation) * accumulation
                group_size = min(accumulation, batches_per_epoch - group_start)
                (outputs.loss / group_size).backward()
                if (step + 1) % accumulation == 0 or step + 1 == batches_per_epoch:
                    torch.nn.utils.clip_grad_norm_(classifier.model.parameters(), 1.0)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad(set_to_none=True)
            except torch.cuda.OutOfMemoryError as exc:
                optimizer.zero_grad(set_to_none=True)
                torch.cuda.empty_cache()
                last_safe = state_path if state_path.exists() else "no completed-epoch resume checkpoint"
                raise RuntimeError(
                    f"CUDA out of memory during QCA epoch {epoch}; gradients were cleared and cache released. "
                    f"Last safe state: {last_safe}. This isolated run keeps its prior best checkpoint."
                ) from exc

        train_loss = running_loss / max(count, 1)
        try:
            validation_loss = evaluate_smart_loss(
                classifier, validation_records, batch_size, device, bf16, "qca", "smart_selector",
                preprocessing["global_attention_mode"], selector,
            )
        except torch.cuda.OutOfMemoryError as exc:
            torch.cuda.empty_cache()
            raise RuntimeError(
                f"CUDA out of memory during validation after epoch {epoch}; "
                f"the last completed-epoch state is {state_path if state_path.exists() else 'not yet available'}."
            ) from exc
        print(f"Epoch {epoch}/{epochs}: train_loss={train_loss:.6f} val_loss={validation_loss:.6f}")
        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
            best_epoch = epoch
            epochs_no_improve = 0
            hashes = _publish_best_checkpoint(
                classifier, selector, output_dir, training_metadata, run_signature,
                epoch, validation_loss,
            )
            run_manifest["best_checkpoint"] = {
                "path": "best_model", "epoch": epoch,
                "validation_loss": validation_loss, "artifact_hashes": hashes,
            }
        else:
            epochs_no_improve += 1

        state = {
            "run_signature": run_signature,
            "epoch": epoch,
            "best_epoch": best_epoch,
            "best_validation_loss": best_validation_loss,
            "epochs_no_improve": epochs_no_improve,
            "model_state_dict": _to_cpu(classifier.model.state_dict()),
            "optimizer_state_dict": _to_cpu(optimizer.state_dict()),
            "scheduler_state_dict": scheduler.state_dict(),
            "rng_state": _capture_rng_state(),
        }
        _atomic_torch_save(state_path, state)
        run_manifest["last_completed_epoch"] = epoch
        _atomic_json(output_dir / PROVENANCE_FILENAME, run_manifest)
        if epochs_no_improve >= patience:
            print(f"Validation-loss early stopping after epoch {epoch}.")
            break

    best_dir = output_dir / "best_model"
    if not best_dir.is_dir():
        raise RuntimeError("No validation-selected checkpoint was saved.")
    best_manifest = load_and_verify_qca_provenance(best_dir)
    run_manifest["status"] = "complete"
    run_manifest["best_checkpoint"] = {
        "path": "best_model",
        "epoch": best_manifest["training"]["best_epoch"],
        "validation_loss": best_manifest["training"]["best_validation_loss"],
        "artifact_hashes": best_manifest["artifact_hashes"],
    }
    _atomic_json(output_dir / PROVENANCE_FILENAME, run_manifest)
    state_path.unlink(missing_ok=True)
    return {
        "output_dir": str(output_dir),
        "best_model_dir": str(best_dir),
        "best_epoch": best_manifest["training"]["best_epoch"],
        "best_validation_loss": best_manifest["training"]["best_validation_loss"],
        "train_records": len(train_records),
        "validation_records": len(validation_records),
        "test_split_opened": False,
        "run_signature": run_signature,
    }


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Train a provenance-packaged RobustRAG Longformer QCA candidate.")
    parser.add_argument("--archive", default="data.zip")
    parser.add_argument("--config", default="configs/experiments/longformer_qca_smart.yaml")
    parser.add_argument("--output-dir", required=True, help="New isolated run directory; existing paths are refused.")
    parser.add_argument("--device", default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    result = train_qca(
        args.archive, args.output_dir, args.config,
        resume=args.resume,
        device_name=args.device,
        batch_size_override=args.batch_size,
        gradient_accumulation_override=args.gradient_accumulation_steps,
    )
    print(json.dumps(result, indent=2))

