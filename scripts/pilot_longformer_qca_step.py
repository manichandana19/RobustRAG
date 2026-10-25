"""Measure a few isolated QCA training microsteps without saving a checkpoint."""
from __future__ import annotations

import json
from pathlib import Path
import sys
import time

import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]

from hallucination_detector.data.sentence_selector import SentenceSelector  # noqa: E402
from hallucination_detector.modeling.longformer_classifier import LongformerClassifier  # noqa: E402
from hallucination_detector.training.longformer_qca_pipeline import load_train_validation_archive  # noqa: E402
from hallucination_detector.training.longformer_smart_train import _make_smart_loader, _resolve_base_model  # noqa: E402


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; refusing to run a GPU timing pilot.")
    config = yaml.safe_load((ROOT / "configs/experiments/longformer_qca_smart.yaml").read_text(encoding="utf-8"))
    model_cfg = config["model"]
    train_cfg = config["training"]
    device = torch.device("cuda:0")
    train_records, _, split_info = load_train_validation_archive(ROOT / "data.zip")
    selector = SentenceSelector.fit_from_records(train_records)
    base_path, revision, local_hash = _resolve_base_model(model_cfg["name"], model_cfg.get("revision", "main"))
    model = LongformerClassifier(model_name=base_path, max_length=model_cfg["max_length"], input_mode="ca",
                                 truncation_strategy="head", global_attention_mode="cls")
    model.enable_gradient_checkpointing()
    model.model.to(device).train()
    loader = _make_smart_loader(model, train_records[:3], 1, False, device, "qca", "smart_selector", "cls", selector)
    optimizer = torch.optim.AdamW(model.model.parameters(), lr=float(train_cfg["learning_rate"]))
    bf16 = bool(train_cfg.get("bf16")) and torch.cuda.is_bf16_supported()
    torch.cuda.reset_peak_memory_stats(device)
    timings, losses = [], []
    started = time.perf_counter()
    for batch in loader:
        step_started = time.perf_counter()
        batch = {key: value.to(device) for key, value in batch.items()}
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=bf16):
            loss = model.model(**batch).loss
        loss.backward()
        optimizer.step()
        torch.cuda.synchronize(device)
        timings.append(time.perf_counter() - step_started)
        losses.append(float(loss.detach().cpu()))
    report = {
        "pilot": "three real train-split QCA microsteps; weights not saved",
        "split": "train only", "training_records_available": len(train_records),
        "split_provenance": split_info, "base_model": model_cfg["name"],
        "resolved_revision": revision, "base_local_artifact_sha256": local_hash,
        "device": torch.cuda.get_device_name(device), "bf16": bf16,
        "gradient_checkpointing": True, "batch_size": 1,
        "microsteps": len(timings), "step_seconds": timings,
        "mean_step_seconds": sum(timings) / len(timings), "losses": losses,
        "peak_allocated_gib": torch.cuda.max_memory_allocated(device) / 2**30,
        "elapsed_seconds_including_load_and_setup": time.perf_counter() - started,
        "three_epoch_training_microbatches": len(train_records) * 3,
        "checkpoint_saved": False,
    }
    print(json.dumps(report, indent=2, default=str))
    del model, optimizer, loader
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
