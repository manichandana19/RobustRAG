from __future__ import annotations

import json
import os
import platform
import random
import shutil
import uuid
import zipfile
from importlib import metadata as package_metadata
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
from torch.utils.data import DataLoader
from huggingface_hub import snapshot_download

from hallucination_detector.data.schema import RAGTruthRecord
from hallucination_detector.data.sentence_selector import SentenceSelector
from hallucination_detector.data.torch_dataset import LongformerDataCollator
from hallucination_detector.data.longformer_smart_dataset import (
    RAGTruthSmartLongformerDataset,
)
from hallucination_detector.data.qca_artifacts import (
    PROVENANCE_FILENAME,
    QCA_ARTIFACT_TYPE,
    QCA_PROVENANCE_SCHEMA,
    artifact_hashes,
    canonical_json_sha256,
    load_and_verify_qca_provenance,
    sha256_bytes,
    sha256_directory,
    sha256_file,
)
from hallucination_detector.modeling.longformer_classifier import LongformerClassifier


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


TRAIN_MEMBER = "data/processed/train.jsonl"
VALIDATION_MEMBER = "data/processed/validation.jsonl"
TRAINING_STATE = "resume_state.pt"


def _records_from_jsonl_bytes(payload: bytes, expected_split: str) -> list[RAGTruthRecord]:
    rows = []
    for line_number, raw_line in enumerate(payload.decode("utf-8").splitlines(), start=1):
        if not raw_line.strip():
            continue
        value = json.loads(raw_line)
        if not isinstance(value, dict):
            raise ValueError(f"Expected a JSON object at {expected_split} line {line_number}.")
        if value.get("split") != expected_split:
            raise ValueError(
                f"Split leakage: {expected_split} archive member contains record "
                f"{value.get('id')!r} labeled {value.get('split')!r}."
            )
        rows.append(RAGTruthRecord(**value))
    if not rows:
        raise ValueError(f"The {expected_split} split is empty.")
    return rows


def load_train_validation_archive(
    archive_path: str | Path,
) -> tuple[list[RAGTruthRecord], list[RAGTruthRecord], dict[str, Any]]:
    """Read only the train and validation members; never open or parse test.jsonl."""
    archive_path = Path(archive_path)
    if not archive_path.is_file():
        raise FileNotFoundError(f"Dataset archive not found: {archive_path}")
    with zipfile.ZipFile(archive_path) as archive:
        train_bytes = archive.read(TRAIN_MEMBER)
        validation_bytes = archive.read(VALIDATION_MEMBER)
    train_records = _records_from_jsonl_bytes(train_bytes, "train")
    validation_records = _records_from_jsonl_bytes(validation_bytes, "validation")
    train_ids = {record.id for record in train_records}
    validation_ids = {record.id for record in validation_records}
    if len(train_ids) != len(train_records) or len(validation_ids) != len(validation_records):
        raise ValueError("Duplicate record IDs found within train or validation split.")
    overlap = train_ids.intersection(validation_ids)
    if overlap:
        raise ValueError(f"Train and validation record IDs overlap: {sorted(overlap)[:5]}")
    split_info = {
        "archive_path": str(archive_path),
        "archive_sha256": sha256_file(archive_path),
        "train_member": TRAIN_MEMBER,
        "train_sha256": sha256_bytes(train_bytes),
        "train_records": len(train_records),
        "train_record_ids_sha256": canonical_json_sha256(sorted(train_ids)),
        "validation_member": VALIDATION_MEMBER,
        "validation_sha256": sha256_bytes(validation_bytes),
        "validation_records": len(validation_records),
        "validation_record_ids_sha256": canonical_json_sha256(sorted(validation_ids)),
        "test_member_opened": False,
    }
    return train_records, validation_records, split_info


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _atomic_torch_save(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)


def recover_checkpoint_publication(output_dir: Path) -> None:
    """Restore a validated prior best checkpoint if interrupted during atomic swap."""
    best_dir = output_dir / "best_model"
    previous_dir = output_dir / ".best_model.previous"
    if previous_dir.exists():
        if best_dir.exists():
            load_and_verify_qca_provenance(best_dir)
            shutil.rmtree(previous_dir)
        else:
            load_and_verify_qca_provenance(previous_dir)
            os.replace(previous_dir, best_dir)
    for staging_dir in output_dir.glob(".best_model.tmp-*"):
        if staging_dir.is_dir():
            shutil.rmtree(staging_dir)


def _to_cpu(value: Any) -> Any:
    if torch.is_tensor(value):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {key: _to_cpu(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_to_cpu(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_to_cpu(item) for item in value)
    return value


def _package_versions() -> dict[str, str]:
    result = {"python": platform.python_version()}
    for name in ("torch", "transformers", "tokenizers", "huggingface-hub", "numpy", "PyYAML"):
        try:
            result[name] = package_metadata.version(name)
        except package_metadata.PackageNotFoundError:
            result[name] = "not-installed"
    return result


def _resolve_base_model(model_name: str, revision: str) -> tuple[str, str, str]:
    local_path = Path(model_name)
    if local_path.exists():
        local_hash = sha256_directory(local_path)
        return str(local_path.resolve()), f"local:{local_hash}", local_hash
    snapshot = Path(snapshot_download(repo_id=model_name, revision=revision))
    resolved_revision = snapshot.name if snapshot.parent.name == "snapshots" else revision
    return str(snapshot), resolved_revision, ""


def _publish_best_checkpoint(
    model: LongformerClassifier,
    selector: SentenceSelector,
    output_dir: Path,
    training_metadata: dict[str, Any],
    run_signature: str,
    best_epoch: int,
    best_validation_loss: float,
) -> dict[str, str]:
    staging_dir = output_dir / f".best_model.tmp-{uuid.uuid4().hex}"
    previous_dir = output_dir / ".best_model.previous"
    try:
        model.save_pretrained(staging_dir)
        selector.save(staging_dir / "sentence_selector.json")
        hashes = artifact_hashes(staging_dir)
        manifest = {
            "schema_version": QCA_PROVENANCE_SCHEMA,
            "artifact_type": QCA_ARTIFACT_TYPE,
            "run_signature": run_signature,
            "training": {**training_metadata, "best_epoch": best_epoch,
                         "best_validation_loss": best_validation_loss},
            "selector": {
                "path": "sentence_selector.json",
                "sha256": hashes["sentence_selector.json"],
                "train_split_sha256": training_metadata["dataset"]["train_sha256"],
                "fit_record_count": training_metadata["dataset"]["train_records"],
            },
            "artifact_hashes": hashes,
        }
        _atomic_json(staging_dir / PROVENANCE_FILENAME, manifest)
        load_and_verify_qca_provenance(staging_dir)

        best_dir = output_dir / "best_model"
        if previous_dir.exists():
            shutil.rmtree(previous_dir)
        if best_dir.exists():
            os.replace(best_dir, previous_dir)
        try:
            os.replace(staging_dir, best_dir)
        except Exception:
            if previous_dir.exists() and not best_dir.exists():
                os.replace(previous_dir, best_dir)
            raise
        if previous_dir.exists():
            shutil.rmtree(previous_dir)
        return hashes
    finally:
        if staging_dir.exists():
            shutil.rmtree(staging_dir)


def _run_signature(
    archive_info: dict[str, Any],
    config: dict[str, Any],
    config_path: Path,
    source_path: Path,
    base_model_id: str,
    resolved_revision: str,
    base_local_hash: str,
    packages: dict[str, str],
) -> str:
    signature_data = {
        "dataset": archive_info,
        "config_sha256": sha256_file(config_path),
        "training_source_sha256": sha256_file(source_path),
        "training_helpers_sha256": sha256_file(Path(__file__).resolve()),
        "base_model_id": base_model_id,
        "base_model_revision": resolved_revision,
        "base_model_local_sha256": base_local_hash,
        "package_versions": packages,
        "config": config,
    }
    return canonical_json_sha256(signature_data)


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
    archive_path: str | Path,
    output_dir: str | Path,
    config_path: str | Path,
    *,
    resume: bool = False,
    device_name: str | None = None,
    batch_size_override: int | None = None,
    gradient_accumulation_override: int | None = None,
) -> dict[str, Any]:
    """Compatibility wrapper for the isolated train/validation-only QCA pipeline."""
    from hallucination_detector.training.longformer_qca_pipeline import train_qca

    return train_qca(
        archive_path=archive_path,
        output_dir=output_dir,
        config_path=config_path,
        resume=resume,
        device_name=device_name,
        batch_size_override=batch_size_override,
        gradient_accumulation_override=gradient_accumulation_override,
    )


def main() -> None:
    from hallucination_detector.training.longformer_qca_pipeline import main as qca_main

    qca_main()


if __name__ == "__main__":
    main()
