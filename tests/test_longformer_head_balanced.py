import json
from zipfile import ZipFile

import numpy as np
import pytest
import torch

from hallucination_detector.data.schema import RAGTruthRecord
from hallucination_detector.training.longformer_head_balanced import (
    read_train_validation,
    task_balanced_weights,
    validation_metrics,
)


def _record(record_id, source, split, annotations=()):
    return RAGTruthRecord(
        id=record_id,
        source_id=source,
        question="question",
        context="context",
        answer="answer",
        hallucinated=bool(annotations),
        split=split,
        annotations=list(annotations),
    )


def _make_archive(path):
    sources = [
        {"source_id": source, "task_type": task}
        for source, task in (("q", "QA"), ("s", "Summary"), ("d", "Data2txt"), ("vq", "QA"))
    ]
    train = [
        {"id": "q0", "source_id": "q", "question": "q", "context": "c", "answer": "a", "hallucinated": False, "split": "train", "annotations": []},
        {"id": "s0", "source_id": "s", "question": "q", "context": "c", "answer": "a", "hallucinated": False, "split": "train", "annotations": []},
        {"id": "d0", "source_id": "d", "question": "q", "context": "c", "answer": "a", "hallucinated": False, "split": "train", "annotations": []},
    ]
    validation = [{"id": "vq0", "source_id": "vq", "question": "q", "context": "c", "answer": "a", "hallucinated": False, "split": "validation", "annotations": []}]
    with ZipFile(path, "w") as archive:
        archive.writestr("data/raw/source_info.jsonl", "\n".join(json.dumps(row) for row in sources))
        archive.writestr("data/processed/train.jsonl", "\n".join(json.dumps(row) for row in train))
        archive.writestr("data/processed/validation.jsonl", "\n".join(json.dumps(row) for row in validation))
        # Malformed on purpose: the training reader must never open held-out test data.
        archive.writestr("data/processed/test.jsonl", "THIS MUST NEVER BE PARSED")


def test_training_reader_uses_only_train_validation_and_checks_source_separation(tmp_path):
    archive = tmp_path / "data.zip"
    _make_archive(archive)
    train, validation, task_by_id, _ = read_train_validation(archive)
    assert [r.id for r in train] == ["q0", "s0", "d0"]
    assert [r.id for r in validation] == ["vq0"]
    assert task_by_id["q0"] == "QA"
    assert {r.source_id for r in train}.isdisjoint({r.source_id for r in validation})


def test_sampler_gives_qa_summary_equal_subtype_mass_and_preserves_data2txt_mass():
    records = []
    tasks = {}
    for task in ("QA", "Summary"):
        for subtype, annotations in (
            ("supported", []),
            ("conflict", [{"label_type": "Evident Conflict"}]),
            ("baseless", [{"label_type": "Evident Baseless Info"}]),
            ("mixed", [{"label_type": "Evident Conflict"}, {"label_type": "Subtle Baseless Info"}]),
        ):
            record_id = f"{task}-{subtype}"
            records.append(_record(record_id, record_id, "train", annotations))
            tasks[record_id] = task
    for i in range(8):
        record_id = f"data-{i}"
        records.append(_record(record_id, record_id, "train"))
        tasks[record_id] = "Data2txt"
    weights, counts = task_balanced_weights(records, tasks)
    mass = {}
    for record, weight in zip(records, weights.tolist()):
        mass[tasks[record.id]] = mass.get(tasks[record.id], 0.0) + weight
    assert mass == pytest.approx({"QA": 0.4, "Summary": 0.4, "Data2txt": 0.2})
    assert all(counts[f"{task}/{kind}"] == 1 for task in ("QA", "Summary") for kind in ("supported", "conflict", "baseless", "mixed"))


def test_validation_metrics_report_task_and_subtype_confusions_at_fixed_threshold():
    records = [
        _record("sup", "s", "validation"),
        _record("conflict", "c", "validation", [{"label_type": "Evident Conflict"}]),
        _record("baseless", "b", "validation", [{"label_type": "Evident Baseless Info"}]),
        _record("sum-sup", "ss", "validation"),
        _record("sum-conflict", "sc", "validation", [{"label_type": "Evident Conflict"}]),
        _record("sum-baseless", "sb", "validation", [{"label_type": "Evident Baseless Info"}]),
    ]
    tasks = {record.id: ("QA" if not record.id.startswith("sum-") else "Summary") for record in records}
    probs = np.asarray([[0.8, 0.2], [0.3, 0.7], [0.4, 0.6], [0.8, 0.2], [0.3, 0.7], [0.4, 0.6]])
    result = validation_metrics(records, probs, tasks)
    assert result["threshold"] == 0.5
    assert result["overall"]["confusion"] == {"tn": 2, "fp": 0, "fn": 0, "tp": 4}
    assert result["by_task_subtype"]["QA/conflict"]["confusion"] == {"tn": 1, "fp": 0, "fn": 0, "tp": 1}
    assert result["by_task_subtype"]["QA/baseless"]["f1"] == 1.0

