import json

from scripts.evaluate_benchmark import DEFAULT_MODEL_KEYS, artifact_provenance, write_markdown_report


def test_benchmark_report_names_average_precision_calculation(tmp_path):
    report = {
        "dataset_archive_sha256": "abc123",
        "record_count": 2_700,
        "inference_device": {
            "transformers": "cuda:0",
            "tfidf-lr": "cpu",
            "bilstm-attention": "cpu",
        },
        "batch_size": 1,
        "requested_models": ["longformer-head"],
        "not_evaluated_models": [],
        "models": {
            "longformer-head": {
                "model_name": "Longformer Head",
                "inference_device": "cuda:0",
                "status": "ok",
                "metrics": {
                    "accuracy": 0.8,
                    "precision": 0.7,
                    "recall": 0.6,
                    "f1": 0.65,
                    "roc_auc": 0.82,
                    "average_precision": 0.75,
                },
            }
        },
        "limitations": [],
    }
    output = tmp_path / "comparison.md"

    write_markdown_report(output, report)

    rendered = output.read_text(encoding="utf-8")
    assert "Average Precision (AP)" in rendered
    assert "| PR-AUC |" not in rendered


def test_candidate_qca_provenance_uses_run_signature(tmp_path):
    (tmp_path / "provenance.json").write_text(
        json.dumps({"run_signature": "candidate-run-123"}), encoding="utf-8"
    )
    result = artifact_provenance(
        "longformer-qca", "nishantwhat/robustrag-longformer-qca", tmp_path
    )
    assert result == {
        "artifact_source": str(tmp_path),
        "artifact_revision": "local-run:candidate-run-123",
    }


def test_optional_bilstm_is_not_mixed_into_the_historical_benchmark_cohort():
    assert DEFAULT_MODEL_KEYS == [
        "tfidf-lr", "bilstm-attention", "deberta", "longformer-head", "longformer-qca"
    ]
    assert "bilstm-optimized" not in DEFAULT_MODEL_KEYS
