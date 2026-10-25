from scripts.analyze_validation_errors import markdown_report, grouped_summary, summarize


def test_validation_summary_and_report_include_answer_length(tmp_path):
    rows = [
        {
            "task_type": "QA",
            "true_label": "SUPPORTED",
            "predicted_label": "SUPPORTED",
            "context_tokens": 240,
            "answer_tokens": 90,
            "context_truncated": False,
            "structured_context": False,
            "context_token_bin": "<512",
            "answer_token_bin": "<128",
        },
        {
            "task_type": "QA",
            "true_label": "HALLUCINATED",
            "predicted_label": "SUPPORTED",
            "context_tokens": 320,
            "answer_tokens": 110,
            "context_truncated": True,
            "structured_context": False,
            "context_token_bin": "<512",
            "answer_token_bin": "<128",
        },
    ]
    overall = summarize(rows)
    report = {
        "dataset_archive_sha256": "test-hash",
        "validation_records": len(rows),
        "model_id": "nishantwhat/robustrag-longformer-head",
        "artifact_revision": "test-revision",
        "device": "cuda:0",
        "batch_size": 4,
        "overall": overall,
        "by_task": grouped_summary(rows, ("task_type",)),
        "by_task_context_bin": grouped_summary(rows, ("task_type", "context_token_bin")),
        "by_task_answer_bin": grouped_summary(rows, ("task_type", "answer_token_bin")),
        "by_task_context_truncated": grouped_summary(rows, ("task_type", "context_truncated")),
        "prediction_csv": "predictions.csv",
    }

    output = tmp_path / "validation.md"
    markdown_report(output, report)

    rendered = output.read_text(encoding="utf-8")
    assert "Median answer tokens" in rendered
    assert "QA / <128" in rendered
    assert overall["median_answer_tokens"] == 100.0
