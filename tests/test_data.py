import json

import pytest

from hallucination_detector.data.loader import JSONLLoadError, load_jsonl


def test_load_jsonl(tmp_path):
    path = tmp_path / "data.jsonl"

    records = [
        {"id": 1, "text": "hello"},
        {"id": 2, "text": "world"},
    ]

    with path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record) + "\n")

    loaded = load_jsonl(path)

    assert loaded == records


def test_load_jsonl_ignores_empty_lines(tmp_path):
    path = tmp_path / "data.jsonl"

    path.write_text(
        '{"id": 1}\n\n{"id": 2}\n',
        encoding="utf-8",
    )

    loaded = load_jsonl(path)

    assert len(loaded) == 2


def test_invalid_json_raises_error(tmp_path):
    path = tmp_path / "invalid.jsonl"

    path.write_text(
        '{"valid": true}\n'
        '{"broken": \n',
        encoding="utf-8",
    )

    with pytest.raises(JSONLLoadError):
        load_jsonl(path)