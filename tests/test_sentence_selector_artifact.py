import json

import pytest

from hallucination_detector.data.sentence_selector import SentenceSelector


def test_sentence_selector_round_trip_preserves_training_statistics(tmp_path):
    original = SentenceSelector(idf={"curie": 2.5, "warsaw": 1.25}, default_idf=0.75)
    path = tmp_path / "checkpoint" / "sentence_selector.json"

    original.save(path)
    loaded = SentenceSelector.load(path)

    assert loaded.idf == original.idf
    assert loaded.default_idf == original.default_idf


def test_sentence_selector_load_rejects_invalid_idf(tmp_path):
    path = tmp_path / "selector.json"
    path.write_text(json.dumps({"idf": {"curie": -1}, "default_idf": 1}), encoding="utf-8")

    with pytest.raises(ValueError, match="Invalid IDF"):
        SentenceSelector.load(path)
