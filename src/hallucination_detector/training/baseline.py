import json
import pickle
from pathlib import Path
from typing import Any

import yaml

from hallucination_detector.data.loader import load_jsonl
from hallucination_detector.data.schema import RAGTruthRecord
from hallucination_detector.evaluation.metrics import (
    evaluate_binary_classifier,
)
from hallucination_detector.modeling.tfidf_logistic import (
    TfidfLogisticClassifier,
)


def load_processed_records(path: str | Path) -> list[RAGTruthRecord]:
    """Load canonical RAGTruth records from processed JSONL."""
    raw_records = load_jsonl(path)

    try:
        return [
            RAGTruthRecord(**record)
            for record in raw_records
        ]
    except TypeError as exc:
        raise ValueError(
            f"Processed dataset does not match RAGTruthRecord schema: {exc}"
        ) from exc


def split_records(
    records: list[RAGTruthRecord],
) -> tuple[list[RAGTruthRecord], list[RAGTruthRecord]]:
    """Use the repository's fixed train/test assignments."""
    train = [record for record in records if record.split == "train"]
    test = [record for record in records if record.split == "test"]

    if not train:
        raise ValueError("No records with split='train'.")
    if not test:
        raise ValueError("No records with split='test'.")

    return train, test


def train_baseline(
    dataset_path: str | Path,
    output_dir: str | Path,
    config: dict[str, Any],
) -> dict[str, Any]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    records = load_processed_records(dataset_path)
    train_records, test_records = split_records(records)

    model_config = config["baseline"]

    classifier = TfidfLogisticClassifier(
        input_mode=model_config["input_mode"],
        ngram_range=tuple(model_config["ngram_range"]),
        min_df=model_config["min_df"],
        max_df=model_config["max_df"],
        sublinear_tf=model_config["sublinear_tf"],
        max_iter=model_config["max_iter"],
        C=model_config["C"],
        random_state=config["project"]["seed"],
    )

    classifier.fit(train_records)

    predictions = classifier.predict(test_records)
    probabilities = classifier.predict_proba(test_records)

    metrics = evaluate_binary_classifier(
        test_records,
        predictions,
        probabilities,
    )

    bundle = {
        "model": classifier,
        "config": config,
        "train_records": len(train_records),
        "test_records": len(test_records),
        "vocabulary_size": len(classifier.vectorizer.vocabulary_),
    }

    with (output_dir / "model.pkl").open("wb") as file:
        pickle.dump(bundle, file)

    with (output_dir / "metrics.json").open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(metrics, file, indent=2)

    with (output_dir / "config.yaml").open(
        "w",
        encoding="utf-8",
    ) as file:
        yaml.safe_dump(config, file, sort_keys=False)

    predictions_output = []
    for record, prediction, probability in zip(
        test_records,
        predictions,
        probabilities[:, 1],
    ):
        predictions_output.append(
            {
                "id": record.id,
                "source_id": record.source_id,
                "hallucinated": record.hallucinated,
                "prediction": int(prediction),
                "probability_hallucinated": float(probability),
            }
        )

    with (output_dir / "test_predictions.jsonl").open(
        "w",
        encoding="utf-8",
    ) as file:
        for prediction in predictions_output:
            file.write(json.dumps(prediction) + "\n")

    return metrics


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)
