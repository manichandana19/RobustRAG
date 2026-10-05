# RobustRAG

A research project for hallucination detection in Retrieval-Augmented Generation (RAG) systems.

---

## Overview

Large language models conditioned on retrieved context frequently produce answers that diverge from or contradict that context — a failure mode commonly called **hallucination**. Detecting these answers reliably is a prerequisite for deploying RAG systems where correctness matters.

**RobustRAG** builds a hallucination detection pipeline on the [RAGTruth](https://arxiv.org/abs/2401.00396) benchmark. Given a source context and a generated answer, the detector predicts whether the answer is *hallucinated* (contains unsupported or contradicted claims) or *supported* (faithfully grounded in the context).

The project follows a structured, stage-by-stage experimental methodology:

1. **Completed:** Dataset preparation, validation, EDA, and a classical (TF-IDF + Logistic Regression) baseline.
2. **Next:** DeBERTa-v3-base transformer baseline.
3. **Planned:** Input ablation, robustness evaluation, external generalization, error analysis, and final comparison.

---

## Current Status

| Stage | Status |
|---|---|
| Dataset ingestion, preprocessing, validation | Complete |
| Exploratory Data Analysis (EDA) | Complete |
| TF-IDF + Logistic Regression baseline | Complete |
| Baseline evaluation and saved artifacts | Complete |
| Unit and integration tests | Complete |
| DeBERTa-v3-base transformer baseline | **Next** |
| Input ablation study | Planned |
| Robustness evaluation | Planned |
| External generalization evaluation | Planned |
| Error analysis | Planned |
| Final model comparison | Planned |

---

## Dataset

**RAGTruth** is a human-annotated hallucination benchmark for RAG outputs. It covers three task types (QA, Summarization, Data-to-text) across six generation models.

| Statistic | Value |
|---|---|
| Total records | 17,790 |
| Train split | 13,578 |
| Validation split | 1,512 |
| Test split | 2,700 |
| Supported responses | 10,126 |
| Hallucinated responses | 7,664 |
| Overall hallucination rate | 43.08% |
| Unique source documents | 2,965 |
| Generation models | 6 |

**Source models:** GPT-3.5-turbo-0613, GPT-4-0613, Llama-2-7B-chat, Llama-2-13B-chat, Llama-2-70B-chat, Mistral-7B-Instruct.

**Annotation types:** Evident Conflict, Evident Baseless Info, Subtle Conflict, Subtle Baseless Info.

**Processed file:** `data/processed/ragtruth.jsonl`

> **Note:** Source document IDs do not cross the train/test boundary. The test set is fixed and used only for final evaluation.

The `context` field in the processed dataset can be either a plain string or a structured dictionary (e.g., for Data-to-text records containing structured business or review data). The pipeline handles both cases via deterministic serialization before vectorization.

---

## Pipeline

```
Raw RAGTruth Files (response.jsonl + source_info.jsonl)
                |
        load_jsonl()                          [data/loader.py]
                |
   build_ragtruth_records()                   [data/preprocessing.py]
   (join response <-> source by source_id,
    construct label from annotations list)
                |
       RAGTruthRecord dataclass               [data/schema.py]
                |
    validate_records() / audit_records()      [data/validation.py]
                |
  source_grouped_train_validation_split()     [data/splitting.py]
                |
       save to ragtruth.jsonl                 [scripts/prepare_data.py]
                |
  ---- BASELINE (scripts/train_baseline.py) ----
                |
  load processed -> filter split="train"/"test"
                |
  build_input_text(record, mode="ca")
  "[CONTEXT] <normalized> [ANSWER] <text>"
                |
  TfidfVectorizer.fit_transform(train_texts)
                |
  LogisticRegression.fit(X_train, y_train)
                |
  predict_proba(X_test) -> hallucination probability
                |
  evaluate_binary_classifier()               [evaluation/metrics.py]
                |
  Save: model.pkl / metrics.json / config.yaml / test_predictions.jsonl
  ----------------------------------------------
```

---

## Repository Structure

```
RobustRAG/
|-- configs/
|   `-- default.yaml              # All experiment settings (baseline + future transformer)
|-- data/
|   |-- processed/
|   |   `-- ragtruth.jsonl        # Canonical processed dataset (17,790 records)
|   `-- raw/                      # Original RAGTruth files (not tracked in git)
|-- reports/
|   |-- eda/                      # EDA figures, statistics, and summary markdown
|   `-- experiments/
|       `-- baseline_tfidf_logreg/
|           |-- model.pkl         # Fitted classifier bundle (pickle)
|           |-- metrics.json      # Full evaluation metrics
|           |-- config.yaml       # Exact config snapshotted at run time
|           `-- test_predictions.jsonl  # Per-record predictions + probabilities
|-- scripts/
|   |-- prepare_data.py           # One-time dataset preparation from raw files
|   |-- audit_dataset.py          # Dataset integrity audit from raw files
|   |-- check_processed_data.py   # Sanity checks on processed JSONL
|   |-- run_eda.py                # EDA figures + summary
|   |-- run_advanced_eda.py       # Cross-feature EDA
|   |-- inspect_data.py           # Lightweight inspection helper
|   |-- inspect_annotations.py    # Annotation inspection helper
|   `-- train_baseline.py         # Train + evaluate the TF-IDF baseline
|-- src/hallucination_detector/
|   |-- data/
|   |   |-- schema.py             # RAGTruthRecord dataclass
|   |   |-- loader.py             # JSONL file loading with error reporting
|   |   |-- preprocessing.py      # Response + source joining, label construction
|   |   |-- validation.py         # Dataset validation and audit
|   |   `-- splitting.py          # Source-grouped train/validation splitting
|   |-- modeling/
|   |   `-- tfidf_logistic.py     # TF-IDF vectorizer + Logistic Regression classifier
|   |-- training/
|   |   `-- baseline.py           # Training orchestration: load -> fit -> evaluate -> save
|   |-- inference/
|   |   `-- baseline.py           # Load saved model; run inference on new records
|   `-- evaluation/
|       |-- metrics.py            # evaluate_binary_classifier() (all metrics)
|       `-- eda.py                # EDA analytics and visualization functions
`-- tests/
    |-- test_data.py              # JSONL loader tests
    |-- test_preprocessing.py     # Record construction and label tests
    |-- test_splitting.py         # Source-grouped split tests
    |-- test_validation.py        # Validation and audit tests
    `-- test_baseline_model.py    # Classifier, probabilities, structured-context tests
```

---

## Baseline Model

**Input:** Context + Answer (mode `ca`)

Before vectorization, all context values are normalized to a flat string:
- Plain string: whitespace-normalized only.
- Structured `dict`: serialized with `json.dumps(..., sort_keys=True)` for deterministic output.
- `list`/`tuple`: similarly JSON-serialized.
- `None`: empty string.

**Input format passed to TF-IDF:**
```
[CONTEXT] <normalized context text> [ANSWER] <answer text>
```

**Feature representation:** TF-IDF, word unigrams and bigrams.

**Classifier:** Logistic Regression (sklearn).

**Configuration (`configs/default.yaml`):**
```yaml
baseline:
  input_mode: ca
  ngram_range: [1, 2]
  min_df: 2
  max_df: 0.95
  sublinear_tf: true
  max_iter: 1000
  C: 1.0
```

**Seed:** 42 (project-level seed from config, applied to `LogisticRegression(random_state=42)`).

The vectorizer is fitted **exclusively on training data**. The test set is only ever transformed, never used during fitting.

---

## Results

Test set: 2,700 records. Positive class = hallucinated (label = 1).

| Metric | Value |
|---|---|
| Accuracy | 76.78% |
| Precision (hallucinated class) | 69.04% |
| Recall (hallucinated class) | 60.76% |
| F1 (hallucinated class) | 64.64% |
| ROC-AUC | 79.77% |
| Average Precision | 73.33% |

**Confusion matrix:**
```
                    Predicted
                 Supported   Hallucinated
Actual Supported    1500          257
Actual Hallucin.     370          573
```

- **257** supported responses incorrectly flagged (false positives).
- **370** hallucinated responses missed (false negatives).

Precision, Recall, and F1 are for the **hallucinated class**. ROC-AUC and Average Precision measure ranking quality across all probability thresholds.

This baseline is a **fixed classical reference experiment**. Its purpose is to quantify how much signal lexical TF-IDF features carry, providing a lower bound against which transformer representations will be compared.

---

## Installation

The project uses `uv` for virtual environment management.

```bash
# Install uv if needed
pip install uv

# Run commands inside the managed environment
uv run python scripts/train_baseline.py
uv run pytest -q
```

Key dependencies (in `.venv`): `scikit-learn 1.9.1`, `numpy`, `scipy`, `pandas`, `pyyaml`, `matplotlib`, `tabulate`, `pytest 9.1.1`.

---

## Running the Project

**Run all tests:**
```bash
uv run pytest -q
```

**Check processed dataset integrity:**
```bash
uv run python scripts/check_processed_data.py
```

**Train and evaluate the TF-IDF baseline:**
```bash
uv run python scripts/train_baseline.py
```
Saves outputs to `reports/experiments/baseline_tfidf_logreg/`.

**Prepare data from raw files** *(requires raw RAGTruth JSONL files)*:
```bash
uv run python scripts/prepare_data.py \
  --responses data/raw/response.jsonl \
  --sources data/raw/source_info.jsonl \
  --output-dir data/processed
```

**Run EDA** *(requires raw source_info.jsonl for task_type metadata)*:
```bash
uv run python scripts/run_eda.py
uv run python scripts/run_advanced_eda.py
```

---

## Testing

Tests are in `tests/`. Coverage includes: JSONL loading, record construction, label correctness, source-grouped splitting determinism, validation errors, classifier training, probability output shape, and structured `dict` context handling.

```bash
uv run pytest -q     # Concise output
uv run pytest -v     # Verbose
```

Current result: **15 tests pass**.

---

## Experiment Artifacts

All artifacts for the classical baseline are in `reports/experiments/baseline_tfidf_logreg/`.

| File | Contents |
|---|---|
| `model.pkl` | Pickle bundle: `TfidfLogisticClassifier`, config, record counts, vocabulary size |
| `metrics.json` | Accuracy, Precision, Recall, F1, ROC-AUC, AP, confusion matrix, classification report |
| `config.yaml` | Snapshot of `configs/default.yaml` at training time |
| `test_predictions.jsonl` | Per-record: `id`, `source_id`, `hallucinated` (true label), `prediction`, `probability_hallucinated` |

---

## Research Roadmap

| Stage | Research Question | Status |
|---|---|---|
| 1. Transformer baseline (DeBERTa-v3-base) | Do contextual pretrained representations improve over lexical features? | **Next** |
| 2. Input ablation (C+A, Q+A, Q+C+A, A-only) | Which inputs contribute most to hallucination detection? | Planned |
| 3. Robustness evaluation | Does performance hold under noisy, incomplete, or adversarial context? | Planned |
| 4. External generalization | Does the detector transfer beyond RAGTruth? | Planned |
| 5. Error analysis | Which hallucination types are systematically missed? | Planned |
| 6. Final comparison | Consolidated metrics across all models and conditions | Planned |

Only the classical baseline (Stage 0) is currently implemented.

---

## Reproducibility

- Global seed `42` is set in `configs/default.yaml` and applied to `LogisticRegression(random_state=42)`.
- The TF-IDF vectorizer is deterministic given fixed training data order.
- The test set is **fixed** via the `split` field in `ragtruth.jsonl`; it is never used during training.
- Source IDs do not appear in both train and test.
- The validation split is produced with a deterministic source-grouped shuffle (same seed).
- The exact config is snapshotted into each experiment directory.
- Per-record predictions are saved for reproducibility auditing.

To reproduce the baseline exactly:
```bash
uv run python scripts/train_baseline.py
```
