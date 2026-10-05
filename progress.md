# RobustRAG - Detailed Project Progress

---

## 1. Project Objective

### Problem Statement

Retrieval-Augmented Generation (RAG) systems retrieve source documents and condition a language model on them to produce answers. Despite conditioning on retrieved evidence, these models regularly produce answers that contradict the context or introduce unsupported claims. Detecting these failures automatically and reliably is the core problem RobustRAG addresses.

### Research Objective

Build a supervised binary classifier that takes a source context and a generated answer as input and predicts whether the answer is hallucinated or supported. The detector should be:

- Accurate: captures genuine hallucination signal, not dataset artifacts
- Robust: performs reliably under variation in context quality, model identity, and task type
- Reproducible: experiments are deterministic and results are archived
- Staged: each experiment is a controlled ablation that builds on the previous

### What Robust Means Here

Robustness is both a design goal and an experimental axis. It will be specifically evaluated against: degraded context (noisy, incomplete, contradictory), context from different domains, and hallucination types not well-represented in training. At this stage, robustness has not yet been tested.

---

## 2. Current Project State

### Completed

- Dataset ingestion: Raw RAGTruth JSONL loaded, responses joined to sources, RAGTruthRecord objects constructed.
- Data validation: Schema enforcement, duplicate ID detection, split label validation, source-split leakage check.
- Data audit: Summary statistics (record counts, label distributions, source overlap, etc.).
- Train/validation/test handling: Fixed test split preserved from RAGTruth. Source-grouped validation split from train data.
- Processed dataset: data/processed/ragtruth.jsonl (17,790 records).
- EDA: 19 figures in reports/eda/. Summary statistics CSV. Written summary in reports/eda/eda_summary.md.
- Classical baseline: TF-IDF (unigrams+bigrams) + Logistic Regression. Input: Context + Answer.
- Structured context handling: normalize_text() in modeling/tfidf_logistic.py handles str and dict context.
- Baseline evaluation: Full metric suite saved to reports/experiments/baseline_tfidf_logreg/metrics.json.
- Artifact archiving: model.pkl, metrics.json, config.yaml, test_predictions.jsonl per experiment.
- Tests: 15 unit tests covering loader, preprocessing, splitting, validation, and baseline model behavior.

### Partially Completed

- configs/default.yaml: Contains both classical baseline config (used) and transformer config fields (not yet used by any code). The transformer config is present as a placeholder only.

### Planned / Next

- DeBERTa-v3-base transformer baseline (next immediate experiment)
- Input ablation: compare Q+A, C+A, Q+C+A
- Robustness evaluation under degraded context conditions
- External generalization evaluation beyond RAGTruth
- Error analysis by hallucination annotation type
- Final cross-model comparison table

### Not Yet Implemented

- Any transformer model code or training loop
- Validation-set-based early stopping
- Any robustness test harness
- Any cross-dataset evaluation
- Any error stratification by annotation type
- Threshold analysis or precision-recall curve generation (beyond saved probabilities)

---

## 3. End-to-End System Flow

```
Raw RAGTruth data
  response.jsonl + source_info.jsonl
        |
  load_jsonl()                   [data/loader.py]
  Returns: list[dict]
        |
  build_ragtruth_records()       [data/preprocessing.py]
  Joins response to source by source_id.
  Label: hallucinated = len(labels) > 0
  Returns: list[RAGTruthRecord]
        |
  RAGTruthRecord dataclass       [data/schema.py]
        |
  validate_records()             [data/validation.py]
  audit_records()                [data/validation.py]
        |
  source_grouped_train_validation_split()  [data/splitting.py]
  Groups train records by source_id.
  10% of sources -> validation. Deterministic seed=42.
        |
  write_jsonl() -> ragtruth.jsonl  [scripts/prepare_data.py]
        |
  === BASELINE TRAINING ===
        |
  load_processed_records()       [training/baseline.py]
  split on record.split: train=13578, test=2700
        |
  build_input_text(record, "ca") [modeling/tfidf_logistic.py]
  "[CONTEXT] <normalized> [ANSWER] <text>"
        |
  TfidfVectorizer.fit_transform(train_texts)
  -> sparse matrix (13578 x 437871)
        |
  LogisticRegression.fit(X_train, y_train)
        |
  vectorizer.transform(test_texts)
  classifier.predict_proba(X_test)
  -> shape (2700, 2); col 1 = P(hallucinated)
        |
  evaluate_binary_classifier()   [evaluation/metrics.py]
        |
  Save: model.pkl / metrics.json / config.yaml / test_predictions.jsonl
```

---

## 4. Repository Map

### configs/default.yaml

Single source of truth for all experiment configuration. Contains:
- project.seed: 42 (global seed)
- baseline.*: TF-IDF baseline settings (USED)
- model.*, training.*: DeBERTa placeholder settings (NOT YET USED)
- experiment.input_mode: qca (future default)

### data/processed/ragtruth.jsonl

Canonical processed dataset. 17,790 JSON objects. Context field is heterogeneous:
- 12,132 records: dict context (Data-to-text task)
- 5,658 records: string context (QA and Summarization tasks)
Split field: train=13578, validation=1512, test=2700.
Source leakage: verified clean.

### src/hallucination_detector/data/schema.py

Defines RAGTruthRecord dataclass. Fields: id, source_id, question, context (Any in practice), answer, hallucinated, model, temperature, split, quality, annotations.

IMPORTANT GAP: context is annotated as str but is dict for ~68% of records. Python dataclasses do not enforce types at runtime so this does not crash, but the type annotation is misleading.

### src/hallucination_detector/data/loader.py

Safe JSONL loading. Skips blank lines. Raises JSONLLoadError (custom exception) with line number on malformed JSON or non-dict values. Used by: training/baseline.py, scripts/prepare_data.py, scripts/audit_dataset.py, scripts/check_processed_data.py.

### src/hallucination_detector/data/preprocessing.py

Joins raw response records with source records. Key operations:
- Builds source_by_id index (string-keyed) for O(1) lookup.
- question = source["prompt"], context = source["source_info"], answer = response["response"]
- hallucinated = len(response["labels"]) > 0
- Annotations preserved verbatim.
Used by: scripts/prepare_data.py, scripts/audit_dataset.py.

### src/hallucination_detector/data/validation.py

Two functions:
- validate_records(): hard checks, raises ValueError on failure. Context emptiness uses str(context_val).strip() to handle both str and dict.
- audit_records(): returns DatasetAudit dataclass with full summary statistics.

KNOWN ISSUE: Line 1 contains "from numpy import record" which is unused and imports a deprecated numpy name. Dead code, no functional impact.
Used by: scripts/prepare_data.py, scripts/audit_dataset.py, tests/test_validation.py.

### src/hallucination_detector/data/splitting.py

Source-grouped train/validation split. Uses random.Random(seed) (pure Python, not numpy) for determinism. Reassigns split="validation" on validation records via dataclasses.replace().
Used by: scripts/prepare_data.py, tests/test_splitting.py.

### src/hallucination_detector/modeling/tfidf_logistic.py

The complete classical baseline model. Key components:

normalize_text(value: Any) -> str
  str -> whitespace-normalize
  dict -> json.dumps(sort_keys=True)
  list/tuple -> json.dumps()
  None -> ""
  other -> str()
  All outputs: re.sub(r"\s+", " ", text).strip()

build_input_text(record, input_mode) -> str
  Modes: "ca" (Context+Answer), "qa" (Question+Answer), "qca" (Question+Context+Answer)
  Output: "[CONTEXT] <ctx> [ANSWER] <ans>"

TfidfLogisticClassifier (dataclass)
  fit(): fit_transform(train_texts), classifier.fit(matrix, labels)
  predict_proba(): transform(texts), classifier.predict_proba(matrix) -> (n, 2)
  predict(): transform(texts), classifier.predict(matrix)
  _check_fitted(): guards against predict before fit

### src/hallucination_detector/training/baseline.py

Orchestrates the full baseline experiment. Key:
- load_processed_records(): loads JSONL, constructs RAGTruthRecord(**record)
- split_records(): filters by record.split (validation records NOT used)
- train_baseline(): load, fit, evaluate, save
- Artifact bundle in model.pkl: {model, config, train_records:13578, test_records:2700, vocabulary_size:437871}
Used by: scripts/train_baseline.py.

### src/hallucination_detector/inference/baseline.py

Load saved model bundle and run inference on new records.
- load_baseline(): opens pickle, validates type is TfidfLogisticClassifier
- predict_record(): returns {"prediction": int, "probability_hallucinated": float}
Status: Implemented but not called by any current script. Infrastructure for future use.

### src/hallucination_detector/evaluation/metrics.py

evaluate_binary_classifier(records, predictions, probabilities) -> dict
- y_true from record.hallucinated (positive=1=hallucinated)
- y_probability = probabilities[:, 1] (sklearn class ordering guarantees col 1 = class 1)
- Returns: accuracy, precision, recall, f1, roc_auc, average_precision, confusion_matrix, classification_report, test_records, positive_rate
Used by: training/baseline.py.

### src/hallucination_detector/evaluation/eda.py

EDA module. Requires both processed JSONL AND raw source_info.jsonl (for task_type column).
Key: load_eda_data(), _calculate_lexical_overlap(), 10 plot functions, calculate_statistics(), generate_summary(), run_eda().
Status: Completed. All 19 figures generated.

### scripts/train_baseline.py

Entry point. Adds src/ to sys.path. Loads config, calls train_baseline(), prints metrics.
Hardcoded paths: configs/default.yaml, data/processed/ragtruth.jsonl, reports/experiments/baseline_tfidf_logreg.

### tests/ (5 test files, 15 tests total)

test_data.py (3): JSONL loading, empty lines, invalid JSON error
test_preprocessing.py (3): supported record, hallucinated record, missing source error
test_splitting.py (2): source disjointness, reproducibility with same seed
test_validation.py (4): valid pass, duplicate IDs fail, invalid split fail, audit detects overlap
test_baseline_model.py (3): CA mode field presence, probability shape and sum, dict context handling

---

## 5. Data Pipeline Deep Dive

### Structured Context

The RAGTruth Data-to-text task uses source_info values that are Python dicts. When the processed JSONL is loaded back with json.loads(), these become Python dicts again. The schema.py type annotation says str, but the actual runtime type is dict for 12,132 of 17,790 records.

The normalize_text() function in modeling/tfidf_logistic.py handles this with json.dumps(sort_keys=True). sort_keys=True is critical - it ensures the same dict always produces the same string regardless of key insertion order.

### Label Construction

hallucinated = len(response["labels"]) > 0

This is the only place labels are determined. Any response with at least one human-annotated hallucination span is labeled positive. This is clean and unambiguous.

### Validation Split vs Baseline Trainer

The processed ragtruth.jsonl contains validation records (1,512). The baseline trainer (train_baseline.py) does NOT use the validation split - it trains on all split=="train" records (13,578) and evaluates on all split=="test" records (2,700). The validation split exists for future use by models requiring early stopping.

---

## 6. Dataset and EDA Findings

### Dataset Facts (verified from processed file)

| Field | Value |
|---|---|
| Total records | 17,790 |
| split=train | 13,578 |
| split=validation | 1,512 |
| split=test | 2,700 |
| hallucinated=True | 7,664 |
| hallucinated=False | 10,126 |
| Unique source_ids | 2,965 |
| Context type=dict | 12,132 |
| Context type=str | 5,658 |
| Test set positive rate | 34.93% (943/2700) |

Note: The overall positive rate is 43.08% but the test set positive rate is 34.93%. This reflects the original RAGTruth split and is inherent to the data.

### EDA Observations (descriptive only - not causal)

Model-level hallucination rates:
- Mistral-7B-Instruct: ~65.9%
- Llama-2-7B-chat: ~61.8%
- Llama-2-13B-chat: ~56.6%
- Llama-2-70B-chat: ~47.1%
- GPT-4: ~13.7%
- GPT-3.5-turbo: ~13.5%

Task-type hallucination rates:
- Data2txt: ~68.6%
- Summarization: ~29.8%
- QA: ~29.1%

Answer length (mean words): Supported ~119, Hallucinated ~146.

Lexical overlap (fraction of unique answer words in context):
- Supported: ~0.683 mean
- Hallucinated: ~0.513 mean

Source-level agreement: 269 sources where all 6 model responses are supported. 86 sources where all 6 model responses are hallucinated.

### Why These Observations Matter

The lexical overlap difference (0.68 vs 0.51) is the primary signal the TF-IDF baseline exploits. The strong model-level stratification means model identity would be a powerful predictor - but the baseline deliberately excludes it to measure only lexical text signal.

These observations motivate future experiments: input ablation (context vs answer contribution), robustness (does performance hold when overlap is artificially altered), error analysis (do subtle vs evident hallucinations have different overlap profiles).

---

## 7. Baseline Model Deep Dive

### Why TF-IDF + Logistic Regression

Establishes what lexical features alone can achieve. No semantic understanding, no pretrained representations. Provides a lower bound. If a transformer cannot beat it, the experiment design is likely flawed. Fast, interpretable, reproducible.

### Why Context + Answer (not Q+C+A)

Research question for the baseline: how much signal is present in the (context, answer) pair alone? Including the question conflates context-answer grounding with question-answer relevance. Q+C+A will be tested in the input ablation stage.

### Input Construction

"[CONTEXT] <normalized context> [ANSWER] <answer>"

The bracket tokens serve as separator tokens and allow the vectorizer to distinguish context-domain features from answer-domain features via bigrams like "context] word" and "word [answer".

### TF-IDF Configuration Rationale

- ngram_range=(1,2): captures keywords and two-word phrases. Bigrams capture collocations like "according to" vs "I believe".
- min_df=2: removes hapax legomena.
- max_df=0.95: removes near-ubiquitous terms with no discriminative value.
- sublinear_tf=True: applies 1+log(tf). Prevents long answers from dominating by raw term count.

Vocabulary size: 437,871 features.

### Train/Test Procedure

1. TfidfVectorizer.fit_transform(train_texts): vocabulary and IDF from training only.
2. LogisticRegression.fit(X_train_sparse, y_train): L2-regularized, C=1.0.
3. vectorizer.transform(test_texts): fixed vocab, unknown tokens silently ignored.
4. classifier.predict_proba(X_test_sparse): softmax probabilities, shape (2700, 2).

---

## 8. Baseline Results

### Metrics (from reports/experiments/baseline_tfidf_logreg/metrics.json)

| Metric | Value |
|---|---|
| Accuracy | 0.7678 |
| Precision (hallucinated) | 0.6904 |
| Recall (hallucinated) | 0.6076 |
| F1 (hallucinated) | 0.6464 |
| ROC-AUC | 0.7977 |
| Average Precision | 0.7333 |

### Confusion Matrix

```
                     Predicted
                  Supported   Hallucinated
Actual Supported    1500          257
Actual Hallucin.     370          573
```

### Per-Class Metrics

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| supported | 0.802 | 0.854 | 0.827 | 1757 |
| hallucinated | 0.690 | 0.608 | 0.646 | 943 |

### Technical Interpretation

Accuracy baseline (always-predict-supported): 65.1%. The baseline at 76.78% meaningfully exceeds this.

False negatives (370 missed hallucinations): 39.2% of hallucinated responses are classified as supported. These are the most dangerous errors in deployment.

False positives (257 wrong flags): 14.6% of supported responses incorrectly flagged.

ROC-AUC (0.798): the model ranks a randomly chosen hallucinated answer above a randomly chosen supported answer 79.8% of the time. Substantially above chance (0.5).

Average Precision (0.733): precision-recall curve substantially above the naive baseline of 0.349 (the positive rate). Probability scores are useful for ranking.

What the baseline establishes: Lexical n-gram features carry meaningful hallucination signal. The gap between hallucinated recall (0.608) and supported recall (0.854) reflects the fundamental challenge: hallucinated answers often look lexically plausible. Semantic reasoning is the natural next step.

---

## 9. Testing and Validation

### Test Coverage

15 tests across 5 files:
- test_data.py: JSONL loading, empty lines, malformed JSON error
- test_preprocessing.py: record construction, label correctness, annotations preserved, missing source error
- test_splitting.py: source disjointness between train/validation, split label reassignment, reproducibility
- test_validation.py: valid records pass, duplicate IDs fail, invalid split fail, audit detects source overlap
- test_baseline_model.py: CA mode field presence, probability shape/sum, dict context handling

### Coverage Gaps

- No test for predict() returning integer labels
- No test that predict before fit raises RuntimeError
- No test for invalid input_mode raising ValueError
- No test for normalize_text() with None, list, or numeric inputs
- No regression test pinning the saved model exact metric values
- No test that evaluate_binary_classifier handles edge cases (all-one-class predictions)

---

## 10. Experimental Validity

### Safeguards Implemented

| Safeguard | Implementation |
|---|---|
| Fixed test set | split=="test" field in ragtruth.jsonl; never modified |
| Source-level separation | No source_id in both train and test (verified) |
| TF-IDF fit on train only | fit_transform(train) then transform(test) |
| Deterministic seed | LogisticRegression(random_state=42), validation Random(seed=42) |
| Config snapshot | config.yaml saved into experiment directory |
| Prediction archive | test_predictions.jsonl for post-hoc analysis |
| Metrics archive | metrics.json is the authoritative result record |

### Known Weaknesses

- context: str type annotation in schema.py is misleading (actual type is str or dict)
- "from numpy import record" in validation.py line 1 is a dead import (deprecated numpy name)
- validate_records() contains inline development comments that should be cleaned up
- Validation split (1,512 records) is not used by the baseline trainer
- No confidence intervals on reported metrics
- Test set positive rate (34.93%) differs from overall rate (43.08%) - inherent to RAGTruth split

---

## 11. Design Decisions and Rationale

### Context + Answer input (not Q+C+A)
Reason: Isolates context-answer grounding signal. Question conflates grounding with relevance.
Future: Q+C+A tested in input ablation.

### TF-IDF over bag-of-words
Reason: Down-weights ubiquitous terms. IDF weighting highlights discriminative vocabulary.
Consequence: 437,871 sparse features. Memory-efficient.

### Source-grouped validation split
Reason: Multiple model responses per source. Without grouping, the model sees the same source context in both train and validation.
Consequence: Slightly uneven split by record count.

### Fixed test set
Reason: All future models compared on the same 2,700 records. Results are not cherry-picked.
Consequence: Test class balance cannot be adjusted without breaking comparability.

### json.dumps(sort_keys=True) for dict contexts
Reason: Dict key iteration order in Python 3.7+ is insertion-order. Different records with the same logical content could produce different strings without sort_keys=True.
Consequence: Serialized context is a flat JSON string; structural information preserved as tokens.

### Save full prediction bundle including probabilities
Reason: Enables post-hoc threshold analysis, calibration checks, per-subgroup analysis without re-running the model.

---

## 12. Current Limitations

### Lexical Feature Limitations

TF-IDF has no concept of: semantic similarity (synonyms), negation, cross-sentence coherence, logical entailment. The 39.2% false-negative rate is partly attributable to hallucinated answers that use similar vocabulary to the context.

### Long-Context Handling

TF-IDF treats the entire context as a bag of n-grams with no notion of position. For long structured dict contexts, the model cannot focus on the most relevant portion.

### No Semantic Grounding

A claim like "the company was founded in 1990" cannot be verified against "the organization was established in 1990" because "company"/"organization" and "founded"/"established" are different tokens.

### Dataset-Specific Behavior

All results are on RAGTruth. Generalizability to other hallucination datasets is unknown.

### No Threshold Analysis

Confusion matrix uses default 0.5 threshold. Saved probabilities enable analysis but no threshold sweep has been performed.

### Technical Debt

- "from numpy import record" in validation.py: dead import, should be removed
- context: str annotation in schema.py: should be context: Any or str | dict
- Inline development comments in validate_records() should be cleaned up

---

## 13. Next Stage: DeBERTa-v3-base

### Why DeBERTa Next

The classical baseline establishes what lexical features achieve. The scientific question is: does a pretrained contextual model improve over this? DeBERTa-v3-base is chosen because it uses disentangled attention (separate content and position embeddings) and is a strong discriminative encoder for classification tasks. The model name and config fields are already in configs/default.yaml.

### What Needs to Be Built

1. Tokenization pipeline: Tokenize "[CONTEXT] <ctx> [ANSWER] <ans>" with DeBERTa tokenizer. Handle max_length=512.
2. Model: DeBERTa-v3-base with linear classification head (num_labels=2).
3. Dataset class: torch.utils.data.Dataset wrapping RAGTruthRecord objects.
4. Training loop: Use validation split (1,512 records) for early stopping. AdamW, lr=2e-5, weight_decay=0.01, epochs=3, batch_size=8.
5. Evaluation: Same evaluate_binary_classifier() function, identical metrics.
6. Artifact saving: Same structure as baseline_tfidf_logreg.

### Evaluation Protocol (must be identical)

- Use the exact same test set (split=="test", 2,700 records)
- Report same metrics (Accuracy, Precision, Recall, F1, ROC-AUC, Average Precision, Confusion Matrix)
- Same positive class definition (1 = hallucinated)
- Save predictions in same test_predictions.jsonl format

### Long-Context Consideration

DeBERTa tokenizer has a max token limit (512 tokens). Some Data-to-text records with large dict contexts will be truncated. The truncation strategy must be documented and applied consistently.

### Suggested File Structure

```
src/hallucination_detector/
    modeling/deberta_classifier.py   # DeBERTa model wrapper
    training/deberta_baseline.py     # Training loop with early stopping
    data/torch_dataset.py            # torch.utils.data.Dataset wrapper
scripts/
    train_deberta.py                 # Entry point
reports/experiments/
    baseline_deberta/                # Same artifact structure
```

---

## 14. Future Experiment Roadmap

### Experiment 1 - Transformer Baseline (DeBERTa)
Research question: Does a contextual pretrained model improve over TF-IDF lexical features?
Independent variable: Model architecture.
Controlled: Input format (C+A), dataset, test set, seed, evaluation metrics.
Status: Not started.

### Experiment 2 - Input Ablation
Research question: Which input fields (A-only, Q+A, C+A, Q+C+A) contribute most?
The input_mode parameter in TfidfLogisticClassifier already supports ca, qa, qca.
Status: Not started.

### Experiment 3 - Robustness Evaluation
Research question: Does the detector hold under degraded context (shuffled, truncated, replaced, contradictory)?
Status: Not started.

### Experiment 4 - External Generalization
Research question: Does the detector transfer beyond RAGTruth?
Status: Not started.

### Experiment 5 - Error Analysis
Research question: Which hallucination types are systematically missed?
Method: Stratify false negatives by annotation type, model, task type, answer length, lexical overlap.
Note: test_predictions.jsonl + annotated dataset enables this without re-running the model.
Status: Not started.

### Experiment 6 - Final Comparison
Research question: Best trade-off across all conditions.
Status: Not started.

---

## 15. File-to-File Dependency Map

```
data/schema.py                     [RAGTruthRecord definition]
  |-- data/preprocessing.py        [creates records]
  |     |-- scripts/prepare_data.py
  |     `-- scripts/audit_dataset.py
  |-- data/validation.py           [validates records]
  |     |-- scripts/prepare_data.py
  |     |-- scripts/audit_dataset.py
  |     `-- tests/test_validation.py
  |-- data/splitting.py            [splits records]
  |     |-- scripts/prepare_data.py
  |     `-- tests/test_splitting.py
  |-- modeling/tfidf_logistic.py   [classifier]
  |     |-- training/baseline.py
  |     |-- inference/baseline.py
  |     `-- tests/test_baseline_model.py
  |-- evaluation/metrics.py        [evaluation]
  |     `-- training/baseline.py
  |-- training/baseline.py         [orchestration]
  |     `-- scripts/train_baseline.py
  `-- inference/baseline.py        [inference; not yet called by any script]

data/loader.py                     [load_jsonl]
  |-- training/baseline.py
  |-- scripts/prepare_data.py
  |-- scripts/audit_dataset.py
  |-- scripts/check_processed_data.py
  `-- tests/test_data.py

configs/default.yaml
  `-- scripts/train_baseline.py    [reads config["baseline"] and config["project"]["seed"]]
```

---

## 16. How to Continue Development

### Immediate: Reproduce and Freeze the Baseline

```bash
uv run pytest -q
uv run python scripts/train_baseline.py
# Expected: accuracy=0.7678, f1=0.6464, roc_auc=0.7977
```

### Next: Implement DeBERTa Baseline

1. Add transformers + torch to the environment.
2. Create src/hallucination_detector/data/torch_dataset.py.
3. Create src/hallucination_detector/modeling/deberta_classifier.py.
4. Create src/hallucination_detector/training/deberta_baseline.py (with validation-based early stopping).
5. Create scripts/train_deberta.py.
6. Use the same evaluate_binary_classifier() function and same test set.
7. Save artifacts to reports/experiments/baseline_deberta/.
8. Compare metrics.json files.

---

## 17. Milestones

### Milestone 1 - Data Infrastructure (Sep 29-30, 2026)
data/loader.py, data/schema.py, data/preprocessing.py, data/validation.py, data/splitting.py, scripts/prepare_data.py, scripts/audit_dataset.py, scripts/check_processed_data.py, data/processed/ragtruth.jsonl

### Milestone 2 - Exploratory Data Analysis (Sep 30, 2026)
evaluation/eda.py, scripts/run_eda.py, scripts/run_advanced_eda.py, reports/eda/ (19 figures, eda_statistics.csv, eda_summary.md)

### Milestone 3 - Classical Baseline (Sep 30, 2026)
modeling/tfidf_logistic.py, training/baseline.py, inference/baseline.py, evaluation/metrics.py, scripts/train_baseline.py, configs/default.yaml, reports/experiments/baseline_tfidf_logreg/ (model.pkl, metrics.json, config.yaml, test_predictions.jsonl)

### Milestone 4 - Tests (Sep 30, 2026)
tests/test_data.py, tests/test_preprocessing.py, tests/test_splitting.py, tests/test_validation.py, tests/test_baseline_model.py. All 15 tests passing.

### Milestone 5 - DeBERTa Transformer Baseline
Status: Not started.
