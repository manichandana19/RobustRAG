# RobustRAG Comprehensive Technical Audit Report

## 1. Executive Summary

This audit report documents the empirical and architectural findings from the **RobustRAG** research repository. RobustRAG investigates hallucination detection in Retrieval-Augmented Generation (RAG) systems using the human-annotated [RAGTruth](https://arxiv.org/abs/2401.00396) benchmark ($17,790$ records; $2,700$ test records).

### Key Audit Findings:
1. **The Answer-Erasure Pathology in 512-Token Encoders:**
   In early transformer baseline implementations (DeBERTa-v3-base), inputs were formatted as a single monolithic string `[CONTEXT] <context> [ANSWER] <answer>` subject to default right-truncation. Because RAG contexts frequently exceed 500 subwords, **50.2% of test answers were completely truncated** (and 97.3% in Data-to-text). This reduced DeBERTa to random guessing on Data-to-text (ROC-AUC 0.5206), while on non-truncated QA records DeBERTa dramatically outperformed classical baselines (ROC-AUC 0.8261 vs. 0.6998).
2. **Answer-Preserving Longformer-1024 (Head Truncation) is Current State-of-the-Art:**
   By explicitly tokenizing the generated answer first and allocating the remaining token budget to context within a 1024-token sliding-window Longformer architecture, the model achieved project-leading performance across Accuracy (**79.78%**), Precision (**74.36%**), F1 (**68.94%**), ROC-AUC (**0.8296**), and PR-AUC (**0.7527**).
3. **Head vs. Tail Truncation Dynamics:**
   Retaining the trailing portion of the context closest to the answer (Head truncation) outperformed retaining the beginning (Tail truncation: 77.56% Acc, 0.8187 ROC-AUC), cutting false positive hallucinations by **30.6%** (209 vs. 301 false positives).
4. **Smart NLP Preprocessing (Q+C+A Longformer):**
   The repository now includes an implemented and evaluated `SentenceSelector` pipeline that scores context sentences based on Question and Answer IDF overlap with explicit contradiction bonuses (divergent numbers, polarity discordance). The evaluated checkpoint (`reports/experiments/longformer_1024_smart_qca/`) achieved **0.8144 ROC-AUC** and **0.7420 Average Precision** on the 2,700-record test benchmark.

---

## 2. Verified Benchmark Comparison

All results are compiled directly from the authoritative `metrics.json` experiment artifacts on the fixed, source-disjoint test set ($N = 2,700$ records, $34.93\%$ positive rate):

| Model Architecture | Context Cap | Input Mode | Attention / Selection Strategy | Accuracy | Precision | Recall | F1-Score | ROC-AUC | PR-AUC |
| :--- | :---: | :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **TF-IDF + Logistic Reg** | Unlimited | C + A | Sublinear TF, unigram+bigram (437k features), $C=1.0$ | 76.78% | 69.04% | 60.76% | 64.64% | 0.7977 | 0.7333 |
| **DeBERTa-v3-base** | 512 | C + A | Single-string right-truncation (Answer erased in 50.2% of test cases) | 75.78% | 64.46% | **68.29%** | 66.32% | 0.7912 | 0.6222 |
| **Longformer (Head)** | 1024 | C + A | Answer-preserved tokenization + keeps trailing context closest to answer | **79.78%** | **74.36%** | 64.26% | **68.94%** | **0.8296** | **0.7527** |
| **Longformer (Tail)** | 1024 | C + A | Answer-preserved tokenization + keeps initial context | 77.56% | 67.94% | 67.66% | 67.80% | 0.8187 | 0.7331 |
| **Longformer Smart QCA** | 1024 | Q + C + A | Q & A preserved + extractive sentence selector (IDF + contradiction bonus) | 76.33% | 66.49% | 65.01% | 65.74% | 0.8144 | 0.7420 |

---

## 3. Detailed Root-Cause & Architectural Analysis

### 3.1 The Answer-Erasure Pathology (DeBERTa 512)
- **Root Mechanism:**
  In `torch_dataset.py` (legacy implementation) and `deberta_classifier.py`, input was formatted as `"[CONTEXT] " + context + " [ANSWER] " + answer`. Hugging Face's tokenizer with `max_length=512, truncation=True` applied default right-truncation (`truncation_side="right"`).
- **Empirical Toll:**
  - In training: $7,206$ of $13,578$ records ($53.1\%$) had zero answer tokens; an additional $19.4\%$ had fragmented answers.
  - In Data-to-text test records: $876$ of $900$ records ($97.3\%$) had zero answer tokens.
  - The model was forced to predict hallucination labels from raw context snippets alone, producing constant high positive probabilities ($\approx 0.78$) and collapsing ROC-AUC to $0.5206$.
- **Lesson:** Sequence classification cross-encoders must never apply unconstrained single-string truncation when verifying evidence-hypothesis pairs.

### 3.2 Dynamic Answer-Preserving Tokenization (Longformer)
- **Implementation in `src/hallucination_detector/data/torch_dataset.py`:**
  ```python
  def encode_pair_with_answer_preservation(
      context, answer, tokenizer, max_length=1024, truncation_strategy="head", global_attention_mode="cls"
  ):
      ans_tokens = tokenizer.encode(normalize_text(answer), add_special_tokens=False)
      ctx_tokens = tokenizer.encode(normalize_text(context), add_special_tokens=False)
      available_budget = max(1, max_length - 4)  # <s> ctx </s></s> ans </s>
      ctx_budget = available_budget - len(ans_tokens)
      # Truncate context only, leaving answer 100% intact
  ```
- **Result:** $100\%$ of answer tokens are guaranteed to reach the self-attention and sliding-window layers across all evaluation records.

### 3.3 Head vs. Tail Context Truncation
- In Longformer-1024 C+A modeling:
  - **Head truncation** keeps the *end* of the context: `ctx_tokens[-ctx_budget:]`.
  - **Tail truncation** keeps the *beginning* of the context: `ctx_tokens[:ctx_budget]`.
- **Finding:** Head truncation outperformed Tail truncation by **+2.22% Accuracy, +6.42% Precision, and +0.0109 ROC-AUC**. Qualitative inspection reveals that for long structured reviews or news articles, the concluding passages often contain summary judgments, ratings, or closing statements that directly align with the generated response.

### 3.4 Longformer Smart QCA Architecture & Preprocessing
- **Location:** `src/hallucination_detector/data/sentence_selector.py` & `longformer_smart_dataset.py`.
- **Design:**
  - Token sequence: `<s> Question </s></s> Context </s></s> Answer </s>` ($6$ delimiter tokens).
  - Both Question and Answer tokens are preserved at 100% fidelity.
  - Context is budgeted dynamically: `ctx_budget = max_length - 6 - (len(q_tokens) + len(ans_tokens))`.
  - Extractive `SentenceSelector`:
    1. Fits smooth IDF statistics strictly on the training split: $\text{IDF}(t) = \ln\left(\frac{1 + N}{1 + \text{DF}(t)}\right) + 1$.
    2. Ranks candidate context sentences via:
       $$\text{Score}(s) = \alpha \cdot \text{Score}_Q(s) + (1 - \alpha) \cdot \text{Score}_A(s) + \text{Bonus}_{\text{contradiction}} + \text{Bias}_{\text{lead}}$$
    3. Contradiction bonus: $+2.0$ for divergent numerical statements ($\text{Nums}(A) \not\subseteq \text{Nums}(s)$) and $+2.5$ for polarity discordance (one clause negated, one affirmative with overlapping terms).
    4. Packs top-ranked sentences up to the token budget and sorts them back into their original document order to preserve narrative flow.
- **Evaluation Status:** Verified. Yields **0.8144 ROC-AUC**, substantially exceeding classical and DeBERTa baselines while processing all three elements (Question, Context, and Answer).

---

## 4. Comprehensive Audit Classification Matrix

Findings and repository characteristics are classified into **Confirmed Facts**, **Potential Risks**, and **Unverified Claims**:

| Item / Finding | Classification | Evidence & Impact |
| :--- | :---: | :--- |
| **Answer-Erasure Bug in DeBERTa** | **Confirmed** | Verified via script audit and task-stratified metrics (ROC-AUC 0.5206 in Data2txt vs. 0.8261 in QA). |
| **Longformer Head 1024 Performance** | **Confirmed** | Verified by `reports/experiments/longformer_1024_head/metrics.json` ($79.78\%$ Acc, $0.8296$ ROC-AUC). |
| **Longformer Smart QCA Evaluation** | **Confirmed** | Verified by `reports/experiments/longformer_1024_smart_qca/metrics.json` ($76.33\%$ Acc, $0.8144$ ROC-AUC). |
| **Dataset Source Disjointness** | **Confirmed** | Verified in `data/splitting.py` and `scripts/check_processed_data.py`. No source passage overlap between train and test. |
| **Lightweight Unit Test Suite** | **Confirmed** | 15 core tests in `test_data.py`, `test_preprocessing.py`, `test_splitting.py`, `test_validation.py`, and `test_baseline_model.py` pass cleanly. |
| **Single-Threshold (0.5) Calibration** | **Potential Risk** | Base hallucination rates vary widely across tasks (Data2txt 64.3%, Summary 22.7%, QA 17.8%). A uniform 0.5 decision boundary produces task-imbalanced precision/recall. |
| **Pytest Full-Suite Timeout / Hanging** | **Potential Risk** | Heavy integration tests (`test_longformer_classifier.py`, `test_bilstm_baseline.py`) instantiate full neural models on CPU without mocks, causing `pytest` to stall on resource-constrained environments. |
| **Dead Code & Type Annotation Drift** | **Potential Risk** | `schema.py` annotates `context: str`, but $68.2\%$ of records have `dict` context (handled safely at runtime via `normalize_text`, but types are inaccurate). Unused legacy comments remain in training loops. |
| **4096-Token Longformer Generalization** | **Unverified** | Longformer configs currently evaluate at `max_length: 1024`. Full 4096-token sequence scaling has not yet been benchmarked on GPU due to memory budget. |
| **Cross-Dataset Generalization** | **Unverified** | All experiments have been conducted on RAGTruth. Generalization to other hallucination benchmarks (e.g., HaluEval, FaithDial) is untested. |

---

## 5. Prioritized Remediation & Recommendations

### Tier 1: Immediate Engineering Polish
1. **Mark Slow Integration Tests in Pytest:**
   Decorate transformer-instantiating tests (`test_longformer_classifier.py`, `test_bilstm_baseline.py`, `test_longformer_smart_nlp.py`) with `@pytest.mark.slow` and provide a mock tokenizer fixture so that standard CI (`pytest -m "not slow"`) runs in under 5 seconds.
2. **Synchronize Dataclass Schema:**
   Update `RAGTruthRecord` in `src/hallucination_detector/data/schema.py` from `context: str` to `context: str | dict[str, Any]` to match true runtime types.
3. **Task-Specific Calibration & Threshold Sweeps:**
   Incorporate threshold tuning per task in `evaluation/metrics.py` to optimize decision boundaries for differing base rates.

### Tier 2: Algorithmic & Modeling Enhancements
1. **Ablation of SentenceSelector Parameters:**
   Evaluate sensitivity of $\alpha$ (balance between Q and A overlap) and contradiction weights (divergent numbers vs. polarity shifts) in `SentenceSelector` to close the gap with pure Head truncation on Longformer.
2. **Full Context Window Expansion (2048 - 4096 Tokens):**
   Run Longformer with 2048 tokens and gradient checkpointing to capture 99% of RAGTruth contexts without any sentence elimination.

---

## 6. Concrete Next Experiment Plan

### Experiment: Longformer QCA SentenceSelector Tuning & Hybrid Windowing
- **Objective:** Determine if combining Head-bias windowing with contradiction-focused sentence selection outperforms purely heuristic Head truncation.
- **Hypothesis:** Preserving the final paragraph of context (which contains summary review/conclusion sentences) while selectively inserting candidate contradictory sentences from earlier in the document will minimize false positives and elevate ROC-AUC above $0.835$.
- **Configuration:**
  - Model: `allenai/longformer-base-4096`
  - Max Length: $1024$ tokens
  - Input Mode: `qca`
  - Selector: `SentenceSelector` with hybrid tail-allocation (50% token budget to contiguous tail context, 50% to high-scoring contradictory sentences).
- **Execution Script:** `hallucination_detector.training.longformer_smart_train` with updated `longformer_qca_smart.yaml`.
- **Target Artifact:** `reports/experiments/longformer_1024_hybrid_qca/metrics.json`.
