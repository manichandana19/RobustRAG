# RobustRAG End-to-End System Audit: DeBERTa-v3 vs. TF-IDF

## Executive Summary

The reported stagnation of DeBERTa-v3-base (F1 66.32% vs. TF-IDF 64.64%, with inferior ROC-AUC and Average Precision) is **not caused by transformer capacity limitations or fundamental model shortcomings**.

Instead, the performance collapse is driven by a **critical tokenization bug**:
> [!CAUTION]
> **The Answer-Erasure Truncation Bug:**
> In the current implementation, inputs are formatted as a single string `[CONTEXT] <context> [ANSWER] <answer>` and passed to the tokenizer with `max_length=512` and default right-truncation. Because context precedes answer and contexts in RAGTruth are long (median 511–530 tokens), **the tokenizer systematically chops off the generated answer from the right**.
> 
> Across the entire test set, **50.2% of answers were 100% deleted** before reaching DeBERTa, and another **18.0% were partially truncated**. Only 31.7% of test inputs retained their full answer. In Data-to-text, **97.3% of answers were completely erased**.

When evaluated on the subset of data where answers remained intact (such as the QA task), **DeBERTa dramatically outperforms TF-IDF across every single metric**. On Data-to-text—where DeBERTa was effectively blind to the answer—its ROC-AUC dropped to **0.5206 (random guessing)**, dragging down the global metrics. Meanwhile, TF-IDF was unaffected by sequence length because sparse bag-of-words vectorization spans arbitrary text lengths without truncation.

---

## Metric Autopsy: Global vs. Task-Stratified Breakdown

The global aggregate numbers obscure what is actually happening across the three constituent tasks of RAGTruth:

### Global Metrics (Test Set, N = 2,700)
| Metric | TF-IDF + Logistic Reg | DeBERTa-v3-base | Delta |
|---|:---:|:---:|:---:|
| **Accuracy** | **76.78%** | 75.78% | -1.00% |
| **Precision (Hallucinated)** | **69.04%** | 64.46% | -4.58% |
| **Recall (Hallucinated)** | 60.76% | **68.29%** | +7.53% |
| **F1-Score (Hallucinated)** | 64.64% | **66.32%** | +1.68% |
| **ROC-AUC** | **0.7977** | 0.7912 | -0.0065 |
| **Average Precision (PR-AUC)** | **0.7333** | 0.6222 | **-0.1111** |

### Task-Level Stratification (N = 900 per task)

```
========================================================================================
TASK        METRIC            TF-IDF        DeBERTa        DELTA        ANSWER INTACT?
========================================================================================
QA          ROC-AUC           0.6998        0.8261        +0.1263       YES (97.3%)
            Avg Precision     0.3173        0.5716        +0.2543       YES
            F1-Score          0.1887        0.5059        +0.3172       YES
            Accuracy          0.8089        0.8611        +0.0522       YES
----------------------------------------------------------------------------------------
Data2txt    ROC-AUC           0.8221        0.5206        -0.3015       NO (0.0% intact,
            Avg Precision     0.8744        0.6545        -0.2199           97.3% erased)
            F1-Score          0.8150        0.7830        -0.0320       NO
            Accuracy          0.7211        0.6433        -0.0778       NO
----------------------------------------------------------------------------------------
Summary     ROC-AUC           0.5904        0.5880        -0.0024       PARTIAL (50.7%
            Avg Precision     0.3132        0.2682        -0.0450                erased)
            F1-Score          0.0000        0.0095        +0.0095       PARTIAL
            Accuracy          0.7733        0.7689        -0.0044       PARTIAL
========================================================================================
```

### Non-Truncated vs. Truncated Subset Comparison
- **Non-Truncated Records (N = 856, length <= 512 tokens)**:
  - TF-IDF Accuracy: **83.29%**
  - DeBERTa Accuracy: **85.40%** (DeBERTa leads by +2.11%)
- **Truncated Records (N = 1,844, length > 512 tokens)**:
  - TF-IDF Accuracy: **73.75%**
  - DeBERTa Accuracy: **71.31%** (DeBERTa collapses because answer is missing)

---

## Detailed Root Cause Analysis

### 1. Confirmed Bug: Answer Erasure via Right-Truncation of Single String
- **Location**: [torch_dataset.py](file:///c:/Users/nishantwhat/CODE/RobustRAG/src/hallucination_detector/data/torch_dataset.py#L112-L121)
- **Mechanism**:
  ```python
  text = build_input_text(record, self.input_mode)  # "[CONTEXT] <ctx> [ANSWER] <ans>"
  encoded = self.tokenizer(
      text,
      max_length=self.max_length,
      truncation=True, # Default: truncation_side="right"
      padding=False,
      return_attention_mask=True,
  )
  ```
- **Consequence**:
  - In Hugging Face tokenizers, `truncation=True` on a single string always truncates from the right.
  - In training: **7,206 of 13,578 training records (53.1%) had their answer completely removed**. Another 2,631 (19.4%) were cut in half. The model spent >70% of its training steps learning to predict hallucination labels from context alone without seeing what the model actually generated.
  - In Data-to-text test: **876 of 900 records (97.3%) had zero answer tokens**. DeBERTa simply saw the first 512 tokens of a raw JSON dict and learned to output a constant high hallucination probability (~0.78), producing an ROC-AUC of 0.5206.

### 2. Confirmed Architectural Bug: Custom Tokens vs. Native Sequence Pairs
- **Location**: [deberta_classifier.py](file:///c:/Users/nishantwhat/CODE/RobustRAG/src/hallucination_detector/modeling/deberta_classifier.py#L250-L261)
- **Mechanism**:
  ```python
  self.tokenizer.add_special_tokens({"additional_special_tokens": ["[CONTEXT]", "[ANSWER]"]})
  self.model.resize_token_embeddings(len(self.tokenizer))
  ```
- **Impact**:
  1. Adding custom tokens forces `resize_token_embeddings`, initializing these new token embeddings with **random Gaussian noise**.
  2. DeBERTa-v3 is a cross-encoder pretrained on sentence pairs separated by native `[SEP]` tokens (`[CLS] text_a [SEP] text_b [SEP]`) with disentangled relative position encodings.
  3. Bypassing sequence-pair encoding disabled HF's native pair truncation mechanisms (e.g. `truncation="only_first"` or `truncation="only_second"`).

### 3. Confirmed Data Format Issue: Naive Dict Serialization for Data-to-Text
- **Location**: [tfidf_logistic.py](file:///c:/Users/nishantwhat/CODE/RobustRAG/src/hallucination_detector/modeling/tfidf_logistic.py#L15-L37)
- **Mechanism**:
  - `normalize_text` serializes dictionary context using `json.dumps(value, ensure_ascii=False, sort_keys=True)`.
  - In Yelp Data-to-text records, the context contains raw customer reviews (`review_info` averages >1,100 characters) plus business attributes, operating hours, and metadata.
  - Naive JSON formatting injects massive syntax token overhead (`"`, `:`, `{`, `}`, `,`), blowing past 700–1,500 subword tokens per example.

### 4. Confirmed Training & Optimization Deficiencies
- **Location**: [deberta_baseline.py](file:///c:/Users/nishantwhat/CODE/RobustRAG/src/hallucination_detector/training/deberta_baseline.py#L873-L877)
- **Deficiencies**:
  - **No Parameter Grouping for Weight Decay**: `torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)` applies weight decay indiscriminately to biases and LayerNorm parameters. Standard transformer practice requires excluding biases and LayerNorm (`no_decay = ["bias", "LayerNorm.weight"]`).
  - **Aggressive Early Stopping on Small Batches**: `early_stopping_patience: 1` with effective batch size 8 caused training to halt after only Epoch 2 when validation loss fluctuated from 0.572 to 0.591.
  - **Validation Loss Plateau**: Validation loss remained high (0.572) because 53% of inputs had no answer, making cross-entropy loss plateau.

### 5. Confirmed Evaluation & Calibration Flaw: Single Global Threshold Across Asymmetric Tasks
- **Observation**:
  - RAGTruth has disparate base hallucination rates: Data2txt (64.3%), Summary (22.7%), QA (17.8%).
  - At the fixed default decision threshold of 0.5:
    - **Summary**: TF-IDF predicted 0 positives (0 TP, 0 FP, F1 = 0.00). DeBERTa predicted 6 positives (1 TP, 5 FP, F1 = 0.009).
    - **QA**: TF-IDF predicted 52 positives (20 TP, 32 FP, F1 = 0.188). DeBERTa predicted 93 positives (64 TP, 29 FP, F1 = 0.505).
    - **Data2txt**: TF-IDF predicted 778 positives. DeBERTa predicted **all 900 instances as hallucinated**.
  - A single uncalibrated 0.5 threshold fails across multi-task mixtures with 3:1 base-rate discrepancies.

### 6. Repository Debt & Artifact Drift
- **Codebase Cleanliness**: Over 700 lines of dead, commented-out legacy code in [torch_dataset.py](file:///c:/Users/nishantwhat/CODE/RobustRAG/src/hallucination_detector/data/torch_dataset.py#L1-L80), [deberta_classifier.py](file:///c:/Users/nishantwhat/CODE/RobustRAG/src/hallucination_detector/modeling/deberta_classifier.py#L1-L198), and [deberta_baseline.py](file:///c:/Users/nishantwhat/CODE/RobustRAG/src/hallucination_detector/training/deberta_baseline.py#L1-L604).
- **Type Inaccuracy**: [schema.py](file:///c:/Users/nishantwhat/CODE/RobustRAG/src/hallucination_detector/data/schema.py#L11) defines `context: str`, whereas 68% of records contain `dict`.
- **Deprecated Imports**: [validation.py](file:///c:/Users/nishantwhat/CODE/RobustRAG/src/hallucination_detector/data/validation.py#L1) contains unused `from numpy import record`.

---

## Actionable Remediation Plan

```mermaid
graph TD
    A["Fix Tokenization & Sequence Pairs"] --> B["Compact Dict Linearization"]
    B --> C["Optimizer & Weight Decay Grouping"]
    C --> D["Task-Stratified & Threshold Evaluation"]
    D --> E["Benchmarking Against Frozen Test Split"]
```

### Phase 1: Small, Testable, Zero-Cost Code Fixes (Immediate)

1. **Adopt Native Sequence-Pair Encoding with Context-Only Truncation**:
   - Reconfigure `torch_dataset.py` to tokenize sequence pairs:
     ```python
     encoded = tokenizer(
         text=context_text,
         text_pair=answer_text,
         max_length=512,
         truncation="only_first",  # Truncates context, NEVER truncates answer
         padding=False,
         return_attention_mask=True,
     )
     ```
   - *Impact*: Guarantees **100% of answer tokens are preserved** across all tasks and records.
   - Remove custom `[CONTEXT]` and `[ANSWER]` tokens and embedding resizing; use native pretrained `[SEP]` separation.

2. **Group Parameters in Optimizer**:
   - In `deberta_baseline.py`, exclude LayerNorm and biases from weight decay:
     ```python
     no_decay = ["bias", "LayerNorm.weight", "layer_norm.weight"]
     optimizer_grouped_parameters = [
         {"params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], "weight_decay": 0.01},
         {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
     ]
     ```

3. **Clean Up Codebase Dead Code & Types**:
   - Delete commented-out blocks in `torch_dataset.py`, `deberta_classifier.py`, and `deberta_baseline.py`.
   - Update `schema.py` to `context: str | dict[str, Any]`.
   - Remove `from numpy import record` in `validation.py`.

### Phase 2: Context Representation Optimization (Pre-Training)

1. **Compact Linearization for Dict Contexts**:
   - Rather than `json.dumps`, format key-values compactly (e.g., `Name: Apna Indian Kitchen | Address: 718 State St | Reviews: ...`).
   - If context exceeds token budget, retain high-density fields (attributes, categories) and truncate trailing reviews, rather than truncating the generated answer.

### Phase 3: Evaluation & Calibration Rigor

1. **Validation Threshold Optimization**:
   - Tune optimal binary classification threshold on the validation split before testing.
2. **Task-Stratified Metrics Reporting**:
   - Always report metrics stratified by task (`QA`, `Summary`, `Data2txt`) alongside aggregate metrics to prevent Simpson's Paradox / task-masking effects.

---

## Conclusion & Verdict

The transformer architecture is **not failing**. When DeBERTa was permitted to see both the context and the answer (the QA task), it outperformed the lexical baseline by **+12.6 points in ROC-AUC, +25.4 points in Average Precision, and +31.7 points in F1**.

The apparent failure on the full dataset is an artifact of the right-truncation bug wiping out the answer in over 50% of the dataset (and 97.3% of Data-to-text). Implementing the sequence-pair tokenization fix will restore the signal and establish a true, superior transformer baseline.
