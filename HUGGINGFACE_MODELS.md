# Hugging Face Models Guide & Remote Inference

This guide documents the five models published to Hugging Face under the `nishantwhat` organization for the **RobustRAG** benchmark, their hardware requirements, and how to use them with the Streamlit app.

---

## 1. Remote Model Repository Catalog

All model weights and configuration artifacts are hosted publicly on Hugging Face:

| # | Model Architecture | Hugging Face Repository | Input Mode | Checkpoint Size | Context Cap |
|---|---|---|---|---|---|
| 1 | **Longformer (Head Truncation)** | [`nishantwhat/robustrag-longformer-head`](https://huggingface.co/nishantwhat/robustrag-longformer-head) | Context + Answer | ~595 MB | 1024 tokens |
| 2 | **Longformer (Smart Q+C+A)** | [`nishantwhat/robustrag-longformer-qca`](https://huggingface.co/nishantwhat/robustrag-longformer-qca) | Question + Context + Answer | ~595 MB | 1024 tokens |
| 3 | **DeBERTa-v3-base** | [`nishantwhat/robustrag-deberta`](https://huggingface.co/nishantwhat/robustrag-deberta) | Context + Answer | ~737 MB | 512 tokens |
| 4 | **BiLSTM + Attention** | [`nishantwhat/robustrag-bilstm-attention`](https://huggingface.co/nishantwhat/robustrag-bilstm-attention) | Context + Answer | ~20.5 MB | 512 tokens |
| 5 | **TF-IDF + Logistic Reg** | [`nishantwhat/robustrag-tfidf-lr`](https://huggingface.co/nishantwhat/robustrag-tfidf-lr) | Context + Answer | ~15.4 MB | Unlimited |

---

## 2. Model Requirements & Modalities

### 1. Longformer (Head Truncation) - Project State-of-the-Art
- **Repo ID**: `nishantwhat/robustrag-longformer-head`
- **Architecture**: `LongformerForSequenceClassification` with local attention sliding window ($w=512$) and global attention on the `<s>` `[CLS]` token.
- **Input Strategy**: Answer tokens are prioritized and preserved at 100% fidelity. Context is truncated from the head (retaining trailing context closest to the answer).
- **Performance**: Test Accuracy: **79.78%**, ROC-AUC: **0.8296**, PR-AUC: **0.7527**.

### 2. Longformer (Smart Q+C+A)
- **Repo ID**: `nishantwhat/robustrag-longformer-qca`
- **Architecture**: Longformer with `SentenceSelector` extractive preprocessing.
- **Input Strategy**: Allocates token budget to Question and Answer first. Context sentences are ranked using IDF relevance and contradiction bonuses (+2.0 numerical discrepancy, +2.5 polarity inversion).
- **Performance**: Test Accuracy: **76.33%**, ROC-AUC: **0.8144**, PR-AUC: **0.7420**.

### 3. DeBERTa-v3-base
- **Repo ID**: `nishantwhat/robustrag-deberta`
- **Architecture**: `DebertaV2ForSequenceClassification` (512 max position embeddings).
- **Input Strategy**: Single-string formatting `[CONTEXT] <ctx> [ANSWER] <ans>`.
- **Limitation**: Prone to answer-erasure when context exceeds 512 tokens. Decisive on QA tasks where answers remain intact (ROC-AUC 0.8261).

### 4. BiLSTM + Bahdanau Attention
- **Repo ID**: `nishantwhat/robustrag-bilstm-attention`
- **Architecture**: 2-layer Bidirectional LSTM ($128$ hidden dimensions) with additive Bahdanau self-attention and 50,000-token custom vocabulary.
- **Artifacts**: PyTorch state dictionary (`bilstm_weights.pt`), vocabulary dictionary (`vocab.json`), and hyperparameter configuration (`model_config.json`).

### 5. TF-IDF + Logistic Regression
- **Repo ID**: `nishantwhat/robustrag-tfidf-lr`
- **Architecture**: Classical n-gram vectorizer (437,871 unigram + bigram features with sublinear TF) + L2-regularized logistic regression ($C=1.0$).
- **Artifacts**: Model bundle (`model.pkl`), split vectorizer (`tfidf_vectorizer.pkl`), and classifier (`logistic_classifier.pkl`).

---

## 3. How Downloads & Caching Work

The unified `ModelRegistry` in `src/hallucination_detector/inference/model_registry.py` handles model retrieval lazily:
1. **Lazy Loading**: Starting the application does **not** download all models. A model is only fetched when selected.
2. **Local Priority**: If local experiment checkpoints are already present in `reports/` or `models/`, the registry loads directly from disk without network calls.
3. **Automatic Hugging Face Caching**: When running in a clean environment without local weights, weights are automatically retrieved via `huggingface_hub` into the standard Hugging Face cache directory (`~/.cache/huggingface/hub/`). Subsequent calls reuse the cached files instantly.
4. **Authentication**: All five repositories are public; no API token or login is required to download or perform inference.

---

## 4. Hardware & Disk Space Considerations

| Model | Download / Disk Footprint | Minimum RAM / VRAM | Recommended Device |
|---|---|---|---|
| **TF-IDF + Logistic Reg** | ~15 MB | < 1 GB RAM | CPU |
| **BiLSTM + Attention** | ~21 MB | < 1 GB RAM | CPU or CUDA |
| **Longformer (Head / QCA)** | ~595 MB | 4 GB RAM / 2 GB VRAM | CUDA (FP16/BF16) or CPU |
| **DeBERTa-v3-base** | ~737 MB | 4 GB RAM / 2 GB VRAM | CUDA or CPU |

---

## 5. Quick-Start & Streamlit UI Launch

### Windows (PowerShell)
```powershell
# 1. Clone repository
git clone https://github.com/nishantwhat/RobustRAG.git
cd RobustRAG

# 2. Set up virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# 3. Install package with UI dependencies
pip install -e ".[ui]"

# 4. Launch Streamlit UI
streamlit run app/streamlit_app.py
```

### Linux / macOS (Bash)
```bash
# 1. Clone repository
git clone https://github.com/nishantwhat/RobustRAG.git
cd RobustRAG

# 2. Set up virtual environment
python3 -m venv .venv
source .venv/bin/activate

# 3. Install package with UI dependencies
pip install -e ".[ui]"

# 4. Launch Streamlit UI
streamlit run app/streamlit_app.py
```

---

## 6. Programmatic Python Inference

You can run detection from Python without launching the UI:

```python
from hallucination_detector.inference.model_registry import ModelRegistry

# Initialize registry (downloads lazily on demand)
registry = ModelRegistry()

# Run prediction with project-best model
result = registry.predict(
    model_key="longformer-head",
    question="What is the capital of Australia?",
    context="Canberra is the capital city of Australia.",
    answer="Sydney is the capital city of Australia.",
)

print(f"Model: {result.model_name}")
print(f"Prediction: {result.prediction}")
print(f"P(Hallucinated): {result.hallucination_probability:.2%}")
```

---

## 7. Troubleshooting

- **CUDA Out of Memory**: Longformer models require ~2 GB VRAM during single-item inference. Pass `device="cpu"` to force CPU computation if VRAM is constrained.
- **Hugging Face Connection Error**: Check network connectivity. Once cached, models run completely offline.
- **Input Formatting**: Ensure context and answer strings are non-empty before executing detection.
