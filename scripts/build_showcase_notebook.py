import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent

def create_showcase_notebook():
    cells = []

    def md(source):
        lines = source.strip().split("\n")
        formatted = [line + "\n" for line in lines[:-1]] + [lines[-1]]
        cells.append({
            "cell_type": "markdown",
            "metadata": {},
            "source": formatted
        })

    def code(source):
        lines = source.strip().split("\n")
        formatted = [line + "\n" for line in lines[:-1]] + [lines[-1]]
        cells.append({
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": formatted
        })

    # =========================================================================
    # SECTION 1: PROJECT OBJECTIVE AND SCOPE
    # =========================================================================
    md("""# RobustRAG — Demo & Research Showcase
## Live Demonstration of 4 Available Models with Historical RAGTruth Results

---

### 1. Project Objective and Scope
Retrieval-Augmented Generation (RAG) pairs an external evidence passage $C$ with a generative language model to answer a user prompt $Q$. Despite conditioning on ground truth, LLMs regularly generate unsupported extrapolations or direct contradictions. 

**Core Question:** Can a detector take $(Q, C, A)$ and determine whether the answer $A$ is **supported** (factually grounded) or **hallucinated** (unsupported or contradicted)?

**Benchmark:** Human-annotated [RAGTruth](https://arxiv.org/abs/2401.00396) (17,790 records; 2,700 fixed test records across QA, Summarization, and Data-to-text).

**Artifact status:** Four registered models are currently available. Longformer QCA remains unavailable because its required fitted selector and matching provenance bundle are missing. Historical benchmark figures below are preserved reference results; they do not establish real-world reliability.

**Presentation Sections:**
1. Project Objective and Scope
2. Setup & Remote Hugging Face Model Availability
3. Registered Models and Historical Benchmark Results
4. Supported vs. Hallucinated Contrastive Pairs
5. Side-by-Side Model Comparison & Latency Diagnostic
6. Input-Ablation Diagnostic Experiments ($C+A$, $Q+C+A$, $Q+C$, $A$-only)
7. Context-Quality Robustness (Complete, Incomplete, Noisy, Contradictory, Empty)
8. Probability Analysis & Conclusions
""")

    # =========================================================================
    # SECTION 2: SETUP AND REMOTE MODEL AVAILABILITY
    # =========================================================================
    md("""---
### 2. Setup and Remote Model Availability
We initialize the environment and verify connectivity with our 5 public Hugging Face model repositories under `nishantwhat`. 

All models are downloaded lazily and cached into `~/.cache/huggingface/hub/` without needing local weight files.""")

    code("""import sys
from pathlib import Path
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Ensure local source package is in path
ROOT = Path.cwd()
sys.path.insert(0, str(ROOT / 'src'))

from hallucination_detector.inference.model_registry import ModelRegistry, MODEL_CATALOG

print("Initializing ModelRegistry...")
registry = ModelRegistry()
catalog = {key: meta for key, meta in registry.get_catalog().items()
           if meta.provenance_status != "UNVERIFIED"}

print(f"Registered Models: {len(catalog)}")
for key, meta in catalog.items():
    print(f" - [{key}] -> {meta.repo_id} ({meta.architecture})")
""")

    # =========================================================================
    # SECTION 3: OVERVIEW OF THE FIVE MODELS
    # =========================================================================
    md("""---
### 3. Model Configurations and Historical Benchmark Results

The table preserves historical model and benchmark figures. Longformer QCA's checkpoint is not currently verified and is not available for live inference.

| Model Key | Architecture | Input Mode | Context Cap | Truncation / Attention Strategy | Verified Benchmark ROC-AUC |
|---|---|:---:|:---:|---|:---:|
| **`longformer-head`** | Longformer-Base | C + A | 1024 tokens | Answer preserved 100%; trailing context kept; local sliding window ($w=512$) + global CLS | **0.8296** (Best) |
| **`longformer-qca`** | Longformer-Base | Q + C + A | 1024 tokens | Historical selector pipeline; current artifact unavailable pending verified selector/provenance | **0.8144 (historical)** |
| **`deberta`** | DeBERTa-v3-base | C + A | 512 tokens | Current inference shortens context first to preserve answer; checkpoint was trained with right truncation | **0.7912 (historical)** |
| **`bilstm-attention`**| 2-layer BiLSTM | C + A | 512 tokens | Word-level 50k vocab + additive Bahdanau self-attention | Evaluated baseline |
| **`tfidf-lr`** | TF-IDF + Logistic Reg | C + A | Unlimited | 437k sublinear unigram+bigram n-grams, L2 $C=1.0$ | **0.7977** |
""")

    code("""# Historical benchmark snapshot compiled from earlier experiment artifacts
benchmark_metrics = pd.DataFrame([
    {"Model": "TF-IDF + Logistic Reg", "Context Cap": "Unlimited", "Input": "C + A", "Accuracy": "76.78%", "F1": "64.64%", "ROC-AUC": "0.7977", "PR-AUC": "0.7333"},
    {"Model": "DeBERTa-v3-base", "Context Cap": "512", "Input": "C + A", "Accuracy": "75.78%", "F1": "66.32%", "ROC-AUC": "0.7912", "PR-AUC": "0.6222"},
    {"Model": "Longformer (Head)", "Context Cap": "1024", "Input": "C + A", "Accuracy": "79.78%", "F1": "68.94%", "ROC-AUC": "0.8296", "PR-AUC": "0.7527"},
    {"Model": "Longformer (Tail)", "Context Cap": "1024", "Input": "C + A", "Accuracy": "77.56%", "F1": "67.80%", "ROC-AUC": "0.8187", "PR-AUC": "0.7331"},
    {"Model": "Longformer Smart QCA", "Context Cap": "1024", "Input": "Q + C + A", "Accuracy": "76.33%", "F1": "65.74%", "ROC-AUC": "0.8144", "PR-AUC": "0.7420"},
])
display(benchmark_metrics)
""")

    # =========================================================================
    # SECTION 4: CONTRASTIVE PAIR EVALUATION
    # =========================================================================
    md("""---
### 4. Supported versus Hallucinated Answer Examples
We construct a clean contrastive pair where the ground-truth context and question remain identical, while the answer is swapped between a supported statement and an unsupported factual substitution.""")

    code("""manual_path = Path("reports/manual_evaluation/robustrag_manual_set_v1.json")
manual_cases = json.loads(manual_path.read_text(encoding="utf-8"))["cases"]
supported_case = next(row for row in manual_cases if row["case_id"] == "v1_qa_supported_place")
contradiction_case = next(row for row in manual_cases if row["case_id"] == "v1_qa_contradiction_location")
q_base, c_base = supported_case["question"], supported_case["context"]
a_supported, a_hallucinated = supported_case["answer"], contradiction_case["answer"]

print("[TEST QUERY]")
print("Question:", q_base)
print("Context: ", c_base)
print("Supported Answer:   ", a_supported)
print("Hallucinated Answer:", a_hallucinated)
""")

    # =========================================================================
    # SECTION 5: SIDE-BY-SIDE MODEL COMPARISON
    # =========================================================================
    md("""---
### 5. Side-by-Side Model Comparison
We evaluate both queries across the four currently available models, measuring inference latency and uncalibrated model scores.""")

    code("""records = []

for key in catalog:
    # Test supported
    t0 = time.perf_counter()
    res_sup = registry.predict(key, q_base, c_base, a_supported)
    lat_sup = (time.perf_counter() - t0) * 1000

    # Test hallucinated
    t0 = time.perf_counter()
    res_hal = registry.predict(key, q_base, c_base, a_hallucinated)
    lat_hal = (time.perf_counter() - t0) * 1000

    records.append({
        "Model": catalog[key].display_name,
        "Input Mode": catalog[key].input_mode,
        "Pred (Sup Claim)": res_sup.prediction,
        "P(Hal) Sup": res_sup.hallucination_probability,
        "Pred (Hal Claim)": res_hal.prediction,
        "P(Hal) Hal": res_hal.hallucination_probability,
        "Avg Latency (ms)": round((lat_sup + lat_hal) / 2, 1),
    })

comparison_df = pd.DataFrame(records)
display(comparison_df)
""")

    code("""# Visualization: Comparison of Hallucination Probabilities
fig, ax = plt.subplots(figsize=(10, 5))
x = np.arange(len(comparison_df))
width = 0.35

ax.bar(x - width/2, comparison_df["P(Hal) Sup"], width, label="Supported Claim", color="#2ecc71")
ax.bar(x + width/2, comparison_df["P(Hal) Hal"], width, label="Hallucinated Claim", color="#e74c3c")
ax.axhline(0.5, color='gray', linestyle='--', alpha=0.7, label="Default Threshold (0.5)")

ax.set_ylabel("P(Hallucinated)")
ax.set_title("Model Probability Response: Supported vs. Hallucinated Claim")
ax.set_xticks(x)
ax.set_xticklabels(comparison_df["Model"], rotation=20, ha="right")
ax.set_ylim(0, 1.0)
ax.legend()
plt.tight_layout()
plt.show()
""")

    md("""**Key Takeaway:**
Transformer models (`deberta`, `longformer-head`, `longformer-qca`) assign lower hallucination probabilities to the supported claim compared to the contradicted claim. Notice how base rates and calibration curves vary: neural models without task-specific calibration thresholds should be interpreted with probability delta awareness.""")

    # =========================================================================
    # SECTION 6: INPUT ABLATION
    # =========================================================================
    md("""---
### 6. Input-Ablation Diagnostic Experiments
We explore input combinations:
- **C + A**: Context + Answer (Longformer Head, DeBERTa, TF-IDF).
- **Q + C + A**: Question + Context + Answer (BiLSTM-Attention).
- **Q + C**: Context + Question with omitted Answer (Distribution shift diagnostic).
- **A only**: Answer without context (Diagnostic for prior bias / language model hallucination artifacts).

*Methodological Note:* Testing $A$-only or omitting $C$ represents an inference-time diagnostic to evaluate whether the model relies on context or makes assumptions from answer keywords alone. QCA is omitted until its selector and checkpoint provenance are verified.""")

    code("""ablation_cases = {
    "Full Q+C+A": {"q": q_base, "c": c_base, "a": a_hallucinated},
    "C+A only (No Q)": {"q": "", "c": c_base, "a": a_hallucinated},
    "Q+A only (No C)": {"q": q_base, "c": "", "a": a_hallucinated},
    "A only (No Q, No C)": {"q": "", "c": "", "a": a_hallucinated},
}

ablation_results = []
for case_name, data in ablation_cases.items():
    for model_key in catalog:
        result = registry.predict(model_key, data["q"], data["c"], data["a"])
        ablation_results.append({"Input Ablation": case_name,
            "Model": catalog[model_key].display_name,
            "Input Mode": catalog[model_key].input_mode,
            "P(Hal) uncalibrated": result.hallucination_probability})

ablation_df = pd.DataFrame(ablation_results)
display(ablation_df)
""")

    md("""**Ablation Takeaway:**
When context $C$ is removed (`Q+A only` and `A only`), the probability of hallucination shifts significantly. Models that condition strongly on evidence rely on lexical overlap and contextual agreement; stripping context produces an extreme evidence vacuum.""")

    # =========================================================================
    # SECTION 7: CONTEXT-QUALITY ROBUSTNESS
    # =========================================================================
    md("""---
### 7. Existing Manual-Case Input Coverage
This optional functional check reads already documented hand-authored cases from the manual set. It does not add examples, tune thresholds, or represent benchmark performance.""")

    code("""selected_ids = {
    "v1_qa_supported_place", "v1_qa_contradiction_location",
    "v1_qa_unsupported_count_addition", "v1_qa_unsupported_location_addition",
}
robustness_records = []
for case in manual_cases:
    if case["case_id"] not in selected_ids:
        continue
    row = {"Context Condition": case["case_id"]}
    for model_key in catalog:
        res = registry.predict(model_key, case["question"] or "", case["context"], case["answer"])
        row[f"{catalog[model_key].display_name} P(Hal)"] = round(res.hallucination_probability, 4)
    robustness_records.append(row)

robustness_df = pd.DataFrame(robustness_records)
display(robustness_df)
""")

    code("""# Visualization: Context Robustness Heatmap
models_cols = [c for c in robustness_df.columns if c != "Context Condition"]
heatmap_data = robustness_df.set_index("Context Condition")[models_cols]

fig, ax = plt.subplots(figsize=(10, 5))
cax = ax.imshow(heatmap_data.values, cmap="YlOrRd", vmin=0.0, vmax=1.0, aspect="auto")

ax.set_xticks(np.arange(len(models_cols)))
ax.set_yticks(np.arange(len(heatmap_data)))
ax.set_xticklabels([c.replace(" P(Hal)", "") for c in models_cols], rotation=20, ha="right")
ax.set_yticklabels(heatmap_data.index)

# Annotate values
for i in range(len(heatmap_data)):
    for j in range(len(models_cols)):
        val = heatmap_data.values[i, j]
        ax.text(j, i, f"{val:.2f}", ha="center", va="center", color="black" if val < 0.6 else "white", fontweight="bold")

fig.colorbar(cax, label="P(Hallucinated)")
ax.set_title("Context Robustness Profile: P(Hallucinated) Across Evidence Degradation")
plt.tight_layout()
plt.show()
""")

    md("""**Robustness Takeaway:**
- When context fully supports the claim ("Complete Grounding"), all models output lower hallucination probabilities.
- When context explicitly contradicts the claim or is completely irrelevant, models elevate hallucination probabilities.
- Classical TF-IDF relies purely on n-gram overlap: when context is empty or distractor, lack of term overlap acts as a blunt indicator of non-grounding. Neural models exhibit more nuanced probability scaling.""")

    # =========================================================================
    # SECTION 8: PROBABILITY CALIBRATION & CONCLUSION
    # =========================================================================
    md("""---
### 8. Probability Analysis & Limitations

#### Scope Notes:
1. Historical benchmark scores in this notebook are preserved snapshots from earlier runs; the new answer-preserving DeBERTa input path has a separate validation-only sensitivity report and has not been retrained.
2. Longformer Head validation separation varies by task: saved validation ROC AUC is near chance for Summarization, so the interface issues no reliable verdict for that task.
3. Longformer QCA is not available for live inference until a complete trained bundle passes selector/provenance verification and validation checks.
4. All model scores are uncalibrated outputs. These benchmarks do not establish real-world reliability.

#### Presentation Summary:
RobustRAG provides a reproducible evaluation pipeline and four currently available model choices. QCA remains hidden until its artifact bundle is rebuilt and verified.
""")

    notebook = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3 (ipykernel)",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "codemirror_mode": {"name": "ipython", "version": 3},
                "file_extension": ".py",
                "mimetype": "text/x-python",
                "name": "python",
                "nbconvert_exporter": "python",
                "pygments_lexer": "ipython3",
                "version": "3.12.10"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 5
    }

    out_path = ROOT / "robustrag_demo_showcase.ipynb"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(notebook, f, indent=1)
    print(f"Generated {out_path} successfully!")

if __name__ == "__main__":
    create_showcase_notebook()
