"""RobustRAG Streamlit Evaluation Interface.

Hallucination detection across the benchmarked models, with task-aware verdicts.
Raw scores from these models are uncalibrated, so verdicts use thresholds that
were measured per task (configs/thresholds.json) instead of a blanket 0.5.
"""

from __future__ import annotations

from pathlib import Path

import streamlit as st

from hallucination_detector.inference.model_registry import MODEL_CATALOG, ModelRegistry
from hallucination_detector.inference.verdict import (
    NOT_SURE,
    TASKS,
    classify,
    load_thresholds,
)

THRESHOLDS_PATH = Path(__file__).resolve().parents[1] / "configs" / "thresholds.json"

st.set_page_config(
    page_title="RobustRAG: Hallucination Detection",
    page_icon="🛡️",
    layout="wide",
)


@st.cache_resource(show_spinner="Preparing model registry...")
def get_model_registry() -> ModelRegistry:
    return ModelRegistry()


@st.cache_data
def get_thresholds() -> dict:
    return load_thresholds(THRESHOLDS_PATH)


def main() -> None:
    st.title("🛡️ RobustRAG: RAG Hallucination Detector")
    st.markdown(
        "Evaluate whether a generated answer is **supported** by retrieved evidence or contains "
        "**hallucinations** using fine-tuned transformers and classical baselines benchmarked on "
        "[RAGTruth](https://arxiv.org/abs/2401.00396)."
    )

    registry = get_model_registry()
    catalog = registry.get_catalog()
    thresholds = get_thresholds()

    # Sidebar: Model + task selection
    st.sidebar.header("Model Configuration")
    model_keys = list(catalog.keys())
    model_labels = [catalog[k].display_name for k in model_keys]

    selected_idx = st.sidebar.selectbox(
        "Choose Detection Model:",
        range(len(model_keys)),
        format_func=lambda i: model_labels[i],
        index=0,
    )
    selected_key = model_keys[selected_idx]
    meta = catalog[selected_key]

    task = st.sidebar.selectbox(
        "What kind of answer are you checking?",
        list(TASKS) + [NOT_SURE],
        help=(
            "The detector's scores mean different things per task, so verdict thresholds are "
            "task-specific. QA = answering a question from passages; Summarization = summarizing "
            "a document; Data-to-text = writing text from structured (JSON-like) data."
        ),
    )

    st.sidebar.markdown(f"**Hugging Face Repo:** [`{meta.repo_id}`](https://huggingface.co/{meta.repo_id})")
    st.sidebar.markdown(f"**Architecture:** `{meta.architecture}`")
    st.sidebar.markdown(f"**Input Modality:** `{meta.input_mode}`")
    st.sidebar.info(f"**Summary:** {meta.description}\n\n**Limitations:** {meta.limitations}")

    # Main input
    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("Query & Context")
        question = st.text_area(
            "User Question / Prompt:",
            value="Where was Marie Curie born?",
            height=120,
            help=(
                "The query submitted to the RAG system. Only the Q+C+A model uses it. In the "
                "training data this field held the full prompt (instruction + passages + question)."
            ),
        )
        context = st.text_area(
            "Retrieved Evidence Passage(s) / Context:",
            value=(
                "Marie Curie was born in Warsaw, in the Kingdom of Poland, on 7 November 1867. "
                "She was a physicist and chemist who conducted pioneering research on radioactivity."
            ),
            height=200,
            help="Passage retrieved from the knowledge base or document store.",
        )

    with col2:
        st.subheader("Generated Response")
        answer = st.text_area(
            "Candidate Answer to Audit:",
            value="Marie Curie was born in Paris, France in 1867.",
            height=330,
            help="The LLM-generated response to test for hallucinations.",
        )

    st.markdown("---")
    run_btn = st.button("🔍 Run Hallucination Detection", type="primary", use_container_width=True)

    if not run_btn:
        return

    if not context.strip() or not answer.strip():
        st.error("Please provide both Context and Answer text before running detection.")
        return

    with st.spinner(f"Evaluating with {meta.display_name} (first run downloads/loads weights)..."):
        try:
            result = registry.predict(
                model_key=selected_key,
                question=question,
                context=context,
                answer=answer,
            )
        except Exception as e:  # noqa: BLE001
            st.error(f"Inference failed on model '{meta.display_name}': {e}")
            return

    p_hal = result.hallucination_probability
    p_sup = result.supported_probability
    verdict = classify(p_hal, selected_key, task, thresholds)

    st.subheader("Diagnostic Results")
    r_col1, r_col2, r_col3 = st.columns(3)

    with r_col1:
        banner = f"### {verdict.label}"
        if verdict.level == "ok":
            st.success(banner)
        elif verdict.level == "warn":
            st.warning(banner)
        elif verdict.level == "bad":
            st.error(banner)
        else:
            st.info(banner)

    with r_col2:
        st.metric(label="Model score P(hallucinated)", value=f"{p_hal:.2%}")
    with r_col3:
        st.metric(label="P(supported)", value=f"{p_sup:.2%}")

    if verdict.explanation:
        st.caption(verdict.explanation)

    trunc = result.details.get("truncation")
    if trunc and trunc.get("truncated"):
        st.warning(
            f"Input was {trunc['total_tokens']} tokens but this model reads at most {trunc['max_length']}. "
            f"Mode: {trunc['mode']}; dropped {trunc['context_tokens_dropped']} context tokens and "
            f"{trunc['answer_tokens_dropped']} answer tokens. Long inputs make this model less reliable."
        )
    if result.details.get("selector_fitted") is False:
        st.warning(
            "configs/qca_selector_idf.json is missing, so context sentences were selected with an "
            "unfitted selector that differs from training. Run scripts/build_selector_idf.py."
        )

    st.write("**Model score (uncalibrated, not a probability of being correct):**")
    st.progress(float(p_hal))

    if selected_key != "longformer-head" and task != NOT_SURE:
        st.caption(
            "Measured thresholds currently exist only for the Longformer (Head) model, so this "
            "verdict uses a plain 0.5 cutoff, which is known to miss many hallucinations."
        )
    if task == NOT_SURE:
        st.caption("Pick a task type in the sidebar to get task-calibrated verdicts.")

    with st.expander("Technical Model & Preprocessing Details"):
        st.json(
            {
                "model_key": selected_key,
                "model_id": result.model_id,
                "task": task,
                "input_mode": meta.input_mode,
                "architecture": meta.architecture,
                "raw_score": result.raw_score,
                "thresholds_used": thresholds.get(selected_key, {}).get(task),
                "thresholds_source": thresholds.get("_meta", {}).get("source"),
                "details": result.details,
            }
        )


if __name__ == "__main__":
    main()
