"""RobustRAG Streamlit Evaluation Interface.

Provides zero-shot hallucination detection across all 5 benchmarked models.
"""

from __future__ import annotations

import streamlit as st
from hallucination_detector.inference.model_registry import ModelRegistry, MODEL_CATALOG


st.set_page_config(
    page_title="RobustRAG: Hallucination Detection",
    page_icon="🛡️",
    layout="wide",
)


@st.cache_resource(show_spinner="Loading model weights...")
def get_model_registry() -> ModelRegistry:
    return ModelRegistry()


def main() -> None:
    st.title("🛡️ RobustRAG: RAG Hallucination Detector")
    st.markdown(
        "Evaluate whether a generated answer is **supported** by retrieved evidence or contains **hallucinations** "
        "using fine-tuned transformers and classical baselines benchmarked on [RAGTruth](https://arxiv.org/abs/2401.00396)."
    )

    registry = get_model_registry()
    catalog = registry.get_catalog()

    # Sidebar: Model Selection
    st.sidebar.header("Model Configuration")
    model_keys = list(catalog.keys())
    model_labels = [f"{catalog[k].display_name}" for k in model_keys]

    selected_idx = st.sidebar.selectbox(
        "Choose Detection Model:",
        range(len(model_keys)),
        format_func=lambda i: model_labels[i],
        index=0,  # Default to Longformer-Head (Project Best)
    )
    selected_key = model_keys[selected_idx]
    meta = catalog[selected_key]

    # Sidebar details
    st.sidebar.markdown(f"**Hugging Face Repo:** [`{meta.repo_id}`](https://huggingface.co/{meta.repo_id})")
    st.sidebar.markdown(f"**Architecture:** `{meta.architecture}`")
    st.sidebar.markdown(f"**Input Modality:** `{meta.input_mode}`")
    st.sidebar.info(f"**Summary:** {meta.description}\n\n**Limitations:** {meta.limitations}")

    # Main Input Columns
    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("Query & Context")
        question = st.text_input(
            "User Question / Prompt:",
            value="Where was Marie Curie born?",
            help="The query submitted to the RAG system.",
        )
        context = st.text_area(
            "Retrieved Evidence Passage(s) / Context:",
            value="Marie Curie was born in Warsaw, in the Kingdom of Poland, on 7 November 1867. She was a physicist and chemist who conducted pioneering research on radioactivity.",
            height=200,
            help="Passage retrieved from the knowledge base or document store.",
        )

    with col2:
        st.subheader("Generated Response")
        answer = st.text_area(
            "Candidate Answer to Audit:",
            value="Marie Curie was born in Paris, France in 1867.",
            height=200,
            help="The LLM-generated response to test for hallucinations.",
        )

    st.markdown("---")
    run_btn = st.button("🔍 Run Hallucination Detection", type="primary", use_container_width=True)

    if run_btn:
        if not context.strip() or not answer.strip():
            st.error("Please provide both Context and Answer text before running detection.")
            return

        with st.spinner(f"Evaluating with {meta.display_name}..."):
            try:
                result = registry.predict(
                    model_key=selected_key,
                    question=question,
                    context=context,
                    answer=answer,
                )

                st.subheader("Diagnostic Results")
                r_col1, r_col2, r_col3 = st.columns(3)

                with r_col1:
                    if result.is_hallucinated:
                        st.error(f"### Verdict: {result.prediction} ⚠️")
                    else:
                        st.success(f"### Verdict: {result.prediction} ✅")

                with r_col2:
                    p_hal = result.hallucination_probability
                    if p_hal is not None:
                        st.metric(
                            label="P(Hallucinated)",
                            value=f"{p_hal:.2%}",
                            delta=f"{p_hal - 0.5:+.2%} vs 0.5 threshold",
                            delta_color="inverse",
                        )

                with r_col3:
                    p_sup = result.supported_probability
                    if p_sup is not None:
                        st.metric(
                            label="P(Supported)",
                            value=f"{p_sup:.2%}",
                        )

                # Visual progress bar of risk
                if p_hal is not None:
                    st.write("**Hallucination Confidence Meter:**")
                    st.progress(float(p_hal))

                # Technical info
                with st.expander("Technical Model & Preprocessing Details"):
                    st.json({
                        "model_key": selected_key,
                        "model_id": result.model_id,
                        "input_mode": meta.input_mode,
                        "architecture": meta.architecture,
                        "raw_score": result.raw_score,
                        "is_calibrated": result.is_calibrated,
                    })

            except Exception as e:
                st.error(f"Inference failed on model '{meta.display_name}': {str(e)}")


if __name__ == "__main__":
    main()
