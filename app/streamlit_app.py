"""RobustRAG: evidence-grounded hallucination detection demo."""

from __future__ import annotations

import csv
import io
import json
from datetime import datetime, timezone
from pathlib import Path

import streamlit as st
from hallucination_detector.inference.model_registry import ModelRegistry
from hallucination_detector.inference.task_verdicts import DEFAULT_CONFIG, interpret_task_score, load_task_verdict_config


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MANUAL_CASES_PATH = PROJECT_ROOT / "reports" / "manual_evaluation" / "robustrag_manual_set_v1.json"
SUPPLIED_CASES_PATH = PROJECT_ROOT / "reports" / "manual_evaluation" / "ragtruth_supplied_cases_v1.json"
QUICK_CASES_PATH = PROJECT_ROOT / "reports" / "manual_evaluation" / "robustrag_quick_cases_v1.json"
TASK_EVIDENCE_PATH = PROJECT_ROOT / "reports" / "validation" / "task_verdict_evidence_20261025" / "task_verdict_evidence.json"
EVALUATION_CSV_FIELDS = [
    "case_id", "source", "task_type", "model", "expected_label", "predicted_label",
    "hallucination_probability_uncalibrated", "supported_probability_uncalibrated",
    "pass_fail", "notes", "recorded_at",
]
TASK_FILTERS = {"QA": "QA", "Summarization": "Summary", "Data-to-text": "Data2txt", "Not sure": None}
QUICK_TEST_CHOICES = [
    ("Supported answer", "v1_qa_supported_place"),
    ("Direct contradiction", "v1_qa_contradiction_location"),
    ("Unsupported factual addition", "v1_qa_unsupported_person_addition"),
    ("Subtle hallucination · cycling passage", "ragtruth_16467"),
    ("Summary · no-reliable-verdict warning", "ragtruth_summary_warning_v1"),
]

st.set_page_config(page_title="RobustRAG: RAG Hallucination Detector", page_icon="🛡️", layout="wide")


@st.cache_resource(show_spinner="Loading model weights...")
def get_model_registry() -> ModelRegistry:
    """Keep the existing lazy, shared model registry behavior."""
    return ModelRegistry()


def _validate_case_file(path: Path, source_name: str, allowed_tasks: set[str]) -> dict[str, dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload.get("cases") if isinstance(payload, dict) else None
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"{source_name} JSON must contain a nonempty cases array.")
    cases: dict[str, dict] = {}
    required = {"case_id", "task_type", "context", "question", "answer", "expected_label", "failure_category", "rationale"}
    for number, item in enumerate(rows, start=1):
        if not isinstance(item, dict) or not required.issubset(item):
            raise ValueError(f"{source_name} case {number} is missing required fields: {sorted(required - set(item or {}))}.")
        cid = item["case_id"]
        if not isinstance(cid, str) or not cid.strip() or cid in cases:
            raise ValueError(f"{source_name} case {number} has an empty or duplicate case_id: {cid!r}.")
        if item["task_type"] not in allowed_tasks:
            raise ValueError(f"{source_name} case {cid} has unsupported task_type {item['task_type']!r}.")
        expected = item["expected_label"]
        valid = expected in {"SUPPORTED", "HALLUCINATED"}
        valid |= expected is None and item.get("label_status") == "ambiguous"
        valid |= expected == "NO RELIABLE VERDICT" and item.get("label_status") == "no_reliable_verdict"
        if not valid:
            raise ValueError(f"{source_name} case {cid} has an invalid expected_label/label_status pair.")
        if not all(isinstance(item[field], str) and item[field].strip() for field in ("failure_category", "rationale")):
            raise ValueError(f"{source_name} case {cid} needs nonempty failure_category and rationale.")
        if item["question"] is not None and not isinstance(item["question"], str):
            raise ValueError(f"{source_name} case {cid}: question must be a string or null.")
        if expected == "NO RELIABLE VERDICT":
            if item["context"] or item["answer"]:
                raise ValueError(f"{source_name} case {cid}: warning-only case must not provide prediction inputs.")
        elif not all(isinstance(item[field], str) and item[field].strip() for field in ("context", "answer")):
            raise ValueError(f"{source_name} case {cid}: context and answer must be nonempty strings.")
        copied = dict(item)
        copied["source"] = source_name
        cases[cid] = copied
    return cases


def load_manual_cases(path: Path = MANUAL_CASES_PATH) -> dict[str, dict]:
    return _validate_case_file(path, "Hand-authored v1", {"QA", "Summary"})


def load_supplied_cases(path: Path = SUPPLIED_CASES_PATH) -> dict[str, dict]:
    return _validate_case_file(path, "User-supplied records", {"QA", "Data2txt", "Summary"})


def load_all_cases() -> dict[str, dict]:
    manual = load_manual_cases()
    supplied = load_supplied_cases()
    overlap = manual.keys() & supplied.keys()
    if overlap:
        raise ValueError(f"Case IDs must be unique across evaluation sources: {sorted(overlap)}")
    return manual | supplied


def _inputs_for(case: dict) -> dict[str, str]:
    return {"question": case.get("question") or "", "context": case.get("context") or "", "answer": case.get("answer") or ""}


def _load_case_callback(case_id: str, case: dict) -> None:
    """Streamlit callback runs before widgets are rebuilt, so assigning their state is safe."""
    values = _inputs_for(case)
    for field, value in values.items():
        st.session_state[f"{field}_input"] = value
    st.session_state["loaded_evaluation_case_id"] = case_id
    st.session_state["loaded_evaluation_inputs"] = values
    st.session_state["last_lab_entry"] = None


def _record(case: dict, model_key: str, predicted: str, probability=None, support=None, notes="") -> dict:
    expected = case["expected_label"]
    binary_prediction = {"LIKELY SUPPORTED": "SUPPORTED", "LIKELY HALLUCINATED": "HALLUCINATED"}.get(predicted, predicted)
    if expected == "NO RELIABLE VERDICT":
        status = "NO RELIABLE VERDICT"
    elif expected is None:
        status = "REVIEW/AMBIGUOUS"
    elif binary_prediction not in {"SUPPORTED", "HALLUCINATED"}:
        status = "REVIEW"
    else:
        status = "PASS" if binary_prediction == expected else "FAIL"
    return {
        "case_id": case["case_id"], "source": case["source"], "task_type": case["task_type"],
        "model": model_key, "expected_label": expected or "", "predicted_label": predicted,
        "hallucination_probability_uncalibrated": "" if probability is None else f"{probability:.8f}",
        "supported_probability_uncalibrated": "" if support is None else f"{support:.8f}",
        "pass_fail": status, "notes": notes, "recorded_at": datetime.now(timezone.utc).isoformat(),
    }


def evaluation_csv(history: list[dict]) -> str:
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=EVALUATION_CSV_FIELDS)
    writer.writeheader()
    writer.writerows({field: row.get(field, "") for field in EVALUATION_CSV_FIELDS} for row in history)
    return output.getvalue()


def inject_styles() -> None:
    st.markdown("""
    <style>
    :root { --rr-bg:#0e1117; --rr-panel:#171b24; --rr-panel2:#20242f; --rr-line:#303643; --rr-text:#f3f5fa; --rr-muted:#c0c7d4; --rr-blue:#3b9cf5; --rr-red:#fa4c50; }
    [data-testid="stAppViewContainer"], [data-testid="stMain"] { background:var(--rr-bg); color:var(--rr-text); }
    [data-testid="stHeader"] { background:rgba(14,17,23,.94); }
    [data-testid="stSidebar"] { background:#1b1e27; border-right:1px solid var(--rr-line); }
    [data-testid="stMainBlockContainer"] { max-width:1500px; padding-top:2rem; padding-bottom:2.5rem; }
    h1,h2,h3,h4,p,label,small { color:var(--rr-text); }
    h1 { font-size:clamp(1.8rem,3vw,2.5rem)!important; font-weight:760!important; letter-spacing:-.025em; }
    h2,h3 { font-weight:700!important; letter-spacing:-.015em; }
    [data-testid="stCaptionContainer"] p { color:var(--rr-muted); }
    [data-testid="stTextInput"] input,[data-testid="stTextArea"] textarea,[data-testid="stSelectbox"] div[data-baseweb="select"]>div { background:#20242f; color:#f3f5fa; border-color:#3b4250; border-radius:9px; }
    [data-testid="stTextArea"] textarea { line-height:1.5; }
    [data-testid="stTextInput"] label,[data-testid="stTextArea"] label,[data-testid="stSelectbox"] label { font-weight:650; color:#eef1f7; }
    [data-testid="stButton"] button,[data-testid="stFormSubmitButton"] button { border-radius:9px; min-height:2.55rem; font-weight:650; }
    [data-testid="stFormSubmitButton"] button[kind="primary"] { background:var(--rr-red); color:#fff; border:1px solid #ff686b; }
    [data-testid="stButton"] button { background:#252b37; color:#f2f4f9; border:1px solid #3b4250; }
    [data-testid="stExpander"] { background:#151922; border:1px solid var(--rr-line); border-radius:9px; }
    .rr-lede { color:#d3d9e3; font-size:.96rem; line-height:1.5; }
    .rr-panel { background:var(--rr-panel); border:1px solid var(--rr-line); border-radius:12px; padding:1rem 1.15rem; }
    .rr-result { background:#222833; border:1px solid #3d4654; border-radius:12px; padding:1.15rem 1.25rem; margin-top:1rem; }
    .rr-result.supported { background:#17352c; border-color:#286049; }
    .rr-result.hallucinated { background:#3b2428; border-color:#764149; }
    .rr-result.uncertain { background:#292b35; border-color:#596072; }
    .rr-verdict { font-size:1.3rem; font-weight:780; color:#fff; }
    .rr-result.supported .rr-verdict { color:#8ce0b3; } .rr-result.hallucinated .rr-verdict { color:#ff888b; }
    .rr-score-grid { display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); gap:1rem; margin-top:1rem; }
    .rr-score { background:rgba(8,11,16,.35); border:1px solid #424b59; border-radius:9px; padding:.85rem; }
    .rr-score-heading { display:flex; justify-content:space-between; gap:.7rem; color:#f4f6fa; font-weight:650; }
    .rr-score-value { font-size:1.1rem; font-weight:750; }
    .rr-track { height:8px; background:#353b47; border-radius:9px; overflow:hidden; margin-top:.65rem; }
    .rr-fill { height:100%; background:var(--rr-blue); border-radius:9px; }
    .rr-score-note { color:#c4cad4; font-size:.82rem; margin:.8rem 0 0; }
    @media(max-width:760px){[data-testid="stMainBlockContainer"]{padding:1rem .8rem 2rem}.rr-score-grid{grid-template-columns:1fr}}
    </style>
    """, unsafe_allow_html=True)


def selectable_model_keys(catalog: dict, loaded_model_keys: set[str]) -> list[str]:
    """Keep optional/unverified entries out until their weights actually load."""
    return [key for key, meta in catalog.items()
            if meta.provenance_status != "UNVERIFIED" or key in loaded_model_keys]


def render_sidebar(registry: ModelRegistry):
    catalog = registry.get_catalog()
    loaded_models = set(registry._loaded_models)
    optional_key = "bilstm-optimized"
    qca_key = "longformer-qca"
    available = selectable_model_keys(catalog, loaded_models)
    with st.sidebar:
        st.subheader("Model Configuration")
        if qca_key in catalog and qca_key not in loaded_models:
            st.caption("Optional detector: Longformer Q+C+A · pinned Hugging Face checkpoint")
            if st.button("Load Longformer QCA", key="load_longformer_qca"):
                try:
                    with st.spinner("Loading Longformer QCA and its fitted selector..."):
                        registry.load_model(qca_key)
                    st.session_state["longformer_qca_loaded_notice"] = True
                    st.session_state.pop("longformer_qca_load_error", None)
                    st.rerun()
                except Exception as exc:
                    st.session_state["longformer_qca_load_error"] = f"{type(exc).__name__}: {exc}"
            error = st.session_state.get("longformer_qca_load_error")
            if error:
                st.warning(f"Longformer QCA was not added. Check the pinned Hub cache/network and configs/qca_selector_idf.json. Details: {error}")
            else:
                st.info("Loads the selector from configs/qca_selector_idf.json. The selector declares training-split metadata, but the published checkpoint has no manifest linking it to these weights; treat QCA scores as experimental and uncalibrated.")
        if st.session_state.pop("longformer_qca_loaded_notice", False):
            st.warning("Longformer QCA loaded. It uses the external fitted selector, but its exact pairing with the checkpoint is unverified; scores remain uncalibrated.")
        if optional_key in catalog and optional_key not in loaded_models:
            st.caption("Optional detector: optimized BiLSTM · separate Hugging Face checkpoint · Q+C+A")
            if st.button("Load Optimized BiLSTM", key="load_optimized_bilstm"):
                try:
                    with st.spinner("Loading the pinned optimized BiLSTM bundle..."):
                        registry.load_model(optional_key, device="cpu")
                    st.session_state["optimized_bilstm_loaded_notice"] = True
                    st.session_state.pop("optimized_bilstm_load_error", None)
                    st.rerun()
                except Exception as exc:
                    st.session_state["optimized_bilstm_load_error"] = f"{type(exc).__name__}: {exc}"
            error = st.session_state.get("optimized_bilstm_load_error")
            if error:
                st.warning(f"Optimized BiLSTM was not added. Check network access or the pinned Hub cache, then retry. Details: {error}")
        if st.session_state.pop("optimized_bilstm_loaded_notice", False):
            st.success("Optimized BiLSTM loaded and is now available in the model selector. Its provenance is partial and its scores are uncalibrated.")
        selected = st.selectbox("Choose Detection Model", options=available, key="model_selector", format_func=lambda key: catalog[key].display_name)
        task = st.selectbox("Answer task", options=list(TASK_FILTERS), index=3, key="task_filter", help="Used for Longformer Head's validation-informed verdict display. Other models keep the uncalibrated 0.50 cutoff.")
        meta = catalog[selected]
        st.markdown(f"**Repository / checkpoint**  \n[{meta.repo_id}](https://huggingface.co/{meta.repo_id})")
        with st.expander("Architecture, input mode & limitations", expanded=False):
            st.markdown(f"**Architecture:** {meta.architecture}")
            st.markdown(f"**Input modality:** `{meta.input_mode}`")
            st.markdown(f"**Limitations:** {meta.limitations}")
            st.caption(meta.description)
    return selected, meta, task


def render_result(result, model_key: str, meta, task: str | None = None) -> None:
    h = result.hallucination_probability
    s = result.supported_probability
    htext = "Unavailable" if h is None else f"{h:.2%}"
    stext = "Unavailable" if s is None else f"{s:.2%}"
    hw = 100 * max(0.0, min(1.0, float(h or 0.0)))
    sw = 100 * max(0.0, min(1.0, float(s or 0.0)))
    task = task or st.session_state.get("active_task", "Not sure")
    verdict = interpret_task_score(model_key, task, float(h or 0.0), load_task_verdict_config())
    cls = verdict.category if verdict.category in {"supported", "hallucinated"} else "uncertain"
    st.markdown(
        f'<section class="rr-result {cls}"><div class="rr-verdict">{verdict.label}</div>'
        f'<div class="rr-score-grid"><div class="rr-score"><div class="rr-score-heading"><span>Hallucination score · uncalibrated</span><span class="rr-score-value">{htext}</span></div><div class="rr-track"><div class="rr-fill" style="width:{hw:.2f}%"></div></div></div>'
        f'<div class="rr-score"><div class="rr-score-heading"><span>Supported score · uncalibrated</span><span class="rr-score-value">{stext}</span></div><div class="rr-track"><div class="rr-fill" style="width:{sw:.2f}%"></div></div></div></div>'
        f'<p class="rr-score-note">{verdict.threshold_description}. Scores are uncalibrated model outputs, not verified confidence or factual confirmation.</p></section>', unsafe_allow_html=True)
    if model_key == "longformer-head":
        _render_task_evidence(task)
    if (model_key == "longformer-qca"
            and result.details.get("selector_provenance_status") == "UNVERIFIED_EXTERNAL_SELECTOR"):
        st.warning(
            "Longformer QCA used the external selector in configs/qca_selector_idf.json. "
            "Its training-split metadata is present, but no manifest links this selector to the "
            f"checkpoint revision {result.details.get('checkpoint_revision')}; treat this output as experimental."
        )
    truncation = result.details.get("truncation", {})
    if truncation.get("context_truncated"):
        st.warning("DeBERTa shortened the context to retain the answer within its 512-token input limit.")
    if truncation.get("answer_truncated"):
        st.warning("The answer itself exceeded the 512-token input budget and was shortened; interpret this score cautiously.")
    with st.expander("Technical Model & Preprocessing Details", expanded=False):
        st.json({"model_key": model_key, "model_id": result.model_id, "input_mode": meta.input_mode,
                 "architecture": meta.architecture, "raw_score": result.raw_score,
                 "is_calibrated": result.is_calibrated, "details": result.details})


def _render_task_evidence(task: str) -> None:
    if not TASK_EVIDENCE_PATH.exists():
        st.warning("Validation evidence report is unavailable; task bands are not independently documented here.")
        return
    evidence = json.loads(TASK_EVIDENCE_PATH.read_text(encoding="utf-8"))
    with st.expander("Validation evidence & limitations", expanded=False):
        st.caption("Saved RAGTruth validation predictions only (1,512 records). These uncalibrated operating bands are not independent benchmark evidence or real-world reliability guarantees.")
        if task == "Summarization":
            auc = evidence["summary_task"]["auc"]
            st.write(f"Longformer Head Summary ROC AUC: {auc['auc']:.3f} (stratified bootstrap 95% interval {auc['bootstrap_95pct_ci'][0]:.3f}–{auc['bootstrap_95pct_ci'][1]:.3f}; n={evidence['summary_task']['n']}). Near-chance separation; no reliable verdict is issued.")
        elif task in {"QA", "Data-to-text"}:
            task_key = "QA" if task == "QA" else "Data-to-text"
            group = evidence["task_bands"][task_key]
            interval = group["auc"]["bootstrap_95pct_ci"]
            st.write(f"{task} validation ROC AUC {group['auc']['auc']:.3f} (95% stratified bootstrap interval {interval[0]:.3f}–{interval[1]:.3f}), n={group['n']}. The table shows outcomes within these predeclared triage bands; scores are not calibrated confidence.")
            band_rows = []
            for key, label in (("likely_supported", "Likely supported"), ("uncertain", "Uncertain"), ("likely_hallucinated", "Likely hallucinated")):
                band = group["bands"]["bands"][key]
                band_rows.append({"Display band": label, "Score range": key.replace("likely_supported", f"< {group['bands']['low_threshold']:.2f}").replace("uncertain", f"{group['bands']['low_threshold']:.2f}–<{group['bands']['high_threshold']:.2f}").replace("likely_hallucinated", f"≥ {group['bands']['high_threshold']:.2f}"),
                    "n": band["n"], "SUPPORTED": band["true_supported"], "HALLUCINATED": band["true_hallucinated"],
                    "Hallucinated share": "—" if band["hallucinated_rate"] is None else f"{band['hallucinated_rate']:.1%}"})
            st.dataframe(band_rows, hide_index=True, use_container_width=True)
        else:
            st.write("Not sure uses the existing 0.50 cutoff. Per-task validation results are in the technical report; this fallback is explicitly uncalibrated.")
        st.caption("Evidence: reports/validation/task_verdict_evidence_20261025/task_verdict_evidence.md. Manual examples and held-out test data were not used to set these rules.")


def render_quick_test_cases(cases: dict[str, dict]) -> None:
    """Offer quick case loading without triggering inference or exposing labels to the model."""
    labels = [label for label, case_id in QUICK_TEST_CHOICES if case_id in cases]
    if not labels:
        return
    case_by_label = {label: case_id for label, case_id in QUICK_TEST_CHOICES if case_id in cases}
    left, right = st.columns([4, 1], vertical_alignment="bottom")
    with left:
        selected_label = st.selectbox("Quick Test Cases", labels, key="quick_test_case_select")
    with right:
        case_id = case_by_label[selected_label]
        st.button("Load into inputs", key="load_quick_test_case", use_container_width=True,
                  on_click=_load_case_callback, args=(case_id, cases[case_id]))
    st.caption("Loading fills the three input fields only; it never runs inference. Use the Evaluation Lab to view expected labels and record actual predictions.")


def render_evaluation_lab(cases: dict[str, dict], registry: ModelRegistry, model_key: str, meta, task_filter: str) -> None:
    history = st.session_state.setdefault("evaluation_history", [])
    with st.expander("Evaluation Lab · supplied records and hand-authored cases", expanded=False):
        st.caption("Manual observations only; this is not a benchmark. User-supplied records retain their source labels; membership was not independently checked against the local archive.")
        selected_task = TASK_FILTERS[task_filter]
        filtered = {cid: case for cid, case in cases.items() if selected_task is None or case["task_type"] == selected_task}
        if not filtered:
            st.info("There are no cases for this task filter.")
            return
        case_id = st.selectbox("Case · ID · task · short description", list(filtered),
            format_func=lambda cid: f"{cid} · {filtered[cid]['task_type']} · {filtered[cid]['failure_category']}", key="evaluation_case_select")
        case = filtered[case_id]
        st.caption(f"Source: {case['source']} · expected label is shown separately and is never sent to the model.")
        if case.get("source_record_id"):
            st.caption(f"Supplied record ID: {case['source_record_id']} · source model: {case.get('source_model', 'not stated')}")
        if case.get("source_warning"):
            st.warning(case["source_warning"])
        if case["expected_label"] == "NO RELIABLE VERDICT":
            st.warning("NO RELIABLE VERDICT — documented summary limitation. No model prediction is made for this warning-only case.")
        st.button("Load case into main inputs", key="load_evaluation_case", use_container_width=True,
                  on_click=_load_case_callback, args=(case_id, case))
        if st.session_state.get("loaded_evaluation_case_id") == case_id:
            st.success("Case loaded. Loading does not run inference.")
        with st.expander("Source annotation and rationale", expanded=False):
            st.markdown(f"**Expected label:** {case['expected_label'] or 'REVIEW / AMBIGUOUS'}")
            st.write(case.get("source_annotation", case["rationale"]))
            if case.get("provenance"):
                st.caption(case["provenance"])
        if st.button("Run selected model on this case", key="run_evaluation_case", disabled=case["expected_label"] == "NO RELIABLE VERDICT", use_container_width=True):
            try:
                values = _inputs_for(case)
                result = registry.predict(model_key=model_key, **values)
                case_task = {"QA": "QA", "Data2txt": "Data-to-text", "Summary": "Summarization"}[case["task_type"]]
                verdict = interpret_task_score(model_key, case_task, float(result.hallucination_probability), load_task_verdict_config())
                row = _record(case, model_key, verdict.label, result.hallucination_probability, result.supported_probability)
                history.append(row)
                st.session_state["evaluation_history"] = history
                st.session_state["last_lab_entry"] = {"case_id": case_id, "model_key": model_key, "result": result, "row": row}
            except Exception as exc:
                st.error(f"Could not run {meta.display_name}. {type(exc).__name__}: {exc}")
        entry = st.session_state.get("last_lab_entry")
        if entry and entry["case_id"] == case_id and entry["model_key"] == model_key:
            row = entry["row"]
            if row["pass_fail"] == "PASS": st.success(f"PASS · expected {row['expected_label']} · predicted {row['predicted_label']}")
            elif row["pass_fail"] == "FAIL": st.error(f"FAIL · expected {row['expected_label']} · predicted {row['predicted_label']}")
            elif row["pass_fail"] == "REVIEW": st.warning(f"REVIEW · model returned {row['predicted_label']}; no binary diagnostic comparison was counted.")
            elif row["pass_fail"] == "REVIEW/AMBIGUOUS": st.warning(f"REVIEW / AMBIGUOUS · predicted {row['predicted_label']}")
            else: st.warning(f"{row['pass_fail']} · predicted {row['predicted_label']}")
            case_task = {"QA": "QA", "Data2txt": "Data-to-text", "Summary": "Summarization"}[case["task_type"]]
            render_result(entry["result"], model_key, meta, case_task)
        if history:
            latest = {}
            for row in history: latest[(row["case_id"], row["model"])] = row
            st.dataframe([{"Case ID": r["case_id"], "Source": r["source"], "Task": r["task_type"],
                "Model": r["model"], "Expected": r["expected_label"] or "REVIEW/AMBIGUOUS", "Predicted": r["predicted_label"],
                "P(HALLUCINATED)": r["hallucination_probability_uncalibrated"] or "—", "Result": r["pass_fail"]} for r in reversed(history)], hide_index=True, use_container_width=True)
            st.download_button("Download recorded results CSV", evaluation_csv(list(latest.values())), "robustrag_manual_evaluation_results.csv", "text/csv", key="download_evaluation_csv")


def main() -> None:
    inject_styles()
    registry = get_model_registry()
    selected_key, meta, task_filter = render_sidebar(registry)
    st.session_state["active_task"] = task_filter
    st.title("🛡️ RobustRAG: RAG Hallucination Detector")
    st.markdown('<div class="rr-lede">Evaluate whether a generated answer is supported by retrieved evidence or contains unsupported claims.</div>', unsafe_allow_html=True)
    try:
        cases = load_all_cases()
    except Exception as exc:
        cases = {}
        st.error(f"Could not load evaluation cases. {type(exc).__name__}: {exc}")
    field_names = {"Q": "Question", "C": "Context", "A": "Answer"}
    used = [field_names[k] for k in meta.input_mode.split("+") if k in field_names]
    unused = [name for name in field_names.values() if name not in used]
    if unused:
        st.info(f"Selected model uses **{' + '.join(used)}**. {' + '.join(unused)} remains available for case entry but does not affect its prediction.")
    else:
        st.caption(f"Selected model uses **{' + '.join(used)}**.")
    render_quick_test_cases(cases)
    st.subheader("Query & Context  ·  Generated Response")
    st.session_state.setdefault("question_input", "")
    st.session_state.setdefault("context_input", "")
    st.session_state.setdefault("answer_input", "")
    with st.form("evidence_check_form", clear_on_submit=False):
        left, right = st.columns([1, 1], gap="large")
        with left:
            question = st.text_area("User Question / Prompt", key="question_input", height=84, placeholder="What did the user ask?")
            context = st.text_area("Retrieved Evidence Passage(s) / Context", key="context_input", height=260, placeholder="Paste retrieved passages or evidence here...")
        with right:
            answer = st.text_area("Candidate Answer to Audit", key="answer_input", height=384, placeholder="Paste the generated answer you want to review...")
        submitted = st.form_submit_button("🔎  Run Hallucination Detection", type="primary", use_container_width=True)
    if submitted:
        if not context.strip() or not answer.strip():
            st.error("Enter both retrieved context and a generated answer before running the check.")
        else:
            try:
                with st.spinner(f"Running {meta.display_name}..."):
                    result = registry.predict(model_key=selected_key, question=question, context=context, answer=answer)
                st.subheader("Diagnostic Results")
                loaded_case_id = st.session_state.get("loaded_evaluation_case_id")
                loaded_case = cases.get(loaded_case_id) if loaded_case_id else None
                if loaded_case and _inputs_for(loaded_case) == {"question": question, "context": context, "answer": answer}:
                    expected = loaded_case["expected_label"] or "REVIEW / AMBIGUOUS"
                    st.caption(f"Reference expected label · {loaded_case_id}: **{expected}** (separate from the model output).")
                elif loaded_case_id:
                    st.caption("The loaded case inputs were edited, so its reference expected label does not apply to this prediction.")
                render_result(result, selected_key, meta, task_filter)
            except Exception as exc:
                st.error(f"Could not run {meta.display_name}. {type(exc).__name__}: {exc}")
                st.caption("Check that the selected checkpoint and its required local artifacts are available.")
    render_evaluation_lab(cases, registry, selected_key, meta, task_filter)


if __name__ == "__main__":
    main()
