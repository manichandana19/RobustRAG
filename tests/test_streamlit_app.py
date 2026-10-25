from pathlib import Path

from streamlit.testing.v1 import AppTest
from app.streamlit_app import selectable_model_keys
from hallucination_detector.inference.model_registry import MODEL_CATALOG

APP_PATH = Path(__file__).resolve().parents[1] / "app" / "streamlit_app.py"


def _widget(widgets, label):
    return next(widget for widget in widgets if widget.label == label)


def test_qca_is_opt_in_and_disclosed_until_successful_registry_load():
    app = AppTest.from_file(APP_PATH).run(timeout=30)

    assert not app.exception
    model_selector = next(widget for widget in app.selectbox if widget.label == "Choose Detection Model")
    assert MODEL_CATALOG["longformer-qca"].display_name not in model_selector.options
    assert any(button.label == "Load Longformer QCA" for button in app.button)
    assert any("no manifest linking it to these weights" in item.value for item in app.info)
    assert "longformer-qca" not in selectable_model_keys(MODEL_CATALOG, set())
    assert "longformer-qca" in selectable_model_keys(MODEL_CATALOG, {"longformer-qca"})


def test_optimized_bilstm_appears_only_after_successful_registry_load():
    assert "bilstm-optimized" not in selectable_model_keys(MODEL_CATALOG, set())
    assert "bilstm-optimized" in selectable_model_keys(MODEL_CATALOG, {"bilstm-optimized"})
    app = AppTest.from_file(APP_PATH).run(timeout=30)
    assert not app.exception
    selector = next(widget for widget in app.selectbox if widget.label == "Choose Detection Model")
    assert "bilstm-optimized" not in selector.options
    assert any(button.label == "Load Optimized BiLSTM" for button in app.button)


def test_primary_screen_has_reference_title_and_no_obsolete_demo_buttons():
    app = AppTest.from_file(APP_PATH).run(timeout=30)
    assert not app.exception
    assert any("RobustRAG: RAG Hallucination Detector" in item.value for item in app.title)
    assert any("Run Hallucination Detection" in item.label for item in app.get("form_submit_button"))
    quick_cases = next(widget for widget in app.selectbox if widget.label == "Quick Test Cases")
    assert "Direct contradiction" in quick_cases.options
    selector = next(widget for widget in app.selectbox if widget.label == "Choose Detection Model")
    assert set(selector.options) == {
        "Longformer (Head Truncation)", "DeBERTa-v3-base (512 tokens)",
        "BiLSTM + Bahdanau Attention", "TF-IDF + Logistic Regression",
    }


def test_each_available_model_keeps_its_documented_input_mode_notice():
    app = AppTest.from_file(APP_PATH).run(timeout=30)
    expected = {
        "longformer-head": "Context + Answer",
        "deberta": "Context + Answer",
        "bilstm-attention": "Question + Context + Answer",
        "tfidf-lr": "Context + Answer",
    }
    for model_key, input_mode in expected.items():
        _widget(app.selectbox, "Choose Detection Model").set_value(model_key).run(timeout=30)
        assert not app.exception
        notices = [item.value for item in app.info] + [item.value for item in app.caption]
        assert any(input_mode in item for item in notices), (model_key, notices)


def test_blank_context_and_answer_show_actionable_input_error():
    app = AppTest.from_file(APP_PATH).run(timeout=30)
    _widget(app.text_area, "Retrieved Evidence Passage(s) / Context").set_value("")
    _widget(app.text_area, "Candidate Answer to Audit").set_value("")
    app.form_submit_button[0].click().run(timeout=30)

    assert any("Enter both retrieved context and a generated answer" in alert.value for alert in app.error)
    assert not any('<section class="rr-result' in markdown.value for markdown in app.markdown)


def test_task_selector_offers_four_tasks_and_summary_head_warns_no_reliable():
    app = AppTest.from_file(APP_PATH).run(timeout=30)
    assert not app.exception
    task = _widget(app.selectbox, "Answer task")
    assert task.options == ["QA", "Summarization", "Data-to-text", "Not sure"]
