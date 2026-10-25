from pathlib import Path

from streamlit.testing.v1 import AppTest

from app.streamlit_app import _inputs_for, _record, load_all_cases, load_supplied_cases

APP_PATH = Path(__file__).resolve().parents[1] / "app" / "streamlit_app.py"


def _widget(widgets, label):
    return next(widget for widget in widgets if widget.label == label)


def test_both_case_sources_are_separate_unique_and_task_mapped():
    cases = load_all_cases()
    supplied = load_supplied_cases()
    assert len(cases) == 25
    assert len(supplied) == 7
    assert [supplied[f"ragtruth_{rid}"]["expected_label"] for rid in (12912, 12917, 16464, 16467, 9000, 9002)] == [
        "SUPPORTED", "HALLUCINATED", "SUPPORTED", "HALLUCINATED", "SUPPORTED", "HALLUCINATED"
    ]
    assert [supplied[f"ragtruth_{rid}"]["task_type"] for rid in (12912, 12917, 16464, 16467, 9000, 9002)] == [
        "QA", "QA", "QA", "QA", "Data2txt", "Data2txt"
    ]
    assert "which may be more stressful on the body" in supplied["ragtruth_16467"]["answer"]
    assert supplied["ragtruth_9000"]["context"] == supplied["ragtruth_9002"]["context"]
    assert supplied["ragtruth_summary_warning_v1"]["expected_label"] == "NO RELIABLE VERDICT"


def test_expected_label_is_not_part_of_production_input_mapping():
    case = load_supplied_cases()["ragtruth_12912"]
    assert _inputs_for(case) == {"question": case["question"], "context": case["context"], "answer": case["answer"]}
    assert "expected_label" not in _inputs_for(case)
    assert "source_annotation" not in _inputs_for(case)


def test_task_aware_verdicts_record_binary_aliases_and_abstentions_correctly():
    case = load_all_cases()["v1_qa_contradiction_location"]
    assert _record(case, "longformer-head", "LIKELY HALLUCINATED")["pass_fail"] == "PASS"
    assert _record(case, "longformer-head", "UNCERTAIN")["pass_fail"] == "REVIEW"


def test_quick_case_selector_loads_existing_contradiction_without_running_inference():
    app = AppTest.from_file(APP_PATH).run(timeout=30)
    selector = _widget(app.selectbox, "Quick Test Cases")
    assert "Direct contradiction" in selector.options
    selector.set_value("Direct contradiction").run(timeout=30)
    _widget(app.button, "Load into inputs").click().run(timeout=30)
    assert not app.exception
    case = load_all_cases()["v1_qa_contradiction_location"]
    assert app.text_area[0].value == case["question"]
    assert app.text_area[1].value == case["context"]
    assert app.text_area[2].value == case["answer"]
    assert not any("Diagnostic Results" in item.value for item in app.subheader)


def test_prediction_after_quick_case_load_shows_reference_label_separately():
    app = AppTest.from_file(APP_PATH).run(timeout=30)
    _widget(app.selectbox, "Choose Detection Model").set_value("tfidf-lr").run(timeout=30)
    _widget(app.selectbox, "Quick Test Cases").set_value("Direct contradiction").run(timeout=30)
    _widget(app.button, "Load into inputs").click().run(timeout=30)
    _widget(app.form_submit_button, "🔎  Run Hallucination Detection").click().run(timeout=60)
    assert not app.exception
    assert any("Reference expected label" in item.value and "HALLUCINATED" in item.value for item in app.caption)
    assert any("Diagnostic Results" in item.value for item in app.subheader)


def test_warning_only_summary_case_is_not_an_inference_case():
    app = AppTest.from_file(APP_PATH).run(timeout=30)
    _widget(app.selectbox, "Case · ID · task · short description").set_value("ragtruth_summary_warning_v1").run(timeout=30)
    assert not app.exception
    assert any("NO RELIABLE VERDICT" in alert.value for alert in app.warning)
    assert _widget(app.button, "Run selected model on this case").disabled
    assert not any("Diagnostic Results" in item.value for item in app.subheader)


def test_case_loads_into_shared_fields_without_automatically_running_prediction():
    app = AppTest.from_file(APP_PATH).run(timeout=30)
    _widget(app.selectbox, "Choose Detection Model").set_value("tfidf-lr").run(timeout=30)
    _widget(app.selectbox, "Case · ID · task · short description").set_value("ragtruth_12912").run(timeout=30)
    _widget(app.button, "Load case into main inputs").click().run(timeout=30)
    case = load_supplied_cases()["ragtruth_12912"]
    assert app.text_area[0].value == case["question"]
    assert app.text_area[1].value == case["context"]
    assert app.text_area[2].value == case["answer"]
    assert not app.exception
    assert not any('<section class="rr-result' in item.value for item in app.markdown)


def test_lab_records_real_registry_prediction_and_csv_for_supplied_case():
    app = AppTest.from_file(APP_PATH).run(timeout=30)
    _widget(app.selectbox, "Choose Detection Model").set_value("tfidf-lr").run(timeout=30)
    _widget(app.selectbox, "Case · ID · task · short description").set_value("ragtruth_12912").run(timeout=30)
    _widget(app.button, "Run selected model on this case").click().run(timeout=60)
    assert not app.exception
    assert any("Evaluation Lab" in item.label for item in app.expander)
    assert any(item.label == "Download recorded results CSV" for item in app.get("download_button"))
    assert any("P(HALLUCINATED)" in row.value for row in app.dataframe)
