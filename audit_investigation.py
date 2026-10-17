import json
from pathlib import Path
from transformers import AutoTokenizer
from hallucination_detector.data.loader import load_jsonl
from hallucination_detector.data.schema import RAGTruthRecord
from hallucination_detector.modeling.tfidf_logistic import normalize_text

data_path = Path("data/processed/ragtruth.jsonl")
records = [RAGTruthRecord(**r) for r in load_jsonl(data_path)]
test_records = [r for r in records if r.split == "test"]
train_records = [r for r in records if r.split == "train"]

tokenizer_path = Path("reports/experiments/baseline_deberta/best_model")
tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_path), local_files_only=True)

print("Tokenizer successfully loaded from local checkpoint!")

# 1. Truncation Analysis
print("\n" + "="*50)
print("PART 1: TRUNCATION & SEQUENCE LENGTH AUDIT")
print("="*50)

def analyze_truncation(recs, name):
    exceeds_512 = 0
    ans_cut = 0
    ans_lost = 0
    ctx_lengths = []
    ans_lengths = []
    
    for r in recs:
        ctx_str = normalize_text(r.context)
        ans_str = normalize_text(r.answer)
        ctx_len = len(tokenizer.encode(ctx_str, add_special_tokens=False))
        ans_len = len(tokenizer.encode(ans_str, add_special_tokens=False))
        ctx_lengths.append(ctx_len)
        ans_lengths.append(ans_len)
        
        # In current single-string input: "[CONTEXT] " + ctx + " [ANSWER] " + ans
        # Total tokens = 1 + 1 + ctx_len + 1 + ans_len + 1 = ctx_len + ans_len + 4
        if ctx_len + ans_len + 4 > 512:
            exceeds_512 += 1
            available_for_ans = 509 - ctx_len
            if available_for_ans <= 0:
                ans_lost += 1
            elif available_for_ans < ans_len:
                ans_cut += 1
                
    ctx_lengths.sort()
    ans_lengths.sort()
    n = len(recs)
    print(f"\n--- {name} ({n} records) ---")
    print(f"Context token length : median={ctx_lengths[n//2]}, p75={ctx_lengths[int(n*0.75)]}, p90={ctx_lengths[int(n*0.9)]}, max={ctx_lengths[-1]}")
    print(f"Answer token length  : median={ans_lengths[n//2]}, p75={ans_lengths[int(n*0.75)]}, p90={ans_lengths[int(n*0.9)]}, max={ans_lengths[-1]}")
    print(f"Total > 512 tokens   : {exceeds_512} ({exceeds_512/n:.1%})")
    print(f"Answer partially cut : {ans_cut} ({ans_cut/n:.1%})")
    print(f"Answer COMPLETELY lost: {ans_lost} ({ans_lost/n:.1%})")
    return exceeds_512, ans_cut, ans_lost

analyze_truncation(train_records, "TRAIN SET")
analyze_truncation(test_records, "TEST SET")

# 2. Performance Breakdown by Truncation Status
print("\n" + "="*50)
print("PART 2: TEST PREDICTIONS COMPARISON (TF-IDF vs DeBERTa)")
print("="*50)

tfidf_preds = load_jsonl("reports/experiments_backup/baseline_tfidf_logreg/test_predictions.jsonl")
deberta_preds = load_jsonl("reports/experiments_backup/baseline_deberta/test_predictions.jsonl")

# Align by id
tfidf_map = {r["id"]: r for r in tfidf_preds}
deberta_map = {r["id"]: r for r in deberta_preds}

correct_tfidf = 0
correct_deberta = 0
both_correct = 0
tfidf_only = 0
deberta_only = 0
both_wrong = 0

trunc_tfidf_correct = 0
trunc_deberta_correct = 0
trunc_total = 0

nontrunc_tfidf_correct = 0
nontrunc_deberta_correct = 0
nontrunc_total = 0

task_perf = {}

for r in test_records:
    rid = r.id
    y_true = int(r.hallucinated)
    p_tf = tfidf_map[rid]["prediction"]
    p_deb = deberta_map[rid]["prediction"]
    
    tf_ok = (p_tf == y_true)
    deb_ok = (p_deb == y_true)
    
    if tf_ok: correct_tfidf += 1
    if deb_ok: correct_deberta += 1
    if tf_ok and deb_ok: both_correct += 1
    elif tf_ok and not deb_ok: tfidf_only += 1
    elif not tf_ok and deb_ok: deberta_only += 1
    else: both_wrong += 1
    
    # Check truncation
    ctx_len = len(tokenizer.encode(normalize_text(r.context), add_special_tokens=False))
    ans_len = len(tokenizer.encode(normalize_text(r.answer), add_special_tokens=False))
    is_trunc = (ctx_len + ans_len + 4 > 512)
    
    if is_trunc:
        trunc_total += 1
        if tf_ok: trunc_tfidf_correct += 1
        if deb_ok: trunc_deberta_correct += 1
    else:
        nontrunc_total += 1
        if tf_ok: nontrunc_tfidf_correct += 1
        if deb_ok: nontrunc_deberta_correct += 1

    task_type = "dict" if isinstance(r.context, dict) else "str"
    if task_type not in task_perf:
        task_perf[task_type] = {"total": 0, "tf_ok": 0, "deb_ok": 0, "halluc": 0, "tf_tp": 0, "deb_tp": 0, "tf_fp": 0, "deb_fp": 0}
    task_perf[task_type]["total"] += 1
    if tf_ok: task_perf[task_type]["tf_ok"] += 1
    if deb_ok: task_perf[task_type]["deb_ok"] += 1
    if y_true == 1:
        task_perf[task_type]["halluc"] += 1
        if p_tf == 1: task_perf[task_type]["tf_tp"] += 1
        if p_deb == 1: task_perf[task_type]["deb_tp"] += 1
    else:
        if p_tf == 1: task_perf[task_type]["tf_fp"] += 1
        if p_deb == 1: task_perf[task_type]["deb_fp"] += 1

print(f"Overall Accuracy: TF-IDF={correct_tfidf/len(test_records):.2%}, DeBERTa={correct_deberta/len(test_records):.2%}")
print(f"Both correct: {both_correct}, TF-IDF only: {tfidf_only}, DeBERTa only: {deberta_only}, Both wrong: {both_wrong}")
print(f"\nNon-truncated ({nontrunc_total} examples):")
print(f"  TF-IDF Accuracy : {nontrunc_tfidf_correct/nontrunc_total:.2%}")
print(f"  DeBERTa Accuracy: {nontrunc_deberta_correct/nontrunc_total:.2%}")
print(f"Truncated ({trunc_total} examples):")
print(f"  TF-IDF Accuracy : {trunc_tfidf_correct/trunc_total:.2%}")
print(f"  DeBERTa Accuracy: {trunc_deberta_correct/trunc_total:.2%}")

print("\n--- Breakdown by Context Type ---")
for task, d in task_perf.items():
    print(f"Context Type: {task} (N={d['total']}, Halluc={d['halluc']}):")
    print(f"  TF-IDF  : Acc={d['tf_ok']/d['total']:.2%}, TP={d['tf_tp']}/{d['halluc']}, FP={d['tf_fp']}")
    print(f"  DeBERTa : Acc={d['deb_ok']/d['total']:.2%}, TP={d['deb_tp']}/{d['halluc']}, FP={d['deb_fp']}")
