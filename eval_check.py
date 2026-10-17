import numpy as np
from pathlib import Path

# Corrected paths explicitly using the src package prefix
from src.hallucination_detector.evaluation.metrics import evaluate_binary_classifier
from src.hallucination_detector.modeling.longformer_classifier import LongformerClassifier
from src.hallucination_detector.data.loader import load_jsonl

def main():
    # 1. Dataset & model paths
    test_data_path = "data/processed/test.jsonl"  # Adjust if your test split file is somewhere else
    model_dir = "reports/experiments/longformer_1024_head/best_model"
    
    print(f"Loading evaluation dataset from: {test_data_path}")
    if not Path(test_data_path).exists():
        print(f"Error: Could not find test data file at: {test_data_path}")
        return
        
    test_records = load_jsonl(test_data_path)
    
    # 2. Load the validated Epoch 2 checkpoint
    print(f"Loading target model checkpoint from: {model_dir}")
    if not Path(model_dir).exists():
        print(f"Error: Checkpoint folder structure not found at: {model_dir}")
        return

    model = LongformerClassifier.from_pretrained(
        model_dir,
        max_length=1024,
        input_mode="cls",
        truncation_strategy="head",
        global_attention_mode="cls"
    )
    
    # 3. Generate predictions
    print("Running sequence evaluation pipeline via CUDA...")
    probabilities = model.predict_proba(
        test_records,
        batch_size=2,
        device="cuda",
        bf16=True
    )
    predictions = np.argmax(probabilities, axis=1)
    
    # 4. Compute metrics using corrected call signature
    metrics = evaluate_binary_classifier(
        records=test_records,
        predictions=predictions,
        probabilities=probabilities
    )
    
    print("\n" + "="*30)
    print("      ACTUAL GENERATED METRICS")
    print("="*30)
    print(f"Accuracy:  {metrics['accuracy']:.4f}")
    print(f"F1-Score:  {metrics['f1']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall:    {metrics['recall']:.4f}")
    print(f"ROC-AUC:   {metrics['roc_auc']:.4f}")
    print("="*30)

if __name__ == "__main__":
    main()
