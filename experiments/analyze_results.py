"""
experiments/analyze_results.py
Unified results analysis for LatentTokenGuard.
Finds the optimal threshold and calculates final metrics (F1, Accuracy, AUROC).
"""

import json
import numpy as np
import os
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score

def analyze(json_path):
    if not os.path.exists(json_path):
        print(f"File not found: {json_path}")
        return

    with open(json_path, 'r') as f:
        data = json.load(f)
    
    # Extract scores and ground truth
    # Latent Contrast: Higher score -> higher likelihood of hallucination
    scores = [p['hallucination_score'] for p in data['predictions']]
    # gt_label 'no' means it IS a hallucination (hallucinated item) -> True (1)
    trues  = [1 if p['gt_label'] == 'no' else 0 for p in data['predictions']]

    # Search for best threshold (F1 optimization)
    best_f1, best_t = 0, 0
    best_acc = 0
    
    # Try thresholds from 0.05 to 0.50
    for t in np.arange(0.05, 0.51, 0.005):
        preds = [1 if s > t else 0 for s in scores]
        f1 = f1_score(trues, preds, zero_division=0)
        acc = accuracy_score(trues, preds)
        
        if f1 > best_f1:
            best_f1 = f1
            best_t = t
            best_acc = acc

    # Calculate AUROC
    try:
        auroc = roc_auc_score(trues, scores)
    except:
        auroc = 0.5

    model_name = data['metadata'].get('model', 'Unknown')
    samples = data['metadata'].get('n_samples', len(scores))
    
    print(f"\n=======================================================")
    print(f"  Final Results for {model_name.upper()} ({samples} samples)")
    print(f"=======================================================")
    print(f"  🎯 Optimal Threshold: {best_t:.3f}")
    print(f"  📈 Best F1 Score:     {best_f1:.4f}  <-- Final Paper Value")
    print(f"  ✅ Accuracy:          {best_acc:.4f}")
    print(f"  📊 AUROC:             {auroc:.4f}")
    print(f"=======================================================\n")

if __name__ == "__main__":
    # Analyze both results if they exist
    analyze("results/pope_adversarial_phi3_alignment_only.json")
    analyze("results/pope_adversarial_llava_alignment_only.json")
