"""
experiments/visualize_results.py
Professional visualization for LatentTokenGuard results.
Generates ROC Curves and F1-Threshold plots for the paper.
"""

import json
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc, f1_score, precision_recall_curve

# Use a clean, professional style
sns.set_theme(style="whitegrid")
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 11

def load_data(json_path):
    if not os.path.exists(json_path):
        return None, None
    with open(json_path, 'r') as f:
        data = json.load(f)
    scores = [p['hallucination_score'] for p in data['predictions']]
    trues = [1 if p['gt_label'] == 'no' else 0 for p in data['predictions']]
    return np.array(scores), np.array(trues)

def plot_combined_roc(llava_res, phi_res, save_path="results/roc_comparison.png"):
    plt.figure(figsize=(8, 6))
    
    # Plot LLaVA ROC
    if llava_res[0] is not None:
        fpr, tpr, _ = roc_curve(llava_res[1], llava_res[0])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, color='darkorange', lw=2.5, label=f'LLaVA-1.5-7B (AUC = {roc_auc:.2f})')
        
    # Plot Phi-3 ROC
    if phi_res[0] is not None:
        fpr, tpr, _ = roc_curve(phi_res[1], phi_res[0])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, color='royalblue', lw=2.5, linestyle='--', label=f'Phi-3 Vision (AUC = {roc_auc:.2f})')
        
    plt.plot([0, 1], [0, 1], color='navy', linestyle=':', label='Random Guessing')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (Incorrectly Flagged)', fontsize=12)
    plt.ylabel('True Positive Rate (Recall/Correct Detections)', fontsize=12)
    plt.title('ROC Curve Comparison of LatentTokenGuard', fontsize=14, pad=15)
    plt.legend(loc="lower right", frameon=True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    print(f"✅ Generated ROC Curve: {save_path}")

def plot_f1_threshold(llava_res, phi_res, save_path="results/f1_v_threshold.png"):
    plt.figure(figsize=(8, 6))
    thresholds = np.arange(0.0, 0.5, 0.005)
    
    # LLaVA F1 curve
    if llava_res[0] is not None:
        f1s = [f1_score(llava_res[1], [1 if s > t else 0 for s in llava_res[0]], zero_division=0) for t in thresholds]
        plt.plot(thresholds, f1s, color='darkorange', lw=2, label='LLaVA-1.5-7B')
        best_t = thresholds[np.argmax(f1s)]
        plt.axvline(x=best_t, color='darkorange', linestyle=':', alpha=0.5, label=f'Best LLaVA T={best_t:.2f}')
        
    # Phi-3 F1 curve
    if phi_res[0] is not None:
        f1s = [f1_score(phi_res[1], [1 if s > t else 0 for s in phi_res[0]], zero_division=0) for t in thresholds]
        plt.plot(thresholds, f1s, color='royalblue', lw=2, linestyle='--', label='Phi-3 Vision')
        best_t = thresholds[np.argmax(f1s)]
        plt.axvline(x=best_t, color='royalblue', linestyle=':', alpha=0.5, label=f'Best Phi T={best_t:.2f}')

    plt.xlabel('Detection Threshold (Confidence)', fontsize=12)
    plt.ylabel('F1 Score', fontsize=12)
    plt.title('F1 Score vs. Latent Contrast Threshold', fontsize=14, pad=15)
    plt.legend(loc="lower left", frameon=True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    print(f"✅ Generated F1 Curve: {save_path}")

def main():
    # Use the Downloads directory the user suggested, or fallback to current results
    downloads_dir = "/Users/sunilkumars/Downloads/LatentTokenGuard_results"
    
    # Check Downloads first
    phi_res = load_data(os.path.join(downloads_dir, "pope_adversarial_phi3_alignment_only.json"))
    llava_res = load_data(os.path.join(downloads_dir, "pope_adversarial_llava_alignment_only.json"))
    
    # Fallback to current results/ folder if downloads doesn't exist
    if phi_res[0] is None:
        phi_res = load_data("results/pope_adversarial_phi3_alignment_only.json")
    if llava_res[0] is None:
        llava_res = load_data("results/pope_adversarial_llava_alignment_only.json")

    if phi_res[0] is None and llava_res[0] is None:
        print("❌ No result files found. Run the benchmark first!")
        return

    # Ensure results folder exists locally
    os.makedirs("results", exist_ok=True)
    
    plot_combined_roc(llava_res, phi_res)
    plot_f1_threshold(llava_res, phi_res)

if __name__ == "__main__":
    main()
