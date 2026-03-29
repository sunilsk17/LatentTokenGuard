"""
experiments/plot_distributions.py
Generates a KDE distribution plot of hallucination scores
to illustrate the separability of factual vs hallucinated tokens.
"""

import json
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid")
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 11

def plot_distribution(json_path, save_path="results/score_distribution.png"):
    if not os.path.exists(json_path):
        print("Data not found.")
        return

    with open(json_path, 'r') as f:
        data = json.load(f)
        
    faithful_scores = []
    hallu_scores = []
    
    for p in data['predictions']:
        # gt_label 'no' means the object isn't there, so the model saying 'yes' is a hallucination
        if p['gt_label'] == 'no':
            hallu_scores.append(p['hallucination_score'])
        else:
            faithful_scores.append(p['hallucination_score'])

    plt.figure(figsize=(8, 5))
    
    # Plot KDEs
    sns.kdeplot(faithful_scores, color="seagreen", fill=True, alpha=0.4, linewidth=2, label="Faithful (Grounded)")
    sns.kdeplot(hallu_scores, color="crimson", fill=True, alpha=0.4, linewidth=2, label="Hallucinated (Ungrounded)")
    
    # Add optimal threshold line (0.15 for LLaVA)
    plt.axvline(x=0.15, color='black', linestyle='--', linewidth=1.5, label='Optimal Threshold (0.15)')

    plt.xlabel(r'Latent Alignment Score ($\mathcal{H}$)', fontsize=12)
    plt.ylabel('Density', fontsize=12)
    plt.title('Distribution of Latent Alignment Scores (LLaVA-1.5-7B)', fontsize=14, pad=15)
    plt.xlim([0, 0.4])
    plt.legend(loc="upper right", frameon=True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    print(f"✅ Generated KDE Distribution Plot: {save_path}")

if __name__ == "__main__":
    # Use the LLaVA local JSON data
    plot_distribution("results/pope_adversarial_llava_alignment_only.json", "results/llava_score_distribution.png")
