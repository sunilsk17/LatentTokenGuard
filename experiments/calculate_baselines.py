import json
import numpy as np

def calculate_pope_qa_metrics(json_path, threshold=0.15):
    with open(json_path, 'r') as f:
        data = json.load(f)
        
    raw_correct = 0
    ltg_correct = 0
    total = len(data['predictions'])
    
    # Standard POPE metrics (Yes/No)
    raw_tp = 0 # GT=Yes, Pred=Yes
    raw_fp = 0 # GT=No, Pred=Yes
    raw_tn = 0 # GT=No, Pred=No
    raw_fn = 0 # GT=Yes, Pred=No
    
    ltg_tp = 0
    ltg_fp = 0
    ltg_tn = 0
    ltg_fn = 0

    for p in data['predictions']:
        gt_is_yes = (p['gt_label'] == 'yes')
        
        # Determine raw model pred
        # usually model_answer starts with Yes or No
        ans = p['model_answer'].lower().strip()
        if ans.startswith('yes'):
            raw_pred_is_yes = True
        elif ans.startswith('no'):
            raw_pred_is_yes = False
        else:
            # fallback
            raw_pred_is_yes = 'yes' in ans
            
        # Determine LTG pred
        score = p['hallucination_score']
        ltg_pred_is_yes = raw_pred_is_yes
        
        # If the model said Yes, but LTG says it's a hallucination (score > threshold), flip to No
        if raw_pred_is_yes and score > threshold:
            ltg_pred_is_yes = False
            
        # Tally Raw
        if gt_is_yes and raw_pred_is_yes: raw_tp += 1
        if not gt_is_yes and raw_pred_is_yes: raw_fp += 1
        if not gt_is_yes and not raw_pred_is_yes: raw_tn += 1
        if gt_is_yes and not raw_pred_is_yes: raw_fn += 1
        
        # Tally LTG
        if gt_is_yes and ltg_pred_is_yes: ltg_tp += 1
        if not gt_is_yes and ltg_pred_is_yes: ltg_fp += 1
        if not gt_is_yes and not ltg_pred_is_yes: ltg_tn += 1
        if gt_is_yes and not ltg_pred_is_yes: ltg_fn += 1

    def get_metrics(tp, fp, tn, fn):
        acc = (tp + tn) / total
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        yes_ratio = (tp + fp) / total
        return acc, f1, yes_ratio

    raw_acc, raw_f1, raw_yes = get_metrics(raw_tp, raw_fp, raw_tn, raw_fn)
    ltg_acc, ltg_f1, ltg_yes = get_metrics(ltg_tp, ltg_fp, ltg_tn, ltg_fn)

    print(f"File: {json_path}")
    print(f"Threshold used for LTG: {threshold}")
    print(f"--- RAW LVLM ---")
    print(f"Accuracy: {raw_acc:.4f} | F1: {raw_f1:.4f} | Yes Proportion: {raw_yes:.2%}")
    print(f"--- LTG CORRECTED ---")
    print(f"Accuracy: {ltg_acc:.4f} | F1: {ltg_f1:.4f} | Yes Proportion: {ltg_yes:.2%}")
    print("------------------------------------------\n")

if __name__ == "__main__":
    import sys
    # For LLaVA optimal threshold is ~0.15
    calculate_pope_qa_metrics("results/pope_adversarial_llava_alignment_only.json", 0.15)
    # For Phi3, optimal threshold is ~0.09
    calculate_pope_qa_metrics("results/pope_adversarial_phi3_alignment_only.json", 0.09)
