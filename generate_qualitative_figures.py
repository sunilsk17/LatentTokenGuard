import sys, os, json, math, yaml
import textwrap
import matplotlib.pyplot as plt

# 1. Setup Path Logic
sys.path.append(os.getcwd())
if os.path.exists("LatentTokenGuard"): sys.path.append(os.path.abspath("LatentTokenGuard"))
from data.pope_loader import POPELoader

def generate_impactful_pdf_figures(config_path="config.yaml"):
    if not os.path.exists(config_path): config_path = os.path.join("LatentTokenGuard", "config.yaml")
    with open(config_path, "r") as f: config = yaml.safe_load(f)
    
    llava_res = "results/pope_adversarial_llava_alignment_only.json"
    if not os.path.exists(llava_res): llava_res = os.path.join("LatentTokenGuard", llava_res)
    
    with open(llava_res, "r") as f:
        data = json.load(f)
        res_map = {str(p['question_id']): p for p in data['predictions']}
        
    loader = POPELoader(config)
    samples = loader.get_all_samples(split="adversarial")

    # CUSTOM THRESHOLD from Table 1 of the paper
    THRESHOLD = 0.150

    # DIVERSE_IDS: Picking a mix of Successes and Hallucinations
    # 8, 26, 72 are Hallucinations the model made and we caught (Red)
    # 31, 55, 63 are Faithful grounding cases (Green)
    DIVERSE_IDS = ["8", "26", "72", "31", "55", "63"]
    KITCHEN_IDS = ["39", "40", "41"]

    # --- FIG 5: DIVERSE GALLERY ---
    fig, axes = plt.subplots(2, 3, figsize=(20, 14))
    items = [s for s in samples if str(s['question_id']) in DIVERSE_IDS]
    # Sort items to match the order in DIVERSE_IDS for consistency
    items.sort(key=lambda x: DIVERSE_IDS.index(str(x['question_id'])))

    for i, sample in enumerate(items):
        r, c = i // 3, i % 3
        res = res_map[str(sample['question_id'])]
        score = res['hallucination_score']
        
        # Pulling out the Ground Truth (GT)
        gt = res['gt_label'].upper()
        
        # Determine status based on the optimized threshold
        status = "Hallucinated" if score > THRESHOLD else "Faithful"
        color = "crimson" if status == "Hallucinated" else "seagreen"
        
        # Wrapping the long questions
        wrapped_q = textwrap.fill(sample['question'], width=45)
        title_text = f"Q: {wrapped_q}\nGT: {gt} | LCE Pred: {status} ({score:.3f})"
        
        axes[r,c].imshow(sample['image']); axes[r,c].axis('off')
        axes[r,c].set_title(title_text, color=color, fontweight='bold', fontsize=14, pad=10)
    
    plt.tight_layout()
    # Changed output to .png
    plt.savefig("results/qualitative_gallery.png", bbox_inches='tight', dpi=300)
    plt.close()

    # --- FIG 6: CONTROLLED CASE STUDY ---
    fig, axes = plt.subplots(1, 3, figsize=(20, 7))
    items = [s for s in samples if str(s['question_id']) in KITCHEN_IDS]
    items.sort(key=lambda x: KITCHEN_IDS.index(str(x['question_id'])))

    for i, sample in enumerate(items):
        res = res_map[str(sample['question_id'])]
        score = res['hallucination_score']
        
        # Pulling out the Ground Truth (GT)
        gt = res['gt_label'].upper()
        
        status = "Hallucinated" if score > THRESHOLD else "Faithful"
        color = "crimson" if status == "Hallucinated" else "seagreen"
        
        # Wrapping the long questions
        wrapped_q = textwrap.fill(sample['question'], width=45)
        title_text = f"Probe: {wrapped_q}\nGT: {gt} | Detect: {status} ({score:.3f})"
        
        axes[i].imshow(sample['image']); axes[i].axis('off')
        axes[i].set_title(title_text, color=color, fontweight='bold', fontsize=14, pad=15)
    
    plt.tight_layout()
    # Changed output to .png
    plt.savefig("results/case_study_kitchen.png", bbox_inches='tight', dpi=300)
    plt.close()
    print("✅ Created: Impactful Gallery and Case Study (Kitchen Probe) PNGs with Ground Truth.")

if __name__ == "__main__":
    generate_impactful_pdf_figures()
