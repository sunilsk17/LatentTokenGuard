# LatentTokenGuard: Experimental Results & Benchmarks

This document records the official benchmark results for the **LatentTokenGuard** hallucination detection framework. All evaluations were conducted on the **POPE-adversarial** split (3,000 samples) to ensure a rigorous assessment against intentionally deceptive prompts.

---

## 📊 1. Primary Metrics (POPE-Adversarial)

Results were obtained using **Layer 15** activations with models loaded in **4-bit quantization** (NF4) and a fixed random seed for reproducibility.

| Model Backbone | Threshold ($\tau$) | Accuracy | F1 Score | AUROC | Latency (ms) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **LLaVA-1.5-7B** | **0.150** | **69.67%** | **0.7415** | **0.7827** | **~747** |
| Phi-3.5-Vision | 0.090 | 51.70% | 0.6727 | 0.5078 | **~599** |

*Note: Latency reflects total inference time (Generation + Detection) on a single NVIDIA A100 (40GB) GPU.*

---

## 📈 2. Visualizations

The following plots provide deeper insights into the latent space characteristics:

- **[Score Distribution](results/llava_score_distribution.png)**: Visualizes the separation between Grounded (Faithful) and Hallucinated text tokens.
- **[ROC Comparison](results/roc_comparison.png)**: Illustrates the diagnostic capability of LatentTokenGuard across architectures.
- **[PR Curve](results/pr_curve.png)**: Shows the Precision-Recall trade-off at various alignment thresholds.
- **[Layer Ablation](results/layer_ablation.png)**: Demonstrates why Layer 15 was selected as the optimal "alignment sweet spot."

---

## 💻 3. Environment & Methodology

### Hardware Specification
- **GPU**: NVIDIA A100 (40GB VRAM)
- **CPU**: High-performance host (Xeon/Epyc) for asynchronous spaCy POS tagging.

### Software Stack
- **Framework**: `torch` + `transformers==4.43.0`
- **NLP Pipeline**: `spacy` (`en_core_web_sm`)
- **Quantization**: `bitsandbytes` (4-bit NF4)

### Qualitative Summary
Our analysis identifies a clear "Architectural Dichotomy": **LLaVA-1.5** maintains high spatial fidelity through its direct MLP projector, enabling powerful post-hoc detection. In contrast, **Phi-3.5** utilizes aggressive visual compression (HD-transform), which abstracts spatial correspondence into semantic mixtures, making simple cosine-similarity alignment significantly less effective.

---

## 🔁 4. Replication Guide

To replicate the LLaVA benchmark reported in the paper:
```bash
python main.py \
    --model llava \
    --dataset pope \
    --split adversarial \
    --threshold 0.150 \
    --mode alignment_only
```

*Verified and finalized: March 29, 2026*
