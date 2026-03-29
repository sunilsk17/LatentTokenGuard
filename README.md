# LatentTokenGuard

**Visual Grounding and Hallucination Detection in LVLMs via Single-Pass Latent Alignment**

LatentTokenGuard is a post-hoc hallucination detection framework designed for Large Vision-Language Models (LVLMs). It identifies ungrounded generations by directly measuring the cosine similarity between generated text tokens and visual patch embeddings within the model's internal latent space.

Unlike prior methods that require multiple generative passes or modify decoding distributions, LatentTokenGuard operates as a single-pass observational detector, preserving linguistic integrity while identifying hallucinations in real-time.

---

## 🛠️ Project Design

The repository is organized into modular components for model wrapping, latent feature extraction, and benchmark evaluation:

-   **`models/`**: Unified wrappers for LLaVA-1.5-7B and Phi-3.5-Vision to extract hidden states and visual patches.
-   **`detection/`**: The core "Latent Contrast Engine" (LCE) which computes alignment scores and makes grounding decisions.
-   **`experiments/`**: Reproducible scripts for evaluating performance on POPE, MMHal-Bench, and automated ablation studies.
-   **`data/`**: Loaders for standard benchmarks and our custom "COCO Indoor" subset.

---

## 📂 Project Structure

```bash
LatentTokenGuard/
├── main.py                # Main CLI entry point
├── config.yaml            # Hyperparameters and dataset paths
├── requirements.txt       # Environment dependencies
│
├── models/                # LVLM Inference Wrappers
│   ├── base_wrapper.py    # Abstract base class
│   ├── llava_wrapper.py   # LLaVA-1.5-7B implementation
│   └── phi3_wrapper.py    # Phi-3.5-Vision implementation
│
├── detection/             # Core Detection Logic
│   ├── token_extractor.py # spaCy content token filtering
│   ├── latent_contrast.py # Core Cosine-Similarity Engine
│   └── decision.py        # Threshold-based classification
│
├── data/                  # Dataset Loaders
│   ├── pope_loader.py     # POPE benchmark loader
│   ├── mmhal_loader.py    # MMHal-Bench loader
│   └── download_coco.py   # Automated COCO annotation setup
│
└── experiments/           # Reproducibility Scripts
    ├── ablation.py        # Layer-wise ablation study
    └── efficiency.py      # Latency benchmarking
```

---

### 1. Installation
```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

### 2. Dataset Setup
Download the COCO validation annotations required for the POPE and Indoor benchmarks:
```bash
python data/download_coco.py
```
*(Note: You will need the COCO val2017 image directory extracted to `data/coco/val2017/` for full evaluation.)*

### 3. Run Inference
To test the detector on a sample of 5 POPE questions using LLaVA:
```bash
python main.py --model llava --dataset pope --split adversarial --n_samples 5 --threshold 0.150
```

---

## 📈 Benchmark Results

The following metrics were achieved on the **POPE-adversarial** split (3,000 samples) using an NVIDIA A100 GPU workstation at optimal decision thresholds.

| Model | Accuracy | F1 Score | AUROC | Latency |
| :--- | :--- | :--- | :--- | :--- |
| **LLaVA-1.5-7B** | **69.67%** | **0.7415** | **0.7827** | **~747 ms** |
| Phi-3.5-Vision | 51.70% | 0.6727 | 0.5078 | **~599 ms** |

For more detailed logs and ablation results, see [RESULTS.md](RESULTS.md).

---

## 📄 Citation

If you use this work in your research, please cite our manuscript:

```bibtex
@article{latenttoken2025,
  title={LatentTokenGuard: Visual Grounding and Hallucination Detection in LVLMs via Single-Pass Latent Alignment},
  author={Sunil Kumar S},
  year={2026}
}
```

---

*This codebase is intended for research purposes only.*
