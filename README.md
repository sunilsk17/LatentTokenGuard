# LatentTokenGuard

**Object hallucination detection in vision-language models from latent image–text alignment**

When a vision-language model describes an image, some of the objects it names are not in the image.
LatentTokenGuard scores each named object by how well the model's own hidden state for that word
matches the hidden states of the image patches, at a middle layer of the language model. An object
that matches no image region is flagged as a likely hallucination. The score is computed from
tensors that generation already produces, so it needs no second model and no extra forward pass,
and it needs no labelled training data: only the layer pair and the statistic are chosen on a
calibration split.

This repository contains the method, a free-form captioning benchmark built to evaluate it
rigorously, the results for three models, and all data needed to reproduce them.

## Results

Free-form captions of 1,000 COCO val2017 images (seed 42) for each model, 4-bit weights.
Every object named in a caption is labelled real or hallucinated with the CHAIR protocol; the first
mention of each object class per caption is scored. AUROC (×100, positive class = hallucinated) is
cross-fitted over 5 folds of images, so every image is scored exactly once by detectors fit on the
other images.

Objects named late in a long caption are hallucinated far more often than early ones, and token
position alone reaches 74–84 AUROC. The column that matters is therefore **AUROC within caption-position
bins** (deciles of the object's token index), where position carries no information.

| Detector | Training data | LLaVA-1.5-7B | LLaVA-1.5-13B | Qwen2.5-VL-7B |
|---|---|---|---|---|
| Token position | none | 83.7 / 47.4 | 83.6 / 53.8 | 74.2 / 49.2 |
| NLL of the object token | none | 73.5 / 70.9 | 72.3 / 73.4 | 65.4 / 60.2 |
| Internal Confidence (Jiang et al., ICLR 2025) | layer choice | 79.5 / 70.8 | 78.0 / 70.2 | 68.1 / 57.9 |
| Max-patch cosine, same layer (ContextualLens-style) | layer choice | 87.6 / 78.0 | 85.7 / 75.4 | 84.1 / 79.0 |
| **LatentTokenGuard** (best latent statistic) | layer pair + statistic choice | **89.1 / 79.7** | **89.1 / 78.7** | **87.4 / 82.8** |
| NLL + LatentTokenGuard (rank average) | as above | 87.2 / 80.1 | 86.9 / 81.9 | 81.7 / 76.0 |
| Linear probe on hidden states (supervised reference) | ~2,600 labelled objects | 92.1 / 86.0 | 92.5 / 84.0 | 94.6 / 91.8 |

Each cell: overall AUROC / AUROC within position bins. Objects scored: 3,281 / 3,174 / 2,570
(837 / 707 / 407 hallucinated).

Paired bootstrap over images, within position bins:

| Comparison | LLaVA-1.5-7B | LLaVA-1.5-13B | Qwen2.5-VL-7B |
|---|---|---|---|
| LatentTokenGuard − NLL | +8.8 [+4.6, +12.8] | +5.3 [−0.9, +10.8] | +22.7 [+16.0, +29.5] |
| LatentTokenGuard − Internal Confidence | +8.8 [+4.3, +13.1] | +8.5 [+1.9, +16.0] | +24.9 [+18.8, +30.3] |
| (NLL + LatentTokenGuard) − NLL | +9.2 [+7.3, +11.2] | +8.5 [+6.1, +10.7] | +15.8 [+11.7, +20.3] |
| LatentTokenGuard − linear probe | −6.3 [−9.5, −3.4] | −5.3 [−13.1, +1.3] | −8.9 [−11.9, −5.9] |

Findings:

- **Best training-free detector on all three models.** At matched caption position, the latent
  alignment score beats the model's own token probability (NLL) and Internal Confidence; on
  Qwen2.5-VL, whose token probabilities carry little signal, by more than 20 points.
- **It adds information that NLL lacks.** Adding it to NLL raises within-position AUROC by
  8.5–15.8 points on every model.
- **The best layers sit in the middle of the network** for all three models (for example text
  layer 16–20 and image layer 13 of 28 for Qwen2.5-VL).
- **A supervised probe is stronger.** A logistic-regression probe trained on the object-token
  hidden state with ~2,600 labelled objects gains another 5–9 points. LatentTokenGuard needs no
  labels and points to the image region an object matches; the probe needs labels for each model,
  and its detect-then-remove threshold transfers less well (it keeps 88–91% of real objects at a
  95% target, against 95% for LatentTokenGuard).
- **Detect-then-remove.** On the held-out test images, dropping flagged objects at a threshold
  that keeps 95% of real calibration objects lowers CHAIR_i (unique objects) from 25.7 to 15.8 (LLaVA-1.5-7B), 22.7 to 14.9 (13B) and 15.4 to
  10.6 (Qwen2.5-VL).

Full tables, including a held-out split, AUPR, every detector variant and the selected layers, are
in `results/caption_runs/<run>/eval/eval.md`. The protocol is described in
[docs/CAPTION_BENCHMARK.md](docs/CAPTION_BENCHMARK.md).

## Reproducing the results

### Setup

```bash
git clone https://github.com/sunilsk17/LatentTokenGuard.git
cd LatentTokenGuard
pip install -r requirements.txt
python -m pytest tests/
```

`requirements.txt` pins transformers 4.43 for the earlier POPE code. The caption benchmark runs on
transformers 4.43 and 5.x; Qwen2.5-VL needs 4.49 or newer.

### Re-run the evaluation from the published data (CPU, minutes)

Captions, labels and evaluation reports are in this repository. The per-object features and the
hidden states are in the
[`caption-runs-v1` release](https://github.com/sunilsk17/LatentTokenGuard/releases/tag/caption-runs-v1).

```bash
RUN=llava15_7b_4bit_n1000_s42        # or llava15_13b_n1000_s42, qwen25vl_7b_n1000_s42
URL=https://github.com/sunilsk17/LatentTokenGuard/releases/download/caption-runs-v1
curl -LO $URL/$RUN.zip && unzip -n $RUN.zip -d results/caption_runs/
# optional, adds the linear probe
curl -LO $URL/hidden_$RUN.zip && unzip -q hidden_$RUN.zip && mv hidden_$RUN/hidden results/caption_runs/$RUN/
python experiments/caption_eval.py --run results/caption_runs/$RUN
```

`SHA256SUMS` in the release lists the checksum of every file.

### Run a model from scratch (GPU)

[`notebooks/caption_benchmark_cloud.ipynb`](notebooks/caption_benchmark_cloud.ipynb) runs on Kaggle
(two T4 GPUs) or Colab. Set `MODEL` to `llava15_7b_4bit`, `llava15_13b` or `qwen25vl_7b`; a
1,000-image run takes about 2 hours on two T4s. Section 8 extracts the hidden states for the probe.
From the command line:

```bash
python experiments/caption_run.py --model llava15_7b_4bit --n_images 1000 \
    --images_dir <coco>/val2017 --annotations_dir <coco>/annotations --out results/caption_runs/<run>
python experiments/caption_hidden.py --model llava15_7b_4bit --run results/caption_runs/<run>   # optional
python experiments/caption_eval.py --run results/caption_runs/<run>
```

Random-weight test models (`experiments/make_tiny_llava.py`, `experiments/make_tiny_qwen25vl.py`)
run the whole pipeline on a laptop CPU.

## Repository layout

```
caption_bench/        caption benchmark
  capture.py          single-pass generation with hidden-state capture (LLaVA-1.5, Qwen2.5-VL)
  chair.py            CHAIR object extraction and COCO ground truth
  features.py         per-object features: NLL, entropy, Internal Confidence, latent alignment
  evaluate.py         detectors, held-out and cross-fitted evaluation, paired bootstrap
experiments/
  caption_run.py      caption images and record features (resumable, multi-GPU shards)
  caption_hidden.py   object-token hidden states for the linear probe
  caption_eval.py     evaluation report (eval.md, eval.json, plots)
configs/caption_bench.yaml   models, prompt and evaluation settings
notebooks/            Kaggle / Colab notebook for GPU runs
results/caption_runs/ captions, run settings and evaluation reports of the published runs
tests/                unit tests and capture-consistency tests
docs/                 benchmark protocol

main.py, models/, detection/, data/, evaluation/   earlier POPE pipeline (see below)
```

## Earlier experiments: POPE

The first version of LatentTokenGuard was evaluated on POPE yes/no questions with token-level
scores (LLaVA-1.5-7B: 78.3 AUROC on POPE-adversarial). Those results and their code are kept for
reference in [RESULTS.md](RESULTS.md) and `main.py`. POPE answers are a single "yes" or "no", so a
detector can score well there by reflecting the model's own answer rather than its grounding; the
caption benchmark above was built to remove that shortcut and is the evaluation to cite.

## Citation

```bibtex
@misc{kumar2026latenttokenguard,
  title  = {LatentTokenGuard: Object Hallucination Detection in Vision-Language Models
            from Latent Image-Text Alignment},
  author = {Sunil Kumar S},
  year   = {2026},
  url    = {https://github.com/sunilsk17/LatentTokenGuard}
}
```

## License

MIT. See [LICENSE](LICENSE).
