# Caption benchmark: object hallucination detection in free-form captions

The model describes COCO images in free form. Each object it names is labelled real or hallucinated with the CHAIR protocol (Rohrbach et al., 2018), and every detector assigns a score to each object. Because the captions contain no yes/no answers, a detector cannot succeed by echoing the model's own answer, which is possible on POPE.

## Running

| Step | Hardware | Command |
|---|---|---|
| Tests | CPU | `python -m pytest tests/test_caption_bench.py tests/test_capture_consistency.py` |
| Smoke test (random-weight LLaVA-1.5, 17 MB) | CPU | `python experiments/make_tiny_llava.py` then `python experiments/caption_run.py --model tiny_llava15 --n_images 3` |
| Smoke test (random-weight Qwen2.5-VL, 67 MB; transformers >= 4.49) | CPU | `python experiments/make_tiny_qwen25vl.py` then `python experiments/caption_run.py --model tiny_qwen25vl --n_images 3` |
| LLaVA-1.5-7B / 13B run | CUDA GPU (T4 with 4-bit, or A100 with fp16) | `notebooks/caption_benchmark_cloud.ipynb`, or `python experiments/caption_run.py --model llava15_7b --n_images 1000` |
| Qwen2.5-VL-7B run | CUDA GPU (T4 with 4-bit) | the same notebook with `MODEL = "qwen25vl_7b"` |
| Hidden states for the linear probe (after a run) | CUDA GPU | notebook section 8, or `python experiments/caption_hidden.py --model <key> --run <run dir>` |
| Evaluation | CPU | `python experiments/caption_eval.py --run <run dir>` |

LLaVA-1.5-7B needs about 14 GB in fp16, and 4-bit loading requires CUDA. Evaluation uses only the saved run folder, so it can run on any machine.

## Pipeline

1. **Capture** (`caption_bench/capture.py`). A single `generate()` call per image, with `output_hidden_states=True`, records:
   - the caption
   - the hidden states of every layer at the image-token positions and at each generated token
   - the logits that produced each token

   No second forward pass is made. `tests/test_capture_consistency.py` checks these tensors against a separate full forward pass on transformers 4.43 and 5.x, for LLaVA-1.5 and Qwen2.5-VL.
   - LLaVA-1.5 has 576 image tokens per image.
   - Qwen2.5-VL has one image token per merged 28×28 pixel block, so the count depends on image size (about 350–400 for a COCO image).
2. **Labelling** (`caption_bench/chair.py`). COCO objects are extracted with the CHAIR synonym list and phrase rules. Ground truth is the instance annotations plus the objects named in the five reference captions.
3. **Features** (`caption_bench/features.py`), taken at each object's first sub-token:
   - NLL and entropy
   - Internal Confidence: logit-lens probability over the image tokens, at every layer
   - for every (text layer, image layer) pair, the cosine similarity between the object token and the image patches, reduced to max, top-k mean (k = 4, 8, 16, 32), mean, std, and cosine with the mean image state
4. **Evaluation** (`caption_bench/evaluate.py`):
   - Images are split into calibration and test halves.
   - Layers, statistic, score sign and thresholds are selected on calibration images only.
   - Test-set AUROC and AUPR are reported, with 95% bootstrap intervals over images. The positive class is a hallucinated object.
   - Objects named late in a long caption are hallucinated far more often than early ones (Zhou et al., ICLR 2024), and hidden-state similarity also drifts with position. Every detector is therefore also scored within caption-position bins: deciles of the object's token index, with edges set on calibration. The reported value is the size-weighted mean AUROC over bins.
   - Detect-then-remove: CHAIR_i and CHAIR_s are reported before and after removing flagged objects, at a threshold that keeps 95% of real calibration objects.
   - Trained baseline: when `hidden/` exists, a linear probe (standardized features, L2-regularized logistic regression) is trained on the object-token hidden state. Its layer and regularization strength are chosen on an inner split of the calibration images, then it is refit on all of them. The latent statistic only selects a layer pair and a statistic, so the probe is the stronger, supervised reference.
   - Cross-fitting (`cv_folds`, default 5) repeats the procedure over folds of images, so every image is scored once as test data. Each fold's scores are converted to percentiles of the fitting folds' scores before pooling. Detectors are compared with a paired bootstrap over images, both overall and within position bins: the best latent statistic against NLL and Internal Confidence, the NLL + latent fusion against NLL alone (does the latent score add information to NLL), and, with hidden states, the latent statistic against the linear probe.

Each run folder contains:
- `run.json`: model, precision, library versions, git commit
- `captions.jsonl`: captions and labelled mentions
- `feats/<image_id>.npz`: per-object features
- `hidden/<image_id>.npz` (optional): object-token hidden states of every layer, first mentions only, from `experiments/caption_hidden.py`. That script re-reads each stored caption in one forward pass; `tests/test_capture_consistency.py` checks that this gives the same states as generation, and the evaluation compares a recomputed max-patch cosine with `feats/` and reports the difference in `eval.md`.
- `eval/`: `eval.json`, `eval.md`, `layer_map.png`, `roc_test.png`

`feats/` and `hidden/` are too large for git. The feature files of the published runs are assets of the [`caption-runs-v1` release](https://github.com/sunilsk17/LatentTokenGuard/releases/tag/caption-runs-v1); unzip one into `results/caption_runs/` to re-run its evaluation.

## Detectors

| Name | Description |
|---|---|
| NLL, Entropy | Next-token probability baselines |
| Token position | Index of the object's token in the caption; uses no model internals |
| Internal Confidence | Jiang et al., ICLR 2025; max over layers, and the best calibration layer |
| Max-patch cosine, same layer | Object-level form of the LatentTokenGuard score and of ContextualLens (Phukan et al., NAACL 2025) |
| Max-patch cosine, layer pair | Same score with text and image layers selected separately |
| Top-k mean cosine, Peakedness, Global cosine | Variants of the patch-similarity score |
| Best latent stat (at / pre) | Best statistic at the object token, or at the position that predicted it |
| Best latent stat, position-corrected | Layers and statistic chosen by within-position AUROC on calibration; the score of a real object at the same position (cubic in log position, fit on real calibration objects) is subtracted |
| Fusion | Rank average of NLL (and Internal Confidence) with the best latent statistic |
| Linear probe on hidden states | Supervised: logistic regression on the object-token hidden state of one layer (SAPLMA-style, Azaria & Mitchell, 2023); needs `hidden/` |

## Notes

- 4-bit (T4) and fp16 (A100) runs produce different hidden states. Compare detectors only within one run.
- Decoding is greedy. Qwen2.5-VL runs use its default `repetition_penalty` of 1.05 (`generate_kwargs` in the config, recorded in `run.json`). NLL, entropy and Internal Confidence use the raw logits, before any penalty.
- `feature_seconds` in `captions.jsonl` times every layer pair plus Internal Confidence. A deployed detector computes a single layer pair, so its overhead should be timed separately against `gen_seconds`.
- Feature and hidden-state arrays are excluded from git (`.gitignore`). The published runs' arrays are in the `caption-runs-v1` release.
