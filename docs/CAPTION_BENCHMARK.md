# Caption benchmark: object hallucination detection in free-form captions

The model describes COCO images in free form. Each object it names is labelled real or hallucinated with the CHAIR protocol (Rohrbach et al., 2018), and every detector assigns a score to each object. Because the captions contain no yes/no answers, a detector cannot succeed by echoing the model's own answer, which is possible on POPE.

## Running

| Step | Hardware | Command |
|---|---|---|
| Tests | CPU | `python -m pytest tests/test_caption_bench.py tests/test_capture_consistency.py` |
| Smoke test (random-weight LLaVA-1.5, 17 MB) | CPU | `python experiments/make_tiny_llava.py` then `python experiments/caption_run.py --model tiny_llava15 --n_images 3` |
| LLaVA-1.5-7B run | CUDA GPU (T4 with 4-bit, or A100 with fp16) | `notebooks/caption_benchmark_cloud.ipynb`, or `python experiments/caption_run.py --model llava15_7b --n_images 1000` |
| Evaluation | CPU | `python experiments/caption_eval.py --run <run dir>` |

LLaVA-1.5-7B needs about 14 GB in fp16, and 4-bit loading requires CUDA. Evaluation uses only the saved run folder, so it can run on any machine.

## Pipeline

1. **Capture** (`caption_bench/capture.py`). A single `generate()` call per image, with `output_hidden_states=True`, records:
   - the caption
   - the hidden states of every layer at the image-token positions and at each generated token
   - the logits that produced each token

   No second forward pass is made. `tests/test_capture_consistency.py` checks these tensors against a separate full forward pass on transformers 4.43 and 5.x.
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
   - Cross-fitting (`cv_folds`, default 5) repeats the procedure over folds of images, so every image is scored once as test data. Each fold's scores are converted to percentiles of the fitting folds' scores before pooling. Detectors are compared with a paired bootstrap over images, both overall and within position bins.

Each run folder contains:
- `run.json`: model, precision, library versions, git commit
- `captions.jsonl`: captions and labelled mentions
- `feats/<image_id>.npz`: per-object features
- `eval/`: `eval.json`, `eval.md`, `layer_map.png`, `roc_test.png`

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

## Notes

- 4-bit (T4) and fp16 (A100) runs produce different hidden states. Compare detectors only within one run.
- `feature_seconds` in `captions.jsonl` times every layer pair plus Internal Confidence. A deployed detector computes a single layer pair, so its overhead should be timed separately against `gen_seconds`.
- Feature arrays are excluded from git (`.gitignore`). Publish them with the release that accompanies the paper.
