# Earlier experiments: POPE

The first version of LatentTokenGuard was developed on POPE yes/no questions ("Is there a
<object> in the image?"). This page records those results and the analysis that led to the
free-form caption benchmark described in the [README](README.md) and
[docs/CAPTION_BENCHMARK.md](docs/CAPTION_BENCHMARK.md), which is the main evaluation.

## What was measured

Each answer was scored by the mean, over its content tokens, of the best cosine similarity between
the token's last-layer hidden state and any image patch (`hidden_layer_index: -1` in `config.yaml`;
the Phi-3.5-Vision run predates the last edit of that file, so its layer setting is not recorded). The positive label is the POPE ground truth
"no", i.e. the queried object is absent from the image. The task is therefore to tell absent
objects from present ones from the model's internal states.

## Results (POPE-adversarial, 3,000 questions, 1,500 absent)

| Model | AUROC, object absent | Accuracy at τ | F1 at τ | τ | Latency |
| :--- | :---: | :---: | :---: | :---: | :---: |
| LLaVA-1.5-7B | 78.3 | 69.67% | 0.7415 | 0.150 | ~747 ms |
| Phi-3.5-Vision | 50.8 | 51.70% | 0.6727 | 0.090 | ~599 ms |

τ was chosen to maximize F1 on the same 3,000 questions, so accuracy and F1 are optimistic.
Latency is generation plus scoring on one NVIDIA A100 (40 GB). LLaVA-1.5-7B used 4-bit NF4 weights
and Phi-3.5-Vision bfloat16 weights.

Two reference points, computed from the same result files:

| | LLaVA-1.5-7B | Phi-3.5-Vision |
| :--- | :---: | :---: |
| The model's own yes/no answer as a detector of absent objects (AUROC) | 79.8 | 83.8 |
| Actual hallucinations: the model answered "yes" to an absent object | 396 | 137 |
| AUROC of the score for those hallucinations, among all "yes" answers | 61.2 | 53.0 |
| AUROC of the score for predicting that the model answered "no" | 91.5 | 48.0 |

On POPE the model's own yes/no answer is itself a strong signal for whether the object is present,
and the answers that are actual hallucinations are few. The benchmark therefore cannot show what a
detector adds beyond the answer. This motivated the free-form caption benchmark, in which every
object the model names is checked against the image annotations and detectors are compared at
matched caption position.

## Files

- `results/pope_adversarial_llava_alignment_only.json`, `results/pope_adversarial_phi3_alignment_only.json`:
  per-question answers, scores, labels and latencies, with the summary metrics at the default threshold
- Replication:

```bash
python main.py --model llava --dataset pope --split adversarial --threshold 0.150 --mode alignment_only
```

Environment: `torch`, `transformers==4.43.0`, `bitsandbytes` (4-bit NF4, LLaVA only), `spacy` with
`en_core_web_sm` for content-token selection.
