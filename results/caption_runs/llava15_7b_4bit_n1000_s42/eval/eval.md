# Object hallucination detection: llava-hf/llava-1.5-7b-hf

Images: 1000 (calib 500, test 500). Object mentions: 3281 (837 hallucinated). Test: 1632 mentions, 419 hallucinated.

CHAIR on test captions: CHAIR_i 25.7, CHAIR_s 46.8

Hallucinated share by caption-position bin, earliest to latest (10 bins): 2%, 1%, 5%, 7%, 14%, 26%, 36%, 55%, 49%, 63%

| Detector | AUROC [95% CI] | AUROC within position bins [95% CI] | AUPR [95% CI] | Selected on calib | CHAIR_i after removal | Real objects kept |
|---|---|---|---|---|---|---|
| NLL | 74.6 [71.8, 76.9] | 72.5 [68.3, 76.0] | 49.9 [44.7, 55.5] | – | 20.8 | 93.6% |
| Entropy | 75.2 [72.6, 77.8] | 69.9 [65.5, 74.4] | 50.1 [45.4, 54.9] | – | 22.5 | 95.6% |
| Token position | 82.7 [80.6, 84.9] | 47.9 [44.4, 52.9] | 54.3 [49.4, 60.9] | – | 21.4 | 94.6% |
| Internal Confidence (max over layers) | 75.5 [72.7, 78.3] | 70.3 [65.1, 74.4] | 52.9 [48.0, 58.9] | – | 20.6 | 93.9% |
| Internal Confidence (calib layer) | 78.6 [76.1, 81.1] | 73.0 [69.2, 76.6] | 56.9 [51.6, 62.2] | layer=32 | 19.8 | 94.2% |
| Max-patch cosine, same layer (LTG/ContextualLens) | 87.0 [85.0, 88.9] | 77.0 [71.9, 81.5] | 68.9 [63.7, 74.2] | state=at, text_layer=24, image_layer=24, stat=max, sign=- | 16.2 | 95.1% |
| Max-patch cosine, layer pair | 87.0 [85.0, 88.9] | 77.0 [71.9, 81.5] | 68.9 [63.7, 74.2] | state=at, text_layer=24, image_layer=24, stat=max, sign=- | 16.2 | 95.1% |
| Top-k mean cosine | 88.7 [86.9, 90.5] | 79.0 [73.7, 83.9] | 71.5 [66.4, 76.5] | state=at, text_layer=21, image_layer=22, stat=top8, sign=- | 15.8 | 95.1% |
| Global cosine | 78.9 [76.3, 81.3] | 63.4 [55.4, 70.5] | 52.6 [47.6, 58.5] | state=at, text_layer=20, image_layer=22, stat=global, sign=- | 21.7 | 95.0% |
| Peakedness | 76.9 [74.1, 79.4] | 65.4 [58.5, 71.1] | 54.9 [49.5, 60.5] | state=at, text_layer=31, image_layer=26, stat=peak_top8, sign=- | 20.4 | 94.6% |
| Best latent stat (at) | 88.7 [86.9, 90.5] | 79.0 [73.7, 83.9] | 71.5 [66.4, 76.5] | state=at, text_layer=21, image_layer=22, stat=top8, sign=- | 15.8 | 95.1% |
| Best latent stat (pre) | 78.2 [75.8, 81.0] | 55.1 [46.5, 62.6] | 54.0 [49.2, 59.3] | state=pre, text_layer=26, image_layer=21, stat=top8, sign=- | 22.0 | 95.1% |
| Best latent stat, position-corrected | 78.0 [75.2, 80.6] | 79.2 [74.2, 84.0] | 50.3 [45.1, 55.3] | state=at, text_layer=22, image_layer=22, stat=top8, sign=- | 22.9 | 94.8% |
| Fusion: NLL + best latent | 87.4 [85.4, 89.2] | 80.7 [77.1, 83.7] | 70.3 [65.5, 75.2] | – | 15.4 | 93.6% |
| Fusion: NLL + IC + best latent | 88.4 [86.4, 90.3] | 83.3 [79.8, 85.8] | 72.0 [67.0, 76.9] | – | 14.6 | 94.5% |

Positive class = hallucinated object. All selections use the calibration images only; numbers are on the held-out test images. CIs: bootstrap over images. Position bins are quantiles of the object's token index on calibration mentions; the within-bin AUROC is the size-weighted mean over bins.
