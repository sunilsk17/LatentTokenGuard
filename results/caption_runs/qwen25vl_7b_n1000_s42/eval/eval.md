# Object hallucination detection: Qwen/Qwen2.5-VL-7B-Instruct

Images: 1000 (calib 500, test 500). Object mentions: 2570 (407 hallucinated). Test: 1261 mentions, 194 hallucinated.

CHAIR on test captions: CHAIR_i 15.4, CHAIR_s 30.2

Hallucinated share by caption-position bin, earliest to latest (10 bins): 0%, 2%, 3%, 9%, 20%, 21%, 18%, 31%, 30%, 29%

| Detector | AUROC [95% CI] | AUROC within position bins [95% CI] | AUPR [95% CI] | Selected on calib | CHAIR_i after removal | Real objects kept |
|---|---|---|---|---|---|---|
| NLL | 65.9 [61.1, 70.5] | 61.9 [55.9, 67.9] | 28.3 [22.8, 35.6] | – | 13.6 | 93.8% |
| Entropy | 66.9 [61.9, 71.5] | 62.5 [56.2, 68.3] | 28.5 [22.9, 36.1] | – | 13.6 | 95.2% |
| Token position | 73.6 [70.2, 76.6] | 49.6 [44.0, 56.3] | 26.8 [22.8, 32.6] | – | 14.8 | 96.1% |
| Internal Confidence (max over layers) | 66.2 [62.3, 69.8] | 58.3 [52.9, 65.0] | 26.5 [22.1, 32.2] | – | 14.1 | 95.7% |
| Internal Confidence (calib layer) | 66.6 [62.8, 70.1] | 58.5 [52.7, 65.4] | 25.6 [21.4, 31.0] | layer=28 | 14.4 | 96.0% |
| Max-patch cosine, same layer (LTG/ContextualLens) | 83.2 [80.2, 86.2] | 78.6 [73.6, 83.4] | 49.3 [41.7, 56.7] | state=at, text_layer=16, image_layer=16, stat=max, sign=- | 11.2 | 96.1% |
| Max-patch cosine, layer pair | 86.6 [84.0, 89.1] | 81.1 [75.2, 85.8] | 56.5 [48.4, 64.1] | state=at, text_layer=20, image_layer=13, stat=max, sign=- | 9.4 | 95.5% |
| Top-k mean cosine | 87.9 [85.7, 90.1] | 83.9 [79.7, 87.6] | 55.4 [47.9, 62.5] | state=at, text_layer=16, image_layer=13, stat=top8, sign=- | 10.6 | 95.3% |
| Global cosine | 78.1 [75.3, 81.0] | 69.9 [63.2, 75.3] | 37.8 [32.0, 44.0] | state=at, text_layer=20, image_layer=27, stat=global, sign=- | 13.5 | 95.9% |
| Peakedness | 78.3 [74.8, 82.0] | 75.6 [68.8, 80.1] | 39.0 [32.6, 47.9] | state=at, text_layer=6, image_layer=13, stat=peak_top8, sign=- | 11.2 | 92.8% |
| Best latent stat (at) | 87.9 [85.7, 90.1] | 83.9 [79.7, 87.6] | 55.4 [47.9, 62.5] | state=at, text_layer=16, image_layer=13, stat=top8, sign=- | 10.6 | 95.3% |
| Best latent stat (pre) | 73.0 [69.2, 76.6] | 63.0 [55.9, 69.8] | 32.1 [26.6, 38.8] | state=pre, text_layer=18, image_layer=16, stat=top16, sign=- | 12.8 | 92.8% |
| Best latent stat, position-corrected | 83.2 [80.1, 86.2] | 83.2 [78.2, 87.0] | 47.1 [39.9, 54.3] | state=at, text_layer=17, image_layer=13, stat=top4, sign=- | 12.0 | 95.4% |
| Fusion: NLL + best latent | 82.4 [79.3, 85.3] | 78.1 [74.1, 81.5] | 47.1 [39.3, 55.6] | – | 10.9 | 94.6% |
| Fusion: NLL + IC + best latent | 81.0 [77.9, 83.8] | 74.5 [70.5, 78.6] | 45.0 [37.3, 52.9] | – | 12.0 | 94.2% |

Positive class = hallucinated object. All selections use the calibration images only; numbers are on the held-out test images. CIs: bootstrap over images. Position bins are quantiles of the object's token index on calibration mentions; the within-bin AUROC is the size-weighted mean over bins.

## Cross-fitted (5 folds over images)

Every image is scored once by detectors fit on the other folds. Mentions: 2570 (407 hallucinated).

| Detector | AUROC [95% CI] | AUROC within position bins [95% CI] | AUPR [95% CI] | Selected per fold |
|---|---|---|---|---|
| NLL | 65.4 [62.1, 68.4] | 60.2 [54.7, 65.1] | 28.5 [24.4, 32.9] | – |
| Entropy | 65.8 [62.6, 68.7] | 59.7 [54.6, 64.2] | 28.9 [25.0, 33.1] | – |
| Token position | 74.2 [72.0, 76.5] | 49.2 [45.1, 55.5] | 29.2 [25.8, 33.2] | – |
| Internal Confidence (max over layers) | 67.4 [64.6, 70.5] | 57.3 [53.6, 62.4] | 27.5 [24.3, 32.0] | – |
| Internal Confidence (calib layer) | 68.1 [65.4, 70.9] | 57.9 [53.7, 63.4] | 26.8 [23.7, 30.9] | L28 (5/5) |
| Max-patch cosine, same layer (LTG/ContextualLens) | 84.1 [82.2, 86.1] | 79.0 [75.5, 82.5] | 53.4 [48.8, 58.8] | at L20/L20 max (-) (3/5), at L13/L13 max (-) (2/5) |
| Max-patch cosine, layer pair | 86.3 [84.6, 88.1] | 81.5 [78.2, 84.9] | 57.3 [52.4, 62.6] | at L20/L13 max (-) (3/5), at L20/L11 max (-) (2/5) |
| Top-k mean cosine | 87.4 [85.9, 89.1] | 82.8 [79.8, 85.8] | 57.6 [52.7, 62.6] | at L16/L13 top8 (-) (3/5), at L20/L12 top8 (-) (1/5), at L20/L13 top4 (-) (1/5) |
| Global cosine | 77.3 [75.0, 79.4] | 69.1 [64.4, 73.5] | 38.8 [34.5, 43.6] | at L20/L27 global (-) (5/5) |
| Peakedness | 77.9 [75.6, 80.1] | 75.7 [71.6, 78.4] | 37.3 [33.3, 42.4] | at L5/L20 peak_top8 (-) (2/5), at L1/L23 peak_top8 (-) (1/5), at L6/L20 peak_top8 (-) (1/5), at L7/L13 peak_top8 (-) (1/5) |
| Best latent stat (at) | 87.4 [85.9, 89.1] | 82.8 [79.8, 85.8] | 57.6 [52.7, 62.6] | at L16/L13 top8 (-) (3/5), at L20/L12 top8 (-) (1/5), at L20/L13 top4 (-) (1/5) |
| Best latent stat (pre) | 75.7 [73.1, 78.0] | 62.0 [58.5, 67.0] | 35.2 [30.6, 40.4] | pre L21/L1 global (+) (5/5) |
| Best latent stat, position-corrected | 82.6 [80.6, 84.5] | 83.5 [80.7, 86.3] | 45.8 [41.3, 51.0] | at L17/L13 top8 (-) (3/5), at L16/L13 top8 (-) (1/5), at L17/L13 top4 (-) (1/5) |
| Fusion: NLL + best latent | 81.7 [79.7, 83.7] | 76.0 [72.4, 79.0] | 47.6 [42.3, 52.8] | – |
| Fusion: NLL + IC + best latent | 81.0 [78.8, 83.1] | 72.8 [68.9, 76.7] | 47.0 [42.2, 52.1] | – |

Paired differences (bootstrap over images; p = share of resamples with difference ≤ 0):

| Comparison | ΔAUROC [95% CI] | p | ΔAUROC within position bins [95% CI] | p |
|---|---|---|---|---|
| Best latent stat (at) − NLL | +22.0 [+18.7, +25.3] | 0.000 | +22.7 [+16.0, +29.5] | 0.000 |
| Best latent stat (at) − Internal Confidence (calib layer) | +19.3 [+16.5, +21.9] | 0.000 | +24.9 [+18.8, +30.3] | 0.000 |
| Fusion: NLL + best latent − NLL | +16.3 [+14.5, +18.1] | 0.000 | +15.8 [+11.7, +20.3] | 0.000 |
