# Object hallucination detection: llava-hf/llava-1.5-13b-hf

Images: 1000 (calib 500, test 500). Object mentions: 3174 (707 hallucinated). Test: 1568 mentions, 356 hallucinated.

CHAIR on test captions: CHAIR_i 22.7, CHAIR_s 45.4

Hallucinated share by caption-position bin, earliest to latest (10 bins): 1%, 1%, 4%, 6%, 6%, 28%, 37%, 41%, 48%, 59%

| Detector | AUROC [95% CI] | AUROC within position bins [95% CI] | AUPR [95% CI] | Selected on calib | CHAIR_i after removal | Real objects kept |
|---|---|---|---|---|---|---|
| NLL | 74.1 [71.3, 77.0] | 76.1 [71.1, 79.4] | 46.8 [41.4, 53.2] | – | 18.9 | 95.1% |
| Entropy | 75.5 [72.6, 78.5] | 74.0 [69.0, 78.4] | 48.0 [42.7, 54.1] | – | 19.1 | 96.3% |
| Token position | 83.3 [81.1, 85.4] | 48.3 [44.2, 53.1] | 51.3 [46.0, 56.8] | – | 19.6 | 95.5% |
| Internal Confidence (max over layers) | 73.3 [70.3, 76.2] | 71.1 [65.3, 74.5] | 48.1 [42.6, 53.6] | – | 18.2 | 94.1% |
| Internal Confidence (calib layer) | 78.0 [75.1, 80.8] | 70.6 [66.8, 75.4] | 52.8 [47.1, 58.3] | layer=40 | 18.0 | 94.4% |
| Max-patch cosine, same layer (LTG/ContextualLens) | 86.1 [83.8, 88.1] | 79.2 [74.1, 82.4] | 65.6 [59.9, 71.1] | state=at, text_layer=26, image_layer=26, stat=max, sign=- | 15.1 | 94.6% |
| Max-patch cosine, layer pair | 86.0 [83.7, 88.0] | 79.3 [74.3, 82.5] | 65.6 [60.3, 71.0] | state=at, text_layer=25, image_layer=26, stat=max, sign=- | 14.8 | 94.9% |
| Top-k mean cosine | 88.9 [87.0, 90.6] | 80.5 [75.3, 84.0] | 66.8 [61.1, 72.3] | state=at, text_layer=25, image_layer=22, stat=top16, sign=- | 14.9 | 94.7% |
| Global cosine | 75.0 [72.1, 77.6] | 57.8 [51.3, 62.7] | 43.6 [38.6, 48.7] | state=at, text_layer=40, image_layer=28, stat=global, sign=- | 20.1 | 95.2% |
| Peakedness | 73.8 [70.7, 76.6] | 66.5 [59.7, 72.3] | 42.0 [36.7, 47.7] | state=at, text_layer=0, image_layer=36, stat=peak_top8, sign=- | 20.3 | 92.9% |
| Best latent stat (at) | 88.9 [87.0, 90.6] | 80.5 [75.3, 84.0] | 66.8 [61.1, 72.3] | state=at, text_layer=25, image_layer=22, stat=top16, sign=- | 14.9 | 94.7% |
| Best latent stat (pre) | 81.3 [78.7, 83.8] | 51.7 [46.5, 59.6] | 55.8 [50.2, 62.1] | state=pre, text_layer=40, image_layer=21, stat=top8, sign=- | 17.4 | 94.1% |
| Best latent stat, position-corrected | 71.7 [68.8, 74.8] | 79.0 [73.1, 83.1] | 39.9 [35.5, 46.0] | state=at, text_layer=31, image_layer=28, stat=top16, sign=- | 21.1 | 96.1% |
| Fusion: NLL + best latent | 87.8 [85.9, 89.7] | 83.6 [79.2, 86.1] | 66.1 [60.2, 71.8] | – | 14.4 | 94.2% |
| Fusion: NLL + IC + best latent | 89.1 [87.2, 91.1] | 85.0 [81.1, 87.4] | 69.9 [64.5, 75.5] | – | 13.0 | 94.6% |

Positive class = hallucinated object. All selections use the calibration images only; numbers are on the held-out test images. CIs: bootstrap over images. Position bins are quantiles of the object's token index on calibration mentions; the within-bin AUROC is the size-weighted mean over bins.

## Cross-fitted (5 folds over images)

Every image is scored once by detectors fit on the other folds. Mentions: 3174 (707 hallucinated).

| Detector | AUROC [95% CI] | AUROC within position bins [95% CI] | AUPR [95% CI] | Selected per fold |
|---|---|---|---|---|
| NLL | 72.3 [70.1, 74.2] | 73.4 [70.0, 76.1] | 43.4 [39.7, 47.2] | – |
| Entropy | 73.4 [71.3, 75.4] | 72.1 [68.6, 75.1] | 43.5 [39.9, 47.2] | – |
| Token position | 83.6 [82.0, 85.2] | 53.8 [47.6, 59.8] | 51.3 [47.1, 55.6] | – |
| Internal Confidence (max over layers) | 74.1 [72.0, 76.2] | 71.9 [67.9, 74.9] | 48.0 [44.2, 51.9] | – |
| Internal Confidence (calib layer) | 78.0 [76.0, 79.9] | 70.2 [66.2, 73.8] | 52.1 [48.1, 56.0] | L40 (5/5) |
| Max-patch cosine, same layer (LTG/ContextualLens) | 85.7 [84.2, 87.2] | 75.4 [69.6, 79.6] | 64.3 [60.0, 68.0] | at L26/L26 max (-) (1/5), at L28/L28 max (-) (1/5), at L27/L27 max (-) (1/5), at L22/L22 max (-) (1/5), at L25/L25 max (-) (1/5) |
| Max-patch cosine, layer pair | 85.5 [83.9, 87.0] | 75.0 [68.6, 79.5] | 64.6 [60.5, 68.4] | at L26/L27 max (-) (1/5), at L28/L28 max (-) (1/5), at L22/L29 max (-) (1/5), at L22/L22 max (-) (1/5), at L25/L25 max (-) (1/5) |
| Top-k mean cosine | 89.1 [87.9, 90.3] | 78.7 [72.8, 83.0] | 67.8 [63.6, 71.7] | at L22/L28 top8 (-) (1/5), at L22/L27 top8 (-) (1/5), at L22/L29 top8 (-) (1/5), at L23/L22 top16 (-) (1/5), at L25/L26 top8 (-) (1/5) |
| Global cosine | 75.2 [73.3, 77.2] | 59.4 [53.6, 64.1] | 43.7 [40.1, 47.8] | at L40/L28 global (-) (5/5) |
| Peakedness | 75.8 [73.8, 77.7] | 66.6 [60.1, 71.4] | 44.4 [40.7, 48.5] | at L39/L38 peak_top8 (-) (4/5), at L0/L36 peak_top8 (-) (1/5) |
| Best latent stat (at) | 89.1 [87.9, 90.3] | 78.7 [72.8, 83.0] | 67.8 [63.6, 71.7] | at L22/L28 top8 (-) (1/5), at L22/L27 top8 (-) (1/5), at L22/L29 top8 (-) (1/5), at L23/L22 top16 (-) (1/5), at L25/L26 top8 (-) (1/5) |
| Best latent stat (pre) | 81.6 [79.7, 83.5] | 56.4 [51.0, 61.5] | 55.4 [51.4, 60.0] | pre L40/L22 top4 (-) (2/5), pre L40/L38 top4 (-) (1/5), pre L40/L24 top8 (-) (1/5), pre L40/L18 top8 (-) (1/5) |
| Best latent stat, position-corrected | 78.6 [76.8, 80.3] | 76.9 [71.2, 83.1] | 46.9 [43.1, 50.9] | at L28/L28 top8 (-) (3/5), at L22/L29 top8 (-) (1/5), at L20/L39 top16 (-) (1/5) |
| Fusion: NLL + best latent | 86.9 [85.6, 88.2] | 81.9 [78.9, 84.1] | 65.3 [61.2, 68.9] | – |
| Fusion: NLL + IC + best latent | 88.4 [87.1, 89.8] | 83.1 [80.4, 85.1] | 67.9 [63.9, 71.7] | – |

Paired differences (bootstrap over images; p = share of resamples with difference ≤ 0):

| Comparison | ΔAUROC [95% CI] | p | ΔAUROC within position bins [95% CI] | p |
|---|---|---|---|---|
| Best latent stat (at) − NLL | +16.8 [+14.6, +19.2] | 0.000 | +5.3 [-0.9, +10.8] | 0.059 |
| Best latent stat (at) − Internal Confidence (calib layer) | +11.2 [+9.5, +13.1] | 0.000 | +8.5 [+1.9, +16.0] | 0.004 |
