# Camelyon17 baseline: `probe/phikon-v2`

6000 tiles (WILDS hospital 2, slides 22, 23, 24, 25, 26), 3000 tumor / 3000 normal. Phikon-v2 features + linear classifier (C=0.001, calibrated, cutoff 0.5) trained on the 10,000-tile practice pool from slides 20, 21, 27-29; no tuning on the test set.

## Accuracy

| metric | value |
|---|---|
| accuracy | 86.0% (95% CI 85.1%-86.9%) |
| balanced accuracy | 86.0% |
| sensitivity (tumors caught) | 74.5% |
| specificity (normals cleared) | 97.5% |
| precision | 96.8% |
| F1 | 84.2% |
| AUROC (stated confidence) | 0.977 |
| predicted tumor rate | 38.5% |
| parse / API errors | 0 / 0 |

Confusion: TP 2234, FN 766, FP 74, TN 2926

### Per slide

| slide | n | tumor | accuracy | sensitivity | specificity |
|---|---|---|---|---|---|
| 22 | 1947 | 1494 | 67.1% | 57.2% | 99.8% |
| 23 | 1281 | 929 | 99.1% | 99.2% | 98.6% |
| 24 | 1215 | 56 | 98.1% | 82.1% | 98.9% |
| 25 | 843 | 335 | 93.7% | 100.0% | 89.6% |
| 26 | 714 | 186 | 84.3% | 40.9% | 99.6% |

## Cost (tokens)

| metric | value |
|---|---|
| input tokens | 0 (0/image) |
| output tokens | 0 (0/image) |
| GPU time | 2.9 min x 1 Mac GPU (MPS) |
| GPU cost (estimate @ $0.00/GPU-h) | $0.00 ($0.00 per 1k images) |
| token-price proxy | $0.00 |

## Speed

| metric | value |
|---|---|
| latency p50 / p95 / max | 29 / 29 / 29 ms |
| wall time | 2.9 min |
| throughput | 34.46 images/s, 0 output tokens/s |
