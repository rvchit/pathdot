# Camelyon17 baseline: `accounts/ramaratnam-rishi-272/models/qwen3p5-9b-camelyon17-aug (Qwen3.5-9B + LoRA, round 2)`

6000 tiles (WILDS hospital 2, slides 22, 23, 24, 25, 26), 3000 tumor / 3000 normal. Round 2: tumor examples from slides 28 and 29 (round 1: 99% slide 28) plus random flips/rotations; prompt v2, 224 px, thinking off. Inference ran 4 rotations per tile; the dev set (1,000 held-out slide-29 tiles) chose the single (unrotated) view and cutoff 0.778, applied once to test.

## Accuracy

| metric | value |
|---|---|
| accuracy | 85.8% (95% CI 84.9%-86.7%) |
| balanced accuracy | 85.8% |
| sensitivity (tumors caught) | 75.7% |
| specificity (normals cleared) | 95.9% |
| precision | 94.9% |
| F1 | 84.2% |
| AUROC (stated confidence) | 0.930 |
| predicted tumor rate | 39.9% |
| parse / API errors | 0 / 0 |

Confusion: TP 2272, FN 728, FP 123, TN 2877

### Per slide

| slide | n | tumor | accuracy | sensitivity | specificity |
|---|---|---|---|---|---|
| 22 | 1947 | 1494 | 72.0% | 63.7% | 99.3% |
| 23 | 1281 | 929 | 96.3% | 97.0% | 94.6% |
| 24 | 1215 | 56 | 92.1% | 1.8% | 96.5% |
| 25 | 843 | 335 | 93.5% | 99.4% | 89.6% |
| 26 | 714 | 186 | 85.0% | 46.2% | 98.7% |

## Cost (tokens)

| metric | value |
|---|---|
| input tokens | 3,168,000 (528/image) |
| output tokens | 57,883 (10/image) |
| GPU time | 1.2 min x 1 NVIDIA_H200_141GB |
| GPU cost (estimate @ $10.00/GPU-h) | $0.21 ($0.03 per 1k images) |
| token-price proxy | $2.90 |

## Speed

| metric | value |
|---|---|
| latency p50 / p95 / max | 506 / 971 / 3492 ms |
| wall time | 4.9 min |
| throughput | 20.27 images/s, 196 output tokens/s |
| deployment spin-up | 5.2 min |
