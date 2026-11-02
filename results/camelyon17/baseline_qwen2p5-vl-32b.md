# Camelyon17 baseline: `accounts/fireworks/models/qwen2p5-vl-32b-instruct`

6000 tiles (WILDS hospital 2, slides 22, 23, 24, 25, 26), 3000 tumor / 3000 normal. Zero-shot, prompt v1, temperature 0.

## Accuracy

| metric | value |
|---|---|
| accuracy | 52.3% (95% CI 51.0%-53.5%) |
| balanced accuracy | 52.3% |
| sensitivity (tumors caught) | 16.6% |
| specificity (normals cleared) | 88.0% |
| precision | 58.0% |
| F1 | 25.8% |
| AUROC (stated confidence) | 0.534 |
| predicted tumor rate | 14.3% |
| parse / API errors | 0 / 0 |

Confusion: TP 498, FN 2502, FP 361, TN 2639

### Per slide

| slide | n | tumor | accuracy | sensitivity | specificity |
|---|---|---|---|---|---|
| 22 | 1947 | 1494 | 34.9% | 18.2% | 89.8% |
| 23 | 1281 | 929 | 33.9% | 13.6% | 87.5% |
| 24 | 1215 | 56 | 85.1% | 25.0% | 88.0% |
| 25 | 843 | 335 | 60.9% | 16.7% | 90.0% |
| 26 | 714 | 186 | 66.8% | 16.1% | 84.7% |

## Cost (tokens)

| metric | value |
|---|---|
| input tokens | 2,922,000 (487/image) |
| output tokens | 290,395 (48/image) |
| GPU time | 12.6 min x 1 NVIDIA_H200_141GB |
| GPU cost (estimate @ $10.00/GPU-h) | $2.10 ($0.35 per 1k images) |
| token-price proxy | $2.89 |

## Speed

| metric | value |
|---|---|
| latency p50 / p95 / max | 2257 / 2676 / 9456 ms |
| wall time | 7.3 min |
| throughput | 13.70 images/s, 663 output tokens/s |
| deployment spin-up | 3.7 min |
