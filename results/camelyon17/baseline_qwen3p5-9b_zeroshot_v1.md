# Camelyon17 baseline: `accounts/fireworks/models/qwen3p5-9b (untrained)`

6000 tiles (WILDS hospital 2, slides 22, 23, 24, 25, 26), 3000 tumor / 3000 normal. Zero-shot, prompt v1 (JSON label+confidence+reason), 448 px, temperature 0, thinking off.

## Accuracy

| metric | value |
|---|---|
| accuracy | 73.9% (95% CI 72.8%-75.0%) |
| balanced accuracy | 73.9% |
| sensitivity (tumors caught) | 95.9% |
| specificity (normals cleared) | 51.9% |
| precision | 66.6% |
| F1 | 78.6% |
| AUROC (stated confidence) | 0.752 |
| predicted tumor rate | 72.0% |
| parse / API errors | 0 / 0 |

Confusion: TP 2877, FN 123, FP 1444, TN 1556

### Per slide

| slide | n | tumor | accuracy | sensitivity | specificity |
|---|---|---|---|---|---|
| 22 | 1947 | 1494 | 90.0% | 93.3% | 79.2% |
| 23 | 1281 | 929 | 85.6% | 99.9% | 47.7% |
| 24 | 1215 | 56 | 52.4% | 82.1% | 51.0% |
| 25 | 843 | 335 | 71.5% | 100.0% | 52.8% |
| 26 | 714 | 186 | 48.2% | 93.5% | 32.2% |

## Cost (tokens)

| metric | value |
|---|---|
| input tokens | 2,580,000 (430/image) |
| output tokens | 258,064 (43/image) |
| GPU time | 3.2 min x 1 NVIDIA_H200_141GB |
| GPU cost (estimate @ $10.00/GPU-h) | $0.54 ($0.09 per 1k images) |
| token-price proxy | $2.55 |

## Speed

| metric | value |
|---|---|
| latency p50 / p95 / max | 809 / 1249 / 12283 ms |
| wall time | 3.2 min |
| throughput | 30.93 images/s, 1330 output tokens/s |
| deployment spin-up | 6.2 min |
