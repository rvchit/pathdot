# Camelyon17 baseline: `accounts/fireworks/models/qwen3p5-9b (untrained)`

6000 tiles (WILDS hospital 2, slides 22, 23, 24, 25, 26), 3000 tumor / 3000 normal. Zero-shot, prompt v2 (one word, P(tumor) from first-token logprobs, cutoff 0.5), 224 px, temperature 0, thinking off.

## Accuracy

| metric | value |
|---|---|
| accuracy | 67.0% (95% CI 65.8%-68.2%) |
| balanced accuracy | 67.0% |
| sensitivity (tumors caught) | 95.3% |
| specificity (normals cleared) | 38.7% |
| precision | 60.9% |
| F1 | 74.3% |
| AUROC (stated confidence) | 0.837 |
| predicted tumor rate | 78.3% |
| parse / API errors | 0 / 0 |

Confusion: TP 2859, FN 141, FP 1838, TN 1162

### Per slide

| slide | n | tumor | accuracy | sensitivity | specificity |
|---|---|---|---|---|---|
| 22 | 1947 | 1494 | 88.9% | 92.2% | 77.9% |
| 23 | 1281 | 929 | 79.4% | 99.9% | 25.3% |
| 24 | 1215 | 56 | 33.6% | 82.1% | 31.2% |
| 25 | 843 | 335 | 65.8% | 100.0% | 43.3% |
| 26 | 714 | 186 | 43.6% | 93.0% | 26.1% |

## Cost (tokens)

| metric | value |
|---|---|
| input tokens | 792,000 (132/image) |
| output tokens | 16,697 (3/image) |
| GPU time | 2.1 min x 1 NVIDIA_H200_141GB |
| GPU cost (estimate @ $10.00/GPU-h) | $0.35 ($0.06 per 1k images) |
| token-price proxy | $0.73 |

## Speed

| metric | value |
|---|---|
| latency p50 / p95 / max | 421 / 745 / 17037 ms |
| wall time | 2.1 min |
| throughput | 47.62 images/s, 133 output tokens/s |
| deployment spin-up | 6.2 min |
