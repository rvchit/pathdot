# Camelyon17 baseline: `accounts/ramaratnam-rishi-272/models/qwen3p5-9b-camelyon17-v2 (Qwen3.5-9B + LoRA)`

6000 tiles (WILDS hospital 2, slides 22, 23, 24, 25, 26), 3000 tumor / 3000 normal. Fine-tuned (LoRA) on 7,607 tiles from slides 20, 21, 27, 28; prompt v2 (one word, P(tumor) from logprobs), 224 px, thinking off; cutoff 0.107 chosen on the 2,393-tile slide-29 dev set (never on test).

## Accuracy

| metric | value |
|---|---|
| accuracy | 85.5% (95% CI 84.5%-86.3%) |
| balanced accuracy | 85.5% |
| sensitivity (tumors caught) | 78.1% |
| specificity (normals cleared) | 92.8% |
| precision | 91.5% |
| F1 | 84.3% |
| AUROC (stated confidence) | 0.943 |
| predicted tumor rate | 42.7% |
| parse / API errors | 0 / 0 |

Confusion: TP 2344, FN 656, FP 217, TN 2783

### Per slide

| slide | n | tumor | accuracy | sensitivity | specificity |
|---|---|---|---|---|---|
| 22 | 1947 | 1494 | 79.6% | 74.0% | 97.8% |
| 23 | 1281 | 929 | 93.2% | 92.2% | 95.7% |
| 24 | 1215 | 56 | 87.0% | 23.2% | 90.1% |
| 25 | 843 | 335 | 92.3% | 99.7% | 87.4% |
| 26 | 714 | 186 | 76.9% | 18.3% | 97.5% |

## Cost (tokens)

| metric | value |
|---|---|
| input tokens | 792,000 (132/image) |
| output tokens | 13,789 (2/image) |
| GPU time | 1.4 min x 1 NVIDIA_H200_141GB |
| GPU cost (estimate @ $10.00/GPU-h) | $0.23 ($0.04 per 1k images) |
| token-price proxy | $0.73 |

## Speed

| metric | value |
|---|---|
| latency p50 / p95 / max | 405 / 561 / 2359 ms |
| wall time | 1.4 min |
| throughput | 71.43 images/s, 164 output tokens/s |
| deployment spin-up | 5.2 min |
