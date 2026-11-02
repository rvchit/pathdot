# PathVQA: `accounts/fireworks/models/qwen3-vl-8b-instruct`

1000 test questions (seeded subset of the 6,719-question PathVQA test set; 499 yes/no, 501 open). Zero-shot.

## Accuracy

| metric | value |
|---|---|
| yes/no accuracy | 67.1% (63.1-71.3) |
| open exact match | 1.2% (0.4-2.2) |
| open token recall (PathVQA metric) | 5.8% (4.2-7.6) |
| overall exact match | 34.1% (31.2-37.1) |
| few-shot examples | 0 |

## Cost

| metric | value |
|---|---|
| deployment GPU time for this eval | 3.0 min x 1 NVIDIA_H100_80GB |
| GPU cost (@ $8.00/GPU-h) | $0.40 ($0.40 per 1k questions) |

## Speed

| metric | value |
|---|---|
| latency p50 / p95 | 305 / 452 ms |
| output tokens per answer | 2.8 |
| wall time (concurrency 16) | 0.3 min |
| deployment | qwen3-vl-8b-instruct-minimal (BF16) |
