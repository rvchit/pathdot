# PathVQA: `accounts/fireworks/models/qwen3-vl-8b-instruct`

1000 test questions (seeded subset of the 6,719-question PathVQA test set; 499 yes/no, 501 open). 8 seeded training examples (4 yes/no, 4 open, images included) prepended as prior turns.

## Accuracy

| metric | value |
|---|---|
| yes/no accuracy | 62.1% (57.7-66.3) |
| open exact match | 1.8% (0.8-3.0) |
| open token recall (PathVQA metric) | 5.8% (4.0-7.7) |
| overall exact match | 31.9% (29.0-34.8) |
| few-shot examples | 8 |

## Cost

| metric | value |
|---|---|
| deployment GPU time for this eval | 3.5 min x 1 NVIDIA_H100_80GB |
| GPU cost (@ $8.00/GPU-h) | $0.47 ($0.47 per 1k questions) |

## Speed

| metric | value |
|---|---|
| latency p50 / p95 | 772 / 1980 ms |
| output tokens per answer | 3.2 |
| wall time (concurrency 16) | 1.0 min |
| deployment | qwen3-vl-8b-instruct-minimal (BF16) |
