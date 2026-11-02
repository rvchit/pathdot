# PathVQA: `accounts/fireworks/models/qwen3p5-9b`

1000 test questions (seeded subset of the 6,719-question PathVQA test set; 499 yes/no, 501 open). Zero-shot, reasoning_effort=none (thinking off).

## Accuracy

| metric | value |
|---|---|
| yes/no accuracy | 75.2% (71.1-79.0) |
| open exact match | 5.2% (3.4-7.2) |
| open token recall (PathVQA metric) | 7.6% (5.6-9.8) |
| overall exact match | 40.1% (37.2-43.1) |
| few-shot examples | 0 |

## Cost

| metric | value |
|---|---|
| deployment GPU time for this eval | 3.0 min x 1 NVIDIA_B200_180GB |
| GPU cost (@ $13.00/GPU-h) | $0.65 ($0.65 per 1k questions) |

## Speed

| metric | value |
|---|---|
| latency p50 / p95 | 312 / 509 ms |
| output tokens per answer | 3.1 |
| wall time (concurrency 16) | 0.3 min |
| deployment | rft-qwen3p5-9b-v2 (BF16) |
