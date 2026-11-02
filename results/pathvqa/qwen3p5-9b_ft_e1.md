# PathVQA: `accounts/rvchit/models/pathvqa-qwen3p5-9b (Qwen3.5-9B + LoRA)`

1000 test questions (seeded subset of the 6,719-question PathVQA test set; 499 yes/no, 501 open). Fine-tuned (LoRA) on 8,000 PathVQA train Q&A pairs with base64 images (test-image near-duplicates removed by perceptual hash); reasoning_effort=none. A 3-epoch run (job uz7n7vzh, metrics in ft_metrics_qwen3p5-9b_e3.jsonl) overfit: eval loss 0.570 / 0.576 / 0.645 after epochs 1/2/3, so it was not deployed.

## Accuracy

| metric | value |
|---|---|
| yes/no accuracy | 90.6% (88.0-93.0) |
| open exact match | 27.5% (23.5-31.3) |
| open token recall (PathVQA metric) | 32.5% (28.5-36.4) |
| overall exact match | 59.0% (56.1-62.2) |
| few-shot examples | 0 |

## Cost

| metric | value |
|---|---|
| deployment GPU time for this eval | 3.5 min x 1 NVIDIA_B200_180GB |
| GPU cost (@ $13.00/GPU-h) | $0.76 ($0.76 per 1k questions) |
| training | LoRA SFT rank 16, 1 epoch, lr 1e-4, 8000 examples, 14 min, ~$0.33 (669797 tokens) |

## Speed

| metric | value |
|---|---|
| latency p50 / p95 | 473 / 995 ms |
| output tokens per answer | 2.9 |
| wall time (concurrency 16) | 0.6 min |
| deployment | rft-qwen3p5-9b-v2 (live merge) (BF16) |
