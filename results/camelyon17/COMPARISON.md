# Camelyon17 (WILDS hospital 2): baseline vs fine-tuned Qwen

Same frozen eval set for every row: 6,000 tiles (3,000 tumor / 3,000 normal) from slides 22-26.
Fine-tuning used slides 20, 21, 27, 28 (7,607 tiles); the decision cutoff was chosen on slide 29
(2,393 tiles). No eval tile was used for training, prompt design or cutoff selection.

| | Qwen2.5-VL-32B baseline | Qwen3.5-9B untrained (v1 prompt) | Qwen3.5-9B untrained (v2 prompt) | **Qwen3.5-9B fine-tuned (v2)** | Phikon-v2 + linear probe (reference, not Qwen) |
|---|---|---|---|---|---|
| accuracy | 52.3% | 73.9% | 67.0% | **85.5%** (84.5-86.3) | 86.0% |
| sensitivity (tumors caught) | 16.6% | 95.9% | 95.3% | 78.1% | 74.5% |
| specificity (normals cleared) | 88.0% | 51.9% | 38.7% | 92.8% | 97.5% |
| precision | 58.0% | 66.6% | 60.9% | 91.5% | 96.8% |
| AUROC | 0.534 | 0.752 | 0.837 | **0.943** | 0.977 |
| input / output tokens per image | 487 / 48 | 430 / 43 | 132 / 3 | **132 / 2** | - |
| GPU-minutes for the 6k run | 7.3 (H200) | 3.2 (H200) | 2.1 (H200) | **1.4** (H200) | 2.9 (Mac MPS) |
| GPU $ per 1k images (est. @ $10/GPU-h) | $0.20 | $0.09 | $0.06 | **$0.04** | $0 (local) |
| throughput (images/s) | 13.7 | 30.9 | 47.6 | **71.4** | 34.5 |

Prompts: v1 = JSON label + confidence + reason at 448 px; v2 = one word ("tumor"/"normal") at
224 px with P(tumor) read from first-token logprobs. Untrained rows use cutoff 0.5; the fine-tuned
row uses cutoff 0.107 from the slide-29 dev set (at 0.5 it scores 79.1%, sensitivity 58.9%).

Paired McNemar tests (fine-tuned vs each Qwen row) all give p < 1e-50.

One-off costs: fine-tuning job `iap42xel` (LoRA rank 16, 2 epochs, ~0.93M tokens/epoch) took
20 min and ~$0.53. Deployment spin-up is ~4-6 min per session and is not in the per-run GPU
minutes above. GPU dollars use a deliberately high $10/GPU-hour; check Fireworks billing for
actual charges.

Weak spots of the fine-tuned model: slides 24 and 26 (tumor sensitivity 23% and 18%), where
tumor patches are rare (56 and 186 tiles); the Phikon probe also misses most tumors on slide 26
(41%) but not on slide 24 (82%).

## Round 2 (no meaningful change)

Training set rebuilt so tumor examples come from slides 28 and 29 (1,826 / 1,827) instead of 99%
slide 28, with one random flip/rotation per image; same size (7,600), same LoRA recipe (job
`i3nc1m5s`, 21 min, ~$0.53). Inference ran 4 rotations per tile; the dev set (1,000 slide-29 tiles
held out of training) chose the single unrotated view and cutoff 0.778, applied once to test.

| | round 1 | round 2 |
|---|---|---|
| accuracy | 85.5% | 85.8% (84.9-86.7) |
| sensitivity / specificity | 78.1% / 92.8% | 75.7% / 95.9% |
| AUROC | 0.943 | 0.930 (4-view average: 0.937) |

Paired McNemar round 2 vs round 1: 335 vs 313 discordant tiles, p = 0.41, i.e. no detectable
difference. Likely cause: slide 29 is now both in training and the dev set, so the model is very
confident on dev (AUROC 0.998) and the dev-chosen cutoff (0.78) is too strict for the unseen test
slides; it trades tumor sensitivity for specificity. Slide 24 tumor sensitivity fell to 2%
(1 of 56). Averaging over rotations did not help on dev (97.8% vs 97.9%).

Per-model reports: `finetuned_qwen3p5-9b_aug_v2.md`, `baseline_qwen2p5-vl-32b.md`, `baseline_qwen3p5-9b_zeroshot_v1.md`,
`baseline_qwen3p5-9b_zeroshot_v2.md`, `finetuned_qwen3p5-9b_v2.md`, `baseline_phikon-v2_probe.md`.
