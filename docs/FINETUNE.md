# Fine-tuning the VLM on Fireworks (vision LoRA SFT)

Slice 5 of `docs/CONTRACTS.md`. Code: `pathfrontier/finetune.py`, `scripts/prepare_sft.py`,
`scripts/launch_finetune.py`, `scripts/deploy_finetuned.py`, tests in `tests/test_finetune.py`.

Everything below was verified against the live Fireworks docs on **2026-10-29/30** (URLs at the
end). Fireworks moves fast; if a command errors, re-check the linked page first.

## TL;DR (hackathon day)

```bash
# 0. credentials (https://fireworks.ai/account/api-keys ; `firectl whoami` prints the account id)
export FIREWORKS_API_KEY=fw_...  FIREWORKS_ACCOUNT_ID=<acct>

# 1. 2,000 balanced PCam tiles (+200 eval) -> data/sft_pcam_train.jsonl / data/sft_pcam_eval.jsonl
.venv/bin/python scripts/prepare_sft.py --dataset pcam --n 2000                 # ~1-2 min, ~60-90 MB

# 2. upload + LoRA SFT + wait + deploy, prints the FIREWORKS_FT_MODEL line at the end
.venv/bin/python scripts/launch_finetune.py --dataset-file data/sft_pcam_train.jsonl \
    --eval-file data/sft_pcam_eval.jsonl --epochs 2 --lora-rank 16 --wait --deploy

# 3. evaluate the fine-tuned model with the teammate's script (stage="vlm_ft")
export FIREWORKS_FT_MODEL=accounts/<acct>/models/<name>        # printed by step 2
.venv/bin/python scripts/run_vlm_ft.py --dataset pcam --split test --limit 500
.venv/bin/python scripts/eval.py results/preds_*.jsonl

# 4. stop the GPU meter
firectl deployment delete <deployment-id>                       # also printed by step 2
```

No API key? `launch_finetune.py` validates the JSONL and prints the exact `firectl` recipe instead
of failing (`--dry-run` forces that even with a key).

## What Fireworks vision SFT looks like today (confirmed facts)

| Topic | Fact | Source |
|---|---|---|
| Dataset format | Ordinary chat `messages` JSONL (OpenAI style). Image parts go on `user` messages, inline as `data:image/<type>;base64,...`; three accepted shapes: `{"type":"image_url","image_url":{"url":...}}`, `{"type":"image_url","image_url":"data:..."}`, `{"type":"image","image":"data:..."}`. `system`/`assistant` turns must not contain images. Raw http(s) image URLs are not supported in training data; nothing is uploaded separately. | [fine-tuning-models#vision-training](https://docs.fireworks.ai/fine-tuning/fine-tuning-models#vision-training), [training-api/vision-inputs](https://docs.fireworks.ai/fine-tuning/training-api/vision-inputs) |
| Dataset limits | 3 .. 3,000,000 lines. Single-request upload (`:upload`, multipart `file`) up to 150 MB; bigger files use `:getUploadEndpoint` (signed URL) + `:validateUpload`. Examples longer than the context cap (>= 32k on most models) are truncated. | [upload-dataset-files](https://docs.fireworks.ai/api-reference/upload-dataset-files), [get-dataset-upload-endpoint](https://docs.fireworks.ai/api-reference/get-dataset-upload-endpoint), [validate-dataset-upload](https://docs.fireworks.ai/api-reference/validate-dataset-upload) |
| REST: dataset | `POST /v1/accounts/{acct}/datasets` body `{"datasetId": ..., "dataset": {"userUploaded": {}, "exampleCount": "N"}}` then `POST /v1/accounts/{acct}/datasets/{id}:upload`; poll `GET .../datasets/{id}` until `state == "READY"`. | [create-dataset](https://docs.fireworks.ai/api-reference/create-dataset) |
| REST: job | `POST /v1/accounts/{acct}/supervisedFineTuningJobs` body: `dataset` (required, `accounts/{acct}/datasets/{id}`), `baseModel`, `outputModel`, `displayName`, `epochs`, `learningRate`, `loraRank`, `evaluationDataset`, `batchSizeSamples`, `maxContextLength`, `lrScheduler`... Response `name = accounts/{acct}/supervisedFineTuningJobs/{id}`, `state` in `JOB_STATE_{CREATING,PENDING,VALIDATING,RUNNING,WRITING_RESULTS,COMPLETED,FAILED,CANCELLED,EXPIRED,EARLY_STOPPED,...}`, `jobProgress.percent`, `estimatedCost`. `GET .../supervisedFineTuningJobs/{id}` to poll. | [create-sftj](https://docs.fireworks.ai/api-reference/create-supervised-fine-tuning-job), [get-sftj](https://docs.fireworks.ai/api-reference/get-supervised-fine-tuning-job) |
| LoRA rank | **Must be set explicitly over REST / Python SDK: omitted or 0 means full-parameter training.** `firectl` defaults to 8; max 128 (shape dependent). | [fine-tuning-models#lora-rank](https://docs.fireworks.ai/fine-tuning/fine-tuning-models#lora-rank) |
| Epochs / LR / batch | Epochs default 1 (integer). Learning rate is auto-selected per model when omitted (docs recommend leaving it). `--batch-size-samples` default 32. If no evaluation dataset is given, Fireworks carves one out of the training set. | [fine-tuning-models](https://docs.fireworks.ai/fine-tuning/fine-tuning-models) |
| firectl | Resource-first syntax: `firectl dataset create <ID> <file>`, `firectl sftj create --base-model <MODEL_ID> --dataset <ID> --output-model <NAME> [--epochs N --lora-rank R --learning-rate LR --evaluation-dataset <ID>]`, `firectl sftj get <JOB_ID>`, `firectl deployment create <MODEL> --deployment-shape default [--wait]`, `firectl deployment delete <ID>`. Install: `brew tap fw-ai/firectl && brew install firectl`, then `firectl signin`. (The old verb-first `firectl create sftj` form in older blog posts is superseded.) | [dataset-create](https://docs.fireworks.ai/tools-sdks/firectl/commands/dataset-create), [sftj-create](https://docs.fireworks.ai/tools-sdks/firectl/commands/supervised-fine-tuning-job-create), [deployment-create](https://docs.fireworks.ai/tools-sdks/firectl/commands/deployment-create), [firectl](https://docs.fireworks.ai/tools-sdks/firectl/firectl) |
| Serving the result | **On-demand (dedicated) deployment is the only way to serve a trained LoRA**; "Serverless deployment is not supported for LoRA models." Default = live-merge (LoRA merged into the base at deploy time, base-model speed, one adapter per deployment). Multi-LoRA (`--enable-addons` + `firectl load-lora`) exists but needs a BF16 shape. REST: `POST /v1/accounts/{acct}/deployments?deploymentId=<id>` body `{"baseModel": "accounts/{acct}/models/<name>", "deploymentShape": "default", ...}`; poll `GET .../deployments/{id}` until `state == "READY"` (`CREATING/READY/FAILED`). `autoscalingPolicy.scaleToZeroWindow` >= 5 min when `minReplicaCount` = 0. | [deploying-loras](https://docs.fireworks.ai/fine-tuning/deploying-loras), [create-deployment](https://docs.fireworks.ai/api-reference/create-deployment), [serving-fees FAQ](https://docs.fireworks.ai/faq-new/billing-pricing/are-there-extra-fees-for-serving-fine-tuned-models) |
| Inference model id | `accounts/<acct>/models/<output-model>` on the normal OpenAI-compatible endpoint `https://api.fireworks.ai/inference/v1/chat/completions` (multi-LoRA: `accounts/<acct>/models/<name>#accounts/<acct>/deployments/<dep>`). This is exactly what `scripts/run_vlm_ft.py` / `vlm_client` expect in `FIREWORKS_FT_MODEL`. | [deploying-loras](https://docs.fireworks.ai/fine-tuning/deploying-loras) |
| Training price | Managed LoRA SFT per 1M training tokens: **$0.50 (<= 16B)**, $3.00 (16.1-80B), $6.00 (80-300B), $10 (>300B); full-parameter is 2x, DPO 2x again. Serving a fine-tuned model costs the same as the base model; on-demand GPUs: H100/H200 $8/h ($0.134/min), B200 $13/h. | [fireworks.ai/pricing](https://fireworks.ai/pricing) |
| Image tokens | Model/processor dependent. Fireworks' table for Qwen2.5-VL: 336x336 -> 144 tokens (28 px per token side). Qwen3-family encoders (16 px patch, 2x2 merge) are ~32 px per token, so a 448x448 tile is ~196-256 tokens. Inference limit: 120 images per request, <10 MB base64 total. | [tokens-per-image FAQ](https://docs.fireworks.ai/faq-new/billing-pricing/how-many-tokens-per-image), [vision models guide](https://docs.fireworks.ai/guides/querying-vision-language-models) |

### Which base models can be vision-fine-tuned (managed SFT) right now

The docs' model matrix (https://docs.fireworks.ai/fine-tuning/models) is rendered from an embedded
catalogue; we parsed it on 2026-10-29 and cross-checked the public model pages
(`https://fireworks.ai/models/fireworks/<id>`, which state "Supporting image input" and
"Fine-tuning" support). Important findings:

* **Qwen2.5-VL (3B/7B/32B/72B) is no longer in the fine-tuning catalogue at all**, even though the
  July-2025 VLM launch blog and the old "fine-tuning VLMs" guide name it. Its model pages now say
  "Fine-tuning: Not Supported". Do not plan on `qwen2p5-vl-*`.
* **Qwen3-VL-8B-Instruct** has LoRA training shapes but `managedSft=false` on all of them: it is
  only trainable through the write-your-own-loop **Training API**, not `firectl sftj`. Qwen3-VL-32B
  / 235B-A22B pages say "Fine-tuning: Not Supported". Qwen3-VL-30B-A3B says "Supported" on its page
  but has no managed shape in the catalogue.
* Managed LoRA SFT (`managedSft=true` on a LoRA shape) **and** image input on the model page:

| Base model id | Params | LoRA shape (smallest) | Train $/1M tok | Notes |
|---|---|---|---|---|
| `accounts/fireworks/models/qwen3p5-9b` | 9.4B dense | `qwen3p5-9b-65k-lora` (2 x B200) | **0.50** | **default** - Qwen3.5 native VLM ("Causal Language Model with Vision Encoder") |
| `accounts/fireworks/models/qwen3p5-27b` | 27.3B dense | `qwen3p5-27b-64k-lora` (1 x B200) | 3.00 | quality upgrade |
| `accounts/fireworks/models/qwen3p6-35b-a3b` | 35B MoE (3B active) | `qwen3p6-35b-a3b-256k-lora` (4 x B200) | 3.00 | base is serverless; LoRA still on-demand |
| `accounts/fireworks/models/qwen3p8-27b` | 27.3B dense | `qwen3p8-27b-32k-b200-lora` (4 x B200) | 3.00 | newest Qwen VLM |
| `accounts/fireworks/models/gemma-4-26b-a4b-it` | 26B MoE | `gemma-4-26b-a4b-256k-b200-lora` (4 x B200) | 3.00 | text+image |
| `accounts/fireworks/models/gemma-4-31b-it` | 32B dense | `gemma-4-31b-256k-b200-lora` (4 x B200) | 3.00 | text+image |
| `accounts/fireworks/models/nvidia-nemotron-3p5-super-vl-120b-a12b-bf16` | 124B MoE | 8 x B300 | 6.00 | too big to serve cheaply |

`scripts/launch_finetune.py --list-models` prints the same table. Before spending money, confirm
with `firectl model get -a fireworks qwen3p5-9b` (look for `Tunable: true`) because the catalogue
changes weekly.

**Why `qwen3p5-9b` is the default:** it is the cheapest tier ($0.50/M tokens), natively multimodal,
managed-SFT-tunable today, small enough that the live-merged deployment fits a single H100/H200
($8/h) with base-model latency, and it is the same family (Qwen3.5/3.6) the baseline VLM client
already prices. `qwen3p5-27b` is the one-flag upgrade (`--base-model`) if the 9B under-performs.

## Step by step

### 1. Prepare the dataset

```bash
.venv/bin/python scripts/prepare_sft.py --dataset pcam --n 2000 --eval-n 200 --image-size 448
# crc: --dataset crc --n 2000 (9 classes, ~222 per class); source split defaults to train
```

* Source: `load_split(dataset, "valid" | "train", limit=n+eval_n, balanced=True, seed)`; a
  stratified seeded split keeps both files balanced; the test split is never touched.
* Each line = `messages`: `system` = the baseline system prompt, `user` = `[image_url data URL,
  baseline user text]`, `assistant` = `{"label": "<class>", "confidence": 0.95, "reason": "<one
  grounded sentence>"}`. The prompt text comes from `vlm_client.build_prompt(dataset, "v1")`, so
  the fine-tuned model is evaluated with exactly the prompt it was trained on. Rationales are drawn
  (seeded by tile id) from 2-3 pathology templates per class (`finetune.RATIONALE_TEMPLATES`).
* Tiles are resized to 448x448 JPEG (PCam is 96x96 natively; upscaling costs tokens but gives the
  vision tower enough pixels; `--image-size 224` halves tokens and file size if you are in a hurry).
* The script prints class counts, file size, a rough token count and the training cost estimate,
  then runs `validate_sft_jsonl` (schema rules above). Extra top-level keys (`tile_id`, `label`)
  are kept for audits; Fireworks ignores them.
* `--synthetic` uses `data.make_synthetic` (no downloads) for smoke tests.

Expected at 2,000 x 448 px: ~0.8-1.0M tokens/epoch (~256 image + ~230 text + ~45 answer tokens per
example), 60-90 MB JSONL (below the 150 MB single-upload limit; the client falls back to the
signed-URL path automatically if you exceed it).

### 2. Launch

```bash
.venv/bin/python scripts/launch_finetune.py \
    --base-model accounts/fireworks/models/qwen3p5-9b \
    --dataset-file data/sft_pcam_train.jsonl --eval-file data/sft_pcam_eval.jsonl \
    --epochs 2 --lora-rank 16 --name pathfrontier-pcam-qwen3p5-9b --wait --deploy
```

What it does (each step prints its `firectl` equivalent):

1. `validate_sft_jsonl` on both files (fails fast on schema errors).
2. `FireworksFT.create_dataset` -> `POST datasets` + `:upload` + poll READY
   (`firectl dataset create sft-pcam-train data/sft_pcam_train.jsonl`).
3. `FireworksFT.create_job` -> `POST supervisedFineTuningJobs` with `loraRank=16`, `epochs=2`,
   learning rate left to Fireworks (`firectl sftj create --base-model ... --lora-rank 16 --epochs 2`).
4. `--wait`: polls every 30 s, prints state/progress, raises on FAILED/CANCELLED/EXPIRED.
5. `--deploy`: `POST deployments` with `deploymentShape: "default"`, `minReplicaCount: 0`,
   `maxReplicaCount: 1`, `scaleToZeroWindow: 600s`, polls until READY, prints
   `export FIREWORKS_FT_MODEL=accounts/<acct>/models/<name>`.

Without `--wait` you can come back later: `firectl sftj get <job>` then
`scripts/deploy_finetuned.py --model accounts/<acct>/models/<name> --wait`.

### 3. Evaluate with the fine-tuned model

`scripts/run_vlm_ft.py` (teammate's script) reads `FIREWORKS_FT_MODEL` and calls the same
OpenAI-compatible endpoint (`config.FIREWORKS_BASE_URL`) with the same `v1` prompt, writing
`results/preds_<model>_test.jsonl` with `stage="vlm_ft"`. Then `scripts/eval.py` /
`scripts/plot_frontier.py` pick it up as the "fine-tuned VLM alone" point and the cascade's
`fallback="vlm_ft"` stage.

Price bookkeeping: `vlm_client.PRICES` has no entry for a custom model id; it falls back to the
parameter-size tier (< 16B -> $0.20/M tokens in+out) which matches "fine-tuned models are served at
the base model price". Latency is measured live.

### 4. Tear down

`firectl deployment delete <deployment-id>` (or `scripts/deploy_finetuned.py --delete <id>`).
Scale-to-zero (`minReplicaCount 0`, 10 min window) limits the damage if you forget, but the
cleanest is to delete right after the 500-tile eval.

## Expected time and cost (2,000 PCam tiles, 448 px, 2 epochs, qwen3p5-9b)

| Step | Time | Cost |
|---|---|---|
| prepare_sft (download valid parquet once + encode) | 2-5 min | 0 |
| dataset upload + validation | 1-3 min | 0 |
| SFT job (queue + 2 epochs x ~0.9M tokens on 2 x B200) | typically 15-45 min; queue can add more | ~1.8M tokens x $0.50/M = **~$0.90** (the job's `estimatedCost` field shows the real figure) |
| deployment creation (live merge + load) | 5-15 min | GPU-time from READY: ~$8/h (1 x H100/H200) |
| run_vlm_ft on 500 test tiles, concurrency 8 | 2-5 min | included in GPU-hour; token cost negligible |
| **Total** | **~45-90 min wall** | **~$10-20** (dominated by keeping the GPU up ~1 h) |

With `qwen3p5-27b`: training ~$5.40, deployment 1 x B200 at $13/h. Do not fine-tune the 120B
Nemotron on hackathon money (8 x B300 to serve).

## Troubleshooting

* `HTTP 400 ... loraRank` / job silently runs full-parameter: you sent `loraRank: 0` or omitted it.
  `FireworksFT.create_job` refuses `lora_rank <= 0`; keep 8-32.
* `base model is not tunable` / "no training shape": the catalogue changed. Run
  `scripts/launch_finetune.py --list-models`, check `firectl model get -a fireworks <id>` for
  `Tunable: true`, and pick another row from the table above (`--base-model`).
* Dataset stuck in `UPLOADING` / validation error mentioning images: make sure every image part is a
  `data:image/jpeg;base64,...` URI on a **user** message (`validate_sft_jsonl` enforces this) and
  the MIME type matches the bytes (we always write JPEG). Remote URLs are rejected.
* `413` or upload timeout: file > 150 MB. Re-run `prepare_sft.py --image-size 224` or let the REST
  client use the signed-URL path (automatic); `firectl dataset create` handles big files too.
* Job `JOB_STATE_FAILED` with a context-length message: an example exceeded `maxContextLength`.
  Only possible with huge images; 448 px tiles are ~300 tokens total.
* Deployment `FAILED` with a live-merge / shape error: run
  `firectl deployment-shape-version match --model accounts/<acct>/models/<name>` and pass one of the
  returned shapes via `--shape` (FP8/FP4 shapes cannot live-merge; use a BF16 one).
* `run_vlm_ft.py` returns 404 / "model not found": the deployment is not READY yet, or you passed the
  job id instead of `accounts/<acct>/models/<output-model>`. `firectl deployment list` shows state.
* Model answers prose instead of JSON: it was trained with the `v1` prompt, so run with
  `--prompt-version v1` (default). The parser falls back to regex and marks `parse_error`.
* Everything looks right but accuracy is at chance: check the class balance printed by
  `prepare_sft.py` and that `tile_id` labels match (`label` key in each JSONL line); try
  `--epochs 3 --lora-rank 32` or the 27B base.
* No `FIREWORKS_ACCOUNT_ID`: `firectl whoami` prints it; the REST paths need it, firectl does not.

## Fallback plan if Fireworks vision SFT is unavailable on the day

**(a) Few-shot baseline (no training, 20 min).** Keep the baseline VLM and prepend 4-8 labelled
tiles as prior user/assistant turns using the same JSONL examples (`build_sft_example` output is
already chat-shaped: just concatenate the `messages` of k exemplars before the query). Run it as
`stage="vlm_ft"` with `model="<baseline id>+fewshot"` so the frontier still shows a "specialised VLM"
point. Cost: k x ~300 extra input tokens per tile; cache the exemplar prefix. Pick exemplars with
the probe's most confident correct tiles per class.

**(b) Local LoRA of Qwen2.5-VL-3B / Qwen3-VL-2B with HF PEFT (sketch, 1-2 h on a rented GPU).**
Rent one A100/H100 (RunPod/Lambda, ~$2-3/h). Then:

```python
# pip install "transformers>=4.49" peft accelerate qwen-vl-utils bitsandbytes
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from peft import LoraConfig, get_peft_model
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2.5-VL-3B-Instruct", torch_dtype="bfloat16", device_map="auto")
processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct",
                                          min_pixels=224*224, max_pixels=448*448)
model = get_peft_model(model, LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]))
# data: read data/sft_pcam_train.jsonl, decode the data URL back to PIL, and build
# processor.apply_chat_template(messages, tokenize=False) + processor(images=..., text=...);
# mask labels to -100 everywhere except the assistant turn; batch 8, lr 1e-4, 2 epochs (~20 min).
# serve: `vllm serve Qwen/Qwen2.5-VL-3B-Instruct --enable-lora --lora-modules ft=./lora_out`
# then point vlm_client at it: FIREWORKS_BASE_URL=http://<gpu-host>:8000/v1 FIREWORKS_FT_MODEL=ft
```

The vLLM OpenAI-compatible server means `run_vlm_ft.py` works unchanged (only `base_url` and the
model id differ); cost per tile is then GPU-hour / throughput, which `vlm_client` can be told via
`VLMConfig.price_*` overrides.

## Sources (all read 2026-10-29/30)

* https://docs.fireworks.ai/fine-tuning/fine-tuning-models (dataset format, vision section, limits, firectl, LoRA rank, defaults)
* https://docs.fireworks.ai/fine-tuning/fine-tuning-vlm (managed VLM guide; same rules, older examples)
* https://docs.fireworks.ai/fine-tuning/training-api/vision-inputs (Training API vision inputs; Qwen3-VL-8B example)
* https://docs.fireworks.ai/fine-tuning/models (model matrix; catalogue parsed for managedSft / shapes)
* https://docs.fireworks.ai/fine-tuning/managed-finetuning-intro
* https://docs.fireworks.ai/fine-tuning/deploying-loras (live merge vs multi-LoRA, no serverless LoRA, inference ids)
* https://docs.fireworks.ai/fine-tuning/training-api/serverless (serverless Training API price table; a different product)
* https://docs.fireworks.ai/api-reference/create-dataset , .../upload-dataset-files , .../get-dataset-upload-endpoint , .../validate-dataset-upload
* https://docs.fireworks.ai/api-reference/create-supervised-fine-tuning-job , .../get-supervised-fine-tuning-job
* https://docs.fireworks.ai/api-reference/create-deployment , .../get-deployment
* https://docs.fireworks.ai/tools-sdks/firectl/firectl , .../commands/dataset-create , .../commands/supervised-fine-tuning-job-create , .../commands/deployment-create
* https://docs.fireworks.ai/faq-new/billing-pricing/are-there-extra-fees-for-serving-fine-tuned-models
* https://docs.fireworks.ai/faq-new/billing-pricing/how-many-tokens-per-image
* https://docs.fireworks.ai/guides/querying-vision-language-models (inference image limits)
* https://docs.fireworks.ai/serverless/pricing (parameter-size inference tiers)
* https://fireworks.ai/pricing (fine-tuning per-token tiers, GPU hourly prices)
* https://fireworks.ai/blog/vlm-fine-tuning (2025-07-29 launch post; Qwen2.5-VL - now stale)
* Model pages: https://fireworks.ai/models/fireworks/qwen3p5-9b , qwen3p5-27b , qwen3p6-35b-a3b , qwen3p8-27b , gemma-4-26b-a4b-it , gemma-4-31b-it , nvidia-nemotron-3p5-super-vl-120b-a12b-bf16 , qwen3-vl-8b-instruct , qwen3-vl-32b-instruct , qwen3-vl-30b-a3b-instruct , qwen3-vl-235b-a22b-instruct , qwen2p5-vl-7b-instruct , qwen2p5-vl-32b-instruct , qwen2p5-vl-72b-instruct
* https://huggingface.co/Qwen/Qwen3.5-9B (vision encoder confirmation)
