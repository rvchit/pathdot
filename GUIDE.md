# GUIDE — "Make your own frontier" (Nov 2, 2026, San Francisco)

Pathology tile triage: a pathology foundation encoder + linear probe handles every tile, a
fine-tuned VLM (or a pathologist) handles only the uncertain band, and we show the
cost-vs-sensitivity frontier against a general-purpose VLM baseline. The demo is your own
slide viewer, Cello, painting a tumor heatmap over a slide.

Everything in this guide was run on this Mac (M3 Pro) unless marked **[needs key]** or
**[needs GPU]**.

---

## 0. What is in the repo

```
pathfrontier/     Python package
  data.py         PCam / NCT-CRC loaders (HF parquet), synthetic fixtures
  encoders.py     Phikon-v2 (ungated), UNI, UNI2-h, CONCH, Virchow2, H-optimus-0, DINOv2 + ResNet50 controls
  probe.py        LogisticRegression head on frozen embeddings, calibrated
  vlm_client.py   OpenAI-compatible VLM client (Fireworks / Anthropic), prices, caching, retries
  finetune.py     Fireworks vision LoRA SFT: dataset builder + REST client + firectl hints
  cascade.py      Router (low/high band), threshold sweep, Pareto front
  metrics.py      AUROC, balanced acc, sens@spec, latency p50/p95, $/1k tiles, $/slide, coverage
  serve.py        FastAPI inference service (tile + whole-slide heatmap jobs), port 8008
  dzi.py          Deep Zoom pyramid helpers + pure-Pillow tiler
scripts/          one CLI per step (see sections 3-6); scripts/dev.sh runs the full stack
cello/            your slide viewer (React + OpenSeadragon + Express) with the AI triage panel
docs/CONTRACTS.md the interfaces every module honours   docs/FINETUNE.md  Fireworks SFT recipe
docs/PITCH.md     slides outline, demo click-script, backup plan, judging-criteria map
results/          preds_*.jsonl, metrics_*.json, leaderboard.md, sweep_*.csv, frontier.png, heatmaps/
data/             pcam/test.parquet (32,768 tiles) and crc/test.parquet (7,180) are already downloaded
tiles/            DZI pyramids; pcam_mosaic_24 (a 24x24 grid of real PCam test tiles) is already built
```

Prediction format (one JSONL line per tile, same for every model): see `pathfrontier/schema.py`.
That one schema is what makes the leaderboard and the frontier chart work for any model.

---

## 1. Before Nov 2 (do these this week)

| When | What | Why |
|---|---|---|
| Today | Register everyone; find 1-4 teammates | solo teams are rejected |
| Today | Request access on Hugging Face: [MahmoodLab/UNI](https://huggingface.co/MahmoodLab/UNI), [MahmoodLab/UNI2-h](https://huggingface.co/MahmoodLab/UNI2-h), [MahmoodLab/CONCH](https://huggingface.co/MahmoodLab/CONCH), [bioptimus/H-optimus-0](https://huggingface.co/bioptimus/H-optimus-0), [paige-ai/Virchow2](https://huggingface.co/paige-ai/Virchow2) | approval takes days; Phikon-v2 is ungated and already works as the fallback |
| Today | Create a Fireworks account, make an API key, note the account id, add a payment method; run `firectl` login | baseline + fine-tune both need it |
| Today | Email organisers: is pre-caching embeddings / having code ready before 10:30 allowed? | the timeline below assumes yes |
| Day before | `cp .env.example .env` and fill FIREWORKS_API_KEY, FIREWORKS_ACCOUNT_ID, HF_TOKEN | |
| Day before | Run section 2 + section 3 end to end once on your laptop | so the day is about results, not setup |
| Day before | Decide: PCam (binary, simplest story; default) vs NCT-CRC (9 classes; everything supports `--dataset crc`) | |
| Optional | `brew install vips && uv pip install pyvips` | only needed to tile real .svs slides; the mosaic demo needs nothing |

Fireworks facts discovered while building this (verified 2026-10-29/30; re-check on the day):

- Serverless **vision** models on Fireworks right now: Kimi K3 ($3/$15 per M tokens), Ember-1,
  Qwen3.8-Max, Muse-Spark, **DeepSeek V4.1 Flash ($0.30/$1.20)**, MiniMax M3.1 Flash, GLM-5.3 Flash.
  Qwen2.5-VL, Qwen3-VL and Llama 4 are on-demand-GPU only now ($8/h H100). The baseline script
  defaults to Kimi K3; `--model accounts/fireworks/models/deepseek-v4p1-flash` is the cheap second point.
- Vision **LoRA fine-tuning**: Qwen2.5-VL is no longer tunable. Tunable multimodal bases with managed
  SFT: `qwen3p5-9b` (default, $0.50/M training tokens), `qwen3p5-27b`, `gemma-4-26b-a4b-it`, others in
  `docs/FINETUNE.md`. Fine-tuned LoRAs serve on an **on-demand deployment** (not serverless), so
  budget ~$8-16 of GPU-hour and delete the deployment after eval.
- Images go inline as base64 data URLs in the SFT JSONL; no separate upload.

---

## 2. Setup (10 minutes, already done on this machine)

```bash
cd "/Users/rachit.bisht/Desktop/personal stuff/path"
make setup                     # uv venv (.venv, Python 3.11) + pip install -e .[dev] + npm install x2
cp .env.example .env           # fill keys
.venv/bin/python -m pytest -q  # offline test suite (no keys, no downloads)
```

Already on disk: `data/pcam/test.parquet`, `data/crc/test.parquet`, `tiles/pcam_mosaic_24/`.
Need the PCam **valid** split too (probe training set): `.venv/bin/python scripts/download_data.py --dataset pcam --splits valid`  (~770 MB).

---

## 3. Day-of timeline (10:30 → 4:30)

Run every command from the repo root with `.venv/bin/python`. Each step writes to `results/` and the
next step reads from there, so people can work in parallel on different steps.

### 10:30 – 11:30  Baseline VLM + eval harness (person A)   **[needs key]**

```bash
.venv/bin/python scripts/list_fireworks_models.py                                  # confirm what is serverless today
.venv/bin/python scripts/run_baseline.py --dataset pcam --split test --limit 500 --concurrency 8
.venv/bin/python scripts/run_baseline.py --dataset pcam --split test --limit 500 \
    --model accounts/fireworks/models/deepseek-v4p1-flash                             # cheap second baseline
.venv/bin/python scripts/eval.py                                                    # -> results/leaderboard.md
```
Resumable: re-running continues from the JSONL cache. Expect ~1-2 s per tile per request, so
500 tiles at concurrency 8 ≈ 2-3 min. Cost at Kimi K3 prices ≈ $1-2 for 500 tiles.
No key? `--dry-run` runs a fake client so the rest of the pipeline can be tested.

### 10:30 – 12:00  Embeddings + linear probe (person B)

```bash
.venv/bin/python scripts/bench_encoders.py                                           # tiles/s per encoder on this machine
.venv/bin/python scripts/embed.py --encoder phikon-v2 --dataset pcam --split valid --limit 8192
.venv/bin/python scripts/embed.py --encoder phikon-v2 --dataset pcam --split test  --limit 4096
.venv/bin/python scripts/train_probe.py --encoder phikon-v2 --dataset pcam --C 0.001 --tiles-per-s 1500
#   -> results/probe_phikon-v2_pcam.joblib  and  results/preds_probe_phikon-v2_test.jsonl
```
`--C 0.001` matters: PCam's test split comes from different scanners than train/valid, and a strongly
regularised head generalises across that shift (measured here: AUROC 0.89 at C=1 vs 0.95 at C=0.001,
while same-distribution cross-validation is 0.99 either way). `--tiles-per-s 1500` prices the probe at
H100 throughput instead of this laptop's 26 tiles/s. **On the hackathon GPU, embed the full PCam train
split** (262k tiles, ~3 min on an H100, `--split train --limit 0`) and train on that: more scanners in
training is the real fix for the shift.
Then repeat with `--encoder uni` (or `uni2-h`, `h-optimus-0`, `virchow2`) once HF access is approved.
Also run `--encoder resnet50` as the "not a pathology model" control; it makes the slide where you
explain *why* the foundation model matters. The 4096 test tiles are a superset of the 500 baseline
tiles (same seed), so the leaderboard compares like with like.

### 11:00 – 1:30  VLM fine-tune, in parallel (person C)   **[needs key, ~$10-20]**

```bash
.venv/bin/python scripts/prepare_sft.py --dataset pcam --split valid --n 2000      # data/sft_pcam_train.jsonl
.venv/bin/python scripts/launch_finetune.py --wait --deploy                         # prints: export FIREWORKS_FT_MODEL=...
export FIREWORKS_FT_MODEL=accounts/<acct>/models/<name>
.venv/bin/python scripts/run_vlm_ft.py --dataset pcam --split test --limit 4096 --concurrency 8
.venv/bin/python scripts/deploy_finetuned.py --delete                               # stop the GPU meter
```
Full recipe, timings and the manual `firectl` commands: `docs/FINETUNE.md`. If Fireworks vision SFT is
unavailable that day, the fallback (few-shot prompt on the baseline model) is in the same doc.

### 1:30 – 2:30  Cascade sweep + THE chart (person A or B)

```bash
.venv/bin/python scripts/sweep.py --probe-preds results/preds_probe_phikon-v2_test.jsonl \
    --vlm-preds results/preds_<ft-model>_test.jsonl --fallback vlm_ft
.venv/bin/python scripts/sweep.py --probe-preds results/preds_probe_phikon-v2_test.jsonl --fallback pathologist
.venv/bin/python scripts/plot_frontier.py --sweep results/sweep_vlm_ft.csv --sweep results/sweep_pathologist.csv \
    --points results/preds_accounts_fireworks_models_kimi-k3_test.jsonl \
    --points results/preds_probe_phikon-v2_test.jsonl --points results/preds_<ft-model>_test.jsonl \
    --out results/frontier.png
.venv/bin/python scripts/eval.py                                                    # final leaderboard
```
`results/frontier.png` is the slide. `results/sweep_*.csv` gives you the operating point sentence:
"auto-clears X% of tiles at Y sensitivity for $Z per slide". A synthetic version of all of this
already exists in `results/demo/` (from `scripts/make_demo_results.py`) so you can rehearse the
slide before real numbers land. Label it synthetic if you ever show it.

### 2:30 – 4:00  Demo in Cello (person D, can start at 10:30 with the stub)

```bash
make dev                  # backend :4001, frontend :4000, ML service :8008 (loads the trained probe)
# or, before the probe exists:  make dev-stub   /   PATHFRONTIER_FAKE=1 make dev
open "http://localhost:4000/?image=pcam_mosaic_24&triage=1&gt=1"
```
Press **Run AI triage** → heatmap (green cleared, red flagged, amber review) → toggle ground truth →
click a cell for the probability and the VLM reason → drag the thresholds and re-run.
To use the fine-tuned VLM for the amber band in the demo: `PATHFRONTIER_VLM_MODEL=$FIREWORKS_FT_MODEL make dev`.
To demo a real slide: `scripts/tile_slide.py --download-sample` (needs libvips for .svs) or tile any
png/jpg with `scripts/tile_slide.py path/to/image.png --out tiles/`.

### 4:00 – 4:30  Slides + submission
`docs/PITCH.md` has the five-slide arc, the click script, and the judging-criteria mapping.
Paste the leaderboard table, the frontier chart, and a screenshot of the heatmap.

---

## 4. The numbers to put on slides

Measured on this laptop (Phikon-v2, probe trained on 8,192 valid tiles, evaluated on 4,096 test tiles;
re-run everything on the day, these are the floor):

| model | AUROC | bal. acc | sens @95% spec | sens @99% spec | $ / slide | note |
|---|---|---|---|---|---|---|
| probe / Phikon-v2 (pathology encoder) | 0.953 | 0.880 | 0.827 | 0.617 | $0.0046 | 0.7 ms/tile at H100 throughput |
| probe / ResNet50 (ImageNet control) | 0.863 | 0.789 | 0.516 | 0.304 | $0.0012 | the "why a pathology model" slide |
| baseline VLM, fine-tuned VLM | run on the day | | | | | needs FIREWORKS_API_KEY |

Cascade with pathologist fallback (`results/sweep_pathologist.csv`), probe = Phikon-v2:

| low / high | tiles auto-decided | system sensitivity | pathologist min / slide (vs 333 full read) |
|---|---|---|---|
| 0.001 / 0.7 | 67% | 0.979 | 111 |
| 0.0003 / 0.7 | 53% | 0.992 | 158 |
| 0.0001 / 0.7 | 43% | 1.000 | 191 |

Demo slide (`pcam_mosaic_24`, 576 real test tiles, low 0.05 / high 0.95): 73% of tiles auto-decided
with zero false positives, slide-level AUROC 0.95, 20 s on the laptop.

Fill the VLM rows from `results/leaderboard.md` after the runs. Columns: AUROC, balanced accuracy, sensitivity at
95% and 99% specificity, p50/p95 latency, $ per 1k tiles, $ per slide (10,000 tiles), coverage.

Business value line (from `results/sweep_pathologist.csv`, pick the row with sensitivity_system ≥ 0.99):
`auto-clears {1 - fraction_routed} of tiles` → `pathologist_minutes_per_slide` saved versus a 333-minute
full read at 2 s/tile. Both constants live in `pathfrontier/config.py`; change them if a pathologist on
your team gives you better ones.

---

## 5. Troubleshooting

| Symptom | Fix |
|---|---|
| `401/403` loading UNI/CONCH | access not approved yet; use `--encoder phikon-v2` |
| MPS out of memory on Virchow2 / H-optimus | `--batch-size 16`, or run on a Fireworks/cloud GPU |
| Fireworks 429s | lower `--concurrency`; the JSONL cache means re-runs cost nothing |
| Baseline answers are not JSON | parser falls back to regex; `parse_errors` is printed; tiles that still fail get uniform probs and `extra.parse_error` |
| `ML service unreachable` in the Cello panel | start it: `make ml` (or `make ml-stub`); check `curl localhost:8008/health` |
| Probe artifact missing at service start | run section 3 person-B steps, or `PATHFRONTIER_FAKE=1` |
| Heatmap looks shifted | DZI level mismatch; the service default is `max_level - 1`; pass `"level": "max"` in the panel's request for full resolution |
| `.svs` won't tile | `brew install vips && uv pip install pyvips` |
| Cascade sensitivity plateaus below 0.99 in the sweep | probabilities are collapsing; the probe blends 2% raw logistic probability into the isotonic output to keep ranking (`RAW_BLEND` in `pathfrontier/probe.py`); extend `DEFAULT_LOWS` in `pathfrontier/cascade.py` if needed |
| Heatmap shows white/pale tumor cells as background | lower `PATHFRONTIER_BG_SAT` (default 0.04; mean HSV saturation below it = glass) |
| Want CRC instead of PCam | add `--dataset crc` everywhere; tumor-vs-rest is the clinical metric, 9-class macro AUROC also reported |

---

## 6. Make it yours (after the hackathon)

The encoder + probe + router is the base for slide-level work: swap the per-tile head for an
attention-MIL aggregator over the cached tile embeddings and you have slide-level biomarker prediction.
The `Prediction` schema, the metrics module and the frontier chart work unchanged for that.
