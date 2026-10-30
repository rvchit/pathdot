# Pitch, demo script and slides outline

## One-liner
**"A general VLM reads a lymph-node tile wrong one time in four and costs dollars per slide.
A pathology foundation model with a 1,000-parameter head reads it right 98% of the time for a
tenth of a cent. We route between them, and only the hard tiles ever reach a human."**

## The story in 5 slides (7 minutes)

1. **The problem (30 s).** A whole-slide image is ~10,000 tiles. A pathologist screens every one.
   Frontier VLMs are impressive on photos but were never trained on H&E; sending 10k tiles
   through one costs $X and takes minutes, and it still misses tumor. (Show one tile the
   baseline VLM got wrong, with its confident rationale.)
2. **The task + eval (45 s).** PatchCamelyon: metastasis yes/no, 96 µm lymph-node patches,
   32k held-out test tiles. Metrics: AUROC, balanced accuracy, **sensitivity at 95%/99%
   specificity** (a missed tumor costs far more than a false alarm), p50/p95 latency,
   **$ per slide**. One harness, one JSONL format, every model on the same tiles.
3. **Three models, three price points (60 s).** Table: baseline VLM zero-shot vs
   Phikon-v2/UNI frozen encoder + linear probe vs LoRA-tuned Qwen-VL on Fireworks.
   The probe wins on accuracy *and* is ~1,000× cheaper. The fine-tuned VLM is in between
   on accuracy but is the only one that explains itself.
4. **The cascade is the frontier (90 s). THE CHART.** Cost per slide (log) vs sensitivity.
   Probe handles every tile; only the uncertain band (typically 5–15% of tiles) goes to
   the VLM or a pathologist. Sweep the band and you get a curve that dominates every
   single-model point. Call out the operating point: "auto-clears N% of tiles at 99%
   sensitivity, $0.0X per slide, saves ~M pathologist-minutes per slide."
5. **Live demo (2 min) in Cello.** Open the mosaic slide, press *Run AI triage*, watch the
   heatmap paint: green = auto-cleared, red = flagged, amber = sent for review. Flip on
   ground truth: the red matches. Click an amber tile: the VLM's one-line reason. Drag the
   threshold slider: the summary numbers move with it. Close on the business value line.

**Close (20 s).** "Every component is an open model on Fireworks or Hugging Face; the whole
thing runs on one GPU. The same encoder + router is the base for slide-level biomarker
prediction, which is where we're taking this next."

## Demo script (exact clicks)
1. `make dev` is already running; browser at http://localhost:4000.
2. Sidebar → pick `pcam_mosaic_24` → slide loads (OpenSeadragon, zoom once to show tiles).
3. Triage panel → **Run AI triage** (low 0.05, high 0.95, fallback vlm_ft). Progress bar ~10 s.
4. Overlay paints. Read the summary aloud: tiles scored, auto-cleared %, review count, cost.
5. Toggle **Ground truth** → outline matches. Zoom into a red cell.
6. Click an amber cell → tooltip with p_tumor and the VLM reason.
7. Slide *high* down to 0.8 → re-run → fewer review tiles, sensitivity reading from the sweep table.
8. Switch to the frontier chart slide (results/frontier.png) and point at the same operating point.

## Backup plan if anything is down
- Fireworks down → set fallback = pathologist in the panel; the cascade still demos, story becomes "triage".
- Encoder download blocked → `PATHFRONTIER_ENCODER=resnet50` (ImageNet control; weaker, still runs).
- Whole ML service down → `PATHFRONTIER_FAKE=1` keeps the UI alive; show the chart from results/.
- Chart not regenerated → `results/demo/frontier.png` from `scripts/make_demo_results.py` (label it as synthetic!).

## Judging criteria mapping
| Criterion | Where we show it |
|---|---|
| Task you define + production deployment | PCam metastasis triage; FastAPI service + Cello viewer, docker-compose |
| Eval on real data | 4,096-tile PCam test subset, same tiles for every model, `results/leaderboard.md` |
| Measured tradeoffs | leaderboard: AUROC/sens@spec vs p50/p95 latency vs $/1k tiles |
| Tuned at least one dimension vs baseline | cascade threshold sweep (`results/sweep_*.csv`), LoRA fine-tune |
| Business value | auto-clear % at 99% sensitivity → pathologist minutes per slide |
| How far you pushed it | frontier chart: cascade curve vs every single-model point |
