# Demo story: two fine-tuned models, one pathologist workflow

**One line.** A pathologist's day is tile-reading and report-writing. We fine-tuned one open 9B
model, twice, on Fireworks, in under 35 minutes of GPU time, to take the bulk out of both: a
triage model that clears most of a slide before a human looks at it, and an assistant that answers
the questions a pathologist asks of a region. Everything runs inside Cello, our slide viewer.

## The two problems and the two models

| problem | what the pathologist does today | our model | measured gain |
|---|---|---|---|
| **Triage**: find metastasis in a lymph-node slide of ~10,000 tiles | reads every tile, ~2 s each, 5+ hours per slide | Phikon-v2 encoder + linear probe on every tile, fine-tuned Qwen3.5-9B second opinion on the uncertain band | PCam test AUROC 0.953; cascade clears 53% of tiles at 99.2% sensitivity. Camelyon17 cross-hospital: fine-tuned Qwen3.5-9B 85.5% vs 52.3% for zero-shot Qwen2.5-VL-32B |
| **Reporting**: describe what a region shows | types the answer from memory and reference texts | Qwen3.5-9B LoRA fine-tuned on 8,000 PathVQA Q&A pairs, 14 min, $0.33 | yes/no accuracy 75% to 91%, open-answer recall 4x, overall exact 40% to 59% (McNemar p < 0.0001), 0.47 s per question |

Same base model, same platform, two LoRAs. General frontier VLMs are the wrong tool for both:
zero-shot Qwen2.5-VL-32B catches 17% of tumors on Camelyon17; in-context examples made PathVQA
worse (-2.2 pts, p = 0.03) and tripled latency.

## Numbers for the slides

PathVQA, same seeded 1,000 test questions for every row:

| model | yes/no acc | open exact | open recall | overall exact | p50 latency |
|---|---|---|---|---|---|
| Qwen3-VL-8B zero-shot | 67.1 | 1.2 | 5.8 | 34.1 | 0.30 s |
| Qwen3-VL-8B 8-shot | 62.1 | 1.8 | 5.8 | 31.9 | 0.77 s |
| Qwen3.5-9B zero-shot | 75.2 | 5.2 | 7.6 | 40.1 | 0.31 s |
| **Qwen3.5-9B LoRA (1 epoch)** | **90.6** | **27.5** | **32.5** | **59.0** | 0.47 s |
| published LLaVA-Med (fine-tuned) | 91.2 | | 38.0 | | |

Camelyon17 (WILDS hospital 2, 6,000 slide-disjoint tiles, 50/50):

| model | accuracy | sensitivity | specificity | AUROC | p50 latency | GPU cost / 1k tiles |
|---|---|---|---|---|---|---|
| Qwen2.5-VL-32B zero-shot | 52.3 | 16.6 | 88.0 | 0.53 | 2.3 s | $0.35 |
| Phikon-v2 probe (ours, no VLM) | 86.0 | | | 0.977 | 0.03 s | ~$0.00 |
| **Qwen3.5-9B LoRA (2 epochs)** | **85.5** | 78.1 | 92.8 | 0.943 | 0.41 s | $0.04 |

Training cost for everything shown: two LoRA jobs, 14 and 20 minutes, under $1 of training tokens.
Serving: one 9B deployment answers a question in half a second for about $0.001.

## Time saved, the arithmetic we say out loud

- A full read is 10,000 tiles x 2 s = 333 minutes. The cascade at the 99% sensitivity operating
  point sends 47% of tiles to the human: 158 minutes, **175 minutes saved per slide**.
- Each report question answered in 0.5 s instead of a 30 to 60 s lookup; at 20 questions per case
  that is **10 to 20 minutes per case**.
- At 8 slides a day, that is a working day returned per week per pathologist.
  (Constants live in `pathfrontier/config.py`; swap in a pathologist's real numbers if one is in the room.)

## Pushing the dimension: epochs
1 epoch: eval loss 1.11 -> 0.57. 3 epochs: 0.570 / 0.576 / 0.645 after epochs 1 / 2 / 3 while train
loss kept falling to 0.24. Held-out loss bottoms at epoch 1; more epochs memorise the 8,000 answers.
The 1-epoch model is the one deployed. Training cost of both runs together: under $1.50.

## Demo click script, gallery page (3 minutes)
Open `http://localhost:4000/demo` (started by `make dev`).
1. **Tumor scan.** Click Tile 1, press *Scan for tumor*: red "TUMOR DETECTED", confidence, who decided,
   latency, then the ground-truth reveal. Tile 3 is the normal one: green "NO TUMOR". These are real
   Camelyon17 test tiles from a hospital the model never trained on.
2. **Ask the assistant.** Click Image 1 (gross kidney). Click the chips: "what does this image show?" ->
   kidney, "where is this?" -> urinary, the yes/no question -> yes, each with the dataset's expected
   answer and a match mark, ~300 ms and $0.0002 per question after the first call.
3. Type a free question of your own. Then jump to the slide viewer for the heatmap if there is time.

## Demo click script, slide viewer (4 minutes)

1. `make dev` is running; open `http://localhost:4000/?image=pcam_mosaic_24`. The slide is 576 real
   PCam test tiles stitched into one image with ground truth.
2. **Run AI triage.** Heatmap paints in ~20 s: green cleared, red flagged, amber review. Read the
   impact strip: % auto-cleared, minutes saved.
3. Toggle **ground truth**: the red cells match the outlines. Zoom into one red cell.
4. Click an amber cell. The tooltip shows p_tumor and that it was routed. (If the Camelyon LoRA is
   deployed, the second opinion and its one-word answer appear here.)
5. In **Ask the pathology assistant**, click "Is tumor present?" then "What tissue is this?" and
   type one free question. Point at latency and cost.
6. Drag the low threshold slider, re-run, watch the auto-cleared % and minutes move: "this is the
   dial a lab sets to its risk tolerance."
7. Close on the results tables above.

## Backups
- ML service down: `PATHFRONTIER_FAKE=1 make dev` keeps the UI alive; show the tables.
- Fireworks down: the assistant panel says "unavailable"; the triage still runs locally on the probe.
- Deployment deleted to save money: redeploy takes ~6 min (`firectl deployment create accounts/rvchit/models/pathvqa-qwen3p5-9b --deployment-shape accounts/fireworks/deploymentShapes/rft-qwen3p5-9b-v2 ...`).
