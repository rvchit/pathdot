# Pathology Q&A on Lymph Node Slides — Hackathon Build Plan

**Event:** Fireworks: Deploy · November 2, 2026 · San Francisco
**One-line pitch:** A specialized open model that reads lymph node tissue, answers a pathologist's routine questions about the tumor, and runs on an ordinary laptop — no GPU, no patient data leaving the building.

---

## 0. The idea in plain terms

A pathologist looking at a lymph node slide asks a small set of routine questions: *Is there tumor? How much? How big is the biggest piece? How many pieces? Where is it?* Answering these by eye on a giant slide is slow and tedious.

We build a system that:

1. Looks at each small square of tissue (a "patch") and predicts **tumor or normal**.
2. Puts those predictions back onto the slide grid using each patch's position.
3. **Computes** the answers to the five questions by counting and measuring — never by guessing.
4. Writes the answer in plain English.

Everything is grounded in either a pathologist's label (for scoring) or a measurement our code made (for answers). Nothing medical is invented.

> **Why this wins points:** the judging sheet puts 40 points on *benchmark evaluation* and 25 on *business value*. This plan is built so that every question has a provably-correct answer (for the 40) and a clear workflow benefit (for the 25).

---

## 1. The five questions and EXACTLY how each is calculated

All five are computed from two things we already have for every patch:

- **its label** — `1` (tumor) or `0` (normal); for scoring this is the pathologist's label, for answers it's our model's prediction
- **its position on the slide** — `x_coord`, `y_coord` (the top-left pixel location of the patch in the full slide), from `metadata.csv`

A slide is just a collection of patches that share the same `(patient, node)` identity. Group patches by `(patient, node)` and you have one slide.

### Q1 — "Is there any tumor on this slide?"
**Calculation:** Is `max(labels) == 1`? If any patch on the slide is tumor, the answer is **yes**.
**Only needs:** labels.
**Clinical meaning:** is this lymph node involved (positive) or clear (negative). The single most important question.

### Q2 — "What fraction of the slide is tumor?" (tumor burden)
**Calculation:** `tumor_fraction = (# patches labeled 1) / (total # patches on the slide)`.
Convert to area: each patch covers a known physical size, so `tumor_area_mm2 ≈ (# tumor patches) × (area of one patch in mm²)`.
**Only needs:** labels (+ patch physical size, a constant).
**Clinical meaning:** "a few scattered cells" vs. "node mostly replaced by tumor" — very different for the patient.

### Q3 — "How large is the biggest tumor focus?"
**Calculation (this is the key geometric one):**
1. Place every tumor patch on a grid using its `(x_coord, y_coord)`.
2. Tumor patches that touch each other (neighbors up/down/left/right) form one **connected cluster** = one focus. Finding these is a standard operation called *connected components* (`scipy.ndimage.label` does it in one call).
3. Measure the biggest cluster's width in millimeters (how many patches across × patch size).
4. Bucket it into the real clinical size categories:
   - **< 0.2 mm** → isolated tumor cells (ITC)
   - **0.2–2.0 mm** → micrometastasis
   - **> 2.0 mm** → macrometastasis
**Only needs:** labels + positions.
**Clinical meaning:** lymph node metastases are officially categorized by size. This is the most clinically credible question we can answer from this data.

### Q4 — "How many separate tumor foci are there?"
**Calculation:** the number of connected clusters found in Q3. (`scipy.ndimage.label` returns this count directly.)
**Only needs:** labels + positions.
**Clinical meaning:** one focus vs. several scattered deposits is a meaningful difference, and counting them by eye is error-prone.

### Q5 — "Where is the tumor, and where should I look first?"
**Calculation, two parts:**
- **Location:** average the `(x_coord, y_coord)` of all tumor patches, then describe which region of the tissue that falls in (e.g., upper-left, center). Simple comparison against the slide's bounding box.
- **Look-here ranking:** sort patches by the **model's probability score** (how confident it is each patch is tumor), highest first. The top patches are where the pathologist should look first.
**Needs:** positions (for location) + the model's probability output (for ranking).
**Clinical meaning:** points the eye straight to the suspicious regions on a huge slide. This is the biggest time-saver and the best demo moment.

> **Important data note:** per-patch tumor/normal (Q1) works from labels alone, but the slide-level questions (Q2–Q5) need each patch's **position**, which lives in the `metadata.csv` of the official `wilds` download. Use `pip install wilds`, not the Hugging Face image-only mirror, so you get `x_coord` and `y_coord`.

---

## 2. How the whole system fits together

```
         ┌─────────────────────────────────────────────────┐
         │  One slide = all patches for a (patient, node)   │
         └─────────────────────────────────────────────────┘
                              │
          each 96×96 patch    ▼
   ┌──────────────────────────────────────────┐
   │ STEP A: Foundation model → embedding      │  (needs GPU once; use Kaggle free tier)
   │  each patch → a list of numbers           │
   └──────────────────────────────────────────┘
                              │
                              ▼
   ┌──────────────────────────────────────────┐
   │ STEP B: Small classifier → tumor prob.    │  (trains in seconds on a laptop)
   │  embedding → probability of tumor (0–1)   │
   └──────────────────────────────────────────┘
                              │
          put patches back    ▼
          on the slide grid
   ┌──────────────────────────────────────────┐
   │ STEP C: Compute the 5 answers             │  (pure counting + geometry, laptop)
   │  burden %, focus count, biggest focus mm, │
   │  location, look-here ranking              │
   └──────────────────────────────────────────┘
                              │
                              ▼
   ┌──────────────────────────────────────────┐
   │ STEP D: Small language model writes the   │  (API call, or template to start)
   │  answer in plain English from the numbers │
   └──────────────────────────────────────────┘
```

**The split of labor that makes this GPU-free:**
- Embeddings (Step A) are the only GPU step. Compute them **once** on Kaggle's free GPU, save to a file, done.
- Everything else (B, C, D) runs on a normal laptop in seconds.

---

## 3. What to prepare BEFORE the event (do this week)

1. **Team Kaggle accounts.** Everyone creates one and verifies their phone to unlock free GPU hours. (More accounts = more total GPU quota.)
2. **Test the data download.** In a Kaggle notebook (internet enabled): `pip install wilds`, then load `camelyon17`. Confirm you can read `metadata.csv` and see columns `patient, node, center, x_coord, y_coord, tumor`.
3. **Pick + request a foundation model.** Try UNI or Phikon on Hugging Face (some need an access request that can take a day). Fallback that needs no approval: **DINOv2**.
4. **Time a small run.** Embed 500 patches on Kaggle, note how long it takes, multiply to estimate the full run.
5. **Read the event rules** on what can be prepared beforehand. If pre-work isn't allowed, just rehearse so the on-the-day run is smooth.
6. **Confirm patch physical size.** Camelyon patches are sampled around 20× magnification; find the microns-per-pixel in the dataset docs so your mm conversions (Q2, Q3) are right. Write the number down.

---

## 4. The hackathon-day timeline (10:30 AM – 4:30 PM)

> Rule for the day: **every block ends with a number saved to a results file.** If something breaks, you still have a story.

### 10:30–11:00 · Lock the plan + data splits
- **Train** on hospitals/centers {0, 3, 4}. **Test** on the unseen center {2}. (This is the "new hospital" test the judges want — held-out.)
- Decide how many patches: with a GPU, ~20k train / ~10k test. CPU-only, ~10k / ~5k.
- Group patches by `(patient, node)` into slides; keep a list of which slides are in test.
- **Output:** fixed patch-ID lists saved to disk so numbers don't drift.

### 11:00–12:00 · Embeddings (the one GPU step)
- Run every selected patch through the **foundation model** AND an **ImageNet/ResNet baseline**. Save both sets of embeddings to files.
- In parallel: one teammate sets up the **frontier baseline** — send ~200 test patches to a frontier vision model via **Fireworks** with a prompt like *"Does this lymph node tissue contain metastatic tumor? Answer TUMOR or NORMAL."* Log answer, time, tokens.
- **Output:** embedding files on disk; frontier baseline running.

### 12:00–12:45 · Train classifier + first real numbers
- Train a logistic-regression (or small MLP) classifier on the foundation-model embeddings. Do the same on the ImageNet embeddings.
- Score both on the **test center**. Record tumor-detection accuracy, sensitivity, specificity **with 95% confidence intervals** (bootstrap — see §5).
- **Output:** first results-table row: frontier vs. ImageNet vs. ours, with CIs. This alone is a credible submission.

### 12:45–2:15 · Build the Q&A engine (the differentiator)
- Write the function that takes a slide's patch predictions + positions and returns the five answers (§1). Use `scipy.ndimage.label` for Q3/Q4.
- Build the **question benchmark**: for each test slide, compute the *true* answers from the pathologist labels. These are your ground-truth Q&A pairs (aim for 100+ — you'll have hundreds).
- Run your **model's** predictions through the same engine → *predicted* answers.
- Score predicted vs. true: Q1 accuracy, Q3 size-category accuracy, Q4 focus-count error, etc.
- **Output:** Q&A accuracy table, model vs. frontier, on 100+ held-out questions with CIs.

### 2:15–3:00 · Cost + efficiency (keep quality honest)
- Measure **time per slide** and **cost per slide** on a normal CPU for your model vs. the frontier model.
- If you have GPU time, also test a smaller backbone and/or lower precision; always report the **quality you kept** next to the cost you saved (the sheet warns: *efficiency gains must retain useful quality*).
- **Output:** cost/latency row for each system.

### 3:00–3:45 · Demo + answer composer
- Step D: turn computed numbers into a written answer. Start with a template ("This node is **positive**. Tumor burden **7.5%**. Largest focus **3.1 mm (macrometastasis)**. **2 foci**, mainly upper-left."), then, if time, have a small Fireworks model phrase it more naturally.
- Build the demo: show a **tumor heatmap** on one test slide (color each patch by predicted probability), then type a question and show the answer. A notebook is fine — the sheet says a notebook or API can earn full marks.
- **3:45 = feature freeze.**

### 3:45–4:15 · Slides + rehearsal
Six slides (see §6). Rehearse once with a timer — the demo must fit **under 120 seconds, ideally under 60**.

### 4:15–4:30 · Submit
- Push code to a **public GitHub repo** with a README (setup + how to reproduce the eval). That's the 5-point documentation criterion outright.
- Record a 60–90s backup screen video of the demo.
- Submit by **4:20**, not 4:29.

---

## 5. The 40-point requirement, handled precisely

The sheet asks for: *meaningful, reliable gains on held-out cases, 100+ examples, confidence intervals or statistical tests.* Do all four:

- **Held-out:** always report the **test center {2}** numbers, never training data.
- **100+ examples:** you'll have hundreds of test slides/questions — **state the exact count** on the slide.
- **Confidence intervals (this is what most teams skip):** report every headline number as `value (95% CI: low–high)`. Simple bootstrap recipe:
  1. You have N test items with correct/incorrect (or error) per item.
  2. Randomly resample N items *with replacement*, recompute the metric. Repeat 1,000 times.
  3. The 2.5th and 97.5th percentiles of those 1,000 values are your 95% CI.
  (`numpy` does this in a few lines — no stats library needed.)
- **Statistical test for "we beat the baseline":** since each test item is scored by both your model and the baseline, use a **paired test** (e.g., McNemar's test for accuracy, or a bootstrap of the *difference*). Report the p-value or the CI of the difference so you can say the gain is real, not noise.

Also follow the fine print: **same task, same test cases** for every system; **disclose quality, cost, and latency tradeoffs**; and never show a cheaper number without the quality beside it.

---

## 6. The six-slide deck (maps to the rubric)

1. **Problem + user** — a pathologist hunting small metastases and measuring by hand. *(business value)*
2. **What we built** — open foundation model + classifier + Q&A engine, running locally. *(technical execution)*
3. **Does it work? Tumor detection** — frontier vs. ImageNet vs. ours on the unseen center, with CIs. *(benchmark — 40)*
4. **Does it work? Q&A** — accuracy on the 5 questions over 100+ held-out slides, vs. frontier, with CIs. *(benchmark — 40)*
5. **Business value** — "% tissue skipped at 99% sensitivity," cost/slide, runs on a laptop with no data leaving the hospital. *(business — 25)*
6. **Tradeoffs + limits** — what got worse, decision-support not diagnosis, patch-level resolution. *(credibility — lifts every rating)*

---

## 7. Why this is useful to pathologists (for the business slide)

- **Triage:** jumps straight to suspicious regions on a huge slide instead of scanning acres of normal tissue.
- **Consistency:** tumor %, focus count, and size measured by eye vary between people and days; an automated measure is a steady second opinion.
- **Speed on routine questions:** "how much, how big, how many, where" answered in seconds.
- **Privacy + cost:** runs locally → sensitive slides never leave the hospital, and there's no per-image API bill. This is the concrete advantage over a frontier API, and it's exactly the "specialized open model beats a frontier baseline on a valuable task" framing the sheet rewards.

**Framing to keep it credible:** a decision-support and measurement tool that a pathologist reviews — never automated diagnosis.

---

## 8. Honest limitations (say these out loud — they raise your rating)

- Patch-level resolution (~96px squares), so very small lesions near the detection limit are hard — report ITC performance separately.
- Labels are binary tumor/normal; we compute burden/size/foci from geometry, we do **not** read cell detail, grade, or subtype.
- The "new hospital" drop is real — show it, and show how a few local labels recover it if you have time.
- Synthetic-free but single-dataset: results are on Camelyon17; other scanners/stains would need revalidation.

---

## 9. If things go wrong (fallbacks)

| Problem | Fallback |
|---|---|
| Foundation model access not approved | Use **DINOv2** (no approval) as "our model" |
| No usable GPU anywhere | Shrink to ~5k train / ~2.5k test, smaller backbone, CPU; the "no-GPU" story gets *stronger* |
| No Fireworks credits | Run frontier baseline on ~100 patches only, and say so |
| Q&A engine runs long | Ship Q1, Q3, Q4 (crisp correct answers); demo Q5; drop Q2 if needed |
| Answer composer (LLM) breaks | Use the plain template — judges score the numbers, not the prose |

---

## 10. What to build with Claude next

1. **`eval_harness.py`** — loads predictions, computes accuracy/sensitivity/specificity + **bootstrap 95% CIs** + a paired test vs. baseline. *(the literal 40-point requirement)*
2. **`slide_questions.py`** — takes a slide's patch labels + `x_coord/y_coord` and returns all five answers (burden %, biggest focus mm + category, focus count, location, look-here ranking). *(the Q&A engine)*
3. **`make_benchmark.py`** — builds the 100+ ground-truth Q&A pairs from the real labels, so day-of you just run and score.

All three run on a laptop with no GPU.
