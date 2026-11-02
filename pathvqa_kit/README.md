# PathVQA fine-tune: Qwen2.5-VL on Fireworks

Goal: beat frontier models on the PathVQA test set by fine-tuning Qwen2.5-VL on PathVQA's train split plus caption-grounded synthetic Q&A.

**Published comparison (Lingshu, arXiv 2506.07044):** GPT-4.1 55.5 · Gemini-2.5-Flash 55.4 · Claude Sonnet 4 54.2 · Qwen2.5-VL-7B 44.1 · Lingshu-7B (fine-tuned Qwen2.5-VL) 61.9.
Check these against the paper's table. Its exact PathVQA protocol isn't stated, so first confirm that your base Qwen2.5-VL-7B score comes out near 44.

## Setup (on a machine that can reach huggingface.co and commons.wikimedia.org)

```bash
pip install -r requirements.txt
cd scripts
```

## Run order

| # | Step | Command | Output |
|---|---|---|---|
| 1 | Download PathVQA, normalize images (max 512 px JPEG), hash test images, report train/test image overlap | `python prepare_pathvqa.py --out ../data` | `data/pathvqa_{train,validation,test}.jsonl`, `data/test_phash.json` |
| 2 | Baseline and frontier evals (can run while steps 3–5 run) | see **Eval** below | `results/*.jsonl` |
| 3 | Collect openly licensed, captioned histopathology images from Wikimedia Commons | `python source_images.py --out ../data --max 1500` (put your email in the User-Agent string first) | `data/commons_images.jsonl` |
| 4 | Remove new images that match a PathVQA test image, or duplicate each other | `python dedupe.py --images ../data/commons_images.jsonl --test-hashes ../data/test_phash.json --out ../data/commons_clean.jsonl` | cleaned list |
| 5 | Claude writes PathVQA-style Q&A grounded in each image's caption | `export ANTHROPIC_API_KEY=...`<br>`python generate_qa.py --images ../data/commons_clean.jsonl --out ../data/synthetic_qa.jsonl` | `data/synthetic_qa.jsonl` |
| 6 | Build the Fireworks datasets | `python build_fireworks_jsonl.py --data ../data --synthetic ../data/synthetic_qa.jsonl --out ../fireworks` | `ft_pathvqa_only.jsonl`, `ft_pathvqa_plus_synth.jsonl`, `ft_val.jsonl` |
| 7 | Fine-tune and deploy on Fireworks | see **Fine-tune** below | deployed model |
| 8 | Evaluate the fine-tuned model on the same test questions and compare | see **Eval** | results table |

Spot-check about 30 synthetic Q&A by hand before step 6. If the answers drift from the captions, tighten the prompt in `generate_qa.py`.

## Fine-tune (Fireworks)

```bash
firectl model get -a fireworks <qwen2.5-vl-7b-or-32b-id>      # confirm "Tunable: true"
firectl dataset create pathvqa-plus-synth ../fireworks/ft_pathvqa_plus_synth.jsonl
firectl dataset create pathvqa-val ../fireworks/ft_val.jsonl
firectl sftj create --base-model <MODEL_ID> --dataset pathvqa-plus-synth \
  --evaluation-dataset pathvqa-val --output-model pathvqa-qwen-synth \
  --epochs 1 --lora-rank 16 --learning-rate 0.0001
firectl sftj get <JOB_ID>
firectl deployment create pathvqa-qwen-synth --deployment-shape default
```

- **Separate the effect of the synthetic data.** If credits allow, also train on `ft_pathvqa_only.jsonl`. Then you can report base → PathVQA-only fine-tune → PathVQA + synthetic fine-tune.
- **Size and speed.** The full PathVQA train file is about 1 GB, which is over the 500 MB UI limit, so upload it with firectl. For a fast first run, use `--max-train 5000` in step 6.
- **7B vs. 32B.** Start with 7B: it's cheaper and quicker, and it's the model the published baseline used.

## Eval

```bash
export FIREWORKS_API_KEY=...
python eval.py run --model <base qwen2.5-vl model id> --name base --data ../data --n 1000
python eval.py run --provider anthropic --model claude-sonnet-4-5 --name frontier --data ../data --n 1000
python eval.py run --model <deployed fine-tuned model id> --name ft_synth --data ../data --n 1000
python eval.py score results/base.jsonl results/frontier.jsonl results/ft_synth.jsonl
```

- **Seeded subset.** `--n 1000` picks the same seeded test questions for every model. Use `--n 0` for the full test set (~6.7k questions).
- **What's reported:**
  - yes/no accuracy
  - open-question exact match and token recall
  - overall exact match
  - 95% bootstrap confidence intervals
  - an exact McNemar test for each pair of models
  - latency p50/p95 and output tokens
- **Resuming.** Reruns pick up where they stopped.
- **Same inputs for every model.** All models get the same system prompt (`common.py`) and the same 512 px images.

## Data integrity notes

- `prepare_pathvqa.py` reports how many PathVQA train images duplicate test images. `build_fireworks_jsonl.py` removes those questions from training. Mention this on your limitations slide.
- Synthetic images are restricted to CC0, public-domain, CC BY and CC BY-SA files. License, author and source URL are kept on every synthetic row, so you can attribute them.
- Synthetic labels are AI-written from expert captions and haven't been reviewed by a pathologist. Present the model as research and decision support, not diagnosis.
