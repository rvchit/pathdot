# pathfrontier + cello: interface contracts

Every component is built by a different person/agent in parallel. These contracts are the
only thing that must line up. Read `pathfrontier/schema.py` and `pathfrontier/config.py` first;
they already exist and are the source of truth for `Prediction`, paths and constants.

Repo layout (monorepo):

```
pathfrontier/          Python package (ML: data, encoders, probe, VLM, cascade, metrics, serve)
scripts/               CLI entry points (typer); each script is runnable as `uv run python scripts/x.py`
tests/                 pytest; MUST run offline in < 60 s using synthetic fixtures (no downloads, no API keys)
cello/                 the slide viewer (React/Vite frontend, Express backend, Lambda tiler) — user's repo
results/               preds_*.jsonl, metrics_*.json, frontier.png, heatmaps/<imageId>.json
cache/                 emb_*.npy embedding caches, downloaded models
data/                  datasets (parquet from HF), sample slides
tiles/                 local DZI pyramids: tiles/<imageId>/<imageId>.dzi + tiles/<imageId>_files/<level>/<x>_<y>.jpeg
```

Python: 3.11, env at `.venv` (uv). Run anything with `uv run python ...` or `.venv/bin/python`.
Device: `pathfrontier.config.device()` -> "cuda" | "mps" | "cpu". Dev machine is an M3 Pro (MPS, 18 GB).

## 1. Data (`pathfrontier/data.py`)

```python
@dataclass
class TileRecord:
    tile_id: str          # f"{dataset}:{split}:{index}"
    image: PIL.Image.Image  # RGB
    label: int            # index into schema.CLASSES[dataset]

def load_split(dataset: Literal["pcam","crc"], split: Literal["train","valid","test"],
               limit: int | None = None, seed: int = 0, balanced: bool = False) -> list[TileRecord]
def iter_split(...same args...) -> Iterator[TileRecord]        # streaming, for big runs
def download(dataset, splits=("valid","test")) -> None           # HF parquet -> data/<dataset>/<split>.parquet
def make_synthetic(dataset="pcam", n=64, seed=0) -> list[TileRecord]   # offline fixture for tests/demos
```
Sources: HF `1aurent/PatchCamelyon` (splits train/valid/test, columns image, label:bool, 96x96) and
HF `1aurent/NCT-CRC-HE` (splits NCT_CRC_HE_100K -> "train", CRC_VAL_HE_7K -> "test"; "valid" = a seeded 10% holdout
of train; 224x224, label ClassLabel). `limit` takes the first N after a seeded shuffle; `balanced` gives equal classes.
Default hackathon sizes: pcam train(valid split)=8192 for probe, test=4096 for eval; baseline VLM on test subset of 500.

## 2. Encoders (`pathfrontier/encoders.py`)

```python
ENCODERS = {  # name -> spec (hub id, loader kind, input size, embed dim, gated?)
  "phikon-v2": ..., "uni": ..., "uni2-h": ..., "conch": ..., "virchow2": ..., "h-optimus-0": ...,
  "dinov2-base": ...  (non-pathology control, ungated), "resnet50": ... (ImageNet control, ungated)
}
class Encoder:
    name: str; dim: int
    def embed(self, images: list[PIL.Image.Image], batch_size: int = 64) -> np.ndarray  # (N, dim) float32, L2-normalised = False
def get_encoder(name: str, device: str | None = None) -> Encoder
```
Cache: `scripts/embed.py --encoder phikon-v2 --dataset pcam --split test --limit 4096` writes
`schema.embedding_paths(...)` -> X.npy (N,dim) float32, labels .npy int64, ids .json list[str].
Also records throughput (tiles/s) and device to `cache/emb_<encoder>_<dataset>_<split>.meta.json`
`{"tiles_per_s":..., "device":..., "n":..., "dim":..., "seconds":...}` — the cost model uses this.

## 3. Linear probe (`pathfrontier/probe.py`)

```python
class LinearProbe:
    def __init__(self, encoder_name: str, dataset: str, C: float = 1.0, calibrate: bool = True)
    def fit(self, X: np.ndarray, y: np.ndarray) -> "LinearProbe"         # sklearn LogisticRegression (+ isotonic/Platt calibration on a held-out 20%)
    def predict_proba(self, X: np.ndarray) -> np.ndarray                 # (N, n_classes)
    def save(self, path: Path) -> None; @classmethod def load(cls, path) -> "LinearProbe"   # joblib
    def predict_records(self, X, y, ids, split, latency_ms_per_tile, cost_per_tile) -> list[Prediction]  # stage="probe", model=f"probe/{encoder_name}"
```
Default artifact path: `results/probe_<encoder>_<dataset>.joblib`. `scripts/train_probe.py` trains on the cached
train/valid embeddings, evaluates on test, writes `results/preds_probe_<encoder>_test.jsonl` and prints metrics.

## 4. VLM client (`pathfrontier/vlm_client.py`)

```python
@dataclass
class VLMConfig: model: str; base_url: str = config.FIREWORKS_BASE_URL; api_key: str = ...; max_tokens: int = 200; temperature: float = 0.0; concurrency: int = 8; price_in_per_m: float; price_out_per_m: float
PRICES = { "<fireworks model id>": (in_per_M_tokens, out_per_M_tokens), ... }   # live list researched by the builder; include a comment with the source URL + date
def classify_tile(image: PIL.Image, dataset: str, cfg: VLMConfig, prompt_version: str = "v1") -> dict
    # returns {"label": int, "probs": list[float], "reason": str, "latency_ms": float, "cost_usd": float, "raw": str, "tokens_in": int, "tokens_out": int}
    # label parse MUST be robust: ask the model for strict JSON {"label": "<class>", "confidence": 0-1, "reason": "<=25 words"}; fall back to regex; unknown -> uniform probs + extra["parse_error"]=True
async def classify_many(records, dataset, cfg, prompt_version, cache_path) -> list[Prediction]  # concurrency via asyncio.Semaphore, exponential backoff on 429/5xx, resumable JSONL cache keyed by (model, prompt_version, tile_id)
```
The OpenAI-compatible endpoint is used (`openai.AsyncOpenAI(base_url=..., api_key=...)`), images sent as base64 data URLs,
upscaled to >= 224 px so the vision tokenizer gets enough pixels. Works for ANY OpenAI-compatible host (Fireworks, a
fine-tuned Fireworks deployment, or local vLLM). Anthropic baseline optional via `provider="anthropic"`.
`scripts/run_baseline.py --model <id> --dataset pcam --split test --limit 500` -> `results/preds_<model>_test.jsonl` (stage="baseline").
`scripts/run_vlm_ft.py` is the same but stage="vlm_ft" and model = the fine-tuned deployment id.

## 5. Fine-tune (`pathfrontier/finetune.py`, `scripts/prepare_sft.py`, `scripts/launch_finetune.py`)

Prepares a Fireworks vision SFT JSONL (`data/sft_<dataset>_train.jsonl`) from `load_split(..., balanced=True)`:
each line = chat `messages` with a user turn containing the tile image (base64 data URL or uploaded URL per current
Fireworks docs) + instruction, and an assistant turn with the strict JSON answer (label + one-sentence
rationale templated per class, e.g. "Sheets of cells with enlarged, hyperchromatic nuclei and loss of normal
lymph-node architecture" for tumor). Launches the job via Fireworks REST / `firectl`, polls status, and prints the
deployable model id. Must degrade gracefully: if no API key, write the JSONL and print the exact manual commands.

## 6. Metrics + eval (`pathfrontier/metrics.py`, `scripts/eval.py`)

```python
def compute_metrics(preds: list[Prediction], dataset: str, specificity_points=(0.90,0.95,0.99), tiles_per_slide=10_000) -> dict
# keys: n, auroc (binary: positive class = POSITIVE_INDEX[dataset]; multiclass: one-vs-rest macro AND tumor-vs-rest),
#       balanced_accuracy, accuracy, sensitivity, specificity, f1, sensitivity_at_spec{0.90,0.95,0.99},
#       threshold_at_spec{...}, latency_p50_ms, latency_p95_ms, latency_mean_ms,
#       cost_per_1k_usd, cost_per_slide_usd, coverage (fraction not routed to pathologist; 1.0 unless cascade),
#       pathologist_minutes_per_slide (routed fraction * tiles_per_slide * PATHOLOGIST_SEC_PER_TILE / 60),
#       confusion (list of lists), per_class (dict)
def roc_points(preds, dataset) -> (fpr, tpr, thr)
```
`scripts/eval.py results/preds_*.jsonl` prints a comparison table (rich/markdown) and writes `results/metrics_<model>.json` per file plus `results/leaderboard.md`.
Pathologist-routed tiles (stage="pathologist" / routed_to="pathologist") count as CORRECT for sensitivity (a human reads
them) but are excluded from "automated" accuracy; report both `sensitivity_automated` and `sensitivity_system`.

## 7. Cascade (`pathfrontier/cascade.py`, `scripts/sweep.py`, `scripts/plot_frontier.py`)

```python
@dataclass
class CascadeConfig:
    low: float = 0.05     # p_tumor below -> auto-clear negative (stage 1 decides)
    high: float = 0.95    # p_tumor above -> auto-flag positive (stage 1 decides)
    fallback: Literal["vlm_ft", "pathologist", "none"] = "vlm_ft"  # what handles the uncertain band
    vlm_cost_per_tile: float; vlm_latency_ms: float; probe_cost_per_tile: float; probe_latency_ms: float
def route(probe_preds: list[Prediction], vlm_preds: dict[str, Prediction] | None, cfg: CascadeConfig) -> list[Prediction]
    # output stage="cascade", routed_to in {"probe","vlm_ft","pathologist"}, latency/cost summed per tile; for routed tiles with no VLM prediction available use the pathologist path
def sweep(probe_preds, vlm_preds, dataset, grid: list[tuple[low,high]] | None, fallback) -> pandas.DataFrame
    # one row per (low, high): metrics from compute_metrics + fraction_routed + cost_per_slide + sensitivity_system
```
`scripts/plot_frontier.py` draws THE chart: x = cost per slide (USD, log scale), y = sensitivity at the operating
point; the cascade sweep is a curve (Pareto front highlighted), the baseline VLM(s), probe-alone and fine-tuned-VLM-alone
are single labelled points; a second panel shows pathologist minutes per slide vs sensitivity. Saves
`results/frontier.png` and `results/frontier.json` (the points). Follow the `dataviz` skill palette rules.

## 8. ML inference service (`pathfrontier/serve.py`, FastAPI, port 8008)

```
GET  /health                      -> {"status":"ok","encoder":..., "probe":..., "vlm": <model or null>, "device":...}
POST /predict/tile   (multipart image file, or JSON {"image_b64":...})
                                  -> {"p_tumor": float, "probs": [...], "classes": [...], "decision": "negative|positive|review",
                                      "routed_to": "probe|vlm_ft|pathologist", "reason": str|null, "latency_ms": float, "cost_usd": float}
POST /predict/slide  JSON {"image_id": str, "level": int|null, "max_tiles": int|null, "low": float, "high": float, "fallback": str}
                                  -> starts a background job; returns {"job_id":..., "status":"running"}
GET  /jobs/{job_id}               -> {"status":"running|done|error","progress":0..1,"result": <heatmap> | null}
GET  /heatmap/{image_id}          -> the last saved heatmap (results/heatmaps/<image_id>.json) or 404
```
Heatmap JSON (consumed by the cello frontend overlay):
```json
{"image_id":"CMU-1-Small-Region","width":2220,"height":2967,"tile_size":256,"level":11,"max_level":12,
 "scale":2,          // image pixels per tile pixel at this level = 2^(max_level-level)
 "cols":5,"rows":6,
 "tiles":[{"x":0,"y":0,"p_tumor":0.03,"decision":"negative","routed_to":"probe","reason":null}, ...],
 "summary":{"n_tiles":30,"auto_cleared":22,"auto_flagged":3,"review":5,"frac_auto_cleared":0.73,
            "cost_usd":0.004,"latency_ms_total":812,"pathologist_minutes_saved":...,"model":"probe/phikon-v2 + vlm_ft"}}
```
The service reads the local DZI pyramid at `tiles/<image_id>_files/<level>/<x>_<y>.jpeg` (env `PATHFRONTIER_TILES_DIR`,
default `tiles/`); it never touches S3. Tile coordinates in the heatmap are DZI tile indices at `level`.
Overlay rectangle in image pixels: `[x*tile_size*scale, y*tile_size*scale, tile_size*scale, tile_size*scale]`.

## 9. Slide tooling (`scripts/tile_slide.py`, `scripts/make_mosaic_slide.py`)

`tile_slide.py <image> --out tiles/` makes a DZI pyramid (tile 256, overlap 0, jpeg) with pyvips when importable
(handles .svs/.tiff/.ndpi), else a pure-Pillow pyramid for png/jpg/tif. Writes `<id>.dzi` (XML, Format="jpeg",
Overlap="0", TileSize="256", Size Width/Height) and `<id>_files/<level>/<x>_<y>.jpeg` using DZI level numbering
(level 0 = 1x1 px, max level = full res). `make_mosaic_slide.py --dataset pcam --grid 24` stitches a seeded
grid of real test tiles into one PNG "slide" + a ground-truth JSON (`data/slides/pcam_mosaic_24.gt.json`:
per-cell label) and tiles it, so the heatmap can be shown next to truth with zero AWS/vips dependencies.

## 10. Cello backend additions (Express, `cello/backend`)

* `STORAGE_MODE=local|s3` (default local). Local mode serves `GET /osd/tile/:imageId/:level/:x/:y` from
  `TILES_DIR` (default `../tiles`) and never constructs an S3 client; s3 mode is the existing behaviour.
* `GET /osd/dzi/:imageId` -> `{width,height,tileSize,overlap,format,maxLevel}` parsed from the .dzi (local) or S3.
* `GET /osd/images` -> list of available imageIds (local dir scan, or S3 prefix list).
* `GET /comments/:imageId`, `POST /comments` `{imageId, annotationId, comment, geometry?}`, `DELETE /comments/:id`
  backed by a JSON file store (`data/comments.json`) when DATABASE_URL is unset, Prisma when set.
* `ALL /ml/*` proxied to `ML_SERVICE_URL` (default http://localhost:8008), so the frontend only talks to :4001.
* `GET /health`.
Frontend: `VITE_BACKEND_URL` (default http://localhost:4001); viewer reads /osd/dzi for dimensions; image list from
/osd/images; "Run AI triage" button -> POST /ml/predict/slide, poll /ml/jobs/{id}, then paint the heatmap overlay
(canvas/SVG overlay on OpenSeadragon, red=positive, amber=review, transparent/green=cleared, toggle + opacity slider),
sidebar panel with the summary numbers, click a tile to see p_tumor / routed_to / reason.
