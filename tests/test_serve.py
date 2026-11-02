"""Offline tests for pathfrontier.serve in PATHFRONTIER_FAKE=1 mode (no models, no API keys)."""
from __future__ import annotations

import base64
import io
import time
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from pathfrontier import dzi, serve


def tissue_image(w: int = 600, h: int = 400, seed: int = 0) -> Image.Image:
    """H&E-ish purple/pink noise so no tile is classified as background."""
    rng = np.random.default_rng(seed)
    base = np.array([190, 110, 170], dtype=np.float32)
    arr = base + rng.normal(0, 30, size=(h, w, 3))
    return Image.fromarray(np.clip(arr, 0, 255).astype("uint8"), "RGB")


def png_bytes(im: Image.Image) -> bytes:
    buf = io.BytesIO()
    im.save(buf, "PNG")
    return buf.getvalue()


@pytest.fixture(scope="module")
def env(tmp_path_factory):
    root = tmp_path_factory.mktemp("serve")
    tiles = root / "tiles"
    dzi.tile_image_pillow(tissue_image(600, 400), tiles, "synth")            # 3x2 at max level
    white = Image.new("RGB", (300, 300), (250, 250, 250))
    dzi.tile_image_pillow(white, tiles, "blank")                               # 2x2 at level 9
    settings = serve.Settings(fake=True, encoder="fake", tiles_dir=tiles,
                              results_dir=root / "results", low=0.05, high=0.95)
    return {"root": root, "tiles": tiles, "settings": settings}


@pytest.fixture(scope="module")
def client(env):
    app = serve.create_app(env["settings"])
    with TestClient(app) as c:
        yield c


def wait_job(client: TestClient, job_id: str, timeout: float = 30.0) -> dict:
    t0 = time.time()
    while time.time() - t0 < timeout:
        r = client.get(f"/jobs/{job_id}")
        assert r.status_code == 200
        j = r.json()
        assert 0.0 <= j["progress"] <= 1.0
        if j["status"] in {"done", "error"}:
            return j
        time.sleep(0.05)
    raise AssertionError("job did not finish")


def test_health(client, env):
    r = client.get("/health")
    assert r.status_code == 200
    j = r.json()
    assert j["status"] == "ok"
    assert j["encoder"] == "fake" and j["fake"] is True
    assert j["vlm"] is None
    assert j["device"] in {"cpu", "mps", "cuda"}
    assert j["classes"] == ["normal", "tumor"]
    assert set(j["images"]) == {"synth", "blank"}


def test_predict_tile_multipart_and_json(client):
    im = tissue_image(96, 96, seed=3)
    r = client.post("/predict/tile", files={"file": ("t.png", png_bytes(im), "image/png")})
    assert r.status_code == 200, r.text
    j = r.json()
    assert set(j) >= {"p_tumor", "probs", "classes", "decision", "routed_to", "reason",
                      "latency_ms", "cost_usd"}
    assert 0.0 <= j["p_tumor"] <= 1.0
    assert abs(sum(j["probs"]) - 1.0) < 1e-4
    assert j["classes"] == ["normal", "tumor"]
    assert j["decision"] in {"negative", "positive", "review"}
    assert j["routed_to"] in {"probe", "pathologist"}
    assert j["cost_usd"] > 0

    b64 = base64.b64encode(png_bytes(im)).decode()
    r2 = client.post("/predict/tile", json={"image_b64": b64})
    assert r2.status_code == 200, r2.text
    assert r2.json()["p_tumor"] == pytest.approx(j["p_tumor"], abs=1e-6)  # deterministic

    # Thresholds can be overridden per call: everything in band -> pathologist review.
    r3 = client.post("/predict/tile", json={"image_b64": b64, "low": 0.0, "high": 1.0})
    assert r3.json()["decision"] == "review" and r3.json()["routed_to"] == "pathologist"
    r4 = client.post("/predict/tile", json={"image_b64": b64, "low": 1.0, "high": 1.0})
    assert r4.json()["decision"] == "negative" and r4.json()["routed_to"] == "probe"

    assert client.post("/predict/tile", json={}).status_code == 400
    assert client.post("/predict/tile", json={"image_b64": "bm90YW5pbWFnZQ=="}).status_code == 400


def test_predict_slide_job_and_heatmap(client, env):
    r = client.post("/predict/slide", json={"image_id": "synth", "level": "max"})
    assert r.status_code == 200, r.text
    job_id = r.json()["job_id"]
    assert r.json()["status"] == "running"

    j = wait_job(client, job_id)
    assert j["status"] == "done", j.get("error")
    hm = j["result"]
    assert hm["image_id"] == "synth"
    assert (hm["width"], hm["height"]) == (600, 400)
    assert hm["tile_size"] == 256 and hm["max_level"] == 10
    assert hm["level"] == 10 and hm["scale"] == 1
    assert (hm["cols"], hm["rows"]) == (3, 2)
    assert len(hm["tiles"]) == 6
    assert {(t["x"], t["y"]) for t in hm["tiles"]} == {(x, y) for y in range(2) for x in range(3)}
    for t in hm["tiles"]:
        assert set(t) >= {"x", "y", "p_tumor", "decision", "routed_to", "reason"}
        assert t["decision"] in {"negative", "positive", "review"}   # tissue image: no background
        assert 0.0 <= t["p_tumor"] <= 1.0
    s = hm["summary"]
    assert s["n_tiles"] == 6
    assert s["auto_cleared"] + s["auto_flagged"] + s["review"] + s["vlm_decided"] == 6
    assert s["frac_auto_cleared"] == pytest.approx(s["auto_cleared"] / 6)
    assert s["pathologist_minutes_saved"] == pytest.approx(s["auto_cleared"] * 2.0 / 60)
    assert s["cost_usd"] > 0 and s["latency_ms_total"] > 0
    assert s["model"] == "probe/fake + pathologist"

    # Saved to results/heatmaps/<image_id>.json and served by /heatmap.
    assert (env["settings"].results_dir / "heatmaps" / "synth.json").is_file()
    r = client.get("/heatmap/synth")
    assert r.status_code == 200
    assert r.json()["summary"] == s and r.json()["tiles"] == hm["tiles"]

    # Default level is max_level - 1 (2x fewer tiles per axis); "auto" is max_level - 2.
    j2 = wait_job(client, client.post("/predict/slide", json={"image_id": "synth"}).json()["job_id"])
    assert j2["result"]["level"] == 9 and j2["result"]["scale"] == 2
    assert (j2["result"]["cols"], j2["result"]["rows"]) == (2, 1)
    j3 = wait_job(client, client.post("/predict/slide",
                                      json={"image_id": "synth", "level": "auto"}).json()["job_id"])
    assert j3["result"]["level"] == 8 and j3["result"]["scale"] == 4

    # max_tiles caps the work.
    j4 = wait_job(client, client.post("/predict/slide",
                                      json={"image_id": "synth", "level": 10,
                                            "max_tiles": 2}).json()["job_id"])
    assert len(j4["result"]["tiles"]) == 2


def test_background_tiles_marked(client):
    j = wait_job(client, client.post("/predict/slide",
                                     json={"image_id": "blank", "level": "max"}).json()["job_id"])
    hm = j["result"]
    assert (hm["cols"], hm["rows"]) == (2, 2)
    assert all(t["decision"] == "background" and t["routed_to"] is None for t in hm["tiles"])
    assert hm["summary"]["n_tiles"] == 0 and hm["summary"]["background"] == 4
    assert hm["summary"]["pathologist_minutes_saved"] == 0


def test_errors(client):
    assert client.get("/jobs/nope").status_code == 404
    assert client.get("/heatmap/never_scored").status_code == 404
    assert client.post("/predict/slide", json={"image_id": "missing"}).status_code == 404
    assert client.post("/predict/slide", json={"image_id": "synth", "level": 99}).status_code == 400
    assert client.post("/predict/slide", json={"image_id": "../etc"}).status_code == 400


def test_vlm_fallback_monkeypatched(env, monkeypatch):
    """With a VLM configured, band tiles go to the VLM (fake here) instead of the pathologist."""
    calls = []

    def fake_vlm(image, settings):
        calls.append(image.size)
        return {"label": 1, "probs": [0.2, 0.8], "reason": "fake rationale",
                "latency_ms": 1.5, "cost_usd": 0.001}

    monkeypatch.setattr(serve, "_vlm_classify", fake_vlm)
    monkeypatch.setenv("FIREWORKS_API_KEY", "test-key")
    settings = serve.Settings(fake=True, encoder="fake", tiles_dir=env["tiles"],
                              results_dir=env["root"] / "results_vlm", vlm_model="fake/vlm",
                              vlm_max_tiles=4)
    with TestClient(serve.create_app(settings)) as c:
        assert c.get("/health").json()["vlm"] == "fake/vlm"
        # whole band -> VLM, capped at vlm_max_tiles=4 of the 6 tiles
        j = wait_job(c, c.post("/predict/slide", json={"image_id": "synth", "level": 10,
                                                       "low": 0.0, "high": 1.0}).json()["job_id"])
        hm = j["result"]
        assert hm["summary"]["model"] == "probe/fake + vlm_ft"
        assert hm["summary"]["vlm_decided"] == 4 and hm["summary"]["review"] == 2
        assert len(calls) == 4
        vlm_tiles = [t for t in hm["tiles"] if t["routed_to"] == "vlm_ft"]
        assert all(t["decision"] == "positive" and t["reason"] == "fake rationale"
                   and t["p_tumor"] == pytest.approx(0.8) for t in vlm_tiles)
        assert hm["summary"]["cost_usd"] == pytest.approx(0.004 + 6 * serve.PROBE_COST_PER_TILE)

        # explicit fallback="pathologist" bypasses the VLM
        j = wait_job(c, c.post("/predict/slide", json={"image_id": "synth", "level": 10,
                                                       "low": 0.0, "high": 1.0,
                                                       "fallback": "pathologist"}).json()["job_id"])
        assert j["result"]["summary"]["review"] == 6 and len(calls) == 4

        # /predict/tile uses it too
        im = tissue_image(96, 96)
        r = c.post("/predict/tile", json={"image_b64": base64.b64encode(png_bytes(im)).decode(),
                                          "low": 0.0, "high": 1.0})
        assert r.json()["routed_to"] == "vlm_ft" and r.json()["decision"] == "positive"


def test_module_app_importable():
    """`uvicorn pathfrontier.serve:app` target exists (lifespan loads models lazily)."""
    assert serve.app.title == "pathfrontier inference"
    assert Path(serve.Settings.from_env().tiles_dir).name == "tiles"


def test_vqa_fake_mode(client, env):
    """/vqa works offline in fake mode for both body shapes, caches, and 404s on missing tiles."""
    assert client.get("/health").json()["vqa"] == "fake/vqa"

    # Tile shape; level null -> max level (10 for the 600x400 synth slide).
    body = {"image_id": "synth", "level": None, "x": 1, "y": 0, "question": "is tumor present?"}
    r = client.post("/vqa", json=body)
    assert r.status_code == 200, r.text
    j = r.json()
    assert set(j) >= {"answer", "question", "model", "latency_ms", "cost_usd", "tokens_in",
                      "tokens_out", "image_id", "level", "x", "y", "cached"}
    assert j["answer"] in {"yes", "no"}
    assert j["question"] == "is tumor present?"
    assert j["model"] == "fake/vqa" and j["cost_usd"] == 0.0
    assert (j["image_id"], j["level"], j["x"], j["y"]) == ("synth", 10, 1, 0)
    assert j["cached"] is False

    # Same request again -> served from the cache.
    r2 = client.post("/vqa", json=body)
    assert r2.status_code == 200 and r2.json()["cached"] is True
    assert {k: v for k, v in r2.json().items() if k != "cached"} == \
        {k: v for k, v in j.items() if k != "cached"}

    # Non yes/no question -> canned description; explicit level works too.
    r3 = client.post("/vqa", json={"image_id": "synth", "level": 9, "x": 0, "y": 0,
                                   "question": "what does this image show?"})
    assert r3.status_code == 200, r3.text
    assert r3.json()["answer"] == "tumor tissue" and r3.json()["level"] == 9
    assert r3.json()["cached"] is False

    # Raw base64 and data-URL image bodies.
    im = tissue_image(96, 96, seed=5)
    b64 = base64.b64encode(png_bytes(im)).decode()
    r4 = client.post("/vqa", json={"image_b64": b64, "question": "does it look malignant?"})
    assert r4.status_code == 200, r4.text
    j4 = r4.json()
    assert j4["answer"] in {"yes", "no"} and j4["image_id"] is None and j4["x"] is None
    assert j4["cached"] is False
    r5 = client.post("/vqa", json={"image_b64": "data:image/png;base64," + b64,
                                   "question": "does it look malignant?"})
    assert r5.status_code == 200 and r5.json()["cached"] is True
    assert r5.json()["answer"] == j4["answer"]

    # Errors: unknown image / tile -> 404, bad bodies -> 400.
    assert client.post("/vqa", json={"image_id": "missing", "x": 0, "y": 0,
                                     "question": "q?"}).status_code == 404
    assert client.post("/vqa", json={"image_id": "synth", "level": 10, "x": 9, "y": 9,
                                     "question": "q?"}).status_code == 404
    assert client.post("/vqa", json={"image_id": "synth", "level": 99, "x": 0, "y": 0,
                                     "question": "q?"}).status_code == 404
    assert client.post("/vqa", json={"question": "q?"}).status_code == 400
    assert client.post("/vqa", json={"image_id": "synth", "x": 0, "y": 0,
                                     "question": "  "}).status_code == 400
    assert client.post("/vqa", json={"image_id": "synth", "x": 0,
                                     "question": "q?"}).status_code == 400
    assert client.post("/vqa", json={"image_b64": "bm90YW5pbWFnZQ==",
                                     "question": "q?"}).status_code == 400
    assert client.post("/vqa", json={"image_id": "../etc", "x": 0, "y": 0,
                                     "question": "q?"}).status_code == 400


def test_vqa_not_configured_and_upstream_error(env, monkeypatch):
    """Real mode without a key -> 503 vqa_not_configured; upstream failure -> 502."""
    monkeypatch.delenv("FIREWORKS_API_KEY", raising=False)
    monkeypatch.setattr(serve.config, "FIREWORKS_API_KEY", "")
    settings = serve.Settings(fake=False, encoder="fake", tiles_dir=env["tiles"],
                              results_dir=env["root"] / "results_vqa")
    # Avoid loading a real encoder/probe: only the /vqa and /health routes are exercised.
    monkeypatch.setattr(serve, "Scorer", lambda s: type("S", (), {"device": "cpu"})())
    app = serve.create_app(settings)
    with TestClient(app) as c:
        assert c.get("/health").json()["vqa"] is None
        r = c.post("/vqa", json={"image_id": "synth", "x": 0, "y": 0, "question": "is it?"})
        assert r.status_code == 503 and r.json()["detail"] == {"error": "vqa_not_configured"}

    monkeypatch.setenv("FIREWORKS_API_KEY", "test-key")

    def boom(image, question, settings):
        raise serve.vqa.VQAError("APIConnectionError: nope")

    monkeypatch.setattr(serve, "_vqa_answer", boom)
    with TestClient(app) as c:
        assert c.get("/health").json()["vqa"] == serve.vqa.DEFAULT_VQA_MODEL
        r = c.post("/vqa", json={"image_id": "synth", "x": 0, "y": 0, "question": "is it?"})
        assert r.status_code == 502 and "nope" in r.json()["detail"]


def test_vlm_classify_prompt_version(env, monkeypatch):
    """_vlm_classify passes v1/v2 + no_thinking through to vlm_client.classify_tile."""
    from pathfrontier import vlm_client

    seen = {}

    def fake_classify(image, dataset, cfg, prompt_version="v1", client=None):
        seen.update(cfg=cfg, prompt_version=prompt_version)
        return {"label": 1, "probs": [0.1, 0.9], "reason": "", "raw": "Tumor\n",
                "latency_ms": 1.0, "cost_usd": 0.0}

    monkeypatch.setattr(vlm_client, "classify_tile", fake_classify)
    im = tissue_image(96, 96)
    s1 = serve.Settings(fake=True, encoder="fake", vlm_model="m", vlm_prompt_version="v1",
                        vlm_no_thinking=False)
    r1 = serve._vlm_classify(im, s1)
    assert seen["prompt_version"] == "v1" and "extra_body" not in seen["cfg"].extra
    assert "logprobs" not in seen["cfg"].extra and r1["reason"] == ""

    s2 = serve.Settings(fake=True, encoder="fake", vlm_model="m", vlm_prompt_version="v2")
    r2 = serve._vlm_classify(im, s2)
    cfg = seen["cfg"]
    assert seen["prompt_version"] == "v2"
    assert cfg.extra["extra_body"] == {"reasoning_effort": "none"}
    assert cfg.extra["logprobs"] is True and cfg.max_tokens == vlm_client.V2_MAX_TOKENS
    assert r2["reason"] == "tumor" and r2["probs"] == [0.1, 0.9]

    monkeypatch.setenv("PATHFRONTIER_VLM_PROMPT_VERSION", "v2")
    monkeypatch.setenv("PATHFRONTIER_VLM_NO_THINKING", "0")
    monkeypatch.setenv("PATHFRONTIER_VQA_MODEL", "acct/models/x")
    s = serve.Settings.from_env()
    assert s.vlm_prompt_version == "v2" and s.vlm_no_thinking is False
    assert s.vqa_model == "acct/models/x"
