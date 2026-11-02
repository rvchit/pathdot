"""Offline tests for pathfrontier.vlm_client (no network, no API keys)."""
from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass

import pytest
from PIL import Image

from pathfrontier import schema
from pathfrontier import vlm_client as vc


@dataclass
class Tile:  # TileRecord-like, independent of pathfrontier.data
    tile_id: str
    image: Image.Image
    label: int


def _tiles(dataset="pcam", n=6, split="test"):
    k = len(schema.CLASSES[dataset])
    return [Tile(f"{dataset}:{split}:{i}", Image.new("RGB", (96, 96), (i * 37 % 255, 80, 120)),
                 i % k) for i in range(n)]


# ---------------------------------------------------------------- parsing ---------------------
@pytest.mark.parametrize("raw,label,conf", [
    ('{"label": "tumor", "confidence": 0.9, "reason": "x"}', 1, 0.9),
    ('```json\n{"label": "normal", "confidence": 0.7, "reason": "clean"}\n```', 0, 0.7),
    ('Sure! {"label":"tumor","confidence":0.6,"reason":"nuclei"} Hope that helps.', 1, 0.6),
    ('{"label": "Tumor", "confidence": 85, "reason": "pct"}', 1, 0.85),
    ('{"label": "normal", "confidence": 1.7}', 0, 1.0),
    ('{"label": "normal"}', 0, 0.5),
    ("label: tumor\nconfidence: 0.8\nreason: 'big nuclei'", 1, 0.8),
    ("I think this is malignant tissue, so tumor. confidence=92%", 1, 0.92),
    ('{"label": "tumour", "confidence": 0.55, "reason": "uk spelling"}', 1, 0.55),
    ('{"label": 1, "confidence": 0.66}', 1, 0.66),
])
def test_parse_pcam_variants(raw, label, conf):
    p = vc.parse_response(raw, "pcam")
    assert p["parse_error"] is False
    assert p["label"] == label
    assert p["confidence"] == pytest.approx(conf)


def test_parse_crc_codes_and_prose():
    p = vc.parse_response('{"label": "TUM", "confidence": 0.8, "reason": "glands"}', "crc")
    assert p["label"] == schema.CRC_CLASSES.index("TUM") and p["reason"] == "glands"
    p = vc.parse_response("Mostly lymphocytes here -> LYM", "crc")
    assert p["label"] == schema.CRC_CLASSES.index("LYM")


def test_parse_garbage_flags_error():
    p = vc.parse_response("I cannot determine this.", "pcam")
    assert p["parse_error"] is True and p["label"] is None
    p = vc.parse_response("", "crc")
    assert p["parse_error"] is True


def test_parse_nested_json_in_fence_with_trailing_text():
    raw = "Analysis:\n```\n{\"label\": \"normal\", \"confidence\": 0.75, \"reason\": \"a, b\"}\n```\nDone."
    p = vc.parse_response(raw, "pcam")
    assert p["label"] == 0 and p["confidence"] == 0.75 and p["reason"] == "a, b"


# ---------------------------------------------------------------- probs -----------------------
def test_probs_binary():
    assert vc.probs_from(1, 0.9, 2) == pytest.approx([0.1, 0.9])
    assert vc.probs_from(0, 0.8, 2) == pytest.approx([0.8, 0.2])


def test_probs_multiclass_uniform_remainder():
    p = vc.probs_from(8, 0.6, 9)
    assert p[8] == pytest.approx(0.6)
    assert all(x == pytest.approx(0.05) for x in p[:8])
    assert sum(p) == pytest.approx(1.0)


def test_probs_unknown_is_uniform():
    assert vc.probs_from(None, 0.9, 9) == pytest.approx([1 / 9] * 9)
    assert vc.probs_from(42, 0.9, 2) == pytest.approx([0.5, 0.5])


# ---------------------------------------------------------------- cost / prices ---------------
def test_cost_math_and_price_lookup():
    cfg = vc.VLMConfig(model="accounts/fireworks/models/deepseek-v4p1-flash")
    assert (cfg.price_in_per_m, cfg.price_out_per_m) == (0.30, 1.20)
    assert vc.cost_usd(1_000_000, 1_000_000, cfg) == pytest.approx(1.50)
    assert vc.cost_usd(1000, 50, cfg) == pytest.approx(0.0003 + 0.00006)
    cfg2 = vc.VLMConfig(model="x/unknown", price_in_per_m=1.0, price_out_per_m=2.0)
    assert vc.cost_usd(500_000, 250_000, cfg2) == pytest.approx(1.0)


def test_price_for_finetuned_id_inherits_base_price():
    assert vc.price_for("accounts/me/models/kimi-k3-pcam-ft-v2") == vc.PRICES[
        "accounts/fireworks/models/kimi-k3"]
    assert vc.price_for("accounts/me/models/totally-new") == vc.UNKNOWN_MODEL_PRICE
    assert vc.DEFAULT_BASELINE_MODEL in vc.PRICES


def test_prices_table_sane():
    for mid, (pin, pout) in vc.PRICES.items():
        assert mid.startswith("accounts/fireworks/models/")
        assert 0 < pin <= 20 and 0 < pout <= 50


# ---------------------------------------------------------------- image -----------------------
def test_encode_upscales_to_448():
    url = vc.encode_image(Image.new("RGB", (96, 96)), 448)
    assert url.startswith("data:image/png;base64,")
    import base64
    import io

    img = Image.open(io.BytesIO(base64.b64decode(url.split(",", 1)[1])))
    assert min(img.size) >= 448
    big = vc.prepare_image(Image.new("L", (600, 500)), 448)
    assert big.size == (600, 500) and big.mode == "RGB"


def test_build_prompt_mentions_classes():
    s, u = vc.build_prompt("pcam")
    assert '"normal" | "tumor"' in s and "JSON" in u
    s, _ = vc.build_prompt("crc")
    for c in schema.CRC_CLASSES:
        assert f"- {c}:" in s
    with pytest.raises(ValueError):
        vc.build_prompt("pcam", "v9")


# ---------------------------------------------------------------- classify --------------------
def test_classify_tile_with_fake_client():
    cfg = vc.VLMConfig(model="accounts/fireworks/models/glm-5p3-flash", api_key="k")
    res = vc.classify_tile(Image.new("RGB", (96, 96)), "pcam", cfg, client=vc.FakeAsyncClient())
    assert set(res) >= {"label", "probs", "reason", "latency_ms", "cost_usd", "raw",
                        "tokens_in", "tokens_out"}
    assert res["tokens_in"] == 900 and res["tokens_out"] == 40
    assert res["cost_usd"] == pytest.approx((900 * 0.15 + 40 * 0.50) / 1e6)
    assert sum(res["probs"]) == pytest.approx(1.0) and res["parse_error"] is False


def test_classify_many_monkeypatched_openai(monkeypatch, tmp_path):
    """AsyncOpenAI is replaced by the fake; oracle makes every answer correct."""
    import openai

    tiles = _tiles("pcam", 6)
    oracle = {vc.image_hash(t.image): t.label for t in tiles}
    fake = vc.FakeAsyncClient("pcam", oracle=oracle)
    monkeypatch.setattr(openai, "AsyncOpenAI", lambda **kw: fake)

    cfg = vc.VLMConfig(model="accounts/fireworks/models/kimi-k3", api_key="k", concurrency=3)
    cache = tmp_path / "cache.jsonl"
    preds = asyncio.run(vc.classify_many(tiles, "pcam", cfg, "v1", cache, progress=False))
    assert [p.tile_id for p in preds] == [t.tile_id for t in tiles]
    assert all(isinstance(p, schema.Prediction) for p in preds)
    assert all(p.y_pred == p.y_true for p in preds)
    assert all(p.stage == "baseline" and p.split == "test" and p.dataset == "pcam" for p in preds)
    assert all(len(p.y_prob) == 2 for p in preds)
    assert fake.calls == 6
    s = vc.summarize(preds)
    assert s["n"] == 6 and s["accuracy"] == 1.0 and s["total_cost_usd"] > 0

    # resume: second run must hit the cache and make zero calls
    fake2 = vc.FakeAsyncClient("pcam")
    preds2 = asyncio.run(vc.classify_many(tiles, "pcam", cfg, "v1", cache, client=fake2,
                                          progress=False))
    assert fake2.calls == 0
    assert [p.y_pred for p in preds2] == [p.y_pred for p in preds]

    # different prompt_version or model -> cache miss
    preds3 = asyncio.run(vc.classify_many(tiles[:2], "pcam", cfg, "v1", cache, client=fake2,
                                          progress=False))
    assert fake2.calls == 0
    cfg_b = vc.VLMConfig(model="accounts/fireworks/models/glm-5p3-flash", api_key="k")
    asyncio.run(vc.classify_many(tiles[:2], "pcam", cfg_b, "v1", cache, client=fake2,
                                 progress=False))
    assert fake2.calls == 2
    assert len(preds3) == 2
    lines = [json.loads(line) for line in cache.read_text().splitlines()]
    assert len(lines) == 8 and {"model", "prompt_version", "tile_id", "result"} <= set(lines[0])


def test_cache_tolerates_corrupt_line(tmp_path):
    cache = tmp_path / "c.jsonl"
    good = {"model": "m", "prompt_version": "v1", "tile_id": "pcam:test:0",
            "result": {"label": 1, "probs": [0.1, 0.9]}}
    cache.write_text(json.dumps(good) + "\n{not json\n")
    loaded = vc.load_cache(cache)
    assert list(loaded) == ["m|v1|pcam:test:0"]


def test_retries_then_succeeds(monkeypatch):
    sleeps: list[float] = []

    async def fake_sleep(s):
        sleeps.append(s)

    monkeypatch.setattr(vc.asyncio, "sleep", fake_sleep)
    fake = vc.FakeAsyncClient("pcam", fail_first_n=2)
    cfg = vc.VLMConfig(model="accounts/fireworks/models/kimi-k3", api_key="k", max_retries=4)
    res = asyncio.run(vc.classify_tile_async(Image.new("RGB", (96, 96)), "pcam", cfg,
                                             client=fake))
    assert fake.calls == 3 and res.get("api_error") is None
    assert len(sleeps) == 2 and sleeps[1] > sleeps[0]  # exponential


def test_retries_exhausted_records_error(monkeypatch):
    async def fake_sleep(s):
        pass

    monkeypatch.setattr(vc.asyncio, "sleep", fake_sleep)
    fake = vc.FakeAsyncClient("pcam", fail_first_n=10)
    cfg = vc.VLMConfig(model="accounts/fireworks/models/kimi-k3", api_key="k", max_retries=2)
    res = asyncio.run(vc.classify_tile_async(Image.new("RGB", (96, 96)), "pcam", cfg,
                                             client=fake))
    assert fake.calls == 3 and res["parse_error"] is True and "api_error" in res
    assert res["probs"] == pytest.approx([0.5, 0.5]) and res["cost_usd"] == 0.0


def test_non_retryable_not_retried():
    class Boom(Exception):
        status_code = 400

    class Client:
        def __init__(self):
            self.calls = 0
            self.chat = vc._obj(completions=vc._obj(create=self._create))

        async def _create(self, **kw):
            self.calls += 1
            raise Boom("bad request")

    c = Client()
    cfg = vc.VLMConfig(model="accounts/fireworks/models/kimi-k3", api_key="k", max_retries=5)
    res = asyncio.run(vc.classify_tile_async(Image.new("RGB", (96, 96)), "pcam", cfg, client=c))
    assert c.calls == 1 and "Boom" in res["api_error"]


def test_to_prediction_marks_parse_error_confidence_none():
    t = _tiles("crc", 1)[0]
    res = {"label": 0, "probs": [1 / 9] * 9, "parse_error": True, "raw": "???", "reason": ""}
    p = vc.to_prediction(t, res, "crc", "m", "baseline", "v1")
    assert p.extra["parse_error"] is True and p.reason is None
    assert p.confidence == pytest.approx(1 / 9)  # schema default: max(y_prob)


def test_run_split_dry_run_writes_file(tmp_path):
    out, summary = vc.run_split("accounts/fireworks/models/kimi-k3", "pcam", "test", 8, 4, "v1",
                                "openai", "baseline", dry_run=True, results_dir=tmp_path,
                                cache_dir=tmp_path)
    assert out == tmp_path / "preds_accounts_fireworks_models_kimi-k3_test.jsonl"
    preds = schema.load_predictions(out)
    assert len(preds) == 8 == summary["n"]
    assert summary["parse_errors"] == 0 and 0.5 <= summary["accuracy"] <= 1.0


def test_p_tumor_from_logprobs_handles_split_tokens():
    from types import SimpleNamespace as NS

    def lp(*cands):
        return NS(content=[NS(top_logprobs=[NS(token=t, logprob=v) for t, v in cands])])

    # Qwen3-VL emits "tumor" as "t" + "umor"
    p = vc.p_tumor_from_logprobs(lp(("t", -0.03), ("normal", -3.53), ("肿瘤", -18.0)))
    assert 0.96 < p < 0.98
    assert vc.p_tumor_from_logprobs(lp(("normal", 0.0), ("t", -9.25))) < 0.001
    assert vc.p_tumor_from_logprobs(lp(("foo", 0.0))) is None
    assert vc.p_tumor_from_logprobs(None) is None
