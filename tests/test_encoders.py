"""Offline tests for pathfrontier.encoders and scripts/embed.py using the `fake` encoder.

The real-download test is marked `slow` (`pytest -m slow tests/test_encoders.py`).
"""
from __future__ import annotations

import importlib.util
import json
import socket
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from pathfrontier.encoders import (
    ENCODERS,
    Encoder,
    EncoderSpec,
    FakeEncoder,
    GatedModelError,
    benchmark,
    get_encoder,
    list_encoders,
)
from pathfrontier.schema import embedding_paths

ROOT = Path(__file__).resolve().parent.parent


def _load_script(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _images(n: int, size: int = 96, seed: int = 0) -> list[Image.Image]:
    rng = np.random.default_rng(seed)
    return [Image.fromarray(rng.integers(0, 256, (size, size, 3), dtype=np.uint8)) for _ in range(n)]


def _records(n: int, dataset: str = "pcam", split: str = "test"):
    return [SimpleNamespace(tile_id=f"{dataset}:{split}:{i}", image=im, label=i % 2)
            for i, im in enumerate(_images(n))]


def _has_network() -> bool:
    try:
        socket.create_connection(("huggingface.co", 443), timeout=3).close()
        return True
    except OSError:
        return False


# -- registry -----------------------------------------------------------------------------------

def test_registry_contract():
    required = {"phikon-v2", "uni", "uni2-h", "conch", "virchow2", "h-optimus-0", "dinov2-base",
                "resnet50", "fake"}
    assert required <= set(ENCODERS)
    for name, spec in ENCODERS.items():
        assert isinstance(spec, EncoderSpec) and spec.name == name
        assert spec.dim > 0 and spec.input_size > 0 and len(spec.mean) == 3 == len(spec.std)
        if name != "fake":
            assert spec.url.startswith("https://huggingface.co/")
    assert ENCODERS["phikon-v2"].dim == 1024 and not ENCODERS["phikon-v2"].gated
    assert ENCODERS["virchow2"].dim == 2560 and ENCODERS["virchow2"].patch_start == 5
    assert ENCODERS["h-optimus-0"].mean == (0.707223, 0.578729, 0.703617)
    assert ENCODERS["resnet50"].dim == 2048 and ENCODERS["dinov2-base"].dim == 768
    assert "fake" not in list_encoders() and "fake" in list_encoders(include_fake=True)


def test_unknown_encoder():
    with pytest.raises(KeyError):
        get_encoder("not-an-encoder")


# -- fake encoder -----------------------------------------------------------------------------

def test_fake_encoder_embeds_and_is_deterministic():
    enc = get_encoder("fake", device="cpu")
    assert isinstance(enc, FakeEncoder) and isinstance(enc, Encoder)
    assert enc.name == "fake" and enc.dim == 64
    imgs = _images(10, size=96) + [Image.new("L", (32, 48))]      # mixed sizes/modes are fine
    X = enc.embed(imgs, batch_size=4)
    assert X.shape == (11, 64) and X.dtype == np.float32
    np.testing.assert_allclose(X, get_encoder("fake").embed(imgs, batch_size=64), rtol=1e-5)
    assert enc.embed([]).shape == (0, 64)
    assert np.abs(X[0] - X[1]).sum() > 0  # different inputs -> different embeddings


def test_preprocess_normalises():
    enc = get_encoder("fake", device="cpu")
    x = enc.preprocess([Image.new("RGB", (40, 40), (124, 116, 104))])  # ~ImageNet mean colour
    assert x.shape == (1, 3, enc.input_size, enc.input_size)
    assert np.abs(x).max() < 0.05


def test_benchmark_returns_throughput():
    tps = benchmark(get_encoder("fake"), n=32, batch_size=16, warmup=4)
    assert tps > 0


def test_gated_message_mentions_fallback():
    from pathfrontier.encoders import _gated_message

    msg = _gated_message(ENCODERS["uni"], RuntimeError("401 Client Error"))
    assert "https://huggingface.co/MahmoodLab/UNI" in msg and "HF_TOKEN" in msg
    assert "phikon-v2" in msg
    assert issubclass(GatedModelError, RuntimeError)


# -- scripts/embed.py --------------------------------------------------------------------------

def test_embed_script_functions_write_cache(tmp_path):
    embed = _load_script("embed")
    enc = get_encoder("fake")
    X, y, ids, seconds = embed.embed_records(enc, _records(12), batch_size=5)
    assert X.shape == (12, 64) and y.tolist() == [i % 2 for i in range(12)]
    meta = embed.write_cache("fake", "pcam", "test", X, y, ids, seconds, enc.device,
                             cache_dir=tmp_path)
    paths = embedding_paths("fake", "pcam", "test", tmp_path)
    assert np.load(paths["X"]).shape == (12, 64) and np.load(paths["y"]).dtype == np.int64
    assert json.loads(paths["ids"].read_text()) == ids
    assert embed.cache_exists("fake", "pcam", "test", cache_dir=tmp_path)
    assert set(meta) >= {"tiles_per_s", "device", "n", "dim", "seconds"}
    assert meta["n"] == 12 and meta["dim"] == 64 and meta["tiles_per_s"] > 0


def test_embed_cli_end_to_end(tmp_path, monkeypatch):
    from typer.testing import CliRunner

    embed = _load_script("embed")
    monkeypatch.setattr(embed.config, "CACHE_DIR", tmp_path)
    monkeypatch.setattr(embed, "load_records",
                        lambda dataset, split, limit, seed, balanced, synthetic: _records(20))
    r = CliRunner().invoke(embed.app, ["--encoder", "fake", "--dataset", "pcam", "--split", "test"])
    assert r.exit_code == 0, r.output
    assert embed.cache_exists("fake", "pcam", "test", cache_dir=tmp_path)
    r2 = CliRunner().invoke(embed.app, ["--encoder", "fake", "--dataset", "pcam", "--split", "test"])
    assert r2.exit_code == 0 and "cache exists" in r2.output


def test_train_probe_cli_end_to_end(tmp_path, monkeypatch):
    from typer.testing import CliRunner

    embed = _load_script("embed")
    train = _load_script("train_probe")
    monkeypatch.setattr(train.config, "CACHE_DIR", tmp_path / "cache")
    monkeypatch.setattr(train.config, "RESULTS_DIR", tmp_path / "results")
    rng = np.random.default_rng(0)
    for split, n in (("valid", 300), ("test", 100)):
        y = np.arange(n) % 2
        X = (rng.standard_normal((n, 64)) + 3 * y[:, None]).astype(np.float32)
        embed.write_cache("fake", "pcam", split, X, y, [f"pcam:{split}:{i}" for i in range(n)],
                          1.0, "cpu", cache_dir=tmp_path / "cache", tiles_per_s=100.0)
    r = CliRunner().invoke(train.app, ["--encoder", "fake", "--dataset", "pcam"])
    assert r.exit_code == 0, r.output
    assert "AUROC" in r.output
    assert (tmp_path / "results" / "probe_fake_pcam.joblib").exists()
    preds = (tmp_path / "results" / "preds_probe_fake_test.jsonl").read_text().splitlines()
    assert len(preds) == 100
    rec = json.loads(preds[0])
    assert rec["model"] == "probe/fake" and rec["stage"] == "probe"
    assert abs(rec["latency_ms"] - 10.0) < 1e-9                      # 1000 / 100 tiles/s
    assert abs(rec["cost_usd"] - (1 / 100.0) * 2.5 / 3600) < 1e-12
    # missing cache -> helpful error naming the embed command
    r2 = CliRunner().invoke(train.app, ["--encoder", "fake", "--dataset", "crc"])
    assert r2.exit_code != 0 and "scripts/embed.py" in r2.output


# -- real download (slow, needs network) -----------------------------------------------------

@pytest.mark.slow
@pytest.mark.skipif(not _has_network(), reason="no network access to huggingface.co")
def test_phikon_v2_real_embedding():
    enc = get_encoder("phikon-v2")
    X = enc.embed(_images(3, size=96), batch_size=2)
    assert X.shape == (3, 1024) and X.dtype == np.float32 and np.isfinite(X).all()
