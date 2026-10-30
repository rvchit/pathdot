"""Offline tests for pathfrontier.finetune: example building, JSONL write/validate, and the REST
client against a mocked httpx transport. Records are built here (no dependency on data.py)."""
from __future__ import annotations

import base64
import io
import json
from dataclasses import dataclass
from pathlib import Path

import httpx
import numpy as np
import pytest
from PIL import Image

from pathfrontier import finetune as ft
from pathfrontier.schema import CLASSES


@dataclass
class Rec:  # duck-typed TileRecord
    tile_id: str
    image: Image.Image
    label: int


def _tile(seed: int, size: int = 96) -> Image.Image:
    rng = np.random.default_rng(seed)
    arr = rng.integers(0, 255, size=(size, size, 3), dtype=np.uint8)
    return Image.fromarray(arr, "RGB")


def _records(dataset: str, n: int) -> list[Rec]:
    k = len(CLASSES[dataset])
    return [Rec(f"{dataset}:valid:{i}", _tile(i), i % k) for i in range(n)]


# ---------------------------------------------------------------------------------------------
# build_sft_example
# ---------------------------------------------------------------------------------------------
def test_build_example_structure_pcam():
    rec = _records("pcam", 2)[1]
    ex = ft.build_sft_example(rec, "pcam", image_size=224)
    msgs = ex["messages"]
    assert [m["role"] for m in msgs] == ["system", "user", "assistant"]
    assert "JSON" in msgs[0]["content"]
    user = msgs[1]["content"]
    assert user[0]["type"] == "image_url"
    url = user[0]["image_url"]["url"]
    assert url.startswith("data:image/jpeg;base64,")
    img = Image.open(io.BytesIO(base64.b64decode(url.split(",", 1)[1])))
    assert img.size == (224, 224)
    assert user[1]["type"] == "text" and user[1]["text"]
    ans = json.loads(msgs[2]["content"])
    assert ans["label"] == "tumor"
    assert 0 < ans["confidence"] <= 1
    assert ans["reason"] in ft.RATIONALE_TEMPLATES["pcam"]["tumor"]
    assert len(ans["reason"].split()) <= 25
    assert ex["tile_id"] == rec.tile_id and ex["label"] == 1


def test_instruction_matches_baseline_prompt():
    from pathfrontier.vlm_client import build_prompt
    assert ft.build_instruction("pcam") == build_prompt("pcam", "v1")
    assert ft.build_instruction("crc") == build_prompt("crc", "v1")


def test_rationale_pick_is_seeded_and_varied():
    recs = _records("pcam", 40)
    a = [ft.build_sft_example(r, "pcam", image_size=64, seed=0)["messages"][2]["content"] for r in recs]
    b = [ft.build_sft_example(r, "pcam", image_size=64, seed=0)["messages"][2]["content"] for r in recs]
    assert a == b  # deterministic across calls
    reasons = {json.loads(x)["reason"] for x in a}
    assert len(reasons) >= 4  # both classes x several templates actually get used
    c = [ft.build_sft_example(r, "pcam", image_size=64, seed=1)["messages"][2]["content"] for r in recs]
    assert c != a  # seed changes the picks


def test_crc_all_classes_have_templates_and_build():
    for i, cls in enumerate(CLASSES["crc"]):
        assert len(ft.RATIONALE_TEMPLATES["crc"][cls]) >= 2
        for t in ft.RATIONALE_TEMPLATES["crc"][cls]:
            assert len(t.split()) <= 25, t
        ex = ft.build_sft_example(Rec(f"crc:train:{i}", _tile(i, 224), i), "crc", image_size=224)
        assert json.loads(ex["messages"][2]["content"])["label"] == cls


def test_custom_templates_override():
    rec = _records("pcam", 1)[0]
    ex = ft.build_sft_example(rec, "pcam", {"normal": ["custom normal"], "tumor": ["x"]}, image_size=32)
    assert json.loads(ex["messages"][2]["content"])["reason"] == "custom normal"


# ---------------------------------------------------------------------------------------------
# write + validate
# ---------------------------------------------------------------------------------------------
def test_write_and_validate_jsonl(tmp_path: Path):
    recs = _records("pcam", 6)
    out = tmp_path / "sft_pcam_train.jsonl"
    assert ft.write_sft_jsonl(recs, "pcam", out, image_size=112) == 6
    rep = ft.validate_sft_jsonl(out)
    assert rep.ok, rep.errors
    assert rep.n_examples == 6 and rep.n_images == 6
    assert rep.label_counts == {"normal": 3, "tumor": 3}
    assert rep.image_tokens == 6 * 4 * 4  # 112/28 = 4 tokens per side
    assert rep.text_tokens > 0 and rep.assistant_tokens > 0
    assert rep.total_tokens == rep.text_tokens + rep.image_tokens
    assert rep.bytes == out.stat().st_size
    assert "6 examples" in rep.summary()
    assert rep.training_cost_usd(epochs=2, usd_per_m_tokens=0.5) == pytest.approx(
        rep.total_tokens * 2 * 0.5 / 1e6)


def test_validate_catches_bad_lines(tmp_path: Path):
    p = tmp_path / "bad.jsonl"
    good = ft.build_sft_example(_records("pcam", 1)[0], "pcam", image_size=32)
    img_on_assistant = json.loads(json.dumps(good))
    img_on_assistant["messages"][2]["content"] = [
        {"type": "image_url", "image_url": {"url": good["messages"][1]["content"][0]["image_url"]["url"]}}]
    http_image = json.loads(json.dumps(good))
    http_image["messages"][1]["content"][0]["image_url"]["url"] = "https://example.com/x.jpg"
    lines = [json.dumps(good), "not json", json.dumps({"foo": 1}), json.dumps(img_on_assistant),
             json.dumps(http_image), json.dumps({"messages": [{"role": "user", "content": "hi"}]})]
    p.write_text("\n".join(lines) + "\n")
    rep = ft.validate_sft_jsonl(p)
    assert not rep.ok
    joined = "\n".join(rep.errors)
    assert "invalid JSON" in joined
    assert "missing non-empty 'messages'" in joined
    assert "images only allowed on user" in joined
    assert "bad image" in joined
    assert "no assistant message" in joined
    assert "at least 3 examples" not in joined  # 5 parsed examples >= 3


def test_validate_min_examples(tmp_path: Path):
    p = tmp_path / "tiny.jsonl"
    ft.write_sft_jsonl(_records("pcam", 2), "pcam", p, image_size=32)
    rep = ft.validate_sft_jsonl(p)
    assert any("at least 3" in e for e in rep.errors)
    assert not ft.validate_sft_jsonl(tmp_path / "missing.jsonl").ok


def test_estimate_training_cost_tiers():
    assert ft.estimate_training_cost(1_000_000, 2, "accounts/fireworks/models/qwen3p5-9b") == pytest.approx(1.0)
    assert ft.estimate_training_cost(1_000_000, 1, "accounts/fireworks/models/qwen3p5-27b") == pytest.approx(3.0)
    assert ft.DEFAULT_BASE_MODEL in ft.VISION_SFT_BASE_MODELS


# ---------------------------------------------------------------------------------------------
# REST client against a mocked transport
# ---------------------------------------------------------------------------------------------
class FakeFireworks:
    """Minimal stateful stub of the Fireworks gateway."""

    def __init__(self):
        self.calls: list[tuple[str, str]] = []
        self.bodies: dict[str, object] = {}
        self.job_states = ["JOB_STATE_PENDING", "JOB_STATE_RUNNING", "JOB_STATE_COMPLETED"]
        self.dep_states = ["CREATING", "READY"]
        self.ds_polls = 0
        self.fail_job = False

    def handler(self, request: httpx.Request) -> httpx.Response:
        path = request.url.path
        self.calls.append((request.method, path))
        assert request.headers["authorization"] == "Bearer test-key"
        acct = "/v1/accounts/acme"
        if request.method == "POST" and path == f"{acct}/datasets":
            body = json.loads(request.content)
            self.bodies["create_dataset"] = body
            return httpx.Response(200, json={"name": f"accounts/acme/datasets/{body['datasetId']}",
                                             "state": "UPLOADING"})
        if request.method == "POST" and path.endswith(":upload"):
            ctype = request.headers["content-type"]
            assert ctype.startswith("multipart/form-data")
            assert b'name="file"' in request.content and b'"messages"' in request.content
            return httpx.Response(200, json={"id": "ds", "object": "file", "bytes": len(request.content)})
        if request.method == "GET" and "/datasets/" in path:
            self.ds_polls += 1
            state = "UPLOADING" if self.ds_polls == 1 else "READY"
            return httpx.Response(200, json={"name": "accounts/acme/datasets/sft-pcam-train",
                                             "state": state})
        if request.method == "POST" and path == f"{acct}/supervisedFineTuningJobs":
            body = json.loads(request.content)
            self.bodies["create_job"] = body
            return httpx.Response(200, json={"name": "accounts/acme/supervisedFineTuningJobs/job123",
                                             "state": "JOB_STATE_CREATING", **body})
        if request.method == "GET" and "/supervisedFineTuningJobs/" in path:
            state = self.job_states.pop(0) if len(self.job_states) > 1 else self.job_states[0]
            if self.fail_job and state == "JOB_STATE_COMPLETED":
                return httpx.Response(200, json={"name": "accounts/acme/supervisedFineTuningJobs/job123",
                                                 "state": "JOB_STATE_FAILED",
                                                 "status": {"code": "INTERNAL", "message": "boom"}})
            return httpx.Response(200, json={"name": "accounts/acme/supervisedFineTuningJobs/job123",
                                             "state": state, "outputModel": "accounts/acme/models/my-ft",
                                             "jobProgress": {"percent": 50}})
        if request.method == "POST" and path == f"{acct}/deployments":
            body = json.loads(request.content)
            self.bodies["create_deployment"] = body
            self.bodies["deployment_id"] = request.url.params.get("deploymentId")
            return httpx.Response(200, json={"name": f"accounts/acme/deployments/{self.bodies['deployment_id']}",
                                             "state": "CREATING"})
        if request.method == "GET" and "/deployments/" in path:
            state = self.dep_states.pop(0) if len(self.dep_states) > 1 else self.dep_states[0]
            return httpx.Response(200, json={"name": path.split("/v1/accounts/acme/")[1]
                                             .replace("deployments/", "accounts/acme/deployments/"),
                                             "state": state})
        if request.method == "DELETE" and "/deployments/" in path:
            return httpx.Response(200, json={})
        return httpx.Response(404, json={"error": f"unhandled {request.method} {path}"})


@pytest.fixture
def fake():
    return FakeFireworks()


@pytest.fixture
def client(fake):
    c = ft.FireworksFT(api_key="test-key", account_id="acme", transport=httpx.MockTransport(fake.handler))
    yield c
    c.close()


def test_client_requires_credentials():
    c = ft.FireworksFT(api_key="", account_id="")
    assert not c.available
    with pytest.raises(RuntimeError, match="FIREWORKS_API_KEY"):
        c.get_job("x")
    c2 = ft.FireworksFT(api_key="k", account_id="")
    with pytest.raises(RuntimeError, match="FIREWORKS_ACCOUNT_ID"):
        c2.get_job("x")


def test_create_dataset_flow(client, fake, tmp_path):
    p = tmp_path / "sft_pcam_train.jsonl"
    ft.write_sft_jsonl(_records("pcam", 4), "pcam", p, image_size=32)
    res = client.create_dataset(p, "sft-pcam-train", poll_s=0)
    assert res["state"] == "READY"
    assert res["name"] == "accounts/acme/datasets/sft-pcam-train"
    assert fake.bodies["create_dataset"]["datasetId"] == "sft-pcam-train"
    assert fake.bodies["create_dataset"]["dataset"]["exampleCount"] == "4"
    assert "userUploaded" in fake.bodies["create_dataset"]["dataset"]
    methods = [m + " " + p_.rsplit("/", 1)[-1] for m, p_ in fake.calls]
    assert methods[:2] == ["POST datasets", "POST sft-pcam-train:upload"]
    assert fake.ds_polls == 2
    assert res.cli_hint == f"firectl dataset create sft-pcam-train {p}"


def test_create_job_body_and_hint(client, fake):
    res = client.create_job("accounts/fireworks/models/qwen3p5-9b", "sft-pcam-train", lora_rank=16,
                            epochs=2, lr=1e-4, name="pcam-ft", eval_dataset_id="sft-pcam-eval")
    body = fake.bodies["create_job"]
    assert body == {
        "dataset": "accounts/acme/datasets/sft-pcam-train",
        "baseModel": "accounts/fireworks/models/qwen3p5-9b",
        "outputModel": "accounts/acme/models/pcam-ft",
        "displayName": "pcam-ft",
        "epochs": 2,
        "loraRank": 16,
        "learningRate": 1e-4,
        "evaluationDataset": "accounts/acme/datasets/sft-pcam-eval",
    }
    assert res["jobId"] == "job123"
    assert res["outputModel"] == "accounts/acme/models/pcam-ft"
    assert res.cli_hint == ("firectl sftj create --base-model accounts/fireworks/models/qwen3p5-9b "
                            "--dataset sft-pcam-train --output-model pcam-ft --lora-rank 16 --epochs 2 "
                            "--learning-rate 0.0001 --evaluation-dataset sft-pcam-eval")


def test_create_job_rejects_full_parameter(client):
    with pytest.raises(ValueError, match="lora_rank"):
        client.create_job("accounts/fireworks/models/qwen3p5-9b", "ds", lora_rank=0)


def test_create_job_omits_lr_when_unset(client, fake):
    client.create_job("accounts/fireworks/models/qwen3p5-9b", "ds", name="n")
    assert "learningRate" not in fake.bodies["create_job"]
    assert "evaluationDataset" not in fake.bodies["create_job"]
    assert fake.bodies["create_job"]["loraRank"] == 16


def test_get_and_wait(client, fake):
    job = client.get_job("accounts/acme/supervisedFineTuningJobs/job123")
    assert job["jobId"] == "job123" and job.cli_hint == "firectl sftj get job123"
    seen = []
    done = client.wait("job123", poll_s=0, on_update=lambda j: seen.append(j["state"]))
    assert done["state"] == "JOB_STATE_COMPLETED"
    assert done["outputModel"] == "accounts/acme/models/my-ft"
    assert seen == ["JOB_STATE_RUNNING", "JOB_STATE_COMPLETED"]


def test_wait_raises_on_failure(client, fake):
    fake.fail_job = True
    with pytest.raises(RuntimeError, match="JOB_STATE_FAILED.*boom"):
        client.wait("job123", poll_s=0)


def test_wait_timeout(client, fake):
    fake.job_states = ["JOB_STATE_RUNNING", "JOB_STATE_RUNNING"]
    with pytest.raises(TimeoutError):
        client.wait("job123", poll_s=0, timeout_s=-1)


def test_deploy_returns_inference_model(client, fake):
    res = client.deploy("my-ft", poll_s=0)
    body = fake.bodies["create_deployment"]
    assert body["baseModel"] == "accounts/acme/models/my-ft"
    assert body["deploymentShape"] == "default"
    assert body["minReplicaCount"] == 0 and body["maxReplicaCount"] == 1
    assert body["autoscalingPolicy"] == {"scaleToZeroWindow": "600s"}
    assert fake.bodies["deployment_id"] == "my-ft"
    assert res["state"] == "READY"
    assert res["name"] == "accounts/acme/deployments/my-ft"
    assert res["inference_model"] == "accounts/acme/models/my-ft"
    assert res.cli_hint == ("firectl deployment create accounts/acme/models/my-ft "
                            "--deployment-shape default --wait --deployment-id my-ft")
    assert client.delete_deployment("my-ft").cli_hint == "firectl deployment delete my-ft"


def test_http_errors_surface(fake):
    def h(request):
        return httpx.Response(403, json={"error": "nope"})
    c = ft.FireworksFT(api_key="k", account_id="acme", transport=httpx.MockTransport(h))
    with pytest.raises(RuntimeError, match="HTTP 403"):
        c.get_job("j")


def test_manual_recipe_mentions_every_step(tmp_path):
    txt = ft.manual_recipe(tmp_path / "sft_pcam_train.jsonl", ft.DEFAULT_BASE_MODEL, "pcam-ft",
                           lora_rank=16, epochs=2, eval_file=tmp_path / "sft_pcam_eval.jsonl")
    assert "firectl dataset create sft-pcam-train" in txt
    assert "firectl dataset create sft-pcam-train-eval" in txt
    assert "firectl sftj create --base-model accounts/fireworks/models/qwen3p5-9b" in txt
    assert "--evaluation-dataset sft-pcam-train-eval" in txt
    assert "firectl deployment create accounts/<ACCOUNT_ID>/models/pcam-ft" in txt
    assert "FIREWORKS_FT_MODEL=accounts/<ACCOUNT_ID>/models/pcam-ft" in txt
    assert "firectl deployment delete" in txt
