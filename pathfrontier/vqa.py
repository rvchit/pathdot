"""PathVQA question answering against the fine-tuned Qwen3.5-9B deployment on Fireworks.

The prompt, image normalisation (longest side <= 512 px, JPEG q90) and message layout match
what the model was fine-tuned and evaluated with (pathvqa_kit/scripts/common.py), so answers
at serve time look like answers at eval time. Qwen3.5 is a hybrid-reasoning model: without
``reasoning_effort: "none"`` it spends ``max_tokens`` thinking and returns empty content.

    PATHFRONTIER_VQA_MODEL   Fireworks model id (default DEFAULT_VQA_MODEL)
"""
from __future__ import annotations

import base64
import io
import time
from typing import Any

from PIL import Image

from pathfrontier import config

DEFAULT_VQA_MODEL = ("accounts/rvchit/models/pathvqa-qwen3p5-9b"
                     "#accounts/rvchit/deployments/pathvqa-q35-9b-ft2")

# Exactly the fine-tuning / eval prompt from pathvqa_kit/scripts/common.py. Do not edit.
SYSTEM_PROMPT = (
    "You are a pathology assistant. Look at the image and answer the question. "
    "If it is a yes/no question, answer exactly 'yes' or 'no'. "
    "Otherwise answer with as few words as possible (usually 1-3 words). "
    "Output only the answer."
)
MAX_SIDE = 512
JPEG_QUALITY = 90
MAX_TOKENS = 32
TEMPERATURE = 0.0
TIMEOUT_S = 60.0

_YES_NO_PREFIXES = ("is ", "are ", "does ", "do ", "was ", "were ", "did ", "can ", "has ",
                    "have ", "will ", "should ", "could ")


class VQAError(RuntimeError):
    """The upstream model call failed (surfaced as HTTP 502 by the service)."""


def normalize_image(img: Image.Image) -> Image.Image:
    """RGB copy with the longest side <= MAX_SIDE (never upscaled), as in training."""
    img = img.convert("RGB")
    w, h = img.size
    s = MAX_SIDE / max(w, h, 1)
    if s < 1:
        img = img.resize((max(1, round(w * s)), max(1, round(h * s))), Image.LANCZOS)
    return img


def image_data_uri(img: Image.Image) -> str:
    buf = io.BytesIO()
    normalize_image(img).save(buf, "JPEG", quality=JPEG_QUALITY)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def build_messages(img: Image.Image, question: str) -> list[dict]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": image_data_uri(img)}},
            {"type": "text", "text": question},
        ]},
    ]


def price_per_token(model: str) -> tuple[float, float]:
    """(input, output) USD per token; fine-tuned ids inherit the base price via vlm_client."""
    try:
        from pathfrontier.vlm_client import price_for
    except ImportError:
        return 0.0, 0.0
    pin, pout = price_for(model.split("#", 1)[0])
    return pin / 1_000_000.0, pout / 1_000_000.0


def canned_answer(question: str) -> str:
    """Offline answer for PATHFRONTIER_FAKE=1: yes/no questions alternate deterministically."""
    q = question.strip().lower()
    if q.startswith(_YES_NO_PREFIXES):
        return "yes" if sum(map(ord, q)) % 2 == 0 else "no"
    return "tumor tissue"


def make_client(api_key: str | None = None, base_url: str | None = None) -> Any:
    import openai

    return openai.OpenAI(base_url=base_url or config.FIREWORKS_BASE_URL,
                         api_key=api_key or config.FIREWORKS_API_KEY or "missing",
                         timeout=TIMEOUT_S, max_retries=1)


def answer_question(img: Image.Image, question: str, model: str = DEFAULT_VQA_MODEL,
                    client: Any = None, fake: bool = False) -> dict:
    """One (image, question) -> {"answer", "model", "latency_ms", "cost_usd", "tokens_in",
    "tokens_out"}. Raises VQAError when the upstream call fails."""
    t0 = time.perf_counter()
    if fake:
        return {"answer": canned_answer(question), "model": "fake/vqa",
                "latency_ms": (time.perf_counter() - t0) * 1000.0, "cost_usd": 0.0,
                "tokens_in": 0, "tokens_out": 0}
    client = client or make_client()
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=build_messages(img, question),
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            extra_body={"reasoning_effort": "none"},  # REQUIRED: Qwen3.5 thinks otherwise
        )
    except Exception as e:  # any SDK/HTTP error -> 502 with the message
        raise VQAError(f"{type(e).__name__}: {e}"[:500]) from e
    latency_ms = (time.perf_counter() - t0) * 1000.0
    choice = resp.choices[0] if getattr(resp, "choices", None) else None
    answer = ((getattr(getattr(choice, "message", None), "content", None) or "")
              if choice is not None else "").strip()
    usage = getattr(resp, "usage", None)
    tin = int(getattr(usage, "prompt_tokens", 0) or 0)
    tout = int(getattr(usage, "completion_tokens", 0) or 0)
    pin, pout = price_per_token(model)
    return {"answer": answer, "model": model, "latency_ms": latency_ms,
            "cost_usd": tin * pin + tout * pout, "tokens_in": tin, "tokens_out": tout}
