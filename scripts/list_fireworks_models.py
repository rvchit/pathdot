"""List vision-capable Fireworks models.

With FIREWORKS_API_KEY set: GET {FIREWORKS_BASE_URL}/models and print the ids that look
vision-capable (by `supports_image_input` / tags, falling back to name heuristics).
Without a key: print the researched PRICES table from pathfrontier.vlm_client.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import httpx
import typer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pathfrontier import config
from pathfrontier.vlm_client import DEFAULT_BASELINE_MODEL, PRICES, price_for

app = typer.Typer(add_completion=False)
_VISION_HINT = re.compile(r"vl|vision|llava|pixtral|maverick|scout|kimi|gemma-?3|flash|glm|"
                          r"ember|muse|mimo|qwen3p8", re.IGNORECASE)


def print_prices() -> None:
    typer.echo(f"{'model id':<62} {'$in/M':>7} {'$out/M':>7}")
    for mid, (pin, pout) in PRICES.items():
        mark = " *" if mid == DEFAULT_BASELINE_MODEL else ""
        typer.echo(f"{mid:<62} {pin:>7.2f} {pout:>7.2f}{mark}")
    typer.echo("* = DEFAULT_BASELINE_MODEL. Source: docs.fireworks.ai/serverless/pricing, 2026-10-29")


def _looks_vision(m: dict) -> bool:
    for key in ("supports_image_input", "vision", "supports_vision"):
        if m.get(key) is True:
            return True
    tags = " ".join(str(t) for t in (m.get("tags") or m.get("capabilities") or []))
    if re.search(r"vision|image|multimodal", tags, re.IGNORECASE):
        return True
    return bool(_VISION_HINT.search(m.get("id", "")))


@app.command()
def main(
    all_models: bool = typer.Option(False, "--all", help="Print every model, not just vision"),
    timeout: float = typer.Option(30.0),
) -> None:
    if not config.FIREWORKS_API_KEY:
        typer.echo("FIREWORKS_API_KEY not set; printing researched price table instead.\n")
        print_prices()
        return
    url = config.FIREWORKS_BASE_URL.rstrip("/") + "/models"
    headers = {"Authorization": f"Bearer {config.FIREWORKS_API_KEY}"}
    try:
        r = httpx.get(url, headers=headers, timeout=timeout)
        r.raise_for_status()
    except httpx.HTTPError as exc:
        typer.echo(f"request failed: {exc}\n")
        print_prices()
        raise typer.Exit(1)
    body = r.json()
    models = body.get("data", body if isinstance(body, list) else [])
    rows = [m for m in models if all_models or _looks_vision(m)]
    typer.echo(f"{len(rows)} of {len(models)} models from {url}")
    typer.echo(f"{'model id':<62} {'$in/M':>7} {'$out/M':>7}  known")
    for m in sorted(rows, key=lambda x: x.get("id", "")):
        mid = m.get("id", "")
        pin, pout = price_for(mid)
        typer.echo(f"{mid:<62} {pin:>7.2f} {pout:>7.2f}  {'yes' if mid in PRICES else 'tier'}")


if __name__ == "__main__":
    app()
