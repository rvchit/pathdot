# pathdot

Pathology tile triage for the "Make your own frontier" hackathon (Fireworks / Microsoft / NVIDIA,
San Francisco, Nov 2 2026).

A pathology foundation encoder with a linear probe scores every tile of a slide for a fraction of a
cent; only the uncertain band goes to a fine-tuned vision-language model or a pathologist. We measure
the cost-vs-sensitivity frontier against a general-purpose VLM baseline, and demo it as a tumor
heatmap inside **Cello**, our slide viewer.

- **`GUIDE.md`** — start here: pre-hackathon checklist, day-of timeline with commands, demo, troubleshooting.
- `pathfrontier/` — the Python package (data, encoders, probe, VLM client, fine-tune, cascade, metrics, inference service).
- `cello/` — the slide viewer (React + OpenSeadragon + Express) with the AI triage panel and heatmap overlay.
- `docs/CONTRACTS.md` — the interfaces every module honours. `docs/FINETUNE.md` — Fireworks vision LoRA recipe. `docs/PITCH.md` — slides and demo script.

```bash
make setup                      # .venv via uv + npm installs
.venv/bin/python -m pytest -q   # offline test suite
make tile                       # build the demo mosaic slide from real PCam tiles
make dev                        # viewer :4000, backend :4001, ML service :8008
open "http://localhost:4000/?image=pcam_mosaic_24&triage=1&gt=1"
```
