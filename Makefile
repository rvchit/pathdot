# pathfrontier + cello — one-stop targets. `make help` lists them.
SHELL := /bin/bash
UV ?= uv
PY := .venv/bin/python
DATASET ?= pcam
GRID ?= 24

.PHONY: help setup setup-py setup-node dev dev-stub backend frontend ml ml-stub tile test test-py test-node lint build docker-up docker-down clean-tiles

help:
	@echo "make setup      install python (.venv via uv) + npm deps for cello/backend and cello/frontend"
	@echo "make dev        run backend (:4001), frontend (:4000) and ML service (:8008) together"
	@echo "make dev-stub   same, but with the canned node ML stub instead of the real model"
	@echo "make tile       build the demo mosaic slide + ground truth and its DZI pyramid (DATASET=$(DATASET) GRID=$(GRID))"
	@echo "make test       pytest + lint + build for both node packages"
	@echo "make docker-up  docker compose up --build (frontend :4000, backend :4001, ml :8008)"

# ---------- setup ----------
setup: setup-py setup-node

setup-py:
	@test -d .venv || $(UV) venv --python 3.11 .venv
	$(UV) pip install -e ".[dev]"

setup-node:
	npm install --prefix cello/backend --no-audit --no-fund
	npm install --prefix cello/frontend --no-audit --no-fund

# ---------- run ----------
dev:
	bash scripts/dev.sh

dev-stub:
	ML_STUB=1 bash scripts/dev.sh

backend:
	npm run dev --prefix cello/backend

frontend:
	npm run dev --prefix cello/frontend

ml:
	$(UV) run uvicorn pathfrontier.serve:app --host 0.0.0.0 --port 8008 --reload

ml-stub:
	npm run ml-stub --prefix cello/backend

# ---------- data ----------
tile:
	$(UV) run python scripts/make_mosaic_slide.py --dataset $(DATASET) --grid $(GRID)

# ---------- quality ----------
test: test-py test-node

test-py:
	$(UV) run pytest

test-node: lint build

lint:
	npm run lint --prefix cello/backend
	npm run lint --prefix cello/frontend

build:
	npm run build --prefix cello/backend
	npm run build --prefix cello/frontend

# ---------- docker ----------
docker-up:
	docker compose up --build

docker-down:
	docker compose down

clean-tiles:
	rm -rf tiles/_fixture tiles/_fixture_files
