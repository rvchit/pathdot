# pathfrontier ML inference service (FastAPI, CPU torch). Build from the monorepo root:
#   docker build -f Dockerfile.ml -t pathfrontier-ml .
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PATHFRONTIER_DATA_DIR=/app/data \
    PATHFRONTIER_CACHE_DIR=/app/cache \
    PATHFRONTIER_RESULTS_DIR=/app/results \
    PATHFRONTIER_TILES_DIR=/app/tiles \
    HF_HOME=/app/cache/hf

WORKDIR /app

# libgl/libglib: OpenCV/pyvips-style wheels sometimes dlopen these; harmless if unused.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 curl \
    && rm -rf /var/lib/apt/lists/*

# CPU-only torch first so `pip install -e .` does not pull the CUDA wheels (~2 GB).
RUN pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

COPY pyproject.toml README.md ./
COPY pathfrontier ./pathfrontier
COPY scripts ./scripts
RUN pip install -e .

RUN mkdir -p /app/data /app/cache /app/results /app/tiles
EXPOSE 8008
HEALTHCHECK --interval=30s --timeout=5s --start-period=120s CMD curl -fsS http://localhost:8008/health || exit 1
CMD ["uvicorn", "pathfrontier.serve:app", "--host", "0.0.0.0", "--port", "8008"]
