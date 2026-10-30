#!/usr/bin/env bash
# Run the three dev servers together and stop them all on Ctrl-C.
#   backend  cello/backend   tsx watch        -> http://localhost:4001
#   frontend cello/frontend  vite             -> http://localhost:4000
#   ml       pathfrontier.serve (uvicorn)     -> http://localhost:8008
# ML_STUB=1 uses cello/backend/scripts/ml-stub.mjs instead of the real service.
set -u
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

pids=()
cleanup() {
  echo
  echo "[dev] stopping..."
  for p in "${pids[@]}"; do kill "$p" 2>/dev/null; done
  wait 2>/dev/null
}
trap cleanup EXIT INT TERM

run() { # name, colour, dir, cmd...
  local name=$1 colour=$2 dir=$3; shift 3
  (
    cd "$dir" && "$@" 2>&1 | while IFS= read -r line; do printf "\033[%sm[%s]\033[0m %s\n" "$colour" "$name" "$line"; done
  ) &
  pids+=($!)
}

run backend  36 "$ROOT/cello/backend"  npm run dev --silent
run frontend 35 "$ROOT/cello/frontend" npm run dev --silent

if [[ "${ML_STUB:-0}" == "1" ]]; then
  run ml 33 "$ROOT/cello/backend" node scripts/ml-stub.mjs 8008
elif [[ -x "$ROOT/.venv/bin/uvicorn" ]]; then
  run ml 33 "$ROOT" "$ROOT/.venv/bin/uvicorn" pathfrontier.serve:app --host 0.0.0.0 --port 8008 --reload
else
  run ml 33 "$ROOT" uv run uvicorn pathfrontier.serve:app --host 0.0.0.0 --port 8008 --reload
fi

echo "[dev] frontend http://localhost:4000  backend http://localhost:4001  ml http://localhost:8008  (Ctrl-C stops all)"
wait
