#!/usr/bin/env bash
# Try the DIRECT multipart dataset upload (<=150 MB), which is a different endpoint from the
# getUploadEndpoint call that firectl uses and that currently returns "uploads temporarily disabled".
# Run from pathvqa_kit/scripts in YOUR terminal with FIREWORKS_API_KEY exported:
#   bash upload_rest.sh pathvqa-val ../fireworks/ft_val.jsonl
#   bash upload_rest.sh pathvqa-train-4k ../fireworks_small/ft_train_4000_384.jsonl
set -euo pipefail
ID=${1:?dataset id}; FILE=${2:?jsonl path}; ACCT=${FIREWORKS_ACCOUNT_ID:-rvchit}
: "${FIREWORKS_API_KEY:?export FIREWORKS_API_KEY first}"
API=https://api.fireworks.ai/v1/accounts/$ACCT
N=$(wc -l < "$FILE" | tr -d ' ')
SIZE=$(stat -f %z "$FILE")
if [ "$SIZE" -gt 150000000 ]; then echo "file is $SIZE bytes > 150 MB; direct upload won't accept it"; exit 1; fi
echo "1/3 create dataset $ID ($N lines, $SIZE bytes)"
curl -sS -X POST "$API/datasets" -H "Authorization: Bearer $FIREWORKS_API_KEY" -H "Content-Type: application/json" \
  -d "{\"datasetId\":\"$ID\",\"dataset\":{\"displayName\":\"$ID\",\"userUploaded\":{},\"exampleCount\":\"$N\"}}" | head -c 600; echo
echo "2/3 direct multipart upload"
curl -sS -X POST "$API/datasets/$ID:upload" -H "Authorization: Bearer $FIREWORKS_API_KEY" \
  -F "file=@$FILE;type=application/jsonl" | head -c 600; echo
echo "3/3 state (poll until READY)"
for i in $(seq 1 30); do
  S=$(curl -sS "$API/datasets/$ID" -H "Authorization: Bearer $FIREWORKS_API_KEY" | python3 -c "import sys,json; j=json.load(sys.stdin); print(j.get('state'), j.get('status',{}))" 2>/dev/null || echo "?")
  echo "  $S"; case "$S" in READY*) exit 0;; FAILED*) exit 1;; esac; sleep 5
done
