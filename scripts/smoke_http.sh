#!/usr/bin/env bash
# Quick HTTP smoke against a running stack (defaults to local Compose ports).
set -euo pipefail
MODEL_URL="${SMOKE_MODEL_URL:-http://127.0.0.1:8000}"
code="$(curl -sS -o /dev/null -w '%{http_code}' "${MODEL_URL}/health")"
if [[ "$code" != "200" ]]; then
  echo "Expected 200 from ${MODEL_URL}/health, got ${code}" >&2
  exit 1
fi
echo "OK ${MODEL_URL}/health -> 200"
