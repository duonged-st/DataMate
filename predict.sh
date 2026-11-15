#!/usr/bin/env bash
set -euo pipefail
cd /code
mkdir -p /result/logs
work="$(mktemp -d /tmp/datamate.XXXXXX)"
trap 'rm -rf "$work"' EXIT
python3 -m datamate predict --data /data --output /result --work "$work" \
  --model-file /code/models/Qwen3.5-9B-Q4_K_M.gguf --run-time-limit 21300 \
  --threshold-config /code/confidence_policy.json
