#!/usr/bin/env bash
set -euo pipefail
model="$1"
shift
exec "${LLAMA_SERVER:-llama-server}" \
  --model "$model" --host 127.0.0.1 --port 8080 --n-gpu-layers 99 \
  --ctx-size 16384 --parallel 1 --batch-size 256 --ubatch-size 256 \
  --reasoning off "$@"
