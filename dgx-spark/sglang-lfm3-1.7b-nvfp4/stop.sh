#!/usr/bin/env bash
set -euo pipefail

# Stop the DGX Spark SGLang container for LiquidAI/LFM3-1.7B.
# Usage: ./stop.sh

CONTAINER_NAME="${CONTAINER_NAME:-sglang-lfm3-1.7b-nvfp4}"

if docker ps --format '{{.Names}}' | grep -qx "${CONTAINER_NAME}"; then
  echo "Stopping ${CONTAINER_NAME}..."
  docker stop "${CONTAINER_NAME}" >/dev/null 2>&1 || true
  echo "Container ${CONTAINER_NAME} stopped."
else
  echo "Container ${CONTAINER_NAME} is not running."
fi
