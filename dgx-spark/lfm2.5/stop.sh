#!/usr/bin/env bash
set -euo pipefail

# Stop SGLang container for LiquidAI/LFM2.5-2.6B.
# Usage: ./stop.sh

CONTAINER_NAME="${CONTAINER_NAME:-sglang-lfm2.5-2.6b}"

if docker ps --format '{{.Names}}' | grep -qx "${CONTAINER_NAME}"; then
  echo "Stopping ${CONTAINER_NAME}..."
  docker stop "${CONTAINER_NAME}" >/dev/null 2>&1 || true
  echo "Container ${CONTAINER_NAME} stopped."
else
  echo "Container ${CONTAINER_NAME} is not running."
fi